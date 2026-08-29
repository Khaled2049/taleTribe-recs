"""retrieve → fuse → score → diversify.

The one path every recommendation mode goes through, extracted so the HTTP
endpoints and the debugging CLI cannot drift apart — if they ranked differently,
the CLI would be useless for exactly the job it exists for.

Order matters and is not arbitrary:

1. **Retrieve** per seed vector, independently and concurrently.
2. **Fuse** by rank (RRF), because per-seed cosine scores are not comparable
   across seeds.
3. **Score** with the three-term blend, which needs one unified candidate list —
   hence after fusion, not before.
4. **Diversify** last, because MMR trades relevance for variety and it can only
   know the relevance it is trading once scoring is done.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from recommendation_engine.fusion import merge_candidate_records
from recommendation_engine.mmr import diversify, intra_list_diversity
from recommendation_engine.retrieval import RetrievalFilters, Retriever
from recommendation_engine.scoring import (
    CatalogStats,
    ScoringConfig,
    blend,
    collaborative_score,
)

logger = logging.getLogger(__name__)

# Rows scored before MMR trims to top_k. Larger gives MMR more room to find
# variety; too large and it starts promoting weak matches for novelty's sake.
CANDIDATE_MULTIPLIER = 5


@dataclass
class RankedItem:
    """One recommendation, with its score fully attributed."""

    id: int
    story_id: str
    title: str
    author: Optional[str]
    genres: List[str]
    themes: List[str]
    tone: List[str]
    core_premise: Optional[str]
    published_year: Optional[int]
    score: float
    breakdown: dict
    matched_query_count: int = 0
    embed_input_sha: Optional[str] = None

    def as_dict(self, include_breakdown: bool = False) -> dict:
        payload = {
            "id": self.id,
            "story_id": self.story_id,
            "title": self.title,
            "author": self.author,
            "genres": self.genres,
            "themes": self.themes,
            "tone": self.tone,
            "core_premise": self.core_premise,
            "published_year": self.published_year,
            "score": round(self.score, 6),
            "matched_query_count": self.matched_query_count,
        }
        if include_breakdown:
            payload["breakdown"] = self.breakdown
        return payload


@dataclass
class RankedResult:
    items: List[RankedItem] = field(default_factory=list)
    degraded: bool = False
    diversity: Optional[float] = None
    candidates_considered: int = 0

    def as_dict(self, include_breakdown: bool = False) -> dict:
        return {
            "items": [item.as_dict(include_breakdown) for item in self.items],
            "degraded": self.degraded,
            "diversity": (
                round(self.diversity, 4) if self.diversity is not None else None
            ),
            "candidates_considered": self.candidates_considered,
        }


async def _cf_scores(
    db,
    candidate_ids: Sequence[int],
    seed_weights: dict,
    config: ScoringConfig,
) -> dict:
    """Collaborative score per candidate.

    Short-circuits to zeros while `w_cf_ceiling` is 0 — no point querying a table
    whose result is multiplied by nothing. The query below is the real one, so
    turning CF on is a config change plus a co-occurrence backfill.
    """
    if config.w_cf_ceiling <= 0 or not seed_weights or not candidate_ids:
        return {item_id: 0.0 for item_id in candidate_ids}

    rows = await db.read_pool.fetch(
        """
        SELECT item_a, item_b, cooc, sim
          FROM recommendations.item_cooccurrence
         WHERE (item_a = ANY($1::bigint[]) AND item_b = ANY($2::bigint[]))
            OR (item_b = ANY($1::bigint[]) AND item_a = ANY($2::bigint[]))
        """,
        list(candidate_ids),
        list(seed_weights),
    )

    # The table stores one direction (item_a < item_b); resolve which end is the
    # candidate per row rather than assuming.
    by_candidate: dict = {item_id: {} for item_id in candidate_ids}
    seeds = set(seed_weights)
    for row in rows:
        a, b = row["item_a"], row["item_b"]
        candidate, seed = (a, b) if b in seeds else (b, a)
        if candidate in by_candidate:
            by_candidate[candidate][seed] = float(row["sim"])

    return {
        item_id: collaborative_score(sims, seed_weights)
        for item_id, sims in by_candidate.items()
    }


async def rank(
    db,
    retriever: Retriever,
    query_vectors: Sequence[Sequence[float]],
    top_k: int,
    config: ScoringConfig,
    stats: CatalogStats,
    filters: Optional[RetrievalFilters] = None,
    seed_weights: Optional[dict] = None,
    candidate_pool: Optional[int] = None,
) -> RankedResult:
    """Run the full pipeline for one or more query vectors."""
    filters = filters or RetrievalFilters()
    pool = candidate_pool or max(top_k * CANDIDATE_MULTIPLIER, top_k)

    if not query_vectors:
        # No vector to search with (no embedder, or a reader with no history):
        # popularity order rather than an error.
        rows = await retriever.popular(limit=pool, filters=filters)
        for row in rows:
            row["semantic"] = 0.0
            row["matched_query_count"] = 0
        degraded = False
    else:
        results = await retriever.knn_many(query_vectors, limit=pool, filters=filters)
        degraded = any(result.degraded for result in results)

        if len(results) == 1:
            # ONE seed: use the cosine similarity directly.
            #
            # RRF must not be applied here. Fusing a single ranked list replaces
            # every similarity with a function of its rank — rank 1 becomes
            # exactly 1.0, rank 2 exactly 61/62, and so on — regardless of whether
            # the top match scored 0.95 or 0.31. Three things break: the score
            # stops meaning anything to a caller, MMR's relevance term goes almost
            # uniform so the diversity term decides the order on its own, and the
            # popularity blend gets weighted against a rank artifact instead of a
            # real similarity.
            rows = results[0].rows
            for row in rows:
                similarity = float(row.get("sem_cos") or 0.0)
                row["semantic"] = max(0.0, min(1.0, similarity))
                row["per_query_similarity"] = [similarity]
                row["matched_query_count"] = 1
        else:
            # MULTIPLE seeds: fuse by rank, because per-seed cosine scores are not
            # comparable — a query about an obscure book yields uniformly lower
            # similarities than one about a well-covered genre, so score-based
            # merging would let the better-covered seed dominate on scale alone.
            rows = merge_candidate_records([result.rows for result in results])

    if not rows:
        return RankedResult(degraded=degraded)

    cf = await _cf_scores(db, [row["id"] for row in rows], seed_weights or {}, config)

    for row in rows:
        breakdown = blend(
            semantic=float(row.get("semantic") or 0.0),
            popularity_score=float(row.get("pop_score") or 0.0),
            collaborative=cf.get(row["id"], 0.0),
            n_interactions=float(row.get("n_interactions") or 0),
            config=config,
        )
        row["score"] = breakdown.score
        row["_breakdown"] = breakdown.as_dict()

    selected = diversify(rows, k=top_k, lambda_=config.mmr_lambda)

    items = [
        RankedItem(
            id=row["id"],
            story_id=str(row["story_id"]),
            title=row["title"],
            author=row.get("author"),
            genres=list(row.get("genres") or []),
            themes=list(row.get("themes") or []),
            tone=list(row.get("tone") or []),
            core_premise=row.get("core_premise"),
            published_year=row.get("published_year"),
            score=row["score"],
            breakdown=row["_breakdown"],
            matched_query_count=int(row.get("matched_query_count") or 0),
            embed_input_sha=row.get("embed_input_sha"),
        )
        for row in selected
    ]

    return RankedResult(
        items=items,
        degraded=degraded,
        diversity=intra_list_diversity(
            [row.get("embedding") or [] for row in selected if row.get("embedding")]
        ),
        candidates_considered=len(rows),
    )
