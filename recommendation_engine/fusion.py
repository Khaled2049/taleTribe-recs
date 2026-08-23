"""Reciprocal Rank Fusion for multi-item queries.

When a reader says "I liked Dune *and* Pride and Prejudice", the tempting move is
to average the two embeddings and search once. That is wrong: the mean of two
distant vectors points at a region of taste space that resembles neither book —
often literally nothing, since embedding spaces are not convex in that way.

So each seed item is retrieved **independently and in parallel**, and the ranked
lists are merged by rank rather than by score:

    RRF(i) = Σ_q  1 / (k + rank_q(i))

Rank-based fusion is the right tool because the per-list cosine scores are not
comparable — a query about an obscure book yields uniformly lower similarities
than one about a well-covered genre, so score-based merging would let the
better-covered seed dominate purely by scale.

The output is normalized back onto [0, 1] so it can be substituted directly for
cosine similarity as the `sem` term in the scoring blend.
"""

from typing import Dict, Hashable, Iterable, List, Mapping, Sequence, Tuple

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Hashable]],
    k: float = DEFAULT_RRF_K,
) -> Dict[Hashable, float]:
    """Fuse ranked ID lists into raw RRF scores.

    `k` damps the influence of top positions: with k=60, ranks 1 and 2 score
    1/61 and 1/62 — close together, so a candidate that places respectably across
    several seeds beats one that places first for a single seed. That is the
    behaviour we want from a multi-book query.

    Items are ranked by their position in each list, so the caller must pass lists
    already ordered best-first.
    """
    k = max(1.0, float(k))
    scores: Dict[Hashable, float] = {}
    for ranked in ranked_lists:
        for position, item_id in enumerate(ranked, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + position)
    return scores


def max_rrf_score(n_lists: int, k: float = DEFAULT_RRF_K) -> float:
    """The score an item would get by placing first in every list.

    This is the normalizer: dividing by it maps RRF onto (0, 1] so the fused
    score is on the same scale as a cosine similarity and can be dropped into the
    blend without a second set of weights.
    """
    k = max(1.0, float(k))
    return max(1, int(n_lists)) / (k + 1.0)


def fuse_normalized(
    ranked_lists: Sequence[Sequence[Hashable]],
    k: float = DEFAULT_RRF_K,
) -> Dict[Hashable, float]:
    """Fuse and normalize to (0, 1].

    An item first in every list scores exactly 1.0.
    """
    lists = [lst for lst in ranked_lists if lst]
    if not lists:
        return {}
    raw = reciprocal_rank_fusion(lists, k)
    ceiling = max_rrf_score(len(lists), k)
    if ceiling <= 0:
        return {item_id: 0.0 for item_id in raw}
    return {item_id: min(1.0, score / ceiling) for item_id, score in raw.items()}


def fuse_and_rank(
    ranked_lists: Sequence[Sequence[Hashable]],
    k: float = DEFAULT_RRF_K,
) -> List[Tuple[Hashable, float]]:
    """Fused (id, normalized_score) pairs, best first.

    Ties break on the id so the ordering is deterministic. Without that, two
    equally-fused candidates could swap places between identical requests, which
    makes both caching and eval reproducibility flaky.
    """
    fused = fuse_normalized(ranked_lists, k)
    return sorted(fused.items(), key=lambda pair: (-pair[1], str(pair[0])))


def merge_candidate_records(
    ranked_lists: Sequence[Sequence[Mapping]],
    id_field: str = "id",
    k: float = DEFAULT_RRF_K,
) -> List[dict]:
    """Fuse lists of row mappings, keeping one record per id.

    Convenience for the retrieval path, which gets full rows back from Postgres
    rather than bare ids. The first-seen record for an id wins; per-seed cosine
    scores are collected into `per_query_similarity` so an explanation can say
    *which* of the reader's books a recommendation came from.
    """
    id_lists: List[List[Hashable]] = []
    records: Dict[Hashable, dict] = {}
    per_query: Dict[Hashable, List[float]] = {}

    for ranked in ranked_lists:
        ids: List[Hashable] = []
        for row in ranked:
            item_id = row[id_field]
            ids.append(item_id)
            if item_id not in records:
                records[item_id] = dict(row)
            similarity = row.get("sem_cos")
            if similarity is not None:
                per_query.setdefault(item_id, []).append(float(similarity))
        id_lists.append(ids)

    out: List[dict] = []
    for item_id, fused_score in fuse_and_rank(id_lists, k):
        record = records[item_id]
        record["semantic"] = fused_score
        record["per_query_similarity"] = per_query.get(item_id, [])
        record["matched_query_count"] = len(per_query.get(item_id, []))
        out.append(record)
    return out


__all__: Iterable[str] = [
    "DEFAULT_RRF_K",
    "fuse_and_rank",
    "fuse_normalized",
    "max_rrf_score",
    "merge_candidate_records",
    "reciprocal_rank_fusion",
]
