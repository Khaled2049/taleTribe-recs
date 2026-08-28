"""Turn raw interactions into the two things scoring actually reads.

`item_stats` — per-item aggregates plus a materialized `pop_score`. Materialized
because it is a *global* property recomputed on a schedule, and volume-damped
arithmetic has no business running inside a retrieval query.

`user_taste` — one precomputed vector per reader. This is what makes behavioral
recommendations need no runtime embedding call at all: the reader's taste is
already a point in the same space as the catalog.

Both are source-agnostic. Whether the interactions came from a synthetic generator
or from Firestore, this code is identical — which is the whole reason the loader and
the source are separated.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, cast

import numpy as np
import numpy.typing as npt

from recommendation_engine.scoring import ScoringConfig, popularity

logger = logging.getLogger(__name__)

# A reader needs at least this many weighted signals before a taste vector means
# anything. Below it, behavioral mode should fall back to popularity rather than
# extrapolate a personality from two data points.
MIN_SIGNALS_FOR_TASTE = 3
FloatArray = npt.NDArray[np.float64]


@dataclass
class RefreshStats:
    items_updated: int = 0
    tastes_built: int = 0
    tastes_skipped_thin: int = 0
    global_mean_rating: Optional[float] = None
    p95_engagement: Optional[float] = None

    def as_dict(self) -> dict:
        return self.__dict__.copy()


async def refresh_item_stats(pool, config: ScoringConfig) -> RefreshStats:
    """Recompute `item_stats` from `interactions`.

    Aggregation happens in SQL; the *scoring* happens in Python via
    `scoring.popularity`, deliberately. Reimplementing the Bayesian damping and log
    compression in SQL would be a second definition of the formula, free to drift
    from the one the unit tests cover.
    """
    stats = RefreshStats()

    rows = await pool.fetch("""
        SELECT
            item_id,
            COUNT(*) FILTER (WHERE kind = 'like')       AS likes,
            COUNT(*) FILTER (WHERE kind = 'rating')     AS ratings_count,
            AVG(value) FILTER (WHERE kind = 'rating')   AS avg_rating,
            COUNT(*) FILTER (WHERE kind = 'completion') AS completions
          FROM recommendations.interactions
         GROUP BY item_id
        """)
    if not rows:
        logger.info("no interactions; item_stats left unchanged")
        return stats

    # Global values the popularity term needs. Computed from the same snapshot the
    # per-item numbers come from, so they cannot disagree.
    rated = [float(r["avg_rating"]) for r in rows if r["avg_rating"] is not None]
    global_mean = sum(rated) / len(rated) if rated else 3.5

    engagement_volumes = sorted(
        float(r["likes"]) + config.pop_completion_mult * float(r["completions"])
        for r in rows
    )
    index = int(round(0.95 * (len(engagement_volumes) - 1)))
    p95 = max(1.0, engagement_volumes[index])

    stats.global_mean_rating = round(global_mean, 4)
    stats.p95_engagement = round(p95, 4)

    payload = []
    for row in rows:
        likes = int(row["likes"])
        ratings_count = int(row["ratings_count"])
        completions = int(row["completions"])
        avg_rating = float(row["avg_rating"]) if row["avg_rating"] is not None else None
        # Drives the cold-start ramp: only these three count, because a mere
        # progress record is too weak to license behavioral weighting.
        n_interactions = likes + ratings_count + completions

        pop_score = popularity(
            likes=likes,
            ratings_count=ratings_count,
            avg_rating=avg_rating,
            completions=completions,
            global_mean_rating=global_mean,
            p95_engagement=p95,
            config=config,
        )
        payload.append(
            (
                row["item_id"],
                likes,
                ratings_count,
                avg_rating,
                completions,
                n_interactions,
                pop_score,
            )
        )

    await pool.executemany(
        """
        INSERT INTO recommendations.item_stats
            (item_id, likes, ratings_count, avg_rating, completions,
             n_interactions, pop_score, refreshed_at)
        VALUES ($1, $2, $3, $4::real, $5, $6, $7::real, now())
        ON CONFLICT (item_id) DO UPDATE SET
            likes = EXCLUDED.likes,
            ratings_count = EXCLUDED.ratings_count,
            avg_rating = EXCLUDED.avg_rating,
            completions = EXCLUDED.completions,
            n_interactions = EXCLUDED.n_interactions,
            pop_score = EXCLUDED.pop_score,
            refreshed_at = now()
        """,
        payload,
    )
    stats.items_updated = len(payload)
    return stats


async def rebuild_user_taste(pool) -> RefreshStats:
    """Rebuild `user_taste` from interactions and item embeddings.

    A reader's taste vector is the **engagement-weighted mean** of the embeddings of
    books they engaged with positively, L2-normalized. Normalizing matters: cosine
    distance ignores magnitude, so an unnormalized mean would give heavy readers
    longer vectors for no reason.

    Books rated 2 or below are excluded from the mean *and* recorded as suppressed.
    Averaging in a book someone disliked would aim their recommendations at exactly
    what they rejected.
    """
    stats = RefreshStats()

    rows = await pool.fetch("""
        SELECT
            i.user_id,
            i.item_id,
            i.kind,
            i.weight,
            i.value,
            it.embedding
          FROM recommendations.interactions i
          JOIN recommendations.items it ON it.id = i.item_id
         WHERE it.embedding IS NOT NULL
         ORDER BY i.user_id
        """)
    if not rows:
        logger.info("no interactions; user_taste left unchanged")
        return stats

    by_user: dict = {}
    for row in rows:
        entry = by_user.setdefault(
            row["user_id"], {"vectors": [], "weights": [], "seeds": set(), "bad": set()}
        )
        # A low rating is a negative signal: suppress the item, contribute nothing.
        if row["kind"] == "rating" and row["value"] is not None and row["value"] <= 2:
            entry["bad"].add(row["item_id"])
            continue
        weight = float(row["weight"] or 0.0)
        if weight <= 0:
            continue
        vector = row["embedding"]
        to_list = getattr(vector, "to_list", None)
        entry["vectors"].append(to_list() if callable(to_list) else list(vector))
        entry["weights"].append(weight)
        entry["seeds"].add(row["item_id"])

    payload = []
    for user_id, entry in by_user.items():
        total_weight = sum(entry["weights"])
        # Weighted count, not row count: three half-hearted partial reads should not
        # look like three completions.
        if not entry["vectors"] or total_weight < MIN_SIGNALS_FOR_TASTE:
            stats.tastes_skipped_thin += 1
            continue

        matrix = cast(FloatArray, np.asarray(entry["vectors"], dtype=np.float64))
        weights = cast(FloatArray, np.asarray(entry["weights"], dtype=np.float64))
        mean = cast(FloatArray, (matrix * weights[:, None]).sum(axis=0) / weights.sum())
        norm = cast(float, np.linalg.norm(mean))
        if norm == 0:
            stats.tastes_skipped_thin += 1
            continue

        payload.append(
            (
                user_id,
                cast(list[float], (mean / norm).tolist()),
                sorted(entry["seeds"]),
                sorted(entry["bad"]),
                # Reported as the weighted total so the endpoint's threshold means
                # "enough evidence", not "enough rows".
                int(round(total_weight)),
            )
        )

    if payload:
        await pool.executemany(
            """
            INSERT INTO recommendations.user_taste
                (user_id, taste_embedding, seed_item_ids, suppressed_item_ids,
                 n_signals, computed_at)
            VALUES ($1, $2::vector, $3::bigint[], $4::bigint[], $5, now())
            ON CONFLICT (user_id) DO UPDATE SET
                taste_embedding = EXCLUDED.taste_embedding,
                seed_item_ids = EXCLUDED.seed_item_ids,
                suppressed_item_ids = EXCLUDED.suppressed_item_ids,
                n_signals = EXCLUDED.n_signals,
                computed_at = now()
            """,
            payload,
        )
    stats.tastes_built = len(payload)
    return stats


async def rebuild_cooccurrence(pool, config: ScoringConfig, min_cooc: int = 3) -> int:
    """Rebuild `item_cooccurrence`.

    Not needed while `w_cf_ceiling` is 0, but written so the CF term can be switched
    on with a config change and a run of this — which was the point of shipping CF as
    a live stub rather than omitting it.

    `min_cooc` discards pairs seen together by fewer than three readers. Below that
    the shrinkage term is doing all the work anyway, and the table would grow
    quadratically in noise.
    """
    await pool.execute("DELETE FROM recommendations.item_cooccurrence")
    result = await pool.execute(
        """
        WITH positive AS (
            SELECT DISTINCT user_id, item_id
              FROM recommendations.interactions
             WHERE weight > 0
        ),
        totals AS (
            SELECT item_id, COUNT(*) AS n FROM positive GROUP BY item_id
        ),
        pairs AS (
            SELECT a.item_id AS item_a, b.item_id AS item_b, COUNT(*) AS cooc
              FROM positive a
              JOIN positive b
                ON a.user_id = b.user_id
               -- One direction only, matching the table's CHECK constraint.
               AND a.item_id < b.item_id
             GROUP BY a.item_id, b.item_id
            HAVING COUNT(*) >= $1
        )
        INSERT INTO recommendations.item_cooccurrence (item_a, item_b, cooc, sim)
        SELECT p.item_a, p.item_b, p.cooc,
               -- Shrunk cosine: without the lambda, two obscure books sharing one
               -- reader would score a perfect 1.0.
               p.cooc / (sqrt(ta.n::float * tb.n::float) + $2)
          FROM pairs p
          JOIN totals ta ON ta.item_id = p.item_a
          JOIN totals tb ON tb.item_id = p.item_b
        """,
        min_cooc,
        config.cf_shrinkage_lambda,
    )
    try:
        return int(str(result).split()[-1])
    except (ValueError, IndexError):
        return 0


__all__: List[str] = [
    "MIN_SIGNALS_FOR_TASTE",
    "RefreshStats",
    "rebuild_cooccurrence",
    "rebuild_user_taste",
    "refresh_item_stats",
]
