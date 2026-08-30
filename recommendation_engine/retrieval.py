"""The KNN primitive: one place where vector search touches the database.

Two decisions here are load-bearing and easy to undo by accident:

**Query-time settings via SET LOCAL, inside an explicit transaction.** Handled by
`Database.vector_query`. A session-level `SET` is discarded by a transaction-mode
pooler, so the query would silently fall back to `ef_search = 40` and return
worse neighbours with nothing in the logs to show it.

**Multi-item retrieval issues N concurrent queries, not one LATERAL join.** A
LATERAL over N query vectors is one planner decision away from N sequential scans
instead of N index probes — a 10ms query becoming 2s, with no error. N separate
statements each get their own plan, and `asyncio.gather` recovers the round-trip
savings the join would have offered.
"""

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

from recommendation_engine.db import Database

logger = logging.getLogger(__name__)

# Columns every retrieval returns. The embedding comes back because MMR needs it
# and re-fetching would cost a second round trip; `sem_cos` is computed in SQL
# because pgvector's operator is the authority on the distance metric.
_SELECT_COLUMNS = """
    i.id, i.story_id, i.title, i.author,
    i.genres, i.themes, i.tone, i.core_premise, i.published_year,
    i.word_count, i.embedding, i.embed_input_sha, i.updated_at,
    1 - (i.embedding <=> $1) AS sem_cos,
    COALESCE(s.n_interactions, 0) AS n_interactions,
    COALESCE(s.pop_score, 0) AS pop_score
"""


@dataclass
class RetrievalFilters:
    """Metadata filters applied alongside the vector search.

    Every filter is optional and NULL-guarded in SQL, so one prepared statement
    covers all combinations rather than assembling SQL per request.
    """

    genres: Optional[Sequence[str]] = None
    themes: Optional[Sequence[str]] = None
    max_word_count: Optional[int] = None
    min_word_count: Optional[int] = None
    author: Optional[str] = None
    published_after: Optional[int] = None
    exclude_ids: Sequence[int] = field(default_factory=list)

    def as_params(self) -> list:
        """Ordered parameters matching the placeholders in `_KNN_SQL`."""
        return [
            list(self.genres) if self.genres else None,
            list(self.themes) if self.themes else None,
            self.max_word_count,
            self.min_word_count,
            f"%{self.author}%" if self.author else None,
            self.published_after,
            list(self.exclude_ids) if self.exclude_ids else [],
        ]


# $1 query vector, $2..$8 filters, $9 limit.
_KNN_SQL = f"""
SELECT {_SELECT_COLUMNS}
  FROM recommendations.items i
  LEFT JOIN recommendations.item_stats s ON s.item_id = i.id
 WHERE i.is_eligible
   AND i.embedding IS NOT NULL
   AND ($2::text[] IS NULL OR i.genres && $2::text[])
   AND ($3::text[] IS NULL OR i.themes && $3::text[])
   AND ($4::int IS NULL OR i.word_count IS NULL OR i.word_count <= $4::int)
   AND ($5::int IS NULL OR i.word_count IS NULL OR i.word_count >= $5::int)
   AND ($6::text IS NULL OR i.author ILIKE $6::text)
   AND ($7::int IS NULL OR i.published_year IS NULL OR i.published_year >= $7::int)
   AND NOT (i.id = ANY($8::bigint[]))
 ORDER BY i.embedding <=> $1
 LIMIT $9
"""

# Resolve a title/author the reader typed to catalog rows. Trigram similarity
# rather than exact match because readers misremember titles and skip subtitles.
_RESOLVE_SQL = """
SELECT i.id, i.story_id, i.title, i.author,
       i.embedding, i.genres, i.themes,
       GREATEST(
           similarity(i.title, $1),
           CASE WHEN $2::text IS NULL THEN 0
                ELSE similarity(COALESCE(i.author, ''), $2::text) END
       ) AS match_score
  FROM recommendations.items i
 WHERE i.is_eligible
   AND i.embedding IS NOT NULL
   AND (
        -- `%` is the pg_trgm similarity operator. A single %, not %%: asyncpg
        -- uses $N placeholders, so it does no %-formatting of the query string.
        i.title % $1
        OR ($2::text IS NOT NULL AND COALESCE(i.author, '') % $2::text)
   )
 ORDER BY match_score DESC, i.title
 LIMIT $3
"""


@dataclass
class RetrievalResult:
    rows: list[dict]
    degraded: bool = False
    reason: Optional[str] = None


def _row_to_dict(row) -> dict:
    """Convert an asyncpg Record to a plain dict, normalizing the embedding.

    pgvector's asyncpg codec hands back a `Vector` object, not a list — it is
    neither iterable nor `len()`-able, so it reaches numpy in MMR as a scalar and
    raises. Converting once here means nothing downstream has to know the vector
    came from pgvector.
    """
    data = dict(row)
    embedding = data.get("embedding")
    if embedding is not None and not isinstance(embedding, list):
        to_list = getattr(embedding, "to_list", None)
        data["embedding"] = to_list() if callable(to_list) else list(embedding)
    return data


class Retriever:
    """Vector retrieval against `recommendations.items`."""

    def __init__(
        self,
        db: Database,
        ef_search: int,
        max_scan_tuples: int,
        statement_timeout_ms: int,
    ) -> None:
        self._db = db
        self._ef_search = ef_search
        self._max_scan_tuples = max_scan_tuples
        self._statement_timeout_ms = statement_timeout_ms

    async def knn(
        self,
        query_vector: Sequence[float],
        limit: int,
        filters: Optional[RetrievalFilters] = None,
    ) -> RetrievalResult:
        """Nearest neighbours for one query vector.

        A statement timeout is treated as *degraded*, not as an error: under a
        selective filter an iterative scan can run long, and returning a partial
        shelf with `degraded: true` is far better for a reader than a 500. The
        caller surfaces the flag rather than hiding it.
        """
        filters = filters or RetrievalFilters()
        params = [list(query_vector), *filters.as_params(), int(limit)]
        try:
            async with self._db.vector_query(
                ef_search=self._ef_search,
                max_scan_tuples=self._max_scan_tuples,
                statement_timeout_ms=self._statement_timeout_ms,
            ) as conn:
                rows = await conn.fetch(_KNN_SQL, *params)
            return RetrievalResult(rows=[_row_to_dict(row) for row in rows])
        except Exception as exc:
            if _is_timeout(exc):
                logger.warning(
                    "knn_timeout limit=%d ef_search=%d", limit, self._ef_search
                )
                return RetrievalResult(rows=[], degraded=True, reason="timeout")
            raise

    async def knn_many(
        self,
        query_vectors: Sequence[Sequence[float]],
        limit: int,
        filters: Optional[RetrievalFilters] = None,
    ) -> list[RetrievalResult]:
        """One independent KNN per query vector, run concurrently.

        Independent by design — see the module docstring on why this is not a
        LATERAL join, and `fusion.py` on why the results are merged by rank rather
        than by averaging the vectors.
        """
        if not query_vectors:
            return []
        return list(
            await asyncio.gather(
                *(self.knn(vector, limit, filters) for vector in query_vectors)
            )
        )

    async def resolve_titles(
        self, title: str, author: Optional[str] = None, limit: int = 3
    ) -> list[dict]:
        """Fuzzy-match a reader-supplied title/author against the catalog."""
        if not (title or "").strip():
            return []
        async with self._db.read_pool.acquire() as conn:
            rows = await conn.fetch(
                _RESOLVE_SQL, title.strip(), (author or "").strip() or None, int(limit)
            )
        return [_row_to_dict(row) for row in rows]

    async def popular(
        self,
        limit: int,
        filters: Optional[RetrievalFilters] = None,
    ) -> list[dict]:
        """Popularity-ranked fallback, used when there is no query vector at all.

        Reached by a reader with almost no history, or when no embedding provider
        is configured. Mirrors `VectorStore`'s posture: degrade to something
        useful rather than erroring.
        """
        filters = filters or RetrievalFilters()
        params = filters.as_params()
        async with self._db.read_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT i.id, i.story_id, i.title,
                       i.author, i.genres, i.themes, i.tone, i.core_premise,
                       i.published_year, i.word_count, i.embedding,
                       i.embed_input_sha, i.updated_at,
                       0.0::float8 AS sem_cos,
                       COALESCE(s.n_interactions, 0) AS n_interactions,
                       COALESCE(s.pop_score, 0) AS pop_score
                  FROM recommendations.items i
                  LEFT JOIN recommendations.item_stats s ON s.item_id = i.id
                 WHERE i.is_eligible
                   AND ($1::text[] IS NULL OR i.genres && $1::text[])
                   AND ($2::text[] IS NULL OR i.themes && $2::text[])
                   AND ($3::int IS NULL OR i.word_count IS NULL OR i.word_count <= $3::int)
                   AND ($4::int IS NULL OR i.word_count IS NULL OR i.word_count >= $4::int)
                   AND ($5::text IS NULL OR i.author ILIKE $5::text)
                   AND ($6::int IS NULL OR i.published_year IS NULL OR i.published_year >= $6::int)
                   AND NOT (i.id = ANY($7::bigint[]))
                 ORDER BY COALESCE(s.pop_score, 0) DESC, i.updated_at DESC
                 LIMIT $8
                """,
                *params,
                int(limit),
            )
        return [_row_to_dict(row) for row in rows]

    async def catalog_stats(self) -> dict[str, Any]:
        """Corpus-level values the popularity term needs.

        Computed here rather than materialized because it is two aggregates over
        a small table.
        """
        async with self._db.read_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT
                    (SELECT AVG(avg_rating) FROM recommendations.item_stats
                      WHERE ratings_count > 0) AS global_mean_rating,
                    (SELECT PERCENTILE_CONT(0.95) WITHIN GROUP (
                                ORDER BY likes + 3 * completions)
                       FROM recommendations.item_stats) AS p95_engagement
                """)
        return dict(row) if row else {}


def _is_timeout(exc: Exception) -> bool:
    """Recognize a statement timeout without importing asyncpg's exception tree
    into every caller."""
    name = type(exc).__name__
    return "QueryCanceled" in name or "statement timeout" in str(exc).lower()
