"""Postgres access layer: connection pools, vector codecs, and the one
transaction helper every KNN query runs inside.

Two pools, because reads and writes have different destinations in production:
serving hits a dedicated read-only compute so vector scans never share compute
with the billing workload, while ingest and sync hit the primary. In local dev
both DSNs are identical and a single pool is shared.
"""

import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Optional

import asyncpg
from pgvector.asyncpg import register_vector

logger = logging.getLogger(__name__)

# pgvector 0.8.0 introduced hnsw.iterative_scan, which is what keeps recall
# usable when a metadata filter runs alongside the KNN. Below that the service
# would still answer, but filtered queries would quietly under-return.
MIN_PGVECTOR_VERSION = (0, 8, 0)

HNSW_INDEX_NAME = "items_embedding_hnsw_idx"


def _parse_version(raw: str) -> tuple:
    """'0.8.1' -> (0, 8, 1). Trailing non-numeric parts are dropped."""
    parts: list = []
    for chunk in raw.split("."):
        digits = ""
        for char in chunk:
            if not char.isdigit():
                break
            digits += char
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


async def _init_connection(conn: asyncpg.Connection) -> None:
    """Per-connection setup: teach asyncpg the `vector` type."""
    await register_vector(conn)


class Database:
    """Owns the pools. Created in the app lifespan, closed on shutdown."""

    def __init__(
        self,
        write_dsn: str,
        read_dsn: str,
        pool_min: int = 1,
        pool_max: int = 10,
        statement_cache_size: int = 0,
    ) -> None:
        self._write_dsn = write_dsn
        self._read_dsn = read_dsn
        self._pool_min = pool_min
        self._pool_max = pool_max
        # Zero disables asyncpg's prepared-statement cache. Required whenever the
        # DSN points at a transaction-mode pooler (Neon's pooled endpoint,
        # PgBouncer): the pooler hands a different backend to each transaction,
        # so a cached prepared statement name eventually collides and raises
        # `prepared statement "__asyncpg_stmt_x__" already exists` — a failure
        # that only appears under concurrency, i.e. never in local testing.
        self._statement_cache_size = statement_cache_size
        self._write_pool: Optional[asyncpg.Pool] = None
        self._read_pool: Optional[asyncpg.Pool] = None
        self._shared = write_dsn == read_dsn

    async def connect(self) -> None:
        self._write_pool = await asyncpg.create_pool(
            self._write_dsn,
            min_size=self._pool_min,
            max_size=self._pool_max,
            init=_init_connection,
            statement_cache_size=self._statement_cache_size,
        )
        if self._shared:
            # One compute in local dev — a second pool to the same DSN would
            # just double idle connections.
            self._read_pool = self._write_pool
            logger.info("db_pools_ready shared=true max_size=%d", self._pool_max)
        else:
            self._read_pool = await asyncpg.create_pool(
                self._read_dsn,
                min_size=self._pool_min,
                max_size=self._pool_max,
                init=_init_connection,
                statement_cache_size=self._statement_cache_size,
            )
            logger.info("db_pools_ready shared=false max_size=%d", self._pool_max)

    async def aclose(self) -> None:
        if self._read_pool is not None and not self._shared:
            await self._read_pool.close()
        if self._write_pool is not None:
            await self._write_pool.close()
        self._read_pool = None
        self._write_pool = None

    @property
    def read_pool(self) -> asyncpg.Pool:
        if self._read_pool is None:
            raise RuntimeError("Database.connect() has not run")
        return self._read_pool

    @property
    def write_pool(self) -> asyncpg.Pool:
        if self._write_pool is None:
            raise RuntimeError("Database.connect() has not run")
        return self._write_pool

    # ------------------------------------------------------------------
    # The retrieval transaction
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def vector_query(
        self,
        ef_search: int,
        max_scan_tuples: int,
        statement_timeout_ms: int,
        iterative_scan: str = "relaxed_order",
    ) -> AsyncIterator[asyncpg.Connection]:
        """Yield a connection inside a transaction with HNSW query-time settings
        applied via SET LOCAL.

        SET LOCAL, never a session-level SET. Under a transaction-mode pooler a
        session setting is discarded or leaks to an unrelated caller, so the
        query would silently run at the default ef_search=40 and return worse
        neighbours with no error to notice. Scoping it to the transaction is
        correct everywhere and costs nothing.

        `relaxed_order` is free here: results are re-scored and MMR-diversified
        downstream, so approximate ordering out of the index changes nothing.
        `statement_timeout` bounds the worst case — under a selective filter an
        iterative scan can run long, and a bounded query that reports itself as
        degraded beats an unbounded one that times out at the load balancer.
        """
        async with self.read_pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(f"SET LOCAL hnsw.ef_search = {int(ef_search)}")
                await conn.execute(
                    f"SET LOCAL hnsw.iterative_scan = {_safe_ident(iterative_scan)}"
                )
                await conn.execute(
                    f"SET LOCAL hnsw.max_scan_tuples = {int(max_scan_tuples)}"
                )
                await conn.execute(
                    f"SET LOCAL statement_timeout = {int(statement_timeout_ms)}"
                )
                yield conn

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    async def health(self) -> dict:
        """Report everything that must be true for retrieval to work at all.

        A missing extension, a too-old pgvector, or a missing HNSW index all
        produce *silently* worse results rather than errors, so each is asserted
        explicitly here instead of being discovered by a user.

        `schema_present` is the check that replaced a schema version: this
        service no longer migrates anything — story-data owns the
        `recommendations` schema — so the failure it has to catch is being
        pointed at a database that has not been migrated yet, which is a
        startup ordering mistake rather than a version skew.

        `catalog_embed_model` is reported so the caller can compare it to the
        *serving* embedder. The ingest already refuses to reuse a vector from a
        different provider, but nothing stopped the query side from drifting:
        embedding a query with one model and searching a catalog built by
        another returns plausible-looking nonsense, ranked confidently, with no
        error and a green health check.
        """
        report: dict = {
            "connected": False,
            "pgvector_version": None,
            "pgvector_ok": False,
            "hnsw_index_present": False,
            "schema_present": False,
            "catalog_embed_model": None,
            "item_count": None,
            "eligible_count": None,
        }
        async with self.read_pool.acquire() as conn:
            report["connected"] = True

            raw_version = await conn.fetchval(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            )
            report["pgvector_version"] = raw_version
            if raw_version:
                report["pgvector_ok"] = (
                    _parse_version(raw_version) >= MIN_PGVECTOR_VERSION
                )

            report["hnsw_index_present"] = bool(
                await conn.fetchval(
                    "SELECT 1 FROM pg_indexes WHERE schemaname = 'recommendations' "
                    "AND indexname = $1",
                    HNSW_INDEX_NAME,
                )
            )

            report["schema_present"] = bool(
                await conn.fetchval(
                    "SELECT 1 FROM information_schema.tables WHERE table_schema = "
                    "'recommendations' AND table_name = 'items'"
                )
            )
            if report["schema_present"]:
                counts = await conn.fetchrow(
                    "SELECT COUNT(*) AS total, "
                    "COUNT(*) FILTER (WHERE is_eligible AND embedding IS NOT NULL) "
                    "AS eligible FROM recommendations.items"
                )
                report["item_count"] = counts["total"]
                report["eligible_count"] = counts["eligible"]
                # The dominant model among rows that can actually be retrieved.
                # A tie or a mixed catalog is itself a problem, and reporting the
                # majority is enough to surface it: whichever the serving
                # embedder is, it will disagree with something.
                report["catalog_embed_model"] = await conn.fetchval(
                    "SELECT embed_model FROM recommendations.items "
                    "WHERE is_eligible AND embedding IS NOT NULL "
                    "AND embed_model IS NOT NULL "
                    "GROUP BY embed_model ORDER BY count(*) DESC LIMIT 1"
                )

        return report

    # ------------------------------------------------------------------
    # Small conveniences used by ingest and scoring
    # ------------------------------------------------------------------

    async def load_config(self) -> dict[str, float]:
        """Read the scoring knobs. Kept in Postgres so ranking can be retuned
        without a redeploy."""
        rows = await self.read_pool.fetch(
            "SELECT key, value FROM recommendations.config"
        )
        return {r["key"]: float(r["value"]) for r in rows}

    async def fetch_read(self, sql: str, *args: object) -> Sequence[asyncpg.Record]:
        return await self.read_pool.fetch(sql, *args)

    async def execute_write(self, sql: str, *args: object) -> str:
        return await self.write_pool.execute(sql, *args)


# hnsw.iterative_scan takes an enum, not a string literal, so it cannot be
# parameterized — hence an allowlist rather than string interpolation of input.
_ITERATIVE_SCAN_MODES = frozenset({"off", "relaxed_order", "strict_order"})


def _safe_ident(mode: str) -> str:
    if mode not in _ITERATIVE_SCAN_MODES:
        raise ValueError(
            f"invalid hnsw.iterative_scan mode {mode!r}; "
            f"expected one of {sorted(_ITERATIVE_SCAN_MODES)}"
        )
    return mode
