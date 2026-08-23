"""Integration tests for the recommendation service against a real Postgres.

Self-skipping on a missing `RECS_TEST_DATABASE_URL`. That matters because
`pytest.ini` puts no `-m` exclusion in `addopts` and CI runs `pytest tests/`, so
these are collected by default — and with no conftest.py in this repo to hold a
fixture, an env guard is the only convention-compatible way to keep CI green.

A *separate* variable from `RECS_DATABASE_URL` on purpose: these tests write and
delete rows, so pointing them at a real deployment must take a deliberate act,
not an inherited environment.

Bring the target up with:
    docker compose -f recommendation_engine/docker-compose.yml up -d
    RECS_TEST_DATABASE_URL=postgresql://recs:recs@localhost:5433/recs pytest tests/test_rec_integration.py
"""

import os

import pytest

TEST_DSN = os.getenv("RECS_TEST_DATABASE_URL", "")

# Point the app at the test database and give it a deterministic offline
# embedder, before importing the server module (which builds the app at import).
if TEST_DSN:
    os.environ["RECS_DATABASE_URL"] = TEST_DSN
    os.environ["RECS_DATABASE_URL_RO"] = ""
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "test-project")
os.environ["USE_MOCK"] = "true"
os.environ["ENVIRONMENT"] = "development"

import asyncpg  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from recommendation_engine.db import Database  # noqa: E402
from recommendation_engine.migrations import migrate  # noqa: E402

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not TEST_DSN,
        reason="RECS_TEST_DATABASE_URL not set; start docker-compose and set it",
    ),
]

EMBEDDING_DIM = 768


async def _ensure_schema() -> None:
    await migrate.run(TEST_DSN)


# ── Migrations ───────────────────────────────────────────────────────────


async def test_migrations_apply_then_are_a_noop():
    await _ensure_schema()
    # Second run must apply nothing. A migration runner that re-applies is how
    # you get duplicate seed rows and failed CREATEs on every boot.
    assert await migrate.run(TEST_DSN) == 0


async def test_schema_migrations_records_every_file():
    await _ensure_schema()
    discovered = {m.version for m in migrate.discover_migrations()}

    conn = await asyncpg.connect(TEST_DSN)
    try:
        rows = await conn.fetch("SELECT version FROM recommendations.schema_migrations")
    finally:
        await conn.close()

    assert {r["version"] for r in rows} == discovered


async def test_modified_applied_migration_is_rejected():
    """git and the database disagreeing is a real bug: every later environment
    would diverge. The runner must refuse rather than guess."""
    await _ensure_schema()
    conn = await asyncpg.connect(TEST_DSN)
    try:
        original = await conn.fetchval(
            "SELECT checksum FROM recommendations.schema_migrations WHERE version = 1"
        )
        await conn.execute(
            "UPDATE recommendations.schema_migrations SET checksum = $1 WHERE version = 1",
            "0" * 64,
        )
        with pytest.raises(RuntimeError, match="modified after it was applied"):
            await migrate.run(TEST_DSN)
    finally:
        # Restore so the rest of the suite sees a clean schema.
        await conn.execute(
            "UPDATE recommendations.schema_migrations SET checksum = $1 WHERE version = 1",
            original,
        )
        await conn.close()


async def test_seeded_config_knobs_are_present():
    await _ensure_schema()
    db = Database(write_dsn=TEST_DSN, read_dsn=TEST_DSN)
    await db.connect()
    try:
        config = await db.load_config()
    finally:
        await db.aclose()

    # The scoring formula reads every one of these; a missing key is a KeyError
    # at request time.
    for key in (
        "w_pop_ceiling",
        "w_cf_ceiling",
        "ramp_n_min",
        "ramp_n50",
        "bayes_prior_c",
        "cf_shrinkage_lambda",
        "rrf_k",
        "mmr_lambda",
        "cmu_target_catalog",
        "cmu_weight_floor",
    ):
        assert key in config, f"missing scoring knob {key}"

    # CF ships inert until an event log exists.
    assert config["w_cf_ceiling"] == 0.0
    # Semantic must remain the ranker: the behavioral ceilings cannot sum to
    # more than 0.45, or vector similarity stops dominating.
    assert config["w_pop_ceiling"] + config["w_cf_ceiling"] <= 0.45


# ── db layer against a real server ───────────────────────────────────────


async def test_health_reports_pgvector_and_index():
    await _ensure_schema()
    db = Database(write_dsn=TEST_DSN, read_dsn=TEST_DSN)
    await db.connect()
    try:
        report = await db.health()
    finally:
        await db.aclose()

    assert report["connected"] is True
    assert report["pgvector_ok"] is True, (
        f"pgvector {report['pgvector_version']} is below 0.8.0; "
        "iterative index scans are unavailable"
    )
    assert report["hnsw_index_present"] is True
    assert report["schema_version"] >= 1


async def test_vector_query_applies_set_local_and_scopes_it_to_the_txn():
    """The whole point of SET LOCAL: the setting must be live inside the
    transaction and gone afterwards, so a pooled connection can never leak a
    tuned ef_search to an unrelated caller."""
    await _ensure_schema()
    db = Database(write_dsn=TEST_DSN, read_dsn=TEST_DSN)
    await db.connect()
    try:
        async with db.vector_query(
            ef_search=137, max_scan_tuples=4242, statement_timeout_ms=1500
        ) as conn:
            assert await conn.fetchval("SHOW hnsw.ef_search") == "137"
            assert await conn.fetchval("SHOW hnsw.iterative_scan") == "relaxed_order"
            assert await conn.fetchval("SHOW hnsw.max_scan_tuples") == "4242"

        # Same pool, fresh transaction: the tuned value must not have survived.
        async with db.read_pool.acquire() as conn:
            assert await conn.fetchval("SHOW hnsw.ef_search") != "137"
    finally:
        await db.aclose()


async def test_invalid_iterative_scan_mode_is_refused_before_reaching_postgres():
    await _ensure_schema()
    db = Database(write_dsn=TEST_DSN, read_dsn=TEST_DSN)
    await db.connect()
    try:
        with pytest.raises(ValueError, match="invalid hnsw.iterative_scan mode"):
            async with db.vector_query(
                ef_search=100,
                max_scan_tuples=1000,
                statement_timeout_ms=1000,
                iterative_scan="off; DROP TABLE recommendations.items",
            ):
                pass
    finally:
        await db.aclose()


async def test_vector_roundtrip_and_cosine_ordering():
    """Proves the pgvector codec is registered and cosine ordering works — if
    `register_vector` were missing, the insert would fail or store a string."""
    await _ensure_schema()
    db = Database(write_dsn=TEST_DSN, read_dsn=TEST_DSN)
    await db.connect()

    # Three orthogonal-ish unit vectors so the expected ordering is unambiguous.
    def unit(index: int) -> list:
        vec = [0.0] * EMBEDDING_DIM
        vec[index] = 1.0
        return vec

    fixtures = [("near", unit(0)), ("mid", unit(1)), ("far", unit(2))]
    try:
        async with db.write_pool.acquire() as conn:
            for name, vec in fixtures:
                await conn.execute(
                    "INSERT INTO recommendations.items "
                    "(source, source_id, title, embed_input, embed_input_sha, "
                    " embedding, embed_task_type) "
                    "VALUES ('cmu', $1, $2, 'x', $3, $4, 'RETRIEVAL_DOCUMENT') "
                    "ON CONFLICT (source, source_id) DO UPDATE SET embedding = $4",
                    f"__test__{name}",
                    name,
                    f"sha-{name}",
                    vec,
                )

            # Query vector closest to "near".
            query = [0.0] * EMBEDDING_DIM
            query[0] = 0.9
            query[1] = 0.4
            rows = await conn.fetch(
                "SELECT title, 1 - (embedding <=> $1) AS sem_cos "
                "FROM recommendations.items "
                "WHERE source_id LIKE '__test__%' "
                "ORDER BY embedding <=> $1",
                query,
            )

            titles = [r["title"] for r in rows]
            assert titles == ["near", "mid", "far"], titles
            # Cosine similarity must be a real number in range, not a string.
            assert 0.0 < rows[0]["sem_cos"] <= 1.0
            assert rows[0]["sem_cos"] > rows[1]["sem_cos"] > rows[2]["sem_cos"]
    finally:
        async with db.write_pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM recommendations.items WHERE source_id LIKE '__test__%'"
            )
        await db.aclose()


# ── The app itself ───────────────────────────────────────────────────────


def test_health_endpoint_is_green_against_a_migrated_database():
    import asyncio

    asyncio.run(_ensure_schema())

    from recommendation_engine.server import create_app

    # TestClient as a context manager runs the lifespan, which is what opens the
    # pool — without it /health would report a disconnected database.
    with TestClient(create_app()) as client:
        response = client.get("/health")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "ok"
    assert body["embedding_dimension"] == EMBEDDING_DIM
    assert body["embedding_dimension_ok"] is True
    assert body["database"]["pgvector_ok"] is True
    assert body["database"]["hnsw_index_present"] is True


def test_health_reports_degraded_when_the_database_is_unreachable():
    """A 503 here is what makes Cloud Run's startup probe fail a broken deploy
    instead of serving empty recommendations."""
    from recommendation_engine.server import create_app

    app = create_app()
    # Swap in a pool-less Database so health() raises internally and is caught.
    app.state.db = Database(
        write_dsn="postgresql://127.0.0.1:1/none",
        read_dsn="postgresql://127.0.0.1:1/none",
    )

    # Call the route directly — entering the lifespan would fail on connect(),
    # which is correct behaviour but not what this test is about.
    from fastapi.testclient import TestClient as _TC

    client = _TC(app)
    response = client.get("/health")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["database"]["connected"] is False
