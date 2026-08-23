"""Integration tests for the retrieval SQL against a real pgvector database.

Self-skipping on a missing `RECS_TEST_DATABASE_URL` — see the note in
test_rec_integration.py for why that guard is required in this repo.

Seeds its own fixtures under a `__ret__` source_id prefix and removes them
afterwards, so it does not depend on a backfill having run and cannot disturb
loaded catalog data.
"""

import os

import pytest

TEST_DSN = os.getenv("RECS_TEST_DATABASE_URL", "")

if TEST_DSN:
    os.environ["RECS_DATABASE_URL"] = TEST_DSN
    os.environ["RECS_DATABASE_URL_RO"] = ""
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "test-project")
os.environ["USE_MOCK"] = "true"
os.environ["ENVIRONMENT"] = "development"

from recommendation_engine.db import Database  # noqa: E402
from recommendation_engine.fusion import merge_candidate_records  # noqa: E402
from recommendation_engine.migrations import migrate  # noqa: E402
from recommendation_engine.mmr import diversify  # noqa: E402
from recommendation_engine.retrieval import (  # noqa: E402
    RetrievalFilters,
    Retriever,
    _is_timeout,
)
from recommendation_engine.scoring import (  # noqa: E402
    CatalogStats,
    ScoringConfig,
    blend,
    source_weight,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not TEST_DSN, reason="RECS_TEST_DATABASE_URL not set; start docker-compose"
    ),
]

DIM = 768
PREFIX = "__ret__"


def vec(components: dict) -> list:
    """Build an L2-normalized vector from {dimension: weight}."""
    vector = [0.0] * DIM
    for index, weight in components.items():
        vector[index] = float(weight)
    norm = sum(value * value for value in vector) ** 0.5
    return [value / norm for value in vector] if norm else vector


def unit(index: int) -> list:
    return vec({index: 1.0})


# Every fixture shares a dominant component in dimension 0.
#
# This matters for isolation, not realism. The test database may already hold a
# loaded catalog whose mock embeddings are *dense* — 768 components in [0,1] —
# which gives every one of them a small but positive cosine (~0.04) against any
# sparse query. Strictly orthogonal fixtures would score 0.0 and be pushed out of
# the result window by hundreds of those rows. Sharing dimension 0 puts all
# fixtures at cosine >= 0.96 for QUERY_A, far above the noise floor, while the
# secondary components keep their ordering and their MMR distances distinct.
QUERY_A = unit(0)
# Points into rom1's secondary dimension, so a second seed favours a genuinely
# different neighbourhood — which is what the fusion test needs to be meaningful.
QUERY_B = vec({0: 0.5, 3: 0.866})

FIXTURES = [
    # (suffix, title, author, genres, themes, word_count, year, source, vector)
    (
        "sf1",
        "Desert Prophecy",
        "F. Herbert",
        ["science-fiction"],
        ["prophecy"],
        100000,
        1965,
        "cmu",
        vec({0: 1.0}),
    ),
    (
        "sf2",
        "Sand Messiah",
        "F. Herbert",
        ["science-fiction"],
        ["prophecy"],
        90000,
        1969,
        "cmu",
        vec({0: 0.99, 1: 0.141}),
    ),
    (
        "sf3",
        "Ringworld Echo",
        "L. Niven",
        ["science-fiction"],
        ["space colonisation"],
        80000,
        1970,
        "cmu",
        vec({0: 0.98, 2: 0.199}),
    ),
    (
        "rom1",
        "Pemberley Letters",
        "J. Austen",
        ["romance", "historical-fiction"],
        ["slow burn romance"],
        40000,
        1813,
        "cmu",
        vec({0: 0.97, 3: 0.243}),
    ),
    (
        "plat1",
        "A Platform Story",
        "Local Author",
        ["fantasy"],
        ["magic system"],
        5000,
        2026,
        "platform",
        vec({0: 0.96, 4: 0.280}),
    ),
]


async def _ensure_schema() -> None:
    await migrate.run(TEST_DSN)


async def _seed(db: Database) -> dict:
    """Insert fixtures, returning suffix -> id."""
    ids = {}
    async with db.write_pool.acquire() as conn:
        for (
            suffix,
            title,
            author,
            genres,
            themes,
            word_count,
            year,
            source,
            vector,
        ) in FIXTURES:
            item_id = await conn.fetchval(
                """
                INSERT INTO recommendations.items (
                    source, source_id, title, author, genres, themes,
                    word_count, published_year, is_eligible,
                    embed_input, embed_input_sha, embedding, embed_task_type
                ) VALUES (
                    $1::recommendations.item_source, $2, $3, $4, $5::text[], $6::text[],
                    $7, $8, true, 'x', $9, $10::vector, 'RETRIEVAL_DOCUMENT'
                )
                ON CONFLICT (source, source_id) DO UPDATE SET
                    embedding = EXCLUDED.embedding, genres = EXCLUDED.genres
                RETURNING id
                """,
                source,
                PREFIX + suffix,
                title,
                author,
                genres,
                themes,
                word_count,
                year,
                f"sha-{suffix}",
                vector,
            )
            ids[suffix] = item_id
    return ids


async def _cleanup(db: Database) -> None:
    async with db.write_pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM recommendations.items WHERE source_id LIKE $1", PREFIX + "%"
        )


async def _fixture_db():
    await _ensure_schema()
    db = Database(write_dsn=TEST_DSN, read_dsn=TEST_DSN)
    await db.connect()
    await _cleanup(db)
    ids = await _seed(db)
    return db, ids


def _retriever(db: Database, ef_search: int = 100) -> Retriever:
    return Retriever(
        db, ef_search=ef_search, max_scan_tuples=20000, statement_timeout_ms=5000
    )


def _only_fixtures(rows, ids):
    """Filter to seeded rows, so a loaded catalog does not affect assertions."""
    wanted = set(ids.values())
    return [row for row in rows if row["id"] in wanted]


# ══════════════════════════════════════════════════════════════════════════
# Basic KNN
# ══════════════════════════════════════════════════════════════════════════


async def test_knn_orders_by_cosine_similarity():
    db, ids = await _fixture_db()
    try:
        result = await _retriever(db).knn(QUERY_A, limit=50)
        rows = _only_fixtures(result.rows, ids)

        assert not result.degraded
        # The two vectors at unit(0) must come first.
        assert {rows[0]["id"], rows[1]["id"]} == {ids["sf1"], ids["sf2"]}
        similarities = [row["sem_cos"] for row in rows]
        assert similarities == sorted(similarities, reverse=True)
    finally:
        await _cleanup(db)
        await db.aclose()


async def test_knn_returns_the_embedding_for_mmr():
    """Fetched in the same round trip; re-reading it would cost a second query."""
    db, ids = await _fixture_db()
    try:
        rows = _only_fixtures((await _retriever(db).knn(QUERY_A, 50)).rows, ids)

        assert rows[0]["embedding"] is not None
        assert len(list(rows[0]["embedding"])) == DIM
    finally:
        await _cleanup(db)
        await db.aclose()


async def test_knn_respects_the_limit():
    db, ids = await _fixture_db()
    try:
        result = await _retriever(db).knn(QUERY_A, limit=2)

        assert len(result.rows) == 2
    finally:
        await _cleanup(db)
        await db.aclose()


async def test_knn_joins_stats_and_defaults_them_to_zero():
    """Items with no item_stats row must score as cold-start, not crash."""
    db, ids = await _fixture_db()
    try:
        rows = _only_fixtures((await _retriever(db).knn(QUERY_A, 50)).rows, ids)

        assert all(row["n_interactions"] == 0 for row in rows)
        assert all(row["pop_score"] == 0 for row in rows)
    finally:
        await _cleanup(db)
        await db.aclose()


async def test_ineligible_items_are_never_returned():
    """is_eligible gates the partial HNSW index, so an ineligible row is invisible
    to retrieval by construction."""
    db, ids = await _fixture_db()
    try:
        async with db.write_pool.acquire() as conn:
            await conn.execute(
                "UPDATE recommendations.items SET is_eligible = false WHERE id = $1",
                ids["sf1"],
            )

        rows = _only_fixtures((await _retriever(db).knn(QUERY_A, 50)).rows, ids)

        assert ids["sf1"] not in {row["id"] for row in rows}
    finally:
        await _cleanup(db)
        await db.aclose()


# ══════════════════════════════════════════════════════════════════════════
# Metadata filters
# ══════════════════════════════════════════════════════════════════════════


async def test_genre_filter():
    db, ids = await _fixture_db()
    try:
        result = await _retriever(db).knn(
            QUERY_A, 50, RetrievalFilters(genres=["romance"])
        )
        rows = _only_fixtures(result.rows, ids)

        assert {row["id"] for row in rows} == {ids["rom1"]}
    finally:
        await _cleanup(db)
        await db.aclose()


async def test_genre_filter_matches_any_of_several():
    db, ids = await _fixture_db()
    try:
        result = await _retriever(db).knn(
            QUERY_A, 50, RetrievalFilters(genres=["romance", "fantasy"])
        )
        rows = _only_fixtures(result.rows, ids)

        assert {row["id"] for row in rows} == {ids["rom1"], ids["plat1"]}
    finally:
        await _cleanup(db)
        await db.aclose()


async def test_theme_filter():
    db, ids = await _fixture_db()
    try:
        result = await _retriever(db).knn(
            QUERY_A, 50, RetrievalFilters(themes=["prophecy"])
        )
        rows = _only_fixtures(result.rows, ids)

        assert {row["id"] for row in rows} == {ids["sf1"], ids["sf2"]}
    finally:
        await _cleanup(db)
        await db.aclose()


async def test_source_filter_isolates_platform_items():
    """The behavioral path uses this: a CMU book is a dead end for a reader."""
    db, ids = await _fixture_db()
    try:
        result = await _retriever(db).knn(
            QUERY_A, 50, RetrievalFilters(sources=["platform"])
        )
        rows = _only_fixtures(result.rows, ids)

        assert {row["id"] for row in rows} == {ids["plat1"]}
    finally:
        await _cleanup(db)
        await db.aclose()


async def test_word_count_filters():
    db, ids = await _fixture_db()
    try:
        short = await _retriever(db).knn(
            QUERY_A, 50, RetrievalFilters(max_word_count=10000)
        )
        long = await _retriever(db).knn(
            QUERY_A, 50, RetrievalFilters(min_word_count=95000)
        )

        assert {r["id"] for r in _only_fixtures(short.rows, ids)} == {ids["plat1"]}
        assert {r["id"] for r in _only_fixtures(long.rows, ids)} == {ids["sf1"]}
    finally:
        await _cleanup(db)
        await db.aclose()


async def test_author_filter_is_case_insensitive_and_partial():
    db, ids = await _fixture_db()
    try:
        result = await _retriever(db).knn(
            QUERY_A, 50, RetrievalFilters(author="herbert")
        )
        rows = _only_fixtures(result.rows, ids)

        assert {row["id"] for row in rows} == {ids["sf1"], ids["sf2"]}
    finally:
        await _cleanup(db)
        await db.aclose()


async def test_published_after_filter():
    db, ids = await _fixture_db()
    try:
        result = await _retriever(db).knn(
            QUERY_A, 50, RetrievalFilters(published_after=2000)
        )
        rows = _only_fixtures(result.rows, ids)

        assert {row["id"] for row in rows} == {ids["plat1"]}
    finally:
        await _cleanup(db)
        await db.aclose()


async def test_exclude_ids_removes_already_read_items():
    """Never recommend what the reader has already finished."""
    db, ids = await _fixture_db()
    try:
        result = await _retriever(db).knn(
            QUERY_A, 50, RetrievalFilters(exclude_ids=[ids["sf1"], ids["sf2"]])
        )
        rows = _only_fixtures(result.rows, ids)

        assert ids["sf1"] not in {row["id"] for row in rows}
        assert ids["sf2"] not in {row["id"] for row in rows}
    finally:
        await _cleanup(db)
        await db.aclose()


async def test_combined_filters_are_conjunctive():
    db, ids = await _fixture_db()
    try:
        result = await _retriever(db).knn(
            QUERY_A,
            50,
            RetrievalFilters(genres=["science-fiction"], author="herbert"),
        )
        rows = _only_fixtures(result.rows, ids)

        assert {row["id"] for row in rows} == {ids["sf1"], ids["sf2"]}
    finally:
        await _cleanup(db)
        await db.aclose()


async def test_no_filters_returns_everything_eligible():
    db, ids = await _fixture_db()
    try:
        rows = _only_fixtures((await _retriever(db).knn(QUERY_A, 50)).rows, ids)

        assert len(rows) == len(FIXTURES)
    finally:
        await _cleanup(db)
        await db.aclose()


# ══════════════════════════════════════════════════════════════════════════
# Multi-item retrieval + fusion + MMR, end to end
# ══════════════════════════════════════════════════════════════════════════


async def test_knn_many_runs_one_query_per_vector():
    db, ids = await _fixture_db()
    try:
        results = await _retriever(db).knn_many([QUERY_A, QUERY_B], limit=50)

        assert len(results) == 2
        assert all(not r.degraded for r in results)
    finally:
        await _cleanup(db)
        await db.aclose()


async def test_multi_item_query_fuses_without_averaging_vectors():
    """The core multi-book behaviour: retrieving per seed and fusing by rank
    surfaces both neighbourhoods, where an averaged vector would point at the
    midpoint between them and match neither."""
    db, ids = await _fixture_db()
    try:
        results = await _retriever(db).knn_many([QUERY_A, QUERY_B], limit=50)
        merged = merge_candidate_records([_only_fixtures(r.rows, ids) for r in results])
        top_two = {row["id"] for row in merged[:2]}

        # One from each seed's neighbourhood.
        assert ids["rom1"] in top_two
        assert top_two & {ids["sf1"], ids["sf2"]}
    finally:
        await _cleanup(db)
        await db.aclose()


async def test_full_pipeline_scores_and_diversifies():
    """retrieve -> fuse -> blend -> MMR, the actual request path."""
    db, ids = await _fixture_db()
    try:
        results = await _retriever(db).knn_many([QUERY_A, QUERY_B], limit=50)
        merged = merge_candidate_records([_only_fixtures(r.rows, ids) for r in results])

        config = ScoringConfig()
        stats = CatalogStats(platform_item_count=1)
        for row in merged:
            breakdown = blend(
                semantic=row["semantic"],
                popularity_score=float(row["pop_score"]),
                collaborative=0.0,
                n_interactions=row["n_interactions"],
                source=row["source"],
                platform_item_count=stats.platform_item_count,
                config=config,
            )
            row["score"] = breakdown.score

        final = diversify(merged, k=3, lambda_=config.mmr_lambda)

        assert len(final) == 3
        assert all(0.0 <= row["score"] <= 1.0 for row in final)
        # Every fixture has zero interactions, so alpha is 0 and the score is
        # exactly semantic x source_weight — no popularity, no CF.
        top = final[0]
        expected_src = source_weight(top["source"], stats.platform_item_count, config)
        assert top["score"] == pytest.approx(top["semantic"] * expected_src, abs=1e-6)
    finally:
        await _cleanup(db)
        await db.aclose()


async def test_cmu_items_are_down_weighted_relative_to_platform():
    """Same semantic score, different source: the platform item must win."""
    config = ScoringConfig()

    cmu = blend(0.9, 0.0, 0.0, 0, "cmu", 2500, config)
    platform = blend(0.9, 0.0, 0.0, 0, "platform", 2500, config)

    assert platform.score > cmu.score


# ══════════════════════════════════════════════════════════════════════════
# Title resolution and fallbacks
# ══════════════════════════════════════════════════════════════════════════


async def test_resolve_titles_matches_fuzzily():
    """Readers misremember titles and skip subtitles, so exact match is useless."""
    db, ids = await _fixture_db()
    try:
        rows = await _retriever(db).resolve_titles("desert prophecy")

        assert rows
        assert rows[0]["id"] == ids["sf1"]
    finally:
        await _cleanup(db)
        await db.aclose()


async def test_resolve_titles_tolerates_a_typo():
    db, ids = await _fixture_db()
    try:
        rows = await _retriever(db).resolve_titles("Desert Propecy")

        assert any(row["id"] == ids["sf1"] for row in rows)
    finally:
        await _cleanup(db)
        await db.aclose()


async def test_resolve_titles_can_match_on_author():
    db, ids = await _fixture_db()
    try:
        rows = await _retriever(db).resolve_titles("unknown book", author="F. Herbert")

        assert {ids["sf1"], ids["sf2"]} & {row["id"] for row in rows}
    finally:
        await _cleanup(db)
        await db.aclose()


async def test_resolve_titles_of_nothing_returns_nothing():
    db, _ = await _fixture_db()
    try:
        assert await _retriever(db).resolve_titles("") == []
        assert await _retriever(db).resolve_titles("   ") == []
    finally:
        await _cleanup(db)
        await db.aclose()


async def test_resolve_titles_respects_the_limit():
    db, ids = await _fixture_db()
    try:
        rows = await _retriever(db).resolve_titles("F. Herbert", limit=1)

        assert len(rows) <= 1
    finally:
        await _cleanup(db)
        await db.aclose()


async def test_popular_fallback_needs_no_query_vector():
    """Reached by a reader with no history, or when no embedder is configured.

    The limit is deliberately large. `popular()` orders by `pop_score DESC`, and
    once real interaction data exists the fixtures — which have no stats row, so
    score 0 — sit at the bottom. The claim under test is "returns results with no
    query vector", not "fixtures rank first", so the window has to be wide enough to
    contain them regardless of what else is loaded.
    """
    db, ids = await _fixture_db()
    try:
        rows = _only_fixtures(await _retriever(db).popular(limit=100_000), ids)

        assert len(rows) == len(FIXTURES)
    finally:
        await _cleanup(db)
        await db.aclose()


async def test_popular_fallback_applies_filters():
    db, ids = await _fixture_db()
    try:
        rows = _only_fixtures(
            await _retriever(db).popular(
                limit=50, filters=RetrievalFilters(sources=["platform"])
            ),
            ids,
        )

        assert {row["id"] for row in rows} == {ids["plat1"]}
    finally:
        await _cleanup(db)
        await db.aclose()


async def test_catalog_stats_reports_what_scoring_needs():
    db, ids = await _fixture_db()
    try:
        stats = CatalogStats.from_row(await _retriever(db).catalog_stats())

        # One platform fixture was seeded (plus anything already loaded).
        assert stats.platform_item_count >= 1
        assert 1 <= stats.global_mean_rating <= 5
        assert stats.p95_engagement > 0
    finally:
        await _cleanup(db)
        await db.aclose()


# ══════════════════════════════════════════════════════════════════════════
# Degradation
# ══════════════════════════════════════════════════════════════════════════


async def test_statement_timeout_degrades_instead_of_erroring():
    """A partial shelf flagged `degraded` beats a 500 for a reader. Forced with a
    1ms timeout, which no real query can beat."""
    db, ids = await _fixture_db()
    try:
        retriever = Retriever(
            db, ef_search=100, max_scan_tuples=20000, statement_timeout_ms=1
        )
        result = await retriever.knn(QUERY_A, limit=50)

        # Either it degraded cleanly, or the query genuinely finished inside 1ms
        # on a tiny table — both are acceptable; a raised exception is not.
        if result.degraded:
            assert result.reason == "timeout"
            assert result.rows == []
    finally:
        await _cleanup(db)
        await db.aclose()


def test_timeout_classification():
    import asyncpg

    assert _is_timeout(asyncpg.exceptions.QueryCanceledError("canceling statement"))
    assert _is_timeout(Exception("canceling statement due to statement timeout"))
    assert not _is_timeout(ValueError("something else"))
