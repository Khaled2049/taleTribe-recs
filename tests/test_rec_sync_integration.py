"""Integration tests for the interaction loader and the derived-table refresh.

Self-skipping on a missing `RECS_TEST_DATABASE_URL` — see test_rec_integration.py.

These cover the *shared* half of the pipeline: everything here runs identically
whether the records came from the synthetic generator or from a future Firestore
export, which is the point of the source/loader split.
"""

import math
import os
from collections.abc import Callable, Iterable
from datetime import datetime, timedelta, timezone
from typing import cast

import pytest

TEST_DSN = os.getenv("RECS_TEST_DATABASE_URL", "")

if TEST_DSN:
    os.environ["RECS_DATABASE_URL"] = TEST_DSN
    os.environ["RECS_DATABASE_URL_RO"] = ""
os.environ["USE_MOCK"] = "true"

import asyncpg
from conftest import (
    drop_stories,
    require_recommendations_schema,
    seed_stories,
    story_id_for,
)
from pgvector.asyncpg import register_vector

from recommendation_engine.scoring import ScoringConfig
from recommendation_engine.sync import stats as stats_mod
from recommendation_engine.sync.interactions import (
    KIND_LIKE,
    KIND_PROGRESS,
    KIND_RATING,
    InteractionRecord,
    load,
    purge_synthetic,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not TEST_DSN, reason="RECS_TEST_DATABASE_URL not set; start docker-compose"
    ),
]

DIM = 768
PREFIX = "__sync__"
USER_A = "synth_test_a"
USER_B = "synth_test_b"
NOW = datetime(2026, 6, 1, tzinfo=timezone.utc)
CONFIG = ScoringConfig()


def unit(index: int) -> list:
    v = [0.0] * DIM
    v[index] = 1.0
    return v


async def _connect():
    conn = await asyncpg.connect(TEST_DSN)
    await register_vector(conn)
    return conn


SUFFIXES = ("x", "y", "z")
KEYS = [PREFIX + s for s in SUFFIXES]


async def _setup():
    """Seed three catalog items and clear any prior test state."""
    await require_recommendations_schema(TEST_DSN)
    conn = await _connect()
    ids = {}
    try:
        await _clear(conn)
        stories = await seed_stories(conn, KEYS)
        for suffix, index in zip(SUFFIXES, range(3)):
            ids[suffix] = await conn.fetchval(
                """
                INSERT INTO recommendations.items
                    (story_id, title, is_eligible, embed_input,
                     embed_input_sha, embedding, embed_task_type)
                VALUES ($1::uuid, $2, true, 'x', $3, $4::vector,
                        'RETRIEVAL_DOCUMENT')
                RETURNING id
                """,
                stories[PREFIX + suffix],
                f"Sync Item {suffix.upper()}",
                f"sha-{PREFIX}{suffix}",
                unit(index),
            )
    finally:
        await conn.close()
    return ids


async def _clear(conn=None):
    own = conn is None
    conn = conn or await _connect()
    try:
        await conn.execute(
            "DELETE FROM recommendations.user_taste WHERE user_id LIKE 'synth_test_%'"
        )
        await conn.execute(
            "DELETE FROM recommendations.interactions WHERE user_id LIKE 'synth_test_%'"
        )
        await drop_stories(conn, KEYS)
    finally:
        if own:
            await conn.close()


def _rec(user, suffix, kind, value=None, chapter=None, total=None, offset=0):
    return InteractionRecord(
        user_id=user,
        story_id=story_id_for(PREFIX + suffix),
        kind=kind,
        occurred_at=NOW - timedelta(days=offset),
        value=value,
        chapter_index=chapter,
        total_chapters=total,
    )


# ══════════════════════════════════════════════════════════════════════════
# Loading
# ══════════════════════════════════════════════════════════════════════════


async def test_load_resolves_items_by_external_identity():
    """Sources key on (source, source_id); the internal id is resolved here."""
    ids = await _setup()
    conn = await _connect()
    try:
        stats = await load(conn, [_rec(USER_A, "x", KIND_LIKE)])

        assert stats.written == 1
        assert stats.unknown_item == 0
        row = await conn.fetchrow(
            "SELECT item_id, kind, weight FROM recommendations.interactions "
            "WHERE user_id = $1",
            USER_A,
        )
        assert row["item_id"] == ids["x"]
        assert row["kind"] == KIND_LIKE
        assert row["weight"] == pytest.approx(1.0)
    finally:
        await _clear(conn)
        await conn.close()


async def test_unknown_items_are_counted_not_fatal():
    """A record naming a book the catalog lacks is normal — a Firestore export can
    legitimately reference a story that has not been synced yet."""
    await _setup()
    conn = await _connect()
    try:
        stats = await load(
            conn,
            [
                _rec(USER_A, "x", KIND_LIKE),
                InteractionRecord(
                    user_id=USER_A,
                    story_id=story_id_for("__nonexistent__"),
                    kind=KIND_LIKE,
                    occurred_at=NOW,
                ),
            ],
        )

        assert stats.written == 1
        assert stats.unknown_item == 1
        assert story_id_for("__nonexistent__") in stats.unknown_examples
    finally:
        await _clear(conn)
        await conn.close()


async def test_completion_is_derived_and_stored_alongside_progress():
    """The platform emits no completion event, so it is inferred once, here."""
    await _setup()
    conn = await _connect()
    try:
        stats = await load(
            conn, [_rec(USER_A, "x", KIND_PROGRESS, value=0.97, chapter=11, total=12)]
        )

        assert stats.completions_derived == 1
        kinds = {
            row["kind"]
            for row in await conn.fetch(
                "SELECT kind FROM recommendations.interactions WHERE user_id = $1",
                USER_A,
            )
        }
        # Both rows: how far they got, and that they finished.
        assert kinds == {KIND_PROGRESS, "completion"}
    finally:
        await _clear(conn)
        await conn.close()


async def test_partial_progress_yields_no_completion():
    await _setup()
    conn = await _connect()
    try:
        stats = await load(
            conn, [_rec(USER_A, "x", KIND_PROGRESS, value=0.4, chapter=3, total=12)]
        )

        assert stats.completions_derived == 0
    finally:
        await _clear(conn)
        await conn.close()


async def test_reload_is_idempotent():
    """readingProgress is current *state*, not an event — re-exporting must update,
    not accumulate."""
    await _setup()
    conn = await _connect()
    try:
        records = [_rec(USER_A, "x", KIND_PROGRESS, value=0.4, chapter=3, total=12)]
        await load(conn, records)
        await load(conn, records)

        count = await conn.fetchval(
            "SELECT COUNT(*) FROM recommendations.interactions WHERE user_id = $1",
            USER_A,
        )
        assert count == 1
    finally:
        await _clear(conn)
        await conn.close()


async def test_newer_progress_replaces_older():
    await _setup()
    conn = await _connect()
    try:
        await load(
            conn,
            [
                _rec(
                    USER_A,
                    "x",
                    KIND_PROGRESS,
                    value=0.2,
                    chapter=1,
                    total=12,
                    offset=10,
                )
            ],
        )
        await load(
            conn,
            [
                _rec(
                    USER_A, "x", KIND_PROGRESS, value=0.8, chapter=9, total=12, offset=0
                )
            ],
        )

        value = await conn.fetchval(
            "SELECT value FROM recommendations.interactions "
            "WHERE user_id = $1 AND kind = $2",
            USER_A,
            KIND_PROGRESS,
        )
        assert value == pytest.approx(0.8)
    finally:
        await _clear(conn)
        await conn.close()


async def test_stale_progress_does_not_overwrite_newer():
    """Out-of-order delivery must not rewind a reader's position."""
    await _setup()
    conn = await _connect()
    try:
        await load(
            conn,
            [
                _rec(
                    USER_A, "x", KIND_PROGRESS, value=0.8, chapter=9, total=12, offset=0
                )
            ],
        )
        await load(
            conn,
            [
                _rec(
                    USER_A,
                    "x",
                    KIND_PROGRESS,
                    value=0.2,
                    chapter=1,
                    total=12,
                    offset=30,
                )
            ],
        )

        value = await conn.fetchval(
            "SELECT value FROM recommendations.interactions "
            "WHERE user_id = $1 AND kind = $2",
            USER_A,
            KIND_PROGRESS,
        )
        assert value == pytest.approx(0.8)
    finally:
        await _clear(conn)
        await conn.close()


async def test_low_ratings_are_counted_as_negative_signals():
    await _setup()
    conn = await _connect()
    try:
        stats = await load(
            conn,
            [
                _rec(USER_A, "x", KIND_RATING, value=1.0),
                _rec(USER_A, "y", KIND_RATING, value=5.0),
            ],
        )

        assert stats.negative_signals == 1
    finally:
        await _clear(conn)
        await conn.close()


async def test_invalid_kinds_are_rejected():
    """The format must not accept data the platform cannot produce."""
    await _setup()
    conn = await _connect()
    try:
        stats = await load(
            conn,
            [
                InteractionRecord(
                    user_id=USER_A,
                    story_id=story_id_for(PREFIX + "x"),
                    kind="dwell_time",
                    occurred_at=NOW,
                )
            ],
        )

        assert stats.invalid_kind == 1
        assert stats.written == 0
    finally:
        await _clear(conn)
        await conn.close()


# ══════════════════════════════════════════════════════════════════════════
# item_stats
# ══════════════════════════════════════════════════════════════════════════


async def test_item_stats_aggregates_and_scores():
    ids = await _setup()
    conn = await _connect()
    try:
        await load(
            conn,
            [
                _rec(USER_A, "x", KIND_LIKE),
                _rec(USER_B, "x", KIND_LIKE),
                _rec(USER_A, "x", KIND_RATING, value=5.0),
                _rec(USER_B, "x", KIND_RATING, value=4.0),
                _rec(USER_A, "x", KIND_PROGRESS, value=0.99, chapter=9, total=10),
            ],
        )
        result = await stats_mod.refresh_item_stats(conn, CONFIG)

        assert result.items_updated >= 1
        row = await conn.fetchrow(
            "SELECT * FROM recommendations.item_stats WHERE item_id = $1", ids["x"]
        )
        assert row["likes"] == 2
        assert row["ratings_count"] == 2
        assert row["avg_rating"] == pytest.approx(4.5, abs=0.01)
        assert row["completions"] == 1
        # likes + ratings + completions — the ramp input.
        assert row["n_interactions"] == 5
        assert 0.0 < row["pop_score"] <= 1.0
    finally:
        await _clear(conn)
        await conn.close()


async def test_n_interactions_drives_the_ramp_past_its_gate():
    """The link that makes the popularity term able to contribute at all: without
    n_interactions >= ramp_n_min, alpha is 0 and w_pop is 0 regardless of pop_score."""
    from recommendation_engine.scoring import behavioral_ramp

    ids = await _setup()
    conn = await _connect()
    try:
        records = []
        for index in range(6):
            records.append(_rec(f"synth_test_{index}", "x", KIND_LIKE, offset=index))
        await load(conn, records)
        await stats_mod.refresh_item_stats(conn, CONFIG)

        n = await conn.fetchval(
            "SELECT n_interactions FROM recommendations.item_stats WHERE item_id = $1",
            ids["x"],
        )
        assert n == 6
        assert behavioral_ramp(n, CONFIG) > 0, "6 likes must clear the n_min=5 gate"
    finally:
        await conn.execute(
            "DELETE FROM recommendations.interactions WHERE user_id LIKE 'synth_test_%'"
        )
        await _clear(conn)
        await conn.close()


async def test_progress_alone_does_not_license_behavioral_weighting():
    """A page-turn is too weak a signal to move the ramp."""
    ids = await _setup()
    conn = await _connect()
    try:
        await load(
            conn,
            [
                _rec(USER_A, "x", KIND_PROGRESS, value=0.3, chapter=2, total=12),
                _rec(USER_B, "x", KIND_PROGRESS, value=0.4, chapter=3, total=12),
            ],
        )
        await stats_mod.refresh_item_stats(conn, CONFIG)

        n = await conn.fetchval(
            "SELECT n_interactions FROM recommendations.item_stats WHERE item_id = $1",
            ids["x"],
        )
        assert n == 0
    finally:
        await _clear(conn)
        await conn.close()


# ══════════════════════════════════════════════════════════════════════════
# user_taste
# ══════════════════════════════════════════════════════════════════════════


async def test_taste_vector_is_built_and_normalized():
    ids = await _setup()
    conn = await _connect()
    try:
        await load(
            conn,
            [
                _rec(USER_A, "x", KIND_LIKE),
                _rec(USER_A, "y", KIND_LIKE),
                _rec(USER_A, "z", KIND_LIKE),
                _rec(USER_A, "x", KIND_RATING, value=5.0),
            ],
        )
        result = await stats_mod.rebuild_user_taste(conn)

        assert result.tastes_built >= 1
        row = await conn.fetchrow(
            "SELECT taste_embedding, seed_item_ids, n_signals "
            "FROM recommendations.user_taste WHERE user_id = $1",
            USER_A,
        )
        raw_vector = cast(object, row["taste_embedding"])
        to_list = cast(
            Callable[[], list[float]] | None,
            getattr(raw_vector, "to_list", None),
        )
        vector = (
            to_list()
            if to_list is not None
            else [float(value) for value in cast(Iterable[float], raw_vector)]
        )

        norm = math.sqrt(sum(v * v for v in vector))
        assert norm == pytest.approx(1.0, abs=1e-6), "must be L2-normalized"
        assert set(row["seed_item_ids"]) == {ids["x"], ids["y"], ids["z"]}
    finally:
        await _clear(conn)
        await conn.close()


async def test_thin_readers_get_no_taste_vector():
    """Two data points is not a personality; behavioral mode should fall back."""
    await _setup()
    conn = await _connect()
    try:
        await load(conn, [_rec(USER_B, "x", KIND_LIKE)])
        result = await stats_mod.rebuild_user_taste(conn)

        assert result.tastes_skipped_thin >= 1
        assert (
            await conn.fetchval(
                "SELECT COUNT(*) FROM recommendations.user_taste WHERE user_id = $1",
                USER_B,
            )
            == 0
        )
    finally:
        await _clear(conn)
        await conn.close()


async def test_disliked_books_are_suppressed_not_averaged_in():
    """Averaging in a rejected book would aim recommendations at what they rejected."""
    ids = await _setup()
    conn = await _connect()
    try:
        await load(
            conn,
            [
                _rec(USER_A, "x", KIND_LIKE),
                _rec(USER_A, "y", KIND_LIKE),
                _rec(USER_A, "z", KIND_LIKE),
                _rec(USER_A, "z", KIND_RATING, value=1.0),
            ],
        )
        await stats_mod.rebuild_user_taste(conn)

        row = await conn.fetchrow(
            "SELECT seed_item_ids, suppressed_item_ids "
            "FROM recommendations.user_taste WHERE user_id = $1",
            USER_A,
        )
        assert ids["z"] in row["suppressed_item_ids"]
    finally:
        await _clear(conn)
        await conn.close()


# ══════════════════════════════════════════════════════════════════════════
# Co-occurrence and purge
# ══════════════════════════════════════════════════════════════════════════


async def test_cooccurrence_stores_one_direction_with_shrunk_similarity():
    ids = await _setup()
    conn = await _connect()
    try:
        records = []
        for index in range(4):
            user = f"synth_test_c{index}"
            records.append(_rec(user, "x", KIND_LIKE))
            records.append(_rec(user, "y", KIND_LIKE))
        await load(conn, records)

        pairs = await stats_mod.rebuild_cooccurrence(conn, CONFIG, min_cooc=3)

        assert pairs >= 1
        row = await conn.fetchrow(
            "SELECT item_a, item_b, cooc, sim FROM recommendations.item_cooccurrence "
            "WHERE item_a = $1 AND item_b = $2",
            min(ids["x"], ids["y"]),
            max(ids["x"], ids["y"]),
        )
        assert row["cooc"] == 4
        # Shrinkage keeps a 4-of-4 co-occurrence well below 1.0.
        assert 0 < row["sim"] < 1.0
    finally:
        await conn.execute("DELETE FROM recommendations.item_cooccurrence")
        await conn.execute(
            "DELETE FROM recommendations.interactions WHERE user_id LIKE 'synth_test_%'"
        )
        await _clear(conn)
        await conn.close()


async def test_rare_pairs_are_excluded():
    """Below the threshold the table would grow quadratically in noise.

    Asserts on *this* pair rather than a global count: `rebuild_cooccurrence`
    rebuilds the whole table, so any other data in the database contributes rows and
    a total of zero is not a claim this test can make.
    """
    ids = await _setup()
    conn = await _connect()
    try:
        # One shared reader only — below min_cooc=3.
        await load(conn, [_rec(USER_A, "x", KIND_LIKE), _rec(USER_A, "y", KIND_LIKE)])

        await stats_mod.rebuild_cooccurrence(conn, CONFIG, min_cooc=3)

        present = await conn.fetchval(
            "SELECT COUNT(*) FROM recommendations.item_cooccurrence "
            "WHERE item_a = $1 AND item_b = $2",
            min(ids["x"], ids["y"]),
            max(ids["x"], ids["y"]),
        )
        assert present == 0
    finally:
        await conn.execute("DELETE FROM recommendations.item_cooccurrence")
        await _clear(conn)
        await conn.close()


async def test_purge_removes_synthetic_readers_and_their_taste():
    """Seed data must never quietly become part of a real measurement."""
    await _setup()
    conn = await _connect()
    try:
        await load(
            conn,
            [
                _rec("synth_purge_me", "x", KIND_LIKE),
                _rec("synth_purge_me", "y", KIND_LIKE),
                _rec("synth_purge_me", "z", KIND_LIKE),
            ],
        )
        await stats_mod.rebuild_user_taste(conn)

        # Scoped to this test's own reader: an unscoped purge would delete a
        # developer's whole seeded dataset as a side effect.
        removed = await purge_synthetic(conn, prefix="synth_purge_me")

        assert removed >= 3
        assert (
            await conn.fetchval(
                "SELECT COUNT(*) FROM recommendations.interactions "
                "WHERE user_id = 'synth_purge_me'"
            )
            == 0
        )
        assert (
            await conn.fetchval(
                "SELECT COUNT(*) FROM recommendations.user_taste "
                "WHERE user_id = 'synth_purge_me'"
            )
            == 0
        )
    finally:
        await _clear(conn)
        await conn.close()
