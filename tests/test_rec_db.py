"""Unit tests for the db layer's pure helpers."""

import pytest

from recommendation_engine.db import (  # noqa: E402
    MIN_PGVECTOR_VERSION,
    Database,
    _parse_version,
    _safe_ident,
)

pytestmark = pytest.mark.unit


# ── Version gate ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("0.8.0", (0, 8, 0)),
        ("0.8.6", (0, 8, 6)),
        ("0.7.4", (0, 7, 4)),
        ("1.0", (1, 0)),
        ("0.8.0-rc1", (0, 8, 0)),  # suffixes dropped, not fatal
        ("", ()),
    ],
)
def test_parse_version(raw, expected):
    assert _parse_version(raw) == expected


def test_version_gate_accepts_080_and_above():
    """0.8.0 is the floor because hnsw.iterative_scan lands there, and without it
    a filtered KNN silently under-returns."""
    assert _parse_version("0.8.0") >= MIN_PGVECTOR_VERSION
    assert _parse_version("0.8.6") >= MIN_PGVECTOR_VERSION
    assert _parse_version("1.0.0") >= MIN_PGVECTOR_VERSION


def test_version_gate_rejects_below_080():
    assert not _parse_version("0.7.4") >= MIN_PGVECTOR_VERSION
    assert not _parse_version("0.5.1") >= MIN_PGVECTOR_VERSION


# ── iterative_scan allowlist ─────────────────────────────────────────────
# The mode is a Postgres enum, so it cannot be passed as a query parameter and
# has to be interpolated. The allowlist is what keeps that from being injectable.


@pytest.mark.parametrize("mode", ["off", "relaxed_order", "strict_order"])
def test_safe_ident_accepts_valid_modes(mode):
    assert _safe_ident(mode) == mode


@pytest.mark.parametrize(
    "mode",
    [
        "relaxed",
        "RELAXED_ORDER",
        "",
        "off; DROP TABLE recommendations.items",
        "strict_order --",
    ],
)
def test_safe_ident_rejects_anything_else(mode):
    with pytest.raises(ValueError, match="invalid hnsw.iterative_scan mode"):
        _safe_ident(mode)


# ── Pool wiring ──────────────────────────────────────────────────────────


def test_shared_pool_when_dsns_match():
    """One compute in local dev — a second pool to the same DSN would just
    double idle connections."""
    db = Database(write_dsn="postgresql://a/db", read_dsn="postgresql://a/db")
    assert db._shared is True


def test_separate_pools_when_dsns_differ():
    db = Database(
        write_dsn="postgresql://primary/db", read_dsn="postgresql://replica/db"
    )
    assert db._shared is False


def test_statement_cache_disabled_by_default():
    """Required for transaction-mode poolers (Neon pooled endpoint, PgBouncer):
    a cached prepared statement name eventually collides across backends, and it
    only fails under concurrency — i.e. never in local testing."""
    assert Database(write_dsn="x", read_dsn="x")._statement_cache_size == 0


def test_accessing_pools_before_connect_raises_clearly():
    db = Database(write_dsn="x", read_dsn="x")
    with pytest.raises(RuntimeError, match="connect\\(\\) has not run"):
        _ = db.read_pool
    with pytest.raises(RuntimeError, match="connect\\(\\) has not run"):
        _ = db.write_pool
