"""Unit tests for the migration runner's pure logic.

The apply path itself needs a database and lives in test_rec_integration.py.
"""

import hashlib
import os

import pytest

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "test-project")

from recommendation_engine.migrations.migrate import (  # noqa: E402
    MIGRATIONS_DIR,
    discover_migrations,
)

pytestmark = pytest.mark.unit


def _write(directory, name, body="SELECT 1;"):
    path = directory / name
    path.write_text(body)
    return path


def test_discovers_and_sorts_by_version(tmp_path):
    _write(tmp_path, "010_third.sql")
    _write(tmp_path, "002_second.sql")
    _write(tmp_path, "001_first.sql")

    found = discover_migrations(tmp_path)

    assert [m.version for m in found] == [1, 2, 10]
    assert [m.name for m in found] == [
        "001_first.sql",
        "002_second.sql",
        "010_third.sql",
    ]


def test_sorts_numerically_not_lexically(tmp_path):
    """Lexical sort would put 010 before 002 — apply order would be wrong."""
    _write(tmp_path, "002_b.sql")
    _write(tmp_path, "010_c.sql")

    assert [m.version for m in discover_migrations(tmp_path)] == [2, 10]


def test_ignores_files_not_matching_the_convention(tmp_path):
    _write(tmp_path, "001_ok.sql")
    _write(tmp_path, "notes.sql")
    _write(tmp_path, "1_too_few_digits.sql")

    found = discover_migrations(tmp_path)

    assert [m.name for m in found] == ["001_ok.sql"]


def test_duplicate_versions_are_rejected(tmp_path):
    """Two files sharing a number means apply order is undefined — refuse rather
    than pick one arbitrarily."""
    _write(tmp_path, "001_a.sql")
    _write(tmp_path, "001_b.sql")

    with pytest.raises(RuntimeError, match="duplicate migration version"):
        discover_migrations(tmp_path)


def test_checksum_is_sha256_of_file_contents(tmp_path):
    body = "CREATE TABLE x (id int);\n"
    _write(tmp_path, "001_x.sql", body)

    (migration,) = discover_migrations(tmp_path)

    assert migration.checksum == hashlib.sha256(body.encode()).hexdigest()


def test_checksum_changes_when_file_changes(tmp_path):
    """This is what lets the runner detect an edited-after-applied migration."""
    _write(tmp_path, "001_x.sql", "SELECT 1;")
    before = discover_migrations(tmp_path)[0].checksum

    _write(tmp_path, "001_x.sql", "SELECT 2;")
    after = discover_migrations(tmp_path)[0].checksum

    assert before != after


def test_empty_directory_yields_nothing(tmp_path):
    assert discover_migrations(tmp_path) == []


# ── The real migration directory ─────────────────────────────────────────


def test_real_migrations_are_discoverable_and_well_formed():
    found = discover_migrations(MIGRATIONS_DIR)

    assert found, "no migrations found in the real migrations directory"
    assert found[0].version == 1
    assert found[0].name == "001_init.sql"
    # Versions must be unique and ascending — discover_migrations raises on
    # duplicates, so reaching here already proves uniqueness.
    versions = [m.version for m in found]
    assert versions == sorted(versions)


def test_init_migration_creates_the_schema_and_hnsw_index():
    """A guard against someone trimming the DDL: without the partial HNSW index
    on is_eligible, filtered retrieval quietly loses recall."""
    (init,) = [m for m in discover_migrations(MIGRATIONS_DIR) if m.version == 1]

    assert "CREATE SCHEMA IF NOT EXISTS recommendations" in init.sql
    assert "CREATE EXTENSION IF NOT EXISTS vector" in init.sql
    assert "USING hnsw (embedding vector_cosine_ops)" in init.sql
    assert "WHERE is_eligible" in init.sql
    assert "vector(768)" in init.sql


def test_init_migration_ships_cf_stubbed_at_zero():
    """The CF term is deliberately inert until an event log exists. If this
    default ever changes silently, ranking shifts with no code change."""
    (init,) = [m for m in discover_migrations(MIGRATIONS_DIR) if m.version == 1]

    assert "('w_cf_ceiling',       0.00" in init.sql
