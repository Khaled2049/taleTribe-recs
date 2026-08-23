"""Migration runner: applies numbered .sql files in order, exactly once.

    python -m recommendation_engine.migrations.migrate            # apply pending
    python -m recommendation_engine.migrations.migrate --status   # show state
    python -m recommendation_engine.migrations.migrate --dsn ...  # override target

Properties that matter:

* **Serialized.** A session-level advisory lock means two concurrent runners
  (two Cloud Run instances booting, or a deploy racing a local run) can never
  apply the same file twice.
* **Checksummed.** Editing an already-applied file is a real bug — the schema in
  the database no longer matches the schema in git, and every later environment
  will diverge. That fails loudly here instead of silently in production.
* **Per-file transaction.** A failing file leaves nothing behind, so a fixed
  file can just be re-run.
"""

import argparse
import asyncio
import hashlib
import logging
import re
import sys
from pathlib import Path
from typing import List, NamedTuple, Optional

import asyncpg

logging.basicConfig(format="%(levelname)s %(message)s", level=logging.INFO)
logger = logging.getLogger("recs.migrate")

MIGRATIONS_DIR = Path(__file__).parent

# Arbitrary but fixed: any 64-bit int works as long as nothing else in this
# database picks the same one. Derived from the schema name for traceability.
ADVISORY_LOCK_KEY = 0x5EC0_11EC_7105_0001

_FILENAME_RE = re.compile(r"^(\d{3,})_.+\.sql$")


class Migration(NamedTuple):
    version: int
    name: str
    sql: str
    checksum: str


def discover_migrations(directory: Path = MIGRATIONS_DIR) -> List[Migration]:
    """Load and sort every NNN_name.sql in the directory."""
    found: List[Migration] = []
    for path in sorted(directory.glob("*.sql")):
        match = _FILENAME_RE.match(path.name)
        if not match:
            logger.warning("skipping %s (does not match NNN_name.sql)", path.name)
            continue
        sql = path.read_text()
        found.append(
            Migration(
                version=int(match.group(1)),
                name=path.name,
                sql=sql,
                checksum=hashlib.sha256(sql.encode()).hexdigest(),
            )
        )

    versions = [m.version for m in found]
    duplicates = {v for v in versions if versions.count(v) > 1}
    if duplicates:
        raise RuntimeError(
            f"duplicate migration version(s) {sorted(duplicates)} — two files "
            "share a number, so apply order is undefined"
        )
    return found


_BOOTSTRAP_SQL = """
CREATE SCHEMA IF NOT EXISTS recommendations;
CREATE TABLE IF NOT EXISTS recommendations.schema_migrations (
    version    integer PRIMARY KEY,
    name       text NOT NULL,
    checksum   text NOT NULL,
    applied_at timestamptz NOT NULL DEFAULT now()
);
"""


async def _applied(conn: asyncpg.Connection) -> dict:
    rows = await conn.fetch(
        "SELECT version, name, checksum FROM recommendations.schema_migrations"
    )
    return {r["version"]: (r["name"], r["checksum"]) for r in rows}


async def run(dsn: str, status_only: bool = False) -> int:
    """Apply pending migrations. Returns the number applied."""
    migrations = discover_migrations()
    if not migrations:
        logger.warning("no migration files found in %s", MIGRATIONS_DIR)
        return 0

    conn = await asyncpg.connect(dsn)
    try:
        # Serialize against other runners for the whole session, not per file:
        # holding it across the batch also prevents interleaved partial applies.
        await conn.execute("SELECT pg_advisory_lock($1)", ADVISORY_LOCK_KEY)
        await conn.execute(_BOOTSTRAP_SQL)
        applied = await _applied(conn)

        # A file that changed after being applied means git and the database
        # disagree. Refuse rather than guess which one is right.
        for migration in migrations:
            if migration.version not in applied:
                continue
            _, recorded_checksum = applied[migration.version]
            if recorded_checksum != migration.checksum:
                raise RuntimeError(
                    f"{migration.name} was modified after it was applied "
                    f"(recorded sha {recorded_checksum[:12]}, file sha "
                    f"{migration.checksum[:12]}). Add a new migration instead of "
                    "editing an applied one."
                )

        pending = [m for m in migrations if m.version not in applied]

        if status_only:
            for migration in migrations:
                state = "applied" if migration.version in applied else "PENDING"
                logger.info("%-40s %s", migration.name, state)
            return 0

        if not pending:
            logger.info("schema up to date (%d applied)", len(applied))
            return 0

        for migration in pending:
            logger.info("applying %s", migration.name)
            # Per-file transaction: a failure rolls the whole file back, so the
            # fixed version can simply be re-run.
            async with conn.transaction():
                await conn.execute(migration.sql)
                await conn.execute(
                    "INSERT INTO recommendations.schema_migrations "
                    "(version, name, checksum) VALUES ($1, $2, $3)",
                    migration.version,
                    migration.name,
                    migration.checksum,
                )

        logger.info("applied %d migration(s)", len(pending))
        return len(pending)
    finally:
        # Best-effort unlock; closing the connection releases it regardless.
        try:
            await conn.execute("SELECT pg_advisory_unlock($1)", ADVISORY_LOCK_KEY)
        finally:
            await conn.close()


def _resolve_dsn(explicit: Optional[str]) -> str:
    if explicit:
        return explicit
    from recommendation_engine.config import RecSettings

    return RecSettings().write_dsn


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Apply recommendations migrations.")
    parser.add_argument("--dsn", help="Postgres DSN (default: RECS_DATABASE_URL)")
    parser.add_argument(
        "--status", action="store_true", help="Report applied/pending and exit"
    )
    args = parser.parse_args(argv)

    try:
        asyncio.run(run(_resolve_dsn(args.dsn), status_only=args.status))
    except Exception as exc:  # surfaced as a clean message, not a traceback wall
        logger.error("migration failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
