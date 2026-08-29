"""Shared integration-test helpers.

The `recommendations` schema is migrated by story-data, not by this repo, so an
integration test can no longer create the schema it needs. It asserts the schema
is present and skips otherwise: a missing schema means the story-data stack was
not started or not migrated, which is a setup problem to report as such rather
than a failure to attribute to the code under test.

Catalog fixtures need a `stories` row to exist first — `items.story_id` is a
foreign key — so `seed_stories` creates them and `drop_stories` removes them.
Cleanup deletes the *story*, never the item: `ON DELETE CASCADE` then takes the
item, its stats, its interactions and its explanations with it, which is both
less code and a standing check that the cascade is wired the way the schema
claims.
"""

import uuid

import asyncpg
import pytest

# Fixture story ids are derived from a fixed namespace so they are stable across
# runs and readable in a failure message: the same key always yields the same
# UUID. A random uuid4 per run would leave orphans behind whenever a test
# crashed before cleanup.
TEST_STORY_NAMESPACE = uuid.UUID("6f1c0f4a-9a2b-4c7d-8e31-0d5b7a9c2e44")


def story_id_for(key: str) -> str:
    """The deterministic story UUID for a fixture key."""
    return str(uuid.uuid5(TEST_STORY_NAMESPACE, key))


async def require_recommendations_schema(dsn: str) -> None:
    """Skip the calling test unless `recommendations.items` exists at `dsn`."""
    conn = await asyncpg.connect(dsn)
    try:
        present = await conn.fetchval(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = 'recommendations' AND table_name = 'items'"
        )
    finally:
        await conn.close()
    if not present:
        pytest.skip(
            "no `recommendations` schema in RECS_TEST_DATABASE_URL; apply "
            "story-data's migrations to that database first (make migrate)"
        )


async def seed_stories(conn, keys, *, published: bool = True) -> dict:
    """Create one published `stories` row per key. Returns key -> story_id.

    Idempotent, so a test whose previous run died before cleanup still starts
    from a known state rather than colliding on the primary key.
    """
    ids = {}
    for key in keys:
        story_id = story_id_for(key)
        await conn.execute(
            """
            INSERT INTO stories (id, owner_id, title, description, author_name,
                                 category, is_published)
            VALUES ($1, 'test-owner', $2, 'A test description.', 'A Test Author',
                    'fantasy', $3)
            ON CONFLICT (id) DO UPDATE SET is_published = EXCLUDED.is_published
            """,
            story_id,
            f"Fixture {key}",
            published,
        )
        ids[key] = story_id
    return ids


async def drop_stories(conn, keys) -> None:
    """Delete fixture stories, cascading to everything derived from them."""
    await conn.execute(
        "DELETE FROM stories WHERE id = ANY($1::uuid[])",
        [story_id_for(key) for key in keys],
    )
