"""Integration tests for what the catalog ingest reads out of story-data.

Self-skipping on a missing `RECS_TEST_DATABASE_URL` — see test_rec_integration.py.

The read is the boundary between an author's private drafts and everything
downstream: whatever it returns is embedded and shown to an LLM.
"""

import os
import uuid

import pytest

TEST_DSN = os.getenv("RECS_TEST_DATABASE_URL", "")

import asyncpg
from conftest import drop_stories, seed_stories

from recommendation_engine.ingest.platform import read_stories

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not TEST_DSN, reason="RECS_TEST_DATABASE_URL not set; start docker-compose"
    ),
]

KEY = "__read__serial"


async def _add_chapter(conn, story_id, position, summary, *, released: bool):
    chapter_id = uuid.uuid4()
    await conn.execute(
        """
        INSERT INTO chapters (id, story_id, title, content, position, word_count,
                              published_at)
        VALUES ($1, $2, 'Chapter', 'x', $3, 100, CASE WHEN $4 THEN now() END)
        """,
        chapter_id,
        story_id,
        position,
        released,
    )
    await conn.execute(
        "INSERT INTO chapter_summaries (chapter_id, source_revision, summary) "
        "VALUES ($1, 1, $2)",
        chapter_id,
        summary,
    )


async def _read_one(conn, story_id, since=None):
    return [r for r in await read_stories(conn, since=since) if r.story_id == story_id]


async def test_draft_chapters_never_reach_the_catalog():
    conn = await asyncpg.connect(TEST_DSN)
    try:
        await drop_stories(conn, [KEY])
        story_id = str((await seed_stories(conn, [KEY]))[KEY])
        await _add_chapter(conn, story_id, 1, "The ship leaves port.", released=True)
        await _add_chapter(
            conn, story_id, 2, "The captain is the traitor.", released=False
        )

        (record,) = await _read_one(conn, story_id)

        assert record.chapter_count == 1
        assert record.word_count == 100
        assert record.chapter_summary == "The ship leaves port."
    finally:
        await drop_stories(conn, [KEY])
        await conn.close()


async def test_a_draft_save_does_not_make_a_story_look_new():
    conn = await asyncpg.connect(TEST_DSN)
    try:
        await drop_stories(conn, [KEY])
        story_id = str((await seed_stories(conn, [KEY]))[KEY])
        (record,) = await _read_one(conn, story_id)
        cursor = record.updated_at

        # What story-data does on a draft save: updated_at moves, the public one does not.
        await conn.execute(
            "UPDATE stories SET updated_at = now() + interval '1 hour' WHERE id = $1",
            story_id,
        )
        assert await _read_one(conn, story_id, since=cursor) == []

        await conn.execute(
            "UPDATE stories SET public_updated_at = now() + interval '1 hour' "
            "WHERE id = $1",
            story_id,
        )
        assert len(await _read_one(conn, story_id, since=cursor)) == 1
    finally:
        await drop_stories(conn, [KEY])
        await conn.close()
