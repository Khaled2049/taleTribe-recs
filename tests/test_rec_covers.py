import uuid

import pytest

from recommendation_engine.routes import attach_cover_urls

pytestmark = pytest.mark.unit

A = "00000000-0000-0000-0000-00000000000a"
B = "00000000-0000-0000-0000-00000000000b"


class _FakePool:
    def __init__(self, rows):
        self._rows = rows
        self.calls = []

    async def fetch(self, sql, *args):
        self.calls.append((sql, args))
        wanted = set(args[0])
        return [row for row in self._rows if str(row["id"]) in wanted]


class _FakeDb:
    def __init__(self, rows):
        self.read_pool = _FakePool(rows)


async def test_one_query_covers_every_item():
    db = _FakeDb(
        [
            {"id": uuid.UUID(A), "cover_url": "https://img/a-thumb.webp"},
            {"id": uuid.UUID(B), "cover_url": None},
        ]
    )
    items = [{"story_id": A}, {"story_id": B}, {"story_id": A}]

    await attach_cover_urls(db, items)

    assert len(db.read_pool.calls) == 1
    assert sorted(db.read_pool.calls[0][1][0]) == [A, B]
    assert [item["cover_url"] for item in items] == [
        "https://img/a-thumb.webp",
        None,
        "https://img/a-thumb.webp",
    ]


async def test_a_story_missing_from_the_lookup_has_no_cover():
    db = _FakeDb([])
    items = [{"story_id": A}]

    await attach_cover_urls(db, items)

    assert items == [{"story_id": A, "cover_url": None}]


async def test_no_items_means_no_query():
    db = _FakeDb([])

    assert await attach_cover_urls(db, []) == []
    assert db.read_pool.calls == []


def test_only_published_stories_expose_a_cover():
    from recommendation_engine.routes import _COVERS_SQL

    assert "is_published" in _COVERS_SQL
