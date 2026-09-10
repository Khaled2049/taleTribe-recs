"""Integration tests for the HTTP routes against a real pgvector database.

Self-skipping on a missing `RECS_TEST_DATABASE_URL` — see test_rec_integration.py.

The LLM is stubbed out on `app.state.llm`, so these tests exercise the real routes,
real SQL and real ranking without spending tokens. Seeds fixtures under a
`__route__` prefix and removes them afterwards.
"""

import json
import math
import os
import uuid
from typing import cast

import pytest

TEST_DSN = os.getenv("RECS_TEST_DATABASE_URL", "")

if TEST_DSN:
    os.environ["RECS_DATABASE_URL"] = TEST_DSN
    os.environ["RECS_DATABASE_URL_RO"] = ""
os.environ["USE_MOCK"] = "true"
os.environ["ENVIRONMENT"] = "development"
# Generous buckets so ordinary assertions do not trip the limiter; the limiter has
# its own dedicated test that sets them low.
os.environ["MAX_REQUESTS_PER_MINUTE_PER_USER"] = "500"
os.environ["MAX_LLM_REQUESTS_PER_MINUTE_PER_USER"] = "500"
# Unlimited daily budgets by default, for two reasons: ordinary assertions must
# not trip them, and 0 skips the meter's round trip entirely, so the rest of this
# file does not depend on `recommendations.llm_usage` existing. The daily budget
# has its own tests below, which require the table and skip without it.
os.environ["RECS_MAX_SEARCHES_PER_DAY_PER_USER"] = "0"
os.environ["RECS_MAX_EXPLANATIONS_PER_DAY_PER_USER"] = "0"
os.environ["RECS_MAX_SEARCHES_PER_DAY_PLATFORM"] = "0"
os.environ["RECS_MAX_EXPLANATIONS_PER_DAY_PLATFORM"] = "0"

import asyncpg
from conftest import (
    drop_stories,
    require_recommendations_schema,
    seed_stories,
)
from fastapi.testclient import TestClient
from pgvector.asyncpg import register_vector

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not TEST_DSN, reason="RECS_TEST_DATABASE_URL not set; start docker-compose"
    ),
]

DIM = 768
PREFIX = "__route__"


def vec(components: dict[int, float]) -> list[float]:
    v = [0.0] * DIM
    for i, w in components.items():
        v[i] = float(w)
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v] if norm else v


# Shared dominant component so fixtures reliably outrank any pre-loaded catalog —
# see the note in test_rec_retrieval_integration.py.
FIXTURES = [
    ("a", "Route Test Alpha", "Test Author", ["science-fiction"], vec({0: 1.0})),
    (
        "b",
        "Route Test Beta",
        "Test Author",
        ["science-fiction"],
        vec({0: 0.99, 1: 0.14}),
    ),
    ("c", "Route Test Gamma", "Other Author", ["horror"], vec({0: 0.98, 2: 0.2})),
]


class _StubLLM:
    """Stands in for GeminiClient — no network, deterministic."""

    model = "stub-model"

    def __init__(self):
        self.text_calls = 0
        self.json_calls = 0

    async def generate_text(self, prompt, **kwargs):
        self.text_calls += 1
        return "A stubbed reason."

    async def generate_json(self, prompt, **kwargs):
        self.json_calls += 1
        return {
            "title": "A Hypothetical Book",
            "genres": ["science-fiction"],
            "core_premise": "An invented premise.",
            "themes": ["isolation"],
            "tone": ["bleak"],
        }

    async def stream_text(self, prompt, **kwargs):
        for chunk in ("A ", "stubbed ", "reason."):
            yield chunk

    async def aclose(self):
        pass


async def _connect():
    """Connect with the pgvector codec registered.

    `Database` installs it through the pool's `init` hook; a bare
    `asyncpg.connect` does not, and passing a Python list for a `vector` column
    then fails with "expected str, got list".
    """
    conn = await asyncpg.connect(TEST_DSN)
    await register_vector(conn)
    return conn


KEYS = [PREFIX + f[0] for f in FIXTURES]


async def _seed():
    await require_recommendations_schema(TEST_DSN)
    conn = await _connect()
    ids = {}
    try:
        await drop_stories(conn, KEYS)
        stories = await seed_stories(conn, KEYS)
        for suffix, title, author, genres, vector in FIXTURES:
            ids[suffix] = await conn.fetchval(
                """
                INSERT INTO recommendations.items (
                    story_id, title, author, genres, themes, tone,
                    core_premise, is_eligible, embed_input, embed_input_sha,
                    embedding, embed_task_type
                ) VALUES ($1::uuid, $2, $3, $4::text[], ARRAY['isolation'],
                          ARRAY['bleak'], 'A premise.', true, 'x', $5,
                          $6::vector, 'RETRIEVAL_DOCUMENT')
                RETURNING id
                """,
                stories[PREFIX + suffix],
                title,
                author,
                genres,
                f"sha-{PREFIX}{suffix}",
                vector,
            )
    finally:
        await conn.close()
    return ids


async def _cleanup():
    conn = await _connect()
    try:
        await drop_stories(conn, KEYS)
        await conn.execute(
            "DELETE FROM recommendations.explanation_cache WHERE explanation = $1",
            "A stubbed reason.",
        )
    finally:
        await conn.close()


def _client(llm=None):
    """A TestClient with the lifespan running and the LLM stubbed."""
    from recommendation_engine.server import create_app

    app = create_app()
    app.state.llm = llm if llm is not None else _StubLLM()
    return TestClient(app)


# ══════════════════════════════════════════════════════════════════════════
# Ad-hoc
# ══════════════════════════════════════════════════════════════════════════


def test_adhoc_resolves_a_named_book_and_excludes_it():
    import asyncio

    ids = asyncio.run(_seed())
    try:
        with _client() as client:
            response = client.post(
                "/recommend/adhoc",
                json={"user_id": "u1", "books": [{"title": "Route Test Alpha"}]},
            )
        assert response.status_code == 200, response.text
        data = response.json()["data"]

        assert data["mode"] == "adhoc"
        assert data["resolved_books"][0]["id"] == ids["a"]
        # Never recommend the book the reader just named.
        assert ids["a"] not in {item["id"] for item in data["items"]}
    finally:
        asyncio.run(_cleanup())


def test_adhoc_reports_unresolved_titles_without_erroring():
    """A reader naming a book we do not carry is a normal outcome."""
    import asyncio

    asyncio.run(_seed())
    try:
        with _client() as client:
            response = client.post(
                "/recommend/adhoc",
                json={
                    "user_id": "u1",
                    "books": [
                        {"title": "Route Test Alpha"},
                        {"title": "Definitely Not In The Catalog XYZQ"},
                    ],
                },
            )
        assert response.status_code == 200
        data = response.json()["data"]

        assert len(data["resolved_books"]) == 1
        assert data["unresolved_books"] == ["Definitely Not In The Catalog XYZQ"]
    finally:
        asyncio.run(_cleanup())


def test_adhoc_requires_a_prompt_or_books():
    with _client() as client:
        response = client.post("/recommend/adhoc", json={"user_id": "u1"})

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_REQUEST"


def test_adhoc_uses_hyde_for_a_free_text_prompt():
    import asyncio

    asyncio.run(_seed())
    llm = _StubLLM()
    try:
        with _client(llm) as client:
            response = client.post(
                "/recommend/adhoc",
                json={"user_id": "u1", "prompt": "a lonely lighthouse keeper"},
            )
        assert response.status_code == 200
        data = response.json()["data"]

        assert data["hyde_used"] is True
        assert data["hypothetical_document"]["title"] == "A Hypothetical Book"
        assert llm.json_calls == 1
    finally:
        asyncio.run(_cleanup())


def test_hyde_can_be_disabled_per_request():
    import asyncio

    asyncio.run(_seed())
    llm = _StubLLM()
    try:
        with _client(llm) as client:
            response = client.post(
                "/recommend/adhoc",
                json={"user_id": "u1", "prompt": "something", "use_hyde": False},
            )
        assert response.json()["data"]["hyde_used"] is False
        assert llm.json_calls == 0
    finally:
        asyncio.run(_cleanup())


def test_adhoc_applies_filters():
    import asyncio

    ids = asyncio.run(_seed())
    try:
        with _client() as client:
            response = client.post(
                "/recommend/adhoc",
                json={
                    "user_id": "u1",
                    "books": [{"title": "Route Test Alpha"}],
                    "filters": {"genres": ["horror"]},
                    "top_k": 20,
                },
            )
        returned = {item["id"] for item in response.json()["data"]["items"]}

        assert ids["c"] in returned  # the only horror fixture
        assert ids["b"] not in returned  # science-fiction, filtered out
    finally:
        asyncio.run(_cleanup())


def test_adhoc_returns_an_explanation_cache_key_per_item():
    """So a client can tell whether an explanation would be a cache hit."""
    import asyncio

    asyncio.run(_seed())
    try:
        with _client() as client:
            response = client.post(
                "/recommend/adhoc",
                json={"user_id": "u1", "books": [{"title": "Route Test Alpha"}]},
            )
        items = response.json()["data"]["items"]

        assert items
        assert all(len(item["explanation_cache_key"]) == 64 for item in items)
    finally:
        asyncio.run(_cleanup())


def test_adhoc_items_name_the_story_they_came_from():
    """A client needs the story id to link anywhere. It replaced the old
    `source`/`source_id` pair when the catalog became platform-only."""
    import asyncio

    asyncio.run(_seed())
    try:
        with _client() as client:
            response = client.post(
                "/recommend/adhoc",
                json={"user_id": "u1", "books": [{"title": "Route Test Alpha"}]},
            )
        items = response.json()["data"]["items"]

        assert items
        assert all(uuid.UUID(item["story_id"]) for item in items)
        assert not any("off_platform" in item for item in items)
    finally:
        asyncio.run(_cleanup())


# ══════════════════════════════════════════════════════════════════════════
# Behavioral
# ══════════════════════════════════════════════════════════════════════════


def test_behavioral_falls_back_to_popular_without_a_taste_vector():
    """Honest reporting: a reader with no signals has no taste vector, so this
    must say `popular` rather than imply personalization."""
    import asyncio

    asyncio.run(_seed())
    try:
        with _client() as client:
            response = client.post(
                "/recommend/behavioral", json={"user_id": "nobody", "top_k": 3}
            )
        assert response.status_code == 200
        data = response.json()["data"]

        assert data["mode"] == "popular"
        assert data["n_signals"] == 0
    finally:
        asyncio.run(_cleanup())


def test_behavioral_uses_a_taste_vector_when_one_exists():
    import asyncio

    ids = asyncio.run(_seed())

    async def add_taste():
        conn = await _connect()
        try:
            await conn.execute(
                "INSERT INTO recommendations.user_taste "
                "(user_id, taste_embedding, seed_item_ids, n_signals) "
                "VALUES ($1, $2::vector, $3::bigint[], $4) "
                "ON CONFLICT (user_id) DO UPDATE SET "
                "taste_embedding = EXCLUDED.taste_embedding, "
                "seed_item_ids = EXCLUDED.seed_item_ids, "
                "n_signals = EXCLUDED.n_signals",
                "__route_reader__",
                vec({0: 1.0}),
                [ids["a"]],
                5,
            )
        finally:
            await conn.close()

    async def drop_taste():
        conn = await _connect()
        try:
            await conn.execute(
                "DELETE FROM recommendations.user_taste WHERE user_id = $1",
                "__route_reader__",
            )
        finally:
            await conn.close()

    try:
        asyncio.run(add_taste())
        with _client() as client:
            response = client.post(
                "/recommend/behavioral",
                json={"user_id": "__route_reader__", "top_k": 5},
            )
        data = response.json()["data"]

        assert data["mode"] == "behavioral"
        assert data["n_signals"] == 5
        # Seed items are excluded — never recommend what they already read.
        assert ids["a"] not in {item["id"] for item in data["items"]}
    finally:
        asyncio.run(drop_taste())
        asyncio.run(_cleanup())


# ══════════════════════════════════════════════════════════════════════════
# Explanations
# ══════════════════════════════════════════════════════════════════════════


def test_explain_sync_returns_one_entry_per_item():
    import asyncio

    ids = asyncio.run(_seed())
    try:
        with _client() as client:
            response = client.post(
                "/recommend/explain",
                json={
                    "user_id": "u1",
                    "item_ids": [ids["a"], ids["b"]],
                    "prompt": "science fiction",
                },
            )
        assert response.status_code == 200
        explanations = response.json()["data"]["explanations"]

        assert len(explanations) == 2
        assert all(e["explanation"] == "A stubbed reason." for e in explanations)
    finally:
        asyncio.run(_cleanup())


def test_explain_sync_second_call_is_cached():
    import asyncio

    ids = asyncio.run(_seed())
    llm = _StubLLM()
    try:
        body = {"user_id": "u1", "item_ids": [ids["a"]], "prompt": "science fiction"}
        with _client(llm) as client:
            first = client.post("/recommend/explain", json=body).json()["data"]
            second = client.post("/recommend/explain", json=body).json()["data"]

        assert first["explanations"][0]["cached"] is False
        assert second["explanations"][0]["cached"] is True
        assert llm.text_calls == 1, "a cache hit must not call the model"
    finally:
        asyncio.run(_cleanup())


def test_explain_ignores_unknown_item_ids():
    import asyncio

    ids = asyncio.run(_seed())
    try:
        with _client() as client:
            response = client.post(
                "/recommend/explain",
                json={"user_id": "u1", "item_ids": [ids["a"], 999999999]},
            )
        assert response.status_code == 200
        assert len(response.json()["data"]["explanations"]) == 1
    finally:
        asyncio.run(_cleanup())


def test_explain_rejects_an_empty_item_list():
    with _client() as client:
        response = client.post(
            "/recommend/explain", json={"user_id": "u1", "item_ids": []}
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


# ── Streaming ────────────────────────────────────────────────────────────


def test_stream_emits_sse_frames_tagged_by_item():
    import asyncio

    ids = asyncio.run(_seed())
    try:
        with _client() as client:
            response = client.get(
                "/recommend/explain/stream",
                params={
                    "user_id": "u1",
                    "item_ids": f"{ids['a']},{ids['b']}",
                    "prompt": "science fiction",
                },
            )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        # Proxies must not buffer a stream into a single response.
        assert response.headers.get("x-accel-buffering") == "no"

        body = response.text
        assert "event: explanation" in body
        assert "event: item_done" in body
        assert "event: done" in body

        streamed: dict[int, str] = {}
        for frame in body.split("\n\n"):
            lines = frame.strip().split("\n")
            if len(lines) < 2 or not lines[0].startswith("event: explanation"):
                continue
            parsed = cast(
                dict[str, object],
                cast(object, json.loads(lines[1].removeprefix("data: "))),
            )
            item_id = parsed.get("item_id")
            delta = parsed.get("delta")
            assert isinstance(item_id, int)
            assert isinstance(delta, str)
            streamed.setdefault(item_id, "")
            streamed[item_id] += delta

        assert set(streamed) == {ids["a"], ids["b"]}
        assert all(text == "A stubbed reason." for text in streamed.values())
    finally:
        asyncio.run(_cleanup())


def test_stream_rejects_non_numeric_item_ids():
    with _client() as client:
        response = client.get(
            "/recommend/explain/stream",
            params={"user_id": "u1", "item_ids": "not,numbers"},
        )

    assert response.status_code == 400


def test_stream_requires_item_ids():
    with _client() as client:
        response = client.get(
            "/recommend/explain/stream", params={"user_id": "u1", "item_ids": ""}
        )

    assert response.status_code == 400


# ══════════════════════════════════════════════════════════════════════════
# Rate limiting
# ══════════════════════════════════════════════════════════════════════════


def test_llm_bucket_is_tighter_than_the_ranking_bucket(monkeypatch):
    """Ranking is one DB round trip; generation spends unmetered tokens. The two
    must not share a budget."""
    import asyncio

    ids = asyncio.run(_seed())
    try:
        from recommendation_engine.server import create_app

        monkeypatch.setenv("MAX_REQUESTS_PER_MINUTE_PER_USER", "100")
        monkeypatch.setenv("MAX_LLM_REQUESTS_PER_MINUTE_PER_USER", "2")
        app = create_app()
        app.state.llm = _StubLLM()

        with TestClient(app) as client:
            body = {"user_id": "burst", "item_ids": [ids["a"]]}
            codes = [
                client.post("/recommend/explain", json=body).status_code
                for _ in range(4)
            ]
            # Ranking on the same user is unaffected by the LLM bucket.
            ranking = client.post(
                "/recommend/adhoc",
                json={"user_id": "burst", "books": [{"title": "Route Test Alpha"}]},
            ).status_code

        assert codes[:2] == [200, 200]
        assert 429 in codes[2:]
        assert ranking == 200
    finally:
        asyncio.run(_cleanup())


def test_rate_limited_response_uses_the_stable_error_envelope():
    import asyncio

    ids = asyncio.run(_seed())
    try:
        from recommendation_engine.server import create_app

        os.environ["MAX_LLM_REQUESTS_PER_MINUTE_PER_USER"] = "1"
        app = create_app()
        app.state.llm = _StubLLM()
        with TestClient(app) as client:
            body = {"user_id": "burst2", "item_ids": [ids["a"]]}
            client.post("/recommend/explain", json=body)
            response = client.post("/recommend/explain", json=body)

        assert response.status_code == 429
        payload = response.json()
        assert payload["success"] is False
        assert payload["error"]["code"] == "LLM_RATE_LIMITED"
    finally:
        os.environ["MAX_LLM_REQUESTS_PER_MINUTE_PER_USER"] = "500"
        asyncio.run(_cleanup())


# ══════════════════════════════════════════════════════════════════════════
# Daily LLM budget
#
# The bucket tests above prove burst control. These prove the thing a bucket
# structurally cannot: a ceiling that survives a restart, because it lives in
# Postgres rather than in one process's memory.
# ══════════════════════════════════════════════════════════════════════════

DAILY_USER = "__daily__"


async def _require_budget_tables():
    # One statement charges both counters, so every budget test needs both.
    await require_recommendations_schema(TEST_DSN, "llm_usage", "llm_platform_usage")


async def _clear_usage():
    """Reset both counters for the day.

    The per-user rows are scoped to this file's fixture users. The platform row
    has no user in its key, so there is nothing to scope it by — it deletes
    today's row outright, which is safe because only tests write it in a
    development database.
    """
    conn = await _connect()
    try:
        await conn.execute(
            "DELETE FROM recommendations.llm_usage WHERE user_id = ANY($1::text[])",
            [DAILY_USER, DAILY_USER + "2"],
        )
        await conn.execute(
            "DELETE FROM recommendations.llm_platform_usage "
            "WHERE day = (now() AT TIME ZONE 'utc')::date"
        )
    finally:
        await conn.close()


async def _platform_usage_row(kind):
    conn = await _connect()
    try:
        return await conn.fetchval(
            "SELECT call_count FROM recommendations.llm_platform_usage "
            "WHERE kind = $1 AND day = (now() AT TIME ZONE 'utc')::date",
            kind,
        )
    finally:
        await conn.close()


async def _usage_row(kind):
    conn = await _connect()
    try:
        return await conn.fetchval(
            "SELECT call_count FROM recommendations.llm_usage WHERE user_id = $1 "
            "AND kind = $2 AND day = (now() AT TIME ZONE 'utc')::date",
            DAILY_USER,
            kind,
        )
    finally:
        await conn.close()


def _search(client, prompt="a lonely lighthouse keeper", **extra):
    return client.post(
        "/recommend/adhoc",
        json={"user_id": DAILY_USER, "prompt": prompt, **extra},
    )


def test_daily_search_budget_blocks_the_request_after_the_limit(monkeypatch):
    import asyncio

    asyncio.run(_seed())
    asyncio.run(_require_budget_tables())
    asyncio.run(_clear_usage())
    try:
        monkeypatch.setenv("RECS_MAX_SEARCHES_PER_DAY_PER_USER", "2")
        with _client() as client:
            codes = [_search(client).status_code for _ in range(3)]
            blocked = _search(client)

        assert codes == [200, 200, 429]
        payload = blocked.json()
        assert payload["success"] is False
        assert payload["error"]["code"] == "LLM_DAILY_LIMIT"
        assert payload["error"]["details"] == {"kind": "search", "limit": 2}
        # The refused calls must not keep incrementing — a counter that ran past
        # its own ceiling would misreport usage and could overflow over time.
        assert asyncio.run(_usage_row("search")) == 2
    finally:
        asyncio.run(_clear_usage())
        asyncio.run(_cleanup())


def test_daily_budget_survives_a_restart(monkeypatch):
    """The point of the whole exercise: an in-memory bucket forgets on deploy."""
    import asyncio

    asyncio.run(_seed())
    asyncio.run(_require_budget_tables())
    asyncio.run(_clear_usage())
    try:
        monkeypatch.setenv("RECS_MAX_SEARCHES_PER_DAY_PER_USER", "1")
        with _client() as client:
            assert _search(client).status_code == 200

        # A brand new app: fresh process state, fresh buckets, same database.
        with _client() as client:
            second = _search(client)

        assert second.status_code == 429
        assert second.json()["error"]["code"] == "LLM_DAILY_LIMIT"
    finally:
        asyncio.run(_clear_usage())
        asyncio.run(_cleanup())


def test_explanations_do_not_spend_the_search_budget(monkeypatch):
    """Separate kinds, so "Why this story?" cannot eat a reader's searches."""
    import asyncio

    ids = asyncio.run(_seed())
    asyncio.run(_require_budget_tables())
    asyncio.run(_clear_usage())
    try:
        monkeypatch.setenv("RECS_MAX_SEARCHES_PER_DAY_PER_USER", "1")
        monkeypatch.setenv("RECS_MAX_EXPLANATIONS_PER_DAY_PER_USER", "5")
        with _client() as client:
            for _ in range(3):
                client.post(
                    "/recommend/explain",
                    json={"user_id": DAILY_USER, "item_ids": [ids["a"]]},
                )
            search = _search(client)

        assert search.status_code == 200, "explanations must not consume searches"
        assert asyncio.run(_usage_row("explain")) == 3
        assert asyncio.run(_usage_row("search")) == 1
    finally:
        asyncio.run(_clear_usage())
        asyncio.run(_cleanup())


def test_a_search_that_spends_no_tokens_is_not_charged(monkeypatch):
    """Seed-book retrieval and `use_hyde: false` never reach the LLM."""
    import asyncio

    asyncio.run(_seed())
    asyncio.run(_require_budget_tables())
    asyncio.run(_clear_usage())
    try:
        monkeypatch.setenv("RECS_MAX_SEARCHES_PER_DAY_PER_USER", "1")
        with _client() as client:
            assert _search(client, use_hyde=False).status_code == 200
            books = client.post(
                "/recommend/adhoc",
                json={
                    "user_id": DAILY_USER,
                    "books": [{"title": "Route Test Alpha"}],
                },
            )
            assert books.status_code == 200
            # The budget is untouched, so a real search still goes through.
            assert _search(client).status_code == 200

        assert asyncio.run(_usage_row("search")) == 1
    finally:
        asyncio.run(_clear_usage())
        asyncio.run(_cleanup())


# ══════════════════════════════════════════════════════════════════════════
# Platform-wide daily cap
#
# The per-user budget bounds one reader's day; multiplied by the user count it
# bounds nothing. These cover the ceiling that does, and the ordering property
# that keeps it from being trivially drained.
# ══════════════════════════════════════════════════════════════════════════


def test_platform_cap_stops_a_second_user_who_has_spent_nothing(monkeypatch):
    """The point of a platform ceiling: it is not about who is asking."""
    import asyncio

    asyncio.run(_seed())
    asyncio.run(_require_budget_tables())
    asyncio.run(_clear_usage())
    try:
        monkeypatch.setenv("RECS_MAX_SEARCHES_PER_DAY_PER_USER", "10")
        monkeypatch.setenv("RECS_MAX_SEARCHES_PER_DAY_PLATFORM", "2")
        with _client() as client:
            first = [_search(client).status_code for _ in range(2)]
            # A different user, with their whole personal allowance untouched.
            other = client.post(
                "/recommend/adhoc",
                json={"user_id": DAILY_USER + "2", "prompt": "a quiet village"},
            )

        assert first == [200, 200]
        assert other.status_code == 429
        assert other.json()["error"]["code"] == "LLM_PLATFORM_DAILY_LIMIT"
        assert other.json()["error"]["details"] == {"kind": "search", "limit": 2}
    finally:
        asyncio.run(_clear_usage())
        asyncio.run(_cleanup())


def test_an_over_budget_user_cannot_drain_the_platform_counter(monkeypatch):
    """The whole reason the SQL charges the user first.

    Were it the other way round, one user who had already spent their own
    allowance could keep incrementing the platform counter with requests that
    are refused anyway — denying the feature to everyone else.
    """
    import asyncio

    asyncio.run(_seed())
    asyncio.run(_require_budget_tables())
    asyncio.run(_clear_usage())
    try:
        monkeypatch.setenv("RECS_MAX_SEARCHES_PER_DAY_PER_USER", "1")
        monkeypatch.setenv("RECS_MAX_SEARCHES_PER_DAY_PLATFORM", "50")
        with _client() as client:
            assert _search(client).status_code == 200
            refused = [_search(client).status_code for _ in range(20)]

            # Everyone else still has the platform budget they started with.
            other = client.post(
                "/recommend/adhoc",
                json={"user_id": DAILY_USER + "2", "prompt": "a quiet village"},
            )

        assert set(refused) == {429}
        assert other.status_code == 200
        # One charge from each user, and nothing from the 20 refusals.
        assert asyncio.run(_platform_usage_row("search")) == 2
    finally:
        asyncio.run(_clear_usage())
        asyncio.run(_cleanup())


def test_platform_caps_are_per_kind(monkeypatch):
    """A flood of searches must not close explanations too."""
    import asyncio

    ids = asyncio.run(_seed())
    asyncio.run(_require_budget_tables())
    asyncio.run(_clear_usage())
    try:
        monkeypatch.setenv("RECS_MAX_SEARCHES_PER_DAY_PLATFORM", "1")
        monkeypatch.setenv("RECS_MAX_EXPLANATIONS_PER_DAY_PLATFORM", "5")
        with _client() as client:
            assert _search(client).status_code == 200
            assert _search(client).status_code == 429
            explanation = client.post(
                "/recommend/explain",
                json={"user_id": DAILY_USER, "item_ids": [ids["a"]]},
            )

        assert explanation.status_code == 200
        assert asyncio.run(_platform_usage_row("search")) == 1
        assert asyncio.run(_platform_usage_row("explain")) == 1
    finally:
        asyncio.run(_clear_usage())
        asyncio.run(_cleanup())


def test_platform_cap_applies_even_with_per_user_budgets_disabled(monkeypatch):
    """Turning off the per-user budget must not turn off the platform ceiling."""
    import asyncio

    asyncio.run(_seed())
    asyncio.run(_require_budget_tables())
    asyncio.run(_clear_usage())
    try:
        monkeypatch.setenv("RECS_MAX_SEARCHES_PER_DAY_PER_USER", "0")
        monkeypatch.setenv("RECS_MAX_SEARCHES_PER_DAY_PLATFORM", "3")
        with _client() as client:
            codes = [_search(client).status_code for _ in range(4)]

        assert codes == [200, 200, 200, 429]
        assert asyncio.run(_platform_usage_row("search")) == 3
    finally:
        asyncio.run(_clear_usage())
        asyncio.run(_cleanup())
