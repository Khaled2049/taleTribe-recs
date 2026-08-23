"""Integration tests for the HTTP routes against a real pgvector database.

Self-skipping on a missing `RECS_TEST_DATABASE_URL` — see test_rec_integration.py.

The LLM is stubbed out on `app.state.llm`, so these tests exercise the real routes,
real SQL and real ranking without spending tokens. Seeds fixtures under a
`__route__` prefix and removes them afterwards.
"""

import json
import os

import pytest

TEST_DSN = os.getenv("RECS_TEST_DATABASE_URL", "")

if TEST_DSN:
    os.environ["RECS_DATABASE_URL"] = TEST_DSN
    os.environ["RECS_DATABASE_URL_RO"] = ""
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "test-project")
os.environ["USE_MOCK"] = "true"
os.environ["ENVIRONMENT"] = "development"
# Generous buckets so ordinary assertions do not trip the limiter; the limiter has
# its own dedicated test that sets them low.
os.environ["MAX_REQUESTS_PER_MINUTE_PER_USER"] = "500"
os.environ["MAX_LLM_REQUESTS_PER_MINUTE_PER_USER"] = "500"

import asyncpg  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from pgvector.asyncpg import register_vector  # noqa: E402

from recommendation_engine.migrations import migrate  # noqa: E402

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not TEST_DSN, reason="RECS_TEST_DATABASE_URL not set; start docker-compose"
    ),
]

DIM = 768
PREFIX = "__route__"


def vec(components: dict) -> list:
    v = [0.0] * DIM
    for i, w in components.items():
        v[i] = float(w)
    norm = sum(x * x for x in v) ** 0.5
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


async def _seed():
    await migrate.run(TEST_DSN)
    conn = await _connect()
    ids = {}
    try:
        await conn.execute(
            "DELETE FROM recommendations.items WHERE source_id LIKE $1", PREFIX + "%"
        )
        for suffix, title, author, genres, vector in FIXTURES:
            ids[suffix] = await conn.fetchval(
                """
                INSERT INTO recommendations.items (
                    source, source_id, title, author, genres, themes, tone,
                    core_premise, is_eligible, embed_input, embed_input_sha,
                    embedding, embed_task_type
                ) VALUES ('cmu', $1, $2, $3, $4::text[], ARRAY['isolation'],
                          ARRAY['bleak'], 'A premise.', true, 'x', $5,
                          $6::vector, 'RETRIEVAL_DOCUMENT')
                RETURNING id
                """,
                PREFIX + suffix,
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
        await conn.execute(
            "DELETE FROM recommendations.items WHERE source_id LIKE $1", PREFIX + "%"
        )
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


def test_adhoc_marks_off_platform_items():
    import asyncio

    asyncio.run(_seed())
    try:
        with _client() as client:
            response = client.post(
                "/recommend/adhoc",
                json={"user_id": "u1", "books": [{"title": "Route Test Alpha"}]},
            )
        items = response.json()["data"]["items"]

        assert any(item["off_platform"] for item in items)
    finally:
        asyncio.run(_cleanup())


# ══════════════════════════════════════════════════════════════════════════
# Behavioral
# ══════════════════════════════════════════════════════════════════════════


def test_behavioral_falls_back_to_popular_without_a_taste_vector():
    """Honest reporting: nothing populates user_taste until the Firestore signals
    export exists, so this must say `popular` rather than imply personalization."""
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

        streamed = {}
        for frame in body.split("\n\n"):
            lines = frame.strip().split("\n")
            if len(lines) < 2 or not lines[0].startswith("event: explanation"):
                continue
            data = json.loads(lines[1].removeprefix("data: "))
            streamed.setdefault(data["item_id"], "")
            streamed[data["item_id"]] += data["delta"]

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
