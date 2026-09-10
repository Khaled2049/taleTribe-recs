"""Unit tests for cache-key derivation, HyDE, and the explanation layer.

No network: the Gemini client is a fake.
"""

import json
import os
from typing import cast

import pytest

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "test-project")

from recommendation_engine import explain as explain_mod  # noqa: E402
from recommendation_engine import hyde as hyde_mod  # noqa: E402
from recommendation_engine.llm import LLMResponseError  # noqa: E402

pytestmark = pytest.mark.unit


def _target(item_id=1, sha="sha1", story_id="story-1"):
    return explain_mod.ExplanationTarget(
        item_id=item_id,
        story_id=story_id,
        title=f"Book {item_id}",
        author="An Author",
        genres=["horror"],
        themes=["ghosts and hauntings"],
        tone=["unsettling"],
        core_premise="A house remembers.",
        embed_input_sha=sha,
    )


class _FakeLLM:
    model = "fake-model"

    def __init__(self, texts=None, chunks=None, json_payload=None, raise_with=None):
        self._texts = list(texts or [])
        self._chunks = list(chunks or [])
        self._json = json_payload
        self._raise = raise_with
        self.text_calls = 0
        self.stream_calls = 0
        self.json_calls = 0

    async def generate_text(self, prompt, **kwargs):
        self.text_calls += 1
        if self._raise:
            raise self._raise
        return self._texts.pop(0) if self._texts else "A reason."

    async def generate_json(self, prompt, **kwargs):
        self.json_calls += 1
        if self._raise:
            raise self._raise
        return self._json or {}

    async def stream_text(self, prompt, **kwargs):
        self.stream_calls += 1
        if self._raise:
            raise self._raise
        for chunk in self._chunks or ["A ", "reason."]:
            yield chunk


class _FakeCache:
    def __init__(self, preloaded=None):
        self.store = dict(preloaded or {})
        self.writes = []

    async def get_many(self, keys):
        return {k: self.store[k] for k in keys if k in self.store}

    async def put(self, key, item_id, explanation, model):
        self.writes.append((key, item_id, explanation))
        self.store[key] = explanation


# ══════════════════════════════════════════════════════════════════════════
# Cache keys
# ══════════════════════════════════════════════════════════════════════════


def test_same_inputs_give_the_same_key():
    a = explain_mod.cache_key("m", "story-1", "sha", "fp")
    b = explain_mod.cache_key("m", "story-1", "sha", "fp")

    assert a == b


@pytest.mark.parametrize(
    "kwargs",
    [
        {"model": "other"},
        {"story_id": "story-2"},
        {"embed_input_sha": "different"},
        {"fingerprint": "other-fp"},
    ],
)
def test_every_component_changes_the_key(kwargs):
    base = dict(model="m", story_id="story-1", embed_input_sha="sha", fingerprint="fp")
    assert explain_mod.cache_key(**base) != explain_mod.cache_key(**{**base, **kwargs})


def test_key_self_invalidates_when_the_item_is_renormalized():
    """The reason embed_input_sha is in the key: if a book's premise or themes are
    re-derived, its stale explanation is abandoned automatically."""
    before = _target(sha="old").key("m", "fp")
    after = _target(sha="new").key("m", "fp")

    assert before != after


def test_seed_order_does_not_change_the_fingerprint():
    """A shelf re-render with reordered seeds must not miss the cache on every
    item."""
    assert explain_mod.query_fingerprint(
        seed_item_ids=[3, 1, 2]
    ) == explain_mod.query_fingerprint(seed_item_ids=[1, 2, 3])


def test_fingerprint_normalizes_query_whitespace_and_case():
    assert explain_mod.query_fingerprint(
        query="  Cozy  MYSTERY "
    ) == explain_mod.query_fingerprint(query="cozy mystery")


def test_different_queries_give_different_fingerprints():
    assert explain_mod.query_fingerprint(
        query="cozy mystery"
    ) != explain_mod.query_fingerprint(query="space opera")


def test_query_and_seeds_are_distinguished():
    assert explain_mod.query_fingerprint(query="dune") != explain_mod.query_fingerprint(
        seed_item_ids=[1]
    )


def test_empty_request_has_a_stable_fingerprint():
    """The popularity fallback still needs a cache key."""
    assert explain_mod.query_fingerprint() == explain_mod.query_fingerprint()


# ══════════════════════════════════════════════════════════════════════════
# Prompt construction
# ══════════════════════════════════════════════════════════════════════════


def test_prompt_carries_only_normalized_fields():
    """Never a raw plot summary — the model cannot spoil what it was not told."""
    prompt = explain_mod.build_prompt(_target(), "gothic horror")

    assert "A house remembers." in prompt
    assert "ghosts and hauntings" in prompt
    assert "gothic horror" in prompt
    assert "Summary" not in prompt


def test_prompt_forbids_spoilers_and_plot_recap():
    prompt = explain_mod.build_prompt(_target(), "x")

    assert "reveal the ending" in prompt
    assert "Do not summarize the plot" in prompt


def test_context_description_prefers_the_query():
    assert explain_mod.describe_context("cozy mystery", ["Dune"]) == "cozy mystery"
    assert explain_mod.describe_context(None, ["Dune", "Foundation"]) == (
        "books like Dune, Foundation"
    )
    assert "popular" in explain_mod.describe_context(None, None)


# ══════════════════════════════════════════════════════════════════════════
# Sync explanations
# ══════════════════════════════════════════════════════════════════════════


async def test_generates_one_explanation_per_target():
    llm = _FakeLLM(texts=["Reason A.", "Reason B."])
    cache = _FakeCache()

    out = await explain_mod.explain_many(
        llm, cache, [_target(1, "a"), _target(2, "b")], "ctx", "fp"
    )

    assert {e["item_id"] for e in out} == {1, 2}
    assert all(e["explanation"] for e in out)
    assert llm.text_calls == 2


async def test_cached_explanations_cost_nothing():
    target = _target(1, "a")
    key = target.key("fake-model", "fp")
    llm = _FakeLLM()
    cache = _FakeCache({key: "Previously written."})

    out = await explain_mod.explain_many(llm, cache, [target], "ctx", "fp")

    assert out[0]["explanation"] == "Previously written."
    assert out[0]["cached"] is True
    assert llm.text_calls == 0


async def test_generated_explanations_are_written_to_cache():
    llm = _FakeLLM(texts=["Fresh."])
    cache = _FakeCache()

    await explain_mod.explain_many(llm, cache, [_target()], "ctx", "fp")

    assert len(cache.writes) == 1


async def test_whitespace_is_collapsed():
    llm = _FakeLLM(texts=["A  reason\nwith   breaks."])

    out = await explain_mod.explain_many(llm, _FakeCache(), [_target()], "ctx", "fp")

    assert out[0]["explanation"] == "A reason with breaks."


async def test_one_failure_does_not_fail_the_batch():
    """A shelf with nine reasons and one blank beats an error page."""
    llm = _FakeLLM(raise_with=LLMResponseError("blocked"))

    out = await explain_mod.explain_many(
        llm, _FakeCache(), [_target(1, "a"), _target(2, "b")], "ctx", "fp"
    )

    assert all(e["explanation"] is None for e in out)
    assert len(out) == 2


async def test_no_llm_configured_returns_nulls_not_errors():
    out = await explain_mod.explain_many(None, _FakeCache(), [_target()], "ctx", "fp")

    assert out[0]["explanation"] is None


async def test_no_targets_makes_no_calls():
    llm = _FakeLLM()

    assert await explain_mod.explain_many(llm, _FakeCache(), [], "ctx", "fp") == []
    assert llm.text_calls == 0


# ══════════════════════════════════════════════════════════════════════════
# Streaming
# ══════════════════════════════════════════════════════════════════════════


def _parse_sse(frames):
    """Turn raw SSE text into (event, data) pairs."""
    events = []
    for frame in "".join(frames).split("\n\n"):
        if not frame.strip():
            continue
        lines = frame.strip().split("\n")
        event = lines[0].removeprefix("event: ")
        data = cast(
            dict[str, object],
            cast(object, json.loads(lines[1].removeprefix("data: "))),
        )
        events.append((event, data))
    return events


async def test_stream_emits_deltas_then_item_done_then_done():
    llm = _FakeLLM(chunks=["Because ", "it is ", "similar."])

    frames = [
        f
        async for f in explain_mod.stream_explanations(
            llm, _FakeCache(), [_target()], "ctx", "fp"
        )
    ]
    events = _parse_sse(frames)

    kinds = [e for e, _ in events]
    assert kinds == ["explanation"] * 3 + ["item_done", "done"]
    assert "".join(d["delta"] for e, d in events if e == "explanation") == (
        "Because it is similar."
    )
    assert events[-1][1]["completed"] == 1


async def test_stream_tags_every_event_with_its_item_id():
    """One multiplexed connection updates many cards, so the id is essential."""
    llm = _FakeLLM(chunks=["x"])

    frames = [
        f
        async for f in explain_mod.stream_explanations(
            llm, _FakeCache(), [_target(1, "a"), _target(2, "b")], "ctx", "fp"
        )
    ]
    events = _parse_sse(frames)

    ids = {d["item_id"] for e, d in events if e != "done"}
    assert ids == {1, 2}


async def test_stream_replays_a_cached_explanation_without_generating():
    target = _target()
    cache = _FakeCache({target.key("fake-model", "fp"): "Cached reason."})
    llm = _FakeLLM()

    frames = [
        f
        async for f in explain_mod.stream_explanations(
            llm, cache, [target], "ctx", "fp"
        )
    ]
    events = _parse_sse(frames)

    assert llm.stream_calls == 0
    assert events[0][1]["delta"] == "Cached reason."
    assert events[1][1]["cached"] is True


async def test_disconnect_stops_generation():
    """The point of the streaming mode: a reader who scrolls away stops paying."""
    llm = _FakeLLM(chunks=["a", "b", "c"])

    async def disconnected():
        return True

    frames = [
        f
        async for f in explain_mod.stream_explanations(
            llm,
            _FakeCache(),
            [_target(1, "a"), _target(2, "b")],
            "ctx",
            "fp",
            is_disconnected=disconnected,
        )
    ]

    # Disconnected before the first item, so nothing was generated at all.
    assert llm.stream_calls == 0
    assert frames == []


async def test_partial_generation_is_never_cached():
    """A truncated explanation must not become the permanent answer."""
    calls = {"n": 0}

    async def disconnected():
        # Connected for the first chunk, gone for the second.
        calls["n"] += 1
        return calls["n"] > 1

    llm = _FakeLLM(chunks=["first ", "second ", "third"])
    cache = _FakeCache()

    _ = [
        f
        async for f in explain_mod.stream_explanations(
            llm, cache, [_target()], "ctx", "fp", is_disconnected=disconnected
        )
    ]

    assert cache.writes == []


async def test_stream_reports_a_per_item_error_and_continues():
    llm = _FakeLLM(raise_with=LLMResponseError("blocked"))

    frames = [
        f
        async for f in explain_mod.stream_explanations(
            llm, _FakeCache(), [_target(1, "a"), _target(2, "b")], "ctx", "fp"
        )
    ]
    events = _parse_sse(frames)

    assert [e for e, _ in events] == ["item_error", "item_error", "done"]
    assert events[-1][1]["completed"] == 0


async def test_stream_without_an_llm_emits_an_error_frame():
    frames = [
        f
        async for f in explain_mod.stream_explanations(
            None, _FakeCache(), [_target()], "ctx", "fp"
        )
    ]
    events = _parse_sse(frames)

    assert events[0][0] == "error"
    assert events[-1][0] == "done"


def test_sse_frame_format():
    frame = explain_mod.sse("explanation", {"item_id": 1, "delta": "hi"})

    assert frame == 'event: explanation\ndata: {"item_id": 1, "delta": "hi"}\n\n'


# ══════════════════════════════════════════════════════════════════════════
# HyDE
# ══════════════════════════════════════════════════════════════════════════


async def test_hyde_renders_in_the_catalog_embedding_format():
    """The whole trick: the hypothetical document must look like a catalog entry,
    or it lands in a different region of the space."""
    llm = _FakeLLM(
        json_payload={
            "title": "The Salt-Stained Mirror",
            "genres": ["horror"],
            "core_premise": "A lighthouse keeper unravels.",
            "themes": ["isolation", "madness and paranoia"],
            "tone": ["bleak", "unsettling"],
        }
    )

    result = await hyde_mod.generate(llm, "lonely lighthouse keeper losing his mind")

    assert result is not None
    assert result.embed_text.startswith("Title: The Salt-Stained Mirror")
    assert "Premise: A lighthouse keeper unravels." in result.embed_text
    assert "Themes & tropes: isolation, madness and paranoia" in result.embed_text
    assert "Tone: bleak, unsettling" in result.embed_text


async def test_hyde_omits_author():
    """A hypothetical book has no real author, and inventing one would add a
    strong meaningless signal."""
    llm = _FakeLLM(
        json_payload={"title": "T", "core_premise": "P", "themes": [], "tone": []}
    )

    result = await hyde_mod.generate(llm, "query")

    assert "Author" not in result.embed_text


async def test_hyde_output_is_vocabulary_filtered():
    """An invented theme would be a token the corpus never uses."""
    llm = _FakeLLM(
        json_payload={
            "title": "T",
            "core_premise": "P",
            "themes": ["isolation", "a theme I invented"],
            "tone": ["bleak", "not a tone"],
        }
    )

    result = await hyde_mod.generate(llm, "query")

    assert "a theme I invented" not in result.embed_text
    assert "not a tone" not in result.embed_text
    assert "isolation" in result.embed_text


async def test_hyde_failure_returns_none_rather_than_raising():
    """HyDE is an enhancement; its failure must degrade to embedding the raw query,
    never fail the recommendation request."""
    llm = _FakeLLM(raise_with=LLMResponseError("blocked"))

    assert await hyde_mod.generate(llm, "query") is None


async def test_hyde_ignores_an_empty_query():
    llm = _FakeLLM(json_payload={})

    assert await hyde_mod.generate(llm, "   ") is None
    assert llm.json_calls == 0


def test_hyde_prompt_asks_for_a_book_not_a_reply():
    prompt = hyde_mod.build_prompt("cozy mystery")

    assert "cozy mystery" in prompt
    assert "Do not address the reader" in prompt
    assert "do not recommend a real existing title" in prompt
    assert "ALLOWED THEMES" in prompt
