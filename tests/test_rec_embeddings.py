"""Unit tests for the task_type / batching extension to the embedding provider.

The headline test here is `test_no_task_type_sends_no_taskType_key`. Every vector
already stored in `chapter_chunks`, `semantic_memory` and `episodic_memory` was
produced by a request with no `taskType` field. If that ever changes, those
vectors stop matching their neighbours — and because `VectorStore` deliberately
has no brute-force fallback, retrieval returns *nothing* with no error raised
anywhere. So the historical request shape is pinned byte for byte.
"""

from typing import Protocol, cast

import pytest
from tenacity import wait_none

from embedding_provider import (  # noqa: E402
    BATCH_MAX_REQUESTS,
    EXPECTED_EMBEDDING_DIM,
    EmbeddingError,
    EmbeddingProvider,
    EmbeddingTransientError,
    GoogleAIEmbeddingProvider,
    MockEmbeddingProvider,
    TaskType,
)

pytestmark = pytest.mark.unit


class _RetryPredicate(Protocol):
    exception_types: type[BaseException] | tuple[type[BaseException], ...]


class _RetryController(Protocol):
    reraise: bool
    retry: _RetryPredicate
    stop: object
    wait: object


def _post_retry() -> _RetryController:
    wrapped = cast(object, GoogleAIEmbeddingProvider._post)
    return cast(_RetryController, getattr(wrapped, "retry"))


class _FakeResponse:
    def __init__(self, payload=None, status_code=200, text=""):
        self._payload = payload if payload is not None else {}
        self.status_code = status_code
        self.text = text

    def json(self):
        return self._payload


class _RecordingClient:
    """Captures every POST so the exact request body can be asserted."""

    def __init__(self, responses):
        # responses: list of _FakeResponse or Exception, consumed in order.
        self._responses = list(responses)
        self.calls = []

    async def post(self, url, headers=None, json=None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        nxt = self._responses.pop(0) if self._responses else _FakeResponse()
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    async def aclose(self):
        pass


def _provider(responses=()):
    provider = GoogleAIEmbeddingProvider(api_key="test-key")
    provider._client = _RecordingClient(responses)
    return provider


def _vec(value=0.1):
    return [value] * EXPECTED_EMBEDDING_DIM


# ══════════════════════════════════════════════════════════════════════════
# The backwards-compatibility pin
# ══════════════════════════════════════════════════════════════════════════


def test_no_task_type_sends_no_taskType_key():
    """The exact historical request body, byte for byte.

    Not `"taskType": None` — the key must be absent entirely.
    """
    provider = GoogleAIEmbeddingProvider(api_key="k")

    body = provider._request_body("hello", None)

    assert body == {
        "model": "models/gemini-embedding-001",
        "content": {"parts": [{"text": "hello"}]},
        "outputDimensionality": 768,
    }
    assert "taskType" not in body


async def test_embed_without_arguments_omits_taskType_on_the_wire():
    """Same guarantee, asserted at the HTTP boundary rather than the helper —
    this is the call shape `VectorStore` and `ChapterRAG` actually make."""
    provider = _provider([_FakeResponse({"embedding": {"values": _vec()}})])

    await provider.embed("some chapter text")

    (call,) = provider._client.calls
    assert "taskType" not in call["json"]
    assert call["url"].endswith(":embedContent")
    assert call["json"]["outputDimensionality"] == 768
    assert call["headers"] == {"x-goog-api-key": "test-key"}


async def test_existing_callers_positional_signature_still_works():
    """agent.py and vector_store.py call `.embed(text)` positionally."""
    provider = _provider([_FakeResponse({"embedding": {"values": _vec()}})])

    result = await provider.embed("text")

    assert len(result) == EXPECTED_EMBEDDING_DIM


# ══════════════════════════════════════════════════════════════════════════
# Provenance
# ══════════════════════════════════════════════════════════════════════════


def test_model_id_distinguishes_mock_from_real():
    """Vectors from different providers occupy different spaces. The catalog
    backfill compares this to decide what needs re-embedding — without it, mock
    md5 vectors survive a switch to Gemini and sit in the index matching nothing,
    with no error raised anywhere.
    """
    assert MockEmbeddingProvider().model_id == "MockEmbeddingProvider"
    assert GoogleAIEmbeddingProvider(api_key="k").model_id == (
        "google:gemini-embedding-001"
    )


def test_model_id_reflects_a_model_change():
    """A model upgrade must be visible too — same provider, different space."""
    a = GoogleAIEmbeddingProvider(api_key="k", model="gemini-embedding-001")
    b = GoogleAIEmbeddingProvider(api_key="k", model="text-embedding-004")

    assert a.model_id != b.model_id


def test_model_id_defaults_to_the_class_name_for_custom_providers():
    class Custom(EmbeddingProvider):
        @property
        def dimension(self):
            return EXPECTED_EMBEDDING_DIM

        async def embed(self, text, task_type=None):
            return _vec()

    assert Custom().model_id == "Custom"


def test_minimal_subclass_gets_embed_batch_for_free():
    """embed_batch is concrete, not abstract, so subclasses defined elsewhere
    (including in the existing test suite) keep instantiating."""

    class Minimal(EmbeddingProvider):
        @property
        def dimension(self):
            return EXPECTED_EMBEDDING_DIM

        async def embed(self, text, task_type=None):
            return [float(len(text))] * EXPECTED_EMBEDDING_DIM

    provider = Minimal()  # would raise TypeError if embed_batch were abstract
    assert hasattr(provider, "embed_batch")


async def test_default_embed_batch_loops_over_embed_in_order():
    class Minimal(EmbeddingProvider):
        @property
        def dimension(self):
            return 3

        async def embed(self, text, task_type=None):
            return [float(len(text))] * 3

    vectors = await Minimal().embed_batch(["a", "bb", "ccc"])

    assert vectors == [[1.0] * 3, [2.0] * 3, [3.0] * 3]


# ══════════════════════════════════════════════════════════════════════════
# Task types
# ══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "task_type",
    [TaskType.RETRIEVAL_DOCUMENT, TaskType.RETRIEVAL_QUERY, TaskType.CLUSTERING],
)
def test_task_type_included_when_given(task_type):
    body = GoogleAIEmbeddingProvider(api_key="k")._request_body("t", task_type)

    assert body["taskType"] == task_type
    # Everything else must be unchanged by the addition.
    assert body["outputDimensionality"] == 768
    assert body["content"] == {"parts": [{"text": "t"}]}


def test_task_type_constants_match_the_api_spelling():
    """These strings go straight into the request; a typo is a 400 at best and a
    silently mis-projected vector at worst."""
    assert TaskType.RETRIEVAL_DOCUMENT == "RETRIEVAL_DOCUMENT"
    assert TaskType.RETRIEVAL_QUERY == "RETRIEVAL_QUERY"


async def test_catalog_and_query_use_different_task_types():
    """The whole point of the asymmetry: a document and a query about it are
    projected differently."""
    provider = _provider(
        [
            _FakeResponse({"embedding": {"values": _vec()}}),
            _FakeResponse({"embedding": {"values": _vec()}}),
        ]
    )

    await provider.embed("doc", task_type=TaskType.RETRIEVAL_DOCUMENT)
    await provider.embed("query", task_type=TaskType.RETRIEVAL_QUERY)

    doc_call, query_call = provider._client.calls
    assert doc_call["json"]["taskType"] == "RETRIEVAL_DOCUMENT"
    assert query_call["json"]["taskType"] == "RETRIEVAL_QUERY"


# ══════════════════════════════════════════════════════════════════════════
# Batching
# ══════════════════════════════════════════════════════════════════════════


async def test_embed_batch_uses_the_batch_endpoint_with_a_requests_array():
    provider = _provider(
        [_FakeResponse({"embeddings": [{"values": _vec(0.1)}, {"values": _vec(0.2)}]})]
    )

    vectors = await provider.embed_batch(
        ["one", "two"], task_type=TaskType.RETRIEVAL_DOCUMENT
    )

    (call,) = provider._client.calls
    assert call["url"].endswith(":batchEmbedContents")
    assert [r["content"]["parts"][0]["text"] for r in call["json"]["requests"]] == [
        "one",
        "two",
    ]
    assert all(r["taskType"] == "RETRIEVAL_DOCUMENT" for r in call["json"]["requests"])
    assert vectors == [_vec(0.1), _vec(0.2)]


async def test_embed_batch_preserves_input_order():
    """Callers zip these vectors back onto rows by position."""
    provider = _provider(
        [_FakeResponse({"embeddings": [{"values": _vec(float(i))} for i in range(5)]})]
    )

    vectors = await provider.embed_batch([f"t{i}" for i in range(5)])

    assert [v[0] for v in vectors] == [0.0, 1.0, 2.0, 3.0, 4.0]


async def test_embed_batch_chunks_at_the_api_limit():
    total = BATCH_MAX_REQUESTS + 25
    provider = _provider(
        [
            _FakeResponse(
                {"embeddings": [{"values": _vec()} for _ in range(BATCH_MAX_REQUESTS)]}
            ),
            _FakeResponse({"embeddings": [{"values": _vec()} for _ in range(25)]}),
        ]
    )

    vectors = await provider.embed_batch([f"t{i}" for i in range(total)])

    assert len(provider._client.calls) == 2
    assert len(provider._client.calls[0]["json"]["requests"]) == BATCH_MAX_REQUESTS
    assert len(provider._client.calls[1]["json"]["requests"]) == 25
    assert len(vectors) == total


async def test_embed_batch_rejects_a_short_response():
    """A count mismatch would silently misalign vectors onto the wrong books —
    far worse than failing the backfill."""
    provider = _provider([_FakeResponse({"embeddings": [{"values": _vec()}]})])

    with pytest.raises(EmbeddingError, match="returned 1 vectors for 3 inputs"):
        await provider.embed_batch(["a", "b", "c"])


async def test_embed_batch_of_nothing_makes_no_request():
    provider = _provider()

    assert await provider.embed_batch([]) == []
    assert provider._client.calls == []


async def test_mock_provider_batches_at_the_production_dimension():
    vectors = await MockEmbeddingProvider().embed_batch(["a", "b"])

    assert len(vectors) == 2
    assert all(len(v) == EXPECTED_EMBEDDING_DIM for v in vectors)


async def test_mock_provider_ignores_task_type():
    """Mock vectors carry no semantics, so making them asymmetric would only
    break tests that embed a document and then query for it."""
    mock = MockEmbeddingProvider()

    as_doc = await mock.embed("text", task_type=TaskType.RETRIEVAL_DOCUMENT)
    as_query = await mock.embed("text", task_type=TaskType.RETRIEVAL_QUERY)
    bare = await mock.embed("text")

    assert as_doc == as_query == bare


# ══════════════════════════════════════════════════════════════════════════
# Retry classification
# ══════════════════════════════════════════════════════════════════════════


def test_retry_is_bounded():
    """Introspected rather than exercised, so the assertion costs no wall time.
    Bounded on purpose: a genuinely broken upstream must fail the backfill, not
    retry for hours."""
    from tenacity.stop import stop_after_attempt, stop_any

    retry_state = _post_retry()

    assert retry_state.reraise is True

    # Only transient failures are retried.
    exception_types = retry_state.retry.exception_types
    if not isinstance(exception_types, tuple):
        exception_types = (exception_types,)
    assert EmbeddingTransientError in exception_types
    assert EmbeddingError not in exception_types

    # Bounded by both an attempt count and a wall-clock deadline.
    assert isinstance(retry_state.stop, stop_any)
    attempt_stops = [
        s for s in retry_state.stop.stops if isinstance(s, stop_after_attempt)
    ]
    assert attempt_stops, "no attempt ceiling — a broken upstream would retry forever"
    assert attempt_stops[0].max_attempt_number == 4


@pytest.mark.parametrize("status", [429, 500, 502, 503])
async def test_transient_statuses_are_retried(status, monkeypatch):
    monkeypatch.setattr(_post_retry(), "wait", wait_none())
    provider = _provider(
        [
            _FakeResponse(status_code=status, text="slow down"),
            _FakeResponse({"embedding": {"values": _vec()}}),
        ]
    )

    result = await provider.embed("text")

    assert len(result) == EXPECTED_EMBEDDING_DIM
    assert len(provider._client.calls) == 2, "should have retried once"


@pytest.mark.parametrize("status", [400, 401, 403, 404])
async def test_permanent_statuses_fail_immediately(status, monkeypatch):
    """Retrying a bad API key just burns the backfill's time budget."""
    monkeypatch.setattr(_post_retry(), "wait", wait_none())
    provider = _provider([_FakeResponse(status_code=status, text="nope")])

    with pytest.raises(EmbeddingError) as exc_info:
        await provider.embed("text")

    assert not isinstance(exc_info.value, EmbeddingTransientError)
    assert len(provider._client.calls) == 1, "must not retry a permanent failure"


async def test_network_errors_are_transient(monkeypatch):
    import httpx

    monkeypatch.setattr(_post_retry(), "wait", wait_none())
    provider = _provider(
        [
            httpx.ConnectError("connection refused"),
            _FakeResponse({"embedding": {"values": _vec()}}),
        ]
    )

    result = await provider.embed("text")

    assert len(result) == EXPECTED_EMBEDDING_DIM
    assert len(provider._client.calls) == 2


async def test_retry_eventually_gives_up_and_reraises(monkeypatch):
    monkeypatch.setattr(_post_retry(), "wait", wait_none())
    provider = _provider([_FakeResponse(status_code=503, text="down")] * 10)

    with pytest.raises(EmbeddingTransientError, match="503"):
        await provider.embed("text")

    # Bounded at 4 attempts, not unbounded.
    assert len(provider._client.calls) == 4
