"""Unit tests for the direct Gemini client.

No network. The httpx client is replaced with a fake that returns canned
responses, including a scripted SSE stream.
"""

import json
import os
from typing import Protocol, cast

import httpx
import pytest
from tenacity import wait_none

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "test-project")

from recommendation_engine.llm import (  # noqa: E402
    DEFAULT_MODEL,
    GeminiClient,
    LLMError,
    LLMResponseError,
    LLMTransientError,
    TokenUsage,
    _extract_text,
    build_client,
)

pytestmark = pytest.mark.unit


class _RetryController(Protocol):
    wait: object


def _post_retry() -> _RetryController:
    wrapped = cast(object, GeminiClient._post)
    return cast(_RetryController, getattr(wrapped, "retry"))


def _candidate(text, finish_reason="STOP"):
    return {
        "candidates": [
            {
                "content": {"parts": [{"text": text}]},
                "finishReason": finish_reason,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 10,
            "candidatesTokenCount": 5,
            "totalTokenCount": 15,
        },
    }


class _FakeResponse:
    def __init__(self, payload=None, status_code=200, text=""):
        self._payload = payload if payload is not None else {}
        self.status_code = status_code
        self.text = text

    def json(self):
        return self._payload


class _FakeStream:
    """Async context manager mimicking httpx's streaming response."""

    def __init__(self, lines, status_code=200, body=b""):
        self._lines = lines
        self.status_code = status_code
        self._body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def aread(self):
        return self._body

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _FakeHttp:
    def __init__(self, responses=(), stream=None):
        self._responses = list(responses)
        self._stream = stream
        self.calls = []

    async def post(self, url, headers=None, json=None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        nxt = self._responses.pop(0) if self._responses else _FakeResponse()
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    def stream(self, method, url, headers=None, json=None):
        self.calls.append(
            {"method": method, "url": url, "headers": headers, "json": json}
        )
        if isinstance(self._stream, Exception):
            raise self._stream
        return self._stream

    async def aclose(self):
        pass


def _client(responses=(), stream=None, model=DEFAULT_MODEL):
    client = GeminiClient(api_key="test-key", model=model)
    client._client = _FakeHttp(responses, stream)
    return client


# ══════════════════════════════════════════════════════════════════════════
# Response extraction — each failure mode gets a specific message
# ══════════════════════════════════════════════════════════════════════════


def test_extract_text_happy_path():
    assert _extract_text(_candidate("hello")) == "hello"


def test_extract_text_joins_multiple_parts():
    payload = {"candidates": [{"content": {"parts": [{"text": "a"}, {"text": "b"}]}}]}

    assert _extract_text(payload) == "ab"


def test_no_candidates_means_blocked_and_says_so():
    payload = {"promptFeedback": {"blockReason": "SAFETY"}}

    with pytest.raises(LLMResponseError, match="no candidates"):
        _extract_text(payload)


def test_empty_text_reports_the_finish_reason():
    payload = {"candidates": [{"content": {"parts": []}, "finishReason": "SAFETY"}]}

    with pytest.raises(LLMResponseError, match="SAFETY"):
        _extract_text(payload)


# ══════════════════════════════════════════════════════════════════════════
# Request construction
# ══════════════════════════════════════════════════════════════════════════


async def test_generate_text_posts_expected_shape():
    client = _client([_FakeResponse(_candidate("out"))])

    result = await client.generate_text(
        "prompt", system="be terse", max_output_tokens=99, temperature=0.3
    )

    assert result == "out"
    (call,) = client._client.calls
    assert call["url"].endswith(f"{DEFAULT_MODEL}:generateContent")
    assert call["headers"] == {"x-goog-api-key": "test-key"}
    body = call["json"]
    assert body["contents"][0]["parts"][0]["text"] == "prompt"
    assert body["systemInstruction"]["parts"][0]["text"] == "be terse"
    assert body["generationConfig"]["maxOutputTokens"] == 99
    assert body["generationConfig"]["temperature"] == 0.3
    # No schema requested, so no JSON mime type.
    assert "responseMimeType" not in body["generationConfig"]


async def test_system_instruction_omitted_when_absent():
    client = _client([_FakeResponse(_candidate("out"))])

    await client.generate_text("prompt")

    assert "systemInstruction" not in client._client.calls[0]["json"]


async def test_generate_json_requests_native_structured_output():
    """Far more reliable than asking for JSON in the prompt, which is what the
    story agent has to do because creditProxy has no schema passthrough."""
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    client = _client([_FakeResponse(_candidate('{"a": "b"}'))])

    result = await client.generate_json("prompt", response_schema=schema)

    assert result == {"a": "b"}
    config = client._client.calls[0]["json"]["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    assert config["responseSchema"] == schema


async def test_generate_json_strips_code_fences():
    """Models sometimes wrap output in ```json despite responseMimeType."""
    client = _client([_FakeResponse(_candidate('```json\n{"a": 1}\n```'))])

    assert await client.generate_json("p", response_schema={}) == {"a": 1}


async def test_generate_json_raises_on_unparseable_output():
    client = _client([_FakeResponse(_candidate("not json at all"))])

    with pytest.raises(LLMResponseError, match="did not return valid JSON"):
        await client.generate_json("p", response_schema={})


async def test_generate_json_defaults_to_low_temperature():
    """Extraction wants reproducibility, not variety."""
    client = _client([_FakeResponse(_candidate("{}"))])

    await client.generate_json("p", response_schema={})

    assert client._client.calls[0]["json"]["generationConfig"]["temperature"] == 0.2


# ══════════════════════════════════════════════════════════════════════════
# Retry classification
# ══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("status", [429, 500, 503])
async def test_transient_statuses_retry(status, monkeypatch):
    monkeypatch.setattr(_post_retry(), "wait", wait_none())
    client = _client(
        [
            _FakeResponse(status_code=status, text="busy"),
            _FakeResponse(_candidate("recovered")),
        ]
    )

    assert await client.generate_text("p") == "recovered"
    assert len(client._client.calls) == 2


@pytest.mark.parametrize("status", [400, 401, 403])
async def test_permanent_statuses_do_not_retry(status, monkeypatch):
    monkeypatch.setattr(_post_retry(), "wait", wait_none())
    client = _client([_FakeResponse(status_code=status, text="bad key")])

    with pytest.raises(LLMError) as exc_info:
        await client.generate_text("p")

    assert not isinstance(exc_info.value, LLMTransientError)
    assert len(client._client.calls) == 1


async def test_network_error_is_transient(monkeypatch):
    monkeypatch.setattr(_post_retry(), "wait", wait_none())
    client = _client([httpx.ConnectError("refused"), _FakeResponse(_candidate("ok"))])

    assert await client.generate_text("p") == "ok"


async def test_retry_is_bounded(monkeypatch):
    monkeypatch.setattr(_post_retry(), "wait", wait_none())
    client = _client([_FakeResponse(status_code=503, text="down")] * 10)

    with pytest.raises(LLMTransientError):
        await client.generate_text("p")

    assert len(client._client.calls) == 4


# ══════════════════════════════════════════════════════════════════════════
# Streaming — the reason this client exists rather than routing via creditProxy
# ══════════════════════════════════════════════════════════════════════════


async def test_stream_yields_text_chunks_in_order():
    lines = [
        f"data: {json.dumps({'candidates': [{'content': {'parts': [{'text': chunk}]}}]})}"
        for chunk in ("Because ", "you liked ", "Dune.")
    ]
    client = _client(stream=_FakeStream(lines))

    chunks = [chunk async for chunk in client.stream_text("p")]

    assert chunks == ["Because ", "you liked ", "Dune."]
    assert "".join(chunks) == "Because you liked Dune."


async def test_stream_requests_sse_alt():
    client = _client(stream=_FakeStream([]))

    _ = [c async for c in client.stream_text("p")]

    call = client._client.calls[0]
    assert call["url"].endswith(":streamGenerateContent?alt=sse")
    assert call["method"] == "POST"


async def test_stream_ignores_non_data_lines_and_sentinels():
    lines = [
        "",
        ": keepalive comment",
        "event: message",
        "data: [DONE]",
        f"data: {json.dumps({'candidates': [{'content': {'parts': [{'text': 'x'}]}}]})}",
    ]
    client = _client(stream=_FakeStream(lines))

    assert [c async for c in client.stream_text("p")] == ["x"]


async def test_stream_survives_an_unparseable_chunk():
    """A malformed frame must not abort a partially-delivered explanation."""
    lines = [
        "data: {not json",
        f"data: {json.dumps({'candidates': [{'content': {'parts': [{'text': 'ok'}]}}]})}",
    ]
    client = _client(stream=_FakeStream(lines))

    assert [c async for c in client.stream_text("p")] == ["ok"]


async def test_stream_accumulates_usage_metadata():
    lines = [
        "data: "
        + json.dumps(
            {
                "candidates": [{"content": {"parts": [{"text": "hi"}]}}],
                "usageMetadata": {
                    "promptTokenCount": 7,
                    "candidatesTokenCount": 3,
                    "totalTokenCount": 10,
                },
            }
        )
    ]
    client = _client(stream=_FakeStream(lines))

    _ = [c async for c in client.stream_text("p")]

    assert client.usage.prompt_tokens == 7
    assert client.usage.output_tokens == 3


async def test_stream_maps_error_statuses():
    client = _client(stream=_FakeStream([], status_code=429, body=b"slow down"))

    with pytest.raises(LLMTransientError, match="429"):
        _ = [c async for c in client.stream_text("p")]


async def test_stream_permanent_error_is_not_transient():
    client = _client(stream=_FakeStream([], status_code=400, body=b"bad"))

    with pytest.raises(LLMError) as exc_info:
        _ = [c async for c in client.stream_text("p")]

    assert not isinstance(exc_info.value, LLMTransientError)


async def test_stream_skips_empty_text_parts():
    lines = [
        "data: " + json.dumps({"candidates": [{"content": {"parts": [{"text": ""}]}}]}),
        "data: "
        + json.dumps({"candidates": [{"content": {"parts": [{"text": "a"}]}}]}),
    ]
    client = _client(stream=_FakeStream(lines))

    assert [c async for c in client.stream_text("p")] == ["a"]


# ══════════════════════════════════════════════════════════════════════════
# Token accounting
# ══════════════════════════════════════════════════════════════════════════


def test_usage_accumulates_across_calls():
    usage = TokenUsage()

    usage.add(
        {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15}
    )
    usage.add(
        {"promptTokenCount": 20, "candidatesTokenCount": 8, "totalTokenCount": 28}
    )

    assert usage.calls == 2
    assert usage.prompt_tokens == 30
    assert usage.output_tokens == 13


def test_usage_counts_thinking_tokens_as_output():
    """candidatesTokenCount excludes thinking tokens, which are still billed."""
    usage = TokenUsage()

    usage.add(
        {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 40}
    )

    assert usage.output_tokens == 30


def test_usage_tolerates_missing_metadata():
    usage = TokenUsage()

    usage.add(None)
    usage.add({})

    assert usage.calls == 2
    assert usage.prompt_tokens == 0


def test_estimated_cost_matches_list_pricing():
    usage = TokenUsage()
    usage.prompt_tokens = 1_000_000
    usage.output_tokens = 1_000_000

    assert usage.estimated_usd() == pytest.approx(0.50)


def test_usage_reported_after_a_call():
    async def go():
        client = _client([_FakeResponse(_candidate("x"))])
        await client.generate_text("p")
        return client.usage.as_dict()

    import asyncio

    report = asyncio.run(go())
    assert report["calls"] == 1
    assert report["prompt_tokens"] == 10


# ══════════════════════════════════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════════════════════════════════


def test_build_client_returns_none_without_a_key():
    """None rather than raising, so the service still starts and serves the
    behavioral path, which runs entirely off precomputed vectors."""
    assert build_client("") is None


def test_build_client_returns_a_client_with_a_key():
    client = build_client("key", model="custom-model")

    assert isinstance(client, GeminiClient)
    assert client.model == "custom-model"
