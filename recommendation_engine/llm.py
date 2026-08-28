"""Direct Gemini client for the recommendation service.

**Deliberately not routed through creditProxy.** The story agent sends every
generation through the gateway for credit metering, but this service needs two
things the gateway does not offer: real token-by-token streaming (so a "why
you'll like this" explanation appears as it is written and can be cancelled
mid-generation when a reader scrolls away) and structured JSON output for the
corpus normalization pass.

The consequence is explicit and accepted: explanations and HyDE are **not
credit-metered**. The cost controls are the deterministic explanation cache, a
dedicated per-user rate bucket, and bounded `max_output_tokens` — not the credit
ledger. If this service ever needs metering, the right fix is a streaming
passthrough in creditProxy, not a quiet unmetered path here.
"""

import json
import logging
import re
from collections.abc import Mapping
from typing import AsyncIterator, List, Optional, cast

import httpx
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential_jitter,
)

logger = logging.getLogger(__name__)

_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-2.5-flash-lite"

# Strips ```json fences when a model ignores responseMimeType and wraps output.
_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$")


class LLMError(RuntimeError):
    """Base class for generation failures."""


class LLMTransientError(LLMError):
    """Rate limit, upstream 5xx, or network fault — worth retrying."""


class LLMResponseError(LLMError):
    """The call succeeded but the payload was unusable (blocked, truncated, or
    not the JSON shape that was asked for)."""


JsonObject = dict[str, object]


def _as_object(value: object) -> JsonObject:
    if not isinstance(value, dict):
        return {}
    raw = cast(dict[object, object], value)
    return {str(key): item for key, item in raw.items()}


def _as_list(value: object | None) -> list[object]:
    return list(cast(list[object], value)) if isinstance(value, list) else []


def _as_int(value: object | None) -> int:
    if isinstance(value, (str, bytes, int, float)):
        return int(value)
    return 0


class TokenUsage:
    """Running token total, so a 15k-record backfill can report what it spent."""

    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.output_tokens = 0
        self.calls = 0

    def add(self, metadata: JsonObject | None) -> None:
        self.calls += 1
        if not metadata:
            return
        self.prompt_tokens += _as_int(metadata.get("promptTokenCount"))
        # candidatesTokenCount excludes thinking tokens, which are billed as
        # output; totalTokenCount - prompt captures both.
        total = _as_int(metadata.get("totalTokenCount"))
        prompt = _as_int(metadata.get("promptTokenCount"))
        candidates = _as_int(metadata.get("candidatesTokenCount"))
        self.output_tokens += max(candidates, total - prompt)

    def estimated_usd(
        self, input_per_million: float = 0.10, output_per_million: float = 0.40
    ) -> float:
        """Rough cost at gemini-2.5-flash-lite list pricing. Indicative only —
        it exists so a backfill prints a number before you run the full corpus."""
        return (
            self.prompt_tokens / 1_000_000 * input_per_million
            + self.output_tokens / 1_000_000 * output_per_million
        )

    def as_dict(self) -> dict[str, int | float]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "estimated_usd": round(self.estimated_usd(), 4),
        }


def _extract_text(payload: JsonObject) -> str:
    """Pull the text out of a generateContent response, with useful errors.

    An empty candidate list means the prompt was blocked; a MAX_TOKENS finish
    reason on a JSON call means the object is truncated and unparseable. Both
    deserve a specific message rather than a KeyError.
    """
    candidates = _as_list(payload.get("candidates"))
    if not candidates:
        feedback = _as_object(payload.get("promptFeedback") or {})
        raise LLMResponseError(f"no candidates returned (promptFeedback={feedback})")

    candidate = _as_object(candidates[0])
    content = _as_object(candidate.get("content") or {})
    parts = _as_list(content.get("parts"))
    text = "".join(
        value
        for part in parts
        if isinstance((value := _as_object(part).get("text")), str)
    )

    if not text:
        raise LLMResponseError(
            f"empty candidate text (finishReason={candidate.get('finishReason')})"
        )
    if candidate.get("finishReason") == "MAX_TOKENS":
        logger.warning("llm_output_truncated finish_reason=MAX_TOKENS")
    return text


class GeminiClient:
    """Minimal async Gemini client: JSON generation, plain text, and SSE streaming."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        timeout: float = 120.0,
    ) -> None:
        self._api_key = api_key
        self._model = model
        # One process-lifetime client so TLS handshakes are not paid per call —
        # the backfill makes ~2,000 of them.
        self._client = httpx.AsyncClient(timeout=timeout)
        self.usage = TokenUsage()

    @property
    def model(self) -> str:
        return self._model

    def _url(self, method: str, sse: bool = False) -> str:
        url = f"{_BASE}/{self._model}:{method}"
        return f"{url}?alt=sse" if sse else url

    def _body(
        self,
        prompt: str,
        system: Optional[str],
        max_output_tokens: int,
        temperature: float,
        response_schema: Mapping[str, object] | None,
    ) -> JsonObject:
        generation_config: JsonObject = {
            "maxOutputTokens": max_output_tokens,
            "temperature": temperature,
        }
        if response_schema is not None:
            # Native structured output. Far more reliable than asking for JSON in
            # the prompt and hoping — which is what the story agent has to do,
            # because creditProxy has no schema passthrough.
            generation_config["responseMimeType"] = "application/json"
            generation_config["responseSchema"] = response_schema

        body: JsonObject = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": generation_config,
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        return body

    @retry(
        stop=stop_after_attempt(4) | stop_after_delay(90),
        wait=wait_exponential_jitter(initial=2, max=20),
        retry=retry_if_exception_type(LLMTransientError),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _post(self, url: str, body: JsonObject) -> JsonObject:
        try:
            response = await self._client.post(
                url, headers={"x-goog-api-key": self._api_key}, json=body
            )
        except httpx.RequestError as exc:
            raise LLMTransientError(f"generation request failed: {exc}") from exc

        if response.status_code == 429 or response.status_code >= 500:
            raise LLMTransientError(
                f"generation upstream returned {response.status_code}: "
                f"{response.text[:200]}"
            )
        if response.status_code >= 400:
            raise LLMError(
                f"generation rejected with {response.status_code}: "
                f"{response.text[:200]}"
            )
        payload = _as_object(cast(object, response.json()))
        self.usage.add(_as_object(payload.get("usageMetadata") or {}))
        return payload

    async def generate_text(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_output_tokens: int = 512,
        temperature: float = 0.7,
    ) -> str:
        payload = await self._post(
            self._url("generateContent"),
            self._body(prompt, system, max_output_tokens, temperature, None),
        )
        return _extract_text(payload)

    async def generate_json(
        self,
        prompt: str,
        *,
        response_schema: Mapping[str, object],
        system: Optional[str] = None,
        max_output_tokens: int = 2048,
        temperature: float = 0.2,
    ) -> object:
        """Generate and parse structured JSON.

        Low temperature by default: the normalization pass wants consistent,
        reproducible extraction, not creative variety.
        """
        payload = await self._post(
            self._url("generateContent"),
            self._body(prompt, system, max_output_tokens, temperature, response_schema),
        )
        text = _extract_text(payload)
        cleaned = _FENCE.sub("", text).strip()
        try:
            return cast(object, json.loads(cleaned))
        except json.JSONDecodeError as exc:
            raise LLMResponseError(
                f"model did not return valid JSON ({exc}): {cleaned[:300]}"
            ) from exc

    async def stream_text(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        max_output_tokens: int = 256,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """Yield text chunks as they arrive.

        Not retried: a partially-streamed response cannot be transparently
        resumed, and re-running it would duplicate text the reader has already
        seen. The caller decides whether to start over.
        """
        body = self._body(prompt, system, max_output_tokens, temperature, None)
        try:
            async with self._client.stream(
                "POST",
                self._url("streamGenerateContent", sse=True),
                headers={"x-goog-api-key": self._api_key},
                json=body,
            ) as response:
                if response.status_code >= 400:
                    detail = (await response.aread())[:200]
                    if response.status_code == 429 or response.status_code >= 500:
                        raise LLMTransientError(
                            f"stream upstream returned {response.status_code}: {detail!r}"
                        )
                    raise LLMError(
                        f"stream rejected with {response.status_code}: {detail!r}"
                    )

                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[len("data:") :].strip()
                    if not raw or raw == "[DONE]":
                        continue
                    try:
                        chunk = _as_object(cast(object, json.loads(raw)))
                    except json.JSONDecodeError:
                        logger.warning("stream_chunk_unparseable chunk=%r", raw[:120])
                        continue

                    if "usageMetadata" in chunk:
                        self.usage.add(_as_object(chunk["usageMetadata"]))

                    for candidate_value in _as_list(chunk.get("candidates")):
                        candidate = _as_object(candidate_value)
                        content = _as_object(candidate.get("content") or {})
                        parts = _as_list(content.get("parts"))
                        for part in parts:
                            text = _as_object(part).get("text")
                            if isinstance(text, str) and text:
                                yield text
        except httpx.RequestError as exc:
            raise LLMTransientError(f"stream request failed: {exc}") from exc

    async def aclose(self) -> None:
        await self._client.aclose()


def build_client(
    api_key: str, model: str = DEFAULT_MODEL, timeout: float = 120.0
) -> Optional[GeminiClient]:
    """Construct a client, or None when no key is configured.

    None rather than raising, so a service with no API key still starts and
    serves the paths that need no LLM (behavioral recommendations run entirely
    off precomputed vectors). Callers check for None and degrade.
    """
    if not api_key:
        logger.warning(
            "no_gemini_api_key: HyDE and explanations are disabled; set "
            "GOOGLE_AI_STUDIO_API_KEY to enable them"
        )
        return None
    return GeminiClient(api_key=api_key, model=model, timeout=timeout)


__all__: List[str] = [
    "DEFAULT_MODEL",
    "GeminiClient",
    "LLMError",
    "LLMResponseError",
    "LLMTransientError",
    "TokenUsage",
    "build_client",
]
