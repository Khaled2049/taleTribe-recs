"""The explanation layer: "why you'll like this".

**The LLM never ranks.** By the time anything here runs, the result list is already
final — chosen by cosine similarity, fused, scored and diversified. This only
narrates a decision that was already made, which is what keeps ranking
reproducible and offline-evaluable.

Two delivery modes, both real:

* **Sync** (`explain_many`) — all explanations in one response. Complete and
  shippable on its own, and the only mode that works cleanly through a Firebase
  Functions proxy, which buffers responses.
* **Streaming** (`stream_explanations`) — one multiplexed SSE stream, events tagged
  by item id, cancellable mid-generation. One connection rather than ten.

Cost control is the deterministic cache key plus a dedicated rate bucket, because
this path deliberately bypasses creditProxy and so is *not* credit-metered.
"""

import asyncio
import hashlib
import json
import logging
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Optional

from recommendation_engine.llm import GeminiClient, LLMError

logger = logging.getLogger(__name__)

# Bump to invalidate every cached explanation.
PROMPT_VERSION = 1

_SYSTEM = (
    "You write one-sentence reasons a reader might enjoy a specific book, given "
    "what they asked for. You are concrete and specific to the book in question. "
    "You never invent plot details beyond what you are told."
)


def query_fingerprint(
    query: Optional[str] = None, seed_item_ids: Optional[Sequence[int]] = None
) -> str:
    """Stable identity for "what the reader asked".

    Seed ids are **sorted** so the same set of books in a different order is the
    same query — otherwise a shelf re-render with reordered seeds would miss the
    cache on every item.
    """
    parts: list[str] = []
    if query:
        parts.append(" ".join(query.split()).casefold())
    if seed_item_ids:
        parts.append(",".join(str(i) for i in sorted(seed_item_ids)))
    if not parts:
        parts.append("__popular__")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def cache_key(
    model: str,
    story_id: str,
    embed_input_sha: str,
    fingerprint: str,
) -> str:
    """Deterministic key for one (item, query) explanation.

    Every component earns its place:

    * `model` + `PROMPT_VERSION` — a different model or prompt writes different
      prose, so the old text must not be served.
    * `story_id` — which story.
    * `embed_input_sha` — **self-invalidating**: if the item's premise, themes or
      tone are re-derived, its hash changes and the stale explanation is abandoned
      automatically. No cache-busting logic to remember.
    * `fingerprint` — the same book explained for a different request needs a
      different sentence.
    """
    payload = f"{model}|{PROMPT_VERSION}|{story_id}|{embed_input_sha}|{fingerprint}"
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class ExplanationTarget:
    """One thing to explain. Built from a pipeline result item."""

    item_id: int
    story_id: str
    title: str
    author: Optional[str]
    genres: list[str]
    themes: list[str]
    tone: list[str]
    core_premise: Optional[str]
    embed_input_sha: str

    @classmethod
    def from_item(cls, item) -> "ExplanationTarget":
        return cls(
            item_id=item.id,
            story_id=item.story_id,
            title=item.title,
            author=item.author,
            genres=list(item.genres or []),
            themes=list(item.themes or []),
            tone=list(item.tone or []),
            core_premise=item.core_premise,
            embed_input_sha=item.embed_input_sha or "",
        )

    def key(self, model: str, fingerprint: str) -> str:
        return cache_key(model, self.story_id, self.embed_input_sha, fingerprint)


def build_prompt(target: ExplanationTarget, context: str) -> str:
    """Prompt for a single explanation.

    Only the *normalized* fields are given — never a raw plot summary. The model
    cannot spoil what it was not told, and it keeps the prompt small enough that
    the token cost of an explanation stays negligible.
    """
    lines = [f"BOOK: {target.title}"]
    if target.author:
        lines.append(f"AUTHOR: {target.author}")
    if target.genres:
        lines.append(f"GENRES: {', '.join(target.genres)}")
    if target.core_premise:
        lines.append(f"PREMISE: {target.core_premise}")
    if target.themes:
        lines.append(f"THEMES: {', '.join(target.themes)}")
    if target.tone:
        lines.append(f"TONE: {', '.join(target.tone)}")

    return (
        f"THE READER ASKED FOR: {context}\n\n"
        + "\n".join(lines)
        + "\n\nIn ONE sentence of at most 30 words, say why this reader might enjoy "
        "this book. Be specific to this book — name the premise, theme or tone that "
        "connects it to what they asked for. Do not summarize the plot, do not "
        "reveal the ending, and do not begin with the book's title."
    )


def describe_context(query: Optional[str], seed_titles: Optional[Sequence[str]]) -> str:
    """Human-readable version of the request, for the prompt."""
    if query:
        return query.strip()
    if seed_titles:
        return "books like " + ", ".join(seed_titles)
    return "popular books they have not read yet"


class ExplanationCache:
    """Postgres-backed cache. Hits are free and instant.

    Holds the `Database`, not a pool, and resolves `.read_pool` per call: the pool
    is opened by the app lifespan, which runs *after* the factory constructs this.
    """

    def __init__(self, db) -> None:
        self._db = db

    @property
    def _pool(self):
        return self._db.read_pool

    async def get_many(self, keys: Sequence[str]) -> dict[str, str]:
        if not keys:
            return {}
        rows = await self._pool.fetch(
            "SELECT cache_key, explanation FROM recommendations.explanation_cache "
            "WHERE cache_key = ANY($1::text[])",
            list(keys),
        )
        found = {row["cache_key"]: row["explanation"] for row in rows}
        if found:
            # Fire-and-forget: hit counting must never delay a response, and losing
            # a count to a cancelled request is harmless.
            asyncio.create_task(self._bump(list(found)))
        return found

    async def _bump(self, keys: Sequence[str]) -> None:
        try:
            await self._pool.execute(
                "UPDATE recommendations.explanation_cache "
                "SET hit_count = hit_count + 1 WHERE cache_key = ANY($1::text[])",
                list(keys),
            )
        except Exception as exc:
            logger.debug("explanation_hit_count_failed error=%s", exc)

    async def put(self, key: str, item_id: int, explanation: str, model: str) -> None:
        try:
            await self._pool.execute(
                "INSERT INTO recommendations.explanation_cache "
                "(cache_key, item_id, explanation, model, prompt_ver) "
                "VALUES ($1, $2, $3, $4, $5) "
                "ON CONFLICT (cache_key) DO NOTHING",
                key,
                item_id,
                explanation,
                model,
                PROMPT_VERSION,
            )
        except Exception as exc:
            # A cache write failure must not fail a request that already has its
            # answer. Worst case the next identical request regenerates.
            logger.warning("explanation_cache_write_failed error=%s", exc)


async def explain_many(
    client: Optional[GeminiClient],
    cache: Optional[ExplanationCache],
    targets: Sequence[ExplanationTarget],
    context: str,
    fingerprint: str,
    max_output_tokens: int = 256,
    concurrency: int = 4,
) -> list[dict]:
    """Sync mode: explanations for every target, cached ones served free.

    A per-item failure yields `explanation: None` rather than failing the batch —
    a shelf with nine reasons and one blank is far better than an error page.
    """
    if not targets:
        return []
    if client is None:
        return [
            {"item_id": t.item_id, "explanation": None, "cached": False}
            for t in targets
        ]

    keys = {t.item_id: t.key(client.model, fingerprint) for t in targets}
    cached = await cache.get_many(list(keys.values())) if cache else {}

    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def one(target: ExplanationTarget) -> dict:
        key = keys[target.item_id]
        hit = cached.get(key)
        if hit is not None:
            return {"item_id": target.item_id, "explanation": hit, "cached": True}
        async with semaphore:
            try:
                text = await client.generate_text(
                    build_prompt(target, context),
                    system=_SYSTEM,
                    max_output_tokens=max_output_tokens,
                    temperature=0.6,
                )
            except LLMError as exc:
                logger.warning(
                    "explanation_failed item_id=%s error=%s",
                    target.item_id,
                    str(exc)[:160],
                )
                return {
                    "item_id": target.item_id,
                    "explanation": None,
                    "cached": False,
                }
        text = " ".join(text.split())
        if cache:
            await cache.put(key, target.item_id, text, client.model)
        return {"item_id": target.item_id, "explanation": text, "cached": False}

    return list(await asyncio.gather(*(one(t) for t in targets)))


def sse(event: str, data: dict) -> str:
    """Format one Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def stream_explanations(
    client: Optional[GeminiClient],
    cache: Optional[ExplanationCache],
    targets: Sequence[ExplanationTarget],
    context: str,
    fingerprint: str,
    is_disconnected=None,
    max_output_tokens: int = 256,
) -> AsyncIterator[str]:
    """Streaming mode: one multiplexed SSE stream over every target.

    Items are streamed **sequentially** rather than concurrently, deliberately: a
    reader reads top to bottom, so finishing the first explanation quickly matters
    more than finishing all of them slightly sooner. It also keeps event ordering
    meaningful and holds only one LLM call open at a time.

    Cancellation is the point of this mode. `is_disconnected` is polled between
    items and between chunks, so a reader who scrolls away stops paying for tokens
    mid-sentence. Partial text is **never cached** — a truncated explanation must
    not become the permanent answer.
    """
    if client is None:
        yield sse("error", {"message": "explanations are not configured"})
        yield sse("done", {"completed": 0})
        return

    completed = 0
    for target in targets:
        if is_disconnected is not None and await is_disconnected():
            logger.info("explanation_stream_cancelled completed=%d", completed)
            return

        key = target.key(client.model, fingerprint)
        hit = (await cache.get_many([key])).get(key) if cache else None
        if hit is not None:
            yield sse("explanation", {"item_id": target.item_id, "delta": hit})
            yield sse("item_done", {"item_id": target.item_id, "cached": True})
            completed += 1
            continue

        pieces: list[str] = []
        cancelled = False
        try:
            async for chunk in client.stream_text(
                build_prompt(target, context),
                system=_SYSTEM,
                max_output_tokens=max_output_tokens,
                temperature=0.6,
            ):
                if is_disconnected is not None and await is_disconnected():
                    cancelled = True
                    break
                pieces.append(chunk)
                yield sse("explanation", {"item_id": target.item_id, "delta": chunk})
        except LLMError as exc:
            logger.warning(
                "explanation_stream_failed item_id=%s error=%s",
                target.item_id,
                str(exc)[:160],
            )
            yield sse("item_error", {"item_id": target.item_id})
            continue

        if cancelled:
            logger.info("explanation_stream_cancelled mid_item=%s", target.item_id)
            return

        text = " ".join("".join(pieces).split())
        if text and cache:
            # Only a completed generation is cached.
            await cache.put(key, target.item_id, text, client.model)
        yield sse("item_done", {"item_id": target.item_id, "cached": False})
        completed += 1

    yield sse("done", {"completed": completed})
