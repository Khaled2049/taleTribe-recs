"""Derive Core Premise / Themes / Tone from a plot summary, with an LLM.

This is the expensive step, so its economics drive the design:

* **Batched ~8 books per call.** Not just for cost — 15,500 individual requests
  would blow the ~1,000 req/day free tier for over two weeks, whereas ~1,940
  batched calls fit in two days free or minutes paid. Batching also amortizes the
  vocabulary block, which is the bulk of the prompt.
* **Cached on the summary hash.** A crashed run resumes instead of re-billing.
  The key includes the prompt version, so changing the instructions correctly
  forces regeneration rather than silently reusing stale extractions.
* **Bounded concurrency.** The API key is shared with the story agent; an
  unthrottled backfill would degrade live chapter generation.

The `confidence` field is a safety gate, not decoration. A 60-word summary will
still yield a fluent, plausible, entirely invented premise — a poisoned vector
that looks perfectly healthy. Anything below the threshold is marked ineligible
instead of being embedded.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

from recommendation_engine.ingest.compose import summary_sha
from recommendation_engine.ingest.vocabularies import (
    filter_themes,
    filter_tones,
    is_genre_noise,
    prompt_vocabulary_block,
    split_misfiled,
)
from recommendation_engine.llm import GeminiClient, LLMError

logger = logging.getLogger(__name__)

# Bump to force regeneration of every cached normalization.
PROMPT_VERSION = 1

# Small enough that one bad book cannot poison many, large enough to amortize the
# ~1,400-token vocabulary block across the batch.
DEFAULT_BATCH_SIZE = 8

# Below this the extraction is treated as unreliable and the item is excluded
# from retrieval rather than embedded.
MIN_CONFIDENCE = 0.5

MAX_PREMISE_WORDS = 60
MAX_THEMES = 8
MAX_TONES = 4

SYSTEM_PROMPT = (
    "You extract structured metadata from book plot summaries for a "
    "recommendation engine. You are precise, literal, and you never invent "
    "detail that is not supported by the summary you are given."
)

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "core_premise": {"type": "string"},
                    "themes": {"type": "array", "items": {"type": "string"}},
                    "tone": {"type": "array", "items": {"type": "string"}},
                    "confidence": {"type": "number"},
                },
                "required": ["id", "core_premise", "themes", "tone", "confidence"],
            },
        }
    },
    "required": ["results"],
}


@dataclass
class Normalization:
    """One book's derived fields.

    `themes`/`tone` are vocabulary-filtered and ready to store on the item.
    `raw_themes`/`raw_tone` are what the model actually said, and are what gets
    written to the cache — see `NormalizationCache` for why that distinction is
    worth carrying.
    """

    core_premise: str
    themes: List[str]
    tone: List[str]
    confidence: float
    raw_themes: List[str] = field(default_factory=list)
    raw_tone: List[str] = field(default_factory=list)

    @property
    def is_reliable(self) -> bool:
        return self.confidence >= MIN_CONFIDENCE and bool(self.core_premise)


@dataclass
class NormalizeStats:
    requested: int = 0
    from_cache: int = 0
    generated: int = 0
    failed: int = 0
    low_confidence: int = 0
    dropped_themes: int = 0
    dropped_tones: int = 0
    batches: int = 0
    content_blocked: int = 0
    split_retries: int = 0
    vocabulary_violations: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        data = self.__dict__.copy()
        data["vocabulary_violations"] = dict(
            sorted(
                self.vocabulary_violations.items(),
                key=lambda kv: kv[1],
                reverse=True,
            )[:25]
        )
        return data


def build_prompt(batch: Sequence[tuple]) -> str:
    """Render the batch prompt.

    `batch` is a sequence of (id, title, author, summary).

    The instruction to return `confidence` low for thin summaries is doing real
    work here: without it the model treats every input as equally answerable and
    happily fabricates a premise for a 60-word stub.
    """
    books = []
    for identifier, title, author, summary in batch:
        byline = f" by {author}" if author else ""
        books.append(
            f"--- BOOK id={identifier} ---\n"
            f"Title: {title}{byline}\n"
            f"Summary: {summary}"
        )

    return (
        f"{prompt_vocabulary_block()}\n\n"
        "For EACH book below, extract:\n"
        f"- core_premise: at most {MAX_PREMISE_WORDS} words. State the central "
        "situation and conflict — not a plot recap, and no spoilers of the "
        "ending.\n"
        f"- themes: {MAX_THEMES} or fewer, chosen ONLY from the allowed themes "
        "list above, ordered most to least central.\n"
        f"- tone: {MAX_TONES} or fewer, chosen ONLY from the allowed tones list "
        "above.\n"
        "- confidence: 0.0 to 1.0, how well the summary supports this "
        "extraction. Use a value below 0.5 when the summary is too short, too "
        "vague, or is mostly bibliographic detail rather than plot. Do NOT "
        "invent premise or themes to fill gaps — report low confidence instead.\n\n"
        "Return one result object per book, echoing its id exactly.\n\n"
        + "\n\n".join(books)
    )


def _coerce(raw: dict, stats: NormalizeStats) -> Normalization:
    """Validate one result object and force it into the vocabularies."""
    premise = " ".join(str(raw.get("core_premise") or "").split())
    words = premise.split(" ")
    if len(words) > MAX_PREMISE_WORDS:
        premise = " ".join(words[:MAX_PREMISE_WORDS])

    raw_themes = [str(t) for t in (raw.get("themes") or [])]
    raw_tones = [str(t) for t in (raw.get("tone") or [])]

    # Relocate cross-axis terms before filtering, so a tone filed under themes is
    # recovered rather than discarded.
    moved_themes, moved_tones = split_misfiled(raw_themes, raw_tones)
    themes = filter_themes(moved_themes)[:MAX_THEMES]
    tones = filter_tones(moved_tones)[:MAX_TONES]

    # Track what the model invented despite being given the list. A rising count
    # here means the prompt or the vocabulary needs work — it is not something to
    # paper over with more aliases.
    for value in raw_themes:
        if filter_themes([value]) or filter_tones([value]):
            continue  # kept, or relocated to tone
        stats.dropped_themes += 1
        if is_genre_noise(value):
            continue  # a genre, not a missing theme — do not inflate the counter
        key = " ".join(value.split()).casefold()[:60]
        stats.vocabulary_violations[key] = stats.vocabulary_violations.get(key, 0) + 1
    for value in raw_tones:
        if not filter_tones([value]) and not filter_themes([value]):
            stats.dropped_tones += 1

    try:
        confidence = float(raw.get("confidence"))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    return Normalization(
        core_premise=premise,
        themes=themes,
        tone=tones,
        confidence=confidence,
        raw_themes=raw_themes,
        raw_tone=raw_tones,
    )


def is_content_block(exc: Exception) -> bool:
    """True when Gemini's safety filter rejected the prompt.

    Distinguished from other failures because the response is *deterministic* —
    retrying the identical batch fails identically. The only useful recovery is to
    isolate which book caused it.
    """
    text = str(exc)
    return "PROHIBITED_CONTENT" in text or "blockReason" in text


class Normalizer:
    """Runs the normalization pass with caching and bounded concurrency."""

    def __init__(
        self,
        client: GeminiClient,
        cache: Optional["NormalizationCache"] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        concurrency: int = 4,
    ) -> None:
        self._client = client
        self._cache = cache
        self._batch_size = max(1, batch_size)
        self._semaphore = asyncio.Semaphore(max(1, concurrency))
        self.stats = NormalizeStats()

    async def normalize_many(
        self, records: Sequence[tuple]
    ) -> Dict[str, Normalization]:
        """Normalize (id, title, author, summary) tuples, keyed by id.

        Cache lookups happen first and in bulk, so a resumed run makes zero API
        calls for work already done.
        """
        self.stats.requested += len(records)
        results: Dict[str, Normalization] = {}

        pending: List[tuple] = []
        if self._cache is not None:
            keys = {
                identifier: summary_sha(summary, PROMPT_VERSION)
                for identifier, _, _, summary in records
            }
            cached = await self._cache.get_many(set(keys.values()))
            for record in records:
                identifier = record[0]
                hit = cached.get(keys[identifier])
                if hit is not None:
                    results[identifier] = hit
                    self.stats.from_cache += 1
                else:
                    pending.append(record)
        else:
            pending = list(records)

        batches = [
            pending[i : i + self._batch_size]
            for i in range(0, len(pending), self._batch_size)
        ]
        if not batches:
            return results

        generated = await asyncio.gather(*(self._run_batch(batch) for batch in batches))
        for batch_result in generated:
            results.update(batch_result)

        if self._cache is not None and generated:
            to_store = {}
            for record in pending:
                identifier, _, _, summary = record
                normalization = results.get(identifier)
                if normalization is not None:
                    to_store[summary_sha(summary, PROMPT_VERSION)] = normalization
            if to_store:
                await self._cache.put_many(to_store, model=self._client.model)

        return results

    async def _run_batch(
        self, batch: Sequence[tuple], allow_split: bool = True
    ) -> Dict[str, Normalization]:
        # allow_split=False means we are already inside a split retry and the
        # semaphore is held by the caller — re-acquiring it would deadlock.
        if not allow_split:
            return await self._execute_batch(batch, allow_split=False)
        async with self._semaphore:
            return await self._execute_batch(batch, allow_split=True)

    async def _execute_batch(
        self, batch: Sequence[tuple], allow_split: bool
    ) -> Dict[str, Normalization]:
        self.stats.batches += 1
        prompt = build_prompt(batch)
        try:
            payload = await self._client.generate_json(
                prompt,
                response_schema=_RESPONSE_SCHEMA,
                system=SYSTEM_PROMPT,
                # ~120 tokens per book plus JSON overhead, with headroom so a
                # verbose batch is not truncated into unparseable JSON.
                max_output_tokens=400 * len(batch) + 256,
                temperature=0.2,
            )
        except (LLMError, json.JSONDecodeError) as exc:
            blocked = is_content_block(exc)
            if blocked and len(batch) > 1 and allow_split:
                # A safety block is a property of one book, but batching makes
                # it fatal for all 8. Measured on a real 2,000-book run: 12
                # blocked batches cost 96 records, of which ~84 were innocent
                # bystanders. Re-running the same batch cannot help — the block
                # is deterministic — so isolate the offender instead.
                logger.warning(
                    "normalization_batch_blocked size=%d — retrying individually",
                    len(batch),
                )
                self.stats.split_retries += 1
                recovered: Dict[str, Normalization] = {}
                for record in batch:
                    recovered.update(await self._run_batch([record], allow_split=False))
                return recovered

            if blocked:
                self.stats.content_blocked += len(batch)
            # One failed batch must not abort a 15k-record run. The records
            # stay uncached, so the next run retries exactly these.
            logger.warning(
                "normalization_batch_failed size=%d blocked=%s error=%s",
                len(batch),
                blocked,
                str(exc)[:160],
            )
            self.stats.failed += len(batch)
            return {}

        by_id = {str(identifier): None for identifier, _, _, _ in batch}
        out: Dict[str, Normalization] = {}
        for raw in payload.get("results") or []:
            identifier = str(raw.get("id") or "")
            if identifier not in by_id:
                # Model echoed an id that was not in the batch — ignore rather
                # than risk attaching a premise to the wrong book.
                continue
            normalization = _coerce(raw, self.stats)
            if not normalization.is_reliable:
                self.stats.low_confidence += 1
            out[identifier] = normalization
            self.stats.generated += 1

        missing = len(batch) - len(out)
        if missing:
            logger.warning(
                "normalization_batch_incomplete size=%d missing=%d",
                len(batch),
                missing,
            )
            self.stats.failed += missing
        return out


class NormalizationCache:
    """Postgres-backed cache for the normalization pass.

    **Stores the model's raw themes and tone, and applies the vocabulary filter on
    read.** That decoupling matters commercially: the vocabulary is meant to be
    tuned iteratively from the `vocabulary_violations` counter, and if the cache
    held post-filtered output then every tuning pass would mean re-generating the
    whole corpus at full token cost. Filtering on read makes vocabulary changes
    free — re-run the backfill and cached rows are simply re-filtered.

    Filtering is idempotent, so rows written by an earlier version that stored
    already-canonical terms still read back correctly.
    """

    def __init__(self, pool) -> None:
        self._pool = pool

    async def get_many(self, shas: Iterable[str]) -> Dict[str, Normalization]:
        shas = list(shas)
        if not shas:
            return {}
        rows = await self._pool.fetch(
            "SELECT summary_sha, core_premise, themes, tone, confidence "
            "FROM recommendations.normalization_cache "
            "WHERE summary_sha = ANY($1::text[]) AND prompt_ver = $2",
            shas,
            PROMPT_VERSION,
        )
        out: Dict[str, Normalization] = {}
        for row in rows:
            raw_themes = list(row["themes"] or [])
            raw_tone = list(row["tone"] or [])
            moved_themes, moved_tones = split_misfiled(raw_themes, raw_tone)
            out[row["summary_sha"]] = Normalization(
                core_premise=row["core_premise"] or "",
                # Re-filtered against the *current* vocabulary, not the one in
                # force when this row was generated.
                themes=filter_themes(moved_themes)[:MAX_THEMES],
                tone=filter_tones(moved_tones)[:MAX_TONES],
                confidence=float(row["confidence"] or 0.0),
                raw_themes=raw_themes,
                raw_tone=raw_tone,
            )
        return out

    async def put_many(self, entries: Dict[str, Normalization], model: str) -> None:
        if not entries:
            return
        await self._pool.executemany(
            "INSERT INTO recommendations.normalization_cache "
            "(summary_sha, core_premise, themes, tone, confidence, model, prompt_ver) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7) "
            "ON CONFLICT (summary_sha) DO UPDATE SET "
            "core_premise = EXCLUDED.core_premise, themes = EXCLUDED.themes, "
            "tone = EXCLUDED.tone, confidence = EXCLUDED.confidence, "
            "model = EXCLUDED.model, prompt_ver = EXCLUDED.prompt_ver",
            [
                (
                    sha,
                    norm.core_premise,
                    # Raw, deliberately — so a later vocabulary change re-filters
                    # instead of re-billing. Falls back to the filtered list for
                    # entries created before raw output was carried.
                    norm.raw_themes or norm.themes,
                    norm.raw_tone or norm.tone,
                    norm.confidence,
                    model,
                    PROMPT_VERSION,
                )
                for sha, norm in entries.items()
            ],
        )
