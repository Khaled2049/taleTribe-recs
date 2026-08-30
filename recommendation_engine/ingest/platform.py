"""Load published TaleTribe stories into `recommendations.items`.

    # what would be read, no API calls, no key needed
    python -m recommendation_engine.ingest.platform --dry-run

    # deterministic offline embeddings — exercises the plumbing, not quality
    USE_MOCK=true python -m recommendation_engine.ingest.platform --skip-normalization

    # the real thing (needs GOOGLE_AI_STUDIO_API_KEY)
    python -m recommendation_engine.ingest.platform

Replaces the CMU corpus backfill. Stories are read straight out of story-data's
database — the `recommendations` schema is a schema in it, so this is a query
rather than an export.

**Incremental by default.** Each run records the newest `updated_at` it saw in
`recommendations.ingest_runs.cursor`, and the next run reads only past it. A
`--full` run ignores the cursor. Either way the work is cheap when nothing has
changed: a row whose `embed_input_sha` is unchanged is never re-embedded, so
re-reading the whole catalog costs one query and no tokens.

**Eligibility is reconciled, not assumed.** The read only sees published
stories, so unpublishing is invisible to it — the story simply stops appearing.
`_retire_unpublished` closes that gap with a join against `stories`, and
`_existing_shas` deliberately ignores rows that are already ineligible so that
re-publishing an unchanged story re-embeds it instead of being skipped as
unchanged. Deletion needs no handling at all: `items.story_id` is a foreign key
with `ON DELETE CASCADE`.
"""

import argparse
import asyncio
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from embedding_provider import (
    get_embedding_provider,
    verify_embedding_dimension,
)
from recommendation_engine.config import RecSettings
from recommendation_engine.db import Database
from recommendation_engine.embeddings import QueryEmbedder
from recommendation_engine.env import load_env
from recommendation_engine.ingest.compose import (
    NormalizedItem,
    compose_embed_input,
    embed_input_sha,
)
from recommendation_engine.ingest.normalize_llm import (
    NormalizationCache,
    Normalizer,
)
from recommendation_engine.llm import build_client

logging.basicConfig(format="%(levelname)s %(message)s", level=logging.INFO)
logger = logging.getLogger("recs.ingest.platform")

RUN_KIND = "platform_sync"
TASK_TYPE_DOCUMENT = "RETRIEVAL_DOCUMENT"

# How many stories are normalized, embedded and committed per round trip.
# Bounds memory and gives incremental, resumable progress.
DEFAULT_CHUNK_SIZE = 200

# Long enough to carry premise and theme, short enough not to spend the tail of
# a novel's worth of chapter summaries on one extraction.
MAX_SUMMARY_WORDS = 1200


@dataclass
class StoryRecord:
    """One published story, as the catalog needs it."""

    story_id: str
    title: str
    author: Optional[str]
    description: str
    category: Optional[str]
    target_audience: Optional[str]
    language: Optional[str]
    tags: list[str]
    word_count: int
    chapter_count: int
    created_at: datetime
    updated_at: datetime
    # Concatenated chapter summaries, when the summarize path has populated
    # them. Sparse by nature — it is opportunistic, not guaranteed.
    chapter_summary: Optional[str] = None


@dataclass
class LoadStats:
    considered: int = 0
    skipped_unchanged: int = 0
    embedded: int = 0
    upserted: int = 0
    retired: int = 0
    ineligible_low_confidence: int = 0
    ineligible_no_normalization: int = 0
    normalize: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


# ══════════════════════════════════════════════════════════════════════════
# Reading from story-data
# ══════════════════════════════════════════════════════════════════════════

# Scalar subqueries rather than two LEFT JOINs: joining `story_tags` and
# `chapters` in one query multiplies their rows together, which silently
# inflates `sum(word_count)` by the number of tags. `count(DISTINCT ...)` hides
# that for the counts and not at all for the sum.
_READ_SQL = """
SELECT s.id, s.title, s.author_name, s.description, s.category,
       s.target_audience, s.language, s.created_at, s.updated_at,
       COALESCE((SELECT array_agg(st.tag ORDER BY st.tag)
                   FROM story_tags st WHERE st.story_id = s.id), '{}') AS tags,
       (SELECT count(*) FROM chapters c WHERE c.story_id = s.id) AS chapter_count,
       (SELECT COALESCE(sum(c.word_count), 0)
          FROM chapters c WHERE c.story_id = s.id) AS word_count,
       (SELECT string_agg(cs.summary, ' ' ORDER BY c.position)
          FROM chapters c
          JOIN chapter_summaries cs ON cs.chapter_id = c.id
         WHERE c.story_id = s.id) AS chapter_summary
  FROM stories s
 WHERE s.is_published
   AND ($1::timestamptz IS NULL OR s.updated_at > $1::timestamptz)
 ORDER BY s.updated_at, s.id
"""


async def read_stories(
    pool, since: Optional[datetime] = None, limit: Optional[int] = None
) -> list[StoryRecord]:
    """Published stories, oldest change first so a cursor can advance safely."""
    sql = _READ_SQL + (f" LIMIT {int(limit)}" if limit else "")
    rows = await pool.fetch(sql, since)
    return [
        StoryRecord(
            story_id=str(row["id"]),
            title=row["title"],
            author=row["author_name"] or None,
            description=row["description"] or "",
            category=row["category"] or None,
            target_audience=row["target_audience"] or None,
            language=row["language"] or None,
            tags=list(row["tags"] or []),
            word_count=int(row["word_count"] or 0),
            chapter_count=int(row["chapter_count"] or 0),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            chapter_summary=row["chapter_summary"],
        )
        for row in rows
    ]


async def _last_cursor(pool) -> Optional[datetime]:
    """The newest `updated_at` a completed run has already covered."""
    return await pool.fetchval(
        "SELECT cursor::timestamptz FROM recommendations.ingest_runs "
        "WHERE kind = $1 AND status = 'completed' AND cursor IS NOT NULL "
        "ORDER BY finished_at DESC LIMIT 1",
        RUN_KIND,
    )


# ══════════════════════════════════════════════════════════════════════════
# Composing what gets embedded
# ══════════════════════════════════════════════════════════════════════════


def summary_text(record: StoryRecord) -> str:
    """The text handed to the LLM to derive premise, themes and tone.

    Chapter summaries when they exist, because a description is a marketing
    blurb and a summary is what actually happens; otherwise the description
    plus tags, which is all there is. Tags are included here rather than in
    `genres` on purpose: `genres` is a filter dimension and stories carry a
    controlled `category` for it, while tags are free text and belong in the
    material the model reads.
    """
    parts: list[str] = []
    if record.description.strip():
        parts.append(record.description.strip())
    if record.chapter_summary and record.chapter_summary.strip():
        parts.append(record.chapter_summary.strip())
    if record.tags:
        parts.append("Tags: " + ", ".join(record.tags))
    text = " ".join(parts)
    words = text.split()
    if len(words) > MAX_SUMMARY_WORDS:
        text = " ".join(words[:MAX_SUMMARY_WORDS])
    return text


def to_normalized_item(record: StoryRecord, normalization) -> NormalizedItem:
    return NormalizedItem(
        title=record.title,
        author=record.author,
        genres=[record.category] if record.category else [],
        core_premise=normalization.core_premise if normalization else None,
        themes=list(normalization.themes) if normalization else [],
        tone=list(normalization.tone) if normalization else [],
    )


# ══════════════════════════════════════════════════════════════════════════
# Writing to the catalog
# ══════════════════════════════════════════════════════════════════════════


async def _existing_shas(
    pool, story_ids: Sequence[str], model_id: str, task_type: str
) -> dict[str, str]:
    """Map story_id -> embed_input_sha for rows already embedded *by this model*.

    Four conditions, each load-bearing:

    * `embedding IS NOT NULL` — a row stored without a vector (previously
      low-confidence) must be reconsidered.
    * `is_eligible` — a row retired by `_retire_unpublished` must be
      reconsidered too. Without this, a story unpublished and then republished
      unchanged would keep its old sha, be skipped as unchanged, and stay
      invisible forever.
    * `embed_model = $2` — vectors from different providers occupy different
      spaces. Without this, switching from the mock embedder to Gemini would
      leave md5-derived vectors in a real corpus, matching nothing and dragging
      down every query, with no error to notice. The text hash is unchanged by a
      provider swap, so the hash alone cannot catch it.
    * `embed_task_type = $3` — RETRIEVAL_DOCUMENT and RETRIEVAL_QUERY project
      differently; a corpus must not mix them.
    """
    if not story_ids:
        return {}
    rows = await pool.fetch(
        "SELECT story_id, embed_input_sha FROM recommendations.items "
        "WHERE story_id = ANY($1::uuid[]) AND embedding IS NOT NULL "
        "AND is_eligible AND embed_model = $2 AND embed_task_type = $3",
        list(story_ids),
        model_id,
        task_type,
    )
    return {str(row["story_id"]): row["embed_input_sha"] for row in rows}


async def _upsert(pool, rows: list[tuple]) -> None:
    """Insert or update catalog rows.

    `updated_at` is refreshed but `created_at` is preserved, so the
    most-recently-updated ordering used elsewhere is not reset by a re-run.
    """
    if not rows:
        return
    await pool.executemany(
        """
        INSERT INTO recommendations.items (
            story_id, title, author, genres,
            core_premise, themes, tone, published_year, word_count,
            chapter_count, language, target_audience,
            is_eligible, confidence,
            embed_input, embed_input_sha, embedding, embed_model, embed_task_type,
            embedded_at, updated_at
        ) VALUES (
            $1::uuid, $2, $3, $4::text[],
            $5, $6::text[], $7::text[], $8::int, $9::int,
            $10::int, $11, $12,
            $13::boolean, $14::real,
            -- Explicit ::vector cast: the parameter is also referenced inside the
            -- CASE below, and without a cast Postgres cannot infer its type.
            $15, $16, $17::vector, $18, $19,
            CASE WHEN $17::vector IS NULL THEN NULL ELSE now() END, now()
        )
        ON CONFLICT (story_id) DO UPDATE SET
            title = EXCLUDED.title,
            author = EXCLUDED.author,
            genres = EXCLUDED.genres,
            core_premise = EXCLUDED.core_premise,
            themes = EXCLUDED.themes,
            tone = EXCLUDED.tone,
            published_year = EXCLUDED.published_year,
            word_count = EXCLUDED.word_count,
            chapter_count = EXCLUDED.chapter_count,
            language = EXCLUDED.language,
            target_audience = EXCLUDED.target_audience,
            is_eligible = EXCLUDED.is_eligible,
            confidence = EXCLUDED.confidence,
            embed_input = EXCLUDED.embed_input,
            embed_input_sha = EXCLUDED.embed_input_sha,
            embedding = COALESCE(EXCLUDED.embedding, recommendations.items.embedding),
            embed_model = EXCLUDED.embed_model,
            embed_task_type = EXCLUDED.embed_task_type,
            embedded_at = CASE
                WHEN EXCLUDED.embedding IS NULL THEN recommendations.items.embedded_at
                ELSE now()
            END,
            updated_at = now()
        """,
        rows,
    )


async def _retire_unpublished(pool) -> int:
    """Mark items whose story is no longer published as ineligible.

    The read never sees them, so nothing else would. Ineligible rows leave the
    partial HNSW index entirely rather than being deleted, so re-publishing is
    an update rather than a re-ingest — and the row keeps its normalization,
    which is the expensive part.
    """
    result = await pool.execute(
        "UPDATE recommendations.items i SET is_eligible = false, updated_at = now() "
        "FROM stories s WHERE s.id = i.story_id AND NOT s.is_published "
        "AND i.is_eligible"
    )
    return int(result.rsplit(" ", 1)[-1]) if result else 0


async def rebuild_hnsw_index(pool) -> None:
    """Drop and rebuild the HNSW graph.

    Not part of a normal run. It was worth it for a 15k-row bulk load; against
    incremental ingest it costs more than it saves, and it must never run
    against the shared database while traffic is being served — the index is
    gone for the duration, so every query falls back to a sequential scan.
    """
    logger.info("rebuilding HNSW index")
    async with pool.acquire() as conn:
        await conn.execute("SET maintenance_work_mem = '512MB'")
        await conn.execute(
            "DROP INDEX IF EXISTS recommendations.items_embedding_hnsw_idx"
        )
        await conn.execute(
            "CREATE INDEX items_embedding_hnsw_idx ON recommendations.items "
            "USING hnsw (embedding vector_cosine_ops) "
            "WITH (m = 16, ef_construction = 200) WHERE is_eligible"
        )
    logger.info("HNSW index rebuilt")


# ══════════════════════════════════════════════════════════════════════════
# Orchestration
# ══════════════════════════════════════════════════════════════════════════


async def _process_chunk(
    db,
    chunk: Sequence[StoryRecord],
    normalizer: Optional[Normalizer],
    query_embedder: QueryEmbedder,
    embedder,
    stats: LoadStats,
) -> None:
    stats.considered += len(chunk)

    normalizations = {}
    if normalizer is not None:
        normalizations = await normalizer.normalize_many(
            [
                (r.story_id, r.title, r.author, summary_text(r))
                for r in chunk
                if summary_text(r)
            ]
        )

    model_id = embedder.model_id
    known_shas = await _existing_shas(
        db.read_pool, [r.story_id for r in chunk], model_id, TASK_TYPE_DOCUMENT
    )

    pending: list[tuple] = []  # (record, normalization, embed_input, sha)
    for record in chunk:
        normalization = normalizations.get(record.story_id)
        item = to_normalized_item(record, normalization)
        embed_input = compose_embed_input(item)
        sha = embed_input_sha(embed_input)

        if known_shas.get(record.story_id) == sha:
            stats.skipped_unchanged += 1
            continue
        pending.append((record, normalization, embed_input, sha))

    if not pending:
        return

    # Only reliable rows are embedded. An unreliable one is still stored (so a
    # later run with a better prompt or a fuller description can fix it) but with
    # no vector and is_eligible=false, which keeps it out of the partial HNSW
    # index entirely. A one-sentence description lands here, by design.
    embeddable = []
    for entry in pending:
        _, normalization, _, _ = entry
        if normalizer is None:
            embeddable.append(entry)
        elif normalization is None:
            stats.ineligible_no_normalization += 1
        elif normalization.is_reliable:
            embeddable.append(entry)
        else:
            stats.ineligible_low_confidence += 1

    vectors: dict[str, list] = {}
    if embeddable:
        computed = await query_embedder.embed_documents(
            [entry[2] for entry in embeddable]
        )
        vectors = {entry[0].story_id: vec for entry, vec in zip(embeddable, computed)}
        stats.embedded += len(computed)

    rows = []
    for record, normalization, embed_input, sha in pending:
        vector = vectors.get(record.story_id)
        rows.append(
            (
                record.story_id,
                record.title,
                record.author,
                [record.category] if record.category else [],
                normalization.core_premise if normalization else None,
                list(normalization.themes) if normalization else [],
                list(normalization.tone) if normalization else [],
                record.created_at.year if record.created_at else None,
                record.word_count,
                record.chapter_count,
                record.language,
                record.target_audience,
                vector is not None,
                normalization.confidence if normalization else None,
                embed_input,
                sha,
                vector,
                model_id,
                TASK_TYPE_DOCUMENT,
            )
        )

    await _upsert(db.write_pool, rows)
    stats.upserted += len(rows)


async def run(
    limit: Optional[int] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    batch_size: int = 8,
    concurrency: int = 4,
    skip_normalization: bool = False,
    rebuild_index: bool = False,
    full: bool = False,
    dry_run: bool = False,
) -> LoadStats:
    settings = RecSettings()
    stats = LoadStats()

    db = Database(write_dsn=settings.write_dsn, read_dsn=settings.read_dsn)
    await db.connect()

    embedder = None
    llm = None
    run_id = None
    try:
        since = None if full else await _last_cursor(db.read_pool)
        records = await read_stories(db.read_pool, since=since, limit=limit)
        logger.info(
            "read %d published stories%s",
            len(records),
            "" if since is None else f" changed since {since.isoformat()}",
        )

        if dry_run:
            logger.info("dry run: no API calls made, nothing written")
            stats.considered = len(records)
            return stats

        embedder = get_embedding_provider(settings.google_ai_studio_api_key)
        verify_embedding_dimension(embedder)
        if embedder is None:
            raise RuntimeError(
                "no embedding provider; set GOOGLE_AI_STUDIO_API_KEY or USE_MOCK=true"
            )
        query_embedder = QueryEmbedder(embedder)

        llm = (
            None
            if skip_normalization
            else build_client(
                settings.google_ai_studio_api_key, settings.recs_gemini_model
            )
        )
        if llm is None and not skip_normalization:
            raise RuntimeError(
                "normalization needs GOOGLE_AI_STUDIO_API_KEY; pass "
                "--skip-normalization to load titles and categories only (much "
                "weaker embeddings, useful only for plumbing tests)"
            )

        run_id = await db.write_pool.fetchval(
            "INSERT INTO recommendations.ingest_runs (kind, status) "
            "VALUES ($1, 'running') RETURNING id",
            RUN_KIND,
        )
        cache = NormalizationCache(db.write_pool)
        normalizer = (
            None
            if llm is None
            else Normalizer(llm, cache, batch_size=batch_size, concurrency=concurrency)
        )

        for start in range(0, len(records), chunk_size):
            chunk = records[start : start + chunk_size]
            await _process_chunk(db, chunk, normalizer, query_embedder, embedder, stats)
            logger.info(
                "progress %d/%d upserted=%d embedded=%d skipped=%d",
                min(start + chunk_size, len(records)),
                len(records),
                stats.upserted,
                stats.embedded,
                stats.skipped_unchanged,
            )

        stats.retired = await _retire_unpublished(db.write_pool)

        if normalizer is not None:
            stats.normalize = normalizer.stats.as_dict()
        if llm is not None:
            stats.usage = llm.usage.as_dict()

        if rebuild_index:
            await rebuild_hnsw_index(db.write_pool)

        # Advance the cursor only on success, and only as far as was actually
        # read — a partial run must not skip what it never saw.
        cursor = records[-1].updated_at.isoformat() if records else None
        await db.write_pool.execute(
            "UPDATE recommendations.ingest_runs SET status = 'completed', "
            "finished_at = now(), counts = $2, "
            "cursor = COALESCE($3, cursor) WHERE id = $1",
            run_id,
            json.dumps(stats.as_dict()),
            cursor,
        )
        return stats
    except Exception as exc:
        if run_id is not None:
            await db.write_pool.execute(
                "UPDATE recommendations.ingest_runs SET status = 'failed', "
                "finished_at = now(), error = $2 WHERE id = $1",
                run_id,
                str(exc)[:2000],
            )
        raise
    finally:
        await db.aclose()
        if llm is not None:
            await llm.aclose()
        if embedder is not None and hasattr(embedder, "aclose"):
            await embedder.aclose()


class _IngestArgs(argparse.Namespace):
    limit: Optional[int]
    chunk_size: int
    batch_size: int
    concurrency: int
    skip_normalization: bool
    rebuild_index: bool
    full: bool
    dry_run: bool


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Load published TaleTribe stories into the recommendation catalog."
    )
    parser.add_argument("--limit", type=int, default=None, help="Stop after N stories")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--skip-normalization",
        action="store_true",
        help="Skip the LLM pass (title + category only; plumbing tests)",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild the HNSW graph afterwards. Never against live traffic.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Ignore the stored cursor and re-read every published story",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be read; no API calls, nothing written",
    )
    args = parser.parse_args(argv, namespace=_IngestArgs())

    load_env()
    try:
        stats = asyncio.run(
            run(
                limit=args.limit,
                chunk_size=args.chunk_size,
                batch_size=args.batch_size,
                concurrency=args.concurrency,
                skip_normalization=args.skip_normalization,
                rebuild_index=args.rebuild_index,
                full=args.full,
                dry_run=args.dry_run,
            )
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("ingest failed: %s", exc)
        return 1
    logger.info("done: %s", json.dumps(stats.as_dict(), default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
