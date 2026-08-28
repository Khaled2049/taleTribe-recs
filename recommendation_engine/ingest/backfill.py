"""Load the CMU corpus into `recommendations.items`.

    # cheap smoke test, no API calls, no cost
    USE_MOCK=true python -m recommendation_engine.ingest.backfill --limit 200

    # validate output quality before committing to the full spend
    python -m recommendation_engine.ingest.backfill --limit 200

    # the real thing (~15.5k records, ~$2, resumable)
    python -m recommendation_engine.ingest.backfill --rebuild-index

Resumable at two levels: derived premises are cached on the summary hash, and
rows whose `embed_input_sha` is unchanged are not re-embedded. So a crashed or
interrupted run can simply be re-run, and it will only pay for what is actually
missing.

Work is processed in chunks rather than all at once so progress is committed
incrementally — an interrupted run leaves a usable partial catalog rather than
nothing.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, cast

from embedding_provider import (
    get_embedding_provider,
    verify_embedding_dimension,
)
from recommendation_engine.config import RecSettings
from recommendation_engine.db import Database
from recommendation_engine.embeddings import QueryEmbedder
from recommendation_engine.env import load_env
from recommendation_engine.ingest.cmu_parse import (
    DEFAULT_CORPUS_PATH,
    CmuRecord,
    ParseStats,
    iter_records,
)
from recommendation_engine.ingest.compose import (
    NormalizedItem,
    compose_embed_input,
    embed_input_sha,
)
from recommendation_engine.ingest.genre_crosswalk import load_crosswalk
from recommendation_engine.ingest.normalize_llm import (
    NormalizationCache,
    Normalizer,
)
from recommendation_engine.llm import build_client

logging.basicConfig(format="%(levelname)s %(message)s", level=logging.INFO)
logger = logging.getLogger("recs.backfill")

SOURCE = "cmu"
TASK_TYPE_DOCUMENT = "RETRIEVAL_DOCUMENT"

# How many records are normalized, embedded and committed per round trip through
# the pipeline. Bounds memory and gives incremental, resumable progress.
DEFAULT_CHUNK_SIZE = 200


@dataclass
class LoadStats:
    considered: int = 0
    skipped_unchanged: int = 0
    embedded: int = 0
    upserted: int = 0
    ineligible_low_confidence: int = 0
    ineligible_no_normalization: int = 0
    parse: dict = field(default_factory=dict)
    normalize: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _to_normalized_item(
    record: CmuRecord, categories: Sequence[str], normalization
) -> NormalizedItem:
    return NormalizedItem(
        title=record.title,
        author=record.author,
        genres=list(categories),
        core_premise=normalization.core_premise if normalization else None,
        themes=list(normalization.themes) if normalization else [],
        tone=list(normalization.tone) if normalization else [],
    )


async def _existing_shas(
    pool, source_ids: Sequence[str], model_id: str, task_type: str
) -> Dict[str, str]:
    """Map source_id -> embed_input_sha for rows already embedded *by this model*.

    Three conditions, each load-bearing:

    * `embedding IS NOT NULL` — a row stored without a vector (previously
      low-confidence) must be reconsidered.
    * `embed_model = $3` — vectors from different providers occupy different
      spaces. Without this, switching from the mock embedder to Gemini would leave
      md5-derived vectors sitting in a real corpus, matching nothing and dragging
      down every query, with no error to notice. The text hash is unchanged by a
      provider swap, so the hash alone cannot catch it.
    * `embed_task_type = $4` — RETRIEVAL_DOCUMENT and RETRIEVAL_QUERY project
      differently; a corpus must not mix them.
    """
    if not source_ids:
        return {}
    rows = await pool.fetch(
        "SELECT source_id, embed_input_sha FROM recommendations.items "
        "WHERE source = $1::recommendations.item_source "
        "AND source_id = ANY($2::text[]) AND embedding IS NOT NULL "
        "AND embed_model = $3 AND embed_task_type = $4",
        SOURCE,
        list(source_ids),
        model_id,
        task_type,
    )
    return {row["source_id"]: row["embed_input_sha"] for row in rows}


async def _upsert(pool, rows: List[tuple]) -> None:
    """Insert or update catalog rows.

    `updated_at` is refreshed but `created_at` is preserved, so the
    most-recently-updated ordering used elsewhere is not reset by a re-run.
    """
    if not rows:
        return
    await pool.executemany(
        """
        INSERT INTO recommendations.items (
            source, source_id, title, author, genres, raw_genres,
            core_premise, themes, tone, published_year, word_count,
            is_eligible, confidence,
            embed_input, embed_input_sha, embedding, embed_model, embed_task_type,
            embedded_at, updated_at
        ) VALUES (
            $1::recommendations.item_source, $2, $3, $4, $5::text[], $6::text[],
            $7, $8::text[], $9::text[], $10::int, $11::int,
            $12::boolean, $13::real,
            -- Explicit ::vector cast: the parameter is also referenced inside the
            -- CASE below, and without a cast Postgres cannot infer its type.
            $14, $15, $16::vector, $17, $18,
            CASE WHEN $16::vector IS NULL THEN NULL ELSE now() END, now()
        )
        ON CONFLICT (source, source_id) DO UPDATE SET
            title = EXCLUDED.title,
            author = EXCLUDED.author,
            genres = EXCLUDED.genres,
            raw_genres = EXCLUDED.raw_genres,
            core_premise = EXCLUDED.core_premise,
            themes = EXCLUDED.themes,
            tone = EXCLUDED.tone,
            published_year = EXCLUDED.published_year,
            word_count = EXCLUDED.word_count,
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


async def rebuild_hnsw_index(pool) -> None:
    """Drop and rebuild the HNSW graph.

    Building once over a loaded table is far faster than maintaining the graph
    across 15k individual inserts, and it produces a better-balanced graph.
    `maintenance_work_mem` is raised for this session only.
    """
    logger.info("rebuilding HNSW index (this takes a few minutes at full corpus size)")
    async with pool.acquire() as conn:
        await conn.execute("SET maintenance_work_mem = '512MB'")
        await conn.execute("DROP INDEX IF EXISTS recommendations.items_embedding_hnsw")
        await conn.execute(
            "CREATE INDEX items_embedding_hnsw ON recommendations.items "
            "USING hnsw (embedding vector_cosine_ops) "
            "WITH (m = 16, ef_construction = 200) WHERE is_eligible"
        )
    logger.info("HNSW index rebuilt")


async def run(
    corpus_path: Path = DEFAULT_CORPUS_PATH,
    limit: Optional[int] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    batch_size: int = 8,
    concurrency: int = 4,
    skip_normalization: bool = False,
    rebuild_index: bool = False,
    dry_run: bool = False,
) -> LoadStats:
    settings = RecSettings()
    stats = LoadStats()
    crosswalk = load_crosswalk()

    # Parse first. It needs no credentials, so --dry-run stays usable with no API
    # key at all — which is the point of a dry run.
    parse_stats = ParseStats()
    records = list(iter_records(corpus_path, parse_stats, limit=limit))
    stats.parse = parse_stats.as_dict()
    logger.info(
        "parsed corpus: %d emitted, %d stubs dropped, %d duplicates collapsed",
        parse_stats.emitted,
        parse_stats.stub_dropped,
        parse_stats.duplicate_dropped,
    )

    unknown = crosswalk.unknown(
        {label for record in records for label in record.raw_genres}
    )
    if unknown:
        # Loud and aggregate rather than a mid-run crash: a corpus refresh that
        # introduces labels must be noticed, but not by dying at record 9,000.
        logger.error(
            "genre labels missing from the crosswalk (%d): %s",
            len(unknown),
            sorted(unknown)[:20],
        )

    if dry_run:
        logger.info("dry run: parsed %d records, no API calls made", len(records))
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
        else build_client(settings.google_ai_studio_api_key, settings.recs_gemini_model)
    )
    if llm is None and not skip_normalization:
        raise RuntimeError(
            "normalization needs GOOGLE_AI_STUDIO_API_KEY; pass "
            "--skip-normalization to load titles and genres only (much weaker "
            "embeddings, useful only for plumbing tests)"
        )

    db = Database(write_dsn=settings.write_dsn, read_dsn=settings.read_dsn)
    await db.connect()
    run_id = None
    try:
        run_id = await db.write_pool.fetchval(
            "INSERT INTO recommendations.ingest_runs (kind, status) "
            "VALUES ('cmu_backfill', 'running') RETURNING id"
        )
        cache = NormalizationCache(db.write_pool)
        normalizer = (
            None
            if llm is None
            else Normalizer(llm, cache, batch_size=batch_size, concurrency=concurrency)
        )

        for start in range(0, len(records), chunk_size):
            chunk = records[start : start + chunk_size]
            await _process_chunk(
                db, chunk, crosswalk, normalizer, query_embedder, embedder, stats
            )
            logger.info(
                "progress %d/%d upserted=%d embedded=%d skipped=%d",
                min(start + chunk_size, len(records)),
                len(records),
                stats.upserted,
                stats.embedded,
                stats.skipped_unchanged,
            )

        if normalizer is not None:
            stats.normalize = normalizer.stats.as_dict()
        if llm is not None:
            stats.usage = llm.usage.as_dict()

        if rebuild_index:
            await rebuild_hnsw_index(db.write_pool)

        await db.write_pool.execute(
            "UPDATE recommendations.ingest_runs SET status = 'completed', "
            "finished_at = now(), counts = $2 WHERE id = $1",
            run_id,
            json.dumps(stats.as_dict()),
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
        if hasattr(embedder, "aclose"):
            await embedder.aclose()


async def _process_chunk(
    db,
    chunk: Sequence[CmuRecord],
    crosswalk,
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
                (record.wiki_id, record.title, record.author, record.summary)
                for record in chunk
            ]
        )

    model_id = embedder.model_id
    known_shas = await _existing_shas(
        db.read_pool,
        [record.wiki_id for record in chunk],
        model_id,
        TASK_TYPE_DOCUMENT,
    )

    pending: List[tuple] = []  # (record, categories, normalization, embed_input, sha)
    for record in chunk:
        categories = crosswalk.map_labels(record.raw_genres)
        normalization = normalizations.get(record.wiki_id)
        item = _to_normalized_item(record, categories, normalization)
        embed_input = compose_embed_input(item)
        sha = embed_input_sha(embed_input)

        if known_shas.get(record.wiki_id) == sha:
            stats.skipped_unchanged += 1
            continue
        pending.append((record, categories, normalization, embed_input, sha))

    if not pending:
        return

    # Only reliable rows are embedded. An unreliable one is still stored (so a
    # later run with a better prompt can fix it) but with no vector and
    # is_eligible=false, which keeps it out of the partial HNSW index entirely.
    embeddable = []
    for entry in pending:
        _, _, normalization, _, _ = entry
        if normalizer is None:
            embeddable.append(entry)
        elif normalization is None:
            stats.ineligible_no_normalization += 1
        elif normalization.is_reliable:
            embeddable.append(entry)
        else:
            stats.ineligible_low_confidence += 1

    vectors: Dict[str, list] = {}
    if embeddable:
        computed = await query_embedder.embed_documents(
            [entry[3] for entry in embeddable]
        )
        vectors = {entry[0].wiki_id: vec for entry, vec in zip(embeddable, computed)}
        stats.embedded += len(computed)

    rows = []
    for record, categories, normalization, embed_input, sha in pending:
        vector = vectors.get(record.wiki_id)
        rows.append(
            (
                SOURCE,
                record.wiki_id,
                record.title,
                record.author,
                categories,
                record.raw_genres,
                normalization.core_premise if normalization else None,
                list(normalization.themes) if normalization else [],
                list(normalization.tone) if normalization else [],
                record.published_year,
                record.summary_word_count,
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


class _BackfillArgs(argparse.Namespace):
    corpus: Path
    limit: int | None
    chunk_size: int
    batch_size: int
    concurrency: int
    skip_normalization: bool
    rebuild_index: bool
    dry_run: bool


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Load the CMU corpus.")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument(
        "--limit",
        type=int,
        help="Only load the first N records (validate quality first)",
    )
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Books per normalization LLM call"
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--skip-normalization",
        action="store_true",
        help="Embed title/author/genres only — no LLM. Much weaker vectors; for "
        "plumbing tests, not for quality evaluation.",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild the HNSW graph after loading (do this after a full load)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Parse only; no API calls, no writes"
    )
    args = cast(_BackfillArgs, parser.parse_args(argv))
    # Before RecSettings() is constructed anywhere, or the key in .env
    # stays invisible and this refuses to run for lack of it.
    load_env()

    if not args.corpus.exists():
        logger.error("corpus not found: %s", args.corpus)
        return 1

    try:
        stats = asyncio.run(
            run(
                corpus_path=args.corpus,
                limit=args.limit,
                chunk_size=args.chunk_size,
                batch_size=args.batch_size,
                concurrency=args.concurrency,
                skip_normalization=args.skip_normalization,
                rebuild_index=args.rebuild_index,
                dry_run=args.dry_run,
            )
        )
    except Exception as exc:
        logger.error("backfill failed: %s", exc)
        return 1

    print(json.dumps(stats.as_dict(), indent=2, default=str))
    if stats.usage:
        print(
            f"\nestimated spend: ${stats.usage.get('estimated_usd', 0):.4f} "
            f"over {stats.usage.get('calls', 0)} LLM calls",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    sys.exit(main())
