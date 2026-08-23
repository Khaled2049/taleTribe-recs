"""Query the recommender from the shell, without an HTTP server.

    # find books like one already in the catalog
    python -m recommendation_engine.query_cli --title "Animal Farm"

    # multi-seed: per-item retrieval + RRF, no vector averaging
    python -m recommendation_engine.query_cli --title "Dune" --title "Foundation"

    # free text (needs a real API key to mean anything)
    python -m recommendation_engine.query_cli --text "cozy small-town mystery, no gore"

    # filters, and show the score attribution
    python -m recommendation_engine.query_cli --title "Dune" --genre science-fiction --scores

Exists so ranking can be debugged without the API in the way, and it shares
`pipeline.rank` with the HTTP endpoints so the two cannot disagree.
"""

import argparse
import asyncio
import json
import logging
import sys
from typing import List, Optional

from embedding_provider import get_embedding_provider
from recommendation_engine.config import RecSettings
from recommendation_engine.db import Database
from recommendation_engine.embeddings import QueryEmbedder
from recommendation_engine.env import load_env
from recommendation_engine.pipeline import rank
from recommendation_engine.retrieval import RetrievalFilters, Retriever
from recommendation_engine.scoring import CatalogStats, ScoringConfig

logging.basicConfig(format="%(levelname)s %(message)s", level=logging.WARNING)
logger = logging.getLogger("recs.query")


def _print_table(result, show_scores: bool) -> None:
    if not result.items:
        print("no results")
        return

    print(
        f"\n{len(result.items)} results  "
        f"(from {result.candidates_considered} candidates, "
        f"diversity={result.diversity if result.diversity is not None else 'n/a'}"
        f"{', DEGRADED' if result.degraded else ''})\n"
    )
    for position, item in enumerate(result.items, start=1):
        flag = " [off-platform]" if item.off_platform else ""
        author = f" — {item.author}" if item.author else ""
        year = f" ({item.published_year})" if item.published_year else ""
        print(f"{position:2d}. {item.score:.4f}  {item.title}{author}{year}{flag}")
        if item.genres:
            print(f"      genres: {', '.join(item.genres)}")
        if item.themes:
            print(f"      themes: {', '.join(item.themes)}")
        if item.core_premise:
            print(f"      premise: {item.core_premise}")
        if show_scores:
            breakdown = item.breakdown
            weights = breakdown["weights"]
            print(
                f"      sem={breakdown['semantic']:.4f}*{weights['semantic']:.2f}  "
                f"pop={breakdown['popularity']:.4f}*{weights['popularity']:.2f}  "
                f"cf={breakdown['collaborative']:.4f}*{weights['collaborative']:.2f}  "
                f"alpha={breakdown['alpha']:.2f}  src={breakdown['source_weight']:.2f}"
            )
    print()


async def run(args) -> int:
    settings = RecSettings()
    embedder = get_embedding_provider(settings.google_ai_studio_api_key)
    query_embedder = QueryEmbedder(embedder)

    db = Database(write_dsn=settings.write_dsn, read_dsn=settings.read_dsn)
    await db.connect()
    try:
        health = await db.health()
        if not health.get("hnsw_index_present"):
            print(
                "warning: HNSW index missing — run the migration first",
                file=sys.stderr,
            )
        if not health.get("eligible_count"):
            print(
                "the catalog is empty; load some records first:\n"
                "  USE_MOCK=true python -m recommendation_engine.ingest.backfill "
                "--limit 300 --skip-normalization",
                file=sys.stderr,
            )
            return 1

        retriever = Retriever(
            db,
            ef_search=settings.recs_hnsw_ef_search,
            max_scan_tuples=settings.recs_hnsw_max_scan_tuples,
            statement_timeout_ms=settings.recs_statement_timeout_ms,
        )
        config = ScoringConfig.from_mapping(await db.load_config())
        stats = CatalogStats.from_row(await retriever.catalog_stats())

        query_vectors: List[List[float]] = []
        exclude: List[int] = []

        # Seed from catalog titles: resolve each, then search with its stored
        # vector. No embedding call needed, so this path works with no API key.
        for title in args.title or []:
            matches = await retriever.resolve_titles(title, limit=1)
            if not matches:
                print(f"no catalog match for {title!r}", file=sys.stderr)
                continue
            match = matches[0]
            print(f"seed: {match['title']} (id={match['id']})", file=sys.stderr)
            embedding = match["embedding"]
            to_list = getattr(embedding, "to_list", None)
            query_vectors.append(to_list() if callable(to_list) else list(embedding))
            # Never recommend the book the reader just named.
            exclude.append(match["id"])

        if args.text:
            if not query_embedder.available:
                print(
                    "--text needs an embedding provider; set "
                    "GOOGLE_AI_STUDIO_API_KEY or USE_MOCK=true (mock gives "
                    "meaningless neighbours)",
                    file=sys.stderr,
                )
                return 1
            query_vectors.append(await query_embedder.embed_query(args.text))

        if not query_vectors:
            print("no query vectors — falling back to popularity", file=sys.stderr)

        result = await rank(
            db=db,
            retriever=retriever,
            query_vectors=query_vectors,
            top_k=args.top_k,
            config=config,
            stats=stats,
            filters=RetrievalFilters(
                genres=args.genre or None,
                themes=args.theme or None,
                sources=args.source or None,
                max_word_count=args.max_words,
                exclude_ids=exclude,
            ),
        )

        if args.json:
            print(json.dumps(result.as_dict(include_breakdown=args.scores), indent=2))
        else:
            _print_table(result, args.scores)
        return 0
    finally:
        await db.aclose()
        if hasattr(embedder, "aclose"):
            await embedder.aclose()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Query the recommender locally.")
    parser.add_argument(
        "--title",
        action="append",
        help="Seed from a catalog title (repeatable → multi-seed RRF)",
    )
    parser.add_argument("--text", help="Free-text query (needs an embedder)")
    parser.add_argument("--genre", action="append", help="Filter by platform category")
    parser.add_argument("--theme", action="append", help="Filter by theme")
    parser.add_argument(
        "--source",
        action="append",
        choices=["platform", "cmu"],
        help="Filter by source",
    )
    parser.add_argument("--max-words", type=int, help="Maximum word count")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--scores", action="store_true", help="Show per-term score attribution"
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON")
    args = parser.parse_args(argv)
    # Before RecSettings() is constructed anywhere, or the key in .env
    # stays invisible and this refuses to run for lack of it.
    load_env()

    if not (args.title or args.text):
        print(
            "note: no --title/--text given; results will be popularity-ranked",
            file=sys.stderr,
        )

    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
