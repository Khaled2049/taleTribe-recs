"""Generate synthetic reader behaviour, and refresh what scoring reads.

    # generate readers, load them, and refresh everything scoring reads
    python -m recommendation_engine.sync.seed --readers 200

    # generate and inspect without touching the database
    python -m recommendation_engine.sync.seed --readers 20 --dry-run

    # remove every synthetic reader
    python -m recommendation_engine.sync.seed --purge

    # recompute aggregates without changing interactions
    python -m recommendation_engine.sync.seed --refresh-only

**Real reader signals do not arrive here.** story-data derives them directly
(`story-data sync-recs`); the JSONL import/export this script used to offer was
the contract between the two services and is gone. `--refresh-only` is the flag
that matters now: run it after a story-data sync to recompute `item_stats`,
`user_taste` and, optionally, the co-occurrence matrix over whatever signals are
in the table — synthetic, real, or both.
"""

import argparse
import asyncio
import json
import logging
import sys
from typing import Optional, cast

from recommendation_engine.config import RecSettings
from recommendation_engine.db import Database
from recommendation_engine.env import load_env
from recommendation_engine.scoring import ScoringConfig
from recommendation_engine.sync import interactions as inter
from recommendation_engine.sync import stats as stats_mod
from recommendation_engine.sync import synthetic

logging.basicConfig(format="%(levelname)s %(message)s", level=logging.INFO)
logger = logging.getLogger("recs.seed")


class _SeedArgs(argparse.Namespace):
    readers: int
    seed: int
    days: int
    catalog_limit: int | None
    purge: bool
    refresh_only: bool
    cooccurrence: bool
    dry_run: bool


async def run(args: _SeedArgs) -> dict:
    settings = RecSettings()
    # Both pools on the primary: a refresh reads what it just wrote, which a
    # read replica may not have yet.
    db = Database(write_dsn=settings.write_dsn, read_dsn=settings.write_dsn)
    await db.connect()
    report: dict = {}
    try:
        pool = db.write_pool

        if args.purge:
            removed = await inter.purge_synthetic(pool)
            report["purged_rows"] = removed
            logger.info("purged %d synthetic interaction rows", removed)

        records: Optional[list[inter.InteractionRecord]] = None

        if not args.refresh_only and not args.purge:
            catalog = await synthetic.load_catalog(pool, limit=args.catalog_limit)
            if not catalog:
                raise RuntimeError("catalog is empty — ingest published stories first")
            logger.info("catalog: %d eligible items", len(catalog))
            records = list(
                synthetic.generate(
                    catalog,
                    readers=args.readers,
                    seed=args.seed,
                    days=args.days,
                )
            )
            report["generated_records"] = len(records)
            logger.info(
                "generated %d records for %d readers", len(records), args.readers
            )

        if args.dry_run:
            report["dry_run"] = True
            if records:
                report["sample"] = [
                    {
                        "user_id": r.user_id,
                        "story_id": r.story_id,
                        "kind": r.kind,
                        "value": r.value,
                    }
                    for r in records[:3]
                ]
            return report

        if records:
            load_stats = await inter.load(pool, records)
            report["load"] = load_stats.as_dict()
            logger.info(
                "loaded: %d rows, %d readers, %d completions derived",
                load_stats.written,
                load_stats.users,
                load_stats.completions_derived,
            )

        # Always refresh: interactions are useless to scoring until aggregated.
        config = ScoringConfig.from_mapping(await db.load_config())
        item_stats = await stats_mod.refresh_item_stats(pool, config)
        report["item_stats"] = item_stats.as_dict()
        logger.info(
            "item_stats: %d items, global_mean_rating=%s, p95_engagement=%s",
            item_stats.items_updated,
            item_stats.global_mean_rating,
            item_stats.p95_engagement,
        )

        taste = await stats_mod.rebuild_user_taste(pool)
        report["user_taste"] = taste.as_dict()
        logger.info(
            "user_taste: %d built, %d skipped as too thin",
            taste.tastes_built,
            taste.tastes_skipped_thin,
        )

        if args.cooccurrence:
            pairs = await stats_mod.rebuild_cooccurrence(pool, config)
            report["cooccurrence_pairs"] = pairs
            logger.info(
                "item_cooccurrence: %d pairs (CF stays inert until "
                "w_cf_ceiling > 0 in recommendations.config)",
                pairs,
            )

        return report
    finally:
        await db.aclose()


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate/load reader behaviour and refresh derived tables."
    )
    parser.add_argument("--readers", type=int, default=200)
    parser.add_argument(
        "--seed", type=int, default=1234, help="RNG seed; same seed = same data"
    )
    parser.add_argument(
        "--days", type=int, default=180, help="Spread activity over this many days"
    )
    parser.add_argument(
        "--catalog-limit", type=int, help="Only let readers see the first N items"
    )
    parser.add_argument(
        "--purge", action="store_true", help="Delete all synth_* readers first"
    )
    parser.add_argument(
        "--refresh-only",
        action="store_true",
        help="Recompute item_stats/user_taste from existing interactions",
    )
    parser.add_argument(
        "--cooccurrence",
        action="store_true",
        help="Also rebuild item_cooccurrence (CF remains weighted 0)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Generate only; no database writes"
    )
    args = cast(_SeedArgs, parser.parse_args(argv))
    load_env()

    try:
        report = asyncio.run(run(args))
    except Exception as exc:
        logger.error("seed failed: %s", exc)
        return 1

    print(json.dumps(report, indent=2, default=str))
    if not args.dry_run:
        print(
            "\nNOTE: synthetic data validates mechanics, not taste. Any quality "
            "metric computed on it is circular — it measures whether the recommender "
            "recovers the generator's own assumptions.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
