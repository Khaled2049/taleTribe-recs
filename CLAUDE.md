# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`taleTribe-recs` is TaleTribe's standalone recommendation service: pgvector
retrieval, HyDE query expansion, RRF fusion across multiple seeds, MMR
diversification, and an LLM explanation layer ("why was this recommended").
It was split out of `taleTribe-agents/recommendation_engine` to keep its
Postgres/pgvector workload and scaling profile independent from that
Firestore-backed service — see `README.md` for why. Its *database* is no
longer separate: the `recommendations` schema now lives in story-data's
Postgres, which story-data migrates. The service, its deploy and its scaling
stay independent; only the schema moved.

It recommends **published TaleTribe stories**, and nothing else. A CMU book
summary corpus used to share the catalog as cold-start scaffolding; it has been
removed, along with its parser, genre crosswalk, the `item_source` enum and the
source down-weighting term. `recommendations.items` is a **derived catalog** —
embeddings + LLM-extracted genres/themes/tone/premise — keyed back to the real
story by a `story_id` foreign key. It is not a copy of story-data's source of
truth, the same way `story_vector_chunks` in story-data isn't a copy of
`chapters`.

Cold start is correspondingly thinner: with a small catalog there is little to
rank and little for MMR to diversify, and an ad-hoc query naming a book
TaleTribe does not host has no anchor. That is a known cost, not an oversight.

Read `recommendation_engine/docs/running-locally.md` first — it has the full
walkthrough (setup, the HTTP API with curl examples, loading the catalog,
cost figures, seed data, useful SQL). This file is the short version plus
repo-specific conventions.

`recommendation_engine/docs/README.md` indexes everything. The four to know:

- **`runbook.md`** — symptom-first diagnostics, and the known issues that block
  a production deploy. Start here when something is wrong.
- **`jobs.md`** — the three background jobs, why their order is fixed, and what
  goes stale when one doesn't run.
- **`security-and-roles.md`** — the `recs_service` grant, the privacy boundary,
  and the threat table.
- **`frontend-integration.md`** — the Firebase Function trust boundary and the
  two UI surfaces that call it.

## Starting the project

```bash
poetry install                                              # or: pip install -e . equivalent deps below
# The `recommendations` schema lives in story-data's database, migrated by it.
(cd ../story-data && docker compose up -d postgres && make migrate)

python -m recommendation_engine.server        # serves on :8100
curl localhost:8100/health                    # 503 until pgvector>=0.8, HNSW index, and embedder dim=768 all check out
```

Zero config needed for local dev — `RECS_DATABASE_URL` defaults to story-data's
local stack (`postgresql://postgres:postgres@localhost:5433/story_data`),
and that default is explicitly rejected when `ENVIRONMENT=production`. Auth
(`verify_internal_token` in `recommendation_engine/server.py`) is a no-op
unless `RECS_SERVICE_URL` is set, so no OIDC token is needed locally either.

To have real data to query:

```bash
# what would be read; no API calls, nothing written
python -m recommendation_engine.ingest.platform --dry-run

# deterministic offline embeddings — exercises the plumbing, not quality
USE_MOCK=true python -m recommendation_engine.ingest.platform --skip-normalization

# real embeddings + premise extraction (needs GOOGLE_AI_STUDIO_API_KEY)
python -m recommendation_engine.ingest.platform
```

Incremental by default: each run stores the newest `stories.updated_at` it saw
in `ingest_runs.cursor` and the next run starts past it. `--full` re-reads
everything, which is cheap — an unchanged `embed_input_sha` is never re-embedded.

Without `GOOGLE_AI_STUDIO_API_KEY`, ranking still works; HyDE and
explanations are disabled (both call Gemini directly, not through
creditProxy — see "Cost controls" below).

## Commands

```bash
pytest tests/test_rec_*.py -q                    # full suite
pytest tests/ -q -m unit                         # no infrastructure needed
RECS_TEST_DATABASE_URL="postgresql://postgres:postgres@localhost:5433/story_data" pytest tests/ -q   # + integration tests

python -m recommendation_engine.query_cli --title "Dune" --top-k 6   # rank without an HTTP server
```

`RECS_TEST_DATABASE_URL` is separate from `RECS_DATABASE_URL` on purpose —
integration tests write and delete rows, and self-skip when it's unset (that
is how CI stays green with no `-m` exclusion in `pytest.ini`).

## Architecture

- **`server.py`** — FastAPI app factory (`create_app()`), port 8100 by
  default. Builds `app.state` singletons (db pool, embedder, LLM client,
  two rate limiters) once at startup.
- **`routes.py`** — the HTTP surface, all five routes go through
  `pipeline.rank`, the same function `query_cli` uses:
  - `POST /recommend/adhoc` — seed by book titles and/or free-text prompt
  - `POST /recommend/behavioral` — from a reader's `user_taste` vector;
    falls back to popularity until the signals sync populates it
  - `POST /recommend/explain` / `GET /recommend/explain/stream` — sync vs.
    SSE-streamed, multiplexed by `item_id`
  - `GET /health` — asserts pgvector version, HNSW index presence, and
    embedder dimension explicitly, because a mismatch on any of them
    otherwise degrades results silently rather than erroring
- **`config.py`** — `RecSettings` (pydantic-settings), no env prefix, field
  names map directly to `SCREAMING_SNAKE` env vars.
- **`db.py` / `retrieval.py`** — asyncpg pool + pgvector HNSW queries.
- **`fusion.py` / `mmr.py`** — RRF fusion across multiple seed vectors, then
  MMR reordering for diversity (results are *not* strictly sorted by score
  — that's intentional, see running-locally.md).
- **`scoring.py`** — blends semantic similarity, popularity, and a
  collaborative-filtering term (currently stubbed at `cf*0.00`); cold-start
  ramps `alpha` from 0 (pure semantic) as `n_signals` grows.
- **`hyde.py` / `explain.py`** — the two paths that call Gemini directly.
  `explain.py` also owns the explanation cache (`ExplanationCache`), which
  together with the tighter LLM rate bucket is the only thing between a
  loop and an ungoverned bill.
- **`ingest/`** — `platform.py` reads published stories out of story-data (same
  database, no HTTP), normalizes them with an LLM (`normalize_llm.py`), composes
  the embed text (`compose.py`) against a controlled theme/tone vocabulary
  (`vocabularies.py`), and upserts `recommendations.items`. Unpublishing is
  reconciled by `_retire_unpublished`; deletion is handled by the `story_id`
  foreign key. The CMU corpus parser, genre crosswalk and backfill are gone.
- **`sync/`** — aggregate refresh (`stats.py`: `item_stats`, `pop_score`,
  `user_taste`, `item_cooccurrence`) plus synthetic reader generation
  (`synthetic.py`, `seed.py`). **Real reader signals are not written here** —
  story-data derives them from `story_likes`/`story_ratings`/`reading_progress`
  into `recommendations.interactions` (`story-data sync-recs`), because
  `reading_progress` is private data this service may not read. Run
  `seed.py --refresh-only` after a story-data sync to recompute the
  aggregates.
- **No `migrations/`** — this service does not migrate anything. The
  `recommendations` schema lives in story-data's database and is created by
  `story-data/migrations/000019_recommendations_schema.sql` under goose. Schema
  changes go there. `/health` fails loudly if the schema is absent, which is
  what catches starting this service before story-data has migrated.

## Duplicated files — read before editing

`embedding_provider.py` and `rate_limit.py` at the repo root are **copies**
from `taleTribe-agents` (`agents/storyAgent/brain/embedding_provider.py` and
root `rate_limit.py`), not a shared package. If you change either file here,
the same change likely needs to land in `taleTribe-agents` too, especially
`EXPECTED_EMBEDDING_DIM = 768` in `embedding_provider.py` — every vector this
service writes and every HNSW index must agree with whatever dimension the
agents service's own embedder produces, or retrieval silently returns
nothing on one side or the other.

## Cost controls

HyDE and explanations bypass creditProxy and call Gemini directly (needed
for token-by-token streaming and structured output), so they are **not**
credit-metered. The only guardrails are the explanation cache and a second,
tighter rate bucket:

- `MAX_REQUESTS_PER_MINUTE_PER_USER` (default 30) — ranking, one DB round trip
- `MAX_LLM_REQUESTS_PER_MINUTE_PER_USER` (default 6) — HyDE + explanations

## Who calls this

The browser never reaches this service. `taleTribe-frontend`'s Firebase
Functions `recommendStories` / `explainRecommendations`
(`functions/src/endpoints/recommendations.ts`) are the only caller: they mint the
Google OIDC token `verify_internal_token` requires, and they set `user_id` from
the **verified Firebase token** rather than the request body, so a reader cannot
ask for another reader's behavioral recommendations. Keep that property if you
touch the route models — `user_id` arriving in a request body is trusted here
precisely because the Function is the trust boundary.

## Ports

| Port | What |
|---|---|
| 8100 | this service |
| 5433 | story-data's Postgres — holds the `recommendations` schema |

## Not built yet

- Scheduling. Both halves — `story-data sync-recs` and
  `seed.py --refresh-only` — run on demand only; nothing invokes them
  periodically yet.
- Eval harness (Phase 6) — see `recommendation_engine/docs/evaluation.md`.
  Until it exists, quality is judged by reading result lists.
- Deployment (Neon, Terraform, Cloud Run) — see
  `recommendation_engine/docs/deployment.md`.
- **Scheduling the ingest.** `ingest.platform` runs on demand only; nothing
  invokes it periodically yet.
