# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`taleTribe-recs` is TaleTribe's standalone recommendation service: pgvector
retrieval, HyDE query expansion, RRF fusion across multiple seeds, MMR
diversification, and an LLM explanation layer ("why was this recommended").
It was split out of `taleTribe-agents/recommendation_engine` to keep its
Postgres/pgvector workload, migrations, and scaling profile independent from
that Firestore-backed service — see `README.md` for why.

It recommends **TaleTribe stories** (owned by the `story-data` repo/service,
`item_source = 'platform'`) plus a bootstrap corpus of CMU book summaries
(`item_source = 'cmu'`) used for cold-start before enough platform
interactions exist. `recommendations.items` is a **derived catalog** —
embeddings + LLM-extracted genres/themes/tone/premise — keyed back to the
real story by `source_id`. It is not a copy of story-data's source of truth,
the same way `story_vector_chunks` in story-data isn't a copy of `chapters`.

Read `recommendation_engine/docs/running-locally.md` first — it has the full
walkthrough (setup, the HTTP API with curl examples, loading the catalog,
cost figures, seed data, useful SQL). This file is the short version plus
repo-specific conventions.

## Starting the project

```bash
poetry install                                              # or: pip install -e . equivalent deps below
docker compose -f recommendation_engine/docker-compose.yml up -d   # pgvector Postgres on :5434
python -m recommendation_engine.migrations.migrate

python -m recommendation_engine.server        # serves on :8100
curl localhost:8100/health                    # 503 until pgvector>=0.8, HNSW index, and embedder dim=768 all check out
```

Zero config needed for local dev — `RECS_DATABASE_URL` defaults to exactly
what `docker-compose.yml` serves (`postgresql://recs:recs@localhost:5434/recs`),
and that default is explicitly rejected when `ENVIRONMENT=production`. Auth
(`verify_internal_token` in `recommendation_engine/server.py`) is a no-op
unless `RECS_SERVICE_URL` is set, so no OIDC token is needed locally either.

To have real data to query:

```bash
# parse-only, no API calls, no key needed
python -m recommendation_engine.ingest.backfill --limit 2000 --dry-run

# deterministic offline embeddings — exercises the plumbing, not quality
USE_MOCK=true python -m recommendation_engine.ingest.backfill --limit 300 --skip-normalization

# real embeddings (needs GOOGLE_AI_STUDIO_API_KEY)
python -m recommendation_engine.ingest.backfill --limit 200
```

Without `GOOGLE_AI_STUDIO_API_KEY`, ranking still works; HyDE and
explanations are disabled (both call Gemini directly, not through
creditProxy — see "Cost controls" below).

## Commands

```bash
pytest tests/test_rec_*.py -q                    # full suite
pytest tests/ -q -m unit                         # no infrastructure needed
RECS_TEST_DATABASE_URL="postgresql://recs:recs@localhost:5434/recs" pytest tests/ -q   # + integration tests

python -m recommendation_engine.query_cli --title "Dune" --top-k 6   # rank without an HTTP server
python -m recommendation_engine.migrations.migrate --status
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
    falls back to popularity until the Firestore signals sync (Phase 3)
    populates it
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
- **`ingest/`** — CMU corpus parsing, genre crosswalk, LLM-based
  normalization, backfill orchestration.
- **`sync/`** — interaction ingestion (`interactions.py`), aggregate refresh
  (`stats.py`), and synthetic reader generation for testing scoring
  mechanics before real signals exist (`synthetic.py`, `seed.py`).
- **`migrations/`** — a small Python migration runner (`migrate.py`), not
  Alembic; SQL files in `migrations/*.sql`, applied under an advisory lock.

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

## Ports

| Port | What |
|---|---|
| 5434 | pgvector Postgres (this repo's `docker-compose.yml`) |
| 8100 | this service |
| 5433 | story-data's Postgres, if running the wider TaleTribe stack alongside this |

## Not built yet

- Firestore interaction sync (Phase 3) — the thing that makes `pop` real
  and `/recommend/behavioral` return actual personalization instead of a
  popularity fallback.
- Eval harness (Phase 6) — see `recommendation_engine/docs/evaluation.md`.
  Until it exists, quality is judged by reading result lists.
- Deployment (Neon, Terraform, Cloud Run) — see
  `recommendation_engine/docs/deployment.md`.
- Ingesting TaleTribe's own stories (`item_source = 'platform'`) as a
  catalog source — the schema and `source`/`source_id` pattern already
  support it, but the ingest path currently only pulls the CMU bootstrap
  corpus.
