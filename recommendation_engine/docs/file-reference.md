# File reference — what each module does

~6,700 lines across 27 files. Line counts are real, so they indicate where the weight
actually sits.

```
recommendation_engine/
├── Serving              config db embeddings env explain fusion hyde llm mmr
│                        pipeline retrieval routes scoring server query_cli
├── ingest/              platform compose normalize_llm vocabularies
├── sync/                interactions seed stats synthetic
└── docs/                these files

The schema lives in story-data (`migrations/000019_recommendations_schema.sql`);
this service does not migrate. The CMU corpus, its parser and its genre
crosswalk have been removed — the catalog is published TaleTribe stories only.
```

---

## Serving layer

### `server.py` — 322 lines
The app factory. `create_app()` wires everything and returns a FastAPI app.

Builds, in order: logging (structlog JSON), `.env`, `RecSettings`, the lifespan
(opens/closes the pool), CORS, then singletons on `app.state` — `db`, `retriever`, two
rate limiters, `embedder`, `query_embedder`, `llm`, `explanation_cache`. Mounts the
router and registers two exception handlers that wrap everything in the
`{success, data, error}` envelope. Owns `verify_internal_token`, and defines `/health`.

`RecSettings()` is instantiated **inside** the factory, not at module level, so tests can
monkeypatch env vars before the app is built.

### `routes.py` — 458 lines
All five HTTP endpoints, plus the pydantic request models and a 60-second TTL cache for
the scoring config and catalog stats (re-reading them per request would add two round
trips to a path whose point is one).

`build_router(verify_internal_token)` takes the auth dependency as an argument so
`server.py` keeps ownership of how callers are verified.

### `pipeline.py` — 246 lines
`rank()` — the one orchestration path: retrieve → fuse → score → diversify. Shared by
the HTTP routes and the CLI, so they cannot drift apart. Also holds the single-seed vs
multi-seed branch (cosine vs RRF) and `_cf_scores`, which short-circuits without
touching the database while `w_cf_ceiling = 0`.

### `scoring.py` — 461 lines
The formula, as pure functions. `ScoringConfig` (knobs, clamped on load),
`behavioral_ramp`, `term_weights`, `bayesian_rating`, `engagement`, `popularity`,
`cooccurrence_similarity`, `collaborative_score`, `source_weight`, `blend`,
`seed_engagement_weight`, `CatalogStats`.

No I/O whatsoever, which is why it's tested against hand-computed values. `blend()`
returns a `ScoreBreakdown` with every input exposed, so a surprising ranking can be
explained from the response instead of re-derived.

### `retrieval.py` — 281 lines
The only place SQL touches the vector index. `Retriever.knn()` (one query vector),
`knn_many()` (N concurrent — deliberately not a `LATERAL` join), `resolve_titles()`
(trigram fuzzy match), `popular()` (the no-vector fallback), `catalog_stats()`.

`RetrievalFilters` maps optional filters to NULL-guarded SQL parameters, so one prepared
statement covers every combination. `_row_to_dict` normalizes pgvector's `Vector` object
to a plain list at the boundary — that object is neither iterable nor `len()`-able, and
it crashed MMR on real rows while passing every list-fixture unit test.

### `db.py` — 245 lines
`Database` owns two asyncpg pools (write → primary, read → replica; shared when the DSNs
match). Registers the pgvector codec per connection.

`vector_query()` is the important piece: an async context manager yielding a connection
inside a transaction with `SET LOCAL hnsw.ef_search / iterative_scan / max_scan_tuples /
statement_timeout`. `SET LOCAL`, never session-level — under a transaction-mode pooler a
session setting is discarded or leaks, and the query would silently run at the default
`ef_search = 40`. Also `health()` and the pgvector version gate.

### `fusion.py` — 137 lines
Reciprocal Rank Fusion. `reciprocal_rank_fusion` (raw), `max_rrf_score` (the
normalizer), `fuse_normalized`, `fuse_and_rank` (deterministic tiebreak on id),
`merge_candidate_records` (fuses row dicts, collecting `per_query_similarity` so an
explanation can say which seed a result came from).

### `mmr.py` — 151 lines
Maximal Marginal Relevance in numpy. `mmr_select` (returns indices), `diversify`
(reorders record dicts), `intra_list_diversity` (the online metric). Zero-norm rows are
left alone rather than producing NaN — a missing embedding should make an item maximally
*dissimilar*, never poison the matrix.

### `explain.py` — 356 lines
The explanation layer. `query_fingerprint` and `cache_key` (deterministic, and
self-invalidating via `embed_input_sha`), `ExplanationTarget`, `build_prompt`,
`ExplanationCache` (Postgres-backed), `explain_many` (sync), `stream_explanations`
(multiplexed SSE with cancellation), `sse` (frame formatting).

Only *normalized* fields reach the prompt — never a raw summary. The model cannot spoil
what it was never told.

### `hyde.py` — 151 lines
Hypothetical Document Embeddings. `build_prompt`, `compose_hyde_text` (renders through
the **same** template as real catalog rows — the whole trick), `generate` (returns `None`
on failure so the caller falls back to embedding the raw query).

### `llm.py` — 325 lines
Direct Gemini client. `generate_text`, `generate_json` (native `responseSchema`
structured output), `stream_text` (SSE parsing), `TokenUsage` (running cost),
`build_client` (returns `None` without a key). Tenacity retry on transient failures
only; streaming is deliberately not retried, since a partially-delivered response can't
be transparently resumed.

Bypasses creditProxy, which is what makes real streaming and structured output possible —
and means explanations are not credit-metered.

### `embeddings.py` — 129 lines
`QueryEmbedder` wraps the shared provider with task-type asymmetry and an LRU over
queries (`embed_query` → `RETRIEVAL_QUERY`, cached; `embed_documents` →
`RETRIEVAL_DOCUMENT`, batched, uncached because catalog vectors live in Postgres and an
in-process copy would just be a staler second store).

### `config.py` — 236 lines
`RecSettings` (pydantic-settings). Clamps hostile values, hard-fails on missing
production config, and exposes derived properties (`oidc_audience` → `None` outside
production, `read_dsn`/`write_dsn`, `parsed_cors_origins`). The `LOCAL_DEV_DSN` default
makes local dev zero-config and is explicitly *rejected* in production.

### `env.py` — 35 lines
`load_env()`, shared by all four entry points. Deliberately not called at package
import: the test suite sets `os.environ` before importing the app, and a package-level
`load_dotenv` would leak a developer's `.env` into tests that depend on a variable being
absent.

### `query_cli.py` — 201 lines
Debug the recommender from a shell, no HTTP. Calls the same `pipeline.rank()` the routes
use. `--title` (repeatable → multi-seed RRF), `--text`, `--genre`, `--scores` (full
per-term attribution), `--json`.

---

## `ingest/` — building the catalog

### `platform.py` — 627 lines
The catalog ingest, and the only thing in this repo that reads story-data's product
tables. Replaced `backfill.py` when the CMU corpus was removed.

Reads published stories (`_READ_SQL` — scalar subqueries rather than joins, because
joining `story_tags` and `chapters` together multiplies their rows and silently
inflates `sum(word_count)`), composes the text an LLM reads (`summary_text` — chapter
summaries preferred over the description, capped at 1,200 words), normalizes,
embeds, and upserts in chunks of 200 so progress commits incrementally.

Owns four things worth knowing:

- **The skip check** (`_existing_shas`) — four conditions, each load-bearing:
  `embedding IS NOT NULL`, `is_eligible`, `embed_model = $2`, `embed_task_type = $3`.
  The last two catch a provider or task-type swap, which the text hash cannot.
- **The incremental cursor** — the newest `stories.updated_at` seen, stored in
  `recommendations.ingest_runs.cursor` and advanced only on success.
- **Eligibility reconciliation** (`_retire_unpublished`) — the read only sees
  published stories, so unpublishing is invisible to it. Deletion needs no handling:
  `items.story_id` is a foreign key with `ON DELETE CASCADE`.
- **`rebuild_hnsw_index()`** — deliberately *not* part of a normal run.

Flags: `--dry-run`, `--full`, `--limit`, `--skip-normalization`, `--rebuild-index`,
`--chunk-size`, `--batch-size`, `--concurrency`. See [jobs.md](jobs.md).

### `normalize_llm.py` — 445 lines
The LLM pass. `Normalizer` (batches ~8 books, bounded concurrency, splits blocked
batches to isolate the offender), `NormalizationCache` (stores **raw** output, filters on
read — which is what makes vocabulary tuning free), `_coerce` (clamps, relocates
cross-axis terms, records violations), `build_prompt`, `PROMPT_VERSION`.

### `vocabularies.py` — 556 lines
223 themes, 33 tones, 79 theme aliases, 21 tone aliases, 20 suppressed genre names.
`filter_themes`/`filter_tones` (canonicalize + drop out-of-vocabulary),
`split_misfiled` (relocates a tone filed under themes, and vice versa), `is_genre_noise`
(a genre arriving as a theme is noise, not a gap), `prompt_vocabulary_block`.

### `compose.py` — 111 lines
`compose_embed_input` (the template; absent fields drop their whole line),
`embed_input_sha`, `summary_sha`, `TEMPLATE_VERSION`. Pure functions, golden-string
tested.

> `genre_crosswalk.py` and its 227-row CSV are **deleted**. They mapped the CMU
> corpus's free-text genre labels onto TaleTribe's controlled categories. A
> TaleTribe story already carries a controlled `category`, so the ingest reads it
> directly and there is nothing to crosswalk.

---

## `sync/` — reader signals

### `interactions.py` — 228 lines
`InteractionRecord` (with `is_complete` inference), `load()` (resolve `story_id` →
`items.id`, weight, derive completion, idempotent upsert), `purge_synthetic(prefix=...)`.

**Real reader signals no longer come through here.** story-data derives them
directly into `recommendations.interactions`; the JSONL format that used to be the
contract between the two services is gone. What remains serves `synthetic.py`.

### `stats.py` — 289 lines
`refresh_item_stats` (SQL aggregates + `pop_score` computed in Python, so the formula has
one definition), `rebuild_user_taste` (weighted, L2-normalized mean; suppresses books
rated ≤2), `rebuild_cooccurrence` (self-join, shrunk cosine, `co_count ≥ 3` floor).

### `synthetic.py` — 313 lines
The generator. `Persona`, `build_personas` (sampled from the catalog's own vocabulary),
`_popularity_weights` (Zipf-like latent popularity), `generate` (emits records with the
correlations downstream aggregation depends on), `load_catalog`. Anchored to
`REFERENCE_NOW` so it's a pure function of its seed.

### `seed.py` — 196 lines
CLI tying it together. `--readers`, `--seed`, `--purge`, `--cooccurrence`,
`--dry-run`, and **`--refresh-only`**, which is the flag that matters in production:
run it after `story-data sync-recs` to recompute `item_stats`, `pop_score` and
`user_taste` over whatever is in `interactions` — synthetic, real, or both.

---

## No `migrations/` — the schema lives in story-data

This service migrates nothing. Both files that used to live here — `001_init.sql`
and a goose-like runner — are deleted.

The `recommendations` schema is created by
**`story-data/migrations/000019_recommendations_schema.sql`** and granted to the
`recs_service` role by **`000020_recommendations_role_grant.sql`**, both applied by
story-data's goose runner at API startup under an advisory lock. Schema changes go
there, as new immutable files.

That is deliberate, not a compromise: `item_stats` and `interactions` are
derivations of `story_likes`, `story_ratings` and `reading_progress`, which across a
database boundary needs an export pipeline and in one database is a query. One
database, one migration runner.

The failure this creates — starting recs before story-data has migrated — is caught
by `schema_present` in `/health`, which is why that check exists. See
[migration-to-story-data.md](migration-to-story-data.md).

---

## Tests

16 files plus `conftest.py`: **431 tests** — 360 unit, 71 integration. Full detail,
including how to run them and what is *not* covered, is in
[testing.md](testing.md). Summary:

| File | Tests | Covers |
|---|---|---|
| `test_rec_scoring.py` | 72 | every formula, against hand-computed values |
| `test_rec_normalize.py` | 44 | vocabularies, prompts, batching, raw-output caching |
| `test_rec_fusion_mmr.py` | 36 | RRF arithmetic, MMR selection, diversity metric |
| `test_rec_explain.py` | 35 | cache keys, sync + streaming explanations, HyDE |
| `test_rec_llm.py` | 33 | request shapes, retry classification, SSE parsing |
| `test_rec_embeddings.py` | 31 | the `task_type=None` byte-for-byte pin, batching, provenance |
| `test_rec_query_embedder.py` | 23 | the query-side cache |
| `test_rec_config.py` | 24 | settings, production validators |
| `test_rec_db.py` | 20 | pool wiring, health probes |
| `test_rec_sync.py` | 20 | record shape, completion inference, generator properties |
| `test_rec_pipeline.py` | 13 | the single-seed cosine vs multi-seed RRF branch |
| `test_rec_platform_ingest.py` | 9 | story → embeddable item shaping |
| `test_rec_retrieval_integration.py` | 27 | real Postgres: SQL, filters, iterative scan |
| `test_rec_routes_integration.py` | 19 | real Postgres: the HTTP surface end to end |
| `test_rec_sync_integration.py` | 18 | real Postgres: loader, aggregation, taste vectors |
| `test_rec_integration.py` | 7 | real Postgres: health, vector round-trip |

Integration tests **self-skip** without `RECS_TEST_DATABASE_URL` — required, not
stylistic: `pytest.ini` has no `-m` exclusion and CI runs `pytest tests/`, so an env
guard is what keeps CI green.

`tests/conftest.py` is new, and load-bearing. Because this repo no longer migrates,
an integration test cannot create the schema it needs — `require_recommendations_schema`
skips instead, so a missing schema reports as a setup problem rather than a code
failure. `seed_stories`/`drop_stories` create the `stories` rows that `items.story_id`
requires, and cleanup deletes the *story*, never the item, so the `ON DELETE CASCADE`
is exercised on every run.

---

## Not written yet

Two entries that used to be here are **done**, and neither landed as a file in this
repo — worth knowing, because it is where you would go looking for them:

| Was planned as | Actually became |
|---|---|
| `sync/platform_sync.py` — import published stories into `items` | **`ingest/platform.py`**, above |
| `sync/firestore.py` — read likes/ratings/progress → `InteractionRecord` | **story-data's `sync-recs`** (`internal/store/recommendations.go`), because `reading_progress` is private data this service may not read |

Still genuinely unwritten:

| File | Purpose |
|---|---|
| `eval/metrics.py` | Precision@K, Recall@K, NDCG@K, MRR |
| `eval/genre_holdout.py` | the cheapest honest offline proxy |
| `eval/llm_judge.py` | LLM-as-judge with a Cohen's κ ≥ 0.4 gate |
| `eval/leave_one_out.py` | with a `--min-users 200` hard gate |
| `Dockerfile`, `.dockerignore`, `terraform/`, `.github/workflows/` | deployment — none exist |
