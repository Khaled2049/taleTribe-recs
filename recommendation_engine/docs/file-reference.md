# File reference — what each module does

~5,300 lines across 31 files. Line counts are real, so they indicate where the weight
actually sits.

```
recommendation_engine/
├── Serving              config db embeddings env explain fusion hyde llm mmr
│                        pipeline retrieval routes scoring server query_cli
├── ingest/              backfill cmu_parse compose genre_crosswalk
│                        normalize_llm vocabularies (+ genre_crosswalk.csv)
├── sync/                interactions seed stats synthetic
├── migrations/          001_init.sql migrate
├── booksummaries/       the CMU corpus (42 MB, not code)
└── docs/                these files
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

### `backfill.py` — 486 lines
The orchestrator: parse → crosswalk → normalize → compose → embed → upsert, in chunks of
200 so progress commits incrementally. Owns the **skip check** (`embed_input_sha` **and**
`embed_model` **and** `embed_task_type`), `ingest_runs` bookkeeping, and
`rebuild_hnsw_index()`. Flags: `--limit`, `--dry-run`, `--skip-normalization`,
`--rebuild-index`, `--batch-size`, `--concurrency`.

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

### `cmu_parse.py` — 241 lines
Streaming TSV parser. `iter_records` (with the four filters), `parse_genres` (guards the
empty-string-not-`{}` trap), `parse_year` (three date precisions → year only),
`normalize_summary` (strip, collapse, truncate), `dedupe_key`, `ParseStats`.

### `compose.py` — 111 lines
`compose_embed_input` (the template; absent fields drop their whole line),
`embed_input_sha`, `summary_sha`, `TEMPLATE_VERSION`. Pure functions, golden-string
tested.

### `genre_crosswalk.py` — 194 lines + `genre_crosswalk.csv`
`load_crosswalk` (validates: no duplicates, no unknown categories, map-vs-drop coherence),
`GenreCrosswalk.map_labels` (including the fictionality resolution), `unknown()`,
`PLATFORM_CATEGORIES`, `FICTION_MARKERS`, `NONFICTION_MARKERS`.

The CSV holds all 227 labels with an action and a note, so the judgement calls are
reviewable in a diff.

---

## `sync/` — reader signals

### `interactions.py` — 299 lines
The canonical contract. `InteractionRecord` (+ JSON round-trip, `is_complete`
inference), `read_jsonl`/`write_jsonl`, `load()` (resolve external identity, weight,
derive completion, idempotent upsert), `purge_synthetic(prefix=...)`.

### `stats.py` — 289 lines
`refresh_item_stats` (SQL aggregates + `pop_score` computed in Python, so the formula has
one definition), `rebuild_user_taste` (weighted, L2-normalized mean; suppresses books
rated ≤2), `rebuild_cooccurrence` (self-join, shrunk cosine, `co_count ≥ 3` floor).

### `synthetic.py` — 313 lines
The generator. `Persona`, `build_personas` (sampled from the catalog's own vocabulary),
`_popularity_weights` (Zipf-like latent popularity), `generate` (emits records with the
correlations downstream aggregation depends on), `load_catalog`. Anchored to
`REFERENCE_NOW` so it's a pure function of its seed.

### `seed.py` — 193 lines
CLI tying it together. `--readers`, `--seed`, `--out`, `--load` (source-agnostic — also
Phase 3's entry point), `--purge`, `--refresh-only`, `--cooccurrence`, `--dry-run`.

---

## `migrations/`

### `001_init.sql` — 262 lines
The whole schema: 9 tables, the partial HNSW index, GIN/trigram/B-tree indexes, the
`item_source` enum, and the seeded `config` rows. Idempotent throughout.

### `migrate.py` — 185 lines
Runner. Session advisory lock (no double-apply), per-file checksum (editing an applied
migration fails loudly), per-file transaction (a failure leaves nothing behind).
`--status`, `--dsn`.

---

## Tests

14 files, ~476 tests for this service (1,025 repo-wide). Repo convention: **no
`conftest.py`** — env vars are set at module import before importing the app,
`pytestmark` at module level, `# noqa: E402` on the late imports.

| File | Covers |
|---|---|
| `test_rec_scoring.py` | every formula, against hand-computed values |
| `test_rec_fusion_mmr.py` | RRF arithmetic, MMR selection, diversity metric |
| `test_rec_cmu_parse.py` | parser filters, crosswalk (incl. full-corpus coverage), compose |
| `test_rec_normalize.py` | vocabularies, prompts, batching, raw-output caching |
| `test_rec_embeddings.py` | the `task_type=None` byte-for-byte pin, batching, provenance |
| `test_rec_explain.py` | cache keys, sync + streaming explanations, HyDE |
| `test_rec_pipeline.py` | the single-seed cosine vs multi-seed RRF branch |
| `test_rec_sync.py` | canonical record format, completion inference, generator properties |
| `test_rec_llm.py` | request shapes, retry classification, SSE parsing |
| `test_rec_config.py`, `test_rec_db.py`, `test_rec_migrations.py`, `test_rec_query_embedder.py` | settings, pool wiring, migration logic, query cache |
| `test_rec_*_integration.py` (4) | real Postgres: SQL, filters, routes, loader, aggregation |

Integration tests **self-skip** without `RECS_TEST_DATABASE_URL` — required, not
stylistic: `pytest.ini` has no `-m` exclusion and CI runs `pytest tests/`, so with no
`conftest.py` an env guard is the only convention-compatible way to keep CI green.

---

## Not written yet

| File | Purpose |
|---|---|
| `sync/firestore.py` | read likes/ratings/readingProgress → `InteractionRecord` |
| `sync/platform_sync.py` | import published stories into `items` |
| `eval/metrics.py` | Precision@K, Recall@K, NDCG@K, MRR |
| `eval/genre_holdout.py` | runnable today on 1,987 real embeddings |
| `eval/llm_judge.py` | LLM-as-judge with a Cohen's κ ≥ 0.4 gate |
| `eval/leave_one_out.py` | with a `--min-users 200` hard gate |
| `Dockerfile`, `terraform/` | deployment |
