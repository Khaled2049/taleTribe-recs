# Testing — what 455 tests cover, and what they don't

17 test files plus `conftest.py`. **455 tests: 376 unit, 79 integration.**

The organising idea: this is a system whose failures are *silent*, so the tests are
weighted toward pinning down things that would otherwise degrade without erroring —
formulas against hand-computed values, byte-for-byte request shapes, and real SQL
against a real Postgres.

---

## Running them

```bash
# unit only — no infrastructure of any kind
poetry run python -m pytest tests/ -q -m unit

# everything, including integration
RECS_TEST_DATABASE_URL="postgresql://postgres:postgres@localhost:5433/story_data" \
  poetry run python -m pytest tests/ -q
```

### ⚠ Use `python -m pytest`, not `pytest`

```
poetry run pytest tests/
→ ModuleNotFoundError: No module named 'recommendation_engine'
```

`pyproject.toml` sets `package-mode = false`, so `recommendation_engine` is never
installed into the virtualenv. Tests import it from the repo root, which lands on
`sys.path` only because `python -m` puts the working directory there. Bare `pytest`
does not.

This costs everyone new to the repo about an hour. Use `python -m pytest`.

### Integration tests self-skip

Without `RECS_TEST_DATABASE_URL` every integration test skips. That is required, not
stylistic: `pytest.ini` has no `-m` exclusion in `addopts` and CI runs `pytest
tests/`, so an env guard is the only thing keeping CI green.

`RECS_TEST_DATABASE_URL` is deliberately **separate** from `RECS_DATABASE_URL`. These
tests write and delete rows; pointing them at a real deployment must take a
deliberate act rather than an inherited environment.

They also skip — with a distinct message — if the `recommendations` schema is absent.
Since this repo no longer migrates, that is a setup problem to report as such rather
than a failure to blame on the code:

```
no `recommendations` schema in RECS_TEST_DATABASE_URL; apply story-data's
migrations to that database first (make migrate)
```

---

## `tests/conftest.py`

New with the story-data migration, and load-bearing. Before the move, an integration
test could create the schema it needed by calling the local migration runner. It
can't any more, so `conftest.py` supplies three things:

**`require_recommendations_schema(dsn)`** — skip unless `recommendations.items`
exists.

**`seed_stories(conn, keys)` / `drop_stories(conn, keys)`** — `items.story_id` is a
foreign key, so a catalog fixture needs a `stories` row to exist first. IDs come from
`uuid5` over a fixed namespace, so the same key always yields the same UUID: stable
across runs, readable in a failure message, and — unlike `uuid4` — leaving no orphans
when a test dies before cleanup.

**Cleanup deletes the *story*, never the item.** `ON DELETE CASCADE` then takes the
item, its stats, its interactions and its explanations with it. That is less code and
a standing check that the cascade is wired the way the schema claims.

---

## What each file covers

### Unit — 360 tests, no infrastructure

| File | Tests | Covers |
|---|---|---|
| `test_rec_scoring.py` | 72 | Every formula, against **hand-computed** values — Bayesian damping, the cold-start ramp, term weights, MMR λ, the `MAX_BEHAVIORAL_WEIGHT` clamp |
| `test_rec_normalize.py` | 44 | Controlled vocabularies, prompt construction, batching, raw-output caching, blocked-response splitting |
| `test_rec_fusion_mmr.py` | 36 | RRF arithmetic across seeds, MMR selection order, the diversity metric |
| `test_rec_explain.py` | 35 | Cache-key derivation and self-invalidation, sync + streamed explanations, HyDE |
| `test_rec_llm.py` | 33 | Request shapes, retry classification, SSE frame parsing |
| `test_rec_embeddings.py` | 31 | The `task_type=None` byte-for-byte pin, batching, provenance |
| `test_rec_config.py` | 24 | Settings parsing, the production validators, clamping |
| `test_rec_query_embedder.py` | 23 | The query-side LRU cache |
| `test_rec_db.py` | 20 | Pool wiring, DSN handling, the health probes |
| `test_rec_sync.py` | 20 | Record shape, completion inference, generator statistical properties |
| `test_rec_pipeline.py` | 13 | The single-seed-cosine vs. multi-seed-RRF branch |
| `test_rec_platform_ingest.py` | 9 | Story → embeddable item shaping |

**Why hand-computed and not snapshots.** `test_rec_scoring.py` derives its expected
values from the formula in the module docstring rather than capturing them from a
run. A snapshot test would happily lock in a wrong formula — and in a recommender, a
wrong formula produces plausible output.

**The `task_type=None` pin** is worth singling out: it asserts the request body
byte-for-byte when no task type is given, because a change there silently shifts
every vector into a different space, and nothing downstream would error.

### Integration — 71 tests, real Postgres

| File | Tests | Covers |
|---|---|---|
| `test_rec_retrieval_integration.py` | 27 | The KNN SQL, every filter predicate, `SET LOCAL` scoping, iterative scan |
| `test_rec_routes_integration.py` | 19 | The HTTP surface end to end against real data and real ranking |
| `test_rec_sync_integration.py` | 18 | The loader, idempotency, stale-vs-newer progress, aggregation, taste vectors, co-occurrence, purge |
| `test_rec_integration.py` | 7 | Health probes, vector round-trip, cosine ordering |

These use `USE_MOCK=true` embeddings — deterministic offline vectors — so they
exercise real SQL and real ranking without spending tokens. Fixtures are namespaced
so a shared database is safe.

Notable coverage: `test_stale_progress_does_not_overwrite_newer` pins the idempotency
rule that lets the signals sync be re-run freely, and
`test_disliked_books_are_suppressed_not_averaged_in` pins the taste-vector rule that
stops a rejected book steering someone's recommendations.

---

## Cross-repo pins

Two rules exist in two languages, and are held together by tests in **story-data**,
not here:

| Rule | Python | Go | Pinned by |
|---|---|---|---|
| Completion = last chapter **and** ≥90% scroll | `COMPLETION_SCROLL_THRESHOLD` | `completionScrollThreshold` | `TestCompletionNeedsLastChapterAndDeepScroll` |
| Signal weights (like 1.0, completion 0.8, rating bands, progress 0.4×) | `seed_engagement_weight` | the SQL `CASE` | `TestRecommendationSignalWeights` |

Both live in `story-data/internal/httpapi/e2e/recommendations_test.go`. Change a
weight here and that suite is what tells you.

A third contract is **not** pinned by any test: `EXPECTED_EMBEDDING_DIM = 768` in
`embedding_provider.py`, which is *duplicated* between this repo and
taleTribe-agents rather than shared. Nothing fails if they drift; retrieval just
returns nothing on one side. `/health` asserting the dimension is the only guard.

---

## What is *not* covered

Stated plainly, because the gaps are in the newest code.

### `ingest/platform.py` has no integration test at all

`test_rec_platform_ingest.py` covers only `summary_text` and `to_normalized_item` —
the pure shaping functions. Untested against a database:

- **`_READ_SQL`**, including the scalar-subquery structure that exists specifically to
  stop `story_tags` and `chapters` multiplying each other's rows and inflating
  `sum(word_count)`. That is a silent-corruption bug with no test.
- **`_existing_shas`** and all four of its conditions — the ones that stop a
  provider swap or a task-type change being skipped as "unchanged".
- **`_upsert`** — the `COALESCE` that preserves an existing vector, and the
  `embedded_at` CASE.
- **`_retire_unpublished`** and the retire/republish cycle.
- **The incremental cursor** round trip through `ingest_runs`.

The deleted CMU ingest had a 628-line test file. Its replacement has 9 unit tests.
Every one of the known issues in [runbook.md](runbook.md) lives in this untested
code — which is not a coincidence.

### `/health` has no green-path test

Two integration tests were removed during the migration review because the shared dev
database makes them unrunnable: `catalog_embed_model` reports the *majority* model
among real rows, and the tests run with the mock embedder, so `/health` is
permanently 503 on any machine that has run a real ingest.

`test_health_ignores_the_embedder_model_on_an_empty_catalog` still exists but its
whole body is inside `if body["catalog_embed_model"] is None:` — against a populated
database it **passes vacuously**, asserting nothing.

So there is currently nothing asserting `/health` ever returns 200. A bug making it
permanently red would ship unnoticed — and it would fail the deploy verification
step, which is exactly when you least want to be debugging it.

**The honest fix is a dedicated test database, not a test change.** The old `:5434`
database was owned by the tests; story-data's shared dev database never will be.

### Other gaps

- **No load or concurrency testing.** `statement_cache_size=0` under a transaction
  pooler is a known production trap (see [deployment.md](deployment.md)) that only
  appears under concurrency — i.e. never in this suite.
- **No test asserts recs cannot read `reading_progress`.** The privacy boundary is a
  grant, and the grant is a no-op locally where everything runs as `postgres`. It is
  currently unverifiable in CI.
- **Quality is untested by construction.** There is no eval harness; see
  [evaluation.md](evaluation.md). Synthetic data validates *mechanics* — the ramp
  engages, the popularity term moves, a taste vector retrieves sensibly — but any
  quality metric computed on it measures whether the recommender recovers the
  generator's own assumptions, which is circular.

---

## CI

Both workflows start `pgvector/pgvector:pg16`, check out story-data, create the
production-only `recs_service` role, and run story-data's migrations before pytest.
That makes the SQL integration suite and the least-privilege boundary part of CI
instead of allowing schema-dependent tests to skip silently.

PR validation also checks formatting, Ruff, Terraform, and the production Docker
build. The deployment workflow repeats the test suite before it can publish or apply.
