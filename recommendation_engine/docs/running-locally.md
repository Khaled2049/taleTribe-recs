# Running locally

Everything except deployment runs on a laptop. All commands below were verified
working.

## Setup

```bash
cd repos/taleTribe-recs

poetry install

# The `recommendations` schema lives in story-data's database and is migrated
# by story-data. Bring that stack up first; this service creates nothing.
(cd ../story-data && docker compose up -d postgres && make migrate)
```

Postgres listens on **5433** — story-data's stack (`dev-new.sh`). This service
had its own Postgres on 5434 until the schema moved; that compose file is gone.
The `pgvector/pgvector:pg16` image ships the extension prebuilt.

Verify:

```bash
python -m recommendation_engine.server          # :8100
curl localhost:8100/health
```

`/health` returns **503 unless** pgvector ≥ 0.8.0, the HNSW index exists, the
`recommendations` schema is present, the embedder reports 768 dimensions, **and the
serving embedder agrees with the model the catalog was built by**. Every one of
those failures is otherwise silent — degraded results, not errors — so each is
asserted explicitly. [runbook.md](runbook.md) has a field-by-field triage table.

## Ports

| Port | What |
|---|---|
| 5433 | story-data's Postgres — holds the `recommendations` schema |
| 8100 | recommendation service |
| 8084 | story-data's HTTP API — not called by recs, but the same stack |
| 8000 | the story agent, for reference |

## Configuration

Config comes from the repo-root `.env` (gitignored), loaded by `recommendation_engine/env.py`
from **all** entry points — servers and CLIs alike. `load_dotenv` doesn't override
already-set variables, so `FOO=bar python -m ...` still wins.

Local dev needs **zero** configuration: `RECS_DATABASE_URL` defaults to
story-data's local stack. That default is explicitly *rejected* when
`ENVIRONMENT=production`, so it can never ship.

For real embeddings and premises:

```bash
GOOGLE_AI_STUDIO_API_KEY=...      # and make sure USE_MOCK is not "true"
```

## Loading the catalog

The catalog is **published TaleTribe stories**, read straight out of story-data's
database. There is no corpus file to download — if `stories` has no published rows,
publish something in the app first (or seed one with SQL).

```bash
# report what would be read — no API calls, no writes, no key needed
python -m recommendation_engine.ingest.platform --dry-run

# plumbing test with deterministic offline embeddings
USE_MOCK=true python -m recommendation_engine.ingest.platform --skip-normalization

# the real thing: LLM-derived premise/themes/tone, then real embeddings
python -m recommendation_engine.ingest.platform

# re-read every published story, ignoring the stored cursor
python -m recommendation_engine.ingest.platform --full
```

**Incremental by default.** Each run records the newest `stories.updated_at` it saw
in `recommendations.ingest_runs.cursor`; the next run starts past it. `--full`
ignores the cursor, which is cheap — a row whose `embed_input_sha` is unchanged is
never re-embedded.

`--rebuild-index` exists but is **not** part of a normal run. It drops the HNSW
graph before recreating it, so every query falls back to a sequential scan for the
duration. It was worth it for a 15k-row bulk load; against an incremental ingest on
a shared database it costs more than it saves. Never run it against live traffic.

**Resumable at two levels**, so re-running is cheap and safe: derived premises are
cached on the summary hash, and rows whose `embed_input_sha` is unchanged aren't
re-embedded. Re-running after a crash pays only for what's missing.

Re-running is also how you recover the ~2% of records lost per pass to safety blocks
and incomplete responses — failures are deliberately not cached.

### `USE_MOCK=true` tests plumbing, not quality

`MockEmbeddingProvider` is md5-seeded: deterministic, and **semantically
meaningless**. Every mechanism is exercised; no semantics are. Use it for tests and
CI; use a real key to judge whether recommendations are any good.

## The HTTP API

```bash
python -m recommendation_engine.server        # :8100

# "I liked these, find me more" — per-seed retrieval + RRF
curl -X POST localhost:8100/recommend/adhoc -H 'Content-Type: application/json' \
  -d '{"user_id":"u1","books":[{"title":"Dune"}],"top_k":5}'

# free text → HyDE writes a hypothetical catalog entry, then embeds that
curl -X POST localhost:8100/recommend/adhoc -H 'Content-Type: application/json' \
  -d '{"user_id":"u1","prompt":"a lonely lighthouse keeper losing his grip"}'

# from a reader's own history (falls back to popularity until the signals jobs run)
curl -X POST localhost:8100/recommend/behavioral -H 'Content-Type: application/json' \
  -d '{"user_id":"reader1","top_k":10}'

# explanations, all at once (Stage A — works through a Functions proxy)
curl -X POST localhost:8100/recommend/explain -H 'Content-Type: application/json' \
  -d '{"user_id":"u1","item_ids":[34,2096],"prompt":"gothic horror"}'

# explanations, streamed token by token (Stage B)
curl -N 'localhost:8100/recommend/explain/stream?user_id=u1&item_ids=34,2096&prompt=gothic%20horror'
```

The stream is **multiplexed** — one connection, events tagged by `item_id`:

```
event: explanation
data: {"item_id": 223, "delta": "This novel offers galactic empire science fiction"}

event: item_done
data: {"item_id": 223, "cached": false}

event: done
data: {"completed": 2}
```

Cancel it (Ctrl-C, or unmount the component) and generation stops mid-sentence.
Partial text is never cached.

### Two rate buckets

Ranking is one DB round trip and gets a generous budget
(`MAX_REQUESTS_PER_MINUTE_PER_USER`, default 30). HyDE and explanations spend
Gemini tokens on a path that bypasses creditProxy, so they draw on a much tighter
one (`MAX_LLM_REQUESTS_PER_MINUTE_PER_USER`, default 6) and return
`LLM_RATE_LIMITED`. Together with the explanation cache, that's the only thing
between a loop and a bill.

## Querying without an HTTP server

`query_cli` exercises the same `pipeline.rank` the endpoints use, so the two can't
drift apart. Useful for debugging ranking with no HTTP in the way.

```bash
export RECS_DATABASE_URL="postgresql://postgres:postgres@localhost:5433/story_data"

# seed from a catalog book (no embedding call — reuses its stored vector)
python -m recommendation_engine.query_cli --title "Dune" --top-k 6

# multi-seed: per-item retrieval + RRF, no vector averaging
python -m recommendation_engine.query_cli --title "Dune" --title "Neuromancer"

# free text
python -m recommendation_engine.query_cli --text "a totalitarian government crushes dissent"

# filters, score attribution, JSON
python -m recommendation_engine.query_cli --title "Dracula" --genre horror --scores
python -m recommendation_engine.query_cli --text "cozy mystery" --json
```

### Reading the output

```
 1. 0.6704  The Salt Road — A. Writer (2026)
      sem=0.6704*1.00  pop=0.0000*0.00  cf=0.0000*0.00  alpha=0.00
```

- `alpha=0.00` and `w_sem=1.00` — cold-start ramp: zero interactions means 100%
  semantic, as designed.
- `pop=0.0000` — `item_stats` is empty until the signals jobs run
  (`story-data sync-recs`, then `seed.py --refresh-only`), so the popularity term is
  implemented but currently inert. See [jobs.md](jobs.md).
- `cf=*0.00` — the deliberate stub.

**Results are not sorted by score.** MMR reorders for diversity, so a lower-scoring
item can appear above a higher one. If the list were strictly descending, MMR
wouldn't be doing anything.

## Tests

```bash
pytest tests/test_rec_*.py -q                    # this service (476)
pytest tests/ -q -m unit                         # no infrastructure needed
RECS_TEST_DATABASE_URL="postgresql://postgres:postgres@localhost:5433/story_data" pytest tests/ -q
```

`RECS_TEST_DATABASE_URL` is deliberately **separate** from `RECS_DATABASE_URL`:
these tests write and delete rows, so pointing them at a real deployment must be a
deliberate act rather than an inherited environment.

Integration tests **self-skip** when it's unset. That's required, not stylistic:
`pytest.ini` has no `-m` exclusion and CI runs `pytest tests/`, and with no
`conftest.py` in this repo an env guard is the only convention-compatible way to keep
CI green.

## What things cost

The old figures here priced a 15,512-book CMU corpus. That corpus is gone, and the
economics changed shape with it: the catalog is now **published TaleTribe stories**
— a tiny set, but one that is re-polled forever rather than loaded once.

### Ingest

The per-story cost is unchanged, because the models and the prompt shape are:
**one normalization call (batched ~8 stories per call) plus one embedding.**
Measured on the 2,000-item CMU run at `gemini-2.5-flash-lite` /
`gemini-embedding-001` list pricing, that came to **$0.33 / 2,000 ≈ $0.00017 per
item**. TaleTribe descriptions and chapter summaries are broadly comparable in
length to CMU plot summaries, so that is a fair estimate until there is a real run
to measure.

| | Cost |
|---|---|
| Per story, first ingest | ~$0.0002 |
| 100 published stories, from empty | ~$0.02 |
| 10,000 published stories, from empty | ~$2 |
| **A steady-state incremental run** | **~$0** |
| Local infrastructure | $0 |

**A steady-state run is essentially free, and that is the important number.** Two
caches make it so:

- **The embedding skip check.** A row whose `embed_input_sha` is unchanged is never
  re-embedded. Re-reading the entire published catalog costs one SQL query and no
  tokens.
- **The normalization cache**, keyed on the hash of the summary text. A story whose
  description and chapter summaries have not changed is not re-extracted.

So the recurring bill is proportional to *stories edited since the last run*, not to
catalog size. A nightly ingest over a stable catalog costs nothing.

Also free, and worth knowing: **vocabulary tuning.** The normalization cache stores
raw model output and filters on read, so changing `vocabularies.py` re-filters
cached extractions rather than regenerating them.

### Query time

| Path | LLM calls |
|---|---|
| `/recommend/behavioral` | **zero** — the taste vector is precomputed by `--refresh-only` |
| `/recommend/adhoc` with seed titles | zero — resolved by trigram match, embeddings already stored |
| `/recommend/adhoc` with free text | one embedding (~free) |
| `/recommend/adhoc` with `use_hyde` | one embedding + **one Gemini generation** |
| `/recommend/explain` | **one Gemini generation per uncached item** |

### The two directly metered paths

**HyDE and explanations bypass creditProxy and call Gemini directly**, so unlike
every other LLM call on the platform they are *not* credit-metered. Nothing debits a
user's balance for them. The burst controls are:

- `ExplanationCache` — keyed on
  `sha256(model | prompt_ver | story_id | embed_input_sha | query_fingerprint)`, so
  the same reader asking the same thing twice is free, and the key self-invalidates
  when the item's normalized text changes.
- `MAX_LLM_REQUESTS_PER_MINUTE_PER_USER` (default **6**) — a second, tighter rate
  bucket than the 30/min one that governs ranking.

The in-process bucket limits bursts but is multiplied by instance count. Durable
Postgres counters therefore add per-user defaults of 10 HyDE searches and 30
explanation requests per UTC day, plus platform-wide defaults of 1,000 for each
kind. Set a daily value to 0 only when intentionally disabling that ceiling.

## Useful inspection queries

```sql
-- provenance: mixed embedders here means trouble (see development-log 1.2)
SELECT embed_model, count(*) FROM recommendations.items
 WHERE embedding IS NOT NULL GROUP BY embed_model;

-- catalog health
SELECT count(*) rows, count(embedding) embedded, count(core_premise) premise,
       round(avg(cardinality(themes)),2) avg_themes
  FROM recommendations.items;

-- why did this book get these categories?
SELECT title, raw_genres, genres FROM recommendations.items WHERE title = 'Jane Eyre';

-- ingest history, including failures
SELECT id, kind, status, counts->>'upserted', error
  FROM recommendations.ingest_runs ORDER BY id DESC LIMIT 5;

-- the tunable knobs (editable at runtime, no redeploy)
SELECT key, value, description FROM recommendations.config ORDER BY key;
```

## Seed data (before you have real users)

The behavioral half of scoring needs reader signals. **Real ones do not come from
this repo** — story-data derives them from `story_likes`, `story_ratings` and
`reading_progress` into `recommendations.interactions`, because `reading_progress`
is private per-user data this service is not permitted to read. See
[jobs.md](jobs.md) and [security-and-roles.md](security-and-roles.md).

What lives here is the *synthetic* generator, which is the only way to exercise
scoring before real traffic exists:

```bash
# generate readers, load them, refresh everything scoring reads
python -m recommendation_engine.sync.seed --readers 250 --cooccurrence

# generate and inspect a sample without touching the database
python -m recommendation_engine.sync.seed --readers 5 --dry-run

# recompute aggregates from whatever is already in `interactions`
# — synthetic, real, or both. Run this after `story-data sync-recs`.
python -m recommendation_engine.sync.seed --refresh-only

# remove every synthetic reader
python -m recommendation_engine.sync.seed --purge
```

The JSONL import/export this script used to offer is gone. It was the contract
between recs and a Firestore exporter that no longer exists; signals now arrive by
SQL, in the same database.

### The record shape

`InteractionRecord` is now an in-memory structure rather than a file format, but
the fields are unchanged and story-data's SQL produces the same rows:

| Field | Why it is this shape |
|---|---|
| `story_id` | The story's own UUID. `items.id` is a surrogate key the database assigns, which no generator can know — the loader resolves it. |
| `kind` | Only `like`, `rating`, `progress` — exactly what the platform records. Nothing richer, because nothing richer exists. |
| `completion` | **Not a kind.** Derived from progress (last chapter + ≥90% scrolled). Two implementations of one rule — here and in story-data's SQL — pinned together by `TestCompletionNeedsLastChapterAndDeepScroll` in story-data. |
| `value` | Rating 1-5, or `scroll_percent` 0-1. Unused for `like`. |
| `occurred_at` | Makes a reload idempotent: a conflict keeps the more recent signal, so re-deriving progress *updates* a position rather than accumulating duplicates. |

### What synthetic data can and cannot tell you

It validates **mechanics**: the popularity term moves, the cold-start ramp engages,
a taste vector retrieves sensibly, co-occurrence builds. Verified end to end —
`interactions → n_interactions → α=0.29 → w_pop=0.071 → score shifted −0.0089`.

It cannot validate **taste**. Any offline quality metric computed on it measures
whether the recommender recovers the generator's own assumptions, which is circular.
Treat it exactly like `USE_MOCK=true` embeddings: the plumbing is real, the
semantics are not.

Every synthetic reader is prefixed `synth_`, so they can be purged and can never
quietly enter a real measurement. That prefix is also load-bearing across the repo
boundary: `story-data sync-recs` excludes `synth_%` from both its delete and its
insert, so a generated cohort survives a real sync instead of being wiped by it.

### Why it is generated in code, not by a language model

It must reference `story_id`s in *this* catalog; it needs a **power-law** popularity
distribution and per-reader affinity that a model will not hold coherently over
thousands of records; and it must be reproducible from a seed or eval runs are not
comparable. Personas (genre/theme affinity, activity tier, rating generosity) are
sampled from the catalog's own vocabulary — and can be supplied explicitly via
`synthetic.Persona` if you want hand-authored or LLM-authored ones.

The power law matters more than it looks: with uniform interaction counts the
popularity term would be flat, the Bayesian damping would never be exercised, and
P95 normalization would be meaningless.

## Production status

| | |
|---|---|
| **Scheduling.** Terraform creates the ordered nightly pipeline paused — see [jobs.md](jobs.md) | ready to enable after initial load |
| Eval harness — see [evaluation.md](evaluation.md) for the full plan | Phase 6 |
| Neon, Terraform, Cloud Run — see [deployment.md](deployment.md) | implemented; first rollout pending |

The signals path itself is *built* (`story-data sync-recs` → `--refresh-only`); it
has simply never run against real traffic. Until it does, `n_interactions` is 0,
the cold-start ramp keeps α at 0, and scoring is 100% semantic — the popularity and
CF terms are inert no matter what `recommendations.config` says.

Until the eval harness exists, quality is judged by reading result lists. *Dune*
returning three Dune sequels at diversity 0.18 is visibly mediocre, but nothing in
the system says so — so every tuning decision is currently a guess.
