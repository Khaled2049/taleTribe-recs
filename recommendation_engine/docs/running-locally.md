# Running locally

Everything except deployment runs on a laptop. All commands below were verified
working.

## Setup

```bash
cd taleTribe-agents

poetry install --with recommendations        # or: pip install asyncpg pgvector
docker compose -f recommendation_engine/docker-compose.yml up -d
python -m recommendation_engine.migrations.migrate
```

Postgres listens on **5434**, not 5432 or 5433, so it can't collide with a local
Postgres, creditProxy's stack, or story-data's stack (`dev-new.sh`), which already
claims 5433. The `pgvector/pgvector:pg16` image ships the extension prebuilt.

Verify:

```bash
python -m recommendation_engine.migrations.migrate --status
python -m recommendation_engine.server          # :8100
curl localhost:8100/health
```

`/health` returns **503 unless** pgvector ≥ 0.8.0, the HNSW index exists, and the
embedder reports 768 dimensions. Those failures are otherwise silent, so they're
asserted at startup.

## Ports

| Port | What |
|---|---|
| 5434 | pgvector Postgres |
| 8100 | recommendation service |
| 8080 | Firestore emulator (only needed for platform sync) |
| 8000 | the story agent, for reference |

## Configuration

Config comes from the repo-root `.env` (gitignored), loaded by `recommendation_engine/env.py`
from **all** entry points — servers and CLIs alike. `load_dotenv` doesn't override
already-set variables, so `FOO=bar python -m ...` still wins.

Local dev needs **zero** configuration: `RECS_DATABASE_URL` defaults to exactly what
docker-compose serves. That default is explicitly *rejected* when
`ENVIRONMENT=production`, so it can never ship.

For real embeddings and premises:

```bash
GOOGLE_AI_STUDIO_API_KEY=...      # and make sure USE_MOCK is not "true"
```

## Loading the catalog

```bash
# parse only — no API calls, no writes, no key needed. ~1.2s over the full 42MB.
python -m recommendation_engine.ingest.backfill --limit 2000 --dry-run

# plumbing test with deterministic offline embeddings
USE_MOCK=true python -m recommendation_engine.ingest.backfill --limit 300 --skip-normalization

# real: validate quality on a small slice first
python -m recommendation_engine.ingest.backfill --limit 200

# then the subset, rebuilding the HNSW graph after bulk load
python -m recommendation_engine.ingest.backfill --limit 2000 --rebuild-index
```

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

# from a reader's own history (falls back to popularity until Phase 3 lands)
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
export RECS_DATABASE_URL="postgresql://recs:recs@localhost:5434/recs"

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
 1. 0.6704  The Open Society and Its Enemies — Karl Popper (1945) [off-platform]
      sem=0.6704*1.00  pop=0.0000*0.00  cf=0.0000*0.00  alpha=0.00  src=1.00
```

- `alpha=0.00` and `w_sem=1.00` — cold-start ramp: zero interactions means 100%
  semantic, as designed.
- `pop=0.0000` — **nothing populates `item_stats` yet** (Phase 3), so the popularity
  term is implemented but currently inert.
- `cf=*0.00` — the deliberate stub.
- `[off-platform]` — a CMU seed-corpus book, not readable on TaleTribe.

**Results are not sorted by score.** MMR reorders for diversity, so a lower-scoring
item can appear above a higher one. If the list were strictly descending, MMR
wouldn't be doing anything.

## Tests

```bash
pytest tests/test_rec_*.py -q                    # this service (476)
pytest tests/ -q -m unit                         # no infrastructure needed
RECS_TEST_DATABASE_URL="postgresql://recs:recs@localhost:5434/recs" pytest tests/ -q
```

`RECS_TEST_DATABASE_URL` is deliberately **separate** from `RECS_DATABASE_URL`:
these tests write and delete rows, so pointing them at a real deployment must be a
deliberate act rather than an inherited environment.

Integration tests **self-skip** when it's unset. That's required, not stylistic:
`pytest.ini` has no `-m` exclusion and CI runs `pytest tests/`, and with no
`conftest.py` in this repo an env guard is the only convention-compatible way to keep
CI green.

## What things cost

Measured on the real 2,000-book run, at gemini-2.5-flash-lite / gemini-embedding-001
list pricing:

| | Cost |
|---|---|
| 200-book quality check | ~$0.03 |
| **2,000-book subset** | **$0.33** |
| Full 15,512-book corpus (projected) | ~$2.40 |
| Local infrastructure | $0 |
| ~1,000 test queries | well under a cent |

Budget **$5** and you can afford the full corpus plus a re-run.

Two things that keep costs down, both learned the hard way — see
[development-log Category 3](development-log.md#category-3-cost-and-economics):

- **Vocabulary tuning is free.** The normalization cache stores raw model output and
  filters on read, so changing the vocabulary re-filters cached extractions instead
  of regenerating them.
- **Batch size is a rate-limit decision.** 8 books/call turns 15,500 requests into
  ~1,940 — the difference between two weeks and two days on a free tier.

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

The behavioral half of scoring needs reader signals. Until Firestore is connected,
generate them — but note the structure, because it is what makes Phase 3 a drop-in:

```
synthetic generator ──┐
                      ├──→ InteractionRecord (JSONL) ──→ loader ──→ interactions
Firestore export ─────┘                                         ──→ item_stats
   (Phase 3)                                                    ──→ user_taste
```

Only the *source* changes later. Completion inference, engagement weighting,
popularity aggregation and taste-vector construction are written and tested once.

```bash
# generate readers, load them, refresh everything scoring reads
python -m recommendation_engine.sync.seed --readers 250 --cooccurrence

# inspect the canonical format without touching the database
python -m recommendation_engine.sync.seed --readers 5 --out /tmp/seed.jsonl --dry-run

# load a file produced anywhere else — this is Phase 3's entry point too
python -m recommendation_engine.sync.seed --load /tmp/seed.jsonl

# recompute aggregates from existing interactions
python -m recommendation_engine.sync.seed --refresh-only

# remove every synthetic reader
python -m recommendation_engine.sync.seed --purge
```

### The canonical record

One JSON object per line. Any source that emits these works:

```json
{"user_id": "synth_00042", "source": "cmu", "source_id": "620",
 "kind": "rating", "value": 4.0, "occurred_at": "2026-07-14T09:31:00Z"}
{"user_id": "synth_00042", "source": "cmu", "source_id": "843",
 "kind": "progress", "value": 0.95, "chapter_index": 11, "total_chapters": 12,
 "occurred_at": "2026-07-15T20:02:00Z"}
```

| Field | Why it is this shape |
|---|---|
| `source` + `source_id` | The **external** identity. `items.id` is a surrogate key the database assigns, which no generator or Firestore export can know. |
| `kind` | Only `like`, `rating`, `progress` — exactly what Firestore has. Nothing richer, because nothing richer exists. |
| `completion` | **Not a kind.** Derived from progress (last chapter + >90% scrolled), once, in the loader — the platform emits no completion event. |
| `value` | Rating 1-5, or `scroll_percent` 0-1. Unused for `like`. |
| `occurred_at` | ISO-8601. Makes a reload idempotent: a conflict keeps the more recent signal, so re-exporting progress *updates* a position rather than accumulating duplicates. |

### What synthetic data can and cannot tell you

It validates **mechanics**: the popularity term moves, the cold-start ramp engages,
a taste vector retrieves sensibly, co-occurrence builds. Verified end to end —
`interactions → n_interactions → α=0.29 → w_pop=0.071 → score shifted −0.0089`.

It cannot validate **taste**. Any offline quality metric computed on it measures
whether the recommender recovers the generator's own assumptions, which is circular.
Treat it exactly like `USE_MOCK=true` embeddings: the plumbing is real, the
semantics are not.

Every synthetic reader is prefixed `synth_`, so they can be purged and can never
quietly enter a real measurement.

### Why it is generated in code, not by a language model

It must reference `source_id`s in *this* catalog; it needs a **power-law** popularity
distribution and per-reader affinity that a model will not hold coherently over
thousands of records; and it must be reproducible from a seed or eval runs are not
comparable. Personas (genre/theme affinity, activity tier, rating generosity) are
sampled from the catalog's own vocabulary — and can be supplied explicitly via
`synthetic.Persona` if you want hand-authored or LLM-authored ones.

The power law matters more than it looks: with uniform interaction counts the
popularity term would be flat, the Bayesian damping would never be exercised, and
P95 normalization would be meaningless.

## Not built yet

| | |
|---|---|
| Firestore sync — the thing that makes `pop` real and lets behavioral mode exist | Phase 3 |
| Eval harness — see [evaluation.md](evaluation.md) for the full plan | Phase 6 |
| Neon, Terraform, Cloud Run, Firebase Functions — see [deployment.md](deployment.md) | deferred |

Until the eval harness exists, quality is judged by reading result lists. *Dune*
returning three Dune sequels at diversity 0.18 is visibly mediocre, but nothing in
the system says so — so every tuning decision is currently a guess.
