# Architecture — layers, storage, and the reasoning

Focused on *why* things are shaped as they are. [concepts.md](concepts.md) covers the
algorithms; [file-reference.md](file-reference.md) lists what each module contains.

---

## The layers

Six of them, each with one job. The layering is what makes the service testable:
the algorithm layer can be tested with no database, and the orchestration layer can
be tested with a fake retriever.

```
6. HTTP            routes.py, server.py
5. Orchestration   pipeline.py                    ← shared by HTTP and CLI
4. Algorithms      scoring.py, fusion.py, mmr.py  ← pure functions, no I/O
3. Data access     retrieval.py, db.py
2. Ingest          ingest/*, sync/*               ← writes the catalog and signals
1. External        llm.py, embeddings.py           ← Gemini, wrapped
```

### Why the layers are split this way

**Algorithms have no I/O.** `scoring.py`, `fusion.py` and `mmr.py` take numbers and
return numbers. No database, no HTTP, no clock. That's why they're tested against
hand-computed expected values rather than snapshots — a snapshot test would happily
lock in a wrong formula.

**Orchestration is shared, not duplicated.** `pipeline.rank()` is called by both the
HTTP routes and `query_cli.py`. If they had separate implementations, the CLI would be
useless for the debugging it exists for, because it could rank differently from
production.

**The catalog is pulled; reader signals are pushed.** This is the most important
boundary in the design, and it is a *privacy* boundary, not a convenience:

```
                    ┌─── recs PULLS ────────────────────────────┐
  stories, story_tags, chapters, chapter_summaries  ──→  ingest/platform.py
                    │       (published stories are public data)  │
                    └───────────────────────────────────────────┘

                    ┌─── story-data PUSHES ─────────────────────┐
  story_likes, story_ratings, reading_progress  ──→  sync-recs  ──→ interactions
                    │  (reading_progress is private; recs may    │
                    │   not read it, so it never does)           │
                    └───────────────────────────────────────────┘

  interactions ──→ sync/stats.py ──→ item_stats, user_taste, item_cooccurrence
                   (stays inside the `recommendations` schema, always)
```

recs may read `stories` because published stories are public data — story-data's own
`ListPublicStories` serves them to any caller. It may **not** read `reading_progress`,
which is strictly private per-user history. So story-data derives the signals it is
allowed to expose (`internal/store/recommendations.go`) and writes them into
`recommendations.interactions`; everything downstream of that table is this service's
and never leaves the schema.

That is enforced in the database, not by convention: migration `000020` grants the
`recs_service` role usage on the `recommendations` schema only. Without a grant,
co-locating the two schemas would silently hand recs a second, unaudited path to
every reader's history. See [security-and-roles.md](security-and-roles.md).

The synthetic generator (`sync/synthetic.py`) writes through the same
`interactions.load()` path, prefixed `synth_`, and story-data's sync excludes that
prefix from both its delete and its insert — so generated and real readers coexist.

**External services return `None` rather than raising.** `build_client()` gives back
`None` when there's no API key, so the service still starts and still serves the
paths that need no LLM. Ranking works; only HyDE and explanations go dark.

---

## Why Postgres + pgvector

This decision was made when the platform was Firestore-first and chapter RAG used
**Firestore's native vector search**. It has aged well: story-data is now Postgres,
the agents service's vector chunks are pgvector, and the Firestore vector path has
since been deleted outright. What follows is the original reasoning, which is still
the reasoning — with a note at the end on which costs went away.

Four reasons, in order of how much they mattered.

### 1. Local development actually works

`VectorStore` in the story agent returns `[]` on the Firestore emulator, because
`find_nearest` doesn't exist there. Chapter RAG therefore *silently degrades* in local
development — retrieval returns nothing and no error is raised.

pgvector has no such gap. `docker compose up` gives the same engine, the same index
type and the same query planner as production. Everything in this service — index
behaviour, `ef_search` tuning, iterative scans, filtered recall — is exercised on a
laptop. Given that the dominant failure mode in retrieval is *silence*, being able to
observe the real thing locally was the deciding factor.

### 2. Filtered recall

Reader-facing recommendations need metadata filters — genre, themes, length, author.
As [concepts.md §2](concepts.md#the-filtering-problem-and-iterative-scans)
explains, a filter applied *after* an approximate vector search silently under-returns.
pgvector 0.8.0's `hnsw.iterative_scan` solves it. Firestore's vector search has no
equivalent, and its `find_nearest` composes with equality filters only via
pre-provisioned composite vector indexes.

### 3. Compute isolation

Vector scans are CPU- and memory-hungry and bursty. Running them on their own
instance or read replica — never sharing compute with the billing workload — was a hard
requirement. Postgres makes that a DSN — `RECS_DATABASE_URL_RO` points at a
dedicated read compute in production — and Firestore doesn't expose the concept.

### 4. Relational work that isn't vector work

Half the service is ordinary SQL: aggregating interactions into `item_stats`, a
self-join for co-occurrence, trigram fuzzy title matching, percentile calculations,
`ON CONFLICT` upserts with precondition guards. In Firestore each of those is a
read-modify-write loop in application code. In Postgres they're one statement each.

### What we gave up — and what we got back

Honest costs of the choice, and their status now that story-data is Postgres too:

| Cost | Status |
|---|---|
| **A new datastore in a Firestore-first codebase** — two mental models, two backup stories | **Gone.** Postgres is the platform's system of record; recs is no longer the odd one out. |
| **Net-new tooling** — no migration harness existed, so we wrote a ~185-line runner | **Gone.** Deleted. story-data's goose runner owns the schema; this service migrates nothing. |
| **A second database to operate** — its own compose file on `:5434`, its own backups | **Gone.** The `recommendations` schema lives in story-data's database. |
| **No real-time listeners.** Firestore's `onSnapshot` is genuinely nice | **Still true**, and fine — recommendations are re-requested, not streamed. |
| **A dimension contract with several consumers.** 768 must agree everywhere | **Still true, and now the sharpest edge.** `embedding_provider.py` is *duplicated* between this repo and taleTribe-agents rather than shared. `/health` asserts the serving dimension and model, which is the only thing making a drift loud. |

The interesting outcome: three of the five costs were paid off not by work in this
repo but by the rest of the platform moving to Postgres independently. See
[migration-to-story-data.md](migration-to-story-data.md).

### Why not a dedicated vector database

Pinecone, Weaviate, Qdrant and friends are excellent, and would have been wrong here:

- **creditProxy already uses Neon Postgres in production**, so Postgres is an existing
  operational dependency. A vector DB would be a genuinely new vendor, new bill, new
  failure mode.
- At the catalog sizes in play — a few thousand published stories, even a few hundred
  thousand — this is comfortably within what pgvector handles well. Dedicated vector
  databases start earning their cost in the millions of rows.
- **Half the workload is relational** (see reason 4). A vector DB would mean *both*
  Postgres and a vector store, joined in application code.

---

## Schema

Nine tables in an isolated `recommendations` schema **inside story-data's database**,
defined by `story-data/migrations/000019_recommendations_schema.sql`. There is no
`schema_migrations` table here — story-data's goose runner owns that, and this
service has no migration runner at all.

The schema isolation is doing real work now that it shares a database: it prevents
name collisions, it is the unit the `recs_service` grant is scoped to, and it is what
lets `/health` answer "has story-data migrated yet?" with one `information_schema`
lookup.

| Table | Holds | Written by |
|---|---|---|
| `items` | the catalog — one row per published story, with its vector | `ingest/platform.py` |
| `interactions` | raw reader signals | **story-data** (`sync-recs`); `sync/interactions.py` for synthetic only |
| `item_stats` | per-item aggregates + materialized `pop_score` | `sync/stats.py`; `views` column by story-data |
| `item_cooccurrence` | item-item CF similarities | `sync/stats.py` (stub — `w_cf_ceiling` is 0) |
| `user_taste` | one precomputed vector per reader | `sync/stats.py` |
| `explanation_cache` | generated "why you'll like this" text | `explain.py` |
| `normalization_cache` | LLM-derived premise/themes/tone, keyed by summary hash | `ingest/normalize_llm.py` |
| `ingest_runs` | run bookkeeping, for resume and audit | `ingest/platform.py`, `sync/stats.py` |
| `config` | tunable scoring knobs | operator, by hand |

**`items.story_id` is a real foreign key** to `stories(id)` with `ON DELETE CASCADE`,
not a loose string key. That is what makes a deleted story disappear from the catalog
by construction rather than by a reconciliation job — the only eligibility change
needing a job is *unpublishing*, which `_retire_unpublished` handles.

Two tables have **two writers**, and both cases are deliberate:

- `interactions` — story-data writes real signals, recs writes `synth_`-prefixed ones.
  story-data's delete is scoped `WHERE user_id NOT LIKE 'synth\_%'`, so neither
  clobbers the other.
- `item_stats` — story-data upserts *only* the `views` column (an anonymous global
  counter recs cannot derive from per-reader signals); recs writes every other column
  and never `views`. A concurrent refresh is therefore safe.

### Two design choices in the schema worth calling out

**`config` lives in the database, not in env vars.** Every scoring knob —
`w_pop_ceiling`, `mmr_lambda`, `ramp_n50`, `rrf_k` — is a row. Ranking can be retuned
with an `UPDATE` and no redeploy, and a bad tune is one `UPDATE` from being reverted.
The service caches it for 60 seconds. `ScoringConfig.from_mapping()` clamps every value
on read, so a typo degrades ranking rather than taking the service down.

**`item_stats` is materialized, not a view.** `pop_score` involves Bayesian damping and
log compression over global aggregates. Recomputing that per query would put a
full-table aggregate inside the retrieval path. It's a global property that changes
slowly, so it's computed on a schedule.

### Provenance columns

`items` carries `embed_model`, `embed_task_type`, `embed_input`, `embed_input_sha` and
`embedded_at`. That's more bookkeeping than it looks like it needs, and each column
prevents a specific silent failure:

- `embed_input_sha` — lets a re-run skip unchanged rows. Resumability.
- `embed_model` — vectors from different providers occupy different spaces. Without
  this column, switching from the mock embedder to Gemini left 292 md5-derived vectors
  in a real corpus, matching nothing, raising nothing. The text hash cannot catch a
  provider swap because the text didn't change.
- `embed_task_type` — documents and queries project differently; a corpus mustn't mix.
- `embed_input` — the exact text embedded, stored verbatim, so a surprising result can
  be explained rather than guessed at.

### Migrations: raw SQL, not Alembic

Numbered `.sql` files plus a ~185-line runner. Deliberate:

- There's **no ORM**, so Alembic's autogenerate — its main value — has nothing to diff.
- Hand-written vector DDL (HNSW build parameters, partial index predicates, extension
  guards) is *more* readable as SQL than as Python.
- It matches `creditProxy/migrations/001_init.sql`, the existing convention.

The runner does three things worth having: a **session advisory lock** so two
concurrent runners can't double-apply; a **checksum** per file, so editing an
already-applied migration fails loudly rather than letting git and the database
diverge silently; and a **per-file transaction**, so a failure leaves nothing behind
and the fixed file can simply be re-run.

---

## Request flow

### Behavioral (a reader's own history)

```
POST /recommend/behavioral
  → rate limit (30/min)
  → SELECT user_taste WHERE user_id = $1
      ├─ found, n_signals ≥ 3 → use the stored vector, exclude seeds + suppressed
      └─ not found            → mode: "popular", no vector
  → pipeline.rank(...)
  → 200
```

**No embedding call on this path at all.** The reader's taste vector was computed at
sync time, so this is one database round trip. That's the whole reason it's the fast
mode.

### Ad-hoc ("I liked these" / free text)

```
POST /recommend/adhoc
  → rate limit (30/min, plus 6/min if HyDE will run)
  → for each named book: trigram resolve → collect its stored vector, exclude it
  → if a prompt is given:
      ├─ HyDE enabled → LLM writes a hypothetical catalog entry → embed that
      └─ otherwise    → embed the prompt directly
  → pipeline.rank(...)
  → attach a deterministic explanation_cache_key per item
  → 200
```

### Inside `pipeline.rank`

```
1. retriever.knn_many(vectors, limit = top_k × 5)   ← concurrent, one per seed
2. one seed  → sem = cosine
   many      → sem = normalized RRF
3. CF lookup (short-circuits to zeros while w_cf_ceiling = 0)
4. blend() per candidate → score + full attribution
5. diversify() → MMR selects top_k
6. RankedResult{items, degraded, diversity, candidates_considered}
```

The candidate pool is `top_k × 5`. MMR needs room to find variety; too much and it
starts promoting weak matches for novelty's sake.

---

## Degradation posture

Every failure path returns something useful, mirroring how the story agent's
`VectorStore` degrades when its index is missing.

| Condition | Behaviour |
|---|---|
| Statement timeout under a selective filter | partial results, `degraded: true` |
| No embedding provider configured | popularity-ranked fallback |
| No LLM configured | ranking works; explanations return `null` |
| Reader with < 3 signals | `mode: "popular"` |
| HyDE generation fails | embed the raw query instead |
| One explanation fails in a batch | that item gets `null`, the rest succeed |
| Low-confidence normalization | row stored, `is_eligible = false`, no vector |

**`/health` is the deliberate exception.** It returns **503** unless pgvector ≥ 0.8.0,
the HNSW index exists, the `recommendations` schema is present, the embedder reports
768 dimensions, and the serving embedder agrees with the model the catalog was built
by. Those are precisely the conditions whose absence causes *silent* degradation, so
they're asserted loudly at startup instead of being discovered by a reader.

Two of the five are new since the schema moved into story-data, and both catch
ordering mistakes the old single-database setup could not produce:

- **`schema_present`** — recs no longer creates its own schema, so starting it before
  story-data has migrated is now possible. Without this check every request would
  return an opaque 500.
- **`embed_model_ok`** — the ingest already refuses to reuse a vector from a different
  provider, but nothing stopped the *query* side drifting. Embedding a query with one
  model and searching a catalog built by another returns plausible, confidently
  ranked nonsense, with no error.

[runbook.md](runbook.md) has the field-by-field triage.

---

## Deployment shape

Summarised here; the full treatment — services, container, security, scale, production
readiness — is in [deployment.md](deployment.md).

The service has its own container, Terraform state, Cloud Run service, two Cloud Run
jobs, deployment workflows, and a paused nightly scheduler. The Workflow preserves
the required ingest → signal sync → aggregate refresh ordering.

Five operational constraints remain important:

1. **Neon, with a dedicated read-only compute** for serving. Vector scans never share
   compute with billing.
2. **`statement_cache_size=0`** on the asyncpg pool. Transaction-mode pooling plus
   asyncpg's prepared-statement cache produces intermittent
   `prepared statement already exists` errors that only appear under concurrency —
   i.e. never in local testing.
3. **Autosuspend disabled** on the read compute. Neon suspends idle computes; resume
   is 500ms–3s, which would dominate every cold request.
4. **SSE across Firebase Functions.** Functions gen2 buffers responses, so a streaming
   endpoint can't be proxied. Hence the two-stage design: sync explanations work
   through the Function bridge today; streaming needs the browser to reach the service
   directly, which in turn needs in-process Firebase token verification.
   `GET /recommend/explain/stream` is built and tested but currently unreachable from
   a browser for exactly this reason.
5. **Deploy ordering is a hard constraint.** story-data must deploy and
   migrate *before* recs starts, or `schema_present` fails the health check and the
   rollout is rejected. The same applies in CI: `pr-check` cannot build its own test
   schema any more, because this repo no longer owns the migrations.
