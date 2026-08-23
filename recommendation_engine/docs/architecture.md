# Architecture — layers, storage, and the reasoning

Focused on *why* things are shaped as they are. [concepts.md](concepts.md) covers the
algorithms; [file-reference.md](file-reference.md) lists what each module contains.

---

## The layers

Six of them, each with one job. The layering is what makes the service testable and
what will make the Firestore work a drop-in rather than a rewrite.

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

**Ingest is split from its sources.** This is the most important boundary in the
codebase:

```
synthetic generator ──┐
                      ├──→ InteractionRecord ──→ loader ──→ interactions
Firestore export ─────┘         (canonical)               ──→ item_stats
   (not written yet)                                      ──→ user_taste
```

Connecting Firestore means writing one reader that emits the canonical record.
Completion inference, engagement weighting, popularity aggregation and taste-vector
construction are already built and tested against whichever source is plugged in.

**External services return `None` rather than raising.** `build_client()` gives back
`None` when there's no API key, so the service still starts and still serves the
paths that need no LLM. Ranking works; only HyDE and explanations go dark.

---

## Why Postgres + pgvector

The parent repo already uses **Firestore's native vector search** for chapter RAG. This
service deliberately doesn't. Four reasons, in order of how much they mattered.

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

Reader-facing recommendations need metadata filters — genre, themes, length, author,
source. As [concepts.md §2](concepts.md#the-filtering-problem-and-iterative-scans)
explains, a filter applied *after* an approximate vector search silently under-returns.
pgvector 0.8.0's `hnsw.iterative_scan` solves it. Firestore's vector search has no
equivalent, and its `find_nearest` composes with equality filters only via
pre-provisioned composite vector indexes.

### 3. Compute isolation

Vector scans are CPU- and memory-hungry and bursty. Running them on their own
instance or read replica — never sharing compute with the billing workload — was a hard
requirement. Postgres makes that a DSN; Firestore doesn't expose the concept.

### 4. Relational work that isn't vector work

Half the service is ordinary SQL: aggregating interactions into `item_stats`, a
self-join for co-occurrence, trigram fuzzy title matching, percentile calculations,
`ON CONFLICT` upserts with precondition guards. In Firestore each of those is a
read-modify-write loop in application code. In Postgres they're one statement each.

### What we gave up

Honest costs of the choice:

- **A new datastore in a Firestore-first codebase.** Two mental models, two backup
  stories, two sets of credentials.
- **No real-time listeners.** Firestore's `onSnapshot` is genuinely nice, and the
  frontend already uses it for job status. Recommendations poll or re-request instead.
- **Net-new tooling.** No SQLAlchemy, no Alembic, no migration harness existed in the
  repo. We wrote a ~185-line runner (see below).
- **A second index to keep in lockstep.** The 768-dimension contract now has three
  consumers: this service, `firestore.indexes.json`, and the shared embedding provider.

### Why not a dedicated vector database

Pinecone, Weaviate, Qdrant and friends are excellent, and would have been wrong here:

- **creditProxy already uses Neon Postgres in production**, so Postgres is an existing
  operational dependency. A vector DB would be a genuinely new vendor, new bill, new
  failure mode.
- At 16.5k rows — even at 200k — this is comfortably within what pgvector handles well.
  Dedicated vector databases start earning their cost in the millions.
- **Half the workload is relational** (see reason 4). A vector DB would mean *both*
  Postgres and a vector store, joined in application code.

---

## Schema

Nine domain tables in an isolated `recommendations` schema, plus `schema_migrations`
created by the migration runner (10 in `information_schema`). The isolation matters:
this can live in its own database, or alongside others, without name collisions.

| Table | Holds | Written by |
|---|---|---|
| `items` | the catalog — one row per book, with its vector | ingest, sync |
| `item_stats` | per-item aggregates + materialized `pop_score` | `sync/stats.py` |
| `interactions` | raw reader signals | `sync/interactions.py` |
| `item_cooccurrence` | item-item CF similarities | `sync/stats.py` (stub) |
| `user_taste` | one precomputed vector per reader | `sync/stats.py` |
| `explanation_cache` | generated "why you'll like this" text | `explain.py` |
| `normalization_cache` | LLM-derived premise/themes/tone, keyed by summary hash | `ingest/normalize_llm.py` |
| `ingest_runs` | run bookkeeping, for resume and audit | `ingest/backfill.py` |
| `config` | tunable scoring knobs | operator, by hand |

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
the HNSW index exists, and the embedder reports 768 dimensions. Those are precisely
the conditions whose absence causes *silent* degradation, so they're asserted loudly at
startup instead of being discovered by a reader.

---

## Deployment shape (designed, not built)

Summarised here; the full treatment — services, container, security, scale, production
readiness — is in [deployment.md](deployment.md).

A separate Cloud Run service built from this package, with its own Dockerfile,
Terraform root and workflow — sharing the repo but not the runtime. Deps live in a
Poetry **optional** group (`--with recommendations`) so the story agent's image never
grows `asyncpg`/`pgvector`.

Deferred by choice, so the design could settle first. Four things will need attention:

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
