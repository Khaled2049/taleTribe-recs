# Runbook — diagnosing the recommendation service

Symptom-first. Find the symptom, follow the tree.

**The thing to internalise before anything else:** in a retrieval system the dominant
failure mode is *silence*. A wrong embedding model, a mixed task type, a missing
index, a stale aggregate — none of these raise an error. They return a confidently
ranked list of the wrong things. Almost everything below is about making a silent
failure visible.

Everything here assumes you can reach the database:

```bash
psql "postgresql://postgres:postgres@localhost:5433/story_data"   # local
curl -s localhost:8100/health | jq                                 # local
```

---

## Start here: read `/health`

`/health` returns **503** if any check fails. It is the fastest triage available and
every field means something specific.

```json
{
  "status": "ok",
  "embedder": "GoogleAIEmbeddingProvider",
  "embedding_dimension": 768,
  "embedding_dimension_ok": true,
  "catalog_embed_model": "google:gemini-embedding-001",
  "embedder_model": "google:gemini-embedding-001",
  "embed_model_ok": true,
  "database": {
    "connected": true,
    "pgvector_version": "0.8.6",
    "pgvector_ok": true,
    "hnsw_index_present": true,
    "schema_present": true,
    "item_count": 19,
    "eligible_count": 15
  }
}
```

| Field | False means | Fix |
|---|---|---|
| `database.connected` | Wrong DSN, database down, or network | Check `RECS_DATABASE_URL`. This is the only failure that is *loud* downstream too. |
| `database.schema_present` | **story-data has not migrated this database.** An ordering mistake, not a fault. | `(cd repos/story-data && make migrate)`. In production, story-data must deploy first. |
| `database.pgvector_ok` | pgvector < 0.8.0 | `hnsw.iterative_scan` is unavailable, so *filtered* queries under-return silently. Upgrade the extension. |
| `database.hnsw_index_present` | The index named `items_embedding_hnsw_idx` is missing | Queries fall back to sequential scan: correct results, terrible latency. Usually means a `--rebuild-index` run died partway. |
| `embedding_dimension_ok` | The embedder is not producing 768 dims | Every vector written from now on is unusable. Check `EXPECTED_EMBEDDING_DIM` in `embedding_provider.py` — it is **duplicated** with taleTribe-agents. |
| `embed_model_ok` | **The serving embedder disagrees with the catalog.** | See below — this is the subtle one. |

### `embed_model_ok: false`

Vectors from different models occupy different spaces. Embedding a query with one
model and searching a catalog built by another returns plausible-looking nonsense,
ranked confidently, with no error. Compare the two fields:

```
"catalog_embed_model": "google:gemini-embedding-001",
"embedder_model":      "MockEmbeddingProvider"
```

Almost always one of two causes:

- **`GOOGLE_AI_STUDIO_API_KEY` is missing**, so the service fell back to the mock
  embedder. Most common in a fresh deploy where the secret was not bound. Set the
  key.
- **`USE_MOCK=true` is set** somewhere it should not be.

The reverse — a mock-built catalog served by Gemini — means the catalog was loaded
with `USE_MOCK=true`. Re-ingest with `--full`.

> **Caveat worth knowing.** `catalog_embed_model` reports the *majority* model among
> retrievable rows. If most rows match the serving embedder and a minority do not,
> health stays green while that slice is unusable. To check properly:
>
> ```sql
> SELECT embed_model, embed_task_type, count(*)
>   FROM recommendations.items
>  WHERE is_eligible AND embedding IS NOT NULL
>  GROUP BY 1, 2;
> ```
>
> **More than one row here is a problem**, whatever `/health` says.

---

## "Recommendations come back empty"

```
                    curl -s localhost:8100/health | jq .database
                                    │
      ┌─────────────────────────────┼─────────────────────────────┐
      │                             │                             │
 item_count = 0              eligible_count = 0           both look fine
      │                      but item_count > 0                   │
      ▼                             ▼                             ▼
The ingest has              Everything is retired         Filters, or an
never run                   or low-confidence             empty result is
                                                          correct
```

### `item_count = 0` — the ingest has never run

```bash
python -m recommendation_engine.ingest.platform --dry-run   # what would be read?
```

If the dry run reports 0 stories, the problem is upstream: there are no *published*
stories. Confirm with `SELECT count(*) FROM stories WHERE is_published;`.

### `eligible_count = 0` but `item_count > 0`

Rows exist but none are in the partial HNSW index. Three causes, distinguishable:

```sql
SELECT is_eligible,
       embedding IS NOT NULL AS has_vector,
       confidence,
       count(*)
  FROM recommendations.items
 GROUP BY 1, 2, 3 ORDER BY 4 DESC;
```

- **`is_eligible = false`, no vector, low `confidence`** → normalization came back
  unreliable. The description was too thin to derive a premise from. Expected for
  one-sentence descriptions; a problem if it is most of the catalog.
- **`is_eligible = false`, has a vector** → `_retire_unpublished` marked them. The
  stories were unpublished. Correct behaviour.
- **`is_eligible = true`, no vector** → should be impossible; the ingest sets
  `is_eligible` from whether a vector was produced. Investigate.

### Both look fine but results are still empty

- **Filters.** Every filter is `AND`-ed, and `genres` comes from the story's
  controlled `category` while `themes` comes from LLM extraction. Asking for a theme
  no story was tagged with returns nothing, correctly. Re-run without filters.
- **Exclusions.** Behavioral mode excludes `seed_item_ids` (already read) and
  `suppressed_item_ids` (rated ≤2). A reader who has read most of a small catalog
  legitimately has nothing left.
- **`--top-k` vs. catalog size.** With 19 stories and a request for 12 after MMR
  diversification, you are near the floor.

---

## "Recommendations are wrong / irrelevant"

The silent-failure family. Work down in order of likelihood:

**1. Is the embedder aligned?** `embed_model_ok` above, plus the mixed-model SQL. This
explains more "the results are nonsense" reports than everything else combined.

**2. Are task types mixed?**

```sql
SELECT embed_task_type, count(*) FROM recommendations.items
 WHERE is_eligible GROUP BY 1;
```

Must be `RETRIEVAL_DOCUMENT` for every catalog row. `RETRIEVAL_DOCUMENT` and
`RETRIEVAL_QUERY` project differently; mixing them degrades recall with no error.

**3. Is the index present but under-tuned?** `RECS_HNSW_EF_SEARCH` (default 100) must
be at least 2× `RECS_CANDIDATE_POOL` or the index cannot return a full pool. The
service logs `ef_search_too_low` at startup — check the logs rather than guessing.

**4. Is it actually MMR?** **Results are deliberately not sorted strictly by score.**
The final pass is MMR (λ = 0.7), which trades some relevance for diversity. A #3
result scoring higher than #2 is the algorithm working. Confirm with:

```bash
python -m recommendation_engine.query_cli --title "..." --scores
```

`--scores` prints per-term attribution. If `sem` is ordered but the output is not,
that is MMR.

**5. Is a poisoned row winning?** A low-confidence normalization produces a
confidently hallucinated premise. Those are supposed to be excluded, but check the
top result's stored text:

```sql
SELECT title, confidence, core_premise, themes, tone, embed_input
  FROM recommendations.items WHERE id = <the id>;
```

`embed_input` is stored verbatim precisely so a result can be explained and
reproduced. If it reads like nonsense, the vector is nonsense.

---

## "Behavioral returns 'popular' instead of personalised"

The response says `"mode": "popular"` and `"n_signals": 0`. This is a fallback, not
an error — but it should be rare once signals flow.

```sql
-- does this reader have a taste vector?
SELECT user_id, n_signals, array_length(seed_item_ids, 1) AS seeds, computed_at
  FROM recommendations.user_taste WHERE user_id = '<firebase-uid>';
```

| Finding | Meaning |
|---|---|
| No row at all | Either the reader has no signals, or `--refresh-only` has not run since they arrived |
| Row exists, `n_signals < 3` | Working as designed — three signals is the floor |
| Row exists with `n_signals ≥ 3` but mode is still `popular` | Real bug. The route reads `user_taste` by `user_id`; check the uid matches exactly. |

If **no reader** has a taste vector, the pipeline has not run:

```sql
SELECT kind, count(*) FROM recommendations.interactions GROUP BY kind;
```

Empty → `story-data sync-recs` has never run. Populated but `user_taste` is empty →
`seed.py --refresh-only` has not run since. See [jobs.md](jobs.md).

**Today this is the expected state platform-wide.** Neither signals job is scheduled,
so every reader falls through to popularity, and popularity itself is flat because
`item_stats` is empty.

---

## "A story I published isn't being recommended"

Follow the path in order — each stage is a place it can stop:

```sql
-- 1. Is it actually published?
SELECT id, title, is_published, updated_at FROM stories WHERE id = '<uuid>';

-- 2. Did the ingest reach it?
SELECT story_id, is_eligible, confidence, embedding IS NOT NULL AS has_vector,
       embed_model, embedded_at, updated_at
  FROM recommendations.items WHERE story_id = '<uuid>';

-- 3. Has the ingest run past its updated_at?
SELECT kind, status, cursor, finished_at
  FROM recommendations.ingest_runs
 WHERE kind = 'platform_sync' AND status = 'completed'
 ORDER BY finished_at DESC LIMIT 1;
```

| Finding | Cause | Fix |
|---|---|---|
| No `items` row, and the story's `updated_at` > cursor | Not ingested yet | Run the ingest |
| No `items` row, and `updated_at` < cursor | The ingest skipped past it — see the stranding gap below | `--full` |
| Row exists, `is_eligible = false`, low `confidence` | Description too thin to normalize reliably | Ask the author to expand the description, or write chapter summaries |
| Row exists, `is_eligible = false`, has a vector | It was unpublished at some point and retired | Re-publish; the next ingest re-embeds it |
| Row is fine but it does not rank | It is competing on content similarity alone | Not a bug — see cold start in [overview.md](overview.md) |

> **⚠ Known gap — stranded rows.** A story stored `is_eligible = false` after a
> low-confidence normalization is **never revisited by an incremental run**: the
> cursor has already advanced past its `updated_at`. Only `--full` recovers it, and
> `--full` is not scheduled. If an author fixes a thin description, the story's
> `updated_at` changes and it *will* be picked up — but a story whose text never
> changes stays invisible forever. Until this is fixed, run `--full` periodically.

---

## "An unpublished story is still being recommended"

`_retire_unpublished` runs at the end of every ingest and flips `is_eligible = false`
for any item whose story is no longer published. If a retired story is still showing:

```sql
SELECT i.story_id, i.is_eligible, s.is_published
  FROM recommendations.items i JOIN stories s ON s.id = i.story_id
 WHERE s.is_published = false AND i.is_eligible;
```

Any rows → the ingest has not run since it was unpublished. Run it; retirement is not
gated by the cursor, so even a no-op incremental run fixes this.

**A deleted story needs no job.** `items.story_id` cascades on delete. If a deleted
story is still being recommended, the foreign key is missing — check the schema.

---

## "Explanations are blank or slow"

`explanation: null` for an item is **by design** under failure — a shelf with nine
reasons and one blank beats an error page. Causes, in order:

- **No `GOOGLE_AI_STUDIO_API_KEY`.** `build_client()` returns `None` without one, and
  the service still starts. Ranking works; explanations go dark entirely.
- **The 6/min LLM bucket.** `MAX_LLM_REQUESTS_PER_MINUTE_PER_USER` is separate from
  and much tighter than the 30/min ranking bucket. A reader clicking through a shelf
  fast will hit it.
- **Cache miss.** The key is
  `sha256(model | prompt_ver | story_id | embed_input_sha | query_fingerprint)`. It
  self-invalidates when the item is re-normalized, which is intended — but it also
  means a `--full` re-ingest that changes `embed_input_sha` invalidates every
  explanation for that item at once.

```sql
SELECT count(*), sum(hit_count) FROM recommendations.explanation_cache;
```

**Slow** is usually just the model. The Function allows 90s for
`explainRecommendations` and 60s for the upstream call, and explanations are
generated per uncached item.

---

## "Requests are timing out" / `degraded: true`

`degraded: true` in a response means a **statement timeout** during retrieval
(`RECS_STATEMENT_TIMEOUT_MS`, default 2000). Partial results are returned rather than
a 500, deliberately.

Under a highly selective filter, HNSW iterative scan can walk a long way looking for
enough matching rows. Options in order of preference:

1. **Loosen the filter.** A filter matching 0.1% of the catalog is the actual cause.
2. **Check the index exists** — without it every query is a sequential scan.
3. Raise `RECS_STATEMENT_TIMEOUT_MS` only if you have confirmed the query is
   legitimately slow rather than the index being absent.

---

## The four silent-failure queries

Worth turning into log-based metrics before going to production:

```sql
-- 1. A failed job = a silently stale catalog
SELECT kind, error, finished_at FROM recommendations.ingest_runs
 WHERE status = 'failed' ORDER BY finished_at DESC LIMIT 5;

-- 2. When did each job last succeed? Staleness is invisible otherwise.
SELECT kind, max(finished_at) FROM recommendations.ingest_runs
 WHERE status = 'completed' GROUP BY kind;

-- 3. Mixed embedding provenance. Should return exactly one row.
SELECT embed_model, embed_task_type, count(*) FROM recommendations.items
 WHERE is_eligible AND embedding IS NOT NULL GROUP BY 1, 2;

-- 4. Catalog health at a glance
SELECT count(*) AS total,
       count(*) FILTER (WHERE is_eligible AND embedding IS NOT NULL) AS servable,
       count(*) FILTER (WHERE NOT is_eligible) AS retired_or_unreliable,
       round(avg(confidence)::numeric, 3) AS avg_confidence
  FROM recommendations.items;
```

---

## What is safe to re-run

| Action | Safe? | Why |
|---|---|---|
| `ingest.platform` | **Yes, always** | Idempotent. Unchanged `embed_input_sha` is skipped; the cursor advances only on success. |
| `ingest.platform --full` | **Yes** | Same, just ignores the cursor. Costs tokens only for genuinely changed rows. |
| `story-data sync-recs` | **Yes** | Transactional full rebuild. Excludes `synth_%`. |
| `seed.py --refresh-only` | **Yes** | Pure derivation from `interactions`. Leaves `item_stats` alone if `interactions` is empty. |
| `seed.py --purge` | Yes, but destructive | Deletes every `synth_*` reader and their taste vectors. Never touches real readers. |
| **`--rebuild-index`** | **No — not against live traffic** | Drops the HNSW graph before recreating it. Every query is a sequential scan for the duration. |

There is no partial-failure state needing cleanup before a retry.

---

## Known issues before deploy

Recorded, not yet fixed. Each is a real failure mode with a known trigger.

1. **`recs_service` cannot run the ingest.** Migration `000020` grants no `SELECT` on
   `public`, but `ingest/platform.py` reads `stories`, `story_tags`, `chapters` and
   `chapter_summaries`, and `_retire_unpublished` does `UPDATE … FROM stories`. The
   first production ingest run fails with `permission denied for table stories`.
   Invisible today only because the role does not exist yet. **Deploy blocker** —
   see [security-and-roles.md](security-and-roles.md) for the two resolutions.

2. **The ingest cursor strands failed rows.** A story stored `is_eligible = false`
   after a low-confidence normalization is never revisited, because the cursor
   advanced past its `updated_at`. The code comment says a later run "can fix it";
   with an incremental cursor it cannot. Only `--full` recovers it.

3. **`catalog_embed_model` reports the majority model.** A minority of rows from a
   different provider leaves health green with an unusable slice. Use query 3 above
   rather than trusting the field.

4. **`--limit` plus a timestamp-only cursor can skip rows permanently.** The read is
   ordered `(updated_at, id)` but the cursor stores only `updated_at`, and the next
   read is strictly `>`. A bulk publish in one transaction gives every row an
   identical `now()`; if a `--limit` cuts between two of them, the second is never
   seen again.

5. **`verify_internal_token` silently no-ops when `RECS_SERVICE_URL` is unset.** A
   production deploy that omits it is completely unauthenticated at the application
   layer. Restrict Cloud Run invoker IAM regardless.

6. **CI cannot build its own test schema.** recs no longer owns migrations, so
   `pr-check` must check out story-data and run goose against the service container —
   otherwise every integration test silently skips and the deploy rides on unit
   tests alone.

7. **No deploy artifacts exist.** No `Dockerfile`, `.dockerignore`, `terraform/` or
   `.github/workflows/` — the only backend repo without them.

8. **No green-path health test.** Two integration tests covering `/health` were
   removed because the shared dev database makes them unrunnable (the catalog's real
   Gemini vectors outvote any fixture, and tests run with the mock embedder). A bug
   that makes health permanently red would now ship unnoticed. Fixing this needs a
   dedicated test database, not a test change. See [testing.md](testing.md).

---

## Related

- [jobs.md](jobs.md) — what each background job does and what breaks without it
- [security-and-roles.md](security-and-roles.md) — the privacy boundary and the grant
- [frontend-integration.md](frontend-integration.md) — the caller, for "the UI shows
  the wrong thing" reports
- [development-log.md](development-log.md) — every bug found while building this, and
  how each was caught. The best single read for *how* silent failures surface.
