# API reference — with a trace of what happens inside

Five endpoints. Every request/response below was captured from a live local server
against the 1,987-book catalog, not written by hand.

**Envelope.** Every JSON response is `{success, data, error}`, matching the story
agent's shape so the Firebase Functions bridge can treat both services identically.
Errors carry a stable `code`.

**Auth.** All routes depend on `verify_internal_token`, which validates a Google OIDC
token and checks the caller's email against an allowlist. It is a **no-op when
`ENVIRONMENT != production`**, which is what lets local development run with no
credentials at all.

**Rate limiting.** Two buckets per user, and the split is deliberate — ranking is one
database round trip; HyDE and explanations spend Gemini tokens on a path that bypasses
creditProxy:

| Bucket | Default | Applies to | Error code |
|---|---|---|---|
| `MAX_REQUESTS_PER_MINUTE_PER_USER` | 30 | every route | `RATE_LIMITED` |
| `MAX_LLM_REQUESTS_PER_MINUTE_PER_USER` | 6 | HyDE + both explain routes | `LLM_RATE_LIMITED` |

---

## `GET /health`

The only route without auth. Returns **503** when anything that would silently degrade
retrieval is wrong.

```bash
curl localhost:8100/health
```

```json
{
  "status": "ok",
  "environment": "development",
  "embedder": "GoogleAIEmbeddingProvider",
  "embedding_dimension": 768,
  "embedding_dimension_ok": true,
  "query_cache": {"size": 3, "capacity": 2000, "hits": 1, "misses": 3, "hit_rate": 0.25},
  "database": {
    "connected": true,
    "pgvector_version": "0.8.6",
    "pgvector_ok": true,
    "hnsw_index_present": true,
    "schema_version": 1,
    "item_count": 2000,
    "eligible_count": 1987
  }
}
```

**Why 503 rather than a warning.** A missing HNSW index, pgvector below 0.8.0, or an
embedder at the wrong dimension each produce *empty or worse* results with no error
anywhere. Failing the health check makes a broken deploy visible to Cloud Run's startup
probe instead of to a reader.

### Backend trace

1. Read `app.state.embedder.dimension`, compare to `EXPECTED_EMBEDDING_DIM` (768).
2. `Database.health()` — one connection, four queries: `pg_extension` for the pgvector
   version, `pg_indexes` for `items_embedding_hnsw`, `schema_migrations` for the applied
   version, and a count of eligible items.
3. If any check fails → status `degraded`, HTTP 503.

---

## `POST /recommend/adhoc`

"I liked these, find me more" — named books, a free-text description, or both.

### Example 1: seeded from a catalog book

```bash
curl -X POST localhost:8100/recommend/adhoc \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"u1","books":[{"title":"Dune"}],"top_k":2}'
```

```json
{
  "success": true,
  "data": {
    "mode": "adhoc",
    "resolved_books": [{"id": 308, "title": "Dune"}],
    "unresolved_books": [],
    "hyde_used": false,
    "hypothetical_document": null,
    "query_fingerprint": "48a1706eca5ee614",
    "items": [
      {
        "id": 3205,
        "source": "cmu",
        "source_id": "867886",
        "title": "Dune: House Harkonnen",
        "author": "Kevin J. Anderson",
        "genres": ["science-fiction"],
        "themes": ["political intrigue", "corporate greed", "family secrets",
                   "betrayal", "revenge"],
        "tone": ["epic", "gritty", "tense"],
        "core_premise": "Amidst political intrigue and the struggle for control of
                         the spice melange, House Harkonnen consolidates power...",
        "published_year": 2000,
        "score": 0.885314,
        "off_platform": true,
        "matched_query_count": 1,
        "explanation_cache_key": "40d8fa1de64df816408dad3d4ed88310f37c0735fb0154cc..."
      }
    ],
    "degraded": false,
    "diversity": 0.1379,
    "candidates_considered": 10
  }
}
```

#### Backend trace

1. **Rate limit.** Ranking bucket only — no prompt, so no HyDE, so no LLM bucket.
2. **Resolve "Dune".** `retriever.resolve_titles()` runs a trigram query:
   ```sql
   WHERE i.title % $1 OR author % $2
   ORDER BY GREATEST(similarity(title,$1), similarity(author,$2)) DESC
   ```
   Trigram, not equality, because readers misremember titles and skip subtitles. Match
   → `id 308`.
3. **Take its stored vector.** No embedding call — the book is already in the catalog,
   so its `RETRIEVAL_DOCUMENT` embedding is reused as the query vector. `id 308` is
   added to `exclude_ids`: never recommend the book the reader just named.
4. **Retrieve.** `knn_many([vector], limit = 2 × 5 = 10)` inside a transaction with
   `SET LOCAL hnsw.ef_search = 100`, `iterative_scan = relaxed_order`,
   `max_scan_tuples = 20000`, `statement_timeout = 2000ms`.
5. **`sem` = raw cosine.** One seed, so no RRF. (Fusing a single list would replace
   every similarity with a rank artifact — see
   [development-log 1.1](development-log.md#11-rrf-applied-to-single-seed-queries-the-worst-one).)
6. **CF short-circuits.** `w_cf_ceiling = 0`, so `_cf_scores` returns zeros without
   touching the database.
7. **Blend.** These neighbours have `n_interactions < 5`, so `α = 0`, `w_sem = 1.0`,
   and `score == sem`. `src = 1.0` because there are no platform items yet.
8. **MMR** over the 10 candidates, λ = 0.7, selecting 2. Reports
   `diversity = 0.1379` — low, i.e. the two results are similar. Honest signal.
9. **Cache keys.** For each item,
   `sha256(model | prompt_ver | source:source_id | embed_input_sha | query_fingerprint)`.

### Example 2: free text, via HyDE

```bash
curl -X POST localhost:8100/recommend/adhoc \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"u1","prompt":"a lonely lighthouse keeper slowly loses his grip on reality","top_k":4}'
```

```json
{
  "hyde_used": true,
  "hypothetical_document": {
    "title": "The Salt-Stained Mirror",
    "core_premise": "Isolated on a remote island, a solitary lighthouse keeper's
                     meticulous routine unravels as the relentless sea and his own
                     solitude begin to warp his perception...",
    "themes": ["isolation", "madness and paranoia", "survival against nature",
               "unreliable narrator", "existential dread"],
    "tone": ["bleak", "suspenseful", "unsettling", "melancholy"]
  },
  "items": [
    {"title": "Pincher Martin",        "score": 0.7384, "tone": ["bleak","unsettling","contemplative"]},
    {"title": "The Whalestoe Letters", "score": 0.7287, "tone": ["unsettling","melancholy","feverish"]},
    {"title": "The Invisible Man",     "score": 0.7233, "tone": ["suspenseful","gritty","unsettling"]},
    {"title": "Blackwood Farm",        "score": 0.7120, "tone": ["melancholy","suspenseful"]}
  ]
}
```

**Pincher Martin** is Golding's novel about a man alone on a rock losing his mind;
**The Whalestoe Letters** is letters from an asylum. Neither shares vocabulary with the
query.

#### Backend trace — the differences

2'. **Both rate buckets** are consumed, because HyDE will call the LLM.
3'. **HyDE.** `hyde.generate()` asks Gemini for a hypothetical catalog entry using a
    JSON response schema, temperature 0.4. Output is rendered through the *same*
    `compose_embed_input` template as real catalog rows, and the themes/tone are
    filtered against the same controlled vocabulary. A hypothetical document in a
    different shape would land elsewhere in the space.
4'. **Embed the hypothetical**, not the prompt, with `RETRIEVAL_QUERY`. Cached in an
    in-process LRU keyed on `sha256(normalized_text | task_type)`.

If HyDE fails, `generate()` returns `None` and the raw prompt is embedded instead —
degraded, not broken.

### Example 3: multiple seeds (RRF)

```bash
curl -X POST localhost:8100/recommend/adhoc \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"u1","books":[{"title":"Dune"},{"title":"Neuromancer"}]}'
```

Two independent KNN searches run concurrently, then merge by rank. `matched_query_count`
on each item tells you how many seeds it matched — which is what lets an explanation say
*which* of the reader's books a recommendation came from.

**Not** an averaged vector: the mean of two distant embeddings points at a region
resembling neither.

### Errors

| Condition | Status | Code |
|---|---|---|
| Neither `prompt` nor `books` | 400 | `INVALID_REQUEST` |
| `top_k` outside 1–50 | 422 | `VALIDATION_ERROR` |
| Free text with no embedder configured | 503 | `EMBEDDER_UNAVAILABLE` |
| Over the ranking bucket | 429 | `RATE_LIMITED` |

A named book that isn't in the catalog is **not** an error — it comes back in
`unresolved_books` and the rest of the request proceeds.

---

## `POST /recommend/behavioral`

Recommendations from a reader's own history.

```bash
curl -X POST localhost:8100/recommend/behavioral \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"synth_00000","top_k":3}'
```

```json
{
  "success": true,
  "data": {
    "mode": "behavioral",
    "n_signals": 23,
    "items": [
      {"title": "Alphabet of Thorn", "score": 0.8891, "off_platform": true},
      {"title": "A Separate Peace",  "score": 0.8734, "off_platform": true},
      {"title": "The Wanting Seed",  "score": 0.8702, "off_platform": true}
    ],
    "degraded": false,
    "diversity": 0.2104,
    "candidates_considered": 15
  }
}
```

A reader with no history:

```json
{"mode": "popular", "n_signals": 0, "items": [...]}
```

**`mode` is load-bearing.** It reports honestly whether this was personalized. Until
the Firestore signals export exists, real readers all get `popular`, and the API says
so rather than implying personalization it can't deliver.

### Backend trace

1. **Rate limit** (ranking bucket only — no LLM on this path).
2. **`SELECT ... FROM user_taste WHERE user_id = $1`.**
   - Found with `n_signals ≥ 3` → use `taste_embedding` as the query vector; add
     `seed_item_ids` (already read) and `suppressed_item_ids` (rated ≤2) to
     `exclude_ids`. `mode = "behavioral"`.
   - Otherwise → no query vector. `mode = "popular"`.
3. **`pipeline.rank`.** With no vector it calls `retriever.popular()`, which orders by
   `pop_score DESC, updated_at DESC` and sets `sem = 0` for every row.
4. Same blend → MMR → response.

**There is no embedding call anywhere on this path.** The taste vector — an
engagement-weighted, L2-normalized mean of the embeddings of books the reader engaged
with — was computed at sync time. That is the entire reason this is the fast mode.

---

## `POST /recommend/explain`

All explanations in one response. **Stage A** — the mode that works through a Firebase
Functions proxy, which buffers responses and therefore cannot stream.

```bash
curl -X POST localhost:8100/recommend/explain \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"u1","item_ids":[34,2096,3460],"prompt":"gothic horror like Dracula"}'
```

```json
{
  "success": true,
  "data": {
    "query_fingerprint": "9c2f...",
    "explanations": [
      {"item_id": 34,   "explanation": "This novel offers a chilling, suspenseful exploration of forbidden love and vampiric menace reminiscent of Dracula.", "cached": false},
      {"item_id": 2096, "explanation": "This unsettling tale of a respectable doctor's dark duality and his monstrous alter ego offers a chilling exploration of good versus evil.", "cached": false},
      {"item_id": 3460, "explanation": "This novel offers gothic horror with a haunted castle and a tale of forbidden love, reminiscent of Dracula's suspenseful atmosphere.", "cached": false}
    ]
  }
}
```

Run the identical request again:

```json
{"explanations": [{"cached": true}, {"cached": true}, {"cached": true}]}
```

Zero tokens spent.

### Backend trace

1. **Rate limit — both buckets.** This always spends tokens on a cache miss.
2. **Load targets.** One query by id, fetching only the *normalized* fields —
   title, author, genres, themes, tone, `core_premise`, `embed_input_sha`. Results are
   re-ordered to the caller's `item_ids` order so explanations arrive in shelf order.
   Unknown ids are silently dropped.
3. **Fingerprint the request.** `query_fingerprint(prompt, seed_item_ids)` — seed ids
   are **sorted**, so the same set in a different order is the same query and doesn't
   miss the cache.
4. **Bulk cache lookup**, one query for all keys. Hits increment `hit_count`
   fire-and-forget, because counting must never delay a response.
5. **Generate misses**, up to 4 concurrently. Prompt asks for one sentence ≤30 words,
   explicitly forbidding plot summary and spoilers. Only the normalized fields are
   supplied — **the model cannot spoil what it was never told.**
6. **Write successes to cache**, then return.

A single failure yields `explanation: null` for that item rather than failing the batch:
a shelf with nine reasons and one blank beats an error page.

### The cache key

```
sha256(model | PROMPT_VERSION | source:source_id | embed_input_sha | query_fingerprint)
```

Each component prevents a specific staleness:

| Component | Prevents |
|---|---|
| `model` | serving text written by a different model |
| `PROMPT_VERSION` | serving text written under different instructions |
| `source:source_id` | mixing books up |
| `embed_input_sha` | **self-invalidating** — if a book's premise or themes are re-derived, its hash changes and the stale explanation is abandoned automatically |
| `query_fingerprint` | reusing "why you'll like this" across unrelated requests |

That fourth one is the elegant part: there is no cache-busting logic to remember,
because the key is derived from the content it describes.

---

## `GET /recommend/explain/stream`

One multiplexed SSE stream. **Stage B** — real token-by-token delivery, cancellable.

```bash
curl -N 'localhost:8100/recommend/explain/stream?user_id=u2&item_ids=223,2283&prompt=galactic%20empire%20science%20fiction'
```

```
event: explanation
data: {"item_id": 223, "delta": "This novel"}

event: explanation
data: {"item_id": 223, "delta": " offers galactic empire science fiction through its suspenseful pursuit of a mathematician whose theories ignite"}

event: explanation
data: {"item_id": 223, "delta": " political intrigue."}

event: item_done
data: {"item_id": 223, "cached": false}

event: explanation
data: {"item_id": 2283, "delta": "This suspense"}

...

event: done
data: {"completed": 2}
```

| Event | Meaning |
|---|---|
| `explanation` | a text fragment, tagged with its `item_id` |
| `item_done` | that item is finished; `cached` says whether it cost tokens |
| `item_error` | that item failed; the stream continues |
| `error` | fatal (e.g. no LLM configured) |
| `done` | stream complete, with a count |

### Design decisions

**GET, not POST.** SSE is a GET-shaped protocol in browsers. Item ids are a
comma-separated query parameter.

**Multiplexed.** One connection serves all ten cards, with events tagged by `item_id` —
not ten parallel connections.

**Sequential, not concurrent.** Items are streamed one after another. A reader reads
top to bottom, so finishing the *first* explanation quickly matters more than finishing
all of them slightly sooner. It also keeps only one LLM call open at a time and keeps
event ordering meaningful.

**Cancellation is the point.** `request.is_disconnected()` is polled between items *and*
between chunks. A reader who scrolls away stops paying for tokens mid-sentence.
**Partial text is never cached** — a truncated explanation must not become the
permanent answer.

**`X-Accel-Buffering: no`.** Without it an intermediary proxy may buffer the whole
response and defeat streaming entirely.

### Errors

| Condition | Status |
|---|---|
| `item_ids` not comma-separated integers | 400 |
| `item_ids` empty | 400 |
| Over the LLM bucket | 429 |

---

## Querying without HTTP

`query_cli.py` calls the same `pipeline.rank()` the endpoints use, so the two cannot
rank differently. Useful for debugging ranking with no API in the way.

```bash
python -m recommendation_engine.query_cli --title "Dune" --scores
```

```
 1. 0.8853  Dune: House Harkonnen — Kevin J. Anderson (2000) [off-platform]
      genres: science-fiction
      sem=0.8853*1.00  pop=0.4395*0.00  cf=0.0000*0.00  alpha=0.00  src=1.00
```

That last line is the full score attribution: each term, its weight, the ramp, and the
source multiplier. `pop=0.4395*0.00` means the item *has* a popularity score but it
contributes nothing, because `α = 0` — fewer than 5 interactions. Exactly the cold-start
behaviour the formula specifies.

**Results are not sorted by score.** MMR reorders for diversity. If the list were
strictly descending, MMR would be doing nothing.
