# Frontend integration — how the browser actually reaches this service

**The browser never calls this service.** Two Firebase Functions do, and they are the
trust boundary. This document traces the whole path so a "the UI shows the wrong
thing" report can be attributed to the right layer.

All frontend paths are in `repos/taleTribe-frontend`.

```
src/routes/Story/AllStories.tsx                    the two UI surfaces
  └─ src/hooks/queries/useRecommendationQueries.ts react-query wrappers
      └─ src/cloudFunctions/recommendations.ts     HTTP client + the item contract
          │  POST /recommendStories, /explainRecommendations
          │  Firebase ID token
          ▼
functions/src/endpoints/recommendations.ts         ← THE TRUST BOUNDARY
  └─ functions/src/recommendations/transport.ts    mints the Google OIDC token
          │  Bearer <OIDC>, audience = RECOMMENDATION_SERVICE_URL
          ▼
recommendation_engine/routes.py                    :8100
```

---

## The two UI surfaces

Both live on the "All stories" page. Only one occupies the column at a time.

### 1. The "For you" shelf — behavioral

A single row at the head of the story list. It stands down whenever search or AI
discovery takes over the column, so only one shelf competes for that slot.

```ts
const forYou = useBehavioralRecommendations(
  user?.uid,
  recommendationFilters,
  RECOMMENDATIONS_ENABLED && searchInput.trim() === "" && !discoveryActive,
);
```

Four conditions must all hold for it to render: the feature flag, a logged-in user,
an empty search box, and no active discovery.

**The title changes with the mode**, and this is the visible face of cold start:

| `data.mode` | Eyebrow | Title |
|---|---|---|
| `behavioral` | "Chosen from your reading" | **For you** |
| `popular` | "A good place to begin" | **Popular on TaleTribe** |

A reader with fewer than three signals gets `mode: "popular"`. That is the intended
experience, not a fallback bug — but note that **today every reader sees it**, because
neither signals job is scheduled. See [jobs.md](jobs.md).

`topK` is `BEHAVIORAL_SHELF_SIZE` (6), sized to the six columns the catalog grid
shows at its widest breakpoint. **The count is baked into the react-query key**, so a
widened shelf is never served a narrower cached payload.

The genre chip flows through as a filter: `selectedCategory` becomes
`{ genres: [selectedCategory] }`, mapping onto the story's controlled `category`.

### 2. "AI story discovery" — ad-hoc

A mutation rather than a query, because it is user-initiated and should not be
cached. Triggered by a prompt, and by "more like this" (`onSimilar`), which re-issues
an ad-hoc request seeded with the story the reader clicked.

### 3. Explanations — lazy, per card

`RecommendationCard` requests an explanation for **one item at a time**, on demand,
and guards against duplicates:

```ts
if (explanation || explain.isPending) return;
explain.mutate({ itemIds: [item.id], prompt, seedItemIds }, …);
```

Until one is fetched the card falls back to `core_premise`, then to
`themes[0] · tone[0]`, then to a generic string — so a card is never blank while an
LLM call is in flight or after one fails.

**This is the main cost-shaped decision in the UI.** The API accepts up to 25
`itemIds` per call, but the card requests one at a time, so a reader browsing a
6-item shelf can issue 6 separate Gemini generations. The 6/min per-user LLM bucket
is what bounds it. Batching a whole shelf into one request would be cheaper and is
worth considering before real traffic.

---

## The Function is the trust boundary

`functions/src/endpoints/recommendations.ts`. Two properties make it one, and both
must survive any future edit to the route models.

### `user_id` comes from the verified token, never the body

```ts
const handleRecommendStories = requireAuth(async (request, response, userId) => {
  …
  body = { user_id: userId, top_k: value.topK, filters: … };
```

`userId` is resolved by `requireAuth` from the caller's Firebase ID token. The Zod
schema is `.strict()`, so a `user_id` in the request body is **rejected outright**
rather than silently overriding.

`/recommend/behavioral` returns recommendations derived from a reader's private
history. Without this, any browser could request any reader's shelf by typing their
uid. **recs trusts the `user_id` in its request body precisely because this Function
put it there.**

### The browser never learns the service URL or holds an OIDC token

`transport.ts` mints the identity token server-side with
`GoogleAuth.getIdTokenClient(audience)`, where the audience is
`RECOMMENDATION_SERVICE_URL`. `getRecommendationServiceUrl()` refuses a localhost URL
outside the emulator, so a misconfigured production deploy fails loudly.

In the emulator, `identityToken()` returns `null` and no `Authorization` header is
sent — which matches `verify_internal_token` being a no-op locally.

### Input ceilings — the real caps on spend

Enforced by Zod here, not by recs:

| Field | Cap |
|---|---|
| `topK` | 1–50, default 12 |
| `books` | ≤ 10 seed titles |
| `prompt` | ≤ 2,000 characters |
| `itemIds` (explain) | 1–25 |
| `genres` / `themes` | ≤ 20 entries, each ≤ 80 chars |

Ad-hoc mode additionally requires a prompt **or** at least one book
(`superRefine`), so an empty discovery request is a 400 rather than a wasted
retrieval.

---

## ⚠ The field-name hop

**Check this first when "a filter isn't working."** The browser speaks camelCase; recs
speaks snake_case. The translation is a hand-written function in
`endpoints/recommendations.ts`:

```ts
function upstreamFilters(filters) {
  return {
    genres: filters.genres,
    themes: filters.themes,
    max_word_count: filters.maxWordCount,
    min_word_count: filters.minWordCount,
    author: filters.author,
    published_after: filters.publishedAfter,
  };
}
```

Because recs' `FilterSpec` is a plain Pydantic model, an unmapped field arrives as
`None` and the filter silently does nothing. **Adding a filter means editing four
places**: `FilterSpec` in `routes.py`, `RetrievalFilters` in `retrieval.py`, the Zod
`filtersSchema`, and `upstreamFilters`. Miss the last one and it fails silently.

---

## The response contract

`RecommendationItem` in `src/cloudFunctions/recommendations.ts` is what the UI
depends on. Anything the frontend reads is effectively frozen:

| Field | Used for |
|---|---|
| `story_id` | **The link target** — `/story/${item.story_id}` — and the cover lookup via `useStoryCovers` |
| `id` | The `items.id` surrogate key, sent back in `itemIds` when requesting an explanation |
| `title`, `author`, `core_premise`, `themes`, `tone` | Card display and the premise fallback chain |
| `score`, `breakdown` | Debugging only; not rendered |

Note the two identifiers are **not interchangeable**. `story_id` is the story's UUID
and is what the platform understands; `id` is a surrogate key local to the
`recommendations` schema and is what the explain endpoint keys on.

`RecommendationData` carries the envelope: `mode`, `degraded`, `diversity`,
`candidates_considered`, `n_signals`, `resolved_books` / `unresolved_books`,
`hyde_used`. Most are unrendered today but are exactly what you want in a bug report.

**`unresolved_books` is the one product should care about.** With a platform-only
catalog, a seed title TaleTribe does not host resolves to nothing and lands here. The
UI currently ignores it, so "I liked Dune, find me more" silently becomes an
unseeded query.

---

## Error handling

`transport.ts` throws `RecommendationServiceError` carrying the upstream status and
payload; the Function passes those straight through, and anything else becomes a
**502 `RECOMMENDATIONS_UNAVAILABLE`**. So a 429 from the rate limiter reaches the
browser intact, while an unexpected failure is masked.

`useBehavioralRecommendations` sets `retry: false` — a failed shelf does not
hammer the service — and `RecommendationCollection` accepts a `quietError` prop that
renders nothing at all on failure. **The shelf disappearing silently is the intended
degradation**, which is worth knowing: "the shelf is gone" and "the shelf is empty"
are different diagnoses.

Timeouts: 30s for ranking, 60s for explanations, with the Functions themselves at 60s
and 90s.

---

## Configuration

| Variable | Where | Purpose |
|---|---|---|
| `VITE_RECOMMENDATIONS_ENABLED` | Vite build | `"false"` removes both surfaces. The kill switch. |
| `RECOMMENDATION_SERVICE_URL` | Functions param | Base URL **and** OIDC audience. Defaults to `http://localhost:8100`. |
| `RECS_SERVICE_URL` | recs | Expected audience. **Unset makes `verify_internal_token` a no-op.** |

`RECOMMENDATION_SERVICE_URL` and `RECS_SERVICE_URL` must be the same string in
production, or every request is a 401.

---

## Known gap: SSE streaming is unreachable

`GET /recommend/explain/stream` is implemented, tested, and **not called by
anything.** The Function only uses the synchronous `/recommend/explain`.

The reason is structural: Firebase Functions gen2 buffers responses, so a streaming
endpoint cannot be proxied through one. Reaching it needs the browser to call recs
directly, which needs in-process Firebase token verification instead of the OIDC
bridge — a different trust model, not a config change. `deployment.md` sketches
gating it behind `VITE_ENABLE_REC_STREAMING`.

Until then, explanations arrive whole after a round trip rather than token by token.

---

## Debugging across the boundary

Reproduce against recs directly, bypassing both frontend layers:

```bash
# behavioral, as a specific reader (no OIDC needed locally)
curl -s localhost:8100/recommend/behavioral \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"<firebase-uid>","top_k":6}' | jq

# ad-hoc with the same filter shape the Function would send — note snake_case
curl -s localhost:8100/recommend/adhoc \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"u1","prompt":"a lonely lighthouse keeper","top_k":6,
       "filters":{"genres":["fantasy"],"max_word_count":50000}}' | jq
```

If the direct call is right and the UI is wrong, the fault is in the Function layer —
start with `upstreamFilters`. If the direct call is also wrong, continue in
[runbook.md](runbook.md).
