# The three jobs

Everything the recommender serves at request time is produced by three background
jobs. At request time the service reads **only their output** — `items.embedding`,
`item_stats.pop_score`, `user_taste.taste_embedding` — and never touches a product
table. That is the whole shape of the system: the expensive and privileged work
happens offline, and the request path is one vector query.

```
┌──────────────────────────────────────────────────────────────────────────┐
│ 1. ingest.platform          recs repo · Python · COSTS MONEY             │
│    stories, story_tags, chapters, chapter_summaries                      │
│                                    └──→ recommendations.items            │
├──────────────────────────────────────────────────────────────────────────┤
│ 2. story-data sync-recs     story-data repo · Go · free                  │
│    story_likes, story_ratings, reading_progress, stories.views           │
│                                    └──→ recommendations.interactions     │
│                                    └──→ item_stats.views (that column)   │
├──────────────────────────────────────────────────────────────────────────┤
│ 3. seed.py --refresh-only   recs repo · Python · free                    │
│    recommendations.interactions + items.embedding                        │
│                                    └──→ item_stats (pop_score, counts)   │
│                                    └──→ user_taste                       │
│                                    └──→ item_cooccurrence (optional)     │
└──────────────────────────────────────────────────────────────────────────┘
```

Terraform creates an ordered Workflow and a nightly Cloud Scheduler trigger. The
scheduler starts paused for the initial rollout; all three jobs can also be run on
demand. See [deployment.md](deployment.md) for resource names and rollout commands.

---

## Why the order is fixed

Each job consumes the previous one's output, and getting it wrong fails quietly
rather than loudly:

1. **ingest before sync-recs.** `sync-recs` joins
   `recommendations.items ON i.story_id = sl.story_id`. A like on a story that has
   not been ingested yet produces **no row at all** — not an error, just a missing
   signal. Run the ingest first and that like lands.
2. **sync-recs before refresh.** `--refresh-only` aggregates
   `recommendations.interactions`. Running it against a stale table recomputes
   yesterday's popularity perfectly.

Run them out of order and nothing breaks; the numbers are just wrong, and nothing
says so. That is the recurring hazard in this service — see
[development-log.md](development-log.md) Category 1.

---

## Job 1 — `ingest.platform`

```bash
cd repos/taleTribe-recs
python -m recommendation_engine.ingest.platform            # incremental
python -m recommendation_engine.ingest.platform --full     # ignore the cursor
python -m recommendation_engine.ingest.platform --dry-run  # read only, no API calls
```

| | |
|---|---|
| **Reads** | `stories` (published only), `story_tags`, `chapters`, `chapter_summaries` |
| **Writes** | `recommendations.items`, `normalization_cache`, `ingest_runs` |
| **Cost** | ~$0.0002 per *changed* story. A run with nothing changed is free. |
| **Runtime** | Seconds for an incremental run; minutes per few thousand stories on `--full` |
| **Suggested cadence** | Hourly to daily. It is cheap and idempotent. |

### What it actually does

For each published story it assembles the text an LLM will read (`summary_text`):
chapter summaries when they exist, because a description is a marketing blurb and a
summary is what actually happens; otherwise the description; plus tags, capped at
1,200 words. Tags go into the material the model reads rather than into `genres`,
because `genres` is a filter dimension backed by the story's controlled `category`
while tags are free text.

That text goes to Gemini, which extracts `core_premise`, `themes`, `tone` and a
confidence score against a controlled vocabulary (`ingest/vocabularies.py`). The
result is composed into a fixed template (`compose_embed_input`), hashed, and
embedded at 768 dimensions.

**Raw prose is never embedded.** Story text is plot incident — who did what to whom,
in order — and embedding it buries the premise/theme/tone signal the recommender
ranks on under a mass of proper nouns.

### Incremental by default

Each successful run records the newest `stories.updated_at` it saw in
`recommendations.ingest_runs.cursor`; the next run reads only past it. `--full`
ignores the cursor.

The cursor advances **only on success**, and only as far as was actually read — a
partial run must not skip what it never saw. If the job throws, the run row is marked
`failed` and the cursor is untouched, so a retry re-reads the same window.

`--full` is cheap regardless, because a row whose `embed_input_sha` is unchanged is
never re-embedded. Re-reading the entire catalog costs one query and no tokens.

### Three things it handles that are easy to miss

**Unpublishing.** The read only sees published stories, so an unpublished one simply
stops appearing — invisible to the ingest. `_retire_unpublished` closes that gap with
a join against `stories`, flipping `is_eligible = false`. Ineligible rows leave the
*partial* HNSW index entirely rather than being deleted, so re-publishing is an update
rather than a re-ingest, and the row keeps its normalization — the expensive part.

**Deletion.** Needs no handling at all. `items.story_id` is a foreign key with
`ON DELETE CASCADE`, so a deleted story cannot survive as a recommendable row.

**Low-confidence normalization.** A thin description yields a confidently
hallucinated premise, which is a poisoned vector that looks perfectly fine. Those
rows are stored **without** a vector and with `is_eligible = false`, keeping them out
of the index. A one-sentence description lands here by design.

### If it doesn't run

Newly published stories are not recommendable, and unpublished ones keep being
recommended. Both are visible to authors, and neither raises an error.

---

## Job 2 — `story-data sync-recs`

```bash
cd repos/story-data
go run ./cmd/api sync-recs
```

| | |
|---|---|
| **Reads** | `story_likes`, `story_ratings`, `reading_progress`, `chapters`, `stories.views` |
| **Writes** | `recommendations.interactions`; the `views` column of `item_stats` |
| **Cost** | Free — one scan of three small tables |
| **Suggested cadence** | Every 15 minutes to hourly |

### Why this one lives in the other repo

**`reading_progress` is strictly private per-user history.** If recs held a
connection that could `SELECT` it, sharing a database would have quietly created a
second, unaudited path to every reader's history — exactly what story-data's
`internal/store` authorization layer exists to prevent.

So story-data reads it and writes only the derived rows recs is allowed to see
(`internal/store/recommendations.go`), and migration `000020` grants `recs_service`
usage on the `recommendations` schema only. The grant is what makes that a boundary
rather than an intention. See [security-and-roles.md](security-and-roles.md).

### Full rebuild, in one transaction

This job **deletes and reinserts everything** each run rather than working
incrementally. That is correct, not lazy: these signals are current *state*, not an
event log. Un-liking a story deletes the `story_likes` row, so an incremental pass
would have nothing to observe and the stale like would live forever.

The delete and the insert share one transaction, so a reader mid-sync never sees a
catalog with likes removed and not yet restored.

### Signal weights, and the one derived kind

| `kind` | Weight | Source |
|---|---|---|
| `like` | 1.0 | `story_likes` |
| `completion` | 0.8 | **derived** |
| `rating` | 1.0 if ≥4, 0.4 if ≥3, else 0.0 | `story_ratings` |
| `progress` | 0.4 × scroll_percent | `reading_progress` |

**`completion` is not a kind any source emits.** The platform records no completion
event, so it is derived — last chapter by `position` order, **and** ≥90% scrolled.
Both halves are needed: chapter index alone counts someone who opened the final
chapter and stopped.

That rule exists in two languages — `completionScrollThreshold` in Go and
`COMPLETION_SCROLL_THRESHOLD` in `sync/interactions.py` — and the two are pinned
together by `TestCompletionNeedsLastChapterAndDeepScroll` and
`TestRecommendationSignalWeights` in story-data. Change one and the test tells you
about the other.

### The `synth_` exclusion

Both the delete and the insert are scoped `WHERE user_id NOT LIKE 'synth\_%'`, so a
synthetic cohort generated by `sync/seed.py` survives a real sync instead of being
wiped by it, and real derivation never runs over synthetic rows. Firebase uids are
28-character alphanumerics and cannot collide with the prefix.

### The `views` column

`stories.views` is an anonymous global counter, not a per-reader signal, so recs
cannot derive it from `interactions`. story-data upserts it into `item_stats` —
**touching only that column**. Job 3 writes every other column and never `views`, so
the two jobs can overlap without either clobbering the other.

### If it doesn't run

`interactions` goes stale. Popularity and taste vectors reflect whenever it last ran.
With it having *never* run, `n_interactions` is 0 everywhere.

---

## Job 3 — `seed.py --refresh-only`

```bash
cd repos/taleTribe-recs
python -m recommendation_engine.sync.seed --refresh-only
python -m recommendation_engine.sync.seed --refresh-only --cooccurrence
```

| | |
|---|---|
| **Reads** | `recommendations.interactions`, `items.embedding` — nothing else |
| **Writes** | `item_stats`, `user_taste`, optionally `item_cooccurrence` |
| **Cost** | Free — SQL plus numpy |
| **Suggested cadence** | Immediately after every `sync-recs` |

Despite living in a file called `seed.py`, `--refresh-only` touches no synthetic
data. It only recomputes derived tables from whatever is already in `interactions` —
synthetic, real, or both.

### `item_stats` and `pop_score`

Aggregation happens in SQL; the **scoring** happens in Python via
`scoring.popularity`. That split is deliberate: reimplementing the Bayesian damping
and log compression in SQL would create a second definition of the formula, free to
drift from the one the unit tests cover.

`pop_score` blends a volume-damped rating (a Bayesian prior stops a single 5-star
rating from beating a well-liked book with fifty) and log-compressed engagement,
where a completion is worth three likes.

If `interactions` is empty the job **leaves `item_stats` unchanged** rather than
zeroing it — a failed upstream sync should not erase yesterday's aggregates.

### `user_taste`

A reader's taste vector is the **engagement-weighted mean** of the embeddings of the
stories they engaged with positively, L2-normalized. Normalizing matters: cosine
distance ignores magnitude, so an unnormalized mean would give heavy readers longer
vectors for no reason.

Stories rated ≤2 are excluded from the mean **and** recorded in
`suppressed_item_ids`. Averaging in something a reader rejected would aim their
recommendations at exactly what they rejected.

**This is why `/recommend/behavioral` makes zero LLM calls.** The reader is already a
point in the catalog's space; serving them is one vector query. Readers with too few
signals get no vector at all, and the route falls back to `mode: "popular"`.

### `item_cooccurrence` — built but inert

`--cooccurrence` rebuilds the item-item CF matrix (shrunk cosine, `cooc ≥ 3` floor,
one direction stored). It contributes **nothing** to ranking: `w_cf_ceiling` is
`0.00` in `recommendations.config`, by design. The platform has no impression or
click log — only binary likes, immutable ratings, and reading-progress state — so
co-occurrence is far too sparse to beat the popularity prior. Keeping the shape means
lighting it up later is a config flip plus a backfill, not a rescoring rewrite.

### If it doesn't run

`pop_score` and `user_taste` reflect the last run. Behavioral recommendations fall
back to popularity for readers whose vectors were never built.

---

## The consequence of none of them being scheduled

The cold-start ramp keys on `n_interactions`, which comes from job 3. With jobs 2 and
3 never having run against real traffic:

```
n_interactions = 0  →  alpha = 0  →  w_semantic = 1.00, w_pop = 0.00, w_cf = 0.00
```

**Every recommendation on the platform is currently 100% content similarity.** The
popularity and collaborative terms are implemented, tested, and completely inert
regardless of what `recommendations.config` says. That is correct behaviour for an
empty signal table — but it means the popularity half of the ranker has never been
exercised against real data, and its first real run is also its first test.

---

## Operating them together

Order matters, so run them as a chain rather than three independent schedules:

```bash
(cd repos/taleTribe-recs && python -m recommendation_engine.ingest.platform) && \
(cd repos/story-data     && go run ./cmd/api sync-recs)                      && \
(cd repos/taleTribe-recs && python -m recommendation_engine.sync.seed --refresh-only)
```

**All three are idempotent and safe to re-run.** The ingest skips unchanged rows,
`sync-recs` is a transactional full rebuild, and `--refresh-only` is pure derivation.
There is no partial-failure state that needs cleaning up before a retry.

### Watching them

```sql
-- when did each job last succeed?
SELECT kind, status, finished_at, counts
  FROM recommendations.ingest_runs
 ORDER BY started_at DESC LIMIT 10;

-- a failed run is a silently stale catalog
SELECT * FROM recommendations.ingest_runs WHERE status = 'failed';

-- did the signals actually land?
SELECT kind, count(*) FROM recommendations.interactions GROUP BY kind;

-- how many readers have a usable taste vector?
SELECT count(*) FROM recommendations.user_taste;
```

[runbook.md](runbook.md) turns these into symptom-first diagnostics.
