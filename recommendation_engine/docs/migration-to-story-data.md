# Migration plan — fold `recommendations` into story-data's database, drop CMU

Status: written 2026-08-28. **All seven phases done.** Remaining work is
listed under "What this migration did not do" at the end.

Two changes that are cheaper together than apart:

1. `recommendations.*` stops living in its own Postgres (`:5434`) and becomes a
   schema inside story-data's database. story-data owns the migrations. recs
   stays its own repo, own service, own connection.
2. The CMU bootstrap corpus is deleted outright. The catalog is TaleTribe
   stories only.

No production data exists, so every step below is a rewrite rather than a data
migration. There is nothing to backfill, dual-write, or cut over.

---

## End state

| Thing | Before | After |
|---|---|---|
| Database | `recs` @ `:5434` (own compose) | `story_data` @ `:5433`, schema `recommendations` |
| Migrations | `recommendation_engine/migrations/migrate.py` | story-data goose, `migrations/000019_*.sql` |
| Item identity | `(source enum, source_id text)` | `story_id UUID REFERENCES stories(id)` |
| Catalog sources | `cmu` + `platform` | platform only |
| Signals | JSONL export → `sync/interactions.py` loader | SQL, in-database |
| Private data access | n/a | recs never reads `reading_progress`; story-data writes the derived rows |

Already compatible, so not work: both databases are `pgvector/pgvector:pg16`,
both use `vector(768)` with cosine HNSW, story-data already runs
`CREATE EXTENSION vector` (`migrations/000003_ai_context.sql`) and `pg_trgm`
(`000012`, `000018`). recs is fully schema-qualified (`recommendations.*`, no
`search_path` dependency) and already has a read/write pool split with a
separate read DSN (`recs_database_url_ro`, `config.py:44`).

---

## Phase 0 — decisions being locked in

These are recommendations, taken as decided unless overridden. Each is
reversible now and expensive later.

**0.1 Collapse `(source, source_id)` to a real FK.** With one source, the enum
and the string key buy nothing and cost a join-by-text plus a class of orphan
rows. `items.story_id UUID NOT NULL UNIQUE REFERENCES stories(id) ON DELETE
CASCADE` gives referential integrity and makes an unpublished-or-deleted story
disappear from the catalog by construction rather than by sync job.

This is the widest-blast-radius change in the plan — it touches `retrieval.py`,
`pipeline.py`, `routes.py`, `explain.py`'s cache key, `query_cli.py`,
`sync/*`, and most integration tests. Do it in Phase 1, not later.

**0.2 recs gets its own database role.** `GRANT USAGE ON SCHEMA
recommendations` and nothing on `public`. See Phase 5 for why this matters more
than it looks.

**0.3 Catalog is pulled; signals are pushed.** Published stories are public data
(`ListPublicStories` already serves them to anyone), so recs polling `stories`
is not a privacy boundary crossing. Reader signals are not, so they arrive via
story-data-owned SQL. This split is the whole point of the co-location design.

**0.4 Rewrite git history to drop the corpus.** `booksummaries.txt` is 43 MB and
tracked. Removing the file in a normal commit leaves it in every clone forever;
only a history rewrite reclaims it. Pre-production is the only cheap moment to
do that, and it has to be run by hand — this plan does not script it.

---

## Phase 1 — schema moves to story-data

**New file: `story-data/migrations/000019_recommendations_schema.sql`** (goose
Up/Down, immutable, applied at startup under the existing advisory lock —
`cmd/api/main.go:171-182`).

Restate `recommendation_engine/migrations/001_init.sql` with these deltas:

- `CREATE SCHEMA recommendations;`
- Drop `CREATE TYPE recommendations.item_source` entirely.
- `items`: replace `source` / `source_id` / `UNIQUE (source, source_id)` with
  `story_id UUID NOT NULL UNIQUE REFERENCES stories(id) ON DELETE CASCADE`.
- `items`: drop `raw_genres` (it existed to audit the CMU crosswalk). `genres`
  now comes from `stories.category` + `story_tags`, which are already
  controlled values.
- `interactions.user_id` / `user_taste.user_id` stay `text` — Firebase uids,
  matching `story_likes.user_id` etc. No FK; story-data has no users table.
- `ingest_runs.kind` loses `cmu_backfill`; keep `platform_sync`,
  `stats_refresh`, `cooc_rebuild`.
- `config`: drop the `cmu_target_catalog` and `cmu_weight_floor` rows. Keep
  everything else — DB-resident knobs are still the right call.
- Keep `normalization_cache`. Platform stories still get LLM-normalized; only
  the corpus changed.
- Keep the `items_title_trgm` / `items_author_trgm` indexes — the ad-hoc
  "I liked these" path still resolves titles fuzzily, now against platform rows.
- `CREATE EXTENSION` lines for `vector` and `pg_trgm` are already satisfied by
  earlier migrations; harmless to restate with `IF NOT EXISTS`.
- Down: `DROP SCHEMA recommendations CASCADE;`

**Note on the pgvector floor.** recs requires >= 0.8.0 for
`hnsw.iterative_scan`; story-data neither requires nor checks it. The shared
local image already satisfies it. Confirm the Neon project does before Phase 7,
and leave the assertion in recs' `/health` as the thing that catches a
regression.

**Deliverable:** start story-data's stack, let it migrate, then `\dn` shows
`recommendations` and `\dt recommendations.*` shows nine tables.

**Done.** Two deliberate departures from the above: `CREATE EXTENSION` is not
restated (000003 and 000018 already guarantee both, and `DROP SCHEMA` does not
remove them), and the recs original's `IF NOT EXISTS` idempotency is dropped
throughout — goose tracks versions, so re-runnability is not this runner's job.
CHECK constraints were added on `ingest_runs.kind` / `.status` and
`interactions.kind`, which is how story-data's other migrations express a closed
value set, and is what makes "loses `cmu_backfill`" a constraint rather than a
comment. `published_year` is retained: `retrieval.py`'s `published_after` filter
reads it.

**recs does not run against this schema yet, by design.** `retrieval.py` still
selects `i.source` and `i.source_id`, which no longer exist. Phases 2–4 are what
make the service work again; until they land, recs runs only against its old
`:5434` database.

---

## Phase 2 — recs stops owning schema

- Delete `recommendation_engine/migrations/` (both `migrate.py` and
  `001_init.sql`). Two auto-running migration runners against one database is
  the actual hazard co-location introduces; the fix is that only one exists.
- Delete `recommendation_engine/docker-compose.yml`. recs now depends on
  story-data's stack being up.
- `config.py`: `LOCAL_DEV_DSN` becomes
  `postgres://postgres:postgres@localhost:5433/story_data?sslmode=disable`.
  Keep the existing production guard (`config.py:177`) that rejects the local
  default when `ENVIRONMENT=production`; only the string changes.
- `server.py` `/health`: keep the pgvector-version, HNSW-presence and
  embedder-dimension assertions. **Add** a check that `recommendations.items`
  exists, so a recs instance started against a story-data that has not migrated
  fails loudly instead of 500-ing per request.
- `CLAUDE.md` (recs): remove the migrate commands and the `:5434` row from the
  ports table.

**Done.** Two things the plan understated:

*The index name had drifted.* `db.HNSW_INDEX_NAME` was `items_embedding_hnsw`;
migration 000019 names it `items_embedding_hnsw_idx` to match story-data's
convention. Left alone, `/health` would have reported a missing HNSW index
against a perfectly good schema — the exact silent degradation that check
exists to prevent. `schema_version` in `db.health()` is replaced by
`schema_present`, since `recommendations.schema_migrations` no longer exists
and goose's `goose_db_version` is not this service's business.

*Five test files depended on the migration runner*, not zero. `tests/conftest.py`
is new (the repo had none) and holds `require_recommendations_schema`, which
skips rather than fails when the schema is absent — an unmigrated database is a
setup problem, not a fault in the code under test.
`tests/test_rec_migrations.py` is deleted along with three tests inside
`test_rec_integration.py` that covered the runner's idempotency, version
recording, and checksum enforcement. Those behaviours are goose's now.

Verified against a throwaway story-data database migrated to 000019: `/health`
is green, `schema_present` and `hnsw_index_present` both true, the goose-seeded
config knobs load, and `SET LOCAL` HNSW tuning works. Suite with no test DSN:
434 passed, 73 skipped.

---

## Phase 3 — excise CMU

**Delete:**

```
recommendation_engine/booksummaries/           (README + 43 MB corpus)
recommendation_engine/ingest/cmu_parse.py
recommendation_engine/ingest/genre_crosswalk.py
recommendation_engine/ingest/genre_crosswalk.csv
tests/test_rec_cmu_parse.py
```

**Edit:**

| File | Change |
|---|---|
| `scoring.py` | Delete `CMU_SOURCE`, `source_weight()` (~line 273-297), the two `cmu_*` fields on the config dataclass, their `DEFAULTS` entries, and the `__all__` export. |
| `retrieval.py` | Drop the `sources` filter field and its two SQL predicates; drop `platform_item_count` from the stats query (~line 260-267) — it existed only to drive the CMU down-weight. |
| `ingest/backfill.py` | Rewritten in Phase 4. `--corpus`, `--skip-normalization` semantics and `rebuild_hnsw_index` need re-examination, not deletion. |
| `ingest/compose.py` | Docstring only; `NormalizedItem` is already source-agnostic. |
| `ingest/normalize_llm.py` | Docstring only; `normalize_many` already takes `(id, title, author, summary)` tuples. |
| `sync/interactions.py` | Largely deleted in Phase 5. |
| `query_cli.py` | Drop source filtering/labels. |
| `pyproject.toml` | Drop any corpus-only dependency. |
| Docs | 10 files under `docs/` plus `CLAUDE.md` and `README.md` mention CMU. |

**Keep:** `ingest/vocabularies.py` — the controlled theme/tone vocabulary is
about the embedding schema, not the corpus.

**Done, and wider than the table above.** Two corrections:

*`backfill.py` was deleted, not stripped.* The plan said "rewritten in phase 4,
not deletion." In the event every function in it referenced `CmuRecord`, the
crosswalk, or the dropped `source`/`source_id` columns — there was no subset
that both compiled and survived decision 0.1. Phase 4 should lift from git
history: `rebuild_hnsw_index` (**and rename the index to
`items_embedding_hnsw_idx`**), the `ingest_runs` start/complete/fail
bookkeeping, chunked commits for resumability, and the `embed_input_sha` skip.

*The `source_id` → `story_id` collapse landed here, not in phase 1.* Decision
0.1 said "do it in phase 1"; phase 1 was schema-only, so the code half had
nowhere to go until the CMU excision forced it. It touched `retrieval.py`
(including renumbering `$7`–`$10` in two queries after the `sources` filter came
out), `pipeline.py`, `routes.py`, `explain.py` (the cache key is now
`model|prompt_ver|story_id|sha|fingerprint`, matching migration 000019's
comment), `sync/synthetic.py` and `sync/interactions.py`.

Also removed: the `off_platform` flag and its API field. It existed solely to
badge CMU items as unreadable; every item is now a readable story.

Suite: 357 passed, 71 skipped. Docs updated across nine files;
`development-log.md` deliberately left alone, being a historical record.

**Cold-start consequence — state it, don't paper over it.** CMU was the answer
to "what do we recommend before the platform has a catalog." Deleting it does
not delete the problem. Post-migration, a reader with no signals against a
catalog of a few dozen stories gets popularity-ranked results over a small pool,
and MMR diversification has little to work with. That is acceptable for a
platform still being built, but it means recommendation quality cannot be
meaningfully evaluated until the catalog grows — which also lowers the value of
the not-yet-built eval harness until then.

---

## Phase 4 — platform ingest (the real new work)

Replaces the CMU backfill. Everything downstream of `NormalizedItem` is
unchanged, so this is a new front end onto an existing pipeline.

**Source query** — same database, no HTTP:

```sql
SELECT s.id, s.title, s.author_name, s.description, s.category,
       s.target_audience, s.language, s.updated_at,
       array_agg(st.tag) FILTER (WHERE st.tag IS NOT NULL) AS tags,
       count(c.id) AS chapter_count, coalesce(sum(c.word_count),0) AS word_count
FROM stories s
LEFT JOIN story_tags st ON st.story_id = s.id
LEFT JOIN chapters  c  ON c.story_id  = s.id
WHERE s.is_published
GROUP BY s.id
```

**What becomes the "summary" fed to `normalize_llm`?** Three options, in
increasing cost and quality:

1. `description` alone. Cheapest; many descriptions are a sentence, which trips
   the existing `confidence < 0.5` gate and marks the row ineligible — the gate
   working correctly, but it may exclude most of the catalog.
2. `description` + tags + category. Recommended starting point.
3. `description` + concatenated `chapter_summaries.summary`. Best signal, but
   that table is populated opportunistically by the summarize path
   (`story-data/internal/store/store.go:328`) and will be sparse. Use it when
   present, fall back to (2).

**Freshness.** Poll on `stories.updated_at` against the last successful
`ingest_runs` row, per decision 0.3. Re-embed only when `embed_input_sha`
changes — that mechanism already exists and does the deduplication for free, so
a poll that re-reads every published story is cheap in LLM and embedding terms.
Do **not** add a `recs_outbox` to story-data's write path yet; recommendation
staleness of minutes is not a product problem, and an outbox is a permanent cost
on every story write.

**Eligibility.** `is_eligible` is set from `is_published` at ingest, and the FK
cascade handles deletion. A story that gets unpublished is caught on the next
poll — acceptable lag for a recommendation surface, and the reader-facing
`ListPublicStories` is unaffected either way.

**HNSW rebuild.** `rebuild_hnsw_index()` drops and recreates the graph with
raised `maintenance_work_mem`. That was sized for a 15.5 k-row bulk load. With
incremental platform ingest it should become a rarely-used maintenance command,
not part of the normal run — and it must never run against the shared database
during traffic. Gate it behind an explicit flag.

**Done** — `recommendation_engine/ingest/platform.py`, plus
`tests/test_rec_platform_ingest.py` (9 unit tests). Three things worth
recording:

*The read SQL in this plan had a fan-out bug.* `LEFT JOIN story_tags` and
`LEFT JOIN chapters` in one query multiply their rows together, so a story with
3 tags and 2 chapters reported triple its word count. `count(DISTINCT c.id)`
hides it for the counts and not at all for `sum(c.word_count)`. The shipped
version uses scalar subqueries; a seeded story with 3 tags and 2 chapters
(1000 + 1500 words) reports 2500, verified.

*Unpublishing needed a reconciliation pass the plan did not account for.* The
read only sees published stories, so an unpublished one simply stops appearing
and would keep its `is_eligible = true` forever. `_retire_unpublished` closes
it with a join against `stories` — trivial only because the schemas share a
database.

*That fix has a trap, and `_existing_shas` guards it.* A story unpublished and
then republished unchanged has an unchanged `embed_input_sha`, so the
skip-unchanged path would skip it and it would stay invisible permanently. The
query therefore ignores rows that are already ineligible. Verified end to end:
retire → `is_eligible = false`, republish → `embedded=1`, not `skipped=1`.

Also verified against a seeded database: unpublished drafts are never read, the
`ingest_runs` cursor advances and makes the next run a no-op, `--full` re-reads
and skips all as unchanged, and the full rank path (trigram title resolution →
KNN → scoring → MMR) returns `story_id` in the payload.

---

## Phase 5 — signals, in SQL

This is the phase that justifies the whole migration: `sync/interactions.py`'s
JSONL contract, `(source, source_id)` resolution, and the entire unbuilt
"Firestore export" step collapse into two queries.

**`recommendations.item_stats`** — periodic `INSERT … SELECT … ON CONFLICT DO
UPDATE` aggregating `story_likes`, `story_ratings` (count + avg),
`reading_progress` (completions), and `stories.views`. The `pop_score`
arithmetic in `sync/stats.py` is unchanged; only its input changes from loaded
rows to a join.

**`recommendations.interactions`** — one row per (user, story, kind) derived
from the same three tables. Note the shape mismatch to resolve:
`reading_progress` has `chapter_id` + `scroll_percent` and **no chapter index**,
while completion derivation needs position and total. Both are derivable in the
same query:

```sql
row_number() over (partition by c.story_id order by c.position) -- chapter_index
count(*)    over (partition by c.story_id)                      -- total_chapters
```

Keep `COMPLETION_SCROLL_THRESHOLD = 0.9` and the last-chapter requirement — that
definition is sound and should not drift during a mechanical port.

**Who runs it, and why it matters.** `reading_progress` is strictly private
per-user data. If recs holds a connection that can `SELECT` it, co-location has
quietly created a second, unaudited path to every reader's history — the exact
thing story-data's `internal/store` authorization layer exists to prevent.

So: **story-data owns these refreshes and writes into `recommendations.*`; recs
reads only its own schema.** Enforce it in the database, not by convention —
decision 0.2's role grant is what makes a future mistake fail rather than work.
Mechanically the refresh can be a Go job in story-data, or SQL functions defined
in migration 000019 and invoked on a schedule; story-data currently has no
background scheduler (`cmd/api/main.go` runs the HTTP server only), so Cloud
Scheduler calling an authenticated endpoint is the smaller addition.

**Delete** the JSONL half of `sync/interactions.py` — `InteractionRecord`,
`from_json`/`to_json`, the file loader. **Keep** `seed_engagement_weight` and
`is_negative_signal` (they live in `scoring.py`) and the `kind` vocabulary.
**Keep** `sync/synthetic.py` and `sync/seed.py`, repointed to write rows
directly; synthetic readers remain the only way to exercise scoring before real
traffic, and Phase 3 just removed the other one.

**Done**, and the boundary landed in a better place than the plan drew it.

*Only `interactions` had to move.* `stats.py` already read `recommendations.interactions`
and nothing else, so `item_stats`, `pop_score`, `user_taste` and the
co-occurrence rebuild all stayed in recs untouched. story-data owns exactly one
thing — `internal/store/recommendations.go`, the only code that reads product
tables on recs's behalf — plus `item_stats.views`, which is a counter on the
story rather than a per-reader signal. The two writers touch disjoint columns,
verified by a refresh that left `views` intact.

*`InteractionRecord` survives; the JSONL does not.* The plan listed the
dataclass for deletion, but `synthetic.py` still needs a shape to generate into
and `load()` still needs one to write. What went is the file format that was
only ever there to carry a Firestore export: `to_json`, `from_json`,
`read_jsonl`, `write_jsonl`, and `seed.py`'s `--load` / `--out`.

*The completion rule is now implemented twice*, in Go SQL and in Python, because
recs cannot compute it without reading `reading_progress`.
`TestRecommendationSignalWeights` and `TestCompletionNeedsLastChapterAndDeepScroll`
in story-data pin both to the Python constants; that duplication is the price of
the privacy boundary and is worth naming rather than hiding.

*A Phase 3 miss surfaced here.* `routes.py` still passed `sources=` to
`RetrievalFilters`, so **every `/recommend/*` call raised a TypeError**. My
Phase 3 audit grepped `\bsource\b`, which does not match `sources`. Only
exercising the endpoint found it — the unit suite was green throughout.

Verified end to end on a seeded database: 9 interactions derived across 2 readers
with correct weights (rating 2 → 0.0, rating 4 and 5 → 1.0, progress 0.30 →
0.12), completion derived only for the reader on the last chapter past 0.9,
`item_stats` aggregated with `views` preserved, a suppression list built from the
low rating, and `/recommend/behavioral` returning **`mode=behavioral`** — which
has never worked before — recommending the one story the reader had not touched.
Suites: recs 360 passed / 71 skipped; story-data all packages green.

---

## Phase 6 — tests

The largest hidden cost, and the reason not to leave tests for last.

- `RECS_TEST_DATABASE_URL` points at a story-data test database. Integration
  tests currently insert `items` rows freely; with the FK they must first create
  a `stories` row. Every integration fixture needs a story factory. Build that
  helper first — roughly a dozen test files depend on it.
- Integration tests write and delete in a schema that now shares a database with
  product tables. Confirm teardown is scoped to `recommendations.*` and cannot
  truncate anything in `public`.
- `tests/test_rec_scoring.py` has the densest CMU coupling (20 references) —
  mostly `source_weight` cases that delete wholesale.
- `tests/test_rec_cmu_parse.py` deletes entirely.
- Keep the self-skip-when-unset behaviour that keeps CI green.

**Done.** `tests/conftest.py` grew `seed_stories` / `drop_stories` / `story_id_for`.
Fixture ids are `uuid5` of a fixed namespace, so the same key always yields the
same UUID — readable in a failure message, and stable across runs, which a
`uuid4` per run would not be (a crash before cleanup would leave orphans that
nothing could find). Cleanup deletes the **story**, never the item, so
`ON DELETE CASCADE` takes the item, its stats, its interactions and its
explanations with it — less code, and a standing check that the cascade is wired
the way migration 000019 claims.

The 60 known-failing tests all pass. Two more Phase 3 leftovers surfaced while
doing it, both invisible to the unit suite:
`test_popular_fallback_applies_filters` still asserted the result of a filter I
had deleted (repointed to `genres`, which the popular path must still honour),
and `test_adhoc_marks_off_platform_items` asserted a field that no longer exists
(now asserts every item carries a `story_id` and no `off_platform`).

**431 passed, 0 failed** against a migrated database.

**Known failing now, and this is the phase that fixes it.** Pointed at a
migrated story-data database, **60 integration tests fail** (measured after
phase 3), every one of them `UndefinedColumnError` on `source` / `source_id` in
a raw `INSERT INTO recommendations.items` inside a test fixture. Phase 3 fixed
every *dataclass* construction, so the remaining failures now have exactly one
cause.

The fix is a story factory: these fixtures cannot insert an item without a
`stories` row to point at, and the identifiers they use (`__sync__x`,
`__ret__…`) have to become real UUIDs. Build it once — `test_rec_sync_`
`integration.py`, `test_rec_retrieval_integration.py`,
`test_rec_routes_integration.py` and `test_rec_integration.py` all need the same
helper — and those 60 come back together.

---

## Phase 7 — surrounding repos

- **`dev-new.sh`** (workspace root): add recs on `:8100` to the integrated
  stack; remove any `:5434` compose invocation. recs now starts after story-data
  has migrated — ordering matters, since recs no longer creates its own schema.
- **Root `CLAUDE.md`**: `taleTribe-recs` is listed under `./repos/` in the table
  but has no section of its own and is absent from the request-flow diagram. Add
  both, including the new database relationship.
- **recs `CLAUDE.md` / `README.md`**: the "split out to keep its Postgres
  workload independent" rationale is now partly reversed; say what is still
  separate (service, scaling, deploy) and what is not (database, migrations).
- **`docs/deployment.md`**: recs targets the same Neon project. Ingest and the
  cooccurrence rebuild should run against a Neon branch or a separate compute
  endpoint, wired through `recs_database_url_ro` where reads allow it, so a
  backfill cannot starve the reader-facing API.
- **Frontend / Functions**: was out of scope here — and has since been built
  separately. `recommendStories` / `explainRecommendations`
  (`functions/src/endpoints/recommendations.ts`) are now the only caller, minting
  the OIDC token in `functions/src/recommendations/transport.ts`. Two properties
  worth preserving: `user_id` comes from the **verified Firebase token**, not the
  request body, and the Zod schema is `.strict()` — together they are what stops
  a browser asking for another reader's behavioral recommendations. The client
  reads `story_id` and links straight to `/story/{id}`, which only works because
  of decision 0.1.

**Done.** `dev-new.sh` starts recs on `:8100` after story-data has migrated
(`SKIP_RECS=1` opts out; it skips with a hint when there is no poetry env). The
root `CLAUDE.md` gained a `taleTribe-recs` section and a corrected
request-flow entry. `docs/deployment.md` was rewritten where it assumed a
separate database and a Firestore signals export: the architecture diagram, the
migration section (recs no longer migrates, and story-data must deploy first),
the three feed jobs and their ordering, the least-privilege role, disaster
recovery, and the checklist.

---

## Sequencing

```
Phase 1 (story-data migration)  ──┬─→ Phase 2 (recs stops migrating)
                                  └─→ Phase 6 (test fixtures: story factory)
Phase 3 (CMU excision)  ─────────────→ Phase 4 (platform ingest)
Phase 1 + 3 ─────────────────────────→ Phase 5 (signals in SQL)
Phase 4 + 5 ─────────────────────────→ Phase 7 (docs, dev-new.sh, deploy)
```

Phases 1 and 3 are independent and can go in parallel. Phase 4 is the only phase
that is net-new engineering rather than deletion or mechanical port; budget
accordingly. Phases 3 and 5 together delete more code than the rest of the plan
adds.

## What this migration did not do

- **Scheduling.** Terraform now creates the two recs Cloud Run Jobs, invokes
  story-data's job between them in a Workflow, and creates a nightly scheduler
  paused for the initial rollout.
- **Deployment.** The container, Terraform root, and GitHub workflows are now
  implemented. Production still has to follow the story-data-first rollout in
  `docs/deployment.md`.
- **Confirming pgvector >= 0.8 on Neon.** The local image satisfies it; the
  production project is unverified. `/health` will catch it, loudly, on first
  deploy.
- **Reclaiming the 43 MB corpus from git history.** `booksummaries.txt` is out of
  the working tree but still in every clone. Needs a history rewrite, run by hand.
- **Cold-start quality.** Removing CMU was the right call for a platform-only
  catalog, but it did not solve the problem CMU was solving. With a few dozen
  stories there is little to rank and little for MMR to diversify.

## Rollback

Per phase, until Phase 7: revert the commit. There is no data to preserve and no
consumer of the recs API — the frontend has never called it. After Phase 7,
rollback also means restoring `dev-new.sh` and the `:5434` compose file, which
is why the tooling and doc changes come last.
