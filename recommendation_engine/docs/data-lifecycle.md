# Data lifecycle — how a book gets into the catalog, and how reader signals get in

Two independent pipelines write to this service. Neither runs during a request.

```
CATALOG      corpus / stories ──→ parse ──→ normalize ──→ compose ──→ embed ──→ items
SIGNALS      synthetic / Firestore ──→ InteractionRecord ──→ load ──→ interactions
                                                                  ──→ item_stats
                                                                  ──→ user_taste
```

---

## Part 1: getting a book into the catalog

### The five stages

Every story goes through the same four stages.

```
1. READ        published stories from story-data → title, author, description, tags
2. NORMALIZE   LLM → core_premise, themes, tone, confidence
3. COMPOSE     the fields → one block of text → sha256
4. EMBED       that text → 768 numbers, tagged RETRIEVAL_DOCUMENT
```

There used to be a fifth stage, CROSSWALK, mapping the CMU corpus's free-text
genre labels onto platform categories. Stories already carry a `category` and
`tags` from a controlled list, so it has no work to do.

### Stage 1 — Read

`ingest/platform.py`. Reads `stories WHERE is_published` straight out of
story-data's database (same database, no HTTP), using `description +
chapter_summaries + tags` as the text and summing chapter `word_count` for
length. `chapter_summaries` is the better input and is used when the summarize
path has populated it; the description is the fallback.

Chapter and tag aggregates are **scalar subqueries, not joins**. Joining
`story_tags` and `chapters` in one query multiplies their rows together, which
leaves `sum(word_count)` inflated by the number of tags — `count(DISTINCT …)`
hides that for the counts and not at all for the sum.

The short-description problem is real and inherited: many stories carry a
one-sentence description, which is exactly what the `confidence < 0.5` gate in
stage 2 is designed to catch. Expect a meaningful share of the catalog to come
back ineligible rather than confidently wrong.

### Stage 2 — Normalize (the expensive one)

An LLM converts a plot summary into three structured fields:

```json
{"core_premise": "≤60 words: the central situation and conflict, no spoilers",
 "themes": ["3-8, from a controlled vocabulary of 223"],
 "tone": ["2-4, from a controlled vocabulary of 33"],
 "confidence": 0.9}
```

Real output for *Animal Farm*:

> **premise:** "After a wise old boar inspires a revolution, the animals of Manor Farm
> overthrow their human farmer. Two pigs, Napoleon and Snowball, vie for control, with
> Napoleon ultimately seizing power and establishing a brutal dictatorship."
> **themes:** rebellion against tyranny, corruption of power, class struggle, allegory
> of politics, propaganda, loss of innocence · **tone:** satirical, bleak, tragic ·
> **confidence:** 0.9

Five design decisions here, each with a measured consequence:

**Batched 8 books per call.** Not primarily for cost — 15,500 individual calls would
exceed a ~1,000 req/day free tier for over two weeks, whereas ~1,940 batched calls fit
in two days free or minutes paid. It also amortizes the ~970-token vocabulary block
across the batch.

**Blocked batches are retried individually.** Gemini's safety filter blocks some
literary plot summaries — unavoidable in a corpus of crime, war and horror. A block
rejects the *whole prompt*, so one problematic book originally destroyed all 8: measured
at 12 blocked batches costing 96 records, ~84 of them innocent. Retrying the same batch
cannot help, because a content block is **deterministic**. Isolating the offender by
splitting cut losses from **96 → 9**.

**The cache stores RAW model output, filtered on read.** This is the highest-leverage
decision in the pipeline. The vocabulary is meant to be tuned iteratively from the
violation counter; if the cache held *filtered* output, every tuning pass would require
regenerating the whole corpus. Storing raw made a vocabulary improvement that recovered
**1,194 → 27 dropped themes** cost **$0.00** — it re-filtered 1,893 cached extractions
instead of regenerating them.

**Controlled vocabularies, enforced twice.** The list is in the prompt *and* the output
is filtered against an allowlist. Step 2 guarantees the invariant; step 1 keeps step 2
from discarding most of the output. Terms landing in the wrong field get **relocated**
(a tone filed under themes moves to tone) rather than discarded — that alone recovered
85 terms. Genre names arriving as themes are recognised as noise and excluded from the
violation counter, so they don't invite bogus vocabulary additions.

**A `confidence` gate.** A 60-word summary yields a fluent, plausible, entirely invented
premise — a poisoned vector that looks perfectly healthy. Anything below 0.5 is marked
`is_eligible = false` and never embedded.

### Stage 3 — Crosswalk

227 Freebase genre labels map to the 10 platform categories, via a **reviewable CSV**
rather than a dict buried in Python — it encodes ~227 judgement calls, and those are far
easier to argue with in a diff.

Two rules:

**Umbrella labels are dropped, not mapped.** `Fiction` (4,747 records),
`Speculative fiction` (4,314) and `Novel` (2,463) each match a third of the corpus, so
they carry no discriminative signal; keeping them would make a genre filter match
almost everything.

**But they still decide fictionality.** Subject labels are genuinely ambiguous — in this
corpus `Philosophy` co-occurs with a fiction marker 46% of the time, `History` 42%,
`Existentialism` 37%. So `Existentialism → non-fiction` put Camus' *The Plague* in the
non-fiction bucket. The fix uses the dropped umbrellas as a **fictionality signal**: an
explicit fiction marker with no explicit non-fiction marker suppresses `non-fiction`.
Result: false `non-fiction` fell 723 → 522, and 1,963 previously uncategorized records
gained a category.

A test fails on **any** label that is neither mapped nor explicitly dropped, so a corpus
refresh that introduces labels breaks loudly instead of silently miscategorizing.

### Stage 4 — Compose

The normalized fields render into one block of text, and *only this* is embedded:

```
Title: Animal Farm
Author: George Orwell
Genres: fiction, young-adult
Premise: After a wise old boar inspires a revolution...
Themes & tropes: rebellion against tyranny, corruption of power, class struggle
Tone: satirical, bleak, tragic
```

An absent field **drops its whole line**. `Author: ` with nothing after it is not
neutral — it's a token sequence the model finds similar across every authorless book,
which would cluster 14% of the corpus by a missing value.

The block is hashed: `embed_input_sha = sha256("v1|" + text)`. That hash is the
resumability mechanism for the whole pipeline.

### Stage 5 — Embed

`embed_batch(texts, task_type=RETRIEVAL_DOCUMENT)` — up to 100 texts per HTTP call to
`batchEmbedContents`, truncated to 768 dimensions via `outputDimensionality`.

Stored with full provenance: `embedding`, `embed_model` (`google:gemini-embedding-001`),
`embed_task_type`, `embed_input`, `embed_input_sha`, `embedded_at`.

The HNSW index is **rebuilt after** a bulk load (`--rebuild-index`) rather than
maintained across 15k inserts — far faster, and it produces a better-balanced graph.

---

## When exactly does an embedding get generated?

This is the question with the most surprising answer, so here it is precisely.

### The rule

**An embedding is generated when `embed_input_sha` differs from what's stored, or when
the stored vector came from a different embedder.**

```sql
-- the ingest's skip check
WHERE story_id = ANY($1)
  AND embedding IS NOT NULL
  AND embed_model = $3          -- current model
  AND embed_task_type = $4      -- RETRIEVAL_DOCUMENT
```

If a row matches on all four *and* its sha matches, it is skipped entirely. No LLM call,
no embedding call, no cost.

### Adding one new story: the walkthrough

Say a TaleTribe author publishes *"The Glass Orchard"*.

```
1. Sync picks it up          stories/{id} where isPublished == true, updatedAt > watermark
2. Text assembled            title + description + tags + category
3. Normalization             cache MISS (new summary hash) → 1 LLM call
                             → premise, themes, tone, confidence
4. Crosswalk                 category "fantasy" → ["fantasy"]
5. Compose                   → text block → sha256 = "a3f9..."
6. Skip check                no row for (platform, {id}) → not skipped
7. Embed                     1 call, RETRIEVAL_DOCUMENT
8. Upsert                    INSERT ... ON CONFLICT (story_id) DO UPDATE
```

Cost: **one LLM call + one embedding call**, roughly $0.0003. It is immediately
retrievable — the partial HNSW index includes it as soon as `is_eligible = true` and the
vector is non-null.

### What happens when the author *edits* it

```
Author fixes a typo in the description
  → sync sees updatedAt changed, re-reads the story
  → normalization: summary hash CHANGED → 1 LLM call (new premise)
  → compose → NEW sha256
  → skip check: stored sha ≠ new sha → NOT skipped
  → re-embed, upsert
```

But if the edit doesn't change any composed field — say they changed the cover image:

```
  → normalization: cache HIT (summary unchanged) → 0 LLM calls
  → compose → SAME sha256
  → skip check: sha matches, model matches → SKIPPED
  → 0 embedding calls, 0 cost
```

**The sha is computed from the composed text, not the source record.** So edits that
don't affect meaning cost nothing, and edits that do are picked up automatically.

### What forces a re-embed of everything

| Change | Effect | Cost |
|---|---|---|
| Edit the crosswalk CSV | `genres` change → sha changes → affected rows re-embed | embeddings only |
| Edit the compose template (`TEMPLATE_VERSION`) | every sha changes → full re-embed | embeddings only |
| Switch embedding model | `embed_model` mismatch → full re-embed | embeddings only |
| Bump `PROMPT_VERSION` | normalization cache invalidated → full re-derive **and** re-embed | LLM **+** embeddings |
| Add a vocabulary term or alias | re-filtered from cached raw output | **$0** |

That last row is the payoff of caching raw model output. That bottom-to-top ordering is
also a rough cost ranking — a `PROMPT_VERSION` bump is the expensive one (~$2.40 for the
full corpus), which is why the axis-confusion prompt fix is deliberately deferred to
ride along with a future full run.

Real numbers observed: editing the crosswalk re-embedded exactly **40 of 300** rows and
skipped 260, because only those 40 had changed genres.

---

## Part 2: getting reader signals in

### Why this half is separate

Scoring reads two things that don't exist in the catalog: how readers have received each
book (`item_stats`), and what each reader likes (`user_taste`). Both derive from raw
signals, which come from outside.

The pipeline is split so the **source is pluggable**:

```
synthetic generator ──┐
                      ├──→ InteractionRecord ──→ load() ──→ interactions
Firestore export ─────┘         (canonical)     refresh() ──→ item_stats
   (not written yet)                                      ──→ user_taste
```

Connecting Firestore means writing **only** the reader. Everything after the canonical
record is built and tested.

### The canonical record

One JSON object per line:

```json
{"user_id": "synth_00042", "story_id": "6f1c…",
 "kind": "rating", "value": 4.0, "occurred_at": "2026-07-14T09:31:00Z"}
{"user_id": "synth_00042", "story_id": "9ab3…",
 "kind": "progress", "value": 0.95, "chapter_index": 11, "total_chapters": 12,
 "occurred_at": "2026-07-15T20:02:00Z"}
```

Four decisions, each load-bearing:

**`story_id`, never the internal `id`.** `items.id` is a surrogate key the
database assigns. No external generator or Firestore export can know it. The pair is the
stable external identity; the loader resolves it. Records naming an unknown book are
counted and skipped, not fatal — a Firestore export can legitimately reference a story
not yet synced.

**`kind` is only `like` | `rating` | `progress`.** Exactly what Firestore has:
`stories/{id}/likes/{uid}`, `stories/{id}/ratings/{uid}`,
`users/{uid}/readingProgress/{storyId}`. Nothing richer, because nothing richer exists.
A format promising data production can't supply would be building on sand.

**`completion` is not a kind — it's derived.** The platform emits no completion event, so
the loader infers it: last chapter **and** >90% scrolled. Both halves are needed —
chapter index alone counts someone who opened the final chapter and stopped. One
definition, shared by every source.

**`occurred_at` makes reload idempotent.** The primary key is
`(user_id, item_id, kind)`, and a conflict keeps whichever signal is **more recent**.
That matters because `readingProgress` is current *state*, not an event: re-exporting it
must update a reader's position, not accumulate duplicates. Out-of-order delivery can't
rewind a reader either.

### What the loader does

```
read records
  → validate kind
  → resolve story_id → items.id
  → compute engagement weight   like 1.0 · rating≥4 1.0 · rating 3 0.4
                                completion 0.8 · progress 0.4×scroll · rating≤2 0.0
  → derive completion from progress → writes a SECOND row
  → upsert, keeping the more recent on conflict
```

A completing `progress` record writes **two** rows — the progress and the derived
completion — because they carry different information: how far they got, and that they
finished.

### Then aggregation

**`item_stats`** — SQL aggregates per item (likes, ratings, avg rating, completions),
then `pop_score` computed **in Python** via `scoring.popularity()`. Deliberately not in
SQL: reimplementing Bayesian damping and log compression in SQL would be a second
definition of the formula, free to drift from the one the unit tests cover.

`n_interactions = likes + ratings + completions` — note **progress alone doesn't count**.
A page-turn is too weak a signal to license behavioral weighting.

**`user_taste`** — the engagement-weighted mean of the embeddings of books a reader
engaged with positively, L2-normalized. Normalizing matters: cosine ignores magnitude,
so an unnormalized mean would give heavy readers longer vectors for no reason. Books
rated ≤2 are excluded from the mean *and* recorded as suppressed — averaging in a
rejected book would aim recommendations at exactly what the reader rejected. Readers
with under 3 weighted signals get no vector at all and fall back to popularity.

**`item_cooccurrence`** — a self-join over readers who engaged with both books, with
shrunk cosine similarity and a `co_count ≥ 3` floor. Built but inert while
`w_cf_ceiling = 0`.

---

## Part 3: synthetic data, and why personas

### Why synthetic data at all

The behavioral half of scoring needs reader signals, and there aren't real users yet.
Without them, two of three scoring terms are inert *twice over*: `pop_score = 0` because
`item_stats` is empty, **and** `w_pop = 0` because `α` derives from `n_interactions`,
also 0. The three-term formula collapses to `score = src × sem`, and
`/recommend/behavioral` returns identical results for every reader.

So synthetic readers let the whole system be developed and exercised before real ones
exist — and, crucially, they flow through the **same loader and the same aggregation**
that Firestore will, so the code being exercised is the code that will run in
production.

### Why personas — the layered structure

A persona is a synthetic reader's taste:

```python
Persona(user_id="synth_00042",
        genres=["horror", "mystery-thriller"],   # affinity
        themes=["isolation", "revenge"],          # sharper affinity
        activity="medium",                        # 6-15 books
        target_count=11,
        generosity=0.3)                           # rates slightly high
```

The naive alternative — pick random books for random users — produces data that is
useless for the thing it's meant to exercise. Each persona field exists because some
downstream mechanism needs the variation it creates:

| Layer | What it creates | What would break without it |
|---|---|---|
| **Genre/theme affinity** | readers with *distinguishable* taste | Every taste vector would converge on the catalog centroid. Behavioral mode could never be seen to personalize, and co-occurrence would be meaningless noise. |
| **Exploration (25%)** | cross-genre overlap | Readers trapped in one genre → no cross-genre co-occurrence ever appears → CF has nothing to find, and diversity has nothing to measure. |
| **Activity tiers** (55% light / 35% medium / 10% heavy) | a realistic spread of reader effort | Uniform activity means every reader clears the `MIN_SIGNALS_FOR_TASTE` gate, so the *fallback* path is never exercised. Today 40 of 250 readers are correctly skipped as too thin — that's the gate being tested. |
| **Rating generosity** | per-reader rating bias | Every book's average rating would converge, leaving the Bayesian prior nothing to shrink *toward* and making `bayes()` untested. |
| **Latent popularity (Zipf)** | a long tail | The most important one. With uniform interaction counts, `pop_score` would be flat, volume damping never exercised, and P95 normalization meaningless. Observed: one book with 85 interactions, most with a handful. |
| **Completion ↔ rating correlation** | finishing predicts liking | `avg_rating` would be independent of `completions`, so a popularity prior blending the two would be blending noise. |

The last one is worth stating as a general principle: **synthetic data has to contain
the correlations the system claims to exploit**, or it validates nothing. A generator
that emits independent random signals will make any aggregation look like it works while
proving nothing.

Personas are sampled from the catalog's *own* genres and themes, so they're always
satisfiable — a persona who loves a genre the catalog lacks would generate nothing. They
can also be supplied explicitly (`synthetic.Persona`) if you want hand-authored or
LLM-authored ones.

### Why it's generated in code, not by a language model

- It must reference `story_id`s in *this* catalog.
- It needs statistical structure a model won't hold coherently over thousands of records
  — the power law, per-reader affinity, and the co-occurrence that emerges from readers
  sharing affinities.
- It must be **reproducible from a seed**, or two eval runs aren't comparable.

An LLM *would* be reasonable for inventing 20–40 persona descriptions with more
character than sampled affinities. The event stream should not come from one.

> A related bug worth knowing: `generate()` originally called `datetime.now()`, which
> made it *almost* deterministic — everything else came from a seeded RNG. Worse than
> obviously broken, because it would have surfaced later as two eval runs mysteriously
> disagreeing. Timestamps now anchor to a fixed `REFERENCE_NOW`.

### What it can and cannot tell you

**Can validate mechanics.** Verified end to end:
`interactions → n_interactions → α = 0.29 → w_pop = 0.071 → score shifted −0.0089`.
`pop_score` spans 0.419–0.867 instead of being uniformly zero. Different readers get
different recommendations; readers below the gate correctly fall back.

**Cannot validate taste.** Any offline quality metric computed on it measures whether
the recommender recovers the generator's own assumptions. That's circular. Treat it
exactly like `USE_MOCK=true` embeddings: the plumbing is real, the semantics are not.

Every synthetic reader is prefixed `synth_` so they can be purged and can never quietly
enter a real measurement.

```bash
python -m recommendation_engine.sync.seed --readers 250 --cooccurrence
python -m recommendation_engine.sync.seed --readers 5 --dry-run
python -m recommendation_engine.sync.seed --purge
```

### The migration to real data — done, and it landed elsewhere

This section used to describe writing a `sync/firestore.py` that read likes, ratings
and reading progress with the Admin SDK and emitted `InteractionRecord`s into the
same loader. **That is not how it went**, and the reason is the interesting part.

The private-data problem that made a server-side reader mandatory under Firestore did
not disappear when the platform moved to Postgres — it got sharper, because recs now
*shares a database* with the data it must not read. So the derivation moved across
the service boundary entirely:

| Planned | Actual |
|---|---|
| `sync/firestore.py` in this repo | `internal/store/recommendations.go` in **story-data** |
| Admin SDK reads → `InteractionRecord` → loader | One SQL statement, in-database |
| Convention keeps recs out of private data | **A role grant** does (`migrations/000020`) |

The steps now:

1. Run `story-data sync-recs` — it derives `recommendations.interactions` directly.
2. Run `seed.py --refresh-only` to recompute the aggregates.
3. Run `--purge` when you no longer want synthetic readers mixed in. Not urgent:
   they are `synth_`-prefixed and story-data's sync excludes that prefix, so real
   and synthetic readers coexist without interfering.

See [jobs.md](jobs.md) and [security-and-roles.md](security-and-roles.md).

The only genuinely new thing needed for **CF** to become useful is still missing: an
append-only reader event log (impression / open / chapter-complete / dismiss).
Without it, co-occurrence stays too sparse to beat the popularity prior — which is
why `w_cf_ceiling` is 0 — and CTR and completion-rate can never be measured at all.
