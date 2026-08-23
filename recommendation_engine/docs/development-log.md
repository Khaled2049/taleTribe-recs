# Development log — bugs, changes, and how they were caught

Every issue hit while building the recommendation service, in the order it was
found, grouped by the *kind* of mistake rather than by module. Each entry has the
symptom, the root cause, the fix, and what generalizes.

Grouped this way on purpose: the categories turned out to matter far more than the
individual bugs. Six separate issues shared one shape — **no error, just quietly
worse results** — and that shape is the defining hazard of a retrieval system.

---

## Category 1: Silent failures

The dangerous class. Nothing raises, nothing logs, results just get worse. Five of
these six would have shipped unnoticed.

### 1.1 RRF applied to single-seed queries (the worst one)

**Symptom.** A live query against real Gemini embeddings returned scores
`1.0000, 0.9839, 0.9683, 0.9531, 0.9385` — *byte-identical* to what the mock
md5-based embedder had produced earlier. That coincidence is what gave it away.

**Root cause.** `pipeline.rank` called `merge_candidate_records` (Reciprocal Rank
Fusion) unconditionally. Fusing a **single** ranked list discards the similarity
and replaces it with a function of rank:

```
score(rank) = (1/(k+rank)) / (1/(k+1))    k = 60
  rank 1 → 1.0000    rank 2 → 61/62 = 0.9839    rank 3 → 61/63 = 0.9683
```

The true cosine similarities were `0.6704, 0.6672, 0.6579, 0.6573, 0.6474`.

**Three consequences, all invisible:**
1. Score magnitude meant nothing — the top result was *always* exactly 1.0, whether
   it was an excellent match or garbage. No caller could threshold on it.
2. MMR's relevance term went nearly uniform (1.0 vs 0.98 vs 0.97), so the
   diversity term decided the ordering essentially on its own.
3. The three-term blend weighed the popularity prior against a rank artifact
   instead of a real similarity.

**Fix.** Single seed → use cosine directly. Multiple seeds → RRF, because per-seed
cosines genuinely aren't comparable (a query about an obscure book yields uniformly
lower similarities than one about a well-covered genre, so score-based merging
would let the better-covered seed dominate on scale alone). This was what the plan
specified; the implementation just didn't branch.

**What to learn.** *The ordering was correct throughout* — only the magnitudes were
wrong. Any test asserting "the right books come back in the right order" passed.
It was caught by noticing two supposedly unrelated systems produced identical
numbers. **Suspiciously round or suspiciously familiar numbers are a signal.**

---

### 1.2 Mock and real embeddings coexisting in one vector space

**Symptom.** None. `SELECT embed_model, count(*)` showed 292 rows from
`MockEmbeddingProvider` and 8 from `GoogleAIEmbeddingProvider` in the same table.

**Root cause.** The backfill's resume logic compared only `embed_input_sha` — the
hash of the *text* being embedded. Swapping the embedder doesn't change the text,
so every mock-embedded row looked "already done" and was skipped. md5-derived
vectors would have sat permanently in a Gemini corpus, matching nothing and
dragging down every query.

**Fix.** Added a public `model_id` property to `EmbeddingProvider`
(`google:gemini-embedding-001` vs `MockEmbeddingProvider`) and made the skip
condition require a matching `embed_model` **and** `embed_task_type`. A provider or
model change now forces re-embedding automatically.

**What to learn.** A content hash answers "did the input change?" — not "is the
stored artifact still valid?". Anything derived from an input needs the
**derivation** identified too, not just the input. This was caught only by
inspecting the table before a paid run.

---

### 1.3 `task_type` would have invalidated every existing vector

**Not a bug — a hazard identified and defended against.** The service needed
asymmetric embeddings (`RETRIEVAL_DOCUMENT` for the catalog, `RETRIEVAL_QUERY` for
user input), which meant editing `embedding_provider.py`, shared with the live
story agent.

Adding `taskType` to requests would silently invalidate every vector already
stored in `chapter_chunks`, `semantic_memory` and `episodic_memory` — and because
`VectorStore` deliberately has no brute-force fallback, chapter RAG would return
**nothing at all**, with no error raised anywhere.

**Defence.** `task_type` defaults to `None`, which omits the key entirely rather
than sending `null`, and a test pins the resulting request body byte-for-byte:

```python
assert body == {
    "model": "models/gemini-embedding-001",
    "content": {"parts": [{"text": "hello"}]},
    "outputDimensionality": 768,
}
assert "taskType" not in body
```

**What to learn.** When extending shared code whose failure mode is silent, pin
the *existing* behaviour with a characterization test before adding anything.

---

### 1.4 Genre crosswalk put Camus in non-fiction

**Symptom.** *The Plague* came back tagged `{fiction, non-fiction}`. A reader
filtering for non-fiction would get a plague allegory.

**Root cause.** Its Freebase labels are `{Existentialism, Fiction, Absurdist
fiction, Novel}`, and the crosswalk mapped `Existentialism → non-fiction`. Checking
the corpus showed the problem was structural, not a one-off:

| Subject label | Also tagged with a fiction marker |
|---|---|
| Philosophy | 46% |
| History | 42% |
| Sociology | 46% |
| Psychology | 40% |
| Existentialism | 37% |

A static label→category map **cannot** decide fictionality for these.

**Fix.** The umbrella labels the crosswalk *drops* as non-discriminative
(`Fiction`, `Novel`, `Speculative fiction` — each matching a third of the corpus)
turn out to be the only reliable **fictionality** signal. So they're dropped for
genre purposes but used to break the tie: an explicit fiction marker with no
explicit non-fiction marker suppresses `non-fiction`. Tagged as both is left alone,
because the corpus is genuinely ambiguous there and guessing is worse.

**Result:** false `non-fiction` fell 723 → 522, and 1,963 previously
*uncategorized* records gained a category (10,010 → 11,973), because the fallback
rescues books whose only labels were subjects plus an umbrella.

**What to learn.** Data you discarded for one purpose may be exactly what another
purpose needs. And: **query the data before encoding a judgement about it** — the
22–46% co-occurrence figure is what turned an opinion into a decision.

---

### 1.5 pgvector returns a `Vector`, not a list

**Symptom.** `TypeError: 'Vector' object is not iterable` and `object of type
'Vector' has no len()` — but *only* in integration tests. Every MMR unit test
passed.

**Root cause.** pgvector's asyncpg codec returns a `Vector` object. MMR did
`len(vector)` and `list(...)`, which works on the list fixtures used in unit tests
and fails on real database rows.

**Fix.** Normalize once at the DB boundary (`_row_to_dict`), so nothing downstream
needs to know the vector came from pgvector.

**What to learn.** A test double that's "close enough" hides real type
incompatibilities. This is the strongest argument in this project for having
integration tests at all — no amount of unit testing would have found it.

---

### 1.6 The index we tuned is not being used

**Symptom.** A first recall probe reported `recall@10 = 1.000` at `ef_search` 10, 40, 100
*and* 200. A suspiciously perfect result that doesn't vary is not a good sign.

**Root cause.** `EXPLAIN` showed `Seq Scan on items` for both the "approximate" and
"exact" queries — they were the same query. The HNSW index is valid (8.3 MB,
`indisvalid = t`), but at **1,987 rows Postgres correctly decides a sequential scan plus
sort is cheaper** than an index scan. Brute force over 2,000 vectors genuinely is fast.

It only picks HNSW with *both* alternatives disabled:

| Planner setting | HNSW used |
|---|---|
| default | no |
| `enable_seqscan=off` | no — falls back to bitmap + sort |
| `enable_seqscan=off, enable_bitmapscan=off` | **yes** |

**Consequences.** Every query today is exact, so `hnsw.ef_search` and
`hnsw.iterative_scan` are **inert** — carefully implemented, correct, and currently doing
nothing. Latency measured now reflects brute-force scanning and will not resemble
production once the catalog grows. And a naive recall sweep produces a vacuous 1.000.

Estimated costs for the two plans are within ~6% (780 vs 828), so the crossover sits
close to the current catalog size.

**Fix.** None to production — the planner is making the right call, and forcing an index
on a 2,000-row table would be slower. The *eval harness* must force the index so it
measures the approximation loss the index will have in production rather than the
planner's current avoidance of it. Documented in
[evaluation.md](evaluation.md), Part 2.

**What to learn.** Two things. First: **a configuration knob having no effect is a state
worth detecting.** Nothing was broken, nothing raised, and the tuning was simply not
reached — the classic shape of a silent failure. Second: **a metric that doesn't vary
when you vary its input is measuring the wrong thing.** The invariant 1.000 was the tell,
not the answer.

---

## Category 2: SQL and driver traps

Loud failures, quick fixes, but each cost time and none were guessable.

### 2.1 `could not determine data type of parameter $16`

The embedding parameter appeared both as an inserted value *and* inside a
`CASE WHEN $16 IS NULL` expression, leaving Postgres unable to infer its type.
Fixed with an explicit `$16::vector` cast in both positions. All array and int
params got explicit casts too, for the same reason.

### 2.2 `operator does not exist: text %% text`

I wrote `%%` to "escape" the pg_trgm similarity operator. That habit comes from
drivers using `%s` placeholders — **asyncpg uses `$N` and does no %-formatting at
all**, so the escape produced a literal `%%`. The operator is a single `%`.

**Learn:** know whether your driver rewrites the query string. asyncpg doesn't.

### 2.3 `SET LOCAL`, never session `SET`

Not a bug that occurred — one designed out. HNSW query tuning (`hnsw.ef_search`,
`hnsw.iterative_scan`) is applied with `SET LOCAL` inside an explicit transaction.
Under a transaction-mode pooler (Neon's pooled endpoint, PgBouncer) a session-level
`SET` is discarded or leaks to an unrelated caller, so queries would silently run
at the default `ef_search = 40` and return worse neighbours. Free to do correctly
now; a genuinely mysterious production bug later. An integration test asserts the
setting is live inside the transaction and gone afterwards.

Same category: `statement_cache_size=0` on the asyncpg pool, because transaction
pooling plus prepared-statement caching produces intermittent
`prepared statement already exists` errors that only appear under concurrency.

### 2.4 A literal null byte in a source file

`SyntaxError: source code string cannot contain null bytes` on import. An f-string
separator had been written as a raw `\x00` instead of a delimiter character. Found
by scanning the file's bytes. Harmless once located, but the error message points
at the import, not the line.

---

## Category 3: Cost and economics

This is where design decisions had direct financial consequences.

### 3.1 The cache stored post-filtered output — fixed *before* it cost anything

**The problem.** Model output is filtered against a controlled vocabulary. The
normalization cache initially stored the **filtered** result. Since the vocabulary
is meant to be tuned iteratively from the violation counter, that meant every
tuning pass would require regenerating the entire corpus at full token cost.

**Fix.** Cache the model's **raw** output; apply the vocabulary filter on read.

**What it saved, measured.** After the 2,000-book run, the diagnostics showed 1,194
dropped themes. Fixing the vocabulary and re-running:

```
from_cache: 1893   generated: 97   calls: 82   dropped_themes: 1194 → 27
```

Vocabulary tuning cost **$0**. Under the original design it would have cost $0.30
per iteration, forever.

**What to learn.** When caching a multi-stage derivation, cache the **earliest**
stage you can still cheaply re-derive from. Ask "which stage am I most likely to
change?" and cache upstream of it.

### 3.2 Safety blocks amplified 12× by batching

**Symptom.** During the real run:

```
normalization_batch_failed size=8
  error=no candidates returned (promptFeedback={'blockReason': 'PROHIBITED_CONTENT'})
```

**Root cause.** Books are batched 8 per LLM call to amortize the ~970-token
vocabulary block. Gemini's safety filter blocks some literary plot summaries —
entirely expected for a corpus of crime, war and horror novels. But a block rejects
the **whole prompt**, so one problematic book destroyed all 8. Twelve blocked
batches cost 96 records, roughly 84 of them innocent bystanders.

Retrying is useless: a content block is *deterministic*, so the same batch fails
identically forever.

**Fix.** On a content block, retry the batch as individual calls to isolate the
offender. Records lost fell **96 → 9**.

**What to learn.** Batching trades cost against **blast radius**. That's fine when
failures are independent and transient, and bad when one item can fail the whole
request. Ask which kind you have — and note the retry policy has to distinguish
*transient* (retry as-is) from *deterministic* (retry differently, or not at all).

### 3.3 The cost estimate, corrected twice

| Stage | Full-corpus estimate | Basis |
|---|---|---|
| Planning | $2.10 | assumed ~570 input tokens/book |
| After measuring corpus | $2.48 | real word counts: 6.42M post-truncation words |
| Extrapolating the 8-book test | $3.24 | **biased high** — the first 8 records are famous novels with unusually long summaries |
| **Actual, from the 2,000 run** | **~$2.40** | $0.33 for 2,000 books |

**What to learn.** A small sample from the *head* of a file is not a random sample.
The grounded estimate (aggregate statistics over the whole corpus) beat the
empirical one from a biased sample. Measure the population, not the first page.

### 3.4 Free-tier limits drove the batch size

15,500 individual calls would exceed a ~1,000 requests/day free tier for over two
weeks. At 8 books per call it's ~1,940 requests — two days free, minutes paid. The
batch size is a **rate-limit** decision first and a cost decision second.

---

## Category 4: Data quality surprises

Things the corpus did that the plan assumed otherwise.

### 4.1 246 duplicate titles, but only 5 actual duplicates

The plan said "246 duplicate titles → dedupe". Deduping on
`(normalized title, normalized author)` collapsed only **5**. The other ~241 are
genuinely different books sharing a title. Title-only deduplication would have
silently deleted hundreds of legitimate records.

### 4.2 The model confused all three of my axes

The 2,000-book run's violation counter was more informative than expected:

| Rejected term | Count | What it actually was |
|---|---|---|
| `philosophical`, `absurdist`, `hopeful` | 85 | **valid tones**, filed under themes |
| `mystery`, `adventure`, `romance`, `science fiction` | ~100 | **genres**, already captured elsewhere |
| `love`, `trauma`, `greed`, `ambition`, `deception` | ~60 | genuine vocabulary gaps |

I had been *discarding* all of it. Three fixes:

- **Relocate, don't discard** (`split_misfiled`): a themes entry that is a valid
  tone moves to tone, and vice versa. Recovered 85 terms.
- **Recognize noise** (`is_genre_noise`): genre names are dropped quietly rather
  than inflating the counter and inviting bogus vocabulary additions.
- **Fill real gaps**: +15 theme terms, +~19 aliases.

Dropped themes fell **1,194 → 27**; dropped tones **427 → 21**. Current vocabulary:
223 themes, 33 tones, 79 theme aliases, 21 tone aliases, 20 suppressed genre names.

A deliberate limit: fixing the *prompt* to prevent the confusion at source would
bump `PROMPT_VERSION` and re-bill the corpus. Post-processing recovers the same
terms for free, so the prompt fix waits and rides along with the eventual
full-corpus run.

**What to learn.** Instrument what you throw away. The violation counter was added
as a nice-to-have and became the most valuable diagnostic in the project. Also:
"the model ignored my instructions" is often "the model disagreed about which
field a value belongs in" — recoverable, not garbage.

### 4.3 Non-fiction has no usable theme vocabulary

A Hume philosophy book received **zero** themes. The model correctly produced
`epistemology`, `empiricism`, `induction`; the fiction vocabulary correctly
rejected all three. Not a bug — a genuine scope limit. Non-fiction recommendations
currently rest almost entirely on premise text. Only ~520 corpus records map to
non-fiction, so it's noted rather than solved.

### 4.4 The empty-string-not-`{}` trap

The Freebase genre column is strict single-line JSON — except when a book has no
genres, where it's an **empty string**, not `{}`. That's the one input a naive
`json.loads` dies on, and it affects 22.5% of records (3,718).

---

## Category 5: Developer-experience bugs

Not user-facing, but each one cost real time.

### 5.1 The API key was in `.env` but unreadable

`backfill` refused to run for lack of `GOOGLE_AI_STUDIO_API_KEY` — which was
sitting in `.env`, populated and valid. Only the HTTP servers called
`load_dotenv`; the CLI entry points constructed `RecSettings()` directly.

Fixed with a shared `env.py`, called from all four entry points. Deliberately *not*
at package import: the test suite sets `os.environ` before importing the app, and a
package-level `load_dotenv` would leak a developer's local `.env` into tests that
depend on a variable being absent.

**Learn:** if config loading isn't in one place, it's in zero places.

### 5.2 `--dry-run` required an API key

Ordering bug: the LLM client was constructed before the dry-run early return, so a
mode that makes no API calls demanded credentials. Fixed by parsing the corpus
first — the part needing no credentials — then returning.

### 5.3 A singleton constructed before the thing it depends on exists

**Symptom.** `AttributeError` on a `read_pool_lazy` attribute I had invented while
wiring the explanation cache into the app factory.

**Root cause.** The connection pool is opened by the FastAPI **lifespan**, which
runs *after* `create_app()` builds the singletons. So `ExplanationCache(db.read_pool)`
at factory time reaches for a pool that doesn't exist yet.

**Fix.** The cache holds the `Database` object and resolves `.read_pool` per call
through a property. Construction-time wiring, call-time resolution.

**What to learn.** In a framework with a startup hook, anything built in the factory
can only capture *references*, not resources. The same trap catches HTTP clients,
caches, and anything else with a connect step.

### 5.4 A bare `asyncpg.connect` has no pgvector codec

**Symptom.** `invalid input for query argument $6: [1.0, 0.0, ...] (expected str,
got list)` — in 15 route tests at once, while the routes themselves worked fine
against the live server.

**Root cause.** `Database` registers the pgvector codec via the pool's `init` hook.
Test helpers that call `asyncpg.connect` directly bypass that, so a Python list
passed for a `vector` column isn't adapted.

**Fix.** A `_connect()` helper in the test module that registers the codec.

**What to learn.** Second cousin to [1.5](#15-pgvector-returns-a-vector-not-a-list):
both are the same lesson from opposite directions. Type adaptation lives on the
*connection*, so any code path that makes its own connection has to repeat the
setup. If setup matters, don't let there be two ways to connect.

### 5.5 A "reproducible" generator that wasn't

**Symptom.** `test_generation_is_reproducible` failed: same seed, timestamps
differing in the microseconds.

**Root cause.** `datetime.now(timezone.utc)` inside `generate()`. Everything else
came from a seeded RNG, so the output was *almost* deterministic — which is worse
than obviously non-deterministic, because the failure would have shown up later as
two eval runs that mysteriously disagreed.

**Fix.** Timestamps anchor to a fixed `REFERENCE_NOW` by default; pass `now=` to opt
into wall-clock recency and give up reproducibility knowingly.

**What to learn.** A seeded RNG does not make a function pure. Any *ambient* input —
clock, environment, filesystem order, dict iteration over unsorted input — leaks
non-determinism. I only caught this because I had written down "reproducible from a
seed" as a property and then tested it.

### 5.6 A test that deleted the developer's dataset

**Symptom.** After a green test run, a `--refresh-only` reported `0 items, 0 taste
vectors`. The 250 seeded readers were gone.

**Root cause.** `test_purge_removes_synthetic_readers` called the production
`purge_synthetic()`, which by design deletes **every** `synth_%` reader. The test
fixture was `synth_purge_me`, but the function's scope was everything.

**Fix.** `purge_synthetic(pool, prefix=...)` — default unchanged (an operator wants
the full wipe), narrowed in the test to its own fixture. It also refuses an empty
prefix, so a falsy argument can't widen the blast radius to every row.

**What to learn.** A test that calls a destructive production function inherits its
full blast radius. Destructive helpers should take a scope parameter *for testability*,
even when production always wants the widest scope.

### 5.7 `timeout` doesn't exist on macOS

`timeout 900 python -m ...` → exit 127, command never ran, and the failure looked
like a JSON parse error downstream. GNU coreutils isn't installed by default on
macOS. Watch for exit 127.

---

## Category 6: Mistakes in the tests themselves

Six of these. Worth recording, because in each case **the test was wrong and the
code was right** — and the temptation to "fix" working code is real.

| Test error | Reality |
|---|---|
| Expected `percentile([1,None,3], 0.5) == 3` | Nearest-rank on 2 values gives index `round(0.5×1) = 0` → `1`. Python's banker's rounding. |
| Expected RRF order `[a, c, b]` | `b` and `c` were exactly symmetric (both `1/62 + 1/63`); only the tiebreak separates them |
| Expected CMU source weight `0.25` | With `platform_item_count=1` it's `max(0.25, 1 − 1/5000) ≈ 0.9998` |
| Expected `matched_query_count == 3` | The item appeared in 2 of the 3 lists |
| Asserted tenacity's `exception_types` is a tuple | It's the bare class when a single type is given |
| Orthogonal test fixtures returned nothing | 300 pre-loaded rows with *dense* mock vectors had small positive cosine against any sparse query, pushing zero-similarity fixtures out of the window |

That last one is the interesting one: it's a **test isolation** failure, and it
happened three separate times as the database filled up:

1. Orthogonal fixtures scored 0.0 and were pushed out of the result window by 300
   rows whose *dense* mock vectors had small positive cosine against any sparse
   query. Fixed by giving fixtures a shared dominant component so they outrank
   background noise, keeping secondary components distinct for ordering and MMR.
2. `test_rare_pairs_are_excluded` asserted a global count of zero, but
   `rebuild_cooccurrence` rebuilds the *whole* table and 250 seeded readers
   contributed 189 pairs. Fixed by asserting on the specific pair under test.
3. `test_popular_fallback` assumed fixtures would rank first — true only while every
   `pop_score` was 0. Once real interaction data existed, 872 scored items pushed
   them past the window. Fixed by widening the window, since the claim under test
   was "works with no query vector", not "fixtures rank first".

The pattern: **an assertion that happens to hold on an empty database is not an
assertion about behaviour.** All three passed initially and broke as data arrived.
Integration tests sharing a database with real data have to be written to be robust
to that data existing.

**What to learn.** When a test fails, decide *first* whether the assertion or the
code is wrong. Four of these six looked like code bugs at first glance.

---

## How these were actually found

Ranked by how much each surfaced:

1. **Running it for real.** The safety blocks, the axis confusion, the cost
   correction and the RRF bug were all invisible until real data and a real API
   were involved. Mock-everything testing found none of them.
2. **Inspecting the database directly.** The mixed embedders, the Camus
   miscategorization and the stale genres all came from ad-hoc `psql` queries, not
   from test output.
3. **Instrumenting rejects.** The vocabulary violation counter drove the single
   biggest quality improvement (1,194 → 27).
4. **Noticing coincidences.** The RRF bug was caught because two unrelated systems
   produced identical numbers.
5. **Integration tests.** Found exactly one thing unit tests couldn't — the
   pgvector `Vector` type — but that one was unfindable any other way.

## The transferable lesson

In a retrieval system, **the default failure mode is silence.** A wrong vector, a
mixed embedding space, a discarded label, a lost query-time setting — none of them
raise. They just make results a bit worse, and "a bit worse" is invisible without
either a measurement or a suspicious coincidence.

Hence three habits this project settled into:

- Assert preconditions that would otherwise fail silently, loudly and at startup
  (`/health` refuses to be green without pgvector ≥ 0.8.0, the HNSW index, and a
  768-dim embedder).
- Record provenance for every derived artifact, so staleness is *detectable*.
- Count and expose what you discard.

And the reason the eval harness (Phase 6) matters: right now, quality is judged by
reading result lists and going "hm, that looks right". *Dune* returning three Dune
sequels at a diversity of 0.18 is visibly mediocre, but nothing in the system says
so. Until offline metrics exist, every tuning decision is a guess.
