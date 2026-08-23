# Evaluation — how you would measure whether this is any good

> **Status: nothing here is implemented.** This is a reference and a plan, deliberately
> written *before* the code, to be picked up once real readers exist. Every number in
> the worked examples is computed, not invented.

## Why it isn't built yet

Evaluating a recommender needs **ground truth** — a record of what readers actually
wanted. There isn't one: no labelled relevance data, no click log, and the only reader
signals are synthetic.

Metrics computed on synthetic data measure whether the recommender recovers the
*generator's* assumptions. That's circular. Worse, it's circular while producing
confident-looking decimals, which is more dangerous than having no number at all —
a fabricated 0.83 NDCG invites tuning decisions that are really just overfitting to a
random number generator.

So the honest position: **quality is currently unmeasured**, and it's stated that way
throughout the docs rather than papered over.

What that costs us concretely: `Dune` returns three Dune sequels at an intra-list
diversity of 0.14. That's visibly mediocre, `mmr_lambda` is the knob that would fix it,
and there is currently no way to tell whether turning it helps or hurts. Every tuning
decision right now is a guess.

---

## Part 1: The metrics, explained

Worth understanding independently of this project. All examples below are computed.

### Precision@K and Recall@K

The two most basic. Given the top K results and a set of genuinely relevant items:

```
retrieved (K=5) = [a, b, c, d, e]
relevant        = {a, c, x, y}
hits            = {a, c} = 2

Precision@5 = 2/5 = 0.40     "of what I showed, how much was good?"
Recall@5    = 2/4 = 0.50     "of what was good, how much did I show?"
```

They trade off. Showing 100 results guarantees high recall and terrible precision.
For a recommendation shelf **precision matters far more** — a reader sees 10 items and
doesn't care that book #4,000 was also relevant. Recall matters for search, not shelves.

Neither cares about **order**, which is their big limitation. A perfect item at position
10 counts the same as at position 1. That's what NDCG fixes.

### NDCG@K — Normalized Discounted Cumulative Gain

The workhorse ranking metric. Three ideas stacked, and the name reads backwards:

- **Gain** — each item has a relevance grade, not just relevant/irrelevant. Typically
  0–3.
- **Discounted** — a good item lower down is worth less, divided by `log2(position + 1)`.
- **Normalized** — divide by the best possible score, so the result is 0–1 and
  comparable across queries.

```
DCG@K  = Σ  rel_i / log2(i + 1)
IDCG@K = the same, computed on the ideally-ordered list
NDCG@K = DCG / IDCG
```

Worked, with grades `[3, 2, 0, 1, 2]`:

| Position | Relevance | `log2(i+1)` | Gain |
|---|---|---|---|
| 1 | 3 | 1.000 | 3.000 |
| 2 | 2 | 1.585 | 1.262 |
| 3 | 0 | 2.000 | 0.000 |
| 4 | 1 | 2.322 | 0.431 |
| 5 | 2 | 2.585 | 0.774 |

```
DCG  = 5.466
ideal ordering [3, 2, 2, 1, 0] → IDCG = 5.693
NDCG = 5.466 / 5.693 = 0.960
```

0.960 says "nearly the best possible arrangement of these particular items". Note it
judges the *ordering* of what you retrieved — a query that misses a great item entirely
can still score well, because IDCG is computed over the retrieved set. Pair it with
Recall@K to catch that.

Why the logarithm specifically: it encodes a belief about attention. Position 1 → 2 is a
big drop (1.000 → 0.631 of the gain), 9 → 10 is small. That roughly matches how people
scan a list.

### MRR — Mean Reciprocal Rank

Only cares where the **first** relevant item appears.

```
first-relevant ranks across 4 queries: [1, 3, 2, none]
reciprocal ranks:                      [1.0, 0.333, 0.5, 0]
MRR = 0.458
```

Right metric when the reader wants *one* answer ("find me that book I half-remember").
Wrong metric for a shelf, where items 2–10 matter as much as item 1. Worth tracking for
the title-resolution path specifically.

### Intra-list diversity

Not a quality metric — a **property** metric. `1 − mean pairwise cosine similarity`
across the returned list.

- near 0 → ten near-identical books (the Dune-sequels failure)
- near 1 → a genuinely varied shelf

This one is measurable **today**, because it needs no ground truth at all — only the
embeddings you already fetched. It's how you tell MMR is doing anything. Currently
implemented in `mmr.intra_list_diversity`.

Don't optimise it alone: perfect diversity is achievable by returning ten random books.
It's only meaningful read alongside a relevance number.

### Catalog coverage

What fraction of the catalog ever gets recommended to anybody. A recommender that only
ever surfaces the same 200 popular books has a coverage problem that no per-query metric
will reveal. Also needs no ground truth — just log which item ids get returned over a
period.

### Cohen's κ — inter-rater agreement

Needed only for the LLM-as-judge approach. Measures agreement between two graders
*beyond what chance would produce*:

```
κ = (p_o − p_e) / (1 − p_e)
```

Worked, on 20 items where a human and the judge agree on 14, the human marks 12
relevant and the judge marks 14:

```
p_o = 14/20                            = 0.700
p_e = (12/20)(14/20) + (8/20)(6/20)    = 0.540
κ   = (0.700 − 0.540) / (1 − 0.540)    = 0.348
```

Raw agreement looks respectable at 70%. But when both graders call most things relevant,
chance alone gets you 54% — so the *real* agreement is 0.348, which is **weak**. That
gap is exactly why raw agreement is a misleading statistic and κ exists.

Rough convention: <0.2 poor, 0.2–0.4 fair, 0.4–0.6 moderate, 0.6–0.8 substantial. The
plan gates at **0.4** — below that, the judge's numbers should not be reported at all.

---

## Part 2: Offline proxies, ranked by rigour

With no ground truth, everything is a proxy. They are not equally trustworthy, and
presenting them as one "quality score" would be dishonest.

### Fully rigorous: the `ef_search` recall sweep

The only measurement here with genuine ground truth, because the ground truth is
*computable*: exact nearest neighbours from a brute-force scan.

```
recall@10 = |approximate_top_10 ∩ exact_top_10| / 10
```

Sweep `ef_search ∈ {40, 80, 100, 200}`, with and without metadata filters, and pick the
smallest value that holds recall above ~0.95. No opinion involved.

**But there's a catch discovered while planning this**, and it's important:

> **At the current catalog size the HNSW index is not being used at all.** The index is
> valid (8.3 MB), but at 1,987 rows Postgres correctly decides a sequential scan plus
> sort is cheaper. It only picks HNSW with *both* `enable_seqscan` and
> `enable_bitmapscan` disabled.
>
> So every query today is **exact**, `ef_search` and `iterative_scan` are inert, and a
> naive recall sweep reports a vacuous `1.000` at every setting — which is exactly what
> a first probe returned.
>
> The two plans are within ~6% on estimated cost (780 vs 828), so the crossover is close
> to the current size. Re-check with `EXPLAIN` as the catalog grows.

Consequence for the harness: it must **force the index** so it measures the
approximation loss the index will have in production, not the planner's current
avoidance of it. And latency measured today reflects brute force, so it will not be
representative later.

### Fully rigorous: latency

p50/p95/p99 over a few hundred queries, per mode (behavioral / ad-hoc in-corpus / HyDE).
Wall clock is ground truth. Must be measured in the forced-index mode too, for the same
reason.

### Rigorous but not about quality: diversity and coverage

Real numbers about real properties. They tell you whether MMR works and whether the long
tail is reachable. They cannot tell you whether recommendations are *good*.

### Semi-rigorous: genre-holdout, and its leak

The idea: hold out a book, query with its premise, and check whether the neighbours
share its genres and themes. Gives Precision/Recall/NDCG/MRR over ~15k books with zero
user data.

**The leak.** `genres` is *part of the embedded text* — `compose_embed_input` includes a
`Genres: fiction, young-adult` line. So retrieving by embedding and then scoring genre
overlap partly measures whether a genre *string* matched, not whether the semantics did.

The honest fix is to measure both ways and report the gap:

| Variant | Query text | Leak |
|---|---|---|
| A | full composed block (with `Genres:`) | high |
| B | premise only | lower — documents still carry genre tokens |

The A−B delta quantifies how much of the score is genre-token matching. Reporting a
single leaky number as "NDCG 0.8" would be misleading; reporting both with the gap is
informative.

What it genuinely measures is **coherence** — does the embedding space cluster sensibly.
Not **taste** — whether a reader would enjoy the result.

### Weak without human calibration: LLM-as-judge

Write ~50 probe queries ("cozy small-town mystery, no gore"), retrieve top-10 for each,
have an LLM grade every result 0–3 against a rubric, compute NDCG@10.

**The structural problem.** If the same model family writes the probes, grades the
results, and calibrates the judge, it's one system's judgment all the way down and the
κ gate is theatre. The calibration sample **must be graded by a human**.

So the workflow is necessarily two-phase:

1. Tooling generates results for the probe set and emits a grading worksheet.
2. A human grades ~20 of them.
3. κ is computed against the LLM's grades. **Below 0.4, refuse to emit metrics.**

Cost is trivial (~$0.05/run). The expensive input is human attention, which is why this
is the last piece to build and the first to skip.

### Rigorous but impossible today: leave-one-out

The classic. Take a reader's liked books, hide one, see whether the recommender surfaces
it from the rest. Real ground truth — an actual human actually liked it.

Needs real users. With synthetic readers it's circular; with zero readers it can't run.
Build it with a hard `--min-users 200` gate that **exits non-zero with an explicit
message** rather than printing a meaningless number from a sample of 3. Refusing is the
correct behaviour, not a failure.

---

## Part 3: Which knobs can honestly be tuned, and when

A metric that changes no decision is waste. Organised by knob:

| Knob | Current | Tunable by | Available |
|---|---|---|---|
| `ef_search` | 100 | recall sweep (index forced) | **now** |
| `mmr_lambda` | 0.70 | relevance-vs-diversity curve | **now**, proxy-based |
| `rrf_k` | 60 | multi-seed judge probes | weakly |
| `ramp_n_min` | 5 | leave-one-out on real likes | needs users |
| `ramp_n50` | 20 | leave-one-out | needs users |
| `w_pop_ceiling` | 0.25 | online CTR | needs users + event log |
| `w_cf_ceiling` | 0.00 | online CTR | needs users + event log |
| `cmu_target_catalog` | 5000 | judgement | never purely empirical |

**Three of the scoring knobs cannot be tuned until Firestore is connected.** Tuning them
against synthetic data would be fitting to the generator's assumptions. That's a real
constraint, not conservatism.

The `mmr_lambda` curve is the most useful thing buildable now: sweep λ ∈ {0.5 … 1.0},
plot proxy-NDCG against diversity, pick the knee. It won't be a *true* quality
optimum, but it will show the shape of the tradeoff and whether 0.70 is anywhere near
sensible — which is more than is known today.

---

## Part 4: Online metrics, once readers exist

Offline metrics approximate. Online metrics measure. But they need instrumentation the
platform doesn't have.

| Metric | Definition | Needs |
|---|---|---|
| **CTR@position** | clicks / impressions, by shelf position | impression + click log |
| **Completion rate** | recommended books actually finished | impression log + progress |
| **Diversity index** | as above, but on what readers actually saw | impression log |
| **Catalog coverage** | fraction of catalog ever surfaced | impression log |
| **Latency SLO** | p95 per mode in production | tracing |

**Everything above the last row needs a `readerEvents` collection** — an append-only log
of `{uid, storyId, kind: impression|open|chapter_complete|dismiss, at}`. It does not
exist. Today's signals are binary likes, create-only ratings, and reading-progress
*state*, none of which record that a recommendation was ever *shown*.

Without impressions you cannot compute CTR, because you have the numerator and not the
denominator. This is the single highest-value addition for making the recommender
measurable, and it's frontend work, not service work.

### Interleaving beats A/B testing for ranking

When there are two ranking variants to compare, the instinct is an A/B test: half the
readers get A, half get B, compare CTR. For *ranking* changes there's a better method.

**Team-draft interleaving**: for a single reader, build one shelf by alternately taking
picks from A and B (like captains picking a team), remember which variant contributed
each slot, and attribute clicks back. Every reader sees a blend, so the comparison is
*within* reader rather than between groups.

It's far more sensitive — it removes between-user variance, which usually dominates —
so it needs roughly an order of magnitude fewer users to reach significance. That
matters a lot when you have hundreds rather than millions.

**One caveat worth knowing:** it measures *preference between rankers*, not absolute
quality. Both could be bad. And it interacts awkwardly with MMR, since diversifying an
interleaved list muddies attribution — the clean approach is to interleave
*pre-diversification* candidates and apply MMR once at the end.

---

## Part 5: The sequence, when you get there

Ordered by dependency, with what each unblocks.

**Stage 0 — now, no users needed.** `metrics.py` as pure functions with hand-computed
tests, and the recall sweep with the index forced. This answers `ef_search` properly and
gives a regression tripwire for the retrieval layer. Genuinely rigorous.

**Stage 1 — after the Firestore reader lands.** Nothing new needed beyond real
`interactions`. Leave-one-out becomes possible once ~200 readers have ≥5 likes each.
That's the first *real* quality number this project will ever produce, and it's worth
waiting for.

**Stage 2 — after `readerEvents` ships.** CTR, completion rate, real diversity, coverage.
Now `w_pop_ceiling` and the ramp parameters become tunable, and CF can be evaluated —
which is what decides whether to lift `w_cf_ceiling` off zero.

**Stage 3 — when there are two variants worth comparing.** Interleaving.

**Optional at any point** — genre-holdout as a regression tripwire, and the LLM judge if
you're willing to grade a calibration sample.

---

## Part 6: Traps to avoid

Collected because each is easy to walk into.

**Reporting a proxy as quality.** Genre-holdout NDCG measures embedding coherence. Call
it that. The moment it's labelled "recommendation quality" in a dashboard, someone will
optimise it and ship something worse.

**Tuning on synthetic data.** It measures recovery of the generator's assumptions. The
generator was written by the same person who wrote the scorer, so the assumptions agree
by construction.

**Trusting an uncalibrated LLM judge.** Without κ against human grades it's a plausible
number generator. The gate must *refuse*, not warn — a warning gets ignored.

**Measuring latency in the wrong regime.** Today's numbers are brute-force scans. When
the catalog crosses the planner's crossover point, the profile changes completely.
Measure both, and label which is which.

**Optimising diversity alone.** Ten random books score perfectly.

**Forgetting the denominator.** CTR without an impression log is not CTR.

**A metric with no decision attached.** If no knob moves as a result, it's a number in a
report. Every entry in Part 3 is paired with the knob it informs — that pairing is the
point.

**Comparing runs that aren't comparable.** Store results (an `eval_runs` table was
planned for this) with the config, catalog size, and probe-set version attached. A
baseline in terminal scrollback is not a baseline.
