# Concepts — embeddings, indexing, and every formula

The main learning document. Builds up from what an embedding *is* to why each term in
the scoring formula has the shape it does. Nothing here assumes prior vector-search
experience.

---

## 1. Embeddings: turning text into position

An **embedding** is a list of numbers that represents a piece of text's *meaning* as a
position in space. Ours have 768 numbers, so every book is a point in 768-dimensional
space.

The useful property is that the model producing them was trained so that **texts about
similar things land near each other**. "A lighthouse keeper losing his mind in
isolation" and "a man alone on a rock, going mad" contain almost no shared words, but
their embeddings are close. That's the entire basis of the recommender: *nearby means
similar*, and we can find nearby points fast.

You cannot read an embedding. Dimension 412 doesn't mean "how gothic it is". The
meaning is distributed across all 768 numbers with no human-interpretable axes. This
matters practically: you can't debug an embedding by inspecting it, only by looking at
what it retrieves.

### Measuring closeness: cosine similarity

We compare by **cosine similarity** — the cosine of the angle between two vectors.

- `1.0` = identical direction
- `0.0` = unrelated (perpendicular)
- `-1.0` = opposite

We use the *angle* rather than the straight-line distance because magnitude is noise
here. A long summary and a short one about the same book may produce vectors of
different length pointing the same way. The direction carries the meaning.

In SQL, pgvector's `<=>` operator gives cosine **distance**, so similarity is
`1 - (a <=> b)`. In practice real similarities cluster surprisingly high — a good
match in our catalog scores around `0.65–0.92`, not `0.99`. Absolute values are only
meaningful relative to each other.

### Asymmetric embeddings (task types)

Gemini's embedding model accepts a `taskType`, and we use two:

- `RETRIEVAL_DOCUMENT` for catalog entries
- `RETRIEVAL_QUERY` for what a reader types

The model then projects the two *differently*, on the theory that a question and an
answer aren't the same kind of object even when they're about the same thing. It
measurably improves retrieval over embedding both identically.

The catch: **a corpus must never mix task types.** Vectors embedded as documents and
vectors embedded as queries live in slightly different places, so mixing them
degrades every search with no error raised. We record `embed_task_type` on every row
and the ingest refuses to mix.

### Matryoshka truncation, and why 768

`gemini-embedding-001` natively emits **3072** numbers, but it's an MRL (Matryoshka
Representation Learning) model: it's trained so that the *first* N numbers are
themselves a usable embedding. So you can ask for 768 via `outputDimensionality` and
get a coherent, if slightly less precise, vector.

We take 768 for three reasons: it matches what the story agent already uses for
chapter RAG (one dimension across the whole platform), Firestore's native vector
index caps at 2048 so the full 3072 was never an option there, and 768 floats per row
is a quarter of the storage and index memory of 3072.

**This is also the escape hatch for scale.** If the HNSW graph outgrows memory at
200k+ books, the move is to add a `vector(256)` **column** with its own index — never
to mutate the 768 one, which would invalidate everything at once.

### What we embed — and what we deliberately don't

We **never embed raw prose.** A plot summary is hundreds of words of plot
incident — who did what to whom, in order. Embedded directly, that buries the signal a
recommender ranks on under a mass of proper nouns and sequence. Two unrelated books
that both feature a train and a betrayal end up neighbours.

So each book is first normalized to a fixed schema, and *only that* is embedded:

```
Title: Animal Farm
Author: George Orwell
Genres: fiction, young-adult
Premise: After a wise old boar inspires a revolution, the animals of Manor Farm
         overthrow their human farmer...
Themes & tropes: rebellion against tyranny, corruption of power, class struggle
Tone: satirical, bleak, tragic
```

One subtlety with real consequences: **an absent field drops its entire line** rather
than emitting `Author: ` with nothing after it. An empty label is not neutral — it's a
token sequence the model finds similar across every authorless book, which would
quietly cluster 14% of the corpus by a missing value.

---

## 2. Indexing: finding neighbours without scanning everything

Comparing a query against 2,000 books is trivial. Against 200,000 it isn't, and the
naive approach is O(n) per query. So we build an index.

### HNSW

**Hierarchical Navigable Small World** builds a multi-layer graph over the vectors.
Sparse upper layers have long-range links; dense lower layers have short local ones.
A search enters at the top, greedily hops toward the query, drops a layer, repeats.
Think of it as travelling by plane, then train, then walking.

It's **approximate**: it can miss a true nearest neighbour. That's the trade — near
enough answers, logarithmically rather than linearly.

Our index:

```sql
CREATE INDEX items_embedding_hnsw
  ON recommendations.items USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 200)
  WHERE is_eligible;
```

Three parameters, and a fourth set at query time:

| Parameter | When | What it does |
|---|---|---|
| `m = 16` | build | Links per node. Higher = better recall, more memory, slower build. 16 is the usual default. |
| `ef_construction = 200` | build | Candidates considered while inserting each node. Higher = better graph, slower build, no runtime cost. |
| `ef_search = 100` | **query** | Candidates held during a search. **The recall dial you actually tune.** |
| `WHERE is_eligible` | build | Partial index — see below. |

`ef_search` is the one that matters day to day, and it must be **at least 2× your
result limit**. Asking for 50 results with `ef_search = 40` means the search never
holds 50 candidates at once, so it cannot return 50 good ones. Our default is 100 for
a 50-row pool, and the settings module logs a warning if that ratio is violated.

### Why the index is partial

`WHERE is_eligible` means ineligible rows — unpublished, deleted, or normalized with
low confidence — **never enter the graph at all**. This isn't a micro-optimisation; it
removes the single highest-selectivity filter from the recall problem below, instead
of asking the index to work around it. It also keeps the graph smaller and better
connected.

### The filtering problem, and iterative scans

Here's the non-obvious failure mode. Suppose you want *horror* books near a query:

```sql
WHERE genres && ARRAY['horror'] ORDER BY embedding <=> $1 LIMIT 10
```

An HNSW search walks the graph collecting the ~100 nearest vectors, *then* the filter
runs. If only 3 of those 100 are horror, you get 3 results — not the 10 nearest horror
books. Worse, you get them with **no error**; the query just under-returns.

**pgvector 0.8.0** fixed this with `hnsw.iterative_scan`: if the filter eliminates too
many candidates, the scan resumes and keeps walking until it has enough. That single
feature is why the service requires ≥ 0.8.0 and why `/health` returns 503 below it.

We use `relaxed_order`, which allows results back slightly out of distance order in
exchange for speed. That's free for us — we re-score and re-order everything
downstream anyway, so the index's ordering is never the final ordering. `max_scan_tuples`
bounds the worst case so a very selective filter can't scan the whole table.

### The pooler trap

These are set with `SET LOCAL` **inside an explicit transaction**, never as a session
setting. Under a transaction-mode connection pooler (PgBouncer, Neon's pooled
endpoint) a session-level `SET` is discarded or leaks to an unrelated caller. The
query would then silently run at pgvector's default `ef_search = 40` and return worse
neighbours, with nothing in the logs. Costs nothing to do correctly; a genuinely
mysterious production bug otherwise.

### The other indexes

Vector search finds *candidates*; the other indexes make *filtering* them cheap.

| Index | Type | For |
|---|---|---|
| `items_genres_gin`, `items_themes_gin` | GIN | array overlap (`genres && ARRAY['horror']`) |
| `items_title_trgm`, `items_author_trgm` | GIN + trigram | fuzzy title matching — readers misremember titles |
| `items_word_count` | B-tree (partial) | length filters |
| `items_source_elig` | B-tree | platform-only queries |
| `items_story_id_key` | unique | one catalog row per story, and idempotent upserts |

**GIN** (Generalized Inverted Index) maps each element to the rows containing it —
the right structure for "which rows have `horror` in this array". **Trigram** indexes
break text into 3-character sequences so `"Desert Propecy"` still matches
`"Desert Prophecy"`; that's what makes `%` (similarity) queries fast.

---

## 3. Retrieval strategies

### Single seed: just cosine

One query vector, one KNN search, sorted by similarity. The `sem` term in the scoring
formula *is* the cosine similarity.

### Multiple seeds: why not to average the vectors

For *"I liked Dune and Pride and Prejudice"*, the tempting move is to average the two
embeddings and search once.

**This is wrong, and instructively so.** The mean of two distant vectors points at a
region resembling neither. Embedding spaces aren't convex that way — the midpoint
between science fiction and Regency romance is not "books fans of both enjoy", it's
nothing in particular. You'd get bland results with no clear relationship to either
input.

So each seed is retrieved **independently and concurrently**, and the ranked lists are
merged.

### Reciprocal Rank Fusion (RRF)

Merge by **rank**, not by score:

```
RRF(i) = Σ  1 / (k + rank_q(i))          k = 60
        q∈Q
```

Each list contributes `1/(60 + position)` for each item. Sum across lists; higher is
better.

**Why rank and not score?** Because per-seed cosine scores aren't comparable. A query
about an obscure book yields uniformly lower similarities than one about a
well-covered genre. Merging on raw score would let the better-covered seed dominate
purely by scale, not by relevance.

**What `k = 60` does.** It flattens the top. Ranks 1 and 2 score `1/61` and `1/62` —
nearly identical. So an item placing respectably for *every* seed beats one topping a
single seed. For a multi-book query that's exactly right: you want books connecting
the reader's tastes, not the single closest match to one of them.

We then normalize by the maximum attainable score (`|Q|/(k+1)`, i.e. first place
everywhere) so the result lands in `(0, 1]` and can substitute directly for a cosine
similarity in the blend, with no second set of weights.

> **This caused our worst bug.** RRF was originally applied unconditionally, including
> to *single*-seed queries. Fusing one list replaces every similarity with a function
> of rank: rank 1 becomes exactly `1.0` whether it scored 0.95 or 0.31. Ordering stayed
> correct, so every test passed. See [development-log 1.1](development-log.md#11-rrf-applied-to-single-seed-queries-the-worst-one).

### HyDE: Hypothetical Document Embeddings

For free-text queries there's a shape mismatch. *"Something cozy where nobody dies"*
is a **request**; the catalog holds **descriptions**. Embedding a request and
comparing it to descriptions asks the index to bridge that gap, and short or vague
requests carry too little signal to land anywhere useful.

HyDE inverts it: ask the LLM to **write the book the reader is describing**, then embed
*that* and search. Query and documents become the same kind of object.

The critical implementation detail: the hypothetical document is rendered in the
**catalog's own embedding format**, field for field — same `Title:/Genres:/Premise:/
Themes:/Tone:` template, same controlled vocabulary. A hypothetical document in a
different shape would land in a different region of the space and defeat the purpose.

Real example. Query: *"a lonely lighthouse keeper slowly loses his grip on reality"* →
the model invents:

```
Title: The Salt-Stained Mirror
Genres: horror
Premise: Isolated on a remote island, a solitary lighthouse keeper's meticulous
         routine unravels as the relentless sea and his own solitude warp his...
Themes & tropes: isolation, madness and paranoia, survival against nature,
                 unreliable narrator, existential dread
Tone: bleak, suspenseful, unsettling, melancholy
```

Embedding that retrieved **Pincher Martin** (Golding — a man alone on a rock, losing
his mind) and **The Whalestoe Letters** (letters from an asylum). Neither shares
vocabulary with the original query.

HyDE failure is non-fatal: if generation fails, we embed the raw query instead.
Degraded, still useful.

### Maximal Marginal Relevance (MMR)

Pure relevance ranking produces a shelf of near-duplicates. Ask for something like
Dune and you get five Dune sequels — each individually a great match, collectively
useless, because a reader who liked Dune has already found them.

MMR builds the list greedily, at each step preferring a candidate that is relevant
*and* unlike what's already chosen:

```
next = argmax [ λ · score(i) − (1−λ) · max cos(e_i, e_j) ]
       i∉S                              j∈S
```

The first term is relevance, the second is the penalty for resembling something
already selected. `λ = 0.7` keeps relevance dominant; `λ = 1.0` degenerates to plain
score order.

Practical consequences worth knowing:

- **API results are not sorted by score.** A lower-scoring item can appear above a
  higher one, because it added variety. If the list were strictly descending, MMR
  would be doing nothing.
- It runs on embeddings **already fetched** by retrieval, so it costs no extra round
  trip. A 50×50 similarity matrix in numpy is microseconds.
- We report `intra_list_diversity = 1 − mean pairwise cosine` so you can see it
  working. Our Dune query scores ~0.14, which is *low* — visibly mediocre, and exactly
  the kind of thing the eval harness needs to exist to tune properly.

---

## 4. The scoring formula, term by term

```
score(i,u) = src(i) · [ w_sem(i)·sem(i) + w_pop(i)·pop(i) + w_cf(i)·cf(i,u) ]
```

Three signals, kept **deliberately separate** because they answer different questions
and fail differently.

### Term 1 — `sem`: semantic similarity

Cosine similarity (single seed) or normalized RRF (multiple seeds). Range `[0,1]`.
Answers: *does this book resemble what was asked for?*

### Term 2 — `pop`: the popularity / engagement prior

Answers: *do readers in general receive this well?* A **global** property of the item.

```
bayes(i)      = [ (n·avg + C·m) / (n + C) − 1 ] / 4          C = 20
engagement(i) = log1p(likes + 3·completions) / log1p(P95)
pop(i)        = 0.6·bayes(i) + 0.4·engagement(i)
```

**Bayesian damping.** The naive approach — sort by average rating — puts a book with
one 5-star rating above a book with 500 ratings averaging 4.8. The fix is to blend
each book's average toward the **global mean** `m`, weighted by a prior `C`. With
`C = 20`, a book effectively starts with 20 imaginary average ratings that real
ratings must outvote:

| Ratings | Average | Damped | Normalized |
|---|---|---|---|
| 1 | 5.0 | 3.57 | **0.64** |
| 500 | 5.0 | 4.94 | **0.99** |

The `(x−1)/4` at the end just maps the 1–5 scale onto `[0,1]` so it can be blended.

**Log compression of engagement.** Raw counts are unusable: a book with 1,000 likes
isn't 100× better than one with 10. `log1p` compresses that, so the difference between
10 and 100 likes matters far more than between 1,000 and 1,090.

**Completions weigh 3× a like.** Finishing a book is the strongest signal the platform
emits. A like is one tap.

**Normalized by the 95th percentile, not the maximum.** One viral outlier would
otherwise compress every other book into the bottom of the range. Values above P95
clamp to 1.0, which is intentional — the difference between "extremely popular" and
"the single most popular" shouldn't drive ranking.

### Term 3 — `cf`: item-item collaborative filtering

This is the term most worth understanding properly, because it's routinely confused
with popularity.

**Collaborative filtering** means using *other people's behaviour* to make a
recommendation, rather than the content of the item. "Readers who liked X also liked
Y." It's collaborative because readers implicitly collaborate; it's filtering because
it narrows a catalog down.

Two classical flavours:

- **User-user**: find readers similar to you, recommend what they liked. Struggles as
  users grow (finding your neighbours is expensive) and with new users.
- **Item-item**: find items that co-occur with the items *you* liked. Item-item
  relationships are far more stable than people's, and precomputable. **This is what
  we use.**

The crucial distinction from popularity:

| | `pop` | `cf` |
|---|---|---|
| Property of | the **item** alone | the **pair** (item, reader) |
| Same for every reader? | Yes | No |
| Question | "do readers like this?" | "do readers *like me* like this?" |

A globally unpopular book can have very high `cf` for one specific reader — and that's
precisely the recommendation worth making. Collapsing the two would destroy exactly
the signal that makes recommendations personal.

**How we compute it.** First, per-pair similarity from co-occurrence:

```
sim(i,j) = cooc(i,j) / ( sqrt(n_i · n_j) + λ )        λ = 10
```

`cooc(i,j)` is how many readers engaged with both. Dividing by `sqrt(n_i · n_j)`
normalizes for popularity — otherwise every book would look "similar" to the most-read
book simply because everyone read it. This is cosine similarity over the co-occurrence
matrix.

**λ = 10 is shrinkage, and it's essential.** Without it, two obscure books read by
exactly one shared reader score `1/1 = 1.0` — a perfect similarity from a single
coincidence, outranking genuinely related pairs with hundreds of shared readers. λ
pulls sparse evidence toward zero:

| cooc | n_i | n_j | Raw | With λ=10 |
|---|---|---|---|---|
| 1 | 1 | 1 | 1.00 | **0.09** |
| 200 | 300 | 300 | 0.67 | **0.65** |

Then, per candidate, a weighted mean over the reader's seed books:

```
cf(i,u) = Σ w_j · sim(i,j) / Σ w_j
          j∈S_u             j∈S_u
```

A **mean, not a max**, so one strong co-occurrence can't carry a candidate on its own.
`w_j` is engagement strength: like `1.0`, rating ≥4 `1.0`, rating 3 `0.4`, completion
`0.8`, partial read `0.4 × scroll`. Books rated ≤2 are excluded *and* suppressed —
treating a rejected book as a taste anchor would recommend more of what the reader
rejected.

**CF currently contributes nothing.** `w_cf_ceiling = 0`. The table, the rebuild query
and the scoring term are all real and tested, but the platform has no impression or
click log — only binary likes, create-only ratings, and reading-progress *state*.
Co-occurrence from that alone is far too sparse to beat the popularity prior. Shipping
it as a live stub means enabling it later is a config `UPDATE` plus a rebuild, not a
rescoring rewrite.

### The cold-start ramp

The three terms can't be weighted equally, because for a new book two of them are
noise. So the weights depend on how much evidence the item has:

```
α(i) = 0                     when n_i < 5      →  100% semantic
α(i) = n_i / (n_i + 20)      otherwise

w_sem = 1 − α·(W_pop + W_cf)
w_pop = α · W_pop            W_pop = 0.25
w_cf  = α · W_cf             W_cf  = 0.00  (stub; 0.20 when enabled)
```

where `n_i = likes + ratings + completions`.

**The hard gate at 5** is what delivers "a new book is judged purely on content".
Below a handful of interactions the popularity estimate is noise, and a smooth curve
alone would still grant it a little weight.

**Saturating rather than linear** above the gate, because a linear ramp is most
aggressive exactly where the data is thinnest.

**`n₅₀ = 20` deliberately equals the Bayesian prior `C = 20`.** Both encode the same
belief: around 20 interactions is where an item's own numbers start to mean something.
Two mechanisms, one assumption, one number.

**Weights always sum to 1**, which keeps scores comparable across items with very
different interaction volumes — otherwise a well-known book would score higher merely
because more terms contributed. And semantic never drops below **0.55** even at full
behavioral weight, so vector search stays the ranker. `ScoringConfig` enforces that
ceiling, so a bad tune in the database can't invert the design.

Observed in production data with 250 synthetic readers:

| n_interactions | α | w_pop | effect on score |
|---|---|---|---|
| 0 | 0.00 | 0.000 | none — pure semantic |
| 5 | 0.20 | 0.050 | slight |
| 20 | 0.50 | 0.125 | moderate |
| 100 | 0.83 | 0.208 | maximum |

### No source multiplier

There used to be one. A CMU bootstrap corpus shared the catalog with TaleTribe
stories and was multiplied down as the real catalog grew. The corpus is gone —
every item is a published TaleTribe story — so the term, its two config knobs,
and the `off_platform` flag went with it.

The problem it solved has not gone away: a small catalog gives a thin shelf, and
an ad-hoc query naming a book TaleTribe does not host now has no anchor to match
against. That is a known cost of platform-only, not an oversight.

---

## 5. Order of operations, and why

```
retrieve → fuse → score → diversify → top-K
```

1. **Retrieve** per seed, concurrently. N separate queries rather than one `LATERAL`
   join: a LATERAL over N query vectors is one planner decision away from N sequential
   scans instead of N index probes — a 10ms query becoming 2s, with no error.
2. **Fuse** (multi-seed only). Must precede scoring, because the blend needs one
   unified candidate list.
3. **Score.** Needs the fused list to exist.
4. **Diversify.** Last, because MMR trades relevance for variety and can only know
   what it's trading once scoring is done.

Each step depends on the previous one's output existing. Reordering any pair breaks
something specific.
