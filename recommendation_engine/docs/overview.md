# Overview — what this service is, for anyone

Written for product, and as the gentlest entry point for engineers. No prior
knowledge of vector search assumed.

## The problem

TaleTribe had no recommendations. Reader-side discovery was **pure recency**: the
browse page listed published stories newest-first, optionally filtered to one
category. `RecommendedReads.tsx` existed as a five-line placeholder. Every AI
feature on the platform was author-side and gated on *owning* the story, so none of
it could be pointed at readers.

## What it does

Two things a reader can ask for:

1. **"Based on what I've read"** — recommendations from their own history.
2. **"I liked these, find me more"** — from a list of books they name, or from a
   free-text description like *"a cozy small-town mystery where nobody dies"*.

And for each result, a one-sentence **"why you'll like this"**, written by an LLM
and streamed in as it's composed.

## The one design decision that matters most

**The LLM does not choose the recommendations. It only explains them.**

The choosing is done by maths over vectors — the same input always produces the same
output. The LLM is handed the finished list and asked to write a sentence about each.

This is worth understanding because it's counterintuitive: the obvious design is to
ask a model "what should this reader read next?". We deliberately don't, for four
reasons:

| | Vector search decides | LLM decides |
|---|---|---|
| Same question, same answer? | Always | No |
| Can we measure if it's good? | Yes, offline, against ground truth | Barely |
| Cost per recommendation | ~$0.00001 | ~100× more |
| Can it recommend a book that doesn't exist? | Impossible — results come from the catalog | Yes, and it will |

That last row is not hypothetical. A model asked to recommend books will invent
plausible titles. A vector search can only return rows that are in the table.

## How the ranking works, in one paragraph

Every book gets converted into a list of 768 numbers that positions it in a
"meaning space" — books about similar things end up near each other. A reader's
request is converted the same way, and we ask the database for the nearest books.
Those candidates are then scored on three separate signals — **how well they match**,
**how well readers generally receive them**, and **whether readers with this
reader's taste liked them** — and finally shuffled slightly so the shelf isn't ten
near-identical books.

## What it's built on, and the state of it

- **1,987 books** currently loaded, each with real premise/theme/tone metadata
  derived by an LLM, and a real embedding.
- Those come from the **CMU Book Summary Corpus** — 16,559 public-domain-ish book
  summaries used as *cold-start scaffolding*. It's a stopgap: the service
  automatically de-emphasises them as real TaleTribe stories are added, and they're
  flagged `off_platform` so the UI can say "not on TaleTribe yet".
- **Postgres + pgvector** for storage and search, not a dedicated vector database.
- Total cost to load 2,000 books: **$0.33**. The full 16.5k corpus projects to ~$2.40.
  Per-recommendation cost is a rounding error.

### What works today

All five API endpoints, both recommendation modes, HyDE for free-text queries,
explanations both batched and streamed, and the full three-term scoring formula.
1,025 automated tests.

### What is honestly not done

- **Real reader data isn't connected.** Behavioral recommendations currently run on
  *synthetic* readers we generate. The Firestore connection is designed for and
  scaffolded, but not written.
- **Quality is unmeasured.** There's no evaluation harness, so "are these good
  recommendations?" is currently answered by reading the output and forming an opinion.
  Every tuning decision is a guess. **Deliberately deferred until there are real
  readers** — metrics computed on synthetic data measure whether the recommender
  recovers the generator's own assumptions, which is circular while producing
  confident-looking decimals. The full plan is written up in
  [evaluation.md](evaluation.md) so it can be picked up rather than re-derived.
- **Nothing is deployed.** It runs locally only, by choice, so the design could
  settle before infrastructure was committed to.

## Things product should know

**CMU books are not readable on TaleTribe.** They exist so the recommender has
enough of a "taste space" to be useful on day one, and so a reader can say "I liked
Dune" about a book we'll never host. Any surface showing them needs to mark them
clearly, or a reader taps through to nothing. The API returns `off_platform: true`
on every such item.

**Explanations are not credit-metered.** They call Gemini directly rather than going
through creditProxy, because that's the only way to get true token-by-token
streaming and cancellation. The cost controls are a deterministic cache (the same
question about the same book is free forever) and a tight per-user rate limit (6/min).

**Cold-start is handled explicitly, not accidentally.** A brand-new story with zero
readers is scored **100% on content similarity**. It is never penalised for being
new. Popularity only begins to matter once a book has at least five real
interactions, and even at maximum it can only move a score by a quarter.

**The seed corpus is licensed CC BY-SA.** LLM-derived premises are plausibly
derivative works. If premise text becomes user-visible, attribution needs settling.

## Where to go next

| You want | Read |
|---|---|
| How embeddings, indexing, CF and the formulas actually work | [concepts.md](concepts.md) |
| How the pieces fit, and why pgvector | [architecture.md](architecture.md) |
| Every endpoint, with a trace of what happens inside | [api.md](api.md) |
| What happens when a story is added; how seed data works | [data-lifecycle.md](data-lifecycle.md) |
| What each Python file is for | [file-reference.md](file-reference.md) |
| Commands that work, and what things cost | [running-locally.md](running-locally.md) |
| How it would be deployed, secured and scaled | [deployment.md](deployment.md) |
| How quality would be measured, and why it isn't yet | [evaluation.md](evaluation.md) |
| Every bug we hit and how it was found | [development-log.md](development-log.md) |
