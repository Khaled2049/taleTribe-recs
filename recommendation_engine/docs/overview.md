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

- **Published TaleTribe stories only.** Each catalog row carries LLM-derived
  premise/theme/tone metadata and an embedding, and keys back to the story by
  `story_id`. A CMU bootstrap corpus used to share this catalog as cold-start
  scaffolding; it has been removed.
- **Postgres + pgvector** for storage and search, not a dedicated vector database.
  The `recommendations` schema lives in story-data's database.
- Per-recommendation cost is a rounding error; the spend is in ingest, which
  embeds each story once and skips unchanged rows on re-run.

### What works today

All five API endpoints, both recommendation modes, HyDE for free-text queries,
explanations both batched and streamed, and the full three-term scoring formula.
455 automated tests.

### What is honestly not done

- **The real reader-signal pipeline has not run in production yet.** story-data now
  derives likes, ratings, and reading progress into the recommendations schema; the
  first ordered production load is still pending.
- **Quality is unmeasured.** There's no evaluation harness, so "are these good
  recommendations?" is currently answered by reading the output and forming an opinion.
  Every tuning decision is a guess. **Deliberately deferred until there are real
  readers** — metrics computed on synthetic data measure whether the recommender
  recovers the generator's own assumptions, which is circular while producing
  confident-looking decimals. The full plan is written up in
  [evaluation.md](evaluation.md) so it can be picked up rather than re-derived.
- **The production stack is implemented but not rolled out yet.** The container,
  Terraform, CI/CD, private Cloud Run service, ordered jobs, and paused scheduler
  are ready for the story-data-first rollout.

## Things product should know

**Every recommendation is a readable TaleTribe story.** That was not true while
the CMU corpus was in the catalog, and the API carried an `off_platform` flag so
a UI could badge the ones that were dead ends. Both are gone; nothing needs
badging now.

**Explanations do not use creditProxy.** They call Gemini directly, so recs applies
its own controls: a deterministic cache, a 6/min per-process burst limit, and
durable per-user and platform-wide daily ceilings stored in Postgres.

**Cold-start is handled explicitly, not accidentally.** A brand-new story with zero
readers is scored **100% on content similarity**. It is never penalised for being
new. Popularity only begins to matter once a book has at least five real
interactions, and even at maximum it can only move a score by a quarter.

**Cold start is a real, open product problem — and removing the seed corpus made it
sharper.** A CMU book-summary corpus used to pad the catalog so there was always
something to rank and something for the diversity pass to work with. It is gone,
because every recommendation should be a story a reader can actually open. The cost
is honest and worth stating to product:

- With a few dozen published stories there is **little to rank and little to
  diversify**. Expect result lists that look thin or repetitive, and a low
  `diversity` number that is reporting reality rather than malfunctioning.
- An ad-hoc query naming a book TaleTribe does not host — *"something like Dune"* —
  **has no anchor**. Seed titles are resolved against the catalog by fuzzy match, so
  an unhosted title resolves to nothing and comes back in `unresolved_books`. Free
  text still works (it is embedded directly, or expanded by HyDE first), so the UI
  should prefer prompts over title-matching while the catalog is small.
- The "For you" shelf silently becomes **"Popular on TaleTribe"** for any reader
  without enough signals. The frontend already switches its title and eyebrow text
  for this — that is the intended experience, not a fallback bug.

**None of the behavioral signals have ever run against real traffic.** The jobs that
derive likes/ratings/progress into the recommender exist and are tested, but nothing
schedules them yet. Until they run, every recommendation on the platform is **100%
content similarity** — popularity and collaborative filtering contribute exactly
nothing regardless of configuration.

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
| **Something is wrong and you need to diagnose it** | **[runbook.md](runbook.md)** |
| The three background jobs and what breaks without them | [jobs.md](jobs.md) |
| How the browser and Firebase Functions call this | [frontend-integration.md](frontend-integration.md) |
| Database roles, the privacy boundary, and threats | [security-and-roles.md](security-and-roles.md) |
| What the 455 tests cover, and what they don't | [testing.md](testing.md) |
