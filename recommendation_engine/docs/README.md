# TaleTribe Recommendation Service — documentation

A microservice that recommends books by **deterministic vector search**, using an LLM
only to *explain* the ranking — never to produce it.

Written to be read by two audiences: engineers learning how a recommender is actually
built, and product understanding what the service does and doesn't do.

## Start here

| If you want | Read | Length |
|---|---|---|
| What this is, in plain language | **[overview.md](overview.md)** | short |
| How embeddings, indexing, CF and every formula work | **[concepts.md](concepts.md)** | long, the main one |
| How the pieces fit, and why pgvector over alternatives | [architecture.md](architecture.md) | medium |
| Every endpoint, with a trace of what happens inside | [api.md](api.md) | medium |
| How a book gets embedded; how seed data and personas work | [data-lifecycle.md](data-lifecycle.md) | long |
| What each Python file is for | [file-reference.md](file-reference.md) | reference |
| Commands that work, and what things cost | [running-locally.md](running-locally.md) | reference |
| How to deploy it: services, security, scale | [deployment.md](deployment.md) | long, not implemented |
| How you would measure whether it works | [evaluation.md](evaluation.md) | long, not implemented |
| Every bug we hit, and how each was found | [development-log.md](development-log.md) | long |

**Product**: [overview.md](overview.md) is written for you. The "Things product should
know" section at the end has the four caveats that affect UI decisions.

**New engineer**: [overview.md](overview.md) → [concepts.md](concepts.md) →
[running-locally.md](running-locally.md), then poke at it with `query_cli`.

**Learning how recommenders work**: [concepts.md](concepts.md) is the substance.
[development-log.md](development-log.md) is where the real lessons are, because it
records what went *wrong*.

## The design in five sentences

Every book is converted into 768 numbers that position it in a space where similar books
sit near each other; a reader's request is converted the same way and the database finds
the nearest books. Those candidates are scored on three deliberately separate signals —
content similarity, a volume-damped popularity prior, and item-item collaborative
filtering — with weights that ramp up only as a book accumulates real reader evidence, so
a brand-new story is judged purely on content and never penalised for being new.
Results are then diversified so a shelf isn't ten near-identical books. An LLM is handed
the finished list and writes one sentence per book about why the reader might like it,
streamed and cancellable. The seed corpus that makes this useful on day one
de-emphasises itself automatically as real stories arrive.

## Current state

**Working:** all five endpoints, both recommendation modes, HyDE, explanations (batched
and streamed), the full three-term formula, 1,987 books with real embeddings and
LLM-derived metadata, synthetic reader signals, 1,025 tests.

**Not done:** the Firestore connection (real reader data), the evaluation harness, and
deployment. All three deliberately, in that order of priority.

**Costs:** $0.33 loaded 2,000 books. Full 16.5k corpus projects to ~$2.40. Local
infrastructure is free.

## If you read one section

[Silent failures](development-log.md#category-1-silent-failures) — five bugs that raised
**no error at all** and just quietly made results worse. In a retrieval system that's the
dominant failure mode, and it shaped how the whole thing is built, instrumented and
tested.
