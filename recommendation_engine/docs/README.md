# TaleTribe Recommendation Service — documentation

A microservice that recommends published TaleTribe stories by **deterministic vector
search**, using an LLM only to *explain* the ranking — never to produce it.

Written for two audiences: engineers learning how a recommender is actually built,
and product understanding what the service does and doesn't do.

## Start here

| If you want | Read | Length |
|---|---|---|
| What this is, in plain language | **[overview.md](overview.md)** | short |
| **Something is broken and you need to fix it** | **[runbook.md](runbook.md)** | reference |
| How embeddings, indexing, CF and every formula work | **[concepts.md](concepts.md)** | long, the main one |
| How the pieces fit, and why pgvector | [architecture.md](architecture.md) | medium |
| Every endpoint, with a trace of what happens inside | [api.md](api.md) | medium |
| The three background jobs, and what breaks without them | [jobs.md](jobs.md) | medium |
| How the browser and Firebase Functions call this | [frontend-integration.md](frontend-integration.md) | medium |
| Database roles, the privacy boundary, threats | [security-and-roles.md](security-and-roles.md) | medium |
| What the 431 tests cover, and what they don't | [testing.md](testing.md) | reference |
| How a story gets embedded; how personas work | [data-lifecycle.md](data-lifecycle.md) | long |
| What each Python file is for | [file-reference.md](file-reference.md) | reference |
| Commands that work, and what things cost | [running-locally.md](running-locally.md) | reference |
| How to deploy it: services, security, scale | [deployment.md](deployment.md) | long, not built |
| How you would measure whether it works | [evaluation.md](evaluation.md) | long, not built |
| Every bug we hit, and how each was found | [development-log.md](development-log.md) | long, historical |
| Why the schema lives in story-data | [migration-to-story-data.md](migration-to-story-data.md) | long, historical |

**Product**: [overview.md](overview.md) is written for you. The "Things product should
know" section at the end has the caveats that affect UI decisions — especially cold
start, which got *harder*, not easier, when the seed corpus was removed.

**New engineer**: [overview.md](overview.md) → [concepts.md](concepts.md) →
[running-locally.md](running-locally.md), then poke at it with `query_cli`.

**On call**: [runbook.md](runbook.md), which is symptom-first and assumes nothing.

**Learning how recommenders work**: [concepts.md](concepts.md) is the substance.
[development-log.md](development-log.md) is where the real lessons are, because it
records what went *wrong*.

## The design in five sentences

Every story is converted into 768 numbers that position it in a space where similar
stories sit near each other; a reader's request is converted the same way and the
database finds the nearest ones. Those candidates are scored on three deliberately
separate signals — content similarity, a volume-damped popularity prior, and
item-item collaborative filtering — with weights that ramp up only as a story
accumulates real reader evidence, so a brand-new story is judged purely on content
and never penalised for being new. Results are then diversified so a shelf isn't ten
near-identical stories. An LLM is handed the finished list and writes one sentence per
story about why the reader might like it. Everything expensive or privileged happens
in three background jobs, so serving a request is one vector query.

## Where things live

This service is one of three moving parts, and only the first is in this repo:

| Part | Repo | Owns |
|---|---|---|
| The service and its ranking | **taleTribe-recs** | `recommendations` schema contents, the HTTP API, the catalog ingest |
| The schema and the reader signals | **story-data** | `migrations/000019`, `000020`, and `sync-recs` |
| The caller | **taleTribe-frontend** | The two Firebase Functions and the UI surfaces |

The `recommendations` schema lives **inside story-data's database** and story-data
migrates it; this service has no migration runner. That is deliberate —
[migration-to-story-data.md](migration-to-story-data.md) has the reasoning, and
[security-and-roles.md](security-and-roles.md) has the boundary that makes sharing a
database safe.

## Current state

**Working:** all five endpoints, both recommendation modes, HyDE, explanations
(batched and streamed), the full three-term scoring formula, the platform catalog
ingest, synthetic reader signals, and the story-data signals derivation. 431 tests.

**Built but never run against real traffic:** the signals pipeline. Nothing schedules
the three jobs, so `n_interactions` is 0 everywhere, the cold-start ramp holds α at 0,
and **every recommendation today is 100% content similarity**. The popularity and CF
terms are implemented, tested, and completely inert.

**Not built:** scheduling, the evaluation harness, and deployment. This is the only
backend repo with no `Dockerfile`, `terraform/` or `.github/workflows/`.

**Costs:** an incremental ingest over an unchanged catalog is **free** — unchanged
rows are never re-embedded. First ingest is ~$0.0002 per story. The uncapped paths are
HyDE and explanations, which bypass creditProxy and are not credit-metered; see
[running-locally.md](running-locally.md#what-things-cost).

## Before merging and deploying

[runbook.md](runbook.md#known-issues-before-deploy) lists eight known issues. The one
that will bite first: **`recs_service` has no `SELECT` on `public`, but the ingest
reads `stories`** — so the first production ingest run fails on permission denied.

## If you read one section

[Silent failures](development-log.md#category-1-silent-failures) — five bugs that
raised **no error at all** and just quietly made results worse. In a retrieval system
that's the dominant failure mode, and it shaped how the whole thing is built,
instrumented and tested. It is also why [runbook.md](runbook.md) is organised the way
it is.
