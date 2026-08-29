# taleTribe-recs

Standalone recommendation service for TaleTribe: pgvector retrieval, HyDE,
RRF fusion, MMR diversification, and LLM-generated explanations.

Split out of `taleTribe-agents/recommendation_engine` to keep its Postgres/
pgvector workload and scaling profile independent from the Firestore-backed
agents service. The database is no longer separate: the `recommendations`
schema lives in story-data's Postgres and story-data migrates it.

`embedding_provider.py` and `rate_limit.py` at the repo root are duplicated
from `taleTribe-agents` (not a shared package) — see that repo if the two
copies need to be reconciled, particularly `EXPECTED_EMBEDDING_DIM`, which
every vector this service writes must agree with.

The catalog is published TaleTribe stories only, ingested with
`python -m recommendation_engine.ingest.platform`. Reader signals are derived by
story-data (`go run ./cmd/api sync-recs`), not by this service — it is not
permitted to read `reading_progress`.

`recommendation_engine/docs/README.md` indexes the full documentation set.
Quick links: `running-locally.md` for setup and the HTTP API, `runbook.md` when
something is broken, `jobs.md` for the three background jobs,
`security-and-roles.md` for the privacy boundary, and
`migration-to-story-data.md` for why the database is shared.
