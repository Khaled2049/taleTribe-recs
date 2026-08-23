# taleTribe-recs

Standalone recommendation service for TaleTribe: pgvector retrieval, HyDE,
RRF fusion, MMR diversification, and LLM-generated explanations.

Split out of `taleTribe-agents/recommendation_engine` to keep its Postgres/
pgvector workload, migrations, and scaling profile independent from the
Firestore-backed agents service.

`embedding_provider.py` and `rate_limit.py` at the repo root are duplicated
from `taleTribe-agents` (not a shared package) — see that repo if the two
copies need to be reconciled, particularly `EXPECTED_EMBEDDING_DIM`, which
every vector this service writes must agree with.

See `recommendation_engine/docs/running-locally.md` for setup, the HTTP API,
loading the catalog, and cost notes.
