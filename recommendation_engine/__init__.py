"""TaleTribe Recommendation Service.

A separate FastAPI service that ranks books with deterministic vector search
(pgvector/HNSW) and uses an LLM only to *explain* the ranking — never to rank.

Deliberately independent of the story agent: its own app factory, its own
settings, and its own datastore (Postgres, not Firestore). It reuses only
process-agnostic pieces from the parent repo — `rate_limit.PerUserRateLimiter`
and `embedding_provider` — so the 768-dim embedding
contract stays single-sourced.
"""

import sys
from pathlib import Path

# Put the repo root on sys.path at package-import time. `recommendation_engine.config`
# imports `embedding_provider` for the canonical embedding
# dimension, and that import happens before any app-factory code can adjust the
# path. `python -m recommendation_engine.server` from the repo root works without
# this; `python recommendation_engine/server.py` does not, because Python puts the
# *script's* directory on the path instead of the repo root.
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
