"""Recommendation service settings via pydantic-settings.

Mirrors the conventions in the repo-root `config.py`: `@field_validator(mode="before")`
for clamping hostile values, `@model_validator(mode="after")` for production
hard-fails, and derived `@property`s instead of extra fields. Instantiated once
inside `create_app()` so tests can monkeypatch env vars before creation.

No env prefix, matching the root Settings — field names map directly to
SCREAMING_SNAKE env vars (`recs_database_url` → `RECS_DATABASE_URL`), which
keeps shared vars like `GOOGLE_CLOUD_PROJECT` spelled the way the Firestore
Admin SDK and the rest of the repo already expect.
"""

import json
import logging
from typing import Optional, cast

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

# story-data's local database, which now holds the `recommendations` schema.
# Deliberately a real, working local DSN so `python -m recommendation_engine.server`
# runs with zero config once that stack is up — and deliberately rejected by the
# production validator below so it can never ship.
LOCAL_DEV_DSN = "postgresql://postgres:postgres@localhost:5433/story_data"

# Every vector we write and every HNSW index must agree on this. Sourced from
# the shared embedding provider rather than redeclared, so there is exactly one
# definition of the dimension across both services.
from embedding_provider import (  # noqa: E402
    EXPECTED_EMBEDDING_DIM,
)

SCHEMA_NAME = "recommendations"


class RecSettings(BaseSettings):
    # ── Datastore ────────────────────────────────────────────────────────
    # RW is used by ingest and sync; RO by request serving. In local dev they
    # are the same DSN; in production RO points at a dedicated read-only
    # compute so vector scans never share compute with the product API. Neither
    # migrates: story-data owns the schema.
    recs_database_url: str = LOCAL_DEV_DSN
    recs_database_url_ro: str = ""  # falls back to recs_database_url (see property)
    recs_db_pool_min: int = 1
    recs_db_pool_max: int = 10

    # ── Runtime environment ──────────────────────────────────────────────
    environment: str = "development"
    google_cloud_project: str = ""  # required only for the Firestore sync path
    firestore_emulator_host: str = ""
    port: int = 8100

    # ── OIDC / service-to-service auth (required in production) ──────────
    recs_service_url: str = ""
    firebase_functions_service_account: str = ""
    allowed_service_accounts: str = ""  # comma-separated

    # ── CORS (JSON array string) ─────────────────────────────────────────
    cors_origins: str = "[]"

    # ── Rate limiting ────────────────────────────────────────────────────
    # Ranking is cheap (one DB round trip). The second bucket guards the paths
    # that spend LLM tokens — HyDE and explanations — which are not credit-metered.
    max_requests_per_minute_per_user: int = 30
    max_llm_requests_per_minute_per_user: int = 6

    # ── LLM / embeddings (direct Gemini; creditProxy is not in this path) ──
    google_ai_studio_api_key: str = ""
    recs_gemini_model: str = "gemini-2.5-flash-lite"
    recs_explanation_max_output_tokens: int = 256
    recs_hyde_max_output_tokens: int = 512

    # ── Retrieval knobs ──────────────────────────────────────────────────
    # ef_search must be >= 2x the candidate pool for usable recall; see the
    # sweep in the eval harness before changing it.
    recs_hnsw_ef_search: int = 100
    recs_hnsw_max_scan_tuples: int = 20000
    recs_statement_timeout_ms: int = 2000
    recs_candidate_pool: int = 50  # rows scored before MMR selects the final K
    recs_default_top_k: int = 10

    # ── Feature flags ────────────────────────────────────────────────────
    recs_enable_hyde: bool = True
    recs_enable_explanations: bool = True

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    @field_validator(
        "max_requests_per_minute_per_user",
        "max_llm_requests_per_minute_per_user",
        mode="before",
    )
    @classmethod
    def clamp_rpm(cls, v: object) -> int:
        """Never negative — a negative bucket would read as 'unlimited'."""
        try:
            return max(0, int(v))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0

    @field_validator("recs_hnsw_ef_search", mode="before")
    @classmethod
    def clamp_ef_search(cls, v: object) -> int:
        """pgvector rejects ef_search < 1; a too-small value silently guts recall."""
        try:
            return max(1, int(v))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 100

    @field_validator("recs_candidate_pool", "recs_default_top_k", mode="before")
    @classmethod
    def clamp_positive(cls, v: object) -> int:
        try:
            return max(1, int(v))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 1

    @field_validator("cors_origins", mode="before")
    @classmethod
    def validate_cors(cls, v: object) -> str:
        """Ensure cors_origins is a valid JSON list; fall back to '[]' (logged)."""
        raw = str(v) if v is not None else "[]"
        try:
            parsed = cast(object, json.loads(raw))
            if not isinstance(parsed, list):
                logger.warning(
                    "cors_origins_invalid_shape: CORS_ORIGINS=%r is not a JSON list; "
                    "browser requests will be blocked",
                    raw,
                )
                return "[]"
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "cors_origins_parse_failed: CORS_ORIGINS=%r is not valid JSON (%s); "
                "browser requests will be blocked",
                raw,
                exc,
            )
            return "[]"
        return raw

    @model_validator(mode="after")
    def warn_ef_search_below_pool(self) -> "RecSettings":
        """ef_search below the candidate pool means the index cannot return a full
        page — recall drops with no error anywhere. Warn loudly rather than
        silently truncating results."""
        if self.recs_hnsw_ef_search < 2 * self.recs_candidate_pool:
            logger.warning(
                "ef_search_too_low: RECS_HNSW_EF_SEARCH=%d is below 2x "
                "RECS_CANDIDATE_POOL=%d; HNSW recall will degrade silently",
                self.recs_hnsw_ef_search,
                self.recs_candidate_pool,
            )
        return self

    @model_validator(mode="after")
    def check_production_fields(self) -> "RecSettings":
        if self.environment != "production":
            return self

        if not self.recs_service_url.strip():
            raise ValueError(
                "RECS_SERVICE_URL must be set when ENVIRONMENT=production "
                "(OIDC token audience for Firebase Functions → recommendations calls)"
            )
        if (
            not self.firebase_functions_service_account.strip()
            and not self.allowed_service_accounts.strip()
        ):
            raise ValueError(
                "FIREBASE_FUNCTIONS_SERVICE_ACCOUNT or ALLOWED_SERVICE_ACCOUNTS must "
                "be set when ENVIRONMENT=production (trusted OIDC caller allowlist)"
            )
        if self.recs_database_url.strip() == LOCAL_DEV_DSN:
            raise ValueError(
                "RECS_DATABASE_URL is still the local story-data default; set a "
                "real DSN when ENVIRONMENT=production"
            )
        if not self.google_ai_studio_api_key.strip():
            raise ValueError(
                "GOOGLE_AI_STUDIO_API_KEY must be set when ENVIRONMENT=production "
                "(embeddings and explanations call Gemini directly)"
            )
        return self

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    @property
    def oidc_audience(self) -> Optional[str]:
        """OIDC audience for service-to-service auth; None outside production,
        which is what makes local dev work with no credentials."""
        if self.environment != "production":
            return None
        return self.recs_service_url.strip().rstrip("/") or None

    @property
    def allowed_callers(self) -> frozenset[str]:
        """Frozenset of trusted caller service account emails."""
        if self.environment != "production":
            return frozenset()
        raw_list = self.allowed_service_accounts.strip()
        if raw_list:
            return frozenset(
                part.strip() for part in raw_list.split(",") if part.strip()
            )
        single = self.firebase_functions_service_account.strip()
        if single:
            return frozenset({single})
        return frozenset()

    @property
    def read_dsn(self) -> str:
        """DSN for request serving. Falls back to the RW DSN in local dev, where
        there is only one compute."""
        return self.recs_database_url_ro.strip() or self.recs_database_url.strip()

    @property
    def write_dsn(self) -> str:
        """DSN for migrations, ingest and sync."""
        return self.recs_database_url.strip()

    @property
    def embedding_dimension(self) -> int:
        return EXPECTED_EMBEDDING_DIM

    @property
    def parsed_cors_origins(self) -> list[str]:
        try:
            parsed = cast(object, json.loads(self.cors_origins))
        except (json.JSONDecodeError, ValueError):
            return []
        if not isinstance(parsed, list):
            return []
        return [str(origin) for origin in parsed]
