"""HTTP server for the TaleTribe recommendation service.

Deliberately a separate app from `server.py` at the repo root: this workload has
its own datastore (Postgres, not Firestore), its own scaling profile (vector
scans, not long LLM generations), and its own audience (readers, not story
owners). It shares the *conventions* of the root app — `create_app()` factory,
`Settings()` built inside it so tests can monkeypatch env first, structlog JSON
output, singletons on `app.state`, and a stable `{success, data, error}` envelope
— but none of its state.

Run locally:
    docker compose -f recommendation_engine/docker-compose.yml up -d
    python -m recommendation_engine.migrations.migrate
    python -m recommendation_engine.server
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token
from pydantic import BaseModel

from embedding_provider import (
    get_embedding_provider,
    verify_embedding_dimension,
)
from rate_limit import PerUserRateLimiter
from recommendation_engine.config import RecSettings
from recommendation_engine.db import Database
from recommendation_engine.embeddings import QueryEmbedder
from recommendation_engine.env import REPO_ROOT, load_env
from recommendation_engine.explain import ExplanationCache
from recommendation_engine.llm import build_client
from recommendation_engine.retrieval import Retriever
from recommendation_engine.routes import build_router

logger = structlog.get_logger(__name__)


def _configure_environment() -> Path:
    """Load the repo-root .env so local runs pick up RECS_* and API keys."""
    load_env()
    return REPO_ROOT


def _configure_logging() -> None:
    """structlog with JSON output, matching the root service so both stream into
    the same log-based metrics."""
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(),
    )
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )


async def verify_internal_token(request: Request) -> None:
    """Verify the caller is Firebase Functions, via Google OIDC.

    No-op outside production, which is what lets local development run with no
    credentials at all. In production: validates signature, expiry and audience
    (RECS_SERVICE_URL), then requires the token's email claim to be on the
    configured allowlist.
    """
    audience: Optional[str] = request.app.state.oidc_audience
    allowed_callers: frozenset = request.app.state.allowed_callers
    if not audience:
        return

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail={
                "code": "UNAUTHORIZED",
                "message": "Missing Authorization header",
                "details": None,
            },
        )

    token = auth_header.split(" ", 1)[1]
    try:
        claims = google_id_token.verify_oauth2_token(
            token,
            request.app.state.google_auth_request,  # cached — not created per call
            audience=audience,
        )
        caller_email = claims.get("email")
        if caller_email not in allowed_callers:
            raise ValueError(f"Unexpected caller email: {caller_email}")
    except Exception as exc:
        logger.warning("token_validation_failed", error=str(exc))
        raise HTTPException(
            status_code=401,
            detail={
                "code": "UNAUTHORIZED",
                "message": "Invalid or unauthorized token",
                "details": None,
            },
        )


class ErrorDetail(BaseModel):
    """Stable error payload returned to API clients."""

    code: str
    message: str
    details: Optional[Any] = None


class RecResponse(BaseModel):
    """Response envelope, matching the root service so the Firebase Functions
    bridge can treat both services identically."""

    success: bool
    data: Optional[Any] = None
    error: Optional[ErrorDetail] = None


def create_app() -> FastAPI:
    """Create and configure the recommendation service app."""
    _configure_logging()
    _configure_environment()

    # Instantiated here, not at module level, so tests can monkeypatch env vars
    # before the app is built.
    settings = RecSettings()

    if settings.environment != "production" and not settings.firestore_emulator_host:
        # Only the platform-catalog sync path touches Firestore; pointing at the
        # emulator by default keeps local runs off real GCP.
        os.environ.setdefault("FIRESTORE_EMULATOR_HOST", "localhost:8080")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # The pool is the one piece of startup that can fail for an operational
        # reason (DB down, wrong DSN). Fail loudly here rather than on the first
        # request: Cloud Run's startup probe then reports the deploy as broken.
        await app.state.db.connect()
        try:
            yield
        finally:
            await app.state.db.aclose()
            embedder = getattr(app.state, "embedder", None)
            if embedder is not None and hasattr(embedder, "aclose"):
                await embedder.aclose()
            llm = getattr(app.state, "llm", None)
            if llm is not None:
                await llm.aclose()

    app = FastAPI(
        title="TaleTribe Recommendation Service",
        description=(
            "Personalized book recommendations: deterministic pgvector ranking "
            "with an LLM explanation layer."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.parsed_cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Authorization", "X-Firebase-Token"],
    )

    app.state.settings = settings
    app.state.oidc_audience = settings.oidc_audience
    app.state.allowed_callers = settings.allowed_callers
    app.state.google_auth_request = google_requests.Request()

    if app.state.oidc_audience:
        logger.info("oidc_audience_set", audience=app.state.oidc_audience)
    if app.state.allowed_callers:
        logger.info(
            "oidc_allowed_callers",
            count=len(app.state.allowed_callers),
            callers=sorted(app.state.allowed_callers),
        )

    app.state.db = Database(
        write_dsn=settings.write_dsn,
        read_dsn=settings.read_dsn,
        pool_min=settings.recs_db_pool_min,
        pool_max=settings.recs_db_pool_max,
    )

    # Two buckets. Ranking is one DB round trip and can be generous; HyDE and
    # explanations spend Gemini tokens on a path that deliberately bypasses
    # creditProxy, so the cache and this bucket are the only cost controls.
    app.state.rate_limiter = PerUserRateLimiter(
        settings.max_requests_per_minute_per_user
    )
    app.state.llm_rate_limiter = PerUserRateLimiter(
        settings.max_llm_requests_per_minute_per_user
    )

    # Shared with the story agent so the 768-dim contract has one definition.
    # A dimension mismatch here means every vector written is unqueryable
    # against the HNSW index, so it is asserted at startup, not at query time.
    app.state.embedder = get_embedding_provider(settings.google_ai_studio_api_key)
    verify_embedding_dimension(app.state.embedder)
    # Wraps the provider with RETRIEVAL_QUERY/RETRIEVAL_DOCUMENT asymmetry and an
    # LRU over queries — ad-hoc prompts repeat heavily across users, and each hit
    # removes an 80-250ms round trip and a token charge from the request path.
    app.state.query_embedder = QueryEmbedder(app.state.embedder)

    app.state.retriever = Retriever(
        app.state.db,
        ef_search=settings.recs_hnsw_ef_search,
        max_scan_tuples=settings.recs_hnsw_max_scan_tuples,
        statement_timeout_ms=settings.recs_statement_timeout_ms,
    )

    # Direct Gemini, not creditProxy — required for token-by-token streaming and
    # structured output. None when no key is configured, in which case ranking
    # still works and only HyDE/explanations are unavailable.
    app.state.llm = (
        build_client(settings.google_ai_studio_api_key, settings.recs_gemini_model)
        if settings.recs_enable_explanations or settings.recs_enable_hyde
        else None
    )
    app.state.explanation_cache = ExplanationCache(app.state.db)

    app.include_router(build_router(verify_internal_token), tags=["Recommendations"])

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(_: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content=RecResponse(
                success=False,
                error=ErrorDetail(
                    code="VALIDATION_ERROR",
                    message="Invalid request",
                    details=exc.errors(),
                ),
            ).model_dump(),
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(_: Request, exc: HTTPException):
        detail = exc.detail
        if isinstance(detail, dict) and {"code", "message"}.issubset(detail.keys()):
            error = ErrorDetail(**detail)
        else:
            error = ErrorDetail(code="HTTP_ERROR", message=str(detail))
        return JSONResponse(
            status_code=exc.status_code,
            content=RecResponse(success=False, error=error).model_dump(),
        )

    @app.get("/health")
    async def health_check(response: JSONResponse = None):  # noqa: ARG001
        """Assert every precondition that would otherwise fail *silently*.

        A missing HNSW index, a pgvector older than 0.8.0, or an embedder at the
        wrong dimension all produce empty or degraded results rather than errors,
        so each is checked explicitly and reflected in the status code.
        """
        embedder = app.state.embedder
        embed_dim = embedder.dimension if embedder is not None else None
        expected_dim = settings.embedding_dimension

        payload: dict = {
            "status": "ok",
            "environment": settings.environment,
            "embedder": type(embedder).__name__ if embedder else None,
            "embedding_dimension": embed_dim,
            "embedding_dimension_ok": embed_dim == expected_dim,
            "query_cache": app.state.query_embedder.stats(),
            "database": {"connected": False},
        }

        try:
            payload["database"] = await app.state.db.health()
        except Exception as exc:
            logger.error("health_db_check_failed", error=str(exc))
            payload["database"] = {"connected": False, "error": str(exc)}

        db_state = payload["database"]
        degraded = (
            not db_state.get("connected")
            or not db_state.get("pgvector_ok")
            or not db_state.get("hnsw_index_present")
            or not payload["embedding_dimension_ok"]
        )
        if degraded:
            payload["status"] = "degraded"
            return JSONResponse(status_code=503, content=payload)
        return payload

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", str(app.state.settings.port)))
    uvicorn.run(app, host="0.0.0.0", port=port)
