"""Typed view of the services stored on FastAPI's dynamic ``app.state``.

Starlette intentionally exposes application state as a dynamic attribute bag.
That is convenient at runtime, but it means a type checker sees every service
resolved through ``request.app.state`` as ``Any``.  Keep the dynamic boundary in
one place and give the rest of the service a precise view of it.
"""

from typing import cast

from fastapi import FastAPI, Request
from google.auth.transport import requests as google_requests

from embedding_provider import EmbeddingProvider
from rate_limit import PerUserRateLimiter
from recommendation_engine.config import RecSettings
from recommendation_engine.db import Database
from recommendation_engine.embeddings import QueryEmbedder
from recommendation_engine.explain import ExplanationCache
from recommendation_engine.llm import GeminiClient
from recommendation_engine.retrieval import Retriever
from recommendation_engine.usage import DailyLlmMeter


class RecommendationAppState:
    """Services installed by :func:`recommendation_engine.server.create_app`."""

    settings: RecSettings
    oidc_audience: str | None
    allowed_callers: frozenset[str]
    google_auth_request: google_requests.Request
    db: Database
    rate_limiter: PerUserRateLimiter
    llm_rate_limiter: PerUserRateLimiter
    llm_meter: DailyLlmMeter
    embedder: EmbeddingProvider | None
    query_embedder: QueryEmbedder
    retriever: Retriever
    llm: GeminiClient | None
    explanation_cache: ExplanationCache


def recommendation_state(request: Request) -> RecommendationAppState:
    """Return the request's dynamic application state with its installed type."""

    app = cast(FastAPI, request.scope["app"])
    return recommendation_app_state(app)


def recommendation_app_state(app: FastAPI) -> RecommendationAppState:
    """Return a FastAPI application's state with its installed service type."""

    return cast(RecommendationAppState, app.state)
