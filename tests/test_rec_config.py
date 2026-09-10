"""Unit tests for recommendation service settings.

Follows the repo convention: env vars set at import time before the module under
test is imported, `pytestmark` at module level, no conftest.
"""

import pytest

from recommendation_engine.config import (  # noqa: E402
    LOCAL_DEV_DSN,
    RecSettings,
)

pytestmark = pytest.mark.unit


# ── Clamping: hostile values must not read as "unlimited" or "disabled" ──


@pytest.mark.parametrize(
    "raw,expected",
    [("-5", 0), ("0", 0), ("30", 30), ("not-a-number", 0), (None, 0)],
)
def test_rpm_clamped_to_non_negative(raw, expected):
    settings = RecSettings(max_requests_per_minute_per_user=raw)
    assert settings.max_requests_per_minute_per_user == expected


def test_ef_search_never_below_one():
    """pgvector rejects ef_search < 1, and a 0 would silently gut recall."""
    assert RecSettings(recs_hnsw_ef_search=0).recs_hnsw_ef_search == 1
    assert RecSettings(recs_hnsw_ef_search=-10).recs_hnsw_ef_search == 1
    assert RecSettings(recs_hnsw_ef_search="garbage").recs_hnsw_ef_search == 100


def test_candidate_pool_and_top_k_never_zero():
    settings = RecSettings(recs_candidate_pool=0, recs_default_top_k=-3)
    assert settings.recs_candidate_pool == 1
    assert settings.recs_default_top_k == 1


# ── CORS parsing degrades to a safe empty list, never a crash ────────────


def test_cors_origins_parsed():
    settings = RecSettings(cors_origins='["https://taletribe.app"]')
    assert settings.parsed_cors_origins == ["https://taletribe.app"]


@pytest.mark.parametrize("raw", ["not json", '{"a": 1}', "", "42"])
def test_cors_origins_invalid_falls_back_to_empty(raw):
    assert RecSettings(cors_origins=raw).parsed_cors_origins == []


# ── OIDC is a no-op outside production; that's what makes local dev work ──


def test_oidc_audience_none_outside_production():
    settings = RecSettings(
        environment="development", recs_service_url="https://recs.example.com"
    )
    assert settings.oidc_audience is None
    assert settings.allowed_callers == frozenset()


def test_oidc_audience_set_in_production():
    settings = RecSettings(
        environment="production",
        recs_service_url="https://recs.example.com/",
        allowed_service_accounts="fn@proj.iam.gserviceaccount.com",
        recs_database_url="postgresql://user:pw@prod-host/recs",
        google_ai_studio_api_key="key",
    )
    # Trailing slash stripped so it matches the audience Functions mints.
    assert settings.oidc_audience == "https://recs.example.com"


def test_allowed_callers_parses_comma_list_and_trims():
    settings = RecSettings(
        environment="production",
        recs_service_url="https://recs.example.com",
        allowed_service_accounts=" a@x.iam.gserviceaccount.com , b@y.iam.gserviceaccount.com ,",
        recs_database_url="postgresql://user:pw@prod-host/recs",
        google_ai_studio_api_key="key",
    )
    assert settings.allowed_callers == frozenset(
        {"a@x.iam.gserviceaccount.com", "b@y.iam.gserviceaccount.com"}
    )


def test_allowed_callers_falls_back_to_single_account():
    settings = RecSettings(
        environment="production",
        recs_service_url="https://recs.example.com",
        firebase_functions_service_account="only@x.iam.gserviceaccount.com",
        recs_database_url="postgresql://user:pw@prod-host/recs",
        google_ai_studio_api_key="key",
    )
    assert settings.allowed_callers == frozenset({"only@x.iam.gserviceaccount.com"})


# ── Production hard-fails: each one is a config mistake that would otherwise
#    surface as a silent security or availability problem in production ──


def _prod_kwargs(**overrides):
    base = dict(
        environment="production",
        recs_service_url="https://recs.example.com",
        allowed_service_accounts="fn@proj.iam.gserviceaccount.com",
        recs_database_url="postgresql://user:pw@prod-host/recs",
        google_ai_studio_api_key="key",
    )
    base.update(overrides)
    return base


def test_production_requires_service_url(monkeypatch):
    monkeypatch.delenv("RECS_SERVICE_URL", raising=False)
    with pytest.raises(ValueError, match="RECS_SERVICE_URL"):
        RecSettings(**_prod_kwargs(recs_service_url=""))


def test_production_requires_caller_allowlist(monkeypatch):
    monkeypatch.delenv("FIREBASE_FUNCTIONS_SERVICE_ACCOUNT", raising=False)
    monkeypatch.delenv("ALLOWED_SERVICE_ACCOUNTS", raising=False)
    with pytest.raises(ValueError, match="ALLOWED_SERVICE_ACCOUNTS"):
        RecSettings(
            **_prod_kwargs(
                allowed_service_accounts="",
                firebase_functions_service_account="",
            )
        )


def test_production_rejects_local_dev_dsn(monkeypatch):
    """story-data's local DSN must never reach production."""
    monkeypatch.delenv("RECS_DATABASE_URL", raising=False)
    with pytest.raises(ValueError, match="local story-data default"):
        RecSettings(**_prod_kwargs(recs_database_url=LOCAL_DEV_DSN))


def test_production_requires_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_AI_STUDIO_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GOOGLE_AI_STUDIO_API_KEY"):
        RecSettings(**_prod_kwargs(google_ai_studio_api_key=""))


def test_development_needs_no_configuration(monkeypatch):
    """Zero-config local dev is the point of the LOCAL_DEV_DSN default."""
    for var in (
        "RECS_SERVICE_URL",
        "RECS_DATABASE_URL",
        "GOOGLE_AI_STUDIO_API_KEY",
        "ALLOWED_SERVICE_ACCOUNTS",
    ):
        monkeypatch.delenv(var, raising=False)
    settings = RecSettings(environment="development")
    assert settings.write_dsn == LOCAL_DEV_DSN
    assert settings.oidc_audience is None


# ── Derived DSNs ─────────────────────────────────────────────────────────


def test_read_dsn_falls_back_to_write_dsn():
    """Local dev has one compute; the RO DSN is optional."""
    settings = RecSettings(
        recs_database_url="postgresql://a/rw", recs_database_url_ro=""
    )
    assert settings.read_dsn == "postgresql://a/rw"
    assert settings.write_dsn == "postgresql://a/rw"


def test_read_dsn_used_when_set():
    settings = RecSettings(
        recs_database_url="postgresql://a/rw",
        recs_database_url_ro="postgresql://b/ro",
    )
    assert settings.read_dsn == "postgresql://b/ro"
    assert settings.write_dsn == "postgresql://a/rw"


def test_embedding_dimension_is_sourced_from_shared_provider():
    """Not redeclared here — a second definition is how a dimension mismatch
    gets introduced, and a mismatch silently kills vector search."""
    from embedding_provider import EXPECTED_EMBEDDING_DIM

    assert RecSettings().embedding_dimension == EXPECTED_EMBEDDING_DIM
    assert RecSettings().embedding_dimension == 768
