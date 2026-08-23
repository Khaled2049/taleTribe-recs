"""Unit tests for the query-embedding cache."""

import os

import pytest

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "test-project")

from embedding_provider import (  # noqa: E402
    EXPECTED_EMBEDDING_DIM,
    EmbeddingProvider,
    TaskType,
)
from recommendation_engine.embeddings import (  # noqa: E402
    QueryEmbedder,
    cache_key,
    normalize_query,
)

pytestmark = pytest.mark.unit


class _CountingProvider(EmbeddingProvider):
    """Records every call so cache behaviour is observable."""

    def __init__(self):
        self.embed_calls = []
        self.batch_calls = []

    @property
    def dimension(self):
        return EXPECTED_EMBEDDING_DIM

    async def embed(self, text, task_type=None):
        self.embed_calls.append((text, task_type))
        seed = float(len(text))
        return [seed] * EXPECTED_EMBEDDING_DIM

    async def embed_batch(self, texts, task_type=None):
        self.batch_calls.append((list(texts), task_type))
        return [[float(len(t))] * EXPECTED_EMBEDDING_DIM for t in texts]


# ── Normalization ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Dune", "dune"),
        ("  Dune  ", "dune"),
        ("DUNE", "dune"),
        ("cozy   small-town   mystery", "cozy small-town mystery"),
        ("line\nbreak", "line break"),
        ("tab\there", "tab here"),
    ],
)
def test_normalize_query_folds_trivial_variation(raw, expected):
    assert normalize_query(raw) == expected


def test_normalize_query_handles_non_ascii():
    """casefold(), not lower() — the catalog has titles like 'Roman à clef'."""
    assert normalize_query("ROMAN À CLEF") == "roman à clef"
    assert normalize_query("STRASSE") == normalize_query("strasse")


def test_variants_share_a_cache_key():
    assert cache_key("Dune", TaskType.RETRIEVAL_QUERY) == cache_key(
        "  dune ", TaskType.RETRIEVAL_QUERY
    )


def test_task_type_is_part_of_the_key():
    """The same string embeds to different vectors as document vs query, so the
    two must never collide in the cache."""
    assert cache_key("Dune", TaskType.RETRIEVAL_QUERY) != cache_key(
        "Dune", TaskType.RETRIEVAL_DOCUMENT
    )


def test_different_text_yields_different_keys():
    assert cache_key("Dune", TaskType.RETRIEVAL_QUERY) != cache_key(
        "Neuromancer", TaskType.RETRIEVAL_QUERY
    )


# ── Caching ──────────────────────────────────────────────────────────────


async def test_repeat_query_is_served_from_cache():
    provider = _CountingProvider()
    embedder = QueryEmbedder(provider)

    first = await embedder.embed_query("cozy mystery")
    second = await embedder.embed_query("cozy mystery")

    assert first == second
    assert len(provider.embed_calls) == 1, "second call should not reach the provider"
    assert embedder.hits == 1
    assert embedder.misses == 1


async def test_normalized_variants_hit_the_same_entry():
    provider = _CountingProvider()
    embedder = QueryEmbedder(provider)

    await embedder.embed_query("Dune")
    await embedder.embed_query("  DUNE  ")

    assert len(provider.embed_calls) == 1
    assert embedder.hits == 1


async def test_queries_are_embedded_as_retrieval_query():
    provider = _CountingProvider()

    await QueryEmbedder(provider).embed_query("find me something like Dune")

    _, task_type = provider.embed_calls[0]
    assert task_type == TaskType.RETRIEVAL_QUERY


async def test_documents_are_embedded_as_retrieval_document_and_batched():
    provider = _CountingProvider()

    vectors = await QueryEmbedder(provider).embed_documents(["a", "bb", "ccc"])

    assert len(provider.batch_calls) == 1, "must use the batch endpoint, not a loop"
    texts, task_type = provider.batch_calls[0]
    assert texts == ["a", "bb", "ccc"]
    assert task_type == TaskType.RETRIEVAL_DOCUMENT
    assert [v[0] for v in vectors] == [1.0, 2.0, 3.0]


async def test_documents_are_not_cached():
    """Catalog vectors live in Postgres; an in-process copy would just be a
    second, staler store."""
    provider = _CountingProvider()
    embedder = QueryEmbedder(provider)

    await embedder.embed_documents(["same"])
    await embedder.embed_documents(["same"])

    assert len(provider.batch_calls) == 2
    assert embedder.stats()["size"] == 0


# ── LRU eviction ─────────────────────────────────────────────────────────


async def test_cache_is_bounded_and_evicts_least_recently_used():
    provider = _CountingProvider()
    embedder = QueryEmbedder(provider, cache_size=2)

    await embedder.embed_query("first")
    await embedder.embed_query("second")
    await embedder.embed_query("first")  # refreshes "first", so "second" is oldest
    await embedder.embed_query("third")  # evicts "second"

    assert embedder.stats()["size"] == 2

    calls_before = len(provider.embed_calls)
    await embedder.embed_query("first")
    assert len(provider.embed_calls) == calls_before, "'first' should still be cached"

    await embedder.embed_query("second")
    assert len(provider.embed_calls) == calls_before + 1, "'second' should be evicted"


async def test_cache_can_be_disabled():
    provider = _CountingProvider()
    embedder = QueryEmbedder(provider, cache_size=0)

    await embedder.embed_query("text")
    await embedder.embed_query("text")

    assert len(provider.embed_calls) == 2
    assert embedder.stats()["size"] == 0


async def test_negative_cache_size_is_clamped_not_crashing():
    embedder = QueryEmbedder(_CountingProvider(), cache_size=-10)
    await embedder.embed_query("text")
    assert embedder.stats()["capacity"] == 0


# ── Degradation when no provider exists ──────────────────────────────────


def test_reports_unavailable_without_a_provider():
    """Mirrors VectorStore's posture: callers degrade to popularity ranking
    rather than surfacing an error."""
    embedder = QueryEmbedder(None)

    assert embedder.available is False
    assert embedder.dimension is None


async def test_embedding_without_a_provider_raises_a_clear_error():
    embedder = QueryEmbedder(None)

    with pytest.raises(RuntimeError, match="GOOGLE_AI_STUDIO_API_KEY"):
        await embedder.embed_query("text")
    with pytest.raises(RuntimeError, match="GOOGLE_AI_STUDIO_API_KEY"):
        await embedder.embed_documents(["text"])


def test_available_true_with_a_provider():
    embedder = QueryEmbedder(_CountingProvider())

    assert embedder.available is True
    assert embedder.dimension == EXPECTED_EMBEDDING_DIM


# ── Stats ────────────────────────────────────────────────────────────────


async def test_hit_rate_is_none_before_any_traffic():
    assert QueryEmbedder(_CountingProvider()).stats()["hit_rate"] is None


async def test_hit_rate_reported():
    embedder = QueryEmbedder(_CountingProvider())

    await embedder.embed_query("a")  # miss
    await embedder.embed_query("a")  # hit
    await embedder.embed_query("a")  # hit

    assert embedder.stats()["hit_rate"] == pytest.approx(2 / 3)
