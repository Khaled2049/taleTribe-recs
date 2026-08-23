"""Embedding access for the recommendation service.

Thin wrapper over the shared `EmbeddingProvider` that adds the two things this
workload needs and the story agent does not:

* **Task-type asymmetry.** Catalog rows embed as `RETRIEVAL_DOCUMENT`, user input
  as `RETRIEVAL_QUERY`. Gemini projects the two differently, which measurably
  improves retrieval over embedding both identically.
* **A query cache.** Ad-hoc queries repeat heavily across users ("find me
  something like Dune"), and every cache hit removes an 80-250ms round trip and
  a token charge from the request path.

The cache is deliberately *query-only*. Catalog vectors are embedded once at
ingestion and live in Postgres; caching them in process would be a second,
staler copy of the authoritative store.
"""

import hashlib
import logging
import re
from collections import OrderedDict
from typing import List, Optional

from embedding_provider import (
    EmbeddingProvider,
    TaskType,
)

logger = logging.getLogger(__name__)

# Bounded so a burst of unique queries cannot grow the process without limit.
# At 768 float64s (~6KB) per entry, 2000 entries is ~12MB — a rounding error
# next to the HNSW graph, and enough to cover the popular-query head.
DEFAULT_CACHE_SIZE = 2000

_WHITESPACE = re.compile(r"\s+")


def normalize_query(text: str) -> str:
    """Fold trivial variation so 'Dune ', 'dune' and 'Dune' share a cache entry.

    Casefold rather than lower() — it handles non-ASCII correctly, which matters
    for a catalog with titles like 'Roman à clef'.
    """
    return _WHITESPACE.sub(" ", text).strip().casefold()


def cache_key(text: str, task_type: Optional[str]) -> str:
    """Task type is part of the key: the same string embedded as a document and
    as a query yields different vectors, so they must not collide."""
    payload = f"{task_type or ''}|{normalize_query(text)}"
    return hashlib.sha256(payload.encode()).hexdigest()


class QueryEmbedder:
    """Embeds user-supplied text, with an LRU cache in front."""

    def __init__(
        self,
        provider: Optional[EmbeddingProvider],
        cache_size: int = DEFAULT_CACHE_SIZE,
    ) -> None:
        self._provider = provider
        self._cache_size = max(0, cache_size)
        self._cache: "OrderedDict[str, List[float]]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    @property
    def available(self) -> bool:
        """False when no embedding provider could be constructed (no API key and
        no local model). Callers degrade to popularity ranking rather than
        returning an error — the same posture `VectorStore` takes when its index
        is missing."""
        return self._provider is not None

    @property
    def dimension(self) -> Optional[int]:
        return self._provider.dimension if self._provider else None

    async def embed_query(self, text: str) -> List[float]:
        """Embed user input as RETRIEVAL_QUERY, caching the result."""
        if self._provider is None:
            raise RuntimeError(
                "no embedding provider available; set GOOGLE_AI_STUDIO_API_KEY or "
                "USE_MOCK=true"
            )

        key = cache_key(text, TaskType.RETRIEVAL_QUERY)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            self.hits += 1
            return cached

        self.misses += 1
        vector = await self._provider.embed(text, task_type=TaskType.RETRIEVAL_QUERY)

        if self._cache_size:
            self._cache[key] = vector
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return vector

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed catalog text as RETRIEVAL_DOCUMENT, in input order.

        Uncached by design (see module docstring) and batched, because the only
        caller is the ingest path embedding thousands of rows at a time.
        """
        if self._provider is None:
            raise RuntimeError(
                "no embedding provider available; set GOOGLE_AI_STUDIO_API_KEY or "
                "USE_MOCK=true"
            )
        return await self._provider.embed_batch(
            texts, task_type=TaskType.RETRIEVAL_DOCUMENT
        )

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "size": len(self._cache),
            "capacity": self._cache_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": (self.hits / total) if total else None,
        }
