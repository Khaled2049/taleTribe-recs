"""Embedding provider abstraction — mirrors the LLM provider pattern."""

import hashlib
import logging
import os
from abc import ABC, abstractmethod

import anyio
import httpx
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential_jitter,
)

logger = logging.getLogger(__name__)

_GOOGLE_EMBED_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent"
)
_GOOGLE_BATCH_EMBED_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents"
)
_GOOGLE_DEFAULT_MODEL = "gemini-embedding-001"

# batchEmbedContents accepts up to 100 sub-requests per call. Callers embedding
# long documents may need a smaller batch to stay under the per-call token cap;
# the recommendation catalog's normalized text is ~150 tokens, so 100 is safe.
BATCH_MAX_REQUESTS = 100


class TaskType:
    """Embedding task types.

    Gemini embedding models produce *asymmetric* embeddings: a document and a
    query about that document are projected differently, which measurably
    improves retrieval over embedding both identically.

    Deliberately opt-in. `embed()` defaults to `task_type=None`, which sends no
    `taskType` field at all and therefore reproduces exactly the vectors already
    stored in `chapter_chunks`, `semantic_memory` and `episodic_memory`. Passing
    a task type for those collections would silently invalidate every stored
    vector — `VectorStore` has no brute-force fallback, so retrieval would
    return nothing at all, with no error anywhere.
    """

    RETRIEVAL_DOCUMENT = "RETRIEVAL_DOCUMENT"
    RETRIEVAL_QUERY = "RETRIEVAL_QUERY"
    SEMANTIC_SIMILARITY = "SEMANTIC_SIMILARITY"
    CLASSIFICATION = "CLASSIFICATION"
    CLUSTERING = "CLUSTERING"


class EmbeddingError(RuntimeError):
    """Base class for embedding failures."""


class EmbeddingTransientError(EmbeddingError):
    """Rate limit, upstream 5xx, or a network fault — worth retrying.

    Separated from permanent failures (bad API key, malformed request) so a
    15k-document backfill retries what it should and fails fast on what it
    shouldn't.
    """


# Single source of truth for the embedding dimension. EVERY vector we write
# (chapter_chunks AND semantic_memory) and EVERY native vector index must use this.
# Keep it in lockstep with:
#   taleTribe-frontend/firestore.indexes.json
#     → fieldOverrides[chapter_chunks].vectorConfig.dimension
# If the embedder's output dim != this, native find_nearest can't match the index's
# vector dimension and queries fail — retrieval silently returns nothing (there is no
# brute-force fallback). That is exactly why this lives in one place and is checked at
# startup (see verify_embedding_dimension). See wiki/chat-scaling-design.md.
EXPECTED_EMBEDDING_DIM = 768

# Known output dimensions per Google embedding model (no network call needed).
# gemini-embedding-001 is Matryoshka (MRL): it natively emits 3072 dims but can be
# truncated via the `outputDimensionality` request param. We pin it to
# EXPECTED_EMBEDDING_DIM so it matches the 768-dim Firestore vector index (Firestore
# native KNN caps at 2048 dims, so the full 3072 isn't indexable anyway).
_GOOGLE_MODEL_DIMS = {
    "gemini-embedding-001": EXPECTED_EMBEDDING_DIM,
    "text-embedding-004": 768,  # retired on AI Studio; kept for reference
}

_MOCK_DIM = EXPECTED_EMBEDDING_DIM  # tests embed at the production dimension


class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed(self, text: str, task_type: str | None = None) -> list[float]: ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Output vector length. Must equal EXPECTED_EMBEDDING_DIM for native
        Firestore vector search to work against the provisioned index."""
        ...

    @property
    def model_id(self) -> str:
        """Stable identifier for whatever produced a vector.

        Recorded alongside every stored embedding so provenance is checkable.
        Vectors from different providers — or different model versions — occupy
        different spaces, and mixing them in one index degrades retrieval with no
        error anywhere. Consumers compare this to decide what needs re-embedding.
        """
        return type(self).__name__

    async def embed_batch(
        self, texts: list[str], task_type: str | None = None
    ) -> list[list[float]]:
        """Embed many texts, returning vectors in the same order as the input.

        Concrete, not abstract, for two reasons: existing subclasses (including
        ones defined in tests) keep working untouched, and a sequential loop is
        the honest fallback for providers with no batch endpoint — a local
        SentenceTransformer gains nothing from batching over HTTP.

        `GoogleAIEmbeddingProvider` overrides this with a real batch call, which
        is the difference between ~166 requests and ~15,500 for a full catalog
        backfill.
        """
        return [await self.embed(text, task_type=task_type) for text in texts]


class GoogleAIEmbeddingProvider(EmbeddingProvider):
    def __init__(self, api_key: str, model: str = _GOOGLE_DEFAULT_MODEL):
        self._api_key = api_key
        self._model = model
        self._url = _GOOGLE_EMBED_URL.format(model=model)
        self._batch_url = _GOOGLE_BATCH_EMBED_URL.format(model=model)
        # One process-lifetime client (connection pool reuse). Creating a fresh
        # AsyncClient per embed() added TLS/handshake latency on every call, which
        # is now multiplied by several retrievals per chat turn. Closed via aclose().
        self._client = httpx.AsyncClient(timeout=30)

    @property
    def dimension(self) -> int:
        return _GOOGLE_MODEL_DIMS.get(self._model, EXPECTED_EMBEDDING_DIM)

    @property
    def model_id(self) -> str:
        return f"google:{self._model}"

    def _request_body(self, text: str, task_type: str | None) -> dict:
        """Build one embedContent request body.

        When `task_type` is None the `taskType` key is omitted entirely rather
        than sent as null. That is what keeps this byte-identical to the requests
        that produced every vector currently stored in Firestore — a test pins
        it, because the failure mode is silent: a task-typed vector simply stops
        matching its neighbours and retrieval returns nothing.
        """
        body: dict = {
            "model": f"models/{self._model}",
            "content": {"parts": [{"text": text}]},
            # Truncate MRL models (e.g. gemini-embedding-001, native 3072) to
            # the dim our vector index expects. Sourced from self.dimension so
            # the request can never drift from the declared/verified dimension.
            "outputDimensionality": self.dimension,
        }
        if task_type is not None:
            body["taskType"] = task_type
        return body

    @retry(
        stop=stop_after_attempt(4) | stop_after_delay(60),
        wait=wait_exponential_jitter(initial=2, max=15),
        retry=retry_if_exception_type(EmbeddingTransientError),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _post(self, url: str, payload: dict) -> dict:
        """POST with retry on transient failures.

        A 15k-document backfill will hit rate limits; without this the whole run
        dies on the first 429. Bounded at 4 attempts / 60s so a genuinely broken
        upstream fails the run instead of retrying for hours.
        """
        try:
            resp = await self._client.post(
                url, headers={"x-goog-api-key": self._api_key}, json=payload
            )
        except httpx.RequestError as exc:  # DNS, connect, read timeout
            raise EmbeddingTransientError(f"embedding request failed: {exc}") from exc

        if resp.status_code == 429 or resp.status_code >= 500:
            raise EmbeddingTransientError(
                f"embedding upstream returned {resp.status_code}: {resp.text[:200]}"
            )
        if resp.status_code >= 400:
            # 400/401/403 are configuration or quota-exhaustion problems. Retrying
            # a bad API key just burns the backfill's time budget.
            raise EmbeddingError(
                f"embedding request rejected with {resp.status_code}: {resp.text[:200]}"
            )
        return resp.json()

    async def embed(self, text: str, task_type: str | None = None) -> list[float]:
        data = await self._post(self._url, self._request_body(text, task_type))
        return data["embedding"]["values"]

    async def embed_batch(
        self, texts: list[str], task_type: str | None = None
    ) -> list[list[float]]:
        """Embed up to BATCH_MAX_REQUESTS texts per HTTP call, in input order.

        The catalog backfill is the reason this exists: 15,500 sequential
        `embedContent` calls is hours of round trips, while batches of 100 is
        ~155 calls.
        """
        if not texts:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(texts), BATCH_MAX_REQUESTS):
            chunk = texts[start : start + BATCH_MAX_REQUESTS]
            data = await self._post(
                self._batch_url,
                {"requests": [self._request_body(text, task_type) for text in chunk]},
            )
            embeddings = data.get("embeddings", [])
            if len(embeddings) != len(chunk):
                # Order and count are the contract callers rely on to zip vectors
                # back onto rows. A short response must not silently misalign them.
                raise EmbeddingError(
                    f"batch embedding returned {len(embeddings)} vectors for "
                    f"{len(chunk)} inputs"
                )
            vectors.extend(item["values"] for item in embeddings)
        return vectors

    async def aclose(self) -> None:
        await self._client.aclose()


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        self._model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    async def embed(self, text: str, task_type: str | None = None) -> list[float]:
        # task_type is accepted for interface parity and ignored: sentence-
        # transformers models are symmetric and have no notion of it.
        result = await anyio.to_thread.run_sync(
            lambda: self._model.encode(text, convert_to_numpy=True)
        )
        return result.tolist()

    async def embed_batch(
        self, texts: list[str], task_type: str | None = None
    ) -> list[list[float]]:
        """One encode() call for the whole list — the local model genuinely
        batches, so this avoids a thread hop per text."""
        if not texts:
            return []
        result = await anyio.to_thread.run_sync(
            lambda: self._model.encode(texts, convert_to_numpy=True)
        )
        return [row.tolist() for row in result]


class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic embeddings for tests — no real API calls."""

    @property
    def dimension(self) -> int:
        return _MOCK_DIM

    async def embed(self, text: str, task_type: str | None = None) -> list[float]:
        # task_type deliberately does NOT change the output: tests that embed a
        # document and then query it must still match, and these vectors carry no
        # semantics to make asymmetric anyway.
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16)
        return [((seed >> i) & 0xFF) / 255.0 for i in range(_MOCK_DIM)]


def get_embedding_provider(api_key: str | None = None) -> EmbeddingProvider | None:
    """Select embedding provider using the same env-var priority as the LLM provider."""
    if os.getenv("USE_MOCK", "").lower() == "true":
        return MockEmbeddingProvider()

    if api_key:
        # text-embedding-004 is the dedicated embedding model; don't use a generative model name
        embed_model = os.getenv("GOOGLE_EMBEDDING_MODEL", _GOOGLE_DEFAULT_MODEL)
        logger.info("Embedding provider: GoogleAI (%s)", embed_model)
        return GoogleAIEmbeddingProvider(api_key, embed_model)

    try:
        provider = SentenceTransformerEmbeddingProvider()
        logger.info("Embedding provider: SentenceTransformer (local)")
        return provider
    except ImportError:
        logger.warning(
            "No embedding provider available; brain memory retrieval disabled"
        )
        return None


def verify_embedding_dimension(embedder: EmbeddingProvider | None) -> None:
    """Assert the active embedder matches the dimension every vector store + index
    expects. Call once at startup so a misconfigured embedder surfaces loudly
    instead of silently rotting recall.

    Default behavior is a loud log.error (so dev with a 384-dim local model still
    runs); set STRICT_EMBEDDING_DIM=true to hard-fail instead — recommended in
    production, where a mismatch means the vector index is effectively dead.
    """
    if embedder is None:
        logger.warning(
            "No embedding provider available — chapter RAG and brain memory "
            "retrieval are DISABLED. Set GOOGLE_AI_STUDIO_API_KEY to enable them."
        )
        return

    dim = embedder.dimension
    if dim == EXPECTED_EMBEDDING_DIM:
        logger.info(
            "Embedding provider %s verified at %d dims (matches vector index).",
            type(embedder).__name__,
            dim,
        )
        return

    msg = (
        "Embedding dimension mismatch: provider %s outputs %d-dim vectors but every "
        "vector store and the Firestore vector index expect %d. Native vector search "
        "(find_nearest) will fail and retrieval will silently return nothing (there is "
        "no brute-force fallback). Use a %d-dim embedding model or "
        "reprovision the index to %d. See wiki/chat-scaling-design.md."
        % (
            type(embedder).__name__,
            dim,
            EXPECTED_EMBEDDING_DIM,
            EXPECTED_EMBEDDING_DIM,
            dim,
        )
    )
    if os.getenv("STRICT_EMBEDDING_DIM", "").lower() == "true":
        raise RuntimeError(msg)
    logger.error(msg)
