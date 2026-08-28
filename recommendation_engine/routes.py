"""HTTP surface for the recommendation service.

Every route goes through `pipeline.rank`, the same function the debugging CLI uses,
so the two cannot rank differently.

Two rate buckets, for a reason. Ranking is one database round trip and can be
generous. HyDE and explanations spend Gemini tokens on a path that deliberately
bypasses creditProxy, so they get a much tighter bucket — together with the
explanation cache, that is the *only* thing standing between a loop and a bill.
"""

import logging
import time
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from recommendation_engine import explain as explain_mod
from recommendation_engine import hyde as hyde_mod
from recommendation_engine.app_state import recommendation_state
from recommendation_engine.pipeline import rank
from recommendation_engine.retrieval import RetrievalFilters
from recommendation_engine.scoring import CatalogStats, ScoringConfig

logger = logging.getLogger(__name__)

# Scoring knobs and catalog aggregates are global, not per-request. Re-reading them
# every request would add two round trips to a path whose whole point is one.
_CONFIG_TTL_SECONDS = 60


class _Cached:
    """Tiny TTL cache for the scoring config and catalog statistics."""

    def __init__(self, ttl: float = _CONFIG_TTL_SECONDS) -> None:
        self._ttl = ttl
        self._at = 0.0
        self.config = ScoringConfig()
        self.stats = CatalogStats()

    async def get(self, db, retriever):
        now = time.monotonic()
        if now - self._at < self._ttl:
            return self.config, self.stats
        try:
            self.config = ScoringConfig.from_mapping(await db.load_config())
            self.stats = CatalogStats.from_row(await retriever.catalog_stats())
            self._at = now
        except Exception as exc:
            # Serve the last known values rather than failing the request; a stale
            # popularity weight is far better than a 500.
            logger.warning("scoring_config_refresh_failed error=%s", exc)
        return self.config, self.stats


# ── Request / response models ────────────────────────────────────────────


class FilterSpec(BaseModel):
    genres: Optional[List[str]] = None
    themes: Optional[List[str]] = None
    sources: Optional[List[str]] = None
    max_word_count: Optional[int] = None
    min_word_count: Optional[int] = None
    author: Optional[str] = None
    published_after: Optional[int] = None

    def to_filters(self, exclude_ids: List[int]) -> RetrievalFilters:
        return RetrievalFilters(
            genres=self.genres,
            themes=self.themes,
            sources=self.sources,
            max_word_count=self.max_word_count,
            min_word_count=self.min_word_count,
            author=self.author,
            published_after=self.published_after,
            exclude_ids=exclude_ids,
        )


class BehavioralRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    top_k: int = Field(default=10, ge=1, le=50)
    filters: Optional[FilterSpec] = None


class SeedBook(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    author: Optional[str] = Field(default=None, max_length=200)


class AdhocRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    prompt: Optional[str] = Field(default=None, max_length=2000)
    # Repeatable seeds are retrieved independently and fused by rank — never by
    # averaging their vectors, which would point at a region resembling neither.
    books: List[SeedBook] = Field(default_factory=list)
    top_k: int = Field(default=10, ge=1, le=50)
    filters: Optional[FilterSpec] = None
    use_hyde: Optional[bool] = None  # None = follow the server default


class ExplainRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    item_ids: List[int] = Field(min_length=1, max_length=25)
    prompt: Optional[str] = Field(default=None, max_length=2000)
    seed_item_ids: List[int] = Field(default_factory=list)


def build_router(verify_internal_token) -> APIRouter:
    """Build the router. The auth dependency is injected so `server.py` keeps
    ownership of how callers are verified."""
    router = APIRouter(dependencies=[Depends(verify_internal_token)])
    cached = _Cached()

    # ── Helpers ──────────────────────────────────────────────────────────

    async def _check_rate(request: Request, user_id: str, llm: bool = False) -> None:
        state = recommendation_state(request)
        if not await state.rate_limiter.allow(user_id):
            raise HTTPException(
                status_code=429,
                detail={
                    "code": "RATE_LIMITED",
                    "message": "Too many requests; slow down.",
                    "details": None,
                },
            )
        if llm and not await state.llm_rate_limiter.allow(user_id):
            raise HTTPException(
                status_code=429,
                detail={
                    "code": "LLM_RATE_LIMITED",
                    "message": (
                        "Too many generation requests; these are not credit-metered "
                        "and are rate limited separately."
                    ),
                    "details": None,
                },
            )

    async def _load_targets(request: Request, item_ids: List[int]) -> list:
        """Fetch the fields an explanation needs, by id."""
        state = recommendation_state(request)
        rows = await state.db.read_pool.fetch(
            "SELECT id, source::text AS source, source_id, title, author, genres, "
            "themes, tone, core_premise, embed_input_sha "
            "FROM recommendations.items WHERE id = ANY($1::bigint[])",
            item_ids,
        )
        by_id = {row["id"]: row for row in rows}
        targets = []
        # Preserve the caller's order so explanations stream in shelf order.
        for item_id in item_ids:
            row = by_id.get(item_id)
            if row is None:
                continue
            targets.append(
                explain_mod.ExplanationTarget(
                    item_id=row["id"],
                    source=row["source"],
                    source_id=row["source_id"],
                    title=row["title"],
                    author=row["author"],
                    genres=list(row["genres"] or []),
                    themes=list(row["themes"] or []),
                    tone=list(row["tone"] or []),
                    core_premise=row["core_premise"],
                    embed_input_sha=row["embed_input_sha"] or "",
                )
            )
        return targets

    # ── Behavioral recommendations ───────────────────────────────────────

    @router.post("/recommend/behavioral")
    async def behavioral(payload: BehavioralRequest, request: Request):
        """Recommendations from a reader's own history.

        Uses the **precomputed** `user_taste` vector, so there is no embedding call
        on this path at all — which is what makes it the fast one.

        Until the Firestore signals export exists nothing populates `user_taste`,
        so every reader currently falls through to popularity. That is reported
        honestly in `mode` rather than dressed up as personalization.
        """
        await _check_rate(request, payload.user_id)
        state = recommendation_state(request)
        config, stats = await cached.get(state.db, state.retriever)

        row = await state.db.read_pool.fetchrow(
            "SELECT taste_embedding, seed_item_ids, suppressed_item_ids, n_signals "
            "FROM recommendations.user_taste WHERE user_id = $1",
            payload.user_id,
        )

        query_vectors: List[List[float]] = []
        exclude: List[int] = []
        mode = "popular"

        if row is not None and row["n_signals"] >= 3:
            vector = row["taste_embedding"]
            to_list = getattr(vector, "to_list", None)
            query_vectors = [to_list() if callable(to_list) else list(vector)]
            # Never recommend what they have already engaged with, or disliked.
            exclude = list(row["seed_item_ids"] or []) + list(
                row["suppressed_item_ids"] or []
            )
            mode = "behavioral"

        filters = (payload.filters or FilterSpec()).to_filters(exclude)
        result = await rank(
            db=state.db,
            retriever=state.retriever,
            query_vectors=query_vectors,
            top_k=payload.top_k,
            config=config,
            stats=stats,
            filters=filters,
        )
        return {
            "success": True,
            "data": {
                "mode": mode,
                "n_signals": row["n_signals"] if row is not None else 0,
                **result.as_dict(),
            },
        }

    # ── Ad-hoc recommendations ───────────────────────────────────────────

    @router.post("/recommend/adhoc")
    async def adhoc(payload: AdhocRequest, request: Request):
        """ "I liked these, find me more" — or a free-text description.

        Named books are resolved against the catalog and retrieved with their stored
        vectors, one query per book, fused by rank. A free-text prompt goes through
        HyDE when enabled.
        """
        if not payload.prompt and not payload.books:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "INVALID_REQUEST",
                    "message": "provide `prompt`, `books`, or both",
                    "details": None,
                },
            )

        state = recommendation_state(request)
        settings = state.settings
        wants_hyde = (
            settings.recs_enable_hyde if payload.use_hyde is None else payload.use_hyde
        )
        # Only the HyDE path spends tokens, so only it draws on the tighter bucket.
        await _check_rate(
            request, payload.user_id, llm=bool(payload.prompt and wants_hyde)
        )
        config, stats = await cached.get(state.db, state.retriever)

        query_vectors: List[List[float]] = []
        exclude: List[int] = []
        resolved: List[dict] = []
        unresolved: List[str] = []

        for book in payload.books:
            matches = await state.retriever.resolve_titles(
                book.title, book.author, limit=1
            )
            if not matches:
                unresolved.append(book.title)
                continue
            match = matches[0]
            vector = match["embedding"]
            to_list = getattr(vector, "to_list", None)
            query_vectors.append(to_list() if callable(to_list) else list(vector))
            exclude.append(match["id"])
            resolved.append({"id": match["id"], "title": match["title"]})

        hyde_used = False
        hyde_doc = None
        if payload.prompt:
            embedder = state.query_embedder
            if not embedder.available:
                raise HTTPException(
                    status_code=503,
                    detail={
                        "code": "EMBEDDER_UNAVAILABLE",
                        "message": "no embedding provider configured",
                        "details": None,
                    },
                )
            text = payload.prompt
            if wants_hyde and state.llm is not None:
                result = await hyde_mod.generate(
                    state.llm,
                    payload.prompt,
                    max_output_tokens=settings.recs_hyde_max_output_tokens,
                )
                if result is not None:
                    # Embed the hypothetical catalog entry instead of the request,
                    # so query and documents are the same kind of text.
                    text = result.embed_text
                    hyde_used = True
                    hyde_doc = result.as_dict()
            query_vectors.append(await embedder.embed_query(text))

        # Every named book missing from the catalog is a legitimate outcome, not an
        # error — the reader may have named something we do not carry.
        if not query_vectors:
            logger.info("adhoc_no_vectors unresolved=%s", unresolved)

        filters = (payload.filters or FilterSpec()).to_filters(exclude)
        result = await rank(
            db=state.db,
            retriever=state.retriever,
            query_vectors=query_vectors,
            top_k=payload.top_k,
            config=config,
            stats=stats,
            filters=filters,
        )

        fingerprint = explain_mod.query_fingerprint(
            query=payload.prompt, seed_item_ids=[r["id"] for r in resolved]
        )
        data = result.as_dict()
        # Returned so a client can tell whether an explanation would be a cache hit
        # before asking for one.
        for item in data["items"]:
            item["explanation_cache_key"] = explain_mod.cache_key(
                state.llm.model if state.llm else "none",
                item["source"],
                item["source_id"],
                next(
                    (
                        i.embed_input_sha or ""
                        for i in result.items
                        if i.id == item["id"]
                    ),
                    "",
                ),
                fingerprint,
            )

        return {
            "success": True,
            "data": {
                "mode": "adhoc",
                "resolved_books": resolved,
                "unresolved_books": unresolved,
                "hyde_used": hyde_used,
                "hypothetical_document": hyde_doc,
                "query_fingerprint": fingerprint,
                **data,
            },
        }

    # ── Explanations: sync ───────────────────────────────────────────────

    @router.post("/recommend/explain")
    async def explain_sync(payload: ExplainRequest, request: Request):
        """All explanations in one response (Stage A).

        The mode that works through a Firebase Functions proxy, which buffers
        responses and so cannot stream.
        """
        await _check_rate(request, payload.user_id, llm=True)
        state = recommendation_state(request)

        targets = await _load_targets(request, payload.item_ids)
        if not targets:
            return {"success": True, "data": {"explanations": []}}

        fingerprint = explain_mod.query_fingerprint(
            query=payload.prompt, seed_item_ids=payload.seed_item_ids
        )
        context = explain_mod.describe_context(payload.prompt, None)
        explanations = await explain_mod.explain_many(
            state.llm,
            state.explanation_cache,
            targets,
            context=context,
            fingerprint=fingerprint,
            max_output_tokens=state.settings.recs_explanation_max_output_tokens,
        )
        return {
            "success": True,
            "data": {"explanations": explanations, "query_fingerprint": fingerprint},
        }

    # ── Explanations: streaming ──────────────────────────────────────────

    @router.get("/recommend/explain/stream")
    async def explain_stream(
        request: Request,
        user_id: str = Query(min_length=1, max_length=128),
        item_ids: str = Query(description="comma-separated item ids"),
        prompt: Optional[str] = Query(default=None, max_length=2000),
        seed_item_ids: Optional[str] = Query(default=None),
    ):
        """One multiplexed SSE stream over every requested item (Stage B).

        GET rather than POST because SSE is a GET-shaped protocol in browsers.
        Events carry `item_id`, so a single connection updates ten cards rather than
        opening ten connections.
        """
        try:
            ids = [int(part) for part in item_ids.split(",") if part.strip()][:25]
            seeds = [
                int(part) for part in (seed_item_ids or "").split(",") if part.strip()
            ]
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "INVALID_REQUEST",
                    "message": "item_ids must be comma-separated integers",
                    "details": None,
                },
            )
        if not ids:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "INVALID_REQUEST",
                    "message": "item_ids is required",
                    "details": None,
                },
            )

        await _check_rate(request, user_id, llm=True)
        state = recommendation_state(request)
        targets = await _load_targets(request, ids)
        fingerprint = explain_mod.query_fingerprint(query=prompt, seed_item_ids=seeds)
        context = explain_mod.describe_context(prompt, None)

        stream = explain_mod.stream_explanations(
            state.llm,
            state.explanation_cache,
            targets,
            context=context,
            fingerprint=fingerprint,
            is_disconnected=request.is_disconnected,
            max_output_tokens=state.settings.recs_explanation_max_output_tokens,
        )
        return StreamingResponse(
            stream,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                # Without this, an intermediary proxy may buffer the whole response
                # and defeat streaming entirely.
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    return router
