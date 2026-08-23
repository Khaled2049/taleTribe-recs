"""Unit tests for the retrieve → fuse → score → diversify pipeline.

The central test here is `test_single_seed_uses_raw_cosine_not_rrf`. Fusing a
single ranked list replaces every similarity with a function of its rank — rank 1
becomes exactly 1.0 whatever it actually scored — which was a real bug found by
noticing that a live query returned the identical score sequence to a mock-
embedded one.
"""

import os

import pytest

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "test-project")

from recommendation_engine.fusion import DEFAULT_RRF_K  # noqa: E402
from recommendation_engine.pipeline import rank  # noqa: E402
from recommendation_engine.retrieval import RetrievalResult  # noqa: E402
from recommendation_engine.scoring import CatalogStats, ScoringConfig  # noqa: E402

pytestmark = pytest.mark.unit

DIM = 8


def _row(item_id, sem_cos, source="cmu", n_interactions=0, pop=0.0):
    vector = [0.0] * DIM
    vector[item_id % DIM] = 1.0
    return {
        "id": item_id,
        "source": source,
        "source_id": f"s{item_id}",
        "title": f"Book {item_id}",
        "author": None,
        "genres": [],
        "themes": [],
        "tone": [],
        "core_premise": None,
        "published_year": None,
        "embedding": vector,
        "embed_input_sha": f"sha{item_id}",
        "sem_cos": sem_cos,
        "n_interactions": n_interactions,
        "pop_score": pop,
    }


class _FakeRetriever:
    def __init__(self, per_query_rows, popular_rows=None):
        self._per_query = per_query_rows
        self._popular = popular_rows or []
        self.knn_many_calls = 0
        self.popular_calls = 0

    async def knn_many(self, query_vectors, limit, filters=None):
        self.knn_many_calls += 1
        return [RetrievalResult(rows=list(rows)) for rows in self._per_query]

    async def popular(self, limit, filters=None):
        self.popular_calls += 1
        return list(self._popular)


CONFIG = ScoringConfig()
STATS = CatalogStats(platform_item_count=0)


async def _rank(retriever, query_vectors, top_k=5, config=CONFIG):
    # db is unused while w_cf_ceiling is 0 — _cf_scores short-circuits.
    return await rank(
        db=None,
        retriever=retriever,
        query_vectors=query_vectors,
        top_k=top_k,
        config=config,
        stats=STATS,
    )


# ══════════════════════════════════════════════════════════════════════════
# The single-seed / multi-seed split
# ══════════════════════════════════════════════════════════════════════════


async def test_single_seed_uses_raw_cosine_not_rrf():
    """With one seed the real similarity must survive to the score.

    RRF on a single list would yield 1.0, 61/62, 61/63 … regardless of whether the
    top match scored 0.95 or 0.31.
    """
    rows = [_row(1, 0.67), _row(2, 0.31), _row(3, 0.12)]
    retriever = _FakeRetriever([rows])

    result = await _rank(retriever, [[1.0] + [0.0] * (DIM - 1)])

    by_id = {item.id: item for item in result.items}
    assert by_id[1].breakdown["semantic"] == pytest.approx(0.67)
    assert by_id[2].breakdown["semantic"] == pytest.approx(0.31)
    assert by_id[3].breakdown["semantic"] == pytest.approx(0.12)


async def test_single_seed_top_score_is_not_pinned_to_one():
    """The specific symptom that exposed the bug."""
    retriever = _FakeRetriever([[_row(1, 0.42)]])

    result = await _rank(retriever, [[1.0] + [0.0] * (DIM - 1)])

    assert result.items[0].score == pytest.approx(0.42)
    assert result.items[0].score != 1.0


async def test_a_poor_match_scores_poorly():
    """Magnitude has to mean something, or a caller cannot threshold on it."""
    good = _FakeRetriever([[_row(1, 0.9)]])
    poor = _FakeRetriever([[_row(1, 0.05)]])

    good_result = await _rank(good, [[1.0] + [0.0] * (DIM - 1)])
    poor_result = await _rank(poor, [[1.0] + [0.0] * (DIM - 1)])

    assert good_result.items[0].score > 0.8
    assert poor_result.items[0].score < 0.1


async def test_single_seed_reports_one_matched_query():
    retriever = _FakeRetriever([[_row(1, 0.5)]])

    result = await _rank(retriever, [[1.0] + [0.0] * (DIM - 1)])

    assert result.items[0].matched_query_count == 1


async def test_multi_seed_uses_rrf():
    """With several seeds, per-seed cosines are not comparable, so rank fusion is
    correct — an item first in every list scores 1.0."""
    top_everywhere = _row(1, 0.4)
    retriever = _FakeRetriever(
        [
            [dict(top_everywhere), _row(2, 0.9)],
            [dict(top_everywhere), _row(3, 0.9)],
        ]
    )

    result = await _rank(retriever, [[1.0] + [0.0] * (DIM - 1)] * 2)

    by_id = {item.id: item for item in result.items}
    # Fused to the maximum despite a middling cosine in both lists.
    assert by_id[1].breakdown["semantic"] == pytest.approx(1.0)
    # And the high-cosine items rank below it, because they appear in one list only.
    assert by_id[2].breakdown["semantic"] < 1.0


async def test_multi_seed_scores_are_rank_derived_not_cosine():
    retriever = _FakeRetriever([[_row(1, 0.9), _row(2, 0.8)], [_row(3, 0.7)]])

    result = await _rank(retriever, [[1.0] + [0.0] * (DIM - 1)] * 2)

    semantics = {item.id: item.breakdown["semantic"] for item in result.items}
    # Rank-2 in a two-list fusion is (1/62)/(2/61); never the raw 0.8.
    assert semantics[2] != pytest.approx(0.8)
    assert semantics[2] == pytest.approx(
        (1 / (DEFAULT_RRF_K + 2)) / (2 / (DEFAULT_RRF_K + 1))
    )


async def test_multi_seed_counts_matched_queries():
    """How many of the reader's seeds each result matched — what lets an
    explanation say *which* of their books a recommendation came from."""
    shared = _row(1, 0.5)
    # Item 1 appears in two of the three lists; item 2 in one.
    retriever = _FakeRetriever([[dict(shared)], [dict(shared)], [_row(2, 0.5)]])

    result = await _rank(retriever, [[1.0] + [0.0] * (DIM - 1)] * 3)

    by_id = {item.id: item for item in result.items}
    assert by_id[1].matched_query_count == 2
    assert by_id[2].matched_query_count == 1


# ══════════════════════════════════════════════════════════════════════════
# Fallback and plumbing
# ══════════════════════════════════════════════════════════════════════════


async def test_no_query_vectors_falls_back_to_popularity():
    retriever = _FakeRetriever([], popular_rows=[_row(1, 0.0, pop=0.8)])

    result = await _rank(retriever, [])

    assert retriever.popular_calls == 1
    assert retriever.knn_many_calls == 0
    assert result.items[0].breakdown["semantic"] == 0.0


async def test_empty_retrieval_returns_no_items():
    result = await _rank(_FakeRetriever([[]]), [[1.0] + [0.0] * (DIM - 1)])

    assert result.items == []


async def test_degraded_flag_propagates():
    class _Degraded(_FakeRetriever):
        async def knn_many(self, query_vectors, limit, filters=None):
            return [RetrievalResult(rows=[_row(1, 0.5)], degraded=True)]

    result = await _rank(_Degraded([]), [[1.0] + [0.0] * (DIM - 1)])

    assert result.degraded is True


async def test_top_k_is_respected():
    rows = [_row(i, 0.9 - i * 0.05) for i in range(1, 11)]
    result = await _rank(_FakeRetriever([rows]), [[1.0] + [0.0] * (DIM - 1)], top_k=3)

    assert len(result.items) == 3
    assert result.candidates_considered == 10


async def test_cf_is_inert_while_stubbed():
    """w_cf_ceiling is 0, so _cf_scores must not even touch the database — db is
    None here and this would raise if it did."""
    result = await _rank(_FakeRetriever([[_row(1, 0.5)]]), [[1.0] + [0.0] * (DIM - 1)])

    assert result.items[0].breakdown["collaborative"] == 0.0
    assert result.items[0].breakdown["weights"]["collaborative"] == 0.0


async def test_off_platform_flag_marks_seed_corpus_items():
    """CMU books are not readable on TaleTribe, so the UI has to badge them."""
    retriever = _FakeRetriever(
        [[_row(1, 0.5, source="cmu"), _row(2, 0.5, source="platform")]]
    )

    result = await _rank(retriever, [[1.0] + [0.0] * (DIM - 1)])

    by_id = {item.id: item for item in result.items}
    assert by_id[1].off_platform is True
    assert by_id[2].off_platform is False


async def test_diversity_is_reported():
    rows = [_row(1, 0.9), _row(2, 0.8), _row(3, 0.7)]

    result = await _rank(_FakeRetriever([rows]), [[1.0] + [0.0] * (DIM - 1)])

    assert result.diversity is not None
    assert 0.0 <= result.diversity <= 1.0
