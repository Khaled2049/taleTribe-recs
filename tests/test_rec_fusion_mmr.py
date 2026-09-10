"""Unit tests for RRF fusion and MMR diversification.

Expected RRF values are computed by hand from 1/(k+rank); MMR cases use
constructed vectors where the correct pick is unambiguous.
"""

import math

import pytest

from recommendation_engine.fusion import (  # noqa: E402
    DEFAULT_RRF_K,
    fuse_and_rank,
    fuse_normalized,
    max_rrf_score,
    merge_candidate_records,
    reciprocal_rank_fusion,
)
from recommendation_engine.mmr import (  # noqa: E402
    diversify,
    intra_list_diversity,
    mmr_select,
)

pytestmark = pytest.mark.unit

K = DEFAULT_RRF_K


# ══════════════════════════════════════════════════════════════════════════
# RRF
# ══════════════════════════════════════════════════════════════════════════


def test_single_list_scores_match_the_formula():
    scores = reciprocal_rank_fusion([["a", "b", "c"]])

    assert scores["a"] == pytest.approx(1 / (K + 1))
    assert scores["b"] == pytest.approx(1 / (K + 2))
    assert scores["c"] == pytest.approx(1 / (K + 3))


def test_appearing_in_multiple_lists_accumulates():
    scores = reciprocal_rank_fusion([["a", "b"], ["b", "a"]])

    expected = 1 / (K + 1) + 1 / (K + 2)
    assert scores["a"] == pytest.approx(expected)
    assert scores["b"] == pytest.approx(expected)


def test_consistent_placement_beats_one_first_place():
    """The behaviour that makes RRF right for a multi-book query: a book that
    places respectably for every seed beats one that tops a single seed."""
    scores = reciprocal_rank_fusion(
        [
            ["specialist", "generalist"],
            ["other", "generalist"],
            ["another", "generalist"],
        ]
    )

    assert scores["generalist"] > scores["specialist"]


def test_k_damps_the_advantage_of_rank_one():
    """With k=60 ranks 1 and 2 are close; with k=1 they are far apart."""
    damped = reciprocal_rank_fusion([["a", "b"]], k=60)
    sharp = reciprocal_rank_fusion([["a", "b"]], k=1)

    assert damped["a"] / damped["b"] < sharp["a"] / sharp["b"]


def test_k_is_floored_at_one():
    """k=0 would make rank 1 score infinity."""
    scores = reciprocal_rank_fusion([["a"]], k=0)

    assert math.isfinite(scores["a"])


def test_empty_input_yields_nothing():
    assert reciprocal_rank_fusion([]) == {}
    assert reciprocal_rank_fusion([[], []]) == {}


# ── Normalization ────────────────────────────────────────────────────────


def test_max_score_is_first_place_in_every_list():
    assert max_rrf_score(3, K) == pytest.approx(3 / (K + 1))


def test_first_in_every_list_normalizes_to_exactly_one():
    """This is what lets the fused score substitute for a cosine similarity in
    the scoring blend without a second set of weights."""
    fused = fuse_normalized([["a", "b"], ["a", "c"], ["a", "d"]])

    assert fused["a"] == pytest.approx(1.0)


def test_normalized_scores_stay_in_unit_range():
    fused = fuse_normalized([["a", "b", "c"], ["c", "a", "b"]])

    assert all(0.0 < score <= 1.0 for score in fused.values())


def test_normalization_accounts_for_list_count():
    """An item first in one of one list scores 1.0; first in one of three does not."""
    one = fuse_normalized([["a"]])
    three = fuse_normalized([["a"], ["b"], ["c"]])

    assert one["a"] == pytest.approx(1.0)
    assert three["a"] < 0.5


def test_empty_lists_are_ignored_in_the_denominator():
    """A seed that matched nothing must not deflate everyone else's score."""
    with_empty = fuse_normalized([["a", "b"], []])
    without = fuse_normalized([["a", "b"]])

    assert with_empty == pytest.approx(without)


def test_all_empty_returns_nothing():
    assert fuse_normalized([[], []]) == {}


# ── Ranking ──────────────────────────────────────────────────────────────


def test_fuse_and_rank_orders_best_first():
    """a: 2/61 = .0328   c: 1/63 + 1/62 = .0320   b: 1/62 = .0161

    Note `c` outranks `b` only because it appears in both lists — inside list one
    it ranks *below* b. Fusion rewarding breadth over a single strong placement is
    the whole point.
    """
    ranked = fuse_and_rank([["a", "b", "c"], ["a", "c"]])

    assert [item for item, _ in ranked] == ["a", "c", "b"]


def test_symmetric_candidates_tie_and_fall_back_to_the_tiebreak():
    """b and c both score 1/62 + 1/63 here, so nothing but the tiebreak separates
    them — and it must be stable rather than arbitrary."""
    ranked = fuse_and_rank([["a", "b", "c"], ["a", "c", "b"]])

    assert [item for item, _ in ranked] == ["a", "b", "c"]
    scores = dict(ranked)
    assert scores["b"] == pytest.approx(scores["c"])


def test_ties_break_deterministically():
    """Without a stable tiebreak, identical requests could return different
    orders — which breaks both caching and eval reproducibility."""
    first = fuse_and_rank([["x", "y"], ["y", "x"]])
    second = fuse_and_rank([["y", "x"], ["x", "y"]])

    assert [item for item, _ in first] == [item for item, _ in second]


def test_fusion_is_order_independent_across_lists():
    """iterative_scan=relaxed_order returns approximately-ordered rows, and the
    lists themselves arrive from concurrent queries in nondeterministic order. If
    fusion ever depended on that, ranking would be a silent coin flip."""
    lists = [["a", "b", "c"], ["b", "c", "a"], ["c", "a", "b"]]

    forward = fuse_normalized(lists)
    reversed_ = fuse_normalized(list(reversed(lists)))

    assert forward == pytest.approx(reversed_)


# ── Record merging ───────────────────────────────────────────────────────


def _row(item_id, sem_cos):
    return {"id": item_id, "title": f"Book {item_id}", "sem_cos": sem_cos}


def test_merge_keeps_one_record_per_id_with_a_fused_score():
    merged = merge_candidate_records(
        [[_row(1, 0.9), _row(2, 0.7)], [_row(2, 0.8), _row(3, 0.6)]]
    )

    assert [row["id"] for row in merged] == [2, 1, 3]
    assert merged[0]["semantic"] == pytest.approx(fuse_normalized([[1, 2], [2, 3]])[2])


def test_merge_records_which_seeds_matched():
    """Lets an explanation say which of the reader's books a result came from."""
    merged = merge_candidate_records([[_row(1, 0.9)], [_row(1, 0.5)], [_row(2, 0.4)]])

    by_id = {row["id"]: row for row in merged}
    assert by_id[1]["matched_query_count"] == 2
    assert by_id[1]["per_query_similarity"] == [0.9, 0.5]
    assert by_id[2]["matched_query_count"] == 1


def test_merge_preserves_other_row_fields():
    merged = merge_candidate_records([[_row(1, 0.9)]])

    assert merged[0]["title"] == "Book 1"


def test_merge_of_nothing_is_empty():
    assert merge_candidate_records([]) == []
    assert merge_candidate_records([[], []]) == []


# ══════════════════════════════════════════════════════════════════════════
# MMR
# ══════════════════════════════════════════════════════════════════════════


def _unit(index, dimension=4):
    vector = [0.0] * dimension
    vector[index] = 1.0
    return vector


def test_highest_scoring_item_is_always_picked_first():
    embeddings = [_unit(0), _unit(0), _unit(1)]
    scores = [0.5, 0.9, 0.4]

    assert mmr_select(embeddings, scores, k=1) == [1]


def test_near_duplicates_are_suppressed_in_favour_of_variety():
    """The behaviour that stops a shelf being five Dune sequels: item 1 is a
    near-duplicate of the top pick, item 2 is orthogonal and scores lower."""
    embeddings = [_unit(0), _unit(0), _unit(1)]
    scores = [0.9, 0.85, 0.6]

    selected = mmr_select(embeddings, scores, k=2, lambda_=0.5)

    assert selected == [0, 2]


def test_lambda_one_degenerates_to_score_order():
    embeddings = [_unit(0), _unit(0), _unit(1)]
    scores = [0.9, 0.85, 0.6]

    assert mmr_select(embeddings, scores, k=3, lambda_=1.0) == [0, 1, 2]


def test_low_lambda_prioritizes_diversity_over_relevance():
    embeddings = [_unit(0), _unit(0), _unit(1)]
    scores = [0.9, 0.89, 0.3]

    relevance_first = mmr_select(embeddings, scores, k=2, lambda_=1.0)
    diversity_first = mmr_select(embeddings, scores, k=2, lambda_=0.1)

    assert relevance_first == [0, 1]
    assert diversity_first == [0, 2]


def test_returns_at_most_k_but_never_more_than_available():
    embeddings = [_unit(0), _unit(1)]

    assert len(mmr_select(embeddings, [0.5, 0.4], k=10)) == 2


def test_empty_and_degenerate_inputs():
    assert mmr_select([], [], k=5) == []
    assert mmr_select([_unit(0)], [0.5], k=0) == []


def test_no_duplicate_indices_are_selected():
    embeddings = [_unit(i % 4) for i in range(8)]
    scores = [0.9 - 0.05 * i for i in range(8)]

    selected = mmr_select(embeddings, scores, k=8)

    assert len(selected) == len(set(selected))


def test_zero_vectors_do_not_produce_nan():
    """A missing embedding must make an item maximally dissimilar, not poison the
    whole similarity matrix."""
    embeddings = [[0.0] * 4, _unit(1), _unit(2)]
    scores = [0.9, 0.5, 0.4]

    selected = mmr_select(embeddings, scores, k=3)

    assert sorted(selected) == [0, 1, 2]


# ── diversify() over record dicts ────────────────────────────────────────


def _record(item_id, score, embedding):
    return {"id": item_id, "score": score, "embedding": embedding}


def test_diversify_reorders_and_trims():
    records = [
        _record(1, 0.9, _unit(0)),
        _record(2, 0.85, _unit(0)),
        _record(3, 0.6, _unit(1)),
    ]

    result = diversify(records, k=2, lambda_=0.5)

    assert [row["id"] for row in result] == [1, 3]


def test_diversify_handles_a_missing_embedding():
    records = [
        _record(1, 0.9, _unit(0)),
        {"id": 2, "score": 0.8},  # no embedding at all
    ]

    result = diversify(records, k=2)

    assert len(result) == 2


def test_diversify_falls_back_to_score_order_with_no_embeddings():
    """Better to admit we cannot diversify than to pretend."""
    records = [{"id": 1, "score": 0.5}, {"id": 2, "score": 0.9}]

    result = diversify(records, k=2)

    assert [row["id"] for row in result] == [2, 1]


def test_diversify_of_nothing_is_empty():
    assert diversify([], k=5) == []


# ── Intra-list diversity metric ──────────────────────────────────────────


def test_identical_items_score_zero_diversity():
    """A shelf of sequels."""
    assert intra_list_diversity([_unit(0), _unit(0), _unit(0)]) == pytest.approx(0.0)


def test_orthogonal_items_score_full_diversity():
    assert intra_list_diversity([_unit(0), _unit(1), _unit(2)]) == pytest.approx(1.0)


def test_diversity_is_undefined_for_fewer_than_two_items():
    """None rather than 1.0 — a one-item shelf is not perfectly diverse."""
    assert intra_list_diversity([]) is None
    assert intra_list_diversity([_unit(0)]) is None


def test_diversity_detects_mmr_working():
    """The end-to-end property the metric exists to measure."""
    embeddings = [_unit(0), _unit(0), _unit(0), _unit(1), _unit(2)]
    scores = [0.9, 0.89, 0.88, 0.5, 0.45]

    undiversified = [embeddings[i] for i in mmr_select(embeddings, scores, 3, 1.0)]
    diversified = [embeddings[i] for i in mmr_select(embeddings, scores, 3, 0.5)]

    assert intra_list_diversity(diversified) > intra_list_diversity(undiversified)
