"""Unit tests for the three-term scoring formula.

Expected values are hand-computed from the formula in the module docstring, not
captured from a run — a snapshot test would happily lock in a wrong formula.
"""

import math
import os

import pytest

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "test-project")

from recommendation_engine.scoring import (
    DEFAULTS,
    MAX_BEHAVIORAL_WEIGHT,
    CatalogStats,
    ScoringConfig,
    bayesian_rating,
    behavioral_ramp,
    blend,
    collaborative_score,
    cooccurrence_similarity,
    engagement,
    is_negative_signal,
    percentile,
    popularity,
    seed_engagement_weight,
    term_weights,
)

pytestmark = pytest.mark.unit

CONFIG = ScoringConfig()


# ══════════════════════════════════════════════════════════════════════════
# Config resolution
# ══════════════════════════════════════════════════════════════════════════


def test_defaults_ship_cf_disabled():
    """CF is inert until an event log exists. If this flips silently, ranking
    changes with no code change."""
    assert ScoringConfig().w_cf_ceiling == 0.0


def test_semantic_stays_dominant_by_construction():
    """Vector search does the ranking; the other terms only adjust it."""
    config = ScoringConfig()

    assert config.w_pop_ceiling + config.w_cf_ceiling <= MAX_BEHAVIORAL_WEIGHT


def test_from_mapping_reads_the_config_table():
    config = ScoringConfig.from_mapping({"w_pop_ceiling": 0.3, "mmr_lambda": 0.5})

    assert config.w_pop_ceiling == 0.3
    assert config.mmr_lambda == 0.5
    # Unspecified knobs fall back rather than becoming zero.
    assert config.ramp_n50 == DEFAULTS["ramp_n50"]


def test_from_mapping_ignores_unknown_keys():
    config = ScoringConfig.from_mapping({"not_a_knob": 99})

    assert config.w_pop_ceiling == DEFAULTS["w_pop_ceiling"]


def test_from_mapping_handles_none():
    assert ScoringConfig.from_mapping(None).w_pop_ceiling == DEFAULTS["w_pop_ceiling"]


def test_behavioral_ceilings_are_scaled_down_if_they_exceed_the_cap():
    """A bad tune in the database must degrade ranking, not invert the design."""
    config = ScoringConfig.from_mapping({"w_pop_ceiling": 0.8, "w_cf_ceiling": 0.8})

    total = config.w_pop_ceiling + config.w_cf_ceiling
    assert total == pytest.approx(MAX_BEHAVIORAL_WEIGHT)
    # Ratio preserved when scaling.
    assert config.w_pop_ceiling == pytest.approx(config.w_cf_ceiling)


def test_out_of_range_knobs_are_clamped_not_fatal():
    config = ScoringConfig.from_mapping(
        {"w_pop_ceiling": -1, "mmr_lambda": 5, "ramp_n50": -3}
    )

    assert config.w_pop_ceiling == 0.0
    assert config.mmr_lambda == 1.0
    assert config.ramp_n50 > 0


# ══════════════════════════════════════════════════════════════════════════
# The cold-start ramp
# ══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4])
def test_cold_start_items_are_purely_semantic(n):
    """The spec requirement: below the gate, scoring is 100% semantic."""
    alpha = behavioral_ramp(n, CONFIG)
    w_sem, w_pop, w_cf = term_weights(alpha, CONFIG)

    assert alpha == 0.0
    assert w_sem == 1.0
    assert w_pop == 0.0
    assert w_cf == 0.0


def test_ramp_engages_at_the_threshold():
    """n_min = 5, n50 = 20 → alpha = 5/25 = 0.2."""
    assert behavioral_ramp(5, CONFIG) == pytest.approx(0.2)


def test_ramp_reaches_half_at_n50():
    """The defining property of n50: alpha = 20/40 = 0.5."""
    assert behavioral_ramp(CONFIG.ramp_n50, CONFIG) == pytest.approx(0.5)


def test_ramp_is_monotonic_and_saturating():
    values = [behavioral_ramp(n, CONFIG) for n in (5, 10, 20, 50, 100, 1000)]

    assert values == sorted(values)
    assert values[-1] < 1.0, "a saturating ramp never reaches 1"


def test_ramp_approaches_one_asymptotically():
    assert behavioral_ramp(1_000_000, CONFIG) > 0.99


def test_negative_interaction_count_is_treated_as_zero():
    assert behavioral_ramp(-10, CONFIG) == 0.0


def test_weights_always_sum_to_one():
    """Keeps scores comparable across items with very different volumes."""
    config = ScoringConfig(w_pop_ceiling=0.25, w_cf_ceiling=0.20)

    for n in (0, 5, 20, 100, 10_000):
        w_sem, w_pop, w_cf = term_weights(behavioral_ramp(n, config), config)
        assert w_sem + w_pop + w_cf == pytest.approx(1.0)


def test_semantic_weight_never_drops_below_the_floor():
    """Even at full behavioral ramp with both ceilings enabled."""
    config = ScoringConfig(w_pop_ceiling=0.25, w_cf_ceiling=0.20)

    w_sem, _, _ = term_weights(1.0, config)

    assert w_sem == pytest.approx(0.55)


def test_with_cf_stubbed_the_semantic_floor_is_higher():
    w_sem, _, w_cf = term_weights(1.0, ScoringConfig())

    assert w_cf == 0.0
    assert w_sem == pytest.approx(0.75)


# ══════════════════════════════════════════════════════════════════════════
# Term 2: popularity
# ══════════════════════════════════════════════════════════════════════════


def test_bayesian_rating_pulls_a_single_rating_toward_the_mean():
    """One 5-star rating with C=20 and m=3.5:
    (1*5 + 20*3.5) / 21 = 75/21 = 3.571 → (3.571-1)/4 = 0.643"""
    result = bayesian_rating(1, 5.0, 3.5, CONFIG)

    assert result == pytest.approx(0.6429, abs=1e-4)


def test_bayesian_rating_barely_moves_a_well_rated_book():
    """500 ratings at 5.0: (500*5 + 20*3.5)/520 = 2570/520 = 4.942 → 0.9856"""
    result = bayesian_rating(500, 5.0, 3.5, CONFIG)

    assert result == pytest.approx(0.9856, abs=1e-4)


def test_volume_beats_a_lucky_single_rating():
    """The whole point of the prior."""
    lucky = bayesian_rating(1, 5.0, 3.5, CONFIG)
    established = bayesian_rating(500, 4.5, 3.5, CONFIG)

    assert established > lucky


def test_no_ratings_falls_back_to_the_global_mean():
    """m=3.5 → (3.5-1)/4 = 0.625"""
    assert bayesian_rating(0, None, 3.5, CONFIG) == pytest.approx(0.625)
    assert bayesian_rating(0, 5.0, 3.5, CONFIG) == pytest.approx(0.625)


def test_bayesian_rating_is_normalized_to_unit_range():
    assert bayesian_rating(1000, 1.0, 3.5, CONFIG) == pytest.approx(0.0, abs=0.02)
    assert bayesian_rating(1000, 5.0, 3.5, CONFIG) == pytest.approx(1.0, abs=0.02)


def test_completions_outweigh_likes():
    """Finishing a book is the strongest signal the platform emits."""
    with_likes = engagement(likes=3, completions=0, p95_engagement=100, config=CONFIG)
    with_completions = engagement(
        likes=0, completions=3, p95_engagement=100, config=CONFIG
    )

    assert with_completions > with_likes


def test_engagement_completion_multiplier_is_exact():
    """3 completions == 9 likes at pop_completion_mult = 3."""
    a = engagement(likes=9, completions=0, p95_engagement=100, config=CONFIG)
    b = engagement(likes=0, completions=3, p95_engagement=100, config=CONFIG)

    assert a == pytest.approx(b)


def test_engagement_is_log_compressed():
    """Doubling raw engagement must not double the score."""
    low = engagement(10, 0, 1000, CONFIG)
    high = engagement(20, 0, 1000, CONFIG)

    assert high < 2 * low


def test_engagement_clamps_above_the_p95():
    """A viral outlier saturates rather than compressing everyone else."""
    at_p95 = engagement(100, 0, 100, CONFIG)
    way_above = engagement(100_000, 0, 100, CONFIG)

    assert at_p95 == pytest.approx(1.0, abs=1e-6)
    assert way_above == 1.0


def test_engagement_matches_the_formula():
    """log1p(10) / log1p(100) = 2.3979 / 4.6151 = 0.5196"""
    result = engagement(likes=10, completions=0, p95_engagement=100, config=CONFIG)

    assert result == pytest.approx(math.log1p(10) / math.log1p(100), abs=1e-6)


def test_zero_engagement_scores_zero():
    assert engagement(0, 0, 100, CONFIG) == 0.0


def test_popularity_blends_quality_and_volume():
    """0.6 * bayes + 0.4 * engagement."""
    result = popularity(
        likes=10,
        ratings_count=500,
        avg_rating=5.0,
        completions=0,
        global_mean_rating=3.5,
        p95_engagement=100,
        config=CONFIG,
    )
    expected = 0.6 * bayesian_rating(500, 5.0, 3.5, CONFIG) + 0.4 * engagement(
        10, 0, 100, CONFIG
    )

    assert result == pytest.approx(expected, abs=1e-6)


def test_popularity_stays_in_unit_range():
    result = popularity(10**9, 10**9, 5.0, 10**9, 3.5, 10, CONFIG)

    assert 0.0 <= result <= 1.0


# ══════════════════════════════════════════════════════════════════════════
# Term 3: collaborative filtering (distinct from popularity)
# ══════════════════════════════════════════════════════════════════════════


def test_shrinkage_stops_a_one_of_one_cooccurrence_scoring_perfectly():
    """Without lambda, two obscure books sharing one reader score 1.0 and outrank
    genuinely related pairs."""
    obscure = cooccurrence_similarity(cooc=1, n_a=1, n_b=1, config=CONFIG)

    assert obscure < 0.15


def test_well_evidenced_pair_beats_a_coincidence():
    coincidence = cooccurrence_similarity(1, 1, 1, CONFIG)
    real = cooccurrence_similarity(200, 300, 300, CONFIG)

    assert real > coincidence


def test_cooccurrence_matches_the_formula():
    """100 / (sqrt(200*300) + 10) = 100 / (244.949 + 10) = 0.39223"""
    result = cooccurrence_similarity(100, 200, 300, CONFIG)

    assert result == pytest.approx(100 / (math.sqrt(60000) + 10), abs=1e-6)


def test_cooccurrence_of_zero_is_zero():
    assert cooccurrence_similarity(0, 100, 100, CONFIG) == 0.0


def test_collaborative_score_is_a_weighted_mean():
    """(1.0*0.8 + 0.4*0.2) / 1.4 = 0.88/1.4 = 0.6286"""
    result = collaborative_score(
        similarities={1: 0.8, 2: 0.2}, seed_weights={1: 1.0, 2: 0.4}
    )

    assert result == pytest.approx(0.88 / 1.4, abs=1e-6)


def test_one_strong_cooccurrence_cannot_carry_a_candidate():
    """A mean, not a max — otherwise a single shared reader dominates."""
    result = collaborative_score(
        similarities={1: 1.0, 2: 0.0, 3: 0.0}, seed_weights={1: 1.0, 2: 1.0, 3: 1.0}
    )

    assert result == pytest.approx(1 / 3)


def test_missing_similarity_counts_as_zero():
    result = collaborative_score(similarities={}, seed_weights={1: 1.0})

    assert result == 0.0


def test_empty_seed_set_scores_zero():
    assert collaborative_score({1: 0.9}, {}) == 0.0


def test_zero_weight_seeds_are_skipped():
    result = collaborative_score({1: 0.9, 2: 0.1}, {1: 1.0, 2: 0.0})

    assert result == pytest.approx(0.9)


# ── Seed weights ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "kind,value,expected",
    [
        ("like", None, 1.0),
        ("rating", 5, 1.0),
        ("rating", 4, 1.0),
        ("rating", 3, 0.4),
        ("rating", 2, 0.0),
        ("rating", 1, 0.0),
        ("completion", None, 0.8),
        ("progress", 1.0, 0.4),
        ("progress", 0.5, 0.2),
        ("unknown", None, 0.0),
    ],
)
def test_seed_engagement_weights(kind, value, expected):
    assert seed_engagement_weight(kind, value) == pytest.approx(expected)


def test_disliked_books_carry_no_seed_weight():
    """Treating a rejected book as a taste anchor would recommend more of it."""
    assert seed_engagement_weight("rating", 1) == 0.0
    assert seed_engagement_weight("rating", 2) == 0.0


@pytest.mark.parametrize(
    "kind,value,expected",
    [
        ("rating", 1, True),
        ("rating", 2, True),
        ("rating", 3, False),
        ("rating", None, False),
        ("like", None, False),
    ],
)
def test_negative_signal_detection(kind, value, expected):
    assert is_negative_signal(kind, value) is expected


# ══════════════════════════════════════════════════════════════════════════
# The blend
# ══════════════════════════════════════════════════════════════════════════


def test_cold_start_score_is_exactly_the_semantic_score():
    result = blend(
        semantic=0.8,
        popularity_score=0.1,
        collaborative=0.0,
        n_interactions=0,
        config=CONFIG,
    )

    assert result.score == pytest.approx(0.8)
    assert result.w_semantic == 1.0


def test_established_item_blends_in_popularity():
    """n=20 → alpha=0.5, w_pop=0.5*0.25=0.125, w_sem=0.875.
    score = 0.875*0.8 + 0.125*1.0 = 0.7 + 0.125 = 0.825"""
    result = blend(
        semantic=0.8,
        popularity_score=1.0,
        collaborative=0.0,
        n_interactions=20,
        config=CONFIG,
    )

    assert result.alpha == pytest.approx(0.5)
    assert result.w_popularity == pytest.approx(0.125)
    assert result.score == pytest.approx(0.825)


def test_cf_contributes_nothing_while_stubbed():
    """Same score whatever the CF value, because w_cf_ceiling is 0."""
    kwargs = dict(
        semantic=0.5,
        popularity_score=0.5,
        n_interactions=100,
        config=CONFIG,
    )

    without = blend(collaborative=0.0, **kwargs)
    with_cf = blend(collaborative=1.0, **kwargs)

    assert without.score == pytest.approx(with_cf.score)
    assert with_cf.w_collaborative == 0.0


def test_cf_contributes_once_enabled():
    """The stub keeps the shape, so enabling it is a config change only."""
    config = ScoringConfig(w_pop_ceiling=0.25, w_cf_ceiling=0.20)
    kwargs = dict(
        semantic=0.5,
        popularity_score=0.5,
        n_interactions=1_000_000,
        config=config,
    )

    without = blend(collaborative=0.0, **kwargs)
    with_cf = blend(collaborative=1.0, **kwargs)

    assert with_cf.score > without.score


def test_score_stays_in_unit_range():
    result = blend(1.0, 1.0, 1.0, 10**9, CONFIG)

    assert 0.0 <= result.score <= 1.0


def test_inputs_are_clamped():
    result = blend(5.0, -2.0, 9.0, 0, CONFIG)

    assert result.semantic == 1.0
    assert result.popularity == 0.0
    assert result.collaborative == 1.0


def test_breakdown_exposes_every_input():
    """A surprising ranking should be explainable from the response, not by
    re-deriving the arithmetic."""
    payload = blend(0.8, 0.5, 0.0, 20, CONFIG).as_dict()

    assert set(payload) == {
        "score",
        "semantic",
        "popularity",
        "collaborative",
        "weights",
        "alpha",
    }
    assert set(payload["weights"]) == {"semantic", "popularity", "collaborative"}


# ══════════════════════════════════════════════════════════════════════════
# Catalog statistics
# ══════════════════════════════════════════════════════════════════════════


def test_catalog_stats_defaults_are_sane():
    stats = CatalogStats()

    assert 1 <= stats.global_mean_rating <= 5
    assert stats.p95_engagement > 0


def test_catalog_stats_from_row():
    stats = CatalogStats.from_row({"global_mean_rating": 4.1, "p95_engagement": 55})

    assert stats.global_mean_rating == 4.1
    assert stats.p95_engagement == 55.0


def test_catalog_stats_from_empty_row_uses_defaults():
    """An empty catalog returns NULL aggregates; those must not become zeros, or
    every rating would normalize against a mean of 0."""
    assert CatalogStats.from_row(None).global_mean_rating == 3.5
    assert CatalogStats.from_row({}).global_mean_rating == 3.5
    assert (
        CatalogStats.from_row(
            {"global_mean_rating": None, "p95_engagement": None}
        ).p95_engagement
        == 10.0
    )


@pytest.mark.parametrize(
    "values,expected",
    [([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 10), ([5], 5), ([], 0.0)],
)
def test_percentile(values, expected):
    assert percentile(values, 0.95) == expected


def test_percentile_ignores_none():
    """None values are dropped, not coerced to 0 — an item with no stats row must
    not drag the P95 engagement ceiling down."""
    assert percentile([1, None, 3, None, 5], 1.0) == 5
    assert percentile([None, None], 0.95) == 0.0
    # Equivalent to the same list with the Nones removed.
    assert percentile([10, None, 20, 30], 0.95) == percentile([10, 20, 30], 0.95)
