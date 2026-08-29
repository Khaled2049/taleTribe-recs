"""The ranking formula: three distinct signals, blended by a cold-start ramp.

    score(i, u) = src(i) · [ w_sem(i)·sem(i) + w_pop(i)·pop(i) + w_cf(i)·cf(i,u) ]

The three terms are kept **separate on purpose**, because they answer different
questions and fail in different ways:

* `sem`  — semantic similarity. Does this book resemble what was asked for?
* `pop`  — a popularity/engagement prior. Do readers in general finish and rate
           it well? Volume-damped, so 2 ratings cannot outrank 500.
* `cf`   — item-item collaborative filtering. Do readers who engaged with *this
           particular reader's* books also engage with it?

`pop` and `cf` are frequently conflated and must not be: `pop` is a global
property of the item, while `cf` is a property of the *pair* (item, reader). A
globally unpopular book can have a very high `cf` for one reader.

`cf` currently contributes nothing — `w_cf_ceiling` is 0 in the `config` table,
because the platform has no impression/click log and co-occurrence built from
binary likes alone is too sparse to beat the popularity prior. The term, its
table and its query are all real, so enabling it later is a config UPDATE plus a
backfill rather than a rescoring rewrite.
"""

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence

# Fallback knob values, used when a key is missing from recommendations.config.
# The table is the source of truth; these keep the service ranking sensibly
# rather than raising if a migration has not seeded a new knob yet.
DEFAULTS: Dict[str, float] = {
    "w_pop_ceiling": 0.25,
    "w_cf_ceiling": 0.00,
    "ramp_n_min": 5,
    "ramp_n50": 20,
    "bayes_prior_c": 20,
    "cf_shrinkage_lambda": 10,
    "pop_w_bayes": 0.60,
    "pop_w_engagement": 0.40,
    "pop_completion_mult": 3.00,
    "rrf_k": 60,
    "mmr_lambda": 0.70,
}

# Semantic similarity must never stop being the dominant signal — vector search
# does the ranking and the other terms only adjust it. Asserted in
# ScoringConfig so a bad tune in the database cannot quietly invert the design.
MAX_BEHAVIORAL_WEIGHT = 0.45


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class ScoringConfig:
    """Knobs, resolved once per request from `recommendations.config`."""

    w_pop_ceiling: float = DEFAULTS["w_pop_ceiling"]
    w_cf_ceiling: float = DEFAULTS["w_cf_ceiling"]
    ramp_n_min: float = DEFAULTS["ramp_n_min"]
    ramp_n50: float = DEFAULTS["ramp_n50"]
    bayes_prior_c: float = DEFAULTS["bayes_prior_c"]
    cf_shrinkage_lambda: float = DEFAULTS["cf_shrinkage_lambda"]
    pop_w_bayes: float = DEFAULTS["pop_w_bayes"]
    pop_w_engagement: float = DEFAULTS["pop_w_engagement"]
    pop_completion_mult: float = DEFAULTS["pop_completion_mult"]
    rrf_k: float = DEFAULTS["rrf_k"]
    mmr_lambda: float = DEFAULTS["mmr_lambda"]

    @classmethod
    def from_mapping(cls, values: Optional[Mapping[str, float]]) -> "ScoringConfig":
        merged = dict(DEFAULTS)
        merged.update({k: v for k, v in (values or {}).items() if k in DEFAULTS})

        # Clamp rather than raise: the config table is editable at runtime, and a
        # typo there should degrade ranking, not take the service down.
        pop_ceiling = _clamp(float(merged["w_pop_ceiling"]))
        cf_ceiling = _clamp(float(merged["w_cf_ceiling"]))
        total = pop_ceiling + cf_ceiling
        if total > MAX_BEHAVIORAL_WEIGHT:
            scale = MAX_BEHAVIORAL_WEIGHT / total
            pop_ceiling *= scale
            cf_ceiling *= scale

        return cls(
            w_pop_ceiling=pop_ceiling,
            w_cf_ceiling=cf_ceiling,
            ramp_n_min=max(0.0, float(merged["ramp_n_min"])),
            ramp_n50=max(1e-9, float(merged["ramp_n50"])),
            bayes_prior_c=max(0.0, float(merged["bayes_prior_c"])),
            cf_shrinkage_lambda=max(0.0, float(merged["cf_shrinkage_lambda"])),
            pop_w_bayes=_clamp(float(merged["pop_w_bayes"])),
            pop_w_engagement=_clamp(float(merged["pop_w_engagement"])),
            pop_completion_mult=max(0.0, float(merged["pop_completion_mult"])),
            rrf_k=max(1.0, float(merged["rrf_k"])),
            mmr_lambda=_clamp(float(merged["mmr_lambda"])),
        )


# ══════════════════════════════════════════════════════════════════════════
# The cold-start ramp
# ══════════════════════════════════════════════════════════════════════════


def behavioral_ramp(n_interactions: float, config: ScoringConfig) -> float:
    """How much to trust this item's own behavioral data, in [0, 1].

        α = 0                      when n < n_min      (100% semantic)
        α = n / (n + n50)          otherwise

    The hard gate at `n_min` is what delivers "cold-start books weight 100% on
    semantic similarity": below a handful of interactions the popularity estimate
    is noise, and a saturating curve alone would still give it a little weight.

    Saturating rather than linear above the gate, because a linear ramp is most
    aggressive exactly where the data is thinnest. `n50` is the interaction count
    at which behavioral reaches half its ceiling, and it deliberately equals
    `bayes_prior_c` — both encode the same belief: about 20 interactions is where
    an item's own numbers start to mean something.
    """
    n = max(0.0, float(n_interactions))
    if n < config.ramp_n_min:
        return 0.0
    return n / (n + config.ramp_n50)


def term_weights(alpha: float, config: ScoringConfig) -> tuple:
    """Return (w_sem, w_pop, w_cf), which always sum to 1.

    Summing to 1 keeps scores comparable across items with very different
    interaction volumes — without it, a well-known book would score higher purely
    because more terms contributed.
    """
    alpha = _clamp(alpha)
    w_pop = alpha * config.w_pop_ceiling
    w_cf = alpha * config.w_cf_ceiling
    return (1.0 - w_pop - w_cf, w_pop, w_cf)


# ══════════════════════════════════════════════════════════════════════════
# Term 2: popularity / engagement prior
# ══════════════════════════════════════════════════════════════════════════


def bayesian_rating(
    ratings_count: float,
    avg_rating: Optional[float],
    global_mean_rating: float,
    config: ScoringConfig,
) -> float:
    """Volume-damped rating, normalized from the 1-5 scale to [0, 1].

        (n·avg + C·m) / (n + C),  then (x - 1) / 4

    A book with one 5-star rating is pulled almost all the way back to the global
    mean; a book with 500 ratings is barely moved. Without this, every new item
    with a single generous rating would top the charts.
    """
    if avg_rating is None or ratings_count <= 0:
        shrunk = global_mean_rating
    else:
        n = max(0.0, float(ratings_count))
        shrunk = (n * float(avg_rating) + config.bayes_prior_c * global_mean_rating) / (
            n + config.bayes_prior_c
        )
    return _clamp((shrunk - 1.0) / 4.0)


def engagement(
    likes: float,
    completions: float,
    p95_engagement: float,
    config: ScoringConfig,
) -> float:
    """Log-compressed engagement volume, normalized to [0, 1].

    A completion counts for `pop_completion_mult` likes — finishing a book is by
    far the strongest signal the platform emits, and a like is one tap.

    Normalized by the **95th percentile**, not the maximum: one viral outlier
    would otherwise compress every other book into the bottom of the range. Values
    above P95 clamp to 1.0, which is the intended behaviour — the difference
    between "extremely popular" and "the single most popular" should not drive
    ranking.
    """
    raw = max(0.0, float(likes)) + config.pop_completion_mult * max(
        0.0, float(completions)
    )
    ceiling = math.log1p(max(1.0, float(p95_engagement)))
    return _clamp(math.log1p(raw) / ceiling) if ceiling > 0 else 0.0


def popularity(
    likes: float,
    ratings_count: float,
    avg_rating: Optional[float],
    completions: float,
    global_mean_rating: float,
    p95_engagement: float,
    config: ScoringConfig,
) -> float:
    """The popularity prior in [0, 1] — quality and volume, weighted."""
    quality = bayesian_rating(ratings_count, avg_rating, global_mean_rating, config)
    volume = engagement(likes, completions, p95_engagement, config)
    total = config.pop_w_bayes + config.pop_w_engagement
    if total <= 0:
        return 0.0
    return _clamp(
        (config.pop_w_bayes * quality + config.pop_w_engagement * volume) / total
    )


# ══════════════════════════════════════════════════════════════════════════
# Term 3: item-item collaborative filtering  (stubbed: w_cf_ceiling == 0)
# ══════════════════════════════════════════════════════════════════════════


def cooccurrence_similarity(
    cooc: float, n_a: float, n_b: float, config: ScoringConfig
) -> float:
    """Shrunk cosine over co-occurrence counts.

        sim = cooc / (sqrt(n_a · n_b) + λ)

    λ is doing the important work. Without it, two obscure books read by exactly
    one shared reader score a perfect 1.0 and outrank genuinely related pairs
    with hundreds of shared readers.
    """
    denominator = math.sqrt(max(0.0, n_a) * max(0.0, n_b)) + config.cf_shrinkage_lambda
    if denominator <= 0:
        return 0.0
    return _clamp(max(0.0, float(cooc)) / denominator)


def collaborative_score(
    similarities: Mapping[int, float],
    seed_weights: Mapping[int, float],
) -> float:
    """Weighted mean similarity to the reader's seed items, in [0, 1].

    `similarities` maps seed item id -> sim(candidate, seed); `seed_weights` maps
    seed item id -> how strongly the reader engaged with it (a like or a 4+ rating
    counts fully, a completion 0.8, a partial read proportionally).

    A weighted *mean* rather than a max, so one strong co-occurrence cannot carry
    a candidate on its own.
    """
    numerator = 0.0
    denominator = 0.0
    for seed_id, weight in seed_weights.items():
        weight = max(0.0, float(weight))
        if weight <= 0:
            continue
        denominator += weight
        numerator += weight * max(0.0, float(similarities.get(seed_id, 0.0)))
    if denominator <= 0:
        return 0.0
    return _clamp(numerator / denominator)


# ══════════════════════════════════════════════════════════════════════════
# The blend
# ══════════════════════════════════════════════════════════════════════════


@dataclass
class ScoreBreakdown:
    """Per-item score with every input exposed.

    Returned rather than a bare float so a surprising ranking can be explained
    from the response instead of by re-deriving it — and so the eval harness can
    attribute a regression to a specific term.
    """

    score: float
    semantic: float
    popularity: float
    collaborative: float
    w_semantic: float
    w_popularity: float
    w_collaborative: float
    alpha: float

    def as_dict(self) -> dict:
        return {
            "score": round(self.score, 6),
            "semantic": round(self.semantic, 6),
            "popularity": round(self.popularity, 6),
            "collaborative": round(self.collaborative, 6),
            "weights": {
                "semantic": round(self.w_semantic, 4),
                "popularity": round(self.w_popularity, 4),
                "collaborative": round(self.w_collaborative, 4),
            },
            "alpha": round(self.alpha, 4),
        }


def blend(
    semantic: float,
    popularity_score: float,
    collaborative: float,
    n_interactions: float,
    config: ScoringConfig,
) -> ScoreBreakdown:
    """Combine the three terms into a final score in [0, 1]."""
    alpha = behavioral_ramp(n_interactions, config)
    w_sem, w_pop, w_cf = term_weights(alpha, config)

    sem = _clamp(semantic)
    pop = _clamp(popularity_score)
    cf = _clamp(collaborative)

    score = w_sem * sem + w_pop * pop + w_cf * cf

    return ScoreBreakdown(
        score=_clamp(score),
        semantic=sem,
        popularity=pop,
        collaborative=cf,
        w_semantic=w_sem,
        w_popularity=w_pop,
        w_collaborative=w_cf,
        alpha=alpha,
    )


# ══════════════════════════════════════════════════════════════════════════
# Global statistics the popularity term needs
# ══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class CatalogStats:
    """Corpus-level values, refreshed on a schedule rather than per request."""

    global_mean_rating: float = 3.5
    p95_engagement: float = 10.0

    @classmethod
    def from_row(cls, row: Optional[Mapping]) -> "CatalogStats":
        if not row:
            return cls()
        return cls(
            # 3.5 rather than 3.0: platform ratings skew high, and a prior below
            # the true mean would penalise every lightly-rated book.
            global_mean_rating=float(row.get("global_mean_rating") or 3.5),
            p95_engagement=float(row.get("p95_engagement") or 10.0),
        )


def percentile(values: Sequence[float], fraction: float = 0.95) -> float:
    """Nearest-rank percentile.

    Hand-rolled rather than numpy so the popularity refresh has no array
    dependency for a one-line calculation.
    """
    cleaned = sorted(float(v) for v in values if v is not None)
    if not cleaned:
        return 0.0
    index = int(round(_clamp(fraction) * (len(cleaned) - 1)))
    return cleaned[index]


def seed_engagement_weight(kind: str, value: Optional[float] = None) -> float:
    """How strongly a reader engaged with an item, for the CF seed set.

    A negative signal (a rating of 2 or below) returns 0.0 and the item belongs
    on the suppression list instead — treating a disliked book as a taste anchor
    would recommend more of what the reader rejected.
    """
    if kind == "like":
        return 1.0
    if kind == "rating":
        if value is None:
            return 0.0
        if value >= 4:
            return 1.0
        if value >= 3:
            return 0.4
        return 0.0
    if kind == "completion":
        return 0.8
    if kind == "progress":
        return 0.4 * _clamp(float(value or 0.0))
    return 0.0


def is_negative_signal(kind: str, value: Optional[float]) -> bool:
    """True when the reader actively disliked the item."""
    return kind == "rating" and value is not None and float(value) <= 2


__all__: Iterable[str] = [
    "DEFAULTS",
    "MAX_BEHAVIORAL_WEIGHT",
    "CatalogStats",
    "ScoreBreakdown",
    "ScoringConfig",
    "bayesian_rating",
    "behavioral_ramp",
    "blend",
    "collaborative_score",
    "cooccurrence_similarity",
    "engagement",
    "is_negative_signal",
    "percentile",
    "popularity",
    "seed_engagement_weight",
    "term_weights",
]
