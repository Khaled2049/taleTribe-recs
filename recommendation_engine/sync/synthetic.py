"""Synthetic reader behaviour, for developing against before real users exist.

**Read this before trusting any number computed on this data.** Synthetic
interactions can validate *mechanics* — does the popularity term move, does the
cold-start ramp engage, does a taste vector retrieve sensibly, does the
co-occurrence rebuild produce a usable table. They cannot validate *taste*. Any
offline quality metric computed here measures whether the recommender recovers this
generator's own assumptions, which is circular. Treat it exactly like
`USE_MOCK=true` embeddings: the plumbing is real, the semantics are not.

Every synthetic reader is prefixed `synth_` so they can be purged and never quietly
enter a real measurement.

## Why it is generated in code rather than by a language model

* It must reference `story_id`s that exist in *this* catalog.
* It needs statistical structure a model will not hold over thousands of records:
  a **power-law** popularity distribution, per-reader genre/theme affinity, and the
  co-occurrence that emerges from readers sharing affinities.
* It must be reproducible from a seed, or eval runs are not comparable.

The power law matters more than it looks. With uniform interaction counts the
popularity term would be flat, the Bayesian damping would never be exercised, and
P95 normalization would be meaningless. A realistic long tail is what makes those
mechanisms actually get tested.
"""

import logging
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from recommendation_engine.sync.interactions import (
    KIND_LIKE,
    KIND_PROGRESS,
    KIND_RATING,
    SYNTHETIC_USER_PREFIX,
    InteractionRecord,
)

logger = logging.getLogger(__name__)

# Readers are not uniformly active. Roughly: a few enthusiasts, a working middle,
# and a majority who read a couple of things.
ACTIVITY_TIERS = (
    ("light", 0.55, (2, 5)),
    ("medium", 0.35, (6, 15)),
    ("heavy", 0.10, (16, 40)),
)

# How strongly a reader sticks to their preferred genres. The remainder is
# exploration — without it every reader is trapped in one genre, no cross-genre
# co-occurrence ever appears, and the diversity metric has nothing to measure.
AFFINITY_ADHERENCE = 0.75

# Timestamps are anchored to a fixed reference instant by default, so `generate()`
# is a pure function of its seed. Using wall-clock time here silently broke
# reproducibility — identical seeds produced different `occurred_at` values — which
# would have made two eval runs incomparable for no visible reason.
#
# Nothing in scoring reads `occurred_at` today; it exists for idempotency on reload
# and for future recency weighting. So a fixed anchor costs nothing. Pass `now=` to
# get wall-clock recency instead, and give up reproducibility knowingly.
REFERENCE_NOW = datetime(2026, 6, 1, tzinfo=timezone.utc)


@dataclass
class CatalogItem:
    """The minimum a generator needs to know about a book."""

    story_id: str
    genres: list[str] = field(default_factory=list)
    themes: list[str] = field(default_factory=list)


@dataclass
class Persona:
    """A synthetic reader's taste. Hand-authorable or LLM-authorable if wanted."""

    user_id: str
    genres: list[str]
    themes: list[str]
    activity: str
    target_count: int
    # Some readers rate generously, some harshly. Without this every average rating
    # converges and the Bayesian prior has nothing to shrink toward.
    generosity: float = 0.0

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def build_personas(
    items: Sequence[CatalogItem],
    count: int,
    rng: random.Random,
) -> list[Persona]:
    """Invent readers whose tastes are drawn from what the catalog actually holds.

    Sampling affinities from real catalog genres/themes rather than a fixed list
    means the personas are always satisfiable — a persona who loves a genre the
    catalog lacks would generate nothing.
    """
    genre_pool = sorted({g for item in items for g in item.genres})
    theme_pool = sorted({t for item in items for t in item.themes})
    if not genre_pool:
        raise ValueError(
            "catalog has no genres; load the corpus before generating readers"
        )

    personas: list[Persona] = []
    for index in range(count):
        tier = _weighted_choice(
            [(name, weight) for name, weight, _ in ACTIVITY_TIERS], rng
        )
        low, high = next(r for name, _, r in ACTIVITY_TIERS if name == tier)
        personas.append(
            Persona(
                user_id=f"{SYNTHETIC_USER_PREFIX}{index:05d}",
                genres=rng.sample(
                    genre_pool, k=min(len(genre_pool), rng.randint(1, 3))
                ),
                themes=(
                    rng.sample(theme_pool, k=min(len(theme_pool), rng.randint(2, 5)))
                    if theme_pool
                    else []
                ),
                activity=tier,
                target_count=rng.randint(low, high),
                generosity=rng.uniform(-0.6, 0.6),
            )
        )
    return personas


def _weighted_choice(pairs, rng: random.Random):
    total = sum(w for _, w in pairs)
    point = rng.uniform(0, total)
    upto = 0.0
    for value, weight in pairs:
        upto += weight
        if point <= upto:
            return value
    return pairs[-1][0]


def _popularity_weights(
    items: Sequence[CatalogItem], rng: random.Random
) -> list[float]:
    """Assign each book a latent popularity on a Zipf-like curve.

    This is what produces a realistic long tail: a handful of books collect many
    interactions, most collect a few. Independent of genre, so a reader's *choice*
    is driven by affinity while *how often a book is chosen at all* is driven by
    popularity — which is how real catalogs behave.
    """
    order = list(range(len(items)))
    rng.shuffle(order)
    weights = [0.0] * len(items)
    for rank, index in enumerate(order, start=1):
        weights[index] = 1.0 / (rank**0.8)
    return weights


def generate(
    items: Sequence[CatalogItem],
    readers: int = 200,
    seed: int = 1234,
    days: int = 180,
    personas: Optional[Sequence[Persona]] = None,
    now: Optional[datetime] = None,
) -> Iterator[InteractionRecord]:
    """Yield interaction records for `readers` synthetic readers.

    Correlations are deliberate, because the aggregates downstream depend on them:

    * A reader who finishes a book is much more likely to like or rate it well.
    * Abandoning a book early correlates with a low rating.
    * Preferred-genre books get better ratings than exploratory picks.

    Without those, `avg_rating` would be independent of completion, and a popularity
    prior blending the two would be blending noise.
    """
    if not items:
        raise ValueError("no catalog items; load the corpus first")

    rng = random.Random(seed)
    people = list(personas) if personas else build_personas(items, readers, rng)
    weights = _popularity_weights(items, rng)

    # Affinity index, so a reader's candidate pool is a lookup rather than a scan
    # over the whole catalog per pick.
    by_genre: dict[str, list[int]] = {}
    for index, item in enumerate(items):
        for genre in item.genres:
            by_genre.setdefault(genre, []).append(index)

    now = now or REFERENCE_NOW

    for persona in people:
        preferred: list[int] = []
        for genre in persona.genres:
            preferred.extend(by_genre.get(genre, ()))
        preferred = list(dict.fromkeys(preferred))

        chosen: set = set()
        attempts = 0
        while (
            len(chosen) < persona.target_count and attempts < persona.target_count * 12
        ):
            attempts += 1
            in_affinity = bool(preferred) and rng.random() < AFFINITY_ADHERENCE
            pool = preferred if in_affinity else range(len(items))
            index = _sample_by_weight(pool, weights, rng)
            if index in chosen:
                continue
            chosen.add(index)

            item = items[index]
            occurred = now - timedelta(
                days=rng.uniform(0, days), hours=rng.uniform(0, 24)
            )

            # Theme overlap sharpens the affinity signal beyond genre alone.
            theme_overlap = len(set(item.themes) & set(persona.themes))
            affinity = (1.0 if in_affinity else 0.0) + min(theme_overlap, 3) * 0.25

            # How far they got. Higher affinity means more likely to finish.
            finish_probability = 0.25 + 0.35 * min(affinity, 2.0)
            finished = rng.random() < finish_probability

            total_chapters = rng.randint(8, 40)
            if finished:
                chapter_index = total_chapters - 1
                scroll = rng.uniform(0.92, 1.0)
            else:
                chapter_index = rng.randint(0, max(0, total_chapters - 2))
                scroll = rng.uniform(0.05, 0.85)

            yield InteractionRecord(
                user_id=persona.user_id,
                story_id=item.story_id,
                kind=KIND_PROGRESS,
                occurred_at=occurred,
                value=round(scroll, 3),
                chapter_index=chapter_index,
                total_chapters=total_chapters,
            )

            # Finishing something you sought out is what earns a like.
            like_probability = (0.55 if finished else 0.08) + 0.15 * min(affinity, 2.0)
            if rng.random() < min(0.95, like_probability):
                yield InteractionRecord(
                    user_id=persona.user_id,
                    story_id=item.story_id,
                    kind=KIND_LIKE,
                    occurred_at=occurred + timedelta(minutes=rng.randint(1, 240)),
                )

            # Not everyone rates. Those who do rate what they finished more kindly.
            if rng.random() < (0.5 if finished else 0.25):
                base = 3.9 if finished else 2.7
                score = base + 0.5 * affinity + persona.generosity
                score += rng.gauss(0, 0.6)
                rating = max(1, min(5, int(round(score))))
                yield InteractionRecord(
                    user_id=persona.user_id,
                    story_id=item.story_id,
                    kind=KIND_RATING,
                    occurred_at=occurred + timedelta(hours=rng.randint(1, 72)),
                    value=float(rating),
                )


def _sample_by_weight(pool, weights: Sequence[float], rng: random.Random) -> int:
    """Pick one index from `pool`, proportional to latent popularity."""
    indices = list(pool)
    if not indices:
        return rng.randrange(len(weights))
    subtotal = sum(weights[i] for i in indices)
    if subtotal <= 0:
        return rng.choice(indices)
    point = rng.uniform(0, subtotal)
    upto = 0.0
    for index in indices:
        upto += weights[index]
        if point <= upto:
            return index
    return indices[-1]


async def load_catalog(pool, limit: Optional[int] = None) -> list[CatalogItem]:
    """Read the eligible catalog, which is what readers can plausibly interact with."""
    rows = await pool.fetch(
        "SELECT story_id, genres, themes "
        "FROM recommendations.items "
        "WHERE is_eligible AND embedding IS NOT NULL "
        "ORDER BY id" + (f" LIMIT {int(limit)}" if limit else "")
    )
    return [
        CatalogItem(
            story_id=str(row["story_id"]),
            genres=list(row["genres"] or []),
            themes=list(row["themes"] or []),
        )
        for row in rows
    ]
