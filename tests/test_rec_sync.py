"""Unit tests for the interaction record format and the synthetic generator.

The loader and aggregation need a database and live in test_rec_sync_integration.py.
"""

import os
import random
from datetime import datetime, timezone

import pytest

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "test-project")

from recommendation_engine.sync import synthetic  # noqa: E402
from recommendation_engine.sync.interactions import (  # noqa: E402
    KIND_COMPLETION,
    KIND_LIKE,
    KIND_PROGRESS,
    KIND_RATING,
    SYNTHETIC_USER_PREFIX,
    VALID_KINDS,
    InteractionRecord,
)

pytestmark = pytest.mark.unit


def _record(**overrides):
    base = dict(
        user_id="synth_00001",
        story_id="story-620",
        kind=KIND_PROGRESS,
        occurred_at=datetime(2026, 7, 14, 9, 31, tzinfo=timezone.utc),
        value=0.5,
        chapter_index=3,
        total_chapters=12,
    )
    base.update(overrides)
    return InteractionRecord(**base)


# ══════════════════════════════════════════════════════════════════════════
# The canonical format
# ══════════════════════════════════════════════════════════════════════════


def test_only_kinds_the_platform_records_are_valid():
    """The format must not promise data the platform cannot produce."""
    assert VALID_KINDS == {KIND_LIKE, KIND_RATING, KIND_PROGRESS}
    assert KIND_COMPLETION not in VALID_KINDS, "completion is derived, never supplied"


# ══════════════════════════════════════════════════════════════════════════
# Completion inference
# ══════════════════════════════════════════════════════════════════════════


def test_last_chapter_and_scrolled_is_complete():
    assert _record(chapter_index=11, total_chapters=12, value=0.95).is_complete


def test_last_chapter_but_barely_scrolled_is_not_complete():
    """Opening the final chapter is not finishing the book."""
    assert not _record(chapter_index=11, total_chapters=12, value=0.2).is_complete


def test_scrolled_but_not_last_chapter_is_not_complete():
    assert not _record(chapter_index=3, total_chapters=12, value=1.0).is_complete


def test_only_progress_records_can_be_complete():
    assert not _record(kind=KIND_LIKE).is_complete
    assert not _record(kind=KIND_RATING, value=5.0).is_complete


def test_missing_chapter_metadata_is_not_complete():
    assert not _record(total_chapters=None).is_complete
    assert not _record(chapter_index=None).is_complete
    assert not _record(total_chapters=0).is_complete


# ══════════════════════════════════════════════════════════════════════════
# The synthetic generator
# ══════════════════════════════════════════════════════════════════════════


def _catalog(n=60):
    genres = ["science-fiction", "fantasy", "romance", "horror"]
    themes = ["revenge", "isolation", "found family", "prophecy"]
    return [
        synthetic.CatalogItem(
            story_id=f"story-{i}",
            genres=[genres[i % len(genres)]],
            themes=[themes[i % len(themes)]],
        )
        for i in range(n)
    ]


def test_generation_is_reproducible():
    """Same seed, same data — otherwise eval runs are not comparable."""
    a = list(synthetic.generate(_catalog(), readers=10, seed=7))
    b = list(synthetic.generate(_catalog(), readers=10, seed=7))

    assert a == b


def test_different_seeds_give_different_data():
    a = list(synthetic.generate(_catalog(), readers=10, seed=1))
    b = list(synthetic.generate(_catalog(), readers=10, seed=2))

    assert a != b


def test_every_reader_is_prefixed_so_they_can_be_purged():
    records = list(synthetic.generate(_catalog(), readers=5, seed=1))

    assert records
    assert all(r.user_id.startswith(SYNTHETIC_USER_PREFIX) for r in records)


def test_only_valid_kinds_are_emitted():
    records = list(synthetic.generate(_catalog(), readers=20, seed=3))

    assert {r.kind for r in records} <= VALID_KINDS


def test_records_reference_real_catalog_items():
    """Keyed on story_id because the internal id is a surrogate the
    generator cannot know."""
    catalog = _catalog()
    known = {item.story_id for item in catalog}

    records = list(synthetic.generate(catalog, readers=20, seed=4))

    assert all(r.story_id in known for r in records)


def test_ratings_are_within_range():
    records = list(synthetic.generate(_catalog(), readers=30, seed=5))
    ratings = [r.value for r in records if r.kind == KIND_RATING]

    assert ratings
    assert all(1 <= value <= 5 for value in ratings)
    assert all(float(value).is_integer() for value in ratings)


def test_scroll_percent_is_a_fraction():
    records = list(synthetic.generate(_catalog(), readers=20, seed=6))
    progress = [r for r in records if r.kind == KIND_PROGRESS]

    assert progress
    assert all(0.0 <= r.value <= 1.0 for r in progress)
    assert all(0 <= r.chapter_index < r.total_chapters for r in progress)


def test_popularity_is_a_long_tail_not_uniform():
    """The property that makes the volume damping and P95 normalization actually
    get exercised — uniform counts would leave both untested."""
    records = list(synthetic.generate(_catalog(80), readers=150, seed=8))

    counts: dict = {}
    for record in records:
        counts[record.story_id] = counts.get(record.story_id, 0) + 1
    ordered = sorted(counts.values(), reverse=True)

    assert len(ordered) > 20
    # The most-touched book should be several times the median.
    median = ordered[len(ordered) // 2]
    assert ordered[0] >= median * 3


def test_readers_have_distinguishable_tastes():
    """If every reader had the same taste, behavioral mode could never be seen to
    personalize and co-occurrence would be meaningless."""
    records = list(synthetic.generate(_catalog(80), readers=40, seed=9))

    by_user: dict = {}
    for record in records:
        by_user.setdefault(record.user_id, set()).add(record.story_id)

    users = [items for items in by_user.values() if len(items) >= 3]
    assert len(users) >= 10
    # At least some pairs of readers should barely overlap.
    disjoint = sum(
        1 for i, a in enumerate(users) for b in users[i + 1 :] if not (a & b)
    )
    assert disjoint > 0


def test_finishing_correlates_with_higher_ratings():
    """A popularity prior blending rating quality and completion volume would be
    blending noise if the two were independent."""
    records = list(synthetic.generate(_catalog(80), readers=200, seed=10))

    completed = set()
    ratings: dict = {}
    for record in records:
        key = (record.user_id, record.story_id)
        if record.is_complete:
            completed.add(key)
        if record.kind == KIND_RATING:
            ratings[key] = record.value

    finished = [v for k, v in ratings.items() if k in completed]
    abandoned = [v for k, v in ratings.items() if k not in completed]

    assert finished and abandoned
    assert sum(finished) / len(finished) > sum(abandoned) / len(abandoned)


def test_generator_refuses_an_empty_catalog():
    with pytest.raises(ValueError, match="no catalog items"):
        list(synthetic.generate([], readers=5))


def test_personas_draw_affinities_from_the_real_catalog():
    """A persona who loves a genre the catalog lacks would generate nothing."""
    catalog = _catalog()
    available = {g for item in catalog for g in item.genres}

    personas = synthetic.build_personas(catalog, 25, random.Random(11))

    assert len(personas) == 25
    assert all(set(p.genres) <= available for p in personas)
    assert all(p.target_count >= 1 for p in personas)


def test_persona_activity_tiers_are_mixed():
    personas = synthetic.build_personas(_catalog(), 200, random.Random(12))

    tiers = {p.activity for p in personas}
    assert len(tiers) >= 2, "all readers in one activity tier is not a realistic mix"


def test_personas_can_be_supplied_explicitly():
    """So hand-authored or LLM-authored personas can be used instead."""
    catalog = _catalog()
    persona = synthetic.Persona(
        user_id="synth_custom",
        genres=["horror"],
        themes=["isolation"],
        activity="medium",
        target_count=4,
    )

    records = list(synthetic.generate(catalog, seed=13, personas=[persona]))

    assert records
    assert {r.user_id for r in records} == {"synth_custom"}
