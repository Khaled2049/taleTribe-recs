"""Unit tests for the platform ingest — the parts that need no database.

The SQL itself is exercised by the integration suite; what is tested here is
the shaping between a story row and an embeddable item, because that is where
a silent quality regression would live.
"""

from datetime import datetime, timezone

import pytest

from recommendation_engine.ingest import platform
from recommendation_engine.ingest.compose import compose_embed_input

pytestmark = pytest.mark.unit

NOW = datetime(2026, 6, 1, tzinfo=timezone.utc)


def _story(**overrides) -> platform.StoryRecord:
    base = dict(
        story_id="3f9c1a2e-5b47-4d18-9f60-1c2b3a4d5e6f",
        title="The Salt Road",
        author="A Writer",
        description="A cartographer walks a road that is not on any map.",
        category="fantasy",
        target_audience="adult",
        language="en",
        tags=["maps", "slow burn"],
        word_count=41000,
        chapter_count=12,
        created_at=NOW,
        updated_at=NOW,
        chapter_summary=None,
    )
    base.update(overrides)
    return platform.StoryRecord(**base)


class _Normalization:
    core_premise = "A mapmaker follows a road that erases itself."
    themes = ["obsession", "isolation"]
    tone = ["dreamlike"]
    confidence = 0.9
    is_reliable = True


# ── summary_text ─────────────────────────────────────────────────────────


def test_description_alone_is_used_when_there_are_no_summaries():
    text = platform.summary_text(_story(tags=[]))

    assert text == "A cartographer walks a road that is not on any map."


def test_chapter_summaries_are_preferred_material_when_present():
    """A description is a blurb; a chapter summary is what actually happens."""
    text = platform.summary_text(_story(chapter_summary="She reaches the coast."))

    assert "She reaches the coast." in text
    assert "cartographer" in text


def test_tags_reach_the_model_but_not_the_genre_filter():
    """Tags are free text, so they inform the extraction without polluting
    `genres`, which is a filter dimension backed by the controlled category."""
    story = _story()

    assert "maps" in platform.summary_text(story)
    assert platform.to_normalized_item(story, None).genres == ["fantasy"]


def test_summary_is_truncated_to_bound_cost():
    story = _story(description="word " * (platform.MAX_SUMMARY_WORDS + 500))

    assert len(platform.summary_text(story).split()) == platform.MAX_SUMMARY_WORDS


def test_a_story_with_nothing_to_say_yields_no_summary():
    """Falsy, so `_process_chunk` never spends a token asking about it."""
    assert platform.summary_text(_story(description="", tags=[])) == ""


# ── to_normalized_item ───────────────────────────────────────────────────


def test_normalization_fields_are_carried_through():
    item = platform.to_normalized_item(_story(), _Normalization())

    assert item.core_premise == "A mapmaker follows a road that erases itself."
    assert item.themes == ["obsession", "isolation"]
    assert item.tone == ["dreamlike"]


def test_missing_normalization_leaves_derived_fields_empty():
    """Not an error: the row is still stored so a later run can fix it."""
    item = platform.to_normalized_item(_story(), None)

    assert item.core_premise is None
    assert item.themes == []
    assert item.tone == []


def test_absent_category_yields_no_genres_rather_than_an_empty_string():
    assert platform.to_normalized_item(_story(category=None), None).genres == []


def test_embed_input_ignores_fields_that_do_not_belong_in_the_vector():
    """word_count and chapter_count are filter metadata. If they reached the
    embed text, publishing one more chapter would change the sha and re-embed
    the whole story for nothing."""
    a = compose_embed_input(platform.to_normalized_item(_story(), _Normalization()))
    b = compose_embed_input(
        platform.to_normalized_item(
            _story(word_count=99999, chapter_count=40), _Normalization()
        )
    )

    assert a == b
