"""Unit tests for the CMU corpus parser, crosswalk and embed-input composer.

All pure functions — no database, no network, no API key.
"""

import os

import pytest

os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "test-project")

from recommendation_engine.ingest.cmu_parse import (  # noqa: E402
    DEFAULT_CORPUS_PATH,
    MAX_SUMMARY_WORDS,
    MIN_SUMMARY_WORDS,
    ParseStats,
    dedupe_key,
    iter_records,
    normalize_summary,
    parse_genres,
    parse_year,
)
from recommendation_engine.ingest.compose import (  # noqa: E402
    NormalizedItem,
    compose_embed_input,
    embed_input_sha,
    summary_sha,
)
from recommendation_engine.ingest.genre_crosswalk import (  # noqa: E402
    PLATFORM_CATEGORIES,
    CrosswalkError,
    load_crosswalk,
)

pytestmark = pytest.mark.unit


def _words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def _row(
    wiki_id="620",
    freebase="/m/0hhy",
    title="Animal Farm",
    author="George Orwell",
    date="1945-08-17",
    genres='{"/m/06nbt": "Satire"}',
    summary=None,
):
    summary = summary if summary is not None else " " + _words(100)
    return "\t".join([wiki_id, freebase, title, author, date, genres, summary])


def _corpus(tmp_path, *rows):
    path = tmp_path / "booksummaries.txt"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


# ══════════════════════════════════════════════════════════════════════════
# Genre column — the empty-string-not-{} trap
# ══════════════════════════════════════════════════════════════════════════


def test_parse_genres_reads_freebase_json():
    raw = '{"/m/016lj8": "Roman \\u00e0 clef", "/m/06nbt": "Satire"}'
    assert sorted(parse_genres(raw)) == ["Roman à clef", "Satire"]


def test_parse_genres_handles_empty_string_not_empty_object():
    """22.5% of records (3,718) have an empty STRING here, not `{}` — the one
    input a naive json.loads dies on."""
    assert parse_genres("") == []
    assert parse_genres("   ") == []
    assert parse_genres(None) == []


def test_parse_genres_survives_malformed_json():
    """One bad row must not kill a 16k-record run."""
    assert parse_genres("{not json") == []
    assert parse_genres("[1, 2, 3]") == []


def test_parse_genres_drops_freebase_mids():
    """MIDs are stable identifiers but meaningless to an embedding model."""
    labels = parse_genres('{"/m/06nbt": "Satire"}')
    assert labels == ["Satire"]
    assert not any(label.startswith("/m/") for label in labels)


# ══════════════════════════════════════════════════════════════════════════
# Dates — three precisions, one field
# ══════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("1945-08-17", 1945),
        ("1962", 1962),
        ("1999-03", 1999),
        ("", None),
        ("   ", None),
        (None, None),
        ("not-a-date", None),
    ],
)
def test_parse_year(raw, expected):
    assert parse_year(raw) == expected


def test_parse_year_rejects_out_of_range_noise():
    assert parse_year("0000") is None
    assert parse_year("9999") is None


# ══════════════════════════════════════════════════════════════════════════
# Summary normalization
# ══════════════════════════════════════════════════════════════════════════


def test_leading_space_is_stripped():
    """Every summary in the corpus starts with a space."""
    text, count, truncated = normalize_summary(" Old Major, the old boar")
    assert text.startswith("Old Major")
    assert not truncated
    assert count == 5


def test_internal_whitespace_is_collapsed():
    text, count, _ = normalize_summary("a  b\n\nc\td")
    assert text == "a b c d"
    assert count == 4


def test_long_summaries_are_truncated():
    text, count, truncated = normalize_summary(_words(MAX_SUMMARY_WORDS + 500))
    assert truncated is True
    assert count == MAX_SUMMARY_WORDS
    assert len(text.split(" ")) == MAX_SUMMARY_WORDS


def test_summary_at_the_limit_is_not_truncated():
    _, count, truncated = normalize_summary(_words(MAX_SUMMARY_WORDS))
    assert truncated is False
    assert count == MAX_SUMMARY_WORDS


def test_empty_summary():
    assert normalize_summary("") == ("", 0, False)
    assert normalize_summary(None) == ("", 0, False)


# ══════════════════════════════════════════════════════════════════════════
# Dedupe identity
# ══════════════════════════════════════════════════════════════════════════


def test_dedupe_key_is_case_and_whitespace_insensitive():
    assert dedupe_key("Animal Farm", "George Orwell") == dedupe_key(
        "  animal   farm ", "GEORGE ORWELL"
    )


def test_dedupe_key_separates_same_title_different_author():
    """Distinct books share titles, so title alone would over-collapse."""
    assert dedupe_key("Ulysses", "James Joyce") != dedupe_key("Ulysses", "Tennyson")


def test_dedupe_key_handles_missing_author():
    assert dedupe_key("Beowulf", None) == dedupe_key("beowulf", "")


# ══════════════════════════════════════════════════════════════════════════
# iter_records — the filters
# ══════════════════════════════════════════════════════════════════════════


def test_parses_a_well_formed_record(tmp_path):
    path = _corpus(tmp_path, _row())
    stats = ParseStats()

    (record,) = list(iter_records(path, stats))

    assert record.wiki_id == "620"
    assert record.freebase_id == "/m/0hhy"
    assert record.title == "Animal Farm"
    assert record.author == "George Orwell"
    assert record.published_year == 1945
    assert record.raw_genres == ["Satire"]
    assert record.summary_word_count == 100
    assert stats.emitted == 1


def test_stubs_are_dropped(tmp_path):
    """1,042 records are under 50 words; the shortest is 11 characters. An LLM
    asked for a premise from that will invent one."""
    path = _corpus(
        tmp_path,
        _row(title="Stub", summary=" " + _words(MIN_SUMMARY_WORDS - 1)),
        _row(title="Fine", summary=" " + _words(MIN_SUMMARY_WORDS)),
    )
    stats = ParseStats()

    records = list(iter_records(path, stats))

    assert [r.title for r in records] == ["Fine"]
    assert stats.stub_dropped == 1


def test_monsters_are_truncated_not_dropped(tmp_path):
    path = _corpus(tmp_path, _row(summary=" " + _words(MAX_SUMMARY_WORDS + 100)))
    stats = ParseStats()

    (record,) = list(iter_records(path, stats))

    assert record.truncated is True
    assert record.summary_word_count == MAX_SUMMARY_WORDS
    assert stats.truncated == 1


def test_missing_author_becomes_none_and_is_counted(tmp_path):
    path = _corpus(tmp_path, _row(author=""))
    stats = ParseStats()

    (record,) = list(iter_records(path, stats))

    assert record.author is None
    assert stats.missing_author == 1


def test_missing_genres_and_year_are_counted(tmp_path):
    path = _corpus(tmp_path, _row(genres="", date=""))
    stats = ParseStats()

    (record,) = list(iter_records(path, stats))

    assert record.raw_genres == []
    assert record.published_year is None
    assert stats.missing_genres == 1
    assert stats.missing_year == 1


def test_malformed_rows_are_counted_not_fatal(tmp_path):
    path = _corpus(tmp_path, "only\tthree\tfields", _row())
    stats = ParseStats()

    records = list(iter_records(path, stats))

    assert len(records) == 1
    assert stats.malformed == 1


def test_empty_title_is_skipped(tmp_path):
    path = _corpus(tmp_path, _row(title="   "), _row(title="Real"))
    stats = ParseStats()

    records = list(iter_records(path, stats))

    assert [r.title for r in records] == ["Real"]
    assert stats.empty_title == 1


def test_duplicates_keep_the_longest_summary(tmp_path):
    """246 titles repeat. Keeping the richest version measurably improves the
    embedding over keeping whichever came first."""
    path = _corpus(
        tmp_path,
        _row(wiki_id="1", summary=" " + _words(60)),
        _row(wiki_id="2", summary=" " + _words(300)),
        _row(wiki_id="3", summary=" " + _words(90)),
    )
    stats = ParseStats()

    (record,) = list(iter_records(path, stats))

    assert record.wiki_id == "2"
    assert record.summary_word_count == 300
    assert stats.duplicate_dropped == 2


def test_quotes_in_summaries_do_not_swallow_fields(tmp_path):
    """Plot summaries are full of bare double quotes; without QUOTE_NONE the csv
    reader would merge fields and the row would look malformed."""
    summary = ' He said "hello" and she said "goodbye" ' + _words(60)
    path = _corpus(tmp_path, _row(summary=summary))
    stats = ParseStats()

    (record,) = list(iter_records(path, stats))

    assert '"hello"' in record.summary
    assert stats.malformed == 0


def test_limit_caps_emitted_records(tmp_path):
    rows = [_row(wiki_id=str(i), title=f"Book {i}") for i in range(10)]
    path = _corpus(tmp_path, *rows)
    stats = ParseStats()

    records = list(iter_records(path, stats, limit=3))

    assert len(records) == 3
    assert stats.emitted == 3


def test_stats_as_dict_sorts_unknown_labels_by_frequency():
    stats = ParseStats()
    stats.unknown_genre_labels = {"rare": 1, "common": 9, "mid": 5}

    assert list(stats.as_dict()["unknown_genre_labels"]) == ["common", "mid", "rare"]


# ══════════════════════════════════════════════════════════════════════════
# Genre crosswalk
# ══════════════════════════════════════════════════════════════════════════


def test_real_crosswalk_loads_and_validates():
    crosswalk = load_crosswalk()

    assert len(crosswalk.known_labels) == 227, (
        "the corpus has exactly 227 distinct genre labels; a change here means "
        "the crosswalk and the corpus have drifted apart"
    )


def test_real_crosswalk_covers_every_label_in_the_corpus():
    """The guard against a silent miscategorization: a corpus refresh that adds
    labels must fail loudly here."""
    crosswalk = load_crosswalk()
    seen = set()

    with DEFAULT_CORPUS_PATH.open(encoding="utf-8", newline="") as handle:
        import csv as _csv

        for row in _csv.reader(handle, delimiter="\t", quoting=_csv.QUOTE_NONE):
            if len(row) == 7:
                seen.update(parse_genres(row[5]))

    unknown = crosswalk.unknown(seen)
    assert not unknown, f"genre labels missing from the crosswalk: {sorted(unknown)}"


def test_umbrella_labels_are_dropped_not_mapped():
    """Fiction (4747), Speculative fiction (4314) and Novel (2463) span so much
    of the corpus that mapping them would make a genre filter match everything."""
    dropped = load_crosswalk().dropped_labels

    for label in ("Fiction", "Speculative fiction", "Novel"):
        assert label in dropped, f"{label} must be dropped as an umbrella"


def test_crosswalk_maps_to_platform_categories_only():
    crosswalk = load_crosswalk()

    assert crosswalk.map_labels(["Science Fiction"]) == ["science-fiction"]
    assert crosswalk.map_labels(["Mystery"]) == ["mystery-thriller"]
    # Multi-category labels fan out.
    assert crosswalk.map_labels(["Dark fantasy"]) == ["fantasy", "horror"]


def test_crosswalk_dedupes_and_sorts_across_labels():
    crosswalk = load_crosswalk()

    result = crosswalk.map_labels(
        ["Science Fiction", "Hard science fiction", "Fantasy"]
    )

    assert result == ["fantasy", "science-fiction"]


def test_dropped_labels_contribute_no_genre_of_their_own():
    """Umbrellas add no *genre*, but they do assert fictionality — so a book
    tagged only `Fiction, Novel` lands in the general `fiction` bucket rather
    than being emitted with no category at all."""
    assert load_crosswalk().map_labels(["Fiction", "Novel"]) == ["fiction"]


def test_dropped_labels_with_no_fictionality_signal_contribute_nothing():
    assert load_crosswalk().map_labels(["Graphic novel", "Serial"]) == []


# ── Fictionality resolution ──────────────────────────────────────────────
# Subject labels are ambiguous in this corpus: Philosophy co-occurs with a
# fiction marker 46% of the time, History 42%, Existentialism 37%. So the
# umbrella markers have to break the tie.


def test_fiction_marker_suppresses_non_fiction():
    """Camus' The Plague is tagged {Existentialism, Fiction, Absurdist fiction,
    Novel}. Without this rule it lands in non-fiction and a reader filtering for
    non-fiction gets a plague allegory."""
    result = load_crosswalk().map_labels(
        ["Existentialism", "Fiction", "Absurdist fiction", "Novel"]
    )

    assert result == ["fiction"]
    assert "non-fiction" not in result


def test_fiction_marker_falls_back_to_fiction_when_nothing_else_remains():
    """{Philosophy, Novel} maps only to non-fiction, which the marker then
    suppresses — leaving nothing. The marker still tells us it is fiction."""
    assert load_crosswalk().map_labels(["Philosophy", "Novel"]) == ["fiction"]


def test_subject_label_alone_stays_non_fiction():
    """The rule must not make everything fiction — an actual philosophy book has
    no fiction marker."""
    assert load_crosswalk().map_labels(["Philosophy"]) == ["non-fiction"]


def test_explicit_non_fiction_suppresses_fiction():
    result = load_crosswalk().map_labels(["Non-fiction", "History"])

    assert result == ["non-fiction"]
    assert "fiction" not in result


def test_both_signals_present_is_left_alone():
    """A non-fiction novel is the corpus being genuinely ambiguous; guessing
    would be worse than preserving what it says."""
    result = load_crosswalk().map_labels(["Non-fiction", "Novel", "True crime"])

    assert "non-fiction" in result
    assert "mystery-thriller" in result


def test_fictionality_rule_leaves_unambiguous_genres_untouched():
    assert load_crosswalk().map_labels(
        ["Hard science fiction", "Science Fiction", "Fiction"]
    ) == ["science-fiction"]


def test_unknown_labels_are_skipped_by_map_but_reported_by_unknown():
    crosswalk = load_crosswalk()

    assert crosswalk.map_labels(["Nonexistent Genre"]) == []
    assert crosswalk.unknown(["Nonexistent Genre"]) == {"Nonexistent Genre"}


def test_every_mapped_category_is_a_real_platform_category():
    crosswalk = load_crosswalk()

    for label in crosswalk.known_labels:
        for category in crosswalk.map_labels([label]):
            assert category in PLATFORM_CATEGORIES


def test_childrens_literature_maps_to_young_adult():
    """The platform has no children's bucket; this is the nearest one and the
    imprecision is documented in the CSV."""
    assert "young-adult" in load_crosswalk().map_labels(["Children's literature"])


# ── Crosswalk table validation ───────────────────────────────────────────


def _crosswalk_file(tmp_path, body):
    path = tmp_path / "crosswalk.csv"
    path.write_text(
        "cmu_label,action,platform_categories,note\n" + body, encoding="utf-8"
    )
    return path


def test_rejects_duplicate_labels(tmp_path):
    path = _crosswalk_file(
        tmp_path, "Mystery,map,mystery-thriller,\nMystery,map,horror,\n"
    )

    with pytest.raises(CrosswalkError, match="duplicate label"):
        load_crosswalk(path)


def test_rejects_unknown_platform_category(tmp_path):
    path = _crosswalk_file(tmp_path, "Mystery,map,not-a-category,\n")

    with pytest.raises(CrosswalkError, match="unknown platform category"):
        load_crosswalk(path)


def test_rejects_map_with_no_categories(tmp_path):
    path = _crosswalk_file(tmp_path, "Mystery,map,,\n")

    with pytest.raises(CrosswalkError, match="marked map but lists no categories"):
        load_crosswalk(path)


def test_rejects_drop_with_categories(tmp_path):
    path = _crosswalk_file(tmp_path, "Fiction,drop,fiction,\n")

    with pytest.raises(CrosswalkError, match="marked drop but also lists categories"):
        load_crosswalk(path)


def test_rejects_unknown_action(tmp_path):
    path = _crosswalk_file(tmp_path, "Mystery,maybe,mystery-thriller,\n")

    with pytest.raises(CrosswalkError, match="has action"):
        load_crosswalk(path)


def test_rejects_a_table_with_no_mappings(tmp_path):
    path = _crosswalk_file(tmp_path, "Fiction,drop,,\n")

    with pytest.raises(CrosswalkError, match="no mapped labels"):
        load_crosswalk(path)


def test_rejects_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("label,categories\nMystery,mystery-thriller\n", encoding="utf-8")

    with pytest.raises(CrosswalkError, match="missing column"):
        load_crosswalk(path)


# ══════════════════════════════════════════════════════════════════════════
# Embed input composition
# ══════════════════════════════════════════════════════════════════════════


def test_full_embed_input_is_a_stable_golden_string():
    item = NormalizedItem(
        title="Animal Farm",
        author="George Orwell",
        genres=["fiction", "satire"],
        core_premise="Farm animals overthrow their owner, then recreate his tyranny.",
        themes=["political allegory", "corruption of power"],
        tone=["bleak", "satirical"],
    )

    assert compose_embed_input(item) == (
        "Title: Animal Farm\n"
        "Author: George Orwell\n"
        "Genres: fiction, satire\n"
        "Premise: Farm animals overthrow their owner, then recreate his tyranny.\n"
        "Themes & tropes: political allegory, corruption of power\n"
        "Tone: bleak, satirical"
    )


def test_absent_author_drops_the_whole_line():
    """`Author: ` with nothing after it is not neutral — it would cluster the
    2,382 authorless records by their missing value."""
    output = compose_embed_input(NormalizedItem(title="Beowulf", author=None))

    assert output == "Title: Beowulf"
    assert "Author" not in output


@pytest.mark.parametrize("blank", [None, "", "   "])
def test_every_optional_field_drops_its_line_when_blank(blank):
    output = compose_embed_input(
        NormalizedItem(title="T", author=blank, core_premise=blank)
    )

    assert output == "Title: T"


def test_empty_lists_drop_their_lines():
    output = compose_embed_input(
        NormalizedItem(title="T", genres=[], themes=[], tone=[])
    )

    assert output == "Title: T"


def test_raw_summary_is_never_part_of_the_embed_input():
    """The central design decision: plot incident swamps premise/theme/tone."""
    item = NormalizedItem(
        title="T", core_premise="A short premise.", themes=["theme"], tone=["wry"]
    )

    output = compose_embed_input(item)

    assert "Premise: A short premise." in output
    assert "Summary" not in output


def test_list_values_are_deduped_case_insensitively_preserving_order():
    item = NormalizedItem(title="T", themes=["Revenge", "revenge", "Loss", "REVENGE"])

    assert "Themes & tropes: Revenge, Loss" in compose_embed_input(item)


def test_whitespace_in_values_is_collapsed():
    item = NormalizedItem(title="  Animal   Farm  ", author="George\tOrwell")

    output = compose_embed_input(item)

    assert output.startswith("Title: Animal Farm")
    assert "Author: George Orwell" in output


# ── Hashes ───────────────────────────────────────────────────────────────


def test_embed_input_sha_is_stable_for_identical_input():
    assert embed_input_sha("Title: X") == embed_input_sha("Title: X")


def test_embed_input_sha_changes_with_content():
    assert embed_input_sha("Title: X") != embed_input_sha("Title: Y")


def test_identical_items_produce_the_same_sha():
    """This is what lets the backfill skip unchanged rows on a re-run."""
    a = NormalizedItem(title="T", author="A", themes=["x"])
    b = NormalizedItem(title="T", author="A", themes=["x"])

    assert embed_input_sha(compose_embed_input(a)) == embed_input_sha(
        compose_embed_input(b)
    )


def test_summary_sha_is_invalidated_by_a_prompt_version_bump():
    """A prompt change must force regeneration rather than silently reusing
    normalizations produced by the old instructions."""
    text = "Some plot summary."

    assert summary_sha(text, 1) != summary_sha(text, 2)


def test_summary_sha_ignores_whitespace_variation():
    assert summary_sha("a  b", 1) == summary_sha("a b", 1)
