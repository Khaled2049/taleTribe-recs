"""Unit tests for controlled vocabularies and the LLM normalization pass.

No network: the LLM is a fake that returns canned payloads.
"""

import pytest

from recommendation_engine.ingest.normalize_llm import (  # noqa: E402
    MAX_PREMISE_WORDS,
    MAX_THEMES,
    MAX_TONES,
    MIN_CONFIDENCE,
    PROMPT_VERSION,
    Normalizer,
    NormalizeStats,
    _coerce,
    build_prompt,
)
from recommendation_engine.ingest.vocabularies import (  # noqa: E402
    THEMES,
    TONES,
    filter_themes,
    filter_tones,
    prompt_vocabulary_block,
)
from recommendation_engine.llm import LLMResponseError  # noqa: E402

pytestmark = pytest.mark.unit


# ══════════════════════════════════════════════════════════════════════════
# Vocabularies
# ══════════════════════════════════════════════════════════════════════════


def test_vocabularies_are_non_trivial():
    """The point is a closed list big enough to describe 15.5k books without
    being so big the model cannot choose consistently."""
    assert 150 <= len(THEMES) <= 400
    assert 20 <= len(TONES) <= 60


def test_vocabulary_terms_are_lowercase_and_trimmed():
    """Canonical form matters — these strings end up in a GIN-indexed array."""
    for term in THEMES | TONES:
        assert term == term.strip()
        assert term == term.lower(), f"{term!r} is not lowercase"


def test_exact_terms_pass_through():
    assert filter_themes(["revenge", "found family"]) == ["revenge", "found family"]
    assert filter_tones(["bleak", "wry"]) == ["bleak", "wry"]


def test_case_and_whitespace_are_normalized():
    assert filter_themes(["  REVENGE  ", "Found   Family"]) == [
        "revenge",
        "found family",
    ]


def test_out_of_vocabulary_terms_are_dropped():
    """This is the enforcement that actually guarantees the invariant — the
    prompt only makes it likely."""
    assert filter_themes(["revenge", "a theme I invented", "quantum ennui"]) == [
        "revenge"
    ]
    assert filter_tones(["bleak", "extremely vibey"]) == ["bleak"]


def test_aliases_are_canonicalized():
    assert filter_themes(["vengeance"]) == ["revenge"]
    assert filter_themes(["AI"]) == ["artificial intelligence"]
    assert filter_themes(["growing up"]) == ["coming of age"]
    assert filter_tones(["funny"]) == ["comic"]
    assert filter_tones(["dark"]) == ["bleak"]


def test_american_spellings_are_accepted():
    """The vocabulary uses British spellings; the model will often emit American."""
    assert filter_themes(["organized crime"]) == ["organised crime"]
    assert filter_themes(["space colonization"]) == ["space colonisation"]


def test_plural_variants_are_tolerated():
    assert filter_themes(["curse"]) == ["curses"]
    assert filter_themes(["dragon"]) == ["dragons"]


def test_duplicates_collapse_after_canonicalization():
    """ "vengeance" and "revenge" must not both survive as separate themes."""
    assert filter_themes(["revenge", "vengeance", "Revenge"]) == ["revenge"]


def test_order_is_preserved():
    """The prompt asks for themes most-to-least central, and leading terms carry
    more weight in the composed embedding input."""
    assert filter_themes(["betrayal", "revenge", "found family"]) == [
        "betrayal",
        "revenge",
        "found family",
    ]


def test_empty_and_none_inputs_are_safe():
    assert filter_themes([]) == []
    assert filter_themes(None) == []
    assert filter_themes(["", "   "]) == []


def test_prompt_vocabulary_block_is_stable():
    """An unstable prompt would change the cache key on every run and re-bill the
    entire corpus."""
    assert prompt_vocabulary_block() == prompt_vocabulary_block()


def test_prompt_vocabulary_block_lists_both_vocabularies():
    block = prompt_vocabulary_block()

    assert "ALLOWED THEMES" in block
    assert "ALLOWED TONES" in block
    assert "revenge" in block
    assert "bleak" in block


# ══════════════════════════════════════════════════════════════════════════
# Prompt construction
# ══════════════════════════════════════════════════════════════════════════


def test_prompt_includes_every_book_with_its_id():
    batch = [
        ("1", "Animal Farm", "George Orwell", "Farm animals revolt."),
        ("2", "Dune", "Frank Herbert", "Desert planet, spice, prophecy."),
    ]

    prompt = build_prompt(batch)

    assert "id=1" in prompt and "id=2" in prompt
    assert "Animal Farm by George Orwell" in prompt
    assert "Dune by Frank Herbert" in prompt
    assert "Farm animals revolt." in prompt


def test_prompt_omits_the_byline_when_author_is_missing():
    prompt = build_prompt([("1", "Beowulf", None, "A hero fights a monster.")])

    assert "Title: Beowulf\n" in prompt
    assert " by " not in prompt.split("--- BOOK")[1].split("\n")[1]


def test_prompt_instructs_low_confidence_rather_than_invention():
    """Without this the model treats every input as answerable and fabricates a
    premise for a 60-word stub."""
    prompt = build_prompt([("1", "T", None, "Short.")])

    assert "Do NOT" in prompt
    assert "invent" in prompt
    assert "below 0.5" in prompt


def test_prompt_carries_the_vocabulary():
    prompt = build_prompt([("1", "T", None, "S")])

    assert "ALLOWED THEMES" in prompt
    assert "ALLOWED TONES" in prompt


# ══════════════════════════════════════════════════════════════════════════
# Coercing a single result
# ══════════════════════════════════════════════════════════════════════════


def test_coerce_happy_path():
    stats = NormalizeStats()

    result = _coerce(
        {
            "id": "1",
            "core_premise": "Farm animals overthrow their owner.",
            "themes": ["allegory of politics", "corruption of power"],
            "tone": ["bleak", "satirical"],
            "confidence": 0.9,
        },
        stats,
    )

    assert result.core_premise == "Farm animals overthrow their owner."
    assert result.themes == ["allegory of politics", "corruption of power"]
    assert result.tone == ["bleak", "satirical"]
    assert result.confidence == 0.9
    assert result.is_reliable is True


def test_coerce_truncates_an_overlong_premise():
    stats = NormalizeStats()
    long_premise = " ".join(f"w{i}" for i in range(MAX_PREMISE_WORDS + 40))

    result = _coerce({"core_premise": long_premise, "confidence": 1.0}, stats)

    assert len(result.core_premise.split(" ")) == MAX_PREMISE_WORDS


def test_coerce_caps_theme_and_tone_counts():
    stats = NormalizeStats()
    many_themes = sorted(THEMES)[: MAX_THEMES + 5]
    many_tones = sorted(TONES)[: MAX_TONES + 5]

    result = _coerce(
        {
            "core_premise": "p",
            "themes": many_themes,
            "tone": many_tones,
            "confidence": 1.0,
        },
        stats,
    )

    assert len(result.themes) == MAX_THEMES
    assert len(result.tone) == MAX_TONES


def test_coerce_records_vocabulary_violations():
    """A rising violation count means the prompt or vocabulary needs work — not
    that more aliases should be bolted on."""
    stats = NormalizeStats()

    _coerce(
        {
            "core_premise": "p",
            "themes": ["revenge", "invented theme"],
            "tone": ["bleak", "invented tone"],
            "confidence": 1.0,
        },
        stats,
    )

    assert stats.dropped_themes == 1
    assert stats.dropped_tones == 1
    assert stats.vocabulary_violations["invented theme"] == 1


@pytest.mark.parametrize(
    "raw,expected", [(1.5, 1.0), (-0.3, 0.0), ("0.7", 0.7), (None, 0.0), ("x", 0.0)]
)
def test_coerce_clamps_confidence(raw, expected):
    result = _coerce({"core_premise": "p", "confidence": raw}, NormalizeStats())

    assert result.confidence == pytest.approx(expected)


def test_low_confidence_is_not_reliable():
    """The gate that keeps hallucinated premises out of the index."""
    stats = NormalizeStats()

    result = _coerce(
        {"core_premise": "invented", "confidence": MIN_CONFIDENCE - 0.01}, stats
    )

    assert result.is_reliable is False


def test_empty_premise_is_not_reliable_even_at_high_confidence():
    result = _coerce({"core_premise": "", "confidence": 1.0}, NormalizeStats())

    assert result.is_reliable is False


def test_missing_fields_do_not_raise():
    result = _coerce({}, NormalizeStats())

    assert result.core_premise == ""
    assert result.themes == []
    assert result.tone == []
    assert result.is_reliable is False


# ══════════════════════════════════════════════════════════════════════════
# Normalizer batching and failure isolation
# ══════════════════════════════════════════════════════════════════════════


class _FakeLLM:
    """Returns a canned payload per call, or raises."""

    model = "fake-model"

    def __init__(self, payloads):
        self._payloads = list(payloads)
        self.prompts = []

    async def generate_json(self, prompt, **kwargs):
        self.prompts.append(prompt)
        nxt = self._payloads.pop(0) if self._payloads else {"results": []}
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def _result(identifier, confidence=0.9, premise="A premise."):
    return {
        "id": identifier,
        "core_premise": premise,
        "themes": ["revenge"],
        "tone": ["bleak"],
        "confidence": confidence,
    }


def _records(n):
    return [(str(i), f"Title {i}", "Author", f"Summary {i}") for i in range(n)]


async def test_normalizes_a_single_batch():
    llm = _FakeLLM([{"results": [_result("0"), _result("1")]}])
    normalizer = Normalizer(llm, cache=None, batch_size=8)

    results = await normalizer.normalize_many(_records(2))

    assert set(results) == {"0", "1"}
    assert len(llm.prompts) == 1
    assert normalizer.stats.generated == 2


async def test_splits_into_batches_of_the_configured_size():
    llm = _FakeLLM(
        [
            {"results": [_result(str(i)) for i in range(3)]},
            {"results": [_result(str(i)) for i in range(3, 5)]},
        ]
    )
    normalizer = Normalizer(llm, cache=None, batch_size=3)

    results = await normalizer.normalize_many(_records(5))

    assert len(llm.prompts) == 2
    assert len(results) == 5
    assert normalizer.stats.batches == 2


async def test_a_failed_batch_does_not_abort_the_run():
    """One bad batch out of ~1,940 must not lose a 15k-record run. The affected
    records stay uncached so the next run retries exactly those."""
    llm = _FakeLLM(
        [
            LLMResponseError("model exploded"),
            {"results": [_result(str(i)) for i in range(2, 4)]},
        ]
    )
    normalizer = Normalizer(llm, cache=None, batch_size=2)

    results = await normalizer.normalize_many(_records(4))

    assert set(results) == {"2", "3"}
    assert normalizer.stats.failed == 2


async def test_ids_not_in_the_batch_are_ignored():
    """Attaching a premise to the wrong book is worse than dropping it."""
    llm = _FakeLLM([{"results": [_result("0"), _result("999")]}])
    normalizer = Normalizer(llm, cache=None, batch_size=8)

    results = await normalizer.normalize_many(_records(1))

    assert set(results) == {"0"}


async def test_missing_results_are_counted_as_failures():
    llm = _FakeLLM([{"results": [_result("0")]}])
    normalizer = Normalizer(llm, cache=None, batch_size=8)

    await normalizer.normalize_many(_records(3))

    assert normalizer.stats.failed == 2


async def test_low_confidence_results_are_counted():
    llm = _FakeLLM([{"results": [_result("0", confidence=0.2)]}])
    normalizer = Normalizer(llm, cache=None, batch_size=8)

    results = await normalizer.normalize_many(_records(1))

    assert normalizer.stats.low_confidence == 1
    assert results["0"].is_reliable is False


async def test_no_records_makes_no_calls():
    llm = _FakeLLM([])
    normalizer = Normalizer(llm, cache=None)

    assert await normalizer.normalize_many([]) == {}
    assert llm.prompts == []


# ── Caching ──────────────────────────────────────────────────────────────


class _FakeCache:
    """In-memory stand-in for the Postgres normalization cache."""

    def __init__(self, preloaded=None):
        self.store = dict(preloaded or {})
        self.put_calls = 0

    async def get_many(self, shas):
        return {sha: self.store[sha] for sha in shas if sha in self.store}

    async def put_many(self, entries, model):
        self.put_calls += 1
        self.store.update(entries)


async def test_cache_hits_avoid_api_calls_entirely():
    """The resume property: a crashed run must not re-bill work already done."""
    from recommendation_engine.ingest.compose import summary_sha
    from recommendation_engine.ingest.normalize_llm import Normalization

    records = _records(2)
    preloaded = {
        summary_sha(summary, PROMPT_VERSION): Normalization(
            core_premise="cached", themes=["revenge"], tone=["bleak"], confidence=0.9
        )
        for _, _, _, summary in records
    }
    llm = _FakeLLM([])
    normalizer = Normalizer(llm, cache=_FakeCache(preloaded), batch_size=8)

    results = await normalizer.normalize_many(records)

    assert llm.prompts == [], "cached records must not reach the API"
    assert normalizer.stats.from_cache == 2
    assert results["0"].core_premise == "cached"


async def test_partial_cache_only_generates_the_misses():
    from recommendation_engine.ingest.compose import summary_sha
    from recommendation_engine.ingest.normalize_llm import Normalization

    records = _records(3)
    cached_sha = summary_sha(records[0][3], PROMPT_VERSION)
    cache = _FakeCache(
        {
            cached_sha: Normalization(
                core_premise="cached", themes=[], tone=[], confidence=0.9
            )
        }
    )
    llm = _FakeLLM([{"results": [_result("1"), _result("2")]}])
    normalizer = Normalizer(llm, cache=cache, batch_size=8)

    results = await normalizer.normalize_many(records)

    assert normalizer.stats.from_cache == 1
    assert len(results) == 3
    # Only the two misses were sent.
    assert "Summary 0" not in llm.prompts[0]
    assert "Summary 1" in llm.prompts[0]


async def test_generated_results_are_written_back_to_the_cache():
    cache = _FakeCache()
    llm = _FakeLLM([{"results": [_result("0")]}])
    normalizer = Normalizer(llm, cache=cache, batch_size=8)

    await normalizer.normalize_many(_records(1))

    assert cache.put_calls == 1
    assert len(cache.store) == 1


async def test_failed_records_are_not_cached():
    """Otherwise a transient failure would be remembered as a permanent one."""
    cache = _FakeCache()
    llm = _FakeLLM([LLMResponseError("boom")])
    normalizer = Normalizer(llm, cache=cache, batch_size=8)

    await normalizer.normalize_many(_records(1))

    assert cache.store == {}


# ── Raw output is what gets cached ───────────────────────────────────────
# The vocabulary is meant to be tuned iteratively from the violation counter. If
# the cache held post-filtered output, every tuning pass would mean regenerating
# the whole corpus at full token cost. Caching raw makes tuning free.


def test_coerce_keeps_the_raw_model_output_alongside_the_filtered_list():
    result = _coerce(
        {
            "core_premise": "p",
            "themes": ["revenge", "a theme I invented"],
            "tone": ["bleak", "invented tone"],
            "confidence": 0.9,
        },
        NormalizeStats(),
    )

    assert result.themes == ["revenge"]
    assert result.raw_themes == ["revenge", "a theme I invented"]
    assert result.tone == ["bleak"]
    assert result.raw_tone == ["bleak", "invented tone"]


async def test_cache_receives_raw_terms_not_filtered_ones():
    """So a later vocabulary change can rescue a term this run discarded."""
    cache = _FakeCache()
    llm = _FakeLLM(
        [
            {
                "results": [
                    {
                        "id": "0",
                        "core_premise": "p",
                        "themes": ["revenge", "not yet in the vocabulary"],
                        "tone": ["bleak"],
                        "confidence": 0.9,
                    }
                ]
            }
        ]
    )
    normalizer = Normalizer(llm, cache=cache, batch_size=8)

    await normalizer.normalize_many(_records(1))

    (stored,) = cache.store.values()
    assert "not yet in the vocabulary" in stored.raw_themes, (
        "raw output must survive into the cache, or vocabulary tuning costs a "
        "full corpus regeneration"
    )


def test_alias_additions_recover_previously_dropped_terms():
    """Observed in a real run: the model reached for shorter forms of terms the
    vocabulary already had, and they were being discarded."""
    for shorthand, canonical in [
        ("free will", "fate versus free will"),
        ("free will versus determinism", "fate versus free will"),
        ("survival", "survival against nature"),
        ("corruption", "corruption of power"),
        ("rebellion", "rebellion against tyranny"),
        ("epidemic", "pandemic"),
        ("violence", "violence and its consequences"),
    ]:
        assert filter_themes([shorthand]) == [
            canonical
        ], f"{shorthand!r} should canonicalize to {canonical!r}"


def test_genuine_vocabulary_gaps_were_added():
    for term in (
        "good versus evil",
        "crime and punishment",
        "violence and its consequences",
    ):
        assert term in THEMES
