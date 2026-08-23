"""Freebase genre labels → TaleTribe platform categories.

The mapping lives in `genre_crosswalk.csv` rather than a dict in this file, on
purpose: it encodes ~227 judgement calls (is "Chivalric romance" the love genre
or the medieval literary form? does "Children's literature" belong in
young-adult?) and those are far easier to review, argue with, and correct in a
diff of a flat table than buried in Python.

Two rules the loader enforces:

* **Every corpus label must be either mapped or explicitly dropped.** An unknown
  label is an error, not a silent no-op — otherwise a corpus refresh introducing
  new labels would quietly produce uncategorized books.
* **Umbrella labels are dropped, not mapped.** `Fiction` (4747 records),
  `Speculative fiction` (4314) and `Novel` (2463) span so much of the corpus that
  they carry no discriminative signal; keeping them would make a genre filter
  match nearly everything.
"""

import csv
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Set

CROSSWALK_PATH = Path(__file__).parent / "genre_crosswalk.csv"

# The reader-facing category vocabulary, mirroring the chips hardcoded in
# taleTribe-frontend/src/routes/Story/AllStories.tsx (minus the "all" pseudo
# category). Note there is deliberately no children's bucket on the platform,
# which is why Children's literature maps to young-adult — see the CSV note.
PLATFORM_CATEGORIES: FrozenSet[str] = frozenset(
    {
        "fiction",
        "non-fiction",
        "poetry",
        "fantasy",
        "science-fiction",
        "romance",
        "mystery-thriller",
        "horror",
        "historical-fiction",
        "young-adult",
    }
)

ACTION_MAP = "map"
ACTION_DROP = "drop"

# Labels that assert a work is fiction. These are exactly the umbrella labels the
# CSV drops — useless for *filtering* (they match a third of the corpus) but the
# only reliable signal of fictionality in the data.
#
# They earn their keep here because subject labels are genuinely ambiguous: in the
# corpus, Philosophy co-occurs with a fiction marker 46% of the time, History 42%,
# Psychology 40%, Existentialism 37%. So "Existentialism" cannot mean non-fiction
# on its own — Camus' *The Plague* is tagged {Existentialism, Fiction, Absurdist
# fiction, Novel} and is unambiguously a novel. Without this rule it lands in
# both `fiction` and `non-fiction`, and a reader filtering for non-fiction gets a
# plague allegory.
FICTION_MARKERS: FrozenSet[str] = frozenset(
    {
        "Fiction",
        "Novel",
        "Novella",
        "Speculative fiction",
        "Short story",
        "Light novel",
    }
)

# The corresponding explicit assertion in the other direction.
NONFICTION_MARKERS: FrozenSet[str] = frozenset({"Non-fiction", "Non-fiction novel"})


class CrosswalkError(ValueError):
    """The crosswalk table is malformed or incomplete."""


class GenreCrosswalk:
    """Loaded crosswalk table."""

    def __init__(self, mapping: Dict[str, List[str]], dropped: Set[str]) -> None:
        self._mapping = mapping
        self._dropped = frozenset(dropped)

    @property
    def known_labels(self) -> FrozenSet[str]:
        return frozenset(self._mapping) | self._dropped

    @property
    def dropped_labels(self) -> FrozenSet[str]:
        return self._dropped

    def unknown(self, labels: Iterable[str]) -> Set[str]:
        """Labels the table has no opinion about — the thing a corpus refresh
        must never introduce silently."""
        return {label for label in labels if label not in self.known_labels}

    def map_labels(self, raw_labels: Iterable[str]) -> List[str]:
        """Crosswalk raw Freebase labels to platform categories.

        Unknown labels are skipped rather than raising: a per-record parse should
        not die on one odd label. Catching new labels is the job of
        `unknown()`, which the backfill calls once over the whole corpus so the
        failure is loud and aggregate instead of a mid-run crash.

        After mapping, fictionality is resolved from the marker labels — see
        FICTION_MARKERS for why a subject label alone cannot decide it.
        """
        labels = list(raw_labels)
        categories: Set[str] = set()
        for label in labels:
            categories.update(self._mapping.get(label, ()))

        label_set = set(labels)
        asserts_fiction = bool(label_set & FICTION_MARKERS)
        asserts_nonfiction = bool(label_set & NONFICTION_MARKERS)

        # Only act when the evidence points one way. Tagged as both (a
        # non-fiction novel, say) means the corpus itself is ambiguous, and
        # guessing would be worse than keeping both.
        if asserts_fiction and not asserts_nonfiction:
            categories.discard("non-fiction")
            if not categories:
                # Every mapped category was non-fiction, e.g. {Philosophy, Novel}.
                # The markers still tell us it is fiction, so fall back to that
                # rather than emitting an uncategorized book.
                categories.add("fiction")
        elif asserts_nonfiction and not asserts_fiction:
            categories.discard("fiction")
            if not categories:
                categories.add("non-fiction")

        return sorted(categories)


def load_crosswalk(path: Path = CROSSWALK_PATH) -> GenreCrosswalk:
    """Read and validate the CSV."""
    mapping: Dict[str, List[str]] = {}
    dropped: Set[str] = set()

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"cmu_label", "action", "platform_categories"}
        missing_columns = required - set(reader.fieldnames or ())
        if missing_columns:
            raise CrosswalkError(
                f"{path.name} is missing column(s): {sorted(missing_columns)}"
            )

        for line_number, row in enumerate(reader, start=2):
            label = (row.get("cmu_label") or "").strip()
            if not label:
                continue
            if label in mapping or label in dropped:
                raise CrosswalkError(
                    f"{path.name}:{line_number} duplicate label {label!r} — two rows "
                    "disagree about the same genre"
                )

            action = (row.get("action") or "").strip()
            raw_categories = (row.get("platform_categories") or "").strip()
            categories = [c.strip() for c in raw_categories.split("|") if c.strip()]

            if action == ACTION_DROP:
                if categories:
                    raise CrosswalkError(
                        f"{path.name}:{line_number} {label!r} is marked drop but also "
                        f"lists categories {categories}"
                    )
                dropped.add(label)
                continue

            if action != ACTION_MAP:
                raise CrosswalkError(
                    f"{path.name}:{line_number} {label!r} has action {action!r}; "
                    f"expected {ACTION_MAP!r} or {ACTION_DROP!r}"
                )
            if not categories:
                raise CrosswalkError(
                    f"{path.name}:{line_number} {label!r} is marked map but lists no "
                    "categories — use action=drop to discard a label"
                )
            invalid = [c for c in categories if c not in PLATFORM_CATEGORIES]
            if invalid:
                raise CrosswalkError(
                    f"{path.name}:{line_number} {label!r} maps to unknown platform "
                    f"category/ies {invalid}; valid: {sorted(PLATFORM_CATEGORIES)}"
                )
            mapping[label] = sorted(set(categories))

    if not mapping:
        raise CrosswalkError(f"{path.name} contains no mapped labels")

    return GenreCrosswalk(mapping, dropped)
