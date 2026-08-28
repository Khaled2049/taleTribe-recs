"""Streaming parser for the CMU Book Summary Corpus.

The file is 42 MB / 16,559 tab-separated rows with no header. Columns:

    1. Wikipedia article ID     4. Author (14.4% empty)
    2. Freebase ID              5. Publication date (33.9% empty, mixed precision)
    3. Book title               6. Genres — single-line JSON, or an EMPTY STRING
                                7. Plot summary (has a LEADING SPACE)

Verified clean: every line has exactly 7 fields and none contain embedded
newlines, so a plain line-by-line split is safe — but it is still streamed rather
than read whole, because holding 42 MB plus derived objects in memory during a
15k-record LLM pass is pointless.

Three filters applied here rather than downstream, because each one costs money
if it slips through to the LLM normalization pass:

* **Stubs dropped.** 1,042 records are under 50 words (the shortest is 11
  *characters*). There is nothing to derive a premise from, and asking an LLM
  anyway produces a confident hallucination — a poisoned vector that looks fine.
* **Monsters truncated.** 397 records exceed 10k characters, the longest 58k
  (~14k tokens). Plot summaries front-load the premise, so the tail is cheap to
  lose and expensive to keep.
* **Duplicates collapsed.** 246 titles repeat; the same book embedded twice would
  occupy two slots in every result list.
"""

import csv
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, cast

logger = logging.getLogger(__name__)

DEFAULT_CORPUS_PATH = (
    Path(__file__).parent.parent / "booksummaries" / "booksummaries.txt"
)

EXPECTED_FIELD_COUNT = 7

# A summary shorter than this cannot support a derived premise.
MIN_SUMMARY_WORDS = 50
# Plot summaries put the premise first; 1,200 words is well past where a premise,
# theme set and tone are established.
MAX_SUMMARY_WORDS = 1200

_WHITESPACE = re.compile(r"\s+")
_YEAR = re.compile(r"^(\d{4})")


@dataclass
class CmuRecord:
    """One parsed, filtered corpus record — before any LLM normalization."""

    wiki_id: str
    freebase_id: str
    title: str
    author: Optional[str]
    published_year: Optional[int]
    raw_genres: List[str]
    summary: str
    summary_word_count: int
    truncated: bool = False


@dataclass
class ParseStats:
    """Where every input line went. Reported at the end of a run so a silent
    drop of half the corpus is impossible to miss."""

    total_lines: int = 0
    malformed: int = 0
    empty_title: int = 0
    empty_summary: int = 0
    stub_dropped: int = 0
    truncated: int = 0
    duplicate_dropped: int = 0
    emitted: int = 0
    missing_author: int = 0
    missing_year: int = 0
    missing_genres: int = 0
    unknown_genre_labels: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        data = self.__dict__.copy()
        data["unknown_genre_labels"] = dict(
            sorted(
                self.unknown_genre_labels.items(), key=lambda kv: kv[1], reverse=True
            )
        )
        return data


def parse_genres(raw: str) -> List[str]:
    """Parse the Freebase genre column into human-readable labels.

    The field is strict single-line JSON (`{"/m/06nbt": "Satire", ...}`) — but
    when a book has no genres it is an **empty string**, not `{}`, which is the
    one thing a naive `json.loads` trips over.
    """
    raw = (raw or "").strip()
    if not raw:
        return []
    try:
        parsed = cast(object, json.loads(raw))
    except (json.JSONDecodeError, ValueError):
        logger.debug("unparseable_genre_field", extra={"raw": raw[:120]})
        return []
    if not isinstance(parsed, dict):
        return []
    # Freebase MIDs are stable identifiers but meaningless to a reader or an
    # embedding model; only the labels are kept.
    parsed_dict = cast(dict[object, object], parsed)
    return [str(name).strip() for name in parsed_dict.values() if str(name).strip()]


def parse_year(raw: str) -> Optional[int]:
    """Extract a publication year.

    Three formats appear — `YYYY` (6,799), `YYYY-MM` (1,479), `YYYY-MM-DD`
    (2,671) — so only the year is kept. Storing a full date would invent
    precision for two thirds of the records that have one.
    """
    match = _YEAR.match((raw or "").strip())
    if not match:
        return None
    year = int(match.group(1))
    # Guard against parse noise rather than asserting a real historical range.
    if not 1 <= year <= 2100:
        return None
    return year


def normalize_summary(raw: str) -> Tuple[str, int, bool]:
    """Collapse whitespace, strip the leading space, and truncate monsters.

    Returns (text, word_count, truncated).
    """
    text = _WHITESPACE.sub(" ", raw or "").strip()
    words = text.split(" ") if text else []
    if len(words) > MAX_SUMMARY_WORDS:
        return " ".join(words[:MAX_SUMMARY_WORDS]), MAX_SUMMARY_WORDS, True
    return text, len(words), False


def dedupe_key(title: str, author: Optional[str]) -> str:
    """Identity for duplicate collapsing.

    Title alone is too aggressive — distinct books share titles — so author is
    part of the key. Both are casefolded and whitespace-collapsed because the
    corpus is inconsistent about both.
    """
    norm_title = _WHITESPACE.sub(" ", title).strip().casefold()
    norm_author = _WHITESPACE.sub(" ", author or "").strip().casefold()
    return f"{norm_title}|{norm_author}"


def iter_records(
    path: Path = DEFAULT_CORPUS_PATH,
    stats: Optional[ParseStats] = None,
    min_summary_words: int = MIN_SUMMARY_WORDS,
    limit: Optional[int] = None,
) -> Iterator[CmuRecord]:
    """Stream filtered records from the corpus.

    Duplicates are resolved by keeping the record with the **longest** summary,
    which means a title cannot be emitted until the whole file has been read.
    That is a deliberate trade: buffering ~15k small records is cheap, and
    picking the richest version of a book materially improves its embedding over
    keeping whichever copy happened to come first.
    """
    stats = stats if stats is not None else ParseStats()
    best: Dict[str, CmuRecord] = {}

    with path.open(encoding="utf-8", newline="") as handle:
        # QUOTE_NONE: the corpus is not quoted, and plot summaries are full of
        # bare double quotes that would otherwise swallow whole fields.
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            stats.total_lines += 1

            if len(row) != EXPECTED_FIELD_COUNT:
                stats.malformed += 1
                continue

            wiki_id, freebase_id, title, author, date, genres, summary = row

            title = title.strip()
            if not title:
                stats.empty_title += 1
                continue

            text, word_count, truncated = normalize_summary(summary)
            if not text:
                stats.empty_summary += 1
                continue
            if word_count < min_summary_words:
                stats.stub_dropped += 1
                continue
            if truncated:
                stats.truncated += 1

            author_clean = author.strip() or None
            year = parse_year(date)
            raw_genres = parse_genres(genres)

            if author_clean is None:
                stats.missing_author += 1
            if year is None:
                stats.missing_year += 1
            if not raw_genres:
                stats.missing_genres += 1

            record = CmuRecord(
                wiki_id=wiki_id.strip(),
                freebase_id=freebase_id.strip(),
                title=title,
                author=author_clean,
                published_year=year,
                raw_genres=raw_genres,
                summary=text,
                summary_word_count=word_count,
                truncated=truncated,
            )

            key = dedupe_key(title, author_clean)
            existing = best.get(key)
            if existing is None:
                best[key] = record
            else:
                stats.duplicate_dropped += 1
                if record.summary_word_count > existing.summary_word_count:
                    best[key] = record

    for index, record in enumerate(best.values()):
        if limit is not None and index >= limit:
            break
        stats.emitted += 1
        yield record
