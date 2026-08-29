"""Compose the text that actually gets embedded.

**Raw prose is never embedded.** A story's text is plot incident — who did what
to whom, in order. Embedding that buries the signal a recommender ranks on
(premise, themes, tone) under a mass of proper nouns and sequence. So every item
is first normalized to a fixed schema — Title, Author, Genres, Core Premise, Key
Themes/Tropes, Tone — and only that is embedded.

The template is a pure function with a stable sha, which is what makes ingest
resumable: a row whose `embed_input_sha` is unchanged does not need re-embedding,
so re-reading every published story on each poll costs nothing.
"""

import hashlib
import re
from dataclasses import dataclass, field
from typing import List, Optional

_WHITESPACE = re.compile(r"\s+")

# Bumping this invalidates every stored embed_input_sha and forces a full
# re-embed. Change it only when the template itself changes in a way that should
# alter the vectors.
TEMPLATE_VERSION = 1


@dataclass
class NormalizedItem:
    """The embeddable shape, whatever the source."""

    title: str
    author: Optional[str] = None
    genres: List[str] = field(default_factory=list)
    core_premise: Optional[str] = None
    themes: List[str] = field(default_factory=list)
    tone: List[str] = field(default_factory=list)


def _clean(value: Optional[str]) -> str:
    return _WHITESPACE.sub(" ", value or "").strip()


def _clean_list(values: Optional[List[str]]) -> List[str]:
    """Dedupe case-insensitively while preserving the given order.

    Order is preserved rather than sorted because the normalization step emits
    themes roughly by salience, and the leading terms carry more weight in the
    embedding than the trailing ones.
    """
    seen = set()
    out: List[str] = []
    for value in values or []:
        cleaned = _clean(value)
        if not cleaned:
            continue
        fingerprint = cleaned.casefold()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        out.append(cleaned)
    return out


def compose_embed_input(item: NormalizedItem) -> str:
    """Render the embedding input.

    Absent fields drop their whole line rather than emitting an empty label.
    `Author: ` with nothing after it is not neutral — it is a token sequence the
    model will happily find similar to every other authorless book, quietly
    clustering 14% of the corpus (2,382 records) by a missing value.
    """
    lines: List[str] = [f"Title: {_clean(item.title)}"]

    author = _clean(item.author)
    if author:
        lines.append(f"Author: {author}")

    genres = _clean_list(item.genres)
    if genres:
        lines.append(f"Genres: {', '.join(genres)}")

    premise = _clean(item.core_premise)
    if premise:
        lines.append(f"Premise: {premise}")

    themes = _clean_list(item.themes)
    if themes:
        lines.append(f"Themes & tropes: {', '.join(themes)}")

    tone = _clean_list(item.tone)
    if tone:
        lines.append(f"Tone: {', '.join(tone)}")

    return "\n".join(lines)


def embed_input_sha(embed_input: str) -> str:
    """Content hash of the embedding input, namespaced by template version."""
    payload = f"v{TEMPLATE_VERSION}|{embed_input}"
    return hashlib.sha256(payload.encode()).hexdigest()


def summary_sha(summary: str, prompt_version: int) -> str:
    """Cache key for the LLM normalization pass.

    Keyed on the *truncated* summary the LLM actually saw, plus the prompt
    version, so a crashed 15k-record run resumes instead of re-billing — and a
    prompt change correctly forces regeneration.
    """
    payload = f"p{prompt_version}|{_clean(summary)}"
    return hashlib.sha256(payload.encode()).hexdigest()
