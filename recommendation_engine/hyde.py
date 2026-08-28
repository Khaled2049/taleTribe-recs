"""HyDE — Hypothetical Document Embeddings.

The problem it solves: a reader's phrasing and a catalog entry are different kinds
of text. "something cozy where nobody dies" is a *request*; the catalog holds
premises, themes and tone. Embedding the request directly asks the index to match
across that gap, and short or vague requests carry too little signal to land
anywhere useful.

So instead: have the LLM write the book the reader is describing — as a
*hypothetical catalog entry* — and embed that. The query and the documents are then
the same kind of object, and retrieval compares like with like.

The generated text deliberately mirrors `compose.compose_embed_input` field for
field. That is the whole trick: a hypothetical document in a different format would
land in a different region of the space and defeat the purpose.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import List, Optional, cast

from recommendation_engine.ingest.vocabularies import prompt_vocabulary_block
from recommendation_engine.llm import GeminiClient, LLMError

logger = logging.getLogger(__name__)

# Bump when the prompt changes: it participates in explanation cache keys via the
# generated text, and a changed prompt should not silently reuse old vectors.
HYDE_PROMPT_VERSION = 1

_SYSTEM = (
    "You invent plausible catalog entries for a book recommendation engine. You "
    "write in the engine's exact metadata format and nothing else."
)

_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "genres": {"type": "array", "items": {"type": "string"}},
        "core_premise": {"type": "string"},
        "themes": {"type": "array", "items": {"type": "string"}},
        "tone": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "core_premise", "themes", "tone"],
}


@dataclass
class HydeResult:
    """The hypothetical document, plus the text that was actually embedded."""

    embed_text: str
    title: str
    core_premise: str
    themes: List[str]
    tone: List[str]

    def as_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "core_premise": self.core_premise,
            "themes": self.themes,
            "tone": self.tone,
        }


def build_prompt(query: str) -> str:
    """Ask for one hypothetical catalog entry matching the reader's request."""
    return (
        f"{prompt_vocabulary_block()}\n\n"
        "A reader asked for book recommendations with this request:\n\n"
        f"  {query}\n\n"
        "Invent ONE hypothetical book that would perfectly satisfy it, and describe "
        "it as a catalog entry:\n"
        "- title: a plausible title for this imagined book\n"
        "- genres: 1-3 broad category names\n"
        "- core_premise: at most 60 words, the central situation and conflict, "
        "written the way a catalog blurb would state it\n"
        "- themes: 3-6, chosen ONLY from the allowed themes list above\n"
        "- tone: 2-4, chosen ONLY from the allowed tones list above\n\n"
        "Describe the book itself. Do not address the reader, do not mention the "
        "request, and do not recommend a real existing title."
    )


def _strings(value: object | None) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in cast(list[object], value)]


def compose_hyde_text(payload: Mapping[str, object]) -> str:
    """Render the hypothetical entry in the catalog's own embedding format.

    Field-for-field identical to `compose.compose_embed_input`, minus Author —
    inventing an author name would add a strong, meaningless signal, and a
    hypothetical book has no real one.
    """
    from recommendation_engine.ingest.compose import (
        NormalizedItem,
        compose_embed_input,
    )
    from recommendation_engine.ingest.vocabularies import filter_themes, filter_tones

    return compose_embed_input(
        NormalizedItem(
            title=str(payload.get("title") or "Untitled"),
            author=None,
            genres=_strings(payload.get("genres")),
            core_premise=str(payload.get("core_premise") or ""),
            # Same vocabulary discipline as the catalog: an invented theme would
            # be a token the corpus never uses.
            themes=filter_themes(_strings(payload.get("themes"))),
            tone=filter_tones(_strings(payload.get("tone"))),
        )
    )


async def generate(
    client: GeminiClient, query: str, max_output_tokens: int = 512
) -> Optional[HydeResult]:
    """Generate a hypothetical document for `query`.

    Returns None on any failure. The caller then embeds the raw query instead —
    degraded but still useful, which is the right trade for an optional
    enhancement. A HyDE failure must never fail the recommendation request.
    """
    query = (query or "").strip()
    if not query:
        return None

    try:
        generated = await client.generate_json(
            build_prompt(query),
            response_schema=_RESPONSE_SCHEMA,
            system=_SYSTEM,
            max_output_tokens=max_output_tokens,
            # Some invention is wanted here, unlike the extraction pass — but not
            # so much that the same request wanders around the space between calls.
            temperature=0.4,
        )
    except LLMError as exc:
        logger.warning("hyde_generation_failed error=%s", str(exc)[:200])
        return None

    if not isinstance(generated, dict):
        logger.warning("hyde_generation_invalid_shape")
        return None
    payload = cast(dict[str, object], generated)

    embed_text = compose_hyde_text(payload)
    if not embed_text.strip():
        return None

    return HydeResult(
        embed_text=embed_text,
        title=str(payload.get("title") or ""),
        core_premise=str(payload.get("core_premise") or ""),
        themes=_strings(payload.get("themes")),
        tone=_strings(payload.get("tone")),
    )
