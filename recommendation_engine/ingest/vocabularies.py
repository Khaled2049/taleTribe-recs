"""Controlled vocabularies for themes/tropes and tone.

Why closed lists rather than letting the model write what it likes: free-text
labels over 15,500 books produce tens of thousands of near-synonyms — "revenge",
"vengeance", "seeking revenge", "revenge plot" — which makes three things
useless at once. The GIN index on `themes` stops being selective, a
themes-based filter matches almost nothing, and the diversity metric can no
longer tell two similar books apart.

Enforced twice, deliberately:

1. The vocabulary is included **in the prompt**, so the model picks from it.
2. Output is filtered against the allowlist **afterwards**, because a model told
   to pick from a list will still occasionally invent a term.

Step 2 is the one that actually guarantees the invariant. Step 1 is what keeps
step 2 from discarding most of the output.
"""

from collections.abc import Iterable

# ── Tone: how the book feels to read ────────────────────────────────────
# Small on purpose. Tone is a coarse signal and a long list would just invite
# the model to split hairs it cannot split consistently.
TONES: frozenset[str] = frozenset(
    {
        "absurdist",
        "austere",
        "bleak",
        "comic",
        "contemplative",
        "cozy",
        "detached",
        "dreamlike",
        "earnest",
        "epic",
        "feverish",
        "gritty",
        "hopeful",
        "intimate",
        "irreverent",
        "lyrical",
        "melancholy",
        "menacing",
        "nostalgic",
        "philosophical",
        "playful",
        "romantic",
        "sardonic",
        "satirical",
        "sentimental",
        "suspenseful",
        "tender",
        "tense",
        "tragic",
        "unsettling",
        "uplifting",
        "whimsical",
        "wry",
    }
)

# ── Themes and tropes: what the book is *about* ─────────────────────────
THEMES: frozenset[str] = frozenset(
    {
        # Growing up, identity
        "coming of age",
        "loss of innocence",
        "identity crisis",
        "secret identity",
        "mistaken identity",
        "doppelganger",
        "coming out",
        "queer identity",
        "gender transition",
        "disguise and cross-dressing",
        # Love and relationships
        "forbidden love",
        "enemies to lovers",
        "love triangle",
        "unrequited love",
        "second chance romance",
        "arranged marriage",
        "fake relationship",
        "slow burn romance",
        "star-crossed lovers",
        "marriage breakdown",
        "infidelity",
        "divorce",
        # Family
        "family secrets",
        "generational trauma",
        "dysfunctional family",
        "sibling rivalry",
        "orphan protagonist",
        "found family",
        "adoption",
        "single parenthood",
        "motherhood",
        "fatherhood",
        "infertility",
        "inheritance dispute",
        "generational saga",
        # Mentors, rivals, companions
        "mentor and protege",
        "unlikely friendship",
        "rivalry",
        "teacher and student",
        "ensemble cast",
        # Heroism and its costs
        "chosen one",
        "reluctant hero",
        "antihero",
        "prophecy",
        "quest",
        "fall from grace",
        "redemption arc",
        "sacrifice",
        "martyrdom",
        "vigilante justice",
        "moral ambiguity",
        "fate versus free will",
        # Added after a real run showed the model reaching for these repeatedly
        # and the filter discarding them. Genuine gaps, not near-misses.
        "good versus evil",
        "crime and punishment",
        "violence and its consequences",
        "love",
        "trauma and recovery",
        "greed",
        "ambition",
        "deception",
        "prejudice",
        "social injustice",
        "conspiracy",
        "jealousy",
        "guilt and atonement",
        "duty and honour",
        "freedom and confinement",
        # Betrayal and revenge
        "revenge",
        "betrayal",
        "blood feud",
        "false accusation",
        "wrongful imprisonment",
        # Crime and investigation
        "amateur sleuth",
        "hardboiled detective",
        "locked room mystery",
        "cold case",
        "serial killer",
        "organised crime",
        "gang loyalty",
        "heist",
        "con artist",
        "courtroom drama",
        "police procedural",
        "investigative journalism",
        "whistleblower",
        # Espionage and war
        "espionage",
        "double agent",
        "defection",
        "war and its aftermath",
        "survivor's guilt",
        "prisoner of war",
        "anti-war",
        "military life",
        # Power and politics
        "dystopian regime",
        "totalitarian surveillance",
        "rebellion against tyranny",
        "corruption of power",
        "political intrigue",
        "court intrigue",
        "class struggle",
        "social climbing",
        "rags to riches",
        "absurd bureaucracy",
        "corporate greed",
        "labour organising",
        "propaganda",
        # Place and displacement
        "journey home",
        "road trip",
        "immigration",
        "exile",
        "diaspora",
        "cultural assimilation",
        "colonialism",
        "decolonisation",
        "slavery and freedom",
        "racial injustice",
        "civil rights",
        "small town secrets",
        "urban alienation",
        "gentrification",
        "poverty",
        "homelessness",
        "farming life",
        "isolated community",
        # Survival
        "survival against nature",
        "shipwreck",
        "desert island",
        "sea voyage",
        "mountaineering",
        "exploration",
        "post-apocalyptic survival",
        "pandemic",
        "ecological collapse",
        "nuclear aftermath",
        # Science fiction
        "first contact",
        "alien invasion",
        "generation ship",
        "space colonisation",
        "terraforming",
        "artificial intelligence",
        "sentient machines",
        "robot uprising",
        "cybernetic enhancement",
        "virtual reality",
        "mind uploading",
        "genetic engineering",
        "cloning",
        "time travel",
        "time loop",
        "parallel worlds",
        "portal to another world",
        "alternate history",
        "scientific hubris",
        "experiment gone wrong",
        "forbidden knowledge",
        # Fantasy and myth
        "magic system",
        "magical academy",
        "dragons",
        "mythical beasts",
        "gods and mortals",
        "mythological retelling",
        "fairy tale retelling",
        "descent to the underworld",
        "immortality",
        "resurrection",
        "curses",
        "witchcraft",
        "talking animals",
        "anthropomorphic society",
        "chivalry",
        # Horror and the uncanny
        "ghosts and hauntings",
        "vampires",
        "werewolves",
        "zombie outbreak",
        "possession",
        "exorcism",
        "cosmic horror",
        "body horror",
        "madness and paranoia",
        "isolation",
        "cult",
        # Mind and body
        "mental illness",
        "addiction",
        "obsession",
        "gaslighting",
        "amnesia",
        "memory and forgetting",
        "unreliable memory",
        "grief and mourning",
        "terminal illness",
        "disability",
        "caregiving",
        "aging",
        "pregnancy and childbirth",
        "medical ethics",
        # Faith and philosophy
        "religious faith and doubt",
        "apostasy",
        "pilgrimage",
        "monastic life",
        "existential dread",
        "utopian experiment",
        "allegory of politics",
        "allegory of religion",
        # Art, learning, institutions
        "artistic ambition",
        "writer's block",
        "musical prodigy",
        "academic rivalry",
        "boarding school",
        "campus politics",
        "sporting triumph",
        "library or archive",
        "translation and mistranslation",
        "language barrier",
        # Storytelling itself
        "unreliable narrator",
        "book within a book",
        "frame narrative",
        "dual timeline",
        "epistolary revelations",
        "diary discovered",
        "storytelling as survival",
        "dreams and visions",
        "trickster figure",
        "satire of manners",
    }
)


def _index(vocabulary: Iterable[str]) -> dict:
    return {term.casefold(): term for term in vocabulary}


_THEME_INDEX = _index(THEMES)
_TONE_INDEX = _index(TONES)

# Variants seen often enough to be worth accepting rather than discarding. Kept
# deliberately short — an ever-growing alias table is a sign the vocabulary or
# the prompt needs fixing, not that more aliases are needed.
_THEME_ALIASES = {
    "vengeance": "revenge",
    "revenge plot": "revenge",
    "growing up": "coming of age",
    "bildungsroman": "coming of age",
    "ai": "artificial intelligence",
    "a.i.": "artificial intelligence",
    "dystopia": "dystopian regime",
    "post-apocalyptic": "post-apocalyptic survival",
    "apocalypse": "post-apocalyptic survival",
    "chosen-one": "chosen one",
    "found-family": "found family",
    "class conflict": "class struggle",
    "class warfare": "class struggle",
    "organized crime": "organised crime",
    "labor organising": "labour organising",
    "labor organizing": "labour organising",
    "space colonization": "space colonisation",
    "decolonization": "decolonisation",
    "haunting": "ghosts and hauntings",
    "ghosts": "ghosts and hauntings",
    "grief": "grief and mourning",
    "mourning": "grief and mourning",
    "madness": "madness and paranoia",
    "paranoia": "madness and paranoia",
    "memory": "memory and forgetting",
    "faith and doubt": "religious faith and doubt",
    "war": "war and its aftermath",
    "childbirth": "pregnancy and childbirth",
    "pregnancy": "pregnancy and childbirth",
    # Observed in a real normalization run: the model reached for a shorter form
    # of a term the vocabulary already has. These are near-misses, not new
    # concepts, so aliasing is the correct fix rather than widening the list.
    "free will": "fate versus free will",
    "free will versus determinism": "fate versus free will",
    "determinism": "fate versus free will",
    "fate": "fate versus free will",
    "destiny": "fate versus free will",
    "disillusionment": "loss of innocence",
    "exploitation": "class struggle",
    "violence": "violence and its consequences",
    "trauma": "trauma and recovery",
    "mentorship": "mentor and protege",
    "mentor": "mentor and protege",
    "atonement": "guilt and atonement",
    "guilt": "guilt and atonement",
    "redemption": "redemption arc",
    "honour": "duty and honour",
    "honor": "duty and honour",
    "duty": "duty and honour",
    "social criticism": "social injustice",
    "social commentary": "social injustice",
    "injustice": "social injustice",
    "racism": "racial injustice",
    "discrimination": "prejudice",
    "conspiracy theory": "conspiracy",
    "romantic love": "love",
    "first love": "love",
    "punishment and rehabilitation": "crime and punishment",
    "survival": "survival against nature",
    "corruption": "corruption of power",
    "abuse of power": "corruption of power",
    "rebellion": "rebellion against tyranny",
    "revolution": "rebellion against tyranny",
    "tyranny": "rebellion against tyranny",
    "totalitarianism": "totalitarian surveillance",
    "surveillance": "totalitarian surveillance",
    "epidemic": "pandemic",
    "plague": "pandemic",
    "quarantine": "pandemic",
    "self-discovery": "identity crisis",
    "identity": "identity crisis",
    "loss": "grief and mourning",
    "bereavement": "grief and mourning",
    "friendship": "unlikely friendship",
    "loneliness": "isolation",
    "solitude": "isolation",
    "imprisonment": "wrongful imprisonment",
    "colonisation": "colonialism",
    "colonization": "colonialism",
    "artificial life": "sentient machines",
    "robots": "sentient machines",
    "dystopian society": "dystopian regime",
}

_TONE_ALIASES = {
    "dark": "bleak",
    "grim": "bleak",
    "funny": "comic",
    "humorous": "comic",
    "humourous": "comic",
    "satiric": "satirical",
    "sad": "melancholy",
    "somber": "melancholy",
    "sombre": "melancholy",
    "tense and suspenseful": "suspenseful",
    "thrilling": "suspenseful",
    "warm": "tender",
    "heartwarming": "uplifting",
    "hopeful and uplifting": "uplifting",
    "creepy": "unsettling",
    "eerie": "unsettling",
    "poetic": "lyrical",
    "reflective": "contemplative",
    "meditative": "contemplative",
    "ironic": "wry",
    "cynical": "sardonic",
}


def _canonicalize(raw: str, index: dict, aliases: dict) -> str:
    """Return the canonical term, or "" when the value is outside the vocabulary."""
    key = " ".join((raw or "").split()).casefold()
    if not key:
        return ""
    if key in index:
        return index[key]
    aliased = aliases.get(key)
    if aliased:
        return aliased
    # Tolerate a trailing plural on an otherwise exact match ("curse" → "curses").
    if key.endswith("s") and key[:-1] in index:
        return index[key[:-1]]
    if f"{key}s" in index:
        return index[f"{key}s"]
    return ""


# Genre names the model routinely puts in `themes`. They are not theme gaps —
# the crosswalk already records genre on the item — so they are dropped quietly
# rather than inflating the violation counter and inviting bogus vocabulary
# additions.
KNOWN_GENRE_NOISE: frozenset[str] = frozenset(
    {
        "mystery",
        "murder mystery",
        "adventure",
        "romance",
        "science fiction",
        "fantasy",
        "horror",
        "thriller",
        "space opera",
        "legal drama",
        "tragedy",
        "comedy",
        "satire",
        "supernatural horror",
        "historical fiction",
        "non-fiction",
        "fiction",
        "drama",
        "biography",
        "philosophy",
    }
)


def is_genre_noise(value: str) -> bool:
    """True when a rejected term is a genre label rather than a missing theme."""
    return " ".join((value or "").split()).casefold() in KNOWN_GENRE_NOISE


def split_misfiled(theme_values: Iterable[str], tone_values: Iterable[str]) -> tuple:
    """Reclassify cross-axis terms instead of discarding them.

    A real 2,000-book run showed the model routinely putting the wrong axis in the
    wrong field: "philosophical" (58 times), "absurdist" (20) and "hopeful" (7)
    appeared under *themes*, where they are not valid themes and were being thrown
    away — even though all three are valid *tones*.

    So: a themes entry that is a recognized tone moves to tone, and vice versa.
    This is post-processing rather than a prompt change on purpose — a prompt
    change bumps PROMPT_VERSION and re-bills the entire corpus, while this
    recovers the same terms for free from cached raw output.

    Returns (themes, tones).
    """
    themes_out: list[str] = []
    tones_out: list[str] = list(tone_values or [])

    for value in theme_values or []:
        as_theme = _canonicalize(str(value), _THEME_INDEX, _THEME_ALIASES)
        if as_theme:
            themes_out.append(as_theme)
            continue
        # Not a theme — is it a tone that landed in the wrong field?
        as_tone = _canonicalize(str(value), _TONE_INDEX, _TONE_ALIASES)
        if as_tone:
            tones_out.append(as_tone)
        else:
            themes_out.append(str(value))  # left for the filter to drop and count

    return themes_out, tones_out


def filter_themes(values: Iterable[str]) -> list[str]:
    """Canonicalize and drop anything outside THEMES, preserving order.

    Order matters: the normalization prompt emits themes roughly by salience, and
    leading terms carry more weight in the composed embedding input.
    """
    return _filter(values, _THEME_INDEX, _THEME_ALIASES)


def filter_tones(values: Iterable[str]) -> list[str]:
    """Canonicalize and drop anything outside TONES, preserving order."""
    return _filter(values, _TONE_INDEX, _TONE_ALIASES)


def _filter(values: Iterable[str], index: dict, aliases: dict) -> list[str]:
    seen = set()
    out: list[str] = []
    for value in values or []:
        canonical = _canonicalize(str(value), index, aliases)
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        out.append(canonical)
    return out


def prompt_vocabulary_block() -> str:
    """The vocabulary as it appears in the normalization prompt.

    Sorted so the prompt text is stable: an unstable prompt would change the
    cache key on every run and re-bill the whole corpus.
    """
    themes = ", ".join(sorted(THEMES))
    tones = ", ".join(sorted(TONES))
    return (
        f"ALLOWED THEMES (choose only from this list):\n{themes}\n\n"
        f"ALLOWED TONES (choose only from this list):\n{tones}"
    )
