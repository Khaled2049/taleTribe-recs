"""Synthetic reader signals, and the loader that writes them.

**Real reader signals do not come through here any more.** story-data derives
them straight from `story_likes`, `story_ratings` and `reading_progress` into
`recommendations.interactions` (`internal/store/recommendations.go`), because
`reading_progress` is private per-user data this service is not permitted to
read. The JSONL export format that used to be the contract between the two is
gone with it.

What remains is the in-memory shape `sync/synthetic.py` generates and `load()`
writes — the only way to exercise scoring before real traffic exists. Those rows
are prefixed `synth_` so the story-data sync neither deletes nor derives over
them.

Field notes, each with a reason:

* **`story_id`, never the internal `id`.** `items.id` is a surrogate key the
  database assigns; no generator can know it. The story's UUID is the stable
  identity, and the loader resolves it. Records naming an unknown story are
  counted and skipped, not fatal.
* **`kind` is one of `like` | `rating` | `progress`.** Exactly what the platform
  records.
* **`completion` is NOT a kind.** It is *derived* from progress, here and in
  story-data's SQL — two implementations of one rule, pinned together by
  `TestCompletionNeedsLastChapterAndDeepScroll` in story-data.
* **`value`** carries the rating (1-5) for `rating`, or `scroll_percent` (0-1)
  for `progress`. Unused for `like`.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence

from recommendation_engine.scoring import is_negative_signal, seed_engagement_weight

logger = logging.getLogger(__name__)

KIND_LIKE = "like"
KIND_RATING = "rating"
KIND_PROGRESS = "progress"
# Derived, never supplied by a source.
KIND_COMPLETION = "completion"

VALID_KINDS = frozenset({KIND_LIKE, KIND_RATING, KIND_PROGRESS})

# A reader who is on the last chapter and near the bottom of it has finished the
# book. Both halves are needed: chapter index alone counts someone who opened the
# final chapter and stopped.
COMPLETION_SCROLL_THRESHOLD = 0.9

# Prefix marking synthetic readers, so they can be found and purged, and never
# silently mixed into a real metric. Firebase uids are 28-char alphanumerics and
# cannot collide with this.
SYNTHETIC_USER_PREFIX = "synth_"


@dataclass
class InteractionRecord:
    """One synthetic reader signal."""

    user_id: str
    story_id: str
    kind: str
    occurred_at: datetime
    value: Optional[float] = None
    chapter_index: Optional[int] = None
    total_chapters: Optional[int] = None

    @property
    def is_complete(self) -> bool:
        """Whether this progress record represents finishing the book.

        Derived rather than reported, because the platform has no completion event.
        """
        if self.kind != KIND_PROGRESS:
            return False
        if self.total_chapters is None or self.chapter_index is None:
            return False
        if self.total_chapters <= 0:
            return False
        on_last_chapter = self.chapter_index >= self.total_chapters - 1
        scrolled = (self.value or 0.0) >= COMPLETION_SCROLL_THRESHOLD
        return on_last_chapter and scrolled


@dataclass
class LoadStats:
    read: int = 0
    invalid_kind: int = 0
    unknown_item: int = 0
    written: int = 0
    completions_derived: int = 0
    negative_signals: int = 0
    users: int = 0
    unknown_examples: List[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        data = self.__dict__.copy()
        data["unknown_examples"] = self.unknown_examples[:10]
        return data



async def _resolve_items(pool, story_ids: Sequence[str]) -> Dict[str, int]:
    """Map story_id -> items.id."""
    if not story_ids:
        return {}
    rows = await pool.fetch(
        "SELECT id, story_id FROM recommendations.items "
        "WHERE story_id = ANY($1::uuid[])",
        list(story_ids),
    )
    return {str(row["story_id"]): row["id"] for row in rows}


async def load(pool, records: Iterable[InteractionRecord]) -> LoadStats:
    """Write records into `recommendations.interactions`.

    Idempotent: the primary key is `(user_id, item_id, kind)`, and a conflict keeps
    whichever signal is **more recent**. That matters because `readingProgress` is
    current *state* rather than an event — re-exporting it must update a reader's
    position, not accumulate duplicates.

    A `progress` record that satisfies the completion rule writes **two** rows: the
    progress itself and a derived `completion`. Both are kept because they carry
    different information — how far they got, and that they finished.
    """
    stats = LoadStats()
    batch: List[InteractionRecord] = []
    users = set()

    for record in records:
        stats.read += 1
        if record.kind not in VALID_KINDS:
            stats.invalid_kind += 1
            continue
        batch.append(record)
        users.add(record.user_id)

    stats.users = len(users)
    if not batch:
        return stats

    resolved = await _resolve_items(pool, list({r.story_id for r in batch}))

    rows: List[tuple] = []
    for record in batch:
        item_id = resolved.get(record.story_id)
        if item_id is None:
            stats.unknown_item += 1
            if len(stats.unknown_examples) < 10:
                stats.unknown_examples.append(record.story_id)
            continue

        if is_negative_signal(record.kind, record.value):
            stats.negative_signals += 1

        rows.append(
            (
                record.user_id,
                item_id,
                record.kind,
                seed_engagement_weight(record.kind, record.value),
                record.value,
                record.occurred_at,
            )
        )

        if record.is_complete:
            stats.completions_derived += 1
            rows.append(
                (
                    record.user_id,
                    item_id,
                    KIND_COMPLETION,
                    seed_engagement_weight(KIND_COMPLETION),
                    1.0,
                    record.occurred_at,
                )
            )

    if rows:
        await pool.executemany(
            """
            INSERT INTO recommendations.interactions
                (user_id, item_id, kind, weight, value, occurred_at)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (user_id, item_id, kind) DO UPDATE SET
                weight = EXCLUDED.weight,
                value = EXCLUDED.value,
                occurred_at = GREATEST(
                    recommendations.interactions.occurred_at, EXCLUDED.occurred_at
                )
            WHERE EXCLUDED.occurred_at >= recommendations.interactions.occurred_at
            """,
            rows,
        )
        stats.written = len(rows)

    return stats


async def purge_synthetic(pool, prefix: str = SYNTHETIC_USER_PREFIX) -> int:
    """Remove synthetic readers' signals and derived state.

    Exists so seed data can never quietly become part of a real measurement. Also
    drops their taste vectors, which are derived from these rows.

    `prefix` is narrowable purely so tests can purge only their own fixtures. The
    default wipes every `synth_` reader, which is what an operator wants — and which
    is exactly why a test calling it unscoped destroyed a developer's seeded dataset
    once already.
    """
    if not prefix:
        raise ValueError("refusing to purge with an empty prefix")
    await pool.execute(
        "DELETE FROM recommendations.user_taste WHERE user_id LIKE $1",
        prefix + "%",
    )
    result = await pool.execute(
        "DELETE FROM recommendations.interactions WHERE user_id LIKE $1",
        prefix + "%",
    )
    try:
        return int(str(result).split()[-1])
    except (ValueError, IndexError):
        return 0
