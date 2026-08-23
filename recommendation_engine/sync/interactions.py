"""The canonical interaction record, and the loader that consumes it.

This is the contract between *any* source of reader behaviour and the database.
Write a source that emits these and everything downstream works.

Shape (one JSON object per line in a `.jsonl` file):

    {"user_id": "synth_0042",
     "source": "cmu",
     "source_id": "620",
     "kind": "rating",
     "value": 4.0,
     "occurred_at": "2026-07-14T09:31:00Z"}

    {"user_id": "synth_0042",
     "source": "cmu",
     "source_id": "843",
     "kind": "progress",
     "value": 0.95,
     "chapter_index": 11,
     "total_chapters": 12,
     "occurred_at": "2026-07-15T20:02:00Z"}

Field notes, each with a reason:

* **`source` + `source_id`, never the internal `id`.** `items.id` is a surrogate key
  the database assigns; no external generator or Firestore export can know it. The
  `(source, source_id)` pair is the stable external identity, and the loader
  resolves it. Records naming an unknown book are counted and skipped, not fatal.
* **`kind` is one of `like` | `rating` | `progress`.** Exactly what Firestore has:
  `stories/{id}/likes/{uid}`, `stories/{id}/ratings/{uid}`, and
  `users/{uid}/readingProgress/{storyId}`. Nothing richer, because nothing richer
  exists.
* **`completion` is NOT a kind.** The platform emits no completion event, so it is
  *derived* here from progress — one definition, shared by every source, rather
  than each source inventing its own.
* **`value`** carries the rating (1-5) for `rating`, or `scroll_percent` (0-1) for
  `progress`. Unused for `like`.
* **`occurred_at`** is ISO-8601. Used for recency and for making a re-load
  idempotent.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

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
    """One reader signal, from any source."""

    user_id: str
    source: str
    source_id: str
    kind: str
    occurred_at: datetime
    value: Optional[float] = None
    chapter_index: Optional[int] = None
    total_chapters: Optional[int] = None

    def to_json(self) -> str:
        data = asdict(self)
        data["occurred_at"] = self.occurred_at.astimezone(timezone.utc).isoformat()
        return json.dumps({k: v for k, v in data.items() if v is not None})

    @classmethod
    def from_json(cls, line: str) -> "InteractionRecord":
        raw = json.loads(line)
        return cls(
            user_id=str(raw["user_id"]),
            source=str(raw["source"]),
            source_id=str(raw["source_id"]),
            kind=str(raw["kind"]),
            occurred_at=_parse_ts(raw["occurred_at"]),
            value=(None if raw.get("value") is None else float(raw["value"])),
            chapter_index=raw.get("chapter_index"),
            total_chapters=raw.get("total_chapters"),
        )

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


def _parse_ts(value) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


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


def read_jsonl(path: Path) -> Iterator[InteractionRecord]:
    """Stream records from a .jsonl file, skipping blank and malformed lines."""
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield InteractionRecord.from_json(line)
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                logger.warning("interaction_line_skipped line=%d error=%s", number, exc)


def write_jsonl(path: Path, records: Iterable[InteractionRecord]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.to_json() + "\n")
            count += 1
    return count


async def _resolve_items(pool, keys: Sequence[Tuple[str, str]]) -> Dict[tuple, int]:
    """Map (source, source_id) -> items.id."""
    if not keys:
        return {}
    sources = [k[0] for k in keys]
    source_ids = [k[1] for k in keys]
    rows = await pool.fetch(
        "SELECT id, source::text AS source, source_id FROM recommendations.items "
        "WHERE (source::text, source_id) IN "
        "(SELECT * FROM UNNEST($1::text[], $2::text[]))",
        sources,
        source_ids,
    )
    return {(row["source"], row["source_id"]): row["id"] for row in rows}


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

    resolved = await _resolve_items(
        pool, list({(r.source, r.source_id) for r in batch})
    )

    rows: List[tuple] = []
    for record in batch:
        item_id = resolved.get((record.source, record.source_id))
        if item_id is None:
            stats.unknown_item += 1
            if len(stats.unknown_examples) < 10:
                stats.unknown_examples.append(f"{record.source}:{record.source_id}")
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
