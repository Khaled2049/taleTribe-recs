"""In-memory token bucket rate limiter, keyed by user id or client IP.

NOTE: state is held in process memory. On horizontally-scaled deployments
(e.g. Cloud Run with N instances), each instance maintains its own buckets,
so the *effective* per-key ceiling is `N * max_per_minute`, not the
configured value. For a true global cap, swap this for a Redis/Memorystore
backed bucket. As a coarse abuse guard this is sufficient.

## Why the bucket table is bounded

Some instances of this limiter are keyed by authenticated user id, whose
cardinality is bounded by the user count. Others (the MCP OAuth endpoints, see
mcp_server/throttle.py) are keyed by **client IP from the open internet**, which
is unbounded and attacker-chosen. A plain dict plus idle-based eviction does not
survive that second key space: during an active flood nothing is idle yet, so
nothing is evictable, and the table grows without limit while every request pays
to scan it.

So the table is a fixed-capacity LRU instead. Insert and eviction are O(1), and
resident memory has a hard ceiling regardless of how many distinct keys arrive.

Eviction is deliberately **fail-open**: the evicted key's next request starts a
fresh bucket with a full budget. The alternative — refusing unknown keys once
the table is full — is fail-closed, and would let anyone who can fill the table
deny service to every legitimate caller. That is far worse than the thing it
would prevent, because a per-IP limiter cannot bound an attacker with unlimited
addresses in the first place: rotating IPs evades it whatever the table size.
This bound exists to protect *this process* from the flood, not to stop it.
Distributed abuse needs an infrastructure answer (Cloud Armor in front of
/register, say), which is out of scope for a process-local bucket.

There is no time-based eviction. A bucket untouched for a full minute has
refilled completely, making it indistinguishable from an absent one, so evicting
it early buys nothing once memory is already capped.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from dataclasses import dataclass

# Measured at 225 B/entry with slotted buckets and IPv6-length keys: ~4.3 MB for
# a full table, ~21 MB across every limiter instance the service builds. The cap
# itself is pinned by tests/test_rate_limit.py; the byte figure is not, so treat
# it as a sizing note rather than a guarantee.
DEFAULT_MAX_TRACKED_KEYS = 20_000


@dataclass(slots=True)
class _Bucket:
    tokens: float
    last_update: float


class PerUserRateLimiter:
    """Token bucket keyed by user id or IP. Limit applies per rolling minute.

    Process-local and capacity-bounded: see the module docstring for the
    horizontal-scaling and LRU-eviction caveats.
    """

    def __init__(
        self,
        max_per_minute: int,
        max_tracked_keys: int = DEFAULT_MAX_TRACKED_KEYS,
    ) -> None:
        self._max_per_minute = max(0, max_per_minute)
        self._max_tracked_keys = max(1, max_tracked_keys)
        # Ordered oldest-touched first, so popitem(last=False) evicts the LRU.
        self._buckets: OrderedDict[str, _Bucket] = OrderedDict()
        self._lock = asyncio.Lock()
        self._evictions = 0

    @property
    def max_per_minute(self) -> int:
        return self._max_per_minute

    @property
    def tracked_keys(self) -> int:
        """Live bucket count. Never exceeds max_tracked_keys."""
        return len(self._buckets)

    @property
    def evictions(self) -> int:
        """Buckets dropped under capacity pressure.

        A steadily climbing value means the key space is larger than the table,
        i.e. the limiter is being outrun — the signal to reach for Cloud Armor
        or a shared store rather than a bigger dict.
        """
        return self._evictions

    async def allow(self, user_id: str) -> bool:
        """Return True if the request is allowed, False if rate limited."""
        if self._max_per_minute <= 0:
            return True

        key = user_id or "anonymous"
        now = time.monotonic()
        refill_rate = self._max_per_minute / 60.0

        async with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                self._admit(key, now)
                return True

            self._buckets.move_to_end(key)
            elapsed = now - bucket.last_update
            bucket.tokens = min(
                float(self._max_per_minute),
                bucket.tokens + elapsed * refill_rate,
            )
            bucket.last_update = now

            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True

            return False

    def reset(self) -> None:
        """Clear all buckets (for tests)."""
        self._buckets.clear()
        self._evictions = 0

    def _admit(self, key: str, now: float) -> None:
        """Insert a bucket for a new key, evicting the LRU one if at capacity."""
        while len(self._buckets) >= self._max_tracked_keys:
            self._buckets.popitem(last=False)
            self._evictions += 1
        self._buckets[key] = _Bucket(
            tokens=float(self._max_per_minute - 1),
            last_update=now,
        )
