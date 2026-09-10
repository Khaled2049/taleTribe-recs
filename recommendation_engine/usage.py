"""Durable daily ceilings for the paths that spend Gemini tokens.

`PerUserRateLimiter` already sits in front of these routes, but it answers a
different question. It is a *burst* guard, held in process memory, so its own
docstring is careful to say the effective ceiling is `instances x limit`; it
also forgets everything when Cloud Run recycles a container. Neither property is
acceptable for a spend limit — a budget that multiplies by the autoscaler and
resets on deploy is not a budget.

So this is the durable half: the bucket bounds the rate, this bounds the *day*.
It lives in Postgres because that is the only state recs shares across
instances, and it is keyed on a UTC day assigned by the database so two
instances cannot disagree about when it resets.

Two ceilings, because they fail differently:

* **Per user** (`recommendations.llm_usage`) — fairness. One reader cannot spend
  everyone else's allowance.
* **Platform-wide** (`recommendations.llm_platform_usage`) — solvency. Per-user
  budgets multiply by the user count, so on their own they bound nothing in
  total. This is the analogue of creditProxy's PLATFORM_DAILY_REQUEST_LIMIT,
  which cannot cover these routes because recs calls Gemini directly.

Within each, `search` (HyDE) and `explain` are budgeted separately: they are
separate user-visible actions with very different fan-out, and a reader who
clicks "Why this story?" should not lose searches for it.

The charge is taken *before* the generation, not after. A reservation can
over-count when a generation then fails, which is the right way round: the
alternative bills nothing for a request that already spent tokens upstream.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

logger = logging.getLogger(__name__)

# Budget names. These are the `kind` values stored in both tables, so they are
# part of the schema contract, not free text.
SEARCH = "search"
EXPLAIN = "explain"

# Substituted for a disabled budget so one statement covers every combination.
# int4 max: a guard that cannot trip, on a column that cannot reach it.
_NO_LIMIT = 2**31 - 1


class DailyBudgetExceeded(Exception):
    """This user has spent their whole daily allowance for this kind."""

    def __init__(self, kind: str, limit: int) -> None:
        super().__init__(f"daily {kind} limit of {limit} reached")
        self.kind = kind
        self.limit = limit


class PlatformBudgetExceeded(Exception):
    """The platform has spent its whole daily allowance for this kind.

    Distinct from `DailyBudgetExceeded` because it means something different to
    everyone involved: the caller did nothing wrong and has budget left, and an
    operator needs to know the service is at capacity rather than that one user
    is enthusiastic.
    """

    def __init__(self, kind: str, limit: int) -> None:
        super().__init__(f"platform daily {kind} limit of {limit} reached")
        self.kind = kind
        self.limit = limit


class MeterUnavailable(Exception):
    """The counters could not be read or written.

    Raised rather than swallowed. The in-memory bucket is deliberately
    fail-open, because a limiter that denies service when its own state is
    unavailable is worse than the abuse it prevents. This one is the opposite:
    its entire job is to stop unmetered spend, so a meter that cannot record a
    charge must not authorize one.
    """


class DailyLlmMeter:
    """Per-user and platform-wide call counters for the token-spending routes.

    A limit of 0 (or less) means unlimited. When *both* budgets for a kind are
    unlimited the charge is skipped entirely, so disabling the meter costs
    nothing rather than costing a round trip that can never refuse anything.
    """

    # One statement for both counters, and the ordering inside it is load
    # bearing: the platform INSERT selects `FROM charged_user`, so it produces
    # no row — and increments nothing — when the user's own guard refused.
    #
    # That direction is the safe one. Charging the platform first would let a
    # single user who has already exhausted their personal budget keep driving
    # the platform counter with requests that are refused anyway, which is a
    # denial-of-service against every other reader's access to the feature.
    #
    # The reverse cost is real but small: when the platform is at capacity, a
    # user's counter still moves for a request that was refused. It is bounded
    # by their own daily budget and only happens while the platform is already
    # capped, which is the moment fairness matters least.
    _CHARGE_SQL = """
        WITH charged_user AS (
            INSERT INTO recommendations.llm_usage AS u
                        (user_id, day, kind, call_count)
            VALUES ($1, (now() AT TIME ZONE 'utc')::date, $2, 1)
            ON CONFLICT (user_id, day, kind) DO UPDATE
               SET call_count = u.call_count + 1
             WHERE u.call_count < $3
            RETURNING call_count
        ), charged_platform AS (
            INSERT INTO recommendations.llm_platform_usage AS p
                        (day, kind, call_count)
            SELECT (now() AT TIME ZONE 'utc')::date, $2, 1 FROM charged_user
            ON CONFLICT (day, kind) DO UPDATE
               SET call_count = p.call_count + 1
             WHERE p.call_count < $4
            RETURNING call_count
        )
        SELECT (SELECT call_count FROM charged_user)     AS user_count,
               (SELECT call_count FROM charged_platform) AS platform_count
    """

    _USER_USED_SQL = """
        SELECT call_count FROM recommendations.llm_usage
         WHERE user_id = $1 AND day = (now() AT TIME ZONE 'utc')::date AND kind = $2
    """

    _PLATFORM_USED_SQL = """
        SELECT call_count FROM recommendations.llm_platform_usage
         WHERE day = (now() AT TIME ZONE 'utc')::date AND kind = $1
    """

    def __init__(
        self,
        db,
        limits: Mapping[str, int],
        platform_limits: Mapping[str, int] | None = None,
    ) -> None:
        self._db = db
        self._limits = dict(limits)
        self._platform_limits = dict(platform_limits or {})

    def limit_for(self, kind: str) -> int:
        return self._limits.get(kind, 0)

    def platform_limit_for(self, kind: str) -> int:
        return self._platform_limits.get(kind, 0)

    async def charge(self, user_id: str, kind: str) -> int | None:
        """Consume one unit of `kind` for `user_id` and for the platform.

        Returns the user's new daily total, or None when both budgets for this
        kind are unlimited. Raises `DailyBudgetExceeded` or
        `PlatformBudgetExceeded` when the respective allowance is spent, and
        `MeterUnavailable` when the counters could not be written.
        """
        user_limit = self.limit_for(kind)
        platform_limit = self.platform_limit_for(kind)
        if user_limit <= 0 and platform_limit <= 0:
            return None

        try:
            row = await self._db.write_pool.fetchrow(
                self._CHARGE_SQL,
                user_id,
                kind,
                user_limit if user_limit > 0 else _NO_LIMIT,
                platform_limit if platform_limit > 0 else _NO_LIMIT,
            )
        except Exception as exc:  # asyncpg raises a wide family here
            logger.warning(
                "llm_meter_unavailable kind=%s error=%s", kind, str(exc)[:200]
            )
            raise MeterUnavailable(str(exc)) from exc

        # A null count means that INSERT's guard refused the increment, which
        # only happens when the stored count already reached its limit.
        if row is None or row["user_count"] is None:
            raise DailyBudgetExceeded(kind, user_limit)
        if row["platform_count"] is None:
            # Worth a log line at warning: unlike a per-user refusal this is an
            # operational event, and the first sign of it is readers being told
            # the feature is closed for the day.
            logger.warning(
                "llm_platform_budget_exhausted kind=%s limit=%s", kind, platform_limit
            )
            raise PlatformBudgetExceeded(kind, platform_limit)
        return row["user_count"]

    async def used_today(self, user_id: str, kind: str) -> int:
        """Read a user's count for the day without charging. Diagnostics."""
        row = await self._db.read_pool.fetchval(self._USER_USED_SQL, user_id, kind)
        return int(row or 0)

    async def platform_used_today(self, kind: str) -> int:
        """Read the platform's count for the day without charging."""
        row = await self._db.read_pool.fetchval(self._PLATFORM_USED_SQL, kind)
        return int(row or 0)
