"""Unit tests for the durable daily LLM budget.

No database: the meter's contract with Postgres is one statement whose *return
value* carries the decision, so a fake pool that returns a count, None, or an
error covers every branch. The statement's own correctness — that the ON
CONFLICT guard really refuses to increment past the limit — is asserted against
a real database in test_rec_routes_integration.py, since only Postgres can
answer that.

Follows the repo convention: env at import time, `pytestmark` at module level.
"""

import pytest
from pydantic import ValidationError

from recommendation_engine.config import RecSettings  # noqa: E402
from recommendation_engine.usage import (  # noqa: E402
    _NO_LIMIT,
    EXPLAIN,
    SEARCH,
    DailyBudgetExceeded,
    DailyLlmMeter,
    MeterUnavailable,
    PlatformBudgetExceeded,
)

pytestmark = pytest.mark.unit


def row(user_count, platform_count):
    """The two-column result `charge` reads. A null count means that INSERT's
    guard refused the increment."""
    return {"user_count": user_count, "platform_count": platform_count}


class _Pool:
    """Records every call and returns a scripted result."""

    def __init__(self, result=None, raises: Exception | None = None) -> None:
        self.result = result
        self.raises = raises
        self.calls: list[tuple] = []

    async def fetchrow(self, sql, *args):
        self.calls.append((sql, args))
        if self.raises is not None:
            raise self.raises
        return self.result


class _Db:
    def __init__(self, pool: _Pool) -> None:
        self.write_pool = pool
        self.read_pool = pool


# ── Charging ────────────────────────────────────────────────────────────


async def test_charge_returns_the_users_running_total():
    pool = _Pool(result=row(3, 40))
    meter = DailyLlmMeter(_Db(pool), {SEARCH: 10}, {SEARCH: 1000})

    assert await meter.charge("u1", SEARCH) == 3

    _, args = pool.calls[0]
    assert args == ("u1", SEARCH, 10, 1000), "both limits are bound params"


async def test_exhausted_user_budget_raises_with_the_limit_that_was_hit():
    meter = DailyLlmMeter(_Db(_Pool(result=row(None, None))), {SEARCH: 10})

    with pytest.raises(DailyBudgetExceeded) as excinfo:
        await meter.charge("u1", SEARCH)

    assert excinfo.value.limit == 10
    assert excinfo.value.kind == SEARCH


async def test_exhausted_platform_budget_is_a_distinct_failure():
    """The caller has budget left; the platform does not. Conflating the two
    would tell a blameless reader they asked for too much."""
    pool = _Pool(result=row(4, None))
    meter = DailyLlmMeter(_Db(pool), {SEARCH: 10}, {SEARCH: 1000})

    with pytest.raises(PlatformBudgetExceeded) as excinfo:
        await meter.charge("u1", SEARCH)

    assert excinfo.value.limit == 1000
    assert excinfo.value.kind == SEARCH


async def test_user_refusal_wins_when_both_are_exhausted():
    """The SQL charges the user first and gates the platform on it, so a user
    who is already over cannot keep driving the platform counter."""
    meter = DailyLlmMeter(_Db(_Pool(result=row(None, None))), {SEARCH: 10}, {SEARCH: 1})

    with pytest.raises(DailyBudgetExceeded):
        await meter.charge("u1", SEARCH)


async def test_budgets_are_independent_per_kind():
    """A reader clicking "Why this story?" must not spend their searches."""
    pool = _Pool(result=row(1, 1))
    meter = DailyLlmMeter(
        _Db(pool), {SEARCH: 10, EXPLAIN: 30}, {SEARCH: 1000, EXPLAIN: 2000}
    )

    await meter.charge("u1", SEARCH)
    await meter.charge("u1", EXPLAIN)

    assert [args[1] for _, args in pool.calls] == [SEARCH, EXPLAIN]
    assert [args[2] for _, args in pool.calls] == [10, 30]
    assert [args[3] for _, args in pool.calls] == [1000, 2000]


# ── The unlimited escape hatch ──────────────────────────────────────────


@pytest.mark.parametrize("limit", [0, -1])
async def test_both_budgets_unlimited_costs_no_round_trip(limit):
    pool = _Pool(result=row(1, 1))
    meter = DailyLlmMeter(_Db(pool), {SEARCH: limit}, {SEARCH: limit})

    assert await meter.charge("u1", SEARCH) is None
    assert pool.calls == [], "an unlimited budget must not touch the database"


async def test_platform_budget_alone_still_meters():
    """Turning off the per-user budget must not turn off the platform ceiling."""
    pool = _Pool(result=row(7, 12))
    meter = DailyLlmMeter(_Db(pool), {SEARCH: 0}, {SEARCH: 1000})

    assert await meter.charge("u1", SEARCH) == 7
    _, args = pool.calls[0]
    assert args[2] == _NO_LIMIT, "a disabled user budget becomes a guard that "
    assert args[3] == 1000


async def test_user_budget_alone_still_meters():
    pool = _Pool(result=row(2, 9))
    meter = DailyLlmMeter(_Db(pool), {SEARCH: 10}, {SEARCH: 0})

    assert await meter.charge("u1", SEARCH) == 2
    _, args = pool.calls[0]
    assert args[2] == 10
    assert args[3] == _NO_LIMIT


async def test_unknown_kind_is_unlimited_rather_than_a_crash():
    pool = _Pool(result=row(1, 1))
    meter = DailyLlmMeter(_Db(pool), {SEARCH: 10}, {SEARCH: 1000})

    assert await meter.charge("u1", "some-future-kind") is None
    assert pool.calls == []


# ── Failure mode ────────────────────────────────────────────────────────


async def test_database_failure_raises_rather_than_admitting_the_request():
    """Fail closed. The in-memory bucket is fail-open on purpose; this is not,
    because it guards real spend outside creditProxy."""
    meter = DailyLlmMeter(_Db(_Pool(raises=RuntimeError("pool closed"))), {SEARCH: 10})

    with pytest.raises(MeterUnavailable):
        await meter.charge("u1", SEARCH)


# ── The statement itself ────────────────────────────────────────────────


async def test_charge_is_a_single_atomic_statement():
    """Read-then-write would let two concurrent requests both see `limit - 1`,
    and two statements would let one commit without the other."""
    pool = _Pool(result=row(1, 1))
    meter = DailyLlmMeter(_Db(pool), {SEARCH: 10}, {SEARCH: 1000})
    await meter.charge("u1", SEARCH)

    sql, _ = pool.calls[0]
    assert len(pool.calls) == 1
    assert "ON CONFLICT" in sql and "WHERE u.call_count < $3" in sql
    assert "WHERE p.call_count < $4" in sql
    assert "AT TIME ZONE 'utc'" in sql, "the day boundary is the database's, not ours"


async def test_the_platform_charge_is_gated_on_the_user_charge():
    """`FROM charged_user` is what stops an over-budget user from draining the
    platform counter with requests that are refused anyway."""
    pool = _Pool(result=row(1, 1))
    meter = DailyLlmMeter(_Db(pool), {SEARCH: 10}, {SEARCH: 1000})
    await meter.charge("u1", SEARCH)

    sql, _ = pool.calls[0]
    assert "FROM charged_user" in sql
    assert sql.index("charged_user AS") < sql.index("charged_platform AS")


# ── Settings ────────────────────────────────────────────────────────────


def test_default_search_budget_is_ten_per_day(monkeypatch):
    """The shipped default, read with the environment out of the way.

    `test_rec_routes_integration` sets this var at import time to opt its own
    assertions out of the budget, and env vars are process-global, so a bare
    `RecSettings()` here would read that value rather than the default.
    """
    for var in (
        "RECS_MAX_SEARCHES_PER_DAY_PER_USER",
        "RECS_MAX_EXPLANATIONS_PER_DAY_PER_USER",
        "RECS_MAX_SEARCHES_PER_DAY_PLATFORM",
        "RECS_MAX_EXPLANATIONS_PER_DAY_PLATFORM",
    ):
        monkeypatch.delenv(var, raising=False)
    settings = RecSettings()
    assert settings.recs_max_searches_per_day_per_user == 10
    assert settings.recs_max_explanations_per_day_per_user == 30
    assert settings.recs_max_searches_per_day_platform == 1000
    assert settings.recs_max_explanations_per_day_platform == 1000


def test_daily_budgets_clamp_negatives_to_unlimited():
    settings = RecSettings(recs_max_searches_per_day_per_user=-5)
    assert settings.recs_max_searches_per_day_per_user == 0


def test_unparseable_daily_budget_is_rejected_not_silently_disabled():
    """0 means unlimited here, so falling back to it would turn the budget off."""
    with pytest.raises(ValidationError):
        RecSettings(recs_max_searches_per_day_per_user="garbage")
