"""
Tests for formal risk invariants: guarded division, Kelly criterion bounds,
monotonic stoploss ratchets, leverage constraints, and drawdown bounds.
"""

import numpy as np
import pytest

from causal_dsp.invariants import (
    InvariantViolationError,
    drawdown_bound,
    guarded_div,
    guarded_div_array,
    kelly_criterion,
    leverage_constraint,
    monotonic_stoploss_ratchet,
)


def test_guarded_div_zero_and_near_zero() -> None:
    # Strict zero
    assert guarded_div(10.0, 0.0, default=99.0) == 99.0
    # Below epsilon
    assert guarded_div(10.0, 1e-15, default=0.0, epsilon=1e-12) == 0.0
    # Valid division
    assert guarded_div(10.0, 2.0) == 5.0
    # Negative zero and signs
    assert guarded_div(-10.0, 2.0) == -5.0
    # Non-finite inputs
    assert guarded_div(np.nan, 2.0, default=-1.0) == -1.0
    assert guarded_div(10.0, np.nan, default=-1.0) == -1.0
    assert guarded_div(np.inf, 2.0, default=-1.0) == -1.0
    assert guarded_div(10.0, np.inf, default=-1.0) == -1.0


def test_guarded_div_array() -> None:
    num = np.array([10.0, 20.0, 30.0, np.nan, 50.0])
    den = np.array([2.0, 0.0, 1e-14, 5.0, np.inf])
    res = guarded_div_array(num, den, default=-999.0, epsilon=1e-12)

    assert res[0] == 10.0 / 2.0
    assert res[1] == -999.0
    assert res[2] == -999.0
    assert res[3] == -999.0
    assert res[4] == -999.0


def test_kelly_criterion_bounds() -> None:
    # 60% winrate, 2.0 win/loss ratio -> f* = (0.6*2 - 0.4)/2 = 0.8/2 = 0.40
    f = kelly_criterion(win_rate=0.6, win_loss_ratio=2.0, max_kelly_fraction=0.5)
    assert abs(f - 0.40) < 1e-6

    # Exceeding safety cap gets clamped
    f_clamped = kelly_criterion(win_rate=0.9, win_loss_ratio=3.0, max_kelly_fraction=0.25)
    assert f_clamped == 0.25

    # Negative expectancy produces 0.0 fraction
    f_loss = kelly_criterion(win_rate=0.3, win_loss_ratio=1.0)
    assert f_loss == 0.0

    # Boundary and invalid conditions
    assert kelly_criterion(0.0, 2.0) == 0.0
    assert kelly_criterion(1.5, 2.0) == 0.0
    assert kelly_criterion(0.5, -1.0) == 0.0
    assert kelly_criterion(0.5, 2.0, max_kelly_fraction=-0.1) == 0.0


def test_monotonic_stoploss_ratchet_long() -> None:
    cur_sl = 100.0
    delta = 5.0

    # Price increases: new candidate is 110 - 5 = 105 > 100 -> tightened to 105
    new_sl = monotonic_stoploss_ratchet(
        cur_sl, new_price=110.0, trailing_delta=delta, is_long=True
    )
    assert new_sl == 105.0

    # Price drops: candidate is 98 - 5 = 93 < 105 -> remains 105! (Never loosens)
    tightened_sl = monotonic_stoploss_ratchet(
        new_sl, new_price=98.0, trailing_delta=delta, is_long=True
    )
    assert tightened_sl == 105.0


def test_monotonic_stoploss_ratchet_short() -> None:
    cur_sl = 100.0
    delta = 5.0

    # Price drops for short: candidate is 90 + 5 = 95 < 100 -> tightened down to 95
    new_sl = monotonic_stoploss_ratchet(
        cur_sl, new_price=90.0, trailing_delta=delta, is_long=False
    )
    assert new_sl == 95.0

    # Price rises for short: candidate is 102 + 5 = 107 > 95 -> remains 95! (Never loosens)
    tightened_sl = monotonic_stoploss_ratchet(
        new_sl, new_price=102.0, trailing_delta=delta, is_long=False
    )
    assert tightened_sl == 95.0


def test_monotonic_stoploss_ratchet_invalid_inputs() -> None:
    with pytest.raises(InvariantViolationError):
        monotonic_stoploss_ratchet(100.0, 110.0, trailing_delta=0.0)

    with pytest.raises(InvariantViolationError):
        monotonic_stoploss_ratchet(100.0, np.nan, trailing_delta=5.0)


def test_leverage_constraint() -> None:
    # Valid range
    assert leverage_constraint(2.5, max_allowed=3.0, min_allowed=1.0) == 2.5
    # Above max
    assert leverage_constraint(10.0, max_allowed=3.0, min_allowed=1.0) == 3.0
    # Below min
    assert leverage_constraint(0.5, max_allowed=3.0, min_allowed=1.0) == 1.0

    with pytest.raises(InvariantViolationError, match="cannot exceed"):
        leverage_constraint(2.0, max_allowed=1.0, min_allowed=3.0)

    with pytest.raises(InvariantViolationError, match="min_allowed must be > 0"):
        leverage_constraint(2.0, max_allowed=3.0, min_allowed=-1.0)

    with pytest.raises(InvariantViolationError, match="positive"):
        leverage_constraint(-1.0)

    with pytest.raises(InvariantViolationError, match="positive"):
        leverage_constraint(np.nan)


def test_drawdown_bound() -> None:
    equity = np.array([100.0, 110.0, 120.0, 96.0, 105.0, 130.0])
    # Peak is 120.0, trough is 96.0 -> DD = (120 - 96)/120 = 24/120 = 0.20 (20%)
    max_dd, is_safe = drawdown_bound(equity, max_drawdown_limit=0.25)
    assert abs(max_dd - 0.20) < 1e-6
    assert is_safe is True

    # Limit of 15% should be flagged as unsafe
    _, is_safe_strict = drawdown_bound(equity, max_drawdown_limit=0.15)
    assert is_safe_strict is False

    # Empty curve
    dd_empty, safe_empty = drawdown_bound([])
    assert dd_empty == 0.0 and safe_empty is True
