"""
Mathematically verified safety invariants: guarded division, Kelly criterion upper bounds,
monotonic stoploss ratchets, leverage constraints, and drawdown bounds.
"""

from typing import Sequence, Tuple, Union
import numpy as np


class InvariantViolationError(ValueError):
    """Raised when an operational parameter or state violates formal mathematical safety invariants."""
    pass


def guarded_div(
    numerator: float,
    denominator: float,
    default: float = 0.0,
    epsilon: float = 1e-12,
) -> float:
    """
    Numerically safe division preventing ZeroDivisionError and NaN/Inf propagation.

    Parameters
    ----------
    numerator : float
        Dividend.
    denominator : float
        Divisor.
    default : float, default=0.0
        Fallback returned if divisor is within epsilon of zero or non-finite.
    epsilon : float, default=1e-12
        Minimum allowable absolute magnitude for denominator.

    Returns
    -------
    float
        numerator / denominator if |denominator| >= epsilon and finite; else default.
    """
    num = float(numerator)
    den = float(denominator)

    if not np.isfinite(den) or abs(den) < epsilon:
        return float(default)
    if not np.isfinite(num):
        return float(default)

    res = num / den
    return float(res) if np.isfinite(res) else float(default)


def guarded_div_array(
    numerator: Union[np.ndarray, Sequence[float]],
    denominator: Union[np.ndarray, Sequence[float]],
    default: float = 0.0,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """
    Vectorized numerically safe division for NumPy arrays.
    """
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)

    safe_mask = np.isfinite(den) & (np.abs(den) >= epsilon) & np.isfinite(num)
    out = np.full_like(num, default, dtype=np.float64)

    # Avoid numpy divide-by-zero warning by dividing only on safe_mask
    np.divide(num, den, out=out, where=safe_mask)
    # Ensure any remaining non-finite output gets default
    out[~np.isfinite(out)] = default
    return out


def kelly_criterion(
    win_rate: float,
    win_loss_ratio: float,
    max_kelly_fraction: float = 0.5,
) -> float:
    """
    Computes optimal Kelly capital growth fraction subject to formal safety caps.

    Mathematical formula:
        f* = (p * b - (1 - p)) / b = p - (1 - p) / b

    Parameters
    ----------
    win_rate : float
        Empirical win probability p in [0.0, 1.0].
    win_loss_ratio : float
        Payoff ratio b = average_win / average_loss.
    max_kelly_fraction : float, default=0.5
        Hard safety cap (Half-Kelly recommended to protect against estimation noise).

    Returns
    -------
    float
        Safe fraction f in [0.0, max_kelly_fraction].
    """
    p = float(win_rate)
    b = float(win_loss_ratio)
    cap = float(max_kelly_fraction)

    if not (0.0 <= p <= 1.0) or not np.isfinite(p):
        return 0.0
    if b <= 0.0 or not np.isfinite(b):
        return 0.0
    if cap <= 0.0 or not np.isfinite(cap):
        return 0.0

    raw_f = (p * b - (1.0 - p)) / b

    if raw_f <= 0.0 or not np.isfinite(raw_f):
        return 0.0

    return min(raw_f, cap)


def monotonic_stoploss_ratchet(
    current_sl: float,
    new_price: float,
    trailing_delta: float,
    is_long: bool = True,
) -> float:
    """
    Monotonic stop-loss tightening ratchet.

    Guarantees that once risk is reduced, the stoploss can NEVER be loosened:
        - For Long positions: SL_{t} = max(SL_{t-1}, new_price - trailing_delta)
        - For Short positions: SL_{t} = min(SL_{t-1}, new_price + trailing_delta)

    Parameters
    ----------
    current_sl : float
        Current stop-loss price level.
    new_price : float
        Current market price.
    trailing_delta : float
        Trailing price distance (> 0).
    is_long : bool, default=True
        True for long position, False for short.

    Returns
    -------
    float
        Updated tightened stoploss level.

    Raises
    ------
    InvariantViolationError
        If trailing_delta is non-positive or prices are non-finite.
    """
    cur = float(current_sl)
    price = float(new_price)
    delta = float(trailing_delta)

    if not np.isfinite(cur) or not np.isfinite(price) or not np.isfinite(delta):
        raise InvariantViolationError("All prices and delta must be finite real numbers.")
    if delta <= 0.0:
        raise InvariantViolationError(f"trailing_delta must be strictly positive, got {delta}")

    if is_long:
        candidate = price - delta
        new_sl = max(cur, candidate)
        # Invariant check: new_sl >= cur
        if new_sl < cur:
            raise InvariantViolationError(
                f"Long stoploss monotonicity violated: {new_sl} < {cur}"
            )
        return new_sl
    else:
        candidate = price + delta
        new_sl = min(cur, candidate)
        # Invariant check: new_sl <= cur
        if new_sl > cur:
            raise InvariantViolationError(
                f"Short stoploss monotonicity violated: {new_sl} > {cur}"
            )
        return new_sl


def leverage_constraint(
    requested_leverage: float,
    max_allowed: float = 3.0,
    min_allowed: float = 1.0,
) -> float:
    """
    Enforces bounded leverage constraints within [min_allowed, max_allowed].

    Parameters
    ----------
    requested_leverage : float
        Nominal desired leverage.
    max_allowed : float, default=3.0
        Maximum allowed leverage cap.
    min_allowed : float, default=1.0
        Minimum allowable leverage.

    Returns
    -------
    float
        Clamped leverage value.

    Raises
    ------
    InvariantViolationError
        If bounds are invalid or requested leverage is non-positive / non-finite.
    """
    req = float(requested_leverage)
    high = float(max_allowed)
    low = float(min_allowed)

    if low > high:
        raise InvariantViolationError(f"min_allowed ({low}) cannot exceed max_allowed ({high})")
    if low <= 0.0:
        raise InvariantViolationError(f"min_allowed must be > 0, got {low}")
    if not np.isfinite(req) or req <= 0.0:
        raise InvariantViolationError(f"requested_leverage must be finite and positive, got {req}")

    return float(np.clip(req, low, high))


def drawdown_bound(
    equity_curve: Union[np.ndarray, Sequence[float]],
    max_drawdown_limit: float = 0.20,
) -> Tuple[float, bool]:
    """
    Calculates maximum drawdown along an equity series and verifies the safety bound.

    Parameters
    ----------
    equity_curve : np.ndarray or Sequence[float]
        Series of total account portfolio value over time.
    max_drawdown_limit : float, default=0.20
        Maximum allowable relative drawdown (e.g. 0.20 = 20%).

    Returns
    -------
    max_dd : float
        Observed maximum peak-to-trough drawdown in [0.0, 1.0].
    is_safe : bool
        True if observed max_dd <= max_drawdown_limit.
    """
    eq = np.asarray(equity_curve, dtype=np.float64)
    if len(eq) == 0:
        return 0.0, True

    peaks = np.maximum.accumulate(eq)
    drawdowns = guarded_div_array(peaks - eq, peaks, default=0.0)
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    return max_dd, max_dd <= max_drawdown_limit
