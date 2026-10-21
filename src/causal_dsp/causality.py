"""
Causality verification protocols, perturbation shock tests, and zero-lookahead validators.
"""

from typing import Callable, Optional, Sequence, Union
import numpy as np


class CausalityViolationError(AssertionError):
    """Raised when a signal transform demonstrates future lookahead or prefix leakage."""

    def __init__(
        self,
        message: str,
        shock_time: Optional[int] = None,
        violation_time: Optional[int] = None,
        delta: Optional[float] = None,
    ) -> None:
        super().__init__(message)
        self.shock_time = shock_time
        self.violation_time = violation_time
        self.delta = delta


def perturbation_shock_test(
    filter_fn: Callable[[np.ndarray], np.ndarray],
    x: Union[np.ndarray, Sequence[float]],
    shock_time: Optional[int] = None,
    shock_magnitude: float = 1e6,
    atol: float = 1e-12,
    shock_type: str = "step",
) -> bool:
    """
    Verifies that modifying future inputs strictly after `shock_time` causes
    0.0 perturbation to output y[0 ... shock_time].

    Parameters
    ----------
    filter_fn : Callable[[np.ndarray], np.ndarray]
        Signal transformation or filter function mapping 1D ndarray -> 1D ndarray.
    x : np.ndarray or Sequence[float]
        Input temporal signal.
    shock_time : int, optional
        Index after which future perturbation is introduced. Defaults to len(x) // 2.
    shock_magnitude : float, default=1e6
        Magnitude of future shock.
    atol : float, default=1e-12
        Absolute tolerance for equality on historical prefix [0 ... shock_time].
    shock_type : {'step', 'impulse', 'random'}, default='step'
        Type of future shock perturbation.

    Returns
    -------
    bool
        True if filter passes prefix-causality test with 0 violation.

    Raises
    ------
    CausalityViolationError
        If any element y[s] for s <= shock_time changes after future perturbation.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim != 1:
        raise ValueError(f"Input signal must be 1D, got shape {x_arr.shape}")
    n = len(x_arr)
    if n < 3:
        raise ValueError(f"Signal length must be >= 3, got {n}")

    if shock_time is None:
        shock_time = n // 2
    if shock_time < 0 or shock_time >= n - 1:
        raise ValueError(
            f"shock_time must be in range [0, {n - 2}], got {shock_time}"
        )

    # 1. Base response
    y_base = np.asarray(filter_fn(x_arr), dtype=np.float64)
    if y_base.shape != x_arr.shape:
        raise ValueError(
            f"Filter output shape {y_base.shape} does not match input shape {x_arr.shape}"
        )

    # 2. Construct future perturbation
    x_shocked = x_arr.copy()
    if shock_type == "step":
        x_shocked[shock_time + 1 :] += shock_magnitude
    elif shock_type == "impulse":
        x_shocked[shock_time + 1] += shock_magnitude
    elif shock_type == "random":
        rng = np.random.default_rng(42)
        x_shocked[shock_time + 1 :] += rng.standard_normal(n - (shock_time + 1)) * shock_magnitude
    else:
        raise ValueError(f"Unknown shock_type: {shock_type}")

    # 3. Perturbed response
    y_shocked = np.asarray(filter_fn(x_shocked), dtype=np.float64)

    # 4. Check historical prefix [0 ... shock_time]
    diff = np.abs(y_base[: shock_time + 1] - y_shocked[: shock_time + 1])
    max_diff_idx = int(np.argmax(diff))
    max_diff = float(diff[max_diff_idx])

    if max_diff > atol:
        raise CausalityViolationError(
            f"Causality Violation Detected! Shock at t={shock_time} leaked backwards "
            f"into t={max_diff_idx} with delta={max_diff:.6e} (tolerance={atol:.1e}).",
            shock_time=shock_time,
            violation_time=max_diff_idx,
            delta=max_diff,
        )

    return True


def is_prefix_causal(
    filter_fn: Callable[[np.ndarray], np.ndarray],
    x: Union[np.ndarray, Sequence[float]],
    shock_time: Optional[int] = None,
    atol: float = 1e-12,
) -> bool:
    """
    Boolean predicate returning True if filter_fn is strictly prefix-causal on x.
    """
    try:
        return perturbation_shock_test(filter_fn, x, shock_time=shock_time, atol=atol)
    except CausalityViolationError:
        return False


def validate_zero_future_leakage(
    filter_fn: Callable[[np.ndarray], np.ndarray],
    x: Union[np.ndarray, Sequence[float]],
    num_checks: int = 5,
    min_history: int = 10,
    atol: float = 1e-12,
    seed: int = 42,
) -> bool:
    """
    Exhaustively audits a filter across multiple horizon cutoffs and multiple shock types.

    Parameters
    ----------
    filter_fn : Callable[[np.ndarray], np.ndarray]
        Filter function to audit.
    x : np.ndarray or Sequence[float]
        Input series.
    num_checks : int, default=5
        Number of distinct cutoff times to test.
    min_history : int, default=10
        Minimum historical length before introducing future shock.
    atol : float, default=1e-12
        Tolerated historical deviation.
    seed : int, default=42
        RNG seed for reproducible cutoff selection.

    Returns
    -------
    bool
        True if all audit shock horizons pass.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    n = len(x_arr)
    if n <= min_history + 2:
        raise ValueError(f"Signal length {n} is too short for min_history={min_history}")

    rng = np.random.default_rng(seed)
    max_t = n - 2
    candidate_times = np.linspace(min_history, max_t, num=min(num_checks, max_t - min_history + 1), dtype=int)

    for t in candidate_times:
        for shock_type in ("step", "impulse", "random"):
            for mag in (1e2, 1e6, -1e6):
                perturbation_shock_test(
                    filter_fn=filter_fn,
                    x=x_arr,
                    shock_time=int(t),
                    shock_magnitude=mag,
                    atol=atol,
                    shock_type=shock_type,
                )

    return True


def assert_prefix_causal(
    filter_fn: Callable[[np.ndarray], np.ndarray],
    x: Union[np.ndarray, Sequence[float]],
    atol: float = 1e-12,
) -> None:
    """
    Convenience assertion function. Raises CausalityViolationError on failure.
    """
    validate_zero_future_leakage(filter_fn, x, atol=atol)
