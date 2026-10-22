"""
Tests for causality verification protocols, perturbation shock tests, and zero future leakage.
"""

import numpy as np
import pytest
from scipy.signal import savgol_filter

from causal_dsp.causality import (
    CausalityViolationError,
    assert_prefix_causal,
    is_prefix_causal,
    perturbation_shock_test,
    validate_zero_future_leakage,
)
from causal_dsp.filters import causal_fractional_diff, causal_savitzky_golay
from causal_dsp.wavelets import causal_daubechies_d4, causal_haar_modwt


def test_identity_is_causal() -> None:
    rng = np.random.default_rng(42)
    x = rng.standard_normal(100)
    assert is_prefix_causal(lambda s: s, x)
    assert perturbation_shock_test(lambda s: s, x, shock_time=50)


def test_centered_moving_average_fails_causality() -> None:
    """Centered moving average queries x[t+1] and MUST fail causality tests."""

    def centered_ma(s: np.ndarray) -> np.ndarray:
        out = s.copy()
        for t in range(1, len(s) - 1):
            out[t] = (s[t - 1] + s[t] + s[t + 1]) / 3.0
        return np.asarray(out, dtype=np.float64)

    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)

    assert not is_prefix_causal(centered_ma, x, shock_time=25)
    with pytest.raises(CausalityViolationError) as exc_info:
        perturbation_shock_test(centered_ma, x, shock_time=25, shock_magnitude=100.0)

    # Violation must be detected at t=25 (because t=25 queried x[26] which was shocked!)
    assert exc_info.value.violation_time == 25
    assert exc_info.value.delta is not None and exc_info.value.delta > 1.0


def test_standard_scipy_savgol_fails_causality() -> None:
    """Scipy's standard Savitzky-Golay filter uses symmetric windows and leaks future data."""
    rng = np.random.default_rng(123)
    x = rng.standard_normal(100)

    def standard_savgol(s: np.ndarray) -> np.ndarray:
        return np.asarray(savgol_filter(s, window_length=15, polyorder=2), dtype=np.float64)

    assert not is_prefix_causal(standard_savgol, x, shock_time=50)
    with pytest.raises(CausalityViolationError):
        perturbation_shock_test(standard_savgol, x, shock_time=50)


def test_causal_savitzky_golay_passes_causality() -> None:
    rng = np.random.default_rng(42)
    x = rng.standard_normal(120)

    def filter_fn(s: np.ndarray) -> np.ndarray:
        return causal_savitzky_golay(s, window_length=15, polyorder=2)

    assert perturbation_shock_test(filter_fn, x, shock_time=60)
    assert validate_zero_future_leakage(filter_fn, x, num_checks=5)
    # Test small num_checks branch (sampling via rng)
    assert validate_zero_future_leakage(filter_fn, x, num_checks=2, min_history=10, seed=123)
    # Test large num_checks branch (num_checks >= span)
    assert validate_zero_future_leakage(
        lambda s: s, np.arange(15, dtype=np.float64), num_checks=20, min_history=10
    )
    assert_prefix_causal(filter_fn, x)


def test_causal_haar_modwt_passes_causality() -> None:
    rng = np.random.default_rng(77)
    x = rng.standard_normal(100)

    def approx_filter(s: np.ndarray) -> np.ndarray:
        _, approx = causal_haar_modwt(s, levels=3)
        return approx

    def detail_filter(s: np.ndarray) -> np.ndarray:
        details, _ = causal_haar_modwt(s, levels=3)
        return details[0]

    assert validate_zero_future_leakage(approx_filter, x)
    assert validate_zero_future_leakage(detail_filter, x)


def test_causal_daubechies_d4_passes_causality() -> None:
    rng = np.random.default_rng(99)
    x = rng.standard_normal(80)

    def d4_filter(s: np.ndarray) -> np.ndarray:
        approx, _ = causal_daubechies_d4(s)
        return approx

    assert validate_zero_future_leakage(d4_filter, x)


def test_causal_fractional_diff_passes_causality() -> None:
    rng = np.random.default_rng(101)
    x = 100.0 + np.cumsum(rng.standard_normal(90))

    def frac_diff(s: np.ndarray) -> np.ndarray:
        return causal_fractional_diff(s, d=0.45)

    assert validate_zero_future_leakage(frac_diff, x)


def test_invalid_arguments_in_causality_test() -> None:
    with pytest.raises(ValueError, match="1D"):
        perturbation_shock_test(lambda s: s, np.ones((10, 2)))

    with pytest.raises(ValueError, match="length must be >= 3"):
        perturbation_shock_test(lambda s: s, [1.0, 2.0])

    with pytest.raises(ValueError, match="shock_time"):
        perturbation_shock_test(lambda s: s, [1.0, 2.0, 3.0], shock_time=10)

    with pytest.raises(ValueError, match="shock_time"):
        perturbation_shock_test(lambda s: s, [1.0, 2.0, 3.0], shock_time=-1)

    with pytest.raises(ValueError, match="does not match input shape"):
        perturbation_shock_test(lambda s: s[:-1], [1.0, 2.0, 3.0, 4.0])

    with pytest.raises(ValueError, match="Unknown shock_type"):
        perturbation_shock_test(lambda s: s, [1.0, 2.0, 3.0, 4.0], shock_type="invalid_type")

    with pytest.raises(ValueError, match="too short for min_history"):
        validate_zero_future_leakage(lambda s: s, [1.0, 2.0, 3.0], min_history=5)


def test_assert_prefix_causal_raises_on_violation() -> None:
    def leak_filter(s: np.ndarray) -> np.ndarray:
        out = s.copy()
        out[0] = s[-1]
        return np.asarray(out, dtype=np.float64)

    x = np.arange(25, dtype=np.float64)
    with pytest.raises(CausalityViolationError):
        assert_prefix_causal(leak_filter, x)


def test_audit_rejects_degenerate_parameters() -> None:
    """A zero-check audit must never report success without running a check."""
    x = np.arange(50, dtype=np.float64)
    with pytest.raises(ValueError, match="num_checks must be >= 1"):
        validate_zero_future_leakage(lambda s: s, x, num_checks=0)
    with pytest.raises(ValueError, match="num_checks must be >= 1"):
        validate_zero_future_leakage(lambda s: s, x, num_checks=-2)
    with pytest.raises(ValueError, match="min_history must be >= 0"):
        validate_zero_future_leakage(lambda s: s, x, min_history=-1)
    with pytest.raises(ValueError, match="num_checks must be an integer"):
        validate_zero_future_leakage(lambda s: s, x, num_checks=2.5)
    with pytest.raises(ValueError, match="min_history must be an integer"):
        validate_zero_future_leakage(lambda s: s, x, min_history=2.5)
    # bool is an int subclass and must behave as one, not reach numpy internals
    assert validate_zero_future_leakage(lambda s: s, x, num_checks=True)
