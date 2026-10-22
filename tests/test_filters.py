"""
Tests for causal fractional differencing, causal Savitzky-Golay, and Kalman estimators.
"""

import numpy as np
import pytest

from causal_dsp.filters import (
    CausalKalman1D,
    CausalKalman2D,
    causal_fractional_diff,
    causal_fractional_weights,
    causal_savitzky_golay,
)


def test_fractional_weights_properties() -> None:
    # d = 0 gives only [1.0] (identity)
    w_0 = causal_fractional_weights(0.0, length=10)
    assert len(w_0) == 1
    assert w_0[0] == 1.0

    # d = 1 gives [1.0, -1.0] (first difference)
    w_1 = causal_fractional_weights(1.0, length=10)
    assert len(w_1) == 2
    assert np.allclose(w_1, [1.0, -1.0])

    # d = 0.5 weights decay monotonically in absolute value
    w_half = causal_fractional_weights(0.5, length=20)
    assert w_half[0] == 1.0
    assert w_half[1] < 0.0
    abs_weights = np.abs(w_half[1:])
    assert np.all(np.diff(abs_weights) <= 0)


def test_fractional_diff_identity_and_first_diff() -> None:
    x = np.array([10.0, 12.0, 15.0, 14.0, 18.0], dtype=np.float64)

    # d = 0.0 -> identity
    y_0 = causal_fractional_diff(x, d=0.0)
    assert np.allclose(y_0, x)

    # d = 1.0 -> standard first difference with y[0] = x[0]
    y_1 = causal_fractional_diff(x, d=1.0)
    expected_diff = np.array([10.0, 2.0, 3.0, -1.0, 4.0])
    assert np.allclose(y_1, expected_diff)


def test_fractional_diff_stationarity_on_random_walk() -> None:
    rng = np.random.default_rng(42)
    # Non-stationary cumulative random walk with zero starting level
    rw = np.cumsum(rng.standard_normal(500))

    # d = 0.45 should produce stationary fluctuations around zero
    y_frac = causal_fractional_diff(rw, d=0.45)
    assert len(y_frac) == len(rw)
    # Mean of increments should be near 0
    assert abs(np.mean(y_frac[50:])) < 1.0
    # Variance should be finite and bounded
    assert 0.5 < np.std(y_frac[50:]) < 3.0


def test_causal_savitzky_golay_polynomial_reconstruction() -> None:
    """A causal SG filter with polyorder 2 should exactly reconstruct a parabola."""
    t = np.arange(50, dtype=np.float64)
    parabola = 2.0 * t**2 - 3.0 * t + 5.0

    smoothed = causal_savitzky_golay(parabola, window_length=15, polyorder=2, deriv=0)
    # After initial window boundary warm-up, fit must be exact
    assert np.allclose(smoothed[20:], parabola[20:], atol=1e-8)


def test_causal_savitzky_golay_derivative() -> None:
    """Derivative order 1 on a linear ramp must yield constant slope."""
    t = np.arange(40, dtype=np.float64)
    ramp = 3.5 * t + 10.0

    slope = causal_savitzky_golay(ramp, window_length=11, polyorder=1, deriv=1)
    assert np.allclose(slope[15:], 3.5, atol=1e-8)


def test_causal_kalman_1d_tracking() -> None:
    kf = CausalKalman1D(dt=1.0, process_noise=1e-4, measurement_noise=1e-2)
    # Test update from uninitialized state directly
    pos0, vel0 = kf.update(15.0)
    assert pos0 == 15.0
    assert vel0 == 0.0

    # Linear ramp with zero noise
    truth = np.linspace(10.0, 60.0, 50)
    filtered = kf.filter(truth)

    assert filtered.shape == truth.shape
    # Kalman filter converges to true ramp trajectory
    assert np.allclose(filtered[10:], truth[10:], atol=0.5)

    # Empty filter check
    empty_res = kf.filter([])
    assert len(empty_res) == 0

    # Reset with None
    kf.reset(None)
    assert not kf.initialized


def test_causal_kalman_2d_tracking() -> None:
    kf2 = CausalKalman2D(process_noise=1e-4, measurement_noise=1e-2)
    p1, p2 = kf2.update(100.0, 50.0)
    assert p1 == 100.0 and p2 == 50.0

    p1_next, p2_next = kf2.update(101.0, 50.5)
    assert abs(p1_next - 101.0) < 1.0
    assert abs(p2_next - 50.5) < 1.0

    # Reset with None
    kf2.reset(None)
    assert not kf2.initialized


def test_invalid_filter_inputs() -> None:
    with pytest.raises(ValueError, match="length must be > 0"):
        causal_fractional_weights(0.4, length=0)

    with pytest.raises(ValueError, match="1D"):
        causal_fractional_diff(np.ones((4, 4)))

    with pytest.raises(ValueError, match="empty"):
        causal_fractional_diff([])

    with pytest.raises(ValueError, match="window_length must be >= 2"):
        causal_savitzky_golay([1.0, 2.0], window_length=1)

    with pytest.raises(ValueError, match="polyorder must be >= 0"):
        causal_savitzky_golay([1.0, 2.0, 3.0], window_length=3, polyorder=-1)

    with pytest.raises(ValueError, match="polyorder must be less than window_length"):
        causal_savitzky_golay([1.0, 2.0, 3.0], window_length=3, polyorder=3)

    with pytest.raises(ValueError, match="deriv must be between 0 and polyorder"):
        causal_savitzky_golay([1.0, 2.0, 3.0, 4.0], window_length=4, polyorder=1, deriv=2)

    with pytest.raises(ValueError, match="deriv must be between 0 and polyorder"):
        causal_savitzky_golay([1.0, 2.0, 3.0, 4.0], window_length=4, polyorder=1, deriv=-1)

    with pytest.raises(ValueError, match="1D"):
        causal_savitzky_golay(np.ones((4, 4)))

    with pytest.raises(ValueError, match="empty"):
        causal_savitzky_golay([])
