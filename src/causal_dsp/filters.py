"""
Memory-preserving Causal Fractional Differencing (De Prado 2018),
One-sided Causal Savitzky-Golay polynomial filters, and Causal Kalman state estimators.
"""

from collections.abc import Sequence
from math import factorial

import numpy as np
from scipy import linalg


def causal_fractional_weights(
    d: float,
    length: int,
    threshold: float = 1e-4,
) -> np.ndarray:
    """
    Computes binomial memory-preserving fractional differentiation weights (De Prado 2018).

    Weights follow recursive expansion of (1 - B)^d:
        w_0 = 1.0
        w_k = -w_{k-1} * (d - k + 1) / k

    Parameters
    ----------
    d : float
        Fractional differentiation order (typically 0.0 <= d <= 1.0).
    length : int
        Maximum number of past lag weights.
    threshold : float, default=1e-4
        Cutoff threshold for memory truncation |w_k| >= threshold.

    Returns
    -------
    np.ndarray
        1D array of weights [w_0, w_1, ..., w_{K-1}].
    """
    if length <= 0:
        raise ValueError("length must be > 0")

    w = [1.0]
    for k in range(1, length):
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < threshold:
            break
        w.append(w_k)

    return np.array(w, dtype=np.float64)


def causal_fractional_diff(
    x: np.ndarray | Sequence[float],
    d: float = 0.40,
    threshold: float = 1e-4,
) -> np.ndarray:
    """
    Applies strictly causal fractional differentiation to input series x.

    Each point y[t] depends strictly on past and present values x[0 ... t]:
        y[t] = sum_{k=0}^{min(t, K-1)} w_k * x[t - k]

    Preserves memory features while approximating stationarity.

    Parameters
    ----------
    x : np.ndarray or Sequence[float]
        1D time series.
    d : float, default=0.40
        Fractional differentiation order.
    threshold : float, default=1e-4
        Weight significance threshold.

    Returns
    -------
    np.ndarray
        Fractionally differentiated causal series (same length as x).

    Notes
    -----
    ``sum(w)`` is only approximately zero, so a constant input is attenuated
    rather than annihilated. Two limits control the residual:

    * ``threshold`` truncates the weight tail (``|w_k| < threshold``);
    * the available history caps the weight count at ``len(x)``.

    Lowering ``threshold`` reduces the residual only until the history limit
    binds (e.g. ``d=0.4`` on a constant series leaves ~7e-2 at ``threshold=1e-4``
    and ~2e-2 once ``n`` exceeds ~5000 with ``threshold=1e-6``). The residual is
    exactly zero at ``d == 1``, where the weights are ``[1, -1]``.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim != 1:
        raise ValueError("Input signal must be 1D")
    n = len(x_arr)
    if n == 0:
        raise ValueError("Signal cannot be empty")

    weights = causal_fractional_weights(d, length=n, threshold=threshold)
    k_len = len(weights)

    out = np.empty(n, dtype=np.float64)
    for t in range(n):
        usable_k = min(t + 1, k_len)
        # lags: x[t], x[t-1], ..., x[t - usable_k + 1]
        past_slice = x_arr[t : t - usable_k : -1] if usable_k <= t else x_arr[t::-1]
        out[t] = np.dot(weights[:usable_k], past_slice)

    return out


def causal_savitzky_golay(
    x: np.ndarray | Sequence[float],
    window_length: int = 15,
    polyorder: int = 2,
    deriv: int = 0,
) -> np.ndarray:
    """
    One-sided causal Savitzky-Golay filter.

    Standard scipy.signal.savgol_filter evaluates polynomials at the *center*
    of the sliding window, causing (window_length - 1) // 2 bars of future lookahead bias.
    This implementation fits the polynomial and evaluates exclusively at the *right edge*
    (offset 0, the current observation) using only past window samples.

    Parameters
    ----------
    x : np.ndarray or Sequence[float]
        1D time series.
    window_length : int, default=15
        Number of past samples in sliding causal window (W >= polyorder + 1).
    polyorder : int, default=2
        Order of fitting polynomial.
    deriv : int, default=0
        Order of derivative to evaluate (0 for smoothing, 1 for velocity).

    Returns
    -------
    np.ndarray
        Filtered causal time series.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim != 1:
        raise ValueError("Input signal must be 1D")
    n = len(x_arr)
    if n == 0:
        raise ValueError("Signal cannot be empty")
    if window_length < 2:
        raise ValueError("window_length must be >= 2")
    if polyorder < 0:
        raise ValueError("polyorder must be >= 0")
    if polyorder >= window_length:
        raise ValueError("polyorder must be less than window_length")
    if deriv < 0 or deriv > polyorder:
        raise ValueError("deriv must be between 0 and polyorder")

    # Time coordinates relative to current point: [-(W-1), -(W-2), ..., -1, 0]
    W = window_length
    t_coords = np.arange(-(W - 1), 1, dtype=np.float64)

    # Vandermonde matrix V[i, j] = t_i^j
    V = np.vander(t_coords, N=polyorder + 1, increasing=True)

    # Moore-Penrose pseudo-inverse: (V^T V)^(-1) V^T
    pinv_V = linalg.pinv(V)

    # For evaluating derivative at t = 0: factorial(deriv) * pinv_V[deriv, :]
    filter_weights = factorial(deriv) * pinv_V[deriv, :]

    # Causal sliding FIR convolution
    out = np.empty(n, dtype=np.float64)

    # Causal prefix extension: pad left with x[0]
    padded = np.empty(n + W - 1, dtype=np.float64)
    padded[: W - 1] = x_arr[0]
    padded[W - 1 :] = x_arr

    for i in range(n):
        # Window of past W samples ending at current point
        window = padded[i : i + W]
        out[i] = np.dot(filter_weights, window)

    return out


class CausalKalman1D:
    """
    Causal 1-Dimensional Linear Kalman Filter for real-time denoising and velocity estimation.
    State vector: [position, velocity]^T.
    """

    def __init__(
        self,
        dt: float = 1.0,
        process_noise: float = 1e-3,
        measurement_noise: float = 1e-2,
    ) -> None:
        self.dt = dt
        self.F = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        self.H = np.array([[1.0, 0.0]], dtype=np.float64)
        self.Q = np.array(
            [[0.25 * dt**4, 0.5 * dt**3], [0.5 * dt**3, dt**2]], dtype=np.float64
        ) * process_noise
        self.R = np.array([[measurement_noise]], dtype=np.float64)
        self.x = np.zeros((2, 1), dtype=np.float64)
        self.P = np.eye(2, dtype=np.float64) * 1.0
        self.initialized = False

    def reset(self, initial_val: float | None = None) -> None:
        """Reset state estimate and covariance."""
        if initial_val is not None:
            self.x = np.array([[initial_val], [0.0]], dtype=np.float64)
            self.P = np.eye(2, dtype=np.float64) * 1.0
            self.initialized = True
        else:
            self.x = np.zeros((2, 1), dtype=np.float64)
            self.P = np.eye(2, dtype=np.float64) * 1.0
            self.initialized = False

    def update(self, z: float) -> tuple[float, float]:
        """
        Process single observation causally.

        Returns
        -------
        est_pos : float
            Filtered state position.
        est_vel : float
            Filtered state velocity.
        """
        if not self.initialized:
            self.reset(z)
            return float(self.x[0, 0]), float(self.x[1, 0])

        # 1. Predict
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q

        # 2. Update
        y = np.array([[z]], dtype=np.float64) - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        self.x = x_pred + K @ y
        identity_2 = np.eye(2, dtype=np.float64)
        self.P = (identity_2 - K @ self.H) @ P_pred

        return float(self.x[0, 0]), float(self.x[1, 0])

    def filter(self, x: np.ndarray | Sequence[float]) -> np.ndarray:
        """
        Causally filters entire series x.
        """
        x_arr = np.asarray(x, dtype=np.float64)
        out = np.empty_like(x_arr)
        if len(x_arr) == 0:
            return out
        self.reset(x_arr[0])
        for i, val in enumerate(x_arr):
            pos, _ = self.update(val)
            out[i] = pos
        return out


class CausalKalman2D:
    """
    Causal 2-State bivariate Kalman Filter for cointegrated pairs or multi-asset tracking.
    """

    def __init__(
        self,
        process_noise: float = 1e-4,
        measurement_noise: float = 1e-2,
    ) -> None:
        self.x = np.zeros((2, 1), dtype=np.float64)
        self.P = np.eye(2, dtype=np.float64) * 1.0
        self.Q = np.eye(2, dtype=np.float64) * process_noise
        self.R = np.eye(2, dtype=np.float64) * measurement_noise
        self.initialized = False

    def reset(self, initial_vals: tuple[float, float] | None = None) -> None:
        if initial_vals is not None:
            self.x = np.array([[initial_vals[0]], [initial_vals[1]]], dtype=np.float64)
            self.P = np.eye(2, dtype=np.float64) * 1.0
            self.initialized = True
        else:
            self.x = np.zeros((2, 1), dtype=np.float64)
            self.P = np.eye(2, dtype=np.float64) * 1.0
            self.initialized = False

    def update(self, z1: float, z2: float) -> tuple[float, float]:
        """
        Update bivariate state strictly causally.
        """
        z = np.array([[z1], [z2]], dtype=np.float64)
        if not self.initialized:
            self.reset((z1, z2))
            return float(self.x[0, 0]), float(self.x[1, 0])

        # Predict (Random walk state)
        x_pred = self.x
        P_pred = self.P + self.Q

        # Update
        y = z - x_pred
        S = P_pred + self.R
        K = P_pred @ np.linalg.inv(S)

        self.x = x_pred + K @ y
        self.P = (np.eye(2) - K) @ P_pred

        return float(self.x[0, 0]), float(self.x[1, 0])
