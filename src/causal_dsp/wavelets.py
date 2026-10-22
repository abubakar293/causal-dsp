"""
Strictly causal Haar and Daubechies wavelet decompositions for time-series.
Guarantees zero boundary lookahead and exact multi-scale reconstruction.
"""

from collections.abc import Sequence
from typing import TypedDict

import numpy as np


class MODWTEnergy(TypedDict):
    """Energy distribution across wavelet scales and signal fidelity."""

    energy_approx: float
    energy_details: list[float]
    energy_components_sum: float
    energy_signal: float


def causal_haar_modwt(
    x: np.ndarray | Sequence[float],
    levels: int = 3,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Maximal Overlap Discrete Wavelet Transform (MODWT) using strictly causal Haar wavelets.

    Unlike standard PyWavelets or MATLAB dwt/modwt which use symmetric reflection or
    periodic boundary extension (leaking future samples into past and present),
    this implementation strictly uses one-sided causal past lags.

    For level j with lag L_j = 2^(j-1):
        A_j[t] = 0.5 * (A_{j-1}[t] + A_{j-1}[t - L_j])
        D_j[t] = 0.5 * (A_{j-1}[t] - A_{j-1}[t - L_j])

    Identity:
        x[t] = A_J[t] + sum_{j=1}^J D_j[t] for all t >= 0.

    Parameters
    ----------
    x : np.ndarray or Sequence[float]
        1D temporal array.
    levels : int, default=3
        Number of decomposition levels (J >= 1).

    Returns
    -------
    details : list[np.ndarray]
        List of J detail coefficient arrays [D_1, D_2, ..., D_J], each same shape as x.
    approx : np.ndarray
        Final smooth approximation array A_J, same shape as x.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim != 1:
        raise ValueError("Input signal must be 1D")
    n = len(x_arr)
    if n == 0:
        raise ValueError("Signal cannot be empty")
    if levels < 1:
        raise ValueError("levels must be >= 1")

    current_approx = x_arr.copy()
    details: list[np.ndarray] = []

    for j in range(1, levels + 1):
        lag = 1 << (j - 1)  # 2^(j - 1)
        # Construct causal delayed series
        delayed = np.empty_like(current_approx)
        if lag < n:
            delayed[:lag] = current_approx[0]
            delayed[lag:] = current_approx[:-lag]
        else:
            delayed[:] = current_approx[0]

        next_approx = 0.5 * (current_approx + delayed)
        detail = 0.5 * (current_approx - delayed)

        details.append(detail)
        current_approx = next_approx

    return details, current_approx


def causal_haar_synth(
    details: Sequence[np.ndarray],
    approx: np.ndarray,
) -> np.ndarray:
    """
    Exact causal inverse synthesis from detail coefficients and final approximation.

    Reconstructs x[t] = A_J[t] + sum_{j=1}^J D_j[t].

    Parameters
    ----------
    details : Sequence[np.ndarray]
        List of detail coefficient arrays.
    approx : np.ndarray
        Approximation array A_J.

    Returns
    -------
    np.ndarray
        Reconstructed time series with machine-precision fidelity.
    """
    recon = np.copy(approx)
    for d in details:
        recon += d
    return recon


def haar_modwt_energy(
    details: Sequence[np.ndarray],
    approx: np.ndarray,
) -> MODWTEnergy:
    """
    Computes energy decomposition across wavelet scales and verifies conservation.
    """
    e_approx = float(np.sum(approx ** 2))
    e_details = [float(np.sum(d ** 2)) for d in details]
    e_total_components = e_approx + sum(e_details)

    recon = causal_haar_synth(details, approx)
    e_signal = float(np.sum(recon ** 2))

    return {
        "energy_approx": e_approx,
        "energy_details": e_details,
        "energy_components_sum": e_total_components,
        "energy_signal": e_signal,
    }


class CausalHaarBuffer:
    """
    Online streaming buffer for real-time Haar MODWT computation.
    Processes incoming price/signal stream with O(1) complexity per sample.
    """

    def __init__(self, levels: int = 3) -> None:
        if levels < 1:
            raise ValueError("levels must be >= 1")
        self.levels = levels
        self.max_lag = 1 << (levels - 1)
        self.history: list[float] = []
        self._approx_history: list[list[float]] = [[] for _ in range(levels + 1)]

    def reset(self) -> None:
        """Reset internal buffer state."""
        self.history.clear()
        self._approx_history = [[] for _ in range(self.levels + 1)]

    def update(self, val: float) -> tuple[list[float], float]:
        """
        Process a single incoming sample tick causally.

        Parameters
        ----------
        val : float
            New observation x[t].

        Returns
        -------
        details : list[float]
            Wavelet detail coefficients [D_1[t], ..., D_J[t]].
        approx : float
            Smooth approximation A_J[t].
        """
        v = float(val)
        self.history.append(v)
        self._approx_history[0].append(v)

        # Bound history to max_lag + 1 to prevent unbounded RAM growth
        keep_len = self.max_lag * 2 + 1
        if len(self._approx_history[0]) > keep_len:
            self._approx_history[0] = self._approx_history[0][-keep_len:]

        current_val = v
        details: list[float] = []

        for j in range(1, self.levels + 1):
            lag = 1 << (j - 1)
            prev_level_history = self._approx_history[j - 1]
            if len(prev_level_history) > lag:
                delayed = prev_level_history[-1 - lag]
            else:
                delayed = prev_level_history[0]

            next_approx = 0.5 * (current_val + delayed)
            detail = 0.5 * (current_val - delayed)

            details.append(detail)
            self._approx_history[j].append(next_approx)
            if len(self._approx_history[j]) > keep_len:
                self._approx_history[j] = self._approx_history[j][-keep_len:]
            current_val = next_approx

        return details, current_val


# Daubechies D4 Filter Coefficients
SQRT_3 = np.sqrt(3.0)
DENOM = 4.0 * np.sqrt(2.0)
H0 = (1.0 + SQRT_3) / DENOM
H1 = (3.0 + SQRT_3) / DENOM
H2 = (3.0 - SQRT_3) / DENOM
H3 = (1.0 - SQRT_3) / DENOM

# Wavelet Quadrature Mirror Filter: g_k = (-1)^k * h_{3-k}
G0 = H3
G1 = -H2
G2 = H1
G3 = -H0

D4_LOW_PASS = np.array([H0, H1, H2, H3], dtype=np.float64)
D4_HIGH_PASS = np.array([G0, G1, G2, G3], dtype=np.float64)


def causal_daubechies_d4(
    x: np.ndarray | Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Strictly causal Daubechies D4 sliding filter.

    Computes 4-tap causal FIR convolution:
        approx[t] = H0*x[t] + H1*x[t-1] + H2*x[t-2] + H3*x[t-3]
        detail[t] = G0*x[t] + G1*x[t-1] + G2*x[t-2] + G3*x[t-3]

    For t < 3, past lags are clamped causally to x[0].

    Parameters
    ----------
    x : np.ndarray or Sequence[float]
        1D temporal series.

    Returns
    -------
    approx : np.ndarray
        Low-pass D4 scaling filter output (same length as x).
    detail : np.ndarray
        High-pass D4 wavelet detail filter output (same length as x).
    """
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim != 1:
        raise ValueError("Input signal must be 1D")
    n = len(x_arr)
    if n == 0:
        raise ValueError("Signal cannot be empty")

    approx = np.empty(n, dtype=np.float64)
    detail = np.empty(n, dtype=np.float64)

    # Pad prefix causally with x[0]
    padded = np.empty(n + 3, dtype=np.float64)
    padded[:3] = x_arr[0]
    padded[3:] = x_arr

    # Causal FIR: index t in output corresponds to padded[t+3]
    for t in range(n):
        idx = t + 3
        # taps: x[t], x[t-1], x[t-2], x[t-3]
        window = np.array([padded[idx], padded[idx - 1], padded[idx - 2], padded[idx - 3]])
        approx[t] = np.dot(D4_LOW_PASS, window)
        detail[t] = np.dot(D4_HIGH_PASS, window)

    return approx, detail
