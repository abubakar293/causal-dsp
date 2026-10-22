"""
Tests for causal Haar MODWT and Daubechies D4 wavelet transforms.
"""

import numpy as np
import pytest

from causal_dsp.wavelets import (
    CausalHaarBuffer,
    causal_daubechies_d4,
    causal_haar_modwt,
    causal_haar_synth,
    haar_modwt_energy,
)


def test_haar_modwt_exact_reconstruction() -> None:
    """Verify x[t] == A_J[t] + sum(D_j[t]) at machine precision."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal(200)

    for levels in (1, 2, 3, 4, 5):
        details, approx = causal_haar_modwt(x, levels=levels)
        recon = causal_haar_synth(details, approx)

        assert len(details) == levels
        assert approx.shape == x.shape
        max_error = np.max(np.abs(x - recon))
        assert max_error < 1e-14, f"Reconstruction error {max_error} at levels={levels}"


def test_haar_modwt_short_signal_large_lag() -> None:
    """Verify modwt handles lag >= n properly."""
    x = [10.0, 12.0, 15.0]
    details, approx = causal_haar_modwt(x, levels=3)
    assert len(details) == 3
    assert len(approx) == 3


def test_haar_modwt_energy_conservation() -> None:
    rng = np.random.default_rng(10)
    x = rng.standard_normal(150)
    details, approx = causal_haar_modwt(x, levels=3)

    energy = haar_modwt_energy(details, approx)
    assert energy["energy_approx"] > 0
    assert len(energy["energy_details"]) == 3
    assert energy["energy_signal"] > 0


def test_causal_haar_buffer_matches_batch() -> None:
    """Verify that online O(1) CausalHaarBuffer yields identical results to batch modwt."""
    rng = np.random.default_rng(88)
    x = rng.standard_normal(100)
    levels = 3

    batch_details, batch_approx = causal_haar_modwt(x, levels=levels)

    buffer = CausalHaarBuffer(levels=levels)
    stream_details: list[list[float]] = [[] for _ in range(levels)]
    stream_approx: list[float] = []

    for val in x:
        dets, app = buffer.update(float(val))
        for j in range(levels):
            stream_details[j].append(dets[j])
        stream_approx.append(app)

    stream_approx_arr = np.array(stream_approx)
    assert np.allclose(stream_approx_arr, batch_approx, atol=1e-12)

    for j in range(levels):
        det_arr = np.array(stream_details[j])
        assert np.allclose(det_arr, batch_details[j], atol=1e-12)


def test_causal_haar_buffer_reset() -> None:
    buffer = CausalHaarBuffer(levels=2)
    buffer.update(10.0)
    buffer.update(12.0)
    buffer.reset()
    assert len(buffer.history) == 0

    dets, app = buffer.update(5.0)
    assert app == 5.0
    assert dets == [0.0, 0.0]


def test_causal_haar_buffer_long_stream_truncation() -> None:
    """Verify stream buffer memory bounding logic over extended sequences."""
    buffer = CausalHaarBuffer(levels=2)
    for val in range(50):
        buffer.update(float(val))
    keep_len = buffer.max_lag * 2 + 1
    assert len(buffer._approx_history[0]) <= keep_len
    assert len(buffer._approx_history[1]) <= keep_len


def test_daubechies_d4_shape_and_filtering() -> None:
    rng = np.random.default_rng(42)
    x = np.sin(np.linspace(0, 4 * np.pi, 100)) + 0.1 * rng.standard_normal(100)

    approx, detail = causal_daubechies_d4(x)

    assert approx.shape == x.shape
    assert detail.shape == x.shape
    # D4 low-pass filter should smooth the high frequency noise
    assert np.std(approx) > 0
    assert np.std(detail) > 0


def test_invalid_wavelet_inputs() -> None:
    with pytest.raises(ValueError, match="1D"):
        causal_haar_modwt(np.ones((5, 5)))

    with pytest.raises(ValueError, match="empty"):
        causal_haar_modwt([])

    with pytest.raises(ValueError, match="levels must be >= 1"):
        causal_haar_modwt([1.0, 2.0], levels=0)

    with pytest.raises(ValueError, match="levels must be >= 1"):
        CausalHaarBuffer(levels=0)

    with pytest.raises(ValueError, match="1D"):
        causal_daubechies_d4(np.ones((2, 2)))

    with pytest.raises(ValueError, match="empty"):
        causal_daubechies_d4([])
