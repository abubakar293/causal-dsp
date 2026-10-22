"""
causal-dsp: Formally Verified Prefix-Causal Signal Processing & Mathematical Risk Invariants.

Copyright (c) 2026 Md Abubakar Siddik
Licensed under the MIT License.
"""

from causal_dsp.causality import (
    CausalityViolationError,
    assert_prefix_causal,
    is_prefix_causal,
    perturbation_shock_test,
    validate_zero_future_leakage,
)
from causal_dsp.filters import (
    CausalKalman1D,
    CausalKalman2D,
    causal_fractional_diff,
    causal_fractional_weights,
    causal_savitzky_golay,
)
from causal_dsp.invariants import (
    InvariantViolationError,
    drawdown_bound,
    guarded_div,
    guarded_div_array,
    kelly_criterion,
    leverage_constraint,
    monotonic_stoploss_ratchet,
)
from causal_dsp.wavelets import (
    CausalHaarBuffer,
    MODWTEnergy,
    causal_daubechies_d4,
    causal_haar_modwt,
    causal_haar_synth,
    haar_modwt_energy,
)

__version__ = "0.1.0"
__author__ = "Md Abubakar Siddik"
__license__ = "MIT"

__all__ = [
    "CausalHaarBuffer",
    "CausalKalman1D",
    "CausalKalman2D",
    "CausalityViolationError",
    "InvariantViolationError",
    "MODWTEnergy",
    "__author__",
    "__license__",
    "__version__",
    "assert_prefix_causal",
    "causal_daubechies_d4",
    "causal_fractional_diff",
    "causal_fractional_weights",
    "causal_haar_modwt",
    "causal_haar_synth",
    "causal_savitzky_golay",
    "drawdown_bound",
    "guarded_div",
    "guarded_div_array",
    "haar_modwt_energy",
    "is_prefix_causal",
    "kelly_criterion",
    "leverage_constraint",
    "monotonic_stoploss_ratchet",
    "perturbation_shock_test",
    "validate_zero_future_leakage",
]
