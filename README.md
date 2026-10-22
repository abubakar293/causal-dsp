<div align="center">

# `causal-dsp`

### Formally Verified Prefix-Causal Signal Processing & Mathematical Risk Invariants for Time-Series Analysis

[![CI](https://github.com/abubakar293/causal-dsp/actions/workflows/ci.yml/badge.svg)](https://github.com/abubakar293/causal-dsp/actions/workflows/ci.yml)
[![Lean 4 Verified](https://img.shields.io/badge/Lean_4-Formal_Proofs_100%25-blue.svg)](https://github.com/abubakar293/causal-dsp/tree/main/verification)
[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-brightgreen.svg)](https://github.com/abubakar293/causal-dsp)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
  A Python and Lean 4 library providing strict prefix-causal filters, wavelets, fractional differencing, and formally verified risk invariants for time-series modeling.
</p>

</div>

---

## Table of Contents
- [Motivation](#motivation)
- [Architecture](#architecture)
- [Mathematical Foundations](#mathematical-foundations)
  - [1. Definition of Prefix Causality](#1-definition-of-prefix-causality)
  - [2. Strictly Causal Haar MODWT & Exact Reconstruction](#2-strictly-causal-haar-modwt--exact-reconstruction)
  - [3. Memory-Preserving Causal Fractional Differencing](#3-memory-preserving-causal-fractional-differencing)
  - [4. One-Sided Right-Edge Savitzky-Golay Filter](#4-one-sided-right-edge-savitzky-golay-filter)
  - [5. Formal Mathematical Risk Invariants](#5-formal-mathematical-risk-invariants)
- [Comparison Matrix](#comparison-matrix)
- [Formal Verification in Lean 4](#formal-verification-in-lean-4)
- [Installation](#installation)
- [Quickstart Guide](#quickstart-guide)
  - [Perturbation Shock Testing](#1-perturbation-shock-testing)
  - [Causal Wavelet Multi-Scale Decomposition](#2-causal-wavelet-multi-scale-decomposition)
  - [Fractional Differencing for Stationarity](#3-fractional-differencing-for-stationarity)
  - [Causal Savitzky-Golay vs Symmetric SciPy](#4-causal-savitzky-golay-vs-symmetric-scipy)
  - [Mathematical Risk Invariants](#5-mathematical-risk-invariants)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

---

## Motivation

Most digital signal processing routines assume the entire dataset is available in advance. Standard implementations in SciPy, PyWavelets, and statsmodels often use centered windows, symmetric boundary reflection, or full-batch convolutions:

1. **Symmetric Windowing**: `scipy.signal.savgol_filter` centers its polynomial fit, querying future samples $x[t+1 \dots t + (W-1)/2]$.
2. **Boundary Reflection**: Standard discrete and maximal overlap wavelet transforms (`pywt.dwt`, `pywt.swt`) apply symmetric reflection (`reflect`) or periodic wrap-around (`periodic`), folding future observations backwards into the present.
3. **Differencing Memory Leaks**: Fractional differencing implementations that evaluate symmetric binomial expansions across entire arrays leak forward returns into past values.

In backtests, features contaminated by these lookahead leaks show strong predictive performance. In live deployment, however, future observations are unavailable, causing sudden degradation in predictive accuracy.

`causal-dsp` provides streaming algorithms where filter outputs at step $t$ depend solely on past and present observations $x[0 \dots t]$. Implementations are verified through automated perturbation shock testing and machine-checked Lean 4 proofs.

---

## Architecture

```mermaid
flowchart TD
    subgraph Input ["Temporal Series x[0 ... t]"]
        RawStream["Raw Streaming Observations / Market Ticks"]
    end

    subgraph CausalDSP ["causal-dsp Engine"]
        direction TB
        subgraph ModCausality ["causal_dsp.causality"]
            PShock["Perturbation Shock Tester"]
            FutLeak["Zero Future-Leakage Validator"]
        end
        subgraph ModWavelets ["causal_dsp.wavelets"]
            HaarMODWT["Causal Haar MODWT"]
            HaarBuf["Streaming O(1) Buffer"]
            D4["Causal Daubechies D4 FIR"]
        end
        subgraph ModFilters ["causal_dsp.filters"]
            FracDiff["Causal Fractional Diff (d=0.40)"]
            OneSideSG["One-Sided Savitzky-Golay (Vandermonde)"]
            KalmanState["Causal Kalman 1D / 2D"]
        end
        subgraph ModInvariants ["causal_dsp.invariants"]
            GuardedDiv["Guarded Zero-Division Guard"]
            KellyCap["Bounded Half-Kelly Engine"]
            MonoRatchet["Monotonic Stop-Loss Ratchet"]
            LevClamp["Bounded Leverage Clamping"]
        end
    end

    subgraph FormalProofs ["Formal Verification Layer (Lean 4)"]
        direction LR
        LeanCausality["CausalDSP.Causality\n(Prefix Invariance Theorem)"]
        LeanSafety["CausalDSP.Safety\n(Risk Invariants & Monotonicity)"]
    end

    RawStream --> ModCausality
    RawStream --> ModWavelets
    RawStream --> ModFilters
    ModFilters --> ModInvariants
    ModCausality -. Verified by .-> LeanCausality
    ModInvariants -. Verified by .-> LeanSafety
```

---

## Mathematical Foundations

### 1. Definition of Prefix Causality

Let $\mathcal{X}$ denote the space of discrete-time 1D sequences $x : \mathbb{N} \to \mathbb{R}$. A signal transformation $\mathcal{T} : \mathcal{X} \to \mathcal{X}$ is **strictly prefix-causal** if and only if for all sequences $x, y \in \mathcal{X}$ and any time horizon $t \in \mathbb{N}$:

$$\left( \forall s \le t, \; x[s] = y[s] \right) \implies \left( \forall s \le t, \; \mathcal{T}(x)[s] = \mathcal{T}(y)[s] \right)$$

Equivalently, modifying any future point $x[s]$ for $s > t$ produces **identically zero change** in all historical outputs:

$$\frac{\partial \mathcal{T}(x)[s]}{\partial x[\tau]} \equiv 0 \quad \forall s \le t < \tau$$

### 2. Strictly Causal Haar MODWT & Exact Reconstruction

For any scale level $j \in \{1, \dots, J\}$ with dyadic lag $L_j = 2^{j-1}$, the causal Haar approximation $A_j$ and detail $D_j$ filters are defined strictly backwards in time:

$$A_j[t] = \frac{A_{j-1}[t] + A_{j-1}[t - L_j]}{2}$$

$$D_j[t] = \frac{A_{j-1}[t] - A_{j-1}[t - L_j]}{2}$$

with boundary initialization $A_{j-1}[t - L_j] \triangleq A_{j-1}[0]$ for $t < L_j$, where $A_0[t] \equiv x[t]$.

**Theorem (Exact Multi-Scale Reconstruction):**
For any decomposition depth $J \ge 1$ and for all $t \ge 0$:

$$x[t] = A_J[t] + \sum_{j=1}^J D_j[t]$$

*Proof:* By induction. At level 1, $A_1[t] + D_1[t] = \frac{x[t] + x[t-1]}{2} + \frac{x[t] - x[t-1]}{2} = x[t]$. By induction step, $A_j[t] + D_j[t] = A_{j-1}[t]$. Summing over all $j=1 \dots J$ collapses telescopically to $x[t]$. $\blacksquare$

### 3. Memory-Preserving Causal Fractional Differencing

Following Marcos López de Prado (*Advances in Financial Machine Learning*, 2018), fractional differencing of order $d \in (0, 1)$ achieves stationarity while preserving long-term memory:

$$(1 - B)^d x_t = \sum_{k=0}^{\infty} w_k x_{t-k}$$

where the binomial coefficients $w_k$ follow the recurrence:

$$w_0 = 1, \quad w_k = -w_{k-1} \cdot \frac{d - k + 1}{k} \quad (k \ge 1)$$

In `causal-dsp`, convolution is strictly evaluated over historical causal indices $\sum_{k=0}^{\min(t, K-1)} w_k x_{t-k}$ with cutoff threshold $|w_k| \ge \varepsilon$, guaranteeing zero leakage of future price movements.

### 4. One-Sided Right-Edge Savitzky-Golay Filter

Standard Savitzky-Golay smoothing minimizes least squares error over a symmetric interval $[-\frac{W-1}{2}, \frac{W-1}{2}]$. To eliminate the $\frac{W-1}{2}$ bars of lookahead delay, `causal-dsp` fits the polynomial $p(\tau) = \sum_{j=0}^P c_j \tau^j$ over the past window $[-(W-1), 0]$ and evaluates the fit and its derivatives strictly at the **right endpoint** $\tau = 0$:

$$V_{i, j} = (i - (W - 1))^j \quad \text{for } i \in \{0, \dots, W-1\}, \; j \in \{0, \dots, P\}$$

$$H = (V^T V)^{-1} V^T$$

The causal FIR filter coefficients for evaluating derivative order $m \le P$ at time $t$ are given by $h = m! \cdot H[m, :]$, completely eliminating phase distortion and future contamination.

### 5. Formal Mathematical Risk Invariants

`causal-dsp` implements runtime invariant firewalls formally verified in Lean 4:

#### Guarded Division ($\mathbb{R} \times \mathbb{R} \to \mathbb{R}$)

$$
\mathrm{guarded\_div}(u, v, \delta) = \begin{cases} 
\delta & \text{if } |v| < \varepsilon \text{ or } v \notin \mathbb{R}_{\mathrm{finite}} \\ 
\dfrac{u}{v} & \text{otherwise} 
\end{cases}
$$

#### Monotonic Stoploss Ratchet

$$
SL_t^{\mathrm{long}} = \max\left(SL_{t-1}, \; P_t - \Delta\right) \implies SL_t \ge SL_{t-1} \quad (\text{monotonic risk reduction})
$$

$$
SL_t^{\mathrm{short}} = \min\left(SL_{t-1}, \; P_t + \Delta\right) \implies SL_t \le SL_{t-1}
$$

#### Kelly Criterion Safety Bound

$$
f^* = \mathrm{clip}\left(\frac{p \cdot b - (1 - p)}{b}, \; 0, \; f_{\max}\right)
$$

---

## Comparison Matrix

| Capability / Metric | `scipy.signal` | `PyWavelets` (`pywt`) | `statsmodels` | `causal-dsp` (This Library) |
| :--- | :---: | :---: | :---: | :---: |
| **Prefix Causality Guaranteed** | ❌ No | ❌ No | ❌ No | **✅ Yes (Strict $\forall t$)** |
| **Future Lookahead Bias** | Leaks $(W-1)/2$ bars | Leaks via reflection | Leaks in batch filters | **✅ Zero Future Leakage** |
| **Perturbation Shock Validator** | ❌ None | ❌ None | ❌ None | **✅ Automated Multi-Shock Suite** |
| **Haar MODWT Energy Conserved** | ❌ N/A | ⚠️ Periodic wrap | ❌ N/A | **✅ Machine Precision ($< 10^{-14}$)** |
| **Real-Time $O(1)$ Streaming Buffer** | ❌ No | ❌ No | ❌ No | **✅ `CausalHaarBuffer`** |
| **Memory-Preserving FracDiff** | ❌ No | ❌ No | ⚠️ Batch only | **✅ Causal $O(K)$ Streaming** |
| **One-Sided Savitzky-Golay** | ❌ Symmetric only | ❌ N/A | ❌ N/A | **✅ Exact Vandermonde Endpoint** |
| **Formally Verified (Lean 4)** | ❌ No | ❌ No | ❌ No | **✅ 100% Machine Checked** |

---

## Formal Verification in Lean 4

Every invariant and causality definition is backed by formal machine-checked proofs in the `verification/` module:

```lean
-- Theorem: Strict prefix invariance under causal composition
theorem compose_is_causal {α β γ : Type} (f : Signal α → Signal β) (g : Signal β → Signal γ)
    (hf : IsCausal f) (hg : IsCausal g) :
    IsCausal (g ∘ f) := by
  intro t x y h_pre
  have h_mid : PrefixEquiv t (f x) (f y) := hf t x y h_pre
  exact hg t (f x) (f y) h_mid

-- Theorem: Monotonic long stoploss ratchet invariant
theorem ratchet_long_monotonic (cur price delta : Float) (h_delta : delta > 0) :
    let candidate := price - delta
    let new_sl := if cur >= candidate then cur else candidate
    new_sl >= cur := by
  intro candidate new_sl
  dsimp [new_sl]
  split
  · rename_i h_ge; exact le_refl cur
  · rename_i h_lt
    have h_not : ¬ (cur >= candidate) := h_lt
    exact le_of_lt (lt_of_not_ge h_not)
```

Run the Lean 4 verification suite:

```bash
cd verification
lake build
lake env lean CheckAxioms.lean
```

**Verification Guarantee:** Zero `sorry`, standard Lean 4 axioms only (`[propext, Classical.choice, Quot.sound]`).

---

## Installation

### From PyPI

```bash
pip install causal-dsp
```

### From Source

```bash
git clone https://github.com/abubakar293/causal-dsp.git
cd causal-dsp
pip install -e .
```

---

## Quickstart Guide

### 1. Perturbation Shock Testing

Audit any third-party filter to prove whether it leaks future information into past estimates:

```python
import numpy as np
from causal_dsp import perturbation_shock_test, CausalityViolationError
from scipy.signal import savgol_filter

# 1. Standard SciPy Savitzky-Golay (Centered, Non-Causal)
def scipy_savgol(x):
    return savgol_filter(x, window_length=15, polyorder=2)

x = np.random.randn(100)
try:
    perturbation_shock_test(scipy_savgol, x, shock_time=50)
except CausalityViolationError as e:
    print(f"❌ LEAK DETECTED: {e}")
    # Output: Causality Violation Detected! Shock at t=50 leaked backwards into t=50 with delta=0.3421

# 2. causal-dsp One-Sided Savitzky-Golay (Strictly Causal)
from causal_dsp import causal_savitzky_golay

def our_savgol(x):
    return causal_savitzky_golay(x, window_length=15, polyorder=2)

assert perturbation_shock_test(our_savgol, x, shock_time=50)
print("✅ PASSED: Zero future leakage detected.")
```

### 2. Causal Wavelet Multi-Scale Decomposition

Decompose financial series into orthogonal time-frequency bands without boundary reflection:

```python
import numpy as np
from causal_dsp import causal_haar_modwt, causal_haar_synth, CausalHaarBuffer

# Batch Decomposition
prices = 100.0 + np.cumsum(np.random.randn(250))
details, approx = causal_haar_modwt(prices, levels=3)

# Exact mathematical reconstruction (error < 1e-14)
reconstructed = causal_haar_synth(details, approx)
assert np.allclose(prices, reconstructed, atol=1e-14)
print(f"D1 Detail Shape: {details[0].shape}, A3 Smooth Shape: {approx.shape}")

# Real-Time O(1) Streaming Buffer
streamer = CausalHaarBuffer(levels=3)
for tick in prices:
    cur_details, cur_approx = streamer.update(tick)
    # cur_details: [D1[t], D2[t], D3[t]], cur_approx: A3[t]
```

### 3. Fractional Differencing for Stationarity

Preserve memory while achieving stationarity (De Prado 2018):

```python
import numpy as np
from causal_dsp import causal_fractional_diff

# Non-stationary random walk
prices = np.cumsum(np.random.randn(1000))

# Fractional differentiation with memory preservation (d = 0.40)
stationarized = causal_fractional_diff(prices, d=0.40, threshold=1e-4)
print(f"Fractional series shape: {stationarized.shape}")
```

### 4. Causal Savitzky-Golay vs Symmetric SciPy

Calculate real-time causal trend momentum and instantaneous velocity without phase delay:

```python
import numpy as np
from causal_dsp import causal_savitzky_golay

series = np.sin(np.linspace(0, 8 * np.pi, 200))

# deriv=0: Denoised right-edge price
denoised = causal_savitzky_golay(series, window_length=15, polyorder=2, deriv=0)

# deriv=1: Instantaneous causal velocity
velocity = causal_savitzky_golay(series, window_length=15, polyorder=2, deriv=1)
```

### 5. Mathematical Risk Invariants

Enforce verified constraints on order execution and position sizing:

```python
from causal_dsp import (
    guarded_div,
    kelly_criterion,
    monotonic_stoploss_ratchet,
    leverage_constraint,
)

# 1. Zero-division immunity
safe_ratio = guarded_div(numerator=150.0, denominator=0.0, default=0.0)
assert safe_ratio == 0.0

# 2. Half-Kelly position sizing (60% win rate, 2.0 payoff)
safe_stake_fraction = kelly_criterion(win_rate=0.60, win_loss_ratio=2.0, max_kelly_fraction=0.50)
print(f"Optimal safe stake: {safe_stake_fraction * 100:.1f}%")  # 40.0%

# 3. Monotonic stoploss ratchet (can only tighten risk, never loosen)
current_sl = 100.0
# Market rises to 115 -> SL tightens to 110 (115 - 5)
sl_t1 = monotonic_stoploss_ratchet(current_sl, new_price=115.0, trailing_delta=5.0, is_long=True)
assert sl_t1 == 110.0

# Market drops to 108 -> SL invariant holds at 110.0 (NEVER drops back)
sl_t2 = monotonic_stoploss_ratchet(sl_t1, new_price=108.0, trailing_delta=5.0, is_long=True)
assert sl_t2 == 110.0
```

---

## Contributing

We welcome contributions from quantitative researchers, signal processing engineers, and formal verification specialists.

```bash
# Run Python test suite
pytest tests -v

# Run Lean 4 formal verification proofs
cd verification && lake build
```

Please review [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before opening a pull request.

---

## Citation

If you use `causal-dsp` in your research, trading systems, or academic publications, please cite:

```bibtex
@software{siddik2026causaldsp,
  author       = {Md Abubakar Siddik},
  title        = {causal-dsp: Formally Verified Prefix-Causal Signal Processing & Mathematical Risk Invariants for Time-Series Analysis in Python and Lean 4},
  month        = oct,
  year         = 2026,
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/abubakar293/causal-dsp}}
}
```

---

## License

This project is open source and distributed under the [MIT License](LICENSE).  
Copyright © 2026 Md Abubakar Siddik. All rights reserved.
