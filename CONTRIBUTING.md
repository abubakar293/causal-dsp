# Contributing to `causal-dsp`

Thank you for your interest in contributing to `causal-dsp`! We are building a formally verified, mathematically rigorous foundation for prefix-causal signal processing in Python and Lean 4.

## Core Design Principles

1. **Zero Future Leakage**: Every filter, estimator, or wavelet decomposition must strictly operate on historical prefix data $x[0 \dots t]$. Never use symmetric or periodic boundary conditions.
2. **Dual-Layer Verification**: Algorithms should be implemented with zero-lookahead guarantees in Python, and their underlying properties must be formally proven in Lean 4.
3. **100% Determinism**: Zero stochastic nondeterminism in unit tests.

## Development Workflow

### Prerequisites
- Python 3.10+
- Lean 4 (`leanprover/lean4:v4.35.0` or higher) with `lake`

### Setup

```bash
git clone https://github.com/abubakar293/causal-dsp.git
cd causal-dsp

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in editable mode with development dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Python unit tests and perturbation audits
pytest -v

# Lean 4 formal proofs compilation
cd verification
lake build
```

## Pull Request Guidelines

- Ensure `pytest` passes with 100% success.
- If you add a new filter, add a corresponding `perturbation_shock_test` and `validate_zero_future_leakage` test case.
- Adhere to PEP 8, add complete type annotations, and maintain docstrings in NumPy style.
- If introducing mathematical theorems or safety invariants, provide Lean 4 proofs with zero `sorry` and standard axioms only.
