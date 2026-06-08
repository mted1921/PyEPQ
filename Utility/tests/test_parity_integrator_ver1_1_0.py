r"""
test_parity_integrator_ver1_1_0.py — parity harness for Integrator_ver1_1_0.py

Integrator is doubly-abstract:
  • extends AdaptiveRungeKutta (abstract)
  • declares abstract getValue(double x)

M4 applies twice over.  Behavioural correctness is validated analytically by
creating Python concrete subclasses.  The integrate() method wraps ARK; a
single-variable derivative vector dydx[0] = getValue(x) is used internally.

Analytical benchmarks:
  ∫₀¹ 1   dx  = 1.0
  ∫₀¹ x   dx  = 0.5
  ∫₀ᵖ sin dx  = 2.0   (p = math.pi)
  ∫₀¹ e^x dx  = e − 1 ≈ 1.71828
"""
from __future__ import annotations

import math

import pytest
from hypothesis import given, strategies as st

from _parity_lib import (
    setup_parity, needs_java, PARITY_ENABLED,
    TOL_LITERAL, TOL_NR_LIB,
    _close,
)

from Integrator_ver1_1_0 import Integrator as PyIntegrator

ctx = setup_parity("gov.nist.microanalysis.Utility.Integrator")
JavaIntegrator = ctx.java_class

_ITOL = 1e-6   # default ARK tolerance carried over by Integrator()


# ---------------------------------------------------------------------------
# Concrete subclasses
# ---------------------------------------------------------------------------

class _ConstantIntegrator(PyIntegrator):
    """getValue(x) = 1.0  →  ∫ = (high − low)"""
    def getValue(self, x: float) -> float:
        return 1.0


class _LinearIntegrator(PyIntegrator):
    """getValue(x) = x  →  ∫₀¹ = 0.5"""
    def getValue(self, x: float) -> float:
        return x


class _SinIntegrator(PyIntegrator):
    """getValue(x) = sin(x)  →  ∫₀ᵖ = 2.0"""
    def getValue(self, x: float) -> float:
        return math.sin(x)


class _ExpIntegrator(PyIntegrator):
    """getValue(x) = exp(x)  →  ∫₀¹ = e − 1"""
    def getValue(self, x: float) -> float:
        return math.exp(x)


class _ConstantTolIntegrator(PyIntegrator):
    """Constructed with explicit tolerance."""
    def __init__(self, tol: float):
        super().__init__(tol)

    def getValue(self, x: float) -> float:
        return 1.0


# ---------------------------------------------------------------------------
# TestConstruction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_constructor(self):
        ig = _ConstantIntegrator()
        assert ig is not None

    def test_tolerance_constructor(self):
        ig = _ConstantTolIntegrator(1e-8)
        assert ig is not None


# ---------------------------------------------------------------------------
# TestIntegrateConstant
# ---------------------------------------------------------------------------

class TestIntegrateConstant:
    def test_unit_interval(self):
        ig = _ConstantIntegrator()
        result = ig.integrate(0.0, 1.0)
        assert _close(result, 1.0, TOL_NR_LIB)

    def test_wider_interval(self):
        ig = _ConstantIntegrator()
        result = ig.integrate(0.0, 5.0)
        assert _close(result, 5.0, TOL_NR_LIB)

    def test_reversed_limits(self):
        ig = _ConstantIntegrator()
        result = ig.integrate(1.0, 0.0)
        assert _close(abs(result), 1.0, TOL_NR_LIB)

    def test_zero_width_interval(self):
        ig = _ConstantIntegrator()
        result = ig.integrate(3.0, 3.0)
        assert _close(result, 0.0, TOL_NR_LIB)


# ---------------------------------------------------------------------------
# TestIntegrateLinear
# ---------------------------------------------------------------------------

class TestIntegrateLinear:
    def test_zero_to_one(self):
        ig = _LinearIntegrator()
        result = ig.integrate(0.0, 1.0)
        assert _close(result, 0.5, TOL_NR_LIB)

    def test_zero_to_two(self):
        ig = _LinearIntegrator()
        result = ig.integrate(0.0, 2.0)
        assert _close(result, 2.0, TOL_NR_LIB)

    def test_one_to_three(self):
        ig = _LinearIntegrator()
        result = ig.integrate(1.0, 3.0)
        # ∫₁³ x dx = [x²/2]₁³ = 9/2 − 1/2 = 4.0
        assert _close(result, 4.0, TOL_NR_LIB)


# ---------------------------------------------------------------------------
# TestIntegrateSin
# ---------------------------------------------------------------------------

class TestIntegrateSin:
    def test_zero_to_pi(self):
        ig = _SinIntegrator()
        result = ig.integrate(0.0, math.pi)
        assert _close(result, 2.0, TOL_NR_LIB)

    def test_zero_to_half_pi(self):
        ig = _SinIntegrator()
        result = ig.integrate(0.0, math.pi / 2.0)
        assert _close(result, 1.0, TOL_NR_LIB)


# ---------------------------------------------------------------------------
# TestIntegrateExp
# ---------------------------------------------------------------------------

class TestIntegrateExp:
    def test_zero_to_one(self):
        ig = _ExpIntegrator()
        result = ig.integrate(0.0, 1.0)
        assert _close(result, math.e - 1.0, TOL_NR_LIB)


# ---------------------------------------------------------------------------
# TestHypothesis
# ---------------------------------------------------------------------------

class TestHypothesis:
    @given(st.floats(0.0, 10.0), st.floats(0.0, 10.0))
    def test_constant_integral_equals_interval_length(self, a, b):
        ig = _ConstantIntegrator()
        result = ig.integrate(a, b)
        expected = abs(b - a)
        assert _close(result, expected, 1e-4)


# ---------------------------------------------------------------------------
# TestIntegratorParity  (M4 — doubly abstract)
# ---------------------------------------------------------------------------

@pytest.mark.skip(
    reason="M4: Integrator extends AdaptiveRungeKutta (abstract) and itself "
           "declares abstract getValue(). JPype cannot extend abstract Java "
           "classes from Python. Correctness validated analytically above."
)
class TestIntegratorParity:
    def test_placeholder(self):
        pass

if __name__ == "__main__":
    from _parity_lib import run_tests; run_tests(__file__)
