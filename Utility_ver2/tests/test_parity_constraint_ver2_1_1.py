r"""
test_parity_constraint_ver2_1_1.py -- parity harness for Constraint

Pre-written harness (Prompt 2): targets the port's expected API per
Constraint_ver1_1_1.spec.md (the spec was repaired from the Java source).

`Constraint` is an interface mapping the optimizer's real line (-inf, inf) onto a
constrained sub-range through invertible, differentiable functions:
  realToConstrained(p)  constrainedToReal(res)  derivative(p)  getResult(uv2: UV2)

Concrete implementations (module-level aliases re-exported from the port):
  Positive(scale)                 c = scale * exp(p)
  Fractional(name, scale, frac)   c = scale * (1 + frac * (2/pi) * atan(p))
  Bounded(center, width)          c = center + (width/2) * (2/pi) * atan(p)
  Unconstrained()                 c = p            (Java `Constraint.None`)

The concrete classes are instantiable, so Part 2 compares them directly against
Java via JPype (Constraint is a Java *interface*; M4 does not block these).
"""
from __future__ import annotations

import math

import pytest
from hypothesis import given, strategies as st

from _parity_lib import (
    setup_parity, jclass, needs_java,
    slow,
    _close, _bdry_close,
)

from Constraint_ver2_1_3 import (  # FIX-3: ver2_1_1 → ver2_1_2 (port renamed after FIX-2/3)
    Positive as PyPositive,
    Fractional as PyFractional,
    Bounded as PyBounded,
    Unconstrained as PyUnconstrained,
)
from UncertainValue2_ver2_1_0 import UncertainValue2 as PyUncertainValue2  # FIX-1

ctx = setup_parity("gov.nist.microanalysis.Utility.Constraint")

_TWO_OVER_PI = 2.0 / math.pi
_TOL = 1e-12

# Inputs well away from the tan() asymptotes used by constrainedToReal.
_real = st.floats(min_value=-50.0, max_value=50.0,
                  allow_nan=False, allow_infinity=False)


# ############################################################################
# PART 1 -- Always-on analytical tests
# ############################################################################


class TestUnconstrained:
    """Java Constraint.None -> port Unconstrained: the identity map."""

    @given(_real)
    @slow
    def test_real_to_constrained_identity(self, p: float) -> None:
        assert PyUnconstrained().realToConstrained(p) == p

    @given(_real)
    @slow
    def test_constrained_to_real_identity(self, p: float) -> None:
        assert PyUnconstrained().constrainedToReal(p) == p

    @given(_real)
    @slow
    def test_derivative_is_one(self, p: float) -> None:
        assert PyUnconstrained().derivative(p) == 1.0

    def test_get_result_passthrough(self) -> None:
        uv = PyUncertainValue2(2.0, 0.3)
        res = PyUnconstrained().getResult(uv)
        assert res.doubleValue() == 2.0


class TestPositive:
    """c = scale * exp(p); inverse log(res/scale); derivative = scale*exp(p)."""

    def test_real_to_constrained_value(self) -> None:
        assert _bdry_close(PyPositive(2.0).realToConstrained(0.0), 2.0, _TOL)
        assert _bdry_close(PyPositive(2.0).realToConstrained(1.0),
                           2.0 * math.e, _TOL)

    def test_always_positive(self) -> None:
        c = PyPositive(3.0)
        for p in (-10.0, -1.0, 0.0, 1.0, 5.0):
            assert c.realToConstrained(p) > 0.0

    def test_derivative_equals_value(self) -> None:
        c = PyPositive(2.5)
        for p in (-2.0, 0.0, 1.5):
            assert _close(c.derivative(p), c.realToConstrained(p), _TOL)

    @given(st.floats(min_value=0.1, max_value=10.0, allow_nan=False),
           _real)
    @slow
    def test_inverse_roundtrip(self, scale: float, p: float) -> None:
        c = PyPositive(scale)
        assert _close(c.constrainedToReal(c.realToConstrained(p)), p,
                      1e-9, rtol=1e-9)

    def test_get_result_nominal_matches_map(self) -> None:
        c = PyPositive(2.0)
        uv = PyUncertainValue2(1.0, 0.1)
        assert _close(c.getResult(uv).doubleValue(),
                      c.realToConstrained(1.0), 1e-12)


class TestFractional:
    """c = scale * (1 + frac * (2/pi) * atan(p))."""

    def test_center_at_zero(self) -> None:
        # atan(0) = 0 -> c = scale
        assert _bdry_close(PyFractional("a", 5.0, 0.2).realToConstrained(0.0),
                           5.0, _TOL)

    def test_real_to_constrained_value(self) -> None:
        c = PyFractional("a", 4.0, 0.5)
        expected = 4.0 * (1.0 + 0.5 * _TWO_OVER_PI * math.atan(2.0))
        assert _bdry_close(c.realToConstrained(2.0), expected, _TOL)

    def test_derivative_value(self) -> None:
        c = PyFractional("a", 4.0, 0.5)
        p = 1.3
        expected = (4.0 * 0.5 * _TWO_OVER_PI) / (1.0 + p * p)
        assert _bdry_close(c.derivative(p), expected, _TOL)

    @given(st.floats(min_value=0.5, max_value=10.0, allow_nan=False),
           st.floats(min_value=0.05, max_value=0.9, allow_nan=False),
           _real)
    @slow
    def test_inverse_roundtrip(self, scale: float, frac: float, p: float) -> None:
        c = PyFractional("a", scale, frac)
        assert _close(c.constrainedToReal(c.realToConstrained(p)), p,
                      1e-7, rtol=1e-7)

    def test_get_result_nominal_matches_map(self) -> None:
        c = PyFractional("a", 4.0, 0.5)
        uv = PyUncertainValue2(1.3, 0.1)
        assert _close(c.getResult(uv).doubleValue(),
                      c.realToConstrained(1.3), 1e-12)


class TestBounded:
    """c = center + (width/2) * (2/pi) * atan(p); the constructor halves width."""

    def test_center_at_zero(self) -> None:
        assert _bdry_close(PyBounded(3.0, 2.0).realToConstrained(0.0),
                           3.0, _TOL)

    def test_real_to_constrained_value(self) -> None:
        c = PyBounded(3.0, 2.0)   # half-width 1.0
        expected = 3.0 + 1.0 * _TWO_OVER_PI * math.atan(1.5)
        assert _bdry_close(c.realToConstrained(1.5), expected, _TOL)

    def test_stays_within_bounds(self) -> None:
        # As p -> +-inf, atan -> +-pi/2, so c -> center +- half-width.
        c = PyBounded(0.0, 2.0)   # half-width 1.0 -> range (-1, 1)
        for p in (-1e6, -100.0, 0.0, 100.0, 1e6):
            v = c.realToConstrained(p)
            assert -1.0 <= v <= 1.0

    def test_derivative_value(self) -> None:
        c = PyBounded(0.0, 4.0)   # half-width 2.0
        p = 0.7
        expected = (2.0 * _TWO_OVER_PI) / (1.0 + p * p)
        assert _bdry_close(c.derivative(p), expected, _TOL)

    @given(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False),
           st.floats(min_value=0.5, max_value=20.0, allow_nan=False),
           _real)
    @slow
    def test_inverse_roundtrip(self, center: float, width: float, p: float) -> None:
        c = PyBounded(center, width)
        assert _close(c.constrainedToReal(c.realToConstrained(p)), p,
                      1e-7, rtol=1e-7)

    def test_get_result_nominal_matches_map(self) -> None:
        c = PyBounded(3.0, 2.0)   # half-width 1.0
        uv = PyUncertainValue2(1.5, 0.2)
        assert _close(c.getResult(uv).doubleValue(),
                      c.realToConstrained(1.5), 1e-12)


# ############################################################################
# PART 2 -- Parity tests (require JVM + EPQ.jar + EPQ_PARITY=1)
# ############################################################################
# Constraint is a Java interface; its concrete inner classes are instantiable,
# so JPype can build them directly (M4 does not apply).

_JFQN = "gov.nist.microanalysis.Utility.Constraint"


@needs_java
class TestPositiveParity:
    def _pair(self, scale: float):
        return PyPositive(scale), jclass(_JFQN + "$Positive")(scale)

    @given(st.floats(0.1, 10.0), _real)
    @slow
    def test_real_to_constrained(self, scale: float, p: float) -> None:
        py, j = self._pair(scale)
        assert _close(j.realToConstrained(p), py.realToConstrained(p),
                      1e-12, rtol=1e-12)

    @given(st.floats(0.1, 10.0), _real)
    @slow
    def test_derivative(self, scale: float, p: float) -> None:
        py, j = self._pair(scale)
        assert _close(j.derivative(p), py.derivative(p), 1e-12, rtol=1e-12)


@needs_java
class TestFractionalParity:
    @given(st.floats(0.5, 10.0), st.floats(0.05, 0.9), _real)
    @slow
    def test_real_to_constrained(self, scale: float, frac: float, p: float) -> None:
        py = PyFractional("a", scale, frac)
        j = jclass(_JFQN + "$Fractional")("a", scale, frac)
        assert _close(j.realToConstrained(p), py.realToConstrained(p),
                      1e-12, rtol=1e-12)

    @given(st.floats(0.5, 10.0), st.floats(0.05, 0.9), _real)
    @slow
    def test_derivative(self, scale: float, frac: float, p: float) -> None:
        py = PyFractional("a", scale, frac)
        j = jclass(_JFQN + "$Fractional")("a", scale, frac)
        assert _close(j.derivative(p), py.derivative(p), 1e-12, rtol=1e-12)


@needs_java
class TestBoundedParity:
    @given(st.floats(-10.0, 10.0), st.floats(0.5, 20.0), _real)
    @slow
    def test_real_to_constrained(self, center: float, width: float, p: float) -> None:
        py = PyBounded(center, width)
        j = jclass(_JFQN + "$Bounded")(center, width)
        assert _close(j.realToConstrained(p), py.realToConstrained(p),
                      1e-12, rtol=1e-12)

    @given(st.floats(-10.0, 10.0), st.floats(0.5, 20.0), _real)
    @slow
    def test_derivative(self, center: float, width: float, p: float) -> None:
        py = PyBounded(center, width)
        j = jclass(_JFQN + "$Bounded")(center, width)
        assert _close(j.derivative(p), py.derivative(p), 1e-12, rtol=1e-12)


@needs_java
class TestUnconstrainedParity:
    @given(_real)
    @slow
    def test_identity_and_derivative(self, p: float) -> None:
        py = PyUnconstrained()
        j = jclass(_JFQN + "$None")()
        assert _close(j.realToConstrained(p), py.realToConstrained(p), 1e-12)
        assert _close(j.derivative(p), py.derivative(p), 1e-12)


if __name__ == "__main__":
    from _parity_lib import run_tests; run_tests(__file__)
