r"""
test_parity_levenbergmarquardtconstrained_ver2_1_0.py -- parity harness for LevenbergMarquardtConstrained

Pre-written harness (Prompt 2): targets the port's expected API per its spec.

LevenbergMarquardtConstrained extends LevenbergMarquardt2. It optimises over the
full real line but maps to a constrained parameter domain through Constraint
objects, wrapped by the inner class ConstrainedFitFunction:
  realToConstrained_matrix(rParams)  per-component realToConstrained
  constrainedToReal_matrix(rParams)  per-component constrainedToReal
  partials(rParams) / compute(rParams)  chain-ruled through the constraints
  realToConstrained_result(fitResult)  map a FitResult back to the domain

compute(ff, ...) dispatches: a ConstrainedFitFunction is transformed; any other
FitFunction falls through to the parent solver.

M4: the inner FitFunction is abstract, so a Python fit model cannot be passed into
the Java solver (Part 2 skipped). Correctness is validated analytically.
"""
from __future__ import annotations

import math

import pytest
from hypothesis import given, strategies as st

from _parity_lib import _close, needs_java, slow

from LevenbergMarquardtConstrained_ver2_1_0 import (
    LevenbergMarquardtConstrained as PyLMC,
    ConstrainedFitFunction as PyCFF,
)
from LevenbergMarquardt2_ver2_1_0 import FitFunction as PyFitFunction
from Constraint_ver2_1_3 import Positive as PyPositive, Unconstrained as PyUnconstrained
from _epq_compat import JamaMatrix


def _col(vals) -> JamaMatrix:
    return JamaMatrix([[float(v)] for v in vals])


class _LinearModel(PyFitFunction):
    """Domain-space y = a + b*x."""

    def __init__(self, xs) -> None:
        self._xs = [float(x) for x in xs]

    def compute(self, params: JamaMatrix) -> JamaMatrix:
        a, b = params.get(0, 0), params.get(1, 0)
        return JamaMatrix([[a + b * x] for x in self._xs])

    def partials(self, params: JamaMatrix) -> JamaMatrix:
        return JamaMatrix([[1.0, x] for x in self._xs])


class _ConstModel(PyFitFunction):
    """Domain-space y = a (single parameter)."""

    def __init__(self, n: int) -> None:
        self._n = n

    def compute(self, params: JamaMatrix) -> JamaMatrix:
        a = params.get(0, 0)
        return JamaMatrix([[a] for _ in range(self._n)])

    def partials(self, params: JamaMatrix) -> JamaMatrix:
        return JamaMatrix([[1.0] for _ in range(self._n)])


_XS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


# ############################################################################
# PART 1 -- Analytical correctness
# ############################################################################


class TestConstraintTransforms:
    """Per-component real<->constrained maps inside ConstrainedFitFunction."""

    def _cff(self):
        cff = PyCFF(_LinearModel(_XS), 2)   # 2 Unconstrained constraints by default
        cff.setConstraint(1, PyPositive(2.0))
        return cff

    def test_real_to_constrained_per_component(self) -> None:
        cff = self._cff()
        out = cff.realToConstrained_matrix(_col([0.5, 1.0]))
        assert _close(out.get(0, 0), 0.5, 1e-12)              # Unconstrained: identity
        assert _close(out.get(1, 0), 2.0 * math.e, 1e-12)    # Positive(2): 2*exp(1)

    def test_roundtrip_constrained_real(self) -> None:
        cff = self._cff()
        r = _col([0.5, 1.0])
        c = cff.realToConstrained_matrix(r)
        back = cff.constrainedToReal_matrix(c)
        assert _close(back.get(0, 0), 0.5, 1e-9)
        assert _close(back.get(1, 0), 1.0, 1e-9)

    def test_default_constraints_are_identity(self) -> None:
        cff = PyCFF(_LinearModel(_XS), 2)   # all Unconstrained
        out = cff.realToConstrained_matrix(_col([3.0, -4.0]))
        assert _close(out.get(0, 0), 3.0, 1e-12)
        assert _close(out.get(1, 0), -4.0, 1e-12)


class TestUnconstrainedFit:
    """All-Unconstrained constraints -> the fit equals the plain LM2 solution."""

    def test_recovers_linear(self) -> None:
        a_true, b_true = 2.0, 3.0
        ys = [a_true + b_true * x for x in _XS]
        cff = PyCFF(_LinearModel(_XS), 2)
        res = PyLMC().compute(cff, _col(ys), _col([1.0] * len(_XS)),
                              _col([0.0, 0.0]))
        params = res.getBestParameters()
        assert _close(params[0], a_true, 1e-5, rtol=1e-5)
        assert _close(params[1], b_true, 1e-5, rtol=1e-5)


class TestPositiveConstrainedFit:
    """A Positive-constrained constant model recovers the domain value."""

    def test_recovers_positive_constant(self) -> None:
        n = 5
        ys = [3.0] * n
        cff = PyCFF(_ConstModel(n), 1)
        cff.setConstraint(0, PyPositive(1.0))   # c = exp(p) > 0
        res = PyLMC().compute(cff, _col(ys), _col([1.0] * n), _col([1.0]))
        params = res.getBestParameters()
        assert _close(params[0], 3.0, 1e-4, rtol=1e-4)
        assert params[0] > 0.0


class TestDispatchToParent:
    """A plain FitFunction (not ConstrainedFitFunction) falls through to LM2."""

    def test_plain_fit_function_passthrough(self) -> None:
        a_true, b_true = 1.0, -0.5
        ys = [a_true + b_true * x for x in _XS]
        res = PyLMC().compute(_LinearModel(_XS), _col(ys),
                              _col([1.0] * len(_XS)), _col([0.0, 0.0]))
        params = res.getBestParameters()
        assert _close(params[0], a_true, 1e-5, rtol=1e-5)
        assert _close(params[1], b_true, 1e-5, rtol=1e-5)


class TestConstraintRoundtripProperty:
    """Property: realToConstrained → constrainedToReal is the identity for any real."""

    _real = st.floats(min_value=-20.0, max_value=20.0,
                      allow_nan=False, allow_infinity=False)

    @given(_real, _real)
    @slow
    def test_unconstrained_roundtrip(self, r0: float, r1: float) -> None:
        cff = PyCFF(_LinearModel(_XS), 2)  # default: all Unconstrained
        r = _col([r0, r1])
        c = cff.realToConstrained_matrix(r)
        back = cff.constrainedToReal_matrix(c)
        assert _close(back.get(0, 0), r0, 1e-9)
        assert _close(back.get(1, 0), r1, 1e-9)

    @given(st.floats(0.1, 10.0, allow_nan=False, allow_infinity=False), _real)
    @slow
    def test_positive_roundtrip(self, scale: float, r: float) -> None:
        from Constraint_ver2_1_3 import Positive as PyPositive
        cff = PyCFF(_LinearModel(_XS), 2)
        cff.setConstraint(0, PyPositive(scale))
        col = _col([r, 0.0])
        c = cff.realToConstrained_matrix(col)
        back = cff.constrainedToReal_matrix(c)
        assert _close(back.get(0, 0), r, 1e-9)


# ############################################################################
# PART 2 -- Java parity
# ############################################################################
# LevenbergMarquardtConstrained is a concrete class (not abstract).
# The inner FitFunction IS abstract, so a Python fit model cannot be passed
# into the Java solver (M4-like limitation on the integration test).
# The @needs_java decorator skips this class when the JVM is unavailable.

@needs_java
class TestLevenbergMarquardtConstrainedParity:
    """Concrete class: needs_java gate is required. Integration blocked by M4
    (inner FitFunction is abstract in Java; JPype cannot pass Python
    implementations). Constraint transforms are parity-tested in
    test_parity_constraint."""

    def test_placeholder(self) -> None:
        pytest.skip(
            "M4: the inner FitFunction is an abstract Java type; JPype cannot "
            "pass a Python fit model into the Java solver. Constraint transforms "
            "and constrained fits are validated analytically in Part 1."
        )


if __name__ == "__main__":
    from _parity_lib import run_tests; run_tests(__file__)
