r"""
test_parity_linearleastsquaresms_ver2_1_0.py -- parity harness for LinearLeastSquaresMS

Pre-written harness (Prompt 2): targets the port's expected API per its spec.

LinearLeastSquaresMS extends LinearLeastSquares with two behaviours:
  * negative fit coefficients are forced to zero (always, even with optimize off);
  * with setOptimize(True), a Bayesian model-selection loop trims the number of
    non-zero parameters to minimise computeMetric().

Still abstract -> M4: no cross-engine Java parity (Part 2 skipped). Correctness is
validated with a concrete polynomial basis. Explicit sigmas are always supplied.
"""
from __future__ import annotations

import math

import pytest
from hypothesis import given, strategies as st

from _parity_lib import _close, slow

from LinearLeastSquaresMS_ver2_1_1 import (  # ver2_1_0 → ver2_1_1 (port renamed after repair round 1)
    LinearLeastSquaresMS as PyLLSMS,
)


class _PolyLLSMS(PyLLSMS):
    """Fit y = sum_k c_k * x^k."""

    def __init__(self, degree: int, xs, ys, sigs) -> None:
        self._degree = degree
        super().__init__(xs, ys, sigs)

    def _fitFunctionCount(self) -> int:
        return self._degree + 1

    def _fitFunction(self, xi: float, afunc) -> None:
        v = 1.0
        for k in range(self._degree + 1):
            afunc[k] = v
            v *= xi


_XS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def _poly(degree: int, coeffs):
    ys = [sum(c * (x ** k) for k, c in enumerate(coeffs)) for x in _XS]
    return _PolyLLSMS(degree, _XS, ys, [1.0] * len(_XS))


# ############################################################################
# PART 1 -- Analytical correctness
# ############################################################################


class TestConstruction:
    def test_optimize_default_false(self) -> None:
        assert _poly(1, [1.0, 1.0]).isOptimize() is False

    def test_set_optimize_roundtrip(self) -> None:
        fit = _poly(1, [1.0, 1.0])
        fit.setOptimize(True)
        assert fit.isOptimize() is True
        fit.setOptimize(False)
        assert fit.isOptimize() is False


class TestNonNegativeFit:
    """All-positive coefficients: nothing is zeroed, fit is exact."""

    def test_quadratic_recovered(self) -> None:
        params = _poly(2, [1.0, 2.0, 0.5]).fitParameters()
        assert _close(params[0], 1.0, 1e-6)
        assert _close(params[1], 2.0, 1e-6)
        assert _close(params[2], 0.5, 1e-6)


class TestNegativeCoefficientZeroing:
    """A negative coefficient in the unconstrained fit is forced to zero."""

    def test_negative_slope_zeroed(self) -> None:
        # Exact data y = 10 - x: unconstrained fit is [10, -1]; the negative
        # slope is zeroed, leaving a constant fit = mean(y) = 7.5.
        params = _poly(1, [10.0, -1.0]).fitParameters()
        assert params[1] == 0.0            # negative coefficient zeroed
        assert _close(params[0], 7.5, 1e-6)


class TestOptimizeRuns:
    """setOptimize(True) runs the model-selection loop without error."""

    def test_optimize_completes(self) -> None:
        fit = _poly(2, [1.0, 2.0, 0.5])
        fit.setOptimize(True)
        params = fit.fitParameters()
        assert len(params) == 3
        assert math.isfinite(fit.chiSquared())


class TestFuzzCoefficients:
    """Property: for all-positive polynomial data, fitParameters recovers coefficients."""

    @given(
        st.floats(0.5, 5.0, allow_nan=False, allow_infinity=False),
        st.floats(0.5, 5.0, allow_nan=False, allow_infinity=False),
    )
    @slow
    def test_positive_linear_recovered(self, c0: float, c1: float) -> None:
        params = _poly(1, [c0, c1]).fitParameters()
        assert _close(params[0], c0, 1e-5, rtol=1e-5)
        assert _close(params[1], c1, 1e-5, rtol=1e-5)


# ############################################################################
# PART 2 -- Java parity (M4 — skipped)
# ############################################################################

@pytest.mark.skip(
    reason="M4: LinearLeastSquaresMS is abstract; JPype cannot extend it from "
           "Python. Correctness is validated against closed-form least-squares "
           "results (with the documented negative-coefficient zeroing) in Part 1."
)
class TestLinearLeastSquaresMSParity:
    def test_placeholder(self) -> None:
        pass


if __name__ == "__main__":
    from _parity_lib import run_tests; run_tests(__file__)
