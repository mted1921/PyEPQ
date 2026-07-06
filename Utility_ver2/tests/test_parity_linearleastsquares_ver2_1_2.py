r"""
test_parity_linearleastsquares_ver2_1_1.py -- parity harness for LinearLeastSquares

Pre-written harness (Prompt 2): targets the port's expected API per its spec.

LinearLeastSquares is an abstract SVD-based linear fitter. A subclass implements
  _fitFunctionCount() -> int
  _fitFunction(xi, afunc) -> None   (fills afunc[k] with the k-th basis value at xi)

M4: the class is abstract; JPype cannot subclass it from Python, so there is no
cross-engine Java parity (Part 2 skipped). Correctness is validated with a concrete
polynomial basis whose least-squares solution on exact polynomial data is the
generating coefficient vector.

All tests pass an explicit sigma array (all 1.0): the `setData(x, y)` /
sig=None path leaves the internal sigma unset in the Java source, so the harness
exercises only the fully-specified `setData(x, y, sig)` form.

Revision ver2_1_1 (2026-07-01):
  - Added TestSetData: exercises the setData R4 dispatcher (FIX-1 in the port).
  - Added TestReducedChiSquared: covers reducedChiSquared(confidenceLevel).
"""
from __future__ import annotations

import math

import pytest
from hypothesis import given, strategies as st

from _parity_lib import slow, _close

from LinearLeastSquares_ver2_1_1 import (  # FIX-2: ver2_1_0 → ver2_1_1 (port renamed after FIX-1)
    LinearLeastSquares as PyLLS,
)


# ---------------------------------------------------------------------------
# Concrete polynomial least-squares model
# ---------------------------------------------------------------------------

class _PolyLLS(PyLLS):
    """Fit y = sum_k c_k * x^k. fitFunctionCount = degree + 1."""

    def __init__(self, degree: int, xs, ys, sigs) -> None:
        self._degree = degree              # set BEFORE the fit is ever triggered
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
    return _PolyLLS(degree, _XS, ys, [1.0] * len(_XS))


# ############################################################################
# PART 1 -- Analytical correctness
# ############################################################################


class TestConstruction:
    """Verify concrete subclass construction and initial state.

    LinearLeastSquares is abstract; all construction goes through a concrete
    subclass that supplies _fitFunctionCount() and _fitFunction(). The tests
    here confirm that a freshly constructed instance has the correct basis size
    and that data can be replaced without error before any fit is triggered.
    """

    def test_fit_function_count_matches_degree(self) -> None:
        # degree=1 -> 2 basis functions (constant + linear)
        fit = _poly(1, [0.0, 1.0])
        assert fit._fitFunctionCount() == 2

    def test_fit_function_count_quadratic(self) -> None:
        fit = _poly(2, [1.0, 2.0, 3.0])
        assert fit._fitFunctionCount() == 3

    def test_setData_before_fit_does_not_raise(self) -> None:
        fit = _poly(1, [2.0, 3.0])
        fit.setData(_XS, [1.0] * len(_XS), [1.0] * len(_XS))


class TestLinearFit:
    """y = a + b*x exact -> fit recovers [a, b]."""

    def test_recovers_coeffs(self) -> None:
        fit = _poly(1, [2.0, 3.0])
        params = fit.fitParameters()
        assert _close(params[0], 2.0, 1e-7)
        assert _close(params[1], 3.0, 1e-7)

    def test_chi_squared_zero_on_exact_data(self) -> None:
        assert _poly(1, [2.0, 3.0]).chiSquared() < 1e-8

    def test_fit_paramter_single(self) -> None:
        fit = _poly(1, [2.0, 3.0])
        assert _close(fit.fitParamter(1), 3.0, 1e-7)   # note Java spelling


class TestQuadraticFit:
    """y = c0 + c1*x + c2*x^2 exact -> fit recovers [c0, c1, c2]."""

    def test_recovers_coeffs(self) -> None:
        fit = _poly(2, [1.0, -2.0, 0.5])
        params = fit.fitParameters()
        assert _close(params[0], 1.0, 1e-6)
        assert _close(params[1], -2.0, 1e-6)
        assert _close(params[2], 0.5, 1e-6)

    def test_results_length(self) -> None:
        res = _poly(2, [1.0, -2.0, 0.5]).getResults()
        assert len(res) == 3

    @given(
        st.floats(-5.0, 5.0, allow_nan=False, allow_infinity=False),
        st.floats(-5.0, 5.0, allow_nan=False, allow_infinity=False),
        st.floats(-2.0, 2.0, allow_nan=False, allow_infinity=False),
    )
    @slow
    def test_quadratic_property(self, c0, c1, c2) -> None:
        params = _poly(2, [c0, c1, c2]).fitParameters()
        assert _close(params[0], c0, 1e-5, rtol=1e-5)
        assert _close(params[1], c1, 1e-5, rtol=1e-5)
        assert _close(params[2], c2, 1e-5, rtol=1e-5)


class TestCovarianceAndErrors:
    def test_covariance_shape(self) -> None:
        cov = _poly(2, [1.0, 1.0, 1.0]).covariance()
        assert cov.getRowDimension() == 3
        assert cov.getColumnDimension() == 3

    def test_errors_length(self) -> None:
        errs = _poly(2, [1.0, 1.0, 1.0]).errors()
        assert len(errs) == 3
        assert all(e >= 0.0 for e in errs)

    def test_correlation_diagonal_is_one(self) -> None:
        corr = _poly(2, [1.0, 1.0, 1.0]).correlation()
        for i in range(corr.getRowDimension()):
            assert _close(corr.get(i, i), 1.0, 1e-9)

    def test_correlation_bounded(self) -> None:
        corr = _poly(2, [1.0, 1.0, 1.0]).correlation()
        for r in range(corr.getRowDimension()):
            for c in range(corr.getColumnDimension()):
                assert -1.0 - 1e-9 <= corr.get(r, c) <= 1.0 + 1e-9

    def test_confidence_intervals_match_errors_at_one_sigma(self) -> None:
        fit = _poly(2, [1.0, 1.0, 1.0])
        fit.fitParameters()
        cov = fit.covariance()
        ci = fit.confidenceIntervals(PyLLS.INTERVAL_MODE.ONE_D_INTERVAL, 0.683, cov)
        errs = fit.errors()
        # ONE_D_INTERVAL at 0.683 uses k=1 -> sqrt(cov[i,i]) == errors().
        for a, b in zip(ci, errs):
            assert _close(a, b, 1e-9, rtol=1e-9)


class TestZeroedCoefficients:
    def test_zeroed_coefficient_count(self) -> None:
        fit = _poly(2, [1.0, 1.0, 1.0])
        assert fit.getNonZeroedCoefficientCount() == 3
        fit.zeroFitCoefficient(2, True)
        assert fit.getNonZeroedCoefficientCount() == 2
        assert fit.isZeroFitCoefficient(2)

    def test_clear_zeroed(self) -> None:
        fit = _poly(2, [1.0, 1.0, 1.0])
        fit.zeroFitCoefficient(2, True)
        fit.clearZeroedCoefficients()
        assert not fit.isZeroFitCoefficient(2)


class TestChiSqrStatic:
    """chiSqr(dof, prob): scipy primary + literal bisection must agree."""

    @pytest.mark.parametrize("dof, prob", [
        (1, 0.5), (1, 0.9), (2, 0.683), (5, 0.95), (10, 0.5),
    ])
    def test_primary_matches_literal(self, dof: int, prob: float) -> None:
        primary = PyLLS.chiSqr(dof, prob)
        literal = PyLLS.chiSqr_literal(dof, prob)
        # Same target; literal bisection converges to ~0.01, so allow slack.
        assert _close(primary, literal, 0.05, rtol=0.01)

    def test_monotonic_in_prob(self) -> None:
        assert PyLLS.chiSqr(5, 0.5) < PyLLS.chiSqr(5, 0.95)

    def test_positive(self) -> None:
        assert PyLLS.chiSqr(3, 0.9) > 0.0


class TestFitQuality:
    def test_fit_quality_in_unit_interval(self) -> None:
        q = _poly(1, [2.0, 3.0]).fitQuality()
        assert 0.0 <= q <= 1.0


class TestConfidenceIntervalsJoint:
    """JOINT_INTERVAL path: inverts covariance, uses sub-determinant for each CI.

    For exact polynomial data the residuals are machine-eps, so the covariance
    is dominated by the SVD pseudo-inverse of the design matrix. The key check
    is that the result is non-negative and has the right length.
    """

    def test_joint_interval_length_matches_params(self) -> None:
        fit = _poly(1, [2.0, 3.0])
        fit.fitParameters()   # trigger SVD so covariance is available
        cov = fit.covariance()
        ci = fit.confidenceIntervals(PyLLS.INTERVAL_MODE.JOINT_INTERVAL, 0.683, cov)
        assert len(ci) == 2

    def test_joint_interval_values_nonneg(self) -> None:
        fit = _poly(2, [1.0, -2.0, 0.5])
        fit.fitParameters()
        cov = fit.covariance()
        ci = fit.confidenceIntervals(PyLLS.INTERVAL_MODE.JOINT_INTERVAL, 0.683, cov)
        assert all(v >= 0.0 for v in ci)


class TestSetData:
    """setData R4 dispatcher routes to setData_xysig / setData_xy (port FIX-1)."""

    def test_setData_replaces_data_and_refit(self) -> None:
        fit = _poly(1, [2.0, 3.0])
        _ = fit.fitParameters()          # trigger initial SVD fit
        new_ys = [5.0 + 7.0 * x for x in _XS]
        fit.setData(_XS, new_ys, [1.0] * len(_XS))
        params = fit.fitParameters()     # must refit on new data
        assert _close(params[0], 5.0, 1e-7)
        assert _close(params[1], 7.0, 1e-7)


class TestReducedChiSquared:
    """reducedChiSquared(cl) = chiSquared(fp) / chiSqr(dof, cl).

    dof = (points with sigma < MAX_ERROR) − (non-zero fit parameters).
    For exact polynomial data all residuals are ≈ 0 so the result is ≈ 0.
    """

    def test_reducedChiSquared_nonneg(self) -> None:
        fit = _poly(1, [2.0, 3.0])
        assert fit.reducedChiSquared(0.683) >= 0.0

    def test_reducedChiSquared_exact_data_near_zero(self) -> None:
        # Exact polynomial data → residuals ≈ machine eps → result ≈ 0.
        fit = _poly(1, [2.0, 3.0])
        assert fit.reducedChiSquared(0.683) < 1e-8

    def test_reducedChiSquared_confidence_changes_denominator(self) -> None:
        # chiSqr(dof, p) is strictly increasing in p, so higher confidence
        # → larger denominator → smaller ratio (numerator fixed: exact data).
        fit = _poly(2, [1.0, -2.0, 0.5])
        low = fit.reducedChiSquared(0.5)
        high = fit.reducedChiSquared(0.95)
        # Both ≈ 0 for exact data, but the ordering must hold (or both == 0).
        assert low >= high or (low < 1e-20 and high < 1e-20)


# ############################################################################
# PART 2 -- Java parity (M4 — skipped)
# ############################################################################

@pytest.mark.skip(
    reason="M4: LinearLeastSquares is abstract; JPype cannot extend it from "
           "Python, so a Python fit basis cannot drive the Java solver. "
           "Correctness is validated against closed-form polynomial least-squares "
           "solutions in Part 1."
)
class TestLinearLeastSquaresParity:
    def test_placeholder(self) -> None:
        pass


if __name__ == "__main__":
    from _parity_lib import run_tests; run_tests(__file__)
