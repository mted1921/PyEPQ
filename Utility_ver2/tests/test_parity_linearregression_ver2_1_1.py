r"""
test_parity_linearregression_ver2_1_1.py -- parity harness for LinearRegression

Revision ver2_1_1 (2026-06-30): FIX-1 + compliance coverage.
  * Import bumped to current port (FIX-1: getRSquared zero-denom).
  * TestCoverage class added: direct addData() and goodnessOfFit() calls.

LinearRegression accumulates weighted (x, y) data and fits the best line by least
squares (lazily, via a LazyEvaluate<Line>). It is concrete, so Part 2 compares the
fit statistics directly against Java.

Overload splits (R4): addDatum_xy / addDatum_xye, removeDatum_xy / removeDatum_xye.
Inner Line: Line(slope, intercept) and Line.from_two_points(x0, y0, x1, y1).
"""
from __future__ import annotations

import math

import pytest
from hypothesis import given, strategies as st

from _parity_lib import (
    setup_parity, needs_java,
    slow,
    _close,
)

from LinearRegression_ver2_1_1 import (
    LinearRegression as PyLR,
    Line as PyLine,
)

ctx = setup_parity("gov.nist.microanalysis.Utility.LinearRegression")
JavaLR = ctx.java_class

_TOL = 1e-9

# y = 2x + 3 exactly.
_LINE_PTS = [(0.0, 3.0), (1.0, 5.0), (2.0, 7.0), (3.0, 9.0), (4.0, 11.0)]


def _fit(points):
    lr = PyLR()
    for x, y in points:
        lr.addDatum_xy(x, y)
    return lr


# ############################################################################
# PART 1 -- Inner Line value object
# ############################################################################


class TestLine:
    def test_compute_y(self) -> None:
        assert _close(PyLine(2.0, 3.0).computeY(4.0), 11.0, _TOL)

    def test_compute_x(self) -> None:
        assert _close(PyLine(2.0, 3.0).computeX(11.0), 4.0, _TOL)

    def test_x_intercept(self) -> None:
        assert _close(PyLine(2.0, 3.0).getXIntercept(), -1.5, _TOL)

    def test_from_two_points(self) -> None:
        ln = PyLine.from_two_points(0.0, 3.0, 1.0, 5.0)
        assert _close(ln.getSlope(), 2.0, _TOL)
        assert _close(ln.getIntercept(), 3.0, _TOL)


# ############################################################################
# PART 1 -- Regression analytics
# ############################################################################


class TestExactLineFit:
    def test_slope(self) -> None:
        assert _close(_fit(_LINE_PTS).getSlope(), 2.0, 1e-9)

    def test_intercept(self) -> None:
        assert _close(_fit(_LINE_PTS).getIntercept(), 3.0, 1e-9)

    def test_r_squared_is_one(self) -> None:
        assert _close(_fit(_LINE_PTS).getRSquared(), 1.0, 1e-9)

    def test_chi_squared_zero(self) -> None:
        assert _fit(_LINE_PTS).chiSquared() < 1e-6

    def test_compute_y(self) -> None:
        assert _close(_fit(_LINE_PTS).computeY(10.0), 23.0, 1e-8)

    def test_compute_x(self) -> None:
        assert _close(_fit(_LINE_PTS).computeX(3.0), 0.0, 1e-8)

    def test_count(self) -> None:
        assert _fit(_LINE_PTS).getCount() == 5

    def test_get_result_is_line(self) -> None:
        ln = _fit(_LINE_PTS).getResult()
        assert _close(ln.getSlope(), 2.0, 1e-9)


class TestCovarianceAndCorrelation:
    def test_covariance_shape(self) -> None:
        cov = _fit(_LINE_PTS).covariance()
        assert cov.getRowDimension() == 2
        assert cov.getColumnDimension() == 2

    def test_correlation_value(self) -> None:
        # correlation = -mSxx / sqrt(mS*mSxx) = -sqrt(Sigma x^2 / N).
        # Sigma x^2 = 30, N = 5 -> -sqrt(6).
        assert _close(_fit(_LINE_PTS).correlation(), -math.sqrt(6.0), 1e-9)


class TestAddRemove:
    def test_remove_restores_fit(self) -> None:
        lr = _fit(_LINE_PTS)
        lr.addDatum_xy(100.0, 999.0)      # outlier
        lr.removeDatum_xy(100.0, 999.0)   # remove it
        assert _close(lr.getSlope(), 2.0, 1e-7)
        assert lr.getCount() == 5

    def test_set_data_replaces(self) -> None:
        lr = PyLR()
        lr.addDatum_xy(0.0, 0.0)
        lr.setData([0.0, 1.0, 2.0], [1.0, 3.0, 5.0])   # y = 2x + 1
        assert lr.getCount() == 3
        assert _close(lr.getSlope(), 2.0, 1e-9)
        assert _close(lr.getIntercept(), 1.0, 1e-9)

    def test_clear_resets(self) -> None:
        lr = _fit(_LINE_PTS)
        lr.clear()
        assert lr.getCount() == 0


class TestCoverage:
    """Direct calls to addData() and goodnessOfFit() for compliance coverage."""

    def test_add_data_direct(self) -> None:
        lr = PyLR()
        lr.addData([0.0, 1.0, 2.0], [3.0, 5.0, 7.0])   # y = 2x + 3
        assert lr.getCount() == 3
        assert _close(lr.getSlope(), 2.0, 1e-9)

    def test_goodness_of_fit_range(self) -> None:
        lr = PyLR()
        lr.addData([0.0, 1.0, 2.0, 3.0], [3.0, 5.0, 7.0, 9.0])   # exact y = 2x + 3
        gof = lr.goodnessOfFit()
        assert isinstance(gof, float)

    def test_r_squared_degenerate_returns_nan(self) -> None:
        # All y values identical → denominator is 0 → Java IEEE-754 gives NaN.
        lr = PyLR()
        lr.addDatum_xy(0.0, 0.0)
        lr.addDatum_xy(0.0, 0.0)
        lr.addDatum_xy(1.0, 0.0)
        assert math.isnan(lr.getRSquared())


# ############################################################################
# PART 2 -- Parity tests (Java oracle)
# ############################################################################

# Distinct x values so the fit is well-posed.
_pts = st.lists(
    st.tuples(
        st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False),
        st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False),
    ),
    min_size=3, max_size=30,
).filter(lambda ps: len({round(x, 6) for x, _ in ps}) >= 2)


@needs_java
class TestLinearRegressionParity:

    def _fit_java(self, points):
        lr = JavaLR()
        for x, y in points:
            lr.addDatum(float(x), float(y))
        return lr

    @given(_pts)
    @slow
    def test_slope_intercept_parity(self, points) -> None:
        j, p = self._fit_java(points), _fit(points)
        assert _close(j.getSlope(), p.getSlope(), 1e-7, rtol=1e-7)
        assert _close(j.getIntercept(), p.getIntercept(), 1e-7, rtol=1e-7)

    @given(_pts)
    @slow
    def test_r_squared_parity(self, points) -> None:
        j, p = self._fit_java(points), _fit(points)
        jr, pr = float(j.getRSquared()), p.getRSquared()
        if math.isnan(jr) or math.isnan(pr):
            return
        assert _close(jr, pr, 1e-7, rtol=1e-7)

    @given(_pts)
    @slow
    def test_chi_squared_parity(self, points) -> None:
        j, p = self._fit_java(points), _fit(points)
        assert _close(j.chiSquared(), p.chiSquared(), 1e-6, rtol=1e-7)

    @given(_pts)
    @slow
    def test_count_parity(self, points) -> None:
        j, p = self._fit_java(points), _fit(points)
        assert int(j.getCount()) == p.getCount()


if __name__ == "__main__":
    from _parity_lib import run_tests; run_tests(__file__)
