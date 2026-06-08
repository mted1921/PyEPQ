r"""
test_parity_simplex_ver1_1_0.py — parity harness for Simplex_ver1_1_0.py

Simplex is abstract (M4: JPype cannot extend Java abstract classes).
Part 1: analytical correctness via Python concrete subclasses.
Part 2: Java parity — @pytest.mark.skip(M4).
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, strategies as st

from _parity_lib import (
    setup_parity, needs_java, PARITY_ENABLED,
    TOL_EXACT, TOL_LITERAL, TOL_NR_LIB,
    finite, positive,
    slow,
    _close,
    _NAN, _INF,
)

from Simplex import Simplex as PySimplex
from _epq_compat import EPQException

ctx = setup_parity("gov.nist.microanalysis.Utility.Simplex")
JavaSimplex = ctx.java_class


# ---------------------------------------------------------------------------
# Concrete subclasses for testing
# ---------------------------------------------------------------------------

class _BowlSimplex(PySimplex):
    """f(x) = x[0]^2 + x[1]^2  — minimum 0.0 at origin."""
    def function(self, x):
        return x[0] ** 2 + x[1] ** 2


class _ShiftedBowlSimplex(PySimplex):
    """f(x) = (x[0]-3)^2 + (x[1]+2)^2  — minimum 0.0 at (3, -2)."""
    def function(self, x):
        return (x[0] - 3.0) ** 2 + (x[1] + 2.0) ** 2


class _ParameterizedSimplex(PySimplex):
    """Reads target coords from parameters."""
    def function(self, x):
        p = self.getParameters()
        return sum((x[i] - p[i]) ** 2 for i in range(len(x)))


class _SaturationSimplex(PySimplex):
    """Returns Double.MAX_VALUE for |x| > 10 — tests out-of-bounds handling."""
    def function(self, x):
        if any(abs(xi) > 10.0 for xi in x):
            return float("inf")
        return x[0] ** 2 + x[1] ** 2


# ---------------------------------------------------------------------------
# TestConstruction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_tolerance(self):
        s = _BowlSimplex()
        assert _close(s.getTolerance(), PySimplex.DEFAULT_TOLERANCE, TOL_EXACT)

    def test_default_max_evaluations(self):
        s = _BowlSimplex()
        assert s.getMaxEvaluations() == PySimplex.DEFAULT_EVALUATIONS

    def test_parameters_none_by_default(self):
        s = _BowlSimplex()
        assert s.getParameters() is None

    def test_parameters_stored(self):
        params = [3.0, -2.0]
        s = _ParameterizedSimplex(params)
        stored = s.getParameters()
        assert list(stored) == params

    def test_parameters_cloned(self):
        params = [1.0, 2.0]
        s = _ParameterizedSimplex(params)
        params[0] = 99.0
        assert s.getParameters()[0] == 1.0


# ---------------------------------------------------------------------------
# TestAccessors
# ---------------------------------------------------------------------------

class TestAccessors:
    def test_set_get_tolerance(self):
        s = _BowlSimplex()
        s.setTolerance(1e-4)
        assert _close(s.getTolerance(), 1e-4, TOL_LITERAL)

    def test_tolerance_negative_becomes_abs(self):
        s = _BowlSimplex()
        s.setTolerance(-1e-5)
        assert s.getTolerance() > 0.0

    def test_set_get_max_evaluations(self):
        s = _BowlSimplex()
        s.setMaxEvaluations(2000)
        assert s.getMaxEvaluations() == 2000

    def test_max_evaluations_floor(self):
        s = _BowlSimplex()
        s.setMaxEvaluations(5)
        assert s.getMaxEvaluations() >= 100


# ---------------------------------------------------------------------------
# TestStaticHelpers
# ---------------------------------------------------------------------------

class TestStaticHelpers:
    def test_regularized_shape(self):
        center = [0.0, 0.0]
        scale = [1.0, 1.0]
        pts = PySimplex.regularizedStartingPoints(center, scale)
        assert len(pts) == 3       # n+1 for n=2
        assert len(pts[0]) == 2

    def test_regularized_first_row_is_center(self):
        center = [5.0, -3.0]
        scale = [1.0, 2.0]
        pts = PySimplex.regularizedStartingPoints(center, scale)
        assert pts[0][0] == 5.0
        assert pts[0][1] == -3.0

    def test_regularized_offsets(self):
        center = [0.0, 0.0]
        scale = [2.0, 3.0]
        pts = PySimplex.regularizedStartingPoints(center, scale)
        assert pts[1][0] == 2.0
        assert pts[2][1] == 3.0

    def test_randomized_shape(self):
        center = [0.0, 0.0, 0.0]
        scale = [1.0, 1.0, 1.0]
        pts = PySimplex.randomizedStartingPoints(center, scale)
        assert len(pts) == 4
        assert len(pts[0]) == 3


# ---------------------------------------------------------------------------
# TestOptimization
# ---------------------------------------------------------------------------

class TestOptimization:
    def test_bowl_minimum_at_origin(self):
        s = _BowlSimplex()
        pts = PySimplex.regularizedStartingPoints([0.5, 0.5], [1.0, 1.0])
        result = s.perform(pts)
        assert abs(result[0]) < 1e-4
        assert abs(result[1]) < 1e-4

    def test_bowl_best_result_near_zero(self):
        s = _BowlSimplex()
        pts = PySimplex.regularizedStartingPoints([1.0, 1.0], [0.5, 0.5])
        s.perform(pts)
        assert s.getBestResult() < 1e-6

    def test_shifted_minimum(self):
        s = _ShiftedBowlSimplex()
        pts = PySimplex.regularizedStartingPoints([2.5, -1.5], [0.5, 0.5])
        result = s.perform(pts)
        assert abs(result[0] - 3.0) < 1e-4
        assert abs(result[1] + 2.0) < 1e-4

    def test_evaluation_count_positive(self):
        s = _BowlSimplex()
        pts = PySimplex.regularizedStartingPoints([1.0, 1.0], [0.5, 0.5])
        s.perform(pts)
        assert s.getEvaluationCount() > 0

    def test_parametrized_minimum(self):
        target = [7.0, -4.0]
        s = _ParameterizedSimplex(target)
        pts = PySimplex.regularizedStartingPoints([6.5, -3.5], [0.5, 0.5])
        result = s.perform(pts)
        assert abs(result[0] - 7.0) < 1e-3
        assert abs(result[1] + 4.0) < 1e-3


# ---------------------------------------------------------------------------
# TestExceptions
# ---------------------------------------------------------------------------

class TestExceptions:
    def test_max_evaluations_raises(self):
        # Bowl from [100, 100] with fatol/xatol=1e-12 needs thousands of evals to converge;
        # maxfev=100 (the enforced floor) is always exhausted first, triggering EPQException.
        s = _BowlSimplex()
        s.setMaxEvaluations(100)   # clamps to floor (100)
        s.setTolerance(1e-12)      # near-zero tolerance: scipy never stops early
        pts = PySimplex.regularizedStartingPoints([100.0, 100.0], [50.0, 50.0])
        with pytest.raises(EPQException):
            s.perform(pts)


# ---------------------------------------------------------------------------
# TestHypothesis
# ---------------------------------------------------------------------------

class TestHypothesis:
    @given(st.floats(-5.0, 5.0), st.floats(-5.0, 5.0))
    @slow
    def test_bowl_finds_finite_minimum(self, cx, cy):
        s = _BowlSimplex()
        pts = PySimplex.regularizedStartingPoints([cx, cy], [0.5, 0.5])
        try:
            result = s.perform(pts)
            assert math.isfinite(result[0])
            assert math.isfinite(result[1])
        except EPQException:
            pass  # max evals: acceptable for extreme starting points


# ---------------------------------------------------------------------------
# TestSimplexParity  (M4 — abstract class)
# ---------------------------------------------------------------------------

@pytest.mark.skip(
    reason="M4: JPype cannot extend Java abstract classes from Python. "
           "Simplex.function() is abstract; no Python callback into JVM possible. "
           "Correctness validated analytically in TestOptimization above."
)
class TestSimplexParity:
    def test_placeholder(self):
        pass

if __name__ == "__main__":
    from _parity_lib import run_tests; run_tests(__file__)
