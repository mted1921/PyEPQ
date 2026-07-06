r"""
test_parity_levenbergmarquardt2_ver2_1_1.py -- parity harness for LevenbergMarquardt2

Pre-written harness (Prompt 2): targets the port's expected API per
LevenbergMarquardt2_ver1_1_1.spec.md (repaired from the Java source).

LevenbergMarquardt2 is the non-linear least-squares engine. The fit model is a
`FitFunction` with Matrix-based `compute(params)` and `partials(params)`:
  compute(m x 1 params)  -> n x 1 model values
  partials(m x 1 params) -> n x m Jacobian
`compute(ff, yData, sigma, p0)` returns a `FitResult`.

M4: `FitFunction` / `AutoPartialsFitFunction` are abstract; JPype cannot subclass
them from Python, so there is no cross-engine Java parity (Part 2 is skipped).
Correctness is validated analytically: for a LINEAR model the least-squares
solution is unique and closed-form, so LM must recover it exactly.
"""
from __future__ import annotations

import math

import pytest
from hypothesis import given, strategies as st

from _parity_lib import needs_java, slow, _close

from LevenbergMarquardt2_ver2_1_0 import (
    LevenbergMarquardt2 as PyLM2,
    FitFunction as PyFitFunction,
    AutoPartialsFitFunction as PyAutoPartials,
)
from _epq_compat import JamaMatrix


# ---------------------------------------------------------------------------
# Concrete Python fit functions
# ---------------------------------------------------------------------------

class _LinearModel(PyFitFunction):
    """y = a + b*x. Jacobian columns: d/da = 1, d/db = x."""

    def __init__(self, xs) -> None:
        self._xs = [float(x) for x in xs]

    def compute(self, params: JamaMatrix) -> JamaMatrix:
        a = params.get(0, 0)
        b = params.get(1, 0)
        return JamaMatrix([[a + b * x] for x in self._xs])

    def partials(self, params: JamaMatrix) -> JamaMatrix:
        return JamaMatrix([[1.0, x] for x in self._xs])


class _ExpModelAuto(PyAutoPartials):
    """y = a*exp(b*x) with finite-difference Jacobian from AutoPartialsFitFunction."""

    def __init__(self, xs) -> None:
        super().__init__()
        self._xs = [float(x) for x in xs]

    def compute(self, params: JamaMatrix) -> JamaMatrix:
        a = params.get(0, 0)
        b = params.get(1, 0)
        return JamaMatrix([[a * math.exp(b * x)] for x in self._xs])


def _col(vals) -> JamaMatrix:
    return JamaMatrix([[float(v)] for v in vals])


def _fit_linear(a_true: float, b_true: float, xs):
    ys = [a_true + b_true * x for x in xs]
    lm = PyLM2()
    return lm.compute(_LinearModel(xs), _col(ys), _col([1.0] * len(xs)),
                      _col([0.0, 0.0]))


# ############################################################################
# PART 1 -- Analytical correctness
# ############################################################################


class TestConstruction:
    def test_default_max_iterations(self) -> None:
        assert PyLM2().getMaxIterations() == 100

    def test_set_max_iterations_roundtrip(self) -> None:
        lm = PyLM2()
        lm.setMaxIterations(250)
        assert lm.getMaxIterations() == 250


class TestLinearFit:
    """Exact linear data -> LM recovers the parameters exactly."""

    XS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    def test_recovers_intercept_and_slope(self) -> None:
        res = _fit_linear(2.0, 3.0, self.XS)
        params = res.getBestParameters()
        # Test bug fix: stopping criterion stop2 fires at ||deltaP||<=_mEps2*||p||=1e-6*3.6,
        # so component error can be ~3.6e-6; 1e-6 was too tight. Matches property-test 1e-5.
        assert _close(params[0], 2.0, 1e-5)
        assert _close(params[1], 3.0, 1e-5)

    def test_zero_chi_squared_on_exact_data(self) -> None:
        res = _fit_linear(1.0, -0.5, self.XS)
        assert res.getChiSquared() < 1e-6

    def test_best_fit_values_match_data(self) -> None:
        res = _fit_linear(2.0, 3.0, self.XS)
        fit = res.getBestFitValues()
        for x, f in zip(self.XS, fit):
            assert _close(f, 2.0 + 3.0 * x, 1e-5)

    def test_covariance_is_square_m_by_m(self) -> None:
        res = _fit_linear(2.0, 3.0, self.XS)
        cov = res.getCovariance()
        assert cov.getRowDimension() == 2
        assert cov.getColumnDimension() == 2

    def test_get_model_returns_engine(self) -> None:
        lm = PyLM2()
        res = lm.compute(_LinearModel(self.XS),
                         _col([2.0 + 3.0 * x for x in self.XS]),
                         _col([1.0] * len(self.XS)), _col([0.0, 0.0]))
        assert res.getModel() is lm

    def test_iteration_count_positive(self) -> None:
        res = _fit_linear(2.0, 3.0, self.XS)
        assert res.getIterationCount() >= 1

    @given(st.floats(-5.0, 5.0, allow_nan=False, allow_infinity=False),
           st.floats(-5.0, 5.0, allow_nan=False, allow_infinity=False))
    @slow
    def test_linear_fit_property(self, a_true: float, b_true: float) -> None:
        res = _fit_linear(a_true, b_true, self.XS)
        params = res.getBestParameters()
        assert _close(params[0], a_true, 1e-5, rtol=1e-5)
        assert _close(params[1], b_true, 1e-5, rtol=1e-5)


class TestAutoPartials:
    """AutoPartialsFitFunction's finite-difference Jacobian matches the analytic one."""

    XS = [0.0, 0.5, 1.0, 1.5, 2.0]

    def test_partials_shape(self) -> None:
        ff = _ExpModelAuto(self.XS)
        j = ff.partials(_col([2.0, 0.5]))
        assert j.getRowDimension() == len(self.XS)
        assert j.getColumnDimension() == 2

    def test_partials_approx_analytic(self) -> None:
        # y = a*exp(b*x): d/da = exp(b*x), d/db = a*x*exp(b*x).
        a, b = 2.0, 0.5
        ff = _ExpModelAuto(self.XS)
        j = ff.partials(_col([a, b]))
        for i, x in enumerate(self.XS):
            assert _close(j.get(i, 0), math.exp(b * x), 1e-4, rtol=1e-4)
            assert _close(j.get(i, 1), a * x * math.exp(b * x), 1e-4, rtol=1e-4)

    def test_auto_partials_fit_recovers_exp(self) -> None:
        a_true, b_true = 1.5, 0.3
        ys = [a_true * math.exp(b_true * x) for x in self.XS]
        lm = PyLM2()
        res = lm.compute(_ExpModelAuto(self.XS), _col(ys),
                         _col([1.0] * len(self.XS)), _col([1.0, 0.1]))
        params = res.getBestParameters()
        assert _close(params[0], a_true, 1e-3, rtol=1e-3)
        assert _close(params[1], b_true, 1e-3, rtol=1e-3)


# ############################################################################
# PART 2 -- Java parity (M4 — skipped)
# ############################################################################

@needs_java
@pytest.mark.skip(
    reason="M4: JPype cannot extend the Java abstract FitFunction / "
           "AutoPartialsFitFunction from Python (@JImplements is interfaces-only; "
           "direct subclassing raises TypeError). No Python fit model can be "
           "passed into the Java engine. Correctness is validated analytically "
           "against closed-form least-squares solutions in Part 1."
)
class TestLevenbergMarquardt2Parity:
    def test_placeholder(self) -> None:
        pass


class TestFitResultExtras:
    """Coverage for FitResult methods not exercised in TestLinearFit."""

    XS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    def test_get_best_parameters_u_length(self) -> None:
        res = _fit_linear(2.0, 3.0, self.XS)
        uparams = res.getBestParametersU()
        assert len(uparams) == 2

    def test_get_fit_function_identity(self) -> None:
        ff = _LinearModel(self.XS)
        lm = PyLM2()
        ys = _col([2.0 + 3.0 * x for x in self.XS])
        res = lm.compute(ff, ys, _col([1.0] * len(self.XS)), _col([0.0, 0.0]))
        assert res.getFitFunction() is ff

    def test_get_improve_count_nonneg(self) -> None:
        res = _fit_linear(2.0, 3.0, self.XS)
        assert res.getImproveCount() >= 0


class TestAddActionListener:
    XS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    def test_add_action_listener_fires(self) -> None:
        calls = []
        lm = PyLM2()
        lm.addActionListener(calls.append)
        ff = _LinearModel(self.XS)
        lm.compute(ff, _col([2.0 + 3.0 * x for x in self.XS]),
                   _col([1.0] * len(self.XS)), _col([0.0, 0.0]))
        assert len(calls) > 0


class TestSetDelta:
    XS = [0.0, 0.5, 1.0, 1.5, 2.0]

    def test_set_delta_overrides_step(self) -> None:
        ff = _ExpModelAuto(self.XS)
        ff.setDelta([1e-6, 1e-6])
        j = ff.partials(_col([2.0, 0.5]))
        assert j.getRowDimension() == len(self.XS)
        assert j.getColumnDimension() == 2


class TestGetIteration:
    """getIteration() is the engine's live iteration counter (distinct from FitResult.getIterationCount).

    After compute() returns, the counter holds the last iteration index.
    """

    XS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]

    def test_get_iteration_nonneg_after_fit(self) -> None:
        lm = PyLM2()
        res = lm.compute(
            _LinearModel(self.XS),
            _col([2.0 + 3.0 * x for x in self.XS]),
            _col([1.0] * len(self.XS)),
            _col([0.0, 0.0]),
        )
        assert lm.getIteration() >= 1

    def test_get_iteration_zero_before_fit(self) -> None:
        lm = PyLM2()
        assert lm.getIteration() == 0


if __name__ == "__main__":
    from _parity_lib import run_tests; run_tests(__file__)
