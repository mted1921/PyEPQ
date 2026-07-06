r"""
test_parity_levenbergmarquardtparameterized_ver2_1_0.py -- parity harness for LevenbergMarquardtParameterized

Pre-written harness (Prompt 2): targets the port's expected API per its spec.

LevenbergMarquardtParameterized tags fit parameters with `Parameter` tokens
(name + Constraint + default + isFit) and fits a `Function` (compute / derivative /
computeU over a Map<Parameter, value>) by delegating to
LevenbergMarquardtConstrained. Results are keyed by Parameter via
`ParameterizedFitResult`.

M4: the `Function` / `FunctionImpl` types are abstract; JPype cannot pass a Python
function model into the Java engine (Part 2 skipped). Correctness is validated with
a concrete Python FunctionImpl over a linear model with an exact solution.

Test-fix: imports updated from stale ver1_1_0 references to the correct ver2
port filenames (LevenbergMarquardtParameterized_ver2_1_0, Constraint_ver2_1_3,
UncertainValue2_ver2_1_0). No port code changed.
"""
from __future__ import annotations

import math

import pytest
from hypothesis import given, strategies as st

from _parity_lib import _close, needs_java, slow
from _epq_compat import JamaMatrix

from LevenbergMarquardtParameterized_ver2_1_0 import (  # test-fix: ver1_1_0 → ver2_1_0
    LevenbergMarquardtParameterized as PyLMP,
    Parameter as PyParameter,
    ParameterObject as PyParameterObject,
    FunctionImpl as PyFunctionImpl,
    InvertableFunction as PyInvertableFunction,
)
from Constraint_ver2_1_3 import Unconstrained as PyUnconstrained  # test-fix: ver1_1_0 → ver2_1_3
from UncertainValue2_ver2_1_0 import UncertainValue2 as PyUV2  # test-fix: ver1_1_0 → ver2_1_0


# ---------------------------------------------------------------------------
# Concrete Function: y = m*x + c
# ---------------------------------------------------------------------------

class _LinearFunction(PyFunctionImpl):
    def __init__(self, m_param, c_param) -> None:
        super().__init__()
        self._m = m_param
        self._c = c_param
        self.add(m_param)
        self.add(c_param)

    def compute(self, arg, param):
        return self._m.getValue(param) * arg + self._c.getValue(param)

    def derivative(self, arg, param, idx):
        if idx == self._m:
            return arg
        if idx == self._c:
            return 1.0
        return 0.0

    def computeU(self, arg, param):
        nominal = {p: param[p].doubleValue() for p in param}
        return PyUV2(self.compute(arg, nominal), 0.0)


# ---------------------------------------------------------------------------
# Concrete InvertableFunction: identity (f(x)=x, f⁻¹(x)=x)
# ---------------------------------------------------------------------------

class _IdentityInvertable(PyInvertableFunction):
    """Minimal concrete InvertableFunction for coverage testing."""

    def isFitParameter(self, idx) -> bool:
        return False

    def getParameters(self, all) -> set:
        return set()

    def compute(self, arg, param) -> float:
        return float(arg)

    def derivative(self, arg, param, idx) -> float:
        return 1.0

    def computeU(self, arg, param) -> PyUV2:
        return PyUV2(float(arg), 0.0)

    def inverse(self, arg, param) -> PyUV2:
        return PyUV2(float(arg), 0.0)


_XS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


# ############################################################################
# PART 1 -- Parameter / ParameterObject / FunctionImpl
# ############################################################################


class TestParameter:
    def test_name_default_isfit(self) -> None:
        p = PyParameter("m", 1.5, True)
        assert p.getName() == "m"
        assert p.getDefaultValue() == 1.5
        assert p.isFit() is True

    def test_fixed_parameter_uses_default(self) -> None:
        p = PyParameter("k", 5.0, False)
        assert p.getValue({}) == 5.0          # not fit -> default, map ignored

    def test_fit_parameter_uses_map(self) -> None:
        p = PyParameter("m", 0.0, True)
        assert p.getValue({p: 3.0}) == 3.0

    def test_equality_by_name(self) -> None:
        assert PyParameter("a", 1.0, True) == PyParameter("a", 9.0, False)
        assert PyParameter("a", 1.0, True) != PyParameter("b", 1.0, True)

    def test_hash_by_name(self) -> None:
        assert hash(PyParameter("a", 1.0, True)) == hash(PyParameter("a", 2.0, False))

    def test_set_default_value(self) -> None:
        p = PyParameter("m", 1.0, True)
        p.setDefaultValue(7.0)
        assert p.getDefaultValue() == 7.0

    def test_constraint_default_unconstrained(self) -> None:
        p = PyParameter("m", 1.0, True)
        assert isinstance(p.getConstraint(), PyUnconstrained)

    def test_set_constraint(self) -> None:
        p = PyParameter("m", 1.0, True)
        new_c = PyUnconstrained()
        p.setConstraint(new_c)
        assert p.getConstraint() is new_c

    def test_set_is_fit(self) -> None:
        p = PyParameter("m", 1.0, True)
        p.setIsFit(False)
        assert p.isFit() is False

    def test_get_uncertain_value_fit(self) -> None:
        p = PyParameter("m", 0.0, True)
        uv_in = PyUV2(2.5, 0.1)
        uv = p.getUncertainValue({p: uv_in})
        assert _close(uv.doubleValue(), 2.5, 1e-9)

    def test_get_uncertain_value_fixed(self) -> None:
        p = PyParameter("k", 7.0, False)
        uv = p.getUncertainValue({})
        assert _close(uv.doubleValue(), 7.0, 1e-9)


class TestParameterObject:
    def test_carries_object(self) -> None:
        tag = ("Fe", "Ka")
        p = PyParameterObject("line", PyUnconstrained(), 1.0, True, tag)
        assert p.getObject() == tag
        assert p.getName() == "line"


class TestFunctionImpl:
    def test_get_parameters_all_vs_fit(self) -> None:
        m = PyParameter("m", 0.0, True)
        c = PyParameter("c", 1.0, False)     # fixed
        f = _LinearFunction(m, c)
        assert len(f.getParameters(True)) == 2     # all
        assert len(f.getParameters(False)) == 1    # fit only (m)

    def test_is_fit_parameter(self) -> None:
        m = PyParameter("m", 0.0, True)
        c = PyParameter("c", 1.0, True)
        f = _LinearFunction(m, c)
        assert f.isFitParameter(m)
        assert f.isFitParameter(c)

    def test_extract_keeps_only_fit(self) -> None:
        m = PyParameter("m", 1.0, True)
        c = PyParameter("c", 0.0, False)
        f = _LinearFunction(m, c)
        full = {m: PyUV2(3.0, 0.1), c: PyUV2(1.0, 0.0)}
        extracted = f.extract(full)
        assert m in extracted
        assert c not in extracted


# ############################################################################
# PART 1b -- InvertableFunction
# ############################################################################


class TestInvertableFunction:
    def test_inverse_identity(self) -> None:
        f = _IdentityInvertable()
        result = f.inverse(3.0, {})
        assert _close(result.doubleValue(), 3.0, 1e-9)


# ############################################################################
# PART 1c -- _ParameterizedFitFunction internals (paramSize / partials)
# ############################################################################


class TestParameterizedFitFunctionInternals:
    """Exercises paramSize and partials on the private _ParameterizedFitFunction
    inner class, which the compliance checker flags as public Java API."""

    def _make_pff(self, m_val: float = 1.0, c_val: float = 0.0):
        m = PyParameter("m", m_val, True)
        c = PyParameter("c", c_val, True)
        f = _LinearFunction(m, c)
        return PyLMP._ParameterizedFitFunction(list(_XS), f)

    def test_param_size(self) -> None:
        pff = self._make_pff()
        assert pff.paramSize() == 2

    def test_partials_shape(self) -> None:
        pff = self._make_pff(m_val=1.0, c_val=0.0)
        p0 = JamaMatrix([[1.0], [0.0]])
        J = pff.partials(p0)
        assert J.getRowDimension() == len(_XS)
        assert J.getColumnDimension() == 2


# ############################################################################
# PART 1d -- addActionListener
# ############################################################################


class TestAddActionListener:
    def test_add_action_listener_no_error(self) -> None:
        lmp = PyLMP()
        lmp.addActionListener(lambda n: None)

    def test_listener_invoked_during_fit(self) -> None:
        calls: list = []
        m = PyParameter("m", 0.0, True)
        c = PyParameter("c", 0.0, True)
        f = _LinearFunction(m, c)
        ys = [3.0 * x + 2.0 for x in _XS]
        lmp = PyLMP()
        lmp.addActionListener(lambda n: calls.append(n))
        lmp.compute(f, list(_XS), ys, [1.0] * len(_XS))
        assert len(calls) > 0


# ############################################################################
# PART 2 -- End-to-end parameterized fit (analytical)
# ############################################################################


class TestParameterizedFit:
    """Fit y = m*x + c to exact linear data; recover m and c by Parameter handle."""

    def _fit(self, m_true: float, c_true: float):
        m = PyParameter("m", 0.0, True)
        c = PyParameter("c", 0.0, True)
        f = _LinearFunction(m, c)
        ys = [m_true * x + c_true for x in _XS]
        res = PyLMP().compute(f, list(_XS), ys, [1.0] * len(_XS))
        return res, m, c

    def test_recovers_parameters(self) -> None:
        res, m, c = self._fit(3.0, 2.0)
        assert _close(res.getBestFit(m), 3.0, 1e-4, rtol=1e-4)
        assert _close(res.getBestFit(c), 2.0, 1e-4, rtol=1e-4)

    def test_best_fit_value_is_uncertain_value2(self) -> None:
        res, m, _c = self._fit(3.0, 2.0)
        uv = res.getBestFitValue(m)
        assert isinstance(uv, PyUV2)
        assert _close(uv.doubleValue(), 3.0, 1e-4, rtol=1e-4)

    def test_get_results_map(self) -> None:
        res, m, c = self._fit(3.0, 2.0)
        results = res.getResults()
        assert _close(results[m], 3.0, 1e-4, rtol=1e-4)
        assert _close(results[c], 2.0, 1e-4, rtol=1e-4)

    def test_parameter_map_keys(self) -> None:
        res, m, c = self._fit(3.0, 2.0)
        pm = res.getParameterMap()
        assert m in pm and c in pm

    def test_index_of(self) -> None:
        res, m, c = self._fit(3.0, 2.0)
        assert res.indexOf(m) in (0, 1)
        assert res.indexOf(m) != res.indexOf(c)

    def test_tabulate_is_string(self) -> None:
        res, _m, _c = self._fit(3.0, 2.0)
        assert isinstance(res.tabulate(), str)

    def test_get_parameters_by_class(self) -> None:
        res, _m, _c = self._fit(3.0, 2.0)
        params = res.getParametersByClass(PyParameter)
        assert len(params) == 2
        assert all(type(p) == PyParameter for p in params)

    def test_get_parameter_by_class(self) -> None:
        res, _m, _c = self._fit(3.0, 2.0)
        p = res.getParameterByClass(PyParameter)
        assert isinstance(p, PyParameter)

    def test_get_parameter_by_class_missing(self) -> None:
        res, _m, _c = self._fit(3.0, 2.0)
        p = res.getParameterByClass(PyParameterObject)
        assert p is None


# ############################################################################
# PART 2b -- Property-based tests
# ############################################################################


class TestParameterProperty:
    """Property: Parameter round-trips default values and names exactly."""

    @given(st.floats(-1e9, 1e9, allow_nan=False, allow_infinity=False))
    @slow
    def test_default_value_roundtrip(self, v: float) -> None:
        p = PyParameter("x", v, False)
        assert p.getDefaultValue() == v

    @given(
        st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False),
        st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False),
    )
    @slow
    def test_linear_fit_recovers_params(self, m_true: float, c_true: float) -> None:
        m = PyParameter("m", 0.0, True)
        c = PyParameter("c", 0.0, True)
        f = _LinearFunction(m, c)
        ys = [m_true * x + c_true for x in _XS]
        res = PyLMP().compute(f, list(_XS), ys, [1.0] * len(_XS))
        assert _close(res.getBestFit(m), m_true, 0.1, rtol=0.01)
        assert _close(res.getBestFit(c), c_true, 0.1, rtol=0.01)


# ############################################################################
# PART 3 -- Java parity (M4 — skipped)
# ############################################################################

# LevenbergMarquardtParameterized is a concrete class: @needs_java gate required.
# The inner Function / FunctionImpl are abstract, so a Python function model
# cannot be passed into the Java engine (M4-like limitation on integration tests).

@needs_java
class TestLevenbergMarquardtParameterizedParity:
    """Concrete class: needs_java gate is required. Integration blocked by M4
    (Function / FunctionImpl are abstract Java types; JPype cannot pass a Python
    model into the Java engine). Correctness is validated analytically in Part 2."""

    def test_placeholder(self) -> None:
        pytest.skip(
            "M4: Function / FunctionImpl are abstract Java types; JPype cannot pass "
            "a Python function model into the Java engine. The parameterized fit is "
            "validated analytically against an exact linear solution in Part 2."
        )


if __name__ == "__main__":
    from _parity_lib import run_tests; run_tests(__file__)
