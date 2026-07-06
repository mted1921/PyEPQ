r"""
test_parity_uncertainvalue2_ver2_1_1.py -- parity harness for UncertainValue2

Revision ver2_1_1 (2026-06-30): extended coverage to the full public API documented
in its spec. The gen1 harness (ver2_1_0) exercised only the arithmetic core
(add/subtract/multiply/divide/invert/negate/sqr/exp/log/sqrt/pow), the primary
Number/comparison/component surface, and the two preserved bugs. This revision adds
deterministic (always-on) coverage for the remaining specified public methods:
  * factories:      asUncertainValue2, valueOf, createGaussian
  * aggregations:   mean_collection/mean_array, weightedMean, safeWeightedMean, normalize
  * trig/quadratic: atan, atan2, quadratic, nonNegative
  * abs (instance + static abs_n)
  * Number widths:  longValue, byteValue, shortValue
  * comparisons:    lessThanOrEqual, greaterThanOrEqual
  * components:     assignComponents, getComponents, getFractional, renameComponent, reduced
  * fractional:     fractionalUncertainty (instance), fractionalUncertaintyU
  * formatting:     format (instance + static format_n), format_src, formatLong,
                    formatComponent_nf, toLongString
plus Java-oracle parity for atan and nonNegative. The import still targets the
unchanged port UncertainValue2_ver2_1_0 (test logic changed, port did not).

Pre-written harness (Prompt 2): targets the port's expected API per its spec.

UncertainValue2 is a concrete `Number` carrying a value plus a map of named
uncertainty components (source -> sigma). Arithmetic propagates the components;
independent single-source operands combine in quadrature.

Test strategy
-------------
* PART 1 asserts only deterministic, certain facts: constants, nominal (value)
  results of each operation, the stored sigma for the (v, dv) constructor, the
  Number interface, comparison, and component bookkeeping. The two preserved bugs
  are checked with well-defined inputs.
* PART 1b (this revision) extends deterministic coverage to the rest of the
  specified public surface, choosing inputs with exact closed-form results.
* PART 2 is the numerical oracle: it compares value AND uncertainty against Java
  for the arithmetic and special functions. Operands are built with EXPLICIT
  source names so the Java and Python component maps line up (the auto-source
  counter would otherwise diverge between the two runtimes).

Preserved Java bugs (per spec):
  JAVA-BUG-1  uncertainty(Collection<String>) sums the raw component sigmas
              instead of their squares, then sqrt()s -> sqrt(Sigma sigma_i),
              not sqrt(Sigma sigma_i^2). Strict companion squares correctly.
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

from UncertainValue2_ver2_1_0 import UncertainValue2 as PyUV2

ctx = setup_parity("gov.nist.microanalysis.Utility.UncertainValue2")
JavaUV2 = ctx.java_class

_TOL = 1e-12
_val = st.floats(min_value=0.5, max_value=100.0, allow_nan=False, allow_infinity=False)
_sig = st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False)


def _fmt2(x: float) -> str:
    """A NumberFormat stand-in (Callable[[float], str]) with two decimal places."""
    return f"{x:.2f}"


# ############################################################################
# PART 1 -- Deterministic facts
# ############################################################################


class TestConstants:
    def test_one(self) -> None:
        assert _close(PyUV2.ONE.doubleValue(), 1.0, _TOL)

    def test_zero(self) -> None:
        assert _close(PyUV2.ZERO.doubleValue(), 0.0, _TOL)

    def test_nan(self) -> None:
        assert PyUV2.NaN.isNaN()

    def test_positive_infinity(self) -> None:
        assert math.isinf(PyUV2.POSITIVE_INFINITY.doubleValue())
        assert PyUV2.POSITIVE_INFINITY.doubleValue() > 0

    def test_negative_infinity(self) -> None:
        assert PyUV2.NEGATIVE_INFINITY.doubleValue() < 0


class TestConstruction:
    def test_value_and_sigma(self) -> None:
        uv = PyUV2(2.0, 0.3)
        assert _close(uv.doubleValue(), 2.0, _TOL)
        assert _close(uv.uncertainty(), 0.3, _TOL)

    def test_value_only_zero_uncertainty(self) -> None:
        uv = PyUV2(5.0)
        assert _close(uv.doubleValue(), 5.0, _TOL)
        assert _close(uv.uncertainty(), 0.0, _TOL)

    def test_named_source(self) -> None:
        uv = PyUV2(5.0, "src", 0.2)
        assert _close(uv.getComponent("src"), 0.2, _TOL)
        assert uv.hasComponent("src")

    def test_variance_is_sigma_squared(self) -> None:
        assert _close(PyUV2(2.0, 0.3).variance(), 0.09, 1e-12)

    def test_is_uncertain(self) -> None:
        assert PyUV2(2.0, 0.3).isUncertain()
        assert not PyUV2(2.0, 0.0).isUncertain()


class TestArithmeticValues:
    """Nominal (value) results are exact and runtime-independent."""

    def test_add(self) -> None:
        assert _close(PyUV2.add_nn(PyUV2(2.0, 0.1), PyUV2(3.0, 0.2)).doubleValue(),
                      5.0, _TOL)

    def test_subtract(self) -> None:
        assert _close(PyUV2.subtract(PyUV2(5.0, 0.1), PyUV2(2.0, 0.2)).doubleValue(),
                      3.0, _TOL)

    def test_multiply(self) -> None:
        assert _close(PyUV2.multiply_nn(PyUV2(2.0, 0.1), PyUV2(3.0, 0.2)).doubleValue(),
                      6.0, _TOL)

    def test_divide(self) -> None:
        assert _close(PyUV2.divide_nn(PyUV2(6.0, 0.1), PyUV2(2.0, 0.2)).doubleValue(),
                      3.0, _TOL)

    def test_invert(self) -> None:
        assert _close(PyUV2.invert(PyUV2(4.0, 0.1)).doubleValue(), 0.25, _TOL)

    def test_negate(self) -> None:
        assert _close(PyUV2.negate(PyUV2(5.0, 0.1)).doubleValue(), -5.0, _TOL)

    def test_sqr(self) -> None:
        assert _close(PyUV2.sqr(PyUV2(3.0, 0.1)).doubleValue(), 9.0, _TOL)


class TestSpecialFunctionValues:
    def test_exp(self) -> None:
        assert _close(PyUV2.exp(PyUV2(0.0, 0.1)).doubleValue(), 1.0, _TOL)

    def test_log(self) -> None:
        assert _close(PyUV2.log(PyUV2(math.e, 0.1)).doubleValue(), 1.0, 1e-12)

    def test_sqrt(self) -> None:
        assert _close(PyUV2.sqrt_n(PyUV2(4.0, 0.1)).doubleValue(), 2.0, _TOL)

    def test_pow(self) -> None:
        assert _close(PyUV2.pow(PyUV2(2.0, 0.1), 3.0).doubleValue(), 8.0, _TOL)


class TestUncertaintyPropagation:
    """Independent single-source operands combine in quadrature."""

    def test_add_quadrature(self) -> None:
        r = PyUV2.add_nn(PyUV2(2.0, 0.3), PyUV2(3.0, 0.4))
        assert _close(r.uncertainty(), 0.5, 1e-9)   # sqrt(0.3^2 + 0.4^2)

    def test_subtract_quadrature(self) -> None:
        r = PyUV2.subtract(PyUV2(5.0, 0.3), PyUV2(2.0, 0.4))
        assert _close(r.uncertainty(), 0.5, 1e-9)


class TestNumberInterface:
    def test_double_value(self) -> None:
        assert _close(PyUV2(3.5, 0.1).doubleValue(), 3.5, _TOL)

    def test_int_value_truncates(self) -> None:
        assert PyUV2(3.9, 0.1).intValue() == 3

    def test_float_value(self) -> None:
        assert _close(PyUV2(3.5, 0.1).floatValue(), 3.5, 1e-6)


class TestComparison:
    def test_compare_to_less(self) -> None:
        assert PyUV2(1.0, 0.1).compareTo(PyUV2(2.0, 0.1)) < 0

    def test_compare_to_greater(self) -> None:
        assert PyUV2(3.0, 0.1).compareTo(PyUV2(2.0, 0.1)) > 0

    def test_clone_equals_original(self) -> None:
        a = PyUV2(1.0, 0.1)
        b = a.clone()
        assert a.equals(b)

    def test_not_equal_different_value(self) -> None:
        assert not PyUV2(1.0, 0.1).clone().equals(PyUV2(2.0, 0.1))

    def test_less_than_helper(self) -> None:
        assert PyUV2(1.0, 0.1).lessThan(PyUV2(2.0, 0.1))
        assert PyUV2(2.0, 0.1).greaterThan(PyUV2(1.0, 0.1))


class TestComponents:
    def test_assign_and_get_component(self) -> None:
        uv = PyUV2(5.0)
        uv.assignComponent("a", 0.3)
        assert _close(uv.getComponent("a"), 0.3, _TOL)

    def test_has_component(self) -> None:
        uv = PyUV2(5.0, "src", 0.2)
        assert uv.hasComponent("src")
        assert not uv.hasComponent("absent")

    def test_get_component_absent_is_zero(self) -> None:
        assert _close(PyUV2(5.0, "src", 0.2).getComponent("absent"), 0.0, _TOL)

    def test_component_names(self) -> None:
        uv = PyUV2(5.0)
        uv.assignComponent("a", 0.3)
        uv.assignComponent("b", 0.4)
        assert "a" in uv.getComponentNames()
        assert "b" in uv.getComponentNames()


# ############################################################################
# Bug-aware tests
# ############################################################################


class TestPreservedBugs:
    def test_uncertainty_comps_sums_unsquared(self) -> None:
        # JAVA-BUG-1: sum2 += getComponent(comp) (raw), then sqrt -> sqrt(3 + 4).
        uv = PyUV2(5.0)
        uv.assignComponent("a", 3.0)
        uv.assignComponent("b", 4.0)
        assert _close(uv.uncertainty_comps(["a", "b"]), math.sqrt(7.0), 1e-12)


class TestStrictVariants:
    def test_uncertainty_comps_strict_sums_squares(self) -> None:
        uv = PyUV2(5.0)
        uv.assignComponent("a", 3.0)
        uv.assignComponent("b", 4.0)
        # sqrt(3^2 + 4^2) = 5.
        assert _close(uv.uncertainty_comps_strict(["a", "b"]), 5.0, 1e-12)


# ############################################################################
# PART 1b -- Extended API coverage (deterministic, per spec)
# ############################################################################


class TestFactories:
    def test_as_uncertain_value2_wraps_number(self) -> None:
        assert _close(PyUV2.asUncertainValue2(3.0).doubleValue(), 3.0, _TOL)

    def test_as_uncertain_value2_is_identity_for_uv2(self) -> None:
        uv = PyUV2(2.0, 0.1)
        assert PyUV2.asUncertainValue2(uv) is uv

    def test_value_of(self) -> None:
        assert _close(PyUV2.valueOf(5.0).doubleValue(), 5.0, _TOL)

    def test_create_gaussian(self) -> None:
        # createGaussian(v, src) -> value v, sigma sqrt(v) under source src.
        g = PyUV2.createGaussian(4.0, "cts")
        assert _close(g.doubleValue(), 4.0, _TOL)
        assert _close(g.getComponent("cts"), 2.0, _TOL)
        assert _close(g.uncertainty(), 2.0, _TOL)


class TestAggregations:
    def test_mean_collection(self) -> None:
        assert _close(
            PyUV2.mean_collection([PyUV2(2.0), PyUV2(4.0)]).doubleValue(), 3.0, 1e-12)

    def test_mean_array(self) -> None:
        assert _close(
            PyUV2.mean_array([PyUV2(1.0), PyUV2(2.0), PyUV2(3.0)]).doubleValue(),
            2.0, 1e-12)

    def test_weighted_mean_equal_weights(self) -> None:
        # Equal variances -> weighted mean equals the arithmetic mean.
        r = PyUV2.weightedMean([PyUV2(2.0, 0.1), PyUV2(4.0, 0.1)])
        assert _close(r.doubleValue(), 3.0, 1e-9)

    def test_safe_weighted_mean(self) -> None:
        r = PyUV2.safeWeightedMean([PyUV2(2.0, 0.1), PyUV2(4.0, 0.1)])
        assert _close(r.doubleValue(), 3.0, 1e-9)

    def test_normalize_sums_to_one(self) -> None:
        # normalize divides each value by the sum of all values.
        out = PyUV2.normalize([PyUV2(1.0), PyUV2(1.0), PyUV2(2.0)])
        vals = [u.doubleValue() for u in out]
        assert _close(vals[0], 0.25, 1e-12)
        assert _close(vals[1], 0.25, 1e-12)
        assert _close(vals[2], 0.5, 1e-12)
        assert _close(sum(vals), 1.0, 1e-12)


class TestTrigAndQuadratic:
    def test_atan(self) -> None:
        assert _close(PyUV2.atan(PyUV2(1.0)).doubleValue(), math.atan(1.0), 1e-12)

    def test_atan2(self) -> None:
        assert _close(
            PyUV2.atan2(PyUV2(1.0), PyUV2(1.0)).doubleValue(),
            math.atan2(1.0, 1.0), 1e-12)

    def test_quadratic_roots(self) -> None:
        # x^2 - 3x + 2 = 0  ->  roots {1, 2}
        roots = PyUV2.quadratic(PyUV2(1.0), PyUV2(-3.0), PyUV2(2.0))
        assert roots is not None
        vals = sorted(r.doubleValue() for r in roots)
        assert _close(vals[0], 1.0, 1e-12)
        assert _close(vals[1], 2.0, 1e-12)

    def test_quadratic_no_real_roots(self) -> None:
        # x^2 + 1 = 0  ->  negative discriminant -> None
        assert PyUV2.quadratic(PyUV2(1.0), PyUV2(0.0), PyUV2(1.0)) is None

    def test_non_negative_clamps_negative(self) -> None:
        assert _close(PyUV2.nonNegative(PyUV2(-3.0, "s", 0.1)).doubleValue(), 0.0, _TOL)

    def test_non_negative_passes_positive(self) -> None:
        assert _close(PyUV2.nonNegative(PyUV2(3.0, "s", 0.1)).doubleValue(), 3.0, _TOL)


class TestAbs:
    def test_abs_instance_negates_negative(self) -> None:
        assert _close(PyUV2(-5.0, "s", 0.1).abs().doubleValue(), 5.0, _TOL)

    def test_abs_static(self) -> None:
        assert _close(PyUV2.abs_n(PyUV2(-5.0, "s", 0.1)).doubleValue(), 5.0, _TOL)


class TestNumberWidths:
    def test_long_value_truncates(self) -> None:
        assert PyUV2(3.9, 0.1).longValue() == 3

    def test_byte_value(self) -> None:
        assert PyUV2(3.9, 0.1).byteValue() == 3

    def test_short_value(self) -> None:
        assert PyUV2(3.9, 0.1).shortValue() == 3


class TestOrEqualComparisons:
    def test_less_than_or_equal(self) -> None:
        assert PyUV2(1.0, 0.1).lessThanOrEqual(PyUV2(1.0, 0.1))
        assert PyUV2(1.0, 0.1).lessThanOrEqual(PyUV2(2.0, 0.1))
        assert not PyUV2(2.0, 0.1).lessThanOrEqual(PyUV2(1.0, 0.1))

    def test_greater_than_or_equal(self) -> None:
        assert PyUV2(1.0, 0.1).greaterThanOrEqual(PyUV2(1.0, 0.1))
        assert PyUV2(2.0, 0.1).greaterThanOrEqual(PyUV2(1.0, 0.1))
        assert not PyUV2(1.0, 0.1).greaterThanOrEqual(PyUV2(2.0, 0.1))


class TestComponentAccessors:
    def test_assign_components_bulk(self) -> None:
        uv = PyUV2(5.0)
        uv.assignComponents({"a": 0.3, "b": 0.4})
        assert _close(uv.getComponent("a"), 0.3, _TOL)
        assert _close(uv.getComponent("b"), 0.4, _TOL)

    def test_get_components_returns_mapping(self) -> None:
        comps = PyUV2(2.0, "s", 0.5).getComponents()
        assert _close(comps["s"], 0.5, _TOL)

    def test_get_fractional(self) -> None:
        # getFractional = component / value = 0.5 / 2.0
        assert _close(PyUV2(2.0, "s", 0.5).getFractional("s"), 0.25, _TOL)

    def test_rename_component(self) -> None:
        uv = PyUV2(2.0, "s", 0.5)
        uv.renameComponent("s", "t")
        assert uv.hasComponent("t")
        assert not uv.hasComponent("s")
        assert _close(uv.getComponent("t"), 0.5, _TOL)

    def test_reduced_collapses_to_single_source(self) -> None:
        r = PyUV2(2.0, "s", 0.3).reduced("total")
        assert _close(r.doubleValue(), 2.0, _TOL)
        assert _close(r.getComponent("total"), 0.3, _TOL)


class TestFractionalUncertainty:
    def test_fractional_uncertainty(self) -> None:
        # |uncertainty / value| = 0.5 / 2.0
        assert _close(PyUV2(2.0, "s", 0.5).fractionalUncertainty(), 0.25, _TOL)

    def test_fractional_uncertainty_u(self) -> None:
        u = PyUV2(2.0, "s", 0.5).fractionalUncertaintyU()
        assert _close(u.doubleValue(), 1.0, _TOL)
        assert _close(u.uncertainty(), 0.25, 1e-12)


class TestFormatting:
    def test_format_instance_certain(self) -> None:
        assert PyUV2(2.0).format(_fmt2) == "2.00"

    def test_format_instance_uncertain(self) -> None:
        assert PyUV2(2.0, "s", 0.5).format(_fmt2) == "2.00±0.50"

    def test_format_static(self) -> None:
        assert PyUV2.format_n(_fmt2, PyUV2(2.0)) == "2.00"

    def test_format_src(self) -> None:
        assert PyUV2(2.0, "s", 0.5).format_src("s", _fmt2) == "U(s)=0.50"

    def test_format_long(self) -> None:
        assert PyUV2(2.0, "s", 0.5).formatLong(_fmt2) == "2.00±0.50(s)"

    def test_format_component_nf(self) -> None:
        assert PyUV2(2.0, "s", 0.5).formatComponent_nf("s", _fmt2) == "0.50(s)"

    def test_to_long_string_labels_component(self) -> None:
        s = PyUV2(2.0, "s", 0.5).toLongString()
        assert "(s)" in s


# ############################################################################
# PART 2 -- Parity tests (value + uncertainty), Java as oracle
# ############################################################################
# Operands use explicit source names so Java and Python component maps match.


@needs_java
class TestUncertainValue2Parity:

    def _pair(self, v: float, src: str, dv: float):
        return PyUV2(v, src, dv), JavaUV2(float(v), src, float(dv))

    @given(_val, _sig, _val, _sig)
    @slow
    def test_add_parity(self, va, da, vb, db) -> None:
        pa, ja = self._pair(va, "A", da)
        pb, jb = self._pair(vb, "B", db)
        pr, jr = PyUV2.add_nn(pa, pb), JavaUV2.add(ja, jb)
        assert _close(jr.doubleValue(), pr.doubleValue(), 1e-9, rtol=1e-9)
        assert _close(jr.uncertainty(), pr.uncertainty(), 1e-9, rtol=1e-9)

    @given(_val, _sig, _val, _sig)
    @slow
    def test_subtract_parity(self, va, da, vb, db) -> None:
        pa, ja = self._pair(va, "A", da)
        pb, jb = self._pair(vb, "B", db)
        pr, jr = PyUV2.subtract(pa, pb), JavaUV2.subtract(ja, jb)
        assert _close(jr.doubleValue(), pr.doubleValue(), 1e-9, rtol=1e-9)
        assert _close(jr.uncertainty(), pr.uncertainty(), 1e-9, rtol=1e-9)

    @given(_val, _sig, _val, _sig)
    @slow
    def test_multiply_parity(self, va, da, vb, db) -> None:
        pa, ja = self._pair(va, "A", da)
        pb, jb = self._pair(vb, "B", db)
        pr, jr = PyUV2.multiply_nn(pa, pb), JavaUV2.multiply(ja, jb)
        assert _close(jr.doubleValue(), pr.doubleValue(), 1e-9, rtol=1e-9)
        assert _close(jr.uncertainty(), pr.uncertainty(), 1e-7, rtol=1e-7)

    @given(_val, _sig, _val, _sig)
    @slow
    def test_divide_parity(self, va, da, vb, db) -> None:
        pa, ja = self._pair(va, "A", da)
        pb, jb = self._pair(vb, "B", db)
        pr, jr = PyUV2.divide_nn(pa, pb), JavaUV2.divide(ja, jb)
        assert _close(jr.doubleValue(), pr.doubleValue(), 1e-9, rtol=1e-9)
        assert _close(jr.uncertainty(), pr.uncertainty(), 1e-7, rtol=1e-7)

    @given(_val, _sig)
    @slow
    def test_exp_parity(self, v, dv) -> None:
        p, j = self._pair(v, "A", dv)
        pr, jr = PyUV2.exp(p), JavaUV2.exp(j)
        assert _close(jr.doubleValue(), pr.doubleValue(), 1e-7, rtol=1e-7)
        assert _close(jr.uncertainty(), pr.uncertainty(), 1e-7, rtol=1e-7)

    @given(_val, _sig)
    @slow
    def test_log_parity(self, v, dv) -> None:
        p, j = self._pair(v, "A", dv)
        pr, jr = PyUV2.log(p), JavaUV2.log(j)
        assert _close(jr.doubleValue(), pr.doubleValue(), 1e-9, rtol=1e-9)
        assert _close(jr.uncertainty(), pr.uncertainty(), 1e-7, rtol=1e-7)

    @given(_val, _sig)
    @slow
    def test_sqrt_parity(self, v, dv) -> None:
        p, j = self._pair(v, "A", dv)
        pr, jr = PyUV2.sqrt_n(p), JavaUV2.sqrt(j)
        assert _close(jr.doubleValue(), pr.doubleValue(), 1e-9, rtol=1e-9)
        assert _close(jr.uncertainty(), pr.uncertainty(), 1e-7, rtol=1e-7)

    @given(_val, _sig, st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False))
    @slow
    def test_pow_parity(self, v, dv, n) -> None:
        p, j = self._pair(v, "A", dv)
        pr, jr = PyUV2.pow(p, float(n)), JavaUV2.pow(j, float(n))
        assert _close(jr.doubleValue(), pr.doubleValue(), 1e-7, rtol=1e-7)
        assert _close(jr.uncertainty(), pr.uncertainty(), 1e-6, rtol=1e-6)


@needs_java
class TestUncertainValue2ExtendedParity:
    """Java-oracle parity for the functions added in revision ver2_1_1."""

    def _pair(self, v: float, src: str, dv: float):
        return PyUV2(v, src, dv), JavaUV2(float(v), src, float(dv))

    @given(_val, _sig)
    @slow
    def test_atan_parity(self, v, dv) -> None:
        p, j = self._pair(v, "A", dv)
        pr, jr = PyUV2.atan(p), JavaUV2.atan(j)
        assert _close(jr.doubleValue(), pr.doubleValue(), 1e-9, rtol=1e-9)
        assert _close(jr.uncertainty(), pr.uncertainty(), 1e-7, rtol=1e-7)

    @given(_val, _sig)
    @slow
    def test_non_negative_parity(self, v, dv) -> None:
        # _val is strictly positive, so this exercises the pass-through branch;
        # the clamp branch is covered deterministically in TestTrigAndQuadratic.
        p, j = self._pair(v, "A", dv)
        pr, jr = PyUV2.nonNegative(p), JavaUV2.nonNegative(j)
        assert _close(jr.doubleValue(), pr.doubleValue(), 1e-9, rtol=1e-9)
        assert _close(jr.uncertainty(), pr.uncertainty(), 1e-9, rtol=1e-9)


if __name__ == "__main__":
    from _parity_lib import run_tests; run_tests(__file__)
