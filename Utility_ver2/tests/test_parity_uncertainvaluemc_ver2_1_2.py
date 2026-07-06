r"""
test_parity_uncertainvaluemc_ver2_1_1.py -- parity harness for UncertainValueMC

Pre-written harness (Prompt 2): targets the port's expected API per
UncertainValueMC_ver1_1_1.spec.md (repaired from the Java source).

UncertainValueMC extends Number and carries a NOMINAL value (`mValue`) and a single
random SAMPLE (`mRandVal`); both are final. The static algebra propagates both
components. Because `mRandVal` is drawn from a wall-clock-seeded RNG, the random
component is non-deterministic across runs.

Test strategy
-------------
* PART 1 builds objects with the private all-fields helper `_internal(v, randVal)`
  so BOTH components are known, then checks the algebra and the two preserved bugs
  deterministically (no RNG involved).
* PART 2 compares against Java only on `nominalValue()`, the RNG-independent
  component (it depends only on `mValue`). The random component cannot be compared
  directly without identical seeds.

Preserved Java bugs (UncertainValueMC_ver1_1_1.spec.md):
  JAVA-BUG-1  exp(n) computes Math.log on BOTH components (copy-paste of log()).
  JAVA-BUG-2  pow(n, exp) uses the exponent's RANDOM sample in the nominal value:
              mValue = pow(n.mValue, exp.mRandVal).
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

from UncertainValueMC_ver2_1_1 import UncertainValueMC as PyUVMC

ctx = setup_parity("gov.nist.microanalysis.Utility.UncertainValueMC")
JavaUVMC = ctx.java_class

_TOL = 1e-12


def _mk(value: float, randval: float):
    """Build a UVMC with both components fixed (bypasses the RNG)."""
    return PyUVMC._internal(value, randval)


# ############################################################################
# PART 1 -- Deterministic algebra via _internal (no RNG)
# ############################################################################


class TestLinearAlgebra:
    """Linear ops act component-wise on (nominal, sample)."""

    def test_add_vv(self) -> None:
        r = PyUVMC.add_vv(_mk(1.0, 2.0), _mk(3.0, 4.0))
        assert _close(r.nominalValue(), 4.0, _TOL)
        assert _close(r.doubleValue(), 6.0, _TOL)

    def test_subtract(self) -> None:
        r = PyUVMC.subtract(_mk(1.0, 2.0), _mk(3.0, 4.0))
        assert _close(r.nominalValue(), -2.0, _TOL)
        assert _close(r.doubleValue(), -2.0, _TOL)

    def test_add_combo(self) -> None:
        # a*uva + b*uvb on each component.
        r = PyUVMC.add_combo(2.0, _mk(1.0, 2.0), 3.0, _mk(4.0, 5.0))
        assert _close(r.nominalValue(), 2 * 1 + 3 * 4, _TOL)
        assert _close(r.doubleValue(), 2 * 2 + 3 * 5, _TOL)

    def test_sum(self) -> None:
        r = PyUVMC.sum([_mk(1.0, 2.0), _mk(3.0, 4.0), _mk(5.0, 6.0)])
        assert _close(r.nominalValue(), 9.0, _TOL)
        assert _close(r.doubleValue(), 12.0, _TOL)

    def test_mean(self) -> None:
        r = PyUVMC.mean([_mk(2.0, 4.0), _mk(4.0, 8.0)])
        assert _close(r.nominalValue(), 3.0, _TOL)
        assert _close(r.doubleValue(), 6.0, _TOL)


class TestMultiplicativeAlgebra:
    def test_multiply_vv(self) -> None:
        r = PyUVMC.multiply_vv(_mk(1.0, 2.0), _mk(3.0, 4.0))
        assert _close(r.nominalValue(), 3.0, _TOL)
        assert _close(r.doubleValue(), 8.0, _TOL)

    def test_multiply_sv(self) -> None:
        r = PyUVMC.multiply_sv(2.0, _mk(3.0, 4.0))
        assert _close(r.nominalValue(), 6.0, _TOL)
        assert _close(r.doubleValue(), 8.0, _TOL)

    def test_divide_vv(self) -> None:
        r = PyUVMC.divide_vv(_mk(6.0, 8.0), _mk(2.0, 4.0))
        assert _close(r.nominalValue(), 3.0, _TOL)
        assert _close(r.doubleValue(), 2.0, _TOL)

    def test_divide_vs(self) -> None:
        r = PyUVMC.divide_vs(_mk(6.0, 8.0), 2.0)
        assert _close(r.nominalValue(), 3.0, _TOL)
        assert _close(r.doubleValue(), 4.0, _TOL)


class TestTranscendentalAndSign:
    def test_log(self) -> None:
        r = PyUVMC.log(_mk(math.e, math.e ** 2))
        assert _close(r.nominalValue(), 1.0, 1e-12)
        assert _close(r.doubleValue(), 2.0, 1e-12)

    def test_sqrt(self) -> None:
        r = PyUVMC.sqrt(_mk(4.0, 9.0))
        assert _close(r.nominalValue(), 2.0, _TOL)
        assert _close(r.doubleValue(), 3.0, _TOL)

    def test_sqrt_clamps_negative_sample(self) -> None:
        # sqrt uses max(0, mRandVal) on the sample component.
        r = PyUVMC.sqrt(_mk(4.0, -1.0))
        assert _close(r.nominalValue(), 2.0, _TOL)
        assert _close(r.doubleValue(), 0.0, _TOL)

    def test_abs(self) -> None:
        r = PyUVMC.abs(_mk(-3.0, -4.0))
        assert _close(r.nominalValue(), 3.0, _TOL)
        assert _close(r.doubleValue(), 4.0, _TOL)

    def test_non_negative_equals_abs(self) -> None:
        r = PyUVMC.nonNegative(_mk(-3.0, -4.0))
        assert _close(r.nominalValue(), 3.0, _TOL)
        assert _close(r.doubleValue(), 4.0, _TOL)


class TestNumberInterface:
    def test_double_value_is_sample(self) -> None:
        assert _close(_mk(5.0, 7.0).doubleValue(), 7.0, _TOL)

    def test_nominal_value_is_value(self) -> None:
        assert _close(_mk(5.0, 7.0).nominalValue(), 5.0, _TOL)

    def test_int_value_truncates_sample(self) -> None:
        assert _mk(5.0, 7.9).intValue() == 7

    def test_long_value_truncates_sample(self) -> None:
        assert _mk(5.0, 7.9).longValue() == 7

    def test_float_value_is_sample(self) -> None:
        assert _close(_mk(5.0, 7.0).floatValue(), 7.0, 1e-6)


class TestConstruction:
    def test_value_uncertainty_ctor_nominal(self) -> None:
        # (v, dv) ctor: nominal is v; sample is dv*normalDeviate() -> 0 when dv==0.
        uv = PyUVMC(5.0, 0.0)
        assert _close(uv.nominalValue(), 5.0, _TOL)
        assert _close(uv.doubleValue(), 0.0, _TOL)

    def test_uv2_deviates_ctor_nominal(self) -> None:
        # (UncertainValue2, deviates) ctor: nominal = uv.doubleValue(); no uncertainty
        # components -> sample = uv.doubleValue() + 0 = nominal.
        from UncertainValue2_ver2_1_0 import UncertainValue2 as PyUV2
        uv2 = PyUV2(3.0, 0.0)   # zero-sigma UV2: no components
        uv = PyUVMC(uv2, {})
        assert _close(uv.nominalValue(), 3.0, _TOL)


# ############################################################################
# Bug-aware tests
# ############################################################################


class TestPreservedBugs:
    def test_exp_actually_computes_log(self) -> None:
        # JAVA-BUG-1: exp() applies log() to both components.
        r = PyUVMC.exp(_mk(math.e, math.e ** 2))
        assert _close(r.nominalValue(), 1.0, 1e-12)   # log(e), not exp(e)
        assert _close(r.doubleValue(), 2.0, 1e-12)    # log(e^2)

    def test_pow_nominal_uses_exponent_sample(self) -> None:
        # JAVA-BUG-2: nominal = pow(n.mValue, exp.mRandVal) -- uses the exponent's
        # RANDOM sample (2.0), not its nominal (5.0).
        n = _mk(2.0, 3.0)
        ex = _mk(5.0, 2.0)
        r = PyUVMC.pow(n, ex)
        assert _close(r.nominalValue(), math.pow(2.0, 2.0), _TOL)   # 4.0, not 2**5
        assert _close(r.doubleValue(), math.pow(3.0, 2.0), _TOL)    # 9.0


class TestStrictVariants:
    def test_exp_strict(self) -> None:
        r = PyUVMC.exp_strict(_mk(0.0, 1.0))
        assert _close(r.nominalValue(), 1.0, 1e-12)        # exp(0)
        assert _close(r.doubleValue(), math.e, 1e-12)      # exp(1)

    def test_pow_strict_uses_exponent_nominal(self) -> None:
        r = PyUVMC.pow_strict(_mk(2.0, 3.0), _mk(5.0, 2.0))
        assert _close(r.nominalValue(), math.pow(2.0, 5.0), _TOL)   # 32.0


class TestLogBoundary:
    """Boundary pins for FIX-1: _java_log handles non-positive mRandVal.

    Java Math.log() returns NaN for negative inputs and -Infinity for zero
    (IEEE 754). The port's _java_log() helper must match those semantics to
    avoid raising ValueError. These tests pin the fixed boundary.
    """

    def test_log_negative_sample_returns_nan(self) -> None:
        # mRandVal = -1.0 -> _java_log(-1.0) must be NaN, not raise ValueError.
        r = PyUVMC.log(_mk(1.0, -1.0))
        assert math.isnan(r.doubleValue())

    def test_log_zero_sample_returns_neg_inf(self) -> None:
        # mRandVal = 0.0 -> _java_log(0.0) must be -Inf (Java IEEE 754).
        r = PyUVMC.log(_mk(1.0, 0.0))
        assert r.doubleValue() == float("-inf")


class TestToString:
    def test_to_string_returns_sample_as_string(self) -> None:
        uv = _mk(5.0, 3.14)
        s = uv.toString()
        assert isinstance(s, str)
        assert "3.14" in s


# ############################################################################
# PART 2 -- Parity tests (nominal value only; RNG-independent)
# ############################################################################
# nominalValue() depends only on mValue (= the `v` constructor argument), so it is
# reproducible across Java and Python despite the wall-clock-seeded RNG. The random
# sample component (doubleValue) is NOT compared. pow() is excluded because its
# nominal uses the exponent's random sample (JAVA-BUG-2).

_val = st.floats(min_value=0.5, max_value=100.0, allow_nan=False, allow_infinity=False)


@needs_java
class TestNominalParity:

    def _pair(self, v: float, dv: float):
        return PyUVMC(v, dv), JavaUVMC(float(v), float(dv))

    @given(_val, _val)
    @slow
    def test_add_nominal(self, va: float, vb: float) -> None:
        pa, ja = self._pair(va, 0.1)
        pb, jb = self._pair(vb, 0.1)
        assert _close(JavaUVMC.add(ja, jb).nominalValue(),
                      PyUVMC.add_vv(pa, pb).nominalValue(), 1e-9, rtol=1e-9)

    @given(_val, _val)
    @slow
    def test_subtract_nominal(self, va: float, vb: float) -> None:
        pa, ja = self._pair(va, 0.1)
        pb, jb = self._pair(vb, 0.1)
        assert _close(JavaUVMC.subtract(ja, jb).nominalValue(),
                      PyUVMC.subtract(pa, pb).nominalValue(), 1e-9, rtol=1e-9)

    @given(_val, _val)
    @slow
    def test_multiply_nominal(self, va: float, vb: float) -> None:
        pa, ja = self._pair(va, 0.1)
        pb, jb = self._pair(vb, 0.1)
        assert _close(JavaUVMC.multiply(ja, jb).nominalValue(),
                      PyUVMC.multiply_vv(pa, pb).nominalValue(), 1e-9, rtol=1e-9)

    @given(_val, _val)
    @slow
    def test_divide_nominal(self, va: float, vb: float) -> None:
        pa, ja = self._pair(va, 0.1)
        pb, jb = self._pair(vb, 0.1)
        assert _close(JavaUVMC.divide(ja, jb).nominalValue(),
                      PyUVMC.divide_vv(pa, pb).nominalValue(), 1e-9, rtol=1e-9)

    @given(_val)
    @slow
    def test_log_nominal(self, v: float) -> None:
        p, j = self._pair(v, 0.1)
        assert _close(JavaUVMC.log(j).nominalValue(),
                      PyUVMC.log(p).nominalValue(), 1e-9, rtol=1e-9)

    @given(_val)
    @slow
    def test_exp_nominal_buggy(self, v: float) -> None:
        # Both sides compute log() in exp() (JAVA-BUG-1); nominal is log(v).
        p, j = self._pair(v, 0.1)
        assert _close(JavaUVMC.exp(j).nominalValue(),
                      PyUVMC.exp(p).nominalValue(), 1e-9, rtol=1e-9)

    @given(_val)
    @slow
    def test_sqrt_nominal(self, v: float) -> None:
        p, j = self._pair(v, 0.1)
        assert _close(JavaUVMC.sqrt(j).nominalValue(),
                      PyUVMC.sqrt(p).nominalValue(), 1e-9, rtol=1e-9)


if __name__ == "__main__":
    from _parity_lib import run_tests; run_tests(__file__)
