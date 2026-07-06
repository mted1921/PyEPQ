r"""
test_parity_mcuncertaintyengine_ver2_1_0.py -- parity harness for MCUncertaintyEngine

Pre-written harness (Prompt 2): targets the port's expected API per its spec.

MCUncertaintyEngine is abstract. The constructor runs `iterations` Monte Carlo
trials immediately: each trial builds an UncertainValueMC per argument (sharing one
deviates map so common components correlate), calls the abstract
`compute(UncertainValueMC[])`, and stores the result. Accessors:
  getResults() -> list           getStatistics() -> DescriptiveStatistics
  nominalValue() -> float        getResult() -> UncertainValue2(mean, stddev)

M4: `compute` is abstract; JPype cannot subclass the Java class from Python
(Part 2 skipped). Correctness is validated with a concrete Python subclass:
  * nominalValue() is RNG-INDEPENDENT (built from each UVMC's nominal component),
    so it is asserted exactly;
  * getResult() is statistical -- checked with large N and tolerances chosen at
    >10 sigma so the wall-clock-seeded RNG does not cause flakiness.
"""
from __future__ import annotations

import math

import pytest
from hypothesis import given, strategies as st

from _parity_lib import _close, slow

from MCUncertaintyEngine_ver2_1_0 import MCUncertaintyEngine as PyMCEngine
from UncertainValueMC_ver2_1_1 import UncertainValueMC as PyUVMC
from UncertainValue2_ver2_1_0 import UncertainValue2 as PyUV2


class _SumEngine(PyMCEngine):
    """z = arg0 + arg1 (independent sources -> var = da^2 + db^2)."""

    def compute(self, arguments):
        return PyUVMC.add_vv(arguments[0], arguments[1])


class _ScaleEngine(PyMCEngine):
    """z = 2 * arg0."""

    def compute(self, arguments):
        return PyUVMC.multiply_sv(2.0, arguments[0])


_N = 50_000   # SE(mean) ~ sqrt(5)/sqrt(N) ~ 0.01; tolerances below are >10 sigma.


# ############################################################################
# PART 1 -- Construction & deterministic nominal value
# ############################################################################


class TestConstruction:
    def test_results_length_equals_iterations(self) -> None:
        eng = _SumEngine(1000, [PyUV2(10.0, "A", 1.0), PyUV2(20.0, "B", 2.0)])
        assert len(eng.getResults()) == 1000

    def test_do_iterations_appends(self) -> None:
        eng = _SumEngine(500, [PyUV2(10.0, "A", 1.0), PyUV2(20.0, "B", 2.0)])
        eng.doIterations(500)
        assert len(eng.getResults()) == 1000


class TestNominalValue:
    """nominalValue() uses each UVMC's nominal component -> deterministic."""

    def test_sum_nominal(self) -> None:
        eng = _SumEngine(100, [PyUV2(10.0, "A", 1.0), PyUV2(20.0, "B", 2.0)])
        assert _close(eng.nominalValue(), 30.0, 1e-9)

    def test_scale_nominal(self) -> None:
        eng = _ScaleEngine(100, [PyUV2(7.0, "A", 0.5)])
        assert _close(eng.nominalValue(), 14.0, 1e-9)

    def test_empty_iterations_nominal_raises(self) -> None:
        eng = _SumEngine(0, [PyUV2(10.0, "A", 1.0), PyUV2(20.0, "B", 2.0)])
        with pytest.raises(IndexError):
            eng.nominalValue()


# ############################################################################
# PART 2 -- Statistical result (large N; >10 sigma tolerances)
# ############################################################################


class TestStatisticalResult:
    def test_sum_mean_and_uncertainty(self) -> None:
        eng = _SumEngine(_N, [PyUV2(10.0, "A", 1.0), PyUV2(20.0, "B", 2.0)])
        r = eng.getResult()
        assert isinstance(r, PyUV2)
        # mean -> 30; SE(mean) ~ 0.01, so 0.2 is ~20 sigma.
        assert _close(r.doubleValue(), 30.0, 0.2)
        # stddev -> sqrt(1^2 + 2^2) = sqrt(5) ~ 2.236; relative 10% is ~15 sigma.
        assert _close(r.uncertainty(), math.sqrt(5.0), 0.0, rtol=0.10)

    def test_get_statistics_count(self) -> None:
        eng = _SumEngine(2000, [PyUV2(10.0, "A", 1.0), PyUV2(20.0, "B", 2.0)])
        stats = eng.getStatistics()
        assert stats.count() == 2000


class TestNominalValueProperty:
    """Property: nominalValue() is the sum of the arguments' nominal values."""

    @given(
        st.floats(1.0, 100.0, allow_nan=False, allow_infinity=False),
        st.floats(1.0, 100.0, allow_nan=False, allow_infinity=False),
        st.floats(0.01, 5.0, allow_nan=False, allow_infinity=False),
        st.floats(0.01, 5.0, allow_nan=False, allow_infinity=False),
    )
    @slow
    def test_sum_nominal_property(self, a: float, b: float, sa: float, sb: float) -> None:
        eng = _SumEngine(10, [PyUV2(a, "A", sa), PyUV2(b, "B", sb)])
        assert _close(eng.nominalValue(), a + b, 1e-9)


# ############################################################################
# PART 3 -- Java parity (M4 — skipped)
# ############################################################################

@pytest.mark.skip(
    reason="M4: MCUncertaintyEngine.compute() is abstract; JPype cannot subclass "
           "the Java class from Python. The wall-clock-seeded RNG also precludes "
           "sample-for-sample comparison. Validated analytically (nominal value) "
           "and statistically (mean/stddev) in Parts 1-2."
)
class TestMCUncertaintyEngineParity:
    def test_placeholder(self) -> None:
        pass


if __name__ == "__main__":
    from _parity_lib import run_tests; run_tests(__file__)
