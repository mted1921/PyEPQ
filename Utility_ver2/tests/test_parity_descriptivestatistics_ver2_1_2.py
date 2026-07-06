r"""
test_parity_descriptivestatistics_ver2_1_1.py -- parity harness for DescriptiveStatistics

Pre-written harness (Prompt 2): targets the port's expected API per
DescriptiveStatistics_ver1_1_1.spec.md (repaired from the Java source).

DescriptiveStatistics accumulates four power sums plus min/max/count and exposes
average / variance / standardDeviation / skewness / kurtosis / minimum / maximum /
count / sum / getValue(name) -> UncertainValue2. It is a concrete class, so Part 2
compares directly against Java.

Java definitions (population-style, dividing by N):
  average()  = sum / N
  variance() = N>1 ? max(0, (sumOfSqrs - sum*avg) / N) : NaN
  getValue(name) = UncertainValue2(average, name, isnan(sd) ? 0 : sd)

JAVA-BUG-1: merge() increments mNPoints TWICE, so after merging an m-point object
into an n-point object, count() reports n + 2m. The `merge_strict` companion
increments once.
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

from DescriptiveStatistics_ver2_1_2 import (  # FIX-1: ver2_1_1 → ver2_1_2 (port renamed after FIX-2)
    DescriptiveStatistics as PyDescriptiveStatistics,
)
from UncertainValue2_ver2_1_0 import UncertainValue2 as PyUncertainValue2

ctx = setup_parity("gov.nist.microanalysis.Utility.DescriptiveStatistics")
JavaDescriptiveStatistics = ctx.java_class

_TOL = 1e-12

_data = st.lists(
    st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
    min_size=1, max_size=50,
)


def _fill(values):
    ds = PyDescriptiveStatistics()
    for v in values:
        ds.add(v)
    return ds


# ############################################################################
# PART 1 -- Always-on analytical tests
# ############################################################################


class TestConstruction:
    def test_empty_count_zero(self) -> None:
        assert PyDescriptiveStatistics().count() == 0

    def test_empty_sum_zero(self) -> None:
        assert PyDescriptiveStatistics().sum() == 0.0


class TestBasicStats:
    DATA = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]   # mean 5, popvar 4, std 2

    def test_count(self) -> None:
        assert _fill(self.DATA).count() == 8

    def test_sum(self) -> None:
        assert _close(_fill(self.DATA).sum(), 40.0, _TOL)

    def test_average(self) -> None:
        assert _close(_fill(self.DATA).average(), 5.0, _TOL)

    def test_variance_population(self) -> None:
        # Java variance divides by N -> population variance = 4.0.
        assert _close(_fill(self.DATA).variance(), 4.0, 1e-9)

    def test_standard_deviation(self) -> None:
        assert _close(_fill(self.DATA).standardDeviation(), 2.0, 1e-9)

    def test_minimum(self) -> None:
        assert _fill(self.DATA).minimum() == 2.0

    def test_maximum(self) -> None:
        assert _fill(self.DATA).maximum() == 9.0

    def test_single_value_variance_is_nan(self) -> None:
        # N == 1 -> variance() returns NaN by definition.
        assert math.isnan(_fill([3.0]).variance())


class TestGetValue:
    def test_get_value_maps_to_uncertain_value2(self) -> None:
        ds = _fill([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        uv = ds.getValue("stat")
        assert isinstance(uv, PyUncertainValue2)
        assert _close(uv.doubleValue(), 5.0, _TOL)
        assert _close(uv.uncertainty(), 2.0, 1e-9)

    def test_single_value_uncertainty_zero(self) -> None:
        # variance() is NaN for N==1; getValue substitutes 0 for the sigma.
        uv = _fill([3.0]).getValue("s")
        assert _close(uv.doubleValue(), 3.0, _TOL)
        assert uv.uncertainty() == 0.0


class TestComputeStatic:
    """Static compute() builds an accumulator from a sequence (R4 split)."""

    def test_compute_collection(self) -> None:
        ds = PyDescriptiveStatistics.compute_collection([1.0, 2.0, 3.0, 4.0])
        assert ds.count() == 4
        assert _close(ds.average(), 2.5, _TOL)

    def test_compute_array(self) -> None:
        ds = PyDescriptiveStatistics.compute_array([1.0, 2.0, 3.0, 4.0])
        assert ds.count() == 4
        assert _close(ds.sum(), 10.0, _TOL)


class TestCombineConstructor:
    """Two-arg constructor combines two accumulators (sums + counts added)."""

    def test_combine_counts_and_sums(self) -> None:
        a = _fill([1.0, 2.0, 3.0])
        b = _fill([4.0, 5.0])
        c = PyDescriptiveStatistics(a, b)
        assert c.count() == 5
        assert _close(c.sum(), 15.0, _TOL)
        assert _close(c.average(), 3.0, _TOL)


class TestCompareTo:
    def test_orders_by_average(self) -> None:
        lo = _fill([1.0, 1.0])
        hi = _fill([9.0, 9.0])
        assert lo.compareTo(hi) == -1
        assert hi.compareTo(lo) == 1

    def test_equal_averages_tiebreak_variance(self) -> None:
        a = _fill([5.0, 5.0])       # variance 0
        b = _fill([4.0, 6.0])       # same average, larger variance
        assert a.compareTo(b) == -1


class TestHypothesis:
    @given(_data)
    @slow
    def test_count_equals_items_added(self, values) -> None:
        assert _fill(values).count() == len(values)

    @given(_data)
    @slow
    def test_sum_equals_total(self, values) -> None:
        assert _close(_fill(values).sum(), sum(values), 1e-6, rtol=1e-9)

    @given(_data)
    @slow
    def test_min_max_bracket_data(self, values) -> None:
        ds = _fill(values)
        assert ds.minimum() == min(values)
        assert ds.maximum() == max(values)


# ############################################################################
# Bug-aware tests -- JAVA-BUG-1 (merge double-counts mNPoints)
# ############################################################################


class TestPreservedBugs:
    """merge() double-counts mNPoints: count becomes n1 + 2*n2 (preserved bug)."""

    def test_merge_double_counts(self) -> None:
        a = _fill([1.0, 2.0, 3.0])   # n1 = 3
        b = _fill([4.0, 5.0])        # n2 = 2
        a.merge(b)
        # Buggy Java behaviour: 3 + 2*2 = 7, not 5.
        assert a.count() == 7


class TestStrictVariants:
    """merge_strict() increments the count once (the corrected behaviour)."""

    def test_merge_strict_counts_once(self) -> None:
        a = _fill([1.0, 2.0, 3.0])
        b = _fill([4.0, 5.0])
        a.merge_strict(b)
        assert a.count() == 5
        assert _close(a.sum(), 15.0, _TOL)


# ############################################################################
# PART 2 -- Parity tests (require JVM + EPQ.jar + EPQ_PARITY=1)
# ############################################################################


def _fill_java(values):
    ds = JavaDescriptiveStatistics()
    for v in values:
        ds.add(float(v))
    return ds


@needs_java
class TestDescriptiveStatisticsParity:

    @given(_data)
    @slow
    def test_average_parity(self, values) -> None:
        assert _close(_fill_java(values).average(), _fill(values).average(),
                      1e-9, rtol=1e-9)

    @given(st.lists(
        st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
        min_size=2, max_size=50))
    @slow
    def test_variance_parity(self, values) -> None:
        assert _close(_fill_java(values).variance(), _fill(values).variance(),
                      1e-7, rtol=1e-7)

    @given(st.lists(
        st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
        min_size=3, max_size=50))
    @slow
    def test_skewness_parity(self, values) -> None:
        j = _fill_java(values).skewness()
        p = _fill(values).skewness()
        # Both may be NaN for degenerate (zero-variance) inputs.
        if math.isnan(float(j)) or math.isnan(p):
            return
        assert _close(j, p, 1e-6, rtol=1e-6)

    @given(_data)
    @slow
    def test_min_max_parity(self, values) -> None:
        jds, pds = _fill_java(values), _fill(values)
        assert _close(jds.minimum(), pds.minimum(), 0.0)
        assert _close(jds.maximum(), pds.maximum(), 0.0)

    @given(_data)
    @slow
    def test_count_sum_parity(self, values) -> None:
        jds, pds = _fill_java(values), _fill(values)
        assert int(jds.count()) == pds.count()
        assert _close(jds.sum(), pds.sum(), 1e-6, rtol=1e-9)


class TestToString:
    def test_to_string_is_string(self) -> None:
        ds = _fill([1.0, 2.0, 3.0])
        assert isinstance(ds.toString(), str)

    def test_to_string_contains_average(self) -> None:
        ds = _fill([2.0, 4.0])
        s = ds.toString()
        assert s is not None and len(s) > 0


class TestSkewnessBoundary:
    """FIX-N discipline: skewness returns NaN when variance is zero (constant data).

    Prior to FIX-1, skewness_literal() raised ZeroDivisionError on constant data.
    The fix returns NaN. This test pins the corrected boundary so a regression
    to ValueError would be immediately caught.
    """

    def test_skewness_constant_data_is_nan(self) -> None:
        ds = _fill([5.0, 5.0, 5.0])
        assert math.isnan(ds.skewness())

    def test_kurtosis_constant_data_is_nan(self) -> None:
        ds = _fill([3.0, 3.0, 3.0])
        assert math.isnan(ds.kurtosis())


class TestAdditionalMethods:
    """Coverage for public methods not exercised in the classes above."""

    DATA = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]

    def test_clone_preserves_state(self) -> None:
        ds = _fill(self.DATA)
        ds2 = ds.clone()
        assert ds2.count() == ds.count()
        assert _close(ds2.average(), ds.average(), _TOL)

    def test_get_last_added(self) -> None:
        ds = _fill([1.0, 2.0, 3.0])
        assert _close(ds.getLastAdded(), 3.0, _TOL)

    def test_kurtosis_finite(self) -> None:
        ds = _fill(self.DATA)
        k = ds.kurtosis()
        assert not math.isnan(k) and not math.isinf(k)

    def test_remove_reduces_count(self) -> None:
        ds = _fill([1.0, 2.0, 3.0])
        ds.remove(2.0)
        assert ds.count() == 2

    def test_remove_last_returns_true(self) -> None:
        ds = _fill([1.0, 2.0, 3.0])
        assert ds.removeLast() is True
        assert ds.count() == 2

    def test_standard_error_of_kurtosis_positive(self) -> None:
        ds = _fill(self.DATA)
        assert ds.standardErrorOfKurtosis() > 0.0

    def test_standard_error_of_skewness_positive(self) -> None:
        ds = _fill(self.DATA)
        assert ds.standardErrorOfSkewness() > 0.0


if __name__ == "__main__":
    from _parity_lib import run_tests; run_tests(__file__)
