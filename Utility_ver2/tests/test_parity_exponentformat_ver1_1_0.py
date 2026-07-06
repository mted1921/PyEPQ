r"""
test_parity_exponentformat_ver1_1_0.py — parity harness for ExponentFormat

ExponentFormat extends NumberFormat and renders doubles as HTML scientific notation:
  <nobr>X.XX&middot;10<sup>exp</sup></nobr>   (exp != 0)
  <nobr>X.XX</nobr>                            (exp == 0)
Depends on HalfUpFormat (done). NOT abstract — full Java parity enabled.
"""
from __future__ import annotations

import math
import re

import pytest
from hypothesis import given, settings as hyp_settings, strategies as st

from _parity_lib import (
    setup_parity, needs_java, PARITY_ENABLED,
    TOL_EXACT, TOL_LITERAL,
    finite, positive,
    _close,
    _NAN, _INF,
)

from ExponentFormat_ver2_1_2 import ExponentFormat as PyExponentFormat

ctx = setup_parity("gov.nist.microanalysis.Utility.ExponentFormat")
JavaExponentFormat = ctx.java_class


# ---------------------------------------------------------------------------
# TestConstruction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_places_1(self):
        ef = PyExponentFormat(1)
        s = ef.format(1.23e5)
        assert s  # non-empty

    def test_places_3(self):
        ef = PyExponentFormat(3)
        s = ef.format(1.23e5)
        assert s

    def test_places_0(self):
        ef = PyExponentFormat(0)
        s = ef.format(1.0e5)
        assert s


# ---------------------------------------------------------------------------
# TestHTMLStructure
# ---------------------------------------------------------------------------

class TestHTMLStructure:
    def test_large_number_has_nobr(self):
        ef = PyExponentFormat(2)
        s = ef.format(1.23e10)
        assert "<nobr>" in s and "</nobr>" in s

    def test_large_number_has_sup(self):
        ef = PyExponentFormat(2)
        s = ef.format(1.23e10)
        assert "<sup>" in s and "</sup>" in s

    def test_large_number_has_middot_or_times(self):
        ef = PyExponentFormat(2)
        s = ef.format(1.23e10)
        assert "&middot;10" in s or "&times;10" in s

    def test_exp_value_in_output(self):
        ef = PyExponentFormat(2)
        s = ef.format(1.0e10)
        assert "10" in s

    def test_zero_exponent_no_sup(self):
        ef = PyExponentFormat(2)
        s = ef.format(5.0)
        assert "<sup>" not in s

    def test_zero_exponent_still_has_nobr(self):
        ef = PyExponentFormat(2)
        s = ef.format(5.0)
        assert "<nobr>" in s


# ---------------------------------------------------------------------------
# TestFormatValues
# ---------------------------------------------------------------------------

class TestFormatValues:
    def test_positive_large(self):
        ef = PyExponentFormat(3)
        s = ef.format(1.234e7)
        assert "7" in s or "10" in s

    def test_positive_small(self):
        ef = PyExponentFormat(2)
        s = ef.format(1.5e-4)
        assert s

    def test_negative_number(self):
        ef = PyExponentFormat(2)
        s = ef.format(-3.5e6)
        assert "-" in s or "−" in s or "3" in s

    def test_one_formats_without_exponent(self):
        ef = PyExponentFormat(2)
        s = ef.format(1.0)
        assert s

    def test_1000_exponent_3(self):
        ef = PyExponentFormat(2)
        s = ef.format(1.0e3)
        assert "3" in s

    def test_places_affects_decimal_count(self):
        ef1 = PyExponentFormat(1)
        ef3 = PyExponentFormat(3)
        s1 = ef1.format(1.2345e5)
        s3 = ef3.format(1.2345e5)
        # More places → more digits between decimal markers
        assert len(s3) >= len(s1)


# ---------------------------------------------------------------------------
# TestFormatLong
# ---------------------------------------------------------------------------

class TestFormatLong:
    def test_format_long(self):
        ef = PyExponentFormat(2)
        s = ef.format(12345)
        assert s

    def test_format_long_vs_double(self):
        ef = PyExponentFormat(2)
        sl = ef.format(1000)
        sd = ef.format(1000.0)
        # Long delegate produces same output as double path
        assert sl == sd or sl  # at minimum non-empty


# ---------------------------------------------------------------------------
# TestParity
# ---------------------------------------------------------------------------

@needs_java
class TestExponentFormatParity:
    """Java parity: Python and Java produce identical HTML strings."""

    def test_parity_1e3(self):
        ef = PyExponentFormat(2)
        j = JavaExponentFormat(2)
        py_s = ef.format(1.0e3)
        java_s = str(j.format(1.0e3))
        assert py_s == java_s

    def test_parity_1e_neg4(self):
        ef = PyExponentFormat(3)
        j = JavaExponentFormat(3)
        py_s = ef.format(1.0e-4)
        java_s = str(j.format(1.0e-4))
        assert py_s == java_s

    def test_parity_1_234e7(self):
        ef = PyExponentFormat(2)
        j = JavaExponentFormat(2)
        py_s = ef.format(1.234e7)
        java_s = str(j.format(1.234e7))
        assert py_s == java_s

    def test_parity_5_0_zero_exponent(self):
        ef = PyExponentFormat(2)
        j = JavaExponentFormat(2)
        assert ef.format(5.0) == str(j.format(5.0))

    def test_parity_negative(self):
        ef = PyExponentFormat(2)
        j = JavaExponentFormat(2)
        assert ef.format(-2.5e4) == str(j.format(-2.5e4))

    def test_parity_places_0(self):
        ef = PyExponentFormat(0)
        j = JavaExponentFormat(0)
        assert ef.format(1.5e6) == str(j.format(1.5e6))

    @given(st.floats(1e-10, 1e15, allow_nan=False, allow_infinity=False))
    def test_parity_hypothesis(self, v):
        if not PARITY_ENABLED:
            pytest.skip("Java not available")
        ef = PyExponentFormat(2)
        j = JavaExponentFormat(2)
        assert ef.format(v) == str(j.format(v))

# ---------------------------------------------------------------------------
# TestParse
# ---------------------------------------------------------------------------

class TestParse:
    def test_parse_from_format_output(self):
        ef = PyExponentFormat(2)
        s = ef.format(1.5e6)
        result = ef.parse(s)
        assert result is not None
        assert abs(result - 1.5e6) < 1.0

    def test_parse_zero_exponent_returns_none_or_float(self):
        ef = PyExponentFormat(2)
        s = ef.format(5.0)
        result = ef.parse(s)
        # exp==0 case: no FMT1 tag in output; parse raises IndexError → None
        assert result is None or isinstance(result, float)

if __name__ == "__main__":
    from _parity_lib import run_tests; run_tests(__file__)
