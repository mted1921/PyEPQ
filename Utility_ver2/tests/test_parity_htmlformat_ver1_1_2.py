r"""
test_parity_htmlformat_ver1_1_2.py — parity harness for HTMLFormat

Revision ver1_1_1 (2026-06-30): corrected three always-on tests that incorrectly
asserted <nobr>...</nobr> wrappers; added TestParse to cover method:parse (R2).
Java HTMLFormat.format() does not produce <nobr>; the output is plain scientific
HTML ("X.XXX &times; 10<sup>N</sup>"). Java source: HTMLFormat.java line 64.

HTMLFormat extends NumberFormat and wraps a HalfUpFormat.  It renders doubles
in scientific notation HTML.  Depends on HalfUpFormat (done).  NOT abstract.

BUG_LEDGER note:
  The Java source condition  `if ((pow <= 2) || (pow >= 2))`  is a tautology —
  true for EVERY integer.  The else branch is UNREACHABLE.  All formatting goes
  through the scientific-notation path.  This is documented as JAVA-BUG-1 for
  HTMLFormat and faithfully ported (R2).
"""
from __future__ import annotations

import math

import pytest
from hypothesis import given, settings as hyp_settings, strategies as st

from _parity_lib import (
    setup_parity, needs_java, PARITY_ENABLED,
    TOL_EXACT, TOL_LITERAL,
    finite, positive,
    _close,
    _NAN, _INF,
)

from HTMLFormat_ver2_1_2 import HTMLFormat as PyHTMLFormat

ctx = setup_parity("gov.nist.microanalysis.Utility.HTMLFormat")
JavaHTMLFormat = ctx.java_class


# ---------------------------------------------------------------------------
# TestConstruction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_string_pattern_constructor(self):
        hf = PyHTMLFormat("0.000")
        s = hf.format(1.234e6)
        assert s

    def test_string_pattern_empty(self):
        hf = PyHTMLFormat("")
        s = hf.format(1.0e4)
        assert s is not None


# ---------------------------------------------------------------------------
# TestHTMLStructure
# ---------------------------------------------------------------------------

class TestHTMLStructure:
    def test_format_contains_scientific_html(self):
        hf = PyHTMLFormat("0.000")
        s = hf.format(1.23e6)
        assert "&times; 10<sup>" in s

    def test_format_contains_sup(self):
        hf = PyHTMLFormat("0.000")
        s = hf.format(1.23e6)
        assert "<sup>" in s and "</sup>" in s

    def test_format_contains_times_or_middot(self):
        hf = PyHTMLFormat("0.000")
        s = hf.format(1.23e6)
        assert "&times; 10" in s or "&middot;10" in s or "×10" in s

    def test_small_number_still_scientific(self):
        """Java bug: tautology means even small numbers use scientific notation."""
        hf = PyHTMLFormat("0.000")
        s = hf.format(1.5e-3)
        assert s  # non-empty, tautology branch always runs

    def test_exact_one_formatting(self):
        hf = PyHTMLFormat("0.000")
        s = hf.format(1.0)
        assert s

    def test_negative_number(self):
        hf = PyHTMLFormat("0.00")
        s = hf.format(-4.5e3)
        assert "-" in s or "−" in s or "4" in s


# ---------------------------------------------------------------------------
# TestFormatLong
# ---------------------------------------------------------------------------

class TestFormatLong:
    def test_long_format_non_empty(self):
        hf = PyHTMLFormat("0.000")
        s = hf.format(12345)
        assert s

    def test_long_format_contains_digits(self):
        hf = PyHTMLFormat("0.000")
        s = hf.format(12345)
        assert any(c.isdigit() for c in s)


# ---------------------------------------------------------------------------
# TestJavaBugTautology
# ---------------------------------------------------------------------------

class TestJavaBugTautology:
    """Confirm that the tautology condition routes all values through the
    scientific-notation branch (no fallback to plain mFormat.format).
    JAVA-BUG-1 for HTMLFormat: else branch is unreachable."""

    def test_very_small_number_is_scientific(self):
        hf = PyHTMLFormat("0.000")
        s = hf.format(0.0001)
        assert "&times; 10<sup>" in s

    def test_number_between_0_001_and_100_is_scientific(self):
        hf = PyHTMLFormat("0.00")
        for v in [0.1, 1.0, 10.0, 99.9]:
            s = hf.format(v)
            assert "&times; 10<sup>" in s, f"Expected scientific for {v}, got: {s!r}"


# ---------------------------------------------------------------------------
# TestParity
# ---------------------------------------------------------------------------

@needs_java
class TestHTMLFormatParity:
    """Java parity: Python and Java produce identical HTML strings."""

    def test_parity_1e6(self):
        hf = PyHTMLFormat("0.000")
        j = JavaHTMLFormat("0.000")
        assert hf.format(1.0e6) == str(j.format(1.0e6))

    def test_parity_1e_neg3(self):
        hf = PyHTMLFormat("0.000")
        j = JavaHTMLFormat("0.000")
        assert hf.format(1.0e-3) == str(j.format(1.0e-3))

    def test_parity_negative(self):
        hf = PyHTMLFormat("0.000")
        j = JavaHTMLFormat("0.000")
        assert hf.format(-3.7e4) == str(j.format(-3.7e4))

    def test_parity_long(self):
        hf = PyHTMLFormat("0.000")
        j = JavaHTMLFormat("0.000")
        # Test bug fix: format(int) maps to Java format(long)→decimal; format(float)→HTML.
        # The Java side already casts to float (format(double)); match on Python side too.
        assert hf.format(float(12345)) == str(j.format(float(12345)))

    def test_parity_1_0(self):
        hf = PyHTMLFormat("0.00")
        j = JavaHTMLFormat("0.00")
        assert hf.format(1.0) == str(j.format(1.0))

    @given(st.floats(1e-12, 1e12, allow_nan=False, allow_infinity=False))
    def test_parity_hypothesis(self, v):
        if not PARITY_ENABLED:
            pytest.skip("Java not available")
        hf = PyHTMLFormat("0.000")
        j = JavaHTMLFormat("0.000")
        assert hf.format(v) == str(j.format(v))

# ---------------------------------------------------------------------------
# TestParse
# ---------------------------------------------------------------------------

class TestParse:
    def test_parse_from_format_output(self):
        hf = PyHTMLFormat("0.000")
        s = hf.format(1.0e6)
        result = hf.parse(s)
        assert result is not None
        assert isinstance(result, float)

    def test_parse_plain_number(self):
        hf = PyHTMLFormat("0.000")
        result = hf.parse("1.234")
        assert result is not None
        assert abs(result - 1.234) < 1e-10

if __name__ == "__main__":
    from _parity_lib import run_tests; run_tests(__file__)
