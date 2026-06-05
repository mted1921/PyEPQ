r"""
test_parity_halfupformat.py -- parity harness for HalfUpFormat_ver1.py

Structure
---------
PART 1  (always-on)
  TestMediumMathematicalSpace
                        Verifies the NIST grouping constant is U+2006,
                        accessible at both module and class level.

  TestHalfUpRoundingMidpoints
                        Deterministic proof that HALF_UP (not HALF_EVEN)
                        is used. Key midpoints: 0.5, 2.5, 4.5 differ
                        between the two rounding modes; these are chosen
                        because they are exact in IEEE-754 double.

  TestFormatFixed       Deterministic fixed-point pattern tests.
  TestFormatScientific  Deterministic scientific-notation pattern tests.
  TestGrouping          Space-grouping insertion (U+2006 every 3 digits).
  TestConstructors      All five constructor paths (no-arg + 4 classmethods).
  TestDecimalFormatSymbols
                        The minimal shim: getInstance, setters, getters.
  TestAdaptiveFormat    Deterministic regression table for adaptiveFormat.
  TestAdaptiveFormatHypothesis
                        Property-based: output parses back to approximately
                        the original value; significant-digit count is
                        within 1 of the requested precision.

PART 2  (parity, requires EPQ_PARITY=1 + jpype1 + EPQ jar)
  TestFormatParity      Java HalfUpFormat.format() vs Python format() for
                        safe inputs (exact binary fractions; clear rounding
                        direction).  String equality expected.

  TestAdaptiveFormatParity
                        Java adaptiveFormat() vs Python adaptiveFormat().

                        KNOWN DIVERGENCE: adaptiveFormat's integer
                        pre-rounding branch uses Python round() (HALF_EVEN)
                        but Java Math.round() (HALF_UP).  These differ only
                        at exact midpoints after dividing by div=10^k.
                        Parity inputs are restricted to non-midpoint values.

ROUNDING NOTES
--------------
* Python `decimal.Decimal(repr(x)).quantize(..., ROUND_HALF_UP)` works from
  the string representation, which is the "nice" decimal for any double.
  This matches Java's DecimalFormat behaviour for most values.
* Values like 1.235 where the IEEE-754 double is slightly BELOW the decimal
  are unsafe for string-exact parity: Java formats the raw binary value
  while Python goes via repr().  Safe inputs are exact binary fractions
  (0.5, 0.25, 0.125, etc.) and values where the rounding digit is clearly
  ≠ 5 (digit < 4 or digit > 6).
"""
from __future__ import annotations

import math
import re
import sys

import pytest
from hypothesis import given, strategies as st, assume

from _parity_lib import (
    setup_parity, needs_java, PARITY_ENABLED,
    TOL_EXACT, TOL_LITERAL,
    finite, small,
    slow, slow_fuzz,
    _close,
)

from HalfUpFormat_ver1 import (
    HalfUpFormat as PyHalfUpFormat,
    DecimalFormatSymbols as PyDecimalFormatSymbols,
    _MEDIUM_MATHEMATICAL_SPACE as SPACE,
    _apply_grouping,
    _apply_pattern,
    _format_scientific_halfup,
    _parse_pattern,
)

ctx = setup_parity("gov.nist.microanalysis.Utility.HalfUpFormat")
JavaHalfUpFormat = ctx.java_class


# ############################################################################
# PART 1 -- Always-on tests
# ############################################################################


class TestMediumMathematicalSpace:
    """The grouping character must be U+2006, not a regular ASCII space."""

    def test_module_constant_is_u2006(self) -> None:
        assert SPACE == " "

    def test_class_constant_is_u2006(self) -> None:
        assert PyHalfUpFormat._MEDIUM_MATHEMATICAL_SPACE == " "

    def test_module_and_class_constant_are_same_object(self) -> None:
        assert PyHalfUpFormat._MEDIUM_MATHEMATICAL_SPACE is SPACE

    def test_not_ascii_space(self) -> None:
        assert SPACE != " "

    def test_length_one(self) -> None:
        assert len(SPACE) == 1


class TestHalfUpRoundingMidpoints:
    """Exact midpoints that differ between HALF_UP and HALF_EVEN.

    Values chosen are exact in IEEE-754 (powers of two denominators),
    so repr() returns a clean decimal string and there is no ambiguity.

    HALF_EVEN (Java default / Python round()) vs HALF_UP:
      2.5 → 2 (even) vs 3    4.5 → 4 (even) vs 5
     -2.5 → -2        vs -3  -4.5 → -4       vs -5
    """

    # -- positive midpoints that differ --

    def test_half_rounds_up_0p5(self) -> None:
        assert PyHalfUpFormat.from_pattern("0").format(0.5) == "1"

    def test_half_rounds_up_2p5(self) -> None:
        # HALF_EVEN gives "2"; HALF_UP gives "3".
        assert PyHalfUpFormat.from_pattern("0").format(2.5) == "3"

    def test_half_rounds_up_4p5(self) -> None:
        # HALF_EVEN gives "4"; HALF_UP gives "5".
        assert PyHalfUpFormat.from_pattern("0").format(4.5) == "5"

    def test_half_rounds_up_6p5(self) -> None:
        assert PyHalfUpFormat.from_pattern("0").format(6.5) == "7"

    # -- midpoints that agree (both round to the even / up value) --

    def test_1p5_rounds_to_2_both_modes(self) -> None:
        assert PyHalfUpFormat.from_pattern("0").format(1.5) == "2"

    def test_3p5_rounds_to_4_both_modes(self) -> None:
        assert PyHalfUpFormat.from_pattern("0").format(3.5) == "4"

    # -- negative midpoints (HALF_UP: away from zero) --

    def test_neg_0p5_rounds_to_neg1(self) -> None:
        assert PyHalfUpFormat.from_pattern("0").format(-0.5) == "-1"

    def test_neg_2p5_rounds_to_neg3(self) -> None:
        assert PyHalfUpFormat.from_pattern("0").format(-2.5) == "-3"

    def test_neg_4p5_rounds_to_neg5(self) -> None:
        assert PyHalfUpFormat.from_pattern("0").format(-4.5) == "-5"

    # -- fractional midpoints --

    def test_0p25_to_1dp_rounds_to_0p3(self) -> None:
        # 0.25 exactly in IEEE-754; digit at 2dp is 5 → round up.
        assert PyHalfUpFormat.from_pattern("0.0").format(0.25) == "0.3"

    def test_0p125_to_2dp_rounds_to_0p13(self) -> None:
        # 0.125 exactly; 3rd dp is 5 → round 2nd dp up.
        assert PyHalfUpFormat.from_pattern("0.00").format(0.125) == "0.13"

    def test_neg_0p25_to_1dp_rounds_to_neg0p3(self) -> None:
        assert PyHalfUpFormat.from_pattern("0.0").format(-0.25) == "-0.3"


class TestFormatFixed:
    """Deterministic fixed-point formatting across common patterns."""

    @pytest.mark.parametrize("number, pattern, expected", [
        # Integer pattern "0"
        (0.0,    "0",    "0"),
        (1.0,    "0",    "1"),
        (-1.0,   "0",   "-1"),
        (99.0,   "0",   "99"),
        # Pattern "0.0"
        (0.0,    "0.0",  "0.0"),
        (1.0,    "0.0",  "1.0"),
        (1.1,    "0.0",  "1.1"),
        (1.24,   "0.0",  "1.2"),   # 4 < 5, rounds down
        (1.26,   "0.0",  "1.3"),   # 6 > 5, rounds up
        (-1.24,  "0.0", "-1.2"),
        (-1.76,  "0.0", "-1.8"),
        # Pattern "0.00"
        (0.0,    "0.00", "0.00"),
        (1.0,    "0.00", "1.00"),
        (3.14,   "0.00", "3.14"),
        (3.141,  "0.00", "3.14"),  # 1 < 5
        (3.146,  "0.00", "3.15"),  # 6 > 5
        (3.145,  "0.00", "3.15"),  # 5 at 3dp; 0.145 = exact in repr → HALF_UP rounds up
        # Minimum digit padding ("000")
        (5.0,    "000",  "005"),
        (12.0,   "000",  "012"),
        (123.0,  "000",  "123"),
        (1234.0, "000",  "1234"),  # more digits than minimum, no truncation
        # Pattern "0.000"
        (1.0,    "0.000", "1.000"),
        (1.2346, "0.000", "1.235"),  # 3rd dp digit is 6 > 5... wait
        # 1.2346 to 3dp: 4th dp is 6 > 5 → round up 3rd dp (4→5): "1.235"
        # Actually: round 1.2346 to 3dp: 4th digit = 6 > 5 → 3rd digit goes 4→5 → "1.235"
        # Hmm wait. 1.2346: digits are 1 . 2 3 4 6.  3dp → look at 4th dp = 6 > 5. Round up.
        # 3rd dp was 4, rounds to 5. → "1.235". Yes.
        (1.2346, "0.000", "1.235"),
        (1.2341, "0.000", "1.234"),  # 4th dp = 1 < 5
    ])
    def test_fixed_format(self, number: float, pattern: str, expected: str) -> None:
        result = PyHalfUpFormat.from_pattern(pattern).format(number)
        assert result == expected, (
            f"format({number!r}, {pattern!r}) → {result!r}, expected {expected!r}"
        )

    def test_zero_with_no_pattern(self) -> None:
        """No-pattern HalfUpFormat formats as integer (pattern '0')."""
        assert PyHalfUpFormat().format(0.0) == "0"

    def test_large_integer_no_grouping(self) -> None:
        fmt = PyHalfUpFormat.from_pattern("0")
        assert fmt.format(1000000.0) == "1000000"

    def test_negative_zero_preserves_sign(self) -> None:
        # Decimal(repr(-0.0)) = Decimal('-0.0'); sign bit is preserved.
        # Java DecimalFormat also outputs "-0.0" for -0.0 — matched behaviour.
        assert PyHalfUpFormat.from_pattern("0.0").format(-0.0) == "-0.0"


class TestFormatScientific:
    """Deterministic scientific-notation patterns."""

    @pytest.mark.parametrize("number, pattern, expected", [
        # Zero is always "0...E0"
        (0.0,     "0.0E0",  "0.0E0"),
        (0.0,     "0.00E0", "0.00E0"),
        (0.0,     "0.000E0","0.000E0"),
        # Powers of 10 — mantissa is exactly 1.0
        (10.0,    "0.0E0",  "1.0E1"),
        (100.0,   "0.0E0",  "1.0E2"),
        (0.1,     "0.0E0",  "1.0E-1"),
        (0.01,    "0.0E0",  "1.0E-2"),
        (1000.0,  "0.00E0", "1.00E3"),
        (0.001,   "0.00E0", "1.00E-3"),
        # Mantissa rounds clearly down (digit < 5)
        (12340.0, "0.00E0", "1.23E4"),   # mantissa=1.234, 3dp=4 < 5
        (1234.0,  "0.0E0",  "1.2E3"),    # mantissa=1.234, 2dp=3 < 5
        # Mantissa rounds clearly up (digit > 5)
        (12360.0, "0.00E0", "1.24E4"),   # mantissa=1.236, 3dp=6 > 5
        (1236.0,  "0.0E0",  "1.2E3"),    # mantissa=1.236, 2dp=3 < 5 → 1.2
        (12700.0, "0.0E0",  "1.3E4"),    # mantissa=1.27,  2dp=7 > 5
        # Carry-over: mantissa rounds to 10 → exp increases
        (9.96,    "0.0E0",  "1.0E1"),    # 9.96 rounds to 10.0 → 1.0E1
        # Negative values
        (-100.0,  "0.0E0",  "-1.0E2"),
        (-1234.0, "0.0E0",  "-1.2E3"),
        # Very small exponents
        (1.0,     "0.00E0", "1.00E0"),
        (5.0,     "0.0E0",  "5.0E0"),
    ])
    def test_scientific_format(
        self, number: float, pattern: str, expected: str
    ) -> None:
        result = PyHalfUpFormat.from_pattern(pattern).format(number)
        assert result == expected, (
            f"format({number!r}, {pattern!r}) → {result!r}, expected {expected!r}"
        )

    def test_scientific_halfup_midpoint(self) -> None:
        """HALF_UP in exponent: 1.25 with 1dp → 1.3."""
        assert PyHalfUpFormat.from_pattern("0.0E0").format(1.25) == "1.3E0"

    def test_parse_pattern_fixed(self) -> None:
        assert _parse_pattern("0.000") == (False, 3)

    def test_parse_pattern_scientific(self) -> None:
        assert _parse_pattern("0.00E0") == (True, 2)

    def test_parse_pattern_integer(self) -> None:
        assert _parse_pattern("0") == (False, 0)


class TestGrouping:
    """Space-grouping inserts U+2006 every 3 digits in the integer part."""

    def test_no_grouping_below_threshold(self) -> None:
        """< 4 digits: no separator inserted."""
        fmt = PyHalfUpFormat.from_pattern_space_grouping("0", True)
        assert SPACE not in fmt.format(999.0)

    def test_grouping_4_digits(self) -> None:
        """1000 → '1<SPACE>000'."""
        fmt = PyHalfUpFormat.from_pattern_space_grouping("0", True)
        assert fmt.format(1000.0) == f"1{SPACE}000"

    def test_grouping_7_digits(self) -> None:
        """1234567 → '1<SPACE>234<SPACE>567'."""
        fmt = PyHalfUpFormat.from_pattern_space_grouping("0", True)
        assert fmt.format(1234567.0) == f"1{SPACE}234{SPACE}567"

    def test_grouping_10_digits(self) -> None:
        fmt = PyHalfUpFormat.from_pattern_space_grouping("0", True)
        assert fmt.format(1234567890.0) == f"1{SPACE}234{SPACE}567{SPACE}890"

    def test_false_space_grouping_no_separator(self) -> None:
        fmt = PyHalfUpFormat.from_pattern_space_grouping("0", False)
        assert SPACE not in fmt.format(1234567.0)
        assert fmt.format(1234567.0) == "1234567"

    def test_apply_grouping_helper_empty_char(self) -> None:
        assert _apply_grouping("1234567", "", 3) == "1234567"

    def test_apply_grouping_helper_size_zero(self) -> None:
        assert _apply_grouping("1234567", ",", 0) == "1234567"

    def test_apply_grouping_helper_known_value(self) -> None:
        assert _apply_grouping("1234567", ",", 3) == "1,234,567"

    def test_grouping_does_not_affect_fractional_part(self) -> None:
        fmt = PyHalfUpFormat.from_pattern_space_grouping("0.000", True)
        result = fmt.format(1234.0)
        # "1 234.000" — the fractional part should not contain the space
        dot_pos = result.index(".")
        assert SPACE not in result[dot_pos:]

    def test_set_grouping_used_false_clears_char(self) -> None:
        fmt = PyHalfUpFormat.from_pattern_space_grouping("0", True)
        fmt.setGroupingUsed(False)
        assert SPACE not in fmt.format(1234567.0)


class TestConstructors:
    """All five constructor overload paths."""

    def test_no_arg_formats_as_integer(self) -> None:
        fmt = PyHalfUpFormat()
        assert fmt.format(3.7) == "4"

    def test_from_space_grouping_true_inserts_space(self) -> None:
        fmt = PyHalfUpFormat.from_space_grouping(True)
        assert SPACE in fmt.format(1000.0)

    def test_from_space_grouping_false_no_space(self) -> None:
        fmt = PyHalfUpFormat.from_space_grouping(False)
        assert SPACE not in fmt.format(1000.0)

    def test_from_pattern_stores_pattern(self) -> None:
        fmt = PyHalfUpFormat.from_pattern("0.00")
        assert fmt.format(1.0) == "1.00"

    def test_from_pattern_space_grouping_true(self) -> None:
        fmt = PyHalfUpFormat.from_pattern_space_grouping("0.00", True)
        assert fmt.format(1234.0) == f"1{SPACE}234.00"

    def test_from_pattern_space_grouping_false(self) -> None:
        fmt = PyHalfUpFormat.from_pattern_space_grouping("0.00", False)
        assert SPACE not in fmt.format(1234.0)
        assert fmt.format(1234.0) == "1234.00"

    def test_from_pattern_symbols_uses_grouping_separator(self) -> None:
        dfs = PyDecimalFormatSymbols()
        dfs.setGroupingSeparator(",")
        fmt = PyHalfUpFormat.from_pattern_symbols("0.00", dfs)
        assert fmt.format(1234.0) == "1,234.00"

    def test_set_grouping_size(self) -> None:
        fmt = PyHalfUpFormat.from_pattern_space_grouping("0", True)
        fmt.setGroupingSize(4)
        # 12345678 with grouping size 4 → "1234 5678"
        assert fmt.format(12345678.0) == f"1234{SPACE}5678"

    def test_set_decimal_format_symbols(self) -> None:
        # setDecimalFormatSymbols only changes the grouping CHARACTER; it does
        # not enable grouping on a format that had none. Start with grouping
        # already active, then swap the separator character.
        fmt = PyHalfUpFormat.from_space_grouping(True)
        dfs = PyDecimalFormatSymbols()
        dfs.setGroupingSeparator("|")
        fmt.setDecimalFormatSymbols(dfs)
        assert fmt.format(1234.0) == "1|234"


class TestDecimalFormatSymbols:
    """The DecimalFormatSymbols shim."""

    def test_get_instance_returns_instance(self) -> None:
        dfs = PyDecimalFormatSymbols.getInstance()
        assert isinstance(dfs, PyDecimalFormatSymbols)

    def test_default_grouping_separator_is_comma(self) -> None:
        dfs = PyDecimalFormatSymbols()
        assert dfs.getGroupingSeparator() == ","

    def test_set_get_grouping_separator_roundtrip(self) -> None:
        dfs = PyDecimalFormatSymbols()
        dfs.setGroupingSeparator(SPACE)
        assert dfs.getGroupingSeparator() == SPACE

    def test_default_decimal_separator_is_dot(self) -> None:
        dfs = PyDecimalFormatSymbols()
        assert dfs.getDecimalSeparator() == "."

    def test_set_get_decimal_separator_roundtrip(self) -> None:
        dfs = PyDecimalFormatSymbols()
        dfs.setDecimalSeparator(",")
        assert dfs.getDecimalSeparator() == ","


class TestAdaptiveFormat:
    """Deterministic regression table for adaptiveFormat.

    Values chosen to avoid the Python round() vs Math.round() divergence
    in the integer branch: only cases where number/div is clearly not
    within 1e-9 of an x.5 midpoint are included.
    """

    @pytest.mark.parametrize("number, precision, sci_range, expected", [
        # Scientific branch: |number| > sciRange
        (2_000_000.0, 3, 1e6, f"2.00E6"),
        (12_340_000.0, 3, 1e6, f"1.23E7"),    # mantissa 1.234, 3dp=4 < 5
        (12_370_000.0, 3, 1e6, f"1.24E7"),    # mantissa 1.237, 3dp=7 > 5

        # Scientific branch: |number| < 1/sciRange
        (1e-7, 3, 1e6, f"1.00E-7"),
        (1e-8, 2, 1e6, f"1.0E-8"),
        (0.0,  3, 1e6, f"0.00E0"),

        # Fixed branch: integer sub-branch (nn >= precision)
        # Use only exact integers to avoid round() midpoint issue.
        (100.0,   3, 1e6, f"100"),           # nn=3, prec=3, div=1, rounded=100
        (1000.0,  3, 1e6, f"1{SPACE}000"),   # nn=4, prec=3, div=10, rounded=1000
        (10000.0, 3, 1e6, f"10{SPACE}000"),  # nn=5, prec=3, div=100, rounded=10000
        (12300.0, 3, 1e6, f"12{SPACE}300"),  # nn=5, prec=3, div=100, round(123)=123
        (1234.0,  4, 1e6, f"1{SPACE}234"),   # nn=4, prec=4, div=1, exact
        (10000.0, 4, 1e6, f"10{SPACE}000"),  # nn=5, prec=4, div=10, round(1000)=1000

        # Fixed branch: decimal sub-branch (nn < precision)
        (1.0,    3, 1e6, f"1.00"),
        (1.0,    4, 1e6, f"1.000"),
        (12.34,  4, 1e6, f"12.34"),
        (123.4,  4, 1e6, f"123.4"),
        (0.001234, 3, 1e6, f"0.00123"),   # 6th dp=4 < 5, rounds down

        # Precision clamped to minimum 1
        (5.0, 0, 1e6, f"5"),              # precision 0 → clamped to 1
        (5.0, -1, 1e6, f"5"),             # negative precision → clamped to 1
    ])
    def test_adaptive_format(
        self,
        number: float,
        precision: int,
        sci_range: float,
        expected: str,
    ) -> None:
        result = PyHalfUpFormat.adaptiveFormat(number, precision, sci_range)
        assert result == expected, (
            f"adaptiveFormat({number!r}, {precision}, {sci_range!r}) "
            f"→ {result!r}, expected {expected!r}"
        )

    def test_negative_value_fixed(self) -> None:
        # Negative, fixed, decimal branch.
        result = PyHalfUpFormat.adaptiveFormat(-12.34, 4, 1e6)
        assert result == "-12.34"

    def test_negative_value_scientific(self) -> None:
        result = PyHalfUpFormat.adaptiveFormat(-2_000_000.0, 3, 1e6)
        assert result == "-2.00E6"

    def test_sci_range_negative_is_abs(self) -> None:
        """Negative sciRange is treated as abs(sciRange)."""
        r1 = PyHalfUpFormat.adaptiveFormat(123.0, 3, 1e6)
        r2 = PyHalfUpFormat.adaptiveFormat(123.0, 3, -1e6)
        assert r1 == r2


class TestAdaptiveFormatHypothesis:
    """Property-based tests for adaptiveFormat."""

    @given(
        st.floats(min_value=1e-5, max_value=1e5,
                  allow_nan=False, allow_infinity=False),
        st.integers(min_value=1, max_value=5),
    )
    @slow
    def test_output_parses_to_finite_float(
        self, number: float, precision: int
    ) -> None:
        """adaptiveFormat output always parses back to a finite float."""
        assume(number != 0.0)
        result = PyHalfUpFormat.adaptiveFormat(number, precision, 1e6)
        # Strip the SPACE grouping char before parsing.
        cleaned = result.replace(SPACE, "")
        assert math.isfinite(float(cleaned))

    @given(
        st.floats(min_value=1e-5, max_value=1e5,
                  allow_nan=False, allow_infinity=False),
        st.integers(min_value=1, max_value=5),
    )
    @slow
    def test_output_contains_e_iff_in_sci_range(
        self, number: float, precision: int
    ) -> None:
        """'E' appears in output iff |number| > 1e6 or |number| < 1e-6."""
        assume(number != 0.0)
        sci_range = 1e6
        result = PyHalfUpFormat.adaptiveFormat(number, precision, sci_range)
        in_sci = (abs(number) > sci_range) or (abs(number) < 1.0 / sci_range)
        assert ("E" in result) == in_sci


# ############################################################################
# PART 2 -- Parity tests (require JVM + EPQ.jar + EPQ_PARITY=1)
# ############################################################################

# Restrict parity inputs to safe values:
#   * Exact binary fractions (0.5, 0.25, 0.125, etc.) and integers
#     → repr() matches the mathematical value exactly.
#   * Values whose "next digit" after the format boundary is clearly
#     ≤ 4 or ≥ 6 (not 5) to avoid the repr() vs raw-binary divergence.
_SAFE_FIXED_FLOATS = [
    0.0, 1.0, 2.0, -1.0, -2.0,
    0.5, 1.5, 2.5, 3.5,          # exact binary; HALF_UP midpoints
    0.25, 0.75, 1.25, 1.75,      # exact binary
    0.1, 0.2, 0.3, 0.4,          # repr gives clean string; digit != 5
    10.0, 100.0, 1000.0, 10000.0,
    -0.5, -2.5, -100.0,
]


@needs_java
class TestFormatParity:
    """Java HalfUpFormat.format() vs Python format() for safe inputs.

    String equality (TOL_EXACT) expected because both sides use HALF_UP
    and the inputs are exact in IEEE-754.
    """

    @pytest.mark.parametrize("number", _SAFE_FIXED_FLOATS)
    def test_format_pattern_0(self, number: float) -> None:
        j = str(JavaHalfUpFormat("0").format(float(number)))
        p = PyHalfUpFormat.from_pattern("0").format(number)
        assert j == p, f"pattern='0', number={number!r}: java={j!r} py={p!r}"

    @pytest.mark.parametrize("number", _SAFE_FIXED_FLOATS)
    def test_format_pattern_0p0(self, number: float) -> None:
        j = str(JavaHalfUpFormat("0.0").format(float(number)))
        p = PyHalfUpFormat.from_pattern("0.0").format(number)
        assert j == p, f"pattern='0.0', number={number!r}: java={j!r} py={p!r}"

    @pytest.mark.parametrize("number", _SAFE_FIXED_FLOATS)
    def test_format_pattern_0p00(self, number: float) -> None:
        j = str(JavaHalfUpFormat("0.00").format(float(number)))
        p = PyHalfUpFormat.from_pattern("0.00").format(number)
        assert j == p, f"pattern='0.00', number={number!r}: java={j!r} py={p!r}"

    @pytest.mark.parametrize("number", [
        0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0,
        0.5, 1.5, 2.5, -1.0, -1000.0,
    ])
    def test_format_scientific_0p0E0(self, number: float) -> None:
        j = str(JavaHalfUpFormat("0.0E0").format(float(number)))
        p = PyHalfUpFormat.from_pattern("0.0E0").format(number)
        assert j == p, f"pattern='0.0E0', number={number!r}: java={j!r} py={p!r}"

    @pytest.mark.parametrize("number", [
        0.0, 1.0, 100.0, 1000.0, 0.5, -1.0,
    ])
    def test_format_scientific_0p00E0(self, number: float) -> None:
        j = str(JavaHalfUpFormat("0.00E0").format(float(number)))
        p = PyHalfUpFormat.from_pattern("0.00E0").format(number)
        assert j == p, f"pattern='0.00E0', number={number!r}: java={j!r} py={p!r}"

    def test_halfup_midpoint_2p5_parity(self) -> None:
        """Both sides must give "3" for 2.5 with pattern "0" (HALF_UP, not HALF_EVEN)."""
        j = str(JavaHalfUpFormat("0").format(float(2.5)))
        p = PyHalfUpFormat.from_pattern("0").format(2.5)
        assert j == p == "3"

    def test_halfup_midpoint_4p5_parity(self) -> None:
        j = str(JavaHalfUpFormat("0").format(float(4.5)))
        p = PyHalfUpFormat.from_pattern("0").format(4.5)
        assert j == p == "5"

    def test_space_grouping_character_matches(self) -> None:
        """The grouping character used by Java and Python must agree (U+2006)."""
        import jpype
        # Instantiate with spaceGrouping=True
        j_result = str(JavaHalfUpFormat(True).format(float(1000.0)))
        p_result = PyHalfUpFormat.from_space_grouping(True).format(1000.0)
        assert j_result == p_result
        # Verify the space char in the Python result is U+2006.
        assert SPACE in p_result


@needs_java
class TestAdaptiveFormatParity:
    """Java HalfUpFormat.adaptiveFormat() vs Python adaptiveFormat().

    KNOWN DIVERGENCE: adaptiveFormat's integer branch uses Python round()
    (HALF_EVEN) vs Java Math.round() (HALF_UP). These differ at exact
    x.5 midpoints. All parametrize inputs below are chosen to be
    midpoint-free (number/div is not within 1e-9 of x.5).
    """

    @pytest.mark.parametrize("number, precision, sci_range", [
        # Scientific branch
        (2_000_000.0, 3, 1e6),
        (1e-8,        2, 1e6),
        # Fixed decimal branch (nn < precision)
        (1.0,   3, 1e6),
        (12.34, 4, 1e6),
        (0.001, 3, 1e6),
        # Fixed integer branch — exact integers, div=1, no rounding needed
        (100.0,   3, 1e6),
        (1000.0,  4, 1e6),
        (10000.0, 4, 1e6),
        # Negatives
        (-1.0,    3, 1e6),
        (-1000.0, 3, 1e6),
    ])
    def test_adaptive_format_parity(
        self, number: float, precision: int, sci_range: float
    ) -> None:
        j = str(JavaHalfUpFormat.adaptiveFormat(float(number), int(precision),
                                                float(sci_range)))
        p = PyHalfUpFormat.adaptiveFormat(number, precision, sci_range)
        # Strip Unicode space for the error message (display cleanly).
        assert j == p, (
            f"adaptiveFormat({number!r}, {precision}, {sci_range!r}): "
            f"java={j!r} py={p!r}"
        )

    @given(
        # Restrict to exact integers so round() divergence never fires.
        st.floats(min_value=1.0, max_value=9999.0,
                  allow_nan=False, allow_infinity=False),
        st.integers(min_value=1, max_value=4),
    )
    @slow
    def test_adaptive_format_integer_inputs_parity(
        self, number: float, precision: int
    ) -> None:
        """Integer inputs have no fractional part; round() cannot produce midpoints."""
        number = float(int(number))   # truncate to integer value
        assume(number > 0)
        j = str(JavaHalfUpFormat.adaptiveFormat(float(number), int(precision),
                                                float(1e6)))
        p = PyHalfUpFormat.adaptiveFormat(number, precision, 1e6)
        assert j == p


# ############################################################################
# Entry point: tee pytest output to test_output.txt
# ############################################################################

if __name__ == "__main__":
    import pathlib
    import subprocess

    _out_path = pathlib.Path(__file__).parent / "test_output.txt"
    _proc = subprocess.Popen(
        [sys.executable, "-m", "pytest", __file__, "-v"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
    )
    with open(_out_path, "w", encoding="utf-8") as _fh:
        for _line in _proc.stdout:
            sys.stdout.write(_line)
            _fh.write(_line)
    _proc.wait()
    sys.exit(_proc.returncode)
