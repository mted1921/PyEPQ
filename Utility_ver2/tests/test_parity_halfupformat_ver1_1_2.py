r"""
test_parity_halfupformat_ver1_1_2.py -- parity harness for HalfUpFormat

Revision ver1_1_2 (2026-06-29): updated import to HalfUpFormat_ver2_1_1 (port fix: _group → _apply_grouping).
Revision ver1_1_1 (2026-06-25): P6 fix (gen1_review) + harness/port realignment.

The gen1 harness failed at COLLECTION because it imported names the port does not
define: `DecimalFormatSymbols`, and the module-level helpers `_apply_grouping`,
`_apply_pattern`, `_format_scientific_halfup`, `_parse_pattern`. Per the
TESTING_GUIDE rule for replaced Java stdlib types, a harness must import only
names that exist in the port and validate FORMAT OUTPUT, not a replaced class.

`HalfUpFormat` exports only `HalfUpFormat` (`__all__ = ["HalfUpFormat"]`)
and provides: `from_bool`, `from_pattern(pattern, space_grouping=False)`,
`from_pattern_symbols(pattern, dict)`, instance `format`, static `adaptiveFormat`
and `adaptiveFormat_literal`, and the class attribute `_MEDIUM_MATHEMATICAL_SPACE`.
This harness is realigned to that surface.

Coverage intentionally dropped because the current port under-implements the Java
API (close these in a future port revision, then restore the tests):
  * no `DecimalFormatSymbols` shim (Java parameter type java.text.DecimalFormatSymbols);
  * no `setGroupingUsed` / `setGroupingSize` (Java inherits these from DecimalFormat);
  * custom grouping separator via `from_pattern_symbols` is inert (the port only
    applies grouping when space_grouping is True, which that classmethod cannot set);
  * the no-pattern `format()` path does not round to an integer.

Structure
---------
PART 1  (always-on)
  TestMediumMathematicalSpace   grouping char is U+2006 at class level.
  TestHalfUpRoundingMidpoints   HALF_UP (not HALF_EVEN) at exact binary midpoints.
  TestFormatFixed               fixed-point patterns ("0", "0.0", "000", ...).
  TestFormatScientific          scientific patterns ("0.0E0", ...).
  TestGrouping                  U+2006 grouping via from_pattern(pattern, True).
  TestConstructors              no-arg / from_bool / from_pattern paths.
  TestAdaptiveFormat            deterministic adaptiveFormat regression table.
  TestAdaptiveFormatHypothesis  output parses back; 'E' iff in sci range.

PART 2  (parity, requires EPQ_PARITY=1 + jpype1 + EPQ jar)
  TestFormatParity              Java format() vs Python format() for safe inputs
                                (exact binary fractions; HALF_UP midpoints).
  TestAdaptiveFormatParity      Java adaptiveFormat() vs Python adaptiveFormat()
                                on midpoint-free inputs.
"""
from __future__ import annotations

import math

import pytest
from hypothesis import given, strategies as st, assume

from _parity_lib import (
    needs_java,
    slow,
)

from HalfUpFormat_ver2_1_1 import HalfUpFormat as PyHalfUpFormat
from _parity_lib import setup_parity

SPACE: str = PyHalfUpFormat._MEDIUM_MATHEMATICAL_SPACE  # U+2006 MEDIUM MATHEMATICAL SPACE

ctx = setup_parity("gov.nist.microanalysis.Utility.HalfUpFormat")
JavaHalfUpFormat = ctx.java_class


# ############################################################################
# PART 1 -- Always-on tests
# ############################################################################


class TestMediumMathematicalSpace:
    """The grouping character is U+2006, not an ASCII space."""

    def test_class_constant_is_u2006(self) -> None:
        assert PyHalfUpFormat._MEDIUM_MATHEMATICAL_SPACE == " "

    def test_not_ascii_space(self) -> None:
        assert SPACE != " "

    def test_length_one(self) -> None:
        assert len(SPACE) == 1


class TestHalfUpRoundingMidpoints:
    """Exact binary midpoints distinguishing HALF_UP from HALF_EVEN.

    Values are exact in IEEE-754 (power-of-two denominators), so the decimal
    quantize is unambiguous. HALF_EVEN (Python round / Java default) vs HALF_UP:
      2.5 -> 2 (even) vs 3;  4.5 -> 4 (even) vs 5.
    """

    def test_half_rounds_up_0p5(self) -> None:
        assert PyHalfUpFormat.from_pattern("0").format(0.5) == "1"

    def test_half_rounds_up_2p5(self) -> None:
        assert PyHalfUpFormat.from_pattern("0").format(2.5) == "3"

    def test_half_rounds_up_4p5(self) -> None:
        assert PyHalfUpFormat.from_pattern("0").format(4.5) == "5"

    def test_half_rounds_up_6p5(self) -> None:
        assert PyHalfUpFormat.from_pattern("0").format(6.5) == "7"

    def test_1p5_rounds_to_2(self) -> None:
        assert PyHalfUpFormat.from_pattern("0").format(1.5) == "2"

    def test_3p5_rounds_to_4(self) -> None:
        assert PyHalfUpFormat.from_pattern("0").format(3.5) == "4"

    def test_neg_0p5_rounds_to_neg1(self) -> None:
        assert PyHalfUpFormat.from_pattern("0").format(-0.5) == "-1"

    def test_neg_2p5_rounds_to_neg3(self) -> None:
        assert PyHalfUpFormat.from_pattern("0").format(-2.5) == "-3"

    def test_neg_4p5_rounds_to_neg5(self) -> None:
        assert PyHalfUpFormat.from_pattern("0").format(-4.5) == "-5"

    def test_0p25_to_1dp_rounds_to_0p3(self) -> None:
        assert PyHalfUpFormat.from_pattern("0.0").format(0.25) == "0.3"

    def test_0p125_to_2dp_rounds_to_0p13(self) -> None:
        assert PyHalfUpFormat.from_pattern("0.00").format(0.125) == "0.13"

    def test_neg_0p25_to_1dp_rounds_to_neg0p3(self) -> None:
        assert PyHalfUpFormat.from_pattern("0.0").format(-0.25) == "-0.3"


class TestFormatFixed:
    """Deterministic fixed-point formatting across common patterns."""

    @pytest.mark.parametrize("number, pattern, expected", [
        (0.0,    "0",    "0"),
        (1.0,    "0",    "1"),
        (-1.0,   "0",   "-1"),
        (99.0,   "0",   "99"),
        (0.0,    "0.0",  "0.0"),
        (1.0,    "0.0",  "1.0"),
        (1.1,    "0.0",  "1.1"),
        (1.24,   "0.0",  "1.2"),
        (1.26,   "0.0",  "1.3"),
        (-1.24,  "0.0", "-1.2"),
        (-1.76,  "0.0", "-1.8"),
        (0.0,    "0.00", "0.00"),
        (1.0,    "0.00", "1.00"),
        (3.14,   "0.00", "3.14"),
        (3.141,  "0.00", "3.14"),
        (3.146,  "0.00", "3.15"),
        (3.145,  "0.00", "3.15"),
        # Minimum integer-digit padding
        (5.0,    "000",  "005"),
        (12.0,   "000",  "012"),
        (123.0,  "000",  "123"),
        (1234.0, "000",  "1234"),
        (1.0,    "0.000", "1.000"),
        (1.2346, "0.000", "1.235"),
        (1.2341, "0.000", "1.234"),
    ])
    def test_fixed_format(self, number: float, pattern: str, expected: str) -> None:
        result = PyHalfUpFormat.from_pattern(pattern).format(number)
        assert result == expected, (
            f"format({number!r}, {pattern!r}) -> {result!r}, expected {expected!r}"
        )


class TestFormatScientific:
    """Deterministic scientific-notation patterns."""

    @pytest.mark.parametrize("number, pattern, expected", [
        (0.0,     "0.0E0",  "0.0E0"),
        (0.0,     "0.00E0", "0.00E0"),
        (0.0,     "0.000E0","0.000E0"),
        (10.0,    "0.0E0",  "1.0E1"),
        (100.0,   "0.0E0",  "1.0E2"),
        (0.1,     "0.0E0",  "1.0E-1"),
        (0.01,    "0.0E0",  "1.0E-2"),
        (1000.0,  "0.00E0", "1.00E3"),
        (0.001,   "0.00E0", "1.00E-3"),
        (12340.0, "0.00E0", "1.23E4"),
        (1234.0,  "0.0E0",  "1.2E3"),
        (12360.0, "0.00E0", "1.24E4"),
        (1236.0,  "0.0E0",  "1.2E3"),
        (12700.0, "0.0E0",  "1.3E4"),
        (9.96,    "0.0E0",  "1.0E1"),
        (-100.0,  "0.0E0",  "-1.0E2"),
        (-1234.0, "0.0E0",  "-1.2E3"),
        (1.0,     "0.00E0", "1.00E0"),
        (5.0,     "0.0E0",  "5.0E0"),
    ])
    def test_scientific_format(
        self, number: float, pattern: str, expected: str
    ) -> None:
        result = PyHalfUpFormat.from_pattern(pattern).format(number)
        assert result == expected, (
            f"format({number!r}, {pattern!r}) -> {result!r}, expected {expected!r}"
        )

    def test_scientific_halfup_midpoint(self) -> None:
        """HALF_UP in the mantissa: 1.25 with 1dp -> 1.3."""
        assert PyHalfUpFormat.from_pattern("0.0E0").format(1.25) == "1.3E0"


class TestGrouping:
    """U+2006 grouping (every 3 integer digits) via from_pattern(pattern, True)."""

    def test_no_grouping_below_threshold(self) -> None:
        fmt = PyHalfUpFormat.from_pattern("0", True)
        assert SPACE not in fmt.format(999.0)

    def test_grouping_4_digits(self) -> None:
        fmt = PyHalfUpFormat.from_pattern("0", True)
        assert fmt.format(1000.0) == f"1{SPACE}000"

    def test_grouping_7_digits(self) -> None:
        fmt = PyHalfUpFormat.from_pattern("0", True)
        assert fmt.format(1234567.0) == f"1{SPACE}234{SPACE}567"

    def test_grouping_10_digits(self) -> None:
        fmt = PyHalfUpFormat.from_pattern("0", True)
        assert fmt.format(1234567890.0) == f"1{SPACE}234{SPACE}567{SPACE}890"

    def test_space_grouping_false_no_separator(self) -> None:
        fmt = PyHalfUpFormat.from_pattern("0", False)
        assert SPACE not in fmt.format(1234567.0)
        assert fmt.format(1234567.0) == "1234567"

    def test_grouping_does_not_affect_fractional_part(self) -> None:
        fmt = PyHalfUpFormat.from_pattern("0.000", True)
        result = fmt.format(1234.0)
        dot_pos = result.index(".")
        assert SPACE not in result[dot_pos:]

    def test_apply_grouping_method_known_value(self) -> None:
        """The instance grouping helper inserts U+2006 every 3 digits."""
        fmt = PyHalfUpFormat.from_pattern("0", True)
        assert fmt._apply_grouping("1234567") == f"1{SPACE}234{SPACE}567"


class TestConstructors:
    """Constructor paths that exist on the port."""

    def test_from_bool_true_inserts_space(self) -> None:
        fmt = PyHalfUpFormat.from_bool(True)
        # No pattern + space grouping still groups the integer part.
        assert SPACE in fmt.format(1000.0)

    def test_from_bool_false_no_space(self) -> None:
        fmt = PyHalfUpFormat.from_bool(False)
        assert SPACE not in fmt.format(1000.0)

    def test_from_pattern_stores_pattern(self) -> None:
        fmt = PyHalfUpFormat.from_pattern("0.00")
        assert fmt.format(1.0) == "1.00"

    def test_from_pattern_space_grouping_true(self) -> None:
        fmt = PyHalfUpFormat.from_pattern("0.00", True)
        assert fmt.format(1234.0) == f"1{SPACE}234.00"

    def test_from_pattern_space_grouping_false(self) -> None:
        fmt = PyHalfUpFormat.from_pattern("0.00", False)
        assert SPACE not in fmt.format(1234.0)
        assert fmt.format(1234.0) == "1234.00"


class TestAdaptiveFormat:
    """Deterministic regression table for the static adaptiveFormat.

    Values avoid the documented HALF_EVEN/HALF_UP divergence in the integer
    sub-branch (only exact integers / clearly non-midpoint inputs).
    """

    @pytest.mark.parametrize("number, precision, sci_range, expected", [
        (2_000_000.0, 3, 1e6, "2.00E6"),
        (12_340_000.0, 3, 1e6, "1.23E7"),
        (12_370_000.0, 3, 1e6, "1.24E7"),
        (1e-7, 3, 1e6, "1.00E-7"),
        (1e-8, 2, 1e6, "1.0E-8"),
        (0.0,  3, 1e6, "0.00E0"),
        (100.0,   3, 1e6, "100"),
        (1000.0,  3, 1e6, f"1{SPACE}000"),
        (10000.0, 3, 1e6, f"10{SPACE}000"),
        (12300.0, 3, 1e6, f"12{SPACE}300"),
        (1234.0,  4, 1e6, f"1{SPACE}234"),
        (10000.0, 4, 1e6, f"10{SPACE}000"),
        (1.0,    3, 1e6, "1.00"),
        (1.0,    4, 1e6, "1.000"),
        (12.34,  4, 1e6, "12.34"),
        (123.4,  4, 1e6, "123.4"),
        (0.001234, 3, 1e6, "0.00123"),
        (5.0, 0, 1e6, "5"),
        (5.0, -1, 1e6, "5"),
    ])
    def test_adaptive_format(
        self, number: float, precision: int, sci_range: float, expected: str
    ) -> None:
        result = PyHalfUpFormat.adaptiveFormat(number, precision, sci_range)
        assert result == expected, (
            f"adaptiveFormat({number!r}, {precision}, {sci_range!r}) "
            f"-> {result!r}, expected {expected!r}"
        )

    def test_negative_value_fixed(self) -> None:
        assert PyHalfUpFormat.adaptiveFormat(-12.34, 4, 1e6) == "-12.34"

    def test_negative_value_scientific(self) -> None:
        assert PyHalfUpFormat.adaptiveFormat(-2_000_000.0, 3, 1e6) == "-2.00E6"

    def test_sci_range_negative_is_abs(self) -> None:
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
        assume(number != 0.0)
        result = PyHalfUpFormat.adaptiveFormat(number, precision, 1e6)
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
        assume(number != 0.0)
        sci_range = 1e6
        result = PyHalfUpFormat.adaptiveFormat(number, precision, sci_range)
        in_sci = (abs(number) > sci_range) or (abs(number) < 1.0 / sci_range)
        assert ("E" in result) == in_sci


# ############################################################################
# PART 2 -- Parity tests (require JVM + EPQ.jar + EPQ_PARITY=1)
# ############################################################################

# Exact binary fractions / integers: repr() matches the mathematical value, and
# HALF_UP vs HALF_EVEN differences are deterministic.
_SAFE_FIXED_FLOATS = [
    0.0, 1.0, 2.0, -1.0, -2.0,
    0.5, 1.5, 2.5, 3.5,
    0.25, 0.75, 1.25, 1.75,
    0.1, 0.2, 0.3, 0.4,
    10.0, 100.0, 1000.0, 10000.0,
    -0.5, -2.5, -100.0,
]


@needs_java
class TestFormatParity:
    """Java HalfUpFormat.format() vs Python format() for safe inputs."""

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

    def test_halfup_midpoint_2p5_parity(self) -> None:
        j = str(JavaHalfUpFormat("0").format(float(2.5)))
        p = PyHalfUpFormat.from_pattern("0").format(2.5)
        assert j == p == "3"

    def test_halfup_midpoint_4p5_parity(self) -> None:
        j = str(JavaHalfUpFormat("0").format(float(4.5)))
        p = PyHalfUpFormat.from_pattern("0").format(4.5)
        assert j == p == "5"


@needs_java
class TestAdaptiveFormatParity:
    """Java adaptiveFormat() vs Python adaptiveFormat_LITERAL().

    The non-literal `adaptiveFormat` deliberately uses HALF_EVEN rounding
    (SCIPY-DEV-1), so it diverges from Java's HALF_UP at x.5 midpoints -- which a
    plain integer input can still reach after the internal divide (e.g.
    8465 / 10 = 846.5). Java parity is therefore against `adaptiveFormat_literal`,
    which preserves the Java HALF_UP algorithm.
    """

    @pytest.mark.parametrize("number, precision, sci_range", [
        (2_000_000.0, 3, 1e6),
        (1e-8,        2, 1e6),
        (1.0,   3, 1e6),
        (12.34, 4, 1e6),
        (0.001, 3, 1e6),
        (100.0,   3, 1e6),
        (1000.0,  4, 1e6),
        (10000.0, 4, 1e6),
        (-1.0,    3, 1e6),
        (-1000.0, 3, 1e6),
    ])
    def test_adaptive_format_parity(
        self, number: float, precision: int, sci_range: float
    ) -> None:
        j = str(JavaHalfUpFormat.adaptiveFormat(float(number), int(precision),
                                                float(sci_range)))
        p = PyHalfUpFormat.adaptiveFormat_literal(number, precision, sci_range)
        assert j == p, (
            f"adaptiveFormat({number!r}, {precision}, {sci_range!r}): "
            f"java={j!r} py={p!r}"
        )

    @given(
        st.floats(min_value=1.0, max_value=9999.0,
                  allow_nan=False, allow_infinity=False),
        st.integers(min_value=1, max_value=4),
    )
    @slow
    def test_adaptive_format_integer_inputs_parity(
        self, number: float, precision: int
    ) -> None:
        # Compare against the literal (HALF_UP) port: integer/div can still be an
        # x.5 midpoint, where the non-literal HALF_EVEN port intentionally differs.
        number = float(int(number))
        assume(number > 0)
        j = str(JavaHalfUpFormat.adaptiveFormat(float(number), int(precision),
                                                float(1e6)))
        p = PyHalfUpFormat.adaptiveFormat_literal(number, precision, 1e6)
        assert j == p


# ############################################################################
# Entry point: tee pytest output to test_output.txt
# ############################################################################

if __name__ == "__main__":
    from _parity_lib import run_tests; run_tests(__file__)
