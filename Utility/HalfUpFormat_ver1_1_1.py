"""
HalfUpFormat_ver1_1_1.py -- Faithful Python port of
gov.nist.microanalysis.Utility.HalfUpFormat.

Java class hierarchy: HalfUpFormat extends DecimalFormat extends NumberFormat.
Python has no DecimalFormat; this module reimplements the behaviour from
scratch using `decimal.Decimal` with ROUND_HALF_UP.

CHANGES
-------
- Inheritance from `DecimalFormat` replaced by standalone class (no equivalent
  exists in Python stdlib).
- Constructor overloads split into classmethods (R4):
    from_space_grouping, from_pattern, from_pattern_space_grouping,
    from_pattern_symbols.
- `DecimalFormatSymbols` implemented as a minimal shim in this module
  (only grouping-separator access is needed by EPQ callers).
- `NumberFormat` type annotation in `adaptiveFormat` replaced by
  `HalfUpFormat` directly (duck typing; no shim needed).
- `Math.*` calls replaced by Python `math.*` / builtins.
- MEDIUM_MATHEMATICAL_SPACE made a module-level constant (still accessible
  as `HalfUpFormat._MEDIUM_MATHEMATICAL_SPACE` for compatibility).

------------------------------------------------------------------------
Original Java source (gov.nist.microanalysis.Utility.HalfUpFormat)
------------------------------------------------------------------------
/**
 * Deals with the annoying default rounding scheme used by DecimalForamt and
 * implements the NIST suggested space grouping scheme.
 *
 * @author nicholas
 */
------------------------------------------------------------------------
"""
from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal
from typing import Optional


# ---------------------------------------------------------------------------
# Module-level constant (mirrors the private static final in Java)
# ---------------------------------------------------------------------------

_MEDIUM_MATHEMATICAL_SPACE: str = " "


# ---------------------------------------------------------------------------
# DecimalFormatSymbols shim
# ---------------------------------------------------------------------------

class DecimalFormatSymbols:
    """Minimal shim for java.text.DecimalFormatSymbols.

    Only the grouping-separator accessor is used by EPQ; all other symbol
    slots are stubbed with their default locale values.
    """

    def __init__(self) -> None:
        self._grouping_separator: str = ","
        self._decimal_separator: str = "."
        self._minus_sign: str = "-"

    @classmethod
    def getInstance(cls) -> "DecimalFormatSymbols":
        """Java: DecimalFormatSymbols.getInstance() -- returns default-locale symbols."""
        return cls()

    def setGroupingSeparator(self, char: str) -> None:
        self._grouping_separator = char

    def getGroupingSeparator(self) -> str:
        return self._grouping_separator

    def setDecimalSeparator(self, char: str) -> None:
        self._decimal_separator = char

    def getDecimalSeparator(self) -> str:
        return self._decimal_separator


# ---------------------------------------------------------------------------
# Internal formatting helpers
# ---------------------------------------------------------------------------

def _apply_grouping(int_str: str, grouping_char: str, grouping_size: int) -> str:
    """Insert grouping_char every grouping_size digits from the right."""
    if not grouping_char or grouping_size <= 0 or len(int_str) <= grouping_size:
        return int_str
    groups: list[str] = []
    s: str = int_str
    while len(s) > grouping_size:
        groups.append(s[-grouping_size:])
        s = s[:-grouping_size]
    groups.append(s)
    return grouping_char.join(reversed(groups))


def _apply_pattern(
    number: float,
    pattern: str,
    grouping_char: str = "",
    grouping_size: int = 3,
) -> str:
    """Apply a fixed-point DecimalFormat-like pattern with HALF_UP rounding.

    Handles patterns of the form ``"0"*m`` (integer) and
    ``"0"*m + "." + "0"*n`` (fixed decimal).  The number of mandatory
    digits is derived from the count of ``'0'`` characters in each section.
    """
    if "." in pattern:
        int_pat, frac_pat = pattern.split(".", 1)
        decimal_places: int = sum(1 for c in frac_pat if c in "0#")
    else:
        int_pat = pattern
        decimal_places = 0

    int_min_digits: int = max(sum(1 for c in int_pat if c == "0"), 1)

    # Round with HALF_UP
    d: Decimal = Decimal(repr(number))
    if decimal_places > 0:
        q: Decimal = Decimal("0." + "0" * decimal_places)
    else:
        q = Decimal("1")
    rounded: Decimal = d.quantize(q, rounding=ROUND_HALF_UP)

    # Render to string (quantize preserves trailing zeros)
    raw: str = str(rounded)

    # Split sign, integer, fractional parts
    if raw.startswith("-"):
        sign: str = "-"
        raw = raw[1:]
    else:
        sign = ""

    if "." in raw:
        int_str, frac_str = raw.split(".")
    else:
        int_str = raw
        frac_str = None

    # Zero-pad integer part to minimum digits
    int_str = int_str.zfill(int_min_digits)

    # Apply grouping
    int_str = _apply_grouping(int_str, grouping_char, grouping_size)

    result: str = sign + int_str
    if frac_str is not None:
        result += "." + frac_str
    return result


def _format_scientific_halfup(
    number: float,
    decimal_places: int,
    grouping_char: str = "",
) -> str:
    """Format *number* in Java-style scientific notation with HALF_UP rounding.

    Output format: ``"1.23E5"``, ``"-1.23E-3"``, ``"0.00E0"``.
    Rules (matching Java DecimalFormat with pattern ``"0.???E0"``):
      * Exactly one digit before the decimal point.
      * No '+' sign on positive exponents.
      * Minimum one digit in the exponent (no leading zeros beyond that).
      * Uppercase ``E``.
    """
    if number == 0.0:
        mantissa_str: str = "0"
        if decimal_places > 0:
            mantissa_str += "." + "0" * decimal_places
        return mantissa_str + "E0"

    exp: int = int(math.floor(math.log10(abs(number))))
    mantissa: float = number / (10.0 ** exp)

    # Round mantissa with HALF_UP
    d: Decimal = Decimal(repr(mantissa))
    q: Decimal = Decimal("0." + "0" * decimal_places) if decimal_places > 0 else Decimal("1")
    rounded: Decimal = d.quantize(q, rounding=ROUND_HALF_UP)

    # Handle carry-over (e.g. 9.999... rounds to 10.000)
    if abs(float(rounded)) >= 10.0:
        exp += 1
        rounded = (rounded / Decimal("10")).quantize(q, rounding=ROUND_HALF_UP)

    # Render mantissa
    raw: str = str(rounded)
    if decimal_places > 0 and "." not in raw:
        raw += "." + "0" * decimal_places

    # Grouping on the mantissa integer part (only 1 digit in the standard pattern;
    # included for completeness when a multi-digit integer pattern is used).
    if "." in raw:
        m_int, m_frac = raw.split(".")
    else:
        m_int, m_frac = raw, None

    sign: str = ""
    if m_int.startswith("-"):
        sign = "-"
        m_int = m_int[1:]

    m_int = _apply_grouping(m_int, grouping_char, 3)
    mantissa_str = sign + m_int + ("." + m_frac if m_frac is not None else "")

    return f"{mantissa_str}E{exp}"


def _parse_pattern(pattern: str) -> tuple[bool, int]:
    """Return (is_scientific, decimal_places) for a DecimalFormat pattern."""
    upper: str = pattern.upper()
    scientific: bool = "E" in upper
    main: str = pattern[: upper.index("E")] if scientific else pattern
    if "." in main:
        decimal_places = sum(1 for c in main.split(".", 1)[1] if c in "0#")
    else:
        decimal_places = 0
    return scientific, decimal_places


# ---------------------------------------------------------------------------
# HalfUpFormat
# ---------------------------------------------------------------------------

class HalfUpFormat:
    """Port of gov.nist.microanalysis.Utility.HalfUpFormat.

    DecimalFormat with HALF_UP rounding and optional NIST space-grouping.
    Java's default rounding mode is HALF_EVEN (banker's); this class
    overrides it to HALF_UP so that 0.5, 1.5, 2.5 … all round away from zero.
    """

    _MEDIUM_MATHEMATICAL_SPACE: str = _MEDIUM_MATHEMATICAL_SPACE

    # ------------------------------------------------------------------
    # Constructor (no-arg) and classmethods for overloads (R4)
    # ------------------------------------------------------------------

    def __init__(
        self,
        pattern: str = "",
        grouping_char: str = "",
        grouping_size: int = 3,
    ) -> None:
        """Base initialiser; called by all classmethods after resolving args."""
        self._pattern: str = pattern
        self._grouping_char: str = grouping_char
        self._grouping_size: int = grouping_size if grouping_char else 0

    @classmethod
    def from_space_grouping(cls, spaceGrouping: bool) -> "HalfUpFormat":
        """Java: HalfUpFormat(boolean spaceGrouping)."""
        if spaceGrouping:
            return cls("", grouping_char=_MEDIUM_MATHEMATICAL_SPACE, grouping_size=3)
        return cls()

    @classmethod
    def from_pattern(cls, pattern: str) -> "HalfUpFormat":
        """Java: HalfUpFormat(String pattern)."""
        return cls(pattern)

    @classmethod
    def from_pattern_space_grouping(cls, pattern: str, spaceGrouping: bool) -> "HalfUpFormat":
        """Java: HalfUpFormat(String pattern, boolean spaceGrouping)."""
        if spaceGrouping:
            return cls(pattern, grouping_char=_MEDIUM_MATHEMATICAL_SPACE, grouping_size=3)
        return cls(pattern)

    @classmethod
    def from_pattern_symbols(
        cls,
        pattern: str,
        symbols: "DecimalFormatSymbols",
    ) -> "HalfUpFormat":
        """Java: HalfUpFormat(String pattern, DecimalFormatSymbols symbols)."""
        return cls(pattern, grouping_char=symbols.getGroupingSeparator(), grouping_size=3)

    # ------------------------------------------------------------------
    # Setters (called by Java constructors via super(); exposed for parity)
    # ------------------------------------------------------------------

    def setGroupingSize(self, size: int) -> None:
        self._grouping_size = size

    def setGroupingUsed(self, used: bool) -> None:
        if not used:
            self._grouping_char = ""

    def setDecimalFormatSymbols(self, symbols: "DecimalFormatSymbols") -> None:
        self._grouping_char = symbols.getGroupingSeparator()

    # ------------------------------------------------------------------
    # format
    # ------------------------------------------------------------------

    def format(self, number: float) -> str:
        """Format *number* according to the stored pattern with HALF_UP rounding."""
        if not self._pattern:
            # No pattern: format with default repr and HALF_UP (no decimal places).
            return _apply_pattern(number, "0", self._grouping_char, self._grouping_size)

        scientific, decimal_places = _parse_pattern(self._pattern)
        if scientific:
            return _format_scientific_halfup(number, decimal_places, self._grouping_char)
        return _apply_pattern(
            number, self._pattern, self._grouping_char, self._grouping_size
        )

    # ------------------------------------------------------------------
    # adaptiveFormat
    # ------------------------------------------------------------------

    @staticmethod
    def adaptiveFormat(number: float, precision: int, sciRange: float) -> str:
        """Adaptively format *number* to *precision* significant digits.

        Switches to scientific notation when ``|number| > sciRange`` or
        ``|number| < 1/sciRange``; otherwise uses fixed-point.  Thousands
        are grouped with MEDIUM MATHEMATICAL SPACE (U+2006).

        Parameters
        ----------
        number    : the value to format
        precision : number of significant digits (clamped to minimum 1)
        sciRange  : threshold for switching to scientific notation (e.g. 1e6)
        """
        precision = max(precision, 1)
        sciRange = abs(sciRange)

        if (abs(number) > sciRange) or (abs(number) < (1.0 / sciRange)):
            # Scientific branch: pattern "0." + "0"*(precision-1) + "E0"
            decimal_places: int = precision - 1
            fmt: HalfUpFormat = HalfUpFormat.from_pattern_space_grouping(
                "0." + "0" * decimal_places + "E0", True
            )
            return fmt.format(number)

        # Fixed branch
        nn: int = int(math.floor(math.log10(abs(number)))) + 1
        if nn >= precision:
            # Round to *precision* significant figures, then format as integer.
            div: float = math.pow(10.0, nn - precision)
            # Java uses Math.round() = floor(x + 0.5), which differs from
            # Python's round() (HALF_EVEN) for even-number tie cases e.g.
            # round(968.5)=968 in Python but Math.round(968.5)=969 in Java.
            rounded: float = math.floor(number / div + 0.5) * div
            pattern: str = "0" * nn
            huf: HalfUpFormat = HalfUpFormat.from_pattern_space_grouping(pattern, True)
            return huf.format(rounded)
        else:
            # Fixed decimal: integer digits + fractional digits.
            int_digits: int = max(nn, 1)
            frac_digits: int = precision - nn
            pattern = "0" * int_digits + "." + "0" * frac_digits
            huf = HalfUpFormat.from_pattern_space_grouping(pattern, True)
            return huf.format(number)
