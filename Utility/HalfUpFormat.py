r"""
HalfUpFormat_ver2_1_1.py — Python port of gov.nist.microanalysis.Utility.HalfUpFormat

Guide version : 2
Generation    : 1
Port-code fixes: 1

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

CHANGES:
* Java `HalfUpFormat extends DecimalFormat`. There is no Python equivalent for
  DecimalFormat, so this is a standalone class implementing the subset of the
  DecimalFormat contract that EPQ uses: HALF_UP rounding (via the `decimal`
  module), fixed patterns ("0", "0.00", "000"), scientific patterns ("0.0E0"),
  NIST space grouping (U+2006) every 3 integer digits, and the default
  ("#,##0.###") pattern. (R10 deviation — Java class hierarchy not portable.)
* `DecimalFormatSymbols` -> a `dict` with optional "grouping_separator" /
  "decimal_separator" keys (R10 — no Python equivalent for the Java type).
* SCIPY-DEV-1: `adaptiveFormat` uses Python `round()` (HALF_EVEN); the
  Java-faithful HALF_UP version is preserved as `adaptiveFormat_literal`.

CHANGES:
* FIX-1 (API-mismatch): renamed internal helper `_group` → `_apply_grouping` so
  the test harness can call `fmt._apply_grouping(digits)` directly.
"""

from __future__ import annotations

import decimal
import math
from typing import Optional, Union

__all__ = ["HalfUpFormat"]


class HalfUpFormat:

    BUG_LEDGER: tuple = (
        ("JAVA-BUG-1", "adaptiveFormat",
         "Java uses Math.round(number/div) = floor(x+0.5) (HALF_UP), which "
         "diverges from Python round() (HALF_EVEN) at x.5 midpoints. The literal "
         "port preserves HALF_UP.", True),
        ("JAVA-BUG-2", "adaptiveFormat",
         "number == 0 makes Math.log10(0) = -Infinity; Java casts floor(-Inf) to "
         "Integer.MIN_VALUE. The path is unreachable for number==0 because the "
         "sciRange branch is taken first.", False),
    )

    serialVersionUID: int = -2503012670987467760   # R1 exception: public, no underscore
    _MEDIUM_MATHEMATICAL_SPACE: str = " "

    def __init__(self, pattern: Union[str, bool] = False,
                 arg2: Union[bool, dict, None] = None) -> None:
        self._pattern: Optional[str] = None
        self._space_grouping: bool = False
        self._grouping_sep: str = ","
        self._decimal_sep: str = "."

        if isinstance(pattern, bool):
            # HalfUpFormat() / HalfUpFormat(boolean spaceGrouping)
            self._space_grouping = pattern
        elif isinstance(pattern, str):
            self._pattern = pattern
            if isinstance(arg2, bool):
                self._space_grouping = arg2
            elif isinstance(arg2, dict):
                # HalfUpFormat(String, DecimalFormatSymbols)
                self._grouping_sep = arg2.get("grouping_separator", ",")
                self._decimal_sep = arg2.get("decimal_separator", ".")
        if self._space_grouping:
            self._grouping_sep = self._MEDIUM_MATHEMATICAL_SPACE

    # ---- classmethods preserving the Java constructor API surface ----

    @classmethod
    def from_bool(cls, space_grouping: bool) -> "HalfUpFormat":
        return cls(space_grouping)

    @classmethod
    def from_pattern(cls, pattern: str, space_grouping: bool = False) -> "HalfUpFormat":
        return cls(pattern, space_grouping)

    @classmethod
    def from_pattern_symbols(cls, pattern: str, symbols: dict) -> "HalfUpFormat":
        return cls(pattern, symbols)

    # ---- formatting ----

    def format(self, number: float) -> str:
        if math.isnan(number):
            return "NaN"
        if math.isinf(number):
            return "Infinity" if number > 0 else "-Infinity"
        if self._pattern is None:
            return self._format_default(number)
        if "E" in self._pattern:
            return self._format_scientific(number, self._pattern)
        return self._format_fixed(number, self._pattern)

    def _apply_grouping(self, int_digits: str) -> str:  # FIX-1: was _group
        """Insert the grouping separator every 3 digits from the right."""
        sep: str = self._grouping_sep
        chunks: list[str] = []
        s: str = int_digits
        while len(s) > 3:
            chunks.insert(0, s[-3:])
            s = s[:-3]
        if s:
            chunks.insert(0, s)
        return sep.join(chunks)

    def _assemble(self, rounded: decimal.Decimal, min_int: int, grouped: bool) -> str:
        s: str = format(rounded, "f")
        neg: bool = s.startswith("-")
        s_abs: str = s[1:] if neg else s
        if "." in s_abs:
            ip, fp = s_abs.split(".")
        else:
            ip, fp = s_abs, None
        ip = ip.zfill(min_int)
        if grouped:
            ip = self._apply_grouping(ip)
        res: str = ip if fp is None else f"{ip}{self._decimal_sep}{fp}"
        return f"-{res}" if neg else res

    def _format_fixed(self, number: float, pattern: str) -> str:
        if "." in pattern:
            int_pat, frac_pat = pattern.split(".")
            frac_digits: int = len(frac_pat)
            min_int: int = max(len(int_pat), 1)
        else:
            frac_digits = 0
            min_int = max(len(pattern), 1)
        with decimal.localcontext() as ctx:
            ctx.rounding = decimal.ROUND_HALF_UP
            quant: decimal.Decimal = (
                decimal.Decimal(1).scaleb(-frac_digits) if frac_digits > 0
                else decimal.Decimal(1)
            )
            rounded: decimal.Decimal = decimal.Decimal.from_float(number).quantize(quant)
        grouped: bool = self._space_grouping or ("," in pattern)
        return self._assemble(rounded, min_int, grouped)

    def _format_scientific(self, number: float, pattern: str) -> str:
        mantissa_pat: str = pattern.split("E")[0]
        frac_digits: int = len(mantissa_pat.split(".")[1]) if "." in mantissa_pat else 0
        if number == 0.0:
            # Java/DecimalFormat renders zero with a zero exponent; Python's
            # Decimal '.Ne' format would emit a non-zero exponent here.
            mant: str = "0" if frac_digits == 0 else "0." + ("0" * frac_digits)
            return f"{mant}E0"
        with decimal.localcontext() as ctx:
            ctx.rounding = decimal.ROUND_HALF_UP
            s: str = format(decimal.Decimal.from_float(number), f".{frac_digits}e")
        mant, exp = s.split("e")
        return f"{mant}E{int(exp)}"

    def _format_default(self, number: float) -> str:
        # Java default DecimalFormat pattern "#,##0.###": grouping on, min 1
        # integer digit, up to 3 fraction digits (trailing zeros trimmed).
        with decimal.localcontext() as ctx:
            ctx.rounding = decimal.ROUND_HALF_UP
            rounded: decimal.Decimal = decimal.Decimal.from_float(number).quantize(
                decimal.Decimal("0.001")
            )
        s: str = format(rounded, "f")
        neg: bool = s.startswith("-")
        s_abs: str = s[1:] if neg else s
        if "." in s_abs:
            ip, fp = s_abs.split(".")
            fp = fp.rstrip("0")
        else:
            ip, fp = s_abs, ""
        ip = self._apply_grouping(ip) if ip else "0"
        res: str = ip if not fp else f"{ip}{self._decimal_sep}{fp}"
        return f"-{res}" if neg else res

    # ---- adaptive formatting ----

    @staticmethod
    def adaptiveFormat(number: float, precision: int, sciRange: float) -> str:
        """Python-idiomatic adaptiveFormat (HALF_EVEN rounding via round()).

        SCIPY-DEV-1: diverges from Java's Math.round (HALF_UP) at x.5 midpoints.
        Use `adaptiveFormat_literal` for Java parity.
        """
        precision = max(precision, 1)
        sciRange = abs(sciRange)
        inv: float = float("inf") if sciRange == 0.0 else (1.0 / sciRange)
        if (abs(number) > sciRange) or (abs(number) < inv):
            fmt: str = "0." + ("0" * (precision - 1)) + "E0"
            return HalfUpFormat(fmt, True).format(number)
        try:
            nn: int = int(math.floor(math.log10(abs(number)))) + 1
        except ValueError:
            nn = 1   # safe default for the (unreachable) zero path
        if nn >= precision:
            fmt = "0" * nn
            div: float = math.pow(10.0, nn - precision)
            # SCIPY-DEV-1: Python round() (HALF_EVEN) rather than Java Math.round.
            rounded: float = round(number / div) * div
            return HalfUpFormat(fmt, True).format(rounded)
        fmt = ("0" * max(nn, 1)) + "." + ("0" * (precision - nn))
        return HalfUpFormat(fmt, True).format(number)

    @staticmethod
    def adaptiveFormat_literal(number: float, precision: int, sciRange: float) -> str:
        """Line-for-line port of Java adaptiveFormat (HALF_UP via Math.round)."""
        precision = max(precision, 1)
        sciRange = abs(sciRange)
        inv: float = float("inf") if sciRange == 0.0 else (1.0 / sciRange)
        if (abs(number) > sciRange) or (abs(number) < inv):
            fmt: str = "0." + ("0" * (precision - 1)) + "E0"
            return HalfUpFormat(fmt, True).format(number)
        try:
            nn: int = int(math.floor(math.log10(abs(number)))) + 1
        except ValueError:
            # JAVA-BUG-2: Java casts floor(-Infinity) to Integer.MIN_VALUE here.
            nn = -2147483648
        if nn >= precision:
            fmt = "0" * nn
            div: float = math.pow(10.0, nn - precision)
            # JAVA-BUG-1: Math.round(x) = floor(x + 0.5) (HALF_UP).
            rounded: float = float(int(math.floor((number / div) + 0.5))) * div
            return HalfUpFormat(fmt, True).format(rounded)
        fmt = ("0" * max(nn, 1)) + "." + ("0" * (precision - nn))
        return HalfUpFormat(fmt, True).format(number)

    # ---- object methods ----

    def __eq__(self, other: object) -> bool:
        return self is other

    def equals(self, other: object) -> bool:
        return self.__eq__(other)

    def __hash__(self) -> int:
        return id(self)

    def hashCode(self) -> int:
        return self.__hash__()

    def __str__(self) -> str:
        return f"HalfUpFormat(pattern={self._pattern!r}, space_grouping={self._space_grouping})"

    def toString(self) -> str:
        return self.__str__()
