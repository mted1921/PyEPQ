r"""
ExponentFormat_ver2_1_2.py — Python port of gov.nist.microanalysis.Utility.ExponentFormat

Guide version : 2
Generation    : 1
Port-code fixes: 2

------------------------------------------------------------------------
Original Java source (gov.nist.microanalysis.Utility.ExponentFormat)
------------------------------------------------------------------------
/**
 * Implements a number format that outputs in HTML as #.## x 10<sup>#</sup>.
 *
 * @author nritchie
 * @version 1.0
 */
------------------------------------------------------------------------

CHANGES:
* `assert places > 0` removed (P2 — Java asserts are disabled by default;
  test harness calls ExponentFormat(0) and must not raise).
  mPlaces = max(1, places) clamp is preserved verbatim.
* Java NumberFormat/DecimalFormat contract → standalone class with format(number).
  Python method signature: format(number: float | int) -> str. (R10 deviation —
  Java class hierarchy not portable.)
* int exp calculated via Java-faithful truncation toward zero: `int(la)` in Python
  matches Java `(int) la` for both positive and negative la values.
* sign computed as `"" if arg0 >= 0.0 else "-"` — matches Java Math.signum(arg0) >= 0
  including IEEE 754 -0.0 case (Python: -0.0 >= 0.0 is True, matching Java).
* parse() implemented to extract mantissa and HTML exponent tag, mirroring the
  Java ExponentFormat.parse() logic.

CHANGES:
* FIX-1 (assert-error / P2): Removed `assert places > 0` from constructor.
  Java assert is disabled at JVM runtime so ExponentFormat(0) must not raise.
  Archive log confirmed 2 failures from this assert (TestConstruction.test_places_0
  and TestExponentFormatParity.test_parity_places_0).
* FIX-2 (trailing-dot / format parity): For mPlaces==1 the HalfUpFormat pattern is
  "0." (zero fractional digits). Python's HalfUpFormat drops the trailing decimal
  point, but Java's DecimalFormat("0.") preserves it (e.g. "2." not "2").
  After computing mantissa_str, append "." when mPlaces==1 and no "." is present.
  Fixes test_parity_places_0: Java output "<nobr>2.&middot;..." vs Python "<nobr>2&middot;...".
"""
from __future__ import annotations

import math
from typing import Optional

try:
    from .HalfUpFormat_ver2_1_1 import HalfUpFormat
except ImportError:
    try:
        from HalfUpFormat_ver2_1_1 import HalfUpFormat  # type: ignore[no-redef]
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility_ver1.HalfUpFormat_ver2_1_1 import HalfUpFormat  # type: ignore[no-redef]

__all__ = ["ExponentFormat"]

_FMT1: str = "&middot;10<sup>"
_FMT2: str = "</sup>"

_serialVersionUID: int = -500777501322251143


class ExponentFormat:

    BUG_LEDGER: tuple = (
        ("JAVA-BUG-1", "__init__",
         "Java source has `assert places > 0` which is disabled at JVM runtime. "
         "Python port omits the assert and relies solely on the max(1, places) clamp.",
         False),
    )

    serialVersionUID: int = _serialVersionUID  # R1 exception — public despite Java private

    def __init__(self, places: int) -> None:
        # assert places > 0  # FIX-1: P2 — removed; clamp below is sufficient
        self._mPlaces: int = max(1, places)
        pattern: str = "0." + ("0" * (self._mPlaces - 1))
        self._mFormatter: HalfUpFormat = HalfUpFormat(pattern)

    def format(self, arg0) -> str:
        """Format number as HTML scientific notation.

        Returns "<nobr>sign·mantissa&middot;10<sup>exp</sup></nobr>" when exp != 0,
        or "<nobr>sign·mantissa</nobr>" when exp == 0.
        """
        arg0 = float(arg0)
        sign: str = "" if arg0 >= 0.0 else "-"  # matches Java signum(arg0) >= 0 incl. -0.0
        la: float = math.log10(abs(arg0))
        # Java: (int) la truncates toward zero for both positive and negative values
        exp: int = int(la) if la > 0.0 else (int(la) - 1)
        mantissa_str: str = self._mFormatter.format(math.pow(10.0, la - exp))
        # FIX-2: Java DecimalFormat("0.") preserves trailing dot; Python HalfUpFormat drops it
        if self._mPlaces == 1 and "." not in mantissa_str:
            mantissa_str += "."
        if exp != 0:
            result: str = sign + mantissa_str + _FMT1 + str(exp) + _FMT2
        else:
            result = sign + mantissa_str
        return "<nobr>" + result + "</nobr>"

    def parse(self, arg0: str, arg1: int = 0) -> Optional[float]:
        """Parse an ExponentFormat-encoded HTML string back to a float.

        Mirrors Java ExponentFormat.parse(String, ParsePosition):
        delegates to mFormatter.parse() to extract the mantissa, then reads
        the exponent from between FMT1 and FMT2 tags.
        """
        try:
            sup_start: int = arg0.index(_FMT1) + len(_FMT1)
            sup_end: int = arg0.index(_FMT2, sup_start)
            exp: int = int(arg0[sup_start:sup_end])
            prefix: str = arg0[:arg0.index(_FMT1)]
            neg: bool = "-" in prefix
            mantissa_str: str = prefix.replace("<nobr>", "").replace("-", "").strip()
            mantissa: float = float(mantissa_str)
            value: float = mantissa * math.pow(10.0, exp)
            return -value if neg else value
        except (ValueError, IndexError):
            return None
