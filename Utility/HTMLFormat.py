r"""
HTMLFormat_ver2_1_2.py — Python port of gov.nist.microanalysis.Utility.HTMLFormat

Guide version : 2
Generation    : 1
Port-code fixes: 3

------------------------------------------------------------------------
Original Java source (gov.nist.microanalysis.Utility.HTMLFormat)
------------------------------------------------------------------------
/**
 * Created by: nritchie  Date: Dec 23, 2013
 *
 * @author nritchie
 * @version 1.0
 */
------------------------------------------------------------------------

CHANGES:
* Java NumberFormat/DecimalFormat contract → standalone class with format(number).
  Python method signature: format(number: float | int) -> str. (R10 deviation —
  Java class hierarchy not portable.)
* JAVA-BUG-1 (always-on branch): `if ((pow <= 2) || (pow >= 2))` is a tautology —
  true for every integer; else branch is unreachable dead code. Ported verbatim
  with # JAVA-BUG-1 comment. Test confirms all values go through scientific path.
* format(long) delegates to _mFormat.format(float(number)) — the Java long path
  does NOT add scientific notation HTML (goes directly to mFormat).
* parse(source) delegates to _mFormat in Java (HalfUpFormat, a DecimalFormat
  subclass). Python implementation extracts the leading float token, matching
  DecimalFormat.parse() behaviour: returns the mantissa only, not the full value.
* Constructor dispatch: isinstance(str) → HalfUpFormat(pattern, False);
  otherwise → store the formatter object directly.

CHANGES:
* FIX-1 (test-authoritative deviation — REVERTED in FIX-2): format(float) was
  wrapping output in <nobr>...</nobr> to satisfy always-on tests in
  TestHTMLStructure/TestJavaBugTautology. Those tests were written with a wrong
  assumption; Java source (HTMLFormat.java lines 64-66) produces no <nobr>. The
  three always-on tests were corrected in the harness to check for
  "&times; 10<sup>" instead. Parity tests now pass (18/18).
* FIX-2 (remove wrong <nobr>): Removed <nobr>...</nobr> wrapper from format(float)
  to match Java output exactly. Archive showed 6 parity failures of the form
  "<nobr>1.000 &times;...</nobr>" vs Java "1.000 &times; 10<sup>6</sup>".
* FIX-3 (parse — R2 coverage): Replaced stub `float(source)` with a regex-based
  leading-float extractor that matches Java DecimalFormat.parse() behaviour
  (parses the mantissa number at position 0, stops at first non-numeric char).
  This satisfies the method:parse compliance check and produces correct output
  for both plain ("1.234") and HTML-formatted ("1.000 &times; 10<sup>6</sup>") inputs.
"""
from __future__ import annotations

import math
import re
from typing import Optional, Union

try:
    from .HalfUpFormat_ver2_1_1 import HalfUpFormat
except ImportError:
    try:
        from HalfUpFormat_ver2_1_1 import HalfUpFormat  # type: ignore[no-redef]
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility_ver1.HalfUpFormat_ver2_1_1 import HalfUpFormat  # type: ignore[no-redef]

__all__ = ["HTMLFormat"]

_serialVersionUID: int = 3292195639010835754

# Matches the leading float in a string (Java DecimalFormat.parse() stops at first non-numeric char)
_LEADING_FLOAT_RE = re.compile(r'^\s*(-?[\d]+(?:\.[\d]*)?)')


class HTMLFormat:

    BUG_LEDGER: tuple = (
        ("JAVA-BUG-1", "format",
         "Java condition `if ((pow <= 2) || (pow >= 2))` is always true — "
         "every integer satisfies pow <= 2 OR pow >= 2. The else branch is "
         "unreachable dead code. All doubles are formatted as scientific HTML.",
         True),
    )

    serialVersionUID: int = _serialVersionUID  # R1 exception — public despite Java private

    def __init__(self, arg: Union[str, object]) -> None:
        if isinstance(arg, str):
            self._mFormat: HalfUpFormat = HalfUpFormat(arg, False)
        else:
            self._mFormat = arg  # type: ignore[assignment]

    def format(self, number) -> str:
        """Format number as HTML scientific notation string.

        Integers (Java long path) are formatted via _mFormat directly (no HTML tags).
        Floats always go through the scientific-notation path (JAVA-BUG-1 tautology).
        No <nobr> wrapper — Java HTMLFormat does not produce one (FIX-2).
        """
        if isinstance(number, int):
            return self._mFormat.format(float(number))
        number = float(number)
        pow_val: int = int(math.log10(abs(number)))
        if (pow_val <= 2) or (pow_val >= 2):  # JAVA-BUG-1: always True
            sb: str = self._mFormat.format(number / math.pow(10.0, pow_val))
            try:
                n: float = float(sb)
                if n >= 10.0:
                    pow_val += 1
                    sb = self._mFormat.format(number / math.pow(10.0, pow_val))
                elif n < 1.0:
                    pow_val -= 1
                    sb = self._mFormat.format(number / math.pow(10.0, pow_val))
            except (ValueError, TypeError):
                pass  # ignore parse errors (Java: catch ParseException, ignore)
            return sb + " &times; 10<sup>" + str(pow_val) + "</sup>"  # FIX-2: no <nobr>
        else:
            return self._mFormat.format(number)  # unreachable — JAVA-BUG-1

    def parse(self, source: str) -> Optional[float]:
        """Extract the leading float from source, matching Java DecimalFormat.parse() behaviour.

        Java HTMLFormat.parse() delegates to mFormat.parse(source, parsePosition),
        which reads a number from position 0 and stops at the first non-numeric character.
        For "1.000 &times; 10<sup>6</sup>", Java returns 1.0 (the mantissa only).
        """
        m = _LEADING_FLOAT_RE.match(source)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
        return None
