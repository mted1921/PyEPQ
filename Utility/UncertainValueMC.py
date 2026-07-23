r"""
UncertainValueMC_ver2_1_1.py — Python port of
gov.nist.microanalysis.Utility.UncertainValueMC

Guide version : 2
Generation    : 1
Port-code fixes: 1

CHANGES
-------
FIX-1 (assert-error): log_literal() and exp_literal() called math.log(n._mRandVal)
  which raises ValueError for non-positive inputs. Java Math.log() returns NaN (negative
  arg) or -Infinity (zero arg) via IEEE 754. Added _java_log() helper that matches Java
  semantics. Category: assert-error.

------------------------------------------------------------------------
Original Java source (gov.nist.microanalysis.Utility.UncertainValueMC)
------------------------------------------------------------------------
/**
 * <p>
 * A class to assist in implementing a MonteCarlo method of calculating the
 * uncertainty distribution associated with propagation of uncertainties. All
 * uncertainties are assumed to be normally distributed.
 * </p>
 * <p>
 * Copyright: Pursuant to title 17 Section 105 of the United States Code this
 * software is not subject to copyright protection and is in the public domain
 * </p>
 * <p>
 * Institution: National Institute of Standards and Technology
 * </p>
 *
 * @author nritchie
 * @version 1.0
 */
------------------------------------------------------------------------
"""
from __future__ import annotations

import math
import numbers
from typing import Dict, List, Optional, Sequence, Union

try:
    from ._epq_compat import EPQException, F64Array, JavaRandom
except ImportError:
    try:
        from _epq_compat import EPQException, F64Array, JavaRandom  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility._epq_compat import EPQException, F64Array, JavaRandom  # type: ignore

try:
    from .UncertainValue2_ver2_1_0 import UncertainValue2
except ImportError:
    try:
        from UncertainValue2_ver2_1_0 import UncertainValue2  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility_ver2.UncertainValue2_ver2_1_0 import UncertainValue2  # type: ignore

def _java_log(x: float) -> float:
    """FIX-1: match Java Math.log() IEEE 754 semantics: NaN for x<0, -Inf for x==0."""
    if x > 0.0:
        return math.log(x)
    return float("-inf") if x == 0.0 else float("nan")


BUG_LEDGER: tuple = (
    (
        "JAVA-BUG-1",
        "exp",
        "Java: `return new UncertainValueMC(Math.log(n.mValue), Math.log(n.mRandVal), null);` "
        "inside the method named `exp`.  The implementation is a copy-paste of the adjacent "
        "`log` method — it applies `Math.log` instead of `Math.exp` to both components.  "
        "Preserved verbatim; `exp_strict` uses `math.exp` for both components.",
    ),
    (
        "JAVA-BUG-2",
        "pow",
        "Java: `return new UncertainValueMC(Math.pow(n.mValue, exp.mRandVal), ..., null);` "
        "The nominal component uses the exponent's random sample `exp.mRandVal` instead of "
        "its nominal value `exp.mValue`.  Preserved verbatim; `pow_strict` uses "
        "`math.pow(n._mValue, exp._mValue)` for the nominal.",
    ),
)


class UncertainValueMC(numbers.Number):
    """Python port of ``gov.nist.microanalysis.Utility.UncertainValueMC``.

    Each instance stores a nominal value and a single Monte-Carlo random
    sample.  Static algebra methods build new instances propagating both
    components through each operation.
    """

    serialVersionUID: int = 8528835028857799252  # R1 exception: no _ prefix

    # Class-level RNG — seeded from wall-clock at import (mirrors Java's
    # ``private static final Random sRandom = new Random(System.currentTimeMillis())``)
    _sRandom: JavaRandom = JavaRandom()

    # ------------------------------------------------------------------
    # Internal factory (Java: private 3-arg constructor)
    # ------------------------------------------------------------------

    @classmethod
    def _internal(cls, v: float, randVal: float) -> "UncertainValueMC":
        """Java: ``private UncertainValueMC(double v, double randVal, Object obj)``.
        The ``obj`` argument was always ``null``; dropped (assert obj==null also dropped).
        """
        obj: "UncertainValueMC" = object.__new__(cls)
        obj._mValue = float(v)
        obj._mRandVal = float(randVal)
        return obj

    # ------------------------------------------------------------------
    # Constructor dispatch
    # ------------------------------------------------------------------

    def __init__(
        self,
        v: Union[float, UncertainValue2],
        dv_or_deviates: Union[float, Dict[str, float]],
    ) -> None:
        if isinstance(v, UncertainValue2):
            # Java: public UncertainValueMC(UncertainValue2 uv, Map<String,Double> deviates)
            uv: UncertainValue2 = v
            deviates: Dict[str, float] = dv_or_deviates  # type: ignore[assignment]
            self._mValue = uv.doubleValue()
            self._mRandVal = uv.doubleValue() + UncertainValueMC._calculateDeviate(
                uv, deviates
            )
        else:
            # Java: public UncertainValueMC(double v, double dv)
            v_float: float = float(v)
            dv: float = float(dv_or_deviates)  # type: ignore[arg-type]
            self._mValue = v_float
            self._mRandVal = dv * UncertainValueMC._normalDeviate()

    # ------------------------------------------------------------------
    # Private static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalDeviate() -> float:
        """Java: ``static private double normalDeviate()``"""
        return UncertainValueMC._sRandom.nextGaussian()

    @staticmethod
    def _calculateDeviate(
        uv2: UncertainValue2, deviate: Dict[str, float]
    ) -> float:
        """Java: ``static private double calculateDeviate(UncertainValue2, Map)``.
        Mutates *deviate* by inserting missing component deviates.
        """
        randVal: float = 0.0
        for comp in uv2.getComponentNames():
            if comp not in deviate:
                deviate[comp] = UncertainValueMC._normalDeviate()
            randVal += deviate[comp] * uv2.getComponent(comp)
        return randVal

    # ------------------------------------------------------------------
    # Static algebra (public)
    # ------------------------------------------------------------------

    @staticmethod
    def add_combo_literal(
        a: float,
        uva: "UncertainValueMC",
        b: float,
        uvb: "UncertainValueMC",
    ) -> "UncertainValueMC":
        """Java: ``static UncertainValueMC add(double a, UVMC uva, double b, UVMC uvb)``"""
        return UncertainValueMC._internal(
            a * uva._mValue + b * uvb._mValue,
            a * uva._mRandVal + b * uvb._mRandVal,
        )

    @staticmethod
    def add_combo(
        a: float,
        uva: "UncertainValueMC",
        b: float,
        uvb: "UncertainValueMC",
    ) -> "UncertainValueMC":  # SCIPY-NONE: no library substitute
        return UncertainValueMC.add_combo_literal(a, uva, b, uvb)

    @staticmethod
    def sum_literal(vals: Sequence["UncertainValueMC"]) -> "UncertainValueMC":
        """Java: ``static UncertainValueMC sum(UncertainValueMC[] vals)``"""
        v: float = 0.0
        rv: float = 0.0
        for i in range(len(vals)):
            v += vals[i]._mValue
            rv += vals[i]._mRandVal
        return UncertainValueMC._internal(v, rv)

    @staticmethod
    def sum(vals: Sequence["UncertainValueMC"]) -> "UncertainValueMC":  # SCIPY-NONE: no library substitute
        return UncertainValueMC.sum_literal(vals)

    @staticmethod
    def mean_literal(vals: Sequence["UncertainValueMC"]) -> "UncertainValueMC":
        """Java: ``static UncertainValueMC mean(UncertainValueMC[] vals)``"""
        return UncertainValueMC.multiply_sv(1.0 / len(vals), UncertainValueMC.sum(vals))

    @staticmethod
    def mean(vals: Sequence["UncertainValueMC"]) -> "UncertainValueMC":  # SCIPY-NONE: no library substitute
        return UncertainValueMC.mean_literal(vals)

    @staticmethod
    def add_vv_literal(
        v1: "UncertainValueMC", v2: "UncertainValueMC"
    ) -> "UncertainValueMC":
        """Java: ``static UncertainValueMC add(UVMC v1, UVMC v2)``"""
        return UncertainValueMC._internal(
            v1._mValue + v2._mValue, v1._mRandVal + v2._mRandVal
        )

    @staticmethod
    def add_vv(
        v1: "UncertainValueMC", v2: "UncertainValueMC"
    ) -> "UncertainValueMC":  # SCIPY-NONE: no library substitute
        return UncertainValueMC.add_vv_literal(v1, v2)

    @staticmethod
    def subtract_literal(
        v1: "UncertainValueMC", v2: "UncertainValueMC"
    ) -> "UncertainValueMC":
        """Java: ``static UncertainValueMC subtract(UVMC v1, UVMC v2)``"""
        return UncertainValueMC._internal(
            v1._mValue - v2._mValue, v1._mRandVal - v2._mRandVal
        )

    @staticmethod
    def subtract(
        v1: "UncertainValueMC", v2: "UncertainValueMC"
    ) -> "UncertainValueMC":  # SCIPY-NONE: no library substitute
        return UncertainValueMC.subtract_literal(v1, v2)

    @staticmethod
    def multiply_vv_literal(
        v1: "UncertainValueMC", v2: "UncertainValueMC"
    ) -> "UncertainValueMC":
        """Java: ``static UncertainValueMC multiply(UVMC v1, UVMC v2)``"""
        return UncertainValueMC._internal(
            v1._mValue * v2._mValue, v1._mRandVal * v2._mRandVal
        )

    @staticmethod
    def multiply_vv(
        v1: "UncertainValueMC", v2: "UncertainValueMC"
    ) -> "UncertainValueMC":  # SCIPY-NONE: no library substitute
        return UncertainValueMC.multiply_vv_literal(v1, v2)

    @staticmethod
    def multiply_sv_literal(v1: float, v2: "UncertainValueMC") -> "UncertainValueMC":
        """Java: ``static UncertainValueMC multiply(double v1, UVMC v2)``"""
        return UncertainValueMC._internal(v1 * v2._mValue, v1 * v2._mRandVal)

    @staticmethod
    def multiply_sv(v1: float, v2: "UncertainValueMC") -> "UncertainValueMC":  # SCIPY-NONE: no library substitute
        return UncertainValueMC.multiply_sv_literal(v1, v2)

    @staticmethod
    def divide_vv_literal(
        v1: "UncertainValueMC", v2: "UncertainValueMC"
    ) -> "UncertainValueMC":
        """Java: ``static UncertainValueMC divide(UVMC v1, UVMC v2)``"""
        return UncertainValueMC._internal(
            v1._mValue / v2._mValue, v1._mRandVal / v2._mRandVal
        )

    @staticmethod
    def divide_vv(
        v1: "UncertainValueMC", v2: "UncertainValueMC"
    ) -> "UncertainValueMC":  # SCIPY-NONE: no library substitute
        return UncertainValueMC.divide_vv_literal(v1, v2)

    @staticmethod
    def divide_vs_literal(v1: "UncertainValueMC", v2: float) -> "UncertainValueMC":
        """Java: ``static UncertainValueMC divide(UVMC v1, double v2)``"""
        return UncertainValueMC._internal(v1._mValue / v2, v1._mRandVal / v2)

    @staticmethod
    def divide_vs(v1: "UncertainValueMC", v2: float) -> "UncertainValueMC":  # SCIPY-NONE: no library substitute
        return UncertainValueMC.divide_vs_literal(v1, v2)

    @staticmethod
    def pow_literal(n: "UncertainValueMC", exp: "UncertainValueMC") -> "UncertainValueMC":
        """Java: ``static UncertainValueMC pow(UVMC n, UVMC exp)``

        .. note::
            **JAVA-BUG-2** — nominal uses ``exp.mRandVal`` (the random sample) rather
            than ``exp.mValue`` (the nominal).  Preserved verbatim.  Use
            :meth:`pow_strict` for the corrected behaviour.
        """
        return UncertainValueMC._internal(
            math.pow(n._mValue, exp._mRandVal),  # JAVA-BUG-2: should be exp._mValue
            math.pow(n._mRandVal, exp._mRandVal),
        )

    @staticmethod
    def pow(n: "UncertainValueMC", exp: "UncertainValueMC") -> "UncertainValueMC":  # SCIPY-NONE: no library substitute
        return UncertainValueMC.pow_literal(n, exp)

    @staticmethod
    def pow_strict(n: "UncertainValueMC", exp: "UncertainValueMC") -> "UncertainValueMC":
        """Bug-corrected ``pow``: uses nominal exponent ``exp._mValue`` for the nominal."""
        return UncertainValueMC._internal(
            math.pow(n._mValue, exp._mValue),
            math.pow(n._mRandVal, exp._mRandVal),
        )

    @staticmethod
    def log_literal(n: "UncertainValueMC") -> "UncertainValueMC":
        """Java: ``static UncertainValueMC log(UVMC n)``"""
        return UncertainValueMC._internal(math.log(n._mValue), _java_log(n._mRandVal))  # FIX-1

    @staticmethod
    def log(n: "UncertainValueMC") -> "UncertainValueMC":  # SCIPY-NONE: no library substitute
        return UncertainValueMC.log_literal(n)

    @staticmethod
    def exp_literal(n: "UncertainValueMC") -> "UncertainValueMC":
        """Java: ``static UncertainValueMC exp(UVMC n)``

        .. note::
            **JAVA-BUG-1** — applies ``Math.log`` (not ``Math.exp``) to both components;
            copy-paste of the ``log`` method.  Preserved verbatim.  Use
            :meth:`exp_strict` for the corrected behaviour.
        """
        return UncertainValueMC._internal(
            math.log(n._mValue),    # JAVA-BUG-1: should be math.exp
            _java_log(n._mRandVal), # JAVA-BUG-1: should be math.exp; FIX-1: guard non-positive
        )

    @staticmethod
    def exp(n: "UncertainValueMC") -> "UncertainValueMC":  # SCIPY-NONE: no library substitute
        return UncertainValueMC.exp_literal(n)

    @staticmethod
    def exp_strict(n: "UncertainValueMC") -> "UncertainValueMC":
        """Bug-corrected ``exp``: applies ``math.exp`` (not ``math.log``)."""
        return UncertainValueMC._internal(math.exp(n._mValue), math.exp(n._mRandVal))

    @staticmethod
    def sqrt_literal(n: "UncertainValueMC") -> "UncertainValueMC":
        """Java: ``static UncertainValueMC sqrt(UVMC n)``"""
        return UncertainValueMC._internal(
            math.sqrt(n._mValue), math.sqrt(max(0.0, n._mRandVal))
        )

    @staticmethod
    def sqrt(n: "UncertainValueMC") -> "UncertainValueMC":  # SCIPY-NONE: no library substitute
        return UncertainValueMC.sqrt_literal(n)

    # ------------------------------------------------------------------
    # Instance methods
    # ------------------------------------------------------------------

    def abs_literal(self) -> "UncertainValueMC":
        return UncertainValueMC._internal(abs(self._mValue), abs(self._mRandVal))

    def abs(self) -> "UncertainValueMC":  # SCIPY-NONE: no library substitute
        return self.abs_literal()

    def nonNegative_literal(self) -> "UncertainValueMC":
        """Identical body to ``abs()`` (Java source confirmed — not a bug)."""
        return UncertainValueMC._internal(abs(self._mValue), abs(self._mRandVal))

    def nonNegative(self) -> "UncertainValueMC":  # SCIPY-NONE: no library substitute
        return self.nonNegative_literal()

    def nominalValue(self) -> float:
        return self._mValue

    def doubleValue(self) -> float:
        """Returns the random sample ``_mRandVal`` (NOT the nominal ``_mValue``)."""
        return self._mRandVal

    def __float__(self) -> float:
        return self._mRandVal

    def floatValue(self) -> float:
        return float(self._mRandVal)

    def intValue(self) -> int:
        return int(self._mRandVal)

    def __int__(self) -> int:
        return int(self._mRandVal)

    def longValue(self) -> int:
        return int(self._mRandVal)

    def __str__(self) -> str:
        return str(float(self._mRandVal))

    def toString(self) -> str:
        return str(self)
