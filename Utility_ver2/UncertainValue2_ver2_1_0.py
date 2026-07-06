r"""
UncertainValue2_ver2_1_0.py — Python port of gov.nist.microanalysis.Utility.UncertainValue2

Guide version : 2
Generation    : 1
Port-code fixes: 0

------------------------------------------------------------------------
Original Java source (gov.nist.microanalysis.Utility.UncertainValue2)
------------------------------------------------------------------------
/**
 * <p>
 * The UncertainValue2 class implements a class for handling values with zero or
 * more component normally distributed uncertainties. The class implements a
 * number of static methods for performing basic mathematical operations on
 * numbers while propagating the component uncertainties.
 * </p>
 * <p>
 * Each uncertain value is represented by a value and a series of named
 * component uncertainties. The component names identify the source of the
 * uncertainty. Uncertainty components associated with the same name are assumed
 * to be 100% correlated (r=1.0) and are accumulated each time an operation is
 * performed. Uncertainties associated with different names are accumulated
 * separately. The named uncertainties can be reduced to a single uncertainty as
 * a final step. This step may either assume the components are independent or
 * correlated using the Correlation imbedded class.
 * </p>
 * <p>
 * Each component of uncertainty propagates through the mathematical operations
 * as though it was the only source of uncertainty.
 * </p>
 * <p>
 * Copyright: Pursuant to title 17 Section 105 of the United States Code this
 * software is not subject to copyright protection and is in the public domain
 * </p>
 * <p>
 * Institution: National Institute of Standards and Technology
 * </p>
 *
 * @author nicholas
 * @version 1.0
 */
------------------------------------------------------------------------

CHANGES:
* Extends `numbers.Number` in place of `java.lang.Number` (R2).
* Implements `Comparable` via `__lt__`, `__le__`, `__gt__`, `__ge__`, `compareTo()` (R2).
* `toString` → `__str__` + `toString()` alias (R2).
* `equals` → `__eq__` + `equals()` alias; `hashCode` → `__hash__` + `hashCode()` alias (R2).
* `clone()` also exposed as `__copy__` for Python copy protocol.
* `intValue()`/`doubleValue()` also exposed as `__int__`/`__float__` (R2).
* Inner class `Correlations.Key` replaced by `frozenset` dictionary keys —
  symmetric pair lookup without a separate Key class.
* `serialVersionUID` discarded per spec.
* Java `assert` in `multiply_dn`, `exp`, `Correlations.add` skipped (disabled at JVM
  runtime); `Math2.bound` in `Correlations.add` is the only production behaviour.
* `from_sigmas` classmethod replaces the `(double v, Map<String,Double> sigmas)`
  constructor (bypasses `_sDefIndex` increment, matches Java) (R4).
* `add` six overloads split as `add_collection/add_array/add_dadb/add_nd/add_dn/add_nn` (R4).
* `normalize`: primary uses numpy matrix ops; `normalize_literal` uses JamaMatrix.
* All other mathematical static and instance methods: primary delegates to `_literal`
  companion with `# SCIPY-NONE` marker.
* R8: `Math.signum(b.mValue)` in `quadratic` → `0.0 if ... == 0.0 else copysign(1.0, ...)`.
* `Correlations` exposed at module level as `Correlations = UncertainValue2.Correlations` (R2).

JAVA-BUG-1 (preserved verbatim):
  `uncertainty_comps` sums raw component values, not their squares.
  Java source: `sum2 += getComponent(comp);`
  Returns sqrt(sum-of-values) instead of sqrt(sum-of-squares). Companion
  `uncertainty_comps_strict()` provides the corrected L2 norm.

JAVA-BUG-2 (preserved verbatim):
  `equals_tol` raises KeyError when a key is present in one but not both maps.
  Java source: `if (Math.abs(mSigmas.get(key) - other.mSigmas.get(key)) >= tolerance)`
  Java `TreeMap.get` returns null for missing key; unboxing null causes NPE.
  Python `dict[key]` raises KeyError. Companion `equals_tol_strict()` uses `.get(k, 0.0)`.

JAVA-BUG-3 (not in spec, preserved verbatim):
  Static `uncertainty_n(Number n)` returns `doubleValue()` instead of `uncertainty()`.
  Java source: `return n instanceof UncertainValue2 ? ((UncertainValue2)n).doubleValue() : 0.0;`
  Companion `uncertainty_n_strict()` returns the uncertainty correctly.

BUG_LEDGER: tuple = (
    ("JAVA-BUG-1", "uncertainty_comps",
     "Java: `sum2 += getComponent(comp)` sums raw values not squares. "
     "Returns sqrt(sum) instead of sqrt(sum-of-squares). Preserved verbatim; "
     "`uncertainty_comps_strict` provides the correct L2 norm."),
    ("JAVA-BUG-2", "equals_tol",
     "Java: `mSigmas.get(key)` returns null for missing keys; unboxing null raises NPE. "
     "Python port raises KeyError identically. Preserved verbatim; "
     "`equals_tol_strict` uses `.get(key, 0.0)`."),
    ("JAVA-BUG-3", "uncertainty_n",
     "Java static `uncertainty(Number n)` returns `((UncertainValue2)n).doubleValue()` "
     "instead of `uncertainty()`. Preserved verbatim; `uncertainty_n_strict` returns "
     "the correct uncertainty."),
)
"""
from __future__ import annotations

import copy
import math
import numbers
from typing import Callable, Optional, Union

import numpy as np

try:
    from ._epq_compat import EPQException, JamaMatrix, F64Array
except ImportError:
    try:
        from _epq_compat import EPQException, JamaMatrix, F64Array  # type: ignore[no-redef]
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility._epq_compat import EPQException, JamaMatrix, F64Array  # type: ignore[no-redef]

try:
    from .Math2_ver8_1_5 import Math2
except ImportError:
    try:
        from Math2_ver8_1_5 import Math2  # type: ignore[no-redef]
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility.Math2_ver8_1_5 import Math2  # type: ignore[no-redef]

try:
    from .HalfUpFormat_ver2_1_1 import HalfUpFormat
except ImportError:
    try:
        from HalfUpFormat_ver2_1_1 import HalfUpFormat  # type: ignore[no-redef]
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility.HalfUpFormat_ver2_1_1 import HalfUpFormat  # type: ignore[no-redef]

try:
    from .ExponentFormat_ver2_1_2 import ExponentFormat
except ImportError:
    try:
        from ExponentFormat_ver2_1_2 import ExponentFormat  # type: ignore[no-redef]
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility.ExponentFormat_ver2_1_2 import ExponentFormat  # type: ignore[no-redef]

try:
    from .UtilException_ver2_1_1 import UtilException
except ImportError:
    try:
        from UtilException_ver2_1_1 import UtilException  # type: ignore[no-redef]
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility.UtilException_ver2_1_1 import UtilException  # type: ignore[no-redef]

__all__ = ["UncertainValue2", "Correlations"]

BUG_LEDGER: tuple = (
    ("JAVA-BUG-1", "uncertainty_comps",
     "Java: `sum2 += getComponent(comp)` sums raw values not squares. "
     "Returns sqrt(sum) instead of sqrt(sum-of-squares). Preserved verbatim; "
     "`uncertainty_comps_strict` provides the correct L2 norm."),
    ("JAVA-BUG-2", "equals_tol",
     "Java: `mSigmas.get(key)` returns null for missing keys; unboxing null raises NPE. "
     "Python port raises KeyError identically. Preserved verbatim; "
     "`equals_tol_strict` uses `.get(key, 0.0)`."),
    ("JAVA-BUG-3", "uncertainty_n",
     "Java static `uncertainty(Number n)` returns `((UncertainValue2)n).doubleValue()` "
     "instead of `uncertainty()`. Preserved verbatim; `uncertainty_n_strict` returns "
     "the correct uncertainty."),
)


class UncertainValue2(numbers.Number):
    """Scalar value with one or more named normally-distributed uncertainty components.

    Components sharing the same name are 100%-correlated (accumulated linearly).
    Components with different names are independent (accumulated in quadrature).
    """

    # ------------------------------------------------------------------
    # Class-level state
    # ------------------------------------------------------------------

    DEFAULT: str = "Default"
    _sDefIndex: int = 0  # private static transient long sDefIndex = 0

    # Class-level sentinel constants — populated after class definition
    ONE: "UncertainValue2"
    ZERO: "UncertainValue2"
    NaN: "UncertainValue2"
    POSITIVE_INFINITY: "UncertainValue2"
    NEGATIVE_INFINITY: "UncertainValue2"
    MAX_VALUE: "UncertainValue2"

    # ------------------------------------------------------------------
    # Inner class: Correlations
    # ------------------------------------------------------------------

    class Correlations:
        """Symmetric correlation matrix between named uncertainty sources."""

        def __init__(self) -> None:
            # Java HashMap<Key, Double>; Key has symmetric equality.
            # frozenset({src1, src2}) is the symmetric-pair key.
            self._mCorrelations: dict[frozenset, float] = {}

        def add(self, src1: str, src2: str, corr: float) -> None:
            # Java assert (corr >= -1.0) && (corr <= 1.0) — disabled at JVM runtime; skip
            self._mCorrelations[frozenset({src1, src2})] = Math2.bound(corr, -1.0, 1.0)

        def get(self, src1: str, src2: str) -> float:
            r: Optional[float] = self._mCorrelations.get(frozenset({src1, src2}))
            return 0.0 if r is None else float(r)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    def __init__(
        self,
        v: float,
        source_or_dv=None,
        dv: float = 0.0,
    ) -> None:
        """
        Overload dispatch:
          (v)             → UncertainValue2(double v) — no uncertainty
          (v, dv)         → UncertainValue2(double v, double dv)
          (v, source, dv) → UncertainValue2(double v, String source, double dv)
        Use from_sigmas() for the Map<String,Double> constructor (no _sDefIndex increment).
        """
        self._mValue: float = float(v)
        self._mSigmas: dict[str, float] = {}

        if source_or_dv is None:
            # UncertainValue2(double v) → calls this(v, 0.0)
            UncertainValue2._sDefIndex += 1
            self.assignComponent(
                UncertainValue2.DEFAULT + str(UncertainValue2._sDefIndex), 0.0
            )
        elif isinstance(source_or_dv, str):
            # UncertainValue2(double v, String source, double dv)
            self.assignComponent(source_or_dv, float(dv))
        else:
            # UncertainValue2(double v, double dv) → calls this(v, DEFAULT+N, dv)
            UncertainValue2._sDefIndex += 1
            self.assignComponent(
                UncertainValue2.DEFAULT + str(UncertainValue2._sDefIndex),
                float(source_or_dv),
            )

    @classmethod
    def from_sigmas(
        cls, v: float, sigmas: Optional[dict[str, float]]
    ) -> "UncertainValue2":
        """UncertainValue2(double v, Map<String,Double> sigmas) — no _sDefIndex increment."""
        obj: "UncertainValue2" = object.__new__(cls)
        obj._mValue = float(v)
        obj._mSigmas = {}
        if sigmas:
            for k, val in sigmas.items():
                obj.assignComponent(k, val)
        return obj

    # ------------------------------------------------------------------
    # Clone / copy
    # ------------------------------------------------------------------

    def clone(self) -> "UncertainValue2":
        return UncertainValue2.from_sigmas(self._mValue, self._mSigmas)

    def __copy__(self) -> "UncertainValue2":
        return self.clone()

    # ------------------------------------------------------------------
    # Component management
    # ------------------------------------------------------------------

    def assignComponent(self, name: str, sigma: float) -> None:
        if sigma != 0.0:
            self._mSigmas[name] = math.fabs(sigma)
        else:
            self._mSigmas.pop(name, None)

    def assignComponents(self, comps: dict[str, float]) -> None:
        for k, v in comps.items():
            self.assignComponent(k, v)

    def getComponent(self, src: str) -> float:
        return float(self._mSigmas.get(src, 0.0))

    def getFractional(self, src: str) -> float:
        v: Optional[float] = self._mSigmas.get(src)
        return float(v) / self._mValue if v is not None else 0.0

    def hasComponent(self, src: str) -> bool:
        return src in self._mSigmas

    def getComponents(self) -> dict[str, float]:
        return dict(self._mSigmas)

    def getComponentNames(self) -> frozenset[str]:
        return frozenset(self._mSigmas.keys())

    def renameComponent(self, oldName: str, newName: str) -> None:
        if newName in self._mSigmas:
            raise EPQException("A component named " + newName + " already exists.")
        val: Optional[float] = self._mSigmas.pop(oldName, None)
        if val is not None:
            self._mSigmas[newName] = val

    # ------------------------------------------------------------------
    # String representation  (R2: dunder + named alias)
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        if self._mSigmas:
            return str(self._mValue) + " ± " + str(self.uncertainty())
        return str(self._mValue)

    def toString(self) -> str:
        return str(self)

    def toLongString(self) -> str:
        if self._mSigmas:
            sb: list[str] = [str(self._mValue)]
            for k, v in self._mSigmas.items():
                sb.append("±")
                sb.append(str(v))
                sb.append("(")
                sb.append(k)
                sb.append(")")
            return "".join(sb)
        return str(self._mValue)

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    @staticmethod
    def format_n(nf: Callable[[float], str], n: object) -> str:
        if isinstance(n, UncertainValue2):
            return n.format(nf)
        return nf(float(n))  # type: ignore[arg-type]

    def format(self, nf: Callable[[float], str]) -> str:
        if self._mSigmas:
            return nf(self._mValue) + "±" + nf(self.uncertainty())
        return nf(self._mValue)

    def formatLong(self, nf: Callable[[float], str]) -> str:
        sb: list[str] = [nf(self._mValue)]
        for k, v in self._mSigmas.items():
            sb.append("±")
            sb.append(nf(v))
            sb.append("(")
            sb.append(k)
            sb.append(")")
        return "".join(sb)

    def format_src(self, src: str, nf: Callable[[float], str]) -> str:
        dv: Optional[float] = self._mSigmas.get(src)
        return "U(" + src + ")=" + nf(dv if dv is not None else 0.0)

    def formatComponent_nf(self, comp: str, nf: Callable[[float], str]) -> str:
        v: Optional[float] = self._mSigmas.get(comp)
        return nf(v if v is not None else 0.0) + "(" + comp + ")"

    def formatComponent_dec(self, comp: str, dec: int) -> str:
        v: Optional[float] = self._mSigmas.get(comp)
        nf: object
        if (v is not None) and (math.fabs(v) < (10.0 * math.pow(10.0, -dec))):
            nf = ExponentFormat(2)
        else:
            sf: str = "0." + "0" * (dec - 1)
            nf = HalfUpFormat(sf)
        return nf.format(v if v is not None else 0.0) + "(" + comp + ")"  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Private helper
    # ------------------------------------------------------------------

    @staticmethod
    def _toUV2(num: object) -> "UncertainValue2":
        if isinstance(num, UncertainValue2):
            return num
        return UncertainValue2(float(num))  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Static factory / conversion
    # ------------------------------------------------------------------

    @staticmethod
    def asUncertainValue2(n: object) -> "UncertainValue2":
        if isinstance(n, UncertainValue2):
            return n
        return UncertainValue2(float(n))  # type: ignore[arg-type]

    @staticmethod
    def createGaussian(v: float, source: str) -> "UncertainValue2":
        return UncertainValue2(v, source, math.sqrt(v))

    @staticmethod
    def valueOf(n: object) -> "UncertainValue2":
        return UncertainValue2.asUncertainValue2(n)

    # ------------------------------------------------------------------
    # Arithmetic — add (six overloads, R4)
    # ------------------------------------------------------------------

    @staticmethod
    def add_collection_literal(uvs: object) -> "UncertainValue2":
        srcs: set[str] = set()
        total: float = 0.0
        for n2 in uvs:  # type: ignore[union-attr]
            uv2: UncertainValue2 = UncertainValue2._toUV2(n2)
            srcs.update(uv2._mSigmas.keys())
            total += uv2._mValue
        res: UncertainValue2 = UncertainValue2(total)
        for src in srcs:
            unc: float = 0.0
            for n2 in uvs:  # type: ignore[union-attr]
                unc += UncertainValue2._toUV2(n2).getComponent(src)
            res.assignComponent(src, unc)
        return res

    @staticmethod
    def add_collection(uvs: object) -> "UncertainValue2":  # SCIPY-NONE
        return UncertainValue2.add_collection_literal(uvs)

    @staticmethod
    def add_array_literal(uvs: list) -> "UncertainValue2":
        return UncertainValue2.add_collection_literal(uvs)

    @staticmethod
    def add_array(uvs: list) -> "UncertainValue2":  # SCIPY-NONE
        return UncertainValue2.add_array_literal(uvs)

    @staticmethod
    def add_dadb_literal(a: float, na: object, b: float, nb: object) -> "UncertainValue2":
        uva: UncertainValue2 = UncertainValue2._toUV2(na)
        uvb: UncertainValue2 = UncertainValue2._toUV2(nb)
        res: UncertainValue2 = UncertainValue2((a * uva._mValue) + (b * uvb._mValue))
        srcs: set[str] = set(uva._mSigmas.keys()) | set(uvb._mSigmas.keys())
        for src in srcs:
            res.assignComponent(
                src,
                math.fabs((a * uva.getComponent(src)) + (b * uvb.getComponent(src))),
            )
        return res

    @staticmethod
    def add_dadb(a: float, na: object, b: float, nb: object) -> "UncertainValue2":  # SCIPY-NONE
        return UncertainValue2.add_dadb_literal(a, na, b, nb)

    @staticmethod
    def add_nd_literal(v1: object, v2: float) -> object:
        if isinstance(v1, UncertainValue2):
            return UncertainValue2.from_sigmas(v1._mValue + v2, v1._mSigmas)
        return float(v1) + v2  # type: ignore[arg-type]

    @staticmethod
    def add_nd(v1: object, v2: float) -> object:  # SCIPY-NONE
        return UncertainValue2.add_nd_literal(v1, v2)

    @staticmethod
    def add_dn_literal(v1: float, v2: object) -> "UncertainValue2":
        if isinstance(v2, UncertainValue2):
            return UncertainValue2.from_sigmas(v2._mValue + v1, v2._mSigmas)
        return UncertainValue2.valueOf(v1 + float(v2))  # type: ignore[arg-type]

    @staticmethod
    def add_dn(v1: float, v2: object) -> "UncertainValue2":  # SCIPY-NONE
        return UncertainValue2.add_dn_literal(v1, v2)

    @staticmethod
    def add_nn_literal(v1: object, v2: object) -> "UncertainValue2":
        return UncertainValue2.add_dadb_literal(1.0, v1, 1.0, v2)

    @staticmethod
    def add_nn(v1: object, v2: object) -> "UncertainValue2":  # SCIPY-NONE
        return UncertainValue2.add_nn_literal(v1, v2)

    @staticmethod
    def subtract_literal(na: object, nb: object) -> "UncertainValue2":
        return UncertainValue2.add_dadb_literal(1.0, UncertainValue2._toUV2(na), -1.0, UncertainValue2._toUV2(nb))

    @staticmethod
    def subtract(na: object, nb: object) -> "UncertainValue2":  # SCIPY-NONE
        return UncertainValue2.subtract_literal(na, nb)

    # ------------------------------------------------------------------
    # Arithmetic — mean
    # ------------------------------------------------------------------

    @staticmethod
    def mean_collection_literal(uvs: object) -> "UncertainValue2":
        lst = list(uvs)  # type: ignore[call-overload]
        return UncertainValue2.divide_nd_literal(UncertainValue2.add_collection(lst), float(len(lst)))

    @staticmethod
    def mean_collection(uvs: object) -> "UncertainValue2":  # SCIPY-NONE
        return UncertainValue2.mean_collection_literal(uvs)

    @staticmethod
    def mean_array_literal(uvs: list) -> "UncertainValue2":
        return UncertainValue2.divide_nd_literal(UncertainValue2.add_array(uvs), float(len(uvs)))

    @staticmethod
    def mean_array(uvs: list) -> "UncertainValue2":  # SCIPY-NONE
        return UncertainValue2.mean_array_literal(uvs)

    # ------------------------------------------------------------------
    # Arithmetic — abs
    # ------------------------------------------------------------------

    def abs_literal(self) -> "UncertainValue2":
        return self if self._mValue >= 0.0 else UncertainValue2.from_sigmas(-self._mValue, self._mSigmas)

    def abs(self) -> "UncertainValue2":  # SCIPY-NONE
        return self.abs_literal()

    @staticmethod
    def abs_n_literal(n: object) -> "UncertainValue2":
        return UncertainValue2._toUV2(n).abs_literal()

    @staticmethod
    def abs_n(n: object) -> "UncertainValue2":  # SCIPY-NONE
        return UncertainValue2.abs_n_literal(n)

    # ------------------------------------------------------------------
    # Arithmetic — weightedMean / safeWeightedMean
    # ------------------------------------------------------------------

    @staticmethod
    def weightedMean_literal(cuv: object) -> "UncertainValue2":
        var_sum: float = 0.0
        total: UncertainValue2 = UncertainValue2.ZERO
        for nuv in cuv:  # type: ignore[union-attr]
            uv: UncertainValue2 = UncertainValue2._toUV2(nuv)
            ivar: float = 1.0 / uv.variance()
            if math.isnan(ivar):
                raise UtilException(
                    "Unable to compute the weighted mean when one or more datapoints have zero uncertainty."
                )
            var_sum += ivar
            total = UncertainValue2.add_nn(total, UncertainValue2.multiply_dn(ivar, uv))
        i_var_sum: float = 1.0 / var_sum
        return UncertainValue2.NaN if math.isnan(i_var_sum) else UncertainValue2.multiply_dn(i_var_sum, total)

    @staticmethod
    def weightedMean(cuv: object) -> "UncertainValue2":  # SCIPY-NONE
        return UncertainValue2.weightedMean_literal(cuv)

    @staticmethod
    def safeWeightedMean_literal(cuv: object) -> "UncertainValue2":
        var_sum: float = 0.0
        total: UncertainValue2 = UncertainValue2.ZERO
        for nuv in cuv:  # type: ignore[union-attr]
            if not isinstance(nuv, UncertainValue2):
                continue
            uv = nuv
            ivar: float = 1.0 / uv.variance()
            if math.isnan(ivar):
                continue
            var_sum += ivar
            total = UncertainValue2.add_nn(total, UncertainValue2.multiply_dn(ivar, uv))
        i_var_sum: float = 1.0 / var_sum
        return UncertainValue2.NaN if math.isnan(i_var_sum) else UncertainValue2.multiply_dn(i_var_sum, total)

    @staticmethod
    def safeWeightedMean(cuv: object) -> "UncertainValue2":  # SCIPY-NONE
        return UncertainValue2.safeWeightedMean_literal(cuv)

    # ------------------------------------------------------------------
    # Arithmetic — min / max
    # ------------------------------------------------------------------

    @staticmethod
    def min_literal(uvs: object) -> Optional["UncertainValue2"]:
        res: Optional[UncertainValue2] = None
        for nuv in uvs:  # type: ignore[union-attr]
            uv: UncertainValue2 = UncertainValue2._toUV2(nuv)
            if res is None:
                res = uv
            elif uv.doubleValue() < res.doubleValue():
                res = uv
            elif uv.doubleValue() == res.doubleValue():
                if uv.uncertainty() > res.uncertainty():
                    res = uv
        return res

    @staticmethod
    def min(uvs: object) -> Optional["UncertainValue2"]:  # SCIPY-NONE
        return UncertainValue2.min_literal(uvs)

    @staticmethod
    def max_literal(uvs: object) -> Optional["UncertainValue2"]:
        res: Optional[UncertainValue2] = None
        for nuv in uvs:  # type: ignore[union-attr]
            uv: UncertainValue2 = UncertainValue2._toUV2(nuv)
            if res is None:
                res = uv
            elif uv.doubleValue() > res.doubleValue():
                res = uv
            elif uv.doubleValue() == res.doubleValue():
                if uv.uncertainty() > res.uncertainty():
                    res = uv
        return res

    @staticmethod
    def max(uvs: object) -> Optional["UncertainValue2"]:  # SCIPY-NONE
        return UncertainValue2.max_literal(uvs)

    # ------------------------------------------------------------------
    # Arithmetic — multiply
    # ------------------------------------------------------------------

    @staticmethod
    def multiply_dn_literal(v1: float, n2: object) -> "UncertainValue2":
        v2: UncertainValue2 = UncertainValue2._toUV2(n2)
        # Java assert v2.uncertainty() >= 0.0 — disabled at JVM runtime; skip
        res: UncertainValue2 = UncertainValue2(v1 * v2._mValue)
        for k, val in v2._mSigmas.items():
            res.assignComponent(k, v1 * val)
        return res

    @staticmethod
    def multiply_dn(v1: float, n2: object) -> "UncertainValue2":  # SCIPY-NONE
        return UncertainValue2.multiply_dn_literal(v1, n2)

    @staticmethod
    def multiply_nn_literal(na: object, nb: object) -> "UncertainValue2":
        a: UncertainValue2 = UncertainValue2._toUV2(na)
        b: UncertainValue2 = UncertainValue2._toUV2(nb)
        res: UncertainValue2 = UncertainValue2(a._mValue * b._mValue)
        srcs: set[str] = set(a._mSigmas.keys()) | set(b._mSigmas.keys())
        ca: float = b._mValue
        cb: float = a._mValue
        for src in srcs:
            ua: float = a.getComponent(src)
            ub: float = b.getComponent(src)
            res.assignComponent(src, math.fabs((ca * ua) + (cb * ub)))
        return res

    @staticmethod
    def multiply_nn(na: object, nb: object) -> "UncertainValue2":  # SCIPY-NONE
        return UncertainValue2.multiply_nn_literal(na, nb)

    # ------------------------------------------------------------------
    # Arithmetic — invert
    # ------------------------------------------------------------------

    @staticmethod
    def invert_literal(nv: object) -> "UncertainValue2":
        v: UncertainValue2 = UncertainValue2._toUV2(nv)
        res: UncertainValue2 = UncertainValue2(1.0 / v._mValue)
        if not math.isnan(res.doubleValue()):
            cb: float = 1.0 / (v._mValue * v._mValue)
            if math.isnan(cb):
                return UncertainValue2.NaN
            for src in v._mSigmas.keys():
                res.assignComponent(src, math.fabs(cb * v.getComponent(src)))
        return res

    @staticmethod
    def invert(nv: object) -> "UncertainValue2":  # SCIPY-NONE
        return UncertainValue2.invert_literal(nv)

    # ------------------------------------------------------------------
    # Arithmetic — divide (three overloads, R4)
    # ------------------------------------------------------------------

    @staticmethod
    def divide_nn_literal(na: object, nb: object) -> "UncertainValue2":
        a: UncertainValue2 = UncertainValue2._toUV2(na)
        b: UncertainValue2 = UncertainValue2._toUV2(nb)
        res: UncertainValue2 = UncertainValue2(a._mValue / b._mValue)
        if not math.isnan(res.doubleValue()):
            srcs: set[str] = set(a._mSigmas.keys()) | set(b._mSigmas.keys())
            ca: float = 1.0 / b._mValue
            cb: float = -a._mValue / (b._mValue * b._mValue)
            if math.isnan(ca) or math.isnan(cb):
                return UncertainValue2.NaN
            for src in srcs:
                ua: float = a.getComponent(src)
                ub: float = b.getComponent(src)
                res.assignComponent(src, math.fabs((ca * ua) + (cb * ub)))
        return res

    @staticmethod
    def divide_nn(na: object, nb: object) -> "UncertainValue2":  # SCIPY-NONE
        return UncertainValue2.divide_nn_literal(na, nb)

    @staticmethod
    def divide_dn_literal(a: float, nb: object) -> "UncertainValue2":
        b: UncertainValue2 = UncertainValue2._toUV2(nb)
        res: UncertainValue2 = UncertainValue2(a / b._mValue)
        if not math.isnan(res.doubleValue()):
            ub: float = math.fabs(a / (b._mValue * b._mValue))
            for k, val in b._mSigmas.items():
                res.assignComponent(k, ub * val)
        return res

    @staticmethod
    def divide_dn(a: float, nb: object) -> "UncertainValue2":  # SCIPY-NONE
        return UncertainValue2.divide_dn_literal(a, nb)

    @staticmethod
    def divide_nd_literal(na: object, b: float) -> "UncertainValue2":
        a: UncertainValue2 = UncertainValue2._toUV2(na)
        den: float = 1.0 / b
        if not math.isnan(den):
            res: UncertainValue2 = UncertainValue2(den * a.doubleValue())
            ua: float = math.fabs(den)
            for k, val in a._mSigmas.items():
                res.assignComponent(k, ua * val)
            return res
        return UncertainValue2.NaN

    @staticmethod
    def divide_nd(na: object, b: float) -> "UncertainValue2":  # SCIPY-NONE
        return UncertainValue2.divide_nd_literal(na, b)

    # ------------------------------------------------------------------
    # Arithmetic — transcendental
    # ------------------------------------------------------------------

    @staticmethod
    def exp_literal(nx: object) -> "UncertainValue2":
        x: UncertainValue2 = UncertainValue2._toUV2(nx)
        # Java assert !Double.isNaN(x.mValue) — disabled at JVM runtime; skip
        ex: float = math.exp(x._mValue)
        res: UncertainValue2 = UncertainValue2(ex)
        for k, val in x._mSigmas.items():
            res.assignComponent(k, ex * val)
        return res

    @staticmethod
    def exp(nx: object) -> "UncertainValue2":  # SCIPY-NONE
        return UncertainValue2.exp_literal(nx)

    @staticmethod
    def log_literal(nx: object) -> "UncertainValue2":
        v2: UncertainValue2 = UncertainValue2._toUV2(nx)
        tmp: float = 1.0 / v2._mValue
        lv: float = math.log(v2._mValue) if v2._mValue > 0.0 else float("nan")
        if not (math.isnan(tmp) or math.isnan(lv)):
            res: UncertainValue2 = UncertainValue2(lv)
            for k, val in v2._mSigmas.items():
                res.assignComponent(k, tmp * val)
            return res
        return UncertainValue2.NaN

    @staticmethod
    def log(nx: object) -> "UncertainValue2":  # SCIPY-NONE
        return UncertainValue2.log_literal(nx)

    @staticmethod
    def pow_literal(n1: object, n: float) -> "UncertainValue2":
        v1: UncertainValue2 = UncertainValue2._toUV2(n1)
        if v1._mValue != 0.0:
            f: float = math.pow(v1._mValue, n)
            df: float = n * math.pow(v1._mValue, n - 1.0)
            res: UncertainValue2 = UncertainValue2(f)
            for k, val in v1._mSigmas.items():
                res.assignComponent(k, val * df)
            return res
        return UncertainValue2.ZERO

    @staticmethod
    def pow(n1: object, n: float) -> "UncertainValue2":  # SCIPY-NONE
        return UncertainValue2.pow_literal(n1, n)

    def sqrt(self) -> "UncertainValue2":  # SCIPY-NONE — instance form
        return UncertainValue2.pow(self, 0.5)

    def sqrt_literal(self) -> "UncertainValue2":
        return UncertainValue2.pow_literal(self, 0.5)

    @staticmethod
    def sqrt_n_literal(uv: object) -> "UncertainValue2":
        return UncertainValue2.pow_literal(uv, 0.5)

    @staticmethod
    def sqrt_n(uv: object) -> "UncertainValue2":  # SCIPY-NONE
        return UncertainValue2.sqrt_n_literal(uv)

    @staticmethod
    def quadratic_literal(
        na: object, nb: object, nc: object
    ) -> Optional[list["UncertainValue2"]]:
        a: UncertainValue2 = UncertainValue2._toUV2(na)
        b: UncertainValue2 = UncertainValue2._toUV2(nb)
        c: UncertainValue2 = UncertainValue2._toUV2(nc)
        r: UncertainValue2 = UncertainValue2.add_dadb(
            1.0, UncertainValue2.pow(b, 2.0),
            -4.0, UncertainValue2.multiply_nn(a, c),
        )
        if r.doubleValue() > 0.0:
            # R8: Math.signum(b.mValue) → 0.0 if 0.0 else copysign(1.0, ...)
            signum_b: float = (
                0.0 if b._mValue == 0.0 else math.copysign(1.0, b._mValue)
            )
            q: UncertainValue2 = UncertainValue2.multiply_dn(
                -0.5,
                UncertainValue2.add_nn(
                    b, UncertainValue2.multiply_dn(signum_b, r.sqrt())
                ),
            )
            return [UncertainValue2.divide_nn(q, a), UncertainValue2.divide_nn(c, q)]
        return None

    @staticmethod
    def quadratic(
        na: object, nb: object, nc: object
    ) -> Optional[list["UncertainValue2"]]:  # SCIPY-NONE
        return UncertainValue2.quadratic_literal(na, nb, nc)

    @staticmethod
    def sqr_literal(uv: "UncertainValue2") -> "UncertainValue2":
        return UncertainValue2.pow_literal(uv, 2.0)

    @staticmethod
    def sqr(uv: "UncertainValue2") -> "UncertainValue2":  # SCIPY-NONE
        return UncertainValue2.sqr_literal(uv)

    @staticmethod
    def negate_literal(n: object) -> "UncertainValue2":
        uv: UncertainValue2 = UncertainValue2._toUV2(n)
        return UncertainValue2.from_sigmas(-uv._mValue, uv._mSigmas)

    @staticmethod
    def negate(n: object) -> "UncertainValue2":  # SCIPY-NONE
        return UncertainValue2.negate_literal(n)

    @staticmethod
    def atan_literal(uv: "UncertainValue2") -> "UncertainValue2":
        f: float = math.atan(uv.doubleValue())
        df: float = 1.0 / (1.0 + (uv.doubleValue() * uv.doubleValue()))
        if not (math.isnan(f) or math.isnan(df)):
            res: UncertainValue2 = UncertainValue2(f)
            for k, val in uv._mSigmas.items():
                res.assignComponent(k, df * val)
            return res
        return UncertainValue2.NaN

    @staticmethod
    def atan(uv: "UncertainValue2") -> "UncertainValue2":  # SCIPY-NONE
        return UncertainValue2.atan_literal(uv)

    @staticmethod
    def atan2_literal(y: "UncertainValue2", x: "UncertainValue2") -> "UncertainValue2":
        f: float = math.atan2(y.doubleValue(), x.doubleValue())
        df: float = 1.0 / (1.0 + Math2.sqr(y.doubleValue() / x.doubleValue()))
        if not (math.isnan(f) or math.isnan(df)):
            res: UncertainValue2 = UncertainValue2(f)
            for k, val in UncertainValue2.divide_nn(y, x)._mSigmas.items():
                res.assignComponent(k, df * val)
            return res
        return UncertainValue2.NaN

    @staticmethod
    def atan2(y: "UncertainValue2", x: "UncertainValue2") -> "UncertainValue2":  # SCIPY-NONE
        return UncertainValue2.atan2_literal(y, x)

    @staticmethod
    def nonNegative_literal(uv: "UncertainValue2") -> "UncertainValue2":
        return uv if uv.doubleValue() >= 0.0 else UncertainValue2.from_sigmas(0.0, uv._mSigmas)

    @staticmethod
    def nonNegative(uv: "UncertainValue2") -> "UncertainValue2":  # SCIPY-NONE
        return UncertainValue2.nonNegative_literal(uv)

    # ------------------------------------------------------------------
    # normalize — primary uses numpy; literal uses JamaMatrix
    # ------------------------------------------------------------------

    @staticmethod
    def normalize(vals: list["UncertainValue2"]) -> list["UncertainValue2"]:
        """Primary: numpy matrix operations (no JamaMatrix)."""
        n: int = len(vals)
        s: float = sum(v.doubleValue() for v in vals)
        s2: float = s * s
        j_arr: np.ndarray = np.zeros((n, n), dtype=np.float64)
        for r_i in range(n):
            j_arr[r_i, r_i] = (s - vals[r_i].doubleValue()) / s2
            other: float = -vals[r_i].doubleValue() / s2
            for c_i in range(n):
                if c_i != r_i:
                    j_arr[r_i, c_i] = other
        cov_arr: np.ndarray = np.zeros((n, n), dtype=np.float64)
        for c_i in range(n):
            cov_arr[c_i, c_i] = vals[c_i].variance()
        r_mat: np.ndarray = j_arr @ cov_arr @ j_arr.T
        return [
            UncertainValue2(vals[i].doubleValue() / s, math.sqrt(float(r_mat[i, i])))
            for i in range(n)
        ]

    @staticmethod
    def normalize_literal(vals: list["UncertainValue2"]) -> list["UncertainValue2"]:
        """Literal: line-for-line Java using JamaMatrix."""
        n: int = len(vals)
        s: float = sum(v.doubleValue() for v in vals)
        s2: float = s * s
        j_data: list[list[float]] = [[0.0] * n for _ in range(n)]
        for r_i in range(n):
            j_data[r_i][r_i] = (s - vals[r_i].doubleValue()) / s2
            other: float = -vals[r_i].doubleValue() / s2
            for c_i in range(n):
                if c_i != r_i:
                    j_data[r_i][c_i] = other
        cov_data: list[list[float]] = [[0.0] * n for _ in range(n)]
        for c_i in range(n):
            cov_data[c_i][c_i] = vals[c_i].variance()
        J: JamaMatrix = JamaMatrix(np.array(j_data, dtype=np.float64))
        C: JamaMatrix = JamaMatrix(np.array(cov_data, dtype=np.float64))
        r_mat: JamaMatrix = J.times(C).times(J.transpose())
        return [
            UncertainValue2(vals[i].doubleValue() / s, math.sqrt(r_mat.get(i, i)))
            for i in range(n)
        ]

    # ------------------------------------------------------------------
    # Instance methods — value and uncertainty accessors
    # ------------------------------------------------------------------

    def doubleValue(self) -> float:
        return float(self._mValue)

    def __float__(self) -> float:
        return self.doubleValue()

    def floatValue(self) -> float:
        return float(self._mValue)

    def intValue(self) -> int:
        return int(self._mValue)

    def __int__(self) -> int:
        return self.intValue()

    def longValue(self) -> int:
        return int(self._mValue)

    def byteValue(self) -> int:
        return int(self._mValue) & 0xFF

    def shortValue(self) -> int:
        return int(self._mValue) & 0xFFFF

    def isUncertain(self) -> bool:
        return bool(self._mSigmas)

    def isNaN(self) -> bool:
        return math.isnan(self._mValue)

    def variance(self) -> float:  # SCIPY-NONE
        return self.variance_literal()

    def variance_literal(self) -> float:
        sigma2: float = 0.0
        for s in self._mSigmas.values():
            sigma2 += s * s
        return sigma2

    def variance_corr(self, corr: "UncertainValue2.Correlations") -> float:  # SCIPY-NONE
        return self.variance_corr_literal(corr)

    def variance_corr_literal(self, corr: "UncertainValue2.Correlations") -> float:
        keys: list[str] = list(self._mSigmas.keys())
        res: float = 0.0
        for i in range(len(keys)):
            res += Math2.sqr(self._mSigmas[keys[i]])
        for i in range(len(keys) - 1):
            for j in range(i + 1, len(keys)):
                res += (
                    2.0
                    * self._mSigmas[keys[i]]
                    * self._mSigmas[keys[j]]
                    * corr.get(keys[i], keys[j])
                )
        return res

    def uncertainty(self) -> float:  # SCIPY-NONE — instance form
        return self.uncertainty_literal()

    def uncertainty_literal(self) -> float:
        return math.sqrt(self.variance())

    def uncertainty_comps(self, comps: object) -> float:
        """JAVA-BUG-1: sums raw values not squares. Returns sqrt(sum-of-values)."""
        sum2: float = 0.0
        for comp in comps:  # type: ignore[union-attr]
            sum2 += self.getComponent(comp)  # JAVA-BUG-1
        return math.sqrt(sum2)

    def uncertainty_comps_literal(self, comps: object) -> float:
        return self.uncertainty_comps(comps)

    def uncertainty_comps_strict(self, comps: object) -> float:
        """Correct L2 norm: sqrt(sum-of-squares)."""
        sum2: float = 0.0
        for comp in comps:  # type: ignore[union-attr]
            sum2 += self.getComponent(comp) ** 2
        return math.sqrt(sum2)

    def uncertainty_corr(self, corr: "UncertainValue2.Correlations") -> float:  # SCIPY-NONE
        return self.uncertainty_corr_literal(corr)

    def uncertainty_corr_literal(self, corr: "UncertainValue2.Correlations") -> float:
        return math.sqrt(self.variance_corr(corr))

    @staticmethod
    def uncertainty_n(n: object) -> float:
        """JAVA-BUG-3: returns doubleValue() instead of uncertainty()."""
        # Java: return n instanceof UncertainValue2 ? ((UncertainValue2)n).doubleValue() : 0.0;
        return float(n._mValue) if isinstance(n, UncertainValue2) else 0.0  # JAVA-BUG-3

    @staticmethod
    def uncertainty_n_strict(n: object) -> float:
        """Correct form: returns uncertainty() for UncertainValue2, else 0.0."""
        return n.uncertainty() if isinstance(n, UncertainValue2) else 0.0

    def fractionalUncertainty(self) -> float:
        return (
            float("1.7976931348623157e+308")
            if math.isnan(1.0 / self._mValue)
            else math.fabs(self.uncertainty() / self._mValue)
        )

    def fractionalUncertainty_literal(self) -> float:
        return self.fractionalUncertainty()

    @staticmethod
    def fractionalUncertainty_n(n: object) -> float:  # SCIPY-NONE
        return UncertainValue2.fractionalUncertainty_n_literal(n)

    @staticmethod
    def fractionalUncertainty_n_literal(n: object) -> float:
        if isinstance(n, UncertainValue2):
            return n.fractionalUncertainty()
        return 0.0

    def fractionalUncertaintyU(self) -> "UncertainValue2":  # SCIPY-NONE
        return self.fractionalUncertaintyU_literal()

    def fractionalUncertaintyU_literal(self) -> "UncertainValue2":
        return UncertainValue2.divide_nd(self, self._mValue)

    def reduced(self, name: str) -> "UncertainValue2":  # SCIPY-NONE
        return self.reduced_literal(name)

    def reduced_literal(self, name: str) -> "UncertainValue2":
        return UncertainValue2(self.doubleValue(), name, self.uncertainty())

    # ------------------------------------------------------------------
    # Equality / ordering  (R2: dunder + named alias)
    # ------------------------------------------------------------------

    def __hash__(self) -> int:
        return hash((self._mValue, frozenset(self._mSigmas.items())))

    def hashCode(self) -> int:
        return self.__hash__()

    def __eq__(self, obj: object) -> bool:
        if self is obj:
            return True
        if not isinstance(obj, UncertainValue2):
            return False
        return self._mSigmas == obj._mSigmas and self._mValue == obj._mValue

    def equals(self, obj: object) -> bool:
        return self.__eq__(obj)

    def equals_tol(self, other: "UncertainValue2", tolerance: float) -> bool:
        """JAVA-BUG-2: raises KeyError when key absent from one map (Java NPE)."""
        if self is other:
            return True
        keys: set[str] = set(other._mSigmas.keys()) | set(self._mSigmas.keys())
        for key in keys:
            # JAVA-BUG-2: dict[key] raises KeyError if absent; Java NPE on null unbox
            if math.fabs(self._mSigmas[key] - other._mSigmas[key]) >= tolerance:  # JAVA-BUG-2
                return False
        return (
            math.fabs(self.uncertainty() - other.uncertainty()) < tolerance
            and math.fabs(self._mValue - other._mValue) < tolerance
        )

    def equals_tol_strict(self, other: "UncertainValue2", tolerance: float) -> bool:
        """Corrected form: uses .get(key, 0.0) to handle absent keys."""
        if self is other:
            return True
        keys: set[str] = set(other._mSigmas.keys()) | set(self._mSigmas.keys())
        for key in keys:
            if math.fabs(self._mSigmas.get(key, 0.0) - other._mSigmas.get(key, 0.0)) >= tolerance:
                return False
        return (
            math.fabs(self.uncertainty() - other.uncertainty()) < tolerance
            and math.fabs(self._mValue - other._mValue) < tolerance
        )

    def compareTo(self, o: "UncertainValue2") -> int:
        if self._mValue < o._mValue:
            res: int = -1
        elif self._mValue > o._mValue:
            res = 1
        else:
            res = 0
        if res != 0:
            return res
        u_self: float = self.uncertainty()
        u_o: float = o.uncertainty()
        if u_self < u_o:
            return -1
        if u_self > u_o:
            return 1
        return 0

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, UncertainValue2):
            return NotImplemented  # type: ignore[return-value]
        return self.compareTo(other) < 0

    def __le__(self, other: object) -> bool:
        if not isinstance(other, UncertainValue2):
            return NotImplemented  # type: ignore[return-value]
        return self.compareTo(other) <= 0

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, UncertainValue2):
            return NotImplemented  # type: ignore[return-value]
        return self.compareTo(other) > 0

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, UncertainValue2):
            return NotImplemented  # type: ignore[return-value]
        return self.compareTo(other) >= 0

    def lessThan(self, uv2: "UncertainValue2") -> bool:
        return self._mValue < uv2._mValue

    def greaterThan(self, uv2: "UncertainValue2") -> bool:
        return self._mValue > uv2._mValue

    def lessThanOrEqual(self, uv2: "UncertainValue2") -> bool:
        return self._mValue <= uv2._mValue

    def greaterThanOrEqual(self, uv2: "UncertainValue2") -> bool:
        return self._mValue >= uv2._mValue


# ------------------------------------------------------------------
# Post-class initialization of static constants (R6: self-referential)
# ------------------------------------------------------------------

UncertainValue2.ONE = UncertainValue2(1.0)
UncertainValue2.ZERO = UncertainValue2(0.0)
UncertainValue2.NaN = UncertainValue2(float("nan"))
UncertainValue2.POSITIVE_INFINITY = UncertainValue2(float("inf"))
UncertainValue2.NEGATIVE_INFINITY = UncertainValue2(float("-inf"))
UncertainValue2.MAX_VALUE = UncertainValue2(float("1.7976931348623157e+308"))

# Module-level alias per spec (R2)
Correlations = UncertainValue2.Correlations
