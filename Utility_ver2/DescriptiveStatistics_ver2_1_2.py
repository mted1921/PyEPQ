r"""
DescriptiveStatistics_ver2_1_1.py — Python port of
gov.nist.microanalysis.Utility.DescriptiveStatistics

Guide version : 2
Generation    : 1
Port-code fixes: 2

CHANGES
-------
FIX-1 (assert-error / R9-violation): skewness_literal() raised ZeroDivisionError
  when variance is 0 (e.g. all-equal data). Java IEEE 754 returns NaN via 0.0/0.0.
  Added explicit denom==0 guard returning float("nan"). Category: assert-error.
FIX-2 (assert-error): kurtosis_literal() raised ZeroDivisionError when variance
  is 0 (e.g. all-equal data). Same IEEE 754 issue as FIX-1. Added explicit
  v==0 guard returning float("nan"). Category: assert-error.

------------------------------------------------------------------------
Original Java source (gov.nist.microanalysis.Utility.DescriptiveStatistics)
------------------------------------------------------------------------
/**
 * <p>
 * A simple class to calculate four basic descriptive statistics - average,
 * variance, skewness and kurtosis. Create an instance of the class, then use
 * addPoint() to record data points. At any time you may call one of the
 * statistic functions to return the current value of the statistic.
 * </p>
 * <p>
 * Copyright: Pursuant to title 17 Section 105 of the United States Code this
 * software is not subject to copyright protection and is in the public domain
 * </p>
 * <p>
 * Company: National Institute of Standards and Technology
 * </p>
 *
 * @author Nicholas W. M. Ritchie
 * @version 1.0
 */
------------------------------------------------------------------------
"""
from __future__ import annotations

import copy
import math
import sys
from typing import Iterable, Optional, Sequence, Union

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

BUG_LEDGER: tuple = (
    (
        "JAVA-BUG-1",
        "merge",
        "Java `merge()` increments `mNPoints += ds.mNPoints` twice (lines 96 and 99). "
        "After merging an m-point object into an n-point object, `count()` reports n+2·m "
        "instead of n+m; `average()` and `variance()` are correspondingly incorrect. "
        "Preserved verbatim; `merge_strict()` companion fixes the double-count.",
    ),
)


class DescriptiveStatistics:
    """Python port of ``gov.nist.microanalysis.Utility.DescriptiveStatistics``.

    Accumulates four power sums plus min/max/last to compute mean, variance,
    skewness and kurtosis incrementally.  Implements ``Comparable`` via
    ``compareTo`` + ``__lt__``/``__le__``/``__gt__``/``__ge__``.
    """

    def __init__(
        self,
        ds1: Optional["DescriptiveStatistics"] = None,
        ds2: Optional["DescriptiveStatistics"] = None,
    ) -> None:
        if ds1 is None and ds2 is None:
            # empty constructor
            self._mLast: float = float("nan")
            self._mSum: float = 0.0
            self._mSumOfSqrs: float = 0.0
            self._mSumOfCubes: float = 0.0
            self._mSumOfQuarts: float = 0.0
            self._mNPoints: int = 0
            self._mMin: float = sys.float_info.max
            self._mMax: float = -sys.float_info.max
        else:
            # two-arg combine constructor
            self._mLast = float("nan")
            self._mSum = ds1._mSum + ds2._mSum  # type: ignore[union-attr]
            self._mSumOfSqrs = ds1._mSumOfSqrs + ds2._mSumOfSqrs  # type: ignore[union-attr]
            self._mSumOfCubes = ds1._mSumOfCubes + ds2._mSumOfCubes  # type: ignore[union-attr]
            self._mSumOfQuarts = ds1._mSumOfQuarts + ds2._mSumOfQuarts  # type: ignore[union-attr]
            self._mNPoints = ds1._mNPoints + ds2._mNPoints  # type: ignore[union-attr]
            self._mMin = min(ds1._mMin, ds2._mMin)  # type: ignore[union-attr]
            self._mMax = max(ds1._mMax, ds2._mMax)  # type: ignore[union-attr]
            self._mNPoints = ds1._mNPoints + ds2._mNPoints  # type: ignore[union-attr]  # Java assigns twice (idempotent =)

    # ------------------------------------------------------------------
    # clone / copy
    # ------------------------------------------------------------------

    def clone(self) -> "DescriptiveStatistics":
        res: "DescriptiveStatistics" = DescriptiveStatistics()
        res._mLast = self._mLast
        res._mSum = self._mSum
        res._mSumOfSqrs = self._mSumOfSqrs
        res._mSumOfCubes = self._mSumOfCubes
        res._mSumOfQuarts = self._mSumOfQuarts
        res._mMin = self._mMin
        res._mMax = self._mMax
        res._mNPoints = self._mNPoints
        return res

    def __copy__(self) -> "DescriptiveStatistics":
        return self.clone()

    # ------------------------------------------------------------------
    # Mutating methods
    # ------------------------------------------------------------------

    def merge(self, ds: "DescriptiveStatistics") -> None:
        """Merge *ds* into this object.

        .. warning::
            **JAVA-BUG-1** — `mNPoints` is incremented twice; `count()` reports
            ``self.count() + 2 * ds.count()`` after the merge.  Preserved
            verbatim.  Use :meth:`merge_strict` for the correct behaviour.
        """
        self._mLast = float("nan")
        self._mSum += ds._mSum
        self._mSumOfSqrs += ds._mSumOfSqrs
        self._mSumOfCubes += ds._mSumOfCubes
        self._mSumOfQuarts += ds._mSumOfQuarts
        self._mNPoints += ds._mNPoints  # JAVA-BUG-1: mNPoints += ds.mNPoints; (first occurrence)
        self._mMin = min(self._mMin, ds._mMin)
        self._mMax = max(self._mMax, ds._mMax)
        self._mNPoints += ds._mNPoints  # JAVA-BUG-1: mNPoints += ds.mNPoints; (duplicate — double-counts)

    def merge_strict(self, ds: "DescriptiveStatistics") -> None:
        """Bug-corrected variant of :meth:`merge` — increments ``mNPoints`` once."""
        self._mLast = float("nan")
        self._mSum += ds._mSum
        self._mSumOfSqrs += ds._mSumOfSqrs
        self._mSumOfCubes += ds._mSumOfCubes
        self._mSumOfQuarts += ds._mSumOfQuarts
        self._mNPoints += ds._mNPoints
        self._mMin = min(self._mMin, ds._mMin)
        self._mMax = max(self._mMax, ds._mMax)

    def add(self, x: float) -> None:
        x = float(x)
        x2: float = x * x
        self._mSum += x
        self._mSumOfSqrs += x2
        self._mSumOfCubes += x2 * x
        self._mSumOfQuarts += x2 * x2
        if math.isnan(self._mMin) or (x < self._mMin):
            self._mMin = x
        if math.isnan(self._mMax) or (x > self._mMax):
            self._mMax = x
        self._mLast = x
        self._mNPoints += 1

    def remove(self, val: float) -> None:
        """Remove a previously added value.

        .. warning::
            Min/max may become ``NaN`` after a call to ``remove``, as documented
            in the Java Javadoc.
        """
        val = float(val)
        if self._mMax == val:
            self._mMax = float("nan")
        if self._mMin == val:
            self._mMin = float("nan")
        self._mNPoints -= 1
        self._mSum -= val
        self._mSumOfSqrs -= val * val
        self._mSumOfCubes -= val * val * val
        self._mSumOfQuarts -= val * val * val * val

    def removeLast(self) -> bool:
        res: bool = not math.isnan(self._mLast)
        if res:
            self.remove(self._mLast)
        self._mLast = float("nan")
        return res

    # ------------------------------------------------------------------
    # Mathematical accessors (primary + _literal companions)
    # ------------------------------------------------------------------

    def average_literal(self) -> float:
        return self._mSum / self._mNPoints

    def average(self) -> float:  # SCIPY-NONE: no library substitute
        return self.average_literal()

    def variance_literal(self) -> float:
        avg: float = self.average()
        return (
            max(0.0, (self._mSumOfSqrs - (self._mSum * avg)) / self._mNPoints)
            if self._mNPoints > 1
            else float("nan")
        )

    def variance(self) -> float:  # SCIPY-NONE: no library substitute
        return self.variance_literal()

    def standardDeviation_literal(self) -> float:
        return math.sqrt(self.variance())

    def standardDeviation(self) -> float:  # SCIPY-NONE: no library substitute
        return self.standardDeviation_literal()

    def getValue_literal(self, name: str) -> UncertainValue2:
        sd: float = self.standardDeviation()
        return UncertainValue2(self.average(), name, 0.0 if math.isnan(sd) else sd)

    def getValue(self, name: str) -> UncertainValue2:  # SCIPY-NONE: no library substitute
        return self.getValue_literal(name)

    def skewness_literal(self) -> float:
        mu: float = self.average()
        denom: float = self._mNPoints * math.pow(self.variance(), 1.5)
        if denom == 0.0:  # FIX-1: Java IEEE 754 returns NaN; Python raises ZeroDivisionError
            return float("nan")
        return (
            (self._mSumOfCubes - (3.0 * mu * self._mSumOfSqrs))
            + (2 * self._mSum * mu * mu)
        ) / denom

    def skewness(self) -> float:  # SCIPY-NONE: no library substitute
        return self.skewness_literal()

    def standardErrorOfSkewness_literal(self) -> float:
        return math.sqrt(6.0 / self._mNPoints)

    def standardErrorOfSkewness(self) -> float:  # SCIPY-NONE: no library substitute
        return self.standardErrorOfSkewness_literal()

    def kurtosis_literal(self) -> float:
        mu: float = self.average()
        v: float = self.variance()
        if v == 0.0:  # FIX-2: Java IEEE 754 returns NaN; Python raises ZeroDivisionError
            return float("nan")
        return (
            (
                (
                    (self._mSumOfQuarts - (4.0 * mu * self._mSumOfCubes))
                    + (6.0 * mu * mu * self._mSumOfSqrs)
                )
                - (3.0 * self._mSum * mu * mu * mu)
            )
            / (self._mNPoints * v * v)
        ) - 3.0

    def kurtosis(self) -> float:  # SCIPY-NONE: no library substitute
        return self.kurtosis_literal()

    def standardErrorOfKurtosis_literal(self) -> float:
        return math.sqrt(24.0 / self._mNPoints)

    def standardErrorOfKurtosis(self) -> float:  # SCIPY-NONE: no library substitute
        return self.standardErrorOfKurtosis_literal()

    # ------------------------------------------------------------------
    # Simple accessors
    # ------------------------------------------------------------------

    def minimum(self) -> float:
        return self._mMin

    def maximum(self) -> float:
        return self._mMax

    def count(self) -> int:
        return self._mNPoints

    def sum(self) -> float:
        return self._mSum

    def getLastAdded(self) -> float:
        return self._mLast

    # ------------------------------------------------------------------
    # toString / compareTo / Comparable
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        return (
            str(self._mNPoints)
            + "\t"
            + str(self.average())
            + "\t"
            + str(self.standardDeviation())
            + "\t"
            + str(self.minimum())
            + "\t"
            + str(self.maximum())
        )

    def toString(self) -> str:
        return str(self)

    def compareTo(self, o: "DescriptiveStatistics") -> int:
        ta: float = self.average()
        oa: float = o.average()
        if ta == oa:
            tv: float = self.variance()
            ov: float = o.variance()
            return 0 if tv == ov else (-1 if tv < ov else 1)
        else:
            return -1 if ta < oa else 1

    def __lt__(self, other: "DescriptiveStatistics") -> bool:
        return self.compareTo(other) < 0

    def __le__(self, other: "DescriptiveStatistics") -> bool:
        return self.compareTo(other) <= 0

    def __gt__(self, other: "DescriptiveStatistics") -> bool:
        return self.compareTo(other) > 0

    def __ge__(self, other: "DescriptiveStatistics") -> bool:
        return self.compareTo(other) >= 0

    # ------------------------------------------------------------------
    # Static factory methods (R4 — split overloads)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_array(ns: Sequence) -> "DescriptiveStatistics":
        """Java: ``public static DescriptiveStatistics compute(Number[] ns)``"""
        ds: DescriptiveStatistics = DescriptiveStatistics()
        for n in ns:
            ds.add(float(n))
        return ds

    @staticmethod
    def compute_collection(ns: Iterable) -> "DescriptiveStatistics":
        """Java: ``public static DescriptiveStatistics compute(Collection<? extends Number> ns)``"""
        ds: DescriptiveStatistics = DescriptiveStatistics()
        for n in ns:
            ds.add(float(n))
        return ds
