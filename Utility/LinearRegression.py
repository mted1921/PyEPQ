r"""
LinearRegression_ver2_1_1.py — Python port of gov.nist.microanalysis.Utility.LinearRegression

Guide version : 2
Generation    : 1
Port-code fixes: 1

------------------------------------------------------------------------
Original Java source (gov.nist.microanalysis.Utility.LinearRegression)
------------------------------------------------------------------------
/**
 * <p>
 * A class implementing a basic LLSQ regression to a line.
 * </p>
 * <p>
 * Copyright: Pursuant to title 17 Section 105 of the United States Code this
 * software is not subject to copyright protection and is in the public domain
 * </p>
 * <p>
 * Institution: National Institute of Standards and Technology
 * </p>
 *
 * @author Nicholas
 * @version 1.0
 */
------------------------------------------------------------------------

CHANGES (from Java):

  R4   — addDatum(x,y) and addDatum(x,y,dy) split → addDatum_xy / addDatum_xye.
          removeDatum(x,y) and removeDatum(x,y,dy) split → removeDatum_xy / removeDatum_xye.

  R4   — Line(slope,intercept) is the primary constructor; two-argument
          Line(x0,y0,x1,y1) becomes a classmethod Line.from_two_points.

  R2   — LazyEvaluate<Line> anonymous inner class → _LineLazy inner class
          (defined inside _make_line_lazy factory) with a captured reference
          to the outer LinearRegression instance.

  FIX-1 (ZeroDivisionError): getRSquared() can produce a zero denominator when
          all y values are identical (degenerate case: mCount*mSyy == mSy^2).
          Java's double arithmetic returns NaN via IEEE-754 (0.0/0.0); Python
          raises ZeroDivisionError.  Guard added: return float('nan') when den==0.

JAVA-NOTE-1 — correlation() divides by sqrt(mS * mSxx).
  If all x are equal then mSxx = 0 → sqrt argument is 0 → result is NaN.
  Java IEEE-754: same NaN propagation.  Disposition: preserve.

JAVA-NOTE-2 — chiSquared() clamps with Math.max(0.0, res).
  Java source: `// assert res >= 0.0 : ...; return Math.max(0.0, res);`
  Floating-point rounding can make res slightly negative for near-perfect
  fits.  Disposition: preserve the clamp; omit the commented-out assert.

BUG_LEDGER: tuple = (
    ("JAVA-NOTE-1", "correlation",
     "NaN when all x equal (mSxx == 0). Preserved per IEEE-754."),
    ("JAVA-NOTE-2", "chiSquared",
     "Negative rounding: Java clamps with Math.max(0, res). Preserved."),
)
"""
from __future__ import annotations

import math
from typing import Optional

try:
    from ._epq_compat import JamaMatrix
except ImportError:
    try:
        from _epq_compat import JamaMatrix  # type: ignore[no-redef]
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility_ver2._epq_compat import JamaMatrix  # type: ignore[no-redef]

try:
    from .LazyEvaluate_ver2_1_2 import LazyEvaluate
except ImportError:
    try:
        from LazyEvaluate_ver2_1_2 import LazyEvaluate  # type: ignore[no-redef]
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility_ver2.LazyEvaluate_ver2_1_2 import LazyEvaluate  # type: ignore[no-redef]

try:
    from .Math2_ver8_1_5 import Math2
except ImportError:
    try:
        from Math2_ver8_1_5 import Math2  # type: ignore[no-redef]
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility_ver2.Math2_ver8_1_5 import Math2  # type: ignore[no-redef]


BUG_LEDGER: tuple = (
    ("JAVA-NOTE-1", "correlation",
     "NaN when all x equal (mSxx == 0). Preserved per IEEE-754."),
    ("JAVA-NOTE-2", "chiSquared",
     "Negative rounding: Java clamps with Math.max(0, res). Preserved."),
)


class LinearRegression:
    """Port of gov.nist.microanalysis.Utility.LinearRegression.

    Accumulates weighted (x, y) data and computes the best-fit line via
    weighted least squares.  The fit result is cached lazily.
    """

    class Line:
        """Inner value object holding slope and intercept."""

        def __init__(self, slope: float, intercept: float) -> None:
            self._mSlope: float = slope
            self._mIntercept: float = intercept

        @classmethod
        def from_two_points(
            cls, x0: float, y0: float, x1: float, y1: float
        ) -> "LinearRegression.Line":
            slope: float = (y1 - y0) / (x1 - x0)
            intercept: float = y0 - slope * x0
            return cls(slope, intercept)

        def getXIntercept(self) -> float:
            return -self._mIntercept / self._mSlope

        def computeY(self, x: float) -> float:
            return self._mIntercept + self._mSlope * x

        def computeX(self, y: float) -> float:
            return (y - self._mIntercept) / self._mSlope

        def getSlope(self) -> float:
            return self._mSlope

        def getIntercept(self) -> float:
            return self._mIntercept

    def _make_line_lazy(self) -> LazyEvaluate:
        lr = self
        class _LineLazy(LazyEvaluate):
            def compute(self_inner) -> "LinearRegression.Line":
                return lr._compute_line()
        return _LineLazy()

    def _compute_line(self) -> "LinearRegression.Line":
        slope: float = math.inf
        intercept: float = float('nan')
        den: float = (self._mS * self._mSxx) - (self._mSx * self._mSx)
        if den != 0.0:
            slope = ((self._mS * self._mSxy) - (self._mSx * self._mSy)) / den
            intercept = ((self._mSxx * self._mSy) - (self._mSx * self._mSxy)) / den
            self._mCovariance = JamaMatrix.zeros(2, 2)
            self._mCovariance.set(0, 0, self._mSxx / den)
            self._mCovariance.set(1, 1, self._mS / den)
            c: float = -self._mSx / den
            self._mCovariance.set(1, 0, c)
            self._mCovariance.set(0, 1, c)
        return LinearRegression.Line(slope, intercept)

    def __init__(self) -> None:
        self._mCovariance: Optional[JamaMatrix] = None
        self._mLine: LazyEvaluate = self._make_line_lazy()
        self.clear()

    def clear(self) -> None:
        self._mS: float = 0.0
        self._mSxx: float = 0.0
        self._mSx: float = 0.0
        self._mSxy: float = 0.0
        self._mSy: float = 0.0
        self._mSyy: float = 0.0
        self._mCount: int = 0
        self._mLine.reset()

    def setData(self, x: list, y: list) -> None:
        self.clear()
        self.addData(x, y)

    def addData(self, x: list, y: list) -> None:
        for xi, yi in zip(x, y):
            self.addDatum_xy(float(xi), float(yi))

    def addDatum_xy(self, x: float, y: float) -> None:
        self.addDatum_xye(x, y, 1.0)

    def addDatum_xye(self, x: float, y: float, dy: float) -> None:
        dy2: float = dy * dy
        self._mS += 1.0 / dy2
        self._mSx += x / dy2
        self._mSxx += (x * x) / dy2
        self._mSxy += (x * y) / dy2
        self._mSy += y / dy2
        self._mSyy += (y * y) / dy2
        self._mCount += 1
        self._mLine.reset()

    def removeDatum_xye(self, x: float, y: float, dy: float) -> None:
        dy2: float = dy * dy
        self._mS -= 1.0 / dy2
        self._mSx -= x / dy2
        self._mSxx -= (x * x) / dy2
        self._mSxy -= (x * y) / dy2
        self._mSy -= y / dy2
        self._mSyy -= (y * y) / dy2
        self._mCount -= 1
        self._mLine.reset()

    def removeDatum_xy(self, x: float, y: float) -> None:
        self.removeDatum_xye(x, y, 1.0)

    def getSlope(self) -> float:
        return self._mLine.get().getSlope()

    def getIntercept(self) -> float:
        return self._mLine.get().getIntercept()

    def getXIntercept(self) -> float:
        return self._mLine.get().getXIntercept()

    def computeY(self, x: float) -> float:
        return self._mLine.get().computeY(x)

    def computeX(self, y: float) -> float:
        return self._mLine.get().computeX(y)

    def covariance(self) -> Optional[JamaMatrix]:
        self._mLine.get()
        return self._mCovariance

    def correlation(self) -> float:
        self._mLine.get()
        return -self._mSxx / math.sqrt(self._mS * self._mSxx)  # JAVA-NOTE-1

    def chiSquared(self) -> float:
        a: float = self.getIntercept()
        b: float = self.getSlope()
        res: float = (
            (self._mSyy - (2.0 * (((a * self._mSy) - (a * b * self._mSx)) + (b * self._mSxy))))
            + (b * b * self._mSxx)
            + (a * a * self._mS)
        )
        return max(0.0, res)  # JAVA-NOTE-2: clamp

    def goodnessOfFit(self) -> float:
        return Math2.gammq(0.5 * (self._mCount - 2), self.chiSquared() / 2.0)

    def getRSquared(self) -> float:
        num: float = Math2.sqr((self._mCount * self._mSxy) - (self._mSx * self._mSy))
        den: float = (
            ((self._mCount * self._mSxx) - (self._mSx * self._mSx))
            * ((self._mCount * self._mSyy) - (self._mSy * self._mSy))
        )
        if den == 0.0:
            return float('nan')  # FIX-1: Java IEEE-754 returns NaN (0.0/0.0); Python raises
        return num / den

    def getCount(self) -> int:
        return self._mCount

    def __str__(self) -> str:
        return "Y=" + str(self.getSlope()) + " X + " + str(self.getIntercept())

    def toString(self) -> str:
        return self.__str__()

    def getResult(self) -> "LinearRegression.Line":
        return self._mLine.get()


Line = LinearRegression.Line
