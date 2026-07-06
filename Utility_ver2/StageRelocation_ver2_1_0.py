r"""
StageRelocation_ver2_1_0.py — Python port of gov.nist.microanalysis.Utility.StageRelocation

Guide version : 2
Generation    : 1
Port-code fixes: 0

------------------------------------------------------------------------
Original Java source (gov.nist.microanalysis.Utility.StageRelocation)
------------------------------------------------------------------------
/**
 * <p>
 * Implements a mechanism for performing relocation on coordinates in one
 * stage's coordinate system (native) in another stage's coordinate system
 * (relocated). The coordinate systems may differ in origin (deltaX, deltaY),
 * scale (xScale, yScale) and rotation. The may also have one or other of the
 * coordinate axes reversed in direction.
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

CHANGES:
* R10-DEVIATION-1: Java field `mX0rigin` (typo, uses zero digit) renamed to
  `_mXOrigin` for readability. Behaviour is identical.
* `optimize` (private in Java) → `_optimize` per R1.
* `distance` (private in Java) → `_distance` per R1.
* Anonymous inner class `OptimizeFit extends Simplex` promoted to a named local
  class inside `_optimize` (Python has no anonymous classes).
* Constructors unified into one `__init__` dispatched by argument type per spec.
* `RelocatedPoint` exposed at module level as an alias per spec (R2).

JAVA-BUG-1 (preserved verbatim):
  The two-point fit residual check uses `0.001 * Math.abs(x2p - x1p)` as the
  threshold for BOTH residual comparisons. For nearly-horizontal translations
  (x2p ≈ x1p), this threshold approaches zero and can reject valid fits.
  Java source:
    if ((distance(x1, y1, x1p, y1p) > (0.001 * Math.abs(x2p - x1p)))
        || (distance(x2, y2, x2p, y2p) > (0.001 * Math.abs(x2p - x1p))))
        throw new EPQException("Erroneous fit...");

BUG_LEDGER: tuple = (
    ("JAVA-BUG-1", "_optimize",
     "Residual check for both fit points uses `0.001 * abs(x2p - x1p)` as "
     "threshold. For nearly-horizontal translations (x2p ≈ x1p) the threshold "
     "approaches zero, causing valid fits to be rejected. Preserved verbatim."),
)
"""
from __future__ import annotations

import math
from typing import Optional

try:
    from ._epq_compat import EPQException, F64Array
except ImportError:
    try:
        from _epq_compat import EPQException, F64Array  # type: ignore[no-redef]
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility._epq_compat import EPQException, F64Array  # type: ignore[no-redef]

try:
    from .Math2_ver8_1_5 import Math2
except ImportError:
    try:
        from Math2_ver8_1_5 import Math2  # type: ignore[no-redef]
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility.Math2_ver8_1_5 import Math2  # type: ignore[no-redef]

try:
    from .Simplex_ver2_1_1 import Simplex
except ImportError:
    try:
        from Simplex_ver2_1_1 import Simplex  # type: ignore[no-redef]
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility.Simplex_ver2_1_1 import Simplex  # type: ignore[no-redef]

import numpy as np

__all__ = ["StageRelocation", "RelocatedPoint"]

BUG_LEDGER: tuple = (
    ("JAVA-BUG-1", "_optimize",
     "Residual check for both fit points uses `0.001 * abs(x2p - x1p)` as "
     "threshold. For nearly-horizontal translations (x2p ≈ x1p) the threshold "
     "approaches zero, causing valid fits to be rejected. Preserved verbatim."),
)


class StageRelocation:
    """SEM stage coordinate relocation: fits origin, scale and rotation from
    matched native/relocated point pairs, then applies the transform forward
    and in reverse.
    """

    class RelocatedPoint:
        """A matched pair of points — same physical location in two coordinate
        systems (native and relocated).
        """

        def __init__(self, nativePt: list[float], relocatedPt: list[float]) -> None:
            self._mPoint1: list[float] = list(nativePt)    # defensive copy
            self._mPoint2: list[float] = list(relocatedPt) # defensive copy

        def getNativePoint(self) -> list[float]:
            return list(self._mPoint1)

        def getRelocatedPoint(self) -> list[float]:
            return list(self._mPoint2)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    def __init__(
        self,
        pts: Optional[list["StageRelocation.RelocatedPoint"]] = None,
        mirror: bool = False,
    ) -> None:
        # Null-transform defaults
        self._mXOrigin: float = 0.0   # R10-DEVIATION-1: Java mX0rigin → _mXOrigin
        self._mYOrigin: float = 0.0
        self._mXScale: float = 1.0
        self._mYScale: float = 1.0
        self._mRotation: float = 0.0
        self._mError: float = 0.0
        self._mMirrored: bool = False

        if pts is not None:
            self._optimize(pts, mirror)

    # ------------------------------------------------------------------
    # Public transform methods
    # ------------------------------------------------------------------

    def apply(self, pt: list[float]) -> list[float]:
        """Forward transform: native → relocated coordinate system."""
        cos_r: float = math.cos(self._mRotation)
        sin_r: float = math.sin(self._mRotation)
        x: float = pt[0] + self._mXOrigin
        y: float = pt[1] + self._mYOrigin
        return [
            ((x * cos_r) - (y * sin_r)) * self._mXScale,
            ((y * cos_r) + (x * sin_r)) * self._mYScale,
        ]

    def inverse(self, pt: list[float]) -> list[float]:
        """Inverse transform: relocated → native coordinate system."""
        cos_r: float = math.cos(self._mRotation)
        sin_r: float = math.sin(self._mRotation)
        return [
            -self._mXOrigin + ((cos_r * pt[0]) / self._mXScale) + ((sin_r * pt[1]) / self._mYScale),
            (-self._mYOrigin - ((sin_r * pt[0]) / self._mXScale)) + ((cos_r * pt[1]) / self._mYScale),
        ]

    def getResidualError(self) -> float:
        """Average residual error per fit point after optimization."""
        return float(self._mError)

    def isMirrored(self) -> bool:
        """Whether one coordinate axis is mirrored in this transform."""
        return bool(self._mMirrored)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _distance(self, x0: float, y0: float, x1: float, y1: float) -> float:
        return math.sqrt(Math2.sqr(x1 - x0) + Math2.sqr(y1 - y0))

    def _optimize(self, pts: list["StageRelocation.RelocatedPoint"], mirror: bool) -> None:
        self._mMirrored = mirror
        n: int = len(pts)

        if n == 0:
            return

        if n == 1:
            pt1 = pts[0]
            self._mXOrigin = pt1._mPoint2[0] - pt1._mPoint1[0]
            self._mYOrigin = pt1._mPoint2[1] - pt1._mPoint1[1]
            return

        # Two-or-more point fit
        pt1 = pts[0]
        pt2 = pts[1]
        x1: float = pt1._mPoint1[0]
        x1p: float = pt1._mPoint2[0]
        y1: float = pt1._mPoint1[1]
        y1p: float = pt1._mPoint2[1]
        x2: float = pt2._mPoint1[0]
        x2p: float = pt2._mPoint2[0]
        y2: float = pt2._mPoint1[1]
        y2p: float = pt2._mPoint2[1]
        dx: float = x1 - x2
        dy: float = y1 - y2
        dxp: float = x1p - x2p
        dyp: float = y1p - y2p
        dp2: float = Math2.sqr(dxp) + Math2.sqr(dyp)
        d2: float = Math2.sqr(dx) + Math2.sqr(dy)

        if not mirror:
            self._mXOrigin = (
                ((-Math2.sqr(x1p) * x2)
                 + (y1p * (((x2p * y1) - (x2 * y1p) - (x2p * y2)) + (x2 * y2p)))
                 + (x1p * (((x1 + x2) * x2p) + (-dy * y2p))))
                - (x1 * ((Math2.sqr(x2p) - (y1p * y2p)) + Math2.sqr(y2p)))
            ) / dp2
            self._mYOrigin = -((
                (Math2.sqr(x2p) * y1) + (dx * x2p * y1p) + (Math2.sqr(x1p) * y2) + (dyp * y1p * y2)
                - (dyp * y1 * y2p)
                - (x1p * ((x2p * (y1 + y2)) + (dx * y2p)))
            ) / dp2)
            self._mXScale = (dp2 * ((dx * dxp) + (dy * dyp))) / (
                math.sqrt(d2 * dp2) * math.fabs((dx * dxp) + (dy * dyp))
            )
            self._mRotation = math.acos(
                math.fabs((dx * dxp) + (dy * dyp)) / math.sqrt(d2 * dp2)
            )
            self._mYScale = self._mXScale
        else:
            self._mXOrigin = -(((
                (Math2.sqr(x1p) * x2)
                + (((dyp * x2) + (dy * x2p)) * y1p)
                - (x1p * ((x1 * x2p) + (x2 * x2p) + (dy * y2p)))
            ) + (x1 * ((Math2.sqr(x2p) - (y1p * y2p)) + Math2.sqr(y2p)))) / dp2)
            self._mYOrigin = (
                (-(Math2.sqr(x2p) * y1) + (dx * x2p * y1p))
                - (Math2.sqr(x1p) * y2)
                - (dyp * y1p * y2)
                + (dyp * y1 * y2p)
                + (x1p * ((x2p * (y1 + y2)) - (dx * y2p)))
            ) / dp2
            self._mXScale = (dp2 * math.fabs((dx * dxp) - (dy * dyp))) / (
                math.sqrt(d2 * dp2) * ((dx * dxp) - (dy * dyp))
            )
            self._mRotation = math.acos(
                math.fabs((dx * dxp) - (dy * dyp)) / math.sqrt(d2 * dp2)
            )
            self._mYScale = -self._mXScale

        threshold: float = 0.001 * math.fabs(x2p - x1p)  # JAVA-BUG-1: same x-diff for both checks
        if self._distance(x1, y1, x1p, y1p) > threshold:
            self._mXScale = -self._mXScale
            self._mYScale = -self._mYScale
            self._mRotation = math.pi - self._mRotation

        # JAVA-BUG-1 preserved verbatim: threshold recomputed the same way
        if (self._distance(x1, y1, x1p, y1p) > threshold) or (self._distance(x2, y2, x2p, y2p) > threshold):
            raise EPQException("Erroneous fit in optimize procedure - Translation invalid.")

        try:
            if n > 2:
                kX0: int = 0
                kY0: int = 1
                kXScale: int = 2
                kYScale: int = 3
                kRotation: int = 4

                # Capture outer self for use inside the local class
                outer = self
                all_pts = pts

                class OptimizeFit(Simplex):
                    def function(self, v: F64Array) -> float:
                        x0: float = float(v[kX0])
                        y0: float = float(v[kY0])
                        xSc: float = float(v[kXScale])
                        ySc: float = float(v[kYScale])
                        rot: float = float(v[kRotation])
                        cos_r: float = math.cos(rot)
                        sin_r: float = math.sin(rot)
                        res: float = 0.0
                        for rp in all_pts:
                            px: float = rp._mPoint1[0] + x0
                            py: float = rp._mPoint1[1] + y0
                            xp: float = ((px * cos_r) - (py * sin_r)) * xSc
                            yp: float = ((py * cos_r) + (px * sin_r)) * ySc
                            res += math.sqrt(
                                Math2.sqr(xp - rp._mPoint2[0]) + Math2.sqr(yp - rp._mPoint2[1])
                            )
                        return res

                of: OptimizeFit = OptimizeFit()

                # Build starting simplex: (kRotation+2) rows × (kRotation+1) cols = 6×5
                p: F64Array = np.zeros((kRotation + 2, kRotation + 1), dtype=np.float64)
                for i in range(kRotation + 2):
                    p[i, kX0] = self._mXOrigin
                    p[i, kY0] = self._mYOrigin
                    p[i, kXScale] = self._mXScale
                    p[i, kYScale] = self._mYScale
                    p[i, kRotation] = self._mRotation
                p[kX0 + 1, kX0] = self._mXOrigin + 0.1
                p[kY0 + 1, kY0] = self._mYOrigin + 0.1
                p[kXScale + 1, kXScale] = self._mXScale + 0.01
                p[kYScale + 1, kYScale] = self._mYScale + 0.01
                p[kRotation + 1, kRotation] = self._mRotation + 0.01

                of.setTolerance(1.0e-6)
                of.setMaxEvaluations(100)
                y: F64Array = of.perform(p)
                self._mError = of.getBestResult() / n

                self._mXOrigin = float(y[kX0])
                self._mYOrigin = float(y[kY0])
                self._mXScale = float(y[kXScale])
                self._mYScale = float(y[kYScale])
                self._mRotation = float(y[kRotation])

        except EPQException:
            raise
        except Exception as ex:
            raise EPQException("Error during Simplex optimization - Using two point fit.") from ex


# Module-level alias per spec (R2)
RelocatedPoint = StageRelocation.RelocatedPoint
