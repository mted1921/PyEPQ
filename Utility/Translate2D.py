r"""
Translate2D_ver2_1_0.py — Python port of gov.nist.microanalysis.Utility.Translate2D

Guide version : 2
Generation    : 1
Port-code fixes: 0

------------------------------------------------------------------------
Original Java source (gov.nist.microanalysis.Utility.Translate2D)
------------------------------------------------------------------------
/**
 * <p>
 * Translate from one planar (2D) coordinate system into another. The coordinate
 * systems may differ in offset, scale and rotation. (2 + 2 + 1 = 5
 * degrees-of-freedom). The two coordinate systems are calibrated relative to
 * each other using a set of points which are located in each coordinate system.
 * One point is sufficient to calibrate the offset; two points the offset, scale
 * and rotation in which the scale is assumed to be equal in both dimensions.
 * Three or more points overspecify the solution for the full 5
 * degree-of-freedom problem and the algorithm attempts to minimize the mean
 * square error using a Simplex algorithm.
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

CHANGES:
* Anonymous inner class `RefineCalibration extends Simplex` promoted to a named
  local class inside `_calibrate_impl` (Python has no anonymous classes).
* Constructors unified into one `__init__` dispatched by argument type/count.
* `createCalibrationPoint` overloads split to `createCalibrationPoint_arr` /
  `createCalibrationPoint_xy` per R4.
* `toString` → `__str__` + `toString()` alias per R2.
* `reset` (private) → `_reset` per R1.
* R7: `Math.round((best[4] / (2.0 * Math.PI)))` → `int(math.floor(v / (2*pi) + 0.5))`.
* `CalibrationPoint` exposed at module level as an alias per spec (R2).

JAVA-BUG-1 (preserved verbatim):
  `inverse` always returns a 2-element array even when `newCoord` has 3 elements.
  Java source: `final double[] res = new double[2];`
  If callers pass a 3-element coordinate, the third component is silently dropped.

BUG_LEDGER: tuple = (
    ("JAVA-BUG-1", "inverse",
     "Java source: `final double[] res = new double[2]` — inverse() always "
     "returns a 2-element array regardless of input length. If a 3-element "
     "coordinate is passed, the third component is silently dropped. Preserved verbatim."),
)
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

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

try:
    from .HalfUpFormat_ver2_1_1 import HalfUpFormat
except ImportError:
    try:
        from HalfUpFormat_ver2_1_1 import HalfUpFormat  # type: ignore[no-redef]
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility.HalfUpFormat_ver2_1_1 import HalfUpFormat  # type: ignore[no-redef]

try:
    from .UtilException_ver2_1_1 import UtilException
except ImportError:
    try:
        from UtilException_ver2_1_1 import UtilException  # type: ignore[no-redef]
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility.UtilException_ver2_1_1 import UtilException  # type: ignore[no-redef]

__all__ = ["Translate2D", "CalibrationPoint"]

BUG_LEDGER: tuple = (
    ("JAVA-BUG-1", "inverse",
     "Java source: `final double[] res = new double[2]` — inverse() always "
     "returns a 2-element array regardless of input length. If a 3-element "
     "coordinate is passed, the third component is silently dropped. Preserved verbatim."),
)


class Translate2D:
    """2D coordinate-system transform: offset, scale, rotation, optional x-axis inversion.

    One calibration point: offset only.
    Two points: offset, scale (equal in both axes), rotation.
    Three or more points: full 5-DOF fit via Nelder-Mead Simplex.
    """

    _SCALE: float = 1.0e-3  # mm

    # ------------------------------------------------------------------
    # Inner class
    # ------------------------------------------------------------------

    class CalibrationPoint:
        """Matched pair of the same physical point in two coordinate systems."""

        def __init__(self, orig: list[float], newPt: list[float]) -> None:
            self._mOldPoint: list[float] = list(orig)    # defensive copy
            self._mNewPoint: list[float] = list(newPt)   # defensive copy

        def getX0(self) -> float:
            return float(self._mOldPoint[0])

        def getY0(self) -> float:
            return float(self._mOldPoint[1])

        def getX1(self) -> float:
            return float(self._mNewPoint[0])

        def getY1(self) -> float:
            return float(self._mNewPoint[1])

        def different(self, cp: "Translate2D.CalibrationPoint") -> bool:
            d: float = max(
                Math2.distance(self._mOldPoint, cp._mOldPoint),
                Math2.distance(self._mNewPoint, cp._mNewPoint),
            )
            return d > (1.0e-3 * Translate2D._SCALE)

    # ------------------------------------------------------------------
    # Static factory methods (R4: overload split _arr / _xy)
    # ------------------------------------------------------------------

    @staticmethod
    def createCalibrationPoint_arr(
        oldCoord: list[float], newCoord: list[float]
    ) -> "Translate2D.CalibrationPoint":
        return Translate2D.CalibrationPoint(oldCoord, newCoord)

    @staticmethod
    def createCalibrationPoint_xy(
        oldX: float, oldY: float, newX: float, newY: float
    ) -> "Translate2D.CalibrationPoint":
        return Translate2D.CalibrationPoint([oldX, oldY], [newX, newY])

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    def __init__(self, *args) -> None:  # type: ignore[no-untyped-def]
        # Java field initializers — all constructors start with these defaults
        self._mOffset: list[float] = [0.0, 0.0]
        self._mScale: list[float] = [1.0, 1.0]
        self._mRotation: float = 0.0
        self._mXAxisInverted: bool = False

        if len(args) == 0:
            # public Translate2D() — null translation
            self._reset()
        elif len(args) == 1 and isinstance(args[0], Translate2D):
            # public Translate2D(Translate2D t2d) — copy constructor
            t2d: Translate2D = args[0]
            self._mOffset[0] = t2d._mOffset[0]
            self._mOffset[1] = t2d._mOffset[1]
            self._mScale[0] = t2d._mScale[0]
            self._mScale[1] = t2d._mScale[1]
            self._mRotation = t2d._mRotation
            self._mXAxisInverted = t2d._mXAxisInverted
        elif len(args) == 2:
            # public Translate2D(double[] offset, double rotation)
            offset_arg, rotation_arg = args[0], args[1]
            self._reset()
            self._mOffset = list(offset_arg)
            self._mRotation = float(rotation_arg)
        elif len(args) == 4:
            # public Translate2D(double[] offset, double[] scale, double rotation, boolean invertXAxis)
            offset_arg, scale_arg, rotation_arg, invert_arg = args
            self._mOffset = list(offset_arg)
            self._mScale = list(scale_arg)
            self._mRotation = float(rotation_arg)
            self._mXAxisInverted = bool(invert_arg)
        else:
            raise TypeError(
                f"Translate2D() accepts 0, 1 (Translate2D), 2, or 4 arguments ({len(args)} given)"
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _reset(self) -> None:
        self._mOffset[0] = 0.0
        self._mOffset[1] = 0.0
        self._mScale[0] = 1.0
        self._mScale[1] = 1.0
        self._mRotation = 0.0
        self._mXAxisInverted = False

    def _calibrate_impl(
        self, calPts: list["Translate2D.CalibrationPoint"], literal: bool
    ) -> float:
        self._reset()
        n: int = len(calPts)

        if n >= 2:
            pt1 = calPts[0]
            pt2 = calPts[1]
            x1: float = pt1._mOldPoint[0]
            x1p: float = pt1._mNewPoint[0]
            y1: float = pt1._mOldPoint[1]
            y1p: float = pt1._mNewPoint[1]
            x2: float = pt2._mOldPoint[0]
            x2p: float = pt2._mNewPoint[0]
            y2: float = pt2._mOldPoint[1]
            y2p: float = pt2._mNewPoint[1]
            dx: float = x1 - x2
            dy: float = y1 - y2
            dxp: float = x1p - x2p
            dyp: float = y1p - y2p
            dp2: float = Math2.sqr(dxp) + Math2.sqr(dyp)
            d2: float = Math2.sqr(dx) + Math2.sqr(dy)
            best_i: int = -1
            best_err: float = float("inf")
            for i in range(5):
                ii: int = best_i if i == 4 else i
                axis_inverted: bool = (ii % 2 == 0)
                invert: bool = (ii < 2)
                self._mXAxisInverted = invert
                if not axis_inverted:
                    # Mirror formula
                    self._mOffset[0] = -(((
                        (Math2.sqr(x1p) * x2)
                        + (((dyp * x2) + (dy * x2p)) * y1p)
                        - (x1p * ((x1 * x2p) + (x2 * x2p) + (dy * y2p)))
                    ) + (x1 * ((Math2.sqr(x2p) - (y1p * y2p)) + Math2.sqr(y2p)))) / dp2)
                    self._mOffset[1] = (
                        (-(Math2.sqr(x2p) * y1) + (dx * x2p * y1p))
                        - (Math2.sqr(x1p) * y2) - (dyp * y1p * y2)
                        + (dyp * y1 * y2p)
                        + (x1p * ((x2p * (y1 + y2)) - (dx * y2p)))
                    ) / dp2
                    ma = math.fabs((dx * dxp) - (dy * dyp))
                    self._mScale[0] = (
                        (dp2 * ma) / (math.sqrt(d2 * dp2) * ((dx * dxp) - (dy * dyp)))
                        if ma > 0.0 else 1.0
                    )
                    self._mRotation = math.acos(
                        math.fabs((dx * dxp) - (dy * dyp)) / math.sqrt(d2 * dp2)
                    )
                    self._mScale[1] = self._mScale[0]
                else:
                    # Non-mirror formula
                    self._mOffset[0] = (
                        ((-Math2.sqr(x1p) * x2)
                         + (y1p * (((x2p * y1) - (x2 * y1p) - (x2p * y2)) + (x2 * y2p)))
                         + (x1p * (((x1 + x2) * x2p) + (-dy * y2p))))
                        - (x1 * ((Math2.sqr(x2p) - (y1p * y2p)) + Math2.sqr(y2p)))
                    ) / dp2
                    self._mOffset[1] = -((
                        (Math2.sqr(x2p) * y1) + (dx * x2p * y1p)
                        + (Math2.sqr(x1p) * y2) + (dyp * y1p * y2)
                        - (dyp * y1 * y2p)
                        - (x1p * ((x2p * (y1 + y2)) + (dx * y2p)))
                    ) / dp2)
                    ma = math.fabs((dx * dxp) + (dy * dyp))
                    self._mScale[0] = (
                        (dp2 * ((dx * dxp) + (dy * dyp))) / (math.sqrt(d2 * dp2) * ma)
                        if ma > 0.0 else 1.0
                    )
                    self._mRotation = math.acos(
                        math.fabs((dx * dxp) + (dy * dyp)) / math.sqrt(d2 * dp2)
                    )
                    self._mScale[1] = self._mScale[0]
                if invert:
                    self._mScale[0] = -self._mScale[0]
                    self._mScale[1] = self._mScale[1]  # no-op; preserved from Java
                if i != 4:
                    err: float = self.error(calPts)
                    if err < best_err:
                        best_i = i
                        best_err = err
        elif n == 1:
            # 1-point fit: offset only
            pt1 = calPts[0]
            self._mOffset[0] = pt1._mNewPoint[0] - pt1._mOldPoint[0]
            self._mOffset[1] = pt1._mNewPoint[1] - pt1._mOldPoint[1]

        if n >= 3:
            all_pts = calPts

            class RefineCalibration(Simplex):
                def function(self, x: F64Array) -> float:
                    sc_x: float = float(x[0])
                    sc_y: float = float(x[1])
                    off_x: float = float(x[2])
                    off_y: float = float(x[3])
                    rot: float = float(x[4])
                    cos_r: float = math.cos(rot)
                    sin_r: float = math.sin(rot)
                    sum_sqr: float = 0.0
                    for cp in all_pts:
                        res_x: float = (
                            ((cp._mOldPoint[0] + off_x) * cos_r)
                            - ((cp._mOldPoint[1] + off_y) * sin_r)
                        ) * sc_x
                        res_y: float = (
                            ((cp._mOldPoint[1] + off_y) * cos_r)
                            + ((cp._mOldPoint[0] + off_x) * sin_r)
                        ) * sc_y
                        sum_sqr += (
                            Math2.sqr(res_x - cp._mNewPoint[0])
                            + Math2.sqr(res_y - cp._mNewPoint[1])
                        )
                    return sum_sqr

            rc: RefineCalibration = RefineCalibration()
            sc_off: float = 0.01
            for cp in calPts:
                sc_off = max(
                    max(sc_off, abs(cp._mOldPoint[0] - cp._mNewPoint[0]) / 100.0),
                    abs(cp._mOldPoint[1] - cp._mNewPoint[1]) / 100.0,
                )
            center: list[float] = [
                self._mScale[0], self._mScale[1],
                self._mOffset[0], self._mOffset[1],
                self._mRotation,
            ]
            scale_arr: list[float] = [
                abs(self._mScale[0]) / 10.0,
                abs(self._mScale[1]) / 10.0,
                sc_off, sc_off, 0.01,
            ]
            try:
                p_arr: F64Array = Simplex.randomizedStartingPoints(center, scale_arr)
                best: F64Array = rc.perform_literal(p_arr) if literal else rc.perform(p_arr)
                self._mScale[0] = float(best[0])
                self._mScale[1] = float(best[1])
                self._mOffset[0] = float(best[2])
                self._mOffset[1] = float(best[3])
                b4: float = float(best[4])
                # R7: Math.round(double) → int(math.floor(x + 0.5))
                self._mRotation = b4 - int(math.floor(b4 / (2.0 * math.pi) + 0.5)) * (2.0 * math.pi)
            except Exception:
                pass  # Can't refine it...

        return self.error(calPts)

    # ------------------------------------------------------------------
    # Public instance methods
    # ------------------------------------------------------------------

    def calibrate(self, calPts: list["Translate2D.CalibrationPoint"]) -> float:
        """Fit the transform to calPts; returns RMS residual. Uses scipy Simplex."""
        return self._calibrate_impl(calPts, literal=False)

    def calibrate_literal(self, calPts: list["Translate2D.CalibrationPoint"]) -> float:
        """Fit the transform using literal Java Nelder-Mead (perform_literal)."""
        return self._calibrate_impl(calPts, literal=True)

    def error(self, calPts: list["Translate2D.CalibrationPoint"]) -> float:  # SCIPY-NONE
        return self.error_literal(calPts)

    def error_literal(self, calPts: list["Translate2D.CalibrationPoint"]) -> float:
        sum_sqr: float = 0.0
        for calib in calPts:
            new_pt: list[float] = self.compute(calib._mOldPoint)
            sum_sqr += (
                Math2.sqr(new_pt[0] - calib._mNewPoint[0])
                + Math2.sqr(new_pt[1] - calib._mNewPoint[1])
            )
        return math.sqrt(sum_sqr / len(calPts))

    def compute(self, oldCoord: list[float]) -> list[float]:  # SCIPY-NONE
        return self.compute_literal(oldCoord)

    def compute_literal(self, oldCoord: list[float]) -> list[float]:
        res: list[float] = list(oldCoord)
        cos_r: float = math.cos(self._mRotation)
        sin_r: float = math.sin(self._mRotation)
        res[0] = (
            ((oldCoord[0] + self._mOffset[0]) * cos_r)
            - ((oldCoord[1] + self._mOffset[1]) * sin_r)
        ) * self._mScale[0]
        res[1] = (
            ((oldCoord[1] + self._mOffset[1]) * cos_r)
            + ((oldCoord[0] + self._mOffset[0]) * sin_r)
        ) * self._mScale[1]
        if len(res) > 2:
            res[2] += self._mRotation
        return res

    def inverse(self, newCoord: list[float]) -> list[float]:  # SCIPY-NONE
        return self.inverse_literal(newCoord)

    def inverse_literal(self, newCoord: list[float]) -> list[float]:
        res: list[float] = [0.0, 0.0]  # JAVA-BUG-1: always 2-element
        cos_r: float = math.cos(self._mRotation)
        sin_r: float = math.sin(self._mRotation)
        res[0] = (
            -self._mOffset[0]
            + ((cos_r * newCoord[0]) / self._mScale[0])
            + ((sin_r * newCoord[1]) / self._mScale[1])
        )
        res[1] = (
            (-self._mOffset[1] - ((sin_r * newCoord[0]) / self._mScale[0]))
            + ((cos_r * newCoord[1]) / self._mScale[1])
        )
        return res

    def isXAxisInverted(self) -> bool:
        return bool(self._mXAxisInverted)

    def getXScale(self) -> float:
        return float(self._mScale[0])

    def getYScale(self) -> float:
        return float(self._mScale[1])

    def getXOffset(self) -> float:
        return float(self._mOffset[0])

    def getYOffset(self) -> float:
        return float(self._mOffset[1])

    def getRotation(self) -> float:
        return float(self._mRotation)

    def __str__(self) -> str:
        nf = HalfUpFormat("0.00")
        sb: list[str] = []
        sb.append("Translation[Scale=[")
        sb.append(nf.format(self._mScale[0]))
        sb.append(",")
        sb.append(nf.format(self._mScale[1]))
        sb.append("],Offset=[")
        sb.append(nf.format(self._mOffset[0]))
        sb.append(",")
        sb.append(nf.format(self._mOffset[1]))
        sb.append("],Rotation=")
        sb.append(nf.format(math.degrees(self._mRotation)))
        sb.append("]]")
        return "".join(sb)

    def toString(self) -> str:  # R2: Java-style call-site alias
        return str(self)


# Module-level alias per spec (R2)
CalibrationPoint = Translate2D.CalibrationPoint
