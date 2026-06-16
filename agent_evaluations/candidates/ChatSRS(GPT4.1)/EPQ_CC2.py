r"""
AdaptiveRungeKutta_ver1_1_2.py — Python port of gov.nist.microanalysis.Utility.AdaptiveRungeKutta

Guide version : 1
Generation    : 1
Port-code fixes: 2

------------------------------------------------------------------------
Original Java source (gov.nist.microanalysis.Utility.AdaptiveRungeKutta)
------------------------------------------------------------------------
/**
 * <p>
 * Adaptive Runge-Kutta ODE integrator base class.
 * </p>
 * <p>
 * This class implements the adaptive Runge-Kutta integration for ordinary
 * differential equations (ODEs) of the form dy/dx = f(x, y), where y is a
 * vector of variables. The step size is automatically adjusted to maintain
 * the specified error tolerance.
 * </p>
 * <p>
 * Not thread-safe.
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
import abc, math, sys
import numpy as np
from typing import Optional, Sequence

try:
    from ._epq_compat import EPQException, F64Array
except ImportError:
    try:
        from _epq_compat import EPQException, F64Array  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility._epq_compat import EPQException, F64Array  # type: ignore

# BUG_LEDGER: tuple = ()  # no bugs identified; see CONVERSION_CONTEXT.md

class AdaptiveRungeKutta(abc.ABC):
    """
    Adaptive Runge-Kutta ODE integrator base class.
    Not thread-safe.
    """

    BUG_LEDGER: tuple = ()  # no bugs identified

    def __init__(self, nVars: int) -> None:
        self.mNVariables: int = nVars
        self.mHDid: float = 0.0
        self.mHNext: float = 0.0
        self.mSaveInterval: float = 0.0
        self.mMinStepSize: float = 0.0
        self.mXSave: Optional[F64Array] = None
        self.mYSave: Optional[F64Array] = None  # shape: (kMax, nVars)
        self.mNSaved: int = 0
        self.mMaxSteps: int = 0
        self.mNOk: int = 0
        self.mNBad: int = 0
        self.mWs2: Optional[F64Array] = None
        self.mWs3: Optional[F64Array] = None
        self.mWs4: Optional[F64Array] = None
        self.mWs5: Optional[F64Array] = None
        self.mWs6: Optional[F64Array] = None
        self.mYTemp: Optional[F64Array] = None
        self.mYErr: Optional[F64Array] = None
        self.mQcYTemp: Optional[F64Array] = None

    @abc.abstractmethod
    def derivatives(self, x: float, y: F64Array, dydx: F64Array) -> None:
        """
        Compute derivatives: dydx = f(x, y)
        This method must be implemented by subclasses.
        """
        pass

    def setSaveInterval(self, interval: float) -> None:
        self.mSaveInterval = interval

    def clearSaveInterval(self) -> None:
        self.mSaveInterval = 0.0
        self.mXSave = None
        self.mYSave = None
        self.mNSaved = 0
        self.mMaxSteps = 0

    def getNSaved(self) -> int:
        return self.mNSaved

    def getX(self, i: int) -> float:
        if self.mXSave is None or i < 0 or i >= self.mNSaved:
            raise EPQException("Index out of range in getX")
        return float(self.mXSave[i])

    def getY(self, i: int) -> F64Array:
        if self.mYSave is None or i < 0 or i >= self.mNSaved:
            raise EPQException("Index out of range in getY")
        return self.mYSave[i, :].copy()

    def setMaxSteps(self, maxSteps: int) -> None:
        self.mMaxSteps = maxSteps

    def setMinStepSize(self, minStep: float) -> None:
        self.mMinStepSize = minStep

    def getNVariables(self) -> int:
        return self.mNVariables

    def getStepCount(self) -> int:
        return self.mNOk + self.mNBad

    def getGoodStepCount(self) -> int:
        return self.mNOk

    def getBadStepCount(self) -> int:
        return self.mNBad

    def integrate(
        self,
        x1: float,
        x2: float,
        ystart: F64Array,
        eps: float,
        h1: float
    ) -> F64Array:
        """
        Integrate from x1 to x2 with initial ystart, error tolerance eps,
        and initial step h1. Mutates ystart in place and returns the final y.
        """
        nvar: int = self.mNVariables
        x: float = x1
        h: float = h1
        y: F64Array = ystart.copy()
        dydx: F64Array = np.zeros(nvar, dtype=np.float64)
        yscal: F64Array = np.zeros(nvar, dtype=np.float64)
        yout: F64Array = np.zeros(nvar, dtype=np.float64)
        yerr: F64Array = np.zeros(nvar, dtype=np.float64)

        if self.mSaveInterval > 0.0:
            kMax: int = int(math.floor(abs(x2 - x1) / self.mSaveInterval + 0.5)) + 2
            self.mXSave = np.zeros(kMax, dtype=np.float64)
            self.mYSave = np.zeros((kMax, nvar), dtype=np.float64)
            self.mNSaved = 0
            self.mMaxSteps = kMax

        self.mNOk = 0
        self.mNBad = 0

        for step in range(self.mMaxSteps if self.mMaxSteps > 0 else 1000000):
            self.derivatives(x, y, dydx)
            for i in range(nvar):
                yscal[i] = abs(y[i]) + abs(dydx[i] * h) + 1.0e-30

            if (x + h - x2) * (x + h - x1) > 0.0:
                h = x2 - x

            hdid, hnext = self._qcStep(x, y, dydx, h, eps, yscal)
            x += hdid
            h = hnext

            if self.mSaveInterval > 0.0 and self.mXSave is not None and self.mYSave is not None:
                if self.mNSaved < self.mMaxSteps:
                    self.mXSave[self.mNSaved] = x
                    self.mYSave[self.mNSaved, :] = y
                    self.mNSaved += 1

            if (x - x2) * (x2 - x1) >= 0.0:
                ystart[:] = y
                return y

            if abs(h) < self.mMinStepSize:
                raise EPQException("AdaptiveRungeKutta: Step size too small.")

        raise EPQException("AdaptiveRungeKutta: Too many steps.")

    # ---- Private helpers ----

    def _sign(self, magnitude: float, sign: float) -> float:
        # See CONVERSION_CONTEXT.md: Java sign(m, -0.0) returns +|m| if sign >= 0.0
        return abs(magnitude) if sign >= 0.0 else -abs(magnitude)

    def _baseStep(
        self,
        x: float,
        y: F64Array,
        dydx: F64Array,
        h: float,
        yout: F64Array,
        yerr: F64Array
    ) -> None:
        """
        Subclass must implement the actual Runge-Kutta step.
        This is a placeholder for the base class.
        """
        raise NotImplementedError("Subclasses must implement _baseStep.")

    def _qcStep(
        self,
        x: float,
        y: F64Array,
        dydx: F64Array,
        htry: float,
        eps: float,
        yscal: F64Array
    ) -> tuple[float, float]:
        """
        Quality-controlled step: tries to step from x using htry, adjusting
        step size to maintain error tolerance eps.
        Returns (hdid, hnext).
        """
        nvar: int = self.mNVariables
        ytemp: F64Array = np.zeros(nvar, dtype=np.float64)
        yerr: F64Array = np.zeros(nvar, dtype=np.float64)
        h: float = htry
        errmax: float = 0.0
        while True:
            self._baseStep(x, y, dydx, h, ytemp, yerr)
            errmax = 0.0
            for i in range(nvar):
                err = abs(yerr[i] / yscal[i])
                if err > errmax:
                    errmax = err
            errmax /= eps
            if errmax <= 1.0:
                break
            htemp = 0.9 * h * (errmax ** -0.2)
            h = self._sign(max(abs(htemp), 0.1 * abs(h)), h)
            self.mNBad += 1
            if abs(h) < self.mMinStepSize:
                raise EPQException("AdaptiveRungeKutta: Step size under minimum.")
        y[:] = ytemp
        self.mNOk += 1
        hnext = 0.9 * h * (errmax ** -0.2)
        hnext = self._sign(max(abs(hnext), 0.1 * abs(h)), h)
        self.mHDid = h
        self.mHNext = hnext
        return h, hnext

    def _clearWorkspace(self) -> None:
        self.mWs2 = None
        self.mWs3 = None
        self.mWs4 = None
        self.mWs5 = None
        self.mWs6 = None
        self.mYTemp = None
        self.mYErr = None
        self.mQcYTemp = None