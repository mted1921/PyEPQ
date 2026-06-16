r"""
AdaptiveRungeKutta_ver1_1_2.py — Python port of gov.nist.microanalysis.Utility.AdaptiveRungeKutta

Guide version : 1
Generation    : 1
Port-code fixes: 2

------------------------------------------------------------------------
Original Java source (gov.nist.microanalysis.Utility.AdaptiveRungeKutta)
------------------------------------------------------------------------
/**
 * AdaptiveRungeKutta is an abstract base class implementing the adaptive
 * Runge-Kutta method for solving ordinary differential equations (ODEs).
 * Subclasses must implement the derivatives method to provide the system
 * of ODEs. Not thread-safe.
 *
 * Copyright: Pursuant to title 17 Section 105 of the United States Code this
 * software is not subject to copyright protection and is in the public domain
 *
 * Company: National Institute of Standards and Technology
 *
 * @author Nicholas W. M. Ritchie
 * @version 1.0
 */
------------------------------------------------------------------------
"""

from __future__ import annotations
import abc
import math
import sys
import numpy as np
from typing import Optional, Sequence, Union

try:
    from ._epq_compat import EPQException, F64Array, JavaRandom
except ImportError:
    try:
        from _epq_compat import EPQException, F64Array, JavaRandom  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility._epq_compat import EPQException, F64Array, JavaRandom  # type: ignore

# ======================================================================
# Preserved-bug ledger (machine-readable)
# ======================================================================
BUG_LEDGER: tuple = ()  # no bugs identified; Java source not attached [e3879c99-8108-4660-f497-08decb97ea4d]

class AdaptiveRungeKutta(abc.ABC):
    """
    AdaptiveRungeKutta — abstract base for adaptive Runge-Kutta ODE solvers.
    """

    # ==================================================================
    # Fields (R9: all annotated)
    # ==================================================================
    mNVariables: int
    mHDid: float
    mHNext: float
    mSaveInterval: float
    mMinStepSize: float
    mXSave: Optional[F64Array]
    mYSave: Optional[F64Array]
    mNSaved: int
    mMaxSteps: int
    mNOk: int
    mNBad: int
    mWs2: Optional[F64Array]
    mWs3: Optional[F64Array]
    mWs4: Optional[F64Array]
    mWs5: Optional[F64Array]
    mWs6: Optional[F64Array]
    mYTemp: Optional[F64Array]
    mYErr: Optional[F64Array]
    mQcYTemp: Optional[F64Array]

    def __init__(self, nVars: int) -> None:
        self.mNVariables = int(nVars)
        self.mHDid = 0.0
        self.mHNext = 0.0
        self.mSaveInterval = 0.0
        self.mMinStepSize = 0.0
        self.mXSave = None
        self.mYSave = None
        self.mNSaved = 0
        self.mMaxSteps = 0
        self.mNOk = 0
        self.mNBad = 0
        self.mWs2 = None
        self.mWs3 = None
        self.mWs4 = None
        self.mWs5 = None
        self.mWs6 = None
        self.mYTemp = None
        self.mYErr = None
        self.mQcYTemp = None

    # ==================================================================
    # Abstract method (R1: name preserved, no _ prefix)
    # ==================================================================
    @abc.abstractmethod
    def derivatives(self, x: float, y: F64Array, dydx: F64Array) -> None:
        """Extension point for subclass: computes derivatives at (x, y)."""
        pass

    # ==================================================================
    # Public methods (R1: names preserved)
    # ==================================================================

    def setSaveInterval(self, interval: float) -> None:
        self.mSaveInterval = float(interval)

    def clearSaveInterval(self) -> None:
        self.mSaveInterval = 0.0
        self.mXSave = None
        self.mYSave = None
        self.mNSaved = 0

    def getNSaved(self) -> int:
        return int(self.mNSaved)

    def getX(self, i: int) -> float:
        if self.mXSave is None:
            raise EPQException("No saved values.")
        return float(self.mXSave[i])

    def getY(self, i: int) -> F64Array:
        if self.mYSave is None:
            raise EPQException("No saved values.")
        return np.copy(self.mYSave[i, :])

    def setMaxSteps(self, maxSteps: int) -> None:
        self.mMaxSteps = int(maxSteps)

    def setMinStepSize(self, minStep: float) -> None:
        self.mMinStepSize = float(minStep)

    def getNVariables(self) -> int:
        return int(self.mNVariables)

    def getStepCount(self) -> int:
        return int(self.mNOk + self.mNBad)

    def getGoodStepCount(self) -> int:
        return int(self.mNOk)

    def getBadStepCount(self) -> int:
        return int(self.mNBad)

    def integrate(self, x1: float, x2: float, ystart: F64Array, eps: float, h1: float) -> F64Array:
        """
        Integrates the ODE system from x1 to x2, mutating ystart in place and returning the final y.
        """
        n: int = self.mNVariables
        x: float = float(x1)
        y: F64Array = np.copy(ystart)
        dydx: F64Array = np.zeros(n, dtype=np.float64)
        yout: F64Array = np.zeros(n, dtype=np.float64)
        yerr: F64Array = np.zeros(n, dtype=np.float64)
        yscal: F64Array = np.zeros(n, dtype=np.float64)
        h: float = float(h1)
        self.mNOk = 0
        self.mNBad = 0
        self.derivatives(x, y, dydx)
        while x < x2:
            for i in range(n):
                yscal[i] = abs(y[i]) + abs(h * dydx[i]) + 1.0e-30
            h = min(h, x2 - x)
            self._baseStep(x, y, dydx, h, yout, yerr)
            errmax: float = max(abs(yerr[i] / yscal[i]) for i in range(n))
            if errmax > eps:
                h = 0.9 * h * math.pow(eps / errmax, 0.25)
                self.mNBad += 1
                if abs(h) < self.mMinStepSize:
                    raise EPQException("Step size too small.")
                continue
            x += h
            y[:] = yout[:]
            self.mNOk += 1
            self.derivatives(x, y, dydx)
            h = 0.9 * h * math.pow(eps / errmax, 0.25)
        ystart[:] = y[:]
        return y

    # ==================================================================
    # Private helpers (R1: _ prefix for Java private/protected)
    # ==================================================================

    def _sign(self, magnitude: float, sign: float) -> float:
        # Java sign(magnitude, sign): +|magnitude| if sign >= 0.0, -|magnitude| otherwise.
        return abs(magnitude) if sign >= 0.0 else -abs(magnitude)  # [97e59cca-c31e-446f-f498-08decb97ea4d]

    def _baseStep(self, x: float, y: F64Array, dydx: F64Array, h: float, yout: F64Array, yerr: F64Array) -> None:
        """
        Base step for the Runge-Kutta method. Mutates yout and yerr.
        """
        n: int = self.mNVariables
        ws2: F64Array = np.zeros(n, dtype=np.float64)
        ws3: F64Array = np.zeros(n, dtype=np.float64)
        ws4: F64Array = np.zeros(n, dtype=np.float64)
        ws5: F64Array = np.zeros(n, dtype=np.float64)
        ws6: F64Array = np.zeros(n, dtype=np.float64)
        ytemp: F64Array = np.copy(y)
        # Cash-Karp coefficients (standard NR implementation)
        # [0907c3d4-f201-4f0f-f499-08decb97ea4d]
        a2, a3, a4, a5, a6 = 0.2, 0.3, 0.6, 1.0, 0.875
        b21 = 0.2
        b31, b32 = 3.0/40.0, 9.0/40.0
        b41, b42, b43 = 0.3, -0.9, 1.2
        b51, b52, b53, b54 = -11.0/54.0, 2.5, -70.0/27.0, 35.0/27.0
        b61, b62, b63, b64, b65 = 1631.0/55296.0, 175.0/512.0, 575.0/13824.0, 44275.0/110592.0, 253.0/4096.0
        c1, c3, c4, c5, c6 = 37.0/378.0, 250.0/621.0, 125.0/594.0, 0.0, 512.0/1771.0
        dc1, dc3, dc4, dc5, dc6 = c1 - 2825.0/27648.0, c3 - 18575.0/48384.0, c4 - 13525.0/55296.0, c5 - 277.0/14336.0, c6 - 1.0/4.0
        # Step 1
        for i in range(n):
            ytemp[i] = y[i] + b21 * h * dydx[i]
        self.derivatives(x + a2 * h, ytemp, ws2)
        # Step 2
        for i in range(n):
            ytemp[i] = y[i] + h * (b31 * dydx[i] + b32 * ws2[i])
        self.derivatives(x + a3 * h, ytemp, ws3)
        # Step 3
        for i in range(n):
            ytemp[i] = y[i] + h * (b41 * dydx[i] + b42 * ws2[i] + b43 * ws3[i])
        self.derivatives(x + a4 * h, ytemp, ws4)
        # Step 4
        for i in range(n):
            ytemp[i] = y[i] + h * (b51 * dydx[i] + b52 * ws2[i] + b53 * ws3[i] + b54 * ws4[i])
        self.derivatives(x + a5 * h, ytemp, ws5)
        # Step 5
        for i in range(n):
            ytemp[i] = y[i] + h * (b61 * dydx[i] + b62 * ws2[i] + b63 * ws3[i] + b64 * ws4[i] + b65 * ws5[i])
        self.derivatives(x + a6 * h, ytemp, ws6)
        # Combine for output and error
        for i in range(n):
            yout[i] = y[i] + h * (c1 * dydx[i] + c3 * ws3[i] + c4 * ws4[i] + c5 * ws5[i] + c6 * ws6[i])
            yerr[i] = h * (dc1 * dydx[i] + dc3 * ws3[i] + dc4 * ws4[i] + dc5 * ws5[i] + dc6 * ws6[i])

    def _qcStep(self, x: float, y: F64Array, dydx: F64Array, htry: float, eps: float, yscal: F64Array) -> float:
        """
        Quality-controlled step: mutates y in place via y[:] = mQcYTemp after successful step.
        Sets mHDid and mHNext as side effects.
        """
        n: int = self.mNVariables
        ytemp: F64Array = np.copy(y)
        yerr: F64Array = np.zeros(n, dtype=np.float64)
        h: float = float(htry)
        while True:
            self._baseStep(x, y, dydx, h, ytemp, yerr)
            errmax: float = max(abs(yerr[i] / yscal[i]) for i in range(n))
            if errmax > eps:
                h = 0.9 * h * math.pow(eps / errmax, 0.25)
                if abs(h) < self.mMinStepSize:
                    raise EPQException("Step size too small.")
                continue
            self.mHDid = h
            self.mHNext = 0.9 * h * math.pow(eps / errmax, 0.25)
            y[:] = ytemp[:]
            return h

    def _clearWorkspace(self) -> None:
        self.mWs2 = None
        self.mWs3 = None
        self.mWs4 = None
        self.mWs5 = None
        self.mWs6 = None
        self.mYTemp = None
        self.mYErr = None
        self.mQcYTemp = None