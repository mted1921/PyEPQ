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
 * Adaptive Runge-Kutta ODE integrator (abstract base class).
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
from typing import Optional, Sequence, Union, Callable

try:
    from ._epq_compat import EPQException, F64Array, JavaRandom
except ImportError:
    try:
        from _epq_compat import EPQException, F64Array, JavaRandom  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility._epq_compat import EPQException, F64Array, JavaRandom  # type: ignore

# No sibling port modules required for AdaptiveRungeKutta

__all__ = ["AdaptiveRungeKutta", "EPQException", "F64Array"]

class AdaptiveRungeKutta(abc.ABC):
    """
    Abstract base class for adaptive Runge-Kutta ODE integrators.
    Not thread-safe.
    """

    BUG_LEDGER: tuple = ()  # no bugs identified; Java source not attached

    # ==================================================================
    # Fields (see CONVERSION_CONTEXT.md for mapping)
    # ==================================================================
    def __init__(self, nVars: int) -> None:
        self.mNVariables: int = nVars
        self.mHDid: float = 0.0
        self.mHNext: float = 0.0
        self.mSaveInterval: float = 0.0
        self.mMinStepSize: float = 0.0
        self.mXSave: Optional[F64Array] = None
        self.mYSave: Optional[F64Array] = None
        self.mNSaved: int = 0
        self.mMaxSteps: int = 1000
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

    # ==================================================================
    # Abstract methods
    # ==================================================================
    @abc.abstractmethod
    def derivatives(self, x: float, y: F64Array, dydx: F64Array) -> None:
        """
        Computes the derivatives at position x for state y.
        Must be implemented by subclasses.
        """
        pass

    # ==================================================================
    # Public methods
    # ==================================================================
    def setSaveInterval(self, interval: float) -> None:
        self.mSaveInterval = float(interval)
        self._clearWorkspace()

    def clearSaveInterval(self) -> None:
        self.mSaveInterval = 0.0
        self._clearWorkspace()

    def getNSaved(self) -> int:
        return self.mNSaved

    def getX(self, i: int) -> float:
        if self.mXSave is None or i < 0 or i >= self.mNSaved:
            raise EPQException("Index out of bounds in getX")
        return float(self.mXSave[i])

    def getY(self, i: int) -> F64Array:
        if self.mYSave is None or i < 0 or i >= self.mNSaved:
            raise EPQException("Index out of bounds in getY")
        # Return a view of row i (shape: (mNVariables,))
        return self.mYSave[i, :]

    def setMaxSteps(self, maxSteps: int) -> None:
        self.mMaxSteps = int(maxSteps)

    def setMinStepSize(self, minStep: float) -> None:
        self.mMinStepSize = float(minStep)

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
        h1: float,
    ) -> F64Array:
        """
        Integrates the ODE from x1 to x2, starting at ystart, with
        accuracy eps and initial stepsize h1.
        Mutates ystart in place and also returns the final y.
        """
        n: int = self.mNVariables
        y = np.array(ystart, dtype=np.float64, copy=True)
        x: float = float(x1)
        h: float = float(h1)
        yscal = np.zeros(n, dtype=np.float64)
        dydx = np.zeros(n, dtype=np.float64)
        yout = np.zeros(n, dtype=np.float64)
        yerr = np.zeros(n, dtype=np.float64)

        self.mNOk = 0
        self.mNBad = 0
        self.mHDid = 0.0
        self.mHNext = 0.0

        if self.mSaveInterval > 0.0:
            kMax = int(math.floor(abs(x2 - x1) / self.mSaveInterval + 0.5)) + 2
            self.mXSave = np.zeros(kMax, dtype=np.float64)
            self.mYSave = np.zeros((kMax, n), dtype=np.float64)
            self.mNSaved = 0
            self.mXSave[0] = x
            self.mYSave[0, :] = y
            self.mNSaved = 1

        for step in range(self.mMaxSteps):
            self.derivatives(x, y, dydx)
            for i in range(n):
                yscal[i] = abs(y[i]) + abs(h * dydx[i]) + 1.0e-30
            if (x + h - x2) * (x + h - x1) > 0.0:
                h = x2 - x
            hDid, hNext = self._qcStep(x, y, dydx, h, eps, yscal)
            self.mHDid = hDid
            self.mHNext = hNext
            if self.mSaveInterval > 0.0 and self.mNSaved < self.mXSave.shape[0]:
                self.mXSave[self.mNSaved] = x + hDid
                self.mYSave[self.mNSaved, :] = y
                self.mNSaved += 1
            if (x + hDid - x2) * (x2 - x1) >= 0.0:
                ystart[:] = y
                return y
            if abs(hNext) <= self.mMinStepSize:
                raise EPQException("Step size too small in integrate")
            x += hDid
            h = hNext
        raise EPQException("Maximum number of steps exceeded in integrate")

    # ==================================================================
    # Private helpers
    # ==================================================================
    def _sign(self, magnitude: float, sign: float) -> float:
        # Java's sign semantics: sign(m, -0.0) returns +|m| (Java: -0.0 >= 0.0 is True)
        return abs(magnitude) if sign >= 0.0 else -abs(magnitude)

    def _baseStep(
        self,
        x: float,
        y: F64Array,
        dydx: F64Array,
        h: float,
        yout: F64Array,
        yerr: F64Array,
    ) -> None:
        """
        Abstract: Subclass must implement the actual Runge-Kutta step.
        Computes yout and yerr for a single step.
        """
        raise NotImplementedError("Subclasses must implement _baseStep")

    def _qcStep(
        self,
        x: float,
        y: F64Array,
        dydx: F64Array,
        htry: float,
        eps: float,
        yscal: F64Array,
    ) -> tuple[float, float]:
        """
        Quality-controlled Runge-Kutta step.
        Returns (hDid, hNext).
        Mutates y in place if step is accepted.
        """
        n: int = self.mNVariables
        ytemp = np.zeros(n, dtype=np.float64)
        yerr = np.zeros(n, dtype=np.float64)
        h = htry
        errmax = 0.0
        while True:
            self._baseStep(x, y, dydx, h, ytemp, yerr)
            errmax = 0.0
            for i in range(n):
                err = abs(yerr[i] / yscal[i])
                if err > errmax:
                    errmax = err
            errmax /= eps
            if errmax <= 1.0:
                y[:] = ytemp
                self.mNOk += 1
                break
            htemp = 0.9 * h * pow(errmax, -0.2)
            h = max(htemp, 0.1 * h)
            self.mNBad += 1
            if abs(h) <= self.mMinStepSize:
                raise EPQException("Step size too small in _qcStep")
        hnext = 0.9 * h * pow(errmax, -0.2) if errmax > 1.89e-4 else 5.0 * h
        return h, hnext

    def _clearWorkspace(self) -> None:
        self.mXSave = None
        self.mYSave = None
        self.mNSaved = 0