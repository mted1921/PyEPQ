r"""
AdaptiveRungeKutta_ver1_1_0.py — Python port of gov.nist.microanalysis.Utility.AdaptiveRungeKutta

Guide version : 1
Generation    : 1
Port-code fixes: 0

------------------------------------------------------------------------
Original Java source (gov.nist.microanalysis.Utility.AdaptiveRungeKutta)
------------------------------------------------------------------------
/**
 * <p>
 * An adaptive step size Runge-Kutta algorithm for numerically evaluating
 * differential equations. This implementation can optionally save intermediate
 * points along the ODE trajectory at a user specified interval. Using this
 * option may limit the step size and thus
 * </p>
 * <p>
 * See Press, Teulolsky, Vetterling &amp; Flannery, Numerical Recipes in C,
 * Second Edition pp 714-722
 * </p>
 * <p>
 * Example:<br>
 * </p>
 * 
 * <pre>
 * AdaptiveRungeKutta trial = new AdaptiveRungeKutta(2) {
 *    void derivatives(double x, double[] y, double[] dydx) {
 *       dydx[0] = -Math.sin(x);
 *       dydx[1] = Math.cos(x);
 *    }
 * };
 * </pre>
 * 
 * <pre>
 * try {
 *    double[] yst = {1.0, 0.0};
 *    trial.setSaveInterval(Math.PI / 16.0);
 *    trial.integrate(0.0, 2.0 * Math.PI, yst, 1.0e-6, 0.01);
 * } catch (UtilException ex) {
 *    System.err.println(ex.toString());
 * }
 * for (int i = 0; i &lt; trial.getNSaved(); ++i)
 *    System.out.println(trial.getX(i) + &quot;\t&quot; + trial.getY(i)[0] + &quot;\t&quot; + trial.getY(i)[1]);
 * </pre>
 * <p>
 * NOTE: This algorithm is not thread-safe. Use each instance in one and only
 * one thread.
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

import abc
import math
import sys
from typing import Any, Optional, Sequence, Union, Callable

import numpy as np

try:
    from ._epq_compat import EPQException, F64Array, JavaRandom
except ImportError:
    try:
        from _epq_compat import EPQException, F64Array, JavaRandom  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility._epq_compat import EPQException, F64Array, JavaRandom  # type: ignore

try:
    from .UtilException_ver1_1_0 import UtilException
except ImportError:
    try:
        from UtilException_ver1_1_0 import UtilException  # type: ignore
    except ImportError:
        try:
            from gov.nist.microanalysis.PyEPQ.Utility.UtilException_ver1_1_0 import UtilException  # type: ignore
        except ImportError:
            UtilException = EPQException  # type: ignore


class AdaptiveRungeKutta(abc.ABC):

    BUG_LEDGER: tuple = ()  # no bugs identified

    @staticmethod
    def _require_mutable_f64(arr: Any, name: str = "arr") -> None:
        """Type guard for in-place methods."""
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{name} must be a numpy ndarray")
        if arr.dtype != np.float64:
            raise TypeError(f"{name} must have dtype float64")
        if not arr.flags.writeable:
            raise TypeError(f"{name} must be writeable")

    def __init__(self, nVars: int) -> None:
        self.mNVariables: int = nVars
        self.mHDid: float = 0.0
        self.mHNext: float = 0.0
        self.mSaveInterval: float = sys.float_info.max
        self.mMinStepSize: float = 0.0
        self.mXSave: Optional[F64Array] = None
        self.mYSave: Optional[F64Array] = None
        self.mNSaved: int = 0
        self.mMaxSteps: int = 10000
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

    def _sign(self, magnitude: float, sign_val: float) -> float:
        return math.fabs(magnitude) if sign_val >= 0.0 else -math.fabs(magnitude)

    def _baseStep(self, x: float, y: F64Array, dydx: F64Array, h: float, yout: F64Array, yerr: F64Array) -> None:
        a2, a3, a4, a5, a6 = 0.2, 0.3, 0.6, 1.0, 0.875
        b21 = 0.2
        b31, b32 = 3.0 / 40.0, 9.0 / 40.0
        b41, b42, b43 = 0.3, -0.9, 1.2
        b51, b52, b53, b54 = -11.0 / 54.0, 2.5, -70.0 / 27.0, 35.0 / 27.0
        b61, b62, b63, b64, b65 = 1631.0 / 55296.0, 175.0 / 512.0, 575.0 / 13824.0, 44275.0 / 110592.0, 253.0 / 4096.0
        c1, c3, c4, c6 = 37.0 / 378.0, 250.0 / 621.0, 125.0 / 594.0, 512.0 / 1771.0
        dc1 = c1 - (2825.0 / 27648.0)
        dc3 = c3 - (18575.0 / 48384.0)
        dc4 = c4 - (13525.0 / 55296.0)
        dc5 = -277.0 / 14336.0
        dc6 = c6 - 0.25

        if self.mWs2 is None:
            self.mWs2 = np.zeros(self.mNVariables, dtype=np.float64)
            self.mWs3 = np.zeros(self.mNVariables, dtype=np.float64)
            self.mWs4 = np.zeros(self.mNVariables, dtype=np.float64)
            self.mWs5 = np.zeros(self.mNVariables, dtype=np.float64)
            self.mWs6 = np.zeros(self.mNVariables, dtype=np.float64)
            self.mYTemp = np.zeros(self.mNVariables, dtype=np.float64)

        assert self.mWs2 is not None
        assert self.mWs3 is not None
        assert self.mWs4 is not None
        assert self.mWs5 is not None
        assert self.mWs6 is not None
        assert self.mYTemp is not None

        for i in range(self.mNVariables):
            self.mYTemp[i] = y[i] + (b21 * h * dydx[i])
            
        self.derivatives(x + (a2 * h), self.mYTemp, self.mWs2)
        for i in range(self.mNVariables):
            self.mYTemp[i] = y[i] + (h * ((b31 * dydx[i]) + (b32 * self.mWs2[i])))
            
        self.derivatives(x + (a3 * h), self.mYTemp, self.mWs3)
        for i in range(self.mNVariables):
            self.mYTemp[i] = y[i] + (h * ((b41 * dydx[i]) + (b42 * self.mWs2[i]) + (b43 * self.mWs3[i])))
            
        self.derivatives(x + (a4 * h), self.mYTemp, self.mWs4)
        for i in range(self.mNVariables):
            self.mYTemp[i] = y[i] + (h * ((b51 * dydx[i]) + (b52 * self.mWs2[i]) + (b53 * self.mWs3[i]) + (b54 * self.mWs4[i])))
            
        self.derivatives(x + (a5 * h), self.mYTemp, self.mWs5)
        for i in range(self.mNVariables):
            self.mYTemp[i] = y[i] + (h * ((b61 * dydx[i]) + (b62 * self.mWs2[i]) + (b63 * self.mWs3[i]) + (b64 * self.mWs4[i]) + (b65 * self.mWs5[i])))
            
        self.derivatives(x + (a6 * h), self.mYTemp, self.mWs6)
        for i in range(self.mNVariables):
            yout[i] = y[i] + (h * ((c1 * dydx[i]) + (c3 * self.mWs3[i]) + (c4 * self.mWs4[i]) + (c6 * self.mWs6[i])))
            
        for i in range(self.mNVariables):
            yerr[i] = h * ((dc1 * dydx[i]) + (dc3 * self.mWs3[i]) + (dc4 * self.mWs4[i]) + (dc5 * self.mWs5[i]) + (dc6 * self.mWs6[i]))

    def _qcStep(self, x: float, y: F64Array, dydx: F64Array, htry: float, eps: float, yscal: F64Array) -> float:
        safety = 0.9
        pgrow = -0.2
        pshrnk = -0.25
        errcon = 1.89e-4

        if self.mYErr is None:
            self.mYErr = np.zeros(self.mNVariables, dtype=np.float64)
            self.mQcYTemp = np.zeros(self.mNVariables, dtype=np.float64)

        assert self.mYErr is not None
        assert self.mQcYTemp is not None

        h = htry
        errmax = 0.0
        while True:
            self._baseStep(x, y, dydx, h, self.mQcYTemp, self.mYErr)
            errmax = 0.0
            for i in range(self.mNVariables):
                errmax = max(errmax, abs(self.mYErr[i] / yscal[i]))
            errmax /= eps
            if errmax > 1.0:
                htemp = safety * h * math.pow(errmax, pshrnk)
                h = max(htemp, 0.1 * h) if h >= 0 else min(htemp, 0.1 * h)
                xnew = x + h
                if xnew == x:
                    raise UtilException("Step size underflow in AdaptiveRungeKutta.qcStep.")
            else:
                break
                
        self.mHNext = safety * h * math.pow(errmax, pgrow) if errmax > errcon else 5.0 * h
        self.mHDid = h
        x += h
        for i in range(self.mNVariables):
            y[i] = self.mQcYTemp[i]
        return x

    def _clearWorkspace(self) -> None:
        self.mWs2 = None
        self.mWs3 = None
        self.mWs4 = None
        self.mWs5 = None
        self.mWs6 = None
        self.mYTemp = None
        self.mYErr = None
        self.mQcYTemp = None

    def setSaveInterval(self, interval: float) -> None:
        self.mSaveInterval = abs(interval)

    def clearSaveInterval(self) -> None:
        self.mSaveInterval = sys.float_info.max

    def getNSaved(self) -> int:
        return self.mNSaved

    def getX(self, i: int) -> float:
        if self.mXSave is None:
            raise IndexError("No saved values available.")
        return float(self.mXSave[i])

    def getY(self, i: int) -> F64Array:
        if self.mYSave is None:
            raise IndexError("No saved values available.")
        return self.mYSave[i, :]

    def setMaxSteps(self, maxSteps: int) -> None:
        self.mMaxSteps = maxSteps

    def setMinStepSize(self, minStep: float) -> None:
        self.mMinStepSize = abs(minStep)

    def getNVariables(self) -> int:
        return self.mNVariables

    def getStepCount(self) -> int:
        return self.mNOk + self.mNBad

    def getGoodStepCount(self) -> int:
        return self.mNOk

    def getBadStepCount(self) -> int:
        return self.mNBad

    def integrate_literal(self, x1: float, x2: float, ystart: F64Array, eps: float, h1: float) -> F64Array:
        AdaptiveRungeKutta._require_mutable_f64(ystart, "ystart")
        tiny = 1.0e-10 * eps
        yscal = np.zeros(self.mNVariables, dtype=np.float64)
        dydx = np.zeros(self.mNVariables, dtype=np.float64)
        y = np.zeros(self.mNVariables, dtype=np.float64)
        x = x1
        h = self._sign(h1, x2 - x1)
        xsav = 0.0
        saveInt = sys.float_info.max
        kMax = 0
        self.mNSaved = 0
        self.mNOk = 0
        self.mNBad = 0

        for i in range(self.mNVariables):
            y[i] = ystart[i]

        if self.mSaveInterval != sys.float_info.max:
            kMax = int(math.floor((abs(x2 - x1) + self.mSaveInterval) / self.mSaveInterval + 0.5))
            saveInt = self._sign(self.mSaveInterval, x2 - x1)
            xsav = x - (2.0 * saveInt)
            self.mXSave = np.zeros(kMax, dtype=np.float64)
            self.mYSave = np.zeros((kMax, self.mNVariables), dtype=np.float64)

        for step in range(self.mMaxSteps):
            if (kMax != 0) and (self.mNSaved < kMax) and (abs(x - xsav) >= (0.9999 * self.mSaveInterval)):
                assert self.mXSave is not None
                assert self.mYSave is not None
                self.mXSave[self.mNSaved] = x
                for i in range(self.mNVariables):
                    self.mYSave[self.mNSaved, i] = y[i]
                xsav = x
                self.mNSaved += 1

            self.derivatives(x, y, dydx)

            hMax = abs((xsav + saveInt) - x)
            if abs(h) > hMax:
                h = self._sign(hMax, h)

            for i in range(self.mNVariables):
                yscal[i] = abs(y[i]) + abs(dydx[i] * h) + tiny

            if (((x + h) - x2) * ((x + h) - x1)) > 0.0:
                h = x2 - x

            x = self._qcStep(x, y, dydx, h, eps, yscal)

            if self.mHDid == h:
                self.mNOk += 1
            else:
                self.mNBad += 1

            if ((x - x2) * (x2 - x1)) >= 0.0:
                for i in range(self.mNVariables):
                    ystart[i] = y[i]
                if kMax != 0:
                    self.mNSaved = min(self.mNSaved, kMax - 1)
                    assert self.mXSave is not None
                    assert self.mYSave is not None
                    self.mXSave[self.mNSaved] = x
                    for i in range(self.mNVariables):
                        self.mYSave[self.mNSaved, i] = y[i]
                    self.mNSaved += 1
                self._clearWorkspace()
                return y

            if abs(self.mHNext) <= self.mMinStepSize:
                raise UtilException("Step size too small in AdaptiveRungeKutta.integrate")
            h = self.mHNext

        raise UtilException("Too many steps in AdaptiveRungeKutta.integrate")

    def integrate(self, x1: float, x2: float, ystart: F64Array, eps: float, h1: float) -> F64Array:
        return self.integrate_literal(x1, x2, ystart, eps, h1)

    @abc.abstractmethod
    def derivatives(self, x: float, y: F64Array, dydx: F64Array) -> None:
        pass
