r"""
AdaptiveRungeKutta_ver1_1_0.py — Python port of gov.nist.microanalysis.Utility.AdaptiveRungeKutta

Guide version : 1
Generation    : 1
Port-code fixes: 0

CHANGES
-------
* SCIPY-DEV-1: `integrate` delegates to `integrate_literal` because the class API tightly
  couples to its internal step-tracking arrays (mXSave, mYSave) and counters (mNOk, mNBad).
  A direct `scipy.integrate.solve_ivp` substitution would violate these state invariants.

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
import numpy as np
from typing import Optional, Sequence, Union, Callable

try:
    from ._epq_compat import EPQException, F64Array, JavaRandom
except ImportError:
    try:
        from _epq_compat import EPQException, F64Array, JavaRandom  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility._epq_compat import EPQException, F64Array, JavaRandom  # type: ignore


class AdaptiveRungeKutta(abc.ABC):
    BUG_LEDGER: tuple = ()  # no bugs identified; Java source not attached

    @staticmethod
    def _require_mutable_f64(arr: np.ndarray, name: str = "arr") -> None:
        """Type guard for in-place methods (CONVERSION_GUIDE R5)."""
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{name} must be a numpy ndarray")
        if arr.dtype != np.float64:
            raise TypeError(f"{name} must have dtype float64")
        if not arr.flags.writeable:
            raise TypeError(f"{name} must be writeable")

    def __init__(self, nVars: int) -> None:
        super().__init__()
        self._mNVariables: int = int(nVars)
        self._mHDid: float = 0.0
        self._mHNext: float = 0.0
        self._mSaveInterval: float = sys.float_info.max
        self._mMinStepSize: float = 0.0
        self._mXSave: Optional[F64Array] = None
        self._mYSave: Optional[F64Array] = None
        self._mNSaved: int = 0
        self._mMaxSteps: int = 10000
        self._mNOk: int = 0
        self._mNBad: int = 0

        # Temporary work space used by baseStep
        self._mWs2: Optional[F64Array] = None
        self._mWs3: Optional[F64Array] = None
        self._mWs4: Optional[F64Array] = None
        self._mWs5: Optional[F64Array] = None
        self._mWs6: Optional[F64Array] = None
        self._mYTemp: Optional[F64Array] = None

        # Temporary work space used by qcStep
        self._mYErr: Optional[F64Array] = None
        self._mQcYTemp: Optional[F64Array] = None

    def _sign(self, magnitude: float, sign: float) -> float:
        return abs(magnitude) if sign >= 0.0 else -abs(magnitude)

    def _baseStep(self, x: float, y: F64Array, dydx: F64Array, h: float, yout: F64Array, yerr: F64Array) -> None:
        self._require_mutable_f64(yout, "yout")
        self._require_mutable_f64(yerr, "yerr")

        a2: float = 0.2
        a3: float = 0.3
        a4: float = 0.6
        a5: float = 1.0
        a6: float = 0.875
        
        b21: float = 0.2
        
        b31: float = 3.0 / 40.0
        b32: float = 9.0 / 40.0
        
        b41: float = 0.3
        b42: float = -0.9
        b43: float = 1.2
        
        b51: float = -11.0 / 54.0
        b52: float = 2.5
        b53: float = -70.0 / 27.0
        b54: float = 35.0 / 27.0
        
        b61: float = 1631.0 / 55296.0
        b62: float = 175.0 / 512.0
        b63: float = 575.0 / 13824.0
        b64: float = 44275.0 / 110592.0
        b65: float = 253.0 / 4096.0
        
        c1: float = 37.0 / 378.0
        c3: float = 250.0 / 621.0
        c4: float = 125.0 / 594.0
        c6: float = 512.0 / 1771.0
        
        dc1: float = c1 - (2825.0 / 27648.0)
        dc3: float = c3 - (18575.0 / 48384.0)
        dc4: float = c4 - (13525.0 / 55296.0)
        dc5: float = -277.0 / 14336.0
        dc6: float = c6 - 0.25

        # Workspace
        if self._mWs2 is None:
            self._mWs2 = np.zeros(self._mNVariables, dtype=np.float64)
            self._mWs3 = np.zeros(self._mNVariables, dtype=np.float64)
            self._mWs4 = np.zeros(self._mNVariables, dtype=np.float64)
            self._mWs5 = np.zeros(self._mNVariables, dtype=np.float64)
            self._mWs6 = np.zeros(self._mNVariables, dtype=np.float64)
            self._mYTemp = np.zeros(self._mNVariables, dtype=np.float64)
            
        assert self._mWs2 is not None
        assert self._mWs3 is not None
        assert self._mWs4 is not None
        assert self._mWs5 is not None
        assert self._mWs6 is not None
        assert self._mYTemp is not None

        # First step
        for i in range(self._mNVariables):
            self._mYTemp[i] = y[i] + (b21 * h * dydx[i])
            
        # Second step
        self.derivatives(x + (a2 * h), self._mYTemp, self._mWs2)
        for i in range(self._mNVariables):
            self._mYTemp[i] = y[i] + (h * ((b31 * dydx[i]) + (b32 * self._mWs2[i])))
            
        # Third step
        self.derivatives(x + (a3 * h), self._mYTemp, self._mWs3)
        for i in range(self._mNVariables):
            self._mYTemp[i] = y[i] + (h * ((b41 * dydx[i]) + (b42 * self._mWs2[i]) + (b43 * self._mWs3[i])))
            
        # Fourth step
        self.derivatives(x + (a4 * h), self._mYTemp, self._mWs4)
        for i in range(self._mNVariables):
            self._mYTemp[i] = y[i] + (h * ((b51 * dydx[i]) + (b52 * self._mWs2[i]) + (b53 * self._mWs3[i]) + (b54 * self._mWs4[i])))
            
        # Fifth step
        self.derivatives(x + (a5 * h), self._mYTemp, self._mWs5)
        for i in range(self._mNVariables):
            self._mYTemp[i] = y[i] + (h * ((b61 * dydx[i]) + (b62 * self._mWs2[i]) + (b63 * self._mWs3[i]) + (b64 * self._mWs4[i]) + (b65 * self._mWs5[i])))
            
        # Sixth step
        self.derivatives(x + (a6 * h), self._mYTemp, self._mWs6)
        for i in range(self._mNVariables):
            yout[i] = y[i] + (h * ((c1 * dydx[i]) + (c3 * self._mWs3[i]) + (c4 * self._mWs4[i]) + (c6 * self._mWs6[i])))
            
        # Estimate the error
        for i in range(self._mNVariables):
            yerr[i] = h * ((dc1 * dydx[i]) + (dc3 * self._mWs3[i]) + (dc4 * self._mWs4[i]) + (dc5 * self._mWs5[i]) + (dc6 * self._mWs6[i]))

    def _qcStep(self, x: float, y: F64Array, dydx: F64Array, htry: float, eps: float, yscal: F64Array) -> float:
        self._require_mutable_f64(y, "y")
        
        safety: float = 0.9
        pgrow: float = -0.2
        pshrnk: float = -0.25
        errcon: float = 1.89e-4
        
        if self._mYErr is None:
            self._mYErr = np.zeros(self._mNVariables, dtype=np.float64)
            self._mQcYTemp = np.zeros(self._mNVariables, dtype=np.float64)
            
        assert self._mYErr is not None
        assert self._mQcYTemp is not None

        errmax: float = 0.0
        h: float = htry
        
        while True:
            self._baseStep(x, y, dydx, h, self._mQcYTemp, self._mYErr)
            errmax = 0.0
            for i in range(self._mNVariables):
                errmax = max(errmax, abs(self._mYErr[i] / yscal[i]))
            errmax /= eps
            
            if errmax > 1.0:
                htemp = safety * h * math.pow(errmax, pshrnk)
                h = max(htemp, 0.1 * h) if h >= 0 else min(htemp, 0.1 * h)
                # Check for step size underflow
                xnew = x + h
                if xnew == x:
                    raise EPQException("Step size underflow in AdaptiveRungeKutta.qcStep.")
            else:
                break
                
        self._mHNext = safety * h * math.pow(errmax, pgrow) if errmax > errcon else 5.0 * h
        self._mHDid = h
        x += h
        y[:self._mNVariables] = self._mQcYTemp[:self._mNVariables]
        return x

    def _clearWorkspace(self) -> None:
        self._mWs2 = None
        self._mWs3 = None
        self._mWs4 = None
        self._mWs5 = None
        self._mWs6 = None
        self._mYTemp = None
        self._mYErr = None
        self._mQcYTemp = None

    def setSaveInterval(self, interval: float) -> None:
        self._mSaveInterval = abs(interval)

    def clearSaveInterval(self) -> None:
        self._mSaveInterval = sys.float_info.max

    def getNSaved(self) -> int:
        return self._mNSaved

    def getX(self, i: int) -> float:
        assert self._mXSave is not None
        return float(self._mXSave[i])

    def getY(self, i: int) -> F64Array:
        assert self._mYSave is not None
        return self._mYSave[i]

    def setMaxSteps(self, maxSteps: int) -> None:
        self._mMaxSteps = int(maxSteps)

    def setMinStepSize(self, minStep: float) -> None:
        self._mMinStepSize = abs(minStep)

    def getNVariables(self) -> int:
        return self._mNVariables

    def getStepCount(self) -> int:
        return self._mNOk + self._mNBad

    def getGoodStepCount(self) -> int:
        return self._mNOk

    def getBadStepCount(self) -> int:
        return self._mNBad

    def integrate_literal(self, x1: float, x2: float, ystart: F64Array, eps: float, h1: float) -> F64Array:
        self._require_mutable_f64(ystart, "ystart")
        
        tiny: float = 1.0e-10 * eps
        yscal: F64Array = np.zeros(self._mNVariables, dtype=np.float64)
        dydx: F64Array = np.zeros(self._mNVariables, dtype=np.float64)
        y: F64Array = np.zeros(self._mNVariables, dtype=np.float64)
        
        x: float = x1
        h: float = self._sign(h1, x2 - x1)
        xsav: float = 0.0
        saveInt: float = sys.float_info.max
        kMax: int = 0
        
        self._mNSaved = 0
        self._mNOk = 0
        self._mNBad = 0
        
        y[:self._mNVariables] = ystart[:self._mNVariables]
        
        if self._mSaveInterval != sys.float_info.max:
            kMax = int(math.floor((abs(x2 - x1) + self._mSaveInterval) / self._mSaveInterval + 0.5))
            saveInt = self._sign(self._mSaveInterval, x2 - x1)
            xsav = x - (2.0 * saveInt)  # to ensure that the first step is saved...
            self._mXSave = np.zeros(kMax, dtype=np.float64)
            self._mYSave = np.zeros((kMax, self._mNVariables), dtype=np.float64)
            
        for step in range(self._mMaxSteps):
            # Save the necessary points
            if (kMax != 0) and (self._mNSaved < kMax) and (abs(x - xsav) >= (0.9999 * self._mSaveInterval)):
                assert self._mXSave is not None
                assert self._mYSave is not None
                self._mXSave[self._mNSaved] = x
                self._mYSave[self._mNSaved, :] = y[:]
                xsav = x
                self._mNSaved += 1
                
            self.derivatives(x, y, dydx)
            
            # Rescale h to ensure we hit desired points
            hMax: float = abs((xsav + saveInt) - x)
            if abs(h) > hMax:
                h = self._sign(hMax, h)
                
            # Scaling to monitor accuracy...
            for i in range(self._mNVariables):
                yscal[i] = abs(y[i]) + abs(dydx[i] * h) + tiny
                
            if (((x + h) - x2) * ((x + h) - x1)) > 0.0:
                h = x2 - x
                
            x = self._qcStep(x, y, dydx, h, eps, yscal)
            
            if self._mHDid == h:
                self._mNOk += 1
            else:
                self._mNBad += 1
                
            if ((x - x2) * (x2 - x1)) >= 0.0:
                ystart[:self._mNVariables] = y[:self._mNVariables]
                if kMax != 0:
                    assert self._mXSave is not None
                    assert self._mYSave is not None
                    self._mNSaved = min(self._mNSaved, kMax - 1)
                    self._mXSave[self._mNSaved] = x
                    self._mYSave[self._mNSaved, :] = y[:]
                    self._mNSaved += 1
                self._clearWorkspace()
                return y
                
            if abs(self._mHNext) <= self._mMinStepSize:
                raise EPQException("Step size too small in AdaptiveRungeKutta.integrate")
            h = self._mHNext
            
        raise EPQException("Too many steps in AdaptiveRungeKutta.integrate")

    def integrate(self, x1: float, x2: float, ystart: F64Array, eps: float, h1: float) -> F64Array:
        """
        SCIPY-DEV-1: Class API heavily relies on internal step-tracking state;
        no direct scipy.integrate substitution preserves the getX/getY/getGoodStepCount API.
        Delegating to literal implementation.
        """
        self._require_mutable_f64(ystart, "ystart")
        return self.integrate_literal(x1, x2, ystart, eps, h1)

    @abc.abstractmethod
    def derivatives(self, x: float, y: F64Array, dydx: F64Array) -> None:
        pass
