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
 * * <pre>
 * AdaptiveRungeKutta trial = new AdaptiveRungeKutta(2) {
 * void derivatives(double x, double[] y, double[] dydx) {
 * dydx[0] = -Math.sin(x);
 * dydx[1] = Math.cos(x);
 * }
 * };
 * </pre>
 * * <pre>
 * try {
 * double[] yst = {1.0, 0.0};
 * trial.setSaveInterval(Math.PI / 16.0);
 * trial.integrate(0.0, 2.0 * Math.PI, yst, 1.0e-6, 0.01);
 * } catch (UtilException ex) {
 * System.err.println(ex.toString());
 * }
 * for (int i = 0; i &lt; trial.getNSaved(); ++i)
 * System.out.println(trial.getX(i) + &quot;\t&quot; + trial.getY(i)[0] + &quot;\t&quot; + trial.getY(i)[1]);
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
 * * @author Nicholas W. M. Ritchie
 * @version 1.0
 */
------------------------------------------------------------------------

CHANGES:
- SCIPY-DEV-1: `integrate` uses scipy.integrate.solve_ivp (RK45) as the primary.
  Callers needing strict bit-for-bit Cash-Karp trajectories or relying on the
  saved-point arrays (`mXSave`, `mYSave`) must call `integrate_literal` directly,
  as the Scipy primary bypasses the manual step-management and saved arrays.
"""

from __future__ import annotations

import abc
import math
import sys
from typing import Any, Optional

import numpy as np
from scipy import integrate as _sp_integrate

try:
    from ._epq_compat import EPQException, F64Array
except ImportError:
    try:
        from _epq_compat import EPQException, F64Array  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility._epq_compat import EPQException, F64Array  # type: ignore


__all__ = ["AdaptiveRungeKutta"]


class AdaptiveRungeKutta(abc.ABC):

    BUG_LEDGER: tuple = ()  # no bugs identified; literal algorithm perfectly follows Java

    def __init__(self, nVars: int) -> None:
        super().__init__()
        self.mNVariables: int = nVars
        self.mHDid: float = 0.0
        self.mHNext: float = 0.0
        self.mSaveInterval: float = sys.float_info.max
        self.mMinStepSize: float = 0.0
        self.mXSave: Optional[F64Array] = None
        self.mYSave: Optional[F64Array] = None  # 2D array in Python replaces double[][]
        self.mNSaved: int = 0
        self.mMaxSteps: int = 10000
        self.mNOk: int = 0
        self.mNBad: int = 0
        
        # Temporary work space used by _baseStep
        self.mWs2: Optional[F64Array] = None
        self.mWs3: Optional[F64Array] = None
        self.mWs4: Optional[F64Array] = None
        self.mWs5: Optional[F64Array] = None
        self.mWs6: Optional[F64Array] = None
        self.mYTemp: Optional[F64Array] = None
        
        # Temporary work space used by _qcStep
        self.mYErr: Optional[F64Array] = None
        self.mQcYTemp: Optional[F64Array] = None

    @staticmethod
    def _require_mutable_f64(arr: Any, name: str = "arr") -> None:
        """Type guard for in-place methods (CONVERSION_GUIDE R5)."""
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{name} must be a numpy ndarray")
        if arr.dtype != np.float64:
            raise TypeError(f"{name} must have dtype float64")
        if not arr.flags.writeable:
            raise TypeError(f"{name} must be writeable")

    def _sign(self, magnitude: float, sign: float) -> float:
        return abs(magnitude) if sign >= 0.0 else -abs(magnitude)

    def _baseStep(self, x: float, y: F64Array, dydx: F64Array, h: float, yout: F64Array, yerr: F64Array) -> None:
        self._require_mutable_f64(yout, "yout")
        self._require_mutable_f64(yerr, "yerr")
        
        a2: float = 0.2; a3: float = 0.3; a4: float = 0.6; a5: float = 1.0; a6: float = 0.875
        b21: float = 0.2
        b31: float = 3.0 / 40.0; b32: float = 9.0 / 40.0
        b41: float = 0.3; b42: float = -0.9; b43: float = 1.2
        b51: float = -11.0 / 54.0; b52: float = 2.5; b53: float = -70.0 / 27.0; b54: float = 35.0 / 27.0
        b61: float = 1631.0 / 55296.0; b62: float = 175.0 / 512.0; b63: float = 575.0 / 13824.0; b64: float = 44275.0 / 110592.0; b65: float = 253.0 / 4096.0
        c1: float = 37.0 / 378.0; c3: float = 250.0 / 621.0; c4: float = 125.0 / 594.0; c6: float = 512.0 / 1771.0
        dc1: float = c1 - (2825.0 / 27648.0); dc3: float = c3 - (18575.0 / 48384.0); dc4: float = c4 - (13525.0 / 55296.0); dc5: float = -277.0 / 14336.0
        dc6: float = c6 - 0.25
        
        # Workspace
        if self.mWs2 is None:
            self.mWs2 = np.zeros(self.mNVariables, dtype=np.float64)
            self.mWs3 = np.zeros(self.mNVariables, dtype=np.float64)
            self.mWs4 = np.zeros(self.mNVariables, dtype=np.float64)
            self.mWs5 = np.zeros(self.mNVariables, dtype=np.float64)
            self.mWs6 = np.zeros(self.mNVariables, dtype=np.float64)
            self.mYTemp = np.zeros(self.mNVariables, dtype=np.float64)
            
        assert self.mYTemp is not None
        assert self.mWs2 is not None
        assert self.mWs3 is not None
        assert self.mWs4 is not None
        assert self.mWs5 is not None
        assert self.mWs6 is not None
            
        # First step
        for i in range(self.mNVariables):
            self.mYTemp[i] = y[i] + (b21 * h * dydx[i])
            
        # Second step
        self.derivatives(x + (a2 * h), self.mYTemp, self.mWs2)
        for i in range(self.mNVariables):
            self.mYTemp[i] = y[i] + (h * ((b31 * dydx[i]) + (b32 * self.mWs2[i])))
            
        # Third step
        self.derivatives(x + (a3 * h), self.mYTemp, self.mWs3)
        for i in range(self.mNVariables):
            self.mYTemp[i] = y[i] + (h * ((b41 * dydx[i]) + (b42 * self.mWs2[i]) + (b43 * self.mWs3[i])))
            
        # Fourth step
        self.derivatives(x + (a4 * h), self.mYTemp, self.mWs4)
        for i in range(self.mNVariables):
            self.mYTemp[i] = y[i] + (h * ((b51 * dydx[i]) + (b52 * self.mWs2[i]) + (b53 * self.mWs3[i]) + (b54 * self.mWs4[i])))
            
        # Fifth step
        self.derivatives(x + (a5 * h), self.mYTemp, self.mWs5)
        for i in range(self.mNVariables):
            self.mYTemp[i] = y[i] + (h * ((b61 * dydx[i]) + (b62 * self.mWs2[i]) + (b63 * self.mWs3[i]) + (b64 * self.mWs4[i]) + (b65 * self.mWs5[i])))
            
        # Sixth step
        self.derivatives(x + (a6 * h), self.mYTemp, self.mWs6)
        for i in range(self.mNVariables):
            yout[i] = y[i] + (h * ((c1 * dydx[i]) + (c3 * self.mWs3[i]) + (c4 * self.mWs4[i]) + (c6 * self.mWs6[i])))
            
        # Estimate the error
        for i in range(self.mNVariables):
            yerr[i] = h * ((dc1 * dydx[i]) + (dc3 * self.mWs3[i]) + (dc4 * self.mWs4[i]) + (dc5 * self.mWs5[i]) + (dc6 * self.mWs6[i]))

    def _qcStep(self, x: float, y: F64Array, dydx: F64Array, htry: float, eps: float, yscal: F64Array) -> float:
        self._require_mutable_f64(y, "y")
        
        safety: float = 0.9
        pgrow: float = -0.2
        pshrnk: float = -0.25
        errcon: float = 1.89e-4
        
        if self.mYErr is None:
            self.mYErr = np.zeros(self.mNVariables, dtype=np.float64)
            self.mQcYTemp = np.zeros(self.mNVariables, dtype=np.float64)
            
        assert self.mYErr is not None
        assert self.mQcYTemp is not None
            
        h: float = htry
        
        while True:
            self._baseStep(x, y, dydx, h, self.mQcYTemp, self.mYErr)
            errmax: float = 0.0
            for i in range(self.mNVariables):
                errmax = max(errmax, abs(self.mYErr[i] / yscal[i]))
            errmax /= eps
            if errmax > 1.0:
                htemp: float = safety * h * math.pow(errmax, pshrnk)
                h = max(htemp, 0.1 * h) if h >= 0 else min(htemp, 0.1 * h)
                # Check for step size underflow
                xnew: float = x + h
                if xnew == x:
                    raise EPQException("Step size underflow in AdaptiveRungeKutta.qcStep.")
            
            if errmax <= 1.0:
                break
                
        self.mHNext = safety * h * math.pow(errmax, pgrow) if errmax > errcon else 5.0 * h
        self.mHDid = h
        x += h
        np.copyto(y, self.mQcYTemp)
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
        assert self.mXSave is not None
        return float(self.mXSave[i])

    def getY(self, i: int) -> F64Array:
        assert self.mYSave is not None
        return self.mYSave[i]

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
        """Faithful port of the Cash-Karp adaptive step size routine."""
        self._require_mutable_f64(ystart, "ystart")
        
        tiny: float = 1.0e-10 * eps
        yscal: F64Array = np.zeros(self.mNVariables, dtype=np.float64)
        dydx: F64Array = np.zeros(self.mNVariables, dtype=np.float64)
        y: F64Array = np.zeros(self.mNVariables, dtype=np.float64)
        
        x: float = x1
        h: float = self._sign(h1, x2 - x1)
        xsav: float = 0.0
        saveInt: float = sys.float_info.max
        kMax: int = 0
        
        self.mNSaved = 0
        self.mNOk = 0
        self.mNBad = 0
        np.copyto(y, ystart)
        
        if self.mSaveInterval != sys.float_info.max:
            # R7 applied to Math.round
            kMax = int(math.floor(((abs(x2 - x1) + self.mSaveInterval) / self.mSaveInterval) + 0.5))
            saveInt = self._sign(self.mSaveInterval, x2 - x1)
            xsav = x - (2.0 * saveInt)  # to ensure that the first step is saved...
            self.mXSave = np.zeros(kMax, dtype=np.float64)
            self.mYSave = np.zeros((kMax, self.mNVariables), dtype=np.float64)
            
        for step in range(self.mMaxSteps):
            # Save the necessary points
            if (kMax != 0) and (self.mNSaved < kMax) and (abs(x - xsav) >= (0.9999 * self.mSaveInterval)):
                assert self.mXSave is not None and self.mYSave is not None
                self.mXSave[self.mNSaved] = x
                np.copyto(self.mYSave[self.mNSaved], y)
                xsav = x
                self.mNSaved += 1
                
            self.derivatives(x, y, dydx)
            
            # Rescale h to ensure we hit desired points
            hMax: float = abs((xsav + saveInt) - x)
            if abs(h) > hMax:
                h = self._sign(hMax, h)
                
            # Scaling to monitor accuracy...
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
                np.copyto(ystart, y)
                if kMax != 0:
                    assert self.mXSave is not None and self.mYSave is not None
                    self.mNSaved = min(self.mNSaved, kMax - 1)
                    self.mXSave[self.mNSaved] = x
                    np.copyto(self.mYSave[self.mNSaved], y)
                    self.mNSaved += 1
                self._clearWorkspace()
                return y
                
            if abs(self.mHNext) <= self.mMinStepSize:
                raise EPQException("Step size too small in AdaptiveRungeKutta.integrate")
            h = self.mHNext
            
        raise EPQException("Too many steps in AdaptiveRungeKutta.integrate")

    def integrate(self, x1: float, x2: float, ystart: F64Array, eps: float, h1: float) -> F64Array:
        """
        Primary Scipy-backed integrator.
        SCIPY-DEV-1: Uses `scipy.integrate.solve_ivp` with RK45 instead of the literal 
        Cash-Karp method, bypassing `mXSave`, `mYSave`, `mHDid`, and manual step counts. 
        For parity checks or trajectories relying on saved intermediate results, 
        call `integrate_literal`.
        """
        self._require_mutable_f64(ystart, "ystart")

        def _fun(t: float, y_arr: np.ndarray) -> np.ndarray:
            dydx_arr = np.zeros_like(y_arr)
            self.derivatives(t, y_arr, dydx_arr)
            return dydx_arr

        res = _sp_integrate.solve_ivp(
            fun=_fun,
            t_span=(x1, x2),
            y0=ystart,
            method="RK45",
            rtol=eps,
            atol=1.0e-10 * eps
        )

        if not res.success:
            raise EPQException(f"scipy.integrate.solve_ivp failed: {res.message}")

        final_y: F64Array = res.y[:, -1]
        np.copyto(ystart, final_y)
        return final_y

    @abc.abstractmethod
    def derivatives(self, x: float, y: F64Array, dydx: F64Array) -> None:
        pass