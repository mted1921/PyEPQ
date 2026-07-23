r"""
AdaptiveRungeKutta_ver2_1_1.py — Python port of gov.nist.microanalysis.Utility.AdaptiveRungeKutta

Guide version : 2
Generation    : 1
Port-code fixes: 1

------------------------------------------------------------------------
Original Java source (gov.nist.microanalysis.Utility.AdaptiveRungeKutta)
------------------------------------------------------------------------
/**
 * An adaptive step size Runge-Kutta algorithm for numerically evaluating
 * differential equations. This implementation can optionally save intermediate
 * points along the ODE trajectory at a user specified interval.
 *
 * See Press, Teukolsky, Vetterling & Flannery, Numerical Recipes in C,
 * 2nd edition, pp 714-722 (Cash-Karp adaptive RK5).
 *
 * @author Nicholas W. M. Ritchie
 * @version 1.0
 */
------------------------------------------------------------------------

CHANGES:
* Abstract class implemented via abc.ABC + @abc.abstractmethod for derivatives() (R2).
* double[] → F64Array (numpy float64 arrays); element-wise loops preserved
  verbatim from Java — no vectorisation in this literal port (R2).
* double[][] mYSave → 2D F64Array of shape (kMax, _mNVariables).
* System.arraycopy(src, 0, dst, 0, n) → dst[:n] = src[:n].
* Double.MAX_VALUE sentinel → sys.float_info.max (same IEEE-754 bit pattern
  as Java's Double.MAX_VALUE). Spec says float('inf') but test is authoritative:
  test asserts _mSaveInterval == sys.float_info.max after clearSaveInterval().
* Java do-while in _qcStep → while-loop pre-seeded with errmax=2.0 (>1.0)
  so the first iteration always executes; exact semantics preserved.
* (int) Math.round(...) → int(math.floor(... + 0.5)) (R7).
* Private methods gain leading underscore (R1): _sign, _baseStep, _qcStep,
  _clearWorkspace.
* _sign(magnitude, sign): faithful inline conditional — Java's sign >= 0.0
  treats -0.0 as non-negative (IEEE 754); math.copysign would differ.
* integrate() is the primary public method backed by scipy.integrate.solve_ivp
  (RK45). The faithful Cash-Karp translation is available as integrate_literal().
  Deviations from Java documented as SCIPY-DEV-1..4 in integrate().

CHANGES:
* FIX-1 (R1 + API-mismatch): All private fields renamed from `mXxx` to `_mXxx`
  throughout — test harness accesses _mSaveInterval, _mHDid, _mHNext, _mWs2,
  _mYErr, _mQcYTemp directly. _mSaveInterval sentinel is sys.float_info.max
  (not float('inf')) per test assertion (test is authoritative over spec).
  integrate_literal() included (R2 — required by TestExceptions and
  TestLiteralPortCorrectness groups in the parity harness).
"""
from __future__ import annotations

import abc
import math
import sys
from typing import Optional

import numpy as np

try:
    from ._epq_compat import F64Array
except ImportError:
    try:
        from _epq_compat import F64Array  # type: ignore[no-redef]
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility_ver1._epq_compat import F64Array  # type: ignore[no-redef]

try:
    from .UtilException_ver2_1_1 import UtilException
except ImportError:
    try:
        from UtilException_ver2_1_1 import UtilException  # type: ignore[no-redef]
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility_ver1.UtilException_ver2_1_1 import UtilException  # type: ignore[no-redef]


_DOUBLE_MAX_VALUE: float = sys.float_info.max  # FIX-1: sys.float_info.max not float('inf')


class AdaptiveRungeKutta(abc.ABC):
    """Port of gov.nist.microanalysis.Utility.AdaptiveRungeKutta.

    Adaptive step-size Runge-Kutta ODE integrator (Cash-Karp 5th order).
    Subclass and implement derivatives(); then call integrate().

    NOTE: Not thread-safe. Use each instance in one and only one thread.
    """

    BUG_LEDGER: tuple = (
        ("SCIPY-DEV-1", "integrate",
         "mNOk is set to result.nfev (function evaluations); mNBad = 0. "
         "scipy does not expose accepted/rejected step counts.", False),
        ("SCIPY-DEV-2", "integrate",
         "mHDid and mHNext are set to 0.0; not tracked by solve_ivp.", False),
        ("SCIPY-DEV-3", "integrate",
         "mMaxSteps and mMinStepSize are ignored by scipy; "
         "UtilException is NOT raised for them by integrate(). "
         "Use integrate_literal() for Java-faithful enforcement.", False),
        ("SCIPY-DEV-4", "integrate",
         "With a save interval, saved x values are at exact multiples "
         "via t_eval interpolation, not the literal port's adaptive "
         "threshold-based sampling.", False),
    )

    def __init__(self, nVars: int) -> None:
        self._mNVariables: int = nVars          # FIX-1: _mXxx naming throughout
        self._mHDid: float = 0.0
        self._mHNext: float = 0.0
        self._mSaveInterval: float = _DOUBLE_MAX_VALUE
        self._mMinStepSize: float = 0.0
        self._mXSave: Optional[F64Array] = None
        self._mYSave: Optional[F64Array] = None
        self._mNSaved: int = 0
        self._mMaxSteps: int = 10000
        self._mNOk: int = 0
        self._mNBad: int = 0
        self._mWs2: Optional[F64Array] = None
        self._mWs3: Optional[F64Array] = None
        self._mWs4: Optional[F64Array] = None
        self._mWs5: Optional[F64Array] = None
        self._mWs6: Optional[F64Array] = None
        self._mYTemp: Optional[F64Array] = None
        self._mYErr: Optional[F64Array] = None
        self._mQcYTemp: Optional[F64Array] = None

    @abc.abstractmethod
    def derivatives(self, x: float, y: F64Array, dydx: F64Array) -> None:
        """Compute derivatives at (x, y) and write them into dydx."""

    def _sign(self, magnitude: float, sign: float) -> float:
        """Return abs(magnitude) with the sign of sign.

        Java: sign >= 0.0 ? Math.abs(magnitude) : -Math.abs(magnitude).
        Java treats -0.0 as non-negative (IEEE 754: -0.0 >= 0.0 is true).
        """
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
        """Single Cash-Karp RK5 step (Numerical Recipes in C, 2nd ed., p.717)."""
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
        if self._mWs2 is None:
            self._mWs2 = np.empty(self._mNVariables, dtype=np.float64)
            self._mWs3 = np.empty(self._mNVariables, dtype=np.float64)
            self._mWs4 = np.empty(self._mNVariables, dtype=np.float64)
            self._mWs5 = np.empty(self._mNVariables, dtype=np.float64)
            self._mWs6 = np.empty(self._mNVariables, dtype=np.float64)
            self._mYTemp = np.empty(self._mNVariables, dtype=np.float64)
        for i in range(self._mNVariables):
            self._mYTemp[i] = y[i] + (b21 * h * dydx[i])
        self.derivatives(x + (a2 * h), self._mYTemp, self._mWs2)
        for i in range(self._mNVariables):
            self._mYTemp[i] = y[i] + (h * ((b31 * dydx[i]) + (b32 * self._mWs2[i])))
        self.derivatives(x + (a3 * h), self._mYTemp, self._mWs3)
        for i in range(self._mNVariables):
            self._mYTemp[i] = y[i] + (h * ((b41 * dydx[i]) + (b42 * self._mWs2[i]) + (b43 * self._mWs3[i])))
        self.derivatives(x + (a4 * h), self._mYTemp, self._mWs4)
        for i in range(self._mNVariables):
            self._mYTemp[i] = y[i] + (h * ((b51 * dydx[i]) + (b52 * self._mWs2[i]) + (b53 * self._mWs3[i]) + (b54 * self._mWs4[i])))
        self.derivatives(x + (a5 * h), self._mYTemp, self._mWs5)
        for i in range(self._mNVariables):
            self._mYTemp[i] = y[i] + (h * ((b61 * dydx[i]) + (b62 * self._mWs2[i]) + (b63 * self._mWs3[i]) + (b64 * self._mWs4[i]) + (b65 * self._mWs5[i])))
        self.derivatives(x + (a6 * h), self._mYTemp, self._mWs6)
        for i in range(self._mNVariables):
            yout[i] = y[i] + (h * ((c1 * dydx[i]) + (c3 * self._mWs3[i]) + (c4 * self._mWs4[i]) + (c6 * self._mWs6[i])))
        for i in range(self._mNVariables):
            yerr[i] = h * ((dc1 * dydx[i]) + (dc3 * self._mWs3[i]) + (dc4 * self._mWs4[i]) + (dc5 * self._mWs5[i]) + (dc6 * self._mWs6[i]))

    def _qcStep(
        self,
        x: float,
        y: F64Array,
        dydx: F64Array,
        htry: float,
        eps: float,
        yscal: F64Array,
    ) -> float:
        """Quality-controlled adaptive step; mutates y in-place."""
        safety: float = 0.9
        pgrow: float = -0.2
        pshrnk: float = -0.25
        errcon: float = 1.89e-4
        if self._mYErr is None:
            self._mYErr = np.empty(self._mNVariables, dtype=np.float64)
            self._mQcYTemp = np.empty(self._mNVariables, dtype=np.float64)
        h: float = htry
        errmax: float = 2.0
        while errmax > 1.0:
            self._baseStep(x, y, dydx, h, self._mQcYTemp, self._mYErr)
            errmax = 0.0
            for i in range(self._mNVariables):
                errmax = max(errmax, abs(self._mYErr[i] / yscal[i]))
            errmax /= eps
            if errmax > 1.0:
                htemp: float = safety * h * math.pow(errmax, pshrnk)
                h = (max(htemp, 0.1 * h) if h >= 0 else min(htemp, 0.1 * h))
                xnew: float = x + h
                if xnew == x:
                    raise UtilException("Step size underflow in AdaptiveRungeKutta.qcStep.")
        self._mHNext = (safety * h * math.pow(errmax, pgrow) if errmax > errcon else 5.0 * h)
        self._mHDid = h
        x += h
        y[:self._mNVariables] = self._mQcYTemp[:self._mNVariables]
        return x

    def _clearWorkspace(self) -> None:
        """Release all lazily-allocated workspace arrays."""
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
        self._mSaveInterval = _DOUBLE_MAX_VALUE

    def setMaxSteps(self, maxSteps: int) -> None:
        self._mMaxSteps = maxSteps

    def setMinStepSize(self, minStep: float) -> None:
        self._mMinStepSize = abs(minStep)

    def getNVariables(self) -> int:
        return self._mNVariables

    def getNSaved(self) -> int:
        return self._mNSaved

    def getX(self, i: int) -> float:
        return float(self._mXSave[i])

    def getY(self, i: int) -> F64Array:
        return self._mYSave[i]

    def getStepCount(self) -> int:
        return self._mNOk + self._mNBad

    def getGoodStepCount(self) -> int:
        return self._mNOk

    def getBadStepCount(self) -> int:
        return self._mNBad

    def integrate(
        self,
        x1: float,
        x2: float,
        ystart: F64Array,
        eps: float,
        h1: float,
    ) -> F64Array:
        """Integrate from x1 to x2 using scipy RK45 (primary public method).

        SCIPY-DEV-1: mNOk = result.nfev; mNBad = 0.
        SCIPY-DEV-2: mHDid = mHNext = 0.0.
        SCIPY-DEV-3: mMaxSteps and mMinStepSize ignored; no UtilException.
        SCIPY-DEV-4: saved x values interpolated at exact multiples.
        """
        from scipy.integrate import solve_ivp

        def fun(x: float, y: np.ndarray) -> F64Array:
            dydx: F64Array = np.empty(self._mNVariables, dtype=np.float64)
            self.derivatives(x, y, dydx)
            return dydx

        t_eval: Optional[F64Array] = None
        if self._mSaveInterval != _DOUBLE_MAX_VALUE:
            direction: float = 1.0 if x2 >= x1 else -1.0
            total: float = abs(x2 - x1)
            n_pts: int = int(math.floor(total / self._mSaveInterval)) + 2
            pts: list[float] = []
            for k in range(n_pts):
                t: float = x1 + direction * k * self._mSaveInterval
                if direction * (t - x2) <= 0.0:
                    pts.append(t)
            if not pts or abs(pts[-1] - x2) > 1e-10 * max(total, 1e-30):
                pts.append(x2)
            t_eval = np.array(pts, dtype=np.float64)

        result = solve_ivp(
            fun,
            (x1, x2),
            np.array(ystart[:self._mNVariables], dtype=np.float64),
            method="RK45",
            t_eval=t_eval,
            rtol=eps,
            atol=eps,
            first_step=abs(h1),
        )

        if not result.success:
            raise UtilException(
                f"scipy solve_ivp failed in AdaptiveRungeKutta.integrate: {result.message}"
            )

        y_final: F64Array = result.y[:, -1].copy()
        ystart[:self._mNVariables] = y_final

        self._mNSaved = 0
        self._mXSave = None
        self._mYSave = None
        if t_eval is not None:
            self._mNSaved = int(result.t.shape[0])
            self._mXSave = result.t.copy()
            self._mYSave = result.y.T.copy()

        self._mNOk = int(result.nfev)
        self._mNBad = 0
        self._mHDid = 0.0
        self._mHNext = 0.0

        return y_final

    def integrate_literal(  # FIX-1: required by TestExceptions + TestLiteralPortCorrectness
        self,
        x1: float,
        x2: float,
        ystart: F64Array,
        eps: float,
        h1: float,
    ) -> F64Array:
        """Faithful Cash-Karp RK5 translation of the Java integrate() method.

        Respects _mMaxSteps, _mMinStepSize, and threshold-based save-interval
        sampling. Raises UtilException on too many steps or step-size underflow.
        """
        tiny: float = 1.0e-10 * eps
        yscal: F64Array = np.empty(self._mNVariables, dtype=np.float64)
        dydx: F64Array = np.empty(self._mNVariables, dtype=np.float64)
        y: F64Array = np.empty(self._mNVariables, dtype=np.float64)
        x: float = x1
        h: float = self._sign(h1, x2 - x1)
        xsav: float = 0.0
        saveInt: float = _DOUBLE_MAX_VALUE
        kMax: int = 0
        self._mNSaved = 0
        self._mNOk = 0
        self._mNBad = 0
        y[:] = ystart[:self._mNVariables]
        if self._mSaveInterval != _DOUBLE_MAX_VALUE:
            kMax = int(math.floor((abs(x2 - x1) + self._mSaveInterval) / self._mSaveInterval + 0.5))
            saveInt = self._sign(self._mSaveInterval, x2 - x1)
            xsav = x - (2.0 * saveInt)
            self._mXSave = np.empty(kMax, dtype=np.float64)
            self._mYSave = np.empty((kMax, self._mNVariables), dtype=np.float64)
        for step in range(self._mMaxSteps):
            if (kMax != 0) and (self._mNSaved < kMax) and (abs(x - xsav) >= (0.9999 * self._mSaveInterval)):
                self._mXSave[self._mNSaved] = x
                self._mYSave[self._mNSaved, :] = y[:]
                xsav = x
                self._mNSaved += 1
            self.derivatives(x, y, dydx)
            hMax: float = abs((xsav + saveInt) - x)
            if abs(h) > hMax:
                h = self._sign(hMax, h)
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
                ystart[:self._mNVariables] = y[:]
                if kMax != 0:
                    self._mNSaved = min(self._mNSaved, kMax - 1)
                    self._mXSave[self._mNSaved] = x
                    self._mYSave[self._mNSaved, :] = y[:]
                    self._mNSaved += 1
                self._clearWorkspace()
                return y
            if abs(self._mHNext) <= self._mMinStepSize:
                raise UtilException("Step size too small in AdaptiveRungeKutta.integrate")
            h = self._mHNext
        raise UtilException("Too many steps in AdaptiveRungeKutta.integrate")
