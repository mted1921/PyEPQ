"""
AdaptiveRungeKutta_ver1.py -- Faithful Python port of
gov.nist.microanalysis.Utility.AdaptiveRungeKutta.

Java class hierarchy: AdaptiveRungeKutta (abstract).
Python equivalent: abc.ABC with @abc.abstractmethod for derivatives().

See Press, Teukolsky, Vetterling & Flannery, Numerical Recipes in C,
2nd edition, pp 714-722 (Cash-Karp adaptive RK5).

CHANGES
-------
- Abstract class implemented via abc.ABC + @abc.abstractmethod (R2).
- double[] → F64Array (numpy float64 arrays); element-wise loops preserved
  verbatim from Java — no vectorisation in this literal port (R2).
- double[][] mYSave → 2D F64Array of shape (kMax, mNVariables).
- System.arraycopy(src, 0, dst, 0, n) → dst[:n] = src[:n].
- Double.MAX_VALUE sentinel for mSaveInterval/saveInt → sys.float_info.max,
  which carries the same IEEE-754 bit pattern as Java's Double.MAX_VALUE.
- Java do-while in _qcStep → while-loop pre-seeded with errmax = 2.0 (> 1.0)
  so the first iteration always executes; exact semantics preserved.
- (int) Math.round(...) for kMax → int(math.floor(... + 0.5)) (R7).
- Private methods gain leading underscore (R1): sign → _sign,
  baseStep → _baseStep, qcStep → _qcStep, clearWorkspace → _clearWorkspace.
- _sign(magnitude, sign): faithful inline conditional — NOT math.copysign —
  because Java's sign >= 0.0 treats -0.0 as non-negative (IEEE 754), while
  math.copysign(x, -0.0) returns a negative result (see spec).
- integrate() is the primary public method backed by scipy.integrate.solve_ivp
  (RK45). The faithful Cash-Karp translation is available as integrate_literal().
  Deviations from Java documented as SCIPY-DEV-1..4 in integrate().

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
from typing import Optional

import numpy as np

from _epq_compat import F64Array
from gov.nist.microanalysis.PyEPQ.Utility.UtilException import UtilException


# IEEE-754 double maximum — identical to Java's Double.MAX_VALUE.
# Used as a sentinel meaning "save interval not set" and as the initial
# value of the local saveInt in integrate() when saving is disabled.
_DOUBLE_MAX_VALUE: float = sys.float_info.max


class AdaptiveRungeKutta(abc.ABC):
    """Port of gov.nist.microanalysis.Utility.AdaptiveRungeKutta.

    Adaptive step-size Runge-Kutta ODE integrator (Cash-Karp 5th order).
    Subclass and implement derivatives(); then call integrate().

    Example::

        class SinCosODE(AdaptiveRungeKutta):
            def derivatives(self, x, y, dydx):
                dydx[0] = -math.sin(x)
                dydx[1] = math.cos(x)

        ode = SinCosODE(2)
        ode.setSaveInterval(math.pi / 16.0)
        y0 = np.array([1.0, 0.0])
        ode.integrate(0.0, 2.0 * math.pi, y0, 1e-6, 0.01)

    NOTE: Not thread-safe. Use each instance in one and only one thread.
    """

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(self, nVars: int) -> None:
        """Construct an integrator for a system of nVars coupled ODEs."""
        self.mNVariables: int = nVars
        self.mHDid: float = 0.0
        self.mHNext: float = 0.0
        self.mSaveInterval: float = _DOUBLE_MAX_VALUE
        self.mMinStepSize: float = 0.0
        self.mXSave: Optional[F64Array] = None
        self.mYSave: Optional[F64Array] = None   # 2D, shape (kMax, mNVariables)
        self.mNSaved: int = 0
        self.mMaxSteps: int = 10000
        self.mNOk: int = 0
        self.mNBad: int = 0
        # Workspace for _baseStep (lazily allocated on first call)
        self.mWs2: Optional[F64Array] = None
        self.mWs3: Optional[F64Array] = None
        self.mWs4: Optional[F64Array] = None
        self.mWs5: Optional[F64Array] = None
        self.mWs6: Optional[F64Array] = None
        self.mYTemp: Optional[F64Array] = None
        # Workspace for _qcStep (lazily allocated on first call)
        self.mYErr: Optional[F64Array] = None
        self.mQcYTemp: Optional[F64Array] = None

    # ------------------------------------------------------------------
    # Abstract extension point
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def derivatives(self, x: float, y: F64Array, dydx: F64Array) -> None:
        """Compute derivatives at (x, y) and write them into dydx.

        x   -- (in)  independent variable
        y   -- (in)  dependent variable values, length getNVariables()
        dydx -- (out) derivatives dy/dx, length getNVariables()
        """

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sign(self, magnitude: float, sign: float) -> float:
        """Return abs(magnitude) with the sign of sign.

        Matches Java: sign >= 0.0 ? Math.abs(magnitude) : -Math.abs(magnitude).
        Differs from math.copysign: Java treats -0.0 as non-negative (IEEE 754
        comparison is true for -0.0 >= 0.0), so _sign(m, -0.0) → +|m|.
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
        """Single Cash-Karp RK5 step.

        Advances the solution from x over interval h. Writes the new
        y values into yout and the error estimate into yerr.
        """
        # Cash-Karp Butcher tableau (Numerical Recipes in C, 2nd ed., p.717)
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
        # Lazy workspace allocation (mirrors Java's null-check pattern)
        if self.mWs2 is None:
            self.mWs2 = np.empty(self.mNVariables, dtype=np.float64)
            self.mWs3 = np.empty(self.mNVariables, dtype=np.float64)
            self.mWs4 = np.empty(self.mNVariables, dtype=np.float64)
            self.mWs5 = np.empty(self.mNVariables, dtype=np.float64)
            self.mWs6 = np.empty(self.mNVariables, dtype=np.float64)
            self.mYTemp = np.empty(self.mNVariables, dtype=np.float64)
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
        # Error estimate (difference between 4th- and 5th-order solutions)
        for i in range(self.mNVariables):
            yerr[i] = h * ((dc1 * dydx[i]) + (dc3 * self.mWs3[i]) + (dc4 * self.mWs4[i]) + (dc5 * self.mWs5[i]) + (dc6 * self.mWs6[i]))

    def _qcStep(
        self,
        x: float,
        y: F64Array,
        dydx: F64Array,
        htry: float,
        eps: float,
        yscal: F64Array,
    ) -> float:
        """Quality-controlled adaptive step.

        Attempts htry; shrinks h and retries if the local truncation error
        exceeds eps. Mutates y in place with the accepted new values.
        Sets mHDid (accepted step) and mHNext (suggested next step).
        Returns the new value of x.

        Raises UtilException if the step size shrinks to machine epsilon.
        """
        safety: float = 0.9
        pgrow: float = -0.2
        pshrnk: float = -0.25
        errcon: float = 1.89e-4
        if self.mYErr is None:
            self.mYErr = np.empty(self.mNVariables, dtype=np.float64)
            self.mQcYTemp = np.empty(self.mNVariables, dtype=np.float64)
        h: float = htry
        # errmax = 2.0 seeds the while-loop to guarantee the first iteration
        # runs — this replicates Java's do-while semantics exactly.
        errmax: float = 2.0
        while errmax > 1.0:
            self._baseStep(x, y, dydx, h, self.mQcYTemp, self.mYErr)
            errmax = 0.0
            for i in range(self.mNVariables):
                errmax = max(errmax, abs(self.mYErr[i] / yscal[i]))
            errmax /= eps
            if errmax > 1.0:
                htemp: float = safety * h * math.pow(errmax, pshrnk)
                h = (max(htemp, 0.1 * h) if h >= 0 else min(htemp, 0.1 * h))
                xnew: float = x + h
                if xnew == x:
                    raise UtilException("Step size underflow in AdaptiveRungeKutta.qcStep.")
        self.mHNext = (safety * h * math.pow(errmax, pgrow) if errmax > errcon else 5.0 * h)
        self.mHDid = h
        x += h
        y[:self.mNVariables] = self.mQcYTemp[:self.mNVariables]
        return x

    def _clearWorkspace(self) -> None:
        """Release all lazily-allocated workspace arrays."""
        self.mWs2 = None
        self.mWs3 = None
        self.mWs4 = None
        self.mWs5 = None
        self.mWs6 = None
        self.mYTemp = None
        self.mYErr = None
        self.mQcYTemp = None

    # ------------------------------------------------------------------
    # Public configuration
    # ------------------------------------------------------------------

    def setSaveInterval(self, interval: float) -> None:
        """Save intermediate trajectory points every abs(interval) in x."""
        self.mSaveInterval = abs(interval)

    def clearSaveInterval(self) -> None:
        """Revert to default: do not save any intermediate points."""
        self.mSaveInterval = _DOUBLE_MAX_VALUE

    def setMaxSteps(self, maxSteps: int) -> None:
        """Set maximum ODE steps before UtilException is raised. Default 10000."""
        self.mMaxSteps = maxSteps

    def setMinStepSize(self, minStep: float) -> None:
        """Set minimum permissible step size. Default 0.0."""
        self.mMinStepSize = abs(minStep)

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def getNVariables(self) -> int:
        """Number of dependent variables (set in constructor)."""
        return self.mNVariables

    def getNSaved(self) -> int:
        """Number of trajectory points saved by the last integrate() call."""
        return self.mNSaved

    def getX(self, i: int) -> float:
        """x-coordinate of the i-th saved point (i < getNSaved())."""
        return float(self.mXSave[i])

    def getY(self, i: int) -> F64Array:
        """y-values of the i-th saved point; array of length getNVariables()."""
        return self.mYSave[i]

    def getStepCount(self) -> int:
        """Total steps taken in the last integrate() call."""
        return self.mNOk + self.mNBad

    def getGoodStepCount(self) -> int:
        """Steps accepted on the first try (mHDid == htry)."""
        return self.mNOk

    def getBadStepCount(self) -> int:
        """Steps that required step-size reduction."""
        return self.mNBad

    # ------------------------------------------------------------------
    # integrate  (primary — scipy RK45)
    # ------------------------------------------------------------------

    def integrate(
        self,
        x1: float,
        x2: float,
        ystart: F64Array,
        eps: float,
        h1: float,
    ) -> F64Array:
        """Integrate from x1 to x2 using scipy RK45 (primary public method).

        ystart -- (in/out) initial y values; mutated in place with final values
        eps    -- tolerance passed as rtol=atol=eps to solve_ivp
        h1     -- initial step size hint (abs value forwarded to first_step)

        Returns the final y array.

        SCIPY-DEV-1: mNOk is set to result.nfev (function evaluations);
                     mNBad = 0. scipy does not expose accepted/rejected counts.
        SCIPY-DEV-2: mHDid and mHNext are set to 0.0; not tracked by solve_ivp.
        SCIPY-DEV-3: mMaxSteps and mMinStepSize are ignored; scipy has no
                     direct equivalent and will not raise UtilException for them.
        SCIPY-DEV-4: With a save interval, saved x values are at exact multiples
                     via t_eval interpolation, not the literal port's adaptive
                     threshold-based sampling.

        Raises UtilException if solve_ivp reports failure.
        """
        from scipy.integrate import solve_ivp

        def fun(x: float, y: np.ndarray) -> F64Array:
            dydx: F64Array = np.empty(self.mNVariables, dtype=np.float64)
            self.derivatives(x, y, dydx)
            return dydx

        # Build t_eval from save-interval spacing when saving is active
        t_eval: Optional[F64Array] = None
        if self.mSaveInterval != _DOUBLE_MAX_VALUE:
            direction: float = 1.0 if x2 >= x1 else -1.0
            total: float = abs(x2 - x1)
            n_pts: int = int(math.floor(total / self.mSaveInterval)) + 2
            pts: list[float] = []
            for k in range(n_pts):
                t: float = x1 + direction * k * self.mSaveInterval
                if direction * (t - x2) <= 0.0:
                    pts.append(t)
            if not pts or abs(pts[-1] - x2) > 1e-10 * max(total, 1e-30):
                pts.append(x2)
            t_eval = np.array(pts, dtype=np.float64)

        result = solve_ivp(
            fun,
            (x1, x2),
            np.array(ystart[:self.mNVariables], dtype=np.float64),
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
        ystart[:self.mNVariables] = y_final

        # Populate saved trajectory from t_eval result
        self.mNSaved = 0
        self.mXSave = None
        self.mYSave = None
        if t_eval is not None:
            self.mNSaved = int(result.t.shape[0])
            self.mXSave = result.t.copy()
            self.mYSave = result.y.T.copy()   # (mNSaved, mNVariables)

        # SCIPY-DEV-1: nfev approximates total evaluations; bad-step unavailable
        self.mNOk = int(result.nfev)
        self.mNBad = 0
        self.mHDid = 0.0    # SCIPY-DEV-2
        self.mHNext = 0.0   # SCIPY-DEV-2

        return y_final

    # ------------------------------------------------------------------
    # integrate_literal  (faithful Cash-Karp translation)
    # ------------------------------------------------------------------

    def integrate_literal(
        self,
        x1: float,
        x2: float,
        ystart: F64Array,
        eps: float,
        h1: float,
    ) -> F64Array:
        """Faithful Cash-Karp RK5 translation of the Java integrate() method.

        Identical contract to integrate() but uses the hand-rolled _qcStep /
        _baseStep implementation. Respects mMaxSteps, mMinStepSize, mHDid,
        mHNext, and the threshold-based save-interval sampling from Java.

        ystart -- (in/out) initial y values; mutated in place with final values
        eps    -- permissible relative error per step
        h1     -- initial step size (sign adjusted to match x2-x1 direction)

        Returns the final y array.

        Raises UtilException on too many steps or step-size underflow.
        """
        tiny: float = 1.0e-10 * eps
        yscal: F64Array = np.empty(self.mNVariables, dtype=np.float64)
        dydx: F64Array = np.empty(self.mNVariables, dtype=np.float64)
        y: F64Array = np.empty(self.mNVariables, dtype=np.float64)
        x: float = x1
        h: float = self._sign(h1, x2 - x1)
        xsav: float = 0.0
        saveInt: float = _DOUBLE_MAX_VALUE
        kMax: int = 0
        self.mNSaved = 0
        self.mNOk = 0
        self.mNBad = 0
        y[:] = ystart[:self.mNVariables]
        if self.mSaveInterval != _DOUBLE_MAX_VALUE:
            kMax = int(math.floor((abs(x2 - x1) + self.mSaveInterval) / self.mSaveInterval + 0.5))
            saveInt = self._sign(self.mSaveInterval, x2 - x1)
            xsav = x - (2.0 * saveInt)   # ensures first point is saved
            self.mXSave = np.empty(kMax, dtype=np.float64)
            self.mYSave = np.empty((kMax, self.mNVariables), dtype=np.float64)
        for step in range(self.mMaxSteps):
            # Save trajectory point if we have travelled at least one interval
            if (kMax != 0) and (self.mNSaved < kMax) and (abs(x - xsav) >= (0.9999 * self.mSaveInterval)):
                self.mXSave[self.mNSaved] = x
                self.mYSave[self.mNSaved, :] = y[:]
                xsav = x
                self.mNSaved += 1
            self.derivatives(x, y, dydx)
            # Limit h so we do not overshoot the next save point
            hMax: float = abs((xsav + saveInt) - x)
            if abs(h) > hMax:
                h = self._sign(hMax, h)
            # Scale for error monitoring
            for i in range(self.mNVariables):
                yscal[i] = abs(y[i]) + abs(dydx[i] * h) + tiny
            # Clamp h to not overshoot x2
            if (((x + h) - x2) * ((x + h) - x1)) > 0.0:
                h = x2 - x
            x = self._qcStep(x, y, dydx, h, eps, yscal)
            if self.mHDid == h:
                self.mNOk += 1
            else:
                self.mNBad += 1
            # Check whether we have reached or passed x2
            if ((x - x2) * (x2 - x1)) >= 0.0:
                ystart[:self.mNVariables] = y[:]
                if kMax != 0:
                    self.mNSaved = min(self.mNSaved, kMax - 1)
                    self.mXSave[self.mNSaved] = x
                    self.mYSave[self.mNSaved, :] = y[:]
                    self.mNSaved += 1
                self._clearWorkspace()
                return y
            if abs(self.mHNext) <= self.mMinStepSize:
                raise UtilException("Step size too small in AdaptiveRungeKutta.integrate")
            h = self.mHNext
        raise UtilException("Too many steps in AdaptiveRungeKutta.integrate")
