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

CHANGES (CONVERSION_GUIDE.md R10)
----------------------------------
* R3: `UtilException` (a sibling, not-yet-standardised Java exception type)
  is mapped onto the shared `EPQException` from `_epq_compat`, consistent
  with every other converted Utility class (e.g. Math2_ver8_1_5.py). This
  keeps a single exception hierarchy across the port so `except
  EPQException` clauses written against one module also catch errors
  raised by this one.
* SCIPY-DEV-1: `integrate()` is a scipy-backed primary that delegates to
  `scipy.integrate.solve_ivp` (RK45 with dense output) for the numerical
  work, then re-derives the same instance-level bookkeeping fields
  (`mNSaved`, `mXSave`, `mYSave`, `mNOk`, `mNBad`, `mHDid`, `mHNext`) that
  Java callers of `getNSaved()` / `getX()` / `getY()` / `getStepCount()`
  rely on, by resampling the dense solution at `mSaveInterval` and
  counting `solve_ivp`'s internal step count. `mNOk`/`mNBad` degrade to
  "steps taken" / 0 because `solve_ivp` does not expose per-step
  accept/reject counts the way the Cash-Karp `qcStep` loop does; this is
  an observable but harmless deviation for accessors whose only
  documented contract is `getStepCount() == getGoodStepCount() +
  getBadStepCount()`. Use `integrate_literal()` when bit-for-bit parity
  with the Java Cash-Karp stepper (including exact good/bad step counts)
  is required.
* Because `derivatives(x, y, dydx)` is an abstract Java callback with an
  *output* parameter (`dydx` is written in place), the scipy code path
  wraps it in a small adapter that allocates a fresh buffer per call
  instead of reusing a work array, since `solve_ivp`'s `fun(t, y)` is
  expected to be side-effect-free and return an array.

BUG_LEDGER: tuple = ()  # no bugs identified in the Java source
------------------------------------------------------------------------
"""

from __future__ import annotations

import abc
import math
import numpy as np
from typing import Optional, Sequence, Union, Callable

try:
    from ._epq_compat import EPQException, F64Array, JavaRandom
except ImportError:
    try:
        from _epq_compat import EPQException, F64Array, JavaRandom  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility._epq_compat import EPQException, F64Array, JavaRandom  # type: ignore

from scipy.integrate import solve_ivp


__all__ = ["AdaptiveRungeKutta", "EPQException", "F64Array"]


class AdaptiveRungeKutta(abc.ABC):
    """Python port of gov.nist.microanalysis.Utility.AdaptiveRungeKutta.

    See the verbatim Javadoc above for algorithm background (Numerical
    Recipes in C, 2nd ed., pp 714-722, adaptive Cash-Karp Runge-Kutta).

    NOTE: This class is not thread-safe (mirrors the Java note): a single
    instance carries mutable workspace and trajectory-save state across
    calls to `integrate()` / `integrate_literal()`.
    """

    # BUG_LEDGER: (id, method, description, has_strict_variant)
    BUG_LEDGER: tuple = ()  # no bugs identified

    def __init__(self, nVars: int) -> None:
        """AdaptiveRungeKutta - Construct an AdaptiveRungeKutta object to solve a
        differential equation of nVars variables. The implementation of
        derivatives should return nVars derivative values for each x & y.

        @param nVars int
        """
        self.mNVariables: int = nVars  # The number of differential equations
        self.mHDid: float = 0.0  # Actual step size accomplished in last call to qcStep
        self.mHNext: float = 0.0  # Next step size to try when calling qcStep
        self.mSaveInterval: float = sys_float_max()
        self.mMinStepSize: float = 0.0
        self.mXSave: Optional[F64Array] = None
        self.mYSave: Optional[F64Array] = None  # shape (kMax, mNVariables)
        self.mNSaved: int = 0
        self.mMaxSteps: int = 10000
        self.mNOk: int = 0  # Number of ok steps
        self.mNBad: int = 0  # Number of repeated steps
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

    # ==================================================================
    # Private helpers
    # ==================================================================
    def _sign(self, magnitude: float, sign: float) -> float:
        return math.fabs(magnitude) if sign >= 0.0 else -math.fabs(magnitude)

    def _baseStep(self, x: float, y: F64Array, dydx: F64Array, h: float,
                  yout: F64Array, yerr: F64Array) -> None:
        """baseStep - Take a single Cash-Karp Runge-Kutta step. Given the
        n=mNDimensions values y[0..n-1] and their derivatives dydx[0..n-1] know at
        x, use a fifth order Cash-Karp Runge-Kutta method to advance the solution
        over an interval h. The resulting y value is returned in yout. An estimate
        of the truncation error is returned in yerr.

        @param x double
        @param y double[]
        @param dydx double[]
        @param h double
        @param yout double[]
        """
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
        if self.mWs2 is None:
            self.mWs2 = np.zeros(self.mNVariables, dtype=np.float64)
            self.mWs3 = np.zeros(self.mNVariables, dtype=np.float64)
            self.mWs4 = np.zeros(self.mNVariables, dtype=np.float64)
            self.mWs5 = np.zeros(self.mNVariables, dtype=np.float64)
            self.mWs6 = np.zeros(self.mNVariables, dtype=np.float64)
            self.mYTemp = np.zeros(self.mNVariables, dtype=np.float64)
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

    def _qcStep(self, x: float, y: F64Array, dydx: F64Array, htry: float,
                eps: float, yscal: F64Array) -> float:
        """qcStep - Take a fifth order Runge-Kutta step with monitoring of local
        truncation error. Input are the dependent variable y[0..mNDimensions-1]
        and its derivatives dydx[0..mNDimensions-1] at the starting value of the
        independent variable x. Also input is the attempted step size htry, the
        required accuracy eps and the vector yscal against which the errors are
        scaled. Upon return, y is replaced with the new values, x is returned and
        mHDid and mHNext are set to the actual step size and the size of the next
        step to try.

        @param x double - (In) independent variable
        @param y double[] - (In,Out) dependent variable
        @param dydx double[] - (In) derivative at x
        @param htry double - The step size to attempt
        @param eps double - Desired accuracy
        @param yscal double[] - (In) Error scaling vector
        @throws EPQException - When the step size becomes too small (Java: UtilException)
        @return double - The new value of x
        """
        safety: float = 0.9
        pgrow: float = -0.2
        pshrnk: float = -0.25
        errcon: float = 1.89e-4
        if self.mYErr is None:
            self.mYErr = np.zeros(self.mNVariables, dtype=np.float64)
            self.mQcYTemp = np.zeros(self.mNVariables, dtype=np.float64)
        errmax: float = 0.0
        h: float = htry
        while True:
            self._baseStep(x, y, dydx, h, self.mQcYTemp, self.mYErr)
            errmax = 0.0
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
            if not (errmax > 1.0):
                break
        self.mHNext = safety * h * math.pow(errmax, pgrow) if errmax > errcon else 5.0 * h
        self.mHDid = h
        x += h
        y[0:self.mNVariables] = self.mQcYTemp[0:self.mNVariables]
        return x

    def _clearWorkspace(self) -> None:
        """clearWorkspace - null all temporary space to free memory"""
        self.mWs2 = None
        self.mWs3 = None
        self.mWs4 = None
        self.mWs5 = None
        self.mWs6 = None
        self.mYTemp = None
        self.mYErr = None
        self.mQcYTemp = None

    # ==================================================================
    # Public API
    # ==================================================================
    def setSaveInterval(self, interval: float) -> None:
        """setSaveInterval - Set the interval on which to save intermediate points on
        the integrated trajectory. (Use clearSaveInterval to not save any
        intermediate points.) Note: The default is not to save any intermediate
        points.

        @param interval double
        """
        self.mSaveInterval = math.fabs(interval)

    def clearSaveInterval(self) -> None:
        """clearSaveInterval - Return to the default of not saving any intermediate
        points."""
        self.mSaveInterval = sys_float_max()

    def getNSaved(self) -> int:
        """getNSaved - Returns the number of saved values.

        @return int
        """
        return self.mNSaved

    def getX(self, i: int) -> float:
        """getX - Returns the x-coordinate of the i-th saved value

        @param i int - Where i<getNSaved()
        @return double
        """
        assert self.mXSave is not None
        return float(self.mXSave[i])

    def getY(self, i: int) -> F64Array:
        """getY - returns the getNVariable x y-coordinates of the i-th saved values.

        @param i int - Where i<getNSaved()
        @return double[] - Of dimension getNVariables
        """
        assert self.mYSave is not None
        return self.mYSave[i]

    def setMaxSteps(self, maxSteps: int) -> None:
        """setMaxSteps - Set the maximum number of ODE steps to allow. Default is
        10000.

        @param maxSteps int
        """
        self.mMaxSteps = maxSteps

    def setMinStepSize(self, minStep: float) -> None:
        """setMinStepSize - Sets the minimum permissible step size. Default is 0.0.

        @param minStep double
        """
        self.mMinStepSize = math.fabs(minStep)

    def getNVariables(self) -> int:
        """getNVariables - Returns the number of variables as set in the constructor.

        @return int
        """
        return self.mNVariables

    def getStepCount(self) -> int:
        """getStepCount - Get the total number of steps required to perform the
        previous integrate operation.

        @return int
        """
        return self.mNOk + self.mNBad

    def getGoodStepCount(self) -> int:
        """getGoodStepCount - Get the number of steps leading to results of the
        desired accuracy.

        @return int
        """
        return self.mNOk

    def getBadStepCount(self) -> int:
        """getBadStepCount - Get the number of steps that were needed to be
        subdivided to attain results of the desired accuracy.

        @return int
        """
        return self.mNBad

    # ------------------------------------------------------------------
    # integrate() / integrate_literal()  (CONVERSION_GUIDE.md R2)
    # ------------------------------------------------------------------
    def integrate_literal(self, x1: float, x2: float, ystart: F64Array,
                           eps: float, h1: float) -> F64Array:
        """integrate - Integrate the ODE specified by derivatives using the adaptive
        step size Runge-Kutta algorithm over the independent variable interval x1
        to x2. ystart contains the initial y values. eps is measure of the
        permissible error. h is the initial step size.

        Literal, line-for-line port of the Java Cash-Karp adaptive stepper.

        @param x1 double - Start of the integration range
        @param x2 double - End of the integration range
        @param ystart double[] - (In & out) The initial y value
        @param eps double - The permissible relative error
        @param h1 double - The initial step size
        @return The final y values as an array of length getNVariables().
        @throws EPQException - Upon too many steps or too small a step (Java: UtilException)
        """
        tiny: float = 1.0e-10 * eps
        yscal: F64Array = np.zeros(self.mNVariables, dtype=np.float64)
        dydx: F64Array = np.zeros(self.mNVariables, dtype=np.float64)
        y: F64Array = np.zeros(self.mNVariables, dtype=np.float64)
        x: float = x1
        h: float = self._sign(h1, x2 - x1)
        xsav: float = 0.0
        saveInt: float = sys_float_max()
        kMax: int = 0
        self.mNSaved = 0
        self.mNOk = 0
        self.mNBad = 0
        y[0:self.mNVariables] = ystart[0:self.mNVariables]
        if self.mSaveInterval != sys_float_max():
            kMax = int(math.floor(((math.fabs(x2 - x1) + self.mSaveInterval) / self.mSaveInterval) + 0.5))
            saveInt = self._sign(self.mSaveInterval, x2 - x1)
            xsav = x - (2.0 * saveInt)  # to ensure that the first step is saved...
            self.mXSave = np.zeros(kMax, dtype=np.float64)
            self.mYSave = np.zeros((kMax, self.mNVariables), dtype=np.float64)
        for _step in range(self.mMaxSteps):
            # Save the necessary points
            if (kMax != 0) and (self.mNSaved < kMax) and (math.fabs(x - xsav) >= (0.9999 * self.mSaveInterval)):
                self.mXSave[self.mNSaved] = x
                self.mYSave[self.mNSaved, 0:self.mNVariables] = y[0:self.mNVariables]
                xsav = x
                self.mNSaved += 1
            self.derivatives(x, y, dydx)
            # Rescale h to ensure we hit desired points
            hMax: float = math.fabs((xsav + saveInt) - x)
            if math.fabs(h) > hMax:
                h = self._sign(hMax, h)
            # Scaling to monitor accuracy...
            for i in range(self.mNVariables):
                yscal[i] = math.fabs(y[i]) + math.fabs(dydx[i] * h) + tiny
            if ((((x + h) - x2) * ((x + h) - x1)) > 0.0):
                h = x2 - x
            x = self._qcStep(x, y, dydx, h, eps, yscal)
            if self.mHDid == h:
                self.mNOk += 1
            else:
                self.mNBad += 1
            if (((x - x2) * (x2 - x1)) >= 0.0):
                ystart[0:self.mNVariables] = y[0:self.mNVariables]
                if kMax != 0:
                    self.mNSaved = min(self.mNSaved, kMax - 1)
                    self.mXSave[self.mNSaved] = x
                    self.mYSave[self.mNSaved, 0:self.mNVariables] = y[0:self.mNVariables]
                    self.mNSaved += 1
                self._clearWorkspace()
                return y
            if math.fabs(self.mHNext) <= self.mMinStepSize:
                raise EPQException("Step size too small in AdaptiveRungeKutta.integrate")
            h = self.mHNext
        raise EPQException("Too many steps in AdaptiveRungeKutta.integrate")

    def integrate(self, x1: float, x2: float, ystart: F64Array,
                  eps: float, h1: float) -> F64Array:
        """Scipy-backed primary. See CHANGES / SCIPY-DEV-1 in the module
        docstring for the deviations this introduces relative to the
        literal Cash-Karp stepper in `integrate_literal()`.

        @param x1 double - Start of the integration range
        @param x2 double - End of the integration range
        @param ystart double[] - (In & out) The initial y value
        @param eps double - The permissible relative error
        @param h1 double - The initial step size (used only as a hint; scipy
               chooses its own initial step)
        @return The final y values as an array of length getNVariables().
        @throws EPQException - Upon integration failure
        """
        n: int = self.mNVariables

        def _fun(t: float, yy: F64Array) -> F64Array:
            dydx: F64Array = np.zeros(n, dtype=np.float64)
            self.derivatives(t, yy, dydx)
            return dydx

        want_save: bool = self.mSaveInterval != sys_float_max()
        sol = solve_ivp(
            _fun,
            (x1, x2),
            np.asarray(ystart, dtype=np.float64)[0:n].copy(),
            method="RK45",
            rtol=eps,
            atol=eps * 1.0e-3,
            first_step=math.fabs(h1) if h1 != 0.0 else None,
            dense_output=want_save,
        )
        if not sol.success:
            # SCIPY-DEV-1: map solve_ivp failure onto the shared EPQException
            raise EPQException("scipy.integrate.solve_ivp failed in AdaptiveRungeKutta.integrate: " + str(sol.message))

        yfinal: F64Array = sol.y[:, -1].astype(np.float64)
        ystart[0:n] = yfinal[0:n]

        # SCIPY-DEV-1: solve_ivp does not distinguish "good" vs "repeated"
        # steps the way the Cash-Karp qcStep loop does; approximate with
        # all accepted steps counted as "good".
        self.mNOk = int(sol.t.shape[0] - 1) if sol.t.shape[0] > 0 else 0
        self.mNBad = 0
        if sol.t.shape[0] >= 2:
            self.mHDid = float(sol.t[-1] - sol.t[-2])
        self.mHNext = self.mHDid

        self.mNSaved = 0
        self.mXSave = None
        self.mYSave = None
        if want_save:
            saveInt: float = self._sign(self.mSaveInterval, x2 - x1)
            kMax: int = int(math.floor(((math.fabs(x2 - x1) + self.mSaveInterval) / self.mSaveInterval) + 0.5))
            xs = [x1]
            xcur = x1
            while True:
                xnext = xcur + saveInt
                if (saveInt > 0 and xnext >= x2) or (saveInt < 0 and xnext <= x2):
                    break
                xs.append(xnext)
                xcur = xnext
                if len(xs) >= kMax:
                    break
            xs.append(x2)
            xs_arr = np.array(xs, dtype=np.float64)
            self.mXSave = xs_arr
            self.mYSave = sol.sol(xs_arr).T.astype(np.float64)
            self.mNSaved = xs_arr.shape[0]

        self._clearWorkspace()
        return yfinal

    @abc.abstractmethod
    def derivatives(self, x: float, y: F64Array, dydx: F64Array) -> None:
        """derivatives - The derived class provides an implementation of the
        derivatives function. x & y[] are input and the user provided
        implementation of derivatives is resposible for returning the derivatives
        in the array dydx. The lengths of y and dydx are equal to mNDimensions.

        @param x double - In
        @param y double[] - In (of dimension mNDimensions)
        @param dydx double[] - Out (of dimension mNDimensions)
        """
        ...


def sys_float_max() -> float:
    """Python equivalent of Java's Double.MAX_VALUE (largest finite double)."""
    import sys
    return sys.float_info.max