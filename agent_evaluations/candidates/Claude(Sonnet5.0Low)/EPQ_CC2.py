r"""
AdaptiveRungeKutta_ver1_1_0.py — Python port of gov.nist.microanalysis.Utility.AdaptiveRungeKutta

Guide version : 1
Generation    : 1
Port-code fixes: 0

CHANGES (see CONVERSION_GUIDE.md rules)
----------------------------------------
* R2-DEVIATION: `integrate` is the sole public "mathematical" entry point.
  Unlike Math2's stateless scalar functions, `integrate` is intrinsically
  stateful (it populates the save-trajectory buffers read back via
  `getNSaved`/`getX`/`getY` and the step counters read via
  `getStepCount`/`getGoodStepCount`/`getBadStepCount`), and its step-control
  logic (adaptive Cash-Karp RK45 with save-interval snapping) has no
  drop-in scipy.integrate equivalent that preserves this exact state
  contract. `integrate()` is therefore implemented as the line-for-line
  Cash-Karp/quality-controlled-step algorithm from the Java source (Numerical
  Recipes in C, 2nd ed., pp 714-722), and `integrate_literal()` is provided
  as an explicit alias of the same code path so the R2 naming contract
  (`foo()` + `foo_literal()`) is satisfied without silently duplicating or
  diverging behaviour.

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
import numpy as np
from typing import Optional, Sequence, Union, Callable

try:
    from ._epq_compat import EPQException, F64Array, JavaRandom
except ImportError:
    try:
        from _epq_compat import EPQException, F64Array, JavaRandom  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility._epq_compat import EPQException, F64Array, JavaRandom  # type: ignore


__all__ = ["AdaptiveRungeKutta", "EPQException", "F64Array"]


class AdaptiveRungeKutta(abc.ABC):
    """Adaptive step size Runge-Kutta (Cash-Karp RK45) ODE solver.

    See Press, Teukolsky, Vetterling & Flannery, Numerical Recipes in C,
    2nd ed., pp 714-722. Subclass and implement :meth:`derivatives`.

    NOTE: Like the Java original, an instance carries mutable per-run
    workspace state (`_mWs2` .. `_mQcYTemp`, save buffers, step counters)
    and is NOT thread-safe. Use each instance in one and only one thread.
    """

    # ==================================================================
    # Preserved-bug ledger (machine-readable)
    # ==================================================================
    # No Java bugs were identified in this class: the step-control
    # algorithm (baseStep / qcStep / integrate) is a faithful, unmodified
    # transcription of the Numerical Recipes Cash-Karp RK45 reference
    # algorithm, and every branch, comparison, and array bound was
    # verified against the cited Java source line-for-line.
    BUG_LEDGER: tuple = ()  # no bugs identified

    def __init__(self, nVars: int) -> None:
        """AdaptiveRungeKutta - Construct an AdaptiveRungeKutta object to solve
        a differential equation of nVars variables. The implementation of
        derivatives should return nVars derivative values for each x & y.

        @param nVars: int
        """
        self._mNVariables: int = nVars  # final int mNVariables
        self._mHDid: float = 0.0  # Actual step size accomplished in last call to qcStep
        self._mHNext: float = 0.0  # Next step size to try when calling qcStep
        self._mSaveInterval: float = float(np.finfo(np.float64).max)  # Double.MAX_VALUE
        self._mMinStepSize: float = 0.0
        self._mXSave: Optional[F64Array] = None
        self._mYSave: Optional[list] = None
        self._mNSaved: int = 0
        self._mMaxSteps: int = 10000
        self._mNOk: int = 0  # Number of ok steps
        self._mNBad: int = 0  # Number of repeated steps
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

    # ==================================================================
    # Private helpers
    # ==================================================================

    def _sign(self, magnitude: float, sign: float) -> float:
        """private double sign(double magnitude, double sign)."""
        return math.fabs(magnitude) if sign >= 0.0 else -math.fabs(magnitude)

    def _baseStep(
        self,
        x: float,
        y: F64Array,
        dydx: F64Array,
        h: float,
        yout: F64Array,
        yerr: F64Array,
    ) -> None:
        """baseStep - Take a single Cash-Karp Runge-Kutta step. Given the
        n=mNDimensions values y[0..n-1] and their derivatives dydx[0..n-1] know
        at x, use a fifth order Cash-Karp Runge-Kutta method to advance the
        solution over an interval h. The resulting y value is returned in
        yout. An estimate of the truncation error is returned in yerr.

        @param x: double
        @param y: double[]
        @param dydx: double[]
        @param h: double
        @param yout: double[]
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
        if self._mWs2 is None:
            self._mWs2 = np.zeros(self._mNVariables, dtype=np.float64)
            self._mWs3 = np.zeros(self._mNVariables, dtype=np.float64)
            self._mWs4 = np.zeros(self._mNVariables, dtype=np.float64)
            self._mWs5 = np.zeros(self._mNVariables, dtype=np.float64)
            self._mWs6 = np.zeros(self._mNVariables, dtype=np.float64)
            self._mYTemp = np.zeros(self._mNVariables, dtype=np.float64)
        ws2, ws3, ws4, ws5, ws6 = self._mWs2, self._mWs3, self._mWs4, self._mWs5, self._mWs6
        yTemp = self._mYTemp
        n: int = self._mNVariables
        # First step
        for i in range(n):
            yTemp[i] = y[i] + (b21 * h * dydx[i])
        # Second step
        self.derivatives(x + (a2 * h), yTemp, ws2)
        for i in range(n):
            yTemp[i] = y[i] + (h * ((b31 * dydx[i]) + (b32 * ws2[i])))
        # Third step
        self.derivatives(x + (a3 * h), yTemp, ws3)
        for i in range(n):
            yTemp[i] = y[i] + (h * ((b41 * dydx[i]) + (b42 * ws2[i]) + (b43 * ws3[i])))
        # Fourth step
        self.derivatives(x + (a4 * h), yTemp, ws4)
        for i in range(n):
            yTemp[i] = y[i] + (h * ((b51 * dydx[i]) + (b52 * ws2[i]) + (b53 * ws3[i]) + (b54 * ws4[i])))
        # Fifth step
        self.derivatives(x + (a5 * h), yTemp, ws5)
        for i in range(n):
            yTemp[i] = y[i] + (h * ((b61 * dydx[i]) + (b62 * ws2[i]) + (b63 * ws3[i]) + (b64 * ws4[i]) + (b65 * ws5[i])))
        # Sixth step
        self.derivatives(x + (a6 * h), yTemp, ws6)
        for i in range(n):
            yout[i] = y[i] + (h * ((c1 * dydx[i]) + (c3 * ws3[i]) + (c4 * ws4[i]) + (c6 * ws6[i])))
        # Estimate the error
        for i in range(n):
            yerr[i] = h * ((dc1 * dydx[i]) + (dc3 * ws3[i]) + (dc4 * ws4[i]) + (dc5 * ws5[i]) + (dc6 * ws6[i]))

    def _qcStep(
        self,
        x: float,
        y: F64Array,
        dydx: F64Array,
        htry: float,
        eps: float,
        yscal: F64Array,
    ) -> float:
        """qcStep - Take a fifth order Runge-Kutta step with monitoring of local
        truncation error. Input are the dependent variable y[0..mNDimensions-1]
        and its derivatives dydx[0..mNDimensions-1] at the starting value of
        the independent variable x. Also input is the attempted step size
        htry, the required accuracy eps and the vector yscal against which the
        errors are scaled. Upon return, y is replaced with the new values, x
        is returned and mHDid and mHNext are set to the actual step size and
        the size of the next step to try.

        @param x: double - (In) independent variable
        @param y: double[] - (In,Out) dependent variable
        @param dydx: double[] - (In) derivative at x
        @param htry: double - The step size to attempt
        @param eps: double - Desired accuracy
        @param yscal: double[] - (In) Error scaling vector
        @raises EPQException: - When the step size becomes too small
        @return: double - The new value of x
        """
        safety: float = 0.9
        pgrow: float = -0.2
        pshrnk: float = -0.25
        errcon: float = 1.89e-4
        if self._mYErr is None:
            self._mYErr = np.zeros(self._mNVariables, dtype=np.float64)
            self._mQcYTemp = np.zeros(self._mNVariables, dtype=np.float64)
        yErr = self._mYErr
        qcYTemp = self._mQcYTemp
        n: int = self._mNVariables
        errmax: float = 0.0
        h: float = htry
        while True:
            self._baseStep(x, y, dydx, h, qcYTemp, yErr)
            errmax = 0.0
            for i in range(n):
                errmax = max(errmax, math.fabs(yErr[i] / yscal[i]))
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
        self._mHNext = safety * h * math.pow(errmax, pgrow) if errmax > errcon else 5.0 * h
        self._mHDid = h
        x += h
        y[:n] = qcYTemp[:n]
        return x

    def _clearWorkspace(self) -> None:
        """clearWorkspace - null all temporary space to free memory"""
        self._mWs2 = None
        self._mWs3 = None
        self._mWs4 = None
        self._mWs5 = None
        self._mWs6 = None
        self._mYTemp = None
        self._mYErr = None
        self._mQcYTemp = None

    # ==================================================================
    # Public API
    # ==================================================================

    def setSaveInterval(self, interval: float) -> None:
        """setSaveInterval - Set the interval on which to save intermediate
        points on the integrated trajectory. (Use clearSaveInterval to not
        save any intermediate points.) Note: The default is not to save any
        intermediate points.

        @param interval: double
        """
        self._mSaveInterval = math.fabs(interval)

    def clearSaveInterval(self) -> None:
        """clearSaveInterval - Return to the default of not saving any
        intermediate points."""
        self._mSaveInterval = float(np.finfo(np.float64).max)

    def getNSaved(self) -> int:
        """getNSaved - Returns the number of saved values.

        @return: int
        """
        return self._mNSaved

    def getX(self, i: int) -> float:
        """getX - Returns the x-coordinate of the i-th saved value

        @param i: int - Where i<getNSaved()
        @return: double
        """
        return self._mXSave[i]

    def getY(self, i: int) -> F64Array:
        """getY - returns the getNVariable x y-coordinates of the i-th saved
        values.

        @param i: int - Where i<getNSaved()
        @return: double[] - Of dimension getNVariables
        """
        return self._mYSave[i]

    def setMaxSteps(self, maxSteps: int) -> None:
        """setMaxSteps - Set the maximum number of ODE steps to allow.
        Default is 10000.

        @param maxSteps: int
        """
        self._mMaxSteps = maxSteps

    def setMinStepSize(self, minStep: float) -> None:
        """setMinStepSize - Sets the minimum permissible step size. Default
        is 0.0.

        @param minStep: double
        """
        self._mMinStepSize = math.fabs(minStep)

    def getNVariables(self) -> int:
        """getNVariables - Returns the number of variables as set in the
        constructor.

        @return: int
        """
        return self._mNVariables

    def getStepCount(self) -> int:
        """getStepCount - Get the total number of steps required to perform
        the previous integrate operation.

        @return: int
        """
        return self._mNOk + self._mNBad

    def getGoodStepCount(self) -> int:
        """getGoodStepCount - Get the number of steps leading to results of
        the desired accuracy.

        @return: int
        """
        return self._mNOk

    def getBadStepCount(self) -> int:
        """getBadStepCount - Get the number of steps that were needed to be
        subdivided to attain results of the desired accuracy.

        @return: int
        """
        return self._mNBad

    def integrate(
        self,
        x1: float,
        x2: float,
        ystart: F64Array,
        eps: float,
        h1: float,
    ) -> F64Array:
        """integrate - Integrate the ODE specified by derivatives using the
        adaptive step size Runge-Kutta algorithm over the independent
        variable interval x1 to x2. ystart contains the initial y values.
        eps is measure of the permissible error. h is the initial step size.

        See CHANGES header: this is the sole implementation (R2-DEVIATION —
        no scipy substitution preserves the save-trajectory / step-counter
        state contract). `integrate_literal` is an alias of this method.

        @param x1: double - Start of the integration range
        @param x2: double - End of the integration range
        @param ystart: double[] - (In & out) The initial y value
        @param eps: double - The permissible relative error
        @param h1: double - The initial step size
        @return: The final y values as an array of length getNVariables().
        @raises EPQException: - Upon too many steps or too small a step
        """
        n: int = self._mNVariables
        tiny: float = 1.0e-10 * eps
        yscal: F64Array = np.zeros(n, dtype=np.float64)
        dydx: F64Array = np.zeros(n, dtype=np.float64)
        y: F64Array = np.zeros(n, dtype=np.float64)
        x: float = x1
        h: float = self._sign(h1, x2 - x1)
        xsav: float = 0.0
        saveInt: float = float(np.finfo(np.float64).max)
        kMax: int = 0
        self._mNSaved = 0
        self._mNOk = 0
        self._mNBad = 0
        y[:n] = np.asarray(ystart, dtype=np.float64)[:n]
        if self._mSaveInterval != float(np.finfo(np.float64).max):
            kMax = int(math.floor((math.fabs(x2 - x1) + self._mSaveInterval) / self._mSaveInterval + 0.5))
            saveInt = self._sign(self._mSaveInterval, x2 - x1)
            xsav = x - (2.0 * saveInt)  # to ensure that the first step is saved...
            self._mXSave = np.zeros(kMax, dtype=np.float64)
            self._mYSave = [np.zeros(n, dtype=np.float64) for _ in range(kMax)]
        for _step in range(self._mMaxSteps):
            # Save the necessary points
            if (kMax != 0) and (self._mNSaved < kMax) and (math.fabs(x - xsav) >= (0.9999 * self._mSaveInterval)):
                self._mXSave[self._mNSaved] = x
                self._mYSave[self._mNSaved][:n] = y[:n]
                xsav = x
                self._mNSaved += 1
            self.derivatives(x, y, dydx)
            # Rescale h to ensure we hit desired points
            hMax: float = math.fabs((xsav + saveInt) - x)
            if math.fabs(h) > hMax:
                h = self._sign(hMax, h)
            # Scaling to monitor accuracy...
            for i in range(n):
                yscal[i] = math.fabs(y[i]) + math.fabs(dydx[i] * h) + tiny
            if (((x + h) - x2) * ((x + h) - x1)) > 0.0:
                h = x2 - x
            x = self._qcStep(x, y, dydx, h, eps, yscal)
            if self._mHDid == h:
                self._mNOk += 1
            else:
                self._mNBad += 1
            if ((x - x2) * (x2 - x1)) >= 0.0:
                ystart[:n] = y[:n]
                if kMax != 0:
                    self._mNSaved = min(self._mNSaved, kMax - 1)
                    self._mXSave[self._mNSaved] = x
                    self._mYSave[self._mNSaved][:n] = y[:n]
                    self._mNSaved += 1
                self._clearWorkspace()
                return y
            if math.fabs(self._mHNext) <= self._mMinStepSize:
                raise EPQException("Step size too small in AdaptiveRungeKutta.integrate")
            h = self._mHNext
        raise EPQException("Too many steps in AdaptiveRungeKutta.integrate")

    def integrate_literal(
        self,
        x1: float,
        x2: float,
        ystart: F64Array,
        eps: float,
        h1: float,
    ) -> F64Array:
        """R2 alias of :meth:`integrate` — see CHANGES header (R2-DEVIATION)."""
        return self.integrate(x1, x2, ystart, eps, h1)

    @abc.abstractmethod
    def derivatives(self, x: float, y: F64Array, dydx: F64Array) -> None:
        """derivatives - The derived class provides an implementation of the
        derivatives function. x & y[] are input and the user provided
        implementation of derivatives is responsible for returning the
        derivatives in the array dydx. The lengths of y and dydx are equal to
        mNDimensions.

        @param x: double - In
        @param y: double[] - In (of dimension mNDimensions)
        @param dydx: double[] - Out (of dimension mNDimensions)
        """
        raise NotImplementedError