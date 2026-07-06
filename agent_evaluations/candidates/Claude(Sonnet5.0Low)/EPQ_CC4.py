r"""
AdaptiveRungeKutta_ver1_1_0.py — Python port of gov.nist.microanalysis.Utility.AdaptiveRungeKutta

Guide version : 1
Generation    : 1
Port-code fixes: 0

CHANGES IN THIS REVISION (ver1_1_0)
------------------------------------------------------------------------
* R2 DEVIATION (documented per R10): `integrate` is not split into a
  scipy-primary `integrate()` plus a literal `integrate_literal()`.
  `baseStep`/`qcStep`/`integrate` together implement a specific,
  stateful Cash-Karp adaptive-step algorithm with side effects
  (mNSaved/mNOk/mNBad bookkeeping, optional trajectory-point saving at
  a caller-specified interval, and in-place mutation of the caller's
  `ystart` array to match Java's `(In & Out)` contract). Substituting
  a scipy ODE solver as the "primary" path would silently change the
  step-accept/reject sequence, the save-point cadence, and exception
  timing relative to the Java reference -- the exact things a parity
  harness for this class needs to check. `integrate()` therefore *is*
  the literal, line-for-line port; no separate `integrate_literal()`
  is provided. `baseStep`/`qcStep`/`sign`/`clearWorkspace` are Java
  `private` helpers and are ported as single (non-split, non-`_literal`)
  methods for the same reason.
* R5 COMPLIANCE: Java's `integrate` mutates the caller-supplied
  `ystart` array in place via `System.arraycopy(y, 0, ystart, 0, ...)`
  at every return path. The Python port preserves this aliasing
  behaviour but first calls `Math2._require_mutable_f64(ystart, ...)`
  so callers passing an immutable / wrong-dtype array get a clear
  `TypeError` instead of a silent no-op or a `ValueError` deep inside
  numpy.
* R7: The single integer-valued `Math.round(...)` call (in computing
  `kMax`) is translated as `int(math.floor(x + 0.5))` rather than
  Python's `round()`, which uses banker's rounding and would diverge
  from Java's `Math.round` on the `*.5` boundary.

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

try:
    from .Math2_ver8_1_5 import Math2
except ImportError:
    try:
        from Math2_ver8_1_5 import Math2  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility.Math2_ver8_1_5 import Math2  # type: ignore


__all__ = ["AdaptiveRungeKutta", "EPQException", "F64Array"]


class AdaptiveRungeKutta(abc.ABC):
    """Port of gov.nist.microanalysis.Utility.AdaptiveRungeKutta.

    An adaptive step size Runge-Kutta (Cash-Karp) algorithm for numerically
    evaluating differential equations. Can optionally save intermediate
    points along the ODE trajectory at a user specified interval.

    NOTE: Like the Java original, an instance carries mutable workspace
    state (`_mWs2`..`_mQcYTemp`, `_mXSave`/`_mYSave`, step counters) that is
    written to during `integrate`. Do not share a single instance across
    threads / concurrent integrations.
    """

    # ==================================================================
    # Preserved-bug ledger (machine-readable)
    # ==================================================================
    BUG_LEDGER: tuple = ()  # no bugs identified

    def __init__(self, nVars: int) -> None:
        """AdaptiveRungeKutta - Construct an AdaptiveRungeKutta object to
        solve a differential equation of nVars variables. The implementation
        of derivatives should return nVars derivative values for each x & y.

        Args:
            nVars: int - the number of differential equations.
        """
        self._mNVariables: int = nVars  # The number of differential equations
        self._mHDid: float = 0.0  # Actual step size accomplished in last call to qcStep
        self._mHNext: float = 0.0  # Next step size to try when calling qcStep
        self._mSaveInterval: float = float(np.finfo(np.float64).max)
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
    # Private helpers (Java `private`)
    # ==================================================================
    def _sign(self, magnitude: float, sign: float) -> float:
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
        """baseStep - Take a single Cash-Karp Runge-Kutta step. Given the
        n=mNVariables values y[0..n-1] and their derivatives dydx[0..n-1]
        known at x, use a fifth order Cash-Karp Runge-Kutta method to
        advance the solution over an interval h. The resulting y value is
        returned in yout. An estimate of the truncation error is returned
        in yerr.

        Args:
            x: float
            y: F64Array
            dydx: F64Array
            h: float
            yout: F64Array - (Out)
            yerr: F64Array - (Out)
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
            self._mYTemp[i] = y[i] + (
                h * ((b51 * dydx[i]) + (b52 * self._mWs2[i]) + (b53 * self._mWs3[i]) + (b54 * self._mWs4[i]))
            )
        # Fifth step
        self.derivatives(x + (a5 * h), self._mYTemp, self._mWs5)
        for i in range(self._mNVariables):
            self._mYTemp[i] = y[i] + (
                h
                * (
                    (b61 * dydx[i])
                    + (b62 * self._mWs2[i])
                    + (b63 * self._mWs3[i])
                    + (b64 * self._mWs4[i])
                    + (b65 * self._mWs5[i])
                )
            )
        # Sixth step
        self.derivatives(x + (a6 * h), self._mYTemp, self._mWs6)
        for i in range(self._mNVariables):
            yout[i] = y[i] + (h * ((c1 * dydx[i]) + (c3 * self._mWs3[i]) + (c4 * self._mWs4[i]) + (c6 * self._mWs6[i])))
        # Estimate the error
        for i in range(self._mNVariables):
            yerr[i] = h * (
                (dc1 * dydx[i]) + (dc3 * self._mWs3[i]) + (dc4 * self._mWs4[i]) + (dc5 * self._mWs5[i]) + (dc6 * self._mWs6[i])
            )

    def _qcStep(
        self,
        x: float,
        y: F64Array,
        dydx: F64Array,
        htry: float,
        eps: float,
        yscal: F64Array,
    ) -> float:
        """qcStep - Take a fifth order Runge-Kutta step with monitoring of
        local truncation error. Input are the dependent variable
        y[0..mNVariables-1] and its derivatives dydx[0..mNVariables-1] at
        the starting value of the independent variable x. Also input is the
        attempted step size htry, the required accuracy eps and the vector
        yscal against which the errors are scaled. Upon return, y is
        replaced with the new values, x is returned and mHDid and mHNext
        are set to the actual step size and the size of the next step to
        try.

        Args:
            x: float - (In) independent variable
            y: F64Array - (In, Out) dependent variable
            dydx: F64Array - (In) derivative at x
            htry: float - The step size to attempt
            eps: float - Desired accuracy
            yscal: F64Array - (In) Error scaling vector

        Raises:
            EPQException: When the step size becomes too small.

        Returns:
            float - The new value of x
        """
        safety: float = 0.9
        pgrow: float = -0.2
        pshrnk: float = -0.25
        errcon: float = 1.89e-4
        if self._mYErr is None:
            self._mYErr = np.zeros(self._mNVariables, dtype=np.float64)
            self._mQcYTemp = np.zeros(self._mNVariables, dtype=np.float64)
        errmax: float
        h: float = htry
        while True:
            self._baseStep(x, y, dydx, h, self._mQcYTemp, self._mYErr)
            errmax = 0.0
            for i in range(self._mNVariables):
                errmax = max(errmax, abs(self._mYErr[i] / yscal[i]))
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
        y[0 : self._mNVariables] = self._mQcYTemp[0 : self._mNVariables]
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
    # Public API (Java `public`)
    # ==================================================================
    def setSaveInterval(self, interval: float) -> None:
        """setSaveInterval - Set the interval on which to save intermediate
        points on the integrated trajectory. (Use clearSaveInterval to not
        save any intermediate points.) Note: The default is not to save any
        intermediate points.

        Args:
            interval: float
        """
        self._mSaveInterval = abs(interval)

    def clearSaveInterval(self) -> None:
        """clearSaveInterval - Return to the default of not saving any
        intermediate points."""
        self._mSaveInterval = float(np.finfo(np.float64).max)

    def getNSaved(self) -> int:
        """getNSaved - Returns the number of saved values.

        Returns:
            int
        """
        return self._mNSaved

    def getX(self, i: int) -> float:
        """getX - Returns the x-coordinate of the i-th saved value.

        Args:
            i: int - Where i < getNSaved()

        Returns:
            float
        """
        return float(self._mXSave[i])

    def getY(self, i: int) -> F64Array:
        """getY - returns the getNVariables x y-coordinates of the i-th
        saved values.

        Args:
            i: int - Where i < getNSaved()

        Returns:
            F64Array - Of dimension getNVariables
        """
        return self._mYSave[i]

    def setMaxSteps(self, maxSteps: int) -> None:
        """setMaxSteps - Set the maximum number of ODE steps to allow.
        Default is 10000.

        Args:
            maxSteps: int
        """
        self._mMaxSteps = maxSteps

    def setMinStepSize(self, minStep: float) -> None:
        """setMinStepSize - Sets the minimum permissible step size. Default
        is 0.0.

        Args:
            minStep: float
        """
        self._mMinStepSize = abs(minStep)

    def getNVariables(self) -> int:
        """getNVariables - Returns the number of variables as set in the
        constructor.

        Returns:
            int
        """
        return self._mNVariables

    def getStepCount(self) -> int:
        """getStepCount - Get the total number of steps required to perform
        the previous integrate operation.

        Returns:
            int
        """
        return self._mNOk + self._mNBad

    def getGoodStepCount(self) -> int:
        """getGoodStepCount - Get the number of steps leading to results of
        the desired accuracy.

        Returns:
            int
        """
        return self._mNOk

    def getBadStepCount(self) -> int:
        """getBadStepCount - Get the number of steps that were needed to be
        subdivided to attain results of the desired accuracy.

        Returns:
            int
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
        eps is measure of the permissible error. h is the initial step
        size.

        Args:
            x1: float - Start of the integration range
            x2: float - End of the integration range
            ystart: F64Array - (In & Out) The initial y value. Mutated
                in place to match the Java `(In & Out)` contract.
            eps: float - The permissible relative error
            h1: float - The initial step size

        Raises:
            EPQException: Upon too many steps or too small a step.

        Returns:
            F64Array - The final y values as an array of length
            getNVariables().
        """
        Math2._require_mutable_f64(ystart, "ystart")  # R5: in-place mutation below
        tiny: float = 1.0e-10 * eps
        yscal: F64Array = np.zeros(self._mNVariables, dtype=np.float64)
        dydx: F64Array = np.zeros(self._mNVariables, dtype=np.float64)
        y: F64Array = np.zeros(self._mNVariables, dtype=np.float64)
        x: float = x1
        h: float = self._sign(h1, x2 - x1)
        xsav: float = 0.0
        saveInt: float = float(np.finfo(np.float64).max)
        kMax: int = 0
        self._mNSaved = 0
        self._mNOk = 0
        self._mNBad = 0
        y[0 : self._mNVariables] = ystart[0 : self._mNVariables]
        if self._mSaveInterval != float(np.finfo(np.float64).max):
            # R7: Math.round(double) -> int(math.floor(x + 0.5))
            kMax = int(
                math.floor(((abs(x2 - x1) + self._mSaveInterval) / self._mSaveInterval) + 0.5)
            )
            saveInt = self._sign(self._mSaveInterval, x2 - x1)
            xsav = x - (2.0 * saveInt)  # to ensure that the first step is saved...
            self._mXSave = np.zeros(kMax, dtype=np.float64)
            self._mYSave = [np.zeros(self._mNVariables, dtype=np.float64) for _ in range(kMax)]
        for _step in range(self._mMaxSteps):
            # Save the necessary points
            if (kMax != 0) and (self._mNSaved < kMax) and (abs(x - xsav) >= (0.9999 * self._mSaveInterval)):
                self._mXSave[self._mNSaved] = x
                self._mYSave[self._mNSaved][0 : self._mNVariables] = y[0 : self._mNVariables]
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
                ystart[0 : self._mNVariables] = y[0 : self._mNVariables]
                if kMax != 0:
                    self._mNSaved = min(self._mNSaved, kMax - 1)
                    self._mXSave[self._mNSaved] = x
                    self._mYSave[self._mNSaved][0 : self._mNVariables] = y[0 : self._mNVariables]
                    self._mNSaved += 1
                self._clearWorkspace()
                return y
            if abs(self._mHNext) <= self._mMinStepSize:
                raise EPQException("Step size too small in AdaptiveRungeKutta.integrate")
            h = self._mHNext
        raise EPQException("Too many steps in AdaptiveRungeKutta.integrate")

    @abc.abstractmethod
    def derivatives(self, x: float, y: F64Array, dydx: F64Array) -> None:
        """derivatives - The derived class provides an implementation of the
        derivatives function. x & y[] are input and the user provided
        implementation of derivatives is responsible for returning the
        derivatives in the array dydx. The lengths of y and dydx are equal
        to mNVariables.

        Args:
            x: float - In
            y: F64Array - In (of dimension mNVariables)
            dydx: F64Array - Out (of dimension mNVariables)
        """
        raise NotImplementedError