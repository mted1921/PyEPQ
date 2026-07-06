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
* SCIPY-DEV-1: No scipy-backed primary is provided for `integrate`.
  Java's `integrate` is stateful: it populates `getStepCount()`,
  `getGoodStepCount()`, `getBadStepCount()`, and (when a save interval
  is set) the `getNSaved()/getX(i)/getY(i)` trajectory buffers as a
  side effect of the exact Cash-Karp adaptive step-doubling algorithm.
  A `scipy.integrate.solve_ivp` substitution would use a different
  step-acceptance sequence and could not honour that contract (R2
  "Faithful AND complete": the getters are part of the public API
  surface). `integrate()` therefore *is* the literal port; `integrate_literal()`
  is provided as an explicit alias per R2's naming convention so
  parity-harness code that looks for `<method>_literal` still finds it.
* No BUG_LEDGER entries were identified in this class (see BUG_LEDGER
  below) — the algorithm is a direct, unmodified Numerical-Recipes-style
  Cash-Karp adaptive Runge-Kutta integrator.

BUG_LEDGER: tuple = ()  # no bugs identified
"""

from __future__ import annotations

import abc
import math
from typing import Optional

import numpy as np
from typing import Sequence, Union, Callable  # noqa: F401  (kept for parity with the style reference)

try:
    from ._epq_compat import EPQException, F64Array, JavaRandom  # noqa: F401
except ImportError:
    try:
        from _epq_compat import EPQException, F64Array, JavaRandom  # type: ignore  # noqa: F401
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility._epq_compat import EPQException, F64Array, JavaRandom  # type: ignore  # noqa: F401

# R3 / "Sibling module filenames taken from UTILITY_LEDGER.md 'Port file' column":
# UtilException's port file is UtilException_ver1_1_0.py.
try:
    from .UtilException_ver1_1_0 import UtilException
except ImportError:
    try:
        from UtilException_ver1_1_0 import UtilException  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility.UtilException_ver1_1_0 import UtilException  # type: ignore


__all__ = ["AdaptiveRungeKutta", "EPQException", "UtilException", "F64Array"]


class AdaptiveRungeKutta(abc.ABC):
    """Python port of gov.nist.microanalysis.Utility.AdaptiveRungeKutta.

    See the verbatim Javadoc in the module docstring above for the
    algorithm description, worked example, and thread-safety note.
    """

    # ==================================================================
    # BUG_LEDGER (CONVERSION_GUIDE R6)
    # ==================================================================
    BUG_LEDGER: tuple = ()  # no bugs identified

    def __init__(self, nVars: int) -> None:
        """AdaptiveRungeKutta - Construct an AdaptiveRungeKutta object to solve a
        differential equation of nVars variables. The implementation of
        derivatives should return nVars derivative values for each x & y.

        @param nVars int
        """
        # -- fields (Java `private`) --
        self._mNVariables: int = nVars  # final int: number of differential equations
        self._mHDid: float = 0.0  # Actual step size accomplished in last call to qcStep
        self._mHNext: float = 0.0  # Next step size to try when calling qcStep
        self._mSaveInterval: float = sys_float_max()
        self._mMinStepSize: float = 0.0
        self._mXSave: Optional[F64Array] = None
        self._mYSave: Optional[np.ndarray] = None  # shape (kMax, mNVariables), float64
        self._mNSaved: int = 0
        self._mMaxSteps: int = 10000
        self._mNOk: int = 0  # Number of ok steps
        self._mNBad: int = 0  # Number of repeated steps
        # Temporary work space used by _baseStep
        self._mWs2: Optional[F64Array] = None
        self._mWs3: Optional[F64Array] = None
        self._mWs4: Optional[F64Array] = None
        self._mWs5: Optional[F64Array] = None
        self._mWs6: Optional[F64Array] = None
        self._mYTemp: Optional[F64Array] = None
        # Temporary work space used by _qcStep
        self._mYErr: Optional[F64Array] = None
        self._mQcYTemp: Optional[F64Array] = None

    # ==================================================================
    # private helpers  (Java `private` -> `_`-prefixed, R1)
    # ==================================================================
    def _sign(self, magnitude: float, sign: float) -> float:
        """Java: private double sign(double magnitude, double sign)."""
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
        if self._mWs2 is None:
            self._mWs2 = np.zeros(self._mNVariables, dtype=np.float64)
            self._mWs3 = np.zeros(self._mNVariables, dtype=np.float64)
            self._mWs4 = np.zeros(self._mNVariables, dtype=np.float64)
            self._mWs5 = np.zeros(self._mNVariables, dtype=np.float64)
            self._mWs6 = np.zeros(self._mNVariables, dtype=np.float64)
            self._mYTemp = np.zeros(self._mNVariables, dtype=np.float64)
        mWs2: F64Array = self._mWs2
        mWs3: F64Array = self._mWs3
        mWs4: F64Array = self._mWs4
        mWs5: F64Array = self._mWs5
        mWs6: F64Array = self._mWs6
        mYTemp: F64Array = self._mYTemp
        # First step
        for i in range(self._mNVariables):
            mYTemp[i] = y[i] + (b21 * h * dydx[i])
        # Second step
        self.derivatives(x + (a2 * h), mYTemp, mWs2)
        for i in range(self._mNVariables):
            mYTemp[i] = y[i] + (h * ((b31 * dydx[i]) + (b32 * mWs2[i])))
        # Third step
        self.derivatives(x + (a3 * h), mYTemp, mWs3)
        for i in range(self._mNVariables):
            mYTemp[i] = y[i] + (h * ((b41 * dydx[i]) + (b42 * mWs2[i]) + (b43 * mWs3[i])))
        # Fourth step
        self.derivatives(x + (a4 * h), mYTemp, mWs4)
        for i in range(self._mNVariables):
            mYTemp[i] = y[i] + (h * ((b51 * dydx[i]) + (b52 * mWs2[i]) + (b53 * mWs3[i]) + (b54 * mWs4[i])))
        # Fifth step
        self.derivatives(x + (a5 * h), mYTemp, mWs5)
        for i in range(self._mNVariables):
            mYTemp[i] = y[i] + (
                h * ((b61 * dydx[i]) + (b62 * mWs2[i]) + (b63 * mWs3[i]) + (b64 * mWs4[i]) + (b65 * mWs5[i]))
            )
        # Sixth step
        self.derivatives(x + (a6 * h), mYTemp, mWs6)
        for i in range(self._mNVariables):
            yout[i] = y[i] + (h * ((c1 * dydx[i]) + (c3 * mWs3[i]) + (c4 * mWs4[i]) + (c6 * mWs6[i])))
        # Estimate the error
        for i in range(self._mNVariables):
            yerr[i] = h * ((dc1 * dydx[i]) + (dc3 * mWs3[i]) + (dc4 * mWs4[i]) + (dc5 * mWs5[i]) + (dc6 * mWs6[i]))

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
        @throws UtilException - When the step size becomes too small
        @return double - The new value of x
        """
        safety: float = 0.9
        pgrow: float = -0.2
        pshrnk: float = -0.25
        errcon: float = 1.89e-4
        if self._mYErr is None:
            self._mYErr = np.zeros(self._mNVariables, dtype=np.float64)
            self._mQcYTemp = np.zeros(self._mNVariables, dtype=np.float64)
        mYErr: F64Array = self._mYErr
        mQcYTemp: F64Array = self._mQcYTemp
        errmax: float
        h: float = htry
        while True:
            self._baseStep(x, y, dydx, h, mQcYTemp, mYErr)
            errmax = 0.0
            for i in range(self._mNVariables):
                errmax = max(errmax, abs(mYErr[i] / yscal[i]))
            errmax /= eps
            if errmax > 1.0:
                htemp: float = safety * h * math.pow(errmax, pshrnk)
                h = max(htemp, 0.1 * h) if h >= 0 else min(htemp, 0.1 * h)
                # Check for step size underflow
                xnew: float = x + h
                if xnew == x:
                    raise UtilException("Step size underflow in AdaptiveRungeKutta.qcStep.")
            if not (errmax > 1.0):
                break
        self._mHNext = safety * h * math.pow(errmax, pgrow) if errmax > errcon else 5.0 * h
        self._mHDid = h
        x += h
        y[: self._mNVariables] = mQcYTemp[: self._mNVariables]
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
    # public API  (Java `public` -> unprefixed, R1)
    # ==================================================================
    def setSaveInterval(self, interval: float) -> None:
        """setSaveInterval - Set the interval on which to save intermediate points on
        the integrated trajectory. (Use clearSaveInterval to not save any
        intermediate points.) Note: The default is not to save any intermediate
        points.

        @param interval double
        """
        self._mSaveInterval = math.fabs(interval)

    def clearSaveInterval(self) -> None:
        """clearSaveInterval - Return to the default of not saving any intermediate
        points.
        """
        self._mSaveInterval = sys_float_max()

    def getNSaved(self) -> int:
        """getNSaved - Returns the number of saved values.

        @return int
        """
        return self._mNSaved

    def getX(self, i: int) -> float:
        """getX - Returns the x-coordinate of the i-th saved value

        @param i int - Where i<getNSaved()
        @return double
        """
        assert self._mXSave is not None
        return float(self._mXSave[i])

    def getY(self, i: int) -> F64Array:
        """getY - returns the getNVariable x y-coordinates of the i-th saved values.

        @param i int - Where i<getNSaved()
        @return double[] - Of dimension getNVariables
        """
        assert self._mYSave is not None
        return self._mYSave[i]

    def setMaxSteps(self, maxSteps: int) -> None:
        """setMaxSteps - Set the maximum number of ODE steps to allow. Default is
        10000.

        @param maxSteps int
        """
        self._mMaxSteps = maxSteps

    def setMinStepSize(self, minStep: float) -> None:
        """setMinStepSize - Sets the minimum permissible step size. Default is 0.0.

        @param minStep double
        """
        self._mMinStepSize = math.fabs(minStep)

    def getNVariables(self) -> int:
        """getNVariables - Returns the number of variables as set in the constructor.

        @return int
        """
        return self._mNVariables

    def getStepCount(self) -> int:
        """getStepCount - Get the total number of steps required to perform the
        previous integrate operation.

        @return int
        """
        return self._mNOk + self._mNBad

    def getGoodStepCount(self) -> int:
        """getGoodStepCount - Get the number of steps leading to results of the
        desired accuracy.

        @return int
        """
        return self._mNOk

    def getBadStepCount(self) -> int:
        """getBadStepCount - Get the number of steps that were needed to be
        subdivided to attain results of the desired accuracy.

        @return int
        """
        return self._mNBad

    # -- R2: public mathematical method -> primary + _literal companion. --
    # SCIPY-DEV-1 (see module docstring CHANGES): no scipy substitution is
    # provided; `integrate` and `integrate_literal` both delegate to the
    # single faithful Cash-Karp implementation so the stateful getters
    # above remain correct.
    def integrate(self, x1: float, x2: float, ystart: F64Array, eps: float, h1: float) -> F64Array:
        """integrate - Integrate the ODE specified by derivatives using the adaptive
        step size Runge-Kutta algorithm over the independent variable interval x1
        to x2. ystart contains the initial y values. eps is measure of the
        permissible error. h is the initial step size.

        @param x1 double - Start of the integration range
        @param x2 double - End of the integration range
        @param ystart double[] - (In & out) The initial y value
        @param eps double - The permissible relative error
        @param h1 double - The initial step size
        @return The final y values as an array of length getNVariables().
        @throws UtilException - Upon too many steps or too small a step
        """
        return self._integrate_impl(x1, x2, ystart, eps, h1)

    def integrate_literal(self, x1: float, x2: float, ystart: F64Array, eps: float, h1: float) -> F64Array:
        """Literal-port alias of `integrate` (see SCIPY-DEV-1 in the module docstring)."""
        return self._integrate_impl(x1, x2, ystart, eps, h1)

    def _integrate_impl(self, x1: float, x2: float, ystart: F64Array, eps: float, h1: float) -> F64Array:
        tiny: float = 1.0e-10 * eps
        yscal: F64Array = np.zeros(self._mNVariables, dtype=np.float64)
        dydx: F64Array = np.zeros(self._mNVariables, dtype=np.float64)
        y: F64Array = np.zeros(self._mNVariables, dtype=np.float64)
        x: float = x1
        h: float = self._sign(h1, x2 - x1)
        xsav: float = 0.0
        saveInt: float = sys_float_max()
        kMax: int = 0
        self._mNSaved = 0
        self._mNOk = 0
        self._mNBad = 0
        y[: self._mNVariables] = ystart[: self._mNVariables]
        if self._mSaveInterval != sys_float_max():
            kMax = int(math.floor((abs(x2 - x1) + self._mSaveInterval) / self._mSaveInterval + 0.5))
            saveInt = self._sign(self._mSaveInterval, x2 - x1)
            xsav = x - (2.0 * saveInt)  # to ensure that the first step is saved...
            self._mXSave = np.zeros(kMax, dtype=np.float64)
            self._mYSave = np.zeros((kMax, self._mNVariables), dtype=np.float64)
        for step in range(self._mMaxSteps):
            # Save the necessary points
            if (kMax != 0) and (self._mNSaved < kMax) and (abs(x - xsav) >= (0.9999 * self._mSaveInterval)):
                assert self._mXSave is not None and self._mYSave is not None
                self._mXSave[self._mNSaved] = x
                self._mYSave[self._mNSaved, : self._mNVariables] = y[: self._mNVariables]
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
                ystart[: self._mNVariables] = y[: self._mNVariables]
                if kMax != 0:
                    assert self._mXSave is not None and self._mYSave is not None
                    self._mNSaved = min(self._mNSaved, kMax - 1)
                    self._mXSave[self._mNSaved] = x
                    self._mYSave[self._mNSaved, : self._mNVariables] = y[: self._mNVariables]
                    self._mNSaved += 1
                self._clearWorkspace()
                return y
            if abs(self._mHNext) <= self._mMinStepSize:
                raise UtilException("Step size too small in AdaptiveRungeKutta.integrate")
            h = self._mHNext
        raise UtilException("Too many steps in AdaptiveRungeKutta.integrate")

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
        raise NotImplementedError


def sys_float_max() -> float:
    """Java's Double.MAX_VALUE -- used as the default/"unset" sentinel for
    mSaveInterval (see the constructor and clearSaveInterval)."""
    import sys as _sys

    return _sys.float_info.max