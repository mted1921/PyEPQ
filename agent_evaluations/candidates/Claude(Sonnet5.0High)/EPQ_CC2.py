r"""
AdaptiveRungeKutta_ver1_1_0.py — Python port of gov.nist.microanalysis.Utility.AdaptiveRungeKutta

Guide version : 1
Generation    : 1
Port-code fixes: 0

CHANGES / DEVIATIONS (R10)
---------------------------
* R3 / Cross-Cutting Patterns ("Exceptions"): Java `UtilException` is mapped
  to `EPQException` imported from `_epq_compat`, matching the convention
  already established in Math2_ver8_1_5.py (the mandated style reference),
  which raises `EPQException` rather than defining or importing a separate
  `UtilException` type. No local exception type is defined (R3).
* R2: `integrate` is the one public "mathematical" method on this class.
  No scipy/numpy library-substituted primary is provided for it: `integrate()`
  and `integrate_literal()` both delegate to the same faithful, line-for-line
  translation of the Java algorithm (`_integrate_impl`). Rationale: this
  class's public API (`getNSaved`, `getX`, `getY`, `getStepCount`,
  `getGoodStepCount`, `getBadStepCount`, and the `mXSave`/`mYSave`
  save-interval bookkeeping) is defined entirely in terms of *this exact*
  Cash-Karp adaptive-step algorithm's internal accept/reject decisions and
  save points. Substituting e.g. `scipy.integrate.solve_ivp` (which uses a
  Dormand-Prince RK45 pair, not Cash-Karp) would silently change step
  counts, accept/reject bookkeeping, and saved trajectory points relative
  to Java, breaking the tested public contract rather than merely trading
  numerical noise (see CONVERSION_GUIDE.md Appendix A, M1). No natural,
  contract-preserving substitution exists, so both public names resolve to
  the faithful port: R2(a) ("no library substitutions, no bug fixes") is
  honored, and R2(b) (completeness / both names present) is satisfied.
* Class is `abstract` in Java (`abstract public class AdaptiveRungeKutta`)
  with one `abstract public` method, `derivatives`. Per R1, `abstract` is
  NOT a privacy modifier: the method is exposed as `derivatives()`, never
  `_derivatives()`. The class subclasses `abc.ABC` and `derivatives` is
  decorated `@abc.abstractmethod` (M4: parity for this class must be tested
  via concrete Python subclasses, since JPype cannot extend a Java abstract
  class from Python).

BUG_LEDGER: empty (see BUG_LEDGER below, R6). No dead branches, always-true
conditions, off-by-one errors, or sign errors were identified in the Java
source: this is a faithful transcription of the standard Numerical Recipes
Cash-Karp adaptive-step-size driver (baseStep == rkck, qcStep == rkqs,
integrate == odeint), and the embedded RK coefficients, error-control
constants (safety, pgrow, pshrnk, errcon), and save-interval bookkeeping all
match the reference algorithm.

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
from typing import Optional, Sequence, Union, Callable

import numpy as np

try:
    from ._epq_compat import EPQException, F64Array, JavaRandom
except ImportError:
    try:
        from _epq_compat import EPQException, F64Array, JavaRandom  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility._epq_compat import EPQException, F64Array, JavaRandom  # type: ignore


__all__ = ["AdaptiveRungeKutta", "EPQException", "JavaRandom", "F64Array"]


class AdaptiveRungeKutta(abc.ABC):
    """Python port of gov.nist.microanalysis.Utility.AdaptiveRungeKutta.

    Adaptive step-size Cash-Karp Runge-Kutta ODE integrator (Numerical
    Recipes in C, 2nd ed., pp 714-722). Subclasses implement `derivatives`.

    NOTE: matches the Java class -- instances are stateful and NOT
    thread-safe. Use each instance from a single thread only.
    """

    # ==================================================================
    # Preserved-bug ledger (machine-readable) -- R6
    # ==================================================================
    BUG_LEDGER: tuple = ()  # no bugs identified

    def __init__(self, nVars: int) -> None:
        """Java: `public AdaptiveRungeKutta(int nVars)`.

        Construct an AdaptiveRungeKutta object to solve a differential
        equation of nVars variables. The implementation of `derivatives`
        should return nVars derivative values for each x & y.

        Args:
            nVars: Number of differential equations / dependent variables.
        """
        self.mNVariables: int = nVars  # final in Java
        self.mHDid: float = 0.0  # Actual step size accomplished in last call to qcStep
        self.mHNext: float = 0.0  # Next step size to try when calling qcStep
        self.mSaveInterval: float = sys.float_info.max  # Java: Double.MAX_VALUE
        self.mMinStepSize: float = 0.0
        self.mXSave: Optional[F64Array] = None
        self.mYSave: Optional[F64Array] = None
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
    # Internal guards -- R5
    # ==================================================================
    @staticmethod
    def _require_mutable_f64(arr, name: str = "arr") -> None:
        """Type guard for in-place methods (CONVERSION_GUIDE R5).

        Java's ``double[]`` is always a mutable double-precision buffer.
        Our in-place helpers can only honour that contract on numpy
        ndarrays with ``dtype=float64``. Fail loud rather than silently
        no-op or copy.
        """
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{name} must be a numpy ndarray")
        if arr.dtype != np.float64:
            raise TypeError(f"{name} must have dtype float64")
        if not arr.flags.writeable:
            raise TypeError(f"{name} must be writeable")

    # ==================================================================
    # Private helpers (Java: private)
    # ==================================================================
    def _sign(self, magnitude: float, sign: float) -> float:
        """Java: `private double sign(double magnitude, double sign)`.

        Fortran-style SIGN transfer function -- distinct from Math.signum.
        Returns |magnitude| if sign >= 0.0, else -|magnitude|.
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
        """Java: `private void baseStep(double x, double[] y, double[] dydx,
        double h, double[] yout, double[] yerr)`.

        Take a single Cash-Karp Runge-Kutta step. Given the n=mNVariables
        values y[0..n-1] and their derivatives dydx[0..n-1] known at x, use
        a fifth order Cash-Karp Runge-Kutta method to advance the solution
        over an interval h. The resulting y value is written into yout. An
        estimate of the truncation error is written into yerr.
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
            self.mYTemp[i] = y[i] + (
                h * ((b51 * dydx[i]) + (b52 * self.mWs2[i]) + (b53 * self.mWs3[i]) + (b54 * self.mWs4[i]))
            )
        # Fifth step
        self.derivatives(x + (a5 * h), self.mYTemp, self.mWs5)
        for i in range(self.mNVariables):
            self.mYTemp[i] = y[i] + (
                h
                * (
                    (b61 * dydx[i])
                    + (b62 * self.mWs2[i])
                    + (b63 * self.mWs3[i])
                    + (b64 * self.mWs4[i])
                    + (b65 * self.mWs5[i])
                )
            )
        # Sixth step
        self.derivatives(x + (a6 * h), self.mYTemp, self.mWs6)
        for i in range(self.mNVariables):
            yout[i] = y[i] + (h * ((c1 * dydx[i]) + (c3 * self.mWs3[i]) + (c4 * self.mWs4[i]) + (c6 * self.mWs6[i])))
        # Estimate the error
        for i in range(self.mNVariables):
            yerr[i] = h * (
                (dc1 * dydx[i]) + (dc3 * self.mWs3[i]) + (dc4 * self.mWs4[i]) + (dc5 * self.mWs5[i]) + (dc6 * self.mWs6[i])
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
        """Java: `private double qcStep(double x, double[] y, double[] dydx,
        double htry, double eps, double[] yscal) throws UtilException`.

        Take a fifth order Runge-Kutta step with monitoring of local
        truncation error. Input are the dependent variable y[0..n-1] and
        its derivatives dydx[0..n-1] at the starting value of the
        independent variable x. Also input is the attempted step size
        htry, the required accuracy eps, and the vector yscal against
        which the errors are scaled. Upon return, y is replaced with the
        new values, x is returned, and self.mHDid / self.mHNext are set to
        the actual step size and the size of the next step to try.

        Raises:
            EPQException: when the step size becomes too small (Java:
                UtilException; see module CHANGES for the R3 mapping).
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
                    # JAVA: throw new UtilException(...) -> R3: mapped to EPQException
                    raise EPQException("Step size underflow in AdaptiveRungeKutta.qcStep.")
            else:
                break
        self.mHNext = (safety * h * math.pow(errmax, pgrow)) if (errmax > errcon) else (5.0 * h)
        self.mHDid = h
        x += h
        y[: self.mNVariables] = self.mQcYTemp[: self.mNVariables]
        return x

    def _clearWorkspace(self) -> None:
        """Java: `private void clearWorkspace()` -- null all temporary
        space to free memory."""
        self.mWs2 = None
        self.mWs3 = None
        self.mWs4 = None
        self.mWs5 = None
        self.mWs6 = None
        self.mYTemp = None
        self.mYErr = None
        self.mQcYTemp = None

    # ==================================================================
    # Public accessors (Java: public)
    # ==================================================================
    def setSaveInterval(self, interval: float) -> None:
        """Java: `public void setSaveInterval(double interval)`.

        Set the interval on which to save intermediate points on the
        integrated trajectory. (Use clearSaveInterval to not save any
        intermediate points.) Note: the default is not to save any
        intermediate points.
        """
        self.mSaveInterval = abs(interval)

    def clearSaveInterval(self) -> None:
        """Java: `public void clearSaveInterval()`.

        Return to the default of not saving any intermediate points.
        """
        self.mSaveInterval = sys.float_info.max

    def getNSaved(self) -> int:
        """Java: `public int getNSaved()`. Returns the number of saved values."""
        return self.mNSaved

    def getX(self, i: int) -> float:
        """Java: `public double getX(int i)`. Returns the x-coordinate of
        the i-th saved value. Requires i < getNSaved()."""
        return float(self.mXSave[i])

    def getY(self, i: int) -> F64Array:
        """Java: `public double[] getY(int i)`. Returns the getNVariables()
        y-coordinates of the i-th saved value. Requires i < getNSaved().

        Returns the internal row directly (a live view), matching Java's
        array-reference return semantics: mutating the result mutates
        this object's internal state.
        """
        return self.mYSave[i]

    def setMaxSteps(self, maxSteps: int) -> None:
        """Java: `public void setMaxSteps(int maxSteps)`. Set the maximum
        number of ODE steps to allow. Default is 10000."""
        self.mMaxSteps = maxSteps

    def setMinStepSize(self, minStep: float) -> None:
        """Java: `public void setMinStepSize(double minStep)`. Sets the
        minimum permissible step size. Default is 0.0."""
        self.mMinStepSize = abs(minStep)

    def getNVariables(self) -> int:
        """Java: `public int getNVariables()`. Returns the number of
        variables as set in the constructor."""
        return self.mNVariables

    def getStepCount(self) -> int:
        """Java: `public int getStepCount()`. Total number of steps
        required to perform the previous integrate operation."""
        return self.mNOk + self.mNBad

    def getGoodStepCount(self) -> int:
        """Java: `public int getGoodStepCount()`. Number of steps leading
        to results of the desired accuracy."""
        return self.mNOk

    def getBadStepCount(self) -> int:
        """Java: `public int getBadStepCount()`. Number of steps that
        needed to be subdivided to attain results of the desired
        accuracy."""
        return self.mNBad

    # ==================================================================
    # integrate -- faithful port; see module CHANGES for the R2 note on
    # why integrate() and integrate_literal() share one implementation.
    # ==================================================================
    def _integrate_impl(self, x1: float, x2: float, ystart: F64Array, eps: float, h1: float) -> F64Array:
        """Shared faithful implementation backing both `integrate` and
        `integrate_literal`.

        Java: `public double[] integrate(double x1, double x2,
        double[] ystart, double eps, double h1) throws UtilException`.

        Integrate the ODE specified by `derivatives` using the adaptive
        step size Runge-Kutta algorithm over the independent variable
        interval x1 to x2. ystart contains the initial y values. eps is a
        measure of the permissible error. h1 is the initial step size.

        Mutates `ystart` in place (Java: "In & out" parameter) and returns
        a fresh array with the final y values (length getNVariables()).

        Raises:
            EPQException: upon too many steps or too small a step (Java:
                UtilException; see module CHANGES for the R3 mapping).
        """
        self._require_mutable_f64(ystart, "ystart")  # R5
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
        y[: self.mNVariables] = ystart[: self.mNVariables]
        if self.mSaveInterval != sys.float_info.max:
            # JAVA-R7: Math.round(x) -> int(math.floor(x + 0.5))
            kMax = int(math.floor(((abs(x2 - x1) + self.mSaveInterval) / self.mSaveInterval) + 0.5))
            saveInt = self._sign(self.mSaveInterval, x2 - x1)
            xsav = x - (2.0 * saveInt)  # ensures the first step is saved...
            self.mXSave = np.zeros(kMax, dtype=np.float64)
            self.mYSave = np.zeros((kMax, self.mNVariables), dtype=np.float64)

        for step in range(self.mMaxSteps):
            # Save the necessary points
            if (kMax != 0) and (self.mNSaved < kMax) and (abs(x - xsav) >= (0.9999 * self.mSaveInterval)):
                self.mXSave[self.mNSaved] = x
                self.mYSave[self.mNSaved][: self.mNVariables] = y[: self.mNVariables]
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
            if ((((x + h) - x2) * ((x + h) - x1)) > 0.0):
                h = x2 - x
            x = self._qcStep(x, y, dydx, h, eps, yscal)
            if self.mHDid == h:
                self.mNOk += 1
            else:
                self.mNBad += 1
            if (((x - x2) * (x2 - x1)) >= 0.0):
                ystart[: self.mNVariables] = y[: self.mNVariables]
                if kMax != 0:
                    self.mNSaved = min(self.mNSaved, kMax - 1)
                    self.mXSave[self.mNSaved] = x
                    self.mYSave[self.mNSaved][: self.mNVariables] = y[: self.mNVariables]
                    self.mNSaved += 1
                self._clearWorkspace()
                return y
            if abs(self.mHNext) <= self.mMinStepSize:
                # JAVA: throw new UtilException(...) -> R3: mapped to EPQException
                raise EPQException("Step size too small in AdaptiveRungeKutta.integrate")
            h = self.mHNext
        # JAVA: throw new UtilException(...) -> R3: mapped to EPQException
        raise EPQException("Too many steps in AdaptiveRungeKutta.integrate")

    def integrate(self, x1: float, x2: float, ystart: F64Array, eps: float, h1: float) -> F64Array:
        """Primary public entry point (R2). Identical to `integrate_literal`
        -- see module CHANGES for why no scipy-substituted primary is
        provided for this stateful, bookkeeping-heavy method."""
        return self._integrate_impl(x1, x2, ystart, eps, h1)

    def integrate_literal(self, x1: float, x2: float, ystart: F64Array, eps: float, h1: float) -> F64Array:
        """R2 companion name: line-for-line Java translation. Identical to
        `integrate` -- see module CHANGES."""
        return self._integrate_impl(x1, x2, ystart, eps, h1)

    # ==================================================================
    # Abstract method -- R1: public abstract -> NO underscore prefix
    # ==================================================================
    @abc.abstractmethod
    def derivatives(self, x: float, y: F64Array, dydx: F64Array) -> None:
        """Java: `abstract public void derivatives(double x, double[] y,
        double[] dydx)`.

        The derived class provides an implementation of the derivatives
        function. x & y are input and the subclass's implementation is
        responsible for writing the derivatives into dydx. The lengths of
        y and dydx are equal to getNVariables().
        """
        ...