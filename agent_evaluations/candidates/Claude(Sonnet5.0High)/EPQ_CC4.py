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
* R1 — `derivatives` is declared `abstract public` in Java, so the Python
  name stays un-prefixed (`derivatives`, never `_derivatives`); every other
  member is Java-`private` and is therefore `_`-prefixed (`_mNVariables`,
  `_sign`, `_baseStep`, `_qcStep`, `_clearWorkspace`, ...).
* Exceptions — Java throws `UtilException`. Per CONVERSION_GUIDE.md
  Cross-Cutting Patterns ("Exceptions | Always EPQException from
  _epq_compat"), both raise sites here use `EPQException` instead of a
  separate `UtilException` port.
* No `foo()` / `foo_literal()` split (R2) was introduced for `integrate`,
  `_baseStep`, or `_qcStep`. These implement a specific, stateful Cash-Karp
  step-doubling algorithm with save-interval bookkeeping (`_mNSaved`,
  `_mNOk`, `_mNBad`, `getX`/`getY`); `scipy.integrate.solve_ivp` uses a
  different (Dormand-Prince) coefficient set and a different step-control
  algorithm, so it is not a "natural substitution" in the R2 sense — using
  it would silently break bit-for-bit reproducibility of published
  trajectories (see CONVERSION_GUIDE.md Appendix A, M1/H1). `integrate` is
  therefore kept as a single faithful (literal) implementation.
* `double[]` / `double[][]` become `F64Array` (numpy float64) buffers.
  Java's `System.arraycopy(src, 0, dst, 0, mNVariables)` calls are ported
  as explicit `dst[:n] = src[:n]` slices so that only the first
  `mNVariables` elements are ever touched, matching Java's explicit-length
  arraycopy semantics regardless of the caller's actual array length.
* R5 — `integrate()` mutates its `ystart` argument in place (mirroring
  Java's `System.arraycopy(y, 0, ystart, 0, mNVariables)` at return), so it
  is guarded with `_require_mutable_f64`.
* `Double.MAX_VALUE` is ported as `sys.float_info.max` (bit-identical IEEE
  754 double). The Java code relies on exact equality against this
  sentinel (`mSaveInterval != Double.MAX_VALUE`) to detect "no save
  interval configured"; that exact-equality comparison is preserved
  verbatim, it is not a bug.
* `equals()` / `hashCode()` / `toString()` are not ported: the Java class
  does not override any of them (default `Object` identity semantics), so
  R2's dunder/alias mapping does not apply here.
* No intra-`Utility` sibling imports were required: the Java source makes
  no calls into `Math2`, `FindRoot`, or any other `gov.nist.microanalysis
  .Utility` class, consistent with UTILITY_LEDGER.md's Tier 1 entry for
  `AdaptiveRungeKutta` ("Unresolved deps: —").
* M4 (CONVERSION_GUIDE.md Appendix A): this is an abstract class — JPype
  cannot extend a Java abstract class from Python, so direct Java/Python
  parity testing of `AdaptiveRungeKutta` itself is not possible; test via
  concrete Python subclasses against known analytic ODE solutions instead.

BUG_LEDGER
----------
BUG_LEDGER: tuple = ()  # no bugs identified — see class body for the
authoritative (machine-readable) empty tuple.
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
# JavaRandom is imported per the project-wide standard import header; this
# class is a deterministic ODE integrator and does not use RNG directly.


__all__ = ["AdaptiveRungeKutta", "EPQException", "JavaRandom", "F64Array"]


class AdaptiveRungeKutta(abc.ABC):
    """Adaptive step-size Cash-Karp Runge-Kutta ODE integrator.

    See the module docstring above for the full, verbatim Java Javadoc.
    Not thread-safe (per the original Java documentation) — use each
    instance from a single thread only.
    """

    # ==================================================================
    # Preserved-bug ledger (machine-readable; CONVERSION_GUIDE.md R6)
    # ==================================================================
    # No bugs (dead branches, always-true conditions, off-by-one, sign
    # errors) were identified in the Java source during this port.
    BUG_LEDGER: tuple = ()  # no bugs identified

    # ==================================================================
    # Internal guards (CONVERSION_GUIDE.md R5)
    # ==================================================================
    @staticmethod
    def _require_mutable_f64(arr: F64Array, name: str = "arr") -> None:
        """Type guard for in-place methods (CONVERSION_GUIDE.md R5).

        Mirrors ``Math2._require_mutable_f64``: Java's ``double[]`` is
        always a mutable double-precision buffer; our in-place helpers can
        only honour that contract on numpy ndarrays with
        ``dtype=float64``. Lists, tuples, and wrong-dtype arrays would
        silently no-op or copy. Fail loud instead.
        """
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{name} must be a numpy ndarray")
        if arr.dtype != np.float64:
            raise TypeError(f"{name} must have dtype float64")
        if not arr.flags.writeable:
            raise TypeError(f"{name} must be writeable")

    # ==================================================================
    # Construction
    # ==================================================================
    def __init__(self, nVars: int) -> None:
        """Java: ``public AdaptiveRungeKutta(int nVars)``.

        Construct an AdaptiveRungeKutta object to solve a differential
        equation of nVars variables. The subclass implementation of
        ``derivatives`` should return nVars derivative values for each x
        and y.

        Parameters
        ----------
        nVars : int
            The number of dependent variables / differential equations.
        """
        self._mNVariables: int = nVars  # Java: private final int mNVariables
        self._mHDid: float = 0.0  # Actual step size accomplished in last call to _qcStep
        self._mHNext: float = 0.0  # Next step size to try when calling _qcStep
        self._mSaveInterval: float = sys.float_info.max  # Java: Double.MAX_VALUE
        self._mMinStepSize: float = 0.0
        self._mXSave: Optional[F64Array] = None
        self._mYSave: Optional[F64Array] = None  # 2-D: (kMax, mNVariables)
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
    # Private helpers
    # ==================================================================
    def _sign(self, magnitude: float, sign: float) -> float:
        """Java: ``private double sign(double magnitude, double sign)``.

        Numerical-Recipes-style SIGN(a, b): returns |magnitude| with the
        sign of `sign`.
        """
        return abs(magnitude) if sign >= 0.0 else -abs(magnitude)

    def _baseStep(self, x: float, y: F64Array, dydx: F64Array, h: float,
                  yout: F64Array, yerr: F64Array) -> None:
        """Java: ``private void baseStep(double x, double[] y, double[]
        dydx, double h, double[] yout, double[] yerr)``.

        Take a single Cash-Karp Runge-Kutta step. Given the n=mNVariables
        values y[0..n-1] and their derivatives dydx[0..n-1] known at x,
        use a fifth order Cash-Karp Runge-Kutta method to advance the
        solution over an interval h. The resulting y value is written
        in-place into `yout`. An estimate of the truncation error is
        written in-place into `yerr`.
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

        n: int = self._mNVariables

        # Workspace — lazily allocated on first call, then reused
        # (matches Java's `if (mWs2 == null) { ... }` allocate-once guard).
        if self._mWs2 is None:
            self._mWs2 = np.zeros(n, dtype=np.float64)
            self._mWs3 = np.zeros(n, dtype=np.float64)
            self._mWs4 = np.zeros(n, dtype=np.float64)
            self._mWs5 = np.zeros(n, dtype=np.float64)
            self._mWs6 = np.zeros(n, dtype=np.float64)
            self._mYTemp = np.zeros(n, dtype=np.float64)

        # First step
        for i in range(n):
            self._mYTemp[i] = y[i] + (b21 * h * dydx[i])
        # Second step
        self.derivatives(x + (a2 * h), self._mYTemp, self._mWs2)
        for i in range(n):
            self._mYTemp[i] = y[i] + (h * ((b31 * dydx[i]) + (b32 * self._mWs2[i])))
        # Third step
        self.derivatives(x + (a3 * h), self._mYTemp, self._mWs3)
        for i in range(n):
            self._mYTemp[i] = y[i] + (h * ((b41 * dydx[i]) + (b42 * self._mWs2[i]) + (b43 * self._mWs3[i])))
        # Fourth step
        self.derivatives(x + (a4 * h), self._mYTemp, self._mWs4)
        for i in range(n):
            self._mYTemp[i] = y[i] + (h * ((b51 * dydx[i]) + (b52 * self._mWs2[i]) + (b53 * self._mWs3[i]) + (b54 * self._mWs4[i])))
        # Fifth step
        self.derivatives(x + (a5 * h), self._mYTemp, self._mWs5)
        for i in range(n):
            self._mYTemp[i] = y[i] + (h * ((b61 * dydx[i]) + (b62 * self._mWs2[i]) + (b63 * self._mWs3[i]) + (b64 * self._mWs4[i]) + (b65 * self._mWs5[i])))
        # Sixth step
        self.derivatives(x + (a6 * h), self._mYTemp, self._mWs6)
        for i in range(n):
            yout[i] = y[i] + (h * ((c1 * dydx[i]) + (c3 * self._mWs3[i]) + (c4 * self._mWs4[i]) + (c6 * self._mWs6[i])))
        # Estimate the error
        for i in range(n):
            yerr[i] = h * ((dc1 * dydx[i]) + (dc3 * self._mWs3[i]) + (dc4 * self._mWs4[i]) + (dc5 * self._mWs5[i]) + (dc6 * self._mWs6[i]))

    def _qcStep(self, x: float, y: F64Array, dydx: F64Array, htry: float,
                eps: float, yscal: F64Array) -> float:
        """Java: ``private double qcStep(double x, double[] y, double[]
        dydx, double htry, double eps, double[] yscal) throws
        UtilException``.

        Take a fifth order Runge-Kutta step with monitoring of local
        truncation error. `y` is replaced in-place with the new values;
        `self._mHDid` / `self._mHNext` are set as side effects to the
        actual step size taken and the size of the next step to try.

        Raises
        ------
        EPQException
            When the step size underflows (ported from Java's
            ``UtilException`` — see module CHANGES).
        """
        safety: float = 0.9
        pgrow: float = -0.2
        pshrnk: float = -0.25
        errcon: float = 1.89e-4

        n: int = self._mNVariables
        if self._mYErr is None:
            self._mYErr = np.zeros(n, dtype=np.float64)
            self._mQcYTemp = np.zeros(n, dtype=np.float64)

        errmax: float
        h: float = htry
        # Java `do { ... } while (errmax > 1.0);` ported as while-True/break.
        while True:
            self._baseStep(x, y, dydx, h, self._mQcYTemp, self._mYErr)
            errmax = 0.0
            for i in range(n):
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
        y[:n] = self._mQcYTemp[:n]
        return x

    def _clearWorkspace(self) -> None:
        """Java: ``private void clearWorkspace()`` — null all temporary
        space to free memory."""
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
        """Java: ``public void setSaveInterval(double interval)``.

        Set the interval on which to save intermediate points on the
        integrated trajectory. (Use ``clearSaveInterval`` to not save any
        intermediate points.) Note: the default is not to save any
        intermediate points.
        """
        self._mSaveInterval = abs(interval)

    def clearSaveInterval(self) -> None:
        """Java: ``public void clearSaveInterval()``.

        Return to the default of not saving any intermediate points.
        """
        self._mSaveInterval = sys.float_info.max  # Java: Double.MAX_VALUE

    def getNSaved(self) -> int:
        """Java: ``public int getNSaved()``. Returns the number of saved
        values."""
        return self._mNSaved

    def getX(self, i: int) -> float:
        """Java: ``public double getX(int i)``. Returns the x-coordinate
        of the i-th saved value, where ``i < getNSaved()``.

        As in Java (unchecked array access), an out-of-range or
        never-populated `i` raises naturally (``IndexError`` /
        ``TypeError`` here in place of Java's
        ``ArrayIndexOutOfBoundsException`` / ``NullPointerException``).
        """
        return float(self._mXSave[i])

    def getY(self, i: int) -> F64Array:
        """Java: ``public double[] getY(int i)``. Returns the
        getNVariables() y-coordinates of the i-th saved value, where
        ``i < getNSaved()``."""
        return self._mYSave[i]

    def setMaxSteps(self, maxSteps: int) -> None:
        """Java: ``public void setMaxSteps(int maxSteps)``. Set the
        maximum number of ODE steps to allow. Default is 10000."""
        self._mMaxSteps = maxSteps

    def setMinStepSize(self, minStep: float) -> None:
        """Java: ``public void setMinStepSize(double minStep)``. Sets the
        minimum permissible step size. Default is 0.0."""
        self._mMinStepSize = abs(minStep)

    def getNVariables(self) -> int:
        """Java: ``public int getNVariables()``. Returns the number of
        variables as set in the constructor."""
        return self._mNVariables

    def getStepCount(self) -> int:
        """Java: ``public int getStepCount()``. Get the total number of
        steps required to perform the previous integrate operation."""
        return self._mNOk + self._mNBad

    def getGoodStepCount(self) -> int:
        """Java: ``public int getGoodStepCount()``. Get the number of
        steps leading to results of the desired accuracy."""
        return self._mNOk

    def getBadStepCount(self) -> int:
        """Java: ``public int getBadStepCount()``. Get the number of
        steps that needed to be subdivided to attain results of the
        desired accuracy."""
        return self._mNBad

    def integrate(self, x1: float, x2: float, ystart: F64Array, eps: float,
                  h1: float) -> F64Array:
        """Java: ``public double[] integrate(double x1, double x2,
        double[] ystart, double eps, double h1) throws UtilException``.

        Integrate the ODE specified by ``derivatives`` using the adaptive
        step size Runge-Kutta algorithm over the independent variable
        interval x1 to x2. `ystart` contains the initial y values and is
        also updated in place (mirroring Java's array-by-reference
        semantics). `eps` is a measure of the permissible error. `h1` is
        the initial step size.

        Returns
        -------
        F64Array
            The final y values, of length ``getNVariables()``.

        Raises
        ------
        EPQException
            Upon too many steps or too small a step (ported from Java's
            ``UtilException`` — see module CHANGES).
        """
        AdaptiveRungeKutta._require_mutable_f64(ystart, "ystart")  # R5
        n: int = self._mNVariables
        tiny: float = 1.0e-10 * eps
        yscal: F64Array = np.zeros(n, dtype=np.float64)
        dydx: F64Array = np.zeros(n, dtype=np.float64)
        y: F64Array = np.zeros(n, dtype=np.float64)
        x: float = x1
        h: float = self._sign(h1, x2 - x1)
        xsav: float = 0.0
        saveInt: float = sys.float_info.max
        kMax: int = 0
        self._mNSaved = 0
        self._mNOk = 0
        self._mNBad = 0
        y[:n] = ystart[:n]

        if self._mSaveInterval != sys.float_info.max:
            # R7: Math.round(x) -> int(math.floor(x + 0.5))
            kMax = int(math.floor(((abs(x2 - x1) + self._mSaveInterval) / self._mSaveInterval) + 0.5))
            saveInt = self._sign(self._mSaveInterval, x2 - x1)
            xsav = x - (2.0 * saveInt)  # to ensure that the first step is saved...
            self._mXSave = np.zeros(kMax, dtype=np.float64)
            self._mYSave = np.zeros((kMax, n), dtype=np.float64)

        for step in range(self._mMaxSteps):
            # Save the necessary points
            if (kMax != 0) and (self._mNSaved < kMax) and (abs(x - xsav) >= (0.9999 * self._mSaveInterval)):
                self._mXSave[self._mNSaved] = x
                self._mYSave[self._mNSaved, :n] = y[:n]
                xsav = x
                self._mNSaved += 1
            self.derivatives(x, y, dydx)
            # Rescale h to ensure we hit desired points
            hMax: float = abs((xsav + saveInt) - x)
            if abs(h) > hMax:
                h = self._sign(hMax, h)
            # Scaling to monitor accuracy...
            for i in range(n):
                yscal[i] = abs(y[i]) + abs(dydx[i] * h) + tiny
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
                    self._mYSave[self._mNSaved, :n] = y[:n]
                    self._mNSaved += 1
                self._clearWorkspace()
                return y
            if abs(self._mHNext) <= self._mMinStepSize:
                raise EPQException("Step size too small in AdaptiveRungeKutta.integrate")
            h = self._mHNext

        raise EPQException("Too many steps in AdaptiveRungeKutta.integrate")

    # ==================================================================
    # Abstract API
    # ==================================================================
    @abc.abstractmethod
    def derivatives(self, x: float, y: F64Array, dydx: F64Array) -> None:
        """Java: ``abstract public void derivatives(double x, double[] y,
        double[] dydx)``.

        The derived class provides an implementation of the derivatives
        function. `x` and `y` are input; the subclass implementation is
        responsible for writing the derivatives in-place into `dydx`. The
        lengths of `y` and `dydx` are equal to ``getNVariables()``.

        R1: ``abstract`` does NOT trigger a leading underscore for a
        `public abstract` Java method — this stays ``derivatives``, never
        ``_derivatives``.
        """
        ...