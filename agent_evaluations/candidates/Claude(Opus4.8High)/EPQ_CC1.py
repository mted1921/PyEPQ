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

CHANGES IN THIS REVISION (ver1.1.0) — see CONVERSION_GUIDE.md R1-R10
-------------------------------------------------------------------
* R1: All Java identifiers preserved verbatim (mNVariables, mHDid, qcStep,
  baseStep, ...). Java `private` methods are `_`-prefixed (`_sign`,
  `_baseStep`, `_qcStep`, `_clearWorkspace`); the `public abstract` method
  keeps its bare Java name `derivatives` (NOT `_derivatives`) so concrete
  subclasses that implement `derivatives()` instantiate correctly.
* R2: `integrate` is the sole public *mathematical* method, so it is provided
  as a scipy-backed primary `integrate()` plus a line-for-line faithful
  `integrate_literal()`. The getters/setters (setSaveInterval, getNSaved, ...)
  are not mathematical and have a single faithful form. The Java class defines
  no equals/hashCode/toString overrides, so no dunder/alias pairs are required.
* R3: EPQException / F64Array imported from `_epq_compat` only (three-tier
  fallback). `UtilException` is imported from its sibling port (see NOTE).
* R5: `_require_mutable_f64` guards the one externally supplied in/out buffer,
  `ystart`, before results are written back into it.
* R7: Java `Math.round((|x2-x1| + mSaveInterval) / mSaveInterval)` ported as
  `int(math.floor(x + 0.5))`. The Java source contains no integer `/` divisions
  (every `/` operates on `double` literals), so no `//` substitutions arise.
* R8: The Java `sign(magnitude, sign)` helper is a Numerical-Recipes SIGN(a,b)
  function, NOT `Math.signum`, so the `signum(0) == 0` zero-case does not occur;
  `_sign` reproduces the original ternary exactly.
* R9: Every parameter, return type, field, and non-obvious local is annotated.
* R6 / R10: SCIPY-DEV-1 (the scipy substitution in `integrate`) is documented
  at its call site and below. No Java bugs were found — BUG_LEDGER is empty.

NOTE — exception type
---------------------
The Java source throws `gov.nist.microanalysis.Utility.UtilException`, which has
its own dedicated port (`UtilException_ver1_1_0.py`, per UTILITY_LEDGER.md). To
remain faithful, this port raises `UtilException` (imported via the standard
three-tier fallback), not a generic `EPQException`. The `_epq_compat` family
imports are retained per the standard port header.

NOTE — SCIPY-DEV-1 (deviation of integrate() from Java)
-------------------------------------------------------
`integrate()` delegates the stepping to `scipy.integrate.solve_ivp` with
`method="RK45"` (the Dormand-Prince embedded pair). The Java algorithm uses the
Cash-Karp embedded pair (RKCK). The two agree on the trajectory to within the
requested relative tolerance, but per-step bookkeeping is NOT reproducible from
scipy's public API: `getGoodStepCount()`/`getBadStepCount()` become best-effort
approximations (accepted-step count and 0 respectively), and the saved abscissae
are resampled at the requested interval from scipy's dense output rather than
falling exactly on the Java stepper's accepted abscissae. For bit-faithful
behaviour and exact step accounting, call `integrate_literal()`.
"""

from __future__ import annotations

import abc
import math
import sys

import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import Optional, Sequence, Union, Callable

from scipy.integrate import solve_ivp

# R3 — shared compat types (single source of truth). Three-tier fallback.
try:
    from ._epq_compat import EPQException, F64Array, JavaRandom
except ImportError:
    try:
        from _epq_compat import EPQException, F64Array, JavaRandom  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility._epq_compat import EPQException, F64Array, JavaRandom  # type: ignore

# Sibling port — filename taken from UTILITY_LEDGER.md "Port file" column.
# Three-tier fallback per the checklist (never define a local replacement).
try:
    from .UtilException_ver1_1_0 import UtilException
except ImportError:
    try:
        from UtilException_ver1_1_0 import UtilException  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility.UtilException_ver1_1_0 import UtilException  # type: ignore


__all__ = ["AdaptiveRungeKutta", "UtilException", "EPQException", "F64Array"]


# Java Double.MAX_VALUE == 1.7976931348623157e308 == sys.float_info.max
_DOUBLE_MAX_VALUE: float = sys.float_info.max


class AdaptiveRungeKutta(abc.ABC):

    # ==================================================================
    # Preserved-bug ledger (machine-readable) — see BUG_GUIDE.md / R6
    # ==================================================================
    # Each tuple: (id, method, description, has_strict_variant).
    BUG_LEDGER: tuple = ()  # no bugs identified — faithful Numerical Recipes
    # (Cash-Karp RKCK `baseStep` / `qcStep` and `odeint`-style `integrate`)
    # port. No dead branches, sign errors, off-by-one, or always-true
    # conditions were found in the Java source.

    # ==================================================================
    # Internal guards (CONVERSION_GUIDE R5)
    # ==================================================================
    @staticmethod
    def _require_mutable_f64(arr: object, name: str = "arr") -> None:
        """Type guard for in-place buffers (CONVERSION_GUIDE R5).

        Java's ``double[]`` is always a mutable double-precision buffer. Our
        in-place write-back can only honour that contract on numpy ndarrays
        with ``dtype=float64``. Lists, tuples, and wrong-dtype arrays would
        silently no-op or copy; fail loud instead.
        """
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{name} must be a numpy ndarray")
        if arr.dtype != np.float64:
            raise TypeError(f"{name} must have dtype float64")
        if not arr.flags.writeable:
            raise TypeError(f"{name} must be writeable")

    # ==================================================================
    # Constructor
    # ==================================================================
    def __init__(self, nVars: int) -> None:
        """AdaptiveRungeKutta - Construct an AdaptiveRungeKutta object to solve a
        differential equation of nVars variables. The implementation of
        derivatives should return nVars derivative values for each x & y.

        :param nVars: int
        """
        # Java: super(); mNVariables = nVars;  (mNVariables is `final`)
        self.mNVariables: int = nVars            # number of differential equations
        self.mHDid: float = 0.0                  # actual step size of last qcStep
        self.mHNext: float = 0.0                 # next step size to try in qcStep
        self.mSaveInterval: float = _DOUBLE_MAX_VALUE
        self.mMinStepSize: float = 0.0
        self.mXSave: Optional[F64Array] = None
        self.mYSave: Optional[NDArray[np.float64]] = None  # 2-D: [kMax][mNVariables]
        self.mNSaved: int = 0
        self.mMaxSteps: int = 10000
        self.mNOk: int = 0                       # number of ok steps
        self.mNBad: int = 0                      # number of repeated steps
        # Temporary work space used by baseStep
        self.mWs2: Optional[F64Array] = None
        self.mWs3: Optional[F64Array] = None
        self.mWs4: Optional[F64Array] = None
        self.mWs5: Optional[F64Array] = None
        self.mWs6: Optional[F64Array] = None
        self.mYTemp: Optional[F64Array] = None
        # Temporary work space used by qcStep
        self.mYErr: Optional[F64Array] = None
        self.mQcYTemp: Optional[F64Array] = None

    # ==================================================================
    # Private helpers
    # ==================================================================
    def _sign(self, magnitude: float, sign: float) -> float:
        # Java: return sign >= 0.0 ? Math.abs(magnitude) : -Math.abs(magnitude);
        # R8: this is the Numerical-Recipes SIGN(a, b) helper, NOT Math.signum;
        # there is no signum(0) zero-case to special-case here.
        return abs(magnitude) if sign >= 0.0 else -abs(magnitude)

    def _baseStep(self, x: float, y: F64Array, dydx: F64Array, h: float,
                  yout: F64Array, yerr: F64Array) -> None:
        """baseStep - Take a single Cash-Karp Runge-Kutta step. Given the
        n=mNDimensions values y[0..n-1] and their derivatives dydx[0..n-1] known
        at x, use a fifth order Cash-Karp Runge-Kutta method to advance the
        solution over an interval h. The resulting y value is returned in yout.
        An estimate of the truncation error is returned in yerr.
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
        # R5: yout / yerr are written in place below.
        self._require_mutable_f64(yout, "yout")
        self._require_mutable_f64(yerr, "yerr")
        n: int = self.mNVariables
        # Workspace
        if self.mWs2 is None:
            self.mWs2 = np.zeros(n, dtype=np.float64)
            self.mWs3 = np.zeros(n, dtype=np.float64)
            self.mWs4 = np.zeros(n, dtype=np.float64)
            self.mWs5 = np.zeros(n, dtype=np.float64)
            self.mWs6 = np.zeros(n, dtype=np.float64)
            self.mYTemp = np.zeros(n, dtype=np.float64)
        # First step
        for i in range(n):
            self.mYTemp[i] = y[i] + (b21 * h * dydx[i])
        # Second step
        self.derivatives(x + (a2 * h), self.mYTemp, self.mWs2)
        for i in range(n):
            self.mYTemp[i] = y[i] + (h * ((b31 * dydx[i]) + (b32 * self.mWs2[i])))
        # Third step
        self.derivatives(x + (a3 * h), self.mYTemp, self.mWs3)
        for i in range(n):
            self.mYTemp[i] = y[i] + (h * ((b41 * dydx[i]) + (b42 * self.mWs2[i]) + (b43 * self.mWs3[i])))
        # Fourth step
        self.derivatives(x + (a4 * h), self.mYTemp, self.mWs4)
        for i in range(n):
            self.mYTemp[i] = y[i] + (h * ((b51 * dydx[i]) + (b52 * self.mWs2[i]) + (b53 * self.mWs3[i]) + (b54 * self.mWs4[i])))
        # Fifth step
        self.derivatives(x + (a5 * h), self.mYTemp, self.mWs5)
        for i in range(n):
            self.mYTemp[i] = y[i] + (h * ((b61 * dydx[i]) + (b62 * self.mWs2[i]) + (b63 * self.mWs3[i]) + (b64 * self.mWs4[i]) + (b65 * self.mWs5[i])))
        # Sixth step
        self.derivatives(x + (a6 * h), self.mYTemp, self.mWs6)
        for i in range(n):
            yout[i] = y[i] + (h * ((c1 * dydx[i]) + (c3 * self.mWs3[i]) + (c4 * self.mWs4[i]) + (c6 * self.mWs6[i])))
        # Estimate the error
        for i in range(n):
            yerr[i] = h * ((dc1 * dydx[i]) + (dc3 * self.mWs3[i]) + (dc4 * self.mWs4[i]) + (dc5 * self.mWs5[i]) + (dc6 * self.mWs6[i]))

    def _qcStep(self, x: float, y: F64Array, dydx: F64Array, htry: float,
                eps: float, yscal: F64Array) -> float:
        """qcStep - Take a fifth order Runge-Kutta step with monitoring of local
        truncation error. Upon return, y is replaced with the new values, x is
        returned and mHDid and mHNext are set to the actual step size and the
        size of the next step to try.

        :raises UtilException: When the step size becomes too small
        :return: float - The new value of x
        """
        safety: float = 0.9
        pgrow: float = -0.2
        pshrnk: float = -0.25
        errcon: float = 1.89e-4
        if self.mYErr is None:
            self.mYErr = np.zeros(self.mNVariables, dtype=np.float64)
            self.mQcYTemp = np.zeros(self.mNVariables, dtype=np.float64)
        h: float = htry
        errmax: float = 0.0
        # Java do { ... } while (errmax > 1.0);
        while True:
            self._baseStep(x, y, dydx, h, self.mQcYTemp, self.mYErr)
            errmax = 0.0
            for i in range(self.mNVariables):
                errmax = max(errmax, abs(self.mYErr[i] / yscal[i]))
            errmax /= eps
            if errmax > 1.0:
                htemp: float = safety * h * math.pow(errmax, pshrnk)
                h = (max(htemp, 0.1 * h) if h >= 0 else min(htemp, 0.1 * h))
                # Check for step size underflow
                xnew: float = x + h
                if xnew == x:
                    raise UtilException("Step size underflow in AdaptiveRungeKutta.qcStep.")
            if not (errmax > 1.0):
                break
        self.mHNext = (safety * h * math.pow(errmax, pgrow)) if errmax > errcon else (5.0 * h)
        self.mHDid = h
        x += h
        # System.arraycopy(mQcYTemp, 0, y, 0, mNVariables)
        y[:self.mNVariables] = self.mQcYTemp[:self.mNVariables]
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
    # Public configuration / accessors
    # ==================================================================
    def setSaveInterval(self, interval: float) -> None:
        """setSaveInterval - Set the interval on which to save intermediate
        points on the integrated trajectory. (Use clearSaveInterval to not save
        any intermediate points.) Note: The default is not to save any
        intermediate points.
        """
        self.mSaveInterval = abs(interval)

    def clearSaveInterval(self) -> None:
        """clearSaveInterval - Return to the default of not saving any
        intermediate points."""
        self.mSaveInterval = _DOUBLE_MAX_VALUE

    def getNSaved(self) -> int:
        """getNSaved - Returns the number of saved values."""
        return self.mNSaved

    def getX(self, i: int) -> float:
        """getX - Returns the x-coordinate of the i-th saved value

        :param i: int - Where i < getNSaved()
        :return: float
        """
        return float(self.mXSave[i])

    def getY(self, i: int) -> F64Array:
        """getY - returns the getNVariable y-coordinates of the i-th saved values.

        :param i: int - Where i < getNSaved()
        :return: double[] - Of dimension getNVariables
        """
        # Java returns the inner array by reference (caller-mutable). The numpy
        # row is a writeable view, preserving that reference semantics.
        return self.mYSave[i]

    def setMaxSteps(self, maxSteps: int) -> None:
        """setMaxSteps - Set the maximum number of ODE steps to allow. Default
        is 10000."""
        self.mMaxSteps = maxSteps

    def setMinStepSize(self, minStep: float) -> None:
        """setMinStepSize - Sets the minimum permissible step size. Default is
        0.0."""
        self.mMinStepSize = abs(minStep)

    def getNVariables(self) -> int:
        """getNVariables - Returns the number of variables as set in the
        constructor."""
        return self.mNVariables

    def getStepCount(self) -> int:
        """getStepCount - Get the total number of steps required to perform the
        previous integrate operation."""
        return self.mNOk + self.mNBad

    def getGoodStepCount(self) -> int:
        """getGoodStepCount - Get the number of steps leading to results of the
        desired accuracy."""
        return self.mNOk

    def getBadStepCount(self) -> int:
        """getBadStepCount - Get the number of steps that were needed to be
        subdivided to attain results of the desired accuracy."""
        return self.mNBad

    # ==================================================================
    # integrate — R2: scipy primary + faithful literal
    # ==================================================================
    def integrate(self, x1: float, x2: float, ystart: F64Array, eps: float,
                  h1: float) -> F64Array:
        """integrate (scipy primary) - Integrate the ODE specified by
        derivatives over [x1, x2].

        SCIPY-DEV-1: stepping is delegated to scipy.integrate.solve_ivp
        (method="RK45", Dormand-Prince) in place of the Java Cash-Karp (RKCK)
        stepper. The trajectory matches Java within the requested relative
        tolerance; step-count bookkeeping and exact saved abscissae cannot be
        reproduced from scipy's public API and are approximated (see the module
        docstring). Use integrate_literal() for bit-faithful behaviour.

        :param x1: Start of the integration range
        :param x2: End of the integration range
        :param ystart: (In & out) The initial y value; mutated in place
        :param eps: The permissible relative error
        :param h1: The initial step size (advisory only for scipy)
        :raises UtilException: Upon integration failure
        :return: The final y values as an array of length getNVariables().
        """
        # R5: ystart is the externally supplied in/out buffer.
        self._require_mutable_f64(ystart, "ystart")
        tiny: float = 1.0e-10 * eps
        n: int = self.mNVariables
        y0: F64Array = np.array(ystart, dtype=np.float64)[:n].copy()

        def _f(xx: float, yv: ArrayLike) -> F64Array:
            dydx: F64Array = np.zeros(n, dtype=np.float64)
            self.derivatives(float(xx), np.asarray(yv, dtype=np.float64), dydx)
            return dydx

        atol: float = tiny if tiny > 0.0 else 1.0e-300
        sol = solve_ivp(_f, (x1, x2), y0, method="RK45", rtol=eps, atol=atol,
                        dense_output=True)
        if not sol.success:
            raise UtilException(
                "Integration failed in AdaptiveRungeKutta.integrate: " + str(sol.message))

        # Approximate step accounting (see SCIPY-DEV-1).
        self.mNOk = max(0, int(sol.t.shape[0]) - 1)  # accepted steps (approx)
        self.mNBad = 0  # scipy does not expose a rejected-step count

        # Save intermediate points if requested, resampled from dense output.
        self.mNSaved = 0
        if self.mSaveInterval != _DOUBLE_MAX_VALUE:
            saveInt: float = self._sign(self.mSaveInterval, x2 - x1)
            xs: list[float] = [x1]
            xc: float = x1 + saveInt
            # advance while xc is strictly between x1 and x2 in the travel direction
            while ((xc - x2) * (x2 - x1)) < 0.0:
                xs.append(xc)
                xc += saveInt
            if xs[-1] != x2:
                xs.append(x2)
            kMax: int = len(xs)
            self.mXSave = np.zeros(kMax, dtype=np.float64)
            self.mYSave = np.zeros((kMax, n), dtype=np.float64)
            for k in range(kMax):
                self.mXSave[k] = xs[k]
                self.mYSave[k][:] = np.asarray(sol.sol(xs[k]), dtype=np.float64)[:n]
            self.mNSaved = kMax

        y: F64Array = np.asarray(sol.y[:, -1], dtype=np.float64)
        ystart[:n] = y[:n]
        self._clearWorkspace()
        return y

    def integrate_literal(self, x1: float, x2: float, ystart: F64Array,
                          eps: float, h1: float) -> F64Array:
        """integrate (faithful literal) - line-for-line port of the Java
        adaptive step size Runge-Kutta driver.

        :param x1: Start of the integration range
        :param x2: End of the integration range
        :param ystart: (In & out) The initial y value; mutated in place
        :param eps: The permissible relative error
        :param h1: The initial step size
        :raises UtilException: Upon too many steps or too small a step
        :return: The final y values as an array of length getNVariables().
        """
        # R5: ystart is the externally supplied in/out buffer.
        self._require_mutable_f64(ystart, "ystart")
        tiny: float = 1.0e-10 * eps
        yscal: F64Array = np.zeros(self.mNVariables, dtype=np.float64)
        dydx: F64Array = np.zeros(self.mNVariables, dtype=np.float64)
        y: F64Array = np.zeros(self.mNVariables, dtype=np.float64)
        x: float = x1
        h: float = self._sign(h1, x2 - x1)
        xsav: float = 0.0
        saveInt: float = _DOUBLE_MAX_VALUE
        kMax: int = 0
        self.mNSaved = 0
        self.mNOk = 0
        self.mNBad = 0
        # System.arraycopy(ystart, 0, y, 0, mNVariables)
        y[:self.mNVariables] = np.asarray(ystart, dtype=np.float64)[:self.mNVariables]
        if self.mSaveInterval != _DOUBLE_MAX_VALUE:
            # R7: (int) Math.round(d) -> int(math.floor(d + 0.5)). The division
            # here is double/double, so it stays a true (non-floor) division.
            kMax = int(math.floor(((abs(x2 - x1) + self.mSaveInterval) / self.mSaveInterval) + 0.5))
            saveInt = self._sign(self.mSaveInterval, x2 - x1)
            xsav = x - (2.0 * saveInt)  # to ensure that the first step is saved...
            self.mXSave = np.zeros(kMax, dtype=np.float64)
            self.mYSave = np.zeros((kMax, self.mNVariables), dtype=np.float64)
        for step in range(self.mMaxSteps):
            # Save the necessary points
            if (kMax != 0) and (self.mNSaved < kMax) and (abs(x - xsav) >= (0.9999 * self.mSaveInterval)):
                self.mXSave[self.mNSaved] = x
                self.mYSave[self.mNSaved][:] = y[:self.mNVariables]
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
                ystart[:self.mNVariables] = y[:self.mNVariables]
                if kMax != 0:
                    self.mNSaved = min(self.mNSaved, kMax - 1)
                    self.mXSave[self.mNSaved] = x
                    self.mYSave[self.mNSaved][:] = y[:self.mNVariables]
                    self.mNSaved += 1
                self._clearWorkspace()
                return y
            if abs(self.mHNext) <= self.mMinStepSize:
                raise UtilException("Step size too small in AdaptiveRungeKutta.integrate")
            h = self.mHNext
        raise UtilException("Too many steps in AdaptiveRungeKutta.integrate")

    # ==================================================================
    # Abstract method — R1: public abstract keeps its bare Java name.
    # ==================================================================
    @abc.abstractmethod
    def derivatives(self, x: float, y: F64Array, dydx: F64Array) -> None:
        """derivatives - The derived class provides an implementation of the
        derivatives function. x & y[] are input and the user provided
        implementation of derivatives is responsible for returning the
        derivatives in the array dydx. The lengths of y and dydx are equal to
        mNDimensions.

        :param x: double - In
        :param y: double[] - In (of dimension mNDimensions)
        :param dydx: double[] - Out (of dimension mNDimensions)
        """
        ...