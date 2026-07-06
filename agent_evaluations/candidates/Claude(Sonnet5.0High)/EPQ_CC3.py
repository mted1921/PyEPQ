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

CHANGES IN THIS REVISION (ver1.1.0)
------------------------------------
* R1: `m`-prefixed instance fields (mNVariables, mHDid, mHNext, mSaveInterval,
  mMinStepSize, mXSave, mYSave, mNSaved, mMaxSteps, mNOk, mNBad, mWs2..mWs6,
  mYTemp, mYErr, mQcYTemp) are preserved verbatim without an added leading
  underscore, matching the worked `mAtomicNumber` example in
  CONVERSION_GUIDE.md R1 -- the `m` prefix already encodes "private" in this
  codebase's own convention. Private *methods* (`sign`, `baseStep`, `qcStep`,
  `clearWorkspace`) do get the `_` prefix per the R1 table. The public
  `abstract void derivatives(...)` keeps its un-prefixed Java name.
* R2: `integrate` is a public mathematical method, so it is split into
  `integrate()` (scipy-primary) and `integrate_literal()` (line-for-line
  Cash-Karp/Numerical-Recipes port). `_baseStep`/`_qcStep`/`_sign` are
  *private* Java helpers -- only a literal translation is provided for them
  (no `_literal` suffix; R2's foo()/foo_literal() split applies to public
  members).
* SCIPY-DEV-1..4 (see `integrate()` docstring): the scipy-primary path uses
  `scipy.integrate.solve_ivp` (explicit RK45 / Dormand-Prince 4(5)) as the
  step engine instead of the Cash-Karp 4(5) pair in `_baseStep`/`_qcStep`.
  Both are adaptive embedded RK pairs of matching order; results agree with
  `integrate_literal()` to within `eps` but are not bit-identical, and the
  step-accounting getters (`getGoodStepCount`, `getBadStepCount`, and the
  values behind `mHDid`/`mHNext`) are best-effort approximations after
  `integrate()` -- call `integrate_literal()` for exact Java-parity
  bookkeeping. `mMaxSteps` (a step-*count* budget in Java) has no direct
  `solve_ivp` equivalent (only `max_step`, a step-*size* bound), so it is
  enforced post-hoc against the solver's natural step grid.
* R3: `UtilException` is NOT one of the shared `_epq_compat` types (see
  UTILITY_LEDGER.md Tier 0); it is a sibling Tier-0 port
  (`UtilException_ver1_1_0.py`) and is imported from there via the standard
  three-tier fallback, never redefined locally.
* R6 / BUG_LEDGER: audited `baseStep` against the standard Cash-Karp
  Butcher tableau (a2..a6, b21..b65, c1/c3/c4/c6, dc1/dc3/dc4/dc5/dc6) and
  `qcStep`/`integrate` against Numerical Recipes' `rkqs`/`odeint` (the
  routines the Javadoc cites, pp 714-722). All coefficients and control
  logic match the reference algorithm exactly; no dead branches,
  always-true conditions, off-by-one errors, or sign errors were found.
  BUG_LEDGER is therefore empty -- see the tuple below.
* R7: The one `Math.round` call (sizing `kMax` for the save buffers) is
  translated to `int(math.floor(x + 0.5))`.
* R9: All parameters, returns, fields, and non-obvious locals are annotated.
  `F64Array` is used for every `double[]`/`double[][]`.
------------------------------------------------------------------------
"""

from __future__ import annotations

import abc
import math
import sys
import numpy as np
from typing import Any, Optional, Sequence, Union, Callable

from scipy.integrate import solve_ivp

try:
    from ._epq_compat import EPQException, F64Array, JavaRandom
except ImportError:
    try:
        from _epq_compat import EPQException, F64Array, JavaRandom  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility._epq_compat import EPQException, F64Array, JavaRandom  # type: ignore

# UtilException is a sibling Tier-0 port (see UTILITY_LEDGER.md "Port file"
# column: UtilException_ver1_1_0.py) -- NOT one of the shared _epq_compat
# types, and must never be redefined locally (R3).
try:
    from .UtilException_ver1_1_0 import UtilException
except ImportError:
    try:
        from UtilException_ver1_1_0 import UtilException  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility.UtilException_ver1_1_0 import UtilException  # type: ignore


__all__ = ["AdaptiveRungeKutta", "UtilException", "EPQException", "JavaRandom", "F64Array"]


class AdaptiveRungeKutta(abc.ABC):
    """An adaptive step size Runge-Kutta algorithm for numerically evaluating
    differential equations. See the module docstring for the verbatim Javadoc,
    including the usage example.

    NOTE: This algorithm is not thread-safe. Use each instance in one and only
    one thread. (Preserved from the Java Javadoc -- Python has no enforcement
    mechanism for this; it is documentation only, exactly as in Java.)
    """

    # ==================================================================
    # Preserved-bug ledger (machine-readable) -- CONVERSION_GUIDE.md R6
    # ==================================================================
    # No Java bugs were identified in this class (see CHANGES above for the
    # audit trail against the Cash-Karp tableau and NR's rkqs/odeint).
    BUG_LEDGER: tuple = ()  # no bugs identified

    # ==================================================================
    # Internal guards -- CONVERSION_GUIDE.md R5
    # ==================================================================
    @staticmethod
    def _require_mutable_f64(arr: Any, name: str = "arr") -> None:
        """Type guard for in-place methods (CONVERSION_GUIDE R5).

        Java's ``double[]`` is always a mutable double-precision buffer.
        Our in-place helpers can only honour that contract on numpy
        ndarrays with ``dtype=float64``. Lists, tuples, and wrong-dtype
        arrays would silently no-op or copy. Fail loud.
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
        """AdaptiveRungeKutta - Construct an AdaptiveRungeKutta object to solve
        a differential equation of nVars variables. The implementation of
        derivatives should return nVars derivative values for each x & y.

        Args:
            nVars: The number of differential equations (Java: int nVars).
        """
        self.mNVariables: int = nVars  # The number of differential equations
        self.mHDid: float = 0.0  # Actual step size accomplished in last call to qcStep
        self.mHNext: float = 0.0  # Next step size to try when calling qcStep
        self.mSaveInterval: float = sys.float_info.max  # Double.MAX_VALUE sentinel
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
    # Private helpers (Java `private` -> `_`-prefixed per R1)
    # ==================================================================
    def _sign(self, magnitude: float, sign: float) -> float:
        """Numerical-Recipes SIGN(a,b) transfer function: returns |magnitude|
        with the algebraic sign of `sign`."""
        return abs(magnitude) if sign >= 0.0 else -abs(magnitude)

    def _baseStep(self, x: float, y: F64Array, dydx: F64Array, h: float,
                  yout: F64Array, yerr: F64Array) -> None:
        """baseStep - Take a single Cash-Karp Runge-Kutta step. Given the
        n=mNDimensions values y[0..n-1] and their derivatives dydx[0..n-1]
        know at x, use a fifth order Cash-Karp Runge-Kutta method to advance
        the solution over an interval h. The resulting y value is returned in
        yout. An estimate of the truncation error is returned in yerr.

        Args:
            x: Independent variable.
            y: (In) dependent variable, length mNVariables.
            dydx: (In) derivative at x, length mNVariables.
            h: Step size to attempt.
            yout: (Out) advanced solution, length mNVariables.
            yerr: (Out) truncation error estimate, length mNVariables.
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
            n: int = self.mNVariables
            self.mWs2 = np.zeros(n, dtype=np.float64)
            self.mWs3 = np.zeros(n, dtype=np.float64)
            self.mWs4 = np.zeros(n, dtype=np.float64)
            self.mWs5 = np.zeros(n, dtype=np.float64)
            self.mWs6 = np.zeros(n, dtype=np.float64)
            self.mYTemp = np.zeros(n, dtype=np.float64)
        nVars: int = self.mNVariables
        # First step
        for i in range(nVars):
            self.mYTemp[i] = y[i] + (b21 * h * dydx[i])
        # Second step
        self.derivatives(x + (a2 * h), self.mYTemp, self.mWs2)
        for i in range(nVars):
            self.mYTemp[i] = y[i] + (h * ((b31 * dydx[i]) + (b32 * self.mWs2[i])))
        # Third step
        self.derivatives(x + (a3 * h), self.mYTemp, self.mWs3)
        for i in range(nVars):
            self.mYTemp[i] = y[i] + (h * ((b41 * dydx[i]) + (b42 * self.mWs2[i]) + (b43 * self.mWs3[i])))
        # Fourth step
        self.derivatives(x + (a4 * h), self.mYTemp, self.mWs4)
        for i in range(nVars):
            self.mYTemp[i] = y[i] + (h * ((b51 * dydx[i]) + (b52 * self.mWs2[i]) + (b53 * self.mWs3[i]) + (b54 * self.mWs4[i])))
        # Fifth step
        self.derivatives(x + (a5 * h), self.mYTemp, self.mWs5)
        for i in range(nVars):
            self.mYTemp[i] = y[i] + (h * ((b61 * dydx[i]) + (b62 * self.mWs2[i]) + (b63 * self.mWs3[i]) + (b64 * self.mWs4[i]) + (b65 * self.mWs5[i])))
        # Sixth step
        self.derivatives(x + (a6 * h), self.mYTemp, self.mWs6)
        for i in range(nVars):
            yout[i] = y[i] + (h * ((c1 * dydx[i]) + (c3 * self.mWs3[i]) + (c4 * self.mWs4[i]) + (c6 * self.mWs6[i])))
        # Estimate the error
        for i in range(nVars):
            yerr[i] = h * ((dc1 * dydx[i]) + (dc3 * self.mWs3[i]) + (dc4 * self.mWs4[i]) + (dc5 * self.mWs5[i]) + (dc6 * self.mWs6[i]))

    def _qcStep(self, x: float, y: F64Array, dydx: F64Array, htry: float,
                eps: float, yscal: F64Array) -> float:
        """qcStep - Take a fifth order Runge-Kutta step with monitoring of
        local truncation error. Input are the dependent variable
        y[0..mNDimensions-1] and its derivatives dydx[0..mNDimensions-1] at
        the starting value of the independent variable x. Also input is the
        attempted step size htry, the required accuracy eps and the vector
        yscal against which the errors are scaled. Upon return, y is replaced
        with the new values, x is returned and mHDid and mHNext are set to
        the actual step size and the size of the next step to try.

        Args:
            x: (In) independent variable.
            y: (In, Out) dependent variable, length mNVariables -- mutated
                in place, matching Java array-reference semantics.
            dydx: (In) derivative at x, length mNVariables.
            htry: The step size to attempt.
            eps: Desired accuracy.
            yscal: (In) Error scaling vector, length mNVariables.

        Returns:
            The new value of x.

        Raises:
            UtilException: When the step size becomes too small.
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
                    raise UtilException("Step size underflow in AdaptiveRungeKutta.qcStep.")
            if not (errmax > 1.0):
                break
        self.mHNext = safety * h * math.pow(errmax, pgrow) if errmax > errcon else 5.0 * h
        self.mHDid = h
        x += h
        y[0:self.mNVariables] = self.mQcYTemp[0:self.mNVariables]
        return x

    def _clearWorkspace(self) -> None:
        """clearWorkspace - null all temporary space to free memory."""
        self.mWs2 = None
        self.mWs3 = None
        self.mWs4 = None
        self.mWs5 = None
        self.mWs6 = None
        self.mYTemp = None
        self.mYErr = None
        self.mQcYTemp = None

    # ==================================================================
    # Public configuration API
    # ==================================================================
    def setSaveInterval(self, interval: float) -> None:
        """setSaveInterval - Set the interval on which to save intermediate
        points on the integrated trajectory. (Use clearSaveInterval to not
        save any intermediate points.) Note: The default is not to save any
        intermediate points."""
        self.mSaveInterval = abs(interval)

    def clearSaveInterval(self) -> None:
        """clearSaveInterval - Return to the default of not saving any
        intermediate points."""
        self.mSaveInterval = sys.float_info.max

    def getNSaved(self) -> int:
        """getNSaved - Returns the number of saved values."""
        return self.mNSaved

    def getX(self, i: int) -> float:
        """getX - Returns the x-coordinate of the i-th saved value.

        Args:
            i: Where i < getNSaved().
        """
        return float(self.mXSave[i])

    def getY(self, i: int) -> F64Array:
        """getY - returns the getNVariable x y-coordinates of the i-th saved
        values.

        Args:
            i: Where i < getNSaved().

        Returns:
            Array of dimension getNVariables(). This is a view onto the
            internal buffer (not a copy), matching Java's array-reference
            semantics.
        """
        return self.mYSave[i]

    def setMaxSteps(self, maxSteps: int) -> None:
        """setMaxSteps - Set the maximum number of ODE steps to allow.
        Default is 10000."""
        self.mMaxSteps = maxSteps

    def setMinStepSize(self, minStep: float) -> None:
        """setMinStepSize - Sets the minimum permissible step size. Default
        is 0.0."""
        self.mMinStepSize = abs(minStep)

    def getNVariables(self) -> int:
        """getNVariables - Returns the number of variables as set in the
        constructor."""
        return self.mNVariables

    def getStepCount(self) -> int:
        """getStepCount - Get the total number of steps required to perform
        the previous integrate operation."""
        return self.mNOk + self.mNBad

    def getGoodStepCount(self) -> int:
        """getGoodStepCount - Get the number of steps leading to results of
        the desired accuracy."""
        return self.mNOk

    def getBadStepCount(self) -> int:
        """getBadStepCount - Get the number of steps that were needed to be
        subdivided to attain results of the desired accuracy."""
        return self.mNBad

    # ==================================================================
    # integrate() / integrate_literal() -- CONVERSION_GUIDE.md R2
    # ==================================================================
    def integrate_literal(self, x1: float, x2: float, ystart: F64Array,
                           eps: float, h1: float) -> F64Array:
        """integrate - Integrate the ODE specified by derivatives using the
        adaptive step size Runge-Kutta algorithm over the independent
        variable interval x1 to x2. ystart contains the initial y values.
        eps is measure of the permissible error. h is the initial step size.

        Line-for-line Cash-Karp / Numerical-Recipes-faithful translation of
        the Java `integrate` method (R2 "literal" companion of `integrate`).

        Args:
            x1: Start of the integration range.
            x2: End of the integration range.
            ystart: (In & out) the initial y value, length mNVariables --
                mutated in place, matching Java's array-reference semantics.
            eps: The permissible relative error.
            h1: The initial step size.

        Returns:
            The final y values as an array of length getNVariables().

        Raises:
            UtilException: Upon too many steps or too small a step.
        """
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
        y[0:self.mNVariables] = ystart[0:self.mNVariables]
        if self.mSaveInterval != sys.float_info.max:
            # R7: Math.round(x) -> int(math.floor(x + 0.5))
            kMax = int(math.floor(((abs(x2 - x1) + self.mSaveInterval) / self.mSaveInterval) + 0.5))
            saveInt = self._sign(self.mSaveInterval, x2 - x1)
            xsav = x - (2.0 * saveInt)  # to ensure that the first step is saved...
            self.mXSave = np.zeros(kMax, dtype=np.float64)
            self.mYSave = np.zeros((kMax, self.mNVariables), dtype=np.float64)
        for _step in range(self.mMaxSteps):
            # Save the necessary points
            if (kMax != 0) and (self.mNSaved < kMax) and (abs(x - xsav) >= (0.9999 * self.mSaveInterval)):
                self.mXSave[self.mNSaved] = x
                self.mYSave[self.mNSaved][0:self.mNVariables] = y[0:self.mNVariables]
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
                ystart[0:self.mNVariables] = y[0:self.mNVariables]
                if kMax != 0:
                    self.mNSaved = min(self.mNSaved, kMax - 1)
                    self.mXSave[self.mNSaved] = x
                    self.mYSave[self.mNSaved][0:self.mNVariables] = y[0:self.mNVariables]
                    self.mNSaved += 1
                self._clearWorkspace()
                return y
            if abs(self.mHNext) <= self.mMinStepSize:
                raise UtilException("Step size too small in AdaptiveRungeKutta.integrate")
            h = self.mHNext
        raise UtilException("Too many steps in AdaptiveRungeKutta.integrate")

    def integrate(self, x1: float, x2: float, ystart: F64Array,
                  eps: float, h1: float) -> F64Array:
        """Scipy-backed primary implementation of Java's `integrate`.

        SCIPY-DEV-1: Uses `scipy.integrate.solve_ivp` (explicit RK45, i.e.
        Dormand-Prince 4(5)) as the adaptive step engine in place of the
        Cash-Karp 4(5) pair in `_baseStep`/`_qcStep`. Both are adaptive
        embedded Runge-Kutta pairs of matching order; results agree with
        `integrate_literal()` to within `eps` but are NOT bit-identical.

        SCIPY-DEV-2: `eps` is passed through as `rtol`. Java's per-component
        error norm scales against `yscal[i] = |y[i]| + |dydx[i]*h| + tiny`;
        there is no exact scipy equivalent, so `atol` is set to a small
        fraction of `eps` instead.

        SCIPY-DEV-3: `mMaxSteps` bounds the *number of adaptive steps* in
        Java; `solve_ivp` has no such parameter (only `max_step`, a step
        *size* bound). It is therefore enforced post-hoc against the number
        of steps the solver's natural (dense-output) grid actually took.

        SCIPY-DEV-4: `mNBad` (rejected steps) cannot be recovered from
        `solve_ivp`'s public result and is always 0 after calling
        `integrate()`; `mHDid`/`mHNext` are likewise approximated from the
        last accepted step in the solver's natural grid. Call
        `integrate_literal()` for exact Java-parity step accounting
        (`getGoodStepCount`, `getBadStepCount`, `mHDid`, `mHNext`).

        Args:
            x1: Start of the integration range.
            x2: End of the integration range.
            ystart: (In & out) the initial y value, length mNVariables --
                mutated in place, matching Java's array-reference semantics.
            eps: The permissible relative error (used as `rtol`).
            h1: The initial step size (used as a hint for scipy's
                `first_step`).

        Returns:
            The final y values as an array of length getNVariables().

        Raises:
            UtilException: If the scipy solver fails, or the number of
                accepted steps exceeds `mMaxSteps`, or the smallest step
                actually taken is at or below `mMinStepSize`.
        """
        self._require_mutable_f64(ystart, "ystart")
        y0: F64Array = np.array(ystart[0:self.mNVariables], dtype=np.float64, copy=True)
        atol: float = max(eps * 1.0e-6, 1.0e-12)
        first_step: Optional[float] = None
        if (h1 != 0.0) and (x2 != x1):
            first_step = float(min(abs(h1), abs(x2 - x1)))

        def _rhs(t: float, yv: F64Array) -> F64Array:
            dydx: F64Array = np.zeros(self.mNVariables, dtype=np.float64)
            self.derivatives(float(t), np.asarray(yv, dtype=np.float64), dydx)
            return dydx

        kMax: int = 0
        saveInt: float = sys.float_info.max
        if self.mSaveInterval != sys.float_info.max:
            kMax = int(math.floor(((abs(x2 - x1) + self.mSaveInterval) / self.mSaveInterval) + 0.5))
            saveInt = self._sign(self.mSaveInterval, x2 - x1)

        sol = solve_ivp(
            fun=_rhs,
            t_span=(x1, x2),
            y0=y0,
            method="RK45",
            rtol=eps,
            atol=atol,
            first_step=first_step,
            dense_output=(kMax != 0),
        )
        if not sol.success:
            raise UtilException(f"Step size underflow in AdaptiveRungeKutta.integrate: {sol.message}")

        nSteps: int = max(len(sol.t) - 1, 0)
        if nSteps > self.mMaxSteps:
            raise UtilException("Too many steps in AdaptiveRungeKutta.integrate")

        self.mNOk = nSteps
        self.mNBad = 0  # SCIPY-DEV-4: not recoverable from solve_ivp's public API
        if nSteps > 0:
            stepSizes: F64Array = np.diff(sol.t)
            nonzero = stepSizes[np.abs(stepSizes) > 0.0]
            if nonzero.size > 0:
                minAbsStep: float = float(np.min(np.abs(nonzero)))
                if minAbsStep <= self.mMinStepSize:
                    raise UtilException("Step size too small in AdaptiveRungeKutta.integrate")
                self.mHDid = float(stepSizes[-1])
                self.mHNext = self.mHDid
            else:
                self.mHDid = 0.0
                self.mHNext = 0.0
        else:
            self.mHDid = 0.0
            self.mHNext = 0.0

        self.mNSaved = 0
        self.mXSave = None
        self.mYSave = None
        if kMax != 0:
            pts: list = [x1 + i * saveInt for i in range(kMax)]
            pts = [p for p in pts if (p - x2) * saveInt <= 0.0]
            if (len(pts) == 0) or (pts[-1] != x2):
                pts.append(x2)
            tSave: F64Array = np.array(pts, dtype=np.float64)
            ySave: F64Array = sol.sol(tSave)  # shape (mNVariables, len(tSave))
            self.mNSaved = tSave.shape[0]
            self.mXSave = tSave
            self.mYSave = np.ascontiguousarray(ySave.T)

        yFinal: F64Array = np.array(sol.y[:, -1], dtype=np.float64)
        ystart[0:self.mNVariables] = yFinal[0:self.mNVariables]
        self._clearWorkspace()
        return yFinal

    # ==================================================================
    # Abstract API -- `public abstract` keeps its un-prefixed Java name (R1)
    # ==================================================================
    @abc.abstractmethod
    def derivatives(self, x: float, y: F64Array, dydx: F64Array) -> None:
        """derivatives - The derived class provides an implementation of the
        derivatives function. x & y[] are input and the user provided
        implementation of derivatives is resposible for returning the
        derivatives in the array dydx. The lengths of y and dydx are equal
        to mNDimensions.

        Args:
            x: (In) independent variable.
            y: (In) dependent variable, length mNVariables.
            dydx: (Out) derivatives, length mNVariables -- must be filled in
                place by the override, matching Java's array-reference
                semantics.
        """
        ...