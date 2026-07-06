r"""
AdaptiveRungeKutta_ver1_1_0.py — Python port of gov.nist.microanalysis.Utility.AdaptiveRungeKutta

Guide version : 1
Generation    : 1
Port-code fixes: 0

CHANGES
-------
* R2: `integrate()` is the scipy-backed primary (scipy.integrate.solve_ivp,
  method="RK45"); `integrate_literal()` is the line-for-line Java port of the
  Numerical-Recipes odeint driver with the Cash-Karp `baseStep`/`qcStep` core.
  Deviations of the scipy primary from Java are tagged SCIPY-DEV-1..4 at the
  call sites:
  - SCIPY-DEV-1: RK45 is Dormand-Prince(4,5), not Cash-Karp(4,5). Same order,
    different embedded pair; trajectories agree within tolerance, step
    sequences do not.
  - SCIPY-DEV-2: scipy does not expose rejected-step counts. `integrate()`
    reports mNOk = accepted steps, mNBad = 0. `getStepCount()` therefore
    counts accepted steps only after `integrate()`; use `integrate_literal()`
    for Java-parity step accounting.
  - SCIPY-DEV-3: saved trajectory points are evaluated on the exact grid
    x1 + k*saveInterval via dense output (plus the endpoint), so the scipy
    primary does not reproduce JAVA-BUG-1 (see BUG_LEDGER) and its saved x
    values sit exactly on the grid rather than within 0.9999*interval of it.
  - SCIPY-DEV-4: mMaxSteps is enforced post-hoc (raise after solve if the
    accepted-step count exceeded it) and mMinStepSize is not enforced (RK45
    has no min-step parameter); scipy raises through sol.success on failure.
    `integrate_literal()` enforces both exactly as Java does.
* R5: `ystart` is mutated (Java `System.arraycopy(y, 0, ystart, 0, n)`), so
  both integrate variants call `_require_mutable_f64(ystart)` on entry, as do
  the private `_qcStep`/`_baseStep` on their in-place output buffers.
* R6: JAVA-BUG-1 recorded in BUG_LEDGER with `integrate_strict()` companion.
* Exceptions: this class throws gov.nist.microanalysis.Utility.UtilException,
  imported from its sibling port `UtilException_ver1_1_0` (filename per
  UTILITY_LEDGER.md "Port file" column), not from `_epq_compat`.

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
from scipy.integrate import solve_ivp as _solve_ivp

try:
    from ._epq_compat import EPQException, F64Array, JavaRandom
except ImportError:
    try:
        from _epq_compat import EPQException, F64Array, JavaRandom  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility._epq_compat import EPQException, F64Array, JavaRandom  # type: ignore

# Sibling port — filename per UTILITY_LEDGER.md "Port file" column (Tier 0).
try:
    from .UtilException_ver1_1_0 import UtilException
except ImportError:
    try:
        from UtilException_ver1_1_0 import UtilException  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility.UtilException_ver1_1_0 import UtilException  # type: ignore


__all__ = ["AdaptiveRungeKutta", "UtilException"]


# Java Double.MAX_VALUE (== 1.7976931348623157e308). Used as the "saving
# disabled" sentinel exactly as Java does.
_DOUBLE_MAX_VALUE: float = sys.float_info.max


class AdaptiveRungeKutta(abc.ABC):
    """Abstract adaptive-step Cash-Karp Runge-Kutta ODE driver.

    Subclasses must implement :meth:`derivatives` (public abstract in Java,
    therefore un-prefixed here per R1). See the module docstring for the
    verbatim Java class Javadoc.
    """

    # ==================================================================
    # Preserved-bug ledger (machine-readable) — see BUG_GUIDE.md
    # ==================================================================
    # Each tuple: (id, method, description, has_strict_variant).
    BUG_LEDGER: tuple = (
        ("JAVA-BUG-1", "integrate",
         "The save buffer is undersized by one slot when the integration "
         "span is not a half-round-up multiple of the save interval. Java "
         "line 376 sizes it as "
         "`kMax = (int) Math.round((Math.abs(x2 - x1) + mSaveInterval) / mSaveInterval);` "
         "but up to floor(span/interval)+1 interior points are saved plus "
         "the endpoint, i.e. floor(span/interval)+2 slots are needed when "
         "frac(span/interval) < 0.5. The endpoint branch at Java line 409 "
         "(`mNSaved = Math.min(mNSaved, kMax - 1);`) then silently "
         "overwrites the last interior trajectory point with the endpoint. "
         "Example: x1=0, x2=1, interval=0.3 saves {0, 0.3, 0.6, 1.0} — the "
         "point near 0.9 is lost. `integrate_literal` preserves this; "
         "`integrate_strict` allocates one extra slot so no interior point "
         "is overwritten; the scipy primary `integrate` sizes its grid "
         "exactly (SCIPY-DEV-3) and is unaffected.", True),
    )

    # ==================================================================
    # Construction
    # ==================================================================
    def __init__(self, nVars: int) -> None:
        """Java: public AdaptiveRungeKutta(int nVars).

        Construct an AdaptiveRungeKutta object to solve a differential
        equation of nVars variables. The implementation of derivatives
        should return nVars derivative values for each x & y.
        """
        # private final int — the number of differential equations
        self.mNVariables: int = int(nVars)
        # Actual step size accomplished in last call to qcStep
        self.mHDid: float = 0.0
        # Next step size to try when calling qcStep
        self.mHNext: float = 0.0
        self.mSaveInterval: float = _DOUBLE_MAX_VALUE
        self.mMinStepSize: float = 0.0
        self.mXSave: Optional[F64Array] = None          # double[]
        self.mYSave: Optional[F64Array] = None          # double[][] -> 2-D
        self.mNSaved: int = 0
        self.mMaxSteps: int = 10000
        self.mNOk: int = 0    # Number of ok steps
        self.mNBad: int = 0   # Number of repeated steps
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
    # Internal guards (CONVERSION_GUIDE R5)
    # ==================================================================
    @staticmethod
    def _require_mutable_f64(arr, name: str = "arr") -> None:
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
    # Private helpers (Java `private` -> `_` prefix, R1)
    # ==================================================================
    def _sign(self, magnitude: float, sign: float) -> float:
        """Java: private double sign(double magnitude, double sign).

        NR's SIGN macro: |magnitude| carrying the sign of `sign`, with
        `sign >= 0.0` mapping to positive. NOTE: this is NOT Math.signum
        (R8 does not apply); the Java ternary is translated verbatim.
        """
        return abs(magnitude) if sign >= 0.0 else -abs(magnitude)

    def _baseStep(self, x: float, y: F64Array, dydx: F64Array, h: float,
                  yout: F64Array, yerr: F64Array) -> None:
        """Java: private void baseStep(double x, double[] y, double[] dydx,
        double h, double[] yout, double[] yerr).

        baseStep - Take a single Cash-Karp Runge-Kutta step. Given the
        n=mNDimensions values y[0..n-1] and their derivatives dydx[0..n-1]
        know at x, use a fifth order Cash-Karp Runge-Kutta method to advance
        the solution over an interval h. The resulting y value is returned
        in yout. An estimate of the truncation error is returned in yerr.
        """
        # R5: yout and yerr are mutated in place.
        AdaptiveRungeKutta._require_mutable_f64(yout, "yout")
        AdaptiveRungeKutta._require_mutable_f64(yerr, "yerr")
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
        """Java: private double qcStep(double x, double[] y, double[] dydx,
        double htry, double eps, double[] yscal) throws UtilException.

        qcStep - Take a fifth order Runge-Kutta step with monitoring of local
        truncation error. Input are the dependent variable y[0..mNDimensions-1]
        and its derivatives dydx[0..mNDimensions-1] at the starting value of
        the independent variable x. Also input is the attempted step size
        htry, the required accuracy eps and the vector yscal against which the
        errors are scaled. Upon return, y is replaced with the new values, x
        is returned and mHDid and mHNext are set to the actual step size and
        the size of the next step to try.

        :param x: (In) independent variable
        :param y: (In,Out) dependent variable — mutated in place
        :param dydx: (In) derivative at x
        :param htry: The step size to attempt
        :param eps: Desired accuracy
        :param yscal: (In) Error scaling vector
        :raises UtilException: When the step size becomes too small
        :return: The new value of x
        """
        # R5: y is mutated in place (System.arraycopy at the end).
        AdaptiveRungeKutta._require_mutable_f64(y, "y")
        safety: float = 0.9
        pgrow: float = -0.2
        pshrnk: float = -0.25
        errcon: float = 1.89e-4
        if self.mYErr is None:
            self.mYErr = np.zeros(self.mNVariables, dtype=np.float64)
            self.mQcYTemp = np.zeros(self.mNVariables, dtype=np.float64)
        errmax: float
        h: float = htry
        # Java do { ... } while (errmax > 1.0);
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
        self.mHNext = (safety * h * math.pow(errmax, pgrow)) if errmax > errcon else (5.0 * h)
        self.mHDid = h
        x += h
        # Java: System.arraycopy(mQcYTemp, 0, y, 0, mNVariables);
        y[0:self.mNVariables] = self.mQcYTemp[0:self.mNVariables]
        return x

    def _clearWorkspace(self) -> None:
        """Java: private void clearWorkspace().

        clearWorkspace - null all temporary space to free memory
        """
        self.mWs2 = None
        self.mWs3 = None
        self.mWs4 = None
        self.mWs5 = None
        self.mWs6 = None
        self.mYTemp = None
        self.mYErr = None
        self.mQcYTemp = None

    # ==================================================================
    # Public accessors / mutators
    # ==================================================================
    def setSaveInterval(self, interval: float) -> None:
        """Java: public void setSaveInterval(double interval).

        setSaveInterval - Set the interval on which to save intermediate
        points on the integrated trajectory. (Use clearSaveInterval to not
        save any intermediate points.) Note: The default is not to save any
        intermediate points.
        """
        self.mSaveInterval = abs(float(interval))

    def clearSaveInterval(self) -> None:
        """Java: public void clearSaveInterval().

        clearSaveInterval - Return to the default of not saving any
        intermediate points.
        """
        self.mSaveInterval = _DOUBLE_MAX_VALUE

    def getNSaved(self) -> int:
        """Java: public int getNSaved().

        getNSaved - Returns the number of saved values.
        """
        return self.mNSaved

    def getX(self, i: int) -> float:
        """Java: public double getX(int i).

        getX - Returns the x-coordinate of the i-th saved value, where
        i < getNSaved().
        """
        # R9: explicit cast — numpy scalar -> Python float (Java double).
        return float(self.mXSave[i])

    def getY(self, i: int) -> F64Array:
        """Java: public double[] getY(int i).

        getY - returns the getNVariable x y-coordinates of the i-th saved
        values, where i < getNSaved(). Of dimension getNVariables().

        Returns a view of the i-th saved row (matching Java's return of the
        array reference: caller mutations propagate to the saved data).
        """
        return self.mYSave[i]

    def setMaxSteps(self, maxSteps: int) -> None:
        """Java: public void setMaxSteps(int maxSteps).

        setMaxSteps - Set the maximum number of ODE steps to allow. Default
        is 10000.
        """
        self.mMaxSteps = int(maxSteps)

    def setMinStepSize(self, minStep: float) -> None:
        """Java: public void setMinStepSize(double minStep).

        setMinStepSize - Sets the minimum permissible step size. Default
        is 0.0.
        """
        self.mMinStepSize = abs(float(minStep))

    def getNVariables(self) -> int:
        """Java: public int getNVariables().

        getNVariables - Returns the number of variables as set in the
        constructor.
        """
        return self.mNVariables

    def getStepCount(self) -> int:
        """Java: public int getStepCount().

        getStepCount - Get the total number of steps required to perform the
        previous integrate operation.
        """
        return self.mNOk + self.mNBad

    def getGoodStepCount(self) -> int:
        """Java: public int getGoodStepCount().

        getGoodStepCount - Get the number of steps leading to results of the
        desired accuracy.
        """
        return self.mNOk

    def getBadStepCount(self) -> int:
        """Java: public int getBadStepCount().

        getBadStepCount - Get the number of steps that were needed to be
        subdivided to attain results of the desired accuracy.
        """
        return self.mNBad

    # ==================================================================
    # integrate — scipy primary (R2), literal Java port, strict variant
    # ==================================================================
    def integrate(self, x1: float, x2: float, ystart: F64Array, eps: float,
                  h1: float) -> F64Array:
        """scipy-backed primary for Java's
        public double[] integrate(double x1, double x2, double[] ystart,
        double eps, double h1) throws UtilException.

        integrate - Integrate the ODE specified by derivatives using an
        adaptive step size Runge-Kutta algorithm over the independent
        variable interval x1 to x2. ystart contains the initial y values
        (In & Out: mutated to the final y on success). eps is the
        permissible relative error. h1 is the initial step size.

        Deviations from the literal Java algorithm: SCIPY-DEV-1..4 (see
        module docstring CHANGES). For bit-faithful Java behaviour use
        :meth:`integrate_literal`.

        :return: The final y values as an array of length getNVariables().
        :raises UtilException: Upon too many steps or solver failure.
        """
        # R5: ystart is mutated on success (Java System.arraycopy).
        AdaptiveRungeKutta._require_mutable_f64(ystart, "ystart")
        n: int = self.mNVariables
        tiny: float = 1.0e-10 * eps   # matches Java's yscal floor
        saving: bool = self.mSaveInterval != _DOUBLE_MAX_VALUE
        self.mNSaved = 0
        self.mNOk = 0
        self.mNBad = 0

        def _fun(t: float, yv: np.ndarray) -> F64Array:
            dydx: F64Array = np.zeros(n, dtype=np.float64)
            self.derivatives(float(t), np.asarray(yv, dtype=np.float64), dydx)
            return dydx

        span: float = abs(x2 - x1)
        # scipy requires 0 < first_step <= |x2 - x1|; Java's h1 is a hint.
        first_step: Optional[float] = None
        if h1 != 0.0 and span > 0.0:
            first_step = min(abs(h1), span)
        # SCIPY-DEV-1: RK45 (Dormand-Prince) instead of Cash-Karp.
        # rtol=eps, atol=tiny mirrors Java's yscal = |y| + |h*dydx| + tiny
        # error scaling as closely as scipy's |y|-proportional model allows.
        sol = _solve_ivp(_fun, (float(x1), float(x2)),
                         np.asarray(ystart[0:n], dtype=np.float64),
                         method="RK45", rtol=float(eps), atol=tiny,
                         first_step=first_step, dense_output=saving)
        if not sol.success:
            raise UtilException(
                "Step size too small in AdaptiveRungeKutta.integrate")
        # SCIPY-DEV-2: rejected steps are not exposed; mNBad stays 0.
        self.mNOk = int(sol.t.shape[0]) - 1
        self.mNBad = 0
        # SCIPY-DEV-4: mMaxSteps enforced post-hoc; mMinStepSize not enforced.
        if self.getStepCount() > self.mMaxSteps:
            raise UtilException("Too many steps in AdaptiveRungeKutta.integrate")
        y: F64Array = np.array(sol.y[:, -1], dtype=np.float64)
        if saving:
            # SCIPY-DEV-3: exact-grid samples via dense output; buffer sized
            # exactly, so JAVA-BUG-1 (endpoint overwriting the last interior
            # point) does not occur here.
            saveInt: float = self._sign(self.mSaveInterval, x2 - x1)
            nInterior: int = int(math.floor(span / self.mSaveInterval)) if self.mSaveInterval > 0.0 else 0
            xs: list = [x1 + (k * saveInt) for k in range(nInterior + 1)]
            if abs(xs[-1] - x2) < (1.0e-4 * self.mSaveInterval):
                xs[-1] = float(x2)   # on-grid endpoint: no duplicate sample
            else:
                xs.append(float(x2))
            xArr: F64Array = np.asarray(xs, dtype=np.float64)
            self.mXSave = xArr
            self.mYSave = np.ascontiguousarray(sol.sol(xArr).T, dtype=np.float64)
            self.mNSaved = int(xArr.shape[0])
        # Java: System.arraycopy(y, 0, ystart, 0, mNVariables);
        ystart[0:n] = y
        return y

    def integrate_literal(self, x1: float, x2: float, ystart: F64Array,
                          eps: float, h1: float) -> F64Array:
        """Line-for-line Java port of
        public double[] integrate(double x1, double x2, double[] ystart,
        double eps, double h1) throws UtilException.

        integrate - Integrate the ODE specified by derivatives using the
        adaptive step size Runge-Kutta algorithm over the independent
        variable interval x1 to x2. ystart contains the initial y values.
        eps is measure of the permissible error. h is the initial step size.

        :param x1: Start of the integration range
        :param x2: End of the integration range
        :param ystart: (In & out) The initial y value — mutated on success
        :param eps: The permissible relative error
        :param h1: The initial step size
        :return: The final y values as an array of length getNVariables().
        :raises UtilException: Upon too many steps or too small a step
        """
        # R5: ystart is mutated on success (Java System.arraycopy).
        AdaptiveRungeKutta._require_mutable_f64(ystart, "ystart")
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
        # Java: System.arraycopy(ystart, 0, y, 0, mNVariables);
        y[0:self.mNVariables] = ystart[0:self.mNVariables]
        if self.mSaveInterval != _DOUBLE_MAX_VALUE:
            # JAVA-BUG-1 (allocation): Java line 376:
            #   `kMax = (int) Math.round((Math.abs(x2 - x1) + mSaveInterval) / mSaveInterval);`
            # Undersized by one slot when frac(|x2-x1|/interval) < 0.5 (and
            # non-zero); the endpoint then overwrites the last interior save
            # (see the termination branch below). R7: Math.round(x) ->
            # int(math.floor(x + 0.5)).
            kMax = int(math.floor(((abs(x2 - x1) + self.mSaveInterval) / self.mSaveInterval) + 0.5))
            saveInt = self._sign(self.mSaveInterval, x2 - x1)
            xsav = x - (2.0 * saveInt)  # to ensure that the first step is saved...
            self.mXSave = np.zeros(kMax, dtype=np.float64)
            self.mYSave = np.zeros((kMax, self.mNVariables), dtype=np.float64)
        for step in range(self.mMaxSteps):
            # Save the necessary points
            if (kMax != 0) and (self.mNSaved < kMax) and (abs(x - xsav) >= (0.9999 * self.mSaveInterval)):
                self.mXSave[self.mNSaved] = x
                # Java: System.arraycopy(y, 0, mYSave[mNSaved], 0, mNVariables);
                self.mYSave[self.mNSaved, 0:self.mNVariables] = y[0:self.mNVariables]
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
                # Java: System.arraycopy(y, 0, ystart, 0, mNVariables);
                ystart[0:self.mNVariables] = y[0:self.mNVariables]
                if kMax != 0:
                    # JAVA-BUG-1 (overwrite): Java line 409:
                    #   `mNSaved = Math.min(mNSaved, kMax - 1);`
                    # When the undersized buffer is already full this clamps
                    # onto the last interior save, which the endpoint then
                    # overwrites. Preserved verbatim.
                    self.mNSaved = min(self.mNSaved, kMax - 1)
                    self.mXSave[self.mNSaved] = x
                    self.mYSave[self.mNSaved, 0:self.mNVariables] = y[0:self.mNVariables]
                    self.mNSaved += 1
                self._clearWorkspace()
                return y
            if abs(self.mHNext) <= self.mMinStepSize:
                raise UtilException("Step size too small in AdaptiveRungeKutta.integrate")
            h = self.mHNext
        raise UtilException("Too many steps in AdaptiveRungeKutta.integrate")

    def integrate_strict(self, x1: float, x2: float, ystart: F64Array,
                         eps: float, h1: float) -> F64Array:
        """Strict companion to :meth:`integrate_literal` fixing JAVA-BUG-1.

        Identical Cash-Karp algorithm and step accounting, but the save
        buffer is allocated with one extra slot (kMax + 1) so the endpoint
        save never overwrites the last interior trajectory point. All other
        behaviour (0.9999*interval save trigger, step limits, exceptions)
        matches integrate_literal exactly.
        """
        # R5: ystart is mutated on success (Java System.arraycopy).
        AdaptiveRungeKutta._require_mutable_f64(ystart, "ystart")
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
        y[0:self.mNVariables] = ystart[0:self.mNVariables]
        if self.mSaveInterval != _DOUBLE_MAX_VALUE:
            # STRICT FIX for JAVA-BUG-1: one extra slot vs Java line 376,
            # sized for floor(span/interval)+1 interior points + endpoint.
            kMax = int(math.floor(((abs(x2 - x1) + self.mSaveInterval) / self.mSaveInterval) + 0.5)) + 1
            saveInt = self._sign(self.mSaveInterval, x2 - x1)
            xsav = x - (2.0 * saveInt)
            self.mXSave = np.zeros(kMax, dtype=np.float64)
            self.mYSave = np.zeros((kMax, self.mNVariables), dtype=np.float64)
        for step in range(self.mMaxSteps):
            if (kMax != 0) and (self.mNSaved < kMax) and (abs(x - xsav) >= (0.9999 * self.mSaveInterval)):
                self.mXSave[self.mNSaved] = x
                self.mYSave[self.mNSaved, 0:self.mNVariables] = y[0:self.mNVariables]
                xsav = x
                self.mNSaved += 1
            self.derivatives(x, y, dydx)
            hMax: float = abs((xsav + saveInt) - x)
            if abs(h) > hMax:
                h = self._sign(hMax, h)
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
                    # Clamp kept purely as an overflow guard; with the extra
                    # slot it never truncates a legitimate interior save.
                    self.mNSaved = min(self.mNSaved, kMax - 1)
                    self.mXSave[self.mNSaved] = x
                    self.mYSave[self.mNSaved, 0:self.mNVariables] = y[0:self.mNVariables]
                    self.mNSaved += 1
                self._clearWorkspace()
                return y
            if abs(self.mHNext) <= self.mMinStepSize:
                raise UtilException("Step size too small in AdaptiveRungeKutta.integrate")
            h = self.mHNext
        raise UtilException("Too many steps in AdaptiveRungeKutta.integrate")

    # ==================================================================
    # Abstract API — public abstract in Java, therefore NOT _-prefixed (R1)
    # ==================================================================
    @abc.abstractmethod
    def derivatives(self, x: float, y: F64Array, dydx: F64Array) -> None:
        """Java: abstract public void derivatives(double x, double[] y, double[] dydx).

        derivatives - The derived class provides an implementation of the
        derivatives function. x & y[] are input and the user provided
        implementation of derivatives is resposible for returning the
        derivatives in the array dydx. The lengths of y and dydx are equal
        to mNDimensions.

        :param x: In
        :param y: In (of dimension mNDimensions)
        :param dydx: Out (of dimension mNDimensions) — fill in place
        """
        ...