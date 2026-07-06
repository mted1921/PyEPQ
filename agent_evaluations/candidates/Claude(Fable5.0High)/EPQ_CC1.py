r"""
AdaptiveRungeKutta_ver1_1_0.py — Python port of gov.nist.microanalysis.Utility.AdaptiveRungeKutta

Guide version : 1
Generation    : 1
Port-code fixes: 0

CHANGES (R10 — deliberate deviations, all confined to the scipy primary
`integrate()`; `integrate_literal()` is the faithful line-for-line port)
------------------------------------------------------------------------
* SCIPY-DEV-1: `integrate()` delegates to `scipy.integrate.solve_ivp`
  with method="RK45" (Dormand–Prince 5(4)), whereas Java implements a
  Cash–Karp 5th-order embedded pair. Trajectories agree within the
  requested tolerance but are not step-for-step identical. Use
  `integrate_literal()` for parity work (TOL_NR_LIB slack per M1).
* SCIPY-DEV-2: Java's error criterion is |yerr_i| <= eps*(|y_i| +
  |h*dydx_i| + tiny); scipy's is |err_i| <= atol + rtol*|y_i|. Mapped as
  rtol=eps, atol=1.0e-10*eps (Java's `tiny`).
* SCIPY-DEV-3: Step accounting. solve_ivp does not expose the
  accepted/rejected split, so `integrate()` sets mNOk to the number of
  accepted steps and mNBad to 0. mHDid/mHNext are set best-effort to the
  final accepted step size. `integrate_literal()` maintains all four
  counters exactly as Java does.
* SCIPY-DEV-4: Intermediate saves. Java saves the *natural* step points
  spaced >= 0.9999*mSaveInterval apart; `integrate()` samples the dense
  output on the exact grid x1 + k*saveInt (plus the endpoint x2), so the
  saved abscissae differ from Java's while the (x, y) pairs themselves
  remain on the same trajectory. Buffer sizing (kMax) matches Java.
* SCIPY-DEV-5: mMinStepSize / mMaxSteps cannot be imposed on solve_ivp
  mid-run; `integrate()` enforces both post-hoc on the accepted-step
  sequence and raises the same UtilException messages as Java.

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
from scipy import integrate as _sp_integrate

try:
    from ._epq_compat import EPQException, F64Array, JavaRandom
except ImportError:
    try:
        from _epq_compat import EPQException, F64Array, JavaRandom  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility._epq_compat import EPQException, F64Array, JavaRandom  # type: ignore

# Sibling port — Java throws gov.nist.microanalysis.Utility.UtilException.
# Filename taken from UTILITY_LEDGER.md "Port file" column (Tier 0).
try:
    from .UtilException_ver1_1_0 import UtilException
except ImportError:
    try:
        from UtilException_ver1_1_0 import UtilException  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility.UtilException_ver1_1_0 import UtilException  # type: ignore


__all__ = ["AdaptiveRungeKutta", "UtilException", "EPQException", "F64Array"]


# Java: java.lang.Double.MAX_VALUE — used by mSaveInterval as the
# "no intermediate saving" sentinel.
_JAVA_DOUBLE_MAX_VALUE: float = sys.float_info.max


class AdaptiveRungeKutta(abc.ABC):
    """Abstract adaptive step size Cash–Karp Runge–Kutta ODE integrator.

    Java: ``abstract public class AdaptiveRungeKutta``. Subclasses must
    implement :meth:`derivatives` (public abstract → un-prefixed name, R1).
    Not thread-safe — use each instance in one and only one thread.
    """

    # ==================================================================
    # Preserved-bug ledger (machine-readable)
    # ==================================================================
    # Each tuple: (id, method, description, has_strict_variant).
    BUG_LEDGER: tuple = ()  # no bugs identified

    # ==================================================================
    # Constructor
    # ==================================================================
    def __init__(self, nVars: int) -> None:
        """AdaptiveRungeKutta - Construct an AdaptiveRungeKutta object to
        solve a differential equation of nVars variables. The implementation
        of derivatives should return nVars derivative values for each x & y.

        Java: ``public AdaptiveRungeKutta(int nVars)``.
        """
        # Java: private final int mNVariables; // The number of differential equations
        self.mNVariables: int = int(nVars)
        # Java: private double mHDid; // Actual step size accomplished in last call to qcStep
        self.mHDid: float = 0.0
        # Java: private double mHNext; // Next step size to try when calling qcStep
        self.mHNext: float = 0.0
        # Java: private double mSaveInterval = Double.MAX_VALUE;
        self.mSaveInterval: float = _JAVA_DOUBLE_MAX_VALUE
        # Java: private double mMinStepSize = 0.0;
        self.mMinStepSize: float = 0.0
        # Java: private double[] mXSave;  (null until integrate() allocates it)
        self.mXSave: Optional[F64Array] = None
        # Java: private double[][] mYSave;  (2-D: kMax x mNVariables)
        self.mYSave: Optional[F64Array] = None
        # Java: private int mNSaved = 0;
        self.mNSaved: int = 0
        # Java: private int mMaxSteps = 10000;
        self.mMaxSteps: int = 10000
        # Java: private int mNOk; // Number of ok steps
        self.mNOk: int = 0
        # Java: private int mNBad; // Number of repeated steps
        self.mNBad: int = 0
        # Temporary work space used by baseStep (Java: lazily allocated, null until first use)
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
    def _require_mutable_f64(arr: object, name: str = "arr") -> None:
        """Type guard for in-place methods (CONVERSION_GUIDE R5).

        Java's ``double[]`` is always a mutable double-precision buffer.
        Our in-place helpers can only honour that contract on numpy
        ndarrays with ``dtype=float64``. Lists, tuples, and wrong-dtype
        arrays would silently no-op or copy. Fail loud.
        (Reference implementation: ``Math2._require_mutable_f64``.)
        """
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{name} must be a numpy ndarray")
        if arr.dtype != np.float64:
            raise TypeError(f"{name} must have dtype float64")
        if not arr.flags.writeable:
            raise TypeError(f"{name} must be writeable")

    # ==================================================================
    # Private helpers (Java `private` → `_` prefix, R1)
    # ==================================================================
    @staticmethod
    def _sign(magnitude: float, sign: float) -> float:
        """Java: ``private double sign(double magnitude, double sign)``.

        NR-style SIGN(a, b): |magnitude| carrying the sign of `sign`,
        with sign == 0.0 treated as positive (Java: ``sign >= 0.0``).
        This is NOT Math.signum — R8 does not apply here.
        """
        return abs(magnitude) if sign >= 0.0 else -abs(magnitude)

    def _baseStep(self, x: float, y: F64Array, dydx: F64Array, h: float,
                  yout: F64Array, yerr: F64Array) -> None:
        """baseStep - Take a single Cash-Karp Runge-Kutta step. Given the
        n=mNDimensions values y[0..n-1] and their derivatives dydx[0..n-1]
        know at x, use a fifth order Cash-Karp Runge-Kutta method to advance
        the solution over an interval h. The resulting y value is returned in
        yout. An estimate of the truncation error is returned in yerr.

        Java: ``private void baseStep(double x, double[] y, double[] dydx,
        double h, double[] yout, double[] yerr)``.
        """
        a2, a3, a4, a5, a6 = 0.2, 0.3, 0.6, 1.0, 0.875
        b21: float = 0.2
        b31, b32 = 3.0 / 40.0, 9.0 / 40.0
        b41, b42, b43 = 0.3, -0.9, 1.2
        b51, b52, b53, b54 = -11.0 / 54.0, 2.5, -70.0 / 27.0, 35.0 / 27.0
        b61, b62, b63, b64, b65 = (1631.0 / 55296.0, 175.0 / 512.0, 575.0 / 13824.0,
                                   44275.0 / 110592.0, 253.0 / 4096.0)
        c1, c3, c4, c6 = 37.0 / 378.0, 250.0 / 621.0, 125.0 / 594.0, 512.0 / 1771.0
        dc1: float = c1 - (2825.0 / 27648.0)
        dc3: float = c3 - (18575.0 / 48384.0)
        dc4: float = c4 - (13525.0 / 55296.0)
        dc5: float = -277.0 / 14336.0
        dc6: float = c6 - 0.25
        # Workspace (Java: lazily allocated on first call)
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
        """qcStep - Take a fifth order Runge-Kutta step with monitoring of
        local truncation error. Input are the dependent variable
        y[0..mNDimensions-1] and its derivatives dydx[0..mNDimensions-1] at
        the starting value of the independent variable x. Also input is the
        attempted step size htry, the required accuracy eps and the vector
        yscal against which the errors are scaled. Upon return, y is replaced
        with the new values, x is returned and mHDid and mHNext are set to
        the actual step size and the size of the next step to try.

        Java: ``private double qcStep(...) throws UtilException``.

        :raises UtilException: When the step size becomes too small.
        :return: The new value of x.
        """
        safety: float = 0.9
        pgrow: float = -0.2
        pshrnk: float = -0.25
        errcon: float = 1.89e-4
        if self.mYErr is None:
            self.mYErr = np.zeros(self.mNVariables, dtype=np.float64)
            self.mQcYTemp = np.zeros(self.mNVariables, dtype=np.float64)
        errmax: float
        h: float = htry
        while True:  # Java: do { ... } while (errmax > 1.0);
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
        """clearWorkspace - null all temporary space to free memory.

        Java: ``private void clearWorkspace()``.
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
    # Public API (trivial accessors / mutators)
    # ==================================================================
    def setSaveInterval(self, interval: float) -> None:
        """setSaveInterval - Set the interval on which to save intermediate
        points on the integrated trajectory. (Use clearSaveInterval to not
        save any intermediate points.) Note: The default is not to save any
        intermediate points.

        Java: ``public void setSaveInterval(double interval)``.
        """
        self.mSaveInterval = abs(float(interval))

    def clearSaveInterval(self) -> None:
        """clearSaveInterval - Return to the default of not saving any
        intermediate points.

        Java: ``public void clearSaveInterval()``.
        """
        self.mSaveInterval = _JAVA_DOUBLE_MAX_VALUE

    def getNSaved(self) -> int:
        """getNSaved - Returns the number of saved values.

        Java: ``public int getNSaved()``.
        """
        return self.mNSaved

    def getX(self, i: int) -> float:
        """getX - Returns the x-coordinate of the i-th saved value
        (where i < getNSaved()).

        Java: ``public double getX(int i)``.
        """
        return float(self.mXSave[i])

    def getY(self, i: int) -> F64Array:
        """getY - returns the getNVariable x y-coordinates of the i-th saved
        values (where i < getNSaved()). Of dimension getNVariables.

        Java: ``public double[] getY(int i)``. Like Java (which returns a
        reference to the inner array), this returns a live view — mutations
        propagate to the saved buffer.
        """
        return self.mYSave[i]

    def setMaxSteps(self, maxSteps: int) -> None:
        """setMaxSteps - Set the maximum number of ODE steps to allow.
        Default is 10000.

        Java: ``public void setMaxSteps(int maxSteps)``.
        """
        self.mMaxSteps = int(maxSteps)

    def setMinStepSize(self, minStep: float) -> None:
        """setMinStepSize - Sets the minimum permissible step size.
        Default is 0.0.

        Java: ``public void setMinStepSize(double minStep)``.
        """
        self.mMinStepSize = abs(float(minStep))

    def getNVariables(self) -> int:
        """getNVariables - Returns the number of variables as set in the
        constructor.

        Java: ``public int getNVariables()``.
        """
        return self.mNVariables

    def getStepCount(self) -> int:
        """getStepCount - Get the total number of steps required to perform
        the previous integrate operation.

        Java: ``public int getStepCount()``.
        """
        return self.mNOk + self.mNBad

    def getGoodStepCount(self) -> int:
        """getGoodStepCount - Get the number of steps leading to results of
        the desired accuracy.

        Java: ``public int getGoodStepCount()``.
        """
        return self.mNOk

    def getBadStepCount(self) -> int:
        """getBadStepCount - Get the number of steps that were needed to be
        subdivided to attain results of the desired accuracy.

        Java: ``public int getBadStepCount()``.
        """
        return self.mNBad

    # ==================================================================
    # integrate — scipy primary (R2a) and faithful literal port (R2b)
    # ==================================================================
    def integrate(self, x1: float, x2: float, ystart: F64Array, eps: float,
                  h1: float) -> F64Array:
        """integrate - Integrate the ODE specified by derivatives using an
        adaptive step size Runge-Kutta algorithm over the independent
        variable interval x1 to x2. ystart contains the initial y values
        (In & out — mutated to hold the final y values). eps is a measure of
        the permissible error. h1 is the initial step size.

        Scipy primary: delegates to ``scipy.integrate.solve_ivp`` (RK45).
        See SCIPY-DEV-1..5 in the module docstring; use
        :meth:`integrate_literal` for the faithful Java port.

        :raises UtilException: Upon too many steps or too small a step.
        :return: The final y values as an array of length getNVariables().
        """
        self._require_mutable_f64(ystart, "ystart")  # R5: ystart is mutated in place
        tiny: float = 1.0e-10 * eps
        self.mNSaved = 0
        self.mNOk = 0
        self.mNBad = 0
        y0: F64Array = np.array(ystart[0:self.mNVariables], dtype=np.float64)

        def _rhs(t: float, yv: F64Array) -> F64Array:
            """Adapter: Java's out-parameter derivatives(x, y, dydx) → f(t, y)."""
            dydx: F64Array = np.zeros(self.mNVariables, dtype=np.float64)
            self.derivatives(float(t), np.asarray(yv, dtype=np.float64), dydx)
            return dydx

        # SCIPY-DEV-2: rtol=eps, atol=tiny approximates Java's
        # eps*(|y| + |h*dydx| + tiny) scaling.
        # first_step: Java uses h = sign(h1, x2 - x1); solve_ivp wants a
        # positive magnitude bounded by the interval width.
        firstStep: Optional[float] = None
        if h1 != 0.0:
            firstStep = min(abs(float(h1)), abs(float(x2) - float(x1)))
        sol = _sp_integrate.solve_ivp(
            _rhs, (float(x1), float(x2)), y0,
            method="RK45",            # SCIPY-DEV-1: Dormand–Prince, not Cash–Karp
            rtol=float(eps), atol=tiny,
            first_step=firstStep,
            dense_output=True,
        )
        if not sol.success:
            raise UtilException(
                f"solve_ivp failed in AdaptiveRungeKutta.integrate: {sol.message}")

        # SCIPY-DEV-5: post-hoc enforcement of mMaxSteps / mMinStepSize.
        nSteps: int = len(sol.t) - 1
        if nSteps > self.mMaxSteps:
            raise UtilException("Too many steps in AdaptiveRungeKutta.integrate")
        if self.mMinStepSize > 0.0 and nSteps > 0:
            minStep: float = float(np.min(np.abs(np.diff(sol.t))))
            if minStep <= self.mMinStepSize:
                raise UtilException("Step size too small in AdaptiveRungeKutta.integrate")

        # SCIPY-DEV-3: rejected-step count unavailable from solve_ivp.
        self.mNOk = nSteps
        self.mNBad = 0
        if nSteps > 0:
            self.mHDid = float(sol.t[-1] - sol.t[-2])
            self.mHNext = self.mHDid

        # SCIPY-DEV-4: intermediate saves on the exact interval grid.
        if self.mSaveInterval != _JAVA_DOUBLE_MAX_VALUE:
            # Java: kMax = (int) Math.round((Math.abs(x2 - x1) + mSaveInterval) / mSaveInterval);
            kMax: int = int(math.floor(((abs(float(x2) - float(x1)) + self.mSaveInterval) / self.mSaveInterval) + 0.5))  # R7: Math.round
            saveInt: float = self._sign(self.mSaveInterval, float(x2) - float(x1))
            self.mXSave = np.zeros(kMax, dtype=np.float64)
            self.mYSave = np.zeros((kMax, self.mNVariables), dtype=np.float64)
            ts: list = []
            k: int = 0
            while k < kMax - 1:
                t: float = float(x1) + k * saveInt
                # Keep grid points strictly inside [x1, x2] (either direction).
                if (t - float(x2)) * (t - float(x1)) > 0.0:
                    break
                ts.append(t)
                k += 1
            ts.append(float(x2))  # final point always saved (matches Java's endpoint save)
            ys: F64Array = sol.sol(np.asarray(ts, dtype=np.float64))  # shape (n, len(ts))
            self.mNSaved = len(ts)
            self.mXSave[0:self.mNSaved] = np.asarray(ts, dtype=np.float64)
            self.mYSave[0:self.mNSaved, :] = ys.T
        else:
            self.mXSave = None
            self.mYSave = None

        yFinal: F64Array = np.array(sol.y[:, -1], dtype=np.float64)
        # Java: System.arraycopy(y, 0, ystart, 0, mNVariables);
        ystart[0:self.mNVariables] = yFinal[0:self.mNVariables]
        self._clearWorkspace()
        return yFinal

    def integrate_literal(self, x1: float, x2: float, ystart: F64Array,
                          eps: float, h1: float) -> F64Array:
        """integrate - Faithful line-for-line port of the Java method.
        Integrate the ODE specified by derivatives using the adaptive step
        size Runge-Kutta algorithm over the independent variable interval x1
        to x2. ystart contains the initial y values (In & out). eps is a
        measure of the permissible error. h1 is the initial step size.

        Java: ``public double[] integrate(double x1, double x2,
        double[] ystart, double eps, double h1) throws UtilException``.

        :raises UtilException: Upon too many steps or too small a step.
        :return: The final y values as an array of length getNVariables().
        """
        self._require_mutable_f64(ystart, "ystart")  # R5: ystart is mutated in place
        tiny: float = 1.0e-10 * eps
        yscal: F64Array = np.zeros(self.mNVariables, dtype=np.float64)
        dydx: F64Array = np.zeros(self.mNVariables, dtype=np.float64)
        y: F64Array = np.zeros(self.mNVariables, dtype=np.float64)
        x: float = float(x1)
        h: float = self._sign(h1, float(x2) - float(x1))
        xsav: float = 0.0
        saveInt: float = _JAVA_DOUBLE_MAX_VALUE
        kMax: int = 0
        self.mNSaved = 0
        self.mNOk = 0
        self.mNBad = 0
        # Java: System.arraycopy(ystart, 0, y, 0, mNVariables);
        y[0:self.mNVariables] = ystart[0:self.mNVariables]
        if self.mSaveInterval != _JAVA_DOUBLE_MAX_VALUE:
            # Java: kMax = (int) Math.round((Math.abs(x2 - x1) + mSaveInterval) / mSaveInterval);
            kMax = int(math.floor(((abs(float(x2) - float(x1)) + self.mSaveInterval) / self.mSaveInterval) + 0.5))  # R7: Math.round
            saveInt = self._sign(self.mSaveInterval, float(x2) - float(x1))
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
            if (((x + h) - float(x2)) * ((x + h) - float(x1))) > 0.0:
                h = float(x2) - x
            x = self._qcStep(x, y, dydx, h, eps, yscal)
            if self.mHDid == h:
                self.mNOk += 1
            else:
                self.mNBad += 1
            if ((x - float(x2)) * (float(x2) - float(x1))) >= 0.0:
                # Java: System.arraycopy(y, 0, ystart, 0, mNVariables);
                ystart[0:self.mNVariables] = y[0:self.mNVariables]
                if kMax != 0:
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
    # Abstract API (public abstract → un-prefixed name, R1)
    # ==================================================================
    @abc.abstractmethod
    def derivatives(self, x: float, y: F64Array, dydx: F64Array) -> None:
        """derivatives - The derived class provides an implementation of the
        derivatives function. x & y[] are input and the user provided
        implementation of derivatives is resposible for returning the
        derivatives in the array dydx. The lengths of y and dydx are equal
        to mNDimensions.

        Java: ``abstract public void derivatives(double x, double[] y,
        double[] dydx)``.

        :param x: float - In
        :param y: F64Array - In (of dimension mNDimensions)
        :param dydx: F64Array - Out (of dimension mNDimensions)
        """
        ...