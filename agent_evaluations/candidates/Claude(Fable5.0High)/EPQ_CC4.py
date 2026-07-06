r"""
AdaptiveRungeKutta_ver1_1_0.py — Python port of gov.nist.microanalysis.Utility.AdaptiveRungeKutta

Guide version : 1
Generation    : 1
Port-code fixes: 0

CHANGES (deliberate deviations, per R10)
----------------------------------------
* SCIPY-DEV-1 (`integrate`): The primary `integrate()` delegates to
  scipy.integrate.solve_ivp with the "RK45" (Dormand-Prince) stepper,
  not the Cash-Karp tableau used by the Java source (scipy ships no
  Cash-Karp integrator). Trajectories agree within tolerance but are
  not step-for-step identical. `integrate_literal()` is the faithful
  Cash-Karp port; the parity harness must use `integrate_literal`.
* SCIPY-DEV-2 (`integrate`): Java scales error per-component as
  yscal[i] = |y[i]| + |dydx[i]*h| + tiny. solve_ivp uses the standard
  atol + rtol*|y| criterion. Mapped as rtol=eps, atol=1.0e-10*eps
  (Java's `tiny`).
* SCIPY-DEV-3 (`integrate`): solve_ivp does not report rejected
  (subdivided) steps. After the scipy path, mNOk is set to the number
  of accepted steps (len(sol.t) - 1) and mNBad to 0; getStepCount()
  therefore counts accepted steps only. mHDid/mHNext are set from the
  last accepted step size. Use `integrate_literal` when exact
  good/bad-step accounting matters.
* SCIPY-DEV-4 (`integrate`): mMinStepSize cannot be imposed on
  solve_ivp; a solver failure (sol.status != 0) is reported as
  UtilException, and exceeding mMaxSteps accepted steps raises the
  same "Too many steps" UtilException as the Java loop.
* R5: `integrate` / `integrate_literal` mutate `ystart` in place
  (Java: System.arraycopy(y, 0, ystart, 0, mNVariables)), so both
  call `_require_mutable_f64(ystart)`. Java accepted any double[];
  the Python port requires a writeable float64 ndarray and fails
  loud on lists/tuples rather than silently not mutating.
* Double.MAX_VALUE is mapped to sys.float_info.max (bit-identical,
  1.7976931348623157e308) as `_DOUBLE_MAX_VALUE`; the Java sentinel
  comparisons (`mSaveInterval != Double.MAX_VALUE`) are preserved as
  exact comparisons against that constant.

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

# Sibling port — filename from UTILITY_LEDGER.md "Port file" column (Tier 0).
try:
    from .UtilException_ver1_1_0 import UtilException
except ImportError:
    try:
        from UtilException_ver1_1_0 import UtilException  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility.UtilException_ver1_1_0 import UtilException  # type: ignore


__all__ = ["AdaptiveRungeKutta", "UtilException", "F64Array"]


# Java Double.MAX_VALUE == sys.float_info.max == 1.7976931348623157e308.
_DOUBLE_MAX_VALUE: float = sys.float_info.max


class AdaptiveRungeKutta(abc.ABC):
    """Adaptive step size Cash-Karp Runge-Kutta ODE integrator.

    Abstract base class (Java: ``abstract public class``). Subclasses
    must implement :meth:`derivatives` — the *public* abstract Java
    name, deliberately NOT underscore-prefixed (R1).

    NOTE: Like the Java original, instances are not thread-safe.
    """

    # ==================================================================
    # Preserved-bug ledger (machine-readable, per BUG_GUIDE.md)
    # ==================================================================
    # Java source reviewed line-by-line (AdaptiveRungeKutta.java, 439
    # lines): no dead branches, always-true conditions, off-by-one or
    # sign errors identified. The exact float comparisons
    # `mHDid == h` (qcStep/integrate) and
    # `mSaveInterval != Double.MAX_VALUE` are correct by construction
    # (both sides assigned, never recomputed) and are preserved as-is.
    BUG_LEDGER: tuple = ()  # no bugs identified

    # ==================================================================
    # Internal guards
    # ==================================================================
    @staticmethod
    def _require_mutable_f64(arr, name: str = "arr") -> None:
        """Type guard for in-place methods (CONVERSION_GUIDE R5).

        Java's ``double[]`` is always a mutable double-precision buffer.
        Our in-place methods can only honour that contract on numpy
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
        """AdaptiveRungeKutta - Construct an AdaptiveRungeKutta object to
        solve a differential equation of nVars variables. The
        implementation of derivatives should return nVars derivative
        values for each x & y.

        Java: ``public AdaptiveRungeKutta(int nVars)``
        """
        super().__init__()
        # private final int — the number of differential equations
        self._mNVariables: int = int(nVars)
        # Actual step size accomplished in last call to qcStep
        self._mHDid: float = 0.0
        # Next step size to try when calling qcStep
        self._mHNext: float = 0.0
        self._mSaveInterval: float = _DOUBLE_MAX_VALUE
        self._mMinStepSize: float = 0.0
        self._mXSave: Optional[F64Array] = None          # double[]
        self._mYSave: Optional[F64Array] = None          # double[][]
        self._mNSaved: int = 0
        self._mMaxSteps: int = 10000
        self._mNOk: int = 0    # Number of ok steps
        self._mNBad: int = 0   # Number of repeated steps
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
    # Private helpers (Java `private` → `_` prefix, R1)
    # ==================================================================
    def _sign(self, magnitude: float, sign: float) -> float:
        """Java: ``private double sign(double magnitude, double sign)``.

        NR-style SIGN: ``sign >= 0.0 ? Math.abs(magnitude) :
        -Math.abs(magnitude)``. This is the class's own helper, not
        Math.signum, so the R8 zero-case mapping does not apply — in
        Java ``-0.0 >= 0.0`` is true and ``abs()`` reproduces that.
        """
        return abs(magnitude) if sign >= 0.0 else -abs(magnitude)

    def _baseStep(self, x: float, y: F64Array, dydx: F64Array, h: float,
                  yout: F64Array, yerr: F64Array) -> None:
        """baseStep - Take a single Cash-Karp Runge-Kutta step. Given the
        n=mNDimensions values y[0..n-1] and their derivatives
        dydx[0..n-1] know at x, use a fifth order Cash-Karp Runge-Kutta
        method to advance the solution over an interval h. The resulting
        y value is returned in yout. An estimate of the truncation error
        is returned in yerr.

        Java: ``private void baseStep(double x, double[] y,
        double[] dydx, double h, double[] yout, double[] yerr)``
        """
        # R5: yout and yerr are mutated in place.
        AdaptiveRungeKutta._require_mutable_f64(yout, "yout")
        AdaptiveRungeKutta._require_mutable_f64(yerr, "yerr")
        a2: float = 0.2; a3: float = 0.3; a4: float = 0.6; a5: float = 1.0; a6: float = 0.875
        b21: float = 0.2
        b31: float = 3.0 / 40.0; b32: float = 9.0 / 40.0
        b41: float = 0.3; b42: float = -0.9; b43: float = 1.2
        b51: float = -11.0 / 54.0; b52: float = 2.5; b53: float = -70.0 / 27.0; b54: float = 35.0 / 27.0
        b61: float = 1631.0 / 55296.0; b62: float = 175.0 / 512.0; b63: float = 575.0 / 13824.0
        b64: float = 44275.0 / 110592.0; b65: float = 253.0 / 4096.0
        c1: float = 37.0 / 378.0; c3: float = 250.0 / 621.0; c4: float = 125.0 / 594.0; c6: float = 512.0 / 1771.0
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
            self._mYTemp[i] = y[i] + (h * ((b51 * dydx[i]) + (b52 * self._mWs2[i]) + (b53 * self._mWs3[i]) + (b54 * self._mWs4[i])))
        # Fifth step
        self.derivatives(x + (a5 * h), self._mYTemp, self._mWs5)
        for i in range(self._mNVariables):
            self._mYTemp[i] = y[i] + (h * ((b61 * dydx[i]) + (b62 * self._mWs2[i]) + (b63 * self._mWs3[i]) + (b64 * self._mWs4[i]) + (b65 * self._mWs5[i])))
        # Sixth step
        self.derivatives(x + (a6 * h), self._mYTemp, self._mWs6)
        for i in range(self._mNVariables):
            yout[i] = y[i] + (h * ((c1 * dydx[i]) + (c3 * self._mWs3[i]) + (c4 * self._mWs4[i]) + (c6 * self._mWs6[i])))
        # Estimate the error
        for i in range(self._mNVariables):
            yerr[i] = h * ((dc1 * dydx[i]) + (dc3 * self._mWs3[i]) + (dc4 * self._mWs4[i]) + (dc5 * self._mWs5[i]) + (dc6 * self._mWs6[i]))

    def _qcStep(self, x: float, y: F64Array, dydx: F64Array, htry: float,
                eps: float, yscal: F64Array) -> float:
        """qcStep - Take a fifth order Runge-Kutta step with monitoring of
        local truncation error. Input are the dependent variable
        y[0..mNDimensions-1] and its derivatives dydx[0..mNDimensions-1]
        at the starting value of the independent variable x. Also input
        is the attempted step size htry, the required accuracy eps and
        the vector yscal against which the errors are scaled. Upon
        return, y is replaced with the new values, x is returned and
        mHDid and mHNext are set to the actual step size and the size of
        the next step to try.

        Java: ``private double qcStep(...) throws UtilException``

        :param x: (In) independent variable
        :param y: (In,Out) dependent variable — mutated in place
        :param dydx: (In) derivative at x
        :param htry: The step size to attempt
        :param eps: Desired accuracy
        :param yscal: (In) Error scaling vector
        :raises UtilException: When the step size becomes too small
        :return: The new value of x
        """
        # R5: y is mutated in place (System.arraycopy into y below).
        AdaptiveRungeKutta._require_mutable_f64(y, "y")
        safety: float = 0.9
        pgrow: float = -0.2
        pshrnk: float = -0.25
        errcon: float = 1.89e-4
        if self._mYErr is None:
            self._mYErr = np.zeros(self._mNVariables, dtype=np.float64)
            self._mQcYTemp = np.zeros(self._mNVariables, dtype=np.float64)
        errmax: float
        h: float = htry
        while True:  # Java do { ... } while (errmax > 1.0);
            self._baseStep(x, y, dydx, h, self._mQcYTemp, self._mYErr)
            errmax = 0.0
            for i in range(self._mNVariables):
                errmax = max(errmax, abs(self._mYErr[i] / yscal[i]))
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
        self._mHNext = (safety * h * math.pow(errmax, pgrow) if errmax > errcon else 5.0 * h)
        self._mHDid = h
        x += h
        # Java: System.arraycopy(mQcYTemp, 0, y, 0, mNVariables);
        y[0:self._mNVariables] = self._mQcYTemp[0:self._mNVariables]
        return x

    def _clearWorkspace(self) -> None:
        """clearWorkspace - null all temporary space to free memory

        Java: ``private void clearWorkspace()``
        """
        self._mWs2 = None
        self._mWs3 = None
        self._mWs4 = None
        self._mWs5 = None
        self._mWs6 = None
        self._mYTemp = None
        self._mYErr = None
        self._mQcYTemp = None

    # ==================================================================
    # Public accessors / mutators
    # ==================================================================
    def setSaveInterval(self, interval: float) -> None:
        """setSaveInterval - Set the interval on which to save intermediate
        points on the integrated trajectory. (Use clearSaveInterval to
        not save any intermediate points.) Note: The default is not to
        save any intermediate points.

        Java: ``public void setSaveInterval(double interval)``
        """
        self._mSaveInterval = abs(float(interval))

    def clearSaveInterval(self) -> None:
        """clearSaveInterval - Return to the default of not saving any
        intermediate points.

        Java: ``public void clearSaveInterval()``
        """
        self._mSaveInterval = _DOUBLE_MAX_VALUE

    def getNSaved(self) -> int:
        """getNSaved - Returns the number of saved values.

        Java: ``public int getNSaved()``
        """
        return self._mNSaved

    def getX(self, i: int) -> float:
        """getX - Returns the x-coordinate of the i-th saved value

        Java: ``public double getX(int i)``

        :param i: Where i < getNSaved()
        """
        return float(self._mXSave[i])

    def getY(self, i: int) -> F64Array:
        """getY - returns the getNVariable x y-coordinates of the i-th
        saved values.

        Java: ``public double[] getY(int i)``

        :param i: Where i < getNSaved()
        :return: Of dimension getNVariables (a view into the internal
                 save buffer, matching Java's reference semantics)
        """
        return self._mYSave[i]

    def setMaxSteps(self, maxSteps: int) -> None:
        """setMaxSteps - Set the maximum number of ODE steps to allow.
        Default is 10000.

        Java: ``public void setMaxSteps(int maxSteps)``
        """
        self._mMaxSteps = int(maxSteps)

    def setMinStepSize(self, minStep: float) -> None:
        """setMinStepSize - Sets the minimum permissible step size.
        Default is 0.0.

        Java: ``public void setMinStepSize(double minStep)``
        """
        self._mMinStepSize = abs(float(minStep))

    def getNVariables(self) -> int:
        """getNVariables - Returns the number of variables as set in the
        constructor.

        Java: ``public int getNVariables()``
        """
        return self._mNVariables

    def getStepCount(self) -> int:
        """getStepCount - Get the total number of steps required to
        perform the previous integrate operation.

        Java: ``public int getStepCount()``
        """
        return self._mNOk + self._mNBad

    def getGoodStepCount(self) -> int:
        """getGoodStepCount - Get the number of steps leading to results
        of the desired accuracy.

        Java: ``public int getGoodStepCount()``
        """
        return self._mNOk

    def getBadStepCount(self) -> int:
        """getBadStepCount - Get the number of steps that were needed to
        be subdivided to attain results of the desired accuracy.

        Java: ``public int getBadStepCount()``
        """
        return self._mNBad

    # ==================================================================
    # integrate — scipy primary AND literal Java port (R2)
    # ==================================================================
    def integrate(self, x1: float, x2: float, ystart: F64Array, eps: float,
                  h1: float) -> F64Array:
        """integrate - Integrate the ODE specified by derivatives using the
        adaptive step size Runge-Kutta algorithm over the independent
        variable interval x1 to x2. ystart contains the initial y values.
        eps is measure of the permissible error. h is the initial step
        size.

        Primary (scipy) variant — delegates to
        ``scipy.integrate.solve_ivp(method="RK45")``. See module
        docstring CHANGES: SCIPY-DEV-1 (Dormand-Prince, not Cash-Karp),
        SCIPY-DEV-2 (error-scaling mapping), SCIPY-DEV-3 (step
        accounting), SCIPY-DEV-4 (mMinStepSize not enforced). For
        step-for-step Java parity use :meth:`integrate_literal`.

        :param x1: Start of the integration range
        :param x2: End of the integration range
        :param ystart: (In & out) The initial y value — mutated in place
        :param eps: The permissible relative error
        :param h1: The initial step size
        :return: The final y values as an array of length getNVariables().
        :raises UtilException: Upon too many steps or solver failure
        """
        # R5: ystart is mutated in place on success (Java arraycopy).
        AdaptiveRungeKutta._require_mutable_f64(ystart, "ystart")
        tiny: float = 1.0e-10 * eps  # Java's `tiny`, reused as atol (SCIPY-DEV-2)
        self._mNSaved = 0
        self._mNOk = 0
        self._mNBad = 0

        def _fun(t: float, yv: F64Array) -> F64Array:
            """Adapt the Java out-parameter contract to scipy's f(t, y)."""
            dydx: F64Array = np.zeros(self._mNVariables, dtype=np.float64)
            self.derivatives(float(t), np.asarray(yv, dtype=np.float64), dydx)
            return dydx

        sol = _sp_integrate.solve_ivp(
            _fun, (float(x1), float(x2)),
            np.asarray(ystart, dtype=np.float64),
            method="RK45",              # SCIPY-DEV-1
            rtol=float(eps),            # SCIPY-DEV-2
            atol=tiny,                  # SCIPY-DEV-2
            first_step=abs(float(h1)) if h1 != 0.0 else None,
            dense_output=(self._mSaveInterval != _DOUBLE_MAX_VALUE),
        )
        if sol.status != 0:
            # SCIPY-DEV-4: solver failure maps onto the Java failure modes.
            raise UtilException("Step size too small in AdaptiveRungeKutta.integrate")
        nAccepted: int = len(sol.t) - 1
        if nAccepted > self._mMaxSteps:
            raise UtilException("Too many steps in AdaptiveRungeKutta.integrate")
        # SCIPY-DEV-3: rejected-step count is unavailable from solve_ivp.
        self._mNOk = nAccepted
        self._mNBad = 0
        self._mHDid = float(sol.t[-1] - sol.t[-2]) if nAccepted > 0 else 0.0
        self._mHNext = self._mHDid
        # Reproduce the Java save-buffer contract via dense output.
        if self._mSaveInterval != _DOUBLE_MAX_VALUE:
            kMax: int = int(math.floor(((abs(x2 - x1) + self._mSaveInterval) / self._mSaveInterval) + 0.5))  # R7
            saveInt: float = self._sign(self._mSaveInterval, x2 - x1)
            self._mXSave = np.zeros(kMax, dtype=np.float64)
            self._mYSave = np.zeros((kMax, self._mNVariables), dtype=np.float64)
            xs: float = float(x1)
            while self._mNSaved < kMax - 1:
                # stop before overrunning x2 in the direction of travel
                if (xs - float(x2)) * (float(x2) - float(x1)) > 0.0:
                    break
                self._mXSave[self._mNSaved] = xs
                self._mYSave[self._mNSaved, :] = sol.sol(xs)
                self._mNSaved += 1
                xs += saveInt
            # Final point, mirroring the Java end-of-integration save.
            self._mNSaved = min(self._mNSaved, kMax - 1)
            self._mXSave[self._mNSaved] = float(sol.t[-1])
            self._mYSave[self._mNSaved, :] = sol.y[:, -1]
            self._mNSaved += 1
        y: F64Array = np.array(sol.y[:, -1], dtype=np.float64)
        # Java: System.arraycopy(y, 0, ystart, 0, mNVariables);
        ystart[0:self._mNVariables] = y[0:self._mNVariables]
        return y

    def integrate_literal(self, x1: float, x2: float, ystart: F64Array,
                          eps: float, h1: float) -> F64Array:
        """integrate - Integrate the ODE specified by derivatives using the
        adaptive step size Runge-Kutta algorithm over the independent
        variable interval x1 to x2. ystart contains the initial y values.
        eps is measure of the permissible error. h is the initial step
        size.

        Faithful line-for-line port of the Java Cash-Karp driver.

        Java: ``public double[] integrate(double x1, double x2,
        double[] ystart, double eps, double h1) throws UtilException``

        :param x1: Start of the integration range
        :param x2: End of the integration range
        :param ystart: (In & out) The initial y value — mutated in place
        :param eps: The permissible relative error
        :param h1: The initial step size
        :return: The final y values as an array of length getNVariables().
        :raises UtilException: Upon too many steps or too small a step
        """
        # R5: ystart is mutated in place on success (Java arraycopy).
        AdaptiveRungeKutta._require_mutable_f64(ystart, "ystart")
        tiny: float = 1.0e-10 * eps
        yscal: F64Array = np.zeros(self._mNVariables, dtype=np.float64)
        dydx: F64Array = np.zeros(self._mNVariables, dtype=np.float64)
        y: F64Array = np.zeros(self._mNVariables, dtype=np.float64)
        x: float = float(x1)
        h: float = self._sign(h1, x2 - x1)
        xsav: float = 0.0
        saveInt: float = _DOUBLE_MAX_VALUE
        kMax: int = 0
        self._mNSaved = 0
        self._mNOk = 0
        self._mNBad = 0
        # Java: System.arraycopy(ystart, 0, y, 0, mNVariables);
        y[0:self._mNVariables] = ystart[0:self._mNVariables]
        if self._mSaveInterval != _DOUBLE_MAX_VALUE:
            # Java: kMax = (int) Math.round((Math.abs(x2 - x1) + mSaveInterval) / mSaveInterval);
            # R7: Math.round(x) → int(math.floor(x + 0.5))
            kMax = int(math.floor(((abs(x2 - x1) + self._mSaveInterval) / self._mSaveInterval) + 0.5))
            saveInt = self._sign(self._mSaveInterval, x2 - x1)
            xsav = x - (2.0 * saveInt)  # to ensure that the first step is
            #                             saved...
            self._mXSave = np.zeros(kMax, dtype=np.float64)
            self._mYSave = np.zeros((kMax, self._mNVariables), dtype=np.float64)
        for step in range(self._mMaxSteps):
            # Save the necessary points
            if (kMax != 0) and (self._mNSaved < kMax) and (abs(x - xsav) >= (0.9999 * self._mSaveInterval)):
                self._mXSave[self._mNSaved] = x
                # Java: System.arraycopy(y, 0, mYSave[mNSaved], 0, mNVariables);
                self._mYSave[self._mNSaved, 0:self._mNVariables] = y[0:self._mNVariables]
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
                # Java: System.arraycopy(y, 0, ystart, 0, mNVariables);
                ystart[0:self._mNVariables] = y[0:self._mNVariables]
                if kMax != 0:
                    self._mNSaved = min(self._mNSaved, kMax - 1)
                    self._mXSave[self._mNSaved] = x
                    self._mYSave[self._mNSaved, 0:self._mNVariables] = y[0:self._mNVariables]
                    self._mNSaved += 1
                self._clearWorkspace()
                return y
            if abs(self._mHNext) <= self._mMinStepSize:
                raise UtilException("Step size too small in AdaptiveRungeKutta.integrate")
            h = self._mHNext
        raise UtilException("Too many steps in AdaptiveRungeKutta.integrate")

    # ==================================================================
    # Abstract method (public abstract → NO underscore, R1)
    # ==================================================================
    @abc.abstractmethod
    def derivatives(self, x: float, y: F64Array, dydx: F64Array) -> None:
        """derivatives - The derived class provides an implementation of
        the derivatives function. x & y[] are input and the user provided
        implementation of derivatives is resposible for returning the
        derivatives in the array dydx. The lengths of y and dydx are
        equal to mNDimensions.

        Java: ``abstract public void derivatives(double x, double[] y,
        double[] dydx)``

        :param x: In
        :param y: In (of dimension mNDimensions)
        :param dydx: Out (of dimension mNDimensions) — implementation
                     must write results into this array in place
        """
        ...