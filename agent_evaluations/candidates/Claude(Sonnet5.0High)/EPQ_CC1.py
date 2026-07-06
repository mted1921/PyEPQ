r"""
AdaptiveRungeKutta_ver1_1_0.py — Python port of gov.nist.microanalysis.Utility.AdaptiveRungeKutta

Guide version : 1
Generation    : 1
Port-code fixes: 0

CHANGES FROM JAVA (see CONVERSION_GUIDE.md R1-R10)
---------------------------------------------------
* R2: `integrate()` is the scipy-primary implementation
  (`scipy.integrate.solve_ivp`, Dormand-Prince RK45). `integrate_literal()`
  is the exact line-for-line Cash-Karp / Numerical-Recipes port (via the
  faithfully-ported private helpers `_baseStep` / `_qcStep`) and is the
  parity-harness reference. See SCIPY-DEV-1..3 in BUG_LEDGER below.
* R3: `EPQException` is imported from `_epq_compat` per the standard
  template, but the two `throws UtilException` sites in the Java source
  raise `UtilException` instead, imported from the sibling port
  `UtilException_ver1_1_0.py` (see UTILITY_LEDGER.md Tier 0), matching
  the Java exception type exactly.
* R7: `Math.round(...)` in the `kMax` calculation is replaced with
  `int(math.floor(x + 0.5))` (exact half-up semantics; Python `round()`
  uses banker's rounding and would diverge at exact .5 boundaries).
* R5: `integrate()` / `integrate_literal()` mutate the caller-supplied
  `ystart` array in place (matching Java's `System.arraycopy(y, 0,
  ystart, 0, mNVariables)`), guarded by `_require_mutable_f64`.
* R9: every field, parameter, return type, and non-obvious local is
  annotated; literal numeric constants (Cash-Karp tableau, NR
  step-control constants) are left unannotated, matching the reference
  style in Math2_ver8_1_5.py.
* R2 member table: `equals`/`hashCode`/`toString` are not overridden in
  the Java source, so no dunder/named-alias pair is added (nothing to
  map).
* R4: not applicable -- the Java source declares no overloaded methods,
  so no type-suffixed (`_vv`/`_arr`/...) split is needed.
* No JAVA-BUG-N entries: `baseStep`, `qcStep`, and `integrate` were
  audited line-by-line against Press/Teukolsky/Vetterling/Flannery,
  "Numerical Recipes in C", 2nd ed., pp 714-722 (`rkck`, `rkqs`,
  `odeint`), which this class is a direct port of. No dead branches,
  always-true conditions, off-by-one errors, or sign errors were found.

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

import abc, math
import numpy as np
from typing import Optional, Sequence, Union, Callable
try:
    from ._epq_compat import EPQException, F64Array, JavaRandom
except ImportError:
    try:
        from _epq_compat import EPQException, F64Array, JavaRandom  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility._epq_compat import EPQException, F64Array, JavaRandom  # type: ignore

# Additional imports beyond the standard template:
import sys
from numpy.typing import ArrayLike
from scipy import integrate as _sp_integrate

# Sibling port (UTILITY_LEDGER.md Tier 0 -- "Port file" column: UtilException_ver1_1_0.py).
# Java's `integrate`/`qcStep` declare `throws UtilException`, not EPQException.
try:
    from .UtilException_ver1_1_0 import UtilException
except ImportError:
    try:
        from UtilException_ver1_1_0 import UtilException  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility.UtilException_ver1_1_0 import UtilException  # type: ignore


__all__ = ["AdaptiveRungeKutta", "EPQException", "UtilException", "JavaRandom", "F64Array"]


class AdaptiveRungeKutta(abc.ABC):
    """Python port of gov.nist.microanalysis.Utility.AdaptiveRungeKutta.

    Abstract base class (Java: `abstract public class`). Subclasses must
    implement `derivatives()`. See the verbatim Javadoc above for the
    algorithm description and usage example.
    """

    # ==================================================================
    # Preserved-bug / deviation ledger (machine-readable)
    # ==================================================================
    # Each tuple: (id, method, description, has_strict_variant).
    # No Java bugs (JAVA-BUG-N) were identified in this class -- see the
    # audit note in the module docstring CHANGES section. The entries
    # below document deliberate R2 scipy-substitution deviations in
    # integrate() relative to integrate_literal().
    BUG_LEDGER: tuple = (
        ("SCIPY-DEV-1", "integrate",
         "Uses scipy.integrate.solve_ivp (Dormand-Prince RK45) instead of "
         "the Cash-Karp RK45 tableau + Numerical-Recipes step controller "
         "used by integrate_literal(). Trajectories agree to eps-scale "
         "tolerance but are NOT bit-identical -- the two RK45 methods have "
         "different embedded-error coefficients. Use integrate_literal() "
         "for exact Java parity.", False),
        ("SCIPY-DEV-2", "integrate",
         "getBadStepCount() (mNBad) is always 0 after integrate(): scipy's "
         "public solve_ivp API exposes only the accepted-step time grid, "
         "not the count of internally rejected steps that "
         "integrate_literal() tracks via qcStep's retry loop.", False),
        ("SCIPY-DEV-3", "integrate",
         "mMinStepSize is not enforced step-by-step (scipy's explicit RK45 "
         "solver accepts no min_step parameter, unlike its implicit "
         "solvers); a step size that collapses is instead surfaced as "
         "result.success == False and raised as UtilException after the "
         "solve, rather than via qcStep's per-step xnew == x underflow "
         "guard.", False),
    )

    def __init__(self, nVars: int) -> None:
        """AdaptiveRungeKutta(int nVars) -- construct a solver for an ODE
        system of `nVars` variables. Subclasses provide `derivatives()`,
        returning `nVars` derivative values for each x & y.
        """
        self._mNVariables: int = nVars  # private final int mNVariables
        self._mHDid: float = 0.0
        self._mHNext: float = 0.0
        self._mSaveInterval: float = sys.float_info.max  # Java: Double.MAX_VALUE
        self._mMinStepSize: float = 0.0
        self._mXSave: Optional[F64Array] = None
        self._mYSave: Optional[F64Array] = None
        self._mNSaved: int = 0
        self._mMaxSteps: int = 10000
        self._mNOk: int = 0
        self._mNBad: int = 0
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
    # Internal guard (CONVERSION_GUIDE R5)
    # ==================================================================
    @staticmethod
    def _require_mutable_f64(arr: object, name: str = "arr") -> None:
        """Type guard for in-place methods.

        Java's `double[]` is always a mutable double-precision buffer.
        Our in-place methods can only honour that contract on numpy
        ndarrays with dtype=float64. Lists, tuples, and wrong-dtype
        arrays would silently no-op or copy. Fail loud.
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
        Numerical Recipes SIGN(a, b) macro: |magnitude| with the sign of
        `sign` (sign >= 0.0 -> positive).
        """
        return abs(magnitude) if sign >= 0.0 else -abs(magnitude)

    def _baseStep(self, x: float, y: F64Array, dydx: F64Array, h: float,
                  yout: F64Array, yerr: F64Array) -> None:
        """baseStep - Take a single Cash-Karp Runge-Kutta step. Given the
        n=mNVariables values y[0..n-1] and their derivatives dydx[0..n-1]
        known at x, use a fifth order Cash-Karp Runge-Kutta method to
        advance the solution over an interval h. The resulting y value is
        returned in yout. An estimate of the truncation error is returned
        in yerr.
        """
        a2 = 0.2
        a3 = 0.3
        a4 = 0.6
        a5 = 1.0
        a6 = 0.875
        b21 = 0.2
        b31 = 3.0 / 40.0
        b32 = 9.0 / 40.0
        b41 = 0.3
        b42 = -0.9
        b43 = 1.2
        b51 = -11.0 / 54.0
        b52 = 2.5
        b53 = -70.0 / 27.0
        b54 = 35.0 / 27.0
        b61 = 1631.0 / 55296.0
        b62 = 175.0 / 512.0
        b63 = 575.0 / 13824.0
        b64 = 44275.0 / 110592.0
        b65 = 253.0 / 4096.0
        c1 = 37.0 / 378.0
        c3 = 250.0 / 621.0
        c4 = 125.0 / 594.0
        c6 = 512.0 / 1771.0
        dc1 = c1 - (2825.0 / 27648.0)
        dc3 = c3 - (18575.0 / 48384.0)
        dc4 = c4 - (13525.0 / 55296.0)
        dc5 = -277.0 / 14336.0
        dc6 = c6 - 0.25
        # Workspace
        n: int = self._mNVariables
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
        """qcStep - Take a fifth order Runge-Kutta step with monitoring of
        local truncation error. Input are the dependent variable
        y[0..mNVariables-1] and its derivatives dydx[0..mNVariables-1] at
        the starting value of the independent variable x. Also input is
        the attempted step size htry, the required accuracy eps and the
        vector yscal against which the errors are scaled. Upon return, y
        is replaced with the new values, x is returned and mHDid and
        mHNext are set to the actual step size and the size of the next
        step to try.

        Raises UtilException when the step size becomes too small.
        """
        safety = 0.9
        pgrow = -0.2
        pshrnk = -0.25
        errcon = 1.89e-4
        n: int = self._mNVariables
        if self._mYErr is None:
            self._mYErr = np.zeros(n, dtype=np.float64)
            self._mQcYTemp = np.zeros(n, dtype=np.float64)
        h: float = htry
        errmax: float = 0.0
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
                    raise UtilException("Step size underflow in AdaptiveRungeKutta.qcStep.")
            if not (errmax > 1.0):
                break
        self._mHNext = safety * h * math.pow(errmax, pgrow) if errmax > errcon else 5.0 * h
        self._mHDid = h
        x += h
        y[:] = self._mQcYTemp  # System.arraycopy(mQcYTemp, 0, y, 0, mNVariables)
        return x

    def _clearWorkspace(self) -> None:
        """clearWorkspace - null all temporary space to free memory."""
        self._mWs2 = None
        self._mWs3 = None
        self._mWs4 = None
        self._mWs5 = None
        self._mWs6 = None
        self._mYTemp = None
        self._mYErr = None
        self._mQcYTemp = None

    # ==================================================================
    # Public API (Java: public)
    # ==================================================================
    def setSaveInterval(self, interval: float) -> None:
        """Set the interval on which to save intermediate points on the
        integrated trajectory. (Use clearSaveInterval() to not save any
        intermediate points.) Note: the default is not to save any
        intermediate points.
        """
        self._mSaveInterval = abs(interval)

    def clearSaveInterval(self) -> None:
        """Return to the default of not saving any intermediate points."""
        self._mSaveInterval = sys.float_info.max

    def getNSaved(self) -> int:
        """Returns the number of saved values."""
        return self._mNSaved

    def getX(self, i: int) -> float:
        """Returns the x-coordinate of the i-th saved value (i < getNSaved())."""
        return self._mXSave[i]

    def getY(self, i: int) -> F64Array:
        """Returns the getNVariables() x y-coordinates of the i-th saved
        values (i < getNSaved()), of dimension getNVariables().
        """
        return self._mYSave[i]

    def setMaxSteps(self, maxSteps: int) -> None:
        """Set the maximum number of ODE steps to allow. Default is 10000."""
        self._mMaxSteps = maxSteps

    def setMinStepSize(self, minStep: float) -> None:
        """Sets the minimum permissible step size. Default is 0.0."""
        self._mMinStepSize = abs(minStep)

    def getNVariables(self) -> int:
        """Returns the number of variables as set in the constructor."""
        return self._mNVariables

    def getStepCount(self) -> int:
        """Get the total number of steps required to perform the previous
        integrate operation.
        """
        return self._mNOk + self._mNBad

    def getGoodStepCount(self) -> int:
        """Get the number of steps leading to results of the desired accuracy."""
        return self._mNOk

    def getBadStepCount(self) -> int:
        """Get the number of steps that needed to be subdivided to attain
        results of the desired accuracy.
        """
        return self._mNBad

    # ------------------------------------------------------------------
    # integrate -- R2: scipy-primary + integrate_literal faithful pair
    # ------------------------------------------------------------------
    def integrate(self, x1: float, x2: float, ystart: F64Array, eps: float, h1: float) -> F64Array:
        """Integrate the ODE specified by derivatives() using scipy's
        adaptive-step Dormand-Prince RK45 solver over the independent
        variable interval x1 to x2. ystart contains the initial y values
        and is updated in place with the final values (matching Java's
        `System.arraycopy(y, 0, ystart, 0, mNVariables)`). eps is a
        measure of the permissible relative/absolute error. h1 is the
        initial step size hint.

        See SCIPY-DEV-1..3 in BUG_LEDGER for documented deviations from
        integrate_literal(). Raises UtilException on solver failure
        (mirrors the Java `throws UtilException`).
        """
        self._require_mutable_f64(ystart, "ystart")
        n: int = self._mNVariables

        def _f(x: float, y: F64Array) -> F64Array:
            dydx: F64Array = np.zeros(n, dtype=np.float64)
            self.derivatives(x, y, dydx)
            return dydx

        t_eval: Optional[F64Array] = None
        if self._mSaveInterval != sys.float_info.max:
            span: float = abs(x2 - x1)
            kMax: int = int(math.floor((span + self._mSaveInterval) / self._mSaveInterval + 0.5))
            step: float = self._sign(self._mSaveInterval, x2 - x1)
            grid: F64Array = x1 + step * np.arange(kMax, dtype=np.float64)
            grid = grid[grid <= x2] if step >= 0 else grid[grid >= x2]
            if grid.size == 0 or grid[-1] != x2:
                grid = np.append(grid, x2)
            t_eval = grid

        result = _sp_integrate.solve_ivp(
            _f, (x1, x2), np.asarray(ystart, dtype=np.float64)[:n].copy(),
            method="RK45", t_eval=t_eval,
            first_step=abs(h1) if h1 != 0.0 else None,
            max_step=abs(self._mSaveInterval) if self._mSaveInterval != sys.float_info.max else np.inf,
            rtol=eps, atol=eps * 1.0e-10,
        )
        if not result.success:
            raise UtilException(f"AdaptiveRungeKutta.integrate (scipy RK45) failed: {result.message}")

        y: F64Array = np.array(result.y[:, -1], dtype=np.float64)
        ystart[:n] = y

        # SCIPY-DEV-2: mNBad always 0 (see BUG_LEDGER).
        self._mNOk = max(result.t.shape[0] - 1, 0)
        self._mNBad = 0
        if t_eval is not None:
            self._mNSaved = result.t.shape[0]
            self._mXSave = np.array(result.t, dtype=np.float64)
            self._mYSave = np.array(result.y.T, dtype=np.float64)
        else:
            self._mNSaved = 0
            self._mXSave = None
            self._mYSave = None
        self._clearWorkspace()
        return y

    def integrate_literal(self, x1: float, x2: float, ystart: F64Array, eps: float, h1: float) -> F64Array:
        """Line-for-line port of Java's `integrate` (Cash-Karp RK45 via
        _baseStep + Numerical-Recipes step control via _qcStep). This is
        the parity-harness reference implementation.

        Integrate the ODE specified by derivatives() using the adaptive
        step size Runge-Kutta algorithm over the independent variable
        interval x1 to x2. ystart contains the initial y values and is
        updated in place. eps is the permissible relative error. h1 is
        the initial step size.

        Returns the final y values as an array of length getNVariables().
        Raises UtilException upon too many steps or too small a step.
        """
        self._require_mutable_f64(ystart, "ystart")
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
        y[:] = ystart[:n]
        if self._mSaveInterval != sys.float_info.max:
            # R7: Math.round(v) -> int(math.floor(v + 0.5))
            kMax = int(math.floor(((abs(x2 - x1) + self._mSaveInterval) / self._mSaveInterval) + 0.5))
            saveInt = self._sign(self._mSaveInterval, x2 - x1)
            xsav = x - (2.0 * saveInt)  # to ensure that the first step is saved...
            self._mXSave = np.zeros(kMax, dtype=np.float64)
            self._mYSave = np.zeros((kMax, n), dtype=np.float64)

        for _step in range(self._mMaxSteps):
            # Save the necessary points
            if (kMax != 0) and (self._mNSaved < kMax) and (abs(x - xsav) >= (0.9999 * self._mSaveInterval)):
                self._mXSave[self._mNSaved] = x
                self._mYSave[self._mNSaved][:] = y
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
                ystart[:n] = y
                if kMax != 0:
                    self._mNSaved = min(self._mNSaved, kMax - 1)
                    self._mXSave[self._mNSaved] = x
                    self._mYSave[self._mNSaved][:] = y
                    self._mNSaved += 1
                self._clearWorkspace()
                return y
            if abs(self._mHNext) <= self._mMinStepSize:
                raise UtilException("Step size too small in AdaptiveRungeKutta.integrate")
            h = self._mHNext

        raise UtilException("Too many steps in AdaptiveRungeKutta.integrate")

    # ------------------------------------------------------------------
    # derivatives -- public abstract void derivatives(...)
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def derivatives(self, x: float, y: F64Array, dydx: F64Array) -> None:
        """derivatives - The derived class provides an implementation of
        the derivatives function. x & y[] are input and the user provided
        implementation of derivatives is responsible for returning the
        derivatives in the array dydx. The lengths of y and dydx are
        equal to getNVariables().

        public abstract -> no leading underscore (R1): overriding
        `abstract void compute()`-style methods must keep the Java name.
        """
        raise NotImplementedError