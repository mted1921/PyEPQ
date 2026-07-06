r"""
AdaptiveRungeKutta_ver1_1_0.py — Python port of gov.nist.microanalysis.Utility.AdaptiveRungeKutta

Guide version : 1
Generation    : 1
Port-code fixes: 0

CHANGES (deliberate deviations, per R10)
----------------------------------------
* SCIPY-DEV-1: `integrate()` (the primary public name, per R2) delegates to
  scipy.integrate.solve_ivp with method="RK45" (Dormand-Prince 5(4)) instead
  of the Java Cash-Karp 5(4) stepper. Same order, different tableau — results
  agree to tolerance, not bit-for-bit. `integrate_literal()` is the faithful
  line-for-line Cash-Karp port; use it for parity testing.
* SCIPY-DEV-2: With a save interval set, `integrate()` records intermediate
  points on the exact grid x1 + k*saveInt (via dense output), whereas Java
  saves at the nearest *actual step positions* at least one interval apart.
  Saved x values therefore differ slightly between the two variants; array
  sizing (kMax) follows the Java formula in both.
* SCIPY-DEV-3: scipy does not report rejected (subdivided) steps, so after
  `integrate()` mNOk = number of accepted internal steps and mNBad = 0.
  `getStepCount()`/`getGoodStepCount()`/`getBadStepCount()` are exact only
  after `integrate_literal()`.
* SCIPY-DEV-4: mHDid / mHNext after `integrate()` are both set to the size of
  the final accepted internal step (scipy does not expose "next h to try").
* SCIPY-DEV-5: mMinStepSize is not enforced inside scipy; step-size underflow
  surfaces as solver failure, re-raised as UtilException. mMaxSteps is
  enforced post-hoc on the accepted-step count.
* R5: `integrate()` / `integrate_literal()` mutate `ystart` in place (Java:
  "In & out" contract), so `ystart` MUST be a writeable float64 ndarray —
  `_require_mutable_f64` raises TypeError for lists/tuples/wrong dtype.
* Exception type: Java `throws UtilException`. The port raises `UtilException`
  imported from the sibling port `UtilException_ver1_1_0.py` (filename taken
  from UTILITY_LEDGER.md "Port file" column, per R3 — never a local stand-in).
* NOTE (cosmetic, no ledger entry): the original class Javadoc sentence
  "Using this option may limit the step size and thus" is truncated in the
  Java source; it is preserved verbatim below.
* `equals`/`hashCode`/`toString`: the Java class does not override them, so
  no dunder/named-alias pairs are required (R2 table applies vacuously).

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
import abc, math, sys
import numpy as np
from typing import Optional, Sequence, Union, Callable
try:
    from ._epq_compat import EPQException, F64Array, JavaRandom
except ImportError:
    try:
        from _epq_compat import EPQException, F64Array, JavaRandom  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility._epq_compat import EPQException, F64Array, JavaRandom  # type: ignore

# Sibling port — filename from UTILITY_LEDGER.md "Port file" column (R3).
try:
    from .UtilException_ver1_1_0 import UtilException
except ImportError:
    try:
        from UtilException_ver1_1_0 import UtilException  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility.UtilException_ver1_1_0 import UtilException  # type: ignore

# scipy is required only by the substituted primary integrate();
# integrate_literal() has no scipy dependency.
try:
    from scipy.integrate import solve_ivp as _solve_ivp
except ImportError:  # pragma: no cover
    _solve_ivp = None  # type: ignore[assignment]


__all__ = ["AdaptiveRungeKutta", "UtilException"]


# Java Double.MAX_VALUE (== 1.7976931348623157e308).
_DOUBLE_MAX: float = sys.float_info.max


class AdaptiveRungeKutta(abc.ABC):
    """Adaptive step size Cash-Karp Runge-Kutta ODE integrator.

    Abstract: subclasses implement ``derivatives(x, y, dydx)`` (public
    abstract in Java → un-prefixed name here, per R1).
    """

    # ==================================================================
    # Preserved-bug ledger (machine-readable) — see BUG_GUIDE.md
    # ==================================================================
    BUG_LEDGER: tuple = ()  # no bugs identified (full Java source reviewed;
    #                         the truncated class-Javadoc sentence is a
    #                         documentation defect with no observable
    #                         behaviour, so it gets no ledger entry)

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
        self.mSaveInterval: float = _DOUBLE_MAX
        self.mMinStepSize: float = 0.0
        self.mXSave: Optional[F64Array] = None
        self.mYSave: Optional[F64Array] = None  # 2-D (kMax, mNVariables)
        self.mNSaved: int = 0
        self.mMaxSteps: int = 10000
        self.mNOk: int = 0   # Number of ok steps
        self.mNBad: int = 0  # Number of repeated steps
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
    # Internal guards (R5)
    # ==================================================================
    @staticmethod
    def _require_mutable_f64(arr, name: str = "arr") -> None:
        """Type guard for in-place mutation targets (CONVERSION_GUIDE R5).

        Java's ``double[]`` is always a mutable double-precision buffer.
        Lists, tuples, and wrong-dtype arrays would silently no-op or
        copy. Fail loud.
        """
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{name} must be a numpy ndarray")
        if arr.dtype != np.float64:
            raise TypeError(f"{name} must have dtype float64")
        if not arr.flags.writeable:
            raise TypeError(f"{name} must be writeable")

    # ==================================================================
    # Private helpers (Java private → `_` prefix, camelCase kept, R1)
    # ==================================================================
    def _sign(self, magnitude: float, sign: float) -> float:
        """Java: private double sign(double magnitude, double sign).

        Fortran-style SIGN transfer: |magnitude| carrying the sign of
        `sign`, with sign == 0.0 treated as positive (Java ternary
        `sign >= 0.0` — this is NOT Math.signum, so R8 does not apply).
        """
        return abs(magnitude) if sign >= 0.0 else -abs(magnitude)

    def _baseStep(self, x: float, y: F64Array, dydx: F64Array, h: float,
                  yout: F64Array, yerr: F64Array) -> None:
        """Java: private void baseStep(double x, double[] y, double[] dydx,
        double h, double[] yout, double[] yerr).

        baseStep - Take a single Cash-Karp Runge-Kutta step. Given the
        n=mNDimensions values y[0..n-1] and their derivatives dydx[0..n-1]
        known at x, use a fifth order Cash-Karp Runge-Kutta method to
        advance the solution over an interval h. The resulting y value is
        returned in yout. An estimate of the truncation error is returned
        in yerr.
        """
        # R5 — yout and yerr are mutated in place.
        AdaptiveRungeKutta._require_mutable_f64(yout, "yout")
        AdaptiveRungeKutta._require_mutable_f64(yerr, "yerr")
        # Cash-Karp tableau (Java local finals, preserved verbatim).
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
        # Workspace (lazily allocated, as in Java)
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

        qcStep - Take a fifth order Runge-Kutta step with monitoring of
        local truncation error. Input are the dependent variable
        y[0..mNDimensions-1] and its derivatives dydx[0..mNDimensions-1]
        at the starting value of the independent variable x. Also input
        is the attempted step size htry, the required accuracy eps and
        the vector yscal against which the errors are scaled. Upon
        return, y is replaced with the new values, x is returned and
        mHDid and mHNext are set to the actual step size and the size of
        the next step to try.

        Raises UtilException when the step size becomes too small.
        Returns the new value of x.
        """
        # R5 — y is mutated in place (Java: In,Out).
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
        while True:  # Java do { ... } while (errmax > 1.0)
            self._baseStep(x, y, dydx, h, self.mQcYTemp, self.mYErr)
            errmax = 0.0
            for i in range(self.mNVariables):
                errmax = max(errmax, abs(float(self.mYErr[i]) / float(yscal[i])))
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
        self.mHNext = (safety * h * math.pow(errmax, pgrow)) if errmax > errcon else 5.0 * h
        self.mHDid = h
        x += h
        y[0:self.mNVariables] = self.mQcYTemp[0:self.mNVariables]  # System.arraycopy
        return x

    def _clearWorkspace(self) -> None:
        """Java: private void clearWorkspace().

        clearWorkspace - null all temporary space to free memory.
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
    # Public API
    # ==================================================================
    def setSaveInterval(self, interval: float) -> None:
        """Java: public void setSaveInterval(double interval).

        Set the interval on which to save intermediate points on the
        integrated trajectory. (Use clearSaveInterval to not save any
        intermediate points.) Note: The default is not to save any
        intermediate points.
        """
        self.mSaveInterval = abs(float(interval))

    def clearSaveInterval(self) -> None:
        """Java: public void clearSaveInterval().

        Return to the default of not saving any intermediate points.
        """
        self.mSaveInterval = _DOUBLE_MAX

    def getNSaved(self) -> int:
        """Java: public int getNSaved(). Returns the number of saved values."""
        return self.mNSaved

    def getX(self, i: int) -> float:
        """Java: public double getX(int i).

        Returns the x-coordinate of the i-th saved value, where
        i < getNSaved().
        """
        return float(self.mXSave[i])

    def getY(self, i: int) -> F64Array:
        """Java: public double[] getY(int i).

        Returns the getNVariable x y-coordinates of the i-th saved
        values (i < getNSaved()), of dimension getNVariables.

        Like Java (which returns a reference into mYSave), this returns
        a numpy row *view* — caller mutations propagate to the internal
        storage.
        """
        return self.mYSave[i]

    def setMaxSteps(self, maxSteps: int) -> None:
        """Java: public void setMaxSteps(int maxSteps).

        Set the maximum number of ODE steps to allow. Default is 10000.
        """
        self.mMaxSteps = int(maxSteps)

    def setMinStepSize(self, minStep: float) -> None:
        """Java: public void setMinStepSize(double minStep).

        Sets the minimum permissible step size. Default is 0.0.
        """
        self.mMinStepSize = abs(float(minStep))

    def getNVariables(self) -> int:
        """Java: public int getNVariables().

        Returns the number of variables as set in the constructor.
        """
        return self.mNVariables

    def getStepCount(self) -> int:
        """Java: public int getStepCount().

        Get the total number of steps required to perform the previous
        integrate operation.
        """
        return self.mNOk + self.mNBad

    def getGoodStepCount(self) -> int:
        """Java: public int getGoodStepCount().

        Get the number of steps leading to results of the desired
        accuracy.
        """
        return self.mNOk

    def getBadStepCount(self) -> int:
        """Java: public int getBadStepCount().

        Get the number of steps that were needed to be subdivided to
        attain results of the desired accuracy.
        """
        return self.mNBad

    # ------------------------------------------------------------------
    # integrate — scipy primary (R2) and literal companion
    # ------------------------------------------------------------------
    def integrate(self, x1: float, x2: float, ystart: F64Array, eps: float,
                  h1: float) -> F64Array:
        """Primary (library-substituted) form of the Java method
        ``public double[] integrate(double x1, double x2, double[] ystart,
        double eps, double h1) throws UtilException``.

        integrate - Integrate the ODE specified by derivatives using an
        adaptive step size Runge-Kutta algorithm over the independent
        variable interval x1 to x2. ystart contains the initial y values
        and receives the final y values (In & out; must be a writeable
        float64 ndarray — R5). eps is the permissible relative error and
        h1 the initial step size. Returns the final y values as an array
        of length getNVariables(). Raises UtilException upon too many
        steps or solver failure.

        SCIPY-DEV-1: uses scipy.integrate.solve_ivp (RK45 Dormand-Prince)
        rather than the Cash-Karp stepper — see the module CHANGES
        section for all documented deviations (SCIPY-DEV-1..5). Use
        ``integrate_literal`` for bit-faithful Java behaviour.
        """
        if _solve_ivp is None:  # pragma: no cover
            raise ImportError("scipy is required for integrate(); "
                              "use integrate_literal() for the scipy-free path")
        AdaptiveRungeKutta._require_mutable_f64(ystart, "ystart")  # R5
        n: int = self.mNVariables
        self.mNSaved = 0
        self.mNOk = 0
        self.mNBad = 0
        saving: bool = self.mSaveInterval != _DOUBLE_MAX
        y0: F64Array = np.array(ystart[0:n], dtype=np.float64, copy=True)

        # Degenerate zero-length interval: mimic the literal path, which
        # saves the initial point (if saving) plus the terminal point and
        # returns immediately with y unchanged.
        if x2 == x1:
            if saving:
                self.mXSave = np.zeros(2, dtype=np.float64)
                self.mYSave = np.zeros((2, n), dtype=np.float64)
                self.mXSave[0] = x1
                self.mYSave[0, :] = y0
                self.mXSave[1] = x2
                self.mYSave[1, :] = y0
                self.mNSaved = 2
            self.mHDid = 0.0
            self.mHNext = 0.0
            return y0

        def _fun(t: float, yv: F64Array) -> F64Array:
            """Adapter: Java's out-parameter `derivatives` → scipy callable."""
            dydx: F64Array = np.zeros(n, dtype=np.float64)
            self.derivatives(float(t), np.asarray(yv, dtype=np.float64), dydx)
            return dydx

        first_step: Optional[float] = None
        if h1 != 0.0:
            first_step = min(abs(float(h1)), abs(x2 - x1))
        sol = _solve_ivp(_fun, (float(x1), float(x2)), y0,
                         method="RK45",
                         rtol=float(eps),
                         atol=1.0e-10 * float(eps),  # Java: tiny = 1.0e-10 * eps
                         dense_output=saving,
                         first_step=first_step)
        if not sol.success:
            # SCIPY-DEV-5: solver failure (incl. step underflow) → UtilException
            raise UtilException(f"AdaptiveRungeKutta.integrate failed: {sol.message}")
        nSteps: int = len(sol.t) - 1
        if nSteps > self.mMaxSteps:
            raise UtilException("Too many steps in AdaptiveRungeKutta.integrate")
        # SCIPY-DEV-3: rejected steps not reported by scipy.
        self.mNOk = nSteps
        self.mNBad = 0
        # SCIPY-DEV-4: mHNext not exposed by scipy; mirror mHDid.
        self.mHDid = float(sol.t[-1] - sol.t[-2]) if len(sol.t) >= 2 else 0.0
        self.mHNext = self.mHDid

        if saving:
            # Array sizing follows the Java formula (Math.round → R7).
            kMax: int = int(math.floor(((abs(x2 - x1) + self.mSaveInterval) / self.mSaveInterval) + 0.5))
            saveInt: float = self._sign(self.mSaveInterval, x2 - x1)
            self.mXSave = np.zeros(kMax, dtype=np.float64)
            self.mYSave = np.zeros((kMax, n), dtype=np.float64)
            # SCIPY-DEV-2: exact grid x1 + k*saveInt, then the terminal x2.
            ts: list[float] = []
            k: int = 0
            while k < kMax and abs(k * saveInt) <= abs(x2 - x1):
                ts.append(x1 + k * saveInt)
                k += 1
            if (not ts) or (ts[-1] != x2):
                if len(ts) < kMax:
                    ts.append(float(x2))
                else:
                    ts[-1] = float(x2)  # clamp, as Java clamps mNSaved to kMax-1
            yGrid: F64Array = sol.sol(np.asarray(ts, dtype=np.float64))  # (n, len(ts))
            self.mNSaved = len(ts)
            for j in range(self.mNSaved):
                self.mXSave[j] = ts[j]
                self.mYSave[j, :] = yGrid[:, j]

        yFinal: F64Array = np.array(sol.y[:, -1], dtype=np.float64, copy=True)
        ystart[0:n] = yFinal  # Java: System.arraycopy(y, 0, ystart, 0, mNVariables)
        return yFinal

    def integrate_literal(self, x1: float, x2: float, ystart: F64Array,
                          eps: float, h1: float) -> F64Array:
        """Faithful line-for-line port of Java integrate (R2).

        integrate - Integrate the ODE specified by derivatives using the
        adaptive step size Runge-Kutta algorithm over the independent
        variable interval x1 to x2. ystart contains the initial y values
        (In & out; must be a writeable float64 ndarray — R5). eps is a
        measure of the permissible error. h1 is the initial step size.
        Returns the final y values as an array of length getNVariables().
        Raises UtilException upon too many steps or too small a step.
        """
        AdaptiveRungeKutta._require_mutable_f64(ystart, "ystart")  # R5
        tiny: float = 1.0e-10 * eps
        yscal: F64Array = np.zeros(self.mNVariables, dtype=np.float64)
        dydx: F64Array = np.zeros(self.mNVariables, dtype=np.float64)
        y: F64Array = np.zeros(self.mNVariables, dtype=np.float64)
        x: float = x1
        h: float = self._sign(h1, x2 - x1)
        xsav: float = 0.0
        saveInt: float = _DOUBLE_MAX
        kMax: int = 0
        self.mNSaved = 0
        self.mNOk = 0
        self.mNBad = 0
        y[0:self.mNVariables] = ystart[0:self.mNVariables]  # System.arraycopy
        if self.mSaveInterval != _DOUBLE_MAX:
            # Java: (int) Math.round(...) → int(floor(x + 0.5)) per R7.
            kMax = int(math.floor(((abs(x2 - x1) + self.mSaveInterval) / self.mSaveInterval) + 0.5))
            saveInt = self._sign(self.mSaveInterval, x2 - x1)
            xsav = x - (2.0 * saveInt)  # to ensure that the first step is saved...
            self.mXSave = np.zeros(kMax, dtype=np.float64)
            self.mYSave = np.zeros((kMax, self.mNVariables), dtype=np.float64)
        for _step in range(self.mMaxSteps):
            # Save the necessary points
            if (kMax != 0) and (self.mNSaved < kMax) and (abs(x - xsav) >= (0.9999 * self.mSaveInterval)):
                self.mXSave[self.mNSaved] = x
                self.mYSave[self.mNSaved, :] = y  # System.arraycopy
                xsav = x
                self.mNSaved += 1
            self.derivatives(x, y, dydx)
            # Rescale h to ensure we hit desired points
            hMax: float = abs((xsav + saveInt) - x)
            if abs(h) > hMax:
                h = self._sign(hMax, h)
            # Scaling to monitor accuracy...
            for i in range(self.mNVariables):
                yscal[i] = abs(float(y[i])) + abs(float(dydx[i]) * h) + tiny
            if (((x + h) - x2) * ((x + h) - x1)) > 0.0:
                h = x2 - x
            x = self._qcStep(x, y, dydx, h, eps, yscal)
            if self.mHDid == h:
                self.mNOk += 1
            else:
                self.mNBad += 1
            if ((x - x2) * (x2 - x1)) >= 0.0:
                ystart[0:self.mNVariables] = y  # System.arraycopy
                if kMax != 0:
                    self.mNSaved = min(self.mNSaved, kMax - 1)
                    self.mXSave[self.mNSaved] = x
                    self.mYSave[self.mNSaved, :] = y  # System.arraycopy
                    self.mNSaved += 1
                self._clearWorkspace()
                return y
            if abs(self.mHNext) <= self.mMinStepSize:
                raise UtilException("Step size too small in AdaptiveRungeKutta.integrate")
            h = self.mHNext
        raise UtilException("Too many steps in AdaptiveRungeKutta.integrate")

    # ------------------------------------------------------------------
    # Abstract member (public abstract → un-prefixed name, R1)
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def derivatives(self, x: float, y: F64Array, dydx: F64Array) -> None:
        """Java: abstract public void derivatives(double x, double[] y, double[] dydx).

        derivatives - The derived class provides an implementation of
        the derivatives function. x & y[] are input and the user
        provided implementation of derivatives is resposible for
        returning the derivatives in the array dydx. The lengths of y
        and dydx are equal to mNDimensions.
        """
        ...