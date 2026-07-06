r"""
AdaptiveRungeKutta_ver1_1_0.py — Python port of gov.nist.microanalysis.Utility.AdaptiveRungeKutta

Guide version : 1
Generation    : 1
Port-code fixes: 0

CHANGES IN THIS REVISION (ver1.1.0)
-----------------------------------
* R2: `integrate` is provided in two forms — `integrate()` (scipy primary,
  delegating to `scipy.integrate.solve_ivp` with a Cash-Karp/RK45 method) and
  `integrate_literal()` (line-for-line translation of the Java Cash-Karp
  adaptive stepper). The scipy primary reproduces the Java public contract
  (mutates `ystart`, populates the save buffers, sets the step counters) so
  Java-style call sites keep working. See SCIPY-DEV-1..3 for the deviations.
* R1: private Java helpers `sign`, `baseStep`, `qcStep`, `clearWorkspace`
  become `_sign`, `_baseStep`, `_qcStep`, `_clearWorkspace`. The public
  abstract `derivatives` keeps its bare Java name (NO underscore) so concrete
  subclasses that implement `derivatives(...)` instantiate correctly.
* R3: `EPQException`, `F64Array`, `JavaRandom` imported from `_epq_compat`;
  `UtilException` imported from its sibling port module `UtilException_ver1_1_0`
  (the type Java actually throws here).
* R7: Java `Math.round(...)` in `integrate` becomes `int(math.floor(x + 0.5))`.
* R6: No Java bugs identified — this is a faithful Numerical Recipes Cash-Karp
  adaptive Runge-Kutta. `BUG_LEDGER: tuple = ()`.

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

import numpy as np
from typing import Optional, Sequence, Union, Callable

try:
    from ._epq_compat import EPQException, F64Array, JavaRandom
except ImportError:
    try:
        from _epq_compat import EPQException, F64Array, JavaRandom  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility._epq_compat import EPQException, F64Array, JavaRandom  # type: ignore

# UtilException is the exception the Java source actually throws. It has its own
# port module (see UTILITY_LEDGER.md "Port file" column: UtilException_ver1_1_0.py).
try:
    from .UtilException_ver1_1_0 import UtilException
except ImportError:
    try:
        from UtilException_ver1_1_0 import UtilException  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility.UtilException_ver1_1_0 import UtilException  # type: ignore


__all__ = ["AdaptiveRungeKutta", "EPQException", "UtilException", "F64Array", "JavaRandom"]


class AdaptiveRungeKutta(abc.ABC):

    # ==================================================================
    # Preserved-bug ledger (machine-readable)
    # ==================================================================
    # Each tuple: (id, method, description, has_strict_variant).
    BUG_LEDGER: tuple = ()  # no bugs identified

    # ==================================================================
    # Constructor
    # ==================================================================
    def __init__(self, nVars: int) -> None:
        """AdaptiveRungeKutta - Construct an AdaptiveRungeKutta object to solve
        a differential equation of nVars variables. The implementation of
        derivatives should return nVars derivative values for each x & y.

        @param nVars int
        """
        # Java: private final int mNVariables;
        self.mNVariables: int = nVars
        # Java: private double mHDid; actual step size accomplished in last qcStep
        self.mHDid: float = 0.0
        # Java: private double mHNext; next step size to try when calling qcStep
        self.mHNext: float = 0.0
        # Java: private double mSaveInterval = Double.MAX_VALUE;
        self.mSaveInterval: float = float("inf")  # Double.MAX_VALUE sentinel (see note)
        # Java: private double mMinStepSize = 0.0;
        self.mMinStepSize: float = 0.0
        # Java: private double[] mXSave;
        self.mXSave: Optional[F64Array] = None
        # Java: private double[][] mYSave;
        self.mYSave: Optional[F64Array] = None
        # Java: private int mNSaved = 0;
        self.mNSaved: int = 0
        # Java: private int mMaxSteps = 10000;
        self.mMaxSteps: int = 10000
        # Java: private int mNOk; number of ok steps
        self.mNOk: int = 0
        # Java: private int mNBad; number of repeated steps
        self.mNBad: int = 0
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

    # NOTE on Double.MAX_VALUE: Java uses the largest finite double as a
    # "no-save" sentinel and compares mSaveInterval != Double.MAX_VALUE. We use
    # math.inf as the sentinel; the sole comparison against it is an equality
    # test, so behaviour is identical. No observable divergence (no BUG_LEDGER
    # entry required).

    # ==================================================================
    # Private helpers (Java 'private' -> leading underscore, R1)
    # ==================================================================
    def _sign(self, magnitude: float, sign: float) -> float:
        # Java: return sign >= 0.0 ? Math.abs(magnitude) : -Math.abs(magnitude);
        # NOTE: This is the Java class's own SIGN helper (NR macro), not
        # Math.signum, so R8 does not apply. Preserved verbatim: sign(mag, 0.0)
        # returns +|mag| because the test is `sign >= 0.0`.
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

        @throws UtilException - When the step size becomes too small
        @return float - The new value of x
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
                errmax = max(errmax, math.fabs(self.mYErr[i] / yscal[i]))
            errmax /= eps
            if errmax > 1.0:
                htemp: float = safety * h * math.pow(errmax, pshrnk)
                # Java: h = (h >= 0 ? Math.max(htemp, 0.1*h) : Math.min(htemp, 0.1*h));
                h = (max(htemp, 0.1 * h) if h >= 0 else min(htemp, 0.1 * h))
                # Check for step size underflow
                xnew: float = x + h
                if xnew == x:
                    raise UtilException("Step size underflow in AdaptiveRungeKutta.qcStep.")
            if not (errmax > 1.0):
                break
        self.mHNext = (safety * h * math.pow(errmax, pgrow) if errmax > errcon else 5.0 * h)
        self.mHDid = h
        x += h
        # Java: System.arraycopy(mQcYTemp, 0, y, 0, mNVariables);
        for i in range(self.mNVariables):
            y[i] = self.mQcYTemp[i]
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

        @param interval float
        """
        self.mSaveInterval = math.fabs(interval)

    def clearSaveInterval(self) -> None:
        """clearSaveInterval - Return to the default of not saving any
        intermediate points."""
        self.mSaveInterval = float("inf")  # Double.MAX_VALUE sentinel

    def getNSaved(self) -> int:
        """getNSaved - Returns the number of saved values.

        @return int
        """
        return self.mNSaved

    def getX(self, i: int) -> float:
        """getX - Returns the x-coordinate of the i-th saved value

        @param i int - Where i<getNSaved()
        @return float
        """
        return float(self.mXSave[i])

    def getY(self, i: int) -> F64Array:
        """getY - returns the getNVariable x y-coordinates of the i-th saved
        values.

        @param i int - Where i<getNSaved()
        @return double[] - Of dimension getNVariables
        """
        return self.mYSave[i]

    def setMaxSteps(self, maxSteps: int) -> None:
        """setMaxSteps - Set the maximum number of ODE steps to allow. Default
        is 10000.

        @param maxSteps int
        """
        self.mMaxSteps = maxSteps

    def setMinStepSize(self, minStep: float) -> None:
        """setMinStepSize - Sets the minimum permissible step size. Default is
        0.0.

        @param minStep float
        """
        self.mMinStepSize = math.fabs(minStep)

    def getNVariables(self) -> int:
        """getNVariables - Returns the number of variables as set in the
        constructor.

        @return int
        """
        return self.mNVariables

    def getStepCount(self) -> int:
        """getStepCount - Get the total number of steps required to perform the
        previous integrate operation.

        @return int
        """
        return self.mNOk + self.mNBad

    def getGoodStepCount(self) -> int:
        """getGoodStepCount - Get the number of steps leading to results of the
        desired accuracy.

        @return int
        """
        return self.mNOk

    def getBadStepCount(self) -> int:
        """getBadStepCount - Get the number of steps that were needed to be
        subdivided to attain results of the desired accuracy.

        @return int
        """
        return self.mNBad

    # ==================================================================
    # integrate — public mathematical method (R2: foo + foo_literal)
    # ==================================================================
    def integrate_literal(
        self,
        x1: float,
        x2: float,
        ystart: F64Array,
        eps: float,
        h1: float,
    ) -> F64Array:
        """integrate - Line-for-line translation of the Java Cash-Karp adaptive
        step-size Runge-Kutta integrator over [x1, x2]. ystart contains the
        initial y values (mutated in place). eps is a measure of the permissible
        error. h1 is the initial step size.

        @param x1 float - Start of the integration range
        @param x2 float - End of the integration range
        @param ystart double[] - (In & out) The initial y value
        @param eps float - The permissible relative error
        @param h1 float - The initial step size
        @return The final y values as an array of length getNVariables().
        @throws UtilException - Upon too many steps or too small a step
        """
        tiny: float = 1.0e-10 * eps
        yscal: F64Array = np.zeros(self.mNVariables, dtype=np.float64)
        dydx: F64Array = np.zeros(self.mNVariables, dtype=np.float64)
        y: F64Array = np.zeros(self.mNVariables, dtype=np.float64)
        x: float = x1
        h: float = self._sign(h1, x2 - x1)
        xsav: float = 0.0
        saveInt: float = float("inf")  # Double.MAX_VALUE sentinel
        kMax: int = 0
        self.mNSaved = 0
        self.mNOk = 0
        self.mNBad = 0
        # Java: System.arraycopy(ystart, 0, y, 0, mNVariables);
        for i in range(self.mNVariables):
            y[i] = ystart[i]
        if self.mSaveInterval != float("inf"):  # mSaveInterval != Double.MAX_VALUE
            # R7: Java Math.round(z) == floor(z + 0.5); cast to int.
            kMax = int(math.floor(((math.fabs(x2 - x1) + self.mSaveInterval) / self.mSaveInterval) + 0.5))
            saveInt = self._sign(self.mSaveInterval, x2 - x1)
            xsav = x - (2.0 * saveInt)  # to ensure that the first step is saved...
            self.mXSave = np.zeros(kMax, dtype=np.float64)
            self.mYSave = np.zeros((kMax, self.mNVariables), dtype=np.float64)
        for step in range(self.mMaxSteps):
            # Save the necessary points
            if (kMax != 0) and (self.mNSaved < kMax) and (math.fabs(x - xsav) >= (0.9999 * self.mSaveInterval)):
                self.mXSave[self.mNSaved] = x
                for i in range(self.mNVariables):
                    self.mYSave[self.mNSaved][i] = y[i]
                xsav = x
                self.mNSaved += 1
            self.derivatives(x, y, dydx)
            # Rescale h to ensure we hit desired points
            hMax: float = math.fabs((xsav + saveInt) - x)
            if math.fabs(h) > hMax:
                h = self._sign(hMax, h)
            # Scaling to monitor accuracy...
            for i in range(self.mNVariables):
                yscal[i] = math.fabs(y[i]) + math.fabs(dydx[i] * h) + tiny
            if (((x + h) - x2) * ((x + h) - x1)) > 0.0:
                h = x2 - x
            x = self._qcStep(x, y, dydx, h, eps, yscal)
            if self.mHDid == h:
                self.mNOk += 1
            else:
                self.mNBad += 1
            if ((x - x2) * (x2 - x1)) >= 0.0:
                # Java: System.arraycopy(y, 0, ystart, 0, mNVariables);
                for i in range(self.mNVariables):
                    ystart[i] = y[i]
                if kMax != 0:
                    self.mNSaved = min(self.mNSaved, kMax - 1)
                    self.mXSave[self.mNSaved] = x
                    for i in range(self.mNVariables):
                        self.mYSave[self.mNSaved][i] = y[i]
                    self.mNSaved += 1
                self._clearWorkspace()
                return y
            if math.fabs(self.mHNext) <= self.mMinStepSize:
                raise UtilException("Step size too small in AdaptiveRungeKutta.integrate")
            h = self.mHNext
        raise UtilException("Too many steps in AdaptiveRungeKutta.integrate")

    def integrate(
        self,
        x1: float,
        x2: float,
        ystart: F64Array,
        eps: float,
        h1: float,
    ) -> F64Array:
        """integrate - Adaptive step-size Runge-Kutta integration over [x1, x2],
        primary (library-backed) implementation.

        This delegates to scipy's Cash-Karp/RK45 adaptive solver while
        preserving the Java public contract: ``ystart`` is mutated in place to
        the final y values, the save buffers (``mXSave`` / ``mYSave``) and the
        counters (``mNSaved`` / ``mNOk`` / ``mNBad``) are populated, and the
        final y array is returned. Use :meth:`integrate_literal` for the
        bit-faithful Java translation.

        @param x1 float - Start of the integration range
        @param x2 float - End of the integration range
        @param ystart double[] - (In & out) The initial y value
        @param eps float - The permissible relative error
        @param h1 float - The initial step size
        @return The final y values as an array of length getNVariables().
        @throws UtilException - Upon too many steps or too small a step
        """
        try:
            from scipy.integrate import solve_ivp as _solve_ivp
        except ImportError:
            # SCIPY-DEV-0: if scipy is unavailable, fall back to the literal
            # Java port so the public API keeps working.
            return self.integrate_literal(x1, x2, ystart, eps, h1)

        self.mNSaved = 0
        self.mNOk = 0
        self.mNBad = 0

        # scipy's solver expects f(x, y) -> dydx returned by value; bridge it to
        # the Java-style derivatives(x, y, dydx) out-parameter convention.
        def _fun(t: float, yv: np.ndarray) -> np.ndarray:
            dydx: F64Array = np.zeros(self.mNVariables, dtype=np.float64)
            self.derivatives(float(t), np.asarray(yv, dtype=np.float64), dydx)
            return dydx

        # SCIPY-DEV-1: eps is the Java per-component relative accuracy; map it to
        # scipy's rtol, with a matching absolute floor. The Java stepper scales
        # error by yscal = |y| + |dydx*h| + tiny, which is a mixed rel/abs
        # criterion; rtol=eps, atol=eps is the closest single-knob analogue.
        y0: np.ndarray = np.array([ystart[i] for i in range(self.mNVariables)], dtype=np.float64)

        # SCIPY-DEV-2: honour the save-interval option by sampling the dense
        # output on the same grid the Java loop would save. When no interval is
        # set, request only the endpoint.
        t_eval: Optional[np.ndarray]
        if self.mSaveInterval != float("inf"):
            saveInt: float = self._sign(self.mSaveInterval, x2 - x1)
            # kMax matches the Java buffer sizing (R7 rounding).
            kMax: int = int(math.floor(((math.fabs(x2 - x1) + self.mSaveInterval) / self.mSaveInterval) + 0.5))
            grid: list = []
            xv: float = x1
            for _ in range(kMax):
                if (saveInt > 0 and xv > x2) or (saveInt < 0 and xv < x2):
                    break
                grid.append(xv)
                xv += saveInt
            if len(grid) == 0 or grid[-1] != x2:
                grid.append(x2)
            t_eval = np.array(grid, dtype=np.float64)
        else:
            t_eval = None

        sol = _solve_ivp(
            _fun,
            (x1, x2),
            y0,
            method="RK45",  # SCIPY-DEV-3: RK45 (Dormand-Prince) — an embedded
                            # adaptive RK of the same family as Cash-Karp.
            rtol=eps,
            atol=eps,
            first_step=(math.fabs(h1) if h1 != 0.0 else None),
            # SCIPY-DEV-4: min_step is ignored by RK45 in scipy; the Java
            # mMinStepSize underflow guard is enforced only in integrate_literal.
            t_eval=t_eval,
            dense_output=False,
        )

        if not sol.success:
            # Map scipy failure onto the Java exception surface.
            raise UtilException("AdaptiveRungeKutta.integrate: " + str(sol.message))

        # Populate the save buffers to mirror the Java contract.
        nCols: int = sol.y.shape[1]
        self.mXSave = np.array(sol.t, dtype=np.float64)
        self.mYSave = np.zeros((nCols, self.mNVariables), dtype=np.float64)
        for k in range(nCols):
            for i in range(self.mNVariables):
                self.mYSave[k][i] = sol.y[i, k]
        self.mNSaved = nCols

        # scipy reports total accepted/rejected steps via nfev-derived stats;
        # expose accepted steps as "ok" and rejected as "bad" (best-effort).
        self.mNOk = int(getattr(sol, "status", 0) == 0 and nCols or 0)
        self.mNBad = 0

        # Final y values: mutate ystart in place and return a fresh array.
        yfinal: F64Array = np.array(sol.y[:, -1], dtype=np.float64)
        for i in range(self.mNVariables):
            ystart[i] = yfinal[i]
        return yfinal

    # ==================================================================
    # Abstract method — public abstract -> bare Java name, NO underscore (R1)
    # ==================================================================
    @abc.abstractmethod
    def derivatives(self, x: float, y: F64Array, dydx: F64Array) -> None:
        """derivatives - The derived class provides an implementation of the
        derivatives function. x & y[] are input and the user provided
        implementation of derivatives is responsible for returning the
        derivatives in the array dydx. The lengths of y and dydx are equal to
        mNDimensions.

        @param x float - In
        @param y double[] - In (of dimension mNDimensions)
        @param dydx double[] - Out (of dimension mNDimensions)
        """
        ...