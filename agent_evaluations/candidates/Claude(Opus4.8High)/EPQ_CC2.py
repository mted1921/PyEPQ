r"""
AdaptiveRungeKutta_ver1_1_0.py — Python port of gov.nist.microanalysis.Utility.AdaptiveRungeKutta

Guide version : 1
Generation    : 1
Port-code fixes: 0

CHANGES IN THIS REVISION (ver1.1.0)
-----------------------------------
* R1: Java identifiers preserved verbatim (camelCase, leading-`m` fields).
  ``derivatives`` is ``public abstract`` → mapped to the un-prefixed name
  ``derivatives`` (NOT ``_derivatives``); ``abstract`` is not an access
  modifier. Private helpers keep their Java spelling under a ``_`` prefix:
  ``_sign``, ``_baseStep``, ``_qcStep``, ``_clearWorkspace``.
* R2: ``integrate`` is the one public mathematical method, so both
  ``integrate()`` (scipy-primary) and ``integrate_literal()`` (line-for-line
  Java) are provided. The scipy primary uses ``scipy.integrate.solve_ivp``
  (RK45 / Dormand–Prince) — see SCIPY-DEV-1. All getters/setters are plain
  literal translations (not "mathematical methods", so no ``_literal`` twin).
  The class defines no ``equals``/``hashCode``/``toString`` in Java, so no
  dunder aliases are added.
* R3: ``EPQException`` / ``JavaRandom`` / ``F64Array`` imported from
  ``_epq_compat`` via the three-tier fallback; ``UtilException`` imported from
  its sibling port ``UtilException_ver1_1_0`` (UTILITY_LEDGER "Port file"
  column) — the Java methods declare ``throws UtilException``, so the port
  raises the ported ``UtilException`` rather than a local stand-in.
* R7: Java ``Math.round`` on the save-buffer size becomes
  ``int(math.floor(x + 0.5))`` (not Python ``round``, which is HALF_EVEN).
* R9: every parameter, return, field, and non-obvious local is annotated;
  ``double[]`` → ``F64Array``.
* R6: no Java bugs identified — ``BUG_LEDGER: tuple = ()``. The private
  ``sign(magnitude, sign)`` helper uses ``sign >= 0.0`` (so a zero sign yields
  ``+|magnitude|``); this is the class's own helper — deliberately NOT
  ``Math.signum`` — and is preserved verbatim, not treated as a bug.

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
from typing import Optional, Sequence, Union, Callable

import numpy as np

try:
    from ._epq_compat import EPQException, F64Array, JavaRandom
except ImportError:
    try:
        from _epq_compat import EPQException, F64Array, JavaRandom  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility._epq_compat import EPQException, F64Array, JavaRandom  # type: ignore

# Sibling port: Java declares `throws UtilException`. Filename taken from
# UTILITY_LEDGER.md "Port file" column (UtilException_ver1_1_0.py) — not guessed.
try:
    from .UtilException_ver1_1_0 import UtilException
except ImportError:
    try:
        from UtilException_ver1_1_0 import UtilException  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility.UtilException_ver1_1_0 import UtilException  # type: ignore


__all__ = ["AdaptiveRungeKutta", "EPQException", "F64Array", "JavaRandom", "UtilException"]


class AdaptiveRungeKutta(abc.ABC):

    # ==================================================================
    # Preserved-bug ledger (machine-readable)
    # ==================================================================
    # Each tuple: (id, method, description, has_strict_variant).
    BUG_LEDGER: tuple = ()  # no bugs identified

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------
    def __init__(self, nVars: int) -> None:
        """AdaptiveRungeKutta - Construct an AdaptiveRungeKutta object to solve
        a differential equation of nVars variables. The implementation of
        derivatives should return nVars derivative values for each x & y.

        :param nVars: int
        """
        # private final int mNVariables; // The number of differential equations
        self.mNVariables: int = nVars
        # private double mHDid; // Actual step size accomplished in last call to qcStep
        self.mHDid: float = 0.0
        # private double mHNext; // Next step size to try when calling qcStep
        self.mHNext: float = 0.0
        # private double mSaveInterval = Double.MAX_VALUE;
        self.mSaveInterval: float = float(np.finfo(np.float64).max)  # Double.MAX_VALUE
        # private double mMinStepSize = 0.0;
        self.mMinStepSize: float = 0.0
        # private double[] mXSave;
        self.mXSave: Optional[F64Array] = None
        # private double[][] mYSave;
        self.mYSave: Optional[F64Array] = None
        # private int mNSaved = 0;
        self.mNSaved: int = 0
        # private int mMaxSteps = 10000;
        self.mMaxSteps: int = 10000
        # private int mNOk; // Number of ok steps
        self.mNOk: int = 0
        # private int mNBad; // Number of repeated steps
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

    # ------------------------------------------------------------------
    # private double sign(double magnitude, double sign)
    # ------------------------------------------------------------------
    def _sign(self, magnitude: float, sign: float) -> float:
        # Java: return sign >= 0.0 ? Math.abs(magnitude) : -Math.abs(magnitude);
        # NOTE: this is the class's own helper (not Math.signum). At sign == 0.0
        # it returns +|magnitude|. Preserved verbatim.
        return abs(magnitude) if sign >= 0.0 else -abs(magnitude)

    # ------------------------------------------------------------------
    # private void baseStep(...)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # private double qcStep(...) throws UtilException
    # ------------------------------------------------------------------
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
        self.mHNext = (safety * h * math.pow(errmax, pgrow) if errmax > errcon else 5.0 * h)
        self.mHDid = h
        x += h
        # System.arraycopy(mQcYTemp, 0, y, 0, mNVariables);
        for i in range(self.mNVariables):
            y[i] = self.mQcYTemp[i]
        return x

    # ------------------------------------------------------------------
    # private void clearWorkspace()
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # public void setSaveInterval(double interval)
    # ------------------------------------------------------------------
    def setSaveInterval(self, interval: float) -> None:
        """setSaveInterval - Set the interval on which to save intermediate
        points on the integrated trajectory. (Use clearSaveInterval to not save
        any intermediate points.) Note: The default is not to save any
        intermediate points.
        """
        self.mSaveInterval = abs(interval)

    # ------------------------------------------------------------------
    # public void clearSaveInterval()
    # ------------------------------------------------------------------
    def clearSaveInterval(self) -> None:
        """clearSaveInterval - Return to the default of not saving any
        intermediate points."""
        self.mSaveInterval = float(np.finfo(np.float64).max)  # Double.MAX_VALUE

    # ------------------------------------------------------------------
    # public int getNSaved()
    # ------------------------------------------------------------------
    def getNSaved(self) -> int:
        """getNSaved - Returns the number of saved values."""
        return self.mNSaved

    # ------------------------------------------------------------------
    # public double getX(int i)
    # ------------------------------------------------------------------
    def getX(self, i: int) -> float:
        """getX - Returns the x-coordinate of the i-th saved value"""
        return float(self.mXSave[i])

    # ------------------------------------------------------------------
    # public double[] getY(int i)
    # ------------------------------------------------------------------
    def getY(self, i: int) -> F64Array:
        """getY - returns the getNVariable x y-coordinates of the i-th saved
        values."""
        return self.mYSave[i]

    # ------------------------------------------------------------------
    # public void setMaxSteps(int maxSteps)
    # ------------------------------------------------------------------
    def setMaxSteps(self, maxSteps: int) -> None:
        """setMaxSteps - Set the maximum number of ODE steps to allow. Default
        is 10000."""
        self.mMaxSteps = maxSteps

    # ------------------------------------------------------------------
    # public void setMinStepSize(double minStep)
    # ------------------------------------------------------------------
    def setMinStepSize(self, minStep: float) -> None:
        """setMinStepSize - Sets the minimum permissible step size. Default is
        0.0."""
        self.mMinStepSize = abs(minStep)

    # ------------------------------------------------------------------
    # public int getNVariables()
    # ------------------------------------------------------------------
    def getNVariables(self) -> int:
        """getNVariables - Returns the number of variables as set in the
        constructor."""
        return self.mNVariables

    # ------------------------------------------------------------------
    # public int getStepCount()
    # ------------------------------------------------------------------
    def getStepCount(self) -> int:
        """getStepCount - Get the total number of steps required to perform the
        previous integrate operation."""
        return self.mNOk + self.mNBad

    # ------------------------------------------------------------------
    # public int getGoodStepCount()
    # ------------------------------------------------------------------
    def getGoodStepCount(self) -> int:
        """getGoodStepCount - Get the number of steps leading to results of the
        desired accuracy."""
        return self.mNOk

    # ------------------------------------------------------------------
    # public int getBadStepCount()
    # ------------------------------------------------------------------
    def getBadStepCount(self) -> int:
        """getBadStepCount - Get the number of steps that were needed to be
        subdivided to attain results of the desired accuracy."""
        return self.mNBad

    # ==================================================================
    # integrate  (R2: scipy primary + `_literal` faithful Java port)
    # ==================================================================
    def integrate_literal(self, x1: float, x2: float, ystart: F64Array,
                          eps: float, h1: float) -> F64Array:
        """integrate - Integrate the ODE specified by derivatives using the
        adaptive step size Runge-Kutta algorithm over the independent variable
        interval x1 to x2. ystart contains the initial y values. eps is a
        measure of the permissible error. h1 is the initial step size.

        Line-for-line faithful translation of the Java Cash-Karp integrator.

        :raises UtilException: Upon too many steps or too small a step
        :return: The final y values as an array of length getNVariables().
        """
        tiny: float = 1.0e-10 * eps
        yscal: F64Array = np.zeros(self.mNVariables, dtype=np.float64)
        dydx: F64Array = np.zeros(self.mNVariables, dtype=np.float64)
        y: F64Array = np.zeros(self.mNVariables, dtype=np.float64)
        x: float = x1
        h: float = self._sign(h1, x2 - x1)
        xsav: float = 0.0
        saveInt: float = float(np.finfo(np.float64).max)  # Double.MAX_VALUE
        kMax: int = 0
        self.mNSaved = 0
        self.mNOk = 0
        self.mNBad = 0
        # System.arraycopy(ystart, 0, y, 0, mNVariables);
        for i in range(self.mNVariables):
            y[i] = ystart[i]
        if self.mSaveInterval != float(np.finfo(np.float64).max):
            # R7: Math.round(x) → int(math.floor(x + 0.5)) (HALF_UP, not HALF_EVEN)
            kMax = int(math.floor(((abs(x2 - x1) + self.mSaveInterval) / self.mSaveInterval) + 0.5))
            saveInt = self._sign(self.mSaveInterval, x2 - x1)
            xsav = x - (2.0 * saveInt)  # to ensure that the first step is saved...
            self.mXSave = np.zeros(kMax, dtype=np.float64)
            self.mYSave = np.zeros((kMax, self.mNVariables), dtype=np.float64)
        for step in range(self.mMaxSteps):
            # Save the necessary points
            if (kMax != 0) and (self.mNSaved < kMax) and (abs(x - xsav) >= (0.9999 * self.mSaveInterval)):
                self.mXSave[self.mNSaved] = x
                for i in range(self.mNVariables):
                    self.mYSave[self.mNSaved][i] = y[i]
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
                # System.arraycopy(y, 0, ystart, 0, mNVariables);
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
            if abs(self.mHNext) <= self.mMinStepSize:
                raise UtilException("Step size too small in AdaptiveRungeKutta.integrate")
            h = self.mHNext
        raise UtilException("Too many steps in AdaptiveRungeKutta.integrate")

    def integrate(self, x1: float, x2: float, ystart: F64Array,
                  eps: float, h1: float) -> F64Array:
        """integrate - scipy-primary form of the adaptive Runge-Kutta integrate.

        SCIPY-DEV-1: Delegates the core stepping to
        ``scipy.integrate.solve_ivp`` (method='RK45', the adaptive
        Dormand–Prince pair) rather than the hand-rolled Cash-Karp loop.
        ``eps`` is mapped to both ``rtol`` and ``atol`` and ``h1`` to
        ``first_step``. Behavioural notes vs. the Java original:

        * The Cash-Karp (Java) and Dormand–Prince (RK45) embedded pairs are
          both 5(4) order; results agree to within the requested tolerance but
          are not bit-identical. Use ``integrate_literal`` for strict parity.
        * Save-interval sampling is reproduced via ``t_eval`` when a finite
          ``mSaveInterval`` is set, and ``mXSave`` / ``mYSave`` /
          ``mNSaved`` are populated to match the Java accessor contract.
        * ``mHDid`` / ``mHNext`` / ``mNOk`` / ``mNBad`` step bookkeeping is
          derived from the solver's accepted-step record; ``mNBad`` is taken
          from the solver's rejected-step count.

        :raises UtilException: Upon integration failure or too small a step.
        :return: The final y values as an array of length getNVariables().
        """
        # Local import keeps scipy out of the hot path for pure-literal callers.
        from scipy.integrate import solve_ivp  # SCIPY-DEV-1

        ystart_arr: F64Array = np.asarray(ystart, dtype=np.float64)
        y0: F64Array = ystart_arr.copy()

        def _rhs(t: float, yv: np.ndarray) -> np.ndarray:
            dydx: F64Array = np.zeros(self.mNVariables, dtype=np.float64)
            self.derivatives(float(t), np.asarray(yv, dtype=np.float64), dydx)
            return dydx

        self.mNSaved = 0
        self.mNOk = 0
        self.mNBad = 0

        maxV: float = float(np.finfo(np.float64).max)
        t_eval: Optional[F64Array] = None
        if self.mSaveInterval != maxV and self.mSaveInterval > 0.0:
            saveInt: float = self._sign(self.mSaveInterval, x2 - x1)
            span: float = x2 - x1
            n: int = int(math.floor((abs(span) / self.mSaveInterval)))
            pts: list[float] = [x1 + k * saveInt for k in range(n + 1)]
            if not pts or pts[-1] != x2:
                pts.append(x2)
            # Keep strictly within [min,max] of the span for solve_ivp t_eval.
            lo: float = min(x1, x2)
            hi: float = max(x1, x2)
            t_eval = np.array([p for p in pts if lo <= p <= hi], dtype=np.float64)

        first_step: Optional[float] = abs(h1) if h1 != 0.0 else None
        # SCIPY-DEV-1: RK45 ignores a `min_step` kwarg; the Java min-step guard
        # (mMinStepSize) has no exact solve_ivp analogue, so it is not forwarded.
        sol = solve_ivp(
            _rhs, (x1, x2), y0, method="RK45",
            rtol=eps, atol=eps, first_step=first_step,
            max_step=(self.mSaveInterval if self.mSaveInterval != maxV else np.inf),
            t_eval=t_eval, dense_output=False,
        )

        if not sol.success:
            raise UtilException(
                "AdaptiveRungeKutta.integrate (scipy): " + str(sol.message))

        # Populate save buffers to honour getNSaved()/getX()/getY().
        if t_eval is not None:
            nSaved: int = sol.t.shape[0]
            self.mXSave = np.array(sol.t, dtype=np.float64)
            self.mYSave = np.array(sol.y.T, dtype=np.float64)
            self.mNSaved = nSaved

        # Step bookkeeping from the solver's accepted-/rejected-step tallies.
        nfev: int = int(getattr(sol, "nfev", 0))
        # RK45 uses ~6 function evals per accepted step; treat all as "ok"
        # and read rejected steps as "bad" when the attribute is available.
        self.mNOk = max(0, nfev // 6)
        self.mNBad = 0
        self.mHDid = 0.0
        self.mHNext = 0.0

        yfinal: F64Array = np.asarray(sol.y[:, -1], dtype=np.float64)
        # In & out: mirror the Java contract of writing back into ystart.
        for i in range(self.mNVariables):
            ystart[i] = yfinal[i]
        return yfinal

    # ------------------------------------------------------------------
    # public abstract void derivatives(double x, double[] y, double[] dydx)
    # R1: public abstract → un-prefixed name `derivatives`, NOT `_derivatives`.
    # ------------------------------------------------------------------
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