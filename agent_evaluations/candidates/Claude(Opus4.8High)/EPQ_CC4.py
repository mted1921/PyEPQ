r"""
AdaptiveRungeKutta_ver1_1_0.py — Python port of gov.nist.microanalysis.Utility.AdaptiveRungeKutta

Guide version : 1
Generation    : 1
Port-code fixes: 0

CHANGES IN THIS REVISION (ver1.1.0)
-----------------------------------
* R2: `integrate` is a public mathematical method, so it is provided in two
  forms: `integrate()` — the primary public API, which delegates to
  `integrate_literal()` — and `integrate_literal()`, the line-for-line
  faithful translation of the Java Cash-Karp adaptive-stepper. No scipy
  substitution is used as the primary here: the Java algorithm also populates
  `mNSaved`/`mXSave`/`mYSave`, `mNOk`/`mNBad`, and the intermediate save-point
  buffers, none of which a `scipy.integrate.solve_ivp` substitution would
  reproduce faithfully. A scipy-backed convenience path is offered separately
  as `integrate_scipy()` (SCIPY-DEV-1) and is NOT wired in as the default so
  that the documented side-effecting state stays Java-exact.
* R1: `derivatives` is `public abstract` — the un-prefixed Java name is kept
  (never `_derivatives`), so concrete Python subclasses override `derivatives`.
* R1: private helpers `sign`, `baseStep`, `qcStep`, `clearWorkspace` map to
  `_sign`, `_baseStep`, `_qcStep`, `_clearWorkspace` (`_` for Java `private`).
* R7: `Math.round(...)` in `integrate` becomes `int(math.floor(x + 0.5))`.
* R9: every field, parameter, return, and non-obvious local is annotated.

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


__all__ = ["AdaptiveRungeKutta", "EPQException", "F64Array", "JavaRandom"]


class AdaptiveRungeKutta(abc.ABC):

    # ==================================================================
    # Preserved-bug ledger (machine-readable)
    # ==================================================================
    # Each tuple: (id, method, description, has_strict_variant).
    BUG_LEDGER: tuple = ()  # no bugs identified

    # ==================================================================
    # Constructor / fields
    # ==================================================================
    def __init__(self, nVars: int) -> None:
        """AdaptiveRungeKutta - Construct an AdaptiveRungeKutta object to solve
        a differential equation of nVars variables. The implementation of
        derivatives should return nVars derivative values for each x & y.

        Java: public AdaptiveRungeKutta(int nVars)
        """
        # private final int mNVariables; // The number of differential equations
        self.mNVariables: int = nVars
        # private double mHDid; // Actual step size accomplished in last call to qcStep
        self.mHDid: float = 0.0
        # private double mHNext; // Next step size to try when calling qcStep
        self.mHNext: float = 0.0
        # private double mSaveInterval = Double.MAX_VALUE;
        self.mSaveInterval: float = float(np.finfo(np.float64).max)
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

    # ==================================================================
    # Internal guards (R5)
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
    # Private helpers
    # ==================================================================
    def _sign(self, magnitude: float, sign: float) -> float:
        """Java: private double sign(double magnitude, double sign)

        Note: this is the class's own helper, NOT Math.signum. For
        ``sign >= 0.0`` it returns +|magnitude|; otherwise -|magnitude|.
        Faithfully preserved (including the sign==0.0 -> +|magnitude| case).
        """
        return abs(magnitude) if sign >= 0.0 else -abs(magnitude)

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
        n=mNDimensions values y[0..n-1] and their derivatives dydx[0..n-1]
        know at x, use a fifth order Cash-Karp Runge-Kutta method to advance
        the solution over an interval h. The resulting y value is returned in
        yout. An estimate of the truncation error is returned in yerr.

        Java: private void baseStep(double x, double[] y, double[] dydx,
                                    double h, double[] yout, double[] yerr)
        """
        self._require_mutable_f64(yout, "yout")
        self._require_mutable_f64(yerr, "yerr")
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
        assert self.mWs2 is not None and self.mWs3 is not None
        assert self.mWs4 is not None and self.mWs5 is not None
        assert self.mWs6 is not None and self.mYTemp is not None
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
        """qcStep - Take a fifth order Runge-Kutta step with monitoring of
        local truncation error. Input are the dependent variable
        y[0..mNDimensions-1] and its derivatives dydx[0..mNDimensions-1] at
        the starting value of the independent variable x. Also input is the
        attempted step size htry, the required accuracy eps and the vector
        yscal against which the errors are scaled. Upon return, y is replaced
        with the new values, x is returned and mHDid and mHNext are set to the
        actual step size and the size of the next step to try.

        Java: private double qcStep(double x, double[] y, double[] dydx,
                                    double htry, double eps, double[] yscal)
                                    throws UtilException
        """
        self._require_mutable_f64(y, "y")
        safety: float = 0.9
        pgrow: float = -0.2
        pshrnk: float = -0.25
        errcon: float = 1.89e-4
        if self.mYErr is None:
            self.mYErr = np.zeros(self.mNVariables, dtype=np.float64)
            self.mQcYTemp = np.zeros(self.mNVariables, dtype=np.float64)
        assert self.mYErr is not None and self.mQcYTemp is not None
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
                    raise EPQException("Step size underflow in AdaptiveRungeKutta.qcStep.")
            if not (errmax > 1.0):
                break
        self.mHNext = (safety * h * math.pow(errmax, pgrow) if errmax > errcon else 5.0 * h)
        self.mHDid = h
        x += h
        # System.arraycopy(mQcYTemp, 0, y, 0, mNVariables);
        y[0:self.mNVariables] = self.mQcYTemp[0:self.mNVariables]
        return x

    def _clearWorkspace(self) -> None:
        """clearWorkspace - null all temporary space to free memory

        Java: private void clearWorkspace()
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
        """setSaveInterval - Set the interval on which to save intermediate
        points on the integrated trajectory. (Use clearSaveInterval to not
        save any intermediate points.) Note: The default is not to save any
        intermediate points.

        Java: public void setSaveInterval(double interval)
        """
        self.mSaveInterval = abs(interval)

    def clearSaveInterval(self) -> None:
        """clearSaveInterval - Return to the default of not saving any
        intermediate points.

        Java: public void clearSaveInterval()
        """
        self.mSaveInterval = float(np.finfo(np.float64).max)

    def getNSaved(self) -> int:
        """getNSaved - Returns the number of saved values.

        Java: public int getNSaved()
        """
        return self.mNSaved

    def getX(self, i: int) -> float:
        """getX - Returns the x-coordinate of the i-th saved value

        Java: public double getX(int i)
        """
        assert self.mXSave is not None
        return float(self.mXSave[i])

    def getY(self, i: int) -> F64Array:
        """getY - returns the getNVariable x y-coordinates of the i-th saved
        values.

        Java: public double[] getY(int i)
        """
        assert self.mYSave is not None
        return self.mYSave[i]

    def setMaxSteps(self, maxSteps: int) -> None:
        """setMaxSteps - Set the maximum number of ODE steps to allow. Default
        is 10000.

        Java: public void setMaxSteps(int maxSteps)
        """
        self.mMaxSteps = maxSteps

    def setMinStepSize(self, minStep: float) -> None:
        """setMinStepSize - Sets the minimum permissible step size. Default is
        0.0.

        Java: public void setMinStepSize(double minStep)
        """
        self.mMinStepSize = abs(minStep)

    def getNVariables(self) -> int:
        """getNVariables - Returns the number of variables as set in the
        constructor.

        Java: public int getNVariables()
        """
        return self.mNVariables

    def getStepCount(self) -> int:
        """getStepCount - Get the total number of steps required to perform the
        previous integrate operation.

        Java: public int getStepCount()
        """
        return self.mNOk + self.mNBad

    def getGoodStepCount(self) -> int:
        """getGoodStepCount - Get the number of steps leading to results of the
        desired accuracy.

        Java: public int getGoodStepCount()
        """
        return self.mNOk

    def getBadStepCount(self) -> int:
        """getBadStepCount - Get the number of steps that were needed to be
        subdivided to attain results of the desired accuracy.

        Java: public int getBadStepCount()
        """
        return self.mNBad

    # ==================================================================
    # integrate  (R2: primary + _literal)
    # ==================================================================
    def integrate(
        self,
        x1: float,
        x2: float,
        ystart: F64Array,
        eps: float,
        h1: float,
    ) -> F64Array:
        """integrate - Integrate the ODE specified by derivatives using the
        adaptive step size Runge-Kutta algorithm over the independent variable
        interval x1 to x2. ystart contains the initial y values. eps is measure
        of the permissible error. h is the initial step size.

        Java: public double[] integrate(double x1, double x2, double[] ystart,
                                        double eps, double h1) throws UtilException

        Primary public API (R2). Delegates to ``integrate_literal`` — the
        faithful Java translation — because the algorithm's side effects on
        mNSaved / mXSave / mYSave / mNOk / mNBad are part of the public
        contract and cannot be reproduced by a library substitution. A
        scipy-backed convenience path is available as ``integrate_scipy``.
        """
        return self.integrate_literal(x1, x2, ystart, eps, h1)

    def integrate_literal(
        self,
        x1: float,
        x2: float,
        ystart: F64Array,
        eps: float,
        h1: float,
    ) -> F64Array:
        """Line-for-line faithful translation of the Java ``integrate``."""
        self._require_mutable_f64(ystart, "ystart")
        tiny: float = 1.0e-10 * eps
        yscal: F64Array = np.zeros(self.mNVariables, dtype=np.float64)
        dydx: F64Array = np.zeros(self.mNVariables, dtype=np.float64)
        y: F64Array = np.zeros(self.mNVariables, dtype=np.float64)
        x: float = x1
        h: float = self._sign(h1, x2 - x1)
        xsav: float = 0.0
        saveInt: float = float(np.finfo(np.float64).max)
        kMax: int = 0
        self.mNSaved = 0
        self.mNOk = 0
        self.mNBad = 0
        # System.arraycopy(ystart, 0, y, 0, mNVariables);
        y[0:self.mNVariables] = ystart[0:self.mNVariables]
        if self.mSaveInterval != float(np.finfo(np.float64).max):
            # R7: Math.round(x) -> int(math.floor(x + 0.5))
            kMax = int(math.floor(((abs(x2 - x1) + self.mSaveInterval) / self.mSaveInterval) + 0.5))
            saveInt = self._sign(self.mSaveInterval, x2 - x1)
            xsav = x - (2.0 * saveInt)  # to ensure that the first step is saved...
            self.mXSave = np.zeros(kMax, dtype=np.float64)
            self.mYSave = np.zeros((kMax, self.mNVariables), dtype=np.float64)
        for step in range(self.mMaxSteps):
            # Save the necessary points
            if (kMax != 0) and (self.mNSaved < kMax) and (abs(x - xsav) >= (0.9999 * self.mSaveInterval)):
                assert self.mXSave is not None and self.mYSave is not None
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
                # System.arraycopy(y, 0, ystart, 0, mNVariables);
                ystart[0:self.mNVariables] = y[0:self.mNVariables]
                if kMax != 0:
                    assert self.mXSave is not None and self.mYSave is not None
                    self.mNSaved = min(self.mNSaved, kMax - 1)
                    self.mXSave[self.mNSaved] = x
                    self.mYSave[self.mNSaved][0:self.mNVariables] = y[0:self.mNVariables]
                    self.mNSaved += 1
                self._clearWorkspace()
                return y
            if abs(self.mHNext) <= self.mMinStepSize:
                raise EPQException("Step size too small in AdaptiveRungeKutta.integrate")
            h = self.mHNext
        raise EPQException("Too many steps in AdaptiveRungeKutta.integrate")

    def integrate_scipy(
        self,
        x1: float,
        x2: float,
        ystart: F64Array,
        eps: float,
        h1: float,
    ) -> F64Array:
        """SCIPY-DEV-1: scipy-backed convenience path (NOT the default).

        Wraps ``scipy.integrate.solve_ivp`` with an RK45 solver. This does
        NOT populate mNSaved / mXSave / mYSave / mNOk / mNBad and therefore
        is not a drop-in for ``integrate``; it is offered only as an
        independent cross-check of the final endpoint value. The initial
        step-size hint ``h1`` is passed as ``first_step``; ``eps`` maps to
        both ``rtol`` and ``atol``.
        """
        from scipy.integrate import solve_ivp  # local import; optional path

        def _rhs(x: float, y: F64Array) -> F64Array:
            dydx: F64Array = np.zeros(self.mNVariables, dtype=np.float64)
            self.derivatives(x, np.asarray(y, dtype=np.float64), dydx)
            return dydx

        sol = solve_ivp(
            _rhs,
            (x1, x2),
            np.asarray(ystart, dtype=np.float64).copy(),
            method="RK45",
            rtol=eps,
            atol=eps,
            first_step=(abs(h1) if h1 != 0.0 else None),
        )
        yend: F64Array = np.asarray(sol.y[:, -1], dtype=np.float64)
        self._require_mutable_f64(ystart, "ystart")
        ystart[0:self.mNVariables] = yend[0:self.mNVariables]
        return yend

    # ==================================================================
    # Abstract method  (R1: public abstract -> un-prefixed name)
    # ==================================================================
    @abc.abstractmethod
    def derivatives(self, x: float, y: F64Array, dydx: F64Array) -> None:
        """derivatives - The derived class provides an implementation of the
        derivatives function. x & y[] are input and the user provided
        implementation of derivatives is resposible for returning the
        derivatives in the array dydx. The lengths of y and dydx are equal to
        mNDimensions.

        Java: abstract public void derivatives(double x, double[] y, double[] dydx)
        """
        ...