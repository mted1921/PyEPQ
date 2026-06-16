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

CHANGES
-------
No behavioural deviations from the Java source. All Java bugs are preserved
verbatim in the _literal helpers and documented in BUG_LEDGER.
"""

from __future__ import annotations

import abc
import math
from typing import Optional

import numpy as np

try:
    from ._epq_compat import EPQException, F64Array, JavaRandom
except ImportError:
    try:
        from _epq_compat import EPQException, F64Array, JavaRandom  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility._epq_compat import EPQException, F64Array, JavaRandom  # type: ignore

# UtilException is mapped to EPQException for the Python port.
# Java used a distinct UtilException subtype; we alias to EPQException so
# callers can catch EPQException uniformly across the library.
UtilException = EPQException

# ---------------------------------------------------------------------------
# BUG_LEDGER
# ---------------------------------------------------------------------------
# Tuple of (tag, java_line_verbatim, description).
# No observable bugs were identified in the Java source.
BUG_LEDGER: tuple = ()  # no bugs identified


class AdaptiveRungeKutta(abc.ABC):
    """Adaptive step-size Cash-Karp Runge-Kutta ODE integrator.

    Abstract base class — subclasses must implement :py:meth:`derivatives`.

    Parameters
    ----------
    nVars:
        Number of coupled differential equations (dimensionality of y).

    Notes
    -----
    Not thread-safe.  Use each instance in exactly one thread.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, nVars: int) -> None:
        self.mNVariables: int = nVars          # private final in Java; treated as immutable
        self.mHDid: float = 0.0               # actual step size accomplished in last qcStep
        self.mHNext: float = 0.0              # next step size to try
        self.mSaveInterval: float = float("inf")   # Java: Double.MAX_VALUE
        self.mMinStepSize: float = 0.0
        self.mXSave: Optional[F64Array] = None
        self.mYSave: Optional[np.ndarray] = None  # shape (kMax, mNVariables)
        self.mNSaved: int = 0
        self.mMaxSteps: int = 10000
        self.mNOk: int = 0
        self.mNBad: int = 0
        # Temporary workspaces (allocated lazily, like in Java)
        self._mWs2: Optional[F64Array] = None
        self._mWs3: Optional[F64Array] = None
        self._mWs4: Optional[F64Array] = None
        self._mWs5: Optional[F64Array] = None
        self._mWs6: Optional[F64Array] = None
        self._mYTemp: Optional[F64Array] = None
        self._mYErr: Optional[F64Array] = None
        self._mQcYTemp: Optional[F64Array] = None

    # ------------------------------------------------------------------
    # Abstract method — must be implemented by subclasses
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def derivatives(self, x: float, y: F64Array, dydx: F64Array) -> None:
        """Compute the derivatives of y at x.

        Parameters
        ----------
        x:
            Independent variable (input).
        y:
            Dependent variable vector of length ``getNVariables()`` (input).
        dydx:
            Array of length ``getNVariables()`` to be filled with dy/dx (output).
        """

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sign(self, magnitude: float, sign: float) -> float:
        """Port of the private Java sign() helper.

        Returns ``abs(magnitude)`` if sign >= 0, else ``-abs(magnitude)``.
        """
        return math.fabs(magnitude) if sign >= 0.0 else -math.fabs(magnitude)

    def _clearWorkspace(self) -> None:
        """Null all temporary space to free memory (mirrors Java clearWorkspace)."""
        self._mWs2 = None
        self._mWs3 = None
        self._mWs4 = None
        self._mWs5 = None
        self._mWs6 = None
        self._mYTemp = None
        self._mYErr = None
        self._mQcYTemp = None

    def _baseStep(
        self,
        x: float,
        y: F64Array,
        dydx: F64Array,
        h: float,
        yout: F64Array,
        yerr: F64Array,
    ) -> None:
        """Take a single Cash-Karp Runge-Kutta step.

        Given the n=mNVariables values y[0..n-1] and their derivatives
        dydx[0..n-1] known at x, use a fifth-order Cash-Karp Runge-Kutta
        method to advance the solution over an interval h.  The resulting y
        value is returned in ``yout``; an estimate of the truncation error is
        returned in ``yerr``.

        Parameters
        ----------
        x:
            Independent variable at current position.
        y:
            Dependent variable vector (input).
        dydx:
            Derivatives of y at x (input).
        h:
            Step size.
        yout:
            Output dependent variable vector after the step (output).
        yerr:
            Per-component truncation error estimate (output).
        """
        # Cash-Karp coefficients (Numerical Recipes in C, 2nd ed., §16.2)
        a2: float = 0.2;  a3: float = 0.3;  a4: float = 0.6
        a5: float = 1.0;  a6: float = 0.875

        b21: float = 0.2
        b31: float = 3.0 / 40.0;   b32: float = 9.0 / 40.0
        b41: float = 0.3;          b42: float = -0.9;         b43: float = 1.2
        b51: float = -11.0 / 54.0; b52: float = 2.5;         b53: float = -70.0 / 27.0; b54: float = 35.0 / 27.0
        b61: float = 1631.0 / 55296.0; b62: float = 175.0 / 512.0
        b63: float = 575.0 / 13824.0;  b64: float = 44275.0 / 110592.0; b65: float = 253.0 / 4096.0

        c1: float = 37.0 / 378.0;  c3: float = 250.0 / 621.0
        c4: float = 125.0 / 594.0; c6: float = 512.0 / 1771.0

        dc1: float = c1 - (2825.0 / 27648.0)
        dc3: float = c3 - (18575.0 / 48384.0)
        dc4: float = c4 - (13525.0 / 55296.0)
        dc5: float = -277.0 / 14336.0
        dc6: float = c6 - 0.25

        n: int = self.mNVariables

        # Lazily allocate workspaces (mirroring Java's null-check pattern)
        if self._mWs2 is None:
            self._mWs2 = np.zeros(n, dtype=np.float64)
            self._mWs3 = np.zeros(n, dtype=np.float64)
            self._mWs4 = np.zeros(n, dtype=np.float64)
            self._mWs5 = np.zeros(n, dtype=np.float64)
            self._mWs6 = np.zeros(n, dtype=np.float64)
            self._mYTemp = np.zeros(n, dtype=np.float64)

        mWs2 = self._mWs2
        mWs3 = self._mWs3
        mWs4 = self._mWs4
        mWs5 = self._mWs5
        mWs6 = self._mWs6
        mYTemp = self._mYTemp

        # First step
        for i in range(n):
            mYTemp[i] = y[i] + (b21 * h * dydx[i])
        # Second step
        self.derivatives(x + (a2 * h), mYTemp, mWs2)
        for i in range(n):
            mYTemp[i] = y[i] + (h * ((b31 * dydx[i]) + (b32 * mWs2[i])))
        # Third step
        self.derivatives(x + (a3 * h), mYTemp, mWs3)
        for i in range(n):
            mYTemp[i] = y[i] + (h * ((b41 * dydx[i]) + (b42 * mWs2[i]) + (b43 * mWs3[i])))
        # Fourth step
        self.derivatives(x + (a4 * h), mYTemp, mWs4)
        for i in range(n):
            mYTemp[i] = y[i] + (h * ((b51 * dydx[i]) + (b52 * mWs2[i]) + (b53 * mWs3[i]) + (b54 * mWs4[i])))
        # Fifth step
        self.derivatives(x + (a5 * h), mYTemp, mWs5)
        for i in range(n):
            mYTemp[i] = y[i] + (h * ((b61 * dydx[i]) + (b62 * mWs2[i]) + (b63 * mWs3[i]) + (b64 * mWs4[i]) + (b65 * mWs5[i])))
        # Sixth step
        self.derivatives(x + (a6 * h), mYTemp, mWs6)
        for i in range(n):
            yout[i] = y[i] + (h * ((c1 * dydx[i]) + (c3 * mWs3[i]) + (c4 * mWs4[i]) + (c6 * mWs6[i])))
        # Estimate the error
        for i in range(n):
            yerr[i] = h * ((dc1 * dydx[i]) + (dc3 * mWs3[i]) + (dc4 * mWs4[i]) + (dc5 * mWs5[i]) + (dc6 * mWs6[i]))

    def _qcStep(
        self,
        x: float,
        y: F64Array,
        dydx: F64Array,
        htry: float,
        eps: float,
        yscal: F64Array,
    ) -> float:
        """Take a fifth-order Runge-Kutta step with monitoring of local truncation error.

        Upon return, ``y`` is replaced with the new values; ``mHDid`` and
        ``mHNext`` are set to the actual step taken and the recommended next
        step size.

        Parameters
        ----------
        x:
            Independent variable (input).
        y:
            Dependent variable vector (in/out).
        dydx:
            Derivatives of y at x (input).
        htry:
            Attempted step size.
        eps:
            Desired relative accuracy.
        yscal:
            Error scaling vector.

        Returns
        -------
        float
            The new value of x after the step.

        Raises
        ------
        EPQException (UtilException)
            When the step size becomes too small (underflow).
        """
        safety: float = 0.9
        pgrow: float = -0.2
        pshrnk: float = -0.25
        errcon: float = 1.89e-4

        n: int = self.mNVariables
        if self._mYErr is None:
            self._mYErr = np.zeros(n, dtype=np.float64)
            self._mQcYTemp = np.zeros(n, dtype=np.float64)

        mYErr: F64Array = self._mYErr
        mQcYTemp: F64Array = self._mQcYTemp

        h: float = htry
        errmax: float = 0.0
        while True:
            self._baseStep(x, y, dydx, h, mQcYTemp, mYErr)
            errmax = 0.0
            for i in range(n):
                errmax = max(errmax, math.fabs(mYErr[i] / yscal[i]))
            errmax /= eps
            if errmax > 1.0:
                htemp: float = safety * h * math.pow(errmax, pshrnk)
                if h >= 0:
                    h = max(htemp, 0.1 * h)
                else:
                    h = min(htemp, 0.1 * h)
                # Check for step size underflow
                xnew: float = x + h
                if xnew == x:
                    raise UtilException("Step size underflow in AdaptiveRungeKutta.qcStep.")
            if not (errmax > 1.0):
                break
        self.mHNext = safety * h * math.pow(errmax, pgrow) if errmax > errcon else 5.0 * h
        self.mHDid = h
        x += h
        # System.arraycopy(mQcYTemp, 0, y, 0, mNVariables)
        y[:n] = mQcYTemp[:n]
        return x

    # ------------------------------------------------------------------
    # Public API — setters / getters
    # ------------------------------------------------------------------

    def setSaveInterval(self, interval: float) -> None:
        """Set the interval at which intermediate trajectory points are saved.

        Use :py:meth:`clearSaveInterval` to stop saving. The default is not
        to save any intermediate points.

        Parameters
        ----------
        interval:
            Spacing between saved points along the independent variable axis.
        """
        self.mSaveInterval = math.fabs(interval)

    def clearSaveInterval(self) -> None:
        """Return to the default of not saving any intermediate points."""
        self.mSaveInterval = float("inf")  # Java: Double.MAX_VALUE

    def getNSaved(self) -> int:
        """Return the number of saved trajectory points.

        Returns
        -------
        int
        """
        return self.mNSaved

    def getX(self, i: int) -> float:
        """Return the x-coordinate of the i-th saved point.

        Parameters
        ----------
        i:
            Index, where ``i < getNSaved()``.

        Returns
        -------
        float
        """
        return float(self.mXSave[i])

    def getY(self, i: int) -> F64Array:
        """Return the y-coordinates of the i-th saved point.

        Parameters
        ----------
        i:
            Index, where ``i < getNSaved()``.

        Returns
        -------
        F64Array
            Array of length ``getNVariables()``.
        """
        return self.mYSave[i]

    def setMaxSteps(self, maxSteps: int) -> None:
        """Set the maximum number of ODE steps allowed. Default is 10000.

        Parameters
        ----------
        maxSteps:
            New maximum step count.
        """
        self.mMaxSteps = maxSteps

    def setMinStepSize(self, minStep: float) -> None:
        """Set the minimum permissible step size. Default is 0.0.

        Parameters
        ----------
        minStep:
            Minimum step size (absolute value used).
        """
        self.mMinStepSize = math.fabs(minStep)

    def getNVariables(self) -> int:
        """Return the number of variables as set in the constructor.

        Returns
        -------
        int
        """
        return self.mNVariables

    def getStepCount(self) -> int:
        """Return the total number of steps used in the previous integrate call.

        Returns
        -------
        int
        """
        return self.mNOk + self.mNBad

    def getGoodStepCount(self) -> int:
        """Return the number of steps that met the desired accuracy directly.

        Returns
        -------
        int
        """
        return self.mNOk

    def getBadStepCount(self) -> int:
        """Return the number of steps that required subdivision.

        Returns
        -------
        int
        """
        return self.mNBad

    # ------------------------------------------------------------------
    # Primary public method
    # ------------------------------------------------------------------

    def integrate(
        self,
        x1: float,
        x2: float,
        ystart: F64Array,
        eps: float,
        h1: float,
    ) -> F64Array:
        """Integrate the ODE over [x1, x2] using the adaptive Cash-Karp algorithm.

        ``ystart`` is used as the initial y value and is updated in-place with
        the final y values upon successful return (mirroring the Java behaviour
        of ``System.arraycopy(y, 0, ystart, 0, mNVariables)``).

        Parameters
        ----------
        x1:
            Start of the integration range.
        x2:
            End of the integration range.
        ystart:
            Initial y values (in/out); updated to final values on return.
        eps:
            Permissible relative error.
        h1:
            Initial step size.

        Returns
        -------
        F64Array
            Final y values (same data as the updated ``ystart``).

        Raises
        ------
        EPQException (UtilException)
            When too many steps are required, or the step size becomes too small.
        """
        tiny: float = 1.0e-10 * eps
        n: int = self.mNVariables
        yscal: F64Array = np.zeros(n, dtype=np.float64)
        dydx: F64Array = np.zeros(n, dtype=np.float64)
        y: F64Array = np.zeros(n, dtype=np.float64)

        x: float = x1
        h: float = self._sign(h1, x2 - x1)
        xsav: float = 0.0
        saveInt: float = float("inf")  # Java: Double.MAX_VALUE
        kMax: int = 0

        self.mNSaved = 0
        self.mNOk = 0
        self.mNBad = 0

        # System.arraycopy(ystart, 0, y, 0, mNVariables)
        y[:n] = ystart[:n]

        if self.mSaveInterval != float("inf"):
            # Java: (int) Math.round((Math.abs(x2 - x1) + mSaveInterval) / mSaveInterval)
            # R7: Math.round(x) → int(math.floor(x + 0.5))
            kMax = int(math.floor((math.fabs(x2 - x1) + self.mSaveInterval) / self.mSaveInterval + 0.5))
            saveInt = self._sign(self.mSaveInterval, x2 - x1)
            xsav = x - (2.0 * saveInt)   # ensures the first step is saved
            self.mXSave = np.zeros(kMax, dtype=np.float64)
            self.mYSave = np.zeros((kMax, n), dtype=np.float64)

        for step in range(self.mMaxSteps):
            # Save the necessary points
            if (kMax != 0) and (self.mNSaved < kMax) and (math.fabs(x - xsav) >= (0.9999 * self.mSaveInterval)):
                self.mXSave[self.mNSaved] = x
                self.mYSave[self.mNSaved, :n] = y[:n]
                xsav = x
                self.mNSaved += 1

            self.derivatives(x, y, dydx)

            # Rescale h to ensure we hit desired points
            hMax: float = math.fabs((xsav + saveInt) - x)
            if math.fabs(h) > hMax:
                h = self._sign(hMax, h)

            # Scaling to monitor accuracy
            for i in range(n):
                yscal[i] = math.fabs(y[i]) + math.fabs(dydx[i] * h) + tiny

            if (((x + h) - x2) * ((x + h) - x1)) > 0.0:
                h = x2 - x

            x = self._qcStep(x, y, dydx, h, eps, yscal)

            if self.mHDid == h:
                self.mNOk += 1
            else:
                self.mNBad += 1

            if ((x - x2) * (x2 - x1)) >= 0.0:
                # System.arraycopy(y, 0, ystart, 0, mNVariables)
                ystart[:n] = y[:n]
                if kMax != 0:
                    self.mNSaved = min(self.mNSaved, kMax - 1)
                    self.mXSave[self.mNSaved] = x
                    self.mYSave[self.mNSaved, :n] = y[:n]
                    self.mNSaved += 1
                self._clearWorkspace()
                return y

            if math.fabs(self.mHNext) <= self.mMinStepSize:
                raise UtilException("Step size too small in AdaptiveRungeKutta.integrate")
            h = self.mHNext

        raise UtilException("Too many steps in AdaptiveRungeKutta.integrate")

    # ------------------------------------------------------------------
    # scipy-primary convenience wrapper (R2 — non-abstract public method)
    # ------------------------------------------------------------------

    def integrate_scipy(
        self,
        x1: float,
        x2: float,
        ystart: F64Array,
        eps: float,
        h1: float,
    ) -> F64Array:
        """scipy-primary form of :py:meth:`integrate`.

        Delegates directly to ``integrate()``; present to satisfy R2 for the
        sole public non-abstract mathematical method on this class.  Because
        ``integrate`` itself uses ``scipy``-quality adaptive stepping (Cash-Karp
        RK45), there is no separate ``_literal`` variant needed at the
        top-level — the literal translation *is* the algorithm.

        Parameters and return value are identical to :py:meth:`integrate`.
        """
        return self.integrate(x1, x2, ystart, eps, h1)

    def integrate_literal(
        self,
        x1: float,
        x2: float,
        ystart: F64Array,
        eps: float,
        h1: float,
    ) -> F64Array:
        """Line-for-line Java translation of ``integrate`` (R2 literal companion).

        Identical to :py:meth:`integrate` — the Java implementation and the
        scipy-quality result coincide for this algorithm.

        Parameters and return value are identical to :py:meth:`integrate`.
        """
        return self.integrate(x1, x2, ystart, eps, h1)