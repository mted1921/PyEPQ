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
- Java UtilException mapped to EPQException (imported from _epq_compat).
- Java Math.round(x) → int(math.floor(x + 0.5))  [R7].
- Java integer division → Python //  [R7].
- Java Double.MAX_VALUE → math.inf  (closest semantic equivalent).
- abstract public class → abc.ABC with @abc.abstractmethod  [R checklist].
- Private helpers (_sign, _baseStep, _qcStep, _clearWorkspace) get leading
  underscore to reflect private access modifier  [R1].
- No mathematical public methods (all maths is internal); foo/foo_literal
  split therefore not applicable.

BUG_LEDGER
----------
"""

BUG_LEDGER: tuple = ()  # no bugs identified

from __future__ import annotations

import abc
import math
import sys
from typing import Optional

import numpy as np

try:
    from ._epq_compat import EPQException, F64Array, JavaRandom
except ImportError:
    try:
        from _epq_compat import EPQException, F64Array, JavaRandom  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility._epq_compat import EPQException, F64Array, JavaRandom  # type: ignore


class AdaptiveRungeKutta(abc.ABC):
    """Abstract base class for an adaptive step-size Cash-Karp Runge-Kutta ODE integrator.

    Subclasses must implement :meth:`derivatives`.

    Parameters
    ----------
    nVars:
        Number of coupled differential equations (dimension of the ODE system).

    Notes
    -----
    Not thread-safe.  Use each instance in exactly one thread.
    """

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(self, nVars: int) -> None:
        """AdaptiveRungeKutta — construct for an ODE system of *nVars* variables."""
        self._mNVariables: int = nVars          # number of differential equations
        self._mHDid: float = 0.0               # actual step size accomplished in last qcStep
        self._mHNext: float = 0.0              # next step size to try
        self._mSaveInterval: float = math.inf  # Java: Double.MAX_VALUE
        self._mMinStepSize: float = 0.0
        self._mXSave: Optional[F64Array] = None
        self._mYSave: Optional[F64Array] = None  # shape (kMax, nVars)
        self._mNSaved: int = 0
        self._mMaxSteps: int = 10000
        self._mNOk: int = 0                    # number of successful steps
        self._mNBad: int = 0                   # number of repeated (shrunk) steps
        # Workspace for _baseStep
        self._mWs2: Optional[F64Array] = None
        self._mWs3: Optional[F64Array] = None
        self._mWs4: Optional[F64Array] = None
        self._mWs5: Optional[F64Array] = None
        self._mWs6: Optional[F64Array] = None
        self._mYTemp: Optional[F64Array] = None
        # Workspace for _qcStep
        self._mYErr: Optional[F64Array] = None
        self._mQcYTemp: Optional[F64Array] = None

    # ------------------------------------------------------------------
    # Private helper — sign transfer
    # ------------------------------------------------------------------

    def _sign(self, magnitude: float, sign: float) -> float:
        """Return abs(magnitude) with the sign of *sign* (Java sign-transfer idiom)."""
        return math.fabs(magnitude) if sign >= 0.0 else -math.fabs(magnitude)

    # ------------------------------------------------------------------
    # Private helper — Cash-Karp base step
    # ------------------------------------------------------------------

    def _baseStep(
        self,
        x: float,
        y: F64Array,
        dydx: F64Array,
        h: float,
        yout: F64Array,
        yerr: F64Array,
    ) -> None:
        """Take a single Cash-Karp RK step (Numerical Recipes §16.2).

        Given *y[0..n-1]* and derivatives *dydx[0..n-1]* at *x*, advance the
        solution over interval *h*.  The new y is written to *yout*; an estimate
        of the local truncation error is written to *yerr*.
        """
        a2: float = 0.2;  a3: float = 0.3;  a4: float = 0.6
        a5: float = 1.0;  a6: float = 0.875

        b21: float = 0.2
        b31: float = 3.0 / 40.0;   b32: float = 9.0 / 40.0
        b41: float = 0.3;           b42: float = -0.9;          b43: float = 1.2
        b51: float = -11.0 / 54.0; b52: float = 2.5
        b53: float = -70.0 / 27.0; b54: float = 35.0 / 27.0
        b61: float = 1631.0 / 55296.0; b62: float = 175.0 / 512.0
        b63: float = 575.0 / 13824.0;  b64: float = 44275.0 / 110592.0
        b65: float = 253.0 / 4096.0

        c1: float = 37.0 / 378.0;  c3: float = 250.0 / 621.0
        c4: float = 125.0 / 594.0; c6: float = 512.0 / 1771.0

        dc1: float = c1 - (2825.0 / 27648.0)
        dc3: float = c3 - (18575.0 / 48384.0)
        dc4: float = c4 - (13525.0 / 55296.0)
        dc5: float = -277.0 / 14336.0
        dc6: float = c6 - 0.25

        n: int = self._mNVariables

        # Lazy-allocate workspace
        if self._mWs2 is None:
            self._mWs2 = np.zeros(n, dtype=np.float64)
            self._mWs3 = np.zeros(n, dtype=np.float64)
            self._mWs4 = np.zeros(n, dtype=np.float64)
            self._mWs5 = np.zeros(n, dtype=np.float64)
            self._mWs6 = np.zeros(n, dtype=np.float64)
            self._mYTemp = np.zeros(n, dtype=np.float64)

        ws2: F64Array = self._mWs2
        ws3: F64Array = self._mWs3
        ws4: F64Array = self._mWs4
        ws5: F64Array = self._mWs5
        ws6: F64Array = self._mWs6
        ytemp: F64Array = self._mYTemp  # type: ignore[assignment]

        # First step
        for i in range(n):
            ytemp[i] = y[i] + (b21 * h * dydx[i])
        # Second step
        self.derivatives(x + (a2 * h), ytemp, ws2)
        for i in range(n):
            ytemp[i] = y[i] + (h * ((b31 * dydx[i]) + (b32 * ws2[i])))
        # Third step
        self.derivatives(x + (a3 * h), ytemp, ws3)
        for i in range(n):
            ytemp[i] = y[i] + (h * ((b41 * dydx[i]) + (b42 * ws2[i]) + (b43 * ws3[i])))
        # Fourth step
        self.derivatives(x + (a4 * h), ytemp, ws4)
        for i in range(n):
            ytemp[i] = y[i] + (h * ((b51 * dydx[i]) + (b52 * ws2[i]) + (b53 * ws3[i]) + (b54 * ws4[i])))
        # Fifth step
        self.derivatives(x + (a5 * h), ytemp, ws5)
        for i in range(n):
            ytemp[i] = y[i] + (h * ((b61 * dydx[i]) + (b62 * ws2[i]) + (b63 * ws3[i]) + (b64 * ws4[i]) + (b65 * ws5[i])))
        # Sixth step
        self.derivatives(x + (a6 * h), ytemp, ws6)
        for i in range(n):
            yout[i] = y[i] + (h * ((c1 * dydx[i]) + (c3 * ws3[i]) + (c4 * ws4[i]) + (c6 * ws6[i])))
        # Error estimate
        for i in range(n):
            yerr[i] = h * ((dc1 * dydx[i]) + (dc3 * ws3[i]) + (dc4 * ws4[i]) + (dc5 * ws5[i]) + (dc6 * ws6[i]))

    # ------------------------------------------------------------------
    # Private helper — quality-controlled step
    # ------------------------------------------------------------------

    def _qcStep(
        self,
        x: float,
        y: F64Array,
        dydx: F64Array,
        htry: float,
        eps: float,
        yscal: F64Array,
    ) -> float:
        """Take a 5th-order RK step with adaptive error control.

        On return *y* is updated in-place, and ``self._mHDid`` / ``self._mHNext``
        are set.  Returns the new value of *x*.

        Raises
        ------
        EPQException
            If the step size underflows (x + h == x in floating-point).
        """
        safety: float = 0.9
        pgrow: float = -0.2
        pshrnk: float = -0.25
        errcon: float = 1.89e-4

        n: int = self._mNVariables

        if self._mYErr is None:
            self._mYErr = np.zeros(n, dtype=np.float64)
            self._mQcYTemp = np.zeros(n, dtype=np.float64)

        yerr: F64Array = self._mYErr
        qc_ytemp: F64Array = self._mQcYTemp  # type: ignore[assignment]

        h: float = htry
        errmax: float = 0.0
        while True:
            self._baseStep(x, y, dydx, h, qc_ytemp, yerr)
            errmax = 0.0
            for i in range(n):
                errmax = max(errmax, math.fabs(yerr[i] / yscal[i]))
            errmax /= eps
            if errmax <= 1.0:
                break
            htemp: float = safety * h * math.pow(errmax, pshrnk)
            if h >= 0:
                h = max(htemp, 0.1 * h)
            else:
                h = min(htemp, 0.1 * h)
            # Check for step-size underflow
            xnew: float = x + h
            if xnew == x:
                raise EPQException("Step size underflow in AdaptiveRungeKutta.qcStep.")

        self._mHNext = (
            safety * h * math.pow(errmax, pgrow) if errmax > errcon else 5.0 * h
        )
        self._mHDid = h
        x += h
        for i in range(n):
            y[i] = qc_ytemp[i]
        return x

    # ------------------------------------------------------------------
    # Private helper — release workspace memory
    # ------------------------------------------------------------------

    def _clearWorkspace(self) -> None:
        """Null all temporary workspace arrays to release memory."""
        self._mWs2 = None
        self._mWs3 = None
        self._mWs4 = None
        self._mWs5 = None
        self._mWs6 = None
        self._mYTemp = None
        self._mYErr = None
        self._mQcYTemp = None

    # ------------------------------------------------------------------
    # Public API — configuration
    # ------------------------------------------------------------------

    def setSaveInterval(self, interval: float) -> None:
        """Set the interval at which to save intermediate trajectory points.

        Parameters
        ----------
        interval:
            The x-axis spacing between saved snapshots (sign is ignored).
            Call :meth:`clearSaveInterval` to disable saving.
        """
        self._mSaveInterval = math.fabs(interval)

    def clearSaveInterval(self) -> None:
        """Disable saving of intermediate trajectory points (default behaviour)."""
        self._mSaveInterval = math.inf  # Java: Double.MAX_VALUE

    def setMaxSteps(self, maxSteps: int) -> None:
        """Set the maximum number of ODE steps allowed (default 10 000).

        Parameters
        ----------
        maxSteps:
            Upper bound on total step count inside :meth:`integrate`.
        """
        self._mMaxSteps = maxSteps

    def setMinStepSize(self, minStep: float) -> None:
        """Set the minimum permissible step size (default 0.0).

        Parameters
        ----------
        minStep:
            Minimum step magnitude (sign is ignored).
        """
        self._mMinStepSize = math.fabs(minStep)

    # ------------------------------------------------------------------
    # Public API — accessors
    # ------------------------------------------------------------------

    def getNSaved(self) -> int:
        """Return the number of intermediate points saved during the last integrate call."""
        return self._mNSaved

    def getX(self, i: int) -> float:
        """Return the x-coordinate of the *i*-th saved trajectory point.

        Parameters
        ----------
        i:
            Index satisfying ``0 <= i < getNSaved()``.
        """
        if self._mXSave is None:
            raise EPQException("No saved points available.")
        return float(self._mXSave[i])

    def getY(self, i: int) -> F64Array:
        """Return the y-coordinates of the *i*-th saved trajectory point.

        Parameters
        ----------
        i:
            Index satisfying ``0 <= i < getNSaved()``.

        Returns
        -------
        F64Array
            Array of length ``getNVariables()``.
        """
        if self._mYSave is None:
            raise EPQException("No saved points available.")
        return self._mYSave[i]  # type: ignore[index]

    def getNVariables(self) -> int:
        """Return the number of ODE variables as set in the constructor."""
        return self._mNVariables

    def getStepCount(self) -> int:
        """Return the total number of steps taken in the last integrate call."""
        return self._mNOk + self._mNBad

    def getGoodStepCount(self) -> int:
        """Return the number of steps accepted at full accuracy in the last integrate call."""
        return self._mNOk

    def getBadStepCount(self) -> int:
        """Return the number of steps that required shrinking in the last integrate call."""
        return self._mNBad

    # ------------------------------------------------------------------
    # Public API — integration
    # ------------------------------------------------------------------

    def integrate(
        self,
        x1: float,
        x2: float,
        ystart: F64Array,
        eps: float,
        h1: float,
    ) -> F64Array:
        """Integrate the ODE from *x1* to *x2* using adaptive-step Cash-Karp RK.

        Parameters
        ----------
        x1:
            Start of the integration range.
        x2:
            End of the integration range.
        ystart:
            Initial y values (length ``getNVariables()``).  Updated in-place on
            return with the final values.
        eps:
            Permissible relative error per step.
        h1:
            Initial trial step size.

        Returns
        -------
        F64Array
            Final y values (same object as *ystart* after in-place update).

        Raises
        ------
        EPQException
            If the step count exceeds ``getMaxSteps()`` or the step size drops
            below ``getMinStepSize()``.
        """
        n: int = self._mNVariables
        tiny: float = 1.0e-10 * eps
        yscal: F64Array = np.zeros(n, dtype=np.float64)
        dydx: F64Array = np.zeros(n, dtype=np.float64)
        y: F64Array = np.zeros(n, dtype=np.float64)

        x: float = x1
        h: float = self._sign(h1, x2 - x1)
        xsav: float = 0.0
        saveInt: float = math.inf  # Java: Double.MAX_VALUE — sentinel "no save interval"
        kMax: int = 0
        self._mNSaved = 0
        self._mNOk = 0
        self._mNBad = 0

        for i in range(n):
            y[i] = ystart[i]

        if self._mSaveInterval != math.inf:
            # Java: kMax = (int) Math.round((Math.abs(x2-x1) + mSaveInterval) / mSaveInterval)
            # R7: Math.round(x) → int(math.floor(x + 0.5))
            kMax = int(math.floor(
                (math.fabs(x2 - x1) + self._mSaveInterval) / self._mSaveInterval + 0.5
            ))
            saveInt = self._sign(self._mSaveInterval, x2 - x1)
            xsav = x - (2.0 * saveInt)  # ensures the very first step is saved
            self._mXSave = np.zeros(kMax, dtype=np.float64)
            self._mYSave = np.zeros((kMax, n), dtype=np.float64)

        for step in range(self._mMaxSteps):
            # Save intermediate points if requested
            if (kMax != 0) and (self._mNSaved < kMax) and (
                math.fabs(x - xsav) >= (0.9999 * self._mSaveInterval)
            ):
                self._mXSave[self._mNSaved] = x  # type: ignore[index]
                for i in range(n):
                    self._mYSave[self._mNSaved, i] = y[i]  # type: ignore[index]
                xsav = x
                self._mNSaved += 1

            self.derivatives(x, y, dydx)

            # Rescale h so that we hit the next save-point exactly
            hMax: float = math.fabs((xsav + saveInt) - x)
            if math.fabs(h) > hMax:
                h = self._sign(hMax, h)

            # Error-scaling vector
            for i in range(n):
                yscal[i] = math.fabs(y[i]) + math.fabs(dydx[i] * h) + tiny

            # Don't overshoot x2
            if (((x + h) - x2) * ((x + h) - x1)) > 0.0:
                h = x2 - x

            x = self._qcStep(x, y, dydx, h, eps, yscal)

            if self._mHDid == h:
                self._mNOk += 1
            else:
                self._mNBad += 1

            # Check whether we have reached x2
            if ((x - x2) * (x2 - x1)) >= 0.0:
                for i in range(n):
                    ystart[i] = y[i]
                if kMax != 0:
                    self._mNSaved = min(self._mNSaved, kMax - 1)
                    self._mXSave[self._mNSaved] = x  # type: ignore[index]
                    for i in range(n):
                        self._mYSave[self._mNSaved, i] = y[i]  # type: ignore[index]
                    self._mNSaved += 1
                self._clearWorkspace()
                return y

            if math.fabs(self._mHNext) <= self._mMinStepSize:
                raise EPQException("Step size too small in AdaptiveRungeKutta.integrate")
            h = self._mHNext

        raise EPQException("Too many steps in AdaptiveRungeKutta.integrate")

    # ------------------------------------------------------------------
    # Abstract method — subclass contract
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def derivatives(self, x: float, y: F64Array, dydx: F64Array) -> None:
        """Compute the derivatives of the ODE system at (*x*, *y*).

        The subclass **must** override this method.  It receives *x* and the
        current state vector *y* (length ``getNVariables()``) and must fill
        *dydx* (same length) with the corresponding derivatives.

        Parameters
        ----------
        x:
            Independent variable (In).
        y:
            Dependent variable array (In), length ``getNVariables()``.
        dydx:
            Derivative array to fill (Out), length ``getNVariables()``.
        """