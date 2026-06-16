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
None — first-generation literal port; no deliberate deviations from Java.

BUG_LEDGER
----------
No bugs identified in the Java source.
"""

from __future__ import annotations

import abc
import math
from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp  # used in integrate() scipy primary

try:
    from ._epq_compat import EPQException, F64Array, JavaRandom
except ImportError:
    try:
        from _epq_compat import EPQException, F64Array, JavaRandom  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility._epq_compat import EPQException, F64Array, JavaRandom  # type: ignore

# UtilException is EPQException in the Python port (UtilException_ver1_1_0.py
# re-exports EPQException under the UtilException name; we import EPQException
# directly because the _epq_compat module is the single source of truth).
UtilException = EPQException

__all__ = ["AdaptiveRungeKutta"]

# ---------------------------------------------------------------------------
# BUG_LEDGER — R6
# ---------------------------------------------------------------------------
BUG_LEDGER: tuple = ()  # no bugs identified


# ---------------------------------------------------------------------------
# AdaptiveRungeKutta
# ---------------------------------------------------------------------------

class AdaptiveRungeKutta(abc.ABC):
    """Adaptive step-size Cash-Karp Runge-Kutta ODE integrator.

    Subclasses must implement :py:meth:`derivatives`.

    NOTE: Not thread-safe — use each instance in exactly one thread.
    """

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(self, nVars: int) -> None:
        """Construct an AdaptiveRungeKutta to solve a system of *nVars*
        differential equations.

        Parameters
        ----------
        nVars:
            Number of dependent variables / differential equations.
        """
        self.mNVariables: int = nVars

        # Step-size tracking (set by _qcStep)
        self.mHDid: float = 0.0
        self.mHNext: float = 0.0

        # Save-interval control
        self.mSaveInterval: float = float("inf")   # Java: Double.MAX_VALUE
        self.mMinStepSize: float = 0.0

        # Saved trajectory arrays (allocated on first integrate call)
        self.mXSave: Optional[F64Array] = None
        self.mYSave: Optional[np.ndarray] = None   # shape (kMax, mNVariables)
        self.mNSaved: int = 0

        # Step-count limits and counters
        self.mMaxSteps: int = 10000
        self.mNOk: int = 0
        self.mNBad: int = 0

        # Workspace arrays (lazy-allocated inside _baseStep / _qcStep)
        self.mWs2: Optional[F64Array] = None
        self.mWs3: Optional[F64Array] = None
        self.mWs4: Optional[F64Array] = None
        self.mWs5: Optional[F64Array] = None
        self.mWs6: Optional[F64Array] = None
        self.mYTemp: Optional[F64Array] = None
        self.mYErr: Optional[F64Array] = None
        self.mQcYTemp: Optional[F64Array] = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sign(self, magnitude: float, sign: float) -> float:
        """Return abs(magnitude) with the sign of *sign*.

        Mirrors Java:
            ``return sign >= 0.0 ? Math.abs(magnitude) : -Math.abs(magnitude);``
        """
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
        """Take a single Cash-Karp Runge-Kutta step (private).

        Given the n=mNVariables values y[0..n-1] and their derivatives
        dydx[0..n-1] at x, advance the solution over interval h.  The
        resulting y values are written into *yout*; a truncation-error
        estimate is written into *yerr*.

        Parameters
        ----------
        x:
            Independent variable at the start of the step.
        y:
            Dependent variable values (input).
        dydx:
            Derivatives of y at x (input).
        h:
            Step size.
        yout:
            Output array for new y values (mutated in-place).
        yerr:
            Output array for error estimates (mutated in-place).
        """
        # Cash-Karp tableau constants (NR pp 717)
        a2: float = 0.2;  a3: float = 0.3;  a4: float = 0.6
        a5: float = 1.0;  a6: float = 0.875

        b21: float = 0.2
        b31: float = 3.0 / 40.0;  b32: float = 9.0 / 40.0
        b41: float = 0.3;   b42: float = -0.9;  b43: float = 1.2
        b51: float = -11.0 / 54.0;  b52: float = 2.5
        b53: float = -70.0 / 27.0;  b54: float = 35.0 / 27.0
        b61: float = 1631.0 / 55296.0;  b62: float = 175.0 / 512.0
        b63: float = 575.0 / 13824.0;   b64: float = 44275.0 / 110592.0
        b65: float = 253.0 / 4096.0

        c1: float = 37.0 / 378.0;   c3: float = 250.0 / 621.0
        c4: float = 125.0 / 594.0;  c6: float = 512.0 / 1771.0

        dc1: float = c1 - (2825.0 / 27648.0)
        dc3: float = c3 - (18575.0 / 48384.0)
        dc4: float = c4 - (13525.0 / 55296.0)
        dc5: float = -277.0 / 14336.0
        dc6: float = c6 - 0.25

        n: int = self.mNVariables

        # Lazy workspace allocation (mirrors Java null-check pattern)
        if self.mWs2 is None:
            self.mWs2 = np.zeros(n, dtype=np.float64)
            self.mWs3 = np.zeros(n, dtype=np.float64)
            self.mWs4 = np.zeros(n, dtype=np.float64)
            self.mWs5 = np.zeros(n, dtype=np.float64)
            self.mWs6 = np.zeros(n, dtype=np.float64)
            self.mYTemp = np.zeros(n, dtype=np.float64)

        mWs2: F64Array = self.mWs2   # type: ignore[assignment]
        mWs3: F64Array = self.mWs3   # type: ignore[assignment]
        mWs4: F64Array = self.mWs4   # type: ignore[assignment]
        mWs5: F64Array = self.mWs5   # type: ignore[assignment]
        mWs6: F64Array = self.mWs6   # type: ignore[assignment]
        mYTemp: F64Array = self.mYTemp  # type: ignore[assignment]

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
            mYTemp[i] = y[i] + (
                h * ((b51 * dydx[i]) + (b52 * mWs2[i]) + (b53 * mWs3[i]) + (b54 * mWs4[i]))
            )

        # Fifth step
        self.derivatives(x + (a5 * h), mYTemp, mWs5)
        for i in range(n):
            mYTemp[i] = y[i] + (
                h * (
                    (b61 * dydx[i])
                    + (b62 * mWs2[i])
                    + (b63 * mWs3[i])
                    + (b64 * mWs4[i])
                    + (b65 * mWs5[i])
                )
            )

        # Sixth step
        self.derivatives(x + (a6 * h), mYTemp, mWs6)
        for i in range(n):
            yout[i] = y[i] + (
                h * ((c1 * dydx[i]) + (c3 * mWs3[i]) + (c4 * mWs4[i]) + (c6 * mWs6[i]))
            )

        # Estimate the truncation error
        for i in range(n):
            yerr[i] = h * (
                (dc1 * dydx[i])
                + (dc3 * mWs3[i])
                + (dc4 * mWs4[i])
                + (dc5 * mWs5[i])
                + (dc6 * mWs6[i])
            )

    def _qcStep(
        self,
        x: float,
        y: F64Array,
        dydx: F64Array,
        htry: float,
        eps: float,
        yscal: F64Array,
    ) -> float:
        """Quality-controlled Runge-Kutta step with adaptive error monitoring.

        Takes a fifth-order Runge-Kutta step, monitors local truncation
        error, and adjusts the step size until the error is within *eps*.
        On return, ``self.mHDid`` holds the actual step taken and
        ``self.mHNext`` holds the recommended next step size.

        Parameters
        ----------
        x:
            Independent variable at the start of the step.
        y:
            Dependent variable values (modified in-place to new values).
        dydx:
            Derivatives at x.
        htry:
            Attempted step size.
        eps:
            Desired relative accuracy.
        yscal:
            Error-scaling vector.

        Returns
        -------
        float
            New value of x after the step.

        Raises
        ------
        EPQException (UtilException)
            If the step size underflows (x + h == x in floating-point).
        """
        safety: float = 0.9
        pgrow: float = -0.2
        pshrnk: float = -0.25
        errcon: float = 1.89e-4

        n: int = self.mNVariables

        if self.mYErr is None:
            self.mYErr = np.zeros(n, dtype=np.float64)
            self.mQcYTemp = np.zeros(n, dtype=np.float64)

        mYErr: F64Array = self.mYErr          # type: ignore[assignment]
        mQcYTemp: F64Array = self.mQcYTemp    # type: ignore[assignment]

        errmax: float
        h: float = htry

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
                # Check for step-size underflow
                xnew: float = x + h
                if xnew == x:
                    raise UtilException(
                        "Step size underflow in AdaptiveRungeKutta.qcStep."
                    )
            else:
                break  # errmax <= 1.0 — exit do-while

        self.mHNext = (
            safety * h * math.pow(errmax, pgrow)
            if errmax > errcon
            else 5.0 * h
        )
        self.mHDid = h
        x += h
        # Java: System.arraycopy(mQcYTemp, 0, y, 0, mNVariables)
        y[:n] = mQcYTemp[:n]
        return x

    def _clearWorkspace(self) -> None:
        """Release all temporary workspace arrays to free memory."""
        self.mWs2 = None
        self.mWs3 = None
        self.mWs4 = None
        self.mWs5 = None
        self.mWs6 = None
        self.mYTemp = None
        self.mYErr = None
        self.mQcYTemp = None

    # ------------------------------------------------------------------
    # Public setters / getters
    # ------------------------------------------------------------------

    def setSaveInterval(self, interval: float) -> None:
        """Set the interval on which to save intermediate trajectory points.

        Use :py:meth:`clearSaveInterval` to stop saving. Default: no saving.

        Parameters
        ----------
        interval:
            Spacing (in x) between saved points; always stored as positive.
        """
        self.mSaveInterval = math.fabs(interval)

    def clearSaveInterval(self) -> None:
        """Restore the default of not saving any intermediate points."""
        self.mSaveInterval = float("inf")   # Java: Double.MAX_VALUE

    def getNSaved(self) -> int:
        """Return the number of trajectory points saved during the last
        :py:meth:`integrate` call."""
        return self.mNSaved

    def getX(self, i: int) -> float:
        """Return the x-coordinate of the *i*-th saved point.

        Parameters
        ----------
        i:
            Index satisfying ``0 <= i < getNSaved()``.
        """
        assert self.mXSave is not None, "No saved points — integrate has not been called."
        return float(self.mXSave[i])

    def getY(self, i: int) -> F64Array:
        """Return the y-coordinates of the *i*-th saved point.

        Parameters
        ----------
        i:
            Index satisfying ``0 <= i < getNSaved()``.

        Returns
        -------
        F64Array
            Array of length ``getNVariables()``.
        """
        assert self.mYSave is not None, "No saved points — integrate has not been called."
        return self.mYSave[i]

    def setMaxSteps(self, maxSteps: int) -> None:
        """Set the maximum number of ODE steps allowed. Default is 10 000.

        Parameters
        ----------
        maxSteps:
            Upper bound on integration steps.
        """
        self.mMaxSteps = maxSteps

    def setMinStepSize(self, minStep: float) -> None:
        """Set the minimum permissible step size. Default is 0.0.

        Parameters
        ----------
        minStep:
            Lower bound on step size; stored as positive.
        """
        self.mMinStepSize = math.fabs(minStep)

    def getNVariables(self) -> int:
        """Return the number of dependent variables set in the constructor."""
        return self.mNVariables

    def getStepCount(self) -> int:
        """Return the total number of steps taken during the last
        :py:meth:`integrate` call."""
        return self.mNOk + self.mNBad

    def getGoodStepCount(self) -> int:
        """Return the number of steps that achieved the desired accuracy on
        the first attempt."""
        return self.mNOk

    def getBadStepCount(self) -> int:
        """Return the number of steps that required subdivision to reach the
        desired accuracy."""
        return self.mNBad

    # ------------------------------------------------------------------
    # Public integrate (scipy primary + literal)
    # ------------------------------------------------------------------

    def integrate(
        self,
        x1: float,
        x2: float,
        ystart: F64Array,
        eps: float,
        h1: float,
    ) -> F64Array:
        """Integrate the ODE from x1 to x2 using adaptive Cash-Karp RK.

        This is the **scipy primary** implementation.  It delegates to
        ``scipy.integrate.solve_ivp`` with ``method='RK45'`` and
        ``rtol=eps``, then reconstructs the save-interval trajectory by
        dense-output evaluation so that the saved-point API
        (``getNSaved`` / ``getX`` / ``getY``) still works.

        The *ystart* array is updated in-place (matching Java semantics).
        The final y values are also returned.

        Parameters
        ----------
        x1:
            Start of the integration range.
        x2:
            End of the integration range.
        ystart:
            Initial y values (modified in-place).
        eps:
            Permissible relative error (passed to scipy as ``rtol``).
        h1:
            Initial step-size hint (passed to scipy as ``first_step``).

        Returns
        -------
        F64Array
            Final y values (same object as *ystart*).

        Raises
        ------
        EPQException (UtilException)
            If scipy's integrator fails (status != 0).

        SCIPY-DEV-1: scipy handles step-size underflow internally; the
            literal's explicit ``UtilException("Step size underflow …")``
            path is replaced by a scipy status-check.
        SCIPY-DEV-2: The save-interval trajectory uses scipy dense output
            (``dense_output=True``) rather than the Java stepping loop.
        SCIPY-DEV-3: mNOk / mNBad / mMaxSteps are not propagated from
            scipy; mNOk is set to the number of evaluations, mNBad = 0.
        """
        n: int = self.mNVariables

        def _rhs(t: float, state: F64Array) -> F64Array:
            dydt: F64Array = np.zeros(n, dtype=np.float64)
            self.derivatives(t, state, dydt)
            return dydt

        # Build t_eval for save-interval points
        t_eval = None
        if self.mSaveInterval != float("inf"):
            import numpy as _np
            # R7: Java Math.round → int(math.floor(x + 0.5))
            kMax: int = int(math.floor(
                (math.fabs(x2 - x1) + self.mSaveInterval) / self.mSaveInterval + 0.5
            ))
            saveInt: float = self._sign(self.mSaveInterval, x2 - x1)
            xsav: float = x1 - (2.0 * saveInt)  # ensure first step is captured
            t_eval_list: list[float] = []
            t: float = xsav + saveInt
            while (len(t_eval_list) < kMax) and (
                (saveInt > 0 and t <= x2 + 1e-12) or
                (saveInt < 0 and t >= x2 - 1e-12)
            ):
                t_eval_list.append(t)
                t += saveInt
            # Always include x2
            if not t_eval_list or abs(t_eval_list[-1] - x2) > 1e-14:
                t_eval_list.append(x2)
            t_eval = _np.array(t_eval_list, dtype=np.float64)

        sol = solve_ivp(
            _rhs,
            (x1, x2),
            ystart.copy(),
            method="RK45",
            rtol=eps,
            atol=eps * 1e-3,
            first_step=abs(h1),
            dense_output=(t_eval is not None),
            t_eval=t_eval,
        )

        if not sol.success:
            raise UtilException(
                f"scipy.integrate.solve_ivp failed: {sol.message}"
            )

        # Populate the saved-point arrays (SCIPY-DEV-2)
        if t_eval is not None and sol.t.size > 0:
            ns: int = sol.t.size
            self.mXSave = np.array(sol.t, dtype=np.float64)
            self.mYSave = np.array(sol.y.T, dtype=np.float64)  # (ns, n)
            self.mNSaved = ns
        else:
            self.mNSaved = 0

        # Update ystart in-place (Java semantics) and return
        final_y: F64Array = np.array(sol.y[:, -1], dtype=np.float64)
        ystart[:n] = final_y
        # SCIPY-DEV-3: approximate step counters from scipy nfev
        self.mNOk = int(sol.nfev)
        self.mNBad = 0

        return final_y

    def integrate_literal(
        self,
        x1: float,
        x2: float,
        ystart: F64Array,
        eps: float,
        h1: float,
    ) -> F64Array:
        """Integrate the ODE from x1 to x2 — **literal Java translation**.

        Line-for-line port of the Java ``integrate`` method using the
        hand-rolled Cash-Karp stepper (``_qcStep`` / ``_baseStep``).

        Parameters
        ----------
        x1:
            Start of the integration range.
        x2:
            End of the integration range.
        ystart:
            Initial y values (modified in-place on success, matching Java).
        eps:
            Permissible relative error.
        h1:
            Initial step size.

        Returns
        -------
        F64Array
            Final y values (same object as *ystart*).

        Raises
        ------
        EPQException (UtilException)
            When the step size becomes too small, or the maximum number of
            steps is exceeded.
        """
        n: int = self.mNVariables
        tiny: float = 1.0e-10 * eps
        yscal: F64Array = np.zeros(n, dtype=np.float64)
        dydx: F64Array = np.zeros(n, dtype=np.float64)
        y: F64Array = np.zeros(n, dtype=np.float64)

        x: float = x1
        h: float = self._sign(h1, x2 - x1)
        xsav: float = 0.0
        saveInt: float = float("inf")   # Java: Double.MAX_VALUE

        kMax: int = 0
        self.mNSaved = 0
        self.mNOk = 0
        self.mNBad = 0

        # Java: System.arraycopy(ystart, 0, y, 0, mNVariables)
        y[:n] = ystart[:n]

        if self.mSaveInterval != float("inf"):
            # R7: Java Math.round → int(math.floor(x + 0.5))
            kMax = int(math.floor(
                (math.fabs(x2 - x1) + self.mSaveInterval) / self.mSaveInterval + 0.5
            ))
            saveInt = self._sign(self.mSaveInterval, x2 - x1)
            xsav = x - (2.0 * saveInt)   # ensure first step is saved
            self.mXSave = np.zeros(kMax, dtype=np.float64)
            self.mYSave = np.zeros((kMax, n), dtype=np.float64)

        for step in range(self.mMaxSteps):
            # Save the necessary points
            if (
                (kMax != 0)
                and (self.mNSaved < kMax)
                and (math.fabs(x - xsav) >= (0.9999 * self.mSaveInterval))
            ):
                self.mXSave[self.mNSaved] = x        # type: ignore[index]
                self.mYSave[self.mNSaved, :n] = y    # type: ignore[index]
                xsav = x
                self.mNSaved += 1

            self.derivatives(x, y, dydx)

            # Rescale h to ensure we hit desired save points
            hMax: float = math.fabs((xsav + saveInt) - x)
            if math.fabs(h) > hMax:
                h = self._sign(hMax, h)

            # Scaling to monitor accuracy
            for i in range(n):
                yscal[i] = math.fabs(y[i]) + math.fabs(dydx[i] * h) + tiny

            # Do not overshoot x2
            if (((x + h) - x2) * ((x + h) - x1)) > 0.0:
                h = x2 - x

            x = self._qcStep(x, y, dydx, h, eps, yscal)

            if self.mHDid == h:
                self.mNOk += 1
            else:
                self.mNBad += 1

            if ((x - x2) * (x2 - x1)) >= 0.0:
                # Java: System.arraycopy(y, 0, ystart, 0, mNVariables)
                ystart[:n] = y[:n]
                if kMax != 0:
                    self.mNSaved = min(self.mNSaved, kMax - 1)
                    self.mXSave[self.mNSaved] = x        # type: ignore[index]
                    self.mYSave[self.mNSaved, :n] = y    # type: ignore[index]
                    self.mNSaved += 1
                self._clearWorkspace()
                return y

            if math.fabs(self.mHNext) <= self.mMinStepSize:
                raise UtilException(
                    "Step size too small in AdaptiveRungeKutta.integrate"
                )
            h = self.mHNext

        raise UtilException("Too many steps in AdaptiveRungeKutta.integrate")

    # ------------------------------------------------------------------
    # Abstract method — public, no _ prefix (R1)
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def derivatives(self, x: float, y: F64Array, dydx: F64Array) -> None:
        """Compute the derivatives of the ODE system at (x, y).

        Subclasses **must** implement this method. On return, *dydx[i]*
        must contain d(y[i])/dx for i in 0..getNVariables()-1.

        Parameters
        ----------
        x:
            Independent variable (input).
        y:
            Dependent variable values of length ``getNVariables()`` (input).
        dydx:
            Output array of length ``getNVariables()``; must be filled with
            the derivative values (output).
        """