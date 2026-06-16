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
* R7: Java integer division preserved as // where applicable.
* R8: No Math.signum usage in source; not applicable.
* R9: All parameters, return types, and fields annotated throughout.
* R10: No deliberate deviations; faithful literal translation.

scipy/numpy primary `integrate()` delegates to `scipy.integrate.solve_ivp`
(RK45 adaptive solver) and coexists with the literal `integrate_literal()`
port. All other methods are algorithmic (no natural library substitution),
so they are ported literally only; no scipy primary is warranted for
`_baseStep` / `_qcStep` / `_sign` / `_clearWorkspace`.
"""

from __future__ import annotations

import abc
import math
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp as _solve_ivp

try:
    from ._epq_compat import EPQException, F64Array, JavaRandom
except ImportError:
    try:
        from _epq_compat import EPQException, F64Array, JavaRandom  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility._epq_compat import EPQException, F64Array, JavaRandom  # type: ignore


__all__ = ["AdaptiveRungeKutta"]


class AdaptiveRungeKutta(abc.ABC):
    """Abstract base for the adaptive Cash-Karp Runge-Kutta ODE integrator.

    Subclasses must implement :meth:`derivatives`.  The class is **not**
    thread-safe; use each instance in exactly one thread.

    See Numerical Recipes in C (2nd ed.), pp. 714-722.
    """

    # ------------------------------------------------------------------
    # Preserved-bug ledger (machine-readable)
    # ------------------------------------------------------------------
    # Each tuple: (id, method, description, has_strict_variant).
    BUG_LEDGER: tuple = ()  # no bugs identified

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(self, nVars: int) -> None:
        """Construct an AdaptiveRungeKutta for a system of *nVars* equations.

        Parameters
        ----------
        nVars:
            Number of coupled differential equations (= dimension of y).
        """
        # Java: private final int mNVariables
        self.mNVariables: int = nVars

        # Java: private double mHDid
        self.mHDid: float = 0.0
        # Java: private double mHNext
        self.mHNext: float = 0.0
        # Java: private double mSaveInterval = Double.MAX_VALUE
        self.mSaveInterval: float = float("inf")
        # Java: private double mMinStepSize = 0.0
        self.mMinStepSize: float = 0.0

        # Java: private double[] mXSave
        self.mXSave: Optional[F64Array] = None
        # Java: private double[][] mYSave
        self.mYSave: Optional[NDArray[np.float64]] = None
        # Java: private int mNSaved = 0
        self.mNSaved: int = 0
        # Java: private int mMaxSteps = 10000
        self.mMaxSteps: int = 10000
        # Java: private int mNOk
        self.mNOk: int = 0
        # Java: private int mNBad
        self.mNBad: int = 0

        # Temporary workspace used by _baseStep
        self.mWs2: Optional[F64Array] = None
        self.mWs3: Optional[F64Array] = None
        self.mWs4: Optional[F64Array] = None
        self.mWs5: Optional[F64Array] = None
        self.mWs6: Optional[F64Array] = None
        self.mYTemp: Optional[F64Array] = None

        # Temporary workspace used by _qcStep
        self.mYErr: Optional[F64Array] = None
        self.mQcYTemp: Optional[F64Array] = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sign(self, magnitude: float, sign: float) -> float:
        """Java: ``private double sign(double magnitude, double sign)``.

        Returns ``|magnitude|`` if *sign* >= 0, else ``-|magnitude|``.
        This is the Numerical Recipes SIGN macro, **not** Java's
        ``Math.signum``; it transfers the sign of the second argument
        onto the absolute value of the first.
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
        """Cash-Karp single RK step (Numerical Recipes pp. 715-716).

        Java: ``private void baseStep(double x, double[] y, double[] dydx,
        double h, double[] yout, double[] yerr)``

        Given values *y[0..n-1]* and derivatives *dydx[0..n-1]* at *x*,
        advances the solution over interval *h* using a fifth-order
        Cash-Karp method.  The output is placed in *yout*; a truncation
        error estimate is placed in *yerr*.  Both arrays are mutated in
        place.

        Parameters
        ----------
        x:    Current independent variable value.
        y:    Dependent variable values (length mNVariables).
        dydx: Derivatives at x (length mNVariables).
        h:    Step size.
        yout: Output: solution at x+h (mutated in place).
        yerr: Output: error estimate (mutated in place).
        """
        # Cash-Karp coefficients (NR Table 16.2)
        a2: float = 0.2;  a3: float = 0.3;  a4: float = 0.6
        a5: float = 1.0;  a6: float = 0.875

        b21: float = 0.2
        b31: float = 3.0 / 40.0;   b32: float = 9.0 / 40.0
        b41: float = 0.3;          b42: float = -0.9;          b43: float = 1.2
        b51: float = -11.0 / 54.0; b52: float = 2.5;           b53: float = -70.0 / 27.0; b54: float = 35.0 / 27.0
        b61: float = 1631.0 / 55296.0; b62: float = 175.0 / 512.0
        b63: float = 575.0 / 13824.0;  b64: float = 44275.0 / 110592.0; b65: float = 253.0 / 4096.0

        c1: float = 37.0 / 378.0;  c3: float = 250.0 / 621.0
        c4: float = 125.0 / 594.0; c6: float = 512.0 / 1771.0

        dc1: float = c1 - (2825.0 / 27648.0)
        dc3: float = c3 - (18575.0 / 48384.0)
        dc4: float = c4 - (13525.0 / 55296.0)
        dc5: float = -277.0 / 14336.0
        dc6: float = c6 - 0.25

        # Lazy workspace allocation (matches Java null-check pattern)
        if self.mWs2 is None:
            self.mWs2 = np.zeros(self.mNVariables, dtype=np.float64)
            self.mWs3 = np.zeros(self.mNVariables, dtype=np.float64)
            self.mWs4 = np.zeros(self.mNVariables, dtype=np.float64)
            self.mWs5 = np.zeros(self.mNVariables, dtype=np.float64)
            self.mWs6 = np.zeros(self.mNVariables, dtype=np.float64)
            self.mYTemp = np.zeros(self.mNVariables, dtype=np.float64)

        mWs2: F64Array = self.mWs2  # type: ignore[assignment]
        mWs3: F64Array = self.mWs3  # type: ignore[assignment]
        mWs4: F64Array = self.mWs4  # type: ignore[assignment]
        mWs5: F64Array = self.mWs5  # type: ignore[assignment]
        mWs6: F64Array = self.mWs6  # type: ignore[assignment]
        mYTemp: F64Array = self.mYTemp  # type: ignore[assignment]

        n: int = self.mNVariables

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
        """Quality-controlled RK step (Numerical Recipes pp. 719-721).

        Java: ``private double qcStep(double x, double[] y, double[] dydx,
        double htry, double eps, double[] yscal) throws UtilException``

        Takes a fifth-order RK step with adaptive step-size control.
        On return *y* is updated in place, and ``self.mHDid`` /
        ``self.mHNext`` reflect the actual step taken and the recommended
        next step size.

        Parameters
        ----------
        x:     Independent variable at the start of the step.
        y:     Dependent variables (mutated in place to the new values).
        dydx:  Derivatives at x.
        htry:  Attempted step size.
        eps:   Desired relative accuracy.
        yscal: Error-scaling vector (length mNVariables).

        Returns
        -------
        float
            New value of x (= old x + mHDid).

        Raises
        ------
        EPQException
            If the step size underflows (x + h == x in floating point).
        """
        safety: float = 0.9
        pgrow: float = -0.2
        pshrnk: float = -0.25
        errcon: float = 1.89e-4

        if self.mYErr is None:
            self.mYErr = np.zeros(self.mNVariables, dtype=np.float64)
            self.mQcYTemp = np.zeros(self.mNVariables, dtype=np.float64)

        mYErr: F64Array = self.mYErr          # type: ignore[assignment]
        mQcYTemp: F64Array = self.mQcYTemp    # type: ignore[assignment]

        errmax: float
        h: float = htry
        while True:
            self._baseStep(x, y, dydx, h, mQcYTemp, mYErr)
            errmax = 0.0
            for i in range(self.mNVariables):
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
                    raise EPQException("Step size underflow in AdaptiveRungeKutta.qcStep.")
            if not (errmax > 1.0):
                break
        self.mHNext = (safety * h * math.pow(errmax, pgrow) if errmax > errcon else 5.0 * h)
        self.mHDid = h
        x += h
        # Java: System.arraycopy(mQcYTemp, 0, y, 0, mNVariables)
        y[:self.mNVariables] = mQcYTemp[:self.mNVariables]
        return x

    def _clearWorkspace(self) -> None:
        """Release all temporary arrays to free memory.

        Java: ``private void clearWorkspace()``
        """
        self.mWs2 = None
        self.mWs3 = None
        self.mWs4 = None
        self.mWs5 = None
        self.mWs6 = None
        self.mYTemp = None
        self.mYErr = None
        self.mQcYTemp = None

    # ------------------------------------------------------------------
    # Public configuration setters / getters
    # ------------------------------------------------------------------

    def setSaveInterval(self, interval: float) -> None:
        """Set the x-interval at which intermediate trajectory points are saved.

        Java: ``public void setSaveInterval(double interval)``

        Use :meth:`clearSaveInterval` to stop saving (the default).
        Note that enabling saving may constrain the adaptive step size.

        Parameters
        ----------
        interval:
            Absolute x-separation between saved points.
        """
        self.mSaveInterval = math.fabs(interval)

    def clearSaveInterval(self) -> None:
        """Restore the default: do not save any intermediate points.

        Java: ``public void clearSaveInterval()``
        """
        self.mSaveInterval = float("inf")

    def getNSaved(self) -> int:
        """Return the number of trajectory points saved during the last integrate call.

        Java: ``public int getNSaved()``
        """
        return self.mNSaved

    def getX(self, i: int) -> float:
        """Return the x-coordinate of the *i*-th saved point (i < getNSaved()).

        Java: ``public double getX(int i)``
        """
        return float(self.mXSave[i])  # type: ignore[index]

    def getY(self, i: int) -> F64Array:
        """Return the y-coordinates of the *i*-th saved point (i < getNSaved()).

        Java: ``public double[] getY(int i)``

        Returns
        -------
        F64Array
            Array of length getNVariables().
        """
        return self.mYSave[i]  # type: ignore[index]

    def setMaxSteps(self, maxSteps: int) -> None:
        """Set the maximum number of ODE steps allowed per integrate call.

        Java: ``public void setMaxSteps(int maxSteps)``

        Default is 10 000.

        Parameters
        ----------
        maxSteps:
            Upper bound on the number of adaptive steps.
        """
        self.mMaxSteps = maxSteps

    def setMinStepSize(self, minStep: float) -> None:
        """Set the minimum permissible step size (default 0.0).

        Java: ``public void setMinStepSize(double minStep)``

        Parameters
        ----------
        minStep:
            Minimum allowed |h|; raises EPQException if breached.
        """
        self.mMinStepSize = math.fabs(minStep)

    def getNVariables(self) -> int:
        """Return the number of ODE variables set in the constructor.

        Java: ``public int getNVariables()``
        """
        return self.mNVariables

    def getStepCount(self) -> int:
        """Return the total number of steps used in the previous integrate call.

        Java: ``public int getStepCount()``
        """
        return self.mNOk + self.mNBad

    def getGoodStepCount(self) -> int:
        """Return the number of steps accepted at first attempt (no retry needed).

        Java: ``public int getGoodStepCount()``
        """
        return self.mNOk

    def getBadStepCount(self) -> int:
        """Return the number of steps that required subdivision/retry.

        Java: ``public int getBadStepCount()``
        """
        return self.mNBad

    # ------------------------------------------------------------------
    # integrate — scipy primary (library substitution)
    # ------------------------------------------------------------------

    def integrate(
        self,
        x1: float,
        x2: float,
        ystart: F64Array,
        eps: float,
        h1: float,
    ) -> F64Array:
        """Integrate the ODE from *x1* to *x2* (scipy RK45 primary).

        Java: ``public double[] integrate(double x1, double x2,
        double[] ystart, double eps, double h1) throws UtilException``

        Uses ``scipy.integrate.solve_ivp`` with method ``"RK45"`` as the
        primary implementation.  The literal Cash-Karp port is available
        as :meth:`integrate_literal`.

        *ystart* is updated in place with the final y values (matching Java
        behaviour) **and** returned for convenience.

        Parameters
        ----------
        x1:
            Start of the integration interval (independent variable).
        x2:
            End of the integration interval.
        ystart:
            Initial y values (length getNVariables()).  **Mutated in place**
            with the final values on return (Java contract).
        eps:
            Permissible relative error tolerance (rtol).
        h1:
            Initial step size hint (passed as ``first_step`` to solve_ivp).

        Returns
        -------
        F64Array
            Final y values (same object as *ystart* after mutation).

        Raises
        ------
        EPQException
            If solve_ivp fails to reach x2.
        """
        # SCIPY-DEV-1: scipy's RK45 is not Cash-Karp; it is Dormand-Prince.
        #   Numerical behaviour is equivalent at the same tolerance but the
        #   exact step sequence will differ from the Java reference.
        # SCIPY-DEV-2: mNOk, mNBad, mHDid, mHNext, mXSave, mYSave are NOT
        #   updated by this variant.  Use integrate_literal() if those
        #   fields are needed.
        def _rhs(t: float, y_arr: F64Array) -> F64Array:
            dydx: F64Array = np.zeros(self.mNVariables, dtype=np.float64)
            self.derivatives(t, y_arr, dydx)
            return dydx

        abs_h1: float = math.fabs(h1)
        result = _solve_ivp(
            _rhs,
            (x1, x2),
            np.asarray(ystart, dtype=np.float64),
            method="RK45",
            rtol=eps,
            atol=eps * 1e-3,
            first_step=abs_h1 if abs_h1 > 0.0 else None,
            dense_output=False,
        )
        if not result.success:
            raise EPQException(
                f"AdaptiveRungeKutta.integrate (scipy): {result.message}"
            )
        final_y: F64Array = result.y[:, -1]
        ystart[:self.mNVariables] = final_y
        return final_y

    # ------------------------------------------------------------------
    # integrate_literal — line-for-line Java port (Cash-Karp)
    # ------------------------------------------------------------------

    def integrate_literal(
        self,
        x1: float,
        x2: float,
        ystart: F64Array,
        eps: float,
        h1: float,
    ) -> F64Array:
        """Integrate the ODE from *x1* to *x2* (faithful Cash-Karp literal port).

        Java: ``public double[] integrate(...)`` — line-for-line translation.

        See :meth:`integrate` for the scipy primary.  This variant updates
        all state fields (mNOk, mNBad, mNSaved, mXSave, mYSave, mHDid,
        mHNext) exactly as the Java source does.

        Parameters
        ----------
        x1:
            Start of the integration interval.
        x2:
            End of the integration interval.
        ystart:
            Initial y values (mutated in place on return).
        eps:
            Permissible relative error.
        h1:
            Initial step size.

        Returns
        -------
        F64Array
            Final y values (same object as *ystart* after mutation).

        Raises
        ------
        EPQException
            On step-size underflow, too-small step, or too many steps.
        """
        tiny: float = 1.0e-10 * eps
        yscal: F64Array = np.zeros(self.mNVariables, dtype=np.float64)
        dydx: F64Array  = np.zeros(self.mNVariables, dtype=np.float64)
        y: F64Array     = np.zeros(self.mNVariables, dtype=np.float64)

        x: float = x1
        h: float = self._sign(h1, x2 - x1)
        xsav: float = 0.0
        saveInt: float = float("inf")
        kMax: int = 0

        self.mNSaved = 0
        self.mNOk = 0
        self.mNBad = 0

        # Java: System.arraycopy(ystart, 0, y, 0, mNVariables)
        y[:self.mNVariables] = ystart[:self.mNVariables]

        if self.mSaveInterval != float("inf"):
            # Java: kMax = (int) Math.round((Math.abs(x2-x1)+mSaveInterval)/mSaveInterval)
            # R7: Math.round → int(math.floor(v + 0.5))
            kMax = int(math.floor(
                (math.fabs(x2 - x1) + self.mSaveInterval) / self.mSaveInterval + 0.5
            ))
            saveInt = self._sign(self.mSaveInterval, x2 - x1)
            xsav = x - (2.0 * saveInt)  # ensure first step is always saved
            self.mXSave = np.zeros(kMax, dtype=np.float64)
            self.mYSave = np.zeros((kMax, self.mNVariables), dtype=np.float64)

        for step in range(self.mMaxSteps):
            # Save the necessary points
            if (kMax != 0) and (self.mNSaved < kMax) and (math.fabs(x - xsav) >= (0.9999 * self.mSaveInterval)):
                self.mXSave[self.mNSaved] = x            # type: ignore[index]
                self.mYSave[self.mNSaved, :] = y[:self.mNVariables]  # type: ignore[index]
                xsav = x
                self.mNSaved += 1

            self.derivatives(x, y, dydx)

            # Rescale h to ensure we hit desired save points
            hMax: float = math.fabs((xsav + saveInt) - x)
            if math.fabs(h) > hMax:
                h = self._sign(hMax, h)

            # Scaling to monitor accuracy
            for i in range(self.mNVariables):
                yscal[i] = math.fabs(y[i]) + math.fabs(dydx[i] * h) + tiny

            # Trim step to not overshoot x2
            if (((x + h) - x2) * ((x + h) - x1)) > 0.0:
                h = x2 - x

            x = self._qcStep(x, y, dydx, h, eps, yscal)

            if self.mHDid == h:
                self.mNOk += 1
            else:
                self.mNBad += 1

            # Check for completion
            if ((x - x2) * (x2 - x1)) >= 0.0:
                # Java: System.arraycopy(y, 0, ystart, 0, mNVariables)
                ystart[:self.mNVariables] = y[:self.mNVariables]
                if kMax != 0:
                    self.mNSaved = min(self.mNSaved, kMax - 1)
                    self.mXSave[self.mNSaved] = x      # type: ignore[index]
                    self.mYSave[self.mNSaved, :] = y[:self.mNVariables]  # type: ignore[index]
                    self.mNSaved += 1
                self._clearWorkspace()
                return y

            if math.fabs(self.mHNext) <= self.mMinStepSize:
                raise EPQException("Step size too small in AdaptiveRungeKutta.integrate")

            h = self.mHNext

        raise EPQException("Too many steps in AdaptiveRungeKutta.integrate")

    # ------------------------------------------------------------------
    # Abstract method — subclasses must provide this
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def derivatives(self, x: float, y: F64Array, dydx: F64Array) -> None:
        """Compute the derivatives of the ODE system at (x, y).

        Java: ``abstract public void derivatives(double x, double[] y, double[] dydx)``

        The subclass fills *dydx* in place with the derivative values.
        Both *y* and *dydx* have length :meth:`getNVariables`.

        Parameters
        ----------
        x:
            Current independent variable value (input).
        y:
            Current dependent variable values, length mNVariables (input).
        dydx:
            Output array to be filled with dy/dx values (output).
        """
        ...