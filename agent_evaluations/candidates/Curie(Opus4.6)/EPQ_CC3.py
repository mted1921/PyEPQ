r"""
AdaptiveRungeKutta_ver1_1_0.py — Python port of gov.nist.microanalysis.Utility.AdaptiveRungeKutta

Guide version : 1
Generation    : 1
Port-code fixes: 0

------------------------------------------------------------------------
Original Java source (gov.nist.microanalysis.Utility.AdaptiveRungeKutta)
------------------------------------------------------------------------
/**
 * AdaptiveRungeKutta - A attempt to implement an adaptive step size Runge-Kutta
 * ODE integrator based on the Cash-Karp method as described in Numerical
 * Recipes in C.
 *
 * Not thread safe!
 *
 * @author nicholas
 */
------------------------------------------------------------------------
"""

from __future__ import annotations

import abc
import math
import sys
from typing import Optional, Sequence, Union, Callable

import numpy as np
from numpy.typing import NDArray

try:
    from ._epq_compat import EPQException, F64Array, JavaRandom
except ImportError:
    try:
        from _epq_compat import EPQException, F64Array, JavaRandom  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility._epq_compat import EPQException, F64Array, JavaRandom  # type: ignore

# UtilException in the Java code is just a RuntimeException subclass.
# We map it to EPQException (the single exception type from _epq_compat)
# since UtilException is functionally identical for port purposes.
# DEVIATION: UtilException → EPQException (same base; no UtilException port exists yet).

__all__ = ["AdaptiveRungeKutta"]

class AdaptiveRungeKutta(abc.ABC):
    """Adaptive step-size Runge-Kutta ODE integrator (Cash-Karp method).

    Translated from the Java implementation which follows the algorithm
    described in *Numerical Recipes in C*.

    CHANGES
    -------
    * UtilException mapped to EPQException (single exception type in the port).
    """

    # ==================================================================
    # Preserved-bug ledger (machine-readable)
    # ==================================================================
    BUG_LEDGER: tuple = ()  # no bugs identified

    # ==================================================================
    # Cash-Karp tableau constants (package-private / class-level)
    # ==================================================================
    _a2: float = 0.2
    _a3: float = 0.3
    _a4: float = 0.6
    _a5: float = 1.0
    _a6: float = 0.875

    _b21: float = 0.2

    _b31: float = 3.0 / 40.0
    _b32: float = 9.0 / 40.0

    _b41: float = 0.3
    _b42: float = -0.9
    _b43: float = 1.2

    _b51: float = -11.0 / 54.0
    _b52: float = 2.5
    _b53: float = -70.0 / 27.0
    _b54: float = 35.0 / 27.0

    _b61: float = 1631.0 / 55296.0
    _b62: float = 175.0 / 512.0
    _b63: float = 575.0 / 13824.0
    _b64: float = 44275.0 / 110592.0
    _b65: float = 253.0 / 4096.0

    _c1: float = 37.0 / 378.0
    _c3: float = 250.0 / 621.0
    _c4: float = 125.0 / 594.0
    _c6: float = 512.0 / 1771.0

    _dc1: float = (37.0 / 378.0) - (2825.0 / 27648.0)
    _dc3: float = (250.0 / 621.0) - (18575.0 / 48384.0)
    _dc4: float = (125.0 / 594.0) - (13525.0 / 55296.0)
    _dc5: float = -277.00 / 14336.0
    _dc6: float = (512.0 / 1771.0) - 0.25

    # Safety / tuning constants for step-size control
    _SAFETY: float = 0.9
    _PGROW: float = -0.2
    _PSHRINK: float = -0.25
    _ERRCON: float = 1.89e-4

    # ==================================================================
    # Constructor
    # ==================================================================
    def __init__(self, nVars: int) -> None:
        """Construct an integrator for a system of *nVars* first-order ODEs.

        Parameters
        ----------
        nVars : int
            Number of dependent variables (dimension of y).
        """
        self.mNVariables: int = nVars
        self.mHDid: float = 0.0
        self.mHNext: float = 0.0
        self.mSaveInterval: float = 0.0
        self.mMinStepSize: float = 0.0
        self.mXSave: Optional[F64Array] = None
        self.mYSave: Optional[F64Array] = None  # 2-D, shape (kMax, mNVariables)
        self.mNSaved: int = 0
        self.mMaxSteps: int = 10000
        self.mNOk: int = 0
        self.mNBad: int = 0

        # Workspace arrays for _baseStep (Cash-Karp RK step)
        self.mWs2: Optional[F64Array] = None
        self.mWs3: Optional[F64Array] = None
        self.mWs4: Optional[F64Array] = None
        self.mWs5: Optional[F64Array] = None
        self.mWs6: Optional[F64Array] = None
        self.mYTemp: Optional[F64Array] = None

        # Workspace arrays for _qcStep (quality-controlled step)
        self.mYErr: Optional[F64Array] = None
        self.mQcYTemp: Optional[F64Array] = None

    # ==================================================================
    # _sign helper  (Java's private sign(double, double))
    # ==================================================================
    def _sign(self, magnitude: float, sign: float) -> float:
        """Return ``abs(magnitude)`` with the sign of *sign*.

        Java semantics: ``sign(m, -0.0)`` returns ``+|m|`` because
        ``-0.0 >= 0.0`` is ``True`` in Java (IEEE-754).  Python's
        ``math.copysign`` would return ``-|m|`` for ``-0.0``, so we
        use the explicit conditional to match Java exactly.
        """
        return abs(magnitude) if sign >= 0.0 else -abs(magnitude)

    # ==================================================================
    # _clearWorkspace  (Java's private clearWorkspace)
    # ==================================================================
    def _clearWorkspace(self) -> None:
        """Release workspace arrays after integration completes."""
        self.mWs2 = None
        self.mWs3 = None
        self.mWs4 = None
        self.mWs5 = None
        self.mWs6 = None
        self.mYTemp = None
        self.mYErr = None
        self.mQcYTemp = None

    # ==================================================================
    # _baseStep  (Java's private baseStep)
    # Cash-Karp Runge-Kutta step — no error control
    # ==================================================================
    def _baseStep(
        self,
        x: float,
        y: F64Array,
        dydx: F64Array,
        h: float,
        yout: F64Array,
        yerr: F64Array,
    ) -> None:
        """Take one Cash-Karp Runge-Kutta step.

        Writes the stepped solution into *yout* and the error estimate
        into *yerr*.  Both must be pre-allocated with length
        ``mNVariables``.
        """
        n: int = self.mNVariables

        # Lazy-allocate workspace vectors
        if self.mWs2 is None:
            self.mWs2 = np.zeros(n, dtype=np.float64)
        if self.mWs3 is None:
            self.mWs3 = np.zeros(n, dtype=np.float64)
        if self.mWs4 is None:
            self.mWs4 = np.zeros(n, dtype=np.float64)
        if self.mWs5 is None:
            self.mWs5 = np.zeros(n, dtype=np.float64)
        if self.mWs6 is None:
            self.mWs6 = np.zeros(n, dtype=np.float64)
        if self.mYTemp is None:
            self.mYTemp = np.zeros(n, dtype=np.float64)

        # --- Stage 1 (dydx already provided) ---
        # ytemp = y + b21*h*dydx
        for i in range(n):
            self.mYTemp[i] = y[i] + self._b21 * h * dydx[i]

        # --- Stage 2 ---
        self.derivatives(x + self._a2 * h, self.mYTemp, self.mWs2)
        for i in range(n):
            self.mYTemp[i] = y[i] + h * (self._b31 * dydx[i] + self._b32 * self.mWs2[i])

        # --- Stage 3 ---
        self.derivatives(x + self._a3 * h, self.mYTemp, self.mWs3)
        for i in range(n):
            self.mYTemp[i] = y[i] + h * (
                self._b41 * dydx[i] + self._b42 * self.mWs2[i] + self._b43 * self.mWs3[i]
            )

        # --- Stage 4 ---
        self.derivatives(x + self._a4 * h, self.mYTemp, self.mWs4)
        for i in range(n):
            self.mYTemp[i] = y[i] + h * (
                self._b51 * dydx[i]
                + self._b52 * self.mWs2[i]
                + self._b53 * self.mWs3[i]
                + self._b54 * self.mWs4[i]
            )

        # --- Stage 5 ---
        self.derivatives(x + self._a5 * h, self.mYTemp, self.mWs5)
        for i in range(n):
            self.mYTemp[i] = y[i] + h * (
                self._b61 * dydx[i]
                + self._b62 * self.mWs2[i]
                + self._b63 * self.mWs3[i]
                + self._b64 * self.mWs4[i]
                + self._b65 * self.mWs5[i]
            )

        # --- Stage 6 ---
        self.derivatives(x + self._a6 * h, self.mYTemp, self.mWs6)

        # --- Accumulate 5th-order solution and error ---
        for i in range(n):
            yout[i] = y[i] + h * (
                self._c1 * dydx[i]
                + self._c3 * self.mWs3[i]
                + self._c4 * self.mWs4[i]
                + self._c6 * self.mWs6[i]
            )
            yerr[i] = h * (
                self._dc1 * dydx[i]
                + self._dc3 * self.mWs3[i]
                + self._dc4 * self.mWs4[i]
                + self._dc5 * self.mWs5[i]
                + self._dc6 * self.mWs6[i]
            )

    # ==================================================================
    # _qcStep  (Java's private qcStep)
    # Quality-controlled adaptive step
    # ==================================================================
    def _qcStep(
        self,
        x: float,
        y: F64Array,
        dydx: F64Array,
        htry: float,
        eps: float,
        yscal: F64Array,
    ) -> float:
        """Take one quality-controlled Runge-Kutta step.

        Adjusts step size to keep the local truncation error below
        *eps*.  Mutates *y* in place upon success.  Sets
        ``self.mHDid`` and ``self.mHNext`` as side effects.

        Returns the new value of x after the step.
        """
        n: int = self.mNVariables

        # Lazy-allocate workspace
        if self.mYErr is None:
            self.mYErr = np.zeros(n, dtype=np.float64)
        if self.mQcYTemp is None:
            self.mQcYTemp = np.zeros(n, dtype=np.float64)

        h: float = htry

        # do { ... } while (errmax > 1.0)  →  prime errmax then loop
        errmax: float = 2.0
        while errmax > 1.0:
            self._baseStep(x, y, dydx, h, self.mQcYTemp, self.mYErr)
            errmax = 0.0
            for i in range(n):
                errmax = max(errmax, abs(self.mYErr[i] / yscal[i]))
            errmax /= eps
            if errmax > 1.0:
                htemp: float = self._SAFETY * h * math.pow(errmax, self._PSHRINK)
                h = max(htemp, 0.1 * h) if h >= 0.0 else min(htemp, 0.1 * h)
                xnew: float = x + h
                if xnew == x:
                    raise EPQException("Stepsize underflow in AdaptiveRungeKutta._qcStep")

        if errmax > self._ERRCON:
            self.mHNext = self._SAFETY * h * math.pow(errmax, self._PGROW)
        else:
            self.mHNext = 5.0 * h

        self.mHDid = h
        x += h
        y[:n] = self.mQcYTemp[:n]
        return x

    # ==================================================================
    # Public API — save-interval management
    # ==================================================================
    def setSaveInterval(self, interval: float) -> None:
        """Enable intermediate-point saving at the given interval."""
        self.mSaveInterval = interval

    def clearSaveInterval(self) -> None:
        """Disable intermediate-point saving."""
        self.mSaveInterval = 0.0

    # ==================================================================
    # Public API — accessors for saved trajectory data
    # ==================================================================
    def getNSaved(self) -> int:
        """Return the number of saved intermediate points."""
        return self.mNSaved

    def getX(self, i: int) -> float:
        """Return the x-value of the *i*-th saved point."""
        assert self.mXSave is not None
        return float(self.mXSave[i])

    def getY(self, i: int) -> F64Array:
        """Return the y-vector of the *i*-th saved point (row view)."""
        assert self.mYSave is not None
        return self.mYSave[i].copy()

    # ==================================================================
    # Public API — tuning parameters
    # ==================================================================
    def setMaxSteps(self, maxSteps: int) -> None:
        """Set the maximum number of integration steps allowed."""
        self.mMaxSteps = maxSteps

    def setMinStepSize(self, minStep: float) -> None:
        """Set the minimum permissible step size (0 disables check)."""
        self.mMinStepSize = minStep

    # ==================================================================
    # Public API — query
    # ==================================================================
    def getNVariables(self) -> int:
        """Return the number of dependent variables."""
        return self.mNVariables

    def getStepCount(self) -> int:
        """Return the total number of steps taken during the last integration."""
        return self.mNOk + self.mNBad

    def getGoodStepCount(self) -> int:
        """Return the number of successful (non-subdivided) steps."""
        return self.mNOk

    def getBadStepCount(self) -> int:
        """Return the number of steps that required subdivision."""
        return self.mNBad

    # ==================================================================
    # Abstract method — derivatives
    # ==================================================================
    @abc.abstractmethod
    def derivatives(self, x: float, y: F64Array, dydx: F64Array) -> None:
        """Evaluate the RHS of the ODE system.

        Subclasses must fill *dydx* with dy/dx evaluated at (*x*, *y*).
        """
        ...

    # ==================================================================
    # integrate — main public entry point
    # ==================================================================
    def integrate(
        self,
        x1: float,
        x2: float,
        ystart: F64Array,
        eps: float,
        h1: float,
    ) -> F64Array:
        """Integrate the ODE from *x1* to *x2*.

        Parameters
        ----------
        x1 : float
            Start of the integration interval.
        x2 : float
            End of the integration interval.
        ystart : F64Array
            Initial y-values; **mutated in place** to hold the final
            y-values on return.
        eps : float
            Desired relative accuracy.
        h1 : float
            Initial step size (sign is forced to match x2 - x1).

        Returns
        -------
        F64Array
            The final y-values (same data written into *ystart*).

        Raises
        ------
        EPQException
            If the maximum number of steps is exceeded or the step
            size underflows.
        """
        n: int = self.mNVariables
        tiny: float = 1.0e-10 * eps

        yscal: F64Array = np.zeros(n, dtype=np.float64)
        dydx: F64Array = np.zeros(n, dtype=np.float64)
        y: F64Array = np.zeros(n, dtype=np.float64)

        x: float = x1
        h: float = self._sign(h1, x2 - x1)
        xsav: float = 0.0
        saveInt: float = float(sys.float_info.max)   # Double.MAX_VALUE
        kMax: int = 0

        self.mNSaved = 0
        self.mNOk = 0
        self.mNBad = 0

        # y[:] = ystart[:] — System.arraycopy
        y[:n] = ystart[:n]

        if self.mSaveInterval != 0.0 and self.mSaveInterval != float(sys.float_info.max):
            # R7: Java integer division via Math.round → int(math.floor(x + 0.5))
            kMax = int(math.floor(
                (abs(x2 - x1) + self.mSaveInterval) / self.mSaveInterval + 0.5
            ))
            saveInt = self._sign(self.mSaveInterval, x2 - x1)
            # Ensure the first step is saved:
            xsav = x - 2.0 * saveInt
            self.mXSave = np.zeros(kMax, dtype=np.float64)
            self.mYSave = np.zeros((kMax, n), dtype=np.float64)

        for step in range(self.mMaxSteps):
            # Save intermediate points if requested
            if (kMax != 0) and (self.mNSaved < kMax) and (
                abs(x - xsav) >= 0.9999 * self.mSaveInterval
            ):
                assert self.mXSave is not None
                assert self.mYSave is not None
                self.mXSave[self.mNSaved] = x
                self.mYSave[self.mNSaved, :n] = y[:n]
                xsav = x
                self.mNSaved += 1

            # Compute derivatives
            self.derivatives(x, y, dydx)

            # Build scaling vector for error monitoring
            for i in range(n):
                yscal[i] = abs(y[i]) + abs(dydx[i] * h) + tiny

            # Reduce step if it would overshoot the endpoint
            if (h > 0.0 and (x + h) > x2) or (h < 0.0 and (x + h) < x2):
                h = x2 - x

            # Take a quality-controlled step
            x = self._qcStep(x, y, dydx, h, eps, yscal)

            if (x - x2) * (x2 - x1) >= 0.0:
                # Reached or passed the endpoint — done
                ystart[:n] = y[:n]
                # Save the final point if saving is enabled
                if kMax != 0 and self.mNSaved < kMax:
                    assert self.mXSave is not None
                    assert self.mYSave is not None
                    self.mXSave[self.mNSaved] = x
                    self.mYSave[self.mNSaved, :n] = y[:n]
                    self.mNSaved += 1
                self._clearWorkspace()
                return y.copy()

            if abs(self.mHNext) <= self.mMinStepSize:
                raise EPQException(
                    "Step size too small in AdaptiveRungeKutta.integrate"
                )

            h = self.mHNext

            if self.mHDid == h:
                self.mNOk += 1
            else:
                self.mNBad += 1

        self._clearWorkspace()
        raise EPQException("Too many steps in AdaptiveRungeKutta.integrate")

    # ==================================================================
    # integrate_literal — identical to integrate (no scipy substitution
    # exists for an adaptive ODE stepper driven by a user-defined
    # abstract method)
    # ==================================================================
    def integrate_literal(
        self,
        x1: float,
        x2: float,
        ystart: F64Array,
        eps: float,
        h1: float,
    ) -> F64Array:
        """Literal line-for-line port — identical to :meth:`integrate`
        because no scipy substitution applies."""
        return self.integrate(x1, x2, ystart, eps, h1)
