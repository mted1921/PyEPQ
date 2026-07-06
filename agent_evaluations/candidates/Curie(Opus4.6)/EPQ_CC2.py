r"""
AdaptiveRungeKutta_ver1_1_0.py — Python port of gov.nist.microanalysis.Utility.AdaptiveRungeKutta

Guide version : 1
Generation    : 1
Port-code fixes: 0

------------------------------------------------------------------------
Original Java source (gov.nist.microanalysis.Utility.AdaptiveRungeKutta)
------------------------------------------------------------------------
/**
 * AdaptiveRungeKutta - An attempt at a robust adaptive step-size Runge-Kutta
 * ODE integrator based on the Cash-Karp method as described in Numerical
 * Recipes in C, section 16.2. The user provides a function that computes the
 * derivatives and this class integrates it from x1 to x2. The class is not
 * thread safe.
 *
 * @author Nicholas
 * @version 1.0
 */
------------------------------------------------------------------------

CHANGES
-------
None — line-for-line faithful port.
"""

from __future__ import annotations

import abc
import math
import sys
from typing import Optional

import numpy as np

try:
    from ._epq_compat import EPQException, F64Array
except ImportError:
    try:
        from _epq_compat import EPQException, F64Array  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility._epq_compat import EPQException, F64Array  # type: ignore

# ---------------------------------------------------------------------------
# UtilException — Java throws gov.nist.microanalysis.Utility.UtilException.
# We reuse EPQException as UtilException for now, since UtilException extends
# EPQException in the Java codebase.
# ---------------------------------------------------------------------------
UtilException = EPQException

# ============================================================================
# BUG_LEDGER
# ============================================================================
BUG_LEDGER: tuple = ()  # no bugs identified

class AdaptiveRungeKutta(abc.ABC):
    """Python port of gov.nist.microanalysis.Utility.AdaptiveRungeKutta.

    An adaptive step-size Runge-Kutta ODE integrator based on the Cash-Karp
    method as described in Numerical Recipes in C, section 16.2.
    """

    # Cash-Karp tableau constants (class-level, matching Java private static final)
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

    _dc1: float = 37.0 / 378.0 - 2825.0 / 27648.0
    _dc3: float = 250.0 / 621.0 - 18575.0 / 48384.0
    _dc4: float = 125.0 / 594.0 - 13525.0 / 55296.0
    _dc5: float = -277.0 / 14336.0
    _dc6: float = 512.0 / 1771.0 - 0.25

    _SAFETY: float = 0.9
    _PGROW: float = -0.2
    _PSHRINK: float = -0.25
    _ERRCON: float = 1.89e-4
    _TINY: float = 1.0e-30

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------
    def __init__(self, nVars: int) -> None:
        """Construct an AdaptiveRungeKutta integrator for *nVars* coupled ODEs.

        Parameters
        ----------
        nVars : int
            Number of dependent variables (dimension of y).
        """
        self.mNVariables: int = nVars

        # Step-size bookkeeping
        self.mHDid: float = 0.0
        self.mHNext: float = 0.0

        # Save-interval state
        self.mSaveInterval: float = 0.0
        self.mMinStepSize: float = 0.0

        # Saved-trajectory arrays (allocated in integrate when kMax > 0)
        self.mXSave: Optional[F64Array] = None
        self.mYSave: Optional[F64Array] = None  # 2-D: shape (kMax, mNVariables)
        self.mNSaved: int = 0

        # Integration bookkeeping
        self.mMaxSteps: int = 10000
        self.mNOk: int = 0
        self.mNBad: int = 0

        # Workspace arrays for _baseStep (allocated lazily in _baseStep)
        self.mWs2: Optional[F64Array] = None
        self.mWs3: Optional[F64Array] = None
        self.mWs4: Optional[F64Array] = None
        self.mWs5: Optional[F64Array] = None
        self.mWs6: Optional[F64Array] = None
        self.mYTemp: Optional[F64Array] = None

        # Workspace arrays for _qcStep (allocated lazily in _qcStep)
        self.mYErr: Optional[F64Array] = None
        self.mQcYTemp: Optional[F64Array] = None

    # ------------------------------------------------------------------
    # Abstract extension point — public abstract in Java → NO underscore
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def derivatives(self, x: float, y: F64Array, dydx: F64Array) -> None:
        """Compute derivatives dydx[] given x and y[].

        The derived class provides an implementation of the derivatives
        function. x & y[] are input and the user-provided implementation
        of derivatives is responsible for returning the derivatives in the
        array dydx.  The lengths of y and dydx are equal to mNVariables.

        Parameters
        ----------
        x : float
            Independent variable.
        y : F64Array
            Dependent variables (length mNVariables), read-only input.
        dydx : F64Array
            Output array for derivatives (length mNVariables), written in place.
        """
        ...

    # ------------------------------------------------------------------
    # _sign  (Java: private double sign(double, double))
    # ------------------------------------------------------------------
    @staticmethod
    def _sign(magnitude: float, sign: float) -> float:
        """Numerical Recipes SIGN function.

        Returns ``+|magnitude|`` when *sign* >= 0.0,  ``-|magnitude|`` otherwise.
        Java ``-0.0 >= 0.0`` is True (IEEE 754), so ``sign(m, -0.0)`` → ``+|m|``.
        Python ``math.copysign(abs(m), -0.0)`` returns ``-|m|``, diverging.
        The conditional form replicates Java exactly.
        """
        return abs(magnitude) if sign >= 0.0 else -abs(magnitude)

    # ------------------------------------------------------------------
    # _baseStep  (Java: private void baseStep(...))
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
        """Fifth-order Cash-Karp Runge-Kutta step (literal NR port).

        Parameters
        ----------
        x : float
            Current independent-variable value.
        y : F64Array
            Current dependent-variable values (length mNVariables).
        dydx : F64Array
            Current derivatives (length mNVariables).
        h : float
            Step size to attempt.
        yout : F64Array
            Output: y values at x + h (length mNVariables).
        yerr : F64Array
            Output: estimated truncation error (length mNVariables).
        """
        n: int = self.mNVariables

        # Lazy workspace allocation
        if self.mWs2 is None or self.mWs2.shape[0] != n:
            self.mWs2 = np.zeros(n, dtype=np.float64)
            self.mWs3 = np.zeros(n, dtype=np.float64)
            self.mWs4 = np.zeros(n, dtype=np.float64)
            self.mWs5 = np.zeros(n, dtype=np.float64)
            self.mWs6 = np.zeros(n, dtype=np.float64)
            self.mYTemp = np.zeros(n, dtype=np.float64)

        # These are guaranteed non-None after the block above
        assert self.mWs2 is not None
        assert self.mWs3 is not None
        assert self.mWs4 is not None
        assert self.mWs5 is not None
        assert self.mWs6 is not None
        assert self.mYTemp is not None

        # First step
        for i in range(n):
            self.mYTemp[i] = y[i] + _b21 * h * dydx[i]
        self.derivatives(x + _a2 * h, self.mYTemp, self.mWs2)

        # Second step
        for i in range(n):
            self.mYTemp[i] = y[i] + h * (_b31 * dydx[i] + _b32 * self.mWs2[i])
        self.derivatives(x + _a3 * h, self.mYTemp, self.mWs3)

        # Third step
        for i in range(n):
            self.mYTemp[i] = y[i] + h * (_b41 * dydx[i] + _b42 * self.mWs2[i] + _b43 * self.mWs3[i])
        self.derivatives(x + _a4 * h, self.mYTemp, self.mWs4)

        # Fourth step
        for i in range(n):
            self.mYTemp[i] = y[i] + h * (
                _b51 * dydx[i] + _b52 * self.mWs2[i] + _b53 * self.mWs3[i] + _b54 * self.mWs4[i]
            )
        self.derivatives(x + _a5 * h, self.mYTemp, self.mWs5)

        # Fifth step
        for i in range(n):
            self.mYTemp[i] = y[i] + h * (
                _b61 * dydx[i]
                + _b62 * self.mWs2[i]
                + _b63 * self.mWs3[i]
                + _b64 * self.mWs4[i]
                + _b65 * self.mWs5[i]
            )
        self.derivatives(x + _a6 * h, self.mYTemp, self.mWs6)

        # Accumulate increments with proper weights
        for i in range(n):
            yout[i] = y[i] + h * (
                _c1 * dydx[i] + _c3 * self.mWs3[i] + _c4 * self.mWs4[i] + _c6 * self.mWs6[i]
            )

        # Estimate error as difference between fourth and fifth order
        for i in range(n):
            yerr[i] = h * (
                _dc1 * dydx[i]
                + _dc3 * self.mWs3[i]
                + _dc4 * self.mWs4[i]
                + _dc5 * self.mWs5[i]
                + _dc6 * self.mWs6[i]
            )

    # ------------------------------------------------------------------
    # _qcStep  (Java: private double qcStep(...))
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
        """Quality-controlled Runge-Kutta step.

        Calls _baseStep repeatedly, shrinking h until error is within *eps*.
        On success, updates *y* in place to the new values and sets
        ``self.mHDid`` and ``self.mHNext``.

        Parameters
        ----------
        x : float
            Current independent variable.
        y : F64Array
            Current dependent variables (mutated in place on success).
        dydx : F64Array
            Current derivatives.
        htry : float
            Initial step size to try.
        eps : float
            Desired fractional accuracy.
        yscal : F64Array
            Scaling array for error monitoring.

        Returns
        -------
        float
            New value of x after the step (x + mHDid).
        """
        n: int = self.mNVariables

        # Lazy workspace allocation
        if self.mYErr is None or self.mYErr.shape[0] != n:
            self.mYErr = np.zeros(n, dtype=np.float64)
            self.mQcYTemp = np.zeros(n, dtype=np.float64)

        assert self.mYErr is not None
        assert self.mQcYTemp is not None

        h: float = htry
        # do { ... } while (errmax > 1.0)  →  errmax = 2.0; while errmax > 1.0
        errmax: float = 2.0
        while errmax > 1.0:
            self._baseStep(x, y, dydx, h, self.mQcYTemp, self.mYErr)
            # Evaluate accuracy
            errmax = 0.0
            for i in range(n):
                errmax = max(errmax, abs(self.mYErr[i] / yscal[i]))
            errmax /= eps
            if errmax > 1.0:
                # Truncation error too large, reduce step size
                htemp: float = _SAFETY * h * math.pow(errmax, _PSHRINK)
                h = self._sign(max(abs(htemp), 0.1 * abs(h)), h)
                xnew: float = x + h
                if xnew == x:
                    raise UtilException("Stepsize underflow in AdaptiveRungeKutta.qcStep")

        # Step succeeded
        if errmax > _ERRCON:
            self.mHNext = _SAFETY * h * math.pow(errmax, _PGROW)
        else:
            self.mHNext = 5.0 * h
        
        self.mHDid = h
        # Copy successful step result into y (in-place mutation)
        for i in range(n):
            y[i] = self.mQcYTemp[i]
        return x + h

    # ------------------------------------------------------------------
    # _clearWorkspace  (Java: private void clearWorkspace())
    # ------------------------------------------------------------------
    def _clearWorkspace(self) -> None:
        """Release workspace arrays so they will be reallocated on next use."""
        self.mWs2 = None
        self.mWs3 = None
        self.mWs4 = None
        self.mWs5 = None
        self.mWs6 = None
        self.mYTemp = None
        self.mYErr = None
        self.mQcYTemp = None

    # ------------------------------------------------------------------
    # Public API — property-style getters / setters
    # ------------------------------------------------------------------
    def setSaveInterval(self, saveInterval: float) -> None:
        """Set the interval at which (x, y) snapshots are stored during
        integration.  A value of ``float('inf')`` or ``sys.float_info.max``
        disables saving."""
        self.mSaveInterval = saveInterval

    def getSaveInterval(self) -> float:
        """Return the current save interval."""
        return self.mSaveInterval

    def getSavedCount(self) -> int:
        """Return the number of (x, y) snapshots saved during the last
        integration."""
        return self.mNSaved

    def getX(self, i: int) -> float:
        """Return the i-th saved x value."""
        assert self.mXSave is not None
        return float(self.mXSave[i])

    def getY(self, i: int) -> F64Array:
        """Return the i-th saved y vector (row view of mYSave)."""
        assert self.mYSave is not None
        return self.mYSave[i]

    def setMaxSteps(self, maxSteps: int) -> None:
        """Set the maximum number of integration steps before raising."""
        self.mMaxSteps = maxSteps

    def setMinStepSize(self, minStep: float) -> None:
        """Set a floor for the step size."""
        self.mMinStepSize = minStep

    def getNVariables(self) -> int:
        """Return the number of dependent variables."""
        return self.mNVariables

    def getStepCount(self) -> int:
        """Return total steps (good + bad) from the last integration."""
        return self.mNOk + self.mNBad

    def getGoodStepCount(self) -> int:
        """Return the number of steps that achieved desired accuracy without
        subdivision."""
        return self.mNOk

    def getBadStepCount(self) -> int:
        """Return the number of steps that required subdivision."""
        return self.mNBad

    # ------------------------------------------------------------------
    # integrate  (Java: public double[] integrate(...))
    # ------------------------------------------------------------------
    def integrate(
        self,
        x1: float,
        x2: float,
        ystart: F64Array,
        eps: float,
        h1: float,
    ) -> F64Array:
        """Integrate the ODE from *x1* to *x2*.

        Uses the adaptive step-size Cash-Karp Runge-Kutta method.

        Parameters
        ----------
        x1 : float
            Start of integration range.
        x2 : float
            End of integration range.
        ystart : F64Array
            Initial y values (length mNVariables).  **Mutated in place** at
            completion to hold the final y values.
        eps : float
            Desired fractional accuracy.
        h1 : float
            Initial step size (sign is adjusted automatically).

        Returns
        -------
        F64Array
            Final y values (same content written into *ystart*).

        Raises
        ------
        UtilException
            If the integrator exceeds ``mMaxSteps`` or the step size
            underflows below ``mMinStepSize``.
        """
        n: int = self.mNVariables
        tiny: float = 1.0e-10 * eps

        yscal: F64Array = np.zeros(n, dtype=np.float64)
        dydx: F64Array = np.zeros(n, dtype=np.float64)
        y: F64Array = np.zeros(n, dtype=np.float64)

        x: float = x1
        h: float = self._sign(h1, x2 - x1)
        xsav: float = 0.0
        saveInt: float = float(sys.float_info.max)
        kMax: int = 0
        self.mNSaved = 0
        self.mNOk = 0
        self.mNBad = 0

        # Copy ystart → local y
        for i in range(n):
            y[i] = ystart[i]

        # Set up saving if requested
        if self.mSaveInterval != float(sys.float_info.max) and self.mSaveInterval != 0.0:
            kMax = int(math.floor(
                (abs(x2 - x1) + self.mSaveInterval) / self.mSaveInterval + 0.5
            ))
            saveInt = self._sign(self.mSaveInterval, x2 - x1)
            xsav = x - 2.0 * saveInt  # ensure the first step is saved
            self.mXSave = np.zeros(kMax, dtype=np.float64)
            self.mYSave = np.zeros((kMax, n), dtype=np.float64)

        for step in range(self.mMaxSteps):
            # Save the necessary points
            if (kMax != 0) and (self.mNSaved < kMax) and (abs(x - xsav) >= 0.9999 * self.mSaveInterval):
                self.mXSave[self.mNSaved] = x
                for i in range(n):
                    self.mYSave[self.mNSaved, i] = y[i]
                xsav = x
                self.mNSaved += 1

            self.derivatives(x, y, dydx)

            # Rescale h to ensure we don't overshoot x2
            for i in range(n):
                yscal[i] = abs(y[i]) + abs(dydx[i] * h) + tiny

            # If we would overshoot, cut h
            if (x + h - x2) * (x + h - x1) > 0.0:
                h = x2 - x

            x = self._qcStep(x, y, dydx, h, eps, yscal)

            if self.mHDid == h:
                self.mNOk += 1
            else:
                self.mNBad += 1

            # Are we done?
            if (x - x2) * (x2 - x1) >= 0.0:
                # Store final values into ystart (in-place mutation)
                for i in range(n):
                    ystart[i] = y[i]
                # Save the final snapshot if saving is active
                if kMax != 0 and self.mNSaved < kMax:
                    self.mXSave[self.mNSaved] = x
                    for i in range(n):
                        self.mYSave[self.mNSaved, i] = y[i]
                    self.mNSaved += 1
                return y

            if abs(self.mHNext) <= self.mMinStepSize:
                raise UtilException("Step size too small in AdaptiveRungeKutta.integrate")

            h = self.mHNext

        raise UtilException("Too many steps in AdaptiveRungeKutta.integrate")

    # ------------------------------------------------------------------
    # integrate_literal — identical to integrate (no scipy substitution
    # possible for an adaptive ODE stepper that relies on user-supplied
    # derivatives via an abstract method).
    # ------------------------------------------------------------------
    def integrate_literal(
        self,
        x1: float,
        x2: float,
        ystart: F64Array,
        eps: float,
        h1: float,
    ) -> F64Array:
        """Literal port of ``integrate`` — identical because no scipy
        substitution is applicable for a user-extensible abstract ODE
        integrator."""
        return self.integrate(x1, x2, ystart, eps, h1)

    # ------------------------------------------------------------------
    # __str__ / toString
    # ------------------------------------------------------------------
    def __str__(self) -> str:
        return (
            f"AdaptiveRungeKutta(nVars={self.mNVariables}, "
            f"ok={self.mNOk}, bad={self.mNBad}, saved={self.mNSaved})"
        )

    def toString(self) -> str:
        """Java-style alias for ``__str__``."""
        return self.__str__()

# =========================================================================
# Module-level constant aliases for the Butcher-tableau values so that
# _baseStep and _qcStep can reference them concisely.  These mirror the
# ``final double`` locals inside the Java methods (which are true constants).
# =========================================================================
_a2: float = AdaptiveRungeKutta._a2
_a3: float = AdaptiveRungeKutta._a3
_a4: float = AdaptiveRungeKutta._a4
_a5: float = AdaptiveRungeKutta._a5
_a6: float = AdaptiveRungeKutta._a6

_b21: float = AdaptiveRungeKutta._b21
_b31: float = AdaptiveRungeKutta._b31
_b32: float = AdaptiveRungeKutta._b32
_b41: float = AdaptiveRungeKutta._b41
_b42: float = AdaptiveRungeKutta._b42
_b43: float = AdaptiveRungeKutta._b43
_b51: float = AdaptiveRungeKutta._b51
_b52: float = AdaptiveRungeKutta._b52
_b53: float = AdaptiveRungeKutta._b53
_b54: float = AdaptiveRungeKutta._b54
_b61: float = AdaptiveRungeKutta._b61
_b62: float = AdaptiveRungeKutta._b62
_b63: float = AdaptiveRungeKutta._b63
_b64: float = AdaptiveRungeKutta._b64
_b65: float = AdaptiveRungeKutta._b65

_c1: float = AdaptiveRungeKutta._c1
_c3: float = AdaptiveRungeKutta._c3
_c4: float = AdaptiveRungeKutta._c4
_c6: float = AdaptiveRungeKutta._c6

_dc1: float = AdaptiveRungeKutta._dc1
_dc3: float = AdaptiveRungeKutta._dc3
_dc4: float = AdaptiveRungeKutta._dc4
_dc5: float = AdaptiveRungeKutta._dc5
_dc6: float = AdaptiveRungeKutta._dc6

_SAFETY: float = AdaptiveRungeKutta._SAFETY
_PGROW: float = AdaptiveRungeKutta._PGROW
_PSHRINK: float = AdaptiveRungeKutta._PSHRINK
_ERRCON: float = AdaptiveRungeKutta._ERRCON

# =========================================================================
# __all__
# =========================================================================
__all__: list[str] = ["AdaptiveRungeKutta"]
