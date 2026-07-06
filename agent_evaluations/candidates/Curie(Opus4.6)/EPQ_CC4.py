r"""
AdaptiveRungeKutta_ver1_1_0.py — Python port of gov.nist.microanalysis.Utility.AdaptiveRungeKutta

Guide version : 1
Generation    : 1
Port-code fixes: 0

------------------------------------------------------------------------
Original Java source (gov.nist.microanalysis.Utility.AdaptiveRungeKutta)
------------------------------------------------------------------------
/**
 * AdaptiveRungeKutta - A attempt to replicate the CashKarp Runge-Kutta
 * adaptive step size ODE integrator from Numerical Recipes in C.
 *
 * Not thread safe!
 *
 * @author nicholas
 */
------------------------------------------------------------------------

CHANGES
-------
None — literal line-for-line port.

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
# UtilException — placeholder until the dedicated port lands.
# Java: gov.nist.microanalysis.Utility.UtilException extends EPQException
# ---------------------------------------------------------------------------
try:
    from .UtilException_ver1_1_0 import UtilException
except ImportError:
    try:
        from UtilException_ver1_1_0 import UtilException  # type: ignore
    except ImportError:
        # Minimal stand-in so the module is self-contained
        class UtilException(EPQException):  # type: ignore[no-redef]
            pass

# ======================================================================
# BUG_LEDGER
# ======================================================================
BUG_LEDGER: tuple = ()  # no bugs identified

# ======================================================================
# _sign helper  (Java: private double sign(double magnitude, double sign))
# ======================================================================
# Java sign(magnitude, sign) returns +|magnitude| when sign >= 0.0 and
# -|magnitude| otherwise.  Critically, -0.0 >= 0.0 is True in Java
# (IEEE 754), so sign(m, -0.0) returns +|m|.
# Python math.copysign(abs(m), -0.0) returns -|m|, diverging.
# The correct Python translation is the conditional form
# (per CONVERSION_GUIDE _sign helper semantics).
def _sign(magnitude: float, sign: float) -> float:
    """Fortran-style SIGN: returns ``abs(magnitude)`` with the sign of *sign*.

    Java ``-0.0 >= 0.0`` is ``True``, so ``sign(m, -0.0)`` → ``+|m|``.
    ``math.copysign`` would give ``-|m|`` for ``-0.0``, so we use the
    conditional form.
    """
    return abs(magnitude) if sign >= 0.0 else -abs(magnitude)

# ======================================================================
# Cash-Karp Butcher tableau constants (class-level, shared by all instances)
# ======================================================================
_A2: float = 0.2
_A3: float = 0.3
_A4: float = 0.6
_A5: float = 1.0
_A6: float = 0.875

_B21: float = 0.2
_B31: float = 3.0 / 40.0
_B32: float = 9.0 / 40.0
_B41: float = 0.3
_B42: float = -0.9
_B43: float = 1.2
_B51: float = -11.0 / 54.0
_B52: float = 2.5
_B53: float = -70.0 / 27.0
_B54: float = 35.0 / 27.0
_B61: float = 1631.0 / 55296.0
_B62: float = 175.0 / 512.0
_B63: float = 575.0 / 13824.0
_B64: float = 44275.0 / 110592.0
_B65: float = 253.0 / 4096.0

_C1: float = 37.0 / 378.0
_C3: float = 250.0 / 621.0
_C4: float = 125.0 / 594.0
_C6: float = 512.0 / 1771.0

_DC1: float = _C1 - (2825.0 / 27648.0)
_DC3: float = _C3 - (18575.0 / 48384.0)
_DC4: float = _C4 - (13525.0 / 55296.0)
_DC5: float = -277.00 / 14336.0
_DC6: float = _C6 - 0.25

_PGROW: float = -0.2
_PSHRINK: float = -0.25
_SAFETY: float = 0.9
_ERRCON: float = 1.89e-4

_MAXSTP: int = 10000
_TINY: float = 1.0e-30

class AdaptiveRungeKutta(abc.ABC):
    """Cash-Karp adaptive-step-size Runge-Kutta ODE integrator.

    Literal port of the Java class
    ``gov.nist.microanalysis.Utility.AdaptiveRungeKutta``.
    Not thread-safe (same as the Java original).
    """

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------
    def __init__(self, nVars: int) -> None:
        self.mNVariables: int = nVars
        self.mHDid: float = 0.0
        self.mHNext: float = 0.0
        self.mSaveInterval: float = 0.0
        self.mMinStepSize: float = 0.0
        self.mXSave: Optional[F64Array] = None
        self.mYSave: Optional[F64Array] = None  # 2-D: shape (kMax, mNVariables)
        self.mNSaved: int = 0
        self.mMaxSteps: int = _MAXSTP
        self.mNOk: int = 0
        self.mNBad: int = 0
        # Workspace arrays for _baseStep
        self.mWs2: Optional[F64Array] = None
        self.mWs3: Optional[F64Array] = None
        self.mWs4: Optional[F64Array] = None
        self.mWs5: Optional[F64Array] = None
        self.mWs6: Optional[F64Array] = None
        self.mYTemp: Optional[F64Array] = None
        # Workspace arrays for _qcStep
        self.mYErr: Optional[F64Array] = None
        self.mQcYTemp: Optional[F64Array] = None

    # ------------------------------------------------------------------
    # Abstract extension point — public abstract, NO underscore (R1)
    # ------------------------------------------------------------------
    @abc.abstractmethod
    def derivatives(self, x: float, y: F64Array, dydx: F64Array) -> None:
        """Compute derivatives.

        *x* and *y* are input; the implementation fills *dydx*.
        Lengths of *y* and *dydx* equal ``mNVariables``.
        """
        ...

    # ------------------------------------------------------------------
    # Save-interval management
    # ------------------------------------------------------------------
    def setSaveInterval(self, interval: float) -> None:
        self.mSaveInterval = interval

    def clearSaveInterval(self) -> None:
        self.mSaveInterval = 0.0

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def getNSaved(self) -> int:
        return self.mNSaved

    def getX(self, i: int) -> float:
        assert self.mXSave is not None
        return float(self.mXSave[i])

    def getY(self, i: int) -> F64Array:
        assert self.mYSave is not None
        return self.mYSave[i]

    def setMaxSteps(self, maxSteps: int) -> None:
        self.mMaxSteps = maxSteps

    def setMinStepSize(self, minStep: float) -> None:
        self.mMinStepSize = minStep

    def getNVariables(self) -> int:
        return self.mNVariables

    def getStepCount(self) -> int:
        return self.mNOk + self.mNBad

    def getGoodStepCount(self) -> int:
        return self.mNOk

    def getBadStepCount(self) -> int:
        return self.mNBad

    # ------------------------------------------------------------------
    # Private: _clearWorkspace
    # ------------------------------------------------------------------
    def _clearWorkspace(self) -> None:
        self.mWs2 = None
        self.mWs3 = None
        self.mWs4 = None
        self.mWs5 = None
        self.mWs6 = None
        self.mYTemp = None
        self.mYErr = None
        self.mQcYTemp = None

    # ------------------------------------------------------------------
    # Private: _baseStep  (Cash-Karp 5th-order step)
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
        """One Cash-Karp Runge-Kutta step.

        Writes the 5th-order result into *yout* and the embedded error
        estimate into *yerr*.
        """
        n: int = self.mNVariables

        # Lazy-allocate workspace arrays
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

        ak2: F64Array = self.mWs2
        ak3: F64Array = self.mWs3
        ak4: F64Array = self.mWs4
        ak5: F64Array = self.mWs5
        ak6: F64Array = self.mWs6
        ytemp: F64Array = self.mYTemp

        # Stage 2
        for i in range(n):
            ytemp[i] = y[i] + _B21 * h * dydx[i]
        self.derivatives(x + _A2 * h, ytemp, ak2)

        # Stage 3
        for i in range(n):
            ytemp[i] = y[i] + h * (_B31 * dydx[i] + _B32 * ak2[i])
        self.derivatives(x + _A3 * h, ytemp, ak3)

        # Stage 4
        for i in range(n):
            ytemp[i] = y[i] + h * (_B41 * dydx[i] + _B42 * ak2[i] + _B43 * ak3[i])
        self.derivatives(x + _A4 * h, ytemp, ak4)

        # Stage 5
        for i in range(n):
            ytemp[i] = y[i] + h * (
                _B51 * dydx[i] + _B52 * ak2[i] + _B53 * ak3[i] + _B54 * ak4[i]
            )
        self.derivatives(x + _A5 * h, ytemp, ak5)

        # Stage 6
        for i in range(n):
            ytemp[i] = y[i] + h * (
                _B61 * dydx[i]
                + _B62 * ak2[i]
                + _B63 * ak3[i]
                + _B64 * ak4[i]
                + _B65 * ak5[i]
            )
        self.derivatives(x + _A6 * h, ytemp, ak6)

        # Accumulate 5th-order result and error estimate
        for i in range(n):
            yout[i] = y[i] + h * (
                _C1 * dydx[i] + _C3 * ak3[i] + _C4 * ak4[i] + _C6 * ak6[i]
            )
        for i in range(n):
            yerr[i] = h * (
                _DC1 * dydx[i]
                + _DC3 * ak3[i]
                + _DC4 * ak4[i]
                + _DC5 * ak5[i]
                + _DC6 * ak6[i]
            )

    # ------------------------------------------------------------------
    # Private: _qcStep  (quality-controlled step with adaptive sizing)
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
        """Attempt a step of size *htry*; shrink until error within *eps*.

        Mutates *y* in place on success.  Sets ``mHDid`` and ``mHNext``
        as side-effects.  Returns the new *x*.
        """
        n: int = self.mNVariables

        if self.mYErr is None:
            self.mYErr = np.zeros(n, dtype=np.float64)
        if self.mQcYTemp is None:
            self.mQcYTemp = np.zeros(n, dtype=np.float64)

        yerr: F64Array = self.mYErr
        ytemp: F64Array = self.mQcYTemp

        h: float = htry

        # do { ... } while (errmax > 1.0)  →  prime errmax then loop
        errmax: float = 2.0  # ensures at least one iteration
        while errmax > 1.0:
            self._baseStep(x, y, dydx, h, ytemp, yerr)
            errmax = 0.0
            for i in range(n):
                errmax = max(errmax, abs(yerr[i] / yscal[i]))
            errmax /= eps
            if errmax > 1.0:
                htemp: float = _SAFETY * h * math.pow(errmax, _PSHRINK)
                h = max(htemp, 0.1 * h) if h >= 0.0 else min(htemp, 0.1 * h)
                xnew: float = x + h
                if xnew == x:
                    raise UtilException(
                        "Stepsize underflow in AdaptiveRungeKutta._qcStep"
                    )

        if errmax > _ERRCON:
            self.mHNext = _SAFETY * h * math.pow(errmax, _PGROW)
        else:
            self.mHNext = 5.0 * h

        self.mHDid = h
        x += h

        # Copy successful result into y (mutates caller's array in place)
        for i in range(n):
            y[i] = ytemp[i]

        return x

    # ------------------------------------------------------------------
    # Public: integrate
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

        *ystart* is mutated in place to hold the final values **and**
        a reference to the final *y* array is returned.

        Parameters
        ----------
        x1 : float
            Start of integration range.
        x2 : float
            End of integration range.
        ystart : F64Array
            Initial (and final) dependent-variable values.
        eps : float
            Desired relative accuracy.
        h1 : float
            Initial step-size guess.

        Returns
        -------
        F64Array
            The final *y* values (length ``mNVariables``).

        Raises
        ------
        UtilException
            When the maximum step count is exceeded or the step size
            underflows.
        """
        tiny: float = 1.0e-10 * eps
        n: int = self.mNVariables

        yscal: F64Array = np.zeros(n, dtype=np.float64)
        dydx: F64Array = np.zeros(n, dtype=np.float64)
        y: F64Array = np.zeros(n, dtype=np.float64)

        x: float = x1
        h: float = _sign(h1, x2 - x1)
        xsav: float = 0.0
        saveInt: float = float("inf")  # Java Double.MAX_VALUE
        kMax: int = 0

        self.mNSaved = 0
        self.mNOk = 0
        self.mNBad = 0

        # System.arraycopy(ystart, 0, y, 0, mNVariables)
        for i in range(n):
            y[i] = ystart[i]

        # Set up save storage if a save interval has been configured
        # Java: if (mSaveInterval != Double.MAX_VALUE)
        # We use 0.0 as the "not set" sentinel (see clearSaveInterval).
        if self.mSaveInterval != 0.0:
            # R7: Java int cast via Math.round → int(math.floor(x + 0.5))
            kMax = int(
                math.floor(
                    (abs(x2 - x1) + self.mSaveInterval) / self.mSaveInterval + 0.5
                )
            )
            saveInt = _sign(self.mSaveInterval, x2 - x1)
            xsav = x - 2.0 * saveInt  # ensure first step is saved
            self.mXSave = np.zeros(kMax, dtype=np.float64)
            self.mYSave = np.zeros((kMax, n), dtype=np.float64)

        for step in range(self.mMaxSteps):
            # --- save points if configured ---
            if (
                kMax != 0
                and self.mNSaved < kMax
                and abs(x - xsav) >= 0.9999 * self.mSaveInterval
            ):
                assert self.mXSave is not None and self.mYSave is not None
                self.mXSave[self.mNSaved] = x
                for i in range(n):
                    self.mYSave[self.mNSaved, i] = y[i]
                xsav = x
                self.mNSaved += 1

            # --- compute derivatives and scaling vector ---
            self.derivatives(x, y, dydx)
            for i in range(n):
                yscal[i] = abs(y[i]) + abs(dydx[i] * h) + tiny

            # --- check for overshooting the endpoint ---
            if (h > 0.0 and (x + h) > x2) or (h < 0.0 and (x + h) < x2):
                h = x2 - x

            # --- quality-controlled step ---
            x = self._qcStep(x, y, dydx, h, eps, yscal)

            if self.mHDid == h:
                self.mNOk += 1
            else:
                self.mNBad += 1

            # --- check for completion ---
            if (x - x2) * (x2 - x1) >= 0.0:
                # Copy final result back into ystart
                for i in range(n):
                    ystart[i] = y[i]
                # Final save if enabled
                if kMax != 0:
                    assert self.mXSave is not None and self.mYSave is not None
                    if self.mNSaved < kMax:
                        self.mXSave[self.mNSaved] = x
                        for i in range(n):
                            self.mYSave[self.mNSaved, i] = y[i]
                        self.mNSaved += 1
                return y

            # --- enforce minimum step size ---
            if abs(self.mHNext) <= self.mMinStepSize:
                raise UtilException(
                    "Step size too small in AdaptiveRungeKutta.integrate"
                )
            h = self.mHNext

        raise UtilException("Too many steps in AdaptiveRungeKutta.integrate")

    # ------------------------------------------------------------------
    # integrate_literal — identical to integrate (already a literal port)
    # ------------------------------------------------------------------
    def integrate_literal(
        self,
        x1: float,
        x2: float,
        ystart: F64Array,
        eps: float,
        h1: float,
    ) -> F64Array:
        """Literal-port alias. ``integrate`` is already a line-for-line
        translation; this alias exists only to satisfy R2 dual-method
        convention for public mathematical methods.
        """
        return self.integrate(x1, x2, ystart, eps, h1)

    # ------------------------------------------------------------------
    # __str__ / toString
    # ------------------------------------------------------------------
    def __str__(self) -> str:
        return f"AdaptiveRungeKutta(nVars={self.mNVariables})"

    def toString(self) -> str:
        """Java-style ``toString()`` alias."""
        return self.__str__()

    # ------------------------------------------------------------------
    # __eq__ / equals  and  __hash__ / hashCode
    # ------------------------------------------------------------------
    def __eq__(self, other: object) -> bool:
        return self is other

    def equals(self, other: object) -> bool:
        """Java-style ``equals()`` alias."""
        return self.__eq__(other)

    def __hash__(self) -> int:
        return id(self)

    def hashCode(self) -> int:
        """Java-style ``hashCode()`` alias."""
        return self.__hash__()

# ======================================================================
# Module-level __all__
# ======================================================================
__all__: list[str] = [
    "AdaptiveRungeKutta",
    "UtilException",
    "BUG_LEDGER",
]

