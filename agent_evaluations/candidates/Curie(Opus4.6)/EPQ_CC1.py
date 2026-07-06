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
 * Attempt to integrate a set of ordinary differential equations using a 5th
 * order Cash-Karp embedded Runge-Kutta method with adaptive step-size control.
 * Based on the Numerical Recipes algorithms rkck and rkqs.
 * </p>
 * <p>
 * This class is not thread-safe.
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
(none — faithful literal port)
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

# UtilException is used in Java; we map it to EPQException since both are
# simple Exception subclasses in the EPQ library.  The spec confirms the class
# only throws UtilException, which we represent as EPQException.
UtilException = EPQException

__all__ = ["AdaptiveRungeKutta", "UtilException"]

class AdaptiveRungeKutta(abc.ABC):
    """5th-order Cash-Karp embedded Runge-Kutta ODE integrator with adaptive
    step-size control.  Abstract: subclasses must implement ``derivatives``.

    Not thread-safe (mirrors Java).
    """

    # ==================================================================
    # BUG_LEDGER
    # ==================================================================
    BUG_LEDGER: tuple = ()  # no bugs identified

    # ==================================================================
    # Cash-Karp tableau constants (class-level, read-only)
    # ==================================================================
    # The Butcher tableau coefficients are taken verbatim from Numerical
    # Recipes / the Java source.

    # a coefficients (node positions)
    _A2: float = 0.2
    _A3: float = 0.3
    _A4: float = 0.6
    _A5: float = 1.0
    _A6: float = 0.875

    # b coefficients (coupling matrix)
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

    # c coefficients (weights for 4th-order solution used in error estimate)
    _C1: float = 37.0 / 378.0
    _C3: float = 250.0 / 621.0
    _C4: float = 125.0 / 594.0
    _C6: float = 512.0 / 1771.0

    # dc coefficients (difference between 5th and 4th order weights for error)
    _DC1: float = 37.0 / 378.0 - 2825.0 / 27648.0
    _DC3: float = 250.0 / 621.0 - 18575.0 / 48384.0
    _DC4: float = 125.0 / 594.0 - 13525.0 / 55296.0
    _DC5: float = -277.0 / 14336.0
    _DC6: float = 512.0 / 1771.0 - 0.25

    # Safety / tuning constants for step-size control
    _SAFETY: float = 0.9
    _PGROW: float = -0.2
    _PSHRINK: float = -0.25
    _ERRCON: float = 1.89e-4  # (5/SAFETY)^(1/PGROW)

    # Default limits
    _DEFAULT_MAX_STEPS: int = 10000

    # ==================================================================
    # Constructor
    # ==================================================================
    def __init__(self, nVariables: int) -> None:
        """Create an integrator for a system of *nVariables* ODEs.

        Parameters
        ----------
        nVariables : int
            Number of dependent variables (dimension of the ODE system).
        """
        self.mNVariables: int = nVariables

        # Step-size information set by _qcStep
        self.mHDid: float = 0.0
        self.mHNext: float = 0.0

        # Dense-output save interval (0 → no saving)
        self.mSaveInterval: float = 0.0

        # Minimum allowed step size
        self.mMinStepSize: float = 0.0

        # Dense-output storage
        self.mXSave: Optional[F64Array] = None
        self.mYSave: Optional[F64Array] = None  # 2-D: shape (kMax, nVariables)
        self.mNSaved: int = 0

        # Integration control
        self.mMaxSteps: int = AdaptiveRungeKutta._DEFAULT_MAX_STEPS

        # Step counters
        self.mNOk: int = 0
        self.mNBad: int = 0

        # Work arrays – allocated lazily in _baseStep / _qcStep
        self.mWs2: Optional[F64Array] = None
        self.mWs3: Optional[F64Array] = None
        self.mWs4: Optional[F64Array] = None
        self.mWs5: Optional[F64Array] = None
        self.mWs6: Optional[F64Array] = None
        self.mYTemp: Optional[F64Array] = None

        self.mYErr: Optional[F64Array] = None
        self.mQcYTemp: Optional[F64Array] = None

    # ==================================================================
    # Abstract method — public, NO underscore prefix (R1)
    # ==================================================================
    @abc.abstractmethod
    def derivatives(self, x: float, y: F64Array, dydx: F64Array) -> None:
        """Compute the derivatives dy/dx at (*x*, *y*) and store into *dydx*.

        Subclasses MUST override this method.

        Parameters
        ----------
        x : float
            The independent variable.
        y : F64Array
            Current values of the dependent variables.
        dydx : F64Array
            Output array — filled with dy_i/dx for each variable.
        """
        ...

    # ==================================================================
    # _sign helper  (private in Java → underscore)
    # ==================================================================
    @staticmethod
    def _sign(magnitude: float, sign: float) -> float:
        """Fortran-style SIGN: return ``|magnitude|`` with the sign of *sign*.

        Java semantics: ``-0.0 >= 0.0`` is ``True``, so ``sign(m, -0.0)``
        returns ``+|m|``.  Python's ``math.copysign`` treats ``-0.0`` as
        negative, so we use the explicit conditional form per
        CONVERSION_GUIDE ``_sign`` section.
        """
        return abs(magnitude) if sign >= 0.0 else -abs(magnitude)

    # ==================================================================
    # _clearWorkspace (private)
    # ==================================================================
    def _clearWorkspace(self) -> None:
        """Release dense-output storage arrays."""
        self.mXSave = None
        self.mYSave = None
        self.mNSaved = 0

    # ==================================================================
    # _baseStep  (private)  —  Cash-Karp Runge-Kutta step
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
        """Take one Cash-Karp RK step of size *h*.

        Parameters
        ----------
        x : float
            Current independent variable.
        y : F64Array
            Current dependent-variable values (length mNVariables).
        dydx : F64Array
            Current derivatives (length mNVariables).
        h : float
            Step size to attempt.
        yout : F64Array
            *Output* — filled with the 5th-order solution.
        yerr : F64Array
            *Output* — filled with the embedded error estimate.
        """
        n: int = self.mNVariables

        # Lazy allocation of work arrays
        if self.mWs2 is None or self.mWs2.shape[0] != n:
            self.mWs2 = np.zeros(n, dtype=np.float64)
            self.mWs3 = np.zeros(n, dtype=np.float64)
            self.mWs4 = np.zeros(n, dtype=np.float64)
            self.mWs5 = np.zeros(n, dtype=np.float64)
            self.mWs6 = np.zeros(n, dtype=np.float64)
            self.mYTemp = np.zeros(n, dtype=np.float64)

        # --- Stage 2 ---
        assert self.mYTemp is not None
        assert self.mWs2 is not None
        assert self.mWs3 is not None
        assert self.mWs4 is not None
        assert self.mWs5 is not None
        assert self.mWs6 is not None

        for i in range(n):
            self.mYTemp[i] = y[i] + AdaptiveRungeKutta._B21 * h * dydx[i]
        self.derivatives(x + AdaptiveRungeKutta._A2 * h, self.mYTemp, self.mWs2)

        # --- Stage 3 ---
        for i in range(n):
            self.mYTemp[i] = y[i] + h * (
                AdaptiveRungeKutta._B31 * dydx[i]
                + AdaptiveRungeKutta._B32 * self.mWs2[i]
            )
        self.derivatives(x + AdaptiveRungeKutta._A3 * h, self.mYTemp, self.mWs3)

        # --- Stage 4 ---
        for i in range(n):
            self.mYTemp[i] = y[i] + h * (
                AdaptiveRungeKutta._B41 * dydx[i]
                + AdaptiveRungeKutta._B42 * self.mWs2[i]
                + AdaptiveRungeKutta._B43 * self.mWs3[i]
            )
        self.derivatives(x + AdaptiveRungeKutta._A4 * h, self.mYTemp, self.mWs4)

        # --- Stage 5 ---
        for i in range(n):
            self.mYTemp[i] = y[i] + h * (
                AdaptiveRungeKutta._B51 * dydx[i]
                + AdaptiveRungeKutta._B52 * self.mWs2[i]
                + AdaptiveRungeKutta._B53 * self.mWs3[i]
                + AdaptiveRungeKutta._B54 * self.mWs4[i]
            )
        self.derivatives(x + AdaptiveRungeKutta._A5 * h, self.mYTemp, self.mWs5)

        # --- Stage 6 ---
        for i in range(n):
            self.mYTemp[i] = y[i] + h * (
                AdaptiveRungeKutta._B61 * dydx[i]
                + AdaptiveRungeKutta._B62 * self.mWs2[i]
                + AdaptiveRungeKutta._B63 * self.mWs3[i]
                + AdaptiveRungeKutta._B64 * self.mWs4[i]
                + AdaptiveRungeKutta._B65 * self.mWs5[i]
            )
        self.derivatives(x + AdaptiveRungeKutta._A6 * h, self.mYTemp, self.mWs6)

        # --- Accumulate: 5th-order solution + error estimate ---
        for i in range(n):
            yout[i] = y[i] + h * (
                AdaptiveRungeKutta._C1 * dydx[i]
                + AdaptiveRungeKutta._C3 * self.mWs3[i]
                + AdaptiveRungeKutta._C4 * self.mWs4[i]
                + AdaptiveRungeKutta._C6 * self.mWs6[i]
            )
            yerr[i] = h * (
                AdaptiveRungeKutta._DC1 * dydx[i]
                + AdaptiveRungeKutta._DC3 * self.mWs3[i]
                + AdaptiveRungeKutta._DC4 * self.mWs4[i]
                + AdaptiveRungeKutta._DC5 * self.mWs5[i]
                + AdaptiveRungeKutta._DC6 * self.mWs6[i]
            )

    # ==================================================================
    # _qcStep  (private)  —  quality-controlled step
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
        """Attempt a step of size *htry*, shrinking until the error is within
        tolerance *eps*.  On success, mutates *y* in-place with the new values
        and sets ``self.mHDid`` / ``self.mHNext``.

        Parameters
        ----------
        x : float
            Current independent variable.
        y : F64Array
            Dependent variables — **mutated in place** on success.
        dydx : F64Array
            Current derivatives.
        htry : float
            Initial step size to try.
        eps : float
            Desired fractional accuracy.
        yscal : F64Array
            Scaling array against which error is measured.

        Returns
        -------
        float
            the new value of the independent variable after the step
            (``x + mHDid``).

        Raises
        ------
        UtilException
            When the step size underflows to zero.
        """
        n: int = self.mNVariables

        # Lazy allocation
        if self.mYErr is None or self.mYErr.shape[0] != n:
            self.mYErr = np.zeros(n, dtype=np.float64)
            self.mQcYTemp = np.zeros(n, dtype=np.float64)

        assert self.mYErr is not None
        assert self.mQcYTemp is not None

        h: float = htry

        while True:
            self._baseStep(x, y, dydx, h, self.mQcYTemp, self.mYErr)

            # Evaluate accuracy
            errmax: float = 0.0
            for i in range(n):
                errmax = max(errmax, abs(self.mYErr[i] / yscal[i]))
            errmax /= eps

            if errmax <= 1.0:
                break

            # Truncation error too large — shrink step
            htemp: float = AdaptiveRungeKutta._SAFETY * h * math.pow(errmax, AdaptiveRungeKutta._PSHRINK)
            h = max(htemp, 0.1 * h) if h >= 0.0 else min(htemp, 0.1 * h)

            # Check for step size underflow
            xnew: float = x + h
            if xnew == x:
                raise UtilException("Step size underflow in _qcStep")

        # Step succeeded
        if errmax > AdaptiveRungeKutta._ERRCON:
            self.mHNext = AdaptiveRungeKutta._SAFETY * h * math.pow(errmax, AdaptiveRungeKutta._PGROW)
        else:
            self.mHNext = 5.0 * h

        self.mHDid = h

        # Update y in place
        for i in range(n):
            y[i] = self.mQcYTemp[i]

        return x + h

    # ==================================================================
    # Public accessors
    # ==================================================================

    def setSaveInterval(self, saveInterval: float) -> None:
        """Set the interval between dense-output saves.

        Parameters
        ----------
        saveInterval : float
            Interval in the independent variable between saves.
            ``float('inf')`` or ``0.0`` disables saving.
        """
        self.mSaveInterval = saveInterval

    def getSaveInterval(self) -> float:
        """Return the current dense-output save interval."""
        return self.mSaveInterval

    def getSavedX(self, i: int) -> float:
        """Return the *i*-th saved independent-variable value.

        Parameters
        ----------
        i : int
            Index into the saved-point array.

        Returns
        -------
        float
        """
        assert self.mXSave is not None
        return float(self.mXSave[i])

    def getY(self, i: int) -> F64Array:
        """Return a view of the *i*-th saved dependent-variable vector.

        Parameters
        ----------
        i : int
            Index into the saved-point array.

        Returns
        -------
        F64Array
            Row view of ``mYSave[i]``.
        """
        assert self.mYSave is not None
        return self.mYSave[i]

    def getSavedCount(self) -> int:
        """Return how many dense-output points were stored by the last
        ``integrate`` call."""
        return self.mNSaved

    def setMaxSteps(self, maxSteps: int) -> None:
        """Set the maximum number of integration steps allowed.

        Parameters
        ----------
        maxSteps : int
        """
        self.mMaxSteps = maxSteps

    def setMinStepSize(self, minStep: float) -> None:
        """Set the minimum allowed step size (0.0 for no floor).

        Parameters
        ----------
        minStep : float
        """
        self.mMinStepSize = minStep

    def getNVariables(self) -> int:
        """Return the number of dependent variables."""
        return self.mNVariables

    def getStepCount(self) -> int:
        """Return total step count (good + bad) from last ``integrate``."""
        return self.mNOk + self.mNBad

    def getGoodStepCount(self) -> int:
        """Return the number of steps that met the accuracy target."""
        return self.mNOk

    def getBadStepCount(self) -> int:
        """Return the number of steps that required subdivision."""
        return self.mNBad

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
        """Integrate the ODE system from *x1* to *x2*.

        Parameters
        ----------
        x1 : float
            Start of the integration range.
        x2 : float
            End of the integration range.
        ystart : F64Array
            Initial y values — **mutated in place** to contain final values.
        eps : float
            Desired fractional accuracy.
        h1 : float
            Initial step size (sign is forced to match ``x2 - x1``).

        Returns
        -------
        F64Array
            The final y values (same data as *ystart* after mutation).

        Raises
        ------
        UtilException
            Upon exceeding ``mMaxSteps`` or when step size becomes too small.
        """
        n: int = self.mNVariables

        # R7: 1.0e-10 * eps — Java used tiny = 1.0e-10 * eps; literal.
        tiny: float = 1.0e-10 * eps

        yscal: F64Array = np.zeros(n, dtype=np.float64)
        dydx: F64Array = np.zeros(n, dtype=np.float64)
        y: F64Array = np.zeros(n, dtype=np.float64)

        x: float = x1
        h: float = AdaptiveRungeKutta._sign(h1, x2 - x1)
        xsav: float = 0.0
        saveInt: float = float("inf")  # Java Double.MAX_VALUE
        kMax: int = 0

        self.mNSaved = 0
        self.mNOk = 0
        self.mNBad = 0

        # Copy ystart → y
        for i in range(n):
            y[i] = ystart[i]

        # Dense-output setup
        # Java: if (mSaveInterval != Double.MAX_VALUE)
        # We treat mSaveInterval > 0 and < inf as "saving enabled", matching
        # the constructor default of 0.0 meaning "disabled" and the Java
        # default of Double.MAX_VALUE meaning "disabled".
        if self.mSaveInterval != 0.0 and not math.isinf(self.mSaveInterval):
            # R7: Math.round → int(math.floor(x + 0.5))
            kMax = int(
                math.floor(
                    (abs(x2 - x1) + self.mSaveInterval) / self.mSaveInterval + 0.5
                )
            )
            saveInt = AdaptiveRungeKutta._sign(self.mSaveInterval, x2 - x1)
            xsav = x - 2.0 * saveInt  # ensure first step is saved
            self.mXSave = np.zeros(kMax, dtype=np.float64)
            self.mYSave = np.zeros((kMax, n), dtype=np.float64)

        for step in range(self.mMaxSteps):
            # --- Save point if appropriate ---
            if (
                kMax != 0
                and self.mNSaved < kMax
                and abs(x - xsav) >= 0.9999 * self.mSaveInterval
            ):
                assert self.mXSave is not None
                assert self.mYSave is not None
                self.mXSave[self.mNSaved] = x
                for i in range(n):
                    self.mYSave[self.mNSaved, i] = y[i]
                xsav = x
                self.mNSaved += 1

            # --- Compute derivatives & scaling ---
            self.derivatives(x, y, dydx)

            # Rescale h to ensure accuracy during special points (turning
            # points etc.).  Java:
            #   for(int i=0;i<mNVariables;++i)
            #      yscal[i] = Math.abs(y[i]) + Math.abs(dydx[i]*h) + tiny;
            for i in range(n):
                yscal[i] = abs(y[i]) + abs(dydx[i] * h) + tiny

            # --- Check if we would overshoot x2 ---
            if (h > 0.0 and (x + h) > x2) or (h < 0.0 and (x + h) < x2):
                h = x2 - x

            # --- Take a quality-controlled step ---
            x = self._qcStep(x, y, dydx, h, eps, yscal)

            if self.mHDid == h:
                self.mNOk += 1
            else:
                self.mNBad += 1

            # --- Are we done? ---
            if (x - x2) * (x2 - x1) >= 0.0:
                # Copy final y back into ystart
                for i in range(n):
                    ystart[i] = y[i]

                # Save last point
                if kMax != 0:
                    assert self.mXSave is not None
                    assert self.mYSave is not None
                    if self.mNSaved < kMax:
                        self.mXSave[self.mNSaved] = x
                        for i in range(n):
                            self.mYSave[self.mNSaved, i] = y[i]
                        self.mNSaved += 1

                return y

            # --- Prepare step size for next iteration ---
            if abs(self.mHNext) <= self.mMinStepSize:
                raise UtilException(
                    "Step size too small in AdaptiveRungeKutta.integrate"
                )
            h = self.mHNext

        raise UtilException("Too many steps in AdaptiveRungeKutta.integrate")

    # ==================================================================
    # integrate_literal — alias (no scipy alternative for this algorithm)
    # ==================================================================
    def integrate_literal(
        self,
        x1: float,
        x2: float,
        ystart: F64Array,
        eps: float,
        h1: float,
    ) -> F64Array:
        """Literal Java port of ``integrate``.  Since ``integrate`` is already
        a line-for-line translation (no scipy substitution), this simply
        delegates to ``integrate``.
        """
        return self.integrate(x1, x2, ystart, eps, h1)

    # ==================================================================
    # Dunder / named-alias pairs (R2)
    # ==================================================================
    def __repr__(self) -> str:
        return (
            f"AdaptiveRungeKutta(nVariables={self.mNVariables}, "
            f"steps={self.getStepCount()}, "
            f"ok={self.mNOk}, bad={self.mNBad})"
        )

    def toString(self) -> str:
        """Java-style ``toString()`` alias for ``__repr__``."""
        return self.__repr__()

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AdaptiveRungeKutta):
            return NotImplemented
        return self is other  # identity semantics, matching Java default

    def equals(self, other: object) -> bool:
        """Java-style ``equals()`` alias for ``__eq__``."""
        return self.__eq__(other)

    def __hash__(self) -> int:
        return id(self)

    def hashCode(self) -> int:
        """Java-style ``hashCode()`` alias for ``__hash__``."""
        return self.__hash__()

