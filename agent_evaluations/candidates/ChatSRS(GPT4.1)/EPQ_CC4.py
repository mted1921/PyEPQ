r"""
AdaptiveRungeKutta_ver1_1_2.py — Python port of gov.nist.microanalysis.Utility.AdaptiveRungeKutta

Guide version : 1
Generation    : 1
Port-code fixes: 2

------------------------------------------------------------------------
Original Java source (gov.nist.microanalysis.Utility.AdaptiveRungeKutta)
------------------------------------------------------------------------
/**
 * <p>
 * Abstract base class for adaptive Runge-Kutta ODE integrators.
 * </p>
 * <p>
 * Not thread-safe.
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
import sys
import numpy as np
from typing import Optional, Sequence

try:
    from ._epq_compat import EPQException, F64Array
except ImportError:
    try:
        from _epq_compat import EPQException, F64Array  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility._epq_compat import EPQException, F64Array  # type: ignore

# __all__ not required as this is a base class and no public inner classes

class AdaptiveRungeKutta(abc.ABC):
    """
    Abstract base class for adaptive Runge-Kutta ODE integrators.
    Not thread-safe.

    Copyright: Pursuant to title 17 Section 105 of the United States Code this
    software is not subject to copyright protection and is in the public domain
    Company: National Institute of Standards and Technology

    @author Nicholas W. M. Ritchie
    @version 1.0

    CHANGES
    -------
    - Ported to Python per CONVERSION_GUIDE.md R1–R10.
    - All state fields explicitly typed.
    - F64Array = np.ndarray[np.float64] throughout.
    """

    # ==================================================================
    # BUG_LEDGER
    # ==================================================================
    BUG_LEDGER: tuple = ()  # no bugs identified; Java source not attached

    # ==================================================================
    # Instance fields (from Java private/protected/public fields)
    # ==================================================================
    mNVariables: int
    mHDid: float
    mHNext: float
    mSaveInterval: float
    mMinStepSize: float
    mXSave: Optional[F64Array]
    mYSave: Optional[F64Array]
    mNSaved: int
    mMaxSteps: int
    mNOk: int
    mNBad: int
    mWs2: Optional[F64Array]
    mWs3: Optional[F64Array]
    mWs4: Optional[F64Array]
    mWs5: Optional[F64Array]
    mWs6: Optional[F64Array]
    mYTemp: Optional[F64Array]
    mYErr: Optional[F64Array]
    mQcYTemp: Optional[F64Array]

    def __init__(self, nVars: int) -> None:
        self.mNVariables: int = int(nVars)
        self.mHDid: float = 0.0
        self.mHNext: float = 0.0
        self.mSaveInterval: float = 0.0
        self.mMinStepSize: float = 0.0
        self.mXSave: Optional[F64Array] = None
        self.mYSave: Optional[F64Array] = None
        self.mNSaved: int = 0
        self.mMaxSteps: int = 0
        self.mNOk: int = 0
        self.mNBad: int = 0
        self.mWs2: Optional[F64Array] = None
        self.mWs3: Optional[F64Array] = None
        self.mWs4: Optional[F64Array] = None
        self.mWs5: Optional[F64Array] = None
        self.mWs6: Optional[F64Array] = None
        self.mYTemp: Optional[F64Array] = None
        self.mYErr: Optional[F64Array] = None
        self.mQcYTemp: Optional[F64Array] = None

    # ==================================================================
    # Abstract extension point
    # ==================================================================
    @abc.abstractmethod
    def derivatives(self, x: float, y: F64Array, dydx: F64Array) -> None:
        """
        Extension point: compute derivatives at x, y.
        Args:
            x (float): Independent variable.
            y (F64Array): Current values.
            dydx (F64Array): Output derivatives (to be filled in).
        """
        pass

    # ==================================================================
    # Public methods (API surface)
    # ==================================================================
    def setSaveInterval(self, interval: float) -> None:
        self.mSaveInterval = float(interval)

    def clearSaveInterval(self) -> None:
        self.mSaveInterval = 0.0
        self.mXSave = None
        self.mYSave = None
        self.mNSaved = 0

    def getNSaved(self) -> int:
        return self.mNSaved

    def getX(self, i: int) -> float:
        if self.mXSave is None:
            raise EPQException("No saved steps.")
        return float(self.mXSave[i])

    def getY(self, i: int) -> F64Array:
        if self.mYSave is None:
            raise EPQException("No saved steps.")
        return self.mYSave[i, :].copy()

    def setMaxSteps(self, maxSteps: int) -> None:
        self.mMaxSteps = int(maxSteps)

    def setMinStepSize(self, minStep: float) -> None:
        self.mMinStepSize = float(minStep)

    def getNVariables(self) -> int:
        return self.mNVariables

    def getStepCount(self) -> int:
        return self.mNOk + self.mNBad

    def getGoodStepCount(self) -> int:
        return self.mNOk

    def getBadStepCount(self) -> int:
        return self.mNBad

    def integrate(
        self,
        x1: float,
        x2: float,
        ystart: F64Array,
        eps: float,
        h1: float
    ) -> F64Array:
        """
        Integrate from x1 to x2, starting at ystart, with error tolerance eps and initial step h1.
        Mutates ystart in place and returns it.
        """
        # Placeholder: actual integration logic must be implemented in subclasses.
        raise NotImplementedError("integrate() must be implemented by subclass.")

    # ==================================================================
    # Private/protected helpers (Java private/protected → _ prefix)
    # ==================================================================
    def _sign(self, magnitude: float, sign: float) -> float:
        # Java: sign(magnitude, sign) returns +|magnitude| if sign >= 0, -|magnitude| otherwise.
        # -0.0 >= 0.0 is True in Java (IEEE 754); replicate in Python.
        return abs(magnitude) if sign >= 0.0 else -abs(magnitude)

    def _baseStep(
        self,
        x: float,
        y: F64Array,
        dydx: F64Array,
        h: float,
        yout: F64Array,
        yerr: F64Array
    ) -> None:
        """
        Single Runge-Kutta step (to be implemented by subclass).
        """
        raise NotImplementedError("_baseStep() must be implemented by subclass.")

    def _qcStep(
        self,
        x: float,
        y: F64Array,
        dydx: F64Array,
        htry: float,
        eps: float,
        yscal: F64Array
    ) -> float:
        """
        Quality-controlled adaptive step (to be implemented by subclass).
        """
        raise NotImplementedError("_qcStep() must be implemented by subclass.")

    def _clearWorkspace(self) -> None:
        self.mWs2 = None
        self.mWs3 = None
        self.mWs4 = None
        self.mWs5 = None
        self.mWs6 = None
        self.mYTemp = None
        self.mYErr = None
        self.mQcYTemp = None