r"""
Integrator_ver1_1_2.py — Python port of gov.nist.microanalysis.Utility.Integrator

Guide version : 1
Generation    : 1
Port-code fixes: 2

CHANGES
-------
* SCIPY-DEV-1: `integrate()` defaults to `scipy.integrate.quad` for standard 
  calls, while `integrate_literal()` preserves the original `AdaptiveRungeKutta` 
  delegation and its failure modes. The scipy variant implicitly corrects JAVA-BUG-1.
* FIX-1 (Test-bug): test_constant_integral_equals_interval_length modified to expect `b - a` rather than `abs(b - a)` since `integrate` (via scipy) correctly computes the signed integral. (Test fix, does not increment Port-code fixes).

------------------------------------------------------------------------
Original Java source (gov.nist.microanalysis.Utility.Integrator)
------------------------------------------------------------------------
/**
 * <p>
 * Implements a numerical integration routine based on an AdaptiveRungeKutta
 * algorithm from Press et al.
 * </p>
 * <p>
 * Copyright: Pursuant to title 17 Section 105 of the United States Code this
 * software is not subject to copyright protection and is in the public domain
 * </p>
 * <p>
 * Institution: National Institute of Standards and Technology
 * </p>
 * 
 * @author Nicholas
 * @version 1.0
 */
------------------------------------------------------------------------
"""

from __future__ import annotations

import abc
import math
from typing import Optional, Union, Callable

import numpy as np

try:
    from ._epq_compat import EPQException, F64Array, JavaRandom
except ImportError:
    try:
        from _epq_compat import EPQException, F64Array, JavaRandom  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility._epq_compat import EPQException, F64Array, JavaRandom  # type: ignore

try:
    from .AdaptiveRungeKutta_ver1_1_2 import AdaptiveRungeKutta
except ImportError:
    try:
        from AdaptiveRungeKutta_ver1_1_2 import AdaptiveRungeKutta  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility.AdaptiveRungeKutta_ver1_1_2 import AdaptiveRungeKutta  # type: ignore

try:
    from .UtilException_ver1_1_0 import UtilException
except ImportError:
    try:
        from UtilException_ver1_1_0 import UtilException  # type: ignore
    except ImportError:
        from UtilException_ver1_1_0 import UtilException  # type: ignore


class Integrator(AdaptiveRungeKutta, abc.ABC):
    """
    Implements a numerical integration routine based on an AdaptiveRungeKutta
    algorithm from Press et al.
    """

    BUG_LEDGER: tuple = (
        ("JAVA-BUG-1", "integrate",
         "Java source: `return highVal > lowVal ? integrate(lowVal, highVal, y, mTolerance, 0.05 * (highVal - lowVal))[0] : 0.0;` "
         "Returns 0.0 when highVal <= lowVal instead of the negative integral. "
         "Preserved verbatim in `integrate_literal`.", True),
    )

    def __init__(self, tol: Optional[float] = None) -> None:
        super().__init__(1)
        self._mTolerance: float = 1.0e-6 if tol is None else float(tol)

    @classmethod
    def from_tol(cls, tol: float) -> "Integrator":
        return cls(tol)

    def integrate_literal(  # type: ignore[override]
        self, lowVal: float, highVal: float,
    ) -> float:
        # R4/Java-hiding: Java's Integrator.integrate(double,double) and
        # AdaptiveRungeKutta.integrate(double,double[],double,double) are distinct
        # overloads resolved at compile time. Python has one slot; this 2-arg form
        # hides the 6-arg base. super().integrate_literal() still calls ARK correctly.
        y: F64Array = np.array([0.0], dtype=np.float64)
        try:
            if highVal > lowVal:
                res: F64Array = super().integrate_literal(
                    lowVal, highVal, y, self._mTolerance, 0.05 * (highVal - lowVal)
                )
                return float(res[0])
            return 0.0
        except UtilException:
            import traceback
            traceback.print_exc()
            return math.nan

    def integrate_strict(self, lowVal: float, highVal: float) -> float:
        """Strict variant: correctly handles reversed limits (fixes JAVA-BUG-1)."""
        y: F64Array = np.array([0.0], dtype=np.float64)
        try:
            if highVal == lowVal:
                return 0.0
            elif highVal > lowVal:
                res: F64Array = super().integrate_literal(
                    lowVal, highVal, y, self._mTolerance, 0.05 * (highVal - lowVal)
                )
                return float(res[0])
            else:
                res_rev: F64Array = super().integrate_literal(
                    highVal, lowVal, y, self._mTolerance, 0.05 * (lowVal - highVal)
                )
                return -float(res_rev[0])
        except UtilException:
            import traceback
            traceback.print_exc()
            return math.nan

    def integrate(  # type: ignore[override]
        self, lowVal: float, highVal: float,
    ) -> float:
        # R4/Java-hiding: same name-slot issue as integrate_literal above.
        # SCIPY-DEV-1: scipy.integrate.quad corrects JAVA-BUG-1 (reversed limits
        # return 0.0 in Java; scipy returns the correct negative integral).
        import scipy.integrate as _sp_integrate
        try:
            res, _ = _sp_integrate.quad(
                self.getValue, lowVal, highVal, epsabs=self._mTolerance,
            )
            return float(res)
        except Exception:
            return math.nan

    def derivatives(self, x: float, y: F64Array, dydx: F64Array) -> None:
        self._require_mutable_f64(dydx, "dydx")
        dydx[0] = self.getValue(x)

    @abc.abstractmethod
    def getValue(self, x: float) -> float:
        pass

    @staticmethod
    def _require_mutable_f64(arr: np.ndarray, name: str = "arr") -> None:
        """Type guard for in-place methods (CONVERSION_GUIDE R5)."""
        if not isinstance(arr, np.ndarray):
            raise TypeError(
                f"{name} must be a numpy ndarray (got {type(arr).__name__}); "
                "in-place helpers cannot mutate Python lists or tuples."
            )
        if arr.dtype != np.float64:
            raise TypeError(
                f"{name} must have dtype float64 (got {arr.dtype}); "
                "wrong-dtype arrays would silently copy."
            )
        if not arr.flags.writeable:
            raise TypeError(f"{name} must be writeable")