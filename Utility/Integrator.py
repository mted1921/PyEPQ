r"""
Integrator_ver2_1_1.py — Python port of gov.nist.microanalysis.Utility.Integrator

Guide version : 2
Generation    : 1
Port-code fixes: 1

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

CHANGES (from Java):

  R2   — getValue() is the sole abstract method; concrete override `derivatives()`
          wires it into the ARK stepper.
         integrate() is the scipy primary (scipy.integrate.quad): handles reversed
          limits with the correct sign (corrects JAVA-BUG-1).
         integrate_literal() is the line-for-line Java port: preserves JAVA-BUG-1
          (returns 0.0 when highVal ≤ lowVal).
         integrate_strict() is the BUG_LEDGER strict variant: signs the reversed-
          limits result by swapping limits and negating.

  R2-alias — Integrator.integrate(low, high) shadows AdaptiveRungeKutta.integrate(
          x1, x2, ystart, eps, h1).  Parent methods aliased before the override
          per the guide's shadowing rule:
            _integrate_ode         = AdaptiveRungeKutta.integrate
            _integrate_ode_literal = AdaptiveRungeKutta.integrate_literal

  R1   — Single abstract method `getValue` uses the un-prefixed Java name.

  FIX-1 (parent-alias-error): In integrate_literal(), the call to the parent alias
          _integrate_ode_literal was incorrectly written as
          `self._integrate_ode_literal(self, ...)` — passing self twice (once via
          bound-method lookup, once explicitly). The explicit `self` argument is
          removed; the bound-method call now correctly supplies self automatically.

JAVA-BUG-1 — `integrate(double lowVal, double highVal)` in Java:
  Source line: `return highVal > lowVal ? integrate(...)[0] : 0.0`
  Returns 0.0 for reversed limits (highVal ≤ lowVal) instead of the
  negative integral.  The literal port preserves this; the scipy primary
  and integrate_strict() correct it.
  Disposition: preserve in integrate_literal(); correct in integrate() and
  integrate_strict().

BUG_LEDGER: tuple = (
    ("JAVA-BUG-1", "integrate",
     "Java: `highVal > lowVal ? ... : 0.0` returns 0.0 for reversed limits. "
     "Corrected in integrate() (scipy.quad) and integrate_strict(). "
     "Preserved in integrate_literal()."),
)
"""
from __future__ import annotations

import abc
import math

import numpy as np
import scipy.integrate

try:
    from ._epq_compat import F64Array
except ImportError:
    try:
        from _epq_compat import F64Array  # type: ignore[no-redef]
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility_ver2._epq_compat import F64Array  # type: ignore[no-redef]

try:
    from .AdaptiveRungeKutta_ver2_1_1 import AdaptiveRungeKutta
except ImportError:
    try:
        from AdaptiveRungeKutta_ver2_1_1 import AdaptiveRungeKutta  # type: ignore[no-redef]
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility_ver2.AdaptiveRungeKutta_ver2_1_1 import AdaptiveRungeKutta  # type: ignore[no-redef]

try:
    from .UtilException_ver2_1_1 import UtilException
except ImportError:
    try:
        from UtilException_ver2_1_1 import UtilException  # type: ignore[no-redef]
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility_ver2.UtilException_ver2_1_1 import UtilException  # type: ignore[no-redef]


BUG_LEDGER: tuple = (
    ("JAVA-BUG-1", "integrate",
     "Java: `highVal > lowVal ? ... : 0.0` returns 0.0 for reversed limits. "
     "Corrected in integrate() (scipy.quad) and integrate_strict(). "
     "Preserved in integrate_literal()."),
)


class Integrator(AdaptiveRungeKutta, abc.ABC):
    """Port of gov.nist.microanalysis.Utility.Integrator.

    Abstract base for one-dimensional numerical integration via AdaptiveRungeKutta.
    Subclass and implement getValue(x); then call integrate(low, high).
    """

    # Alias parent's integrate/integrate_literal before we shadow them (R2-alias).
    _integrate_ode = AdaptiveRungeKutta.integrate
    _integrate_ode_literal = AdaptiveRungeKutta.integrate_literal

    def __init__(self, tol: float = 1.0e-6) -> None:
        super().__init__(1)  # one ODE dimension: y[0] = running integral
        self._mTolerance: float = tol

    @abc.abstractmethod
    def getValue(self, x: float) -> float:
        """Integrand: returns the function value at x."""

    def derivatives(self, x: float, y: F64Array, dydx: F64Array) -> None:
        """Override of AdaptiveRungeKutta.derivatives: dydx[0] = getValue(x)."""
        dydx[0] = self.getValue(x)

    def integrate(self, lowVal: float, highVal: float) -> float:
        """Scipy primary: scipy.integrate.quad for signed result (corrects JAVA-BUG-1)."""
        result, _ = scipy.integrate.quad(self.getValue, lowVal, highVal)
        return result

    def integrate_literal(self, lowVal: float, highVal: float) -> float:
        """Line-for-line Java translation. Preserves JAVA-BUG-1: returns 0.0 for
        reversed limits (highVal ≤ lowVal)."""
        if not (highVal > lowVal):
            return 0.0  # JAVA-BUG-1: preserved
        y: F64Array = np.zeros(1, dtype=np.float64)
        try:
            return float(self._integrate_ode_literal(  # FIX-1: removed erroneous explicit `self`
                lowVal, highVal, y, self._mTolerance,
                0.05 * (highVal - lowVal),
            )[0])
        except UtilException:
            return float('nan')

    def integrate_strict(self, lowVal: float, highVal: float) -> float:
        """Strict variant correcting JAVA-BUG-1: returns signed integral for
        reversed limits by swapping and negating."""
        if lowVal == highVal:
            return 0.0
        if highVal > lowVal:
            return self.integrate_literal(lowVal, highVal)
        return -self.integrate_literal(highVal, lowVal)
