r"""
Constraint_ver2_1_1.py — Python port of gov.nist.microanalysis.Utility.Constraint

Guide version : 2
Generation    : 1
Port-code fixes: 3

CHANGES
-------
- Java `Constraint.None` inner class renamed to `Constraint.Unconstrained` (R1/R10):
  `None` is a Python builtin; the Java `toString()` already returns `"Unconstrained[]"`.
  Module-level alias `Unconstrained` provided; no `None` alias exported.
- Inner classes defined at module level and attached as `Constraint.<Name>` after
  `Constraint` is fully constructed (Python cannot inherit from an outer class while
  nested inside it; attaching post-hoc reproduces the Java nested-class access pattern).
- FIX-1: R4-violation in `_Positive.getResult_literal` — `UncertainValue2.multiply()`
  does not exist; the port split Java overloads into `multiply_dn`, `multiply_nd`,
  `multiply_nn`.  `self._mScale` (float) × `exp(param)` (UncertainValue2) → `multiply_dn`.
- FIX-2: R4-violation in `_Fractional.getResult_literal` — BOTH the outer `add`
  and the inner `multiply` were split by the port.  `add(float, UV2) → add_dn`;
  `self._mScale * self._mFraction * self._TWO_OVER_PI` (float) × `atan(param)`
  (UncertainValue2) → `multiply_dn`.  (The `add` split was masked until the
  Fractional `getResult` coverage was added to the harness.)
- FIX-3: R4-violation in `_Bounded.getResult_literal` — BOTH the outer `add`
  and the inner `multiply` were split by the port.  `add(float, UV2) → add_dn`;
  `self._mWidth * self._TWO_OVER_PI` (float) × `atan(param)` (UncertainValue2)
  → `multiply_dn`.  (The `add` split was masked until the Bounded `getResult`
  coverage was added to the harness.)

------------------------------------------------------------------------
Original Java source (gov.nist.microanalysis.Utility.Constraint)
------------------------------------------------------------------------
/**
 * <p>
 * Provides a generic mechanism for constraining the fit parameters. The
 * parameter the optimizer sees can range over the full extent of the real
 * numbers but the fit parameter can be constrained within a sub-set of the
 * reals. Typically, the constraints are either within a bounded range of
 * values, strictly positive or strictly negative.
 * </p>
 * <p>
 * Constraints are implemented by mapping the [-&#8734;,&#8734;] onto the
 * desired sub-range through some invertible, differentiable function.
 * </p>
 * <p>
 * Copyright: Pursuant to title 17 Section 105 of the United States Code this
 * software is not subject to copyright protection and is in the public domain
 * </p>
 * <p>
 * Institution: National Institute of Standards and Technology
 * </p>
 *
 * @author nritchie
 * @version 1.0
 */
------------------------------------------------------------------------
"""
from __future__ import annotations

import abc
import math
from typing import Optional

try:
    from ._epq_compat import EPQException, F64Array, JavaRandom
except ImportError:
    try:
        from _epq_compat import EPQException, F64Array, JavaRandom  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility._epq_compat import EPQException, F64Array, JavaRandom  # type: ignore

try:
    from .UncertainValue2_ver2_1_0 import UncertainValue2
except ImportError:
    try:
        from UncertainValue2_ver2_1_0 import UncertainValue2  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility_ver2.UncertainValue2_ver2_1_0 import UncertainValue2  # type: ignore

BUG_LEDGER: tuple = ()  # no bugs identified


class Constraint(abc.ABC):
    """Python port of the Java `Constraint` interface.

    Maps an unconstrained real parameter (range ℝ) onto a sub-range through an
    invertible, differentiable function.  Four concrete implementations are
    provided as class attributes after this class is defined:
    ``Positive``, ``Fractional``, ``Bounded``, ``Unconstrained``.
    """

    @abc.abstractmethod
    def realToConstrained(self, param: float) -> float:
        """Map *param* from the full real line onto the constrained range."""

    @abc.abstractmethod
    def constrainedToReal(self, param: float) -> float:
        """Invert `realToConstrained`; `constrainedToReal(realToConstrained(p)) == p`."""

    @abc.abstractmethod
    def derivative(self, param: float) -> float:
        """d(`realToConstrained`)/d(`param`) at *param*."""

    @abc.abstractmethod
    def getResult(self, param: UncertainValue2) -> UncertainValue2:
        """Apply `realToConstrained` with uncertainty propagation."""


# ---------------------------------------------------------------------------
# Concrete implementations — defined at module level so they can inherit from
# Constraint, then attached as nested-class attributes (mirrors Java inner classes).
# ---------------------------------------------------------------------------

class _Positive(Constraint):
    """Constrains the fit parameter to be strictly positive via exp mapping.

    Java: ``public class Positive implements Constraint``
    """

    def __init__(self, scale: float) -> None:
        self._mScale: float = float(scale)

    # --- realToConstrained ---

    def realToConstrained_literal(self, param: float) -> float:
        return self._mScale * math.exp(param)

    def realToConstrained(self, param: float) -> float:  # SCIPY-NONE: no library substitute
        return self.realToConstrained_literal(param)

    # --- constrainedToReal ---

    def constrainedToReal_literal(self, param: float) -> float:
        return math.log(param / self._mScale)

    def constrainedToReal(self, param: float) -> float:  # SCIPY-NONE: no library substitute
        return self.constrainedToReal_literal(param)

    # --- derivative ---

    def derivative_literal(self, param: float) -> float:
        return self._mScale * math.exp(param)

    def derivative(self, param: float) -> float:  # SCIPY-NONE: no library substitute
        return self.derivative_literal(param)

    # --- getResult ---

    def getResult_literal(self, param: UncertainValue2) -> UncertainValue2:
        return UncertainValue2.multiply_dn(self._mScale, UncertainValue2.exp(param))  # FIX-1: R4-violation

    def getResult(self, param: UncertainValue2) -> UncertainValue2:  # SCIPY-NONE: no library substitute
        return self.getResult_literal(param)

    # --- toString / __str__ ---

    def __str__(self) -> str:
        return "Positive[scale=" + str(float(self._mScale)) + "]"

    def toString(self) -> str:
        return str(self)


class _Fractional(Constraint):
    """Constrains the fit parameter within a fractional range of *scale* via atan mapping.

    Java: ``public class Fractional implements Constraint``
    """

    _TWO_OVER_PI: float = 2.0 / math.pi

    def __init__(self, name: str, scale: float, fraction: float) -> None:
        self._mName: str = name
        self._mScale: float = float(scale)
        self._mFraction: float = float(fraction)

    # --- realToConstrained ---

    def realToConstrained_literal(self, param: float) -> float:
        return self._mScale * (1.0 + (self._mFraction * self._TWO_OVER_PI * math.atan(param)))

    def realToConstrained(self, param: float) -> float:  # SCIPY-NONE: no library substitute
        return self.realToConstrained_literal(param)

    # --- constrainedToReal ---

    def constrainedToReal_literal(self, param: float) -> float:
        return math.tan((param - self._mScale) / (self._mScale * self._mFraction * self._TWO_OVER_PI))

    def constrainedToReal(self, param: float) -> float:  # SCIPY-NONE: no library substitute
        return self.constrainedToReal_literal(param)

    # --- derivative ---

    def derivative_literal(self, param: float) -> float:
        return (self._mScale * self._mFraction * self._TWO_OVER_PI) / (1.0 + (param * param))

    def derivative(self, param: float) -> float:  # SCIPY-NONE: no library substitute
        return self.derivative_literal(param)

    # --- getResult ---

    def getResult_literal(self, param: UncertainValue2) -> UncertainValue2:
        return UncertainValue2.add_dn(  # FIX-2: R4-violation; float × UV2 → add_dn
            self._mScale,
            UncertainValue2.multiply_dn(  # FIX-2: R4-violation; float × UV2 → multiply_dn
                self._mScale * self._mFraction * self._TWO_OVER_PI,
                UncertainValue2.atan(param),
            ),
        )

    def getResult(self, param: UncertainValue2) -> UncertainValue2:  # SCIPY-NONE: no library substitute
        return self.getResult_literal(param)

    # --- toString / __str__ ---

    def __str__(self) -> str:
        return (
            "Fraction["
            + self._mName
            + ","
            + str(float(self._mScale))
            + " +- "
            + str(float(self._mScale * self._mFraction))
            + "]"
        )

    def toString(self) -> str:
        return str(self)


class _Bounded(Constraint):
    """Constrains the fit parameter within *width* of *center* via atan mapping.

    Java: ``public class Bounded implements Constraint``

    The constructor stores ``0.5 * width`` in ``_mWidth``; the visible range is
    ``[center - 0.5*width, center + 0.5*width]``.
    """

    _TWO_OVER_PI: float = 2.0 / math.pi

    def __init__(self, center: float, width: float) -> None:
        self._mCenter: float = float(center)
        self._mWidth: float = 0.5 * float(width)  # Java: mWidth = 0.5 * width

    # --- realToConstrained ---

    def realToConstrained_literal(self, param: float) -> float:
        return self._mCenter + (self._mWidth * self._TWO_OVER_PI * math.atan(param))

    def realToConstrained(self, param: float) -> float:  # SCIPY-NONE: no library substitute
        return self.realToConstrained_literal(param)

    # --- constrainedToReal ---

    def constrainedToReal_literal(self, param: float) -> float:
        return math.tan((param - self._mCenter) / (self._mWidth * self._TWO_OVER_PI))

    def constrainedToReal(self, param: float) -> float:  # SCIPY-NONE: no library substitute
        return self.constrainedToReal_literal(param)

    # --- derivative ---

    def derivative_literal(self, param: float) -> float:
        return (self._mWidth * self._TWO_OVER_PI) / (1.0 + (param * param))

    def derivative(self, param: float) -> float:  # SCIPY-NONE: no library substitute
        return self.derivative_literal(param)

    # --- getResult ---

    def getResult_literal(self, param: UncertainValue2) -> UncertainValue2:
        return UncertainValue2.add_dn(  # FIX-3: R4-violation; float × UV2 → add_dn
            self._mCenter,
            UncertainValue2.multiply_dn(  # FIX-3: R4-violation; float × UV2 → multiply_dn
                self._mWidth * self._TWO_OVER_PI,
                UncertainValue2.atan(param),
            ),
        )

    def getResult(self, param: UncertainValue2) -> UncertainValue2:  # SCIPY-NONE: no library substitute
        return self.getResult_literal(param)

    # --- toString / __str__ ---

    def __str__(self) -> str:
        return (
            "Bounded[min="
            + str(float(self._mCenter - self._mWidth))
            + ",max="
            + str(float(self._mCenter + self._mWidth))
            + "]"
        )

    def toString(self) -> str:
        return str(self)


class _Unconstrained(Constraint):
    """Does not constrain the fit parameter in any way (identity mapping).

    Java: ``public class None implements Constraint`` — renamed to ``Unconstrained``
    to avoid shadowing the Python builtin.  ``toString()`` returns ``"Unconstrained[]"``
    exactly as the Java source does.
    """

    def __init__(self) -> None:
        pass

    # --- realToConstrained ---

    def realToConstrained_literal(self, param: float) -> float:
        return param

    def realToConstrained(self, param: float) -> float:  # SCIPY-NONE: no library substitute
        return self.realToConstrained_literal(param)

    # --- constrainedToReal ---

    def constrainedToReal_literal(self, param: float) -> float:
        return param

    def constrainedToReal(self, param: float) -> float:  # SCIPY-NONE: no library substitute
        return self.constrainedToReal_literal(param)

    # --- derivative ---

    def derivative_literal(self, param: float) -> float:
        return 1.0

    def derivative(self, param: float) -> float:  # SCIPY-NONE: no library substitute
        return self.derivative_literal(param)

    # --- getResult ---

    def getResult_literal(self, param: UncertainValue2) -> UncertainValue2:
        return param

    def getResult(self, param: UncertainValue2) -> UncertainValue2:  # SCIPY-NONE: no library substitute
        return self.getResult_literal(param)

    # --- toString / __str__ ---

    def __str__(self) -> str:
        return "Unconstrained[]"

    def toString(self) -> str:
        return str(self)


# Attach as nested class attributes to mirror Java's inner-class namespace
Constraint.Positive = _Positive  # type: ignore[attr-defined]
Constraint.Fractional = _Fractional  # type: ignore[attr-defined]
Constraint.Bounded = _Bounded  # type: ignore[attr-defined]
Constraint.Unconstrained = _Unconstrained  # type: ignore[attr-defined]

# Module-level aliases (R2)
Positive = _Positive
Fractional = _Fractional
Bounded = _Bounded
Unconstrained = _Unconstrained  # Java: Constraint.None — no `None` alias (builtin clash)
