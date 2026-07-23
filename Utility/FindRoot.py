r"""
FindRoot_ver2_1_0.py — Python port of gov.nist.microanalysis.Utility.FindRoot

Guide version : 2
Generation    : 1
Port-code fixes: 0

------------------------------------------------------------------------
Original Java source (gov.nist.microanalysis.Utility.FindRoot)
------------------------------------------------------------------------
/**
 * <p>
 * A simple root finder based on Jack Crenshaw's "world's best root finder."
 * Derive a class from this that implements function (and optionally also
 * initialize). Anonymous classes work nicely for this.
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

CHANGES:
* IllegalArgumentException -> ValueError; ArithmeticException -> ArithmeticError
  (Java stdlib exceptions mapped to their nearest Python equivalents, per spec).
* EvaluationCount() casts the int field to float to mirror Java's int->double
  widening on a `double`-declared accessor (R9).
"""

from __future__ import annotations

import abc

try:
    from ._epq_compat import F64Array
except ImportError:
    try:
        from _epq_compat import F64Array  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility._epq_compat import F64Array  # type: ignore

__all__ = ["FindRoot"]


class FindRoot(abc.ABC):

    BUG_LEDGER: tuple = ()  # no bugs identified

    def __init__(self) -> None:
        self._mBestX: float = 0.0
        self._mBestY: float = 0.0
        self._mNEvals: int = 0

    @abc.abstractmethod
    def function(self, x0: float) -> float:
        """The function for which the root will be found. Implement in a subclass."""
        ...

    def initialize(self, vars: F64Array) -> None:
        """Override to perform any required initialization. No-op by default."""
        pass

    def bestX(self) -> float:
        """Returns the x value closest to the root."""
        return self._mBestX

    def bestY(self) -> float:
        """Returns the y value associated with the best x value."""
        return self._mBestY

    def EvaluationCount(self) -> float:
        # R9: Java declares this `double` but returns the int field mNEvals.
        return float(self._mNEvals)

    def perform(self, x0: float, x2: float, eps: float, iMax: int) -> float:
        """Implements the Crenshaw root-find algorithm.

        Raises ValueError if function(x0)*function(x2) > 0.0 (range does not
        straddle a zero), and ArithmeticError if not converged after iMax
        iterations.
        """
        xmlast: float = x0
        self._mNEvals = 1
        y0: float = self.function(x0)
        if y0 == 0.0:
            self._mBestX = x0
            self._mBestY = y0
            return x0
        y2: float = self.function(x2)
        self._mNEvals += 1
        if y2 == 0.0:
            self._mBestX = x2
            self._mBestY = y2
            return x2
        if (y2 * y0) > 0.0:
            raise ValueError(
                "Input range does not straddle a zero in FindRoot.perform()"
            )
        for i in range(iMax):
            x1: float = 0.5 * (x2 + x0)
            y1: float = self.function(x1)
            self._mNEvals += 1
            if y1 == 0.0:
                return x1
            if abs(x1 - x0) < eps:
                self._mBestX = x1
                self._mBestY = y1
                return x1
            if (y1 * y0) > 0.0:
                temp: float = x0
                x0 = x2
                x2 = temp
                temp = y0
                y0 = y2
                y2 = temp
            y10: float = y1 - y0
            y21: float = y2 - y1
            y20: float = y2 - y0
            if (y2 * y20) < (2.0 * y1 * y10):
                x2 = x1
                y2 = y1
            else:
                b: float = (x1 - x0) / y10
                c: float = (y10 - y21) / (y21 * y20)
                xm: float = x0 - (b * y0 * (1.0 - (c * y1)))
                ym: float = self.function(xm)
                self._mNEvals += 1
                if ym == 0.0:
                    return xm
                if abs(xm - xmlast) < eps:
                    self._mBestX = xm
                    self._mBestY = ym
                    return xm
                xmlast = xm
                if (ym * y0) < 0.0:
                    x2 = xm
                    y2 = ym
                else:
                    x0 = xm
                    y0 = ym
                    x2 = x1
                    y2 = y1
            self._mBestX = x1
            self._mBestY = y1
        raise ArithmeticError(
            f"FindRoot.perform has not converged after {iMax} iterations."
        )
