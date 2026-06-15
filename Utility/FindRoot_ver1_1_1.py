"""
FindRoot_ver1_1_1.py -- Faithful Python port of
gov.nist.microanalysis.Utility.FindRoot.

CHANGES
-------
- Abstract class expressed with `abc.ABC` + `@abstractmethod` (R2).
- `IllegalArgumentException` → `ValueError` (closest Python stdlib equivalent).
- `ArithmeticException`      → `ArithmeticError`.
- `EvaluationCount()` return type kept as `float` to match Java's declared
  `public double` signature; internal `mNEvals` field remains `int`.

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
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class FindRoot(ABC):
    """Port of gov.nist.microanalysis.Utility.FindRoot.

    Jack Crenshaw's "world's best root finder" (bisection fused with
    inverse quadratic interpolation). Subclass and implement `function`.
    """

    def __init__(self) -> None:
        self.mBestX: float = 0.0
        self.mBestY: float = 0.0
        self.mNEvals: int = 0

    @abstractmethod
    def function(self, x0: float) -> float:
        """The function whose root is sought. Must be implemented by subclasses."""
        ...

    def initialize(self, vars: list[float]) -> None:
        """Override to perform any pre-`perform` initialisation. No-op by default."""
        pass

    def bestX(self) -> float:
        """Returns the x value closest to the root found so far."""
        return self.mBestX

    def bestY(self) -> float:
        """Returns the y value at bestX."""
        return self.mBestY

    def EvaluationCount(self) -> float:
        """Returns the number of function evaluations made by the last perform call."""
        return float(self.mNEvals)

    def perform(self, x0: float, x2: float, eps: float, iMax: int) -> float:
        """Find a root in [x0, x2] to within absolute error eps.

        Parameters
        ----------
        x0   : lower bound
        x2   : upper bound
        eps  : absolute convergence tolerance
        iMax : maximum iterations

        Raises
        ------
        ValueError
            If function(x0) * function(x2) > 0 (range does not straddle a root).
        ArithmeticError
            If the algorithm has not converged after iMax iterations.
        """
        xmlast: float = x0
        self.mNEvals = 1
        y0: float = self.function(x0)
        if y0 == 0.0:
            self.mBestX = x0
            self.mBestY = y0
            return x0
        y2: float = self.function(x2)
        self.mNEvals += 1
        if y2 == 0.0:
            self.mBestX = x2
            self.mBestY = y2
            return x2
        if (y2 * y0) > 0.0:
            raise ValueError(
                "Input range does not straddle a zero in FindRoot.perform()"
            )
        for _i in range(iMax):
            x1: float = 0.5 * (x2 + x0)
            y1: float = self.function(x1)
            self.mNEvals += 1
            if y1 == 0.0:
                return x1
            if abs(x1 - x0) < eps:
                self.mBestX = x1
                self.mBestY = y1
                return x1
            if (y1 * y0) > 0.0:
                x0, x2 = x2, x0
                y0, y2 = y2, y0
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
                self.mNEvals += 1
                if ym == 0.0:
                    return xm
                if abs(xm - xmlast) < eps:
                    self.mBestX = xm
                    self.mBestY = ym
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
            self.mBestX = x1
            self.mBestY = y1
        raise ArithmeticError(
            "FindRoot.perform has not converged after " + str(iMax) + " iterations."
        )
