r"""
MCIntegrator_ver2_1_0.py — Python port of gov.nist.microanalysis.Utility.MCIntegrator

Guide version : 2
Generation    : 1
Port-code fixes: 0

------------------------------------------------------------------------
Original Java source (gov.nist.microanalysis.Utility.MCIntegrator)
------------------------------------------------------------------------
/**
 * <p>
 * A simple class for performing a one dimensional integration.
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

CHANGES (from Java):
  R10-1 — Empty-domain guard: Java `Math2.timesEquals(scalar, null)` would throw
           NullPointerException when no sample lands inside the integration volume.
           Python port calls function() once with the first corner point to probe
           the output dimension, then returns np.zeros(dim).  This is also the
           correct behaviour for nTests=0.
  R2    — compute() is the primary entry point (# SCIPY-NONE: Monte Carlo with
           Java-matching Random is the only available algorithm); compute_literal()
           is the line-for-line Java translation.
  R1    — Abstract methods `function` and `inside` use un-prefixed Java names
           (abstract does NOT add leading underscore per R1).
  R3    — java.util.Random (unseeded) replaced by Python random.Random().
           Parity tests are M4-skipped (JPype cannot extend abstract Java classes),
           so JavaRandom is not required here.
  R9    — F64Array (numpy float64) used for double[] throughout compute_literal().
           Abstract method signatures accept any Sequence[float] or F64Array from
           subclasses; results are coerced to F64Array inside compute_literal().

BUG_LEDGER: tuple = ()  # no bugs identified
"""
from __future__ import annotations

import abc
import random as _random_module
from typing import List, Optional, Sequence

import numpy as np

try:
    from ._epq_compat import F64Array
except ImportError:
    try:
        from _epq_compat import F64Array  # type: ignore[no-redef]
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility_ver2._epq_compat import F64Array  # type: ignore[no-redef]

try:
    from .Math2_ver8_1_5 import Math2
except ImportError:
    try:
        from Math2_ver8_1_5 import Math2  # type: ignore[no-redef]
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility_ver2.Math2_ver8_1_5 import Math2  # type: ignore[no-redef]


BUG_LEDGER: tuple = ()  # no bugs identified


class MCIntegrator(abc.ABC):
    """Port of gov.nist.microanalysis.Utility.MCIntegrator.

    Abstract base for Monte Carlo integration over a bounded region.
    Subclass and implement function() and inside(); then call compute().
    """

    def __init__(self, point1: List[float], point2: List[float]) -> None:
        # Java assert (point1.length == point2.length) — disabled by default; omitted per guide.
        self._mPoint1: List[float] = list(point1)
        self._mPoint2: List[float] = list(point2)
        self._mRand: _random_module.Random = _random_module.Random()

    @abc.abstractmethod
    def function(self, args: F64Array) -> F64Array:
        """User-supplied integrand.  Returns function value(s) at args."""

    @abc.abstractmethod
    def inside(self, args: F64Array) -> bool:
        """Returns True if args is within the integration volume."""

    def compute(self, nTests: int) -> F64Array:
        # SCIPY-NONE: no library substitute for Java-parity Monte Carlo integration
        return self.compute_literal(nTests)

    def compute_literal(self, nTests: int) -> F64Array:
        """Line-for-line translation of MCIntegrator.compute()."""
        volume: float = self._mPoint2[0] - self._mPoint1[0]
        for i in range(1, len(self._mPoint1)):
            volume *= self._mPoint2[i] - self._mPoint1[i]

        inner: Optional[F64Array] = None
        tempPoint: F64Array = np.array(self._mPoint1, dtype=np.float64)
        for _ in range(nTests):
            for index in range(len(self._mPoint1)):
                tempPoint[index] = self._mPoint1[index] + (
                    self._mRand.random() * (self._mPoint2[index] - self._mPoint1[index])
                )
            if self.inside(tempPoint):
                f: F64Array = np.asarray(self.function(tempPoint), dtype=np.float64).copy()
                if inner is None:
                    inner = f
                else:
                    inner = Math2.plusEquals(inner, f)

        if inner is None:
            # R10-1: empty domain or nTests=0 — probe output dimension via function().
            probe: F64Array = np.asarray(self.function(tempPoint), dtype=np.float64)
            return np.zeros(len(probe), dtype=np.float64)

        return Math2.timesEquals(volume / nTests, inner)
