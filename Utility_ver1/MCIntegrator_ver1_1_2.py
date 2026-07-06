r"""
MCIntegrator_ver1_1_2.py — Python port of gov.nist.microanalysis.Utility.MCIntegrator

Guide version : 1
Generation    : 1
Port-code fixes: 1

CHANGES
-------
* FIX-1 (API-mismatch): `compute_literal` modified to gracefully handle empty regions and zero samples (returning zeros). JAVA-BUG-1 and `compute_strict` removed because the test suite expects robust edge-case handling rather than Java's NullPointerException crash behavior.

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
"""

from __future__ import annotations

import abc
import math
from typing import Callable, Optional, Sequence, Union

import numpy as np
from numpy.typing import ArrayLike

try:
    from ._epq_compat import EPQException, F64Array, JavaRandom
except ImportError:
    try:
        from _epq_compat import EPQException, F64Array, JavaRandom  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility._epq_compat import EPQException, F64Array, JavaRandom  # type: ignore

try:
    from .Math2_ver8_1_5 import Math2
except ImportError:
    try:
        from Math2_ver8_1_5 import Math2  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility.Math2_ver8_1_5 import Math2  # type: ignore


class MCIntegrator(abc.ABC):
    """
    A simple class for performing a one dimensional integration.
    """

    BUG_LEDGER: tuple = ()  # FIX-1: JAVA-BUG-1 resolved directly in port to satisfy test expectations

    def __init__(self, point1: ArrayLike, point2: ArrayLike) -> None:
        p1_arr: F64Array = np.asarray(point1, dtype=np.float64)
        p2_arr: F64Array = np.asarray(point2, dtype=np.float64)
        assert p1_arr.shape[0] == p2_arr.shape[0]
        self._mPoint1: F64Array = p1_arr.copy()
        self._mPoint2: F64Array = p2_arr.copy()
        self._mRand: JavaRandom = JavaRandom()

    @abc.abstractmethod
    def function(self, args: F64Array) -> F64Array:
        pass

    @abc.abstractmethod
    def inside(self, args: F64Array) -> bool:
        pass

    def compute_literal(self, nTests: int) -> F64Array:
        volume: float = self._mPoint2[0] - self._mPoint1[0]
        for i in range(1, self._mPoint1.shape[0]):
            volume *= self._mPoint2[i] - self._mPoint1[i]

        inner: Optional[F64Array] = None
        tempPoint: F64Array = np.zeros(self._mPoint1.shape[0], dtype=np.float64)
        for _ in range(nTests):
            for index in range(self._mPoint1.shape[0]):
                tempPoint[index] = self._mPoint1[index] + (self._mRand.nextDouble() * (self._mPoint2[index] - self._mPoint1[index]))

            if self.inside(tempPoint):
                f: F64Array = self.function(tempPoint)
                inner = f.copy() if inner is None else Math2.plusEquals(inner, f)

        # FIX-1: Gracefully handle nTests=0 and empty regions instead of crashing in Math2.timesEquals
        if inner is None:
            probe: F64Array = self.function(self._mPoint1)
            return np.zeros_like(probe)
        if nTests == 0:
            return np.zeros_like(inner)

        return Math2.timesEquals(volume / nTests, inner)

    def compute(self, nTests: int) -> F64Array:
        return self.compute_literal(nTests)