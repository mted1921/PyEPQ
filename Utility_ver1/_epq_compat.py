"""
_epq_compat.py
==============

Single source of truth for cross-module types in the EPQ Python port.

Every converted module MUST import these from here. Defining local
stand-in classes (as the initial Math2_ver1 revision did) creates
incompatible types: ``except EPQException`` written against module A's
stand-in does not catch ``raise EPQException()`` from module B.

Exported types
--------------
EPQException  : Exception subclass, port of the Java EPQException.
JavaRandom    : Bit-exact reimplementation of java.util.Random.
JavaTreeSet   : Sorted-set replacement for java.util.TreeSet.
JamaMatrix    : Minimal numpy-backed shim for Jama.Matrix.
F64Array      : Type alias for NDArray[np.float64].

This file has no project-internal dependencies; it is safe to import
from any other converted module without circular-import risk.
"""

from __future__ import annotations

import bisect
import math
import time
from typing import Any, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray


__all__ = ["EPQException", "JavaRandom", "JavaTreeSet", "JamaMatrix", "F64Array"]


F64Array = NDArray[np.float64]


# ======================================================================
# EPQException
# ======================================================================

class EPQException(Exception):
    """Port of gov.nist.microanalysis.EPQLibrary.EPQException.

    This is the single source of truth for the EPQException type during
    the migration. When the dedicated EPQException port lands, replace
    the body of this class but keep the import path stable so call sites
    do not need to change.
    """
    pass


# ======================================================================
# JavaRandom
# ======================================================================

class JavaRandom:
    """Bit-exact reimplementation of java.util.Random.

    A given seed produces the same sequence as ``new Random(seed)`` on
    the JVM. Required for:
      * Parity testing the Python port against reference Java outputs.
      * Reproducing Monte Carlo runs cited in publications.
      * Any test that asserts specific RNG-dependent values.

    Python's ``random.Random`` (Mersenne Twister) is statistically
    excellent but produces a *different* stream for the same seed --
    unsuitable as a drop-in for parity work.

    Algorithm (per JDK source):
        scramble:  s' = (seed ^ MULT) & MASK
        step:      s' = (s * MULT + INCR) & MASK
        next(b):   top b bits of s', sign-extended when b == 32.

    Public method names follow Java conventions so call sites translate
    verbatim; ``random()`` is provided as a Python-convention alias for
    ``nextDouble()``.
    """

    _MULTIPLIER: int = 0x5DEECE66D
    _INCREMENT: int = 0xB
    _MASK: int = (1 << 48) - 1

    def __init__(self, seed: Optional[int] = None) -> None:
        if seed is None:
            # Java default: nanoTime XOR'd with a "uniquifier". For
            # parity-critical work, always pass an explicit seed.
            seed = time.time_ns()
        self._seed: int = (int(seed) ^ self._MULTIPLIER) & self._MASK
        self._have_next_gaussian: bool = False
        self._next_gaussian: float = 0.0

    def setSeed(self, seed: int) -> None:
        self._seed = (int(seed) ^ self._MULTIPLIER) & self._MASK
        self._have_next_gaussian = False

    def _next(self, bits: int) -> int:
        """Internal LCG step returning the top `bits` bits.

        Java's protected ``next(int)`` returns a *signed* 32-bit int.
        We return the unsigned value and let public methods sign-extend
        as needed (see ``nextInt`` no-arg form).
        """
        self._seed = (self._seed * self._MULTIPLIER + self._INCREMENT) & self._MASK
        return self._seed >> (48 - bits)

    def nextDouble(self) -> float:
        # Java: ((next(26) << 27) + next(27)) / 2^53
        return ((self._next(26) << 27) + self._next(27)) / (1 << 53)

    def random(self) -> float:
        """Python-convention alias for nextDouble()."""
        return self.nextDouble()

    def nextFloat(self) -> float:
        return self._next(24) / (1 << 24)

    def nextInt(self, bound: Optional[int] = None) -> int:
        """One-arg form: uniform in [0, bound).
        No-arg form: uniform signed 32-bit int.

        The bounded form implements Java's exact rejection-sampling
        algorithm including the power-of-two fast path -- this affects
        which seeds map to which outputs.
        """
        if bound is None:
            r = self._next(32)
            return r - (1 << 32) if r >= (1 << 31) else r  # sign-extend
        if bound <= 0:
            raise ValueError("bound must be positive")
        m = bound - 1
        if (bound & m) == 0:  # power of two
            return (bound * self._next(31)) >> 31
        while True:
            r = self._next(31)
            v = r % bound
            if r - v + m >= 0:
                return v

    def nextLong(self) -> int:
        r = (self._next(32) << 32) + self._next(32)
        return r - (1 << 64) if r >= (1 << 63) else r  # sign-extend

    def nextBoolean(self) -> bool:
        return self._next(1) != 0

    def nextGaussian(self) -> float:
        """Marsaglia polar method, matching Java's java.util.Random."""
        if self._have_next_gaussian:
            self._have_next_gaussian = False
            return self._next_gaussian
        while True:
            v1 = 2.0 * self.nextDouble() - 1.0
            v2 = 2.0 * self.nextDouble() - 1.0
            s = v1 * v1 + v2 * v2
            if 0.0 < s < 1.0:
                break
        mult = math.sqrt(-2.0 * math.log(s) / s)
        self._next_gaussian = v2 * mult
        self._have_next_gaussian = True
        return v1 * mult


# ======================================================================
# JavaTreeSet
# ======================================================================

class JavaTreeSet:
    """Replacement for java.util.TreeSet<T> where T implements Comparable.

    Provides the exact subset of TreeSet's API used by EPQ converted classes:
    - ``add(item)``   — insert maintaining sorted order; returns False if duplicate
    - ``floor(item)`` — greatest element ≤ item, or None
    - ``__iter__``    — iterate in ascending order
    - ``__len__``     — element count

    **Requirements on stored elements:**
    Elements must implement both:
    - ``__lt__`` (used by bisect for insertion position)
    - ``compareTo(other) -> int`` (returns -1 / 0 / +1, used for equality in add/floor)

    These are the Python equivalents of Java's ``Comparable<T>`` contract. Any
    correctly ported Java class that ``implements Comparable`` will satisfy them
    via the R2 ``compareTo`` → ``__lt__``/``compareTo()`` mapping.

    Unlike Python's ``SortedList`` (sortedcontainers), this class requires no
    third-party dependency and implements floor() directly, matching Java's
    NavigableSet.floor() semantics.
    """

    def __init__(self) -> None:
        self._list: list[Any] = []

    def add(self, item: Any) -> bool:
        """Insert item in sorted order. Returns False (no-op) if already present."""
        idx: int = bisect.bisect_left(self._list, item)
        if idx < len(self._list) and self._list[idx].compareTo(item) == 0:
            return False
        self._list.insert(idx, item)
        return True

    def floor(self, item: Any) -> Optional[Any]:
        """Return the greatest element ≤ item, or None if no such element exists."""
        idx: int = bisect.bisect_right(self._list, item)
        if idx == 0:
            return None
        return self._list[idx - 1]

    def __iter__(self):
        return iter(self._list)

    def __len__(self) -> int:
        return len(self._list)

    def __repr__(self) -> str:
        return f"JavaTreeSet({self._list!r})"


# ======================================================================
# JamaMatrix
# ======================================================================

class JamaMatrix:
    """Minimal Jama-compatible matrix wrapper around numpy.

    Implements the subset of Jama's API actually used by EPQ. Exotic
    decompositions (SVD, eig, Cholesky) intentionally are NOT wrapped --
    use scipy.linalg directly; the result types are well-documented and
    there is no value in re-boxing them.

    Storage is numpy float64. ``getArray()`` exposes the underlying
    ndarray for native interop (caller mutations propagate to the
    matrix). Use ``getArrayCopy()`` if isolation is required.
    """

    def __init__(self, data: ArrayLike) -> None:
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise ValueError("JamaMatrix requires 1-D or 2-D input")
        self._A: NDArray[np.float64] = arr

    # ---- factory methods (Jama's alternate constructors) ----

    @classmethod
    def from_flat(cls, vals: ArrayLike, m: int) -> "JamaMatrix":
        """Java: ``new Matrix(double[] vals, int m)`` -- column-packed flat array."""
        arr = np.asarray(vals, dtype=np.float64)
        n = arr.shape[0] // m
        return cls(arr.reshape(n, m).T)

    @classmethod
    def zeros(cls, m: int, n: int) -> "JamaMatrix":
        return cls(np.zeros((m, n), dtype=np.float64))

    @classmethod
    def filled(cls, m: int, n: int, s: float) -> "JamaMatrix":
        return cls(np.full((m, n), float(s), dtype=np.float64))

    @classmethod
    def identity(cls, m: int, n: Optional[int] = None) -> "JamaMatrix":
        return cls(np.eye(m, n if n is not None else m, dtype=np.float64))

    @classmethod
    def random(cls, m: int, n: int,
               rng: Optional["JavaRandom"] = None) -> "JamaMatrix":
        """Jama's random matrix. Pass a JavaRandom for seed parity."""
        if rng is None:
            return cls(np.random.rand(m, n))
        out = np.empty((m, n), dtype=np.float64)
        for i in range(m):
            for j in range(n):
                out[i, j] = rng.nextDouble()
        return cls(out)

    # ---- size and access ----

    def getRowDimension(self) -> int:
        return self._A.shape[0]

    def getColumnDimension(self) -> int:
        return self._A.shape[1]

    def get(self, i: int, j: int) -> float:
        return float(self._A[i, j])

    def set(self, i: int, j: int, val: float) -> None:
        self._A[i, j] = val

    def getArray(self) -> NDArray[np.float64]:
        return self._A

    def getArrayCopy(self) -> NDArray[np.float64]:
        return self._A.copy()

    def getMatrix(self, i0: int, i1: int, j0: int, j1: int) -> "JamaMatrix":
        """Submatrix. Jama is *inclusive* on both endpoints (Python slice +1)."""
        return JamaMatrix(self._A[i0:i1 + 1, j0:j1 + 1].copy())

    # ---- arithmetic ----

    def plus(self, other: "JamaMatrix") -> "JamaMatrix":
        return JamaMatrix(self._A + other._A)

    def plusEquals(self, other: "JamaMatrix") -> "JamaMatrix":
        self._A += other._A
        return self

    def minus(self, other: "JamaMatrix") -> "JamaMatrix":
        return JamaMatrix(self._A - other._A)

    def minusEquals(self, other: "JamaMatrix") -> "JamaMatrix":
        self._A -= other._A
        return self

    def times(self, other: Union["JamaMatrix", float]) -> "JamaMatrix":
        if isinstance(other, JamaMatrix):
            return JamaMatrix(self._A @ other._A)
        return JamaMatrix(self._A * float(other))

    def timesEquals(self, scalar: float) -> "JamaMatrix":
        self._A *= float(scalar)
        return self

    def transpose(self) -> "JamaMatrix":
        return JamaMatrix(self._A.T.copy())

    def inverse(self) -> "JamaMatrix":
        return JamaMatrix(np.linalg.inv(self._A))

    def solve(self, B: "JamaMatrix") -> "JamaMatrix":
        """Solve A X = B. Square -> direct solve; rectangular -> lstsq (matches Jama)."""
        if self._A.shape[0] == self._A.shape[1]:
            return JamaMatrix(np.linalg.solve(self._A, B._A))
        return JamaMatrix(np.linalg.lstsq(self._A, B._A, rcond=None)[0])

    def det(self) -> float:
        return float(np.linalg.det(self._A))

    def trace(self) -> float:
        return float(np.trace(self._A))

    def norm1(self) -> float:
        return float(np.linalg.norm(self._A, 1))

    def norm2(self) -> float:
        return float(np.linalg.norm(self._A, 2))

    def normInf(self) -> float:
        return float(np.linalg.norm(self._A, np.inf))

    def normF(self) -> float:
        return float(np.linalg.norm(self._A, "fro"))

    # ---- conveniences ----

    def __repr__(self) -> str:
        return f"JamaMatrix(shape={self._A.shape})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, JamaMatrix) and np.array_equal(self._A, other._A)
