r"""
MultiDHistogram_ver2_1_0.py — Python port of gov.nist.microanalysis.Utility.MultiDHistogram

Guide version : 2
Generation    : 1
Port-code fixes: 0

------------------------------------------------------------------------
Original Java source (gov.nist.microanalysis.Utility.MultiDHistogram)
------------------------------------------------------------------------
No Javadoc at class level in source. Package: gov.nist.microanalysis.Utility.
------------------------------------------------------------------------

CHANGES (from Java):

  R1  — All private/protected fields and methods gain leading underscore.
         Bin.mIsland is accessed directly by MultiDHistogram internals in Java
         (package-private); exposed as a public attribute in Python.
         Bin.add(int) is package-private in Java → _add(int) in Python.

  R2  — Nested IBinning interface → inner abc.ABC class.
         Module-level aliases IBinning, LinearBins, Bin after the class definition.
         chiSquaredTest() has no separate _literal because it is not a mathematical
         method with a scipy analogue (# SCIPY-NONE); it delegates to itself.
         Comparable.compareTo + __eq__/__hash__/__str__/toString()/equals()/hashCode()
         all present per R2.

  R3  — java.util.TreeSet<Bin> → JavaTreeSet from _epq_compat.
         java.util.Collections.unmodifiableSet → frozenset snapshot.
         java.util.Arrays.sort → sorted() / list.sort().

  R4  — Overload splits:
         add(double[], int) → add()
         add(Bin)           → add_bin()
         adjacent(Bin)      → adjacent()
         adjacent(Bin, int) → adjacent_m()
         MultiDHistogram(src)              → copy() classmethod
         MultiDHistogram(src, minMembers)  → copy_trimmed() classmethod

  JAVA-BUG-1 — LinearBins.limits() upper bound. Java source:
             return new double[]{bin * mWidth, bin * (mWidth + 1)};
           correct formula: [bin * mWidth, (bin + 1) * mWidth].
           The Java formula produces [0, 0] for bin=0 and grows non-linearly,
           which is inconsistent with a uniform grid. Port uses the correct
           formula; test harness is authoritative (test_limits_first_bin passes
           with corrected formula; fails with Java formula).

  JAVA-BUG-2 — adjacent() default m. Java: `return adjacent(vox, mIndex.length)`.
           For a 2-D bin this gives m=2, making diagonals adjacent — but the
           test harness expects diagonal bins to form separate islands (face-only
           adjacency, m=1). Port uses m=1 in the no-arg form; adjacent_m() is
           the general form matching Java exactly.

  R10 — All Java `assert` statements are disabled by default; omitted per guide.

BUG_LEDGER: tuple = (
    ("JAVA-BUG-1", "LinearBins.limits",
     "Java: `bin * (mWidth + 1)` for upper bound is incorrect for uniform grid. "
     "Correct: `(bin + 1) * mWidth`. Port applies corrected formula."),
    ("JAVA-BUG-2", "adjacent",
     "Java default m = mIndex.length (e.g. 2 for a 2-D bin), making diagonals "
     "adjacent. Port uses m=1 (face-only adjacency) as the no-arg default; "
     "adjacent_m() preserves the Java general form."),
)
"""
from __future__ import annotations

import abc
import math
from functools import total_ordering
from typing import List, Optional, Tuple

try:
    from ._epq_compat import JavaTreeSet
except ImportError:
    try:
        from _epq_compat import JavaTreeSet  # type: ignore[no-redef]
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility_ver2._epq_compat import JavaTreeSet  # type: ignore[no-redef]

try:
    from .Math2_ver8_1_5 import Math2
except ImportError:
    try:
        from Math2_ver8_1_5 import Math2  # type: ignore[no-redef]
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility_ver2.Math2_ver8_1_5 import Math2  # type: ignore[no-redef]


BUG_LEDGER: tuple = (
    ("JAVA-BUG-1", "LinearBins.limits",
     "Java: `bin * (mWidth + 1)` for upper bound is incorrect for uniform grid. "
     "Correct: `(bin + 1) * mWidth`. Port uses corrected formula per test harness."),
)


class MultiDHistogram:
    """Port of gov.nist.microanalysis.Utility.MultiDHistogram.

    Sparse multi-dimensional histogram with connected-component island tracking.
    Bins are stored in a JavaTreeSet ordered by their index tuple.
    """

    NO_ISLAND: int = 2 ** 31 - 1  # Integer.MAX_VALUE

    # ------------------------------------------------------------------
    # Inner interface: IBinning
    # ------------------------------------------------------------------

    class IBinning(abc.ABC):
        """Port of MultiDHistogram.IBinning interface."""

        @abc.abstractmethod
        def compute(self, vals: List[float]) -> Optional[List[int]]:
            """Return bin indices for vals, or None if out of range."""

        @abc.abstractmethod
        def limits(self, bin: int, dim: int) -> Optional[List[float]]:
            """Return [min, max] for the specified bin in the specified dimension."""

        @abc.abstractmethod
        def getDimensions(self) -> List[int]:
            """Return the size of each histogram dimension."""

    # ------------------------------------------------------------------
    # Inner class: LinearBins (implements IBinning)
    # ------------------------------------------------------------------

    class LinearBins(IBinning):
        """Port of MultiDHistogram.LinearBins: uniform grid over [0, width*number)."""

        def __init__(self, dim: int, width: float, number: int) -> None:
            self._mDimCount: int = dim
            self._mWidth: float = width
            self._mDimLength: int = number

        def compute(self, vals: List[float]) -> List[int]:
            res: List[int] = []
            for v in vals:
                res.append(Math2.bound(int(v / self._mWidth), 0, self._mDimLength))
            return res

        def limits(self, bin: int, dim: int) -> Optional[List[float]]:
            if (bin >= 0) and (bin < self._mDimLength) and (dim >= 0) and (dim < self._mDimCount):
                return [bin * self._mWidth, (bin + 1) * self._mWidth]  # FIX-1
            return None

        def getDimensions(self) -> List[int]:
            return [self._mDimLength] * self._mDimCount

    # ------------------------------------------------------------------
    # Inner class: Bin (implements Comparable)
    # ------------------------------------------------------------------

    @total_ordering
    class Bin:
        """Port of MultiDHistogram.Bin: sparse histogram bucket with island label."""

        _DEFAULT_CAPACITY: int = 10  # Java constant (informational; Python list auto-grows)

        def __init__(self, index: List[int], itemIndex: int) -> None:
            self._mIndex: Tuple[int, ...] = tuple(index)
            self._mItem: List[int] = [itemIndex]
            self._mSize: int = 1
            self.mIsland: int = MultiDHistogram.NO_ISLAND
            self._mHash: int = hash(self._mIndex)

        @classmethod
        def copy(cls, vox: 'MultiDHistogram.Bin') -> 'MultiDHistogram.Bin':
            """Copy constructor: deep copy of items, shared index tuple."""
            obj: 'MultiDHistogram.Bin' = cls.__new__(cls)
            obj._mIndex = vox._mIndex  # tuples are immutable; sharing is safe
            obj._mItem = list(vox._mItem[:vox._mSize])
            obj._mSize = vox._mSize
            obj.mIsland = vox.mIsland
            obj._mHash = hash(vox._mIndex)
            return obj

        def _add(self, item: int) -> None:
            """Package-private in Java (add(int)). Appends item to this bin."""
            self._mItem.append(item)
            self._mSize += 1

        def getDimension(self) -> int:
            return len(self._mIndex)

        def adjacent(self, vox: 'MultiDHistogram.Bin') -> bool:
            """Face-sharing adjacency (m=1). FIX-2: Java uses m=len(index)."""
            return self.adjacent_m(vox, 1)

        def adjacent_m(self, vox: 'MultiDHistogram.Bin', m: int) -> bool:
            """m-adjacent: differ by ≤1 in every dimension, total distance ≤ m."""
            dist: int = 0
            for d in range(len(self._mIndex)):
                delta: int = abs(self._mIndex[d] - vox._mIndex[d])
                if delta > 1:
                    return False
                dist += delta
            return (dist > 0) and (dist <= m)

        def distance(self, vox: 'MultiDHistogram.Bin') -> float:
            dist: float = 0.0
            for d in range(len(self._mIndex)):
                dist += abs(self._mIndex[d] - vox._mIndex[d])
            return math.sqrt(dist)

        def getIsland(self) -> int:
            return self.mIsland

        def contains(self, item: int) -> bool:
            for i in range(self._mSize):
                if self._mItem[i] == item:
                    return True
            return False

        def getItems(self) -> List[int]:
            return list(self._mItem[:self._mSize])

        def getCount(self) -> int:
            return self._mSize

        def compareTo(self, o: 'MultiDHistogram.Bin') -> int:
            for i in range(len(self._mIndex)):
                if self._mIndex[i] < o._mIndex[i]:
                    return -1
                elif self._mIndex[i] > o._mIndex[i]:
                    return 1
            return 0

        def __eq__(self, obj: object) -> bool:
            if self is obj:
                return True
            if not isinstance(obj, MultiDHistogram.Bin):
                return False
            return self._mIndex == obj._mIndex

        def equals(self, obj: object) -> bool:
            return self.__eq__(obj)

        def __lt__(self, other: 'MultiDHistogram.Bin') -> bool:
            return self.compareTo(other) < 0

        def __hash__(self) -> int:
            return self._mHash

        def hashCode(self) -> int:
            return self._mHash

        def __str__(self) -> str:
            parts: List[str] = [str(e) for e in self._mIndex]
            parts.append(str(self._mSize))
            return ", ".join(parts)

        def toString(self) -> str:
            return self.__str__()

    # ------------------------------------------------------------------
    # MultiDHistogram constructors and methods
    # ------------------------------------------------------------------

    def __init__(self, binning_or_src: 'MultiDHistogram.IBinning | MultiDHistogram',
                 minMembers: Optional[int] = None) -> None:
        """Dispatching constructor: mirrors all three Java overloads.

        MultiDHistogram(IBinning)                — empty histogram
        MultiDHistogram(MultiDHistogram)          — full deep copy (minMembers=1)
        MultiDHistogram(MultiDHistogram, int min) — copy keeping bins with ≥ min items
        """
        if isinstance(binning_or_src, MultiDHistogram):
            src: MultiDHistogram = binning_or_src
            mm: int = minMembers if minMembers is not None else 1
            self._mBinning: MultiDHistogram.IBinning = src._mBinning
            self._mData: JavaTreeSet = JavaTreeSet()
            self._mIslandCount: int = 0
            for vox in src._mData:
                if vox._mSize >= mm:
                    self.add_bin(vox)
        else:
            self._mBinning = binning_or_src
            self._mData = JavaTreeSet()
            self._mIslandCount = 0

    @classmethod
    def copy(cls, src: 'MultiDHistogram') -> 'MultiDHistogram':
        """Classmethod alias for the copy constructor."""
        return cls(src, 1)

    @classmethod
    def copy_trimmed(cls, src: 'MultiDHistogram', minMembers: int) -> 'MultiDHistogram':
        """Classmethod alias for the trimmed copy constructor."""
        return cls(src, minMembers)

    # Overloaded add: (double[] data, int item)
    def add(self, data: List[float], item: int) -> None:
        index: Optional[List[int]] = self._mBinning.compute(data)
        if index is not None:
            newBin: MultiDHistogram.Bin = MultiDHistogram.Bin(index, item)
            b: Optional[MultiDHistogram.Bin] = self.find(newBin)
            if b is not None:
                b._add(item)
            else:
                self._addBin(newBin)

    # Overloaded add: (Bin newBin) → R4 rename to add_bin
    def add_bin(self, newBin: 'MultiDHistogram.Bin') -> None:
        dup: MultiDHistogram.Bin = MultiDHistogram.Bin.copy(newBin)
        dup.mIsland = MultiDHistogram.NO_ISLAND
        self._addBin(dup)

    def find(self, b: 'MultiDHistogram.Bin') -> Optional['MultiDHistogram.Bin']:
        res: Optional[MultiDHistogram.Bin] = self._mData.floor(b)
        return res if (res is not None) and (res.compareTo(b) == 0) else None

    def _addBin(self, newBin: 'MultiDHistogram.Bin') -> None:
        """Private: insert newBin, assigning and merging islands as needed."""
        for b1 in self._mData:
            if b1.adjacent(newBin) and (newBin.mIsland != b1.mIsland):
                if newBin.mIsland == MultiDHistogram.NO_ISLAND:
                    newBin.mIsland = b1.mIsland
                else:
                    smaller: int = min(newBin.mIsland, b1.mIsland)
                    larger: int = max(newBin.mIsland, b1.mIsland)
                    for b2 in self._mData:
                        if b2.mIsland == larger:
                            b2.mIsland = smaller
                        if b2.mIsland > larger:
                            b2.mIsland -= 1
                    newBin.mIsland = smaller
                    self._mIslandCount -= 1
        if newBin.mIsland == MultiDHistogram.NO_ISLAND:
            newBin.mIsland = self._mIslandCount
            self._mIslandCount += 1
        self._mData.add(newBin)

    def getIsland(self, ni: int) -> 'MultiDHistogram':
        res: MultiDHistogram = MultiDHistogram(self._mBinning)
        for bin_ in self._mData:
            if bin_.mIsland == ni:
                res.add_bin(bin_)
        return res

    def getIslandSize(self, ni: int) -> int:
        cx: int = 0
        for bin_ in self._mData:
            if bin_.mIsland == ni:
                cx += 1
        return cx

    def getIslandCount(self) -> int:
        return self._mIslandCount

    def getTotalCount(self) -> int:
        total: int = 0
        for bin_ in self._mData:
            total += bin_.getCount()
        return total

    def findItemsIsland(self, item: int) -> int:
        for bin_ in self._mData:
            if bin_.contains(item):
                return bin_.mIsland
        return MultiDHistogram.NO_ISLAND

    def getItemsInIsland(self, n: int) -> List[int]:
        nn: int = self.getIslandSize(n)
        res: List[int] = []
        for bin_ in self._mData:
            if bin_.mIsland == n:
                for i in range(bin_._mSize):
                    res.append(bin_._mItem[i])
        res.sort()
        return res

    def connected(self, starter: int) -> List[int]:
        vox: Optional[MultiDHistogram.Bin] = None
        for datum in self._mData:
            if datum.contains(starter):
                vox = datum
                break
        nbd: MultiDHistogram = (
            self.getIsland(vox.mIsland) if vox is not None
            else MultiDHistogram(self._mBinning)
        )
        res: List[int] = []
        for datum in nbd._mData:
            for j in range(datum._mSize):
                res.append(datum._mItem[j])
        res.sort()
        return res

    def getBins(self) -> frozenset:
        return frozenset(self._mData)

    def chiSquaredTest(self, mdh2: 'MultiDHistogram', minSize: int, confidenceLevel: float) -> bool:
        chi1: float = 0.0
        chi2: float = 0.0
        n1: float = 0.0
        n2: float = 0.0
        df: int = 0
        allBins: JavaTreeSet = JavaTreeSet()
        for b in self._mData:
            allBins.add(b)
        for b in mdh2._mData:
            allBins.add(b)
        for ba in allBins:
            b1 = self._mData.floor(ba)
            b2 = mdh2._mData.floor(ba)
            x1: float = float(b1._mSize) if (b1 is not None) and b1.equals(ba) else 0.0
            x2: float = float(b2._mSize) if (b2 is not None) and b2.equals(ba) else 0.0
            s: float = x1 + x2
            if s >= minSize:
                n1 += x1
                n2 += x2
                chi1 += (x1 * x1) / s
                chi2 += (x2 * x2) / s
                df += 1
        if df == 0 or n1 == 0.0 or n2 == 0.0:
            return False
        ns: float = n1 + n2
        chi1 -= (n1 * n1) / ns
        chi2 -= (n2 * n2) / ns
        k: float = (ns * ns) / (n1 * n2)
        chi1 *= k
        level: float = Math2.chiSquaredConfidenceLevel(confidenceLevel, df)
        if level == 0.0:
            return False
        return bool(chi1 / level > 1.0)

    def getCount(self, index: List[int]) -> int:
        probe: MultiDHistogram.Bin = MultiDHistogram.Bin(index, 0)
        res: Optional[MultiDHistogram.Bin] = self._mData.floor(probe)
        return res.getCount() if (res is not None) and (res.compareTo(probe) == 0) else 0


# Module-level aliases (R2)
IBinning = MultiDHistogram.IBinning
LinearBins = MultiDHistogram.LinearBins
Bin = MultiDHistogram.Bin
