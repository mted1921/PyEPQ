r"""
MultiDHistogram_ver1_1_0.py — Python port of gov.nist.microanalysis.Utility.MultiDHistogram

Guide version : 1
Generation    : 1
Port-code fixes: 8

CHANGES:
* FIX-1 (R1/R4): LinearBins constructor was `(int dim, float width, int number)` but Java
  constructor is `(double[] lo, double[] hi, int[] nBins)`. Fixed to three-array signature.
* FIX-2 (R2): LinearBins.compute_literal rewritten to use per-dimension range calculation
  matching the Java implementation.
* FIX-3 (R1): LinearBins.dimensions() method missing; port had getDimensions() with wrong
  return type (Sequence[int] instead of int). Added dimensions()→int matching Java IBinning
  interface; getDimensions() retained as utility returning per-dimension bin counts.
* FIX-4 (R4): MultiDHistogram.__init__ now dispatches on argument type, handling the Java
  copy constructors MultiDHistogram(MultiDHistogram) and MultiDHistogram(MultiDHistogram,int).
* FIX-5 (R1): Public method add_data renamed to add (Java method is public void add(double[],int)).
  add_data kept as backward-compatibility alias.
* FIX-6 (R1): Bin.adjacent_bin passed m=len(mIndex) to adjacent_bin_m, making diagonal
  neighbours adjacent. Java uses face-sharing adjacency (m=1). Fixed to pass m=1.
* FIX-7: __init__ copy path called _addBin(from_bin(vox)) directly. from_bin preserves
  mIsland; _addBin asserts mIsland==NO_ISLAND → AssertionError. Fixed to call add_bin()
  which already resets mIsland before delegating to _addBin.
* FIX-8: getItemsInIsland allocated result buffer using getIslandSize() (bin count), not
  item count. Bins with multiple items overflowed the buffer → IndexError. Fixed to sum
  bin_obj.mSize across bins in the island.
* FIX-9 (R2): chiSquaredTest_literal returned float ratio; Java method returns boolean
  (chi1 >= chiSquaredConfidenceLevel). Fixed to return bool.
* TreeSet replacement: Python lacks a built-in TreeSet. The Java `TreeSet<Bin> mData` is
  reproduced using a private `_TreeSet` class built on `bisect` and a sorted list,
  providing O(log N) inserts and the exact `.floor()` semantics used in the Java class.

------------------------------------------------------------------------
Original Java source (gov.nist.microanalysis.Utility.MultiDHistogram)
------------------------------------------------------------------------
/**
 * (No class-level Javadoc in original source)
 */
------------------------------------------------------------------------
"""

from __future__ import annotations
import abc
import math
import numpy as np
from typing import Optional, Sequence, Union

try:
    from ._epq_compat import EPQException, F64Array, JavaRandom, JavaTreeSet
except ImportError:
    try:
        from _epq_compat import EPQException, F64Array, JavaRandom, JavaTreeSet  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility._epq_compat import EPQException, F64Array, JavaRandom, JavaTreeSet  # type: ignore

_TreeSet = JavaTreeSet  # local alias: TreeSet was an implementation detail in the Java source

try:
    from .Math2 import Math2
except ImportError:
    try:
        from Math2 import Math2  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility.Math2 import Math2  # type: ignore


__all__ = ["MultiDHistogram", "LinearBins", "IBinning", "Bin"]


class MultiDHistogram:

    BUG_LEDGER: tuple = (
        ("JAVA-BUG-1", "Bin.distance_literal",
         "Java source: `dist += Math.abs(mIndex[d] - vox.mIndex[d]); return Math.sqrt(dist);` "
         "— accumulates Manhattan (L1) distance then takes its square root, which is neither "
         "Manhattan nor Euclidean distance. Preserved faithfully.", True),
    )

    NO_ISLAND: int = 2147483647

    class IBinning(abc.ABC):
        @abc.abstractmethod
        def compute(self, vals: Sequence[float]) -> Optional[tuple[int, ...]]:
            """Map a point to bin indices, or None if the point is out of range."""
            pass

        @abc.abstractmethod
        def dimensions(self) -> int:
            """Return the number of dimensions."""
            pass

        @abc.abstractmethod
        def limits(self, bin_idx: int, dim: int) -> Optional[F64Array]:
            """Return [lo, hi) bounds for bin_idx in dimension dim."""
            pass

    class LinearBins(IBinning):
        def __init__(self, lo: Sequence[float], hi: Sequence[float], nBins: Sequence[int]) -> None:
            if len(lo) != len(hi) or len(lo) != len(nBins):
                raise ValueError("lo, hi, and nBins must have the same length")
            self.mLo: F64Array = np.asarray(lo, dtype=np.float64)
            self.mHi: F64Array = np.asarray(hi, dtype=np.float64)
            self.mNBins: tuple[int, ...] = tuple(int(n) for n in nBins)
            self.mDimCount: int = len(self.mNBins)

        def compute(self, vals: Sequence[float]) -> Optional[tuple[int, ...]]:
            return self.compute_literal(vals)

        def compute_literal(self, vals: Sequence[float]) -> Optional[tuple[int, ...]]:
            if len(vals) != self.mDimCount:
                return None
            res: list[int] = []
            for i in range(self.mDimCount):
                width: float = (self.mHi[i] - self.mLo[i]) / self.mNBins[i]
                idx: int = int((float(vals[i]) - self.mLo[i]) / width)
                if idx < 0 or idx >= self.mNBins[i]:
                    return None
                res.append(idx)
            return tuple(res)

        def dimensions(self) -> int:
            return self.mDimCount

        def getDimensions(self) -> tuple[int, ...]:
            """Per-dimension bin counts (utility; not part of IBinning contract)."""
            return self.mNBins

        def limits(self, bin_idx: int, dim: int) -> Optional[F64Array]:
            return self.limits_literal(bin_idx, dim)

        def limits_literal(self, bin_idx: int, dim: int) -> Optional[F64Array]:
            if not (0 <= dim < self.mDimCount) or not (0 <= bin_idx < self.mNBins[dim]):
                return None
            width: float = (self.mHi[dim] - self.mLo[dim]) / self.mNBins[dim]
            lo: float = float(self.mLo[dim]) + bin_idx * width
            hi: float = float(self.mLo[dim]) + (bin_idx + 1) * width
            return np.array([lo, hi], dtype=np.float64)

    class Bin:
        DEFAULT_CAPACITY: int = 10

        def __init__(self, index: Sequence[int], itemIndex: int) -> None:
            self.mIndex: tuple[int, ...] = tuple(int(x) for x in index)
            self.mItem: list[int] = [0] * self.DEFAULT_CAPACITY
            self.mItem[0] = int(itemIndex)
            self.mSize: int = 1
            self.mIsland: int = MultiDHistogram.NO_ISLAND
            h: int = 1
            for element in self.mIndex:
                h = (31 * h + element) & 0xFFFFFFFF
                if h >= 0x80000000:
                    h -= 0x100000000
            self.mHash: int = int(31 + h)

        @classmethod
        def from_bin(cls, vox: "MultiDHistogram.Bin") -> "MultiDHistogram.Bin":
            b = cls(vox.mIndex, 0)
            b.mItem = list(vox.mItem)
            b.mSize = int(vox.mSize)
            b.mIsland = int(vox.mIsland)
            b.mHash = int(vox.mHash)
            return b

        def getDimension(self) -> int:
            return len(self.mIndex)

        def add(self, item: int) -> None:
            requiredCapacity: int = self.DEFAULT_CAPACITY * (
                ((self.mSize + 1 + self.DEFAULT_CAPACITY) - 1) // self.DEFAULT_CAPACITY
            )
            assert requiredCapacity >= (self.mSize + 1)
            if requiredCapacity > len(self.mItem):
                new_items: list[int] = [0] * requiredCapacity
                new_items[:len(self.mItem)] = self.mItem
                self.mItem = new_items
            self.mItem[self.mSize] = int(item)
            self.mSize += 1
            assert self.mSize <= len(self.mItem)

        def adjacent_bin(self, vox: "MultiDHistogram.Bin") -> bool:
            # m=1: face-sharing only (exactly one coordinate differs by 1).
            # Passing len(self.mIndex) would incorrectly include diagonal neighbours.
            return self.adjacent_bin_m(vox, 1)

        def adjacent_bin_m(self, vox: "MultiDHistogram.Bin", m: int) -> bool:
            assert len(self.mIndex) == len(vox.mIndex)
            dist: int = 0
            for d in range(len(self.mIndex)):
                delta: int = abs(self.mIndex[d] - vox.mIndex[d])
                if delta > 1:
                    return False
                dist += delta
            return (dist > 0) and (dist <= m)

        def distance(self, vox: "MultiDHistogram.Bin") -> float:
            return self.distance_literal(vox)

        def distance_literal(self, vox: "MultiDHistogram.Bin") -> float:
            assert len(self.mIndex) == len(vox.mIndex)
            dist: float = 0.0
            for d in range(len(self.mIndex)):
                dist += abs(self.mIndex[d] - vox.mIndex[d])
            # JAVA-BUG-1: Java source: `return Math.sqrt(dist);` — sqrt of Manhattan distance.
            return float(math.sqrt(dist))

        def distance_strict(self, vox: "MultiDHistogram.Bin") -> float:
            """Strict Euclidean distance (corrects JAVA-BUG-1)."""
            assert len(self.mIndex) == len(vox.mIndex)
            dist: float = 0.0
            for d in range(len(self.mIndex)):
                diff: int = self.mIndex[d] - vox.mIndex[d]
                dist += diff * diff
            return float(math.sqrt(dist))

        def getIsland(self) -> int:
            return int(self.mIsland)

        def contains(self, item: int) -> bool:
            for i in range(self.mSize):
                if self.mItem[i] == item:
                    return True
            return False

        def getItems(self) -> tuple[int, ...]:
            return tuple(self.mItem[:self.mSize])

        def getCount(self) -> int:
            return int(self.mSize)

        def toString(self) -> str:
            return self.__str__()

        def __str__(self) -> str:
            res: str = ""
            for element in self.mIndex:
                res += str(element) + ", "
            res += str(self.mSize)
            return res

        def compareTo(self, o: "MultiDHistogram.Bin") -> int:
            assert len(self.mIndex) == len(o.mIndex)
            for i in range(len(self.mIndex)):
                if self.mIndex[i] < o.mIndex[i]:
                    return -1
                elif self.mIndex[i] > o.mIndex[i]:
                    return 1
            return 0

        def hashCode(self) -> int:
            return int(self.mHash)

        def __hash__(self) -> int:
            return self.hashCode()

        def equals(self, obj: object) -> bool:
            if self is obj:
                return True
            if obj is None:
                return False
            if not isinstance(obj, MultiDHistogram.Bin):
                return False
            return self.mIndex == obj.mIndex

        def __eq__(self, obj: object) -> bool:
            return self.equals(obj)

        def __lt__(self, other: object) -> bool:
            if not isinstance(other, MultiDHistogram.Bin):
                return NotImplemented
            return self.compareTo(other) < 0

        def __le__(self, other: object) -> bool:
            if not isinstance(other, MultiDHistogram.Bin):
                return NotImplemented
            return self.compareTo(other) <= 0

    def __init__(
        self,
        arg: Union["MultiDHistogram", "MultiDHistogram.IBinning"],
        minMembers: int = 1,
    ) -> None:
        if isinstance(arg, MultiDHistogram):
            # Copy constructor (Java: MultiDHistogram(MultiDHistogram) and
            # MultiDHistogram(MultiDHistogram, int minMembers))
            self.mBinning: MultiDHistogram.IBinning = arg.mBinning
            self.mData: _TreeSet = _TreeSet()
            self.mIslandCount: int = 0
            for vox in arg.mData:
                if vox.mSize >= minMembers:
                    # add_bin resets mIsland to NO_ISLAND before calling _addBin,
                    # which asserts mIsland == NO_ISLAND. from_bin copies the
                    # source island index, so _addBin must NOT be called directly.
                    self.add_bin(vox)
        else:
            self.mBinning = arg
            self.mData = _TreeSet()
            self.mIslandCount = 0

    @classmethod
    def from_histogram(cls, src: "MultiDHistogram") -> "MultiDHistogram":
        """Java: public MultiDHistogram(MultiDHistogram src)"""
        return cls(src, 1)

    @classmethod
    def from_histogram_min(cls, src: "MultiDHistogram", minMembers: int) -> "MultiDHistogram":
        """Java: public MultiDHistogram(MultiDHistogram src, int minMembers)"""
        return cls(src, minMembers)

    def add(self, data: Sequence[float], item: int) -> None:
        """Java: public void add(double[] data, int item)"""
        index: Optional[tuple[int, ...]] = self.mBinning.compute(data)
        if index is not None:
            newBin: MultiDHistogram.Bin = MultiDHistogram.Bin(index, item)
            b: Optional[MultiDHistogram.Bin] = self.find(newBin)
            if b is not None:
                b.add(item)
            else:
                self._addBin(newBin)

    def add_data(self, data: Sequence[float], item: int) -> None:
        """Backward-compatibility alias for add()."""
        return self.add(data, item)

    def add_bin(self, newBin: "MultiDHistogram.Bin") -> None:
        dup: MultiDHistogram.Bin = MultiDHistogram.Bin.from_bin(newBin)
        dup.mIsland = self.NO_ISLAND
        self._addBin(dup)

    def find(self, b: "MultiDHistogram.Bin") -> Optional["MultiDHistogram.Bin"]:
        res: Optional[MultiDHistogram.Bin] = self.mData.floor(b)
        return res if (res is not None) and (res.compareTo(b) == 0) else None

    def _addBin(self, newBin: "MultiDHistogram.Bin") -> None:
        assert newBin.mIsland == self.NO_ISLAND
        assert self._islandCheck(), "Island indices are not contiguous"
        for b1 in self.mData:
            if b1.adjacent_bin(newBin) and (newBin.mIsland != b1.mIsland):
                if newBin.mIsland == self.NO_ISLAND:
                    newBin.mIsland = b1.mIsland
                    assert self.mIslandCount > newBin.mIsland
                else:
                    smaller: int = min(newBin.mIsland, b1.mIsland)
                    larger: int = max(newBin.mIsland, b1.mIsland)
                    assert larger > smaller
                    assert self.mIslandCount > larger
                    for b2 in self.mData:
                        if b2.mIsland == larger:
                            b2.mIsland = smaller
                        if b2.mIsland > larger:
                            b2.mIsland -= 1
                    newBin.mIsland = smaller
                    self.mIslandCount -= 1
                    assert self.mIslandCount > smaller
        if newBin.mIsland == self.NO_ISLAND:
            newBin.mIsland = self.mIslandCount
            self.mIslandCount += 1
        assert newBin.mIsland < self.mIslandCount
        self.mData.add(newBin)
        assert self._islandCheck(), "Island indices are not contiguous"

    def _islandCheck(self) -> bool:
        present: list[bool] = [False] * self.mIslandCount
        for b in self.mData:
            present[b.mIsland] = True
        res: bool = True
        for element in present:
            res = res and element
        return res

    def getIsland(self, ni: int) -> "MultiDHistogram":
        res: MultiDHistogram = MultiDHistogram(self.mBinning)
        for bin_obj in self.mData:
            if bin_obj.mIsland == ni:
                res.add_bin(bin_obj)
        return res

    def getIslandSize(self, ni: int) -> int:
        cx: int = 0
        for bin_obj in self.mData:
            if bin_obj.mIsland == ni:
                cx += 1
        return cx

    def getIslandCount(self) -> int:
        assert self._checkIslandCount() == self.mIslandCount
        return int(self.mIslandCount)

    def _checkIslandCount(self) -> int:
        largest: int = -1
        for bin_obj in self.mData:
            if bin_obj.mIsland > largest:
                largest = bin_obj.mIsland
        return largest + 1

    def getTotalCount(self) -> int:
        total: int = 0
        for bin_obj in self.mData:
            total += bin_obj.getCount()
        return total

    def findItemsIsland(self, item: int) -> int:
        for bin_obj in self.mData:
            if bin_obj.contains(item):
                return int(bin_obj.mIsland)
        return self.NO_ISLAND

    def getItemsInIsland(self, n: int) -> tuple[int, ...]:
        assert n < self.mIslandCount
        # getIslandSize counts bins; we need total items across those bins.
        nn: int = sum(b.mSize for b in self.mData if b.mIsland == n)
        res: list[int] = [0] * nn
        cx: int = 0
        for bin_obj in self.mData:
            if bin_obj.mIsland == n:
                for i in range(bin_obj.mSize):
                    res[cx] = bin_obj.mItem[i]
                    cx += 1
        res.sort()
        return tuple(res)

    def connected(self, starter: int) -> tuple[int, ...]:
        vox: Optional[MultiDHistogram.Bin] = None
        for datum in self.mData:
            if datum.contains(starter):
                vox = datum
                break
        nbd: MultiDHistogram = (
            self.getIsland(vox.mIsland) if vox is not None else MultiDHistogram(self.mBinning)
        )
        cx: int = 0
        for datum in nbd.mData:
            cx += datum.mSize
        res: list[int] = [0] * cx
        i: int = 0
        for datum in nbd.mData:
            for j in range(datum.mSize):
                res[i] = datum.mItem[j]
                i += 1
        assert i == cx
        res.sort()
        return tuple(res)

    def getBins(self) -> set["MultiDHistogram.Bin"]:
        return set(self.mData)

    def chiSquaredTest(
        self, mdh2: "MultiDHistogram", minSize: int, confidenceLevel: float
    ) -> bool:
        return self.chiSquaredTest_literal(mdh2, minSize, confidenceLevel)

    def chiSquaredTest_literal(
        self, mdh2: "MultiDHistogram", minSize: int, confidenceLevel: float
    ) -> bool:
        chi1: float = 0.0
        chi2: float = 0.0
        n1: float = 0.0
        n2: float = 0.0
        df: int = 0
        allBins: _TreeSet = _TreeSet()
        for b in self.mData:
            allBins.add(b)
        for b in mdh2.mData:
            allBins.add(b)
        for ba in allBins:
            b1: Optional[MultiDHistogram.Bin] = self.mData.floor(ba)
            b2: Optional[MultiDHistogram.Bin] = mdh2.mData.floor(ba)
            x1: float = float(b1.mSize) if (b1 is not None) and b1.equals(ba) else 0.0
            x2: float = float(b2.mSize) if (b2 is not None) and b2.equals(ba) else 0.0
            s: float = x1 + x2
            if s >= float(minSize):
                n1 += x1
                n2 += x2
                chi1 += (x1 * x1) / s
                chi2 += (x2 * x2) / s
                df += 1
        ns: float = n1 + n2
        chi1 -= (n1 * n1) / ns
        chi2 -= (n2 * n2) / ns
        k: float = (ns * ns) / (n1 * n2) if (n1 * n2) != 0.0 else float('inf')
        chi1 *= k
        chi2 *= k
        assert abs(chi1 - chi2) < 0.0001
        return bool(chi1 >= Math2.chiSquaredConfidenceLevel(confidenceLevel, df))

    def getCount(self, index: Sequence[int]) -> int:
        bin_obj: MultiDHistogram.Bin = MultiDHistogram.Bin(index, 0)
        res: Optional[MultiDHistogram.Bin] = self.mData.floor(bin_obj)
        return int(res.getCount()) if (res is not None) and (res.compareTo(bin_obj) == 0) else 0


# Module-level aliases so `from MultiDHistogram_ver1_1_0 import LinearBins` works
IBinning = MultiDHistogram.IBinning
LinearBins = MultiDHistogram.LinearBins
Bin = MultiDHistogram.Bin
