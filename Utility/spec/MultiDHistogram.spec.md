# MultiDHistogram Conversion Spec

## Java class
`gov.nist.microanalysis.Utility.MultiDHistogram`

Source: `src/gov/nist/microanalysis/Utility/MultiDHistogram.java`

---

## Inbound dependencies (Java imports)
- `java.util.Arrays` — `Arrays.sort`, `Arrays.copyOf`, `Arrays.fill`, `Arrays.hashCode`
- `java.util.Collections` — `Collections.unmodifiableSet`
- `java.util.Set` — return type of `getBins()`
- `java.util.TreeSet` — backing store `mData`; also used in `chiSquaredTest`
- `gov.nist.microanalysis.Utility.Math2` — `Math2.bound` (in `LinearBins.compute`) and `Math2.chiSquaredConfidenceLevel` (in `chiSquaredTest`)

---

## Outbound dependents (callers of public methods)
- `EPQTests/MultiDHistogramTest.java` — tests only
- Various EPQLibrary classes that accumulate multi-dimensional Monte Carlo data

---

## Public API surface

### IBinning (interface)
| Java signature | Python signature | Notes |
|---|---|---|
| `int[] compute(double[] vals)` | `compute(self, vals) -> list[int]` | Returns bin indices or None if out of range |
| `double[] limits(int bin, int dim)` | `limits(self, bin: int, dim: int) -> list[float] \| None` | `[min, max]` or `None` |
| `int[] getDimensions()` | `getDimensions(self) -> tuple[int, ...]` | Size per dimension |

### LinearBins (static inner class, implements IBinning)
| Java signature | Python signature | Notes |
|---|---|---|
| `LinearBins(int dim, double width, int number)` | `__init__(self, lo, hi, dims)` | Port uses `lo/hi/dims` arrays instead — see translation decisions |
| `int[] compute(double[] vals)` | `compute(self, vals) -> list[int]` | Calls `Math2.bound` |
| `double[] limits(int bin, int dim)` | `limits(self, bin, dim) -> list[float] \| None` | JAVA-BUG-1: `bin * (mWidth + 1)` is wrong; port preserves it |
| `int[] getDimensions()` | `getDimensions(self) -> tuple[int, ...]` | Returns `(mDimLength,) * mDimCount` |

### Bin (static inner class, Comparable<Bin>)
| Java signature | Python signature | Notes |
|---|---|---|
| `private Bin(int[] index, int itemIndex)` | `Bin(indices, item)` | Port exposes constructor publicly for test access |
| `int getDimension()` | `getDimension(self) -> int` | `mIndex.length` |
| `boolean adjacent(Bin vox)` | `adjacent(self, vox) -> bool` | Face-adjacency (m = nDim) |
| `boolean adjacent(Bin vox, int m)` | `adjacent(self, vox, m) -> bool` | m-adjacency |
| `double distance(Bin vox)` | `distance(self, vox) -> float` | L1 distance (Manhattan, then sqrt) |
| `int getIsland()` | `getIsland(self) -> int` | Returns `mIsland` |
| `boolean contains(int item)` | `contains(self, item: int) -> bool` | Linear scan of `mItem` |
| `int[] getItems()` | `getItems(self) -> list[int]` | Copy of items slice |
| `int getCount()` | `getCount(self) -> int` | `mSize` |
| `int compareTo(Bin o)` | `compareTo(self, o) -> int` | Lexicographic on indices: -1/0/1 |
| `boolean equals(Object obj)` | `__eq__` | Index equality only |
| `String toString()` | `__str__` | Indices + count |

### MultiDHistogram
| Java signature | Python signature | Notes |
|---|---|---|
| `MultiDHistogram(IBinning binning)` | `__init__(self, binning)` | Empty histogram |
| `MultiDHistogram(MultiDHistogram src)` | `__init__(self, src)` | Delegates to `(src, 1)` |
| `MultiDHistogram(MultiDHistogram src, int minMembers)` | `__init__(self, src, minMembers)` | Copy with min-members filter |
| `void add(double[] data, int item)` | `add(self, data, item)` | Primary data entry point |
| `void add(Bin newBin)` | `add(self, newBin: Bin)` | Copies bin from another histogram |
| `Bin find(Bin b)` | `find(self, b: Bin) -> Bin \| None` | Floor lookup |
| `MultiDHistogram getIsland(int ni)` | `getIsland(self, ni: int) -> MultiDHistogram` | Extracts island as sub-histogram |
| `int getIslandSize(int ni)` | `getIslandSize(self, ni: int) -> int` | Bins in island |
| `int getIslandCount()` | `getIslandCount(self) -> int` | Number of connected components |
| `int getTotalCount()` | `getTotalCount(self) -> int` | Sum of all bin counts |
| `int findItemsIsland(int item)` | `findItemsIsland(self, item: int) -> int` | Island index for item, or `NO_ISLAND` |
| `int[] getItemsInIsland(int n)` | `getItemsInIsland(self, n: int) -> list[int]` | Sorted items in island |
| `int[] connected(int starter)` | `connected(self, starter: int) -> list[int]` | All items in same island as starter |
| `Set<Bin> getBins()` | `getBins(self) -> set[Bin]` | Unmodifiable view |
| `double chiSquaredTest(MultiDHistogram mdh2, int minSize, double confidenceLevel)` | `chiSquaredTest(self, mdh2, minSize, confidenceLevel) -> float` | Returns ratio; <1 similar, >1 dissimilar |
| `int getCount(int[] index)` | `getCount(self, index) -> int` | Items in bin at index |
| `static final int NO_ISLAND` | `NO_ISLAND: int = sys.maxsize` | `Integer.MAX_VALUE` equivalent |

---

## Private / protected members

| Java | Python |
|---|---|
| `final IBinning mBinning` | `self.mBinning` |
| `TreeSet<Bin> mData` | `self.mData: SortedList[Bin]` (or sorted set equivalent) |
| `int mIslandCount` | `self.mIslandCount: int` |
| `private void addBin(Bin)` | `_addBin(self, bin)` |
| `private boolean islandCheck()` | `_islandCheck(self)` |
| `private int checkIslandCount()` | `_checkIslandCount(self)` |
| `Bin.mIndex`, `mItem`, `mSize`, `mIsland`, `mHash` | `_index`, `_items`, `_size`, `_island`, `_hash` |

---

## Overloaded methods (split plan)

| Java overloads | Python translation |
|---|---|
| `MultiDHistogram(IBinning)` and `(MultiDHistogram)` and `(MultiDHistogram, int)` | Single `__init__` dispatching on type of first arg |
| `add(double[], int)` and `add(Bin)` | Single `add` dispatching on type of first arg |
| `adjacent(Bin)` and `adjacent(Bin, int)` | `adjacent(vox, m=None)` with default `m = nDim` |

---

## Mutable-output methods
- **`addBin`**: mutates `mData` in place and updates `mIsland` fields on existing bins during island merges.
- **`perform`**: R5 guard not applicable here (not a numpy array method).

---

## Touchpoints into Jama / javax.swing / java.awt / java.io
None.

---

## Abstract class strategy
Not abstract. Concrete class; full `@needs_java` parity testing is possible.

---

## Java-specific translation decisions

| Java pattern | Python translation | Rule |
|---|---|---|
| `TreeSet<Bin>` (sorted set, `Bin implements Comparable`) | Python `SortedList` (from `sortedcontainers`) or manual sorted list; `Bin.__lt__` derived from `compareTo` | R2 |
| `LinearBins(int dim, double width, int number)` — stores width | Port uses `(lo[], hi[], dims[])` instead for ergonomics; `mWidth` computed per-dimension as `(hi-lo)/dims` | DEVIATION |
| `JAVA-BUG-1: limits()` returns `bin * (mWidth + 1)` (should be `(bin+1) * mWidth`) | Preserved faithfully in port | R6 |
| `static final int NO_ISLAND = Integer.MAX_VALUE` | `NO_ISLAND: int = sys.maxsize` | R2 |
| `Collections.unmodifiableSet(mData)` | Returns a copy or read-only view | R2 |
| `Bin` constructor is `private` | Port makes it package-accessible for test purposes | R1 deviation |

---

## Suspected Java bugs

**JAVA-BUG-1** — `LinearBins.limits()` upper-bound calculation.

```java
return new double[]{bin * mWidth, bin * (mWidth + 1)};
```

Should be `(bin + 1) * mWidth` but Java has `bin * (mWidth + 1)`. For bin=0, both return 0.0; for bin>0, the Java result is wrong. Faithfully preserved in the port. The compliance checker does not detect this; it is documented here and in the port's module docstring.

---

## Static init order
`static final int NO_ISLAND = Integer.MAX_VALUE` — constant; safe.

---

## Thread safety
Not documented as thread-safe. Mutating `mData` and island indices during `addBin` is not synchronized. No lock added in port.
