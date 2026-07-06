# MultiDHistogram Conversion Spec

## Java class
`gov.nist.microanalysis.Utility.MultiDHistogram`

Source: `src/gov/nist/microanalysis/Utility/MultiDHistogram.java`

---

## Inbound dependencies (Java imports)
- `java.util.Arrays` — `Arrays.fill`, `Arrays.sort`, `Arrays.hashCode`, `Arrays.copyOf`, `Arrays.equals`
- `java.util.Collections` — `Collections.unmodifiableSet`
- `java.util.Set`, `java.util.TreeSet` — sorted set of `Bin` objects
- `gov.nist.microanalysis.Utility.Math2` — `Math2.bound` in `LinearBins.compute`; `Math2.chiSquaredConfidenceLevel` in `chiSquaredTest`; import from sibling port

No EPQException, Jama, or java.awt imports.

---

## Outbound dependents (callers of public methods)
Not audited.

---

## Public API surface

### Inner interface: `MultiDHistogram.IBinning`

| Java signature | Python signature | Notes |
|---|---|---|
| `int[] compute(double[] vals)` | `@abc.abstractmethod def compute(self, vals: list[float]) -> Optional[list[int]]` | Returns `None` if out of range |
| `double[] limits(int bin, int dim)` | `@abc.abstractmethod def limits(self, bin: int, dim: int) -> Optional[list[float]]` | Returns `[min, max]` for specified bin/dim; `None` if out of range |
| `int[] getDimensions()` | `@abc.abstractmethod def getDimensions(self) -> list[int]` | Size of each dimension |

Port as `class IBinning(abc.ABC)` nested inside `MultiDHistogram`. Module-level alias: `IBinning = MultiDHistogram.IBinning`. (R2)

### Inner class: `MultiDHistogram.LinearBins` (implements `IBinning`)

| Java signature | Python signature | Notes |
|---|---|---|
| `public LinearBins(int dim, double width, int number)` | `__init__(self, dim: int, width: float, number: int)` | |
| `public int[] compute(double[] vals)` | `def compute(self, vals: list[float]) -> list[int]` | `Math2.bound((int)(vals[i]/width), 0, mDimLength)` for each i |
| `public double[] limits(int bin, int dim)` | `def limits(self, bin: int, dim: int) -> Optional[list[float]]` | JAVA-BUG-1 applies |
| `public int[] getDimensions()` | `def getDimensions(self) -> list[int]` | Returns `[mDimLength] * mDimCount` |

Module-level alias: `LinearBins = MultiDHistogram.LinearBins`. (R2)

### Inner class: `MultiDHistogram.Bin` (implements `Comparable<Bin>`)

`Bin` is a public inner class with a private constructor. Used as a data structure inside `MultiDHistogram`.

| Java signature | Python signature | Notes |
|---|---|---|
| `private Bin(int[] index, int itemIndex)` | `def __init__(self, index: list[int], itemIndex: int)` — prefix with `_` convention but keep accessible | Package-private in Java; expose as public in Python for copy constructor |
| `private Bin(Bin vox)` | classmethod `copy(cls, vox: Bin) -> Bin` | Copy constructor |
| `public int getDimension()` | `def getDimension(self) -> int` | |
| `public boolean adjacent(Bin vox)` | `def adjacent(self, vox: Bin) -> bool` | |
| `public boolean adjacent(Bin vox, int m)` | `def adjacent_m(self, vox: Bin, m: int) -> bool` | R4 |
| `public double distance(Bin vox)` | `def distance(self, vox: Bin) -> float` | |
| `public int getIsland()` | `def getIsland(self) -> int` | |
| `public boolean contains(int item)` | `def contains(self, item: int) -> bool` | Linear scan |
| `public int[] getItems()` | `def getItems(self) -> list[int]` | Returns copy |
| `public int getCount()` | `def getCount(self) -> int` | |
| `public int compareTo(Bin o)` | `def compareTo(self, o: Bin) -> int` + `__lt__` etc. | R2 |
| `public int hashCode()` | `def __hash__(self) -> int` + `def hashCode(self) -> int` | R2 — based on `tuple(mIndex)` |
| `public boolean equals(Object obj)` | `def __eq__(self, obj) -> bool` + `def equals(self, obj) -> bool` | R2 |
| `public String toString()` | `def __str__(self) -> str` + `def toString(self) -> str` | R2 |

Module-level alias: `Bin = MultiDHistogram.Bin`. (R2)

### `MultiDHistogram` constructor and methods

| Java signature | Python signature | Notes |
|---|---|---|
| `public MultiDHistogram(IBinning binning)` | `__init__(self, binning: IBinning)` | |
| `public MultiDHistogram(MultiDHistogram src)` | classmethod `copy(cls, src: MultiDHistogram) -> MultiDHistogram` — or dispatch in `__init__` | Copy constructor |
| `public MultiDHistogram(MultiDHistogram src, int minMembers)` | classmethod `copy_trimmed(cls, src: MultiDHistogram, minMembers: int) -> MultiDHistogram` | R4 |
| `public void add(double[] data, int item)` | `def add(self, data: list[float], item: int) -> None` | |
| `public void add(Bin newBin)` | `def add_bin(self, newBin: Bin) -> None` | R4 |
| `public Bin find(Bin b)` | `def find(self, b: Bin) -> Optional[Bin]` | |
| `public MultiDHistogram getIsland(int ni)` | `def getIsland(self, ni: int) -> MultiDHistogram` | |
| `public int getIslandSize(int ni)` | `def getIslandSize(self, ni: int) -> int` | |
| `public int getIslandCount()` | `def getIslandCount(self) -> int` | |
| `public int getTotalCount()` | `def getTotalCount(self) -> int` | |
| `public int findItemsIsland(int item)` | `def findItemsIsland(self, item: int) -> int` | Returns `NO_ISLAND` if not found |
| `public int[] getItemsInIsland(int n)` | `def getItemsInIsland(self, n: int) -> list[int]` | Returns sorted array |
| `public int[] connected(int starter)` | `def connected(self, starter: int) -> list[int]` | Returns sorted array of connected items |
| `public Set<Bin> getBins()` | `def getBins(self) -> frozenset[Bin]` | Returns unmodifiable view — use `frozenset` |
| `public double chiSquaredTest(MultiDHistogram mdh2, int minSize, double confidenceLevel)` | `def chiSquaredTest(self, mdh2: MultiDHistogram, minSize: int, confidenceLevel: float) -> float` | |
| `public int getCount(int[] index)` | `def getCount(self, index: list[int]) -> int` | |

---

## Constants

| Java | Python |
|---|---|
| `public static final int NO_ISLAND = Integer.MAX_VALUE` | `NO_ISLAND: int = 2**31 - 1` |
| `static private final int DEFAULT_CAPACITY = 10` (in `Bin`) | `_DEFAULT_CAPACITY: int = 10` (in `Bin`) |

---

## Private / protected members

| Java (MultiDHistogram) | Python |
|---|---|
| `private final IBinning mBinning` | `self._mBinning: IBinning` |
| `private final TreeSet<Bin> mData` | `self._mData: SortedSet[Bin]` — Python has no `TreeSet`; use `sortedcontainers.SortedList` or maintain a sorted `list` |
| `private int mIslandCount` | `self._mIslandCount: int` |

**TreeSet translation note**: Java `TreeSet<Bin>` uses `Bin.compareTo` for ordering and provides O(log n) `floor()`. Python's `sortedcontainers.SortedList` is the closest equivalent. If `sortedcontainers` is unavailable, maintain a plain Python `list` kept sorted via `bisect`. The `find(b)` method requires a "floor" (largest element ≤ b) — implement via `bisect_right` and stepping back one.

| Java (Bin) | Python |
|---|---|
| `private int[] mIndex` | `self._mIndex: tuple[int, ...]` — use `tuple` for hashability |
| `private int mSize` | `self._mSize: int` |
| `private int[] mItem` | `self._mItem: list[int]` — dynamic resizing handled by `list.append` |
| `private int mIsland` | `self.mIsland: int` — **public** in Python (Bin.mIsland is accessed directly by MultiDHistogram internals in Java) |
| `private final transient int mHash` | `self._mHash: int` — cached hash |

---

## Overloaded methods (split plan)

`add(double[], int)` and `add(Bin)` → `add` / `add_bin`.

`adjacent(Bin)` and `adjacent(Bin, int)` → `adjacent` / `adjacent_m`.

`MultiDHistogram(src)` and `MultiDHistogram(src, minMembers)` → classmethods `copy` / `copy_trimmed`.

---

## Mutable-output methods

`add` and `add_bin` mutate `self._mData` and island assignments — intentional.

`addBin` (private) mutates island indices of existing bins — R5 does not apply; this is core algorithm behavior.

---

## Touchpoints into Jama / javax.swing / java.awt / java.io
None.

---

## Abstract class strategy
IS_ABSTRACT = False for `MultiDHistogram`. IS_ABSTRACT = True for `IBinning` (interface). `LinearBins` is concrete.

M4 does not apply to `MultiDHistogram` or `LinearBins`. Parity tests can instantiate `LinearBins` directly.

---

## Java-specific translation decisions

| Java pattern | Python translation | Rule |
|---|---|---|
| `TreeSet<Bin>` | `sortedcontainers.SortedList` with key=`Bin.compareTo` | No Python stdlib equivalent for ordered set with floor |
| `mData.floor(b)` | `SortedList._floor(b)` helper — find largest element ≤ b | |
| `Collections.unmodifiableSet(mData)` | `frozenset(self._mData)` — returns a snapshot | |
| `Arrays.copyOf(mItem, mSize)` | `self._mItem[:self._mSize]` | |
| `Arrays.hashCode(mIndex)` | `hash(tuple(self._mIndex))` | |
| `Arrays.equals(mIndex, other.mIndex)` | `self._mIndex == other._mIndex` | |
| Dynamic array resize (`Arrays.copyOf` + size check) | Replace with `list.append` | |
| `int[] mIndex` used as key | Store as `tuple[int, ...]` for hashability | |

---

## Suspected Java bugs

**JAVA-BUG-1 — `LinearBins.limits` computes wrong upper bin boundary.**
Java source line:
```java
return new double[]{bin * mWidth, bin * (mWidth + 1)};
```
The upper bound should be `(bin + 1) * mWidth`, not `bin * (mWidth + 1)`. The current formula gives `bin * mWidth + bin` for `mWidth=5, bin=3` → `[15, 18]` instead of `[15, 20]`. For `bin=0` both formulas give `[0, 0]`.
Disposition: **Preserve** — port the buggy formula verbatim; document as `# JAVA-BUG-1`. The strict variant uses `(bin + 1) * mWidth`.

---

## Static init order
`NO_ISLAND = Integer.MAX_VALUE` and `DEFAULT_CAPACITY = 10` are compile-time constants — initialize in class body. No ordering concern.

---

## Thread safety
Not thread-safe. `_mData`, `_mIslandCount`, and `Bin.mIsland` fields are all mutable. Concurrent `add` calls would corrupt island assignments.
