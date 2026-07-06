# DescriptiveStatistics Conversion Spec

> **Spec revision ver1_1_1** (2026-06-25): repaired against the Java source. The prior
> ver1_1_0 invented methods (`add(double,double)`, `meanAndSigma`, `meanAndSigmaOfMean`,
> `total`, `isOk`, `min`/`max`), invented fields (`mWeightSum`, `mWeightSumSqr`), and a
> one-arg copy constructor — none exist. The real class accumulates four power sums plus
> min/max/count, has a **two-arg combine** constructor, and computes skewness/kurtosis.
> One genuine bug (`merge` double-counts `mNPoints`) is retained.

## Java class
`gov.nist.microanalysis.Utility.DescriptiveStatistics`

Source: `src/gov/nist/microanalysis/Utility/DescriptiveStatistics.java`

---

## Inbound dependencies (Java imports)
- `java.util.Collection` — the `compute(Collection)` overload; use Python iterable/`list`.
- `gov.nist.microanalysis.Utility.UncertainValue2` — return type of `getValue(String)` (3-arg constructor); import from the sibling port. (Same-package use; graph: `DescriptiveStatistics → UncertainValue2`.) **No `Math2` dependency** — squares/cubes are computed inline.

`DescriptiveStatistics implements Comparable<DescriptiveStatistics>`.

---

## Outbound dependents (callers of public methods)
Running accumulator for mean/variance/skewness/kurtosis. `MCUncertaintyEngine.getStatistics()` builds one via the static `compute(Collection)`. Not exhaustively audited.

---

## Public API surface

### Constructors

| Java signature | Python signature | Notes |
|---|---|---|
| `public DescriptiveStatistics()` | `__init__(self)` | empty: sums 0, `mNPoints` 0, `mMin = MAX_VALUE`, `mMax = -MAX_VALUE`, `mLast = NaN` |
| `public DescriptiveStatistics(DescriptiveStatistics ds1, DescriptiveStatistics ds2)` | `__init__(self, ds1, ds2)` | **two-arg combine** — sums and counts added, min/max combined; dispatch by arg count |

`__init__` dispatches by arity: no args → empty; two `DescriptiveStatistics` args → combine. (There is **no** one-arg copy constructor; copying is via `clone()`.)

### Public instance methods

| Java signature | Python signature | Notes |
|---|---|---|
| `@Override public DescriptiveStatistics clone()` | `def clone(self) -> DescriptiveStatistics` + `def __copy__(self)` | copies all eight fields |
| `public void merge(DescriptiveStatistics ds)` | `def merge(self, ds) -> None` | **JAVA-BUG-1** — double-counts `mNPoints` |
| `public void add(double x)` | `def add(self, x: float) -> None` | updates four power sums, min/max (NaN-aware), `mLast`, `++mNPoints` |
| `public double average()` | `def average(self) -> float` | `mSum / mNPoints` |
| `public double variance()` | `def variance(self) -> float` | `mNPoints>1 ? max(0,(mSumOfSqrs - mSum*avg)/mNPoints) : NaN` |
| `public double standardDeviation()` | `def standardDeviation(self) -> float` | `sqrt(variance())` |
| `public UncertainValue2 getValue(String name)` | `def getValue(self, name: str) -> UncertainValue2` | `UncertainValue2(average(), name, isnan(sd) ? 0.0 : sd)` |
| `public double skewness()` | `def skewness(self) -> float` | `(mSumOfCubes - 3·mu·mSumOfSqrs + 2·mSum·mu²) / (mNPoints·variance()^1.5)` |
| `public double standardErrorOfSkewness()` | `def standardErrorOfSkewness(self) -> float` | `sqrt(6/mNPoints)` |
| `public double kurtosis()` | `def kurtosis(self) -> float` | `((mSumOfQuarts - 4·mu·mSumOfCubes + 6·mu²·mSumOfSqrs - 3·mSum·mu³)/(mNPoints·v²)) - 3` |
| `public double standardErrorOfKurtosis()` | `def standardErrorOfKurtosis(self) -> float` | `sqrt(24/mNPoints)` |
| `public double minimum()` | `def minimum(self) -> float` | returns `mMin` (note name: `minimum`, not `min`) |
| `public double maximum()` | `def maximum(self) -> float` | returns `mMax` (note name: `maximum`, not `max`) |
| `@Override public String toString()` | `def __str__(self) -> str` + `def toString(self) -> str` | tab-separated `count\taverage\tstdDev\tmin\tmax` (R2) |
| `public int count()` | `def count(self) -> int` | `mNPoints` |
| `public double sum()` | `def sum(self) -> float` | `mSum` (note name: `sum`, not `total`) |
| `public double getLastAdded()` | `def getLastAdded(self) -> float` | `mLast` |
| `public void remove(double val)` | `def remove(self, val: float) -> None` | subtracts power sums, `--mNPoints`; **may corrupt min/max to NaN** (documented Java caveat) |
| `public boolean removeLast()` | `def removeLast(self) -> bool` | removes `mLast` if not NaN; resets `mLast = NaN` |
| `@Override public int compareTo(DescriptiveStatistics o)` | `def compareTo(self, o) -> int` + `__lt__/__le__/__gt__/__ge__` | order by `average()`, tie-break by `variance()` (R2 Comparable) |

### Static methods

| Java signature | Python signature | Notes |
|---|---|---|
| `public static DescriptiveStatistics compute(Number[] ns)` | `@staticmethod def compute_array(ns) -> DescriptiveStatistics` | R4 |
| `public static DescriptiveStatistics compute(Collection<? extends Number> ns)` | `@staticmethod def compute_collection(ns) -> DescriptiveStatistics` | R4 — `MCUncertaintyEngine` calls this form |

---

## Private / protected members

| Java | Python |
|---|---|
| `private double mLast` | `self._mLast: float = float('nan')` |
| `private double mSum` | `self._mSum: float = 0.0` |
| `private double mSumOfSqrs` | `self._mSumOfSqrs: float = 0.0` |
| `private double mSumOfCubes` | `self._mSumOfCubes: float = 0.0` |
| `private double mSumOfQuarts` | `self._mSumOfQuarts: float = 0.0` |
| `private double mMin` | `self._mMin: float = sys.float_info.max` |
| `private double mMax` | `self._mMax: float = -sys.float_info.max` |
| `private int mNPoints` | `self._mNPoints: int = 0` |

---

## Overloaded methods (split plan)
- Constructor: `()` / `(DescriptiveStatistics, DescriptiveStatistics)` → `__init__` arity dispatch.
- `compute`: `(Number[])` / `(Collection)` → `compute_array` / `compute_collection` (R4). Callers in this package use the collection form.

---

## Mutable-output methods
`add`, `merge`, `remove`, `removeLast` mutate `self`. None mutate caller-supplied buffers (the combine constructor and `merge` only read `ds`/`ds1`/`ds2`). No R5 guard needed.

---

## Touchpoints into Jama / javax.swing / java.awt / java.io
None.

---

## Abstract class strategy
Not applicable — concrete class. IS_ABSTRACT = False.

---

## Java-specific translation decisions

| Java pattern | Python translation | Rule |
|---|---|---|
| `implements Comparable<DescriptiveStatistics>` | `compareTo` + `__lt__/__le__/__gt__/__ge__` | R2 |
| `Double.MAX_VALUE` / `-Double.MAX_VALUE` init | `sys.float_info.max` / `-sys.float_info.max` | |
| `Double.NaN`, `Double.isNaN` | `float('nan')`, `math.isnan` | |
| `x2 = x*x; mSumOfCubes += x2*x; mSumOfQuarts += x2*x2` | inline, faithful (no `Math2.sqr`) | |
| `Math.pow(variance(), 1.5)` in `skewness` | `math.pow(self.variance(), 1.5)` | |
| `mSum / mNPoints` (int divisor, float numerator) | `self._mSum / self._mNPoints` — Java promotes to double; this is **float** division, **not** `//` | R7 caveat — `mSum` is `double`, so result is float; keep `/` |
| `Integer.toString` / `Double.toString` in `toString` | match Java formatting in parity assertions or compare field-by-field | exact-string note |
| `getValue` 3-arg `UncertainValue2(avg, name, sd)` | sibling-port constructor | R3 |

---

## Suspected Java bugs

**JAVA-BUG-1 — `merge()` double-counts `mNPoints`.**
Java source: `merge` contains `mNPoints += ds.mNPoints;` **twice** (the statement appears at the start and again at the end of the method). After merging an `m`-point object into an `n`-point object, `count()` reports `n + 2·m` instead of `n + m`; `average()`/`variance()` (which divide by `mNPoints`) are correspondingly wrong.
Disposition: **Preserve** behind `# JAVA-BUG-1` quoting the duplicated line; `BUG_LEDGER` entry `has_strict_variant=True`; `merge_strict` increments `mNPoints` once.

**Not a bug — two-arg constructor's repeated `mNPoints = ...`.** The combine constructor assigns `mNPoints = ds1.mNPoints + ds2.mNPoints` **twice**, but both are plain assignments (`=`), so the second is idempotent — the count is correct. Port verbatim; do **not** log it as a bug and do **not** "fix" it to a single line in a way that changes observable behaviour (there is none).

**Caveat (documented, not a bug) — `remove`/`removeLast` can set `mMin`/`mMax` to NaN.** The Javadoc warns min/max may become corrupt after `remove`. Preserve the behaviour exactly.

---

## Static init order
All fields initialise in `__init__`. The static `compute_*` methods construct instances at call time. No cross-class static cycle (`getValue` touches `UncertainValue2` only at call time).

---

## Thread safety
Not thread-safe. All accumulator fields are mutable and unguarded; concurrent `add`/`merge`/`remove` corrupts the sums and count.
