# UncertainValue2 Conversion Spec

## Java class
`gov.nist.microanalysis.Utility.UncertainValue2`

Source: `src/gov/nist/microanalysis/Utility/UncertainValue2.java`

---

## Inbound dependencies (Java imports)
- `java.text.NumberFormat` — parameter to `format(NumberFormat)` etc.; port as `Callable[[float], str]`
- `java.util.ArrayList`, `java.util.Arrays`, `java.util.Collection`, `java.util.Collections` — standard collections
- `java.util.HashMap`, `java.util.HashSet`, `java.util.Map`, `java.util.Objects`, `java.util.Set`, `java.util.TreeMap`, `java.util.TreeSet` — use `dict`, `set`, `sorted()` as appropriate; `TreeMap` → `dict` (Python 3.7+ dicts preserve insertion order; sorted key order requires `sorted()`)
- `Jama.Matrix` — used in `normalize(UncertainValue2[])` and `atan2`; import `JamaMatrix` from `_epq_compat`
- `gov.nist.microanalysis.EPQLibrary.EPQException` — thrown by `renameComponent`; import from `_epq_compat`
- `gov.nist.microanalysis.Utility.UtilException` — thrown by `weightedMean`; import from sibling port
- `gov.nist.microanalysis.Utility.Math2` — `Math2.sqr`, `Math2.bound` used in `variance(Correlations)` and `Correlations.add`; import from sibling port
- `gov.nist.microanalysis.Utility.HalfUpFormat` — used in `formatComponent(String, int)`; import from sibling port
- `gov.nist.microanalysis.Utility.ExponentFormat` — used in `formatComponent(String, int)`; import from sibling port

---

## Outbound dependents (callers of public methods)
Most depended-upon class in the Utility package after Math2 (in_degree 9). Used by `LevenbergMarquardt2`, `DescriptiveStatistics`, `Constraint`, and throughout EPQLibrary. Not exhaustively audited.

---

## Public API surface

### Static constants

| Java signature | Python signature | Notes |
|---|---|---|
| `public static final String DEFAULT = "Default"` | `DEFAULT: str = "Default"` | |
| `public static final UncertainValue2 ONE` | `ONE: UncertainValue2` | See Static init order |
| `public static final UncertainValue2 ZERO` | `ZERO: UncertainValue2` | |
| `public static final UncertainValue2 NaN` | `NaN: UncertainValue2` | Conflicts with Python builtin `float('nan')` — use `UV2_NaN` internally but expose as `NaN` at module level |
| `public static final UncertainValue2 POSITIVE_INFINITY` | `POSITIVE_INFINITY: UncertainValue2` | |
| `public static final UncertainValue2 NEGATIVE_INFINITY` | `NEGATIVE_INFINITY: UncertainValue2` | |
| `public static final UncertainValue2 MAX_VALUE` | `MAX_VALUE: UncertainValue2` | |

### Constructors

| Java signature | Python signature | Notes |
|---|---|---|
| `public UncertainValue2(double v, double dv)` | `__init__(self, v: float, dv: float)` | Auto-generates source name `"Default{n}"` |
| `public UncertainValue2(double v)` | `__init__(self, v: float, dv: float = 0.0)` | No uncertainty — collapses with above |
| `public UncertainValue2(double v, String source, double dv)` | `__init__(self, v: float, source: str, dv: float)` | Named source — dispatched by type |
| `public UncertainValue2(double v, Map<String,Double> sigmas)` | classmethod `from_sigmas(cls, v: float, sigmas: dict[str, float])` | R4 |

### Public static methods

| Java signature | Python signature | Notes |
|---|---|---|
| `static public UncertainValue2 asUncertainValue2(Number n)` | `@staticmethod def asUncertainValue2(n) -> UncertainValue2` | |
| `static public UncertainValue2 createGaussian(double v, String source)` | `@staticmethod def createGaussian(v: float, source: str) -> UncertainValue2` | |
| `static public UncertainValue2 add(Collection<? extends Number> uvs)` | `@staticmethod def add_collection(uvs) -> UncertainValue2` | R4 |
| `static public UncertainValue2 add(Number[] uvs)` | `@staticmethod def add_array(uvs) -> UncertainValue2` | R4 |
| `static public UncertainValue2 add(double a, Number na, double b, Number nb)` | `@staticmethod def add_dadb(a: float, na, b: float, nb) -> UncertainValue2` | R4 — weighted linear combination |
| `static public Number add(Number v1, double v2)` | `@staticmethod def add_nd(v1, v2: float)` | R4 — returns `Number` (may be `float` or `UncertainValue2`) |
| `static public UncertainValue2 add(double v1, Number v2)` | `@staticmethod def add_dn(v1: float, v2) -> UncertainValue2` | R4 |
| `static public UncertainValue2 add(Number v1, Number v2)` | `@staticmethod def add_nn(v1, v2) -> UncertainValue2` | R4 |
| `static public UncertainValue2 subtract(Number na, Number nb)` | `@staticmethod def subtract(na, nb) -> UncertainValue2` | |
| `public static UncertainValue2 mean(Collection<Number> uvs)` | `@staticmethod def mean_collection(uvs) -> UncertainValue2` | R4 |
| `public static UncertainValue2 mean(Number[] uvs)` | `@staticmethod def mean_array(uvs) -> UncertainValue2` | R4 |
| `public static UncertainValue2 abs(Number n)` | `@staticmethod def abs_n(n) -> UncertainValue2` | R4 — static form; instance form is `abs()` |
| `static public UncertainValue2 weightedMean(Collection<? extends Number> cuv) throws UtilException` | `@staticmethod def weightedMean(cuv) -> UncertainValue2` | Throws `UtilException` |
| `static public UncertainValue2 safeWeightedMean(Collection<? extends Number> cuv)` | `@staticmethod def safeWeightedMean(cuv) -> UncertainValue2` | |
| `public static UncertainValue2 min(Collection<Number> uvs)` | `@staticmethod def min(uvs) -> Optional[UncertainValue2]` | Returns `None` for empty |
| `public static UncertainValue2 max(Collection<UncertainValue2> uvs)` | `@staticmethod def max(uvs) -> Optional[UncertainValue2]` | Returns `None` for empty |
| `static public UncertainValue2 multiply(double v1, Number n2)` | `@staticmethod def multiply_dn(v1: float, n2) -> UncertainValue2` | R4 |
| `static public UncertainValue2 multiply(Number na, Number nb)` | `@staticmethod def multiply_nn(na, nb) -> UncertainValue2` | R4 |
| `static public UncertainValue2 invert(Number nv)` | `@staticmethod def invert(nv) -> UncertainValue2` | |
| `static public UncertainValue2 divide(Number na, Number nb)` | `@staticmethod def divide_nn(na, nb) -> UncertainValue2` | R4 |
| `static public UncertainValue2 divide(double a, Number nb)` | `@staticmethod def divide_dn(a: float, nb) -> UncertainValue2` | R4 |
| `static public UncertainValue2 divide(Number na, double b)` | `@staticmethod def divide_nd(na, b: float) -> UncertainValue2` | R4 |
| `static public UncertainValue2 exp(Number nx)` | `@staticmethod def exp(nx) -> UncertainValue2` | |
| `static public UncertainValue2 log(Number nx)` | `@staticmethod def log(nx) -> UncertainValue2` | |
| `static public UncertainValue2 pow(Number n1, double n)` | `@staticmethod def pow(n1, n: float) -> UncertainValue2` | |
| `public static UncertainValue2 sqrt(Number uv)` | `@staticmethod def sqrt_n(uv) -> UncertainValue2` | R4 — static form; instance form is `sqrt()` |
| `public static UncertainValue2[] quadratic(Number na, Number nb, Number nc)` | `@staticmethod def quadratic(na, nb, nc) -> Optional[list[UncertainValue2]]` | Returns `None` if discriminant < 0 |
| `static public UncertainValue2 sqr(UncertainValue2 uv)` | `@staticmethod def sqr(uv: UncertainValue2) -> UncertainValue2` | |
| `static public UncertainValue2 negate(Number n)` | `@staticmethod def negate(n) -> UncertainValue2` | |
| `static public UncertainValue2 atan(UncertainValue2 uv)` | `@staticmethod def atan(uv: UncertainValue2) -> UncertainValue2` | |
| `static public UncertainValue2 atan2(UncertainValue2 y, UncertainValue2 x)` | `@staticmethod def atan2(y: UncertainValue2, x: UncertainValue2) -> UncertainValue2` | |
| `static public UncertainValue2 nonNegative(UncertainValue2 uv)` | `@staticmethod def nonNegative(uv: UncertainValue2) -> UncertainValue2` | |
| `static public double uncertainty(Number n)` | `@staticmethod def uncertainty_n(n) -> float` | R4 — static form; instance form is `uncertainty()` |
| `static public double fractionalUncertainty(Number n)` | `@staticmethod def fractionalUncertainty_n(n) -> float` | R4 — static form |
| `static public UncertainValue2 valueOf(Number n)` | `@staticmethod def valueOf(n) -> UncertainValue2` | |
| `public static UncertainValue2[] normalize(UncertainValue2[] vals)` | `@staticmethod def normalize(vals: list[UncertainValue2]) -> list[UncertainValue2]` | Uses JamaMatrix |
| `static public String format(NumberFormat nf, Number n)` | `@staticmethod def format_n(nf: Callable[[float], str], n) -> str` | R4 — static form; instance form is `format(nf)` |

### Public instance methods

| Java signature | Python signature | Notes |
|---|---|---|
| `public UncertainValue2 clone()` | `def clone(self) -> UncertainValue2` | Also `__copy__` |
| `public String toString()` | `def __str__(self) -> str` + `def toString(self) -> str` | R2 — dunder + named alias |
| `public String toLongString()` | `def toLongString(self) -> str` | |
| `public String format(NumberFormat nf)` | `def format(self, nf: Callable[[float], str]) -> str` | |
| `public String formatLong(NumberFormat nf)` | `def formatLong(self, nf: Callable[[float], str]) -> str` | |
| `public String format(String src, NumberFormat nf)` | `def format_src(self, src: str, nf: Callable[[float], str]) -> str` | R4 — distinct from format(nf) |
| `public void assignComponent(String name, double sigma)` | `def assignComponent(self, name: str, sigma: float) -> None` | |
| `public void assignComponents(Map<String,Double> comps)` | `def assignComponents(self, comps: dict[str, float]) -> None` | |
| `public double getComponent(String src)` | `def getComponent(self, src: str) -> float` | Returns 0.0 if not present |
| `public double getFractional(String src)` | `def getFractional(self, src: str) -> float` | |
| `public String formatComponent(String comp, NumberFormat nf)` | `def formatComponent_nf(self, comp: str, nf: Callable[[float], str]) -> str` | R4 |
| `public String formatComponent(String comp, int dec)` | `def formatComponent_dec(self, comp: str, dec: int) -> str` | R4 |
| `public boolean hasComponent(String src)` | `def hasComponent(self, src: str) -> bool` | |
| `public Map<String,Double> getComponents()` | `def getComponents(self) -> dict[str, float]` | Returns unmodifiable view — return copy |
| `public Set<String> getComponentNames()` | `def getComponentNames(self) -> frozenset[str]` | |
| `public void renameComponent(String oldName, String newName) throws EPQException` | `def renameComponent(self, oldName: str, newName: str) -> None` | Raises EPQException |
| `public double doubleValue()` | `def doubleValue(self) -> float` + `def __float__(self) -> float` | From Number |
| `public float floatValue()` | `def floatValue(self) -> float` | |
| `public int intValue()` | `def intValue(self) -> int` + `def __int__(self) -> int` | |
| `public long longValue()` | `def longValue(self) -> int` | |
| `public byte byteValue()` | `def byteValue(self) -> int` | |
| `public short shortValue()` | `def shortValue(self) -> int` | |
| `public boolean isUncertain()` | `def isUncertain(self) -> bool` | |
| `public double uncertainty()` | `def uncertainty(self) -> float` | Instance form |
| `public double uncertainty(Collection<String> comps)` | `def uncertainty_comps(self, comps) -> float` | R4 — JAVA-BUG-1 |
| `public double uncertainty(Correlations corr)` | `def uncertainty_corr(self, corr: Correlations) -> float` | R4 |
| `public double variance()` | `def variance(self) -> float` | |
| `public double variance(Correlations corr)` | `def variance_corr(self, corr: Correlations) -> float` | R4 |
| `public double fractionalUncertainty()` | `def fractionalUncertainty(self) -> float` | Instance form |
| `public UncertainValue2 fractionalUncertaintyU()` | `def fractionalUncertaintyU(self) -> UncertainValue2` | |
| `public boolean isNaN()` | `def isNaN(self) -> bool` | |
| `public int hashCode()` | `def __hash__(self) -> int` + `def hashCode(self) -> int` | R2 |
| `public boolean equals(Object obj)` | `def __eq__(self, obj) -> bool` + `def equals(self, obj) -> bool` | R2 |
| `public boolean equals(UncertainValue2 other, double tolerance)` | `def equals_tol(self, other: UncertainValue2, tolerance: float) -> bool` | R4 |
| `public int compareTo(UncertainValue2 o)` | `def compareTo(self, o: UncertainValue2) -> int` + `def __lt__` etc. | R2 — also implement `__lt__`, `__le__`, `__gt__`, `__ge__` |
| `public boolean lessThan(UncertainValue2 uv2)` | `def lessThan(self, uv2: UncertainValue2) -> bool` | |
| `public boolean greaterThan(UncertainValue2 uv2)` | `def greaterThan(self, uv2: UncertainValue2) -> bool` | |
| `public boolean lessThanOrEqual(UncertainValue2 uv2)` | `def lessThanOrEqual(self, uv2: UncertainValue2) -> bool` | |
| `public boolean greaterThanOrEqual(UncertainValue2 uv2)` | `def greaterThanOrEqual(self, uv2: UncertainValue2) -> bool` | |
| `public UncertainValue2 abs()` | `def abs(self) -> UncertainValue2` | Instance form |
| `public UncertainValue2 sqrt()` | `def sqrt(self) -> UncertainValue2` | Instance form |
| `public UncertainValue2 reduced(String name)` | `def reduced(self, name: str) -> UncertainValue2` | |

---

## Private / protected members

| Java | Python |
|---|---|
| `private static transient long sDefIndex = 0` | `_sDefIndex: int = 0` — class-level counter |
| `private static final long serialVersionUID` | Discard |
| `private final double mValue` | `self._mValue: float` |
| `private final TreeMap<String, Double> mSigmas` | `self._mSigmas: dict[str, float]` — kept sorted via `sorted()` on access |
| `static private UncertainValue2 toUV2(Number num)` | `@staticmethod def _toUV2(num) -> UncertainValue2` |

---

## Overloaded methods (split plan)
`add` has six overloads resolved by argument types — split into `add_collection`, `add_array`, `add_dadb`, `add_nd`, `add_dn`, `add_nn`. A unified `add` dispatcher may be provided but must document ambiguity hazards.

`format` has instance and static forms; two instance overloads (`format(nf)` and `format(src, nf)`) split as `format` / `format_src`.

`formatComponent` splits as `formatComponent_nf` / `formatComponent_dec`.

`uncertainty` has instance and static forms plus two instance overloads — split as `uncertainty`, `uncertainty_comps`, `uncertainty_corr`, `uncertainty_n`.

`multiply` splits as `multiply_dn` / `multiply_nn`.

`divide` splits as `divide_nn` / `divide_dn` / `divide_nd`.

Constructor overloads: `__init__` dispatches on argument types (`str` second arg → named-source path; `dict` second arg unavailable in `__init__` — use `from_sigmas` classmethod).

---

## Mutable-output methods
None. `assignComponent` and `assignComponents` mutate `self` but do not accept caller-supplied mutable arrays.

---

## Touchpoints into Jama / javax.swing / java.awt / java.io
- `Jama.Matrix` — used in `normalize()` and `atan2()`. Import `JamaMatrix` from `_epq_compat`.
- `gov.nist.microanalysis.EPQLibrary.EPQException` — `renameComponent`. Import from `_epq_compat`.
- No `javax.swing` or `java.awt`.

---

## Abstract class strategy
Not applicable. `UncertainValue2` is declared `final public class`.

---

## Java-specific translation decisions

| Java pattern | Python translation | Rule |
|---|---|---|
| `extends Number` | Inherit from `numbers.Number` or provide `__float__`, `__int__` | Numeric ABC |
| `implements Comparable<UncertainValue2>` | Implement `__lt__`, `__le__`, `__gt__`, `__ge__` + `compareTo()` | R2 |
| `TreeMap<String, Double> mSigmas` | `dict[str, float]` — sort keys when output order matters | Java `TreeMap` is always-sorted; Python dict preserves insertion order. Call `sorted(self._mSigmas)` where sorted iteration is required. |
| `NumberFormat` parameters | `Callable[[float], str]` | R10 deviation |
| `Jama.Matrix` in `normalize` | `JamaMatrix` | R3 |
| `++sDefIndex` (static counter) | `UncertainValue2._sDefIndex += 1` | Not thread-safe — matches Java |
| Inner class `Correlations` | Nested class `UncertainValue2.Correlations` + module-level alias `Correlations = UncertainValue2.Correlations` | R2 inner-class alias |

---

## Suspected Java bugs

**JAVA-BUG-1 — `uncertainty(Collection<String> comps)` omits the squared sum.**
Java source line: `sum2 += getComponent(comp);`
Sums the raw component values, not their squares. A correct L2 norm would be `sum2 += getComponent(comp)**2`. The return `Math.sqrt(sum2)` then produces sqrt(sum) rather than sqrt(sum-of-squares).
Disposition: **Preserve** — literal port copies the un-squared sum; `uncertainty_comps_strict` companion computes the correct value.

**JAVA-BUG-2 — `equals(UncertainValue2 other, double tolerance)` NullPointerException.**
Java source line: `if (Math.abs(mSigmas.get(key) - other.mSigmas.get(key)) >= tolerance)`
`mSigmas.get(key)` returns `null` (unboxed) when `key` is present in `other` but not `this`, causing a NullPointerException.
Disposition: **Preserve** in literal — `_mSigmas.get(key, 0.0)` used in the strict variant.

---

## Static init order
`ONE`, `ZERO`, `NaN`, `POSITIVE_INFINITY`, `NEGATIVE_INFINITY`, `MAX_VALUE` are initialized via the constructor — a self-referential class-body assignment. In Python, initialize these after the class definition:

```python
UncertainValue2.ONE = UncertainValue2(1.0)
UncertainValue2.ZERO = UncertainValue2(0.0)
# etc.
```

`sDefIndex` is a class-level counter; initialize to `0` in the class body.

---

## Thread safety
`sDefIndex` is a shared mutable class variable incremented without synchronization — not thread-safe. `mSigmas` is a `final` field but its `TreeMap` is mutable; `assignComponent` mutates it without synchronization.
