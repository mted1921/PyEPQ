# Java → Python 3 Conversion Problem List
## Scope: `gov.nist.microanalysis.Utility`

This document catalogues conversion problems found across the Utility package,
organized by severity. It is intended to be iterated upon as more files are
inspected, and to serve as a briefing document for a coding agent carrying out
the conversion.

Severity scale:
- **CRITICAL** — blocks conversion entirely without a design decision first
- **HIGH** — significant rework; no 1-to-1 mapping exists
- **MEDIUM** — mechanical translation with careful attention; common pitfalls
- **LOW** — trivial; drop or rename with no logic change

---

## Global Issues (apply across all files)

### CRITICAL

#### G-C1 — External library: `Jama.Matrix` / `Jama.SingularValueDecomposition`
- **Files affected**: `Math2.java`, `UncertainValue2.java`, `LevenbergMarquardt2.java`,
  `LevenbergMarquardtConstrained.java`, `LevenbergMarquardtParameterized.java`,
  `LinearLeastSquares.java`, `LinearLeastSquaresMS.java`
- **Problem**: Jama is a Java-only linear algebra library. All matrix construction,
  multiplication, decomposition, and solve operations must be re-expressed using
  `numpy` / `scipy`.
- **API differences**:
  - `new Matrix(n, m)` → `np.zeros((n, m))`
  - `matrix.times(other)` → `matrix @ other`
  - `matrix.solve(b)` → `np.linalg.solve(matrix, b)`
  - `new SingularValueDecomposition(m)` → `np.linalg.svd(m, full_matrices=False)`
  - `svd.getV()`, `svd.getS()`, `svd.getU()` → unpacked tuple from `np.linalg.svd`
- **Action**: Before converting any file that uses Jama, define a thin numpy-backed
  shim or convert all usages inline. Recommend inline conversion.

#### G-C2 — Java Swing/AWT GUI classes
- **Files affected**: `AutoComplete.java`, `ComboBoxCellEditor.java`,
  `EachRowEditor.java`, `ElementComboBoxModel.java`, `ElementTreePanel.java`,
  `HtmlSelection.java`, `PrintUtilities.java`, `SpectrumPropertiesTableModel.java`
- **Problem**: These files are built entirely on `javax.swing` and `java.awt`. There
  is no 1-to-1 Python equivalent. Conversion requires choosing a GUI framework
  (PyQt6, tkinter, wxPython) and rewriting from scratch.
- **Action**: Exclude from the first conversion pass. Port computational classes
  first. Revisit GUI classes as a separate workstream once the backend is stable.

---

### HIGH

#### G-H1 — Method overloading
- **Files affected**: All files with multiple constructors or same-named methods
  with different signatures.
- **Examples**:
  - `Integrator()` and `Integrator(double tol)` — two constructors
  - `Math2.initializeRandom()` and `Math2.initializeRandom(long seed)`
  - `UtilException(String)`, `UtilException(String, Throwable)`,
    `UtilException(Throwable)`, `UtilException()`
- **Problem**: Python does not support method overloading. All variants must be
  collapsed into one method using default parameters or `*args`/`**kwargs`.
- **Pattern**:
  ```python
  # Java: Integrator() and Integrator(double tol)
  def __init__(self, tol: float = 1e-6):
      ...

  # Java: initializeRandom() and initializeRandom(long seed)
  def initialize_random(seed: int | None = None):
      ...
  ```
- **Risk**: Where overloads differ in type (not just presence of argument),
  use `isinstance` checks or separate functions with clear names.

#### G-H2 — Mutable array output parameters
- **Files affected**: `AdaptiveRungeKutta.java`, `Integrator.java`, `Math2.java`,
  and most numerical classes.
- **Problem**: Java routinely passes arrays as output parameters, mutating them
  in-place (e.g., `derivatives(double x, double[] y, double[] dydx)` writes
  results into `dydx`). Python lists and numpy arrays are also mutable and
  passed by reference, so the mechanics work — but the idiom is unnatural and
  error-prone if the caller forgets to pre-allocate.
- **Recommendation**: Where the output array is always the same size, return a
  numpy array instead. Reserve in-place mutation only where performance requires
  it (tight numerical loops).
  ```python
  # Java: void derivatives(double x, double[] y, double[] dydx)
  def derivatives(self, x: float, y: np.ndarray) -> np.ndarray:
      dydx = np.empty_like(y)
      ...
      return dydx
  ```
- **Caveat**: `AdaptiveRungeKutta` and `Integrator` rely on in-place mutation
  for correctness — change carefully and test numerically.

#### G-H3 — Abstract classes and interfaces → `abc.ABC` / `typing.Protocol`
- **Files affected**: `AdaptiveRungeKutta.java`, `Integrator.java`,
  `LevenbergMarquardt2.FitFunction`, `LevenbergMarquardt2.AutoPartialsFitFunction`
- **Problem**: Java abstract classes and interfaces map to Python ABCs or Protocols,
  but the semantics differ:
  - Java enforces abstract method implementation at compile time; Python enforces
    at instantiation time.
  - Java interfaces become `typing.Protocol` (structural subtyping) or `abc.ABC`
    (nominal subtyping) depending on whether the codebase checks `isinstance`.
- **Pattern**:
  ```python
  from abc import ABC, abstractmethod

  class AdaptiveRungeKutta(ABC):
      @abstractmethod
      def derivatives(self, x: float, y: np.ndarray, dydx: np.ndarray) -> None: ...
  ```

#### G-H4 — Inner interfaces and anonymous inner classes
- **Files affected**: `LevenbergMarquardt2.java` (`FitFunction` inner interface,
  `AutoPartialsFitFunction` inner abstract class), `Pair.java` (`Comparator`
  anonymous classes in `compareA()` / `compareB()`).
- **Problem**: Python supports nested classes but anonymous classes don't exist.
  `Comparator` anonymous implementations become lambdas or `functools.cmp_to_key`.
- **Pattern**:
  ```python
  # Java anonymous Comparator -> Python key function
  def compare_a(pair): return pair.first
  pairs.sort(key=compare_a)
  ```

#### G-H5 — `java.awt.event` callbacks (`ActionListener`, `ActionEvent`)
- **Files affected**: `LevenbergMarquardt2.java`
- **Problem**: The Levenberg-Marquardt optimizer fires `ActionEvent` via registered
  `ActionListener` objects to report iteration progress. This is a Swing event
  pattern with no direct Python equivalent.
- **Recommendation**: Replace with a plain Python callback: accept an optional
  `on_iteration: Callable[[int, float], None] = None` parameter.

---

### MEDIUM

#### G-M1 — Java generics → Python type hints
- **Files affected**: `Pair<A,B>`, `UncertainValue2` (uses `TreeMap<String, Double>`),
  `LevenbergMarquardt2` (uses `ArrayList<ActionListener>`).
- **Problem**: Java generics are erased at runtime; Python type hints are also
  not enforced at runtime. The conversion is mostly mechanical, but wildcard
  types (`<?>`, `<? extends T>`) have no direct equivalent.
- **Pattern**:
  ```python
  from typing import TypeVar, Generic
  A = TypeVar('A')
  B = TypeVar('B')
  class Pair(Generic[A, B]):
      def __init__(self, first: A, second: B): ...
  ```
- **Simpler alternative**: `Pair` can become a `dataclass` or `NamedTuple` for
  most uses.

#### G-M2 — Java collections → Python built-ins
| Java | Python |
|------|--------|
| `ArrayList<T>` | `list` |
| `HashMap<K,V>` | `dict` |
| `TreeMap<K,V>` | `dict` (Python 3.7+ preserves insertion order) or `sortedcontainers.SortedDict` |
| `HashSet<T>` | `set` |
| `TreeSet<T>` | `sortedcontainers.SortedSet` |
| `Collection<T>` | `list` or `Iterable` |
- **Risk**: `TreeMap`/`TreeSet` guarantee sorted order on every operation.
  Plain `dict`/`set` do not sort by key. `UncertainValue2` uses `TreeMap<String, Double>`
  for named uncertainties — if sorted iteration matters, use `sortedcontainers`.

#### G-M3 — `Comparable` interface → dunder methods
- **Files affected**: `UncertainValue2` implements `Comparable<UncertainValue2>`.
- **Pattern**:
  ```python
  from functools import total_ordering
  @total_ordering
  class UncertainValue2:
      def __eq__(self, other): ...
      def __lt__(self, other): ...
  ```

#### G-M4 — `extends Number` → numeric dunder methods
- **Files affected**: `UncertainValue2 extends Number`
- **Problem**: Java's `Number` abstract class requires `intValue()`, `doubleValue()`,
  `floatValue()`, `longValue()`. Python equivalent is implementing `__float__`,
  `__int__`, and registering with `numbers.Number` ABC if needed.
- **Pattern**:
  ```python
  import numbers
  class UncertainValue2(numbers.Number):
      def __float__(self): return self.mValue
      def __int__(self): return int(self.mValue)
  ```

#### G-M5 — `Double` special values
| Java | Python |
|------|--------|
| `Double.NaN` | `float('nan')` or `math.nan` |
| `Double.MAX_VALUE` | `sys.float_info.max` |
| `Double.POSITIVE_INFINITY` | `math.inf` |
| `Double.NEGATIVE_INFINITY` | `-math.inf` |
| `Double.isNaN(x)` | `math.isnan(x)` |

#### G-M6 — `Math.*` → `math` / `numpy`
- `Math.abs(x)` → `abs(x)` or `np.abs(x)`
- `Math.max(a,b)` / `Math.min(a,b)` → `max(a,b)` / `min(a,b)`
- `Math.sqrt(x)` → `math.sqrt(x)` or `np.sqrt(x)`
- `Math.pow(x,y)` → `x ** y`
- `Math.round(x)` → `round(x)` (**caution**: Java rounds half-up; Python rounds
  half-to-even — results may differ for .5 cases)
- `Math.PI`, `Math.E` → `math.pi`, `math.e`

#### G-M7 — `System.arraycopy` → slice assignment
- `System.arraycopy(src, 0, dst, 0, n)` → `dst[:n] = src[:n]`
- With numpy: `np.copyto(dst[:n], src[:n])` or `dst[:n] = src[:n]`

#### G-M8 — Checked exceptions → unchecked
- **Files affected**: `AdaptiveRungeKutta`, `Integrator`, `LevenbergMarquardt2`,
  `LinearLeastSquares`
- Java `throws UtilException` declarations disappear in Python. The exception
  classes themselves (`UtilException`, `EPQException`) translate directly:
  ```python
  class UtilException(Exception):
      pass
  ```
- All `try/catch` blocks become `try/except`. Callers that currently ignore
  checked exceptions will silently swallow errors — audit each catch site.

#### G-M9 — `final` fields → enforced-by-convention immutability
- `private final double mTolerance` → `self._tolerance: float` with no
  enforcement. For truly immutable data classes, consider `@dataclass(frozen=True)`.

#### G-M10 — Package structure → Python module structure
- `gov.nist.microanalysis.Utility` → a Python package directory `utility/`
  with an `__init__.py`.
- Class-per-file structure can be preserved or classes can be grouped into
  thematic modules (e.g., `numerical.py`, `statistics.py`).
- Circular import risk: `Math2` imports from `EPQLibrary.EPQException` —
  Python circular imports require careful ordering or lazy imports.

---

### LOW

#### G-L1 — Drop with no replacement
| Java | Action |
|------|--------|
| `serialVersionUID` | Drop — Python has `pickle` but no version field |
| `transient` | Drop — use `__getstate__`/`__setstate__` if pickling matters |
| `@Override` | Drop — Python has no equivalent annotation |
| `final` local variables | Drop — no enforcement in Python |
| `private`/`protected` | Rename to `_name` (protected) or `__name` (private) by convention |

#### G-L2 — Naming conventions
- Java uses camelCase for methods/fields; Python uses snake_case.
- Java `mFieldName` Hungarian prefix → drop the `m`, use `snake_case`.
- Example: `mTolerance` → `_tolerance`, `getNSaved()` → `get_n_saved()` or
  expose as a `@property`.

---

## Per-File Notes

### `AdaptiveRungeKutta.java`
- Abstract class with one abstract method (`derivatives`) — straightforward ABC.
- All numerical work uses raw `double[]` arrays in tight loops — strong candidate
  for numpy vectorization, but the loop structure is inherently sequential (ODE
  stepping). Keep as loops or use `scipy.integrate.ode` / `solve_ivp` instead
  of porting the algorithm directly.
- `clearWorkspace()` sets arrays to `null` to free memory — not needed in Python
  (GC handles it); can be omitted or replaced with `del`.
- `sign(magnitude, sign)` private helper → `math.copysign(magnitude, sign)`.

### `Integrator.java`
- Thin wrapper over `AdaptiveRungeKutta` — converts to Python cleanly once
  the base class is ported.
- `Double.NaN` return on exception → `float('nan')`.
- Single abstract method `getValue(double x)` → `@abstractmethod def get_value(self, x: float) -> float`.

### `Math2.java`
- Large static utility class — becomes a Python module (`math2.py`) with
  module-level functions, not a class. Static constants become module globals.
- `public static Random rgen` mutable global → `_rgen = random.Random()` at
  module level; thread-safety concern if used from multiple threads.
- Jama `Matrix` usage for any linear algebra methods — see G-C1.
- `initializeRandom()` overload — see G-H1.
- 3D vector helpers (`v3`, `normalize`, `dot`, `cross`) → numpy operations;
  these are the highest-value candidates for numpy vectorization.

### `UncertainValue2.java`
- Complex class: implements `Comparable`, extends `Number`, uses `TreeMap`
  for named uncertainty components, has many static factory methods and
  mathematical operators.
- `TreeMap<String, Double>` for `mSigmas` — sorted order may matter for
  deterministic output; use `sortedcontainers.SortedDict` or verify that
  plain `dict` suffices.
- Static constants `ONE`, `ZERO`, `NaN` etc. — translate directly as class
  attributes.
- Overloaded math operations (add, multiply, etc.) → Python dunder methods
  (`__add__`, `__mul__`, etc.) — high effort but clean result.
- Jama dependency — see G-C1.
- Cross-package dependency on `EPQLibrary.EPQException` — see G-M10.

### `LevenbergMarquardt2.java`
- Inner interface `FitFunction` and inner abstract class `AutoPartialsFitFunction`
  — see G-H3, G-H4.
- `ActionListener`/`ActionEvent` for iteration callbacks — see G-H5.
- Jama `Matrix` and `SingularValueDecomposition` throughout — see G-C1.
- Consider replacing entire implementation with `scipy.optimize.least_squares`
  which provides Levenberg-Marquardt natively, exposing the same interface.

### `Pair.java`
- Generic `Pair<A,B>` — simplest conversion target.
- Replace with `@dataclass` or `typing.NamedTuple`:
  ```python
  from dataclasses import dataclass
  from typing import TypeVar, Generic
  A, B = TypeVar('A'), TypeVar('B')
  @dataclass(frozen=True)
  class Pair(Generic[A, B]):
      first: A
      second: B
  ```
- Anonymous `Comparator` in `compareA`/`compareB` → key functions (see G-H4).

### `UtilException.java`
- Trivial conversion:
  ```python
  class UtilException(Exception):
      pass
  ```
- Multiple constructors collapse naturally since Python `Exception.__init__`
  already accepts `(message)` and `(message, cause)` via chaining (`raise X from Y`).

---

## Recommended Conversion Order (Utility package)

Port in this order to minimize broken dependencies at each step:

1. `UtilException` — no dependencies
2. `Pair` — no dependencies
3. `Math2` — depends only on Jama (numpy replacement) and `EPQException`
4. `AdaptiveRungeKutta` → `Integrator` — depends only on `UtilException`
5. `LinearLeastSquares` / `LinearLeastSquaresMS` — Jama only
6. `LevenbergMarquardt2` → `LevenbergMarquardtConstrained` → `LevenbergMarquardtParameterized`
7. `UncertainValue2` — depends on Jama and `EPQException`
8. Remaining numerical classes (`Histogram`, `DescriptiveStatistics`,
   `FindRoot`, `Simplex`, `MCIntegrator`, etc.)
9. GUI classes — defer or omit

---

*Last updated: 2026-05-22. Files inspected: `AdaptiveRungeKutta.java`,
`Integrator.java`, `Math2.java` (partial), `UncertainValue2.java` (partial),
`LevenbergMarquardt2.java` (partial), `Pair.java`, `UtilException.java`.*
