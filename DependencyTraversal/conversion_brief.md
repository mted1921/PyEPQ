# Java→Python3 Conversion Brief: `gov.nist.microanalysis.Utility`

## Blockers (resolve before converting any file)
- **Jama.Matrix / SingularValueDecomposition** → replace with `numpy`/`scipy` inline.
  `new Matrix(n,m)`→`np.zeros((n,m))`, `.times()`→`@`, `.solve()`→`np.linalg.solve()`,
  `SVD.getU/S/V()`→`np.linalg.svd(full_matrices=False)`.
- **Swing/AWT files** (`AutoComplete`, `ComboBoxCellEditor`, `EachRowEditor`,
  `ElementComboBoxModel`, `ElementTreePanel`, `HtmlSelection`, `PrintUtilities`,
  `SpectrumPropertiesTableModel`) — **skip entirely** in this pass.

## Global Translation Rules

| Java | Python |
|------|--------|
| Overloaded constructors/methods | Single method with default params; `isinstance` if types differ |
| `abstract class` / `interface` | `abc.ABC` + `@abstractmethod` / `typing.Protocol` |
| `implements Comparable<T>` | `@functools.total_ordering` + `__eq__`, `__lt__` |
| `extends Number` | `numbers.Number` + `__float__`, `__int__` |
| Mutable `double[]` output param | Return `np.ndarray`; keep in-place only in hot loops |
| `System.arraycopy(s,0,d,0,n)` | `d[:n] = s[:n]` |
| Inner interface / abstract class | Nested class or top-level ABC |
| Anonymous `Comparator` | Lambda or `key=` function |
| `ActionListener` callbacks | `on_iteration: Callable[[int,float],None] = None` |
| `ArrayList` / `HashMap` / `HashSet` | `list` / `dict` / `set` |
| `TreeMap` / `TreeSet` | `sortedcontainers.SortedDict` / `SortedList` if order matters |
| `Double.NaN/MAX_VALUE/POSITIVE_INFINITY` | `math.nan` / `sys.float_info.max` / `math.inf` |
| `Math.pow(x,y)` | `x**y` |
| `Math.round(x)` | `round(x)` (**half-even**, not half-up — check .5 cases) |
| `Math.*` (abs/max/min/sqrt/sin…) | `math.*` or `np.*` |
| `sign(magnitude, sign)` helper | `math.copysign(magnitude, sign)` |
| `static` utility class | Module-level functions, no class wrapper |
| `static final` constant | Module-level or class-level constant |
| `private final` field | `self._name` (convention only) |
| `mFieldName` Hungarian prefix | Drop `m`, use `snake_case` |
| `serialVersionUID`, `transient`, `@Override`, `final` locals | Drop |
| `private`/`protected` | `__name` / `_name` prefix |
| Checked exceptions (`throws X`) | Drop declaration; keep `try/except`; `UtilException`→`class UtilException(Exception): pass` |
| Package `gov.nist.microanalysis.Utility` | Python package `utility/` with `__init__.py` |

## Per-File Notes

**`UtilException`** — `class UtilException(Exception): pass`. Multiple constructors collapse naturally.

**`Pair<A,B>`** — `@dataclass(frozen=True)` + `Generic[A,B]`. `compareA`/`compareB` → `key=` functions.

**`AdaptiveRungeKutta`** — ABC with abstract `derivatives()`. Consider replacing entire class with `scipy.integrate.solve_ivp`. If porting directly: `clearWorkspace()` → omit; `sign()` → `math.copysign()`.

**`Integrator`** — Thin subclass of `AdaptiveRungeKutta`; ports cleanly once base is done. `Double.NaN` return on exception → `float('nan')`.

**`Math2`** — Becomes module `math2.py` (no class). `rgen` → module-level `random.Random()`. 3D vector helpers → numpy. Overloaded `initializeRandom` → default param `seed=None`.

**`UncertainValue2`** — `@total_ordering`, `numbers.Number`, `__add__`/`__mul__` etc. for operator overloads. `TreeMap<String,Double>` for `mSigmas` → `SortedDict`. Cross-dependency on `EPQLibrary.EPQException` — use lazy import or port `EPQException` first.

**`LevenbergMarquardt2`** — Consider replacing with `scipy.optimize.least_squares` (has LM built-in) rather than porting. If porting: `FitFunction` → ABC, `ActionListener` → callable param.

## Conversion Order
1. `UtilException` → `Pair` → `Math2`
2. `AdaptiveRungeKutta` → `Integrator`
3. `LinearLeastSquares` → `LinearLeastSquaresMS`
4. `LevenbergMarquardt2` → `LevenbergMarquardtConstrained` → `LevenbergMarquardtParameterized`
5. `UncertainValue2`
6. Remaining numerical classes (`Histogram`, `DescriptiveStatistics`, `FindRoot`, `Simplex`, `MCIntegrator`, …)
7. GUI classes — defer
