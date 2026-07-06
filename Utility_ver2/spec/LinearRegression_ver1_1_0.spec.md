# LinearRegression Conversion Spec

## Java class
`gov.nist.microanalysis.Utility.LinearRegression`

Source: `src/gov/nist/microanalysis/Utility/LinearRegression.java`

---

## Inbound dependencies (Java imports)
- `Jama.Matrix` — used for the 2×2 covariance matrix; import `JamaMatrix` from `_epq_compat`
- `gov.nist.microanalysis.Utility.LazyEvaluate` — used to memoize the fitted `Line` result; import from sibling port
- `gov.nist.microanalysis.Utility.Math2` — `Math2.sqr`, `Math2.gammq`; import from sibling port

**Note**: The dependency graph dot file shows `LinearRegression` → `LazyEvaluate` and `LinearRegression` → `Math2` but omits the `Jama.Matrix` dependency (Jama is external, not in the Utility package). The spec must include `JamaMatrix`.

---

## Outbound dependents (callers of public methods)
Not audited (in_degree = 0 — no Utility-package callers; callers are in EPQLibrary or external).

---

## Public API surface

### Inner class: `LinearRegression.Line`

A simple value object holding slope and intercept.

| Java signature | Python signature | Notes |
|---|---|---|
| `public Line(double slope, double intercept)` | `__init__(self, slope: float, intercept: float)` | |
| `public Line(double x0, double y0, double x1, double y1)` | classmethod `from_two_points(cls, x0: float, y0: float, x1: float, y1: float) -> Line` | R4 |
| `public double getXIntercept()` | `def getXIntercept(self) -> float` | `-intercept / slope` |
| `public double computeY(double x)` | `def computeY(self, x: float) -> float` | `intercept + slope * x` |
| `public double computeX(double y)` | `def computeX(self, y: float) -> float` | `(y - intercept) / slope` |
| `public double getSlope()` | `def getSlope(self) -> float` | |
| `public double getIntercept()` | `def getIntercept(self) -> float` | |

Module-level alias: `Line = LinearRegression.Line`. (R2)

### `LinearRegression` constructor and methods

| Java signature | Python signature | Notes |
|---|---|---|
| `public LinearRegression()` | `__init__(self)` | Calls `clear()` |
| `public void clear()` | `def clear(self) -> None` | Resets all accumulators and invalidates the cached line |
| `public void setData(double[] x, double[] y)` | `def setData(self, x: list[float], y: list[float]) -> None` | Clears then calls `addData` |
| `public void addData(double[] x, double[] y)` | `def addData(self, x: list[float], y: list[float]) -> None` | |
| `public void addDatum(double x, double y)` | `def addDatum_xy(self, x: float, y: float) -> None` | R4 — unit weight |
| `public void addDatum(double x, double y, double dy)` | `def addDatum_xye(self, x: float, y: float, dy: float) -> None` | R4 — weighted |
| `public void removeDatum(double x, double y)` | `def removeDatum_xy(self, x: float, y: float) -> None` | R4 |
| `public void removeDatum(double x, double y, double dy)` | `def removeDatum_xye(self, x: float, y: float, dy: float) -> None` | R4 |
| `public double getSlope()` | `def getSlope(self) -> float` | Triggers lazy recompute |
| `public double getIntercept()` | `def getIntercept(self) -> float` | Triggers lazy recompute |
| `public double getXIntercept()` | `def getXIntercept(self) -> float` | |
| `public double computeY(double x)` | `def computeY(self, x: float) -> float` | |
| `public double computeX(double y)` | `def computeX(self, y: float) -> float` | |
| `public Matrix covariance()` | `def covariance(self) -> Optional[JamaMatrix]` | `None` until 2+ data points with nonzero `den` |
| `public double correlation()` | `def correlation(self) -> float` | `-mSxx / sqrt(mS * mSxx)` |
| `public double chiSquared()` | `def chiSquared(self) -> float` | Clamped to `≥ 0.0` |
| `public double goodnessOfFit()` | `def goodnessOfFit(self) -> float` | Uses `Math2.gammq` |
| `public double getRSquared()` | `def getRSquared(self) -> float` | |
| `public int getCount()` | `def getCount(self) -> int` | |
| `public Line getResult()` | `def getResult(self) -> Line` | Returns the cached `Line` |
| `public String toString()` | `def __str__(self) -> str` + `def toString(self) -> str` | R2 |

---

## Private / protected members

| Java | Python |
|---|---|
| `private double mS` | `self._mS: float` |
| `private double mSxx` | `self._mSxx: float` |
| `private double mSx` | `self._mSx: float` |
| `private double mSxy` | `self._mSxy: float` |
| `private double mSy` | `self._mSy: float` |
| `private double mSyy` | `self._mSyy: float` |
| `private int mCount` | `self._mCount: int` |
| `private LazyEvaluate<Line> mLine` | `self._mLine: LazyEvaluate` (parameterized with `Line`) |
| `private Matrix mCovariance` | `self._mCovariance: Optional[JamaMatrix] = None` |

---

## Overloaded methods (split plan)

`addDatum` has two overloads → `addDatum_xy` / `addDatum_xye`.

`removeDatum` has two overloads → `removeDatum_xy` / `removeDatum_xye`.

`Line` constructor has two overloads: `__init__(slope, intercept)` and `from_two_points`.

---

## Mutable-output methods

`setData`, `addData`, `addDatum_*`, `removeDatum_*`, `clear` mutate accumulator fields and invalidate `_mLine` cache. `covariance()` computes and caches `_mCovariance` on first call.

`addData` and `setData` accept caller-supplied arrays — read-only input; no defensive copy needed (values are immediately accumulated).

---

## Touchpoints into Jama / javax.swing / java.awt / java.io
`Jama.Matrix` — used for the 2×2 covariance matrix inside the lazy `mLine.compute()` closure. Import `JamaMatrix` from `_epq_compat`.

---

## Abstract class strategy
Not applicable. Concrete class.

### `LazyEvaluate` anonymous subclass

Java source creates an anonymous `LazyEvaluate<Line>` instance with an overridden `compute()` method. In Python, subclass `LazyEvaluate` with a local class or pass a `Callable` to a `LazyEvaluate` factory:

```python
def _make_line_lazy(self):
    lr = self
    class _LineLazy(LazyEvaluate):
        def compute(self):
            return lr._compute_line()
    return _LineLazy()
```

Or if `LazyEvaluate` supports a `Callable` constructor (check the spec), pass `lambda: self._compute_line()` directly.

---

## Suspected Java bugs

**JAVA-NOTE-1 — `correlation()` divides by `sqrt(mS * mSxx)` but mSxx may be zero.**
If all x values are identical, `mSxx = 0.0` and `correlation()` returns `NaN` via division by zero. No guard exists.
Disposition: **Preserve** — port verbatim; the NaN propagation is consistent with Java IEEE-754 behavior.

**JAVA-NOTE-2 — `chiSquared()` asserts non-negative (commented out) then clamps.**
Java source: `// assert res >= 0.0 : ...; return Math.max(0.0, res);`
The assert is commented out, relying on the clamp. Due to floating-point rounding, `res` can be slightly negative for near-perfect fits.
Disposition: **Preserve** — port the clamp `return max(0.0, res)` verbatim; the commented-out assert is discarded.

---

## Static init order
None. All state is instance-level.

---

## Thread safety
Not thread-safe. Accumulator fields and the lazy cache are all mutable. Concurrent `addDatum` / `getSlope` calls on the same instance would produce incorrect results.
