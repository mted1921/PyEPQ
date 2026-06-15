# Simplex Conversion Spec

## Java class
`gov.nist.microanalysis.Utility.Simplex`

Source: `src/gov/nist/microanalysis/Utility/Simplex.java`

---

## Inbound dependencies (Java imports)
- `java.util.Arrays` — `Arrays.toString` (in error messages only)
- `java.util.Random` — used in `randomizedStartingPoints` (unseeded, `System.currentTimeMillis()`)
- `gov.nist.microanalysis.Utility.UtilException` — thrown by `perform` and `evaluateFunction`

---

## Outbound dependents (callers of public methods)
- EPQLibrary fitting classes (spectrum fitting, composition refinement)
- All callers subclass `Simplex`, implement `function()`, and call `perform()`
- `regularizedStartingPoints` and `randomizedStartingPoints` are used to build initial simplex

---

## Public API surface

| Java signature | Python signature | Notes |
|---|---|---|
| `Simplex()` | `__init__(self, params=None)` | No-arg + params constructors merged |
| `Simplex(Object[] params)` | `__init__(self, params: Sequence[Any])` | Clones params array |
| `Object[] getParameters()` | `getParameters(self) -> list[Any] \| None` | Returns `mParameters` |
| `abstract double function(double[] x)` | `@abc.abstractmethod function(self, x: F64Array) -> float` | Extension point; no `_` prefix (R1) |
| `static double[][] randomizedStartingPoints(double[] center, double[] scale)` | `randomizedStartingPoints(center, scale, seed=None) -> F64Array` | RNG-DEVIATION-1; seed arg added |
| `static double[][] regularizedStartingPoints(double[] center, double[] scale)` | `regularizedStartingPoints(center, scale) -> F64Array` | Direct port; no deviation |
| `double[] perform(double[][] p)` | `perform(self, p: F64Array) -> F64Array` | SCIPY-DEV-1: uses `scipy.optimize.minimize` |
| *(added)* | `perform_literal(self, p: F64Array) -> F64Array` | Exact Java Nelder-Mead port |
| `int getEvaluationCount()` | `getEvaluationCount(self) -> int` | |
| `double getBestResult()` | `getBestResult(self) -> float` | `mResult` after last `perform` |
| `void setTolerance(double t)` | `setTolerance(self, t: float) -> None` | Clamps to < `LARGEST_TOLERANCE` |
| `double getTolerance()` | `getTolerance(self) -> float` | |
| `void setMaxEvaluations(int n)` | `setMaxEvaluations(self, n: int) -> None` | Clamps to ≥ `MIN_EVALUATIONS` |
| `int getMaxEvaluations()` | `getMaxEvaluations(self) -> int` | |

---

## Private / protected members

| Java | Python |
|---|---|
| `private double[] mPTry` | `self.mPTry: Optional[F64Array]` |
| `private double mTolerance` | `self.mTolerance: float` |
| `private int mMaxEvaluations` | `self.mMaxEvaluations: int` |
| `private double[] mBest` | `self.mBest: Optional[F64Array]` |
| `private double mResult` | `self.mResult: float = float('nan')` |
| `private Object[] mParameters` | `self.mParameters: Optional[list[Any]]` |
| `private int mEvaluationCount` | `self.mEvaluationCount: int` |
| `private double ooze(...)` | `_ooze_literal(self, ...)` |
| `private double evaluateFunction(double[])` | `_evaluateFunction_literal(self, x)` |
| `static final double LARGEST_TOLERANCE = 0.01` | `_LARGEST_TOLERANCE: float = 0.01` |
| `static final int MIN_EVALUATIONS = 100` | `_MIN_EVALUATIONS: int = 100` |

---

## Overloaded methods (split plan)

| Java overloads | Python translation |
|---|---|
| `Simplex()` and `Simplex(Object[] params)` | Single `__init__(self, params=None)` |

---

## Mutable-output methods
- **`perform` / `perform_literal`**: mutates `p` in place (Java contract) — the simplex vertices are updated as the algorithm wanders. `_require_mutable_f64(p, "p")` guard applied (R5).

---

## Touchpoints into Jama / javax.swing / java.awt / java.io
None.

---

## Abstract class strategy
`Simplex` declares `abstract double function(double[] x)`. Direct JPype parity testing is **blocked by M4**. The parity harness uses concrete Python subclasses (e.g. a quadratic bowl `f(x) = x·x`) to verify convergence, then skips the `@needs_java` class with `M4`.

---

## Java-specific translation decisions

| Java pattern | Python translation | Rule |
|---|---|---|
| `abstract public class Simplex` | `class Simplex(abc.ABC)` | R2 |
| `abstract double function(double[] x)` | `@abc.abstractmethod function(self, x: F64Array) -> float` — no `_` prefix | R1 |
| `throw new UtilException(...)` | `raise EPQException(...)` | R2 |
| `new Random(System.currentTimeMillis())` in `randomizedStartingPoints` | `JavaRandom(time.time_ns())` unless `seed` is given | RNG-DEVIATION-1 |
| Java Nelder-Mead (`perform`) | Two variants: `perform` (scipy), `perform_literal` (exact Java) | SCIPY-DEV-1 |
| `p[i][j]` 2D Java array | `p[i, j]` numpy 2D array (F64Array shape `[n+1, n]`) | R2 |
| `assert (rtol < mTolerance)` — rtol undefined when `y[ihi]+y[ilo]==0` | Port guards `if denom != 0.0` before rtol computation | FIX |

---

## Deviations

**RNG-DEVIATION-1** — `randomizedStartingPoints` RNG seed.

Java uses `new Random(System.currentTimeMillis())` — a wall-clock seed, producing different starting points each call. This is intentional (exploration). The port adds an optional `seed` parameter so tests can be deterministic. When `seed=None`, `time.time_ns()` is used to match Java's intent.

**SCIPY-DEV-1** — `perform` uses `scipy.optimize.minimize`.

`scipy.optimize.minimize(method='Nelder-Mead')` evaluates stopping criteria differently from the Java NR implementation and does not mutate all simplex vertices into `p` on completion — only `p[0]` is set to the best result. Use `perform_literal` for exact Java mutation semantics (important for callers that read the final simplex state from `p`).

---

## Constants

| Java | Value | Python |
|---|---|---|
| `DEFAULT_TOLERANCE` | `1.0e-8` | `Simplex.DEFAULT_TOLERANCE` |
| `DEFAULT_EVALUATIONS` | `5000` | `Simplex.DEFAULT_EVALUATIONS` |
| `LARGEST_TOLERANCE` | `0.01` | `Simplex._LARGEST_TOLERANCE` |
| `MIN_EVALUATIONS` | `100` | `Simplex._MIN_EVALUATIONS` |

---

## Suspected Java bugs
None. The `rtol = 2·tol / (|y[ihi]| + |y[ilo]|)` computation is undefined when both `y[ihi]` and `y[ilo]` are 0 (converged exactly to zero). Java produces `NaN` via IEEE-754 division; the conditional `rtol < mTolerance` then evaluates to false, falling through to `tol < mTolerance` which catches it correctly. The port adds an explicit `denom != 0` guard to make this safe without changing behaviour.

---

## Static init order
Constants only; no cross-class static references.

---

## Thread safety
Not thread-safe. `mPTry`, `mBest`, `mResult`, `mEvaluationCount` are all mutable instance state modified during `perform`.
