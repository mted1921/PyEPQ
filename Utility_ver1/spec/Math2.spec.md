# Math2 Conversion Spec

## Java class
`gov.nist.microanalysis.Utility.Math2`

Source: `src/gov/nist/microanalysis/Utility/Math2.java`

---

## Inbound dependencies (Java imports)
- `java.text.NumberFormat`
- `java.util.Arrays`
- `java.util.Random`
- `gov.nist.microanalysis.EPQLibrary.EPQException`
- `gov.nist.microanalysis.Utility.FindRoot` (used via anonymous inner class)
- `Jama.Matrix`

---

## Outbound dependents (callers of public methods)

Core foundational library, heavily used throughout EPQ (e.g. `Material`, `Element`,
`AlgorithmClass`, various numerical solvers).

---

## Public API surface

Math2 exposes ~70 public static methods. Listed by category:

- **Constants:** `ORIGIN_3D`, `ONE`, `X_AXIS`, `Y_AXIS`, `Z_AXIS`, `MINUS_X_AXIS`, `MINUS_Y_AXIS`, `MINUS_Z_AXIS`, `SQRT_PI`, `rgen`
- **RNG:** `initializeRandom`, `expRand`, `randomDir`
- **Element-wise arithmetic:** `sqr`, `add`, `plus`, `minus`, `multiply`, `divide`, `ebeDivide`, `negative`, `abs`, `cubeRoot`
- **In-place arithmetic:** `addInPlace`, `plusEquals`, `timesEquals`, `divideEquals`
- **Reductions / metrics:** `sum`, `max`, `min`, `pNorm`, `infinityNorm`, `magnitude`, `distance`, `distanceSqr`
- **Vector geometry:** `cross`, `dot`, `normalize`, `isUnitVector`, `angleBetween`, `pointBetween`
- **Special functions:** `erf`, `erfc`, `gammq`, `gammap`, `gammaln`, `chiSquaredConfidenceLevel`, `Legendre`, `li`
- **Polynomial / root finding:** `quadraticSolver`, `cubicSolver`, `cubicSolver2`, `solvePoly`, `findRoot`, `polynomial`, `solveQuadratic`, `solveCubic`
- **Misc:** `bound`, `positive`, `binomialCoefficient`, `slice`, `closestTo`, `approxEquals`, `convolve`, `toString`, `isNaN`, `toContinuedFraction`, `toDecimal`, `toFraction`, `createRowMatrix`, `gcd`, `v3`, `x3`, `y3`, `z3`, `transpose`

---

## Private / protected members
None beyond the static constants listed in the public API surface. Internal
helpers (e.g. anonymous `FindRoot.Function` for `chiSquaredConfidenceLevel`) have
no named Java members.

---

## Overloaded methods (split plan)
- `initializeRandom()` / `initializeRandom(long)` → default argument `seed=None`.
- `expRand()` / `expRand(double)` → default argument `lambda_=1.0`.
- `sum(double[])` / `sum(int[])` → type-preserving single function.
- `max`, `min` (arrays / matrices) → type-preserving single function.
- `plus`, `minus` (vector/vector vs vector/scalar) → split `_vv` / `_vs` with dispatcher.
- `multiply` (scalar/vector vs vector/vector) → split `_sv` / `_vv` with dispatcher.
- `negative` (double[] vs double) → split `negative_arr` / `negative_scalar` with dispatcher.
- `bound` (double vs int/long) → split `bound_double` / `bound_int` with dispatcher.
- `solvePoly(double[])` / `solvePoly(double[], double)` → default argument `y=None`.

---

## Mutable-output methods
- `addInPlace`, `plusEquals`, `timesEquals`, `divideEquals` — all mutate a
  caller-supplied array in place. `_require_mutable_f64` guard required on each.

---

## Touchpoints into Jama / javax.swing / java.awt / java.io
- **`Jama.Matrix`** — use `JamaMatrix` from `_epq_compat` (R3).
- **`FindRoot`** anonymous inner class in `chiSquaredConfidenceLevel` — bridged
  with `scipy.optimize.brentq` (temporary; replace with concrete `FindRoot` subclass
  once `FindRoot` is listed as a dependency).
- **`java.text.NumberFormat`** — `NumberFormat.getInstance()` is replaced by
  Python f-string formatting in `toString`.

---

## Java-specific translation decisions

| Java pattern | Python translation | Rule |
|---|---|---|
| `static` methods | `@staticmethod` inside class body | R2 |
| `double[]` parameters / returns | `F64Array` (numpy float64) | R9 |
| `Jama.Matrix` | `JamaMatrix` from `_epq_compat` | R3 |
| `java.util.Random rgen` | `JavaRandom` from `_epq_compat` | R3 |
| Integer `/` between `int` types | `//` | R7 |
| `Math.round(x)` | `int(math.floor(x + 0.5))` | R7 |
| `Math.signum(0)` edge case | `0.0 if v == 0.0 else math.copysign(1.0, v)` | R8 |
| `plus(double[], double[])` vs `plus(double[], double)` | `plus_vv` / `plus_vs` dispatcher | R4 |
| `negative(double[])` vs `negative(double)` | `negative_arr` / `negative_scalar` dispatcher | R4 |
| `bound(double, double, double)` vs `bound(long, long, long)` | `bound_double` / `bound_int` dispatcher | R4 |
| `new FindRoot.Function() { double function(x) {...} }` (anonymous) | `scipy.optimize.brentq(lambda x: ..., ...)` | R3/bridge |

---

## Suspected Java bugs

- **JAVA-BUG-1** (`abs`): Clamps negatives to zero rather than computing absolute value.
  Preserve buggy behaviour in `abs`; add `abs_real` strict variant.
- **JAVA-BUG-2** (`ebeDivide`): Uses modulo indexing for divisor despite assuming equal length.
  Preserve in `ebeDivide`; add `ebeDivide_strict`.
- **JAVA-BUG-3** (`cubicSolver`): Exact float equality check `h == 0`. Preserve.
- **JAVA-BUG-4** (`toContinuedFraction`): Prints debug output to `stdout`. Preserve
  with optional `verbose` toggle (default `True`).
- **JAVA-BUG-5** (`createRowMatrix`): Returns an N×1 column matrix instead of a 1×N row
  matrix. Preserve buggy behaviour; add `createRowMatrix_strict` for correct shape.
- **JAVA-BUG-6** (`solveCubic`): Wrong Cardano formula in 3-real-roots branch. Preserve.
- **RNG-DEVIATION-1** (`randomDir`): Uses `rgen` instead of `Math.random()` to allow
  sequence determinism. Document deviation.

---

## Static init order

Static final array constants (`ORIGIN_3D`, `ONE`, `X_AXIS`, etc.) are initialised
at class load via `np.array([...])` calls at module level. `rgen = JavaRandom()` is
also module-level. No ordering constraints among them.

---

## Thread safety

`rgen` is shared mutable state on the class. Not thread-safe: concurrent calls to
any method that reads or writes `rgen` (e.g. `expRand`, `randomDir`,
`initializeRandom`) will race. All other methods are stateless and safe.
