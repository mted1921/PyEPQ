# Math2_ver1 Conversion Specification (Literal Port)

## 1. Inbound dependencies (Java imports)
* `java.text.NumberFormat`
* `java.util.Arrays`
* `java.util.Random`
* `gov.nist.microanalysis.EPQLibrary.EPQException`
* `gov.nist.microanalysis.Utility.FindRoot` (Used via anonymous inner class)
* `Jama.Matrix`

## 2. Outbound dependents
Utility class heavily used throughout EPQ (e.g. `Material`, `Element`, `AlgorithmClass`, various numerical solvers). Core foundational library.

## 3. Public API surface
* Constants: `ORIGIN_3D`, `ONE`, `X_AXIS`, `Y_AXIS`, `Z_AXIS`, `MINUS_X_AXIS`, `MINUS_Y_AXIS`, `MINUS_Z_AXIS`, `SQRT_PI`, `rgen`
* RNG: `initializeRandom`, `expRand`, `randomDir`
* Element-wise arithmetic: `sqr`, `add`, `plus`, `minus`, `multiply`, `divide`, `ebeDivide`, `negative`, `abs`, `cubeRoot`
* In-place arithmetic: `addInPlace`, `plusEquals`, `timesEquals`, `divideEquals`
* Reductions / Metrics: `sum`, `max`, `min`, `pNorm`, `infinityNorm`, `magnitude`, `distance`, `distanceSqr`
* Vector geometry: `cross`, `dot`, `normalize`, `isUnitVector`, `angleBetween`, `pointBetween`
* Special functions: `erf`, `erfc`, `gammq`, `gammap`, `gammaln`, `chiSquaredConfidenceLevel`, `Legendre`, `li`
* Polynomial / Root finding: `quadraticSolver`, `cubicSolver`, `cubicSolver2`, `solvePoly`, `findRoot`, `polynomial`, `solveQuadratic`, `solveCubic`
* Misc: `bound`, `positive`, `binomialCoefficient`, `slice`, `closestTo`, `approxEquals`, `convolve`, `toString`, `isNaN`, `toContinuedFraction`, `toDecimal`, `toFraction`, `createRowMatrix`, `gcd`, `v3`, `x3`, `y3`, `z3`, `transpose`

## 4. Overloaded methods (split plan)
* `initializeRandom()`, `initializeRandom(long)` -> Default argument `seed=None`.
* `expRand()`, `expRand(double)` -> Default argument `lambda_=1.0`.
* `sum(double[])`, `sum(int[])` -> Type-preserving single function covers both.
* `max`, `min` (arrays/matrices) -> Type-preserving single function covers both.
* `plus`, `minus` (vector/vector vs vector/scalar) -> Split into `_vv` and `_vs`, with dispatcher.
* `multiply` (scalar/vector vs vector/vector) -> Split into `_sv` and `_vv`, with dispatcher.
* `negative` (double[] vs double) -> Split into `negative_arr` and `negative_scalar`, with dispatcher. Semantics differ entirely!
* `bound` (double vs int/long) -> Split into `bound_double` and `bound_int`, with dispatcher. Semantics differ entirely!
* `solvePoly(double[])`, `solvePoly(double[], double)` -> Default argument `y=None`.

## 5. Mutable-output methods
* `addInPlace`, `plusEquals`, `timesEquals`, `divideEquals`. All require `_require_mutable_f64` guard.

## 6. Touchpoints into external libraries
* `Jama.Matrix`: Use `JamaMatrix` from `_epq_compat`.
* `FindRoot`: Simulated inline via `scipy.optimize.brentq` as a temporary bridge for `chiSquaredConfidenceLevel`.
* Number formatting in `toString`: Use Python string formatting/callbacks.

## 7. Suspected Java bugs and disposition
* `JAVA-BUG-1` (`abs`): Clamps negatives to zero rather than computing absolute value. Preserve buggy behaviour in `abs`, add `abs_real` strict variant.
* `JAVA-BUG-2` (`ebeDivide`): Uses modulo indexing for divisor despite assuming equal length. Preserve buggy behaviour in `ebeDivide`, add `ebeDivide_strict`.
* `JAVA-BUG-3` (`cubicSolver`): Exact float equality check for `h==0`. Preserve.
* `JAVA-BUG-4` (`toContinuedFraction`): Prints debug output to `stdout`. Preserve with optional `verbose` toggle (default True).
* `JAVA-BUG-5` (`createRowMatrix`): Returns an Nx1 column matrix instead of a 1xN row matrix. Preserve buggy behaviour, add `createRowMatrix_strict` for correct shape.
* `JAVA-BUG-6` (`solveCubic`): Wrong Cardano formula in 3-real-roots branch. Preserve.
* `RNG-DEVIATION-1` (`randomDir`): Uses `rgen` instead of global `Math.random()` to allow for sequence determinism. Document deviation.