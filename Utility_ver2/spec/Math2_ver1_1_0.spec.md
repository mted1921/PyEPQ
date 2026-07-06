# Math2 Conversion Spec

## Java class
`gov.nist.microanalysis.Utility.Math2`

Source: `src/gov/nist/microanalysis/Utility/Math2.java`

---

## Inbound dependencies (Java imports)
- `java.text.NumberFormat` — used as parameter type in `toString(double[], NumberFormat)`; port as `Callable[[float], str]`
- `java.util.Arrays` — used internally in private methods only
- `java.util.Random` — `rgen` field; port as `JavaRandom` from `_epq_compat`
- `gov.nist.microanalysis.EPQLibrary.EPQException` — thrown by `solvePoly`, `solveQuadratic`, `findRoot`; import from `_epq_compat`
- `Jama.Matrix` — returned by `createRowMatrix`; port as `JamaMatrix` from `_epq_compat`

---

## Outbound dependents (callers of public methods)
Most depended-upon class in the Utility package (in_degree 10). Callers include `UncertainValue2`, `LevenbergMarquardt2`, `DescriptiveStatistics`, and many EPQLibrary classes. Not exhaustively audited.

---

## Public API surface

| Java signature | Python signature | Notes |
|---|---|---|
| `public static final double[] ORIGIN_3D` | `ORIGIN_3D: F64Array` | Initialised via `v3(0,0,0)` — see Static init order |
| `public static final double[] ONE` | `ONE: F64Array` | `v3(1,1,1)` |
| `public static final double[] X_AXIS` | `X_AXIS: F64Array` | `v3(1,0,0)` |
| `public static final double[] Y_AXIS` | `Y_AXIS: F64Array` | `v3(0,1,0)` |
| `public static final double[] Z_AXIS` | `Z_AXIS: F64Array` | `v3(0,0,1)` |
| `public static final double[] MINUS_X_AXIS` | `MINUS_X_AXIS: F64Array` | `v3(-1,0,0)` |
| `public static final double[] MINUS_Y_AXIS` | `MINUS_Y_AXIS: F64Array` | `v3(0,-1,0)` |
| `public static final double[] MINUS_Z_AXIS` | `MINUS_Z_AXIS: F64Array` | `v3(0,0,-1)` |
| `public static final double SQRT_PI` | `SQRT_PI: float` | `math.sqrt(math.pi)` |
| `public static Random rgen` | `rgen: JavaRandom` | Mutable; replaced by `initializeRandom` |
| `public static void initializeRandom(long seed)` | `@staticmethod def initializeRandom_seed(seed: int) -> None` | R4 overload split |
| `public static void initializeRandom()` | `@staticmethod def initializeRandom() -> None` | Unseeded form |
| `public static final double sqr(double x)` | `@staticmethod def sqr(x: float) -> float` | |
| `public static final double erf(double x)` | `@staticmethod def erf(x: float) -> float` | NR port; also write `erf_literal` |
| `public static final double erfc(double x)` | `@staticmethod def erfc(x: float) -> float` | NR port; also write `erfc_literal` |
| `public static final double gammq(double a, double x)` | `@staticmethod def gammq(a: float, x: float) -> float` | NR port; also write `gammq_literal` |
| `public static final double gammap(double a, double x)` | `@staticmethod def gammap(a: float, x: float) -> float` | NR port; also write `gammap_literal` |
| `public static double chiSquaredConfidenceLevel(double confidence, int degreesOfFreedom)` | `@staticmethod def chiSquaredConfidenceLevel(confidence: float, degreesOfFreedom: int) -> float` | Uses FindRoot internally |
| `public static final double gammaln(double xx)` | `@staticmethod def gammaln(xx: float) -> float` | NR port; also write `gammaln_literal` |
| `static final public double expRand()` | `@staticmethod def expRand() -> float` | |
| `static final public double expRand(double lambda)` | `@staticmethod def expRand_lambda(lambda_: float) -> float` | R4: `lambda` is a Python keyword; rename parameter to `lambda_` |
| `static final public double[] randomDir()` | `@staticmethod def randomDir() -> F64Array` | Uses `Math.random()` — port with `rgen` |
| `static final public double distance(double[] p1, double[] p2)` | `@staticmethod def distance(p1: F64Array, p2: F64Array) -> float` | |
| `static final public double distanceSqr(double[] p1, double[] p2)` | `@staticmethod def distanceSqr(p1: F64Array, p2: F64Array) -> float` | |
| `static final public double magnitude(double[] p)` | `@staticmethod def magnitude(p: F64Array) -> float` | |
| `static final public double[] normalize(double[] p)` | `@staticmethod def normalize(p: F64Array) -> F64Array` | |
| `static final public double sum(double[] da)` | `@staticmethod def sum_d(da: F64Array) -> float` | R4 overload split with int form |
| `static final public int sum(int[] da)` | `@staticmethod def sum_i(da: Sequence[int]) -> int` | R4 overload split |
| `static final public double[] add(double[] da, double[] db)` | `@staticmethod def add(da: F64Array, db: F64Array) -> F64Array` | Returns new array |
| `static final public double[] addInPlace(double[] da, double[] db)` | `@staticmethod def addInPlace(da: F64Array, db: F64Array) -> F64Array` | **MUTABLE** — R5 guard on `da` |
| `static final public double[] plus(double[] a, double[] b)` | `@staticmethod def plus_vv(a: F64Array, b: F64Array) -> F64Array` | R4 overload split |
| `static final public double[] plus(double[] a, double b)` | `@staticmethod def plus_vs(a: F64Array, b: float) -> F64Array` | R4 overload split |
| `static final public double[] plusEquals(double[] a, double[] b)` | `@staticmethod def plusEquals(a: F64Array, b: F64Array) -> F64Array` | **MUTABLE** — R5 guard on `a` |
| `static final public double[] minus(double[] a, double[] b)` | `@staticmethod def minus_vv(a: F64Array, b: F64Array) -> F64Array` | R4 overload split |
| `static final public double[] minus(double[] a, double b)` | `@staticmethod def minus_vs(a: F64Array, b: float) -> F64Array` | R4 overload split |
| `static final public double dot(double[] a, double[] b)` | `@staticmethod def dot(a: F64Array, b: F64Array) -> float` | |
| `static final public double[] negative(double[] a)` | `@staticmethod def negative_v(a: F64Array) -> F64Array` | R4: vector form |
| `static public final double negative(double x)` | `@staticmethod def negative_scalar(x: float) -> float` | R4: scalar form — same Java name |
| `static final public double[] cross(double[] a, double[] b)` | `@staticmethod def cross(a: F64Array, b: F64Array) -> F64Array` | Requires length-3 inputs |
| `static final public double[] multiply(double a, double[] b)` | `@staticmethod def multiply_sv(a: float, b: F64Array) -> F64Array` | R4 scalar×vector |
| `static final public double[] multiply(double[] a, double[] b)` | `@staticmethod def multiply_vv(a: F64Array, b: F64Array) -> F64Array` | R4 element-wise |
| `static final public double[] timesEquals(double a, double[] b)` | `@staticmethod def timesEquals(a: float, b: F64Array) -> F64Array` | **MUTABLE** — R5 guard on `b` |
| `static final public double[] abs(double[] data)` | `@staticmethod def abs(data: F64Array) -> F64Array` | **JAVA-BUG-1** — clamps negatives to 0; also write `abs_real` |
| `static final public double[] pointBetween(double[] a, double[] b, double f)` | `@staticmethod def pointBetween(a: F64Array, b: F64Array, f: float) -> F64Array` | |
| `static final public double angleBetween(double[] a, double[] b)` | `@staticmethod def angleBetween(a: F64Array, b: F64Array) -> float` | |
| `static final public double[] divide(double[] a, double b)` | `@staticmethod def divide_vs(a: F64Array, b: float) -> F64Array` | Scalar denominator |
| `static final public double[] divideEquals(double[] a, double b)` | `@staticmethod def divideEquals(a: F64Array, b: float) -> F64Array` | **MUTABLE** — R5 guard on `a` |
| `static final public double[] ebeDivide(double[] a, double[] b)` | `@staticmethod def ebeDivide(a: F64Array, b: F64Array) -> F64Array` | Element-by-element |
| `static final public double[] quadraticSolver(double a, double b, double c)` | `@staticmethod def quadraticSolver(a: float, b: float, c: float) -> Optional[F64Array]` | Returns `None` if no real roots |
| `public static final double cubeRoot(double x)` | `@staticmethod def cubeRoot(x: float) -> float` | |
| `public static double[] cubicSolver(double a, double b, double c, double d)` | `@staticmethod def cubicSolver(a: float, b: float, c: float, d: float) -> F64Array` | |
| `public static double[] cubicSolver2(double a, double b, double c, double d)` | `@staticmethod def cubicSolver2(a: float, b: float, c: float, d: float) -> F64Array` | |
| `static final public double polynomial(double[] coeff, double x)` | `@staticmethod def polynomial(coeff: F64Array, x: float) -> float` | Horner's method |
| `static public double closestTo(double[] vals, double val)` | `@staticmethod def closestTo(vals: F64Array, val: float) -> float` | |
| `static final public double[] solvePoly(double[] coeff) throws EPQException` | `@staticmethod def solvePoly_coeffs(coeff: F64Array) -> Optional[F64Array]` | R4 overload split; raises EPQException |
| `static final public double[] solvePoly(double[] coeff, double y) throws EPQException` | `@staticmethod def solvePoly_coeffs_y(coeff: F64Array, y: float) -> Optional[F64Array]` | R4 overload split |
| `static final public double li(double x)` | `@staticmethod def li(x: float) -> float` | Logarithmic integral; `x > 1.0` required |
| `static final public double bound(double x, double x0, double x1)` | `@staticmethod def bound_d(x: float, x0: float, x1: float) -> float` | R4 overload split |
| `static final public int bound(int x, int lowerInc, int upperExc)` | `@staticmethod def bound_i(x: int, lowerInc: int, upperExc: int) -> int` | R4 overload split |
| `static final public long bound(long x, long lowerInc, long upperExc)` | `@staticmethod def bound_l(x: int, lowerInc: int, upperExc: int) -> int` | R4 overload split; Python `int` covers `long` |
| `static public final double positive(double x)` | `@staticmethod def positive(x: float) -> float` | |
| `static final public int binomialCoefficient(int n, int m)` | `@staticmethod def binomialCoefficient(n: int, m: int) -> int` | |
| `static final public double max(double[] da)` | `@staticmethod def max_arr(da: F64Array) -> float` | R4 overload split |
| `static final public double max(double[][] m)` | `@staticmethod def max_mat(m: Sequence[F64Array]) -> float` | R4 overload split |
| `static final public int max(int[] da)` | `@staticmethod def max_iarr(da: Sequence[int]) -> int` | R4 overload split |
| `static final public double min(double[] da)` | `@staticmethod def min_arr(da: F64Array) -> float` | R4 overload split |
| `static final public double min(double[][] m)` | `@staticmethod def min_mat(m: Sequence[F64Array]) -> float` | R4 overload split |
| `static final public int min(int[] da)` | `@staticmethod def min_iarr(da: Sequence[int]) -> int` | R4 overload split |
| `static public double[] slice(double[] data, int st, int len)` | `@staticmethod def slice(data: F64Array, st: int, len: int) -> F64Array` | |
| `static public double pNorm(double[] data, double p)` | `@staticmethod def pNorm(data: F64Array, p: float) -> float` | |
| `static public double infinityNorm(double[] data)` | `@staticmethod def infinityNorm(data: F64Array) -> float` | |
| `static public double Legendre(double x, int n)` | `@staticmethod def Legendre(x: float, n: int) -> float` | R1: preserve capital L |
| `public static boolean approxEquals(double a, double b, double frac)` | `@staticmethod def approxEquals(a: float, b: float, frac: float) -> bool` | |
| `static public double[] convolve(double[] v, double[] kernel)` | `@staticmethod def convolve(v: F64Array, kernel: F64Array) -> F64Array` | |
| `static public String toString(double[] vec)` | `@staticmethod def toString_vec(vec: F64Array) -> str` | R4 overload split; also exposes `toString` dispatcher |
| `static public String toString(double[] vec, NumberFormat nf)` | `@staticmethod def toString_nf(vec: F64Array, nf: Callable[[float], str]) -> str` | R4 overload split |
| `static public boolean isNaN(double[] arr)` | `@staticmethod def isNaN(arr: F64Array) -> bool` | |
| `public static long[] toContinuedFraction(double val, double tol)` | `@staticmethod def toContinuedFraction(val: float, tol: float, verbose: bool = False) -> list[int]` | **JAVA-BUG-2** — contains `System.out.println`; scipy primary adds `verbose` param |
| `public static double toDecimal(long[] cf)` | `@staticmethod def toDecimal(cf: Sequence[int]) -> float` | |
| `public static long[] toFraction(long[] cf)` | `@staticmethod def toFraction(cf: Sequence[int]) -> list[int]` | |
| `public static Matrix createRowMatrix(double[] vals)` | `@staticmethod def createRowMatrix(vals: F64Array) -> JamaMatrix` | Returns JamaMatrix |
| `public static long gcd(long a, long b)` | `@staticmethod def gcd(a: int, b: int) -> int` | |
| `public static double[] solveQuadratic(double a, double b, double c) throws EPQException` | `@staticmethod def solveQuadratic(a: float, b: float, c: float) -> Optional[F64Array]` | Delegates to solvePoly |
| `public static double[] solveCubic(double a, double b, double c)` | `@staticmethod def solveCubic(a: float, b: float, c: float) -> F64Array` | |
| `static public double findRoot(double[] coeffs, double x1, double x2, double xacc) throws EPQException` | `@staticmethod def findRoot(coeffs: F64Array, x1: float, x2: float, xacc: float) -> float` | |
| `static public double[] v3(double x, double y, double z)` | `@staticmethod def v3(x: float, y: float, z: float) -> F64Array` | |
| `static public double[] x3(double x)` | `@staticmethod def x3(x: float) -> F64Array` | |
| `static public double[] y3(double y)` | `@staticmethod def y3(y: float) -> F64Array` | |
| `static public double[] z3(double z)` | `@staticmethod def z3(z: float) -> F64Array` | |
| `public static double[][] transpose(double[][] mat)` | `@staticmethod def transpose(mat: Sequence[F64Array]) -> list[F64Array]` | |

---

## Private / protected members

| Java | Python |
|---|---|
| `private static final double gser(double a, double x)` | `@staticmethod def _gser(a: float, x: float) -> float` |
| `private static final double gcf(double a, double x)` | `@staticmethod def _gcf(a: float, x: float) -> float` |
| `static final boolean isUnitVector(double[] a)` | Not public (package-private in Java — no Python equivalent of package visibility) — omit or keep as `_isUnitVector` |

---

## Overloaded methods (split plan)

| Java overload group | Split strategy |
|---|---|
| `initializeRandom()` / `initializeRandom(long)` | `initializeRandom()` (no-arg) + `initializeRandom_seed(seed)` |
| `expRand()` / `expRand(double lambda)` | `expRand()` + `expRand_lambda(lambda_)` |
| `sum(double[])` / `sum(int[])` | `sum_d(da)` + `sum_i(da)` |
| `plus(double[], double[])` / `plus(double[], double)` | `plus_vv(a, b)` + `plus_vs(a, b)` |
| `minus(double[], double[])` / `minus(double[], double)` | `minus_vv(a, b)` + `minus_vs(a, b)` |
| `negative(double[])` / `negative(double)` | `negative_v(a)` + `negative_scalar(x)` |
| `multiply(double, double[])` / `multiply(double[], double[])` | `multiply_sv(a, b)` + `multiply_vv(a, b)` |
| `bound(double,...)` / `bound(int,...)` / `bound(long,...)` | `bound_d` + `bound_i` + `bound_l` |
| `max(double[])` / `max(double[][])` / `max(int[])` | `max_arr` + `max_mat` + `max_iarr` |
| `min(double[])` / `min(double[][])` / `min(int[])` | `min_arr` + `min_mat` + `min_iarr` |
| `toString(double[])` / `toString(double[], NumberFormat)` | `toString_vec` + `toString_nf` |
| `solvePoly(double[])` / `solvePoly(double[], double)` | `solvePoly_coeffs` + `solvePoly_coeffs_y` |

Unified dispatchers (with AMBIGUITY HAZARD comments) may optionally be provided for the most commonly called overload groups (`plus`, `minus`, `multiply`).

---

## Mutable-output methods
All of the following mutate their first array argument in place (R5: `_require_mutable_f64` guard required):

| Java method | Mutated parameter |
|---|---|
| `addInPlace(double[] da, double[] db)` | `da` |
| `plusEquals(double[] a, double[] b)` | `a` |
| `timesEquals(double a, double[] b)` | `b` |
| `divideEquals(double[] a, double b)` | `a` |

---

## Touchpoints into Jama / javax.swing / java.awt / java.io
- `Jama.Matrix` — `createRowMatrix` returns a `Matrix`. Port returns `JamaMatrix`.
- `gov.nist.microanalysis.EPQLibrary.EPQException` — thrown by `solvePoly`, `solveQuadratic`, `findRoot`. Import from `_epq_compat`.
- No `javax.swing` or `java.awt` usage.

---

## Abstract class strategy
Not applicable. Class is not abstract.

---

## Java-specific translation decisions

| Java pattern | Python translation | Rule |
|---|---|---|
| `public static Random rgen` (mutable public field) | `rgen: JavaRandom` at class level; `Math2.rgen` for callers | R3 — use `JavaRandom` |
| `Math.random()` in `randomDir()` | `Math2.rgen.nextDouble()` | Use the class RNG, not Python `random` |
| `new Random(System.currentTimeMillis())` in `randomDir()` | `Math2.rgen` (reuse global) | R3 — Java creates a local RNG; port uses the shared one |
| `NumberFormat` parameter in `toString` | `Callable[[float], str]` | R10 deviation |
| `Matrix` return from `createRowMatrix` | `JamaMatrix` | R3 |
| `EPQException` | `EPQException` from `_epq_compat` | R3 |
| `System.out.println(...)` in `toContinuedFraction` | Suppress (add `verbose: bool = False` param, print only if True) | R10 |
| Java int→double widening | Explicit `float(...)` casts where return type is `double` | R9 |

---

## Suspected Java bugs

**JAVA-BUG-1 — `abs(double[])` clamps negatives to zero instead of computing `|x|`.**
Java source line: `res[i] = (data[i] > 0.0 ? data[i] : 0.0);`
Returns `data[i]` when positive, `0.0` when negative or zero. True absolute value would return `-data[i]` for negative inputs.
Disposition: **Preserve** — literal port keeps the clamp behavior; add `abs_real` companion (scipy primary: `np.abs(data)`).

**JAVA-BUG-2 — `toContinuedFraction` contains a debug `System.out.println`.**
Java source line: `System.out.println(num[i + 2] / den[i + 2]);`
Prints intermediate convergents to stdout on every call. Clearly unintentional debug output.
Disposition: **Preserve** (literal port uses `verbose=True`); scipy primary defaults `verbose=False` and suppresses the print.

---

## Static init order
The eight `double[]` constants (`ORIGIN_3D`, `ONE`, `X_AXIS`, etc.) are initialized via `Math2.v3(...)` — a self-referential static call. In Python this requires the `v3` method to be defined before the class-level constant assignments. Initialize them after the `v3` definition in the class body (or as `ClassVar` initialized after class definition).

---

## Thread safety
`rgen` is a shared mutable static. Concurrent calls to any method that uses `rgen` (`expRand`, `randomDir`, plus anything that indirectly uses `Math.random()`) are not thread-safe. `initializeRandom` replaces `rgen` non-atomically. No synchronization present.
