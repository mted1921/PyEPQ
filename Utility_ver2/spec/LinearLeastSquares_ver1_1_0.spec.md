# LinearLeastSquares Conversion Spec

## Java class
`gov.nist.microanalysis.Utility.LinearLeastSquares`

Source: `src/gov/nist/microanalysis/Utility/LinearLeastSquares.java`

---

## Inbound dependencies (Java imports)
- `java.util.Arrays` — `Arrays.fill`, `Arrays.sort`, `Arrays.binarySearch`; use Python equivalents (`list`/`np`).
- `gov.nist.microanalysis.EPQLibrary.EPQException` — thrown by `perform`/`performFit` and the result getters; import from `_epq_compat`.
- `Jama.Matrix` — covariance matrix, design matrix, sub-matrix operations; import `JamaMatrix` from `_epq_compat`.
- `Jama.SingularValueDecomposition` — `a.svd()` in `performFit`. **`JamaMatrix` does NOT wrap SVD** (see `_epq_compat` docstring). Use `numpy.linalg.svd` directly — see SVD-substitution note below.
- `gov.nist.microanalysis.Utility.Math2` — `Math2.gammq`, `Math2.sqr`, `Math2.chiSquaredConfidenceLevel`; import from the sibling port.
- `gov.nist.microanalysis.Utility.UncertainValue2` — `mFitCoefficients` element type; import from the sibling port.

---

## Outbound dependents (callers of public methods)
Base class for linear least-squares fitting. Extended by `LinearLeastSquaresMS` (Bayesian model selection) and by concrete fit-model subclasses that implement `fitFunction`/`fitFunctionCount`. Not exhaustively audited.

---

## Public API surface

`LinearLeastSquares` is an **abstract class**. Subclasses implement the two abstract methods (`fitFunctionCount`, `fitFunction`). The fit is computed lazily on first access via the protected `perform()`/private `performFit()`.

### Constants

| Java declaration | Python name | Notes |
|---|---|---|
| `static protected final double TOLERANCE = 1.0e-12` | `_TOLERANCE: float = 1.0e-12` | `protected` → `_`; subclasses (MS) read it |
| `private static final double MAX_ERROR = Double.MAX_VALUE` | `_MAX_ERROR: float = sys.float_info.max` | Used as the "exclude this point" sentinel in `reducedChiSquared` |

### Enum

| Java | Python | Notes |
|---|---|---|
| `public enum INTERVAL_MODE { ONE_D_INTERVAL, JOINT_INTERVAL }` | `class INTERVAL_MODE(enum.Enum): ONE_D_INTERVAL = auto(); JOINT_INTERVAL = auto()` | Nested in the class; add module-level alias `INTERVAL_MODE = LinearLeastSquares.INTERVAL_MODE` (R2) |

### Constructors

| Java signature | Python signature | Notes |
|---|---|---|
| `public LinearLeastSquares()` | `__init__(self)` | Blank object; data assigned later via `setData` |
| `public LinearLeastSquares(double[] x, double[] y, double[] sig)` | `__init__(self, x, y, sig)` | Calls `setData(x, y, sig)` |
| `public LinearLeastSquares(double[] x, double[] y)` | `__init__(self, x, y)` | `sig=None` → counting-statistics path |

`__init__` must dispatch by argument count/type (P4 discipline): no-arg → blank; `(x, y)` → `setData(x, y, None)`; `(x, y, sig)` → `setData(x, y, sig)`.

### Public instance methods

| Java signature | Python signature | Notes |
|---|---|---|
| `public void setData(double[] x, double[] y, double[] sig)` | `def setData_xysig(self, x, y, sig) -> None` | R4. Filters points with `sig[i] >= 1e300` (treated as excluded). Then `reevaluateAll()` |
| `public void setData(double[] x, double[] y)` | `def setData_xy(self, x, y) -> None` | R4 — delegates with `sig=None` |
| `public double[] fitParameters() throws EPQException` | `def fitParameters(self) -> list[float]` | Triggers `performFit()`; returns nominal coefficients |
| `public UncertainValue2[] getResults() throws EPQException` | `def getResults(self) -> list[UncertainValue2]` | Fit coefficients with uncertainties |
| `public double fitParamter(int i) throws EPQException` | `def fitParamter(self, i: int) -> float` | **Keep the Java spelling `fitParamter`** (source typo) verbatim per R1 |
| `public double[] errors() throws EPQException` | `def errors(self) -> list[float]` | `sqrt` of covariance diagonal |
| `public Matrix covariance() throws EPQException` | `def covariance(self) -> JamaMatrix` | Returns `_mCovariance` |
| `public static double chiSqr(int degsOfFree, double prob)` | `@staticmethod def chiSqr(degsOfFree, prob) -> float` **and** `chiSqr_literal(...)` | **R2/P1 — scipy primary + literal pair.** See library-substitution note |
| `public double[] confidenceIntervals(INTERVAL_MODE mode, double prob, Matrix cov) throws EPQException` | `def confidenceIntervals(self, mode, prob, cov) -> list[float]` | Uses `chiSqr`, matrix `inverse`/`det` |
| `public Matrix correlation() throws EPQException` | `def correlation(self) -> JamaMatrix` | Normalised covariance |
| `public double chiSquared() throws EPQException` | `def chiSquared(self) -> float` | Delegates to `_chiSquared_d(fitParameters())` |
| `public double reducedChiSquared(double confidenceLevel) throws EPQException` | `def reducedChiSquared(self, confidenceLevel: float) -> float` | Uses `Math2.chiSquaredConfidenceLevel` |
| `public void clearZeroedCoefficients()` | `def clearZeroedCoefficients(self) -> None` | |
| `public void zeroFitCoefficient(int i, boolean b)` | `def zeroFitCoefficient(self, i: int, b: bool) -> None` | Allocates `_mZeroThese` lazily |
| `public int getNonZeroedCoefficientCount()` | `def getNonZeroedCoefficientCount(self) -> int` | |
| `public boolean isZeroFitCoefficient(int i)` | `def isZeroFitCoefficient(self, i: int) -> bool` | |
| `public double fitQuality() throws EPQException` | `def fitQuality(self) -> float` | Delegates to `fitQuality(fitParameters())` |
| `public double fitQuality(double[] fp)` | `def fitQuality_fp(self, fp: list[float]) -> float` | R4 — uses `Math2.gammq` |

### Protected methods (subclass extension points / overrides)

| Java signature | Python signature | Notes |
|---|---|---|
| `protected void perform() throws EPQException` | `def _perform(self) -> None` | Overridden by `LinearLeastSquaresMS`. Default delegates to `_performFit()` |
| `protected double[] editSingularValues(double[] wi)` | `def _editSingularValues(self, wi: list[float]) -> list[float]` | Overridden by MS. Zeroes singular values below `wMax * TOLERANCE` |
| `protected void reevaluate()` | `def _reevaluate(self) -> None` | Clears `_mFitCoefficients`, `_mCovariance` |
| `protected void reevaluateAll()` | `def _reevaluateAll(self) -> None` | Clears the SVD too; overridden by MS |
| `protected double chiSquared(double[] fitCoeff)` | `def _chiSquared_d(self, fitCoeff: list[float]) -> float` | R4 |
| `protected double chiSquared(UncertainValue2[] fitCoeff)` | `def _chiSquared_uv(self, fitCoeff: list[UncertainValue2]) -> float` | R4 |

### Abstract methods (extension points)

| Java signature | Python signature | Notes |
|---|---|---|
| `abstract protected int fitFunctionCount()` | `@abc.abstractmethod def _fitFunctionCount(self) -> int` | `protected` → `_` |
| `abstract protected void fitFunction(double xi, double[] afunc)` | `@abc.abstractmethod def _fitFunction(self, xi: float, afunc: F64Array) -> None` | Fills the caller-supplied `afunc` in place (mutable output — see below) |

---

## Private / protected members

| Java | Python |
|---|---|
| `protected double[] mXCoordinate` | `self._mXCoordinate: Optional[F64Array]` |
| `protected double[] mData` | `self._mData: Optional[F64Array]` |
| `protected double[] mSigma` | `self._mSigma: Optional[F64Array]` |
| `protected UncertainValue2[] mFitCoefficients` | `self._mFitCoefficients: Optional[list[UncertainValue2]]` — `None` flags "not yet fit" |
| `protected Matrix mCovariance` | `self._mCovariance: Optional[JamaMatrix]` |
| `private SingularValueDecomposition mSVD` | `self._mSVD` — cache the numpy SVD result as a tuple `(U, s, V)` (see SVD note); `None` flags "recompute SVD" |
| `private boolean[] mZeroThese` | `self._mZeroThese: Optional[list[bool]]` |
| `synchronized (this)` in `performFit` | `self._mLock: threading.Lock` guarding the lazy fit |

---

## Overloaded methods (split plan)
- `setData(x,y,sig)` / `setData(x,y)` → `setData_xysig` / `setData_xy` (R4). Constructor dispatches to these.
- `chiSquared()` (public, no-arg) vs `chiSquared(double[])` / `chiSquared(UncertainValue2[])` (protected) → public `chiSquared`, protected `_chiSquared_d` / `_chiSquared_uv`.
- `fitQuality()` / `fitQuality(double[])` → `fitQuality` / `fitQuality_fp`.
- Constructor: three overloads dispatched by arg count/type.

---

## Mutable-output methods
- `fitFunction(double xi, double[] afunc)` (abstract) **fills the caller-supplied `afunc`** in place. The concrete subclass implementation must call `_require_mutable_f64(afunc, "afunc")` before writing (R5). The parent `_performFit`/`_chiSquared_*` callers pass a freshly allocated array each time.
- `setData_*` stores references to / filtered copies of caller arrays. When the no-exclusion branch is taken Java aliases the input arrays directly (`mXCoordinate = x`); the port may keep this aliasing for fidelity but should prefer defensive copies and note any deviation (R10).

---

## Touchpoints into Jama / javax.swing / java.awt / java.io
- `Jama.Matrix` → `JamaMatrix` (covariance, design matrix, sub-matrix, `inverse`, `det`, `copy`). Use `JamaMatrix.zeros(m, n)` for `new Matrix(m, n)`; use `.getMatrix(...)` for sub-matrix extraction (inclusive endpoints).
- `Jama.SingularValueDecomposition` → **numpy**: `U, s, Vt = numpy.linalg.svd(A, full_matrices=False)`. No `javax.swing`/`java.awt`/`java.io`.

### SVD substitution note (R2 / P1 — read carefully)
`performFit` calls `a.svd()` and then reads `U`, `S` (diagonal singular values), and `V` with `a = U·S·Vᵀ`. numpy returns `U, s, Vt` with `a = U·diag(s)·Vt`, so **Jama's `V` equals numpy's `Vt.T`** (`V[j,i] == Vt[i,j]`). Map indices accordingly — this is the highest-risk transcription point. Singular values are descending in both, and `full_matrices=False` reproduces Jama's economy dimensions (`U` is `dataLen×nFit`, `S`/`V` are `nFit×nFit`).

- Mark the `numpy.linalg.svd` call `# SCIPY-DEV-1: Jama SingularValueDecomposition replaced by numpy.linalg.svd` and record it in `BUG_LEDGER` and the docstring `CHANGES`.
- **`performFit` has no `_literal` twin.** Java itself delegates the decomposition to Jama's SVD; numpy's SVD *is* the faithful equivalent. Do **not** hand-roll a Golub–Kahan SVD to manufacture a `_literal`. The `_literal` pairing requirement (R2) applies only to `chiSqr` below.

---

## Abstract class strategy
IS_ABSTRACT = True (`fitFunctionCount`, `fitFunction` are abstract).

Port as `class LinearLeastSquares(abc.ABC)` with the two methods `@abc.abstractmethod`.

**M4 applies.** JPype cannot extend the Java abstract class from Python, so the parity harness cannot build a Java-side subclass from Python. Strategy for the parity test:
- Mark the direct parity class `@pytest.mark.skip(reason="M4: ...")`.
- Validate analytically with a **concrete Python subclass** — e.g. a polynomial basis `fitFunction(xi, afunc) → afunc[k] = xi**k` with `fitFunctionCount() = degree+1`. Fit known data with a known closed-form least-squares answer and assert `fitParameters()`, `errors()`, `chiSquared()`, and `correlation()` against the analytic/`numpy.polyfit` ground truth.
- Where a Java-side concrete subclass exists in the EPQ jar, drive it through a concrete Java method for a true parity comparison.

---

## Java-specific translation decisions

| Java pattern | Python translation | Rule |
|---|---|---|
| `a.svd()` (Jama) | `numpy.linalg.svd(A, full_matrices=False)`; Jama `V` = numpy `Vt.T` | SCIPY-DEV-1 |
| `new Matrix(m, n)` | `JamaMatrix.zeros(m, n)` | R3 |
| `mCovariance.copy()` | `JamaMatrix(m.getArrayCopy())` | R3 |
| `cov.inverse()`, `cov.det()`, `sub.det()` | `JamaMatrix.inverse()`, `.det()` | R3 |
| many `assert ...` lines (sigma≥0, !NaN/!Inf, SVD reconstruction, dim checks, correlation∈[-1,1]) | **Omit** — Java `assert` is disabled by default; these are sanity checks, not production behaviour. Do **not** emit Python `assert` | Cross-Cutting — Java `assert` |
| `synchronized (this)` lazy double-checked fit | `with self._mLock:` preserving the two-phase null check | Cross-Cutting — concurrency |
| `Arrays.fill(mFitCoefficients, UncertainValue2.ZERO)` | `[UncertainValue2.ZERO] * nTot` | |
| `Arrays.sort` / `Arrays.binarySearch` (used in MS override) | `sorted()` / `bisect` — see MS spec | |
| integer division in index math (`j`, `k` loops) | already integer; guard any `/` between ints with `//` | R7 |
| `Math.sqrt`, `Math.abs`, `Math.max`, `Math.pow` | `math.*` / `np.*` | |
| `Double.MAX_VALUE` | `sys.float_info.max` | |
| `fitParamter` (typo) | keep verbatim | R1 |

---

## Suspected Java bugs
None identified that change correct-path output. The numerous `assert` statements are debug-only (disabled by default) and are intentionally dropped, not logged as `JAVA-BUG` entries. If the literal SVD-index mapping is implemented incorrectly it will surface as a parity failure — that is a port error, not a Java bug.

`BUG_LEDGER`: one `SCIPY-DEV-1` entry for the numpy-SVD substitution (no `JAVA-BUG-N`).

---

## Static init order
`TOLERANCE` and `MAX_ERROR` are compile-time constants — initialize in the class body. `chiSqr` is a pure static method. No cross-class static initialization ordering concern.

---

## Thread safety
`performFit` uses double-checked locking on `this` to compute the fit once. The port must preserve this with `threading.Lock`. Mutators (`setData`, `zeroFitCoefficient`, `reevaluate*`) invalidate the cache and are not safe to call concurrently with `performFit` on the same instance.
