# LevenbergMarquardt2 Conversion Spec

> **Spec revision ver1_1_1** (2026-06-25): repaired against the Java source. The prior
> ver1_1_0 described a `double[]`-based `FitFunction` and a large set of methods
> (`setConstraints`, `setInitial`, `setData`, `setTolerance`, `perform`,
> `getChiSquared`, `getIterations`, `_DEFAULT_MAX_ITER=200`, …) that **do not exist**.
> The real class uses a **`Matrix`-based** `FitFunction { partials, compute }`, a
> **non-static inner** `FitResult`, a single `compute(ff, yData, sigma, p0)` entry point,
> and `mKMax=100`.

## Java class
`gov.nist.microanalysis.Utility.LevenbergMarquardt2`

Source: `src/gov/nist/microanalysis/Utility/LevenbergMarquardt2.java`

---

## Inbound dependencies (Java imports)
- `java.awt.event.ActionEvent`, `java.awt.event.ActionListener` — progress callback fired each iteration; **no Python equivalent** → replace with `Callable` (R10 deviation).
- `java.util.ArrayList` — `mListeners`; use `list`.
- `gov.nist.microanalysis.EPQLibrary.EPQException` — thrown by `solve`/`compute`; import from `_epq_compat`.
- `Jama.Matrix` — all parameter/Jacobian/covariance matrices; import `JamaMatrix` from `_epq_compat`.
- `Jama.SingularValueDecomposition` — `a.svd()` in `solve` and `compute`. **`JamaMatrix` does NOT wrap SVD** — use `numpy.linalg.svd` directly (see substitution note).
- `gov.nist.microanalysis.Utility.Math2` — `Math2.sqr`, `Math2.min`; sibling port.
- `gov.nist.microanalysis.Utility.UncertainValue2` — `FitResult.mBestParams` element type; sibling port.

---

## Outbound dependents (callers of public methods)
The non-linear Levenberg–Marquardt fitting engine. Extended by `LevenbergMarquardtConstrained`; used by `LevenbergMarquardtParameterized`. Callers implement `FitFunction` and call `compute`. Not exhaustively audited.

---

## Public API surface

`LevenbergMarquardt2` is a **concrete** class (IS_ABSTRACT = False). It nests one interface, one abstract helper, and one inner result class.

### Inner interface `LevenbergMarquardt2.FitFunction`

| Java signature | Python signature | Notes |
|---|---|---|
| `Matrix partials(Matrix params)` | `@abc.abstractmethod def partials(self, params: JamaMatrix) -> JamaMatrix` | n×m Jacobian; `params` is m×1 |
| `Matrix compute(Matrix params)` | `@abc.abstractmethod def compute(self, params: JamaMatrix) -> JamaMatrix` | n×1 fit values |

Port as `class FitFunction(abc.ABC)` nested in `LevenbergMarquardt2`; module-level alias `FitFunction = LevenbergMarquardt2.FitFunction` (R2). IS_ABSTRACT = True → M4 applies.

### Inner abstract class `LevenbergMarquardt2.AutoPartialsFitFunction` (`static abstract class implements FitFunction`)

| Java signature | Python signature | Notes |
|---|---|---|
| `private final double DELTA` | `self._DELTA: float` | finite-difference base step |
| `private double[] mDelta` | `self._mDelta: Optional[F64Array]` | per-parameter steps, lazily sized |
| `protected AutoPartialsFitFunction(double delta)` | `__init__(self, delta: float)` | |
| `protected AutoPartialsFitFunction()` | `__init__(self)` → `DELTA = 1.0e-8` | dispatch by arity |
| `@Override public Matrix partials(Matrix params)` | `def partials(self, params) -> JamaMatrix` | forward-difference Jacobian; `v2 = mDelta[p] if v1==0 else v1*(1+mDelta[p])` |
| `public void setDelta(double[] delta)` | `def setDelta(self, delta) -> None` | `self._mDelta = list(delta)` (clone) |
| *(abstract)* `compute` inherited | remains `@abc.abstractmethod` | |

Module-level alias `AutoPartialsFitFunction = LevenbergMarquardt2.AutoPartialsFitFunction` (R2). IS_ABSTRACT = True → M4 applies.

### Inner class `LevenbergMarquardt2.FitResult` (**non-static** inner class)

`FitResult` is a **non-static** inner class — `getModel()` returns the enclosing `LevenbergMarquardt2.this`. Port as a nested class that takes an explicit `model` reference (Python nested classes do not auto-bind to an outer instance). Constructor signature: `__init__(self, model: "LevenbergMarquardt2", ff: FitFunction)`. Module-level alias `FitResult = LevenbergMarquardt2.FitResult` (R2).

| Java member | Python | Notes |
|---|---|---|
| `protected double mChiSq` | `self._mChiSq: float` | |
| `protected UncertainValue2[] mBestParams` | `self._mBestParams: list[UncertainValue2]` | |
| `protected double[] mBestY` | `self._mBestY: list[float]` | |
| `protected Matrix mCovariance` | `self._mCovariance: JamaMatrix` | |
| `protected int mIterCount` | `self._mIterCount: int` | |
| `protected int mImproveCount` | `self._mImproveCount: int` | |
| `protected final FitFunction mFunction` | `self._mFunction: FitFunction` | |
| `FitResult(FitFunction ff)` *(package-private)* | `__init__(self, model, ff)` | store `model` for `getModel()` |
| `public UncertainValue2[] getBestParametersU()` | `def getBestParametersU(self) -> list[UncertainValue2]` | |
| `public double[] getBestParameters()` | `def getBestParameters(self) -> list[float]` | nominal values of `mBestParams` |
| `public double[] getBestFitValues()` | `def getBestFitValues(self) -> list[float]` | returns `mBestY` |
| `public double getChiSquared()` | `def getChiSquared(self) -> float` | |
| `public Matrix getCovariance()` | `def getCovariance(self) -> JamaMatrix` | |
| `public int getIterationCount()` | `def getIterationCount(self) -> int` | |
| `public int getImproveCount()` | `def getImproveCount(self) -> int` | |
| `public LevenbergMarquardt2 getModel()` | `def getModel(self) -> "LevenbergMarquardt2"` | returns the stored `model` |
| `public FitFunction getFitFunction()` | `def getFitFunction(self) -> FitFunction` | |

> Subclasses `LevenbergMarquardtConstrained.realToConstrained(FitResult)` and
> `LevenbergMarquardtParameterized.ParameterizedFitResult` **write** the protected
> fields (`_mBestParams`, `_mBestY`, `_mChiSq`, `_mCovariance`, `_mImproveCount`,
> `_mIterCount`) directly and instantiate `FitResult` via the model. Keep the
> `__init__(self, model, ff)` signature stable — those ports depend on it.

### `LevenbergMarquardt2` constructor and public methods

| Java signature | Python signature | Notes |
|---|---|---|
| *(implicit no-arg constructor)* | `__init__(self)` | initialise constants/`_mListeners` |
| `public int getIteration()` | `def getIteration(self) -> int` | |
| `public void addActionListener(ActionListener al)` | `def addActionListener(self, al: Callable) -> None` | R10 — append a progress `Callable` |
| `public FitResult compute(FitFunction ff, Matrix yData, Matrix sigma, Matrix p0) throws EPQException` | `def compute(self, ff, yData, sigma, p0) -> FitResult` | core LM loop; **constructs `FitResult(self, ff)`** |
| `public int getMaxIterations()` | `def getMaxIterations(self) -> int` | |
| `public void setMaxIterations(int max)` | `def setMaxIterations(self, max: int) -> None` | sets `_mKMax` |

---

## Private / protected members

| Java | Python |
|---|---|
| `private final double mEps1 = 1.0e-15` | `self._mEps1: float = 1.0e-15` |
| `private final double mEps2 = 1.0e-6` | `self._mEps2: float = 1.0e-6` |
| `private final double mEps3 = 1.0e-15` | `self._mEps3: float = 1.0e-15` |
| `private final double mTau = 1.0e-3` | `self._mTau: float = 1.0e-3` |
| `private int mKMax = 100` | `self._mKMax: int = 100` |
| `private int mIteration` | `self._mIteration: int` |
| `private final ArrayList<ActionListener> mListeners` | `self._mListeners: list[Callable]` |
| `private Matrix jTj(Matrix j, Matrix sigma)` | `def _jTj(self, j, sigma) -> JamaMatrix` — symmetric JᵀJ weighted by 1/σ² |
| `private Matrix g(Matrix j, Matrix eps, Matrix sigma)` | `def _g(self, j, eps, sigma) -> JamaMatrix` |
| `private Matrix eps(Matrix yData, Matrix fp, Matrix sigma)` | `def _eps(self, yData, fp, sigma) -> JamaMatrix` |
| `private Matrix solve(Matrix a, double mu, Matrix g) throws EPQException` | `def _solve(self, a, mu, g) -> JamaMatrix` — adds `mu` to the diagonal, SVD pseudo-inverse solve |
| `private Matrix updateSingularValues(Matrix w)` | `def _updateSingularValues(self, w) -> JamaMatrix` — invert singular values above `1e-10*max`, else 0 |
| `private double maxDiagonal(Matrix a)` | `def _maxDiagonal(self, a) -> float` |
| `private double chiSqr(Matrix eps)` | `def _chiSqr(self, eps) -> float` |

---

## Overloaded methods (split plan)
- `AutoPartialsFitFunction` constructor: `(double delta)` / `()` → `__init__` dispatch by arity (`()` uses `DELTA = 1e-8`).
- No other overloads. `compute`/`partials` are interface methods — keep their names exactly (do not suffix).

---

## Mutable-output methods
`solve` mutates its `a` argument's diagonal (`a.set(i,i, a.get(i,i)+mu)`) before decomposing — note this in-place write (`_require_mutable` not applicable to `JamaMatrix`, but the port must operate on the live underlying array via `.set`). `setDelta` clones its input (`delta.clone()`); port as `list(delta)` to avoid aliasing.

---

## Touchpoints into Jama / javax.swing / java.awt / java.io
- `Jama.Matrix` throughout → `JamaMatrix` (`get`/`set`, `getRowDimension`/`getColumnDimension`, `copy`, `plus`, `times`, `transpose`, `normF`, `normInf`, `getColumnPackedCopy`). Build new matrices with `JamaMatrix.zeros(m, n)`.
- `Jama.SingularValueDecomposition` → numpy (see below).
- `java.awt.event.ActionListener`/`ActionEvent` → `Callable` (R10). No `javax.swing`/`java.io`.

### SVD substitution note (R2 / P1)
`solve` and the final covariance step call `a.svd()` and use `getU()`, `getS()` (singular values), `getV()`. The pseudo-inverse is `V · W⁺ · Uᵀ` where `W⁺` inverts singular values above `1e-10·max` (else 0). With `U, s, Vt = numpy.linalg.svd(A, full_matrices=False)`: **Jama `V` = numpy `Vt.T`**, and `S` is the diagonal of `s`. Build `W⁺ = diag(updateSingularValues(s))`. Mark the numpy call `# SCIPY-DEV-1: Jama SVD → numpy.linalg.svd` and record it in `BUG_LEDGER`/`CHANGES`. There is **no `_literal` twin** — Java itself delegates to Jama's SVD; numpy SVD is the faithful equivalent. Do not hand-roll an SVD.

`getColumnPackedCopy()` (used for `mBestY`) → flatten the matrix column-major: `A.getArray().flatten(order="F").tolist()`.

---

## Abstract class strategy
`LevenbergMarquardt2` IS_ABSTRACT = False.

`FitFunction` and `AutoPartialsFitFunction` IS_ABSTRACT = True → **M4 applies**. JPype cannot subclass them from Python (only `@JImplements` works for the interface). Parity strategy:
- Mark cross-engine Java parity classes `@pytest.mark.skip(reason="M4: ...")`.
- Validate analytically: implement a concrete Python `FitFunction` for a model with a known least-squares solution (e.g. linear `y = a + b·x`, where LM should converge to the normal-equation answer), or subclass `AutoPartialsFitFunction` and check its finite-difference Jacobian against the analytic one. Assert `getBestParameters()`, `getChiSquared()`, and `getCovariance()` against the closed-form / `scipy.optimize.least_squares` reference within `TOL_NR_LIB`.

---

## Java-specific translation decisions

| Java pattern | Python translation | Rule |
|---|---|---|
| `a.svd()` (Jama) | `numpy.linalg.svd(A, full_matrices=False)`; Jama `V` = numpy `Vt.T` | SCIPY-DEV-1 |
| `new Matrix(m, n)` | `JamaMatrix.zeros(m, n)` | R3 |
| `params.copy()` | `JamaMatrix(params.getArrayCopy())` | R3 |
| `mListeners.get(i).actionPerformed(new ActionEvent(this, (100*mIteration)/mKMax, null))` | `callback((100*self._mIteration)//self._mKMax)` for each callback | R10 + R7 (integer `//`) |
| `assert n > m`, `assert ...ColumnDimension()==1`, `assert den.get(0,0) > 0.0`, etc. | **Omit** — disabled-by-default Java asserts | Cross-Cutting — Java `assert` |
| `throw new EPQException("mu is NaN ...")` / `a(i,i) is NaN` | `raise EPQException(...)` | R3 |
| `Math2.sqr`, `Math2.min(sigma.getColumnPackedCopy())` | sibling-port statics | R3 |
| `g.normInf()`, `deltaP.normF()`, `p.normF()` | `JamaMatrix.normInf()` / `.normF()` | R3 |
| `mu *= Math.max(1.0/3.0, 1.0 - Math.pow((2*rho)-1, 3))` | faithful `math.pow`/`max` | |
| nested `FitResult` via `new FitResult(ff)` (non-static) | `FitResult(self, ff)` — pass the model | R2 / inner class |

---

## Suspected Java bugs
None identified in the LM iteration. The numerous `assert` statements are debug-only and intentionally dropped (not `JAVA-BUG` entries).

`BUG_LEDGER`: one `SCIPY-DEV-1` entry (numpy SVD). No `JAVA-BUG-N`.

---

## Static init order
`mEps1..mTau`, `mKMax` are instance constants set at construction. No class-level statics with cross-class references. No ordering concern.

---

## Thread safety
Not thread-safe. `compute` mutates `_mIteration` and internal matrices; `addActionListener` mutates `_mListeners`. Concurrent `compute` calls on one instance corrupt iteration state.
