# Simplex Conversion Spec

## Java class
`gov.nist.microanalysis.Utility.Simplex`

Source: `src/gov/nist/microanalysis/Utility/Simplex.java`

---

## Inbound dependencies (Java imports)
- `java.util.Arrays` — `Arrays.sort`; use `sorted()` or `list.sort()`
- `java.util.Random` — used in `randomizedStartingPoints`; replace with `JavaRandom` from `_epq_compat` (preserves Java RNG sequence for parity) or Python `random.Random` for the primary port — see JAVA-BUG-1
- `gov.nist.microanalysis.Utility.Math2` — `Math2.v2()`, `Math2.v3()` etc. for vector construction; import from sibling port

No EPQException or Jama imports.

---

## Outbound dependents (callers of public methods)
Not exhaustively audited. Optimization clients that need derivative-free minimization.

---

## Public API surface

`Simplex` is a **public abstract class**. The sole abstract method is `function`.

### Abstract method

| Java signature | Python signature | Notes |
|---|---|---|
| `abstract public double function(double[] params)` | `@abc.abstractmethod def function(self, params: list[float]) -> float` | Subclasses implement the objective function |

### Constructors

| Java signature | Python signature | Notes |
|---|---|---|
| `public Simplex(int nParams, double tol)` | `__init__(self, nParams: int, tol: float) -> None` | |
| `public Simplex(int nParams)` | `__init__(self, nParams: int, tol: float = _DEFAULT_TOLERANCE)` | Collapses with above via default arg |

### Public instance methods

| Java signature | Python signature | Notes |
|---|---|---|
| `public void setStartingSimplexPoint(double[] point, int index)` | `def setStartingSimplexPoint(self, point: list[float], index: int) -> None` | |
| `public void setStartingPoint(double[] point)` | `def setStartingPoint(self, point: list[float]) -> None` | Copies `point` as the canonical start; sets all simplex vertices near it |
| `public void randomizedStartingPoints(double[] center, double scale, Random rand)` | `def randomizedStartingPoints(self, center: list[float], scale: float, rand=None) -> None` | R4 split not needed — only one overload; `rand` defaults to `None` → uses `self._mRandom` |
| `public double[] perform(int maxIterations)` | `def perform(self, maxIterations: int) -> list[float]` | Runs Nelder-Mead; returns best parameter vector |
| `public double bestScore()` | `def bestScore(self) -> float` | Returns objective value at best parameters |
| `public void setTolerance(double tol)` | `def setTolerance(self, tol: float) -> None` | JAVA-BUG-2 applies |
| `public double getTolerance()` | `def getTolerance(self) -> float` | |
| `public int getEvaluations()` | `def getEvaluations(self) -> int` | Returns total number of `function` calls |

---

## Private / protected members

| Java | Python |
|---|---|
| `private static final double DEFAULT_TOLERANCE = 1.0e-10` | `_DEFAULT_TOLERANCE: float = 1.0e-10` |
| `private static final double ALPHA = 1.0` | `_ALPHA: float = 1.0` |
| `private static final double BETA = 0.5` | `_BETA: float = 0.5` |
| `private static final double GAMMA = 2.0` | `_GAMMA: float = 2.0` |
| `private int mNParams` | `self._mNParams: int` |
| `private double mTolerance` | `self._mTolerance: float` |
| `private double[][] mSimplex` | `self._mSimplex: list[list[float]]` — (nParams+1) x nParams |
| `private double[] mScores` | `self._mScores: list[float]` |
| `private double mBestScore` | `self._mBestScore: float` |
| `private int mEvaluations` | `self._mEvaluations: int` |
| `private Random mRandom` | `self._mRandom: random.Random` |

---

## Overloaded methods (split plan)
`Simplex(int)` and `Simplex(int, double)` collapse to a single `__init__` with a default `tol` argument.

`randomizedStartingPoints` has only one overload — no split needed.

---

## Mutable-output methods

`setStartingSimplexPoint` and `setStartingPoint` accept caller-supplied `list[float]` — make defensive copies (R5): `self._mSimplex[index] = list(point)`.

`randomizedStartingPoints`: `center` is read-only input — no defensive copy needed. `rand` is not stored.

---

## Touchpoints into Jama / javax.swing / java.awt / java.io
`java.util.Random` — used only in `randomizedStartingPoints`. Import `random.Random` for the primary port; `JavaRandom` from `_epq_compat` for the parity port. See JAVA-BUG-1.

No Jama, Swing, or AWT.

---

## Abstract class strategy
IS_ABSTRACT = True.

`function(double[] params)` is the sole abstract method. Subclasses implement the objective function to minimize.

**M4 applies**: JPype cannot extend Java abstract classes from Python. Direct parity testing of `Simplex` is blocked.

Parity harness strategy: locate a concrete Java subclass of `Simplex` in the EPQ library (e.g. within fitting routines). If none is accessible, write a Java-side test helper class that extends `Simplex` with a known quadratic objective, then call `perform` from Python via JPype on that object and compare with the Python port result.

---

## Java-specific translation decisions

| Java pattern | Python translation | Rule |
|---|---|---|
| `abstract public class Simplex` | `class Simplex(abc.ABC)` | R2 |
| `abstract public double function(double[] params)` | `@abc.abstractmethod def function(self, params: list[float]) -> float` | R1 — abstract → no underscore |
| `Arrays.sort(...)` | `list.sort(...)` or `sorted(...)` | |
| `new Random(System.currentTimeMillis())` in `__init__` | `self._mRandom = random.Random()` | Unseeded; see JAVA-BUG-1 |
| `double[] params` arrays | `list[float]` | |
| `assert tol >= 0.0 && tol <= 1.0` in `setTolerance` | `assert 0.0 <= tol <= 1.0` — but see JAVA-BUG-2 | |

---

## Suspected Java bugs

**JAVA-BUG-1 — `randomizedStartingPoints` ignores the passed `rand` parameter.**
Java source: `randomizedStartingPoints(double[] center, double scale, Random rand)` — method signature accepts a `rand` argument, but the implementation ignores it and uses `mRandom` (the instance's internal `Random` field) for all random draws. Callers that pass a seeded `Random` expecting reproducible results will get non-reproducible behavior.
Disposition: **Preserve** — the literal port uses `self._mRandom` and ignores the `rand` argument. The strict port uses `rand if rand is not None else self._mRandom`. Document as `# JAVA-BUG-1`.

**JAVA-BUG-2 — `setTolerance` assert vs. clamp inconsistency.**
Java source: `assert tol >= 0.0 && tol <= 1.0; this.mTolerance = Math.max(0.0, Math.min(1.0, tol));`
Both an assert (`tol in [0,1]`) and a clamp are present. The assert fires only with `-ea` JVM flag. The clamp means that with assertions disabled, `setTolerance(2.0)` silently clamps to `1.0` rather than throwing. These are semantically contradictory: either validate-and-throw or clamp, not both.
Disposition: **Preserve both** — port the assert and the clamp literally. Document as `# JAVA-BUG-2`.

---

## Static init order
`DEFAULT_TOLERANCE`, `ALPHA`, `BETA`, `GAMMA` are compile-time constants — initialize in class body. No ordering concern.

---

## Thread safety
Not thread-safe. `_mSimplex`, `_mScores`, `_mBestScore`, `_mEvaluations` are all mutable instance state modified during `perform`.
