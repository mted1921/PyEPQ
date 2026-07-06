# FindRoot Conversion Spec

## Java class
`gov.nist.microanalysis.Utility.FindRoot`

Source: `src/gov/nist/microanalysis/Utility/FindRoot.java`

---

## Inbound dependencies (Java imports)
- None (uses only `java.lang.Math.abs`, mapped to Python built-in `abs()`)

---

## Outbound dependents (callers of public methods)

- `Math2` — uses `FindRoot` via an anonymous inner class for `chiSquaredConfidenceLevel`
  and exposes a `findRoot` wrapper. `Math2_ver8_1_5` bridges this with
  `scipy.optimize.brentq`; a concrete `FindRoot` subclass can replace it once `FindRoot`
  is a dependency of `Math2`.
- Any EPQLibrary class that calls `Math2.findRoot(...)` or subclasses `FindRoot` directly.

---

## Public API surface

| Java signature | Python signature | Notes |
|---|---|---|
| `abstract double function(double x0)` | `@abc.abstractmethod function(self, x0: float) -> float` | Extension point; un-prefixed per R1 |
| `void initialize(double[] vars)` | `initialize(self, vars: list[float]) -> None` | Virtual no-op default |
| `double bestX()` | `bestX(self) -> float` | Getter |
| `double bestY()` | `bestY(self) -> float` | Getter |
| `double EvaluationCount()` | `EvaluationCount(self) -> float` | Returns `float` to match Java declared `double`; internal field is `int` |
| `double perform(double x0, double x2, double eps, int iMax)` | `perform(self, x0: float, x2: float, eps: float, iMax: int) -> float` | Jack Crenshaw's bisection + inverse quadratic interpolation |

---

## Private / protected members

| Java | Python |
|---|---|
| `private double mBestX` | `self._mBestX: float` |
| `private double mBestY` | `self._mBestY: float` |
| `private int mNEvals` | `self._mNEvals: int` |

---

## Overloaded methods (split plan)
None.

---

## Mutable-output methods
None. `perform` modifies `_mBestX`, `_mBestY`, `_mNEvals` on `self` (instance
state, not a passed buffer — no `_require_mutable_f64` guard needed).

---

## Touchpoints into Jama / javax.swing / java.awt / java.io
None. Only `java.lang.Math.abs` → Python `abs()`.

---

## Abstract class strategy
`FindRoot` is abstract: declares `abstract double function(double x0)`. In Python:
`abc.ABC` + `@abc.abstractmethod` on `function`. Direct JPype parity testing is
**blocked by M4** (JPype cannot extend Java abstract classes from Python). The
parity harness marks the Java-facing section `@pytest.mark.skip(M4)` and validates
against concrete Python subclasses (e.g. `f(x) = x - 2`).

---

## Java-specific translation decisions

| Java pattern | Python translation | Rule |
|---|---|---|
| `abstract public class FindRoot` | `class FindRoot(abc.ABC)` | R2 |
| `abstract double function(double x0)` | `@abc.abstractmethod function(self, x0)` — no `_` prefix | R1 |
| `IllegalArgumentException` | `ValueError` | R2 |
| `ArithmeticException` | `ArithmeticError` | R2 |
| `Integer.toString(int)` | `str(int)` | R2 |
| `Math.abs(double)` | `abs()` | R2 |
| `public double EvaluationCount()` return type | `-> float` (not `int`) | R9 — matches Java declared signature |

---

## Suspected Java bugs
None. The algorithm is Jack Crenshaw's "world's best root finder".
`BUG_LEDGER` is empty.

---

## Static init order
None.

---

## Thread safety
Not documented. `_mBestX`, `_mBestY`, `_mNEvals` are mutable instance state;
separate instances are safe to use concurrently, but a single instance is not.
