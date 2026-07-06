# FindRoot Conversion Spec

## Java class
`gov.nist.microanalysis.Utility.FindRoot`

Source: `src/gov/nist/microanalysis/Utility/FindRoot.java`

---

## Inbound dependencies (Java imports)
None. No imports in the Java source.

---

## Outbound dependents (callers of public methods)
`Math2.chiSquaredConfidenceLevel` instantiates an anonymous `FindRoot` subclass and calls `perform()`. Not exhaustively audited.

---

## Public API surface

| Java signature | Python signature | Notes |
|---|---|---|
| `abstract public double function(double x0)` | `def function(self, x0: float) -> float` | `@abc.abstractmethod` — public abstract → NO underscore prefix (R1) |
| `public void initialize(double[] vars)` | `def initialize(self, vars: F64Array) -> None` | No-op base implementation |
| `public double bestX()` | `def bestX(self) -> float` | |
| `public double bestY()` | `def bestY(self) -> float` | |
| `public double EvaluationCount()` | `def EvaluationCount(self) -> float` | Returns `float(self.mNEvals)` — R9: int field, double return |
| `public double perform(double x0, double x2, double eps, int iMax)` | `def perform(self, x0: float, x2: float, eps: float, iMax: int) -> float` | Throws `ValueError` (IllegalArgumentException) and `ArithmeticError` (ArithmeticException) |

---

## Private / protected members

| Java | Python |
|---|---|
| `private double mBestX` | `self._mBestX: float = 0.0` |
| `private double mBestY` | `self._mBestY: float = 0.0` |
| `private int mNEvals = 0` | `self._mNEvals: int = 0` |

---

## Overloaded methods (split plan)
No overloaded methods.

---

## Mutable-output methods
None. `perform` writes to instance fields `_mBestX`, `_mBestY`, `_mNEvals` but does not mutate caller-supplied arrays.

---

## Touchpoints into Jama / javax.swing / java.awt / java.io
None.

---

## Abstract class strategy
IS_ABSTRACT = True.

`function(double x0)` is the sole abstract method. Subclasses override it to define the function whose root is sought.

**M4 applies**: JPype cannot extend Java abstract classes from Python. Direct parity testing of `FindRoot` is blocked.

Parity harness strategy: Create a concrete Python subclass of `FindRoot` that implements `function` as a known polynomial (e.g. `x^2 - 2`, root at `sqrt(2)`). Test `perform` against the closed-form root and verify `bestX`, `bestY`, `EvaluationCount` match expectations analytically.

---

## Java-specific translation decisions

| Java pattern | Python translation | Rule |
|---|---|---|
| `abstract public class FindRoot` | `class FindRoot(abc.ABC)` | R2 |
| `abstract public double function(double x0)` | `@abc.abstractmethod def function(self, x0: float) -> float` | R1 — `abstract` does NOT add underscore |
| `public double EvaluationCount()` | `return float(self._mNEvals)` | R9 — Java implicit widening int→double |
| `new IllegalArgumentException(...)` | `raise ValueError(...)` | Java stdlib exception → nearest Python equivalent |
| `new ArithmeticException(...)` | `raise ArithmeticError(...)` | Java stdlib exception → nearest Python equivalent |
| `int i` loop variables | `i: int` | No integer division in the algorithm; all arithmetic is float |

---

## Suspected Java bugs
None identified. The Crenshaw algorithm is implemented faithfully and the branching logic is correct.

---

## Static init order
None.

---

## Thread safety
Not thread-safe. `mNEvals`, `mBestX`, `mBestY` are instance fields mutated during `perform`. Concurrent calls on the same instance would corrupt state.
