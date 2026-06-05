# FindRoot Conversion Specification

## 1. Inbound dependencies
- None (uses only `java.lang.Math.abs`, mapped to Python built-in `abs()`)

## 2. Outbound dependents
- `Math2` — uses `FindRoot` via an anonymous inner class for `chiSquaredConfidenceLevel`
  and exposes a `findRoot` wrapper. The Python port in `Math2_ver1` currently bridges
  this with `scipy.optimize.brentq`; once `FindRoot_ver1` exists, that bridge can be
  replaced with a concrete `FindRoot` subclass.
- Any EPQLibrary class that calls `Math2.findRoot(...)` or subclasses `FindRoot` directly.

## 3. Public API surface
| Member | Kind | Signature |
|---|---|---|
| `function` | abstract | `function(x0: float) -> float` |
| `initialize` | virtual (no-op default) | `initialize(vars: list[float]) -> None` |
| `bestX` | getter | `bestX() -> float` |
| `bestY` | getter | `bestY() -> float` |
| `EvaluationCount` | getter | `EvaluationCount() -> float` |
| `perform` | algorithm | `perform(x0, x2, eps, iMax) -> float` |

## 4. Overloaded methods
None.

## 5. Mutable-output methods
None. `perform` modifies `mBestX`, `mBestY`, `mNEvals` on `self` (instance state,
not a passed buffer — no `_require_mutable_f64` guard needed).

## 6. External touchpoints
- `Math.abs(double)` → Python built-in `abs()`.
- `Integer.toString(int)` → Python `str(int)`.
- `IllegalArgumentException` → Python `ValueError`.
- `ArithmeticException` → Python `ArithmeticError`.

## 7. Suspected Java bugs
None. The algorithm is Jack Crenshaw's "world's best root finder"
(bisection + inverse quadratic interpolation). No deviations from the
reference implementation are needed.

Note: `EvaluationCount()` is declared `public double` in Java (return type `double`)
but `mNEvals` is an `int`. The Python port declares the return type as `float` to
match Java's declared signature; the underlying field remains `int`.
