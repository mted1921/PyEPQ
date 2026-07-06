# Integrator Conversion Spec

## Java class
`gov.nist.microanalysis.Utility.Integrator`

Source: `src/gov/nist/microanalysis/Utility/Integrator.java`

---

## Inbound dependencies (Java imports)
- `gov.nist.microanalysis.Utility.AdaptiveRungeKutta` — base class; `Integrator extends AdaptiveRungeKutta`
- `gov.nist.microanalysis.Utility.UtilException` — caught inside `integrate(double, double)`, swallowed as `Double.NaN`

No EPQException, Jama, or java.awt imports.

---

## Outbound dependents (callers of public methods)
Not exhaustively audited. Callers that need numerical integration provide a concrete subclass implementing `getValue(double x)`.

---

## Public API surface

`Integrator` is a **public abstract class** that extends `AdaptiveRungeKutta`. Its only abstract method is `getValue`.

### Constructors

| Java signature | Python signature | Notes |
|---|---|---|
| `public Integrator()` | `__init__(self, tol: float = 1.0e-6)` | Calls `super().__init__(1)` (one dimension) |
| `public Integrator(double tol)` | collapses with above via default arg | |

### Public instance methods

| Java signature | Python signature | Notes |
|---|---|---|
| `abstract public double getValue(double x)` | `@abc.abstractmethod def getValue(self, x: float) -> float` | Subclasses define the integrand |
| `public double integrate(double lowVal, double highVal)` | `def integrate(self, lowVal: float, highVal: float) -> float` | Returns `float('nan')` on `UtilException`; returns `0.0` if `highVal <= lowVal` |
| `public void derivatives(double x, double[] y, double[] dydx)` | `def derivatives(self, x: float, y: list[float], dydx: list[float]) -> None` | Override of `AdaptiveRungeKutta.derivatives`; sets `dydx[0] = self.getValue(x)` |

---

## Private / protected members

| Java | Python |
|---|---|
| `private final double mTolerance` | `self._mTolerance: float` |

---

## Overloaded methods (split plan)
Constructors collapse into one `__init__` with a default `tol` argument.

`integrate` has one overload. No split needed.

---

## Mutable-output methods
`derivatives` mutates `dydx[0]` in-place — this is required by the `AdaptiveRungeKutta` contract. No R5 guard needed (the contract specifies mutation).

---

## Touchpoints into Jama / javax.swing / java.awt / java.io
None.

---

## Abstract class strategy
IS_ABSTRACT = True.

`getValue(double x)` is the sole abstract method. Subclasses implement it to define the function being integrated.

`derivatives` is a concrete override that wires `getValue` into the Runge-Kutta stepper.

**M4 applies**: JPype cannot extend Java abstract classes from Python. Parity harness strategy: construct a concrete Java subclass of `Integrator` with a known analytic integrand (e.g. `x^2`, integral = `x^3/3`), then compare `integrate(a, b)` against the closed-form result.

---

## Java-specific translation decisions

| Java pattern | Python translation | Rule |
|---|---|---|
| `abstract public class Integrator extends AdaptiveRungeKutta` | `class Integrator(AdaptiveRungeKutta, abc.ABC)` | R2 |
| `super(1)` in constructor | `super().__init__(n_dims=1)` | AdaptiveRungeKutta takes dimension count |
| `catch (final UtilException e) { e.printStackTrace(); return Double.NaN; }` | `except UtilException: return float('nan')` | Swallow, return NaN |
| `highVal > lowVal ? ... : 0.0` | `if highVal > lowVal: ... else: return 0.0` | |
| `integrate(lowVal, highVal, y, mTolerance, 0.05 * (highVal - lowVal))[0]` | `self.integrate_ode(lowVal, highVal, y, self._mTolerance, 0.05 * (highVal - lowVal))[0]` | Calls parent `AdaptiveRungeKutta.integrate` method; rename to avoid shadowing |

**Naming hazard**: `Integrator.integrate(low, high)` (public API) shadows `AdaptiveRungeKutta.integrate(...)` (ODE integrator). In the Java code these are distinct by signature (argument count). In Python, use `_integrate_ode` for the parent-level ODE call inside `Integrator.integrate`.

---

## Suspected Java bugs
None. The logic is straightforward.

---

## Static init order
None.

---

## Thread safety
Not thread-safe. `mTolerance` is read-only after construction. `AdaptiveRungeKutta` internal state (step arrays) is mutable during integration — concurrent calls on the same instance would corrupt state.
