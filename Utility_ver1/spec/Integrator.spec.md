# Integrator Conversion Spec

## Java class
`gov.nist.microanalysis.Utility.Integrator`

Source: `src/gov/nist/microanalysis/Utility/Integrator.java`

---

## Inbound dependencies (Java imports)
- `gov.nist.microanalysis.Utility.AdaptiveRungeKutta` — superclass; `integrate(x1,x2,y,eps,h1)` is called internally
- `gov.nist.microanalysis.Utility.UtilException` — caught in `integrate(double,double)`

No Jama, javax.swing, java.awt, or java.io imports.

---

## Outbound dependents (callers of public methods)

`grep -r "extends Integrator" src/` reveals subclasses in:
- `EPQLibrary/Armstrong1982ParticleCorrection.java` — anonymous subclass providing `getValue`
- `EPQLibrary/Armstrong1982ParticleMC.java` — anonymous subclass providing `getValue`
- `JMONSEL/ChargingListener.java` — anonymous subclass providing `getValue`
- `EPQTests/IntegratorTest.java` — test only

All callers subclass `Integrator` and call `integrate(double, double)`. The `getValue` extension point is the only public contract callers depend on.

---

## Public API surface

| Java signature | Python signature | Notes |
|---|---|---|
| `Integrator()` | `__init__(self, tol=None)` | Default tol = 1e-6 |
| `Integrator(double tol)` | `__init__(self, tol: float)` | Explicit tolerance; merged with default via `tol=None` |
| `double integrate(double lowVal, double highVal)` | `integrate(self, lowVal: float, highVal: float) -> float` | JAVA-BUG-1; see below. Hides ARK 6-arg overload — `# type: ignore[override]` required |
| `void derivatives(double x, double[] y, double[] dydx)` | `derivatives(self, x: float, y: F64Array, dydx: F64Array) -> None` | `@Override`; writes `getValue(x)` into `dydx[0]` |
| `abstract double getValue(double x)` | `@abc.abstractmethod getValue(self, x: float) -> float` | Extension point; un-prefixed per R1 |

---

## Private / protected members

| Java | Python |
|---|---|
| `private final double mTolerance` | `self._mTolerance: float` |

---

## Overloaded methods (split plan)
None. The two Java constructors collapse into a single `__init__` with `tol=None` default.

The inherited `integrate` overloads (6-arg ARK vs 2-arg Integrator) are a **Java method-hiding** case across the inheritance hierarchy, not a same-class overload. Python resolves this with `# type: ignore[override]` — see translation decisions.

---

## Mutable-output methods
- **`derivatives`**: mutates `dydx` in place (`dydx[0] = getValue(x)`). R5 guard (`_require_mutable_f64`) applied.

---

## Touchpoints into Jama / javax.swing / java.awt / java.io
None.

---

## Abstract class strategy
`Integrator` is doubly abstract: it declares `abstract getValue()` and extends the abstract `AdaptiveRungeKutta`. Direct JPype parity testing is **blocked by M4** twice over. The parity harness marks the Java-facing section `@pytest.mark.skip(M4)` and validates correctness using concrete Python subclasses against closed-form analytical integrals.

---

## Java-specific translation decisions

| Java pattern | Python translation | Rule |
|---|---|---|
| `abstract public class Integrator` | `class Integrator(AdaptiveRungeKutta, abc.ABC)` | R2 |
| `abstract public double getValue(double x)` | `@abc.abstractmethod getValue(self, x)` — no `_` prefix | R1 |
| `integrate(double,double)` hides `ARK.integrate(double[],double,double[],double,double)` | `# type: ignore[override]` on 2-arg form; `super().integrate_literal()` calls ARK correctly | R4/Java-hiding |
| JAVA-BUG-1: returns `0.0` when `highVal <= lowVal` | Preserved in `integrate_literal()`; `integrate()` uses `scipy.integrate.quad` (SCIPY-DEV-1) | R6, R10 |
| `e.printStackTrace(); return Double.NaN` | `traceback.print_exc(); return math.nan` | R2 |

---

## Suspected Java bugs

**JAVA-BUG-1** — `integrate` returns 0.0 for reversed limits.

Exact Java source line:
```java
return highVal > lowVal ? integrate(lowVal, highVal, y, mTolerance, 0.05 * (highVal - lowVal))[0] : 0.0;
```

Disposition: **Preserve in `integrate_literal`** (faithful copy of Java behaviour).
**Fix in `integrate`** via `scipy.integrate.quad`, which returns the correct signed integral (SCIPY-DEV-1).

---

## Static init order
None. No static blocks or cross-class static references.

---

## Thread safety
Not documented in Javadoc. `mTolerance` is `final` so it is safe, but the inherited ARK workspace arrays (`mWs2`…`mYSave`) are mutable instance state — not thread-safe.
