# MCIntegrator Conversion Spec

## Java class
`gov.nist.microanalysis.Utility.MCIntegrator`

Source: `src/gov/nist/microanalysis/Utility/MCIntegrator.java`

---

## Inbound dependencies (Java imports)
- `java.util.Random` — bounding-box sampler (`mRand`)
- `gov.nist.microanalysis.Utility.Math2` — `plusEquals` (accumulate function values), `timesEquals` (scale result by volume/nTests)

No Jama, javax.swing, java.awt, or java.io imports.

---

## Outbound dependents (callers of public methods)

`grep -r "extends MCIntegrator" src/` reveals:
- `EPQLibrary/Armstrong1982ParticleMC.java` — concrete subclass; calls `compute(int)`
- `EPQTests/MCIntegratorTest.java` — test only

All callers subclass `MCIntegrator`, provide `function` and `inside`, and call `compute`.

---

## Public API surface

| Java signature | Python signature | Notes |
|---|---|---|
| `protected MCIntegrator(double[] point1, double[] point2)` | `__init__(self, point1, point2)` | Protected in Java → public `__init__` in Python (no enforcement); assert `len(point1) == len(point2)` preserved |
| `abstract double[] function(double[] args)` | `@abc.abstractmethod function(self, args) -> np.ndarray` | Extension point; un-prefixed per R1 |
| `abstract boolean inside(double[] args)` | `@abc.abstractmethod inside(self, args) -> bool` | Extension point; un-prefixed per R1 |
| `double[] compute(int nTests)` | `compute(self, nTests: int) -> np.ndarray` | JAVA-BUG-1; see below |

---

## Private / protected members

| Java | Python |
|---|---|
| `private final double[] mPoint1` | `self._mPoint1: list[float]` |
| `private final double[] mPoint2` | `self._mPoint2: list[float]` |
| `private final Random mRand` | `self._mRand: JavaRandom` (from `_epq_compat`) |

---

## Overloaded methods (split plan)
None.

---

## Mutable-output methods
None in MCIntegrator's own public API. `Math2.plusEquals` and `Math2.timesEquals` mutate arrays internally within `compute`, but these are not exposed as mutable-output methods to callers.

---

## Touchpoints into Jama / javax.swing / java.awt / java.io
None.

---

## Abstract class strategy
`MCIntegrator` declares `abstract function()` and `abstract inside()`. Direct JPype parity testing is **blocked by M4** (JPype cannot extend Java abstract classes from Python). The parity harness marks the Java-facing section `@pytest.mark.skip(M4)` and validates correctness using concrete Python subclasses against known analytical integrals (sphere volume, unit square, etc.).

---

## Java-specific translation decisions

| Java pattern | Python translation | Rule |
|---|---|---|
| `abstract public class MCIntegrator` | `class MCIntegrator(abc.ABC)` | R2 |
| `abstract public double[] function(double[] args)` | `@abc.abstractmethod function(self, args)` — no `_` prefix | R1 |
| `abstract public boolean inside(double[] args)` | `@abc.abstractmethod inside(self, args)` — no `_` prefix | R1 |
| `private final Random mRand` | `self._mRand = JavaRandom()` from `_epq_compat` | R3 |
| `Math2.plusEquals(inner, f)` | import from sibling Math2 port (filename from UTILITY_LEDGER.md) | R3/sibling import |
| `Math2.timesEquals(volume / nTests, inner)` | same | R3/sibling import |
| JAVA-BUG-1: `timesEquals(v/n, null)` raises when inner is null | Preserved in `compute()`; `compute_strict()` handles gracefully | R6, R10 |

---

## Suspected Java bugs

**JAVA-BUG-1** — `compute` raises when no points land inside the region, or when `nTests=0`.

Two triggering conditions, same root cause (`inner` remains `null`):

1. **Empty region** (`inside()` always returns `False`): `inner` is never set from `null`. Final call `Math2.timesEquals(volume/nTests, null)` raises `NullPointerException` in Java / `TypeError` in Python.

   Exact Java source lines:
   ```java
   double[] inner = null;
   ...
   inner = (inner == null ? f : Math2.plusEquals(inner, f));
   ...
   return Math2.timesEquals(volume / nTests, inner);  // inner still null
   ```

2. **Zero samples** (`nTests=0`): loop body never executes; `inner` stays `null`. Additionally, `volume / 0` produces `Infinity` in Java (IEEE 754 double division), compounding into `Math2.timesEquals(Inf, null)`.

Disposition: **Preserve in `compute()`** (faithful Java behaviour — raises `TypeError` in Python).
**Fix in `compute_strict()`** — return a zero array when `inner` is `None`.

---

## Static init order
None. No static blocks or cross-class static references.

---

## Thread safety
Not thread-safe. `mRand` is a shared `Random` instance with no synchronisation. Each `MCIntegrator` instance has its own `mRand`, so separate instances are safe to use concurrently.
