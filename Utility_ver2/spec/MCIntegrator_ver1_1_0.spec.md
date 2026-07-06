# MCIntegrator Conversion Spec

## Java class
`gov.nist.microanalysis.Utility.MCIntegrator`

Source: `src/gov/nist/microanalysis/Utility/MCIntegrator.java`

---

## Inbound dependencies (Java imports)
- `java.util.Random` — internal RNG; replace with Python `random.Random` for the primary port
- `gov.nist.microanalysis.Utility.Math2` — `Math2.plusEquals` and `Math2.timesEquals` used in `compute`; import from sibling port

No EPQException, Jama, or java.awt imports.

---

## Outbound dependents (callers of public methods)
Not audited. Callers implement concrete subclasses for Monte Carlo integration of multidimensional domains.

---

## Public API surface

`MCIntegrator` is a **public abstract class** with two abstract methods.

### Abstract methods

| Java signature | Python signature | Notes |
|---|---|---|
| `abstract public double[] function(double[] args)` | `@abc.abstractmethod def function(self, args: list[float]) -> list[float]` | Integrand — returns a vector of function values |
| `abstract public boolean inside(double[] args)` | `@abc.abstractmethod def inside(self, args: list[float]) -> bool` | Domain indicator — `True` if `args` is inside the integration volume |

### Constructor

| Java signature | Python signature | Notes |
|---|---|---|
| `protected MCIntegrator(double[] point1, double[] point2)` | `__init__(self, point1: list[float], point2: list[float])` | Constructor is `protected` in Java — in Python, prefix `__init__` with no visibility restriction; document that subclasses should call `super().__init__(point1, point2)` |

### Public instance methods

| Java signature | Python signature | Notes |
|---|---|---|
| `public double[] compute(int nTests)` | `def compute(self, nTests: int) -> Optional[list[float]]` | Monte Carlo integration; returns `None` if no sample landed inside the domain |

---

## Private / protected members

| Java | Python |
|---|---|
| `private final double[] mPoint1` | `self._mPoint1: list[float]` — defensive copy in `__init__` (R5) |
| `private final double[] mPoint2` | `self._mPoint2: list[float]` — defensive copy |
| `private final Random mRand` | `self._mRand: random.Random` |

---

## Overloaded methods (split plan)
No overloads.

---

## Mutable-output methods

`compute` calls `Math2.plusEquals(inner, f)` and `Math2.timesEquals(volume/nTests, inner)` — these mutate their first argument in-place. Both calls use local temporaries; no caller-supplied arrays are mutated.

Port: use `inner = Math2.plusEquals(inner, f)` (reassign to returned reference, matching Java semantics where these return the mutated array).

---

## Touchpoints into Jama / javax.swing / java.awt / java.io
None. `java.util.Random` maps to `random.Random`.

---

## Abstract class strategy
IS_ABSTRACT = True.

Two abstract methods: `function` and `inside`.

**M4 applies**: JPype cannot extend Java abstract classes from Python. Parity harness strategy: construct a concrete Java subclass of `MCIntegrator` with a known domain and integrand (e.g. integrand = 1.0 inside the unit hypercube, so the integral equals the volume), compare `compute(nTests)` against the known volume using a fixed seed for reproducibility.

---

## Java-specific translation decisions

| Java pattern | Python translation | Rule |
|---|---|---|
| `protected MCIntegrator(double[] point1, double[] point2)` | `def __init__(self, point1: list[float], point2: list[float])` | No Python equivalent for `protected`; prefix docstring note |
| `new Random()` unseeded | `random.Random()` — unseeded | Matches Java `new Random()` behavior |
| `assert (point1.length == point2.length)` | `assert len(point1) == len(point2)` | |
| `inner == null` first-call check | `if inner is None: inner = f[:]  else: ...` | |
| `Math2.plusEquals(inner, f)` | `inner = Math2.plusEquals(inner, f)` | Mutable in-place + returns the mutated array |
| `Math2.timesEquals(volume / nTests, inner)` | `return Math2.timesEquals(volume / nTests, inner)` | |

---

## Suspected Java bugs

**No confirmed bugs.** However:

**JAVA-NOTE-1 — `compute` returns `None` if no sample is inside the domain.**
Java source: `return Math2.timesEquals(volume / nTests, inner)` where `inner` may still be `null`. If no random point landed inside the domain, `Math2.timesEquals(scalar, null)` is called. The behavior of `Math2.timesEquals(double, double[])` when the array argument is `null` is undefined (likely `NullPointerException`).
Disposition: **Guard** — in the Python port, check `if inner is None: return None` before scaling and document as R10 deviation.

---

## Static init order
None.

---

## Thread safety
Not thread-safe. `_mRand` is a shared mutable `Random` instance; concurrent calls to `compute` on the same instance would produce incorrect results.
