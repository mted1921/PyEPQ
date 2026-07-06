# MCUncertaintyEngine Conversion Spec

## Java class
`gov.nist.microanalysis.Utility.MCUncertaintyEngine`

Source: `src/gov/nist/microanalysis/Utility/MCUncertaintyEngine.java`

> **Dependency note.** The `UncertainValueMC` and `DescriptiveStatistics` specs were
> repaired to **ver1_1_1** to match the Java source (adding the
> `UncertainValueMC(UncertainValue2, Map)` constructor and the static
> `DescriptiveStatistics.compute(Collection)` → `compute_collection`). This spec is
> consistent with those corrected APIs.

---

## Inbound dependencies (Java imports)
- `java.util.ArrayList`, `java.util.List` — `mResults` storage; use Python `list`.
- `java.util.Map`, `java.util.TreeMap` — the per-iteration deviates map shared across arguments; use `dict` (insertion-ordered; sorted-key semantics are not relied upon here — keys are correlation tags).
- `gov.nist.microanalysis.Utility.UncertainValueMC` — sample carrier; import from the sibling port. Uses constructor `UncertainValueMC(UncertainValue2 uv, Map<String, Double> deviates)` and `nominalValue()`.
- `gov.nist.microanalysis.Utility.UncertainValue2` — `asUncertainValue2(Number)` and the `(double, double)` constructor; import from the sibling port.
- `gov.nist.microanalysis.Utility.DescriptiveStatistics` — `compute(Collection)`, `average()`, `standardDeviation()`; import from the sibling port.

---

## Outbound dependents (callers of public methods)
Base class for Monte Carlo uncertainty propagation: a subclass implements `compute(UncertainValueMC[])` to evaluate the model once per random draw. Not exhaustively audited.

---

## Public API surface

`MCUncertaintyEngine` is an **abstract class**. The constructor immediately runs the Monte Carlo iterations by calling the abstract `compute` repeatedly.

### Constructor

| Java signature | Python signature | Notes |
|---|---|---|
| `public MCUncertaintyEngine(int iterations, Number[] arguments)` | `__init__(self, iterations: int, arguments: list)` | Stores `arguments`, initialises `_mResults = []`, then calls `doIterations(iterations)`. `arguments` elements are `float`/`int`/`UncertainValue2` (anything `asUncertainValue2` accepts) |

### Public instance methods

| Java signature | Python signature | Notes |
|---|---|---|
| `public void doIterations(int iterations)` | `def doIterations(self, iterations: int) -> None` | Per iteration: new `dict` `rd`; build `args[j] = UncertainValueMC(UncertainValue2.asUncertainValue2(self._mArguments[j]), rd)`; append `self.compute(args)` to `_mResults`. **The single shared `rd` per iteration is intentional** — it correlates the random draws across arguments |
| `public List<UncertainValueMC> getResults()` | `def getResults(self) -> list[UncertainValueMC]` | Returns the live `_mResults` list (Java returns the field directly; preserve reference semantics, do not copy) |
| `public DescriptiveStatistics getStatistics()` | `def getStatistics(self) -> DescriptiveStatistics` | `DescriptiveStatistics.compute_collection(self._mResults)` — the `Collection` overload (ver1_1_1) |
| `public double nominalValue()` | `def nominalValue(self) -> float` | `self._mResults[0].nominalValue()` — see empty-results edge case |
| `public UncertainValue2 getResult()` | `def getResult(self) -> UncertainValue2` | `UncertainValue2(stats.average(), stats.standardDeviation())` |

### Abstract method (extension point)

| Java signature | Python signature | Notes |
|---|---|---|
| `abstract public UncertainValueMC compute(UncertainValueMC[] arguments)` | `@abc.abstractmethod def compute(self, arguments: list[UncertainValueMC]) -> UncertainValueMC` | `public abstract` → **no underscore** (R1); subclasses implement `compute` |

---

## Private / protected members

| Java | Python |
|---|---|
| `private final ArrayList<UncertainValueMC> mResults` | `self._mResults: list[UncertainValueMC]` |
| `private final Number[] mArguments` | `self._mArguments: list` — stored reference to the constructor argument |

---

## Overloaded methods (split plan)
No overloads in this class. **Dependency note:** `DescriptiveStatistics.compute` is overloaded in its own class (`compute(Number[])` and `compute(Collection)`); the DescriptiveStatistics ver1_1_1 spec R4-splits them as `compute_array` / `compute_collection`. Call **`compute_collection`** here.

---

## Mutable-output methods
None expose caller buffers. `doIterations` mutates `self._mResults` (its purpose). The shared `rd` dict is mutated by the `UncertainValueMC` constructor as it registers deviates — by design; no R5 guard.

---

## Touchpoints into Jama / javax.swing / java.awt / java.io
None.

---

## Abstract class strategy
IS_ABSTRACT = True (`compute` is abstract).

Port as `class MCUncertaintyEngine(abc.ABC)` with `compute` decorated `@abc.abstractmethod`. Note the constructor calls `self.compute(...)` during `__init__` (via `doIterations`) — a subclass instance therefore runs the simulation at construction time; this matches Java.

**M4 applies.** JPype cannot extend the Java abstract class from Python. Parity strategy:
- Mark the direct parity class `@pytest.mark.skip(reason="M4: ...")`.
- Validate with a **concrete Python subclass** whose `compute` implements a known closed-form propagation (e.g. `z = a + b`, or `z = a*b`), seed the underlying RNG (`JavaRandom`) deterministically, and assert `getResult()` (mean ± sigma) and `nominalValue()` against analytic uncertainty-propagation values within a statistical tolerance (`TOL_NR_LIB`/`TOL_COMPOUND`).

---

## Java-specific translation decisions

| Java pattern | Python translation | Rule |
|---|---|---|
| `abstract public UncertainValueMC compute(...)` | `@abc.abstractmethod def compute(...)` — un-prefixed | R1 — `public abstract` does NOT add `_` |
| `new TreeMap<String, Double>()` | `{}` (`dict`); keys are correlation tags, sorted order not required | Cross-Cutting — collections |
| `new UncertainValueMC(UncertainValue2.asUncertainValue2(x), rd)` | `UncertainValueMC(UncertainValue2.asUncertainValue2(x), rd)` | depends on UncertainValueMC port's 2-arg ctor |
| `mResults.iterator().next()` | `self._mResults[0]` | first element |
| `Number[] arguments` | `list` of numbers / `UncertainValue2` | |
| constructor runs `doIterations` | keep the side-effecting `__init__` | faithful port |

---

## Suspected Java bugs
None identified.

---

## Edge cases (return value on every code path — P10 discipline)
- **`nominalValue()` with zero iterations** (`_mResults` empty): Java's `mResults.iterator().next()` throws `NoSuchElementException`. Port: `self._mResults[0]` raises `IndexError`. Do **not** silently return `0.0` or `None` — preserve the raised exception (document it; the parity test asserts the raise).
- **`getResult()` / `getStatistics()` with zero iterations**: delegates to `DescriptiveStatistics.compute([])`; the return/raise behaviour is **whatever that port does for an empty collection**. Confirm against the `DescriptiveStatistics` port and pin the expected result (value or exception) explicitly in the test — do not assume.
- **Normal path** (`iterations > 0`): `_mResults` has length `iterations`; all accessors return well-defined values.

---

## Static init order
None. All state is instance state set in `__init__`.

---

## Thread safety
Not thread-safe. The constructor and `doIterations` mutate `_mResults`; concurrent calls corrupt it. Each iteration's `rd` map is local to that iteration, so iterations are independent in principle but the shared `_mResults` accumulator is not guarded.
