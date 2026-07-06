# LazyEvaluate Conversion Spec

## Java class
`gov.nist.microanalysis.Utility.LazyEvaluate<H>`

Source: `src/gov/nist/microanalysis/Utility/LazyEvaluate.java`

---

## Inbound dependencies (Java imports)
None. No external imports — only `java.lang` types.

---

## Outbound dependents (callers of public methods)

`grep -r "LazyEvaluate" src/` reveals usage in:
- Numerous EPQLibrary classes that cache expensive computed values (e.g. composition caches, spectrum caches)
- All dependents use the same pattern: subclass `LazyEvaluate<T>`, implement `compute()`, call `get()` and `reset()`

---

## Public API surface

| Java signature | Python signature | Notes |
|---|---|---|
| `LazyEvaluate()` | `__init__(self)` | Sets `mValue = None`; adds `threading.Lock` |
| `void reset()` | `reset(self) -> None` | Synchronized on `this`; Python: `with self._lock` |
| `boolean evaluated()` | `evaluated(self) -> bool` | Returns `mValue != null` |
| `boolean equals(Object val)` | `equals(self, val: object) -> bool` | FIX-2: added; also exposed via `__eq__` |
| `H get()` | `get(self) -> H` | Double-checked locking; calls `compute()` once |
| `abstract protected H compute()` | `@abc.abstractmethod compute(self) -> H` | FIX-1: `_compute` → `compute` (R1) |

---

## Private / protected members

| Java | Python |
|---|---|
| `private transient H mValue` | `self.mValue: Optional[H]` |
| `synchronized (this)` blocks | `self._lock: threading.Lock` |

---

## Overloaded methods (split plan)
None.

---

## Mutable-output methods
None. `get()` returns a reference; `reset()` only nulls the internal reference.

---

## Touchpoints into Jama / javax.swing / java.awt / java.io
None.

---

## Abstract class strategy
`LazyEvaluate<H>` is generic-abstract: declares `abstract protected H compute()`. Direct JPype parity testing is **blocked by M4** — JPype cannot extend Java abstract classes from Python. The parity harness marks the Java-facing test class `@pytest.mark.skip(reason="M4: ...")` and validates correctness using concrete Python subclasses (`_ConstantLazy`, `_MutableStateLazy`, `_ListLazy`) against known return values.

---

## Java-specific translation decisions

| Java pattern | Python translation | Rule |
|---|---|---|
| `abstract class LazyEvaluate<H>` | `class LazyEvaluate(abc.ABC, Generic[H])` | R2 |
| `abstract protected H compute()` | `@abc.abstractmethod compute(self) -> H` — no `_` prefix | FIX-1 / R1 |
| `synchronized (this) { mValue = null; }` | `with self._lock: self.mValue = None` | R10 |
| Double-checked locking in `get()` | Same pattern with `threading.Lock` | R10 |
| Java `equals(Object val)` checks `mValue == val` (identity) | Python `equals` uses `==` for two `LazyEvaluate` instances; identity short-circuit preserved | FIX-2 |
| `H mValue = null` as "not computed" sentinel | `mValue: Optional[H] = None`; `compute()` must not return `None` | R2 |

---

## Suspected Java bugs
None identified. The `None`-as-sentinel constraint (compute() returning null causes infinite recomputation in Java via the `assert mValue != null`) is faithfully preserved as an `AssertionError` in the Python port.

---

## Static init order
None.

---

## Thread safety
Java uses `synchronized (this)` for double-checked locking. Python uses `threading.Lock()`. The unsynchronized read of `mValue` before the lock (`res = mValue`) is technically a data race in Java < 5 memory models, but is safe under Java 5+ (`transient` + double-checked locking with volatile). The Python port uses the same pattern; `threading.Lock` provides the same memory-visibility guarantees.
