# LazyEvaluate Conversion Spec

## Java class
`gov.nist.microanalysis.Utility.LazyEvaluate`

Source: `src/gov/nist/microanalysis/Utility/LazyEvaluate.java`

---

## Inbound dependencies (Java imports)
None. The Java source has no imports. It is a self-contained generic abstract base class `LazyEvaluate<H>`.

---

## Outbound dependents (callers of public methods)
Used wherever an expensive value should be computed once on first access and cached. `LinearRegression` extends/uses this lazy-evaluation pattern. Not exhaustively audited.

---

## Public API surface

`LazyEvaluate<H>` is a **generic abstract class**. Subclasses implement `compute()` to produce the cached value.

### Constructor

| Java signature | Python signature | Notes |
|---|---|---|
| `public LazyEvaluate()` | `__init__(self) -> None` | Sets the cached value to `None` and creates the lock |

### Public instance methods

| Java signature | Python signature | Notes |
|---|---|---|
| `public void reset()` | `def reset(self) -> None` | Clears the cached value under the lock; next `get()` recomputes |
| `public boolean evaluated()` | `def evaluated(self) -> bool` | `return self._mValue is not None` |
| `public boolean equals(Object val)` | `def __eq__(self, val) -> bool` + `def equals(self, val) -> bool` | R2 — dunder + named alias. See translation note: Java uses reference identity (`mValue == val`), so port as `self.evaluated() and (self._mValue is val)` |
| `public H get()` | `def get(self) -> H` | Double-checked-locking lazy compute; calls `_compute()` once and caches |

### Abstract method (extension point)

| Java signature | Python signature | Notes |
|---|---|---|
| `abstract protected H compute()` | `@abc.abstractmethod def _compute(self) -> H` | `protected` → `_` prefix (R1). Subclasses override `_compute()` |

---

## Private / protected members

| Java | Python |
|---|---|
| `private transient H mValue` | `self._mValue: Optional[H]` — `transient` has no Python equivalent (serialization-only); port as a plain attribute initialized to `None` |
| `synchronized (this)` blocks in `reset()` / `get()` | `self._mLock: threading.Lock` guarding the same critical sections |

---

## Overloaded methods (split plan)
No overloads.

---

## Mutable-output methods
None. `get()` and `reset()` mutate only `self._mValue` (internal cache state); no caller-supplied buffers are mutated. No R5 guard needed.

---

## Touchpoints into Jama / javax.swing / java.awt / java.io
None.

---

## Abstract class strategy
IS_ABSTRACT = True.

Port as `class LazyEvaluate(abc.ABC, Generic[H])` with `_compute` decorated `@abc.abstractmethod`. Define `H = TypeVar("H")` at module level.

**M4 applies.** JPype cannot extend the Java abstract class from Python, so the parity harness cannot subclass `LazyEvaluate` on the Java side directly. The parity test class must be marked `@pytest.mark.skip(reason="M4: ...")`. Validate correctness analytically with a **concrete Python subclass** whose `_compute()` returns a known sentinel and asserts:
- `get()` invokes `_compute()` exactly once across repeated calls (use a call counter);
- `evaluated()` is `False` before the first `get()` and `True` after;
- `reset()` forces the next `get()` to recompute (counter increments again).

---

## Java-specific translation decisions

| Java pattern | Python translation | Rule |
|---|---|---|
| `abstract class LazyEvaluate<H>` | `class LazyEvaluate(abc.ABC, Generic[H])` + `H = TypeVar("H")` | Generics |
| `abstract protected H compute()` | `@abc.abstractmethod def _compute(self) -> H` | R1 — `protected` adds `_`; `abstract` does not change the name |
| `synchronized (this) { ... }` | `with self._mLock:` around the same block | Cross-Cutting — `synchronized` → `threading.Lock` |
| Double-checked locking in `get()` | Preserve the two-phase check (`res = self._mValue; if res is None: with lock: if self._mValue is None: ...`) | Faithful port |
| `assert mValue != null;` after compute (inside `get()`) | **Omit** — Java `assert` is disabled by default; it is a post-condition sanity check, not load-bearing production behaviour. Do not emit a Python `assert` | Cross-Cutting — Java `assert` |
| `equals(Object val)` → `mValue == val` | `self.evaluated() and (self._mValue is val)` — Java `==` on objects is reference identity, so use `is`, not `==` | R2; identity semantics |
| `transient H mValue` | plain attribute; document that Python serialization is out of scope | R10 deviation note |

---

## Suspected Java bugs
None identified. `equals()` comparing the cached value to an arbitrary argument by identity is unusual but is the intended (if odd) Java behaviour — preserve it, do not "fix" it.

---

## Static init order
None. No class-level state; `_mValue` and `_mLock` are instance state set in `__init__`.

---

## Thread safety
Thread-safe by design via the `synchronized`/lock-guarded `reset()` and `get()`. The double-checked-locking `get()` ensures `_compute()` runs at most once even under concurrent first access. The port must preserve this with `threading.Lock`; do not drop the lock for "simplicity."
