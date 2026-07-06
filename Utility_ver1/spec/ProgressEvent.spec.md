# ProgressEvent Conversion Spec

## Java class
`gov.nist.microanalysis.Utility.ProgressEvent`

Source: `src/gov/nist/microanalysis/Utility/ProgressEvent.java`

---

## Inbound dependencies (Java imports)
- `java.awt.event.ActionEvent` — superclass; not available in Python (DEVIATION-1)
- `gov.nist.microanalysis.Utility.Math2` — `Math2.bound(progress, 0, 101)` in constructor

---

## Outbound dependents (callers of public methods)
- EPQLibrary simulation classes that report Monte Carlo progress to GUI listeners
- All callers use only `getProgress()` and the inherited `ActionEvent` fields (`source`, `id`, `command`)

---

## Public API surface

| Java signature | Python signature | Notes |
|---|---|---|
| `ProgressEvent(Object src, int id, int progress)` | `__init__(self, src: Any, id: int, progress: int)` | Bounds progress to [0, 100] via `bound_int(progress, 0, 101)` |
| `int getProgress()` | `getProgress(self) -> int` | Returns bounded `mProgress` |
| *(inherited)* `Object getSource()` | `getSource(self) -> Any` | DEVIATION-1: simulated; returns `self.source` |
| *(inherited)* `int getID()` | `getID(self) -> int` | DEVIATION-1: simulated; returns `self.id` |
| *(inherited)* `String getActionCommand()` | `getActionCommand(self) -> str` | DEVIATION-1: simulated; returns `self.command` |

---

## Private / protected members

| Java | Python |
|---|---|
| `private final int mProgress` | `self.mProgress: int` |
| `private static final long serialVersionUID` | `_serialVersionUID: int = 7157720139752608198` |
| *(inherited ActionEvent)* source, id, actionCommand | `self.source`, `self.id`, `self.command` |

---

## Overloaded methods (split plan)
None.

---

## Mutable-output methods
None. All fields set in constructor and read-only thereafter.

---

## Touchpoints into Jama / javax.swing / java.awt / java.io
**`java.awt.event.ActionEvent`** — superclass. Not available in Python.

---

## Abstract class strategy
Not abstract. Concrete class. However, `java.awt.event.ActionEvent` cannot be extended in Python (DEVIATION-1), so Java parity comparison is **not feasible** — the port is a standalone reimplementation that simulates the required fields.

The parity harness tests the Python port behaviourally (constructor bounds, `getProgress`, simulated accessors) with no Java comparison.

---

## Java-specific translation decisions (DEVIATION-1)

`java.awt.event.ActionEvent` has no Python equivalent. The port does not extend any base class and instead simulates the three fields callers actually use:

| Java field/method (via ActionEvent) | Python simulation |
|---|---|
| `super(src, id, commandStr)` | `self.source = src; self.id = id; self.command = commandStr` |
| `getSource()` | `return self.source` |
| `getID()` | `return self.id` |
| `getActionCommand()` | `return self.command` |

The `command` string is constructed as `str(bounded_progress) + "%"` to match Java's `Integer.toString(Math2.bound(progress, 0, 101)) + "%"`.

`Math2.bound(progress, 0, 101)` uses the **int overload** (upper bound exclusive), so valid range is [0, 100] inclusive. Progress = 100 returns 100; progress = 101 returns 100; progress < 0 returns 0.

---

## Suspected Java bugs
None. The `bound(progress, 0, 101)` call is correct: Java's `int` overload of `bound` is upper-exclusive, so `bound(x, 0, 101)` returns values in [0, 100].

---

## Static init order
`private static final long serialVersionUID = 7157720139752608198L` — constant; safe.

---

## Thread safety
Not documented as thread-safe. All fields are set in constructor; reads after construction are safe.
