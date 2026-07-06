# ProgressEvent Conversion Spec

## Java class
`gov.nist.microanalysis.Utility.ProgressEvent`

Source: `src/gov/nist/microanalysis/Utility/ProgressEvent.java`

---

## Inbound dependencies (Java imports)
- `java.awt.event.ActionEvent` — base class; no Python equivalent
- `gov.nist.microanalysis.Utility.Math2` — `Math2.bound(progress, 0, 101)` used in constructor; import from sibling port

No EPQException or Jama imports.

---

## Outbound dependents (callers of public methods)
Not audited. Used by progress-reporting infrastructure to communicate percentage completion to registered `ActionListener`s.

---

## Public API surface

`ProgressEvent` is a **concrete class** that extends `java.awt.event.ActionEvent`. In Python there is no `ActionEvent` hierarchy; port as a plain dataclass.

### Constructor

| Java signature | Python signature | Notes |
|---|---|---|
| `public ProgressEvent(Object src, int id, int progress)` | `__init__(self, src: object, id: int, progress: int)` | Clamps `progress` to `[0, 101]` via `Math2.bound`; stores clamped value |

### Public instance methods

| Java signature | Python signature | Notes |
|---|---|---|
| `public int getProgress()` | `def getProgress(self) -> int` | Returns clamped `_mProgress` |

---

## Private / protected members

| Java | Python |
|---|---|
| `private static final long serialVersionUID = 7157720139752608198L` | Discard |
| `private final int mProgress` | `self._mProgress: int` |

The parent `ActionEvent` fields (`src`, `id`, `command` string) exist in Java but have no meaning in the Python port. Retain `src` and `id` as stored attributes for API surface completeness; the `command` string computed in Java (`Integer.toString(Math2.bound(progress, 0, 101)) + "%"`) can be exposed via a `getCommand()` method or property.

---

## Overloaded methods (split plan)
No overloads.

---

## Mutable-output methods
None.

---

## Touchpoints into Jama / javax.swing / java.awt / java.io

`java.awt.event.ActionEvent` — the base class carries `source`, `id`, and `command` string. **No Python equivalent.** Port `ProgressEvent` as a standalone class (not inheriting from anything meaningful) that stores `src`, `id`, `_mProgress`, and exposes `getProgress()`. Document as R10 deviation.

---

## Abstract class strategy
Not applicable. Concrete class.

---

## Java-specific translation decisions

| Java pattern | Python translation | Rule |
|---|---|---|
| `extends ActionEvent` | Standalone class; no base | R10 deviation — no Java AWT in Python |
| `super(src, id, Integer.toString(Math2.bound(progress, 0, 101)) + "%")` | Store `self._src = src`, `self._id = id`, `self._command = str(Math2.bound(progress, 0, 101)) + "%"` | Simulate ActionEvent fields |
| `Math2.bound(progress, 0, 101)` | `Math2.bound(progress, 0, 101)` | Sibling import |
| `private final int mProgress` | `self._mProgress: int = Math2.bound(progress, 0, 101)` | |

---

## Suspected Java bugs

**JAVA-NOTE-1 — Clamp upper bound is 101, not 100.**
`Math2.bound(progress, 0, 101)` clamps to `[0, 101]`, so a `progress` of `101` is valid. The percentage range should logically be `[0, 100]`. This may be intentional (101 = "done plus one step") or a typo.
Disposition: **Preserve** — port as `Math2.bound(progress, 0, 101)` verbatim; document as `# JAVA-NOTE-1`.

---

## Static init order
None.

---

## Thread safety
Immutable after construction. Safe for concurrent read-only use.
