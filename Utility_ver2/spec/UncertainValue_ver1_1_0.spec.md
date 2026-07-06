# UncertainValue Conversion Spec

## Java class
`gov.nist.microanalysis.Utility.UncertainValue`  *(`@Deprecated`)*

Source: `src/gov/nist/microanalysis/Utility/UncertainValue.java`

---

## Inbound dependencies (Java imports)
- `gov.nist.microanalysis.Utility.UncertainValue2` — produced by `readResolve()`; import from the sibling port (filename from `UTILITY_LEDGER.md`).

No other imports.

---

## Outbound dependents (callers of public methods)
None call this class directly. It exists solely as a **legacy XStream deserialization shim**: old serialized `UncertainValue` blobs are read back and `readResolve()` maps them onto the modern `UncertainValue2`. Not exhaustively audited.

---

## Public API surface

`UncertainValue` is a `@Deprecated` value carrier with two fields and a single serialization hook. It has **no public constructor** and **no getters/setters** in the Java source — the fields are populated by the XStream deserializer via reflection.

### Constructor

| Java signature | Python signature | Notes |
|---|---|---|
| *(none in Java; fields set by deserialization)* | `__init__(self, mValue: float = 0.0, mSigma: float = 0.0) -> None` | **Added for Python usability** — Java relied on reflective field injection during deserialization; Python needs an explicit constructor to populate the fields. Document as an R10 deviation (no Java equivalent) |

### Public instance methods

| Java signature | Python signature | Notes |
|---|---|---|
| `public Object readResolve()` | `def readResolve(self) -> UncertainValue2` | Returns `UncertainValue2(self._mValue, self._mSigma)`. Java serialization hook (`readResolve`) — keep the Java name; it is not a Python protocol method (see translation note) |

---

## Private / protected members

| Java | Python |
|---|---|
| `private double mValue` | `self._mValue: float` |
| `private double mSigma` | `self._mSigma: float` |

---

## Overloaded methods (split plan)
No overloads.

---

## Mutable-output methods
None.

---

## Touchpoints into Jama / javax.swing / java.awt / java.io
None directly. **Serialization touchpoint:** `readResolve()` is part of the Java object-serialization protocol (`java.io.Serializable`). Python's serialization protocol differs (`__reduce__` / `__setstate__`); the Java hook is ported as an ordinary named method, not wired into `pickle`. Document as an R10 deviation.

---

## Abstract class strategy
Not applicable. `UncertainValue` is a concrete (deprecated) class.

IS_ABSTRACT = False.

---

## Java-specific translation decisions

| Java pattern | Python translation | Rule |
|---|---|---|
| `@Deprecated` class | Add a module/class docstring deprecation note; optionally emit `warnings.warn(..., DeprecationWarning)` in `__init__`. Document the choice in `CHANGES` | R10 |
| `public Object readResolve()` | `def readResolve(self) -> UncertainValue2` returning `UncertainValue2(self._mValue, self._mSigma)` | Keep Java name; not a Python pickle hook |
| No Java constructor (reflective field set) | Add `__init__(self, mValue=0.0, mSigma=0.0)` | R10 deviation — needed to construct/test in Python |
| `new UncertainValue2(mValue, mSigma)` | `UncertainValue2(self._mValue, self._mSigma)` from sibling port | R3 / sibling import |

---

## Suspected Java bugs
None identified. The class is a trivial legacy shim.

---

## Static init order
None. Both fields are instance state.

---

## Thread safety
Effectively immutable after construction (in practice, after deserialization). `readResolve()` only reads the two fields and constructs a new `UncertainValue2`. Safe for concurrent read-only use.
