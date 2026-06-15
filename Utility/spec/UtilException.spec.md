# UtilException Conversion Spec

## Java class
`gov.nist.microanalysis.Utility.UtilException`

Source: `src/gov/nist/microanalysis/Utility/UtilException.java`

---

## Inbound dependencies (Java imports)
- None (`java.lang.Exception` is implicit in Java)

---

## Outbound dependents (callers of public methods)

Used throughout the Utility package wherever non-fatal utility errors are thrown.
Callers include `Math2`, `LinearLeastSquares`, and other Utility classes that
need a package-specific checked exception distinct from `EPQException`.

---

## Public API surface

| Java signature | Python signature | Notes |
|---|---|---|
| `UtilException()` | `__init__(self)` | No-message constructor |
| `UtilException(String string)` | `from_string(cls, string: str)` | `@classmethod` |
| `UtilException(String string, Throwable throwable)` | `from_string_throwable(cls, string: str, throwable: BaseException)` | `@classmethod`; sets `__cause__` |
| `UtilException(Throwable throwable)` | `from_throwable(cls, throwable: BaseException)` | `@classmethod`; `str(throwable)` as message; sets `__cause__` |

---

## Private / protected members
None.

---

## Overloaded methods (split plan)
Four constructor overloads per R4:
- **`__init__(self)`** — maps to `super().__init__()`.
- **`from_string`** — maps to `super().__init__(string)`.
- **`from_string_throwable`** — maps to `super().__init__(string)`; sets `__cause__ = throwable`.
- **`from_throwable`** — Java calls `throwable.toString()` as the message; Python uses `str(throwable)` and sets `__cause__ = throwable`.

---

## Mutable-output methods
None.

---

## Touchpoints into Jama / javax.swing / java.awt / java.io
None. `java.lang.Exception` → Python `Exception` (stdlib).

---

## Java-specific translation decisions

| Java pattern | Python translation | Rule |
|---|---|---|
| `java.lang.Exception` superclass | `Exception` | R2 |
| Four constructor overloads | `__init__` + three `@classmethod` factory methods | R4 |
| `throwable.toString()` as message | `str(throwable)` | R2 |
| Exception chaining | `exception.__cause__ = throwable` | R2 |

---

## Suspected Java bugs
None.

---

## Static init order
None.

---

## Thread safety
Not applicable. Exception objects are not shared across threads.
