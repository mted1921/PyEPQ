# UtilException Conversion Specification

## 1. Inbound dependencies
- None (uses only `java.lang.Exception`, implicit in java.lang)

## 2. Outbound dependents
- Used throughout the Utility package wherever non-fatal utility errors are thrown
- Callers include `Math2`, `LinearLeastSquares`, and other Utility classes that need
  a package-specific checked exception distinct from `EPQException`

## 3. Public API surface
| Java constructor | Python equivalent |
|---|---|
| `UtilException()` | `__init__(self)` |
| `UtilException(String string)` | `@classmethod from_string(cls, string: str)` |
| `UtilException(String string, Throwable throwable)` | `@classmethod from_string_throwable(cls, string: str, throwable: BaseException)` |
| `UtilException(Throwable throwable)` | `@classmethod from_throwable(cls, throwable: BaseException)` |

## 4. Overloaded constructors (split plan)
Four constructor overloads per R4:
- **`__init__(self)`** — maps to `super()`.
- **`from_string`** — maps to `super(string)`.
- **`from_string_throwable`** — maps to `super(string, throwable)`; sets `__cause__`.
- **`from_throwable`** — maps to `super(throwable)`; Java calls `throwable.toString()`
  as the message, so Python uses `str(throwable)` and sets `__cause__`.

## 5. Mutable-output methods
None.

## 6. External touchpoints
None. `java.lang.Exception` → Python `Exception` (stdlib).

## 7. Suspected Java bugs
None.
