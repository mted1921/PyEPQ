`spec/UtilException.spec.md`

# UtilException Conversion Spec

## Java class
`gov.nist.microanalysis.Utility.UtilException`

Source: `src/gov/nist/microanalysis/Utility/UtilException.java`

---

## Inbound dependencies (Java imports)
- None. Implicitly relies on `java.lang.Exception`, `java.lang.String`, and `java.lang.Throwable`.

---

## Outbound dependents (callers of public methods)
Not audited.

---

## Public API surface

| Java signature | Python signature | Notes |
|---|---|---|
| `public UtilException()` | `__init__(self, message=None, cause=None)` | Standard exception constructor |
| `public UtilException(String string)` | handled by `__init__` default args | |
| `public UtilException(String string, Throwable throwable)` | handled by `__init__` default args | Maps to Python exception chaining |
| `public UtilException(Throwable throwable)` | handled by `__init__` default args | |

---

## Private / protected members

| Java | Python |
|---|---|
| `private static final long serialVersionUID = 0x1;` | Discard / Not applicable in Python |

---

## Overloaded methods (split plan)
The overloaded constructors in Java (`UtilException()`, `UtilException(String)`, `UtilException(String, Throwable)`, `UtilException(Throwable)`) translate to a single Python constructor using default arguments (`message=None`, `cause=None`). Python's native `Exception` class handles arbitrary arguments natively, but explicit mapping of the `cause` parameter will align with Python 3's exception chaining (`raise ... from cause`).

---

## Mutable-output methods
None.

---

## Touchpoints into Jama / javax.swing / java.awt / java.io
None.

---

## Abstract class strategy
Not applicable. Class is not abstract.

---

## Java-specific translation decisions

| Java pattern | Python translation | Rule |
|---|---|---|
| `extends Exception` | Inherits from standard Python `Exception` | Subclassing core language exceptions |
| `Throwable throwable` | Mapped to Python `Exception` | Java `Throwable` equates to Python `BaseException` / `Exception` |
| `super(string, throwable)` | Call `super().__init__(message)` and assign/chain the cause | Standard Python 3 exception chaining paradigm |

---

## Suspected Java bugs
None identified.

---

## Static init order
None.

---

## Thread safety
No synchronization docstrings present. As an Exception class, instances are typically instantiated and immediately thrown, functioning as mostly immutable state carriers on the local thread's stack. Thread safety issues are not applicable under normal usage.
