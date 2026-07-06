# Constraint Conversion Spec

> **Spec revision ver1_1_1** (2026-06-25): repaired against the Java source. The prior
> ver1_1_0 documented a non-existent `limit(double)` clamp API and invented constructor
> signatures. The real `Constraint` interface has four methods
> (`realToConstrained` / `constrainedToReal` / `derivative` / `getResult`) and the inner
> classes use `exp`/`atan` mappings, not `min`/`max` clamps.

## Java class
`gov.nist.microanalysis.Utility.Constraint`

Source: `src/gov/nist/microanalysis/Utility/Constraint.java`

---

## Inbound dependencies (Java imports)
No Java `import` statements, but the interface **uses `UncertainValue2`** (same package) in `getResult` and in every inner class's `getResult`. The dependency graph records `Constraint → UncertainValue2`. Import `UncertainValue2` from the sibling port (filename from `UTILITY_LEDGER.md`); its `multiply`, `exp`, `add`, `atan` statics are required by the inner classes.

---

## Outbound dependents (callers of public methods)
`LevenbergMarquardtConstrained` and `LevenbergMarquardtParameterized` map fit parameters between the optimizer's real space `(-∞,∞)` and a constrained sub-range through these invertible, differentiable functions. Not exhaustively audited.

---

## Public API surface

`Constraint` is a **public interface** with four abstract methods and four public nested implementing classes (interface members are implicitly `public static`). It has no fields and no constructor of its own.

### Interface methods

| Java signature | Python signature | Notes |
|---|---|---|
| `double realToConstrained(double param)` | `@abc.abstractmethod def realToConstrained(self, param: float) -> float` | Maps real `param` onto the constrained range |
| `double constrainedToReal(double param)` | `@abc.abstractmethod def constrainedToReal(self, param: float) -> float` | Inverse of `realToConstrained` |
| `double derivative(double param)` | `@abc.abstractmethod def derivative(self, param: float) -> float` | d(`realToConstrained`)/d(param) |
| `UncertainValue2 getResult(UncertainValue2 param)` | `@abc.abstractmethod def getResult(self, param: UncertainValue2) -> UncertainValue2` | `realToConstrained` with uncertainty propagation |

### Inner classes

Each is `public class <X> implements Constraint` (implicitly static). Port each as a nested class inside `Constraint` plus a module-level alias (R2).

#### `Constraint.Positive`

| Java | Python | Notes |
|---|---|---|
| `private final double mScale` | `self._mScale: float` | |
| `public Positive(double scale)` | `__init__(self, scale: float)` | |
| `realToConstrained(p)` = `mScale * Math.exp(p)` | `self._mScale * math.exp(param)` | |
| `constrainedToReal(res)` = `Math.log(res / mScale)` | `math.log(res / self._mScale)` | |
| `derivative(p)` = `mScale * Math.exp(p)` | `self._mScale * math.exp(param)` | |
| `getResult(param)` = `UncertainValue2.multiply(mScale, UncertainValue2.exp(param))` | same via sibling port | |
| `toString()` = `"Positive[scale=" + mScale + "]"` | `__str__` + `toString` alias | R2 |

#### `Constraint.Fractional`

| Java | Python | Notes |
|---|---|---|
| `private final String mName` | `self._mName: str` | |
| `private final double mScale` | `self._mScale: float` | |
| `private final double mFraction` | `self._mFraction: float` | |
| `static private final double TWO_OVER_PI = 2.0 / Math.PI` | `_TWO_OVER_PI: float = 2.0 / math.pi` | class-level, `_`-prefixed |
| `public Fractional(String name, double scale, double fraction)` | `__init__(self, name: str, scale: float, fraction: float)` | |
| `realToConstrained(p)` = `mScale*(1.0 + mFraction*TWO_OVER_PI*Math.atan(p))` | faithful | |
| `constrainedToReal(res)` = `Math.tan((res - mScale)/(mScale*mFraction*TWO_OVER_PI))` | faithful | |
| `derivative(p)` = `(mScale*mFraction*TWO_OVER_PI)/(1.0 + p*p)` | faithful | |
| `getResult(param)` = `UncertainValue2.add(mScale, UncertainValue2.multiply(mScale*mFraction*TWO_OVER_PI, UncertainValue2.atan(param)))` | via sibling port | |
| `toString()` = `"Fraction[" + mName + "," + mScale + " +- " + (mScale*mFraction) + "]"` | `__str__` + alias | R2 |

#### `Constraint.Bounded`

| Java | Python | Notes |
|---|---|---|
| `private final double mCenter` | `self._mCenter: float` | |
| `private final double mWidth` | `self._mWidth: float` | **Constructor stores `0.5 * width`**, not `width` |
| `static private final double TWO_OVER_PI = 2.0 / Math.PI` | `_TWO_OVER_PI: float = 2.0 / math.pi` | |
| `public Bounded(double center, double width)` | `__init__(self, center: float, width: float)` → `self._mWidth = 0.5 * width` | |
| `realToConstrained(p)` = `mCenter + mWidth*TWO_OVER_PI*Math.atan(p)` | faithful | |
| `constrainedToReal(res)` = `Math.tan((res - mCenter)/(mWidth*TWO_OVER_PI))` | faithful | |
| `derivative(p)` = `(mWidth*TWO_OVER_PI)/(1.0 + p*p)` | faithful | |
| `getResult(param)` = `UncertainValue2.add(mCenter, UncertainValue2.multiply(mWidth*TWO_OVER_PI, UncertainValue2.atan(param)))` | via sibling port | |
| `toString()` = `"Bounded[min=" + (mCenter-mWidth) + ",max=" + (mCenter+mWidth) + "]"` | `__str__` + alias | R2 |

#### `Constraint.None` → port as `Constraint.Unconstrained`

| Java | Python | Notes |
|---|---|---|
| `public None()` | `__init__(self)` | **`None` is a Python builtin** — rename the class to `Unconstrained` (R1/R10) |
| `realToConstrained(p)` = `p` | `return param` | identity |
| `constrainedToReal(res)` = `res` | `return param` | identity |
| `derivative(p)` = `1.0` | `return 1.0` | |
| `getResult(param)` = `param` | `return param` | identity |
| `toString()` = `"Unconstrained[]"` | `__str__` + alias | R2 — Java already returns `"Unconstrained[]"` |

---

## Module-level aliases (R2)

```python
Positive      = Constraint.Positive
Fractional    = Constraint.Fractional
Bounded       = Constraint.Bounded
Unconstrained = Constraint.Unconstrained   # Java Constraint.None
```

Callers that used `Constraint.None` must use `Constraint.Unconstrained`. No `None` alias (builtin clash).

---

## Private / protected members
Covered per inner class above (`_mScale`, `_mName`, `_mFraction`, `_mCenter`, `_mWidth`, `_TWO_OVER_PI`). The interface itself has no state.

---

## Overloaded methods (split plan)
No overloads. Each inner class implements exactly one form of each of the four interface methods.

---

## Mutable-output methods
None. All methods are pure.

---

## Touchpoints into Jama / javax.swing / java.awt / java.io
None.

---

## Abstract class strategy
IS_ABSTRACT = True (it is an interface). Port `Constraint` as `class Constraint(abc.ABC)` with the four methods `@abc.abstractmethod`. Each inner class subclasses `Constraint` and implements all four.

**M4 does not block parity.** `Constraint` is a Java *interface*, but the four concrete inner classes are instantiable; JPype can wrap them. Parity tests construct `Positive(scale)`, `Fractional(name, scale, fraction)`, `Bounded(center, width)`, `None()` on the Java side and compare `realToConstrained` / `constrainedToReal` / `derivative` outputs (and `getResult` value+uncertainty) against the Python port.

---

## Java-specific translation decisions

| Java pattern | Python translation | Rule |
|---|---|---|
| `public interface Constraint` | `class Constraint(abc.ABC)` | R2 |
| `public class None implements Constraint` | `class Unconstrained(Constraint)` | R1/R10 — builtin clash |
| `Math.exp` / `Math.log` / `Math.atan` / `Math.tan` | `math.exp` / `math.log` / `math.atan` / `math.tan` | |
| `UncertainValue2.multiply/exp/add/atan` | sibling-port statics | R3 |
| `2.0 / Math.PI` constant | `2.0 / math.pi` | |
| `Double.toString(x)` in `toString` | match Java's formatting in the parity assertion, or compare numerically | exact-string note |

---

## Suspected Java bugs
None. The four mappings are mathematically consistent (`constrainedToReal` inverts `realToConstrained`; `derivative` matches d/dp of `realToConstrained`). `Positive.constrainedToReal` will raise on non-positive input (`log` of ≤0) — that is the intended domain restriction, not a bug.

`BUG_LEDGER`: `()`.

---

## Static init order
`_TWO_OVER_PI` constants are compile-time — initialize in the respective class bodies. No cross-class ordering concern. `getResult` calls into `UncertainValue2` statics at call time (not import time), so no static-init cycle with `UncertainValue2`.

---

## Thread safety
All implementations are immutable after construction (`Positive`, `Fractional`, `Bounded`) or stateless (`Unconstrained`). Safe for concurrent read-only use.
