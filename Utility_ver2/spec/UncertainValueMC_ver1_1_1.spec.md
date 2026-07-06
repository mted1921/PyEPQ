# UncertainValueMC Conversion Spec

> **Spec revision ver1_1_1** (2026-06-25): repaired against the Java source. The prior
> ver1_1_0 invented constructors (`(double,String,double)`, `(UncertainValue2)`),
> methods (`getRandVal`/`setRandVal`, `clone`, `sample`, `add_list`, …) and a `_SOURCE`
> field that **do not exist**, and mis-stated the bugs. The real class extends `Number`,
> stores two `final` doubles, and exposes the static algebra below.

## Java class
`gov.nist.microanalysis.Utility.UncertainValueMC`

Source: `src/gov/nist/microanalysis/Utility/UncertainValueMC.java`

---

## Inbound dependencies (Java imports)
- `java.util.Map` — the deviates dictionary in the `(UncertainValue2, Map)` constructor; use `dict`.
- `java.util.Random` — `sRandom.nextGaussian()` for normal deviates; **use `JavaRandom` from `_epq_compat`** for sequence parity (see RNG note).
- `gov.nist.microanalysis.Utility.UncertainValue2` — `getComponentNames`, `getComponent`, `doubleValue` in `calculateDeviate`; import from the sibling port. (Same-package use; no Java `import` line, but the dependency is real — graph: `UncertainValueMC → UncertainValue2`.)

`UncertainValueMC extends Number`.

---

## Outbound dependents (callers of public methods)
`MCUncertaintyEngine` builds `UncertainValueMC` instances from arguments sharing a deviates map and combines them through the static algebra. Not exhaustively audited.

---

## Public API surface

`UncertainValueMC extends Number`. Each instance carries a **nominal** value (`mValue`) and a single **random sample** (`mRandVal`); both are `final` — there are no setters. The static methods build new instances; the random component propagates linearly/non-linearly per operation.

### Static constants

| Java declaration | Python name | Notes |
|---|---|---|
| `private static final long serialVersionUID = 8528835028857799252L` | `serialVersionUID: int = 8528835028857799252` | **R1 exception** — public, no `_` |
| `private static final Random sRandom = new Random(System.currentTimeMillis())` | `_sRandom: JavaRandom` | seeded from wall-clock → non-deterministic; see RNG note |

### Constructors

| Java signature | Python signature | Notes |
|---|---|---|
| `private UncertainValueMC(double v, double randVal, Object obj)` | `@classmethod def _internal(cls, v: float, randVal: float) -> UncertainValueMC` | internal all-fields builder; Java's `obj` is always `null` (drop it and its `assert`) |
| `public UncertainValueMC(double v, double dv)` | `__init__(self, v: float, dv: float)` | `this(v, dv * normalDeviate(), null)` → nominal `_mValue = v`, sample `_mRandVal = dv * normalDeviate()` (centred at 0, **not** at `v` — contrast the `(UncertainValue2, Map)` ctor which centres at the value; preserve both verbatim) |
| `public UncertainValueMC(UncertainValue2 uv, Map<String,Double> deviates)` | `__init__(self, uv: UncertainValue2, deviates: dict)` | `mValue = uv.doubleValue()`; `mRandVal = uv.doubleValue() + calculateDeviate(uv, deviates)` |

`__init__` dispatches by type: `(float, float)` → value/uncertainty path; `(UncertainValue2, dict)` → component-deviate path. Both delegate to `_internal`.

### Static algebra

| Java signature | Python signature | Notes |
|---|---|---|
| `static UncertainValueMC add(double a, UncertainValueMC uva, double b, UncertainValueMC uvb)` | `add_combo(a, uva, b, uvb)` | R4 — `a·uva + b·uvb`, both value and randVal |
| `static UncertainValueMC sum(UncertainValueMC[] vals)` | `sum(vals)` | element-wise sum of value and randVal |
| `static UncertainValueMC mean(UncertainValueMC[] vals)` | `mean(vals)` | `multiply(1/len, sum(vals))` |
| `static UncertainValueMC add(UncertainValueMC v1, UncertainValueMC v2)` | `add_vv(v1, v2)` | R4 |
| `static UncertainValueMC subtract(UncertainValueMC v1, UncertainValueMC v2)` | `subtract(v1, v2)` | |
| `static UncertainValueMC multiply(UncertainValueMC v1, UncertainValueMC v2)` | `multiply_vv(v1, v2)` | R4 |
| `static UncertainValueMC multiply(double v1, UncertainValueMC v2)` | `multiply_sv(v1, v2)` | R4 |
| `static UncertainValueMC divide(UncertainValueMC v1, UncertainValueMC v2)` | `divide_vv(v1, v2)` | R4 |
| `static UncertainValueMC divide(UncertainValueMC v1, double v2)` | `divide_vs(v1, v2)` | R4 |
| `static UncertainValueMC pow(UncertainValueMC n, UncertainValueMC exp)` | `pow(n, exp)` | **JAVA-BUG-2** (see bugs) |
| `static UncertainValueMC log(UncertainValueMC n)` | `log(n)` | `log(value)`, `log(randVal)` |
| `static UncertainValueMC exp(UncertainValueMC n)` | `exp(n)` | **JAVA-BUG-1** — computes `log`, not `exp` |
| `static UncertainValueMC sqrt(UncertainValueMC n)` | `sqrt(n)` | `sqrt(value)`, `sqrt(max(0, randVal))` |

### Instance methods

| Java signature | Python signature | Notes |
|---|---|---|
| `public UncertainValueMC abs()` | `def abs(self) -> UncertainValueMC` | `abs(value)`, `abs(randVal)` |
| `public UncertainValueMC nonNegative()` | `def nonNegative(self) -> UncertainValueMC` | identical body to `abs()` (see note) |
| `public double nominalValue()` | `def nominalValue(self) -> float` | returns `_mValue` |
| `@Override public double doubleValue()` | `def doubleValue(self) -> float` + `def __float__(self) -> float` | **returns `_mRandVal`** (the random sample), NOT the nominal |
| `@Override public float floatValue()` | `def floatValue(self) -> float` | `float(_mRandVal)` |
| `@Override public int intValue()` | `def intValue(self) -> int` + `def __int__(self) -> int` | `int(_mRandVal)` (truncation) |
| `@Override public long longValue()` | `def longValue(self) -> int` | `int(_mRandVal)` |
| `@Override public String toString()` | `def __str__(self) -> str` + `def toString(self) -> str` | `str(_mRandVal)` (R2) |

### Private static helpers

| Java | Python |
|---|---|
| `static private double calculateDeviate(UncertainValue2 uv2, Map<String,Double> deviate)` | `@staticmethod def _calculateDeviate(uv2, deviate) -> float` — for each component name: if absent, store `normalDeviate()`; accumulate `deviate[comp] * uv2.getComponent(comp)` |
| `static private double normalDeviate()` | `@staticmethod def _normalDeviate() -> float` — `_sRandom.nextGaussian()` |

---

## Private / protected members

| Java | Python |
|---|---|
| `private final double mValue` | `self._mValue: float` — nominal value |
| `private final double mRandVal` | `self._mRandVal: float` — single random sample |

---

## Overloaded methods (split plan)
- `add`: `(double,UVMC,double,UVMC)` and `(UVMC,UVMC)` → `add_combo` / `add_vv` (R4).
- `multiply`: `(UVMC,UVMC)` / `(double,UVMC)` → `multiply_vv` / `multiply_sv` (R4).
- `divide`: `(UVMC,UVMC)` / `(UVMC,double)` → `divide_vv` / `divide_vs` (R4).
- Constructor: `(double,double)` / `(UncertainValue2,Map)` → `__init__` type dispatch; the private 3-arg form → `_internal` classmethod.

---

## Mutable-output methods
`calculateDeviate` **mutates** the caller-supplied `deviate` map (inserts new deviates) — this is intentional and shared across arguments in an `MCUncertaintyEngine` iteration to correlate samples. No R5 guard (it is a `dict`, by design). No method mutates a numeric buffer.

---

## Touchpoints into Jama / javax.swing / java.awt / java.io
None.

---

## Abstract class strategy
Not applicable — `UncertainValueMC` is concrete. IS_ABSTRACT = False.

---

## RNG note (parity)
`sRandom` is seeded from `System.currentTimeMillis()`, so `mRandVal` is **non-deterministic** across runs. For parity:
- Use `JavaRandom` from `_epq_compat` so that, *when seeded identically*, `nextGaussian()` reproduces Java's sequence (Mersenne-vs-Java divergence otherwise — Cross-Cutting RNG rule).
- Parity tests must either inject a fixed-seed `JavaRandom` (replace `_sRandom`) and compare sample-for-sample against a Java instance seeded the same way, or compare **statistical properties** (mean, variance) over many draws rather than exact values. Record the wall-clock seeding as `RNG-DEVIATION-1` if the port changes seeding strategy.

---

## Java-specific translation decisions

| Java pattern | Python translation | Rule |
|---|---|---|
| `extends Number` | provide `__float__`/`__int__`; optionally inherit `numbers.Number` | numeric ABC |
| `sRandom.nextGaussian()` | `_sRandom.nextGaussian()` (`JavaRandom`) | Cross-Cutting RNG |
| `Math.pow/log/sqrt/abs/max` | `math.pow`/`math.log`/`math.sqrt`/`abs`/`max` | |
| private 3-arg ctor with `assert obj == null` | `_internal` classmethod; **omit** the assert | Cross-Cutting — Java `assert` |
| `serialVersionUID` (private) | public attribute, no `_` | R1 exception |
| `Double.valueOf(mRandVal).floatValue()` etc. | `float(self._mRandVal)`, `int(self._mRandVal)` | |

---

## Suspected Java bugs

**JAVA-BUG-1 — `exp(n)` computes `log`, not `exp`.**
Java source line: `return new UncertainValueMC(Math.log(n.mValue), Math.log(n.mRandVal), null);` (inside `exp`).
The method named `exp` applies `Math.log` to both the nominal value and the random sample — a copy-paste of the adjacent `log` method.
Disposition: **Preserve** behind `# JAVA-BUG-1` quoting the line; `BUG_LEDGER` entry with `has_strict_variant=True`; `exp_strict` uses `math.exp` for both components.

**JAVA-BUG-2 — `pow(n, exp)` uses the exponent's *random sample* in the nominal value.**
Java source line: `return new UncertainValueMC(Math.pow(n.mValue, exp.mRandVal), Math.pow(n.mRandVal, exp.mRandVal), null);`
The nominal value is `pow(n.mValue, exp.mRandVal)` — it should use the exponent's nominal (`exp.nominalValue()` / `exp.mValue`), not its random deviate. The random component `pow(n.mRandVal, exp.mRandVal)` is internally consistent.
Disposition: **Preserve** behind `# JAVA-BUG-2`; `BUG_LEDGER` entry with `has_strict_variant=True`; `pow_strict` uses `math.pow(n._mValue, exp._mValue)` for the nominal.

**Note (not a bug): `nonNegative()` is identical to `abs()`.** Both return `(abs(value), abs(randVal))`. Port both verbatim; do not collapse or "correct" `nonNegative`.

---

## Static init order
`serialVersionUID` is a compile-time constant. `_sRandom` is a module/class-level singleton — initialise once at class-body load (mirrors Java's `static final`); note the parity-seeding override hook. No cross-class static cycle (`calculateDeviate` touches `UncertainValue2` at call time only).

---

## Thread safety
`_sRandom` is a shared singleton; `nextGaussian()` is not synchronized here → concurrent construction races the RNG (matches Java). Instances are immutable after construction (`mValue`, `mRandVal` final). The shared deviates map in `calculateDeviate` is mutated without synchronization.
