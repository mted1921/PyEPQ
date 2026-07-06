# PoissonDeviate Conversion Spec

## Java class
`gov.nist.microanalysis.Utility.PoissonDeviate`

Source: `src/gov/nist/microanalysis/Utility/PoissonDeviate.java`

---

## Inbound dependencies (Java imports)
- `java.util.Random` — internal RNG seeded in constructor; replace with Python `random.Random`
- `gov.nist.microanalysis.Utility.Math2` — `Math2.gammaln` used in the large-mean branch; import from sibling port

No EPQException, Jama, or java.awt imports.

---

## Outbound dependents (callers of public methods)
Not audited. Used wherever Poisson-distributed random counts are needed (e.g. shot-noise simulation).

---

## Public API surface

`PoissonDeviate` is a **concrete class** with no subclass or interface constraints.

### Constructor

| Java signature | Python signature | Notes |
|---|---|---|
| `public PoissonDeviate(long seed)` | `__init__(self, seed: int)` | Seeds an internal `random.Random` instance; sets `_mPrevMean = -1.0` |

### Public instance methods

| Java signature | Python signature | Notes |
|---|---|---|
| `public double randomDeviate(double mean)` | `def randomDeviate(self, mean: float) -> float` | Returns a Poisson-distributed float (whole-number value) |

---

## Private / protected members

| Java | Python |
|---|---|
| `transient private double mPrevMean` | `self._mPrevMean: float = -1.0` |
| `transient private double mG` | `self._mG: float` |
| `transient private double mSqr` | `self._mSqr: float` |
| `transient private double mLogMean` | `self._mLogMean: float` |
| `private final transient Random mRandom` | `self._mRandom: random.Random` |

All fields are declared `transient` in Java (not serialized). In Python, discard `transient` — no equivalent.

---

## Overloaded methods (split plan)
No overloads.

---

## Mutable-output methods
None. `randomDeviate` returns a new value; instance fields `_mG`, `_mSqr`, `_mLogMean` are write-through caches for the previous mean — not caller-supplied.

---

## Touchpoints into Jama / javax.swing / java.awt / java.io
None. `java.util.Random` maps to `random.Random`.

---

## Abstract class strategy
Not applicable. Concrete class.

---

## Java-specific translation decisions

| Java pattern | Python translation | Rule |
|---|---|---|
| `new Random(); mRandom.setSeed(seed)` | `random.Random(seed)` | Equivalent seeded RNG |
| `mRandom.nextDouble()` | `self._mRandom.random()` | |
| `Math.exp(-mean)` | `math.exp(-mean)` | |
| `Math.sqrt(2.0 * mean)` | `math.sqrt(2.0 * mean)` | |
| `Math.log(mean)` | `math.log(mean)` | |
| `Math2.gammaln(mean + 1.0)` | `Math2.gammaln(mean + 1.0)` | Sibling import |
| `Math.tan(Math.PI * mRandom.nextDouble())` | `math.tan(math.pi * self._mRandom.random())` | |
| `Math.floor(em)` | `math.floor(em)` | |
| `assert mean > 0.0` | `assert mean > 0.0` | |
| `assert (em >= 0)` | `assert em >= 0` | |
| Return type is `double` but always a whole number | Return type is `float`; R9 does not apply (the whole-number is intentional) | |

---

## Two algorithm branches

The Java implementation uses two algorithms depending on `mean`:

**Small mean (mean < 12.0):** Direct multiplied-uniform method. Caches `mG = exp(-mean)` when `mean` changes. Produces integer deviates.

**Large mean (mean >= 12.0):** Rejection method using a Lorentzian comparison function (Numerical Recipes algorithm). Caches `mSqr`, `mLogMean`, `mG`. Requires `Math2.gammaln`.

Both branches must match exactly for parity testing (same RNG sequence → same output). Use `JavaRandom` from `_epq_compat` in the parity port to match Java's `Random` sequence.

---

## Suspected Java bugs

**JAVA-BUG-1 — Constructor ignores `seed` parameter.**
Java source:
```java
public PoissonDeviate(long seed) {
    mRandom = new Random();
    mRandom.setSeed(seed);
    ...
}
```
`new Random()` seeds from `System.currentTimeMillis()`, then `setSeed(seed)` overrides it. This is effectively correct (the first seed is discarded), but it is wasteful. No behavioral bug.
Disposition: **Not a bug** — port as `random.Random(seed)`.

---

## Static init order
None.

---

## Thread safety
Not thread-safe. `_mPrevMean`, `_mG`, `_mSqr`, `_mLogMean` are cached instance state; `_mRandom` is a shared mutable RNG. Concurrent calls to `randomDeviate` on the same instance would produce incorrect or interleaved sequences.
