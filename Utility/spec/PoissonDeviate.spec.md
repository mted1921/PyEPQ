# PoissonDeviate Conversion Spec

## Java class
`gov.nist.microanalysis.Utility.PoissonDeviate`

Source: `src/gov/nist/microanalysis/Utility/PoissonDeviate.java`

---

## Inbound dependencies (Java imports)
- `java.util.Random` — seeded RNG (`mRandom`)
- `gov.nist.microanalysis.Utility.Math2` — `Math2.gammaln` (large-mean branch only)

---

## Outbound dependents (callers of public methods)
- `EPQLibrary/MonteCarloSS.java` — samples Poisson counts for x-ray generation
- `EPQLibrary/PoissonDistribution.java` — wraps PoissonDeviate
- Other Monte Carlo simulation classes

---

## Public API surface

| Java signature | Python signature | Notes |
|---|---|---|
| `PoissonDeviate(long seed)` | `__init__(self, seed: int)` | Seeds `java.util.Random`; Python uses `JavaRandom` |
| `double randomDeviate(double mean)` | `randomDeviate(self, mean: float) -> float` | Returns Poisson sample as `float` (Java `double`, always integer-valued) |

---

## Private / protected members

| Java | Python |
|---|---|
| `transient double mPrevMean` | `self.mPrevMean: float = -1.0` |
| `transient double mG` | `self.mG: float` |
| `transient double mSqr` | `self.mSqr: float` |
| `transient double mLogMean` | `self.mLogMean: float` |
| `final transient Random mRandom` | `self.mRandom: JavaRandom` |

---

## Overloaded methods (split plan)
None. Single constructor, single method.

---

## Mutable-output methods
None.

---

## Touchpoints into Jama / javax.swing / java.awt / java.io
None.

---

## Abstract class strategy
Not abstract. Concrete class; full `@needs_java` parity testing is possible. Java parity requires bit-identical `java.util.Random` sequences — provided by `JavaRandom` in `_epq_compat`.

---

## Java-specific translation decisions

| Java pattern | Python translation | Rule |
|---|---|---|
| `new Random(); mRandom.setSeed(seed)` | `JavaRandom(seed)` — `_epq_compat` provides a bit-exact Java LCG RNG | R10 |
| `Math.exp(-mean)` cached in `mG` | Identical; `math.exp(-mean)` | R2 |
| `Math.tan(Math.PI * mRandom.nextDouble())` | `math.tan(math.pi * self.mRandom.nextDouble())` | R2 |
| `Math.floor(em)` | `math.floor(em)` | R2 |
| `Math2.gammaln(...)` | `Math2.gammaln(...)` from port | R2 |
| `assert mean > 0.0` (Java assert, disabled by default) | Not reproduced — assert is compile-time disabled in prod Java; port omits it per R2 | R2 |

---

## Algorithm branches

**Small mean (mean < 12) — Knuth algorithm:**
```
G = exp(-mean)          [cached when mean changes]
em = -1; t = 1.0
do { em++; t *= U() } while t > G
return em
```

**Large mean (mean ≥ 12) — Rejection sampling (NR §7.3):**
```
sqr = sqrt(2*mean); logMean = log(mean)
G = mean*logMean - gammaln(mean+1)  [cached when mean changes]
repeat:
    repeat: y = tan(π*U()); em = sqr*y + mean; until em >= 0
    em = floor(em)
    t = 0.9*(1+y²)*exp(em*logMean - gammaln(em+1) - G)
until U() <= t
return em
```

The cache (`mPrevMean`) avoids recomputing `G`, `sqr`, `logMean` when called repeatedly with the same mean. The port reproduces this caching faithfully.

---

## Suspected Java bugs
None. The `transient` fields mean the cached state is not serialized — if a deserialized instance is used, `mPrevMean = -1.0` forces a recompute on the first call. This is correct behavior, not a bug.

---

## Static init order
None.

---

## Thread safety
Not thread-safe. `mRandom` and cached state (`mPrevMean`, `mG`, etc.) are mutable instance fields with no synchronization.
