# StageRelocation Conversion Spec

## Java class
`gov.nist.microanalysis.Utility.StageRelocation`

Source: `src/gov/nist/microanalysis/Utility/StageRelocation.java`

---

## Inbound dependencies (Java imports)
- `gov.nist.microanalysis.EPQLibrary.EPQException` — thrown by `StageRelocation(List, boolean)` constructor and raised inside `optimize`; import from `_epq_compat`
- `gov.nist.microanalysis.Utility.Math2` — `Math2.sqr` used in `distance` helper and `optimize`; import from sibling port
- `gov.nist.microanalysis.Utility.Simplex` — used in anonymous inner class `OptimizeFit` inside `optimize` (3+ point case); import from sibling port
- `java.util.List` — parameter type for `optimize` and constructor

No Jama or java.awt imports.

---

## Outbound dependents (callers of public methods)
Not audited. Used for SEM stage coordinate translation.

---

## Public API surface

### Inner class: `StageRelocation.RelocatedPoint`

| Java signature | Python signature | Notes |
|---|---|---|
| `public RelocatedPoint(double[] nativePt, double[] relocatedPt)` | `__init__(self, nativePt: list[float], relocatedPt: list[float])` | Makes defensive copies (`.clone()` in Java) |
| `public double[] getNativePoint()` | `def getNativePoint(self) -> list[float]` | Returns copy |
| `public double[] getRelocatedPoint()` | `def getRelocatedPoint(self) -> list[float]` | Returns copy |

Module-level alias: `RelocatedPoint = StageRelocation.RelocatedPoint`. (R2)

### Constructors

| Java signature | Python signature | Notes |
|---|---|---|
| `public StageRelocation()` | `__init__(self)` | Null-transform: origin=(0,0), scale=(1,1), rotation=0, error=0 |
| `public StageRelocation(List<RelocatedPoint> pts, boolean mirror) throws EPQException` | `__init__(self, pts: list[RelocatedPoint], mirror: bool)` | Dispatched by arg type; raises `EPQException` |

### Public instance methods

| Java signature | Python signature | Notes |
|---|---|---|
| `public double[] apply(double[] pt)` | `def apply(self, pt: list[float]) -> list[float]` | Forward transform; returns new 2-element list |
| `public double[] inverse(double[] pt)` | `def inverse(self, pt: list[float]) -> list[float]` | Inverse transform; returns new 2-element list |
| `public double getResidualError()` | `def getResidualError(self) -> float` | RMS residual per fit point after `optimize` |
| `public boolean isMirrored()` | `def isMirrored(self) -> bool` | |

---

## Private / protected members

| Java | Python |
|---|---|
| `private double mX0rigin, mYOrigin` | `self._mX0rigin: float`, `self._mYOrigin: float` — **note typo in Java**: `mX0rigin` uses zero not O |
| `private double mXScale, mYScale` | `self._mXScale: float`, `self._mYScale: float` |
| `private double mRotation` | `self._mRotation: float` |
| `private double mError` | `self._mError: float` |
| `private boolean mMirrored` | `self._mMirrored: bool` |
| `private void optimize(List<RelocatedPoint>, boolean) throws EPQException` | `def _optimize(self, pts: list[RelocatedPoint], mirror: bool) -> None` | R1 — private → underscore |
| `private double distance(double, double, double, double)` | `def _distance(self, x0: float, y0: float, x1: float, y1: float) -> float` | R1 |

---

## Overloaded methods (split plan)
Both constructors collapse into one `__init__` dispatched by argument type.

No other method overloads.

---

## Mutable-output methods

`_optimize` mutates `self._mX0rigin`, `self._mYOrigin`, `self._mXScale`, `self._mYScale`, `self._mRotation`, `self._mError`, `self._mMirrored`. Called from constructor only.

`apply` and `inverse` return new arrays — no caller-supplied mutation.

`RelocatedPoint.__init__` makes defensive copies of `nativePt` and `relocatedPt` (R5).

---

## Touchpoints into Jama / javax.swing / java.awt / java.io
`EPQException` — raised in `_optimize` when two-point fit yields inconsistent distances, and when Simplex optimization fails. Import from `_epq_compat`.

---

## Abstract class strategy
Not applicable. Concrete class.

### Anonymous Simplex subclass inside `_optimize`

When `len(pts) > 2`, Java uses an anonymous inner class `OptimizeFit extends Simplex` to perform Nelder-Mead minimization. In Python, create a named local class inside `_optimize`:

```python
def _optimize(self, pts, mirror):
    ...
    if len(pts) > 2:
        class OptimizeFit(Simplex):
            def function(self, v):
                ...
        of = OptimizeFit()
        ...
```

**Note**: Java code calls `of.setMaxEvaluations(100)` and `of.getBestResult()` on the Simplex instance. These methods were **not captured** in the Simplex spec written from the prior source read. The Simplex spec must be cross-checked against the full `Simplex.java` source to confirm whether `setMaxEvaluations` and `getBestResult` exist. Flag as `# REVIEW: Simplex.setMaxEvaluations / getBestResult`.

Java calls `of.perform(p)` where `p` is a `double[][]` (a pre-specified simplex matrix). The Simplex spec shows `perform(int maxIterations)`. This is another discrepancy — `Simplex` may have a second `perform(double[][])` overload accepting a starting simplex.

---

## Suspected Java bugs

**JAVA-NOTE-1 — `mX0rigin` is a typo for `mXOrigin`.**
Java field: `private double mX0rigin` — uses zero digit `0` instead of uppercase O. Preserved verbatim in the Java source. In Python, use `self._mX0rigin` to exactly match (preserving the typo) for parity testing, or rename to `self._mXOrigin` as an R10 deviation.
Disposition: **Rename** — `self._mXOrigin` in the port; document as R10 deviation.

**JAVA-BUG-1 — `_optimize` raises `EPQException` with misleading message when the fit check triggers on either of two inconsistent conditions combined with OR.**
Java source:
```java
if ((distance(x1, y1, x1p, y1p) > (0.001 * Math.abs(x2p - x1p)))
    || (distance(x2, y2, x2p, y2p) > (0.001 * Math.abs(x2p - x1p))))
    throw new EPQException("Erroneous fit...");
```
The threshold `0.001 * Math.abs(x2p - x1p)` uses the x-difference of the second pair for checking residuals of both points. For nearly-horizontal translations (where `x2p ≈ x1p`), the threshold approaches zero, causing valid fits to be rejected.
Disposition: **Preserve** — port verbatim; document as `# JAVA-BUG-1`.

---

## Static init order
None.

---

## Thread safety
Not thread-safe. `_mX0rigin`, `_mYOrigin`, `_mXScale`, `_mYScale`, `_mRotation`, `_mError`, `_mMirrored` are all mutable instance state. Concurrent construction or `apply` calls on the same instance would be unsafe during construction; read-only after construction is safe.
