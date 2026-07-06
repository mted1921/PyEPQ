# AdaptiveRungeKutta Conversion Spec

## Java class
`gov.nist.microanalysis.Utility.AdaptiveRungeKutta`

Source: `src/gov/nist/microanalysis/Utility/AdaptiveRungeKutta.java`

---

## Inbound dependencies (Java imports)
- `gov.nist.microanalysis.Utility.UtilException` — thrown by `integrate` and `qcStep`; import from sibling port

No EPQException, Jama, or java.awt imports.

---

## Outbound dependents (callers of public methods)
- `Integrator` extends `AdaptiveRungeKutta` (depth 2)
- `Math2.chiSquaredConfidenceLevel` calls `integrate` via an anonymous subclass of `FindRoot`; not `AdaptiveRungeKutta` directly

---

## Public API surface

`AdaptiveRungeKutta` is a **public abstract class**. Its single abstract method is `derivatives`.

### Constructor

| Java signature | Python signature | Notes |
|---|---|---|
| `public AdaptiveRungeKutta(int nVars)` | `__init__(self, nVars: int)` | Sets number of ODE variables |

### Abstract method

| Java signature | Python signature | Notes |
|---|---|---|
| `abstract public void derivatives(double x, double[] y, double[] dydx)` | `@abc.abstractmethod def derivatives(self, x: float, y: list[float], dydx: list[float]) -> None` | Subclass fills `dydx[i]` in-place; `y` and `dydx` have length `nVars` |

### Public instance methods (configuration)

| Java signature | Python signature | Notes |
|---|---|---|
| `public void setSaveInterval(double interval)` | `def setSaveInterval(self, interval: float) -> None` | Enables trajectory saving at this spacing; `abs(interval)` stored |
| `public void clearSaveInterval()` | `def clearSaveInterval(self) -> None` | Disables trajectory saving (restores `float('inf')` default) |
| `public void setMaxSteps(int maxSteps)` | `def setMaxSteps(self, maxSteps: int) -> None` | Default 10 000 |
| `public void setMinStepSize(double minStep)` | `def setMinStepSize(self, minStep: float) -> None` | Default 0.0; `abs(minStep)` stored |

### Public instance methods (results)

| Java signature | Python signature | Notes |
|---|---|---|
| `public int getNVariables()` | `def getNVariables(self) -> int` | Returns `nVars` from constructor |
| `public int getNSaved()` | `def getNSaved(self) -> int` | Number of trajectory points saved after `integrate` |
| `public double getX(int i)` | `def getX(self, i: int) -> float` | x-coordinate of i-th saved point |
| `public double[] getY(int i)` | `def getY(self, i: int) -> list[float]` | y-coordinates of i-th saved point |
| `public int getStepCount()` | `def getStepCount(self) -> int` | `mNOk + mNBad` after last `integrate` |
| `public int getGoodStepCount()` | `def getGoodStepCount(self) -> int` | Steps accepted without retry |
| `public int getBadStepCount()` | `def getBadStepCount(self) -> int` | Steps requiring subdivision |

### Public instance methods (integration)

| Java signature | Python signature | Notes |
|---|---|---|
| `public double[] integrate(double x1, double x2, double[] ystart, double eps, double h1) throws UtilException` | `def integrate(self, x1: float, x2: float, ystart: list[float], eps: float, h1: float) -> list[float]` | Raises `UtilException`; mutates `ystart` on exit (R5 note below) |

---

## Private / protected members

| Java | Python |
|---|---|
| `private final int mNVariables` | `self._mNVariables: int` |
| `private double mHDid` | `self._mHDid: float` |
| `private double mHNext` | `self._mHNext: float` |
| `private double mSaveInterval = Double.MAX_VALUE` | `self._mSaveInterval: float = float('inf')` |
| `private double mMinStepSize = 0.0` | `self._mMinStepSize: float = 0.0` |
| `private double[] mXSave` | `self._mXSave: Optional[list[float]] = None` |
| `private double[][] mYSave` | `self._mYSave: Optional[list[list[float]]] = None` |
| `private int mNSaved = 0` | `self._mNSaved: int = 0` |
| `private int mMaxSteps = 10000` | `self._mMaxSteps: int = 10000` |
| `private int mNOk` | `self._mNOk: int` |
| `private int mNBad` | `self._mNBad: int` |
| `private double[] mWs2..mWs6, mYTemp` | `self._mWs2..._mWs6, self._mYTemp: Optional[list[float]]` — lazy-allocated workspace |
| `private double[] mYErr, mQcYTemp` | `self._mYErr, self._mQcYTemp: Optional[list[float]]` — lazy-allocated workspace |
| `private void baseStep(...)` | `def _baseStep(self, ...) -> None` | R1 — private → underscore |
| `private double qcStep(...)` | `def _qcStep(self, ...) -> float` | R1 — raises `UtilException` |
| `private void clearWorkspace()` | `def _clearWorkspace(self) -> None` | R1 — sets workspace refs to `None` |
| `private double sign(double magnitude, double sign)` | `def _sign(self, magnitude: float, sign: float) -> float` | R1 — `abs(magnitude) if sign >= 0 else -abs(magnitude)` |

---

## Overloaded methods (split plan)
No overloads.

---

## Mutable-output methods

**`integrate` mutates `ystart` in-place.** Java source:
```java
System.arraycopy(y, 0, ystart, 0, mNVariables);
```
On successful completion, the final y values are copied back into the caller-supplied `ystart` array. This is documented behavior (marked `In & out` in the Javadoc). The Python port must replicate this mutation: `ystart[:] = y`.

**R5 note**: Do NOT make a defensive copy of `ystart` on entry — the mutation is the contract.

`_qcStep` mutates `y` in-place via `System.arraycopy(mQcYTemp, 0, y, 0, mNVariables)`. Port: `y[:] = self._mQcYTemp`.

`derivatives` (abstract) — callers (e.g. `_baseStep`) pass `dydx` for in-place mutation. Contract-required; no R5 guard.

---

## Touchpoints into Jama / javax.swing / java.awt / java.io
None.

---

## Abstract class strategy
IS_ABSTRACT = True.

`derivatives(double x, double[] y, double[] dydx)` is the sole abstract method. Subclasses fill `dydx[i]` to define the ODE.

**M4 applies**: JPype cannot extend Java abstract classes from Python. Parity harness strategy: use a concrete Java subclass of `AdaptiveRungeKutta` with a known ODE (e.g. `dydx[0] = -sin(x)`, `dydx[1] = cos(x)`, exact solution `cos(x)` and `sin(x)`) and compare `integrate(0, 2π, [1,0], 1e-6, 0.01)` against the analytic result.

---

## Java-specific translation decisions

| Java pattern | Python translation | Rule |
|---|---|---|
| `abstract public class AdaptiveRungeKutta` | `class AdaptiveRungeKutta(abc.ABC)` | R2 |
| `System.arraycopy(src, 0, dst, 0, n)` | `dst[:n] = src[:n]` | |
| `Double.MAX_VALUE` | `float('inf')` for `mSaveInterval`; `sys.float_info.max` if exact value needed | |
| `Math.abs(magnitude)` | `abs(magnitude)` | |
| `Math.max(a, b)` | `max(a, b)` | |
| `Math.min(a, b)` | `min(a, b)` | |
| `Math.pow(x, p)` | `x ** p` | |
| `Math.round(x)` | `round(x)` — Python `round()` uses HALF_EVEN, Java `Math.round` uses HALF_UP (floor(x+0.5)). In the `integrate` method: `kMax = int(round(...))` — discrepancy possible at 0.5 boundaries | R10 — note in BUG_LEDGER |
| Lazy workspace allocation (null check → allocate) | `if self._mWs2 is None: self._mWs2 = [0.0] * self._mNVariables` | |
| `mWs2 = null` in `clearWorkspace` | `self._mWs2 = None` | |

---

## Cash-Karp coefficients

The `_baseStep` method implements the Cash-Karp embedded Runge-Kutta scheme. All 15 constants (`a2..a6`, `b21..b65`, `c1..c6`, `dc1..dc6`) are defined as local `final double` in Java. In Python, define them as local variables inside `_baseStep` or as class-level constants — class-level constants preferred for readability:

```python
_A2, _A3, _A4, _A5, _A6 = 0.2, 0.3, 0.6, 1.0, 0.875
_B21 = 0.2
# etc.
```

---

## Suspected Java bugs

**JAVA-NOTE-1 — `Math.round` vs Python `round` in `integrate`.**
Java source: `kMax = (int) Math.round((Math.abs(x2 - x1) + mSaveInterval) / mSaveInterval);`
Java `Math.round(x)` is `floor(x + 0.5)`; Python `round(x)` uses HALF_EVEN. Diverges at 0.5 boundaries.
Disposition: **Preserve** — port as `int(math.floor(val + 0.5))` to match Java semantics; document as `# JAVA-NOTE-1`.

**JAVA-NOTE-2 — `xnew == x` floating-point equality check for step size underflow.**
Java source: `final double xnew = x + h; if (xnew == x) throw ...`
This is the standard NR recipe idiom for detecting when `h` is so small that `x + h == x` in IEEE 754. Port verbatim — the equality check is intentional.

---

## Static init order
None. All state is instance-level.

---

## Thread safety
Not thread-safe (documented explicitly in the Javadoc). `_mHDid`, `_mHNext`, `_mNOk`, `_mNBad`, `_mNSaved`, and all workspace arrays are mutable during `integrate`. Use one instance per thread.
