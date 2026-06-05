# AdaptiveRungeKutta Conversion Spec

## Java class
`gov.nist.microanalysis.Utility.AdaptiveRungeKutta`

Source: `src/gov/nist/microanalysis/Utility/AdaptiveRungeKutta.java`

---

## Inbound dependencies (Java imports)
- `gov.nist.microanalysis.Utility.UtilException` — thrown by `qcStep` and `integrate`

No other EPQ-internal or Jama imports.

---

## Outbound dependents (callers of public methods)

`grep -r "AdaptiveRungeKutta" src/` reveals callers in:
- `EPQTests/AdaptiveRungeKuttaTest.java` — test only; creates anonymous subclass
- Various simulation / transport classes throughout EPQLibrary that subclass
  `AdaptiveRungeKutta` and call `integrate` (deferred; Phase 3).

The public API surface is stable: all callers use `integrate`, `setSaveInterval`,
`getNSaved`, `getX`, `getY`. The abstract `derivatives` is the extension point.

---

## Public API surface

| Java signature | Python signature | Notes |
|---|---|---|
| `AdaptiveRungeKutta(int nVars)` | `__init__(self, nVars: int)` | Base constructor |
| `abstract void derivatives(double x, double[] y, double[] dydx)` | `@abstractmethod derivatives(self, x, y, dydx)` | Extension point |
| `void setSaveInterval(double interval)` | `setSaveInterval(self, interval: float)` | |
| `void clearSaveInterval()` | `clearSaveInterval(self)` | |
| `int getNSaved()` | `getNSaved(self) -> int` | |
| `double getX(int i)` | `getX(self, i: int) -> float` | |
| `double[] getY(int i)` | `getY(self, i: int) -> F64Array` | Returns row-view of mYSave |
| `void setMaxSteps(int maxSteps)` | `setMaxSteps(self, maxSteps: int)` | |
| `void setMinStepSize(double minStep)` | `setMinStepSize(self, minStep: float)` | |
| `int getNVariables()` | `getNVariables(self) -> int` | |
| `int getStepCount()` | `getStepCount(self) -> int` | mNOk + mNBad |
| `int getGoodStepCount()` | `getGoodStepCount(self) -> int` | |
| `int getBadStepCount()` | `getBadStepCount(self) -> int` | |
| `double[] integrate(x1,x2,ystart,eps,h1)` | `integrate(self,...) -> F64Array` | Mutates ystart in place AND returns y |

---

## Private members

| Java | Python |
|---|---|
| `double sign(double magnitude, double sign)` | `_sign(self, magnitude, sign)` |
| `void baseStep(x, y, dydx, h, yout, yerr)` | `_baseStep(self, x, y, dydx, h, yout, yerr)` |
| `double qcStep(x, y, dydx, htry, eps, yscal)` | `_qcStep(self, x, y, dydx, htry, eps, yscal)` |
| `void clearWorkspace()` | `_clearWorkspace(self)` |
| `int mNVariables` | `self.mNVariables` |
| `double mHDid` | `self.mHDid` |
| `double mHNext` | `self.mHNext` |
| `double mSaveInterval` | `self.mSaveInterval` |
| `double mMinStepSize` | `self.mMinStepSize` |
| `double[] mXSave` | `self.mXSave: Optional[F64Array]` |
| `double[][] mYSave` | `self.mYSave: Optional[F64Array]` (2D, shape kMax×nVars) |
| `int mNSaved` | `self.mNSaved` |
| `int mMaxSteps` | `self.mMaxSteps` |
| `int mNOk` | `self.mNOk` |
| `int mNBad` | `self.mNBad` |
| `double[] mWs2..mWs6, mYTemp` | `self.mWs2..self.mWs6, self.mYTemp: Optional[F64Array]` |
| `double[] mYErr, mQcYTemp` | `self.mYErr, self.mQcYTemp: Optional[F64Array]` |

---

## Overloaded methods
None.

---

## Mutable-output methods

- **`_qcStep`**: mutates `y` in place via `y[:] = mQcYTemp` after successful step.
  Sets `mHDid` and `mHNext` as side effects.
- **`integrate`**: mutates `ystart` in place at successful completion
  (`ystart[:] = y[:]`). Also returns the final `y` array.
- **`_baseStep`**: writes into `yout` and `yerr` (caller-supplied output arrays).

No guard needed for `_baseStep` inputs (all internal; never caller-supplied raw lists).

---

## Touchpoints into Jama / javax.swing / java.awt / java.io
None. This class only uses `java.lang.Math` and throws `UtilException`.

---

## Abstract class strategy
`AdaptiveRungeKutta` is `abstract` in Java. In Python: `abc.ABC` with
`@abc.abstractmethod` on `derivatives`. Direct parity testing via JPype is
**blocked by methodology limit M4** (JPype cannot extend Java abstract classes
from Python). The parity test file marks Part 2 with `@pytest.mark.skip` and
documents M4. Part 1 tests correctness against closed-form analytical solutions.

---

## Java-specific translation decisions

| Java pattern | Python translation | Rule |
|---|---|---|
| `abstract public class` | `abc.ABC` + `@abc.abstractmethod` | R2 |
| `double[]` fields and locals | `F64Array` (numpy float64) | R9 |
| `double[][] mYSave` | 2D `F64Array`, shape `(kMax, mNVariables)` | R9 |
| `System.arraycopy(src, 0, dst, 0, n)` | `dst[:n] = src[:n]` | R2 |
| `Double.MAX_VALUE` sentinel | `sys.float_info.max` (same IEEE-754 bits) | R2 |
| `do { ... } while (errmax > 1.0)` | `errmax = 2.0; while errmax > 1.0: ...` | R2 |
| `(int) Math.round(...)` in kMax | `int(math.floor(... + 0.5))` | R7 |
| `Math.abs`, `Math.max`, `Math.min` | `abs()`, `max()`, `min()` | R2 |
| `Math.pow` | `math.pow` | R2 |

---

## `_sign` helper semantics
Java `sign(magnitude, sign)` returns `+|magnitude|` when `sign >= 0.0` and
`-|magnitude|` otherwise. Critically, `-0.0 >= 0.0` is `True` in Java (IEEE 754),
so `sign(m, -0.0)` returns `+|m|`. Python `math.copysign(abs(m), -0.0)` returns
`-|m|`, diverging. The correct Python translation is the conditional:
`abs(magnitude) if sign >= 0.0 else -abs(magnitude)`.

---

## Suspected Java bugs
None identified. The Cash-Karp tableau coefficients and error control logic are
standard Numerical Recipes implementations with no obvious algorithmic defects.
`BUG_LEDGER` is empty.

---

## Static init order
No static initialisers. No cross-class static references. No `_initialize()` needed.

---

## Thread safety
Java docstring says "not thread-safe". Python port preserves this: no locks added.
