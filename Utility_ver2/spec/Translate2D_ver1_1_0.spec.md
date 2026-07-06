# Translate2D Conversion Spec

## Java class
`gov.nist.microanalysis.Utility.Translate2D`

Source: `src/gov/nist/microanalysis/Utility/Translate2D.java`

---

## Inbound dependencies (Java imports)
- `java.text.NumberFormat` — parameter type in `toString()` (only as a local variable inside `toString`; uses `HalfUpFormat`)
- `java.util.Collection`, `java.util.Iterator` — parameter type for `calibrate` and `error`
- `gov.nist.microanalysis.Utility.Math2` — `Math2.sqr`, `Math2.distance`; import from sibling port
- `gov.nist.microanalysis.Utility.Simplex` — used in an anonymous inner class inside `calibrate` (3+ point case); import from sibling port
- `gov.nist.microanalysis.Utility.HalfUpFormat` — used inside `toString()`; import from sibling port
- `gov.nist.microanalysis.Utility.UtilException` — caught in `calibrate`, swallowed silently; import from sibling port

No EPQException, Jama, or java.awt imports directly.

---

## Outbound dependents (callers of public methods)
`StageRelocation` uses a similar transformation but is independent. Not audited further.

---

## Public API surface

### Inner class: `Translate2D.CalibrationPoint`

`CalibrationPoint` is declared `static final public` but its constructor is **package-private** (no access modifier):

```java
CalibrationPoint(double[] orig, double[] newPt) { ... }
```

In Java, this means callers outside the package must use the static factory methods. In Python, expose `__init__` normally but note that direct construction is only intended via the factory methods.

| Java signature | Python signature | Notes |
|---|---|---|
| `public double getX0()` | `def getX0(self) -> float` | |
| `public double getY0()` | `def getY0(self) -> float` | |
| `public double getX1()` | `def getX1(self) -> float` | |
| `public double getY1()` | `def getY1(self) -> float` | |
| `public boolean different(CalibrationPoint cp)` | `def different(self, cp: CalibrationPoint) -> bool` | Uses `Math2.distance` internally |

Module-level alias: `CalibrationPoint = Translate2D.CalibrationPoint`. (R2)

### Static factory methods (on `Translate2D`)

| Java signature | Python signature | Notes |
|---|---|---|
| `static public CalibrationPoint createCalibrationPoint(double[] oldCoord, double[] newCoord)` | `@staticmethod def createCalibrationPoint_arr(oldCoord: list[float], newCoord: list[float]) -> CalibrationPoint` | R4 |
| `static public CalibrationPoint createCalibrationPoint(double oldX, double oldY, double newX, double newY)` | `@staticmethod def createCalibrationPoint_xy(oldX: float, oldY: float, newX: float, newY: float) -> CalibrationPoint` | R4 |

### Constructors

| Java signature | Python signature | Notes |
|---|---|---|
| `public Translate2D()` | `__init__(self)` | Null-translation: offset=(0,0), scale=(1,1), rotation=0 |
| `public Translate2D(Translate2D t2d)` | `__init__(self, t2d: Translate2D)` | Copy constructor — dispatch by isinstance |
| `public Translate2D(double[] offset, double rotation)` | `__init__(self, offset: list[float], rotation: float)` | Dispatch: 2-element `list` + `float` |
| `public Translate2D(double[] offset, double[] scale, double rotation, boolean invertXAxis)` | `__init__(self, offset: list[float], scale: list[float], rotation: float, invertXAxis: bool)` | Full specification |

All four constructors must be dispatched in `__init__` by argument type and count.

### Public instance methods

| Java signature | Python signature | Notes |
|---|---|---|
| `public double calibrate(Collection<CalibrationPoint> calPts)` | `def calibrate(self, calPts: list[CalibrationPoint]) -> float` | Returns RMS residual error; uses anonymous `Simplex` subclass internally for 3+ points |
| `public double error(Collection<CalibrationPoint> calPts)` | `def error(self, calPts: list[CalibrationPoint]) -> float` | RMS error without recalibrating |
| `public double[] compute(double[] oldCoord)` | `def compute(self, oldCoord: list[float]) -> list[float]` | Forward transform; accepts 2- or 3-element input |
| `public double[] inverse(double[] newCoord)` | `def inverse(self, newCoord: list[float]) -> list[float]` | Inverse transform; always returns 2-element |
| `public boolean isXAxisInverted()` | `def isXAxisInverted(self) -> bool` | |
| `public double getXScale()` | `def getXScale(self) -> float` | |
| `public double getYScale()` | `def getYScale(self) -> float` | |
| `public double getXOffset()` | `def getXOffset(self) -> float` | |
| `public double getYOffset()` | `def getYOffset(self) -> float` | |
| `public double getRotation()` | `def getRotation(self) -> float` | Radians |
| `public String toString()` | `def __str__(self) -> str` + `def toString(self) -> str` | R2 — uses `HalfUpFormat` internally |

---

## Private / protected members

| Java | Python |
|---|---|
| `private static final double SCALE = 1.0e-3` | `_SCALE: float = 1.0e-3` |
| `private double[] mOffset = new double[2]` | `self._mOffset: list[float] = [0.0, 0.0]` |
| `private double[] mScale = new double[2]` | `self._mScale: list[float] = [1.0, 1.0]` |
| `private double mRotation` | `self._mRotation: float` |
| `private boolean mXAxisInverted` | `self._mXAxisInverted: bool` |
| `private void reset()` | `def _reset(self) -> None` | R1 — private → underscore prefix |

---

## Overloaded methods (split plan)

`createCalibrationPoint` has two static overloads → `createCalibrationPoint_arr` / `createCalibrationPoint_xy`.

All four constructors collapse into one `__init__` dispatched by argument type and count.

---

## Mutable-output methods

`compute` calls `oldCoord.clone()` — returns a new array, does not mutate input. Port: `res = list(oldCoord)`, then overwrite `res[0]`, `res[1]` (and optionally `res[2]`).

Constructor copy paths make defensive copies of `offset` and `scale` arrays (R5): `self._mOffset = list(offset)`.

---

## Touchpoints into Jama / javax.swing / java.awt / java.io
None directly. `HalfUpFormat` used in `toString()` is a sibling import.

---

## Abstract class strategy
Not applicable. `Translate2D` is a concrete class.

### Anonymous Simplex subclass inside `calibrate`

When `calPts.size() > 2`, Java uses an anonymous inner class `RefineCalibration extends Simplex` to minimize fit residuals via Nelder-Mead. In Python, create a named local class inside `calibrate`:

```python
def calibrate(self, calPts):
    ...
    if len(calPts) > 2:
        class RefineCalibration(Simplex):
            def function(self, x):
                ...
        rc = RefineCalibration()
        ...
```

This local class implements `Simplex.function` and overrides `getValue`.

**Note**: The Java source calls `rc.perform(Simplex.randomizedStartingPoints(center, scale))`. This signature (`randomizedStartingPoints(double[], double[])` returning `double[][]`) was not found in the `Simplex` spec. It may be a static overload of `randomizedStartingPoints` not captured in the prior analysis — requires cross-check with full `Simplex.java` source. Flag as `# REVIEW: Simplex.randomizedStartingPoints signature`.

---

## Suspected Java bugs

**JAVA-NOTE-1 — `calibrate` tries 5 candidate orientations in a loop (i=0..4) but commits to `bestI` after 4 trials.**
The loop runs `i = 0..4`; when `i == 4`, it sets `ii = bestI` (the best-so-far index) and applies that configuration. This is correct — the fifth iteration applies the winner. Not a bug; just unusual loop structure.

**JAVA-BUG-1 — `inverse` always returns 2-element array even when input is 3-element.**
`compute` handles 3-element input by appending `mRotation` to `res[2]`. `inverse` always returns `new double[2]`, ignoring a potential 3-element input. If callers pass a 3D coordinate to `inverse`, the third component is silently dropped.
Disposition: **Preserve** — port verbatim; document as `# JAVA-BUG-1`.

---

## Static init order
`_SCALE = 1.0e-3` is a compile-time constant. No ordering concern.

---

## Thread safety
Not thread-safe. `_mOffset`, `_mScale`, `_mRotation`, `_mXAxisInverted` are all mutable. `calibrate` writes these fields; concurrent calibration calls on the same instance would corrupt state.
