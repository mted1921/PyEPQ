# LinearLeastSquaresMS Conversion Spec

## Java class
`gov.nist.microanalysis.Utility.LinearLeastSquaresMS`

Source: `src/gov/nist/microanalysis/Utility/LinearLeastSquaresMS.java`

---

## Inbound dependencies (Java imports)
- `java.util.Arrays` — `Arrays.sort`, `Arrays.binarySearch` in `editSingularValues`; `Math.*` elsewhere.
- `gov.nist.microanalysis.EPQLibrary.EPQException` — propagated from `perform`/`computeMetric`; import from `_epq_compat`.
- **Superclass** `gov.nist.microanalysis.Utility.LinearLeastSquares` — import the sibling port (filename from `UTILITY_LEDGER.md`); inherits all SVD/fit machinery, `Math2`, `UncertainValue2`, `JamaMatrix` transitively.

---

## Outbound dependents (callers of public methods)
A drop-in `LinearLeastSquares` that adds Bayesian model selection ("MS") to trim the number of fit parameters. Concrete subclasses still implement `fitFunction`/`fitFunctionCount`. Not exhaustively audited.

---

## Public API surface

`LinearLeastSquaresMS extends LinearLeastSquares`. It is still **abstract** (does not implement `fitFunction`/`fitFunctionCount`). It overrides `editSingularValues`, `perform`, `reevaluateAll` and adds the optimisation controls.

### Constructors

| Java signature | Python signature | Notes |
|---|---|---|
| `public LinearLeastSquaresMS()` | `__init__(self)` | `super().__init__()` |
| `public LinearLeastSquaresMS(double[] x, double[] y, double[] sig)` | `__init__(self, x, y, sig)` | `super().__init__(x, y, sig)` |
| `public LinearLeastSquaresMS(double[] x, double[] y)` | `__init__(self, x, y)` | `super().__init__(x, y)` |

`__init__` dispatches by arg count/type and forwards to the parent constructor (same dispatch contract as `LinearLeastSquares`).

### Public instance methods

| Java signature | Python signature | Notes |
|---|---|---|
| `public boolean isOptimize()` | `def isOptimize(self) -> bool` | |
| `public void setOptimize(boolean optimize)` | `def setOptimize(self, optimize: bool) -> None` | Toggling forces `_reevaluateAll()` |

### Protected methods (overrides + new)

| Java signature | Python signature | Notes |
|---|---|---|
| `@Override protected double[] editSingularValues(double[] wi)` | `def _editSingularValues(self, wi) -> list[float]` | Keeps the largest `mNParams` singular values, zeroes the smallest `nDrop`; also applies `_mZero` mask and the `wMax * TOLERANCE` floor |
| `@Override protected void perform() throws EPQException` | `def _perform(self) -> None` | Model-selection loop — see below |
| `@Override protected void reevaluateAll()` | `def _reevaluateAll(self) -> None` | Resets `_mNParams = _INT_MIN` then `super()._reevaluateAll()` |
| `protected double computeMetric() throws EPQException` | `def _computeMetric(self) -> float` | Bayesian model-selection metric (Sivia eqn 4.20) |

---

## Private / protected members

| Java | Python |
|---|---|
| `private int mNParams = Integer.MIN_VALUE` | `self._mNParams: int = _INT_MIN` — sentinel "not initialised"; define module const `_INT_MIN = -2147483648` |
| `private boolean[] mZero = null` | `self._mZero: Optional[list[bool]] = None` — per-parameter zeroing mask |
| `private double mAMax = Double.MAX_VALUE` | `self._mAMax: float = sys.float_info.max` — assumed parameter prior scale |
| `private double[] mMetric` | `self._mMetric: Optional[list[float]]` |
| `private boolean mOptimize = false` | `self._mOptimize: bool = False` |

---

## Overloaded methods (split plan)
No new overloads beyond the inherited constructor split. The three constructors forward to the parent.

---

## Mutable-output methods
`_editSingularValues` clones its input (`wi.clone()`) and returns a new list — it does **not** mutate the caller's array. No R5 guard required. The inherited `fitFunction` contract (mutable `afunc`) is unchanged.

---

## Touchpoints into Jama / javax.swing / java.awt / java.io
- `Jama.Matrix` via the inherited `covariance()` (`_mCovariance.det()` in `computeMetric`) → `JamaMatrix.det()`.
- SVD substitution is inherited from `LinearLeastSquares` (numpy SVD; SCIPY-DEV-1). No new Jama/SVD touchpoints. No `javax.swing`/`java.awt`/`java.io`.

---

## Abstract class strategy
IS_ABSTRACT = True (inherits the two abstract methods unimplemented).

Port as `class LinearLeastSquaresMS(LinearLeastSquares)` — it remains `abc.ABC` by inheritance.

**M4 applies** (same as the parent). Parity strategy:
- Skip the direct parity class (`@pytest.mark.skip(reason="M4: ...")`).
- Validate analytically with a **concrete Python subclass** (e.g. polynomial basis). Test two regimes explicitly:
  - `setOptimize(False)` (default): behaves like `LinearLeastSquares` with the MS singular-value editing — verify fit parameters and that negative coefficients are zeroed.
  - `setOptimize(True)`: verify the parameter-count selection runs and the chosen `mNParams` minimises `computeMetric()` over the tested range.

---

## Java-specific translation decisions

| Java pattern | Python translation | Rule |
|---|---|---|
| `mNParams == Integer.MIN_VALUE` sentinel | `self._mNParams == _INT_MIN` with `_INT_MIN = -2147483648` | Preserve literal sentinel value |
| `Arrays.sort(dup)` | `dup = sorted(w)` (ascending, matches Java) | |
| `Arrays.binarySearch(dup, w[j]) < nDrop` | `bisect.bisect_left(dup, w[j]) < nDrop` — **note:** Java `binarySearch` returns *an* index of a matching element (unspecified which when duplicates exist); `bisect_left` returns the leftmost. For distinct singular values the result is identical; for tied values flag a possible 1-index divergence with an R10 comment | R10 note |
| `assert wMax >= 0.0` | **Omit** (disabled-by-default Java assert) | Cross-Cutting — Java `assert` |
| `Math.pow((4.0 * Math.PI) / mAMax, mNParams)` | `math.pow((4.0 * math.pi) / self._mAMax, self._mNParams)` | |
| `Math.exp(-0.5 * chiSquared())` | `math.exp(-0.5 * self.chiSquared())` | |
| `covariance().det()` | `self.covariance().det()` (`JamaMatrix.det()`) | R3 |
| repeated `super.perform()` calls inside the loop | `super()._perform()` — parent method is `_perform`; call it directly | Faithful port |
| `reevaluate()` (parent protected) | `self._reevaluate()` | R1 — protected → `_` |

---

## Suspected Java bugs
None identified. The model-selection loop and metric are intentional. The only transcription hazard is the `binarySearch`→`bisect_left` tie-handling noted above (a faithfulness nuance, not a Java bug).

`BUG_LEDGER`: inherits the parent's `SCIPY-DEV-1` (numpy SVD); no new entries unless the `bisect` divergence is made observable, in which case record it as `DEVIATION-1`.

---

## Static init order
`_INT_MIN` is a module constant. Instance fields initialise in `__init__` (after `super().__init__()`, which may already trigger `setData`/`reevaluateAll`). **Ordering caveat:** the parent constructor calls `reevaluateAll()`, which is overridden here to touch `_mNParams`; ensure `_mNParams` (and the other MS fields) are assigned **before** `super().__init__(...)` runs, or guard the override against missing attributes. Document this in `CHANGES`.

---

## Thread safety
Not thread-safe. `_perform` mutates `_mNParams`, `_mZero`, `_mAMax`, `_mMetric` and repeatedly re-runs the inherited (lock-guarded) fit. Concurrent use on one instance corrupts the selection state.
