# LevenbergMarquardtConstrained Conversion Spec

## Java class
`gov.nist.microanalysis.Utility.LevenbergMarquardtConstrained`

Source: `src/gov/nist/microanalysis/Utility/LevenbergMarquardtConstrained.java`

> **Dependency note.** The `Constraint` and `LevenbergMarquardt2` specs were repaired
> to **ver1_1_1** to match the Java source; this spec is consistent with those corrected
> APIs (real `Constraint`: `realToConstrained` / `constrainedToReal` / `derivative` /
> `getResult`; real `FitFunction`: `Matrix partials(Matrix)` / `Matrix compute(Matrix)`;
> non-static inner `FitResult` constructed as `FitResult(model, ff)`).

---

## Inbound dependencies (Java imports)
- `Jama.Matrix` — parameter/result matrices throughout; import `JamaMatrix` from `_epq_compat`.
- `gov.nist.microanalysis.EPQLibrary.EPQException` — thrown by `compute`; import from `_epq_compat`.
- **Superclass** `gov.nist.microanalysis.Utility.LevenbergMarquardt2` — import the sibling port; supplies the real `FitFunction` interface (`Matrix partials(Matrix)`, `Matrix compute(Matrix)`), the `FitResult` inner class, and `compute(FitFunction, Matrix yData, Matrix sigma, Matrix p0)`.
- `gov.nist.microanalysis.Utility.Constraint` — the real interface (`realToConstrained`, `constrainedToReal`, `derivative`, `getResult(UncertainValue2)`) and its `None` inner class; import from the sibling port.
- `gov.nist.microanalysis.Utility.UncertainValue2` — `FitResult.mBestParams` element type and `Constraint.getResult` return type; import from the sibling port.

---

## Outbound dependents (callers of public methods)
Used to fit when parameters must be bounded/positive/fractional. `LevenbergMarquardtParameterized` builds a `ConstrainedFitFunction` and calls this class's `compute`. Not exhaustively audited.

---

## Public API surface

`LevenbergMarquardtConstrained extends LevenbergMarquardt2`. It overrides `compute` and adds the public static inner class `ConstrainedFitFunction`.

### `LevenbergMarquardtConstrained` (outer class)

| Java signature | Python signature | Notes |
|---|---|---|
| `public LevenbergMarquardtConstrained()` | `__init__(self)` | `super().__init__()` |
| `@Override public FitResult compute(FitFunction ff, Matrix yData, Matrix sigma, Matrix p0) throws EPQException` | `def compute(self, ff, yData, sigma, p0) -> FitResult` | **Same signature as the parent** — a clean override; call `super().compute(...)`. No parent-alias trick needed (contrast Integrator P9). See dispatch note |

**`compute` dispatch.** If `ff` is a `ConstrainedFitFunction`, transform the start point to real space (`cff.constrainedToReal_matrix(p0)`), call `super().compute(cff, yData, sigma, ...)`, then transform the result back with `cff.realToConstrained_result(tmp)`. Otherwise, delegate straight to `super().compute(ff, yData, sigma, p0)`.

### Inner class `LevenbergMarquardtConstrained.ConstrainedFitFunction` (implements `FitFunction`)

Port as a nested `class ConstrainedFitFunction(FitFunction)` plus module-level alias `ConstrainedFitFunction = LevenbergMarquardtConstrained.ConstrainedFitFunction` (R2). Enumerate **every** method (P5):

| Java signature | Python signature | Notes |
|---|---|---|
| `public ConstrainedFitFunction(FitFunction ff, int paramDim)` | `__init__(self, ff, paramDim: int)` | Fills `_mConstraints` with `Constraint.None()` ×`paramDim` |
| `public void setConstraint(int paramIdx, Constraint c)` | `def setConstraint(self, paramIdx: int, c: Constraint) -> None` | |
| `public Matrix realToConstrained(Matrix rParams)` | `def realToConstrained_matrix(self, rParams) -> JamaMatrix` | R4 — overload split |
| `public Matrix constrainedToReal(Matrix rParams)` | `def constrainedToReal_matrix(self, rParams) -> JamaMatrix` | R4 — there is one `constrainedToReal` (Matrix); keep the `_matrix` suffix for symmetry |
| `@Override public Matrix partials(Matrix rParams)` | `def partials(self, rParams) -> JamaMatrix` | Chain rule: parent partials × constraint derivative per column |
| `@Override public Matrix compute(Matrix rParams)` | `def compute(self, rParams) -> JamaMatrix` | `mFitFunction.compute(realToConstrained_matrix(rParams))` |
| `public FitResult realToConstrained(FitResult fr)` | `def realToConstrained_result(self, fr) -> FitResult` | R4 — overload split; **inner-class instantiation caveat below** |

---

## Private / protected members

| Java | Python |
|---|---|
| `private final Constraint[] mConstraints` (in `ConstrainedFitFunction`) | `self._mConstraints: list[Constraint]` |
| `private final FitFunction mFitFunction` (in `ConstrainedFitFunction`) | `self._mFitFunction: FitFunction` |

The outer `LevenbergMarquardtConstrained` adds no fields of its own.

---

## Overloaded methods (split plan)
`ConstrainedFitFunction.realToConstrained` is **overloaded** by argument type:
- `realToConstrained(Matrix)` → `realToConstrained_matrix` (parameter-space map, per element)
- `realToConstrained(FitResult)` → `realToConstrained_result` (result transform, with covariance chain rule)

`constrainedToReal(Matrix)` has a single form → `constrainedToReal_matrix` (suffix kept for naming symmetry with its inverse).

`compute(Matrix)` (FitFunction method) and `partials(Matrix)` are single-form overrides — keep the interface names (`compute`, `partials`); do **not** suffix them, or the `FitFunction` contract breaks.

---

## Mutable-output methods
None expose caller buffers for in-place writes. `realToConstrained_*` / `constrainedToReal_*` allocate new `JamaMatrix` results. `realToConstrained_result` copies `fr.mCovariance` (`fr.mCovariance.copy()` → `JamaMatrix(fr._mCovariance.getArrayCopy())`) before scaling — preserve the copy so the source `FitResult` is untouched.

---

## Touchpoints into Jama / javax.swing / java.awt / java.io
- `Jama.Matrix` everywhere → `JamaMatrix` (`get`/`set`, `getRowDimension`/`getColumnDimension`, `copy`). Build new matrices with `JamaMatrix.zeros(rows, 1)`.
- No `javax.swing`/`java.awt`/`java.io`. (The inherited `addActionListener` lives in `LevenbergMarquardt2`; see that port's listener deviation.)

---

## Abstract class strategy
IS_ABSTRACT = False for `LevenbergMarquardtConstrained`.

`FitFunction` (inherited interface) IS_ABSTRACT = True — **M4 applies** to it. `ConstrainedFitFunction` is concrete but *wraps* a user `FitFunction`; parity tests must supply a concrete `FitFunction`. Because JPype cannot subclass the Java `FitFunction` interface from Python via plain subclassing (only `@JImplements` works for interfaces), prefer:
- a Java-side concrete `FitFunction` if the jar provides one, driven through `compute`; or
- analytic validation in pure Python: build a Python `FitFunction` (linear/exponential model), wrap it in `ConstrainedFitFunction` with known constraints, and assert the fitted parameters against a closed-form answer; mark the cross-engine Java parity portion `@pytest.mark.skip(reason="M4: ...")`.

---

## Java-specific translation decisions

| Java pattern | Python translation | Rule |
|---|---|---|
| `extends LevenbergMarquardt2` | `class LevenbergMarquardtConstrained(LevenbergMarquardt2)` | inheritance |
| `super.compute(...)` (same signature) | `super().compute(...)` — safe; identical signature | Not a P9 case |
| `ff instanceof ConstrainedFitFunction` | `isinstance(ff, ConstrainedFitFunction)` | |
| `new Constraint.None()` ×paramDim | `[Constraint.Unconstrained() for _ in range(paramDim)]` — Java `None` → port `Unconstrained` (Python builtin clash). **Confirm the Constraint port's chosen name** | R1/R10 |
| `mConstraints[i].realToConstrained(double)` etc. | call the real `Constraint` methods (`realToConstrained`, `constrainedToReal`, `derivative`, `getResult`) | depends on corrected Constraint port |
| `fr.getModel().new FitResult(mFitFunction)` | **Inner-class instantiation:** Java creates a `FitResult` bound to the model instance `fr.getModel()`. In Python the nested `FitResult` does not auto-bind to an outer instance — construct it explicitly, e.g. `FitResult(fr.getModel(), self._mFitFunction)`, matching the LM2 port's `FitResult.__init__(self, model, ff)` signature. **Confirm that signature in the LM2 port** | R2 / inner class |
| `res.mBestParams`, `res.mBestY`, `res.mChiSq`, `res.mCovariance`, `res.mImproveCount`, `res.mIterCount` | access the LM2 `FitResult` fields. These are `protected` in Java → `_mBestParams`, `_mBestY`, `_mChiSq`, `_mCovariance`, `_mImproveCount`, `_mIterCount` (R1). Cross-class field writes are allowed in Python | R1 |
| `covar.set(r, c, fr.mCovariance.get(r,c) * dp[r] * dp[c])` | element-wise covariance chain-rule scaling | faithful port |
| `rParams.getColumnDimension() == 1` asserts | **Omit** (disabled-by-default Java assert) | Cross-Cutting — Java `assert` |
| reverse loops `for (i = n-1; i >= 0; --i)` | `for i in range(n - 1, -1, -1)` — order is immaterial here but preserve for fidelity | |

---

## Suspected Java bugs
None identified. The constraint chain-rule transforms are standard.

**Note on `FitResult` field access (`mBestParams` etc.).** These are `protected` fields of the LM2 `FitResult`. The port must reference them with the R1 underscore (`_mBestParams`). A parity/unit test that reads them must also use the underscored name (TESTING_GUIDE "Accessing port internals") — the Integrator/ARK gen-1 failures (P3) came from reading `mField` without the underscore.

`BUG_LEDGER`: `()` — no `JAVA-BUG-N`. (The inherited LM2 `compute` carries the numpy-SVD substitution; that is recorded in the LM2 port, not here.)

---

## Static init order
None. `ConstrainedFitFunction` builds its `_mConstraints` array in its constructor; the outer class has no static state.

---

## Thread safety
Not thread-safe. `compute` mutates inherited LM2 iteration state; `ConstrainedFitFunction.setConstraint` mutates `_mConstraints`. Concurrent use on one instance corrupts state.
