# LevenbergMarquardtParameterized Conversion Spec

## Java class
`gov.nist.microanalysis.Utility.LevenbergMarquardtParameterized`

Source: `src/gov/nist/microanalysis/Utility/LevenbergMarquardtParameterized.java`

> **Dependency note.** This class builds on `LevenbergMarquardt2`
> (`FitFunction`, `FitResult`), `LevenbergMarquardtConstrained`
> (`ConstrainedFitFunction`), `Constraint`, and `UncertainValue2`. The `Constraint`
> and `LevenbergMarquardt2` specs were repaired to **ver1_1_1** to match the Java
> source; this spec is consistent with those corrected APIs (real `Constraint`:
> `realToConstrained` / `constrainedToReal` / `derivative` / `getResult`; real
> `FitFunction`: `Matrix partials(Matrix)` / `Matrix compute(Matrix)`; non-static
> inner `FitResult` constructed as `FitResult(model, ff)`).

---

## Inbound dependencies (Java imports)
- `java.awt.event.ActionListener` — progress callback; **no Python equivalent** → replace with `Callable` (R10 deviation; same strategy as `LevenbergMarquardt2` / `ProgressEvent`).
- `java.util.ArrayList`, `java.util.Collection`, `java.util.Collections`, `java.util.HashMap`, `java.util.HashSet`, `java.util.List`, `java.util.Map`, `java.util.Set` — collections; use `list`/`dict`/`set` and `frozenset`/unmodifiable copies where Java uses `Collections.unmodifiable*`.
- `gov.nist.microanalysis.EPQLibrary.EPQException` — thrown by `compute`; import from `_epq_compat`.
- `Jama.Matrix` — fit matrices; import `JamaMatrix` from `_epq_compat`.
- `gov.nist.microanalysis.Utility.LevenbergMarquardt2` (`FitFunction`, `FitResult`) — sibling port.
- `gov.nist.microanalysis.Utility.LevenbergMarquardtConstrained` (`ConstrainedFitFunction`) — sibling port.
- `gov.nist.microanalysis.Utility.Constraint` (real interface + `None`) — sibling port.
- `gov.nist.microanalysis.Utility.UncertainValue2` — uncertainties throughout; sibling port.

---

## Outbound dependents (callers of public methods)
Top-level convenience fitter: callers tag parameters with `Parameter` tokens (name + `Constraint` + default + isFit) and a `Function`, and receive results keyed by `Parameter`. Not exhaustively audited.

---

## Public API surface

The outer class is small; the bulk is six inner types. **Every inner-class member is enumerated below (P5 discipline).**

### `LevenbergMarquardtParameterized` (outer class)

| Java signature | Python signature | Notes |
|---|---|---|
| *(implicit no-arg constructor)* | `__init__(self)` | Only field is `_mListener = None` |
| `public ParameterizedFitResult compute(Function f, double[] xVals, double[] yData, double[] sigma) throws EPQException` | `def compute(self, f, xVals, yData, sigma) -> ParameterizedFitResult` | Builds a `_ParameterizedFitFunction`, wraps it in a `ConstrainedFitFunction`, sets per-parameter constraints, runs `LevenbergMarquardtConstrained.compute`, wraps the result |
| `public void addActionListener(ActionListener al)` | `def addActionListener(self, al: Callable) -> None` | R10 — stores a progress `Callable`; passed through to the `LevenbergMarquardtConstrained` instance in `compute` |

### Inner class `Parameter` (`static public class`)

Module-level alias `Parameter = LevenbergMarquardtParameterized.Parameter` (R2).

| Java signature | Python signature | Notes |
|---|---|---|
| `public Parameter(String name, double defValue, boolean isFit)` | `__init__` path (3 args) | Delegates to the 4-arg form with `Constraint.Unconstrained()` |
| `public Parameter(String name, Constraint constraint, double defValue, boolean isFit)` | `__init__` path (4 args) | Primary; dispatch on whether the 2nd arg is a `Constraint` or a `float` |
| `public Constraint getConstraint()` | `def getConstraint(self) -> Constraint` | |
| `public void setConstraint(Constraint constraint)` | `def setConstraint(self, constraint) -> None` | |
| `public double getDefaultValue()` | `def getDefaultValue(self) -> float` | |
| `public void setDefaultValue(double iv)` | `def setDefaultValue(self, iv: float) -> None` | |
| `public double getValue(Map<Parameter,Double> param)` | `def getValue(self, param: dict) -> float` | `param[self]` if `_mIsFit` else `_mDefaultValue` (drop the Java `assert`) |
| `public UncertainValue2 getUncertainValue(Map<Parameter,UncertainValue2> param)` | `def getUncertainValue(self, param: dict) -> UncertainValue2` | Re-wraps with `_mName` as source |
| `public boolean isFit()` | `def isFit(self) -> bool` | |
| `public void setIsFit(boolean b)` | `def setIsFit(self, b: bool) -> None` | |
| `public int hashCode()` | `def __hash__(self) -> int` + `def hashCode(self) -> int` | R2 — hash from `_mName` only (`31*1 + hash(name)`) |
| `public boolean equals(Object obj)` | `def __eq__(self, obj) -> bool` + `def equals(self, obj) -> bool` | R2 — equal iff same class and same `_mName` |
| `public String toString()` | `def __str__(self) -> str` + `def toString(self) -> str` | R2 — `"{name}[{constraint},{default}]"` |
| `public String getName()` | `def getName(self) -> str` | |

### Inner class `ParameterObject<T>` (`static public class extends Parameter`)

Module-level alias `ParameterObject = LevenbergMarquardtParameterized.ParameterObject` (R2). Port as `class ParameterObject(Parameter, Generic[T])` with `T = TypeVar("T")`.

| Java signature | Python signature | Notes |
|---|---|---|
| `public ParameterObject(String name, Constraint constraint, double defValue, boolean isFit, T obj)` | `__init__` (5 args, public) | `super().__init__(name, constraint, defValue, isFit)` then store `_mObject` |
| `protected ParameterObject(String name, double defValue, boolean isFit, T obj)` | `__init__` (4 args) | Dispatch by arity/type; `protected` — keep accessible |
| `public T getObject()` | `def getObject(self) -> T` | |

### Inner interface `Function` (`public interface`)

Port as `class Function(abc.ABC)`; module-level alias. **M4 applies** (interface).

| Java signature | Python signature | Notes |
|---|---|---|
| `boolean isFitParameter(Parameter idx)` | `@abc.abstractmethod def isFitParameter(self, idx) -> bool` | |
| `Set<Parameter> getParameters(boolean all)` | `@abc.abstractmethod def getParameters(self, all: bool) -> set` | |
| `double compute(double arg, Map<Parameter,Double> param)` | `@abc.abstractmethod def compute(self, arg: float, param: dict) -> float` | |
| `double derivative(double arg, Map<Parameter,Double> param, Parameter idx)` | `@abc.abstractmethod def derivative(self, arg: float, param: dict, idx) -> float` | |
| `UncertainValue2 computeU(double arg, Map<Parameter,UncertainValue2> param)` | `@abc.abstractmethod def computeU(self, arg: float, param: dict) -> UncertainValue2` | |

### Inner interface `InvertableFunction extends Function` (`public interface`)

Port as `class InvertableFunction(Function)`; module-level alias. **M4 applies.**

| Java signature | Python signature | Notes |
|---|---|---|
| `UncertainValue2 inverse(double arg, Map<Parameter,UncertainValue2> param)` | `@abc.abstractmethod def inverse(self, arg: float, param: dict) -> UncertainValue2` | |

### Inner class `FunctionImpl` (`public static abstract class implements Function`)

Module-level alias; **M4 applies** (abstract — `compute`/`derivative`/`computeU` remain abstract).

| Java signature | Python signature | Notes |
|---|---|---|
| `final public Parameter add(Parameter p)` | `def add(self, p) -> Parameter` | R4 — single-parameter form; returns `p`. `final` has no Python analogue (note only) |
| `final public void add(Collection<Parameter> cp)` | `def addAll(self, cp) -> None` | R4 — collection form |
| `@Override final public Set<Parameter> getParameters(boolean all)` | `def getParameters(self, all: bool) -> set` | Returns fit-or-all subset |
| `@Override final public boolean isFitParameter(Parameter p)` | `def isFitParameter(self, p) -> bool` | |
| `final public Map<Parameter,UncertainValue2> extract(Map<Parameter,UncertainValue2> fitResult)` | `def extract(self, fitResult: dict) -> dict` | |
| `@Override public String toString()` | `def __str__(self) -> str` + `def toString(self) -> str` | R2 |
| *(inherited abstract)* `compute`, `derivative`, `computeU` | remain `@abc.abstractmethod` | subclasses implement |

### Inner class `ParameterizedFitFunction` (`private class implements FitFunction`)

Private → port as `class _ParameterizedFitFunction(FitFunction)` (R1). No module-level alias (private). It is an **inner (non-static) class** in Java but uses no outer state — port as an ordinary nested class.

| Java signature | Python signature | Notes |
|---|---|---|
| `private ParameterizedFitFunction(double[] x, Function f)` | `__init__(self, x, f)` | Copies `x` to `_mOrdinate`; `_mParameters = list(f.getParameters(False))` |
| `private Map<Parameter,Double> getUpdatedParam(Matrix params)` | `def _getUpdatedParam(self, params) -> dict` | Maps each parameter to `params.get(i,0)` |
| `@Override public Matrix partials(Matrix params)` | `def partials(self, params) -> JamaMatrix` | n×m Jacobian via `_mFunction.derivative` |
| `@Override public Matrix compute(Matrix params)` | `def compute(self, params) -> JamaMatrix` | n×1 values via `_mFunction.compute` |
| `public int paramSize()` | `def paramSize(self) -> int` | `len(self._mParameters)` |

### Inner class `ParameterizedFitResult` (`public static class extends FitResult`)

Module-level alias (R2); **inherits from the LM2 `FitResult` port**.

| Java signature | Python signature | Notes |
|---|---|---|
| `private ParameterizedFitResult(LevenbergMarquardt2 lm2, FitResult fr, List<Parameter> params)` | `__init__(self, lm2, fr, params)` | **`lm2.super(fr.mFunction)` caveat below**; copies the 6 `FitResult` fields; stores an unmodifiable copy of `params` |
| `public int indexOf(Parameter p)` | `def indexOf(self, p) -> int` | |
| `public UncertainValue2 getBestFitValue(Parameter p)` | `def getBestFitValue(self, p) -> UncertainValue2` | |
| `public double getBestFit(Parameter p)` | `def getBestFit(self, p) -> float` | |
| `public List<Parameter> getParameters()` | `def getParameters(self) -> list` | R4 — distinct from `Function.getParameters(bool)`; no-arg here |
| `public List<Parameter> getParametersByClass(Class<? extends Parameter> pc)` | `def getParametersByClass(self, pc: type) -> list` | `type(p) == pc` |
| `public Parameter getParameterByClass(Class<? extends Parameter> pc)` | `def getParameterByClass(self, pc: type) -> Optional[Parameter]` | Returns `None` if absent |
| `public Map<Parameter,UncertainValue2> getParameterMap()` | `def getParameterMap(self) -> dict` | |
| `public Map<Parameter,Double> getResults()` | `def getResults(self) -> dict` | nominal values |
| `public String tabulate()` | `def tabulate(self) -> str` | Tab-separated table |

---

## Private / protected members

| Java | Python |
|---|---|
| `private ActionListener mListener` (outer) | `self._mListener: Optional[Callable]` |
| `Parameter`: `mConstraint`, `mDefaultValue`, `mName` (final), `mIsFit` | `_mConstraint`, `_mDefaultValue`, `_mName`, `_mIsFit` |
| `ParameterObject`: `mObject` (final) | `_mObject` |
| `FunctionImpl`: `mParameters` (HashSet) | `_mParameters: set` |
| `_ParameterizedFitFunction`: `mParameters`, `mFunction`, `mOrdinate` | `_mParameters`, `_mFunction`, `_mOrdinate` |
| `ParameterizedFitResult`: `mParameters` (List) | `_mParameters: list` |

---

## Overloaded methods (split plan)
- `Parameter` constructor: `(name, defValue, isFit)` vs `(name, constraint, defValue, isFit)` → `__init__` dispatch on 2nd-arg type (`Constraint` vs `float`).
- `ParameterObject` constructor: 5-arg public vs 4-arg protected → `__init__` dispatch by arity.
- `FunctionImpl.add`: `add(Parameter)` / `add(Collection)` → `add` / `addAll` (R4).
- `getParameters`: `Function.getParameters(boolean)` vs `ParameterizedFitResult.getParameters()` are on **different classes** — no conflict; keep both names.

---

## Mutable-output methods
None expose caller buffers for in-place writes. `add`/`addAll`/`setConstraint`/`setIsFit`/`setDefaultValue` mutate `self`. `extract`, `getParameterMap`, `getResults`, `getParametersByClass` return new `dict`/`list` objects (Java wraps several in `Collections.unmodifiable*`; port as plain copies, optionally `MappingProxyType`/`tuple` if read-only enforcement is desired — document as R10 if you deviate).

---

## Touchpoints into Jama / javax.swing / java.awt / java.io
- `Jama.Matrix` in `_ParameterizedFitFunction.partials`/`compute` and the `compute` driver → `JamaMatrix` (`JamaMatrix.zeros(rows, cols)`, `get`/`set`).
- `java.awt.event.ActionListener` → `Callable` (R10 deviation). The outer `compute` wires `_mListener` into the `LevenbergMarquardtConstrained` instance via its `addActionListener`/listener mechanism.

---

## Abstract class strategy
`LevenbergMarquardtParameterized` IS_ABSTRACT = False.

Abstract/interface inner types: `Function`, `InvertableFunction`, `FunctionImpl` — all IS_ABSTRACT = True. **M4 applies** to each: JPype cannot subclass these from Python directly.

Parity strategy:
- Mark cross-engine Java parity classes `@pytest.mark.skip(reason="M4: ...")`.
- Validate analytically: implement a concrete Python `FunctionImpl` subclass (e.g. `y = a·exp(b·x)` or a polynomial) with `Parameter` tokens (mix of fit and fixed, with `Constraint.Positive`/`Unconstrained`), run `compute(...)`, and assert the recovered best-fit values (via `getBestFit`/`getParameterMap`) against a closed-form or `scipy.optimize.curve_fit` reference within `TOL_NR_LIB`.

---

## Java-specific translation decisions

| Java pattern | Python translation | Rule |
|---|---|---|
| `lm2.super(fr.mFunction)` (qualified inner-superclass ctor) | `ParameterizedFitResult` extends the LM2 `FitResult`. Java's `lm2.super(ff)` invokes the non-static `FitResult(FitFunction)` constructor bound to instance `lm2`. In Python call `super().__init__(lm2, fr._mFunction)` matching the LM2 `FitResult.__init__(self, model, ff)` signature. **Confirm that signature in the LM2 port** | R2 / inner class |
| `fr.mFunction`, `fr.mBestParams`, … | LM2 `FitResult` `protected` fields → `_mFunction`, `_mBestParams`, etc. (R1) | R1 |
| `new Constraint.None()` | `Constraint.Unconstrained()` (Python builtin clash) | R1/R10 |
| `Constraint mConstraint` token usage | real `Constraint` API (the parameter carries a constraint object; the fit transforms through `ConstrainedFitFunction`) | depends on corrected Constraint/LMConstrained ports |
| `ActionListener` / `addActionListener` | `Callable` / `addActionListener(callback)` | R10 deviation |
| `getClass().equals(pc)` | `type(p) == pc` | |
| `Collections.unmodifiableList/Map(...)` | plain `list(...)` / `dict(...)` copy (or read-only proxy) | R10 note |
| `mName.hashCode()`, `31*result + ...` | `hash(self._mName)` folded into the documented `31*1 + h` form; ensure `__hash__`/`__eq__` stay consistent | R2 |
| `assert (!mIsFit) || param.containsKey(this)` in `getValue`/`getUncertainValue` | **Omit** the assert; keep only the conditional return | Cross-Cutting — Java `assert` |
| `StringBuffer` in `tabulate` | build a Python `str`/`io.StringIO` | |

---

## Suspected Java bugs
None identified. `Parameter.equals`/`hashCode` intentionally key on `mName` only (two parameters with the same name are "equal"); preserve this — it is the design, used for `Map<Parameter,…>` lookups.

`BUG_LEDGER`: `()` — no `JAVA-BUG-N` here. (Numpy-SVD substitution lives in the LM2 port.)

---

## Static init order
`Parameter` delegating constructor builds a `Constraint.Unconstrained()` default. `_ParameterizedFitResult` depends on the LM2 `FitResult` port being importable. No cross-class static initialiser ordering concern; all state is instance state.

---

## Thread safety
Not thread-safe. `compute` constructs fresh helper objects per call (so distinct `compute` calls on one engine are largely independent), but `addActionListener` mutates `_mListener` and `Parameter`/`FunctionImpl` mutators alter shared token state. Treat instances as single-threaded.
