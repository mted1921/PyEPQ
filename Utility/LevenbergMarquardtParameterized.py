r"""
LevenbergMarquardtParameterized_ver2_1_0.py — Python port of
gov.nist.microanalysis.Utility.LevenbergMarquardtParameterized

Guide version : 2
Generation    : 1
Port-code fixes: 0

CHANGES
-------
- `java.awt.event.ActionListener` → `Callable[[int], None]` (R10): no Python
  equivalent of ``ActionListener.actionPerformed(ActionEvent)``; same strategy
  as ``LevenbergMarquardt2``.  ``addActionListener`` stores the callable and
  passes it through to the inner ``LevenbergMarquardtConstrained`` instance.
- `Constraint.None` → `Constraint.Unconstrained` / ``Unconstrained()``
  throughout (R1/R10: ``None`` is a Python built-in; the ``Constraint`` port
  already renamed this inner class).
- `lm2.super(fr.mFunction)` (qualified inner super-constructor) →
  ``super().__init__(lm2, fr._mFunction)`` to match
  ``LevenbergMarquardt2.FitResult.__init__(model, ff)``.
- ``Parameter`` constructor overloads dispatched by the presence of the
  optional *isfit* keyword (R4): ``(name, defValue, isFit)`` uses
  ``Unconstrained()``; ``(name, constraint, defValue, isFit)`` uses the
  supplied constraint.
- ``ParameterObject`` constructor overloads dispatched by arity (R4):
  4-arg protected form ``(name, defValue, isFit, obj)`` vs 5-arg public form
  ``(name, constraint, defValue, isFit, obj)``.
- Java ``assert`` statements in ``getValue``/``getUncertainValue`` dropped
  (disabled by default — Cross-Cutting rule).
- ``FunctionImpl.add(Collection<Parameter>)`` renamed ``addAll`` (R4 split for
  the two overloads of Java ``add``).
- ``Collections.unmodifiableList/Map(...)`` → plain ``list``/``dict`` copy
  (R10: Python has no read-only-view equivalent in the standard library;
  callers should not mutate returned collections).
- ``getClass().equals(pc)`` → ``type(p) == pc`` (strict class equality, not
  ``isinstance``, to match Java's behaviour).
- ``ParameterizedFitFunction`` (private Java inner class) → ``_ParameterizedFitFunction``
  (R1 private naming); uses no outer-class state and is implemented as a nested
  class of the port (no closure needed).

------------------------------------------------------------------------
Original Java source (gov.nist.microanalysis.Utility.LevenbergMarquardtParameterized)
------------------------------------------------------------------------
/**
 * <p>
 * This class is similar to LevenbergMarquardtConstrained but adds the ability
 * to tag the parameters with tokens (represented by Parameter objects)
 * containing a Constraint, a default value and a name. The bookkeeping of
 * parameters is then handled transparently and the resulting fit values can be
 * accessed via the Parameter handle. Further specializations of the Parameter
 * class can be created to associate Parameters with specific objects like
 * XRayTransition objects, XRayTransitionSet objects or other.
 * </p>
 * <p>
 * The other critical component is the Function interface. The Function
 * interface provides a mechanism for computing the fit function and its partial
 * derivatives. The <code>compute</code>, <code>derivative</code> and
 * <code>getResult</code> functions are handed a map connecting Parameter to
 * value which they they use to perform the computation.
 * </p>
 * <p>
 * Copyright: Pursuant to title 17 Section 105 of the United States Code this
 * software is not subject to copyright protection and is in the public domain
 * </p>
 * <p>
 * Institution: National Institute of Standards and Technology
 * </p>
 *
 * @author nritchie
 * @version 1.0
 */
------------------------------------------------------------------------
"""
from __future__ import annotations

import abc
from typing import Callable, Dict, Generic, Iterable, List, Optional, Set, Type, TypeVar

import numpy as np

try:
    from ._epq_compat import EPQException, F64Array, JamaMatrix
except ImportError:
    try:
        from _epq_compat import EPQException, F64Array, JamaMatrix  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility_ver2._epq_compat import EPQException, F64Array, JamaMatrix  # type: ignore

try:
    from .LevenbergMarquardt2_ver2_1_0 import LevenbergMarquardt2
except ImportError:
    try:
        from LevenbergMarquardt2_ver2_1_0 import LevenbergMarquardt2  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility_ver2.LevenbergMarquardt2_ver2_1_0 import LevenbergMarquardt2  # type: ignore

try:
    from .LevenbergMarquardtConstrained_ver2_1_0 import LevenbergMarquardtConstrained, ConstrainedFitFunction
except ImportError:
    try:
        from LevenbergMarquardtConstrained_ver2_1_0 import LevenbergMarquardtConstrained, ConstrainedFitFunction  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility_ver2.LevenbergMarquardtConstrained_ver2_1_0 import LevenbergMarquardtConstrained, ConstrainedFitFunction  # type: ignore

try:
    from .Constraint_ver2_1_3 import Constraint, Unconstrained
except ImportError:
    try:
        from Constraint_ver2_1_3 import Constraint, Unconstrained  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility_ver2.Constraint_ver2_1_3 import Constraint, Unconstrained  # type: ignore

try:
    from .UncertainValue2_ver2_1_0 import UncertainValue2
except ImportError:
    try:
        from UncertainValue2_ver2_1_0 import UncertainValue2  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility_ver2.UncertainValue2_ver2_1_0 import UncertainValue2  # type: ignore

BUG_LEDGER: tuple = ()  # no bugs identified

# TypeVar for ParameterObject's generic type parameter
T = TypeVar("T")

# FitResult inner class of LevenbergMarquardt2
_FitResult = LevenbergMarquardt2.FitResult


class LevenbergMarquardtParameterized:
    """Python port of ``gov.nist.microanalysis.Utility.LevenbergMarquardtParameterized``.

    A parameterised Levenberg-Marquardt fitter.  Parameters are tagged with
    :class:`Parameter` tokens carrying a :class:`Constraint`, a default value
    and a name.  A :class:`Function` supplies the model; results are returned
    as a :class:`ParameterizedFitResult` keyed by :class:`Parameter`.
    """

    # ------------------------------------------------------------------
    # Inner class: Parameter  (static public class)
    # ------------------------------------------------------------------

    class Parameter:
        """Fit-parameter token with a name, Constraint, default value, and isFit flag.

        Port of ``LevenbergMarquardtParameterized.Parameter`` (static inner class).

        Constructor dispatch (R4):
          ``Parameter(name, defValue, isFit)``            — uses Unconstrained()
          ``Parameter(name, constraint, defValue, isFit)`` — uses supplied constraint
        """

        def __init__(
            self,
            name: str,
            constraint_or_defvalue,
            defvalue_or_isfit,
            isfit: Optional[bool] = None,
        ) -> None:
            self._mName: str = name
            if isfit is not None:
                # 4-arg form: (name, constraint, defValue, isFit)
                self._mConstraint: Constraint = constraint_or_defvalue
                self._mDefaultValue: float = float(defvalue_or_isfit)
                self._mIsFit: bool = bool(isfit)
            else:
                # 3-arg form: (name, defValue, isFit) — Unconstrained default
                # Java: this(name, new Constraint.None(), defValue, isFit)
                self._mConstraint = Unconstrained()
                self._mDefaultValue = float(constraint_or_defvalue)
                self._mIsFit = bool(defvalue_or_isfit)

        def getConstraint(self) -> Constraint:
            """Java: ``public Constraint getConstraint()``"""
            return self._mConstraint

        def setConstraint(self, constraint: Constraint) -> None:
            """Java: ``public void setConstraint(Constraint constraint)``"""
            if self._mConstraint is not constraint:
                self._mConstraint = constraint

        def getDefaultValue(self) -> float:
            """Java: ``public double getDefaultValue()``"""
            return self._mDefaultValue

        def setDefaultValue(self, iv: float) -> None:
            """Java: ``public void setDefaultValue(double iv)``"""
            self._mDefaultValue = float(iv)

        def getValue(
            self,
            param: Dict["LevenbergMarquardtParameterized.Parameter", float],
        ) -> float:
            """Return the value from *param* when fit, else the default.

            Java: ``public double getValue(Map<Parameter,Double> param)``
            Java assert ``(!mIsFit) || param.containsKey(this)`` omitted (disabled).
            """
            return param[self] if self._mIsFit else self._mDefaultValue

        def getUncertainValue(
            self,
            param: Dict["LevenbergMarquardtParameterized.Parameter", UncertainValue2],
        ) -> UncertainValue2:
            """Return an UncertainValue2 re-labelled with this parameter's name.

            Java: ``public UncertainValue2 getUncertainValue(Map<Parameter,UncertainValue2> param)``
            Java assert ``(!mIsFit) || param.containsKey(this)`` omitted (disabled).
            """
            res: UncertainValue2 = param[self] if self._mIsFit else UncertainValue2(self._mDefaultValue)
            return UncertainValue2(res.doubleValue(), self._mName, res.uncertainty())

        def isFit(self) -> bool:
            """Java: ``public boolean isFit()``"""
            return self._mIsFit

        def setIsFit(self, b: bool) -> None:
            """Java: ``public void setIsFit(boolean b)``"""
            self._mIsFit = bool(b)

        def getName(self) -> str:
            """Java: ``public String getName()``"""
            return self._mName

        # R2: hashCode / equals / toString — dunder + named aliases

        def __hash__(self) -> int:
            # Java: prime=31; result=1; result = 31*result + mName.hashCode()
            return 31 + hash(self._mName)

        def hashCode(self) -> int:
            """Java: ``@Override public int hashCode()``"""
            return self.__hash__()

        def __eq__(self, obj: object) -> bool:
            if self is obj:
                return True
            if obj is None or type(obj) is not type(self):
                return False
            return self._mName == obj._mName  # type: ignore[union-attr]

        def equals(self, obj: object) -> bool:
            """Java: ``@Override public boolean equals(Object obj)``"""
            return self.__eq__(obj)

        def __str__(self) -> str:
            return f"{self._mName}[{self._mConstraint},{self._mDefaultValue}]"

        def toString(self) -> str:
            """Java: ``@Override public String toString()``"""
            return self.__str__()

    # ------------------------------------------------------------------
    # Inner class: ParameterObject<T>  (static public class extends Parameter)
    # ------------------------------------------------------------------

    class ParameterObject(Parameter, Generic[T]):
        """Parameter subclass that carries an associated object of type T.

        Port of ``LevenbergMarquardtParameterized.ParameterObject<T>``
        (static inner class).

        Constructor dispatch (R4):
          5-arg public:    ``(name, constraint, defValue, isFit, obj)``
          4-arg protected: ``(name, defValue, isFit, obj)``
        """

        def __init__(self, name: str, *args) -> None:
            if len(args) == 4:
                # 5-arg public: (name, constraint, defValue, isFit, obj)
                constraint, defvalue, isfit, obj = args
                super().__init__(name, constraint, float(defvalue), bool(isfit))
                self._mObject: T = obj
            elif len(args) == 3:
                # 4-arg protected: (name, defValue, isFit, obj)
                defvalue, isfit, obj = args
                super().__init__(name, float(defvalue), bool(isfit))
                self._mObject = obj
            else:
                raise TypeError(
                    f"ParameterObject() takes 4 or 5 positional arguments "
                    f"but {1 + len(args)} were given"
                )

        def getObject(self) -> T:
            """Java: ``public T getObject()``"""
            return self._mObject

    # ------------------------------------------------------------------
    # Inner interface: Function  (public interface)
    # ------------------------------------------------------------------

    class Function(abc.ABC):
        """Abstract base for parameterised fit functions.

        Port of ``LevenbergMarquardtParameterized.Function`` (public interface).
        M4: JPype cannot subclass this from Python to pass into the Java solver.
        Concrete Python subclasses via :class:`FunctionImpl` are the typical path.
        """

        @abc.abstractmethod
        def isFitParameter(
            self, idx: "LevenbergMarquardtParameterized.Parameter"
        ) -> bool:
            """Java: ``boolean isFitParameter(Parameter idx)``"""
            ...

        @abc.abstractmethod
        def getParameters(self, all: bool) -> Set["LevenbergMarquardtParameterized.Parameter"]:
            """Java: ``Set<Parameter> getParameters(boolean all)``"""
            ...

        @abc.abstractmethod
        def compute(
            self,
            arg: float,
            param: Dict["LevenbergMarquardtParameterized.Parameter", float],
        ) -> float:
            """Java: ``double compute(double arg, Map<Parameter,Double> param)``"""
            ...

        @abc.abstractmethod
        def derivative(
            self,
            arg: float,
            param: Dict["LevenbergMarquardtParameterized.Parameter", float],
            idx: "LevenbergMarquardtParameterized.Parameter",
        ) -> float:
            """Java: ``double derivative(double arg, Map<Parameter,Double> param, Parameter idx)``"""
            ...

        @abc.abstractmethod
        def computeU(
            self,
            arg: float,
            param: Dict["LevenbergMarquardtParameterized.Parameter", UncertainValue2],
        ) -> UncertainValue2:
            """Java: ``UncertainValue2 computeU(double arg, Map<Parameter,UncertainValue2> param)``"""
            ...

    # ------------------------------------------------------------------
    # Inner interface: InvertableFunction  (public interface extends Function)
    # ------------------------------------------------------------------

    class InvertableFunction(Function):
        """Extension of :class:`Function` supporting inversion.

        Port of ``LevenbergMarquardtParameterized.InvertableFunction``
        (public interface).
        """

        @abc.abstractmethod
        def inverse(
            self,
            arg: float,
            param: Dict["LevenbergMarquardtParameterized.Parameter", UncertainValue2],
        ) -> UncertainValue2:
            """Java: ``public UncertainValue2 inverse(double arg, Map<Parameter,UncertainValue2> param)``"""
            ...

    # ------------------------------------------------------------------
    # Inner class: FunctionImpl  (public static abstract class implements Function)
    # ------------------------------------------------------------------

    class FunctionImpl(Function):
        """Partial implementation of :class:`Function` with parameter bookkeeping.

        Port of ``LevenbergMarquardtParameterized.FunctionImpl``
        (``public static abstract class implements Function``).

        ``compute``, ``derivative``, and ``computeU`` remain abstract; subclasses
        implement them.  ``add``/``addAll``/``getParameters``/``isFitParameter``/
        ``extract``/``toString`` are final (Java ``final``; no Python analogue).
        """

        def __init__(self) -> None:
            self._mParameters: Set[LevenbergMarquardtParameterized.Parameter] = set()

        def add(
            self,
            p: "LevenbergMarquardtParameterized.Parameter",
        ) -> "LevenbergMarquardtParameterized.Parameter":
            """Register a single parameter; return it for chaining.

            Java: ``final public Parameter add(Parameter p)``
            """
            self._mParameters.add(p)
            return p

        def addAll(
            self,
            cp: Iterable["LevenbergMarquardtParameterized.Parameter"],
        ) -> None:
            """Register a collection of parameters.

            Java: ``final public void add(Collection<Parameter> cp)`` — R4 split.
            """
            self._mParameters.update(cp)

        def getParameters(
            self, all: bool
        ) -> Set["LevenbergMarquardtParameterized.Parameter"]:
            """Return all parameters (``all=True``) or fit-only (``all=False``).

            Java: ``@Override final public Set<Parameter> getParameters(boolean all)``
            """
            if all:
                return set(self._mParameters)
            return {p for p in self._mParameters if p.isFit()}

        def isFitParameter(
            self, p: "LevenbergMarquardtParameterized.Parameter"
        ) -> bool:
            """True iff *p* is registered and ``isFit() == True``.

            Java: ``@Override final public boolean isFitParameter(Parameter p)``
            """
            return p in self._mParameters and p.isFit()

        def extract(
            self,
            fitResult: Dict["LevenbergMarquardtParameterized.Parameter", UncertainValue2],
        ) -> Dict["LevenbergMarquardtParameterized.Parameter", UncertainValue2]:
            """Subset *fitResult* to this function's fit parameters.

            Java: ``final public Map<Parameter,UncertainValue2> extract(Map<Parameter,UncertainValue2> fitResult)``
            """
            return {p: fitResult[p] for p in self.getParameters(False)}

        def __str__(self) -> str:
            return f"[all={len(self._mParameters)},fit={len(self.getParameters(False))}]"

        def toString(self) -> str:
            """Java: ``@Override public String toString()``"""
            return self.__str__()

    # ------------------------------------------------------------------
    # Inner class: _ParameterizedFitFunction  (private — implements FitFunction)
    # ------------------------------------------------------------------

    class _ParameterizedFitFunction(LevenbergMarquardt2.FitFunction):
        """Maps a :class:`Function` into a :class:`~LevenbergMarquardt2.FitFunction`.

        Port of the private inner class ``ParameterizedFitFunction``.
        Uses no outer-class state and is implemented as an ordinary nested class.
        """

        def __init__(
            self,
            x: List[float],
            f: "LevenbergMarquardtParameterized.Function",
        ) -> None:
            self._mFunction: LevenbergMarquardtParameterized.Function = f
            self._mParameters: List[LevenbergMarquardtParameterized.Parameter] = list(
                f.getParameters(False)
            )
            self._mOrdinate: List[float] = list(x)

        def _getUpdatedParam(
            self,
            params: JamaMatrix,
        ) -> Dict["LevenbergMarquardtParameterized.Parameter", float]:
            """Build a Parameter→float map from column-vector *params*.

            Java: ``private Map<Parameter,Double> getUpdatedParam(Matrix params)``
            """
            result: Dict[LevenbergMarquardtParameterized.Parameter, float] = {}
            for i, p in enumerate(self._mParameters):
                result[p] = params.get(i, 0)
            return result

        def partials(self, params: JamaMatrix) -> JamaMatrix:
            """Jacobian: n×m matrix of partial derivatives.

            Java: ``@Override public Matrix partials(Matrix params)``
            """
            n: int = len(self._mOrdinate)
            m: int = params.getRowDimension()
            res: JamaMatrix = JamaMatrix.zeros(n, m)
            param: Dict[LevenbergMarquardtParameterized.Parameter, float] = (
                self._getUpdatedParam(params)
            )
            for j, p in enumerate(self._mParameters):
                for ch in range(n):
                    res.set(ch, j, self._mFunction.derivative(self._mOrdinate[ch], param, p))
            return res

        def compute(self, params: JamaMatrix) -> JamaMatrix:
            """Evaluate the fit function at each ordinate.

            Java: ``@Override public Matrix compute(Matrix params)``
            """
            n: int = len(self._mOrdinate)
            res: JamaMatrix = JamaMatrix.zeros(n, 1)
            param: Dict[LevenbergMarquardtParameterized.Parameter, float] = (
                self._getUpdatedParam(params)
            )
            for j in range(n):
                res.set(j, 0, self._mFunction.compute(self._mOrdinate[j], param))
            return res

        def paramSize(self) -> int:
            """Number of fit parameters.

            Java: ``public int paramSize()``
            """
            return len(self._mParameters)

    # ------------------------------------------------------------------
    # Inner class: ParameterizedFitResult  (public static class extends FitResult)
    # ------------------------------------------------------------------

    class ParameterizedFitResult(LevenbergMarquardt2.FitResult):
        """Extends :class:`~LevenbergMarquardt2.FitResult` with Parameter-keyed access.

        Port of ``LevenbergMarquardtParameterized.ParameterizedFitResult``
        (``public static class extends FitResult``).

        Java ``lm2.super(fr.mFunction)`` (qualified inner super-constructor) →
        Python ``super().__init__(lm2, fr._mFunction)`` carrying the explicit
        model reference required by the LM2 ``FitResult`` port.
        """

        def __init__(
            self,
            lm2: LevenbergMarquardt2,
            fr: LevenbergMarquardt2.FitResult,
            params: List["LevenbergMarquardtParameterized.Parameter"],
        ) -> None:
            super().__init__(lm2, fr._mFunction)
            self._mBestParams = fr._mBestParams
            self._mBestY = fr._mBestY
            self._mChiSq = fr._mChiSq
            self._mCovariance = fr._mCovariance
            self._mImproveCount = fr._mImproveCount
            self._mIterCount = fr._mIterCount
            self._mParameters: List[LevenbergMarquardtParameterized.Parameter] = list(params)

        def indexOf(
            self, p: "LevenbergMarquardtParameterized.Parameter"
        ) -> int:
            """Index of *p* in the ordered parameter list.

            Java: ``public int indexOf(Parameter p)``
            """
            return self._mParameters.index(p)

        def getBestFitValue(
            self, p: "LevenbergMarquardtParameterized.Parameter"
        ) -> UncertainValue2:
            """Best-fit UncertainValue2 for parameter *p*.

            Java: ``public UncertainValue2 getBestFitValue(Parameter p)``
            """
            return self.getBestParametersU()[self.indexOf(p)]

        def getBestFit(
            self, p: "LevenbergMarquardtParameterized.Parameter"
        ) -> float:
            """Best-fit nominal value for parameter *p*.

            Java: ``public double getBestFit(Parameter p)``
            """
            return self.getBestParametersU()[self.indexOf(p)].doubleValue()

        def getParameters(
            self,
        ) -> List["LevenbergMarquardtParameterized.Parameter"]:
            """Ordered list of fit parameters.

            Java: ``public List<Parameter> getParameters()``
            R4 note: distinct from ``Function.getParameters(boolean)`` — no conflict.
            Returns a plain list copy (R10: Java wraps in Collections.unmodifiableList).
            """
            return list(self._mParameters)

        def getParametersByClass(
            self,
            pc: Type["LevenbergMarquardtParameterized.Parameter"],
        ) -> List["LevenbergMarquardtParameterized.Parameter"]:
            """Parameters whose exact class equals *pc*.

            Java: ``public List<Parameter> getParametersByClass(Class<? extends Parameter> pc)``
            ``type(p) == pc`` matches Java ``p.getClass().equals(pc)`` (strict, not isinstance).
            Returns a plain list copy (R10: Java wraps in Collections.unmodifiableList).
            """
            return [p for p in self._mParameters if type(p) == pc]

        def getParameterByClass(
            self,
            pc: Type["LevenbergMarquardtParameterized.Parameter"],
        ) -> Optional["LevenbergMarquardtParameterized.Parameter"]:
            """First parameter whose exact class equals *pc*, or ``None``.

            Java: ``public Parameter getParameterByClass(Class<? extends Parameter> pc)``
            """
            for p in self._mParameters:
                if type(p) == pc:
                    return p
            return None

        def getParameterMap(
            self,
        ) -> Dict["LevenbergMarquardtParameterized.Parameter", UncertainValue2]:
            """Map from each parameter to its best-fit UncertainValue2.

            Java: ``public Map<Parameter,UncertainValue2> getParameterMap()``
            Returns a plain dict copy (R10: Java wraps in Collections.unmodifiableMap).
            """
            bpu: List[UncertainValue2] = self.getBestParametersU()
            return {self._mParameters[i]: bpu[i] for i in range(len(self._mParameters))}

        def getResults(
            self,
        ) -> Dict["LevenbergMarquardtParameterized.Parameter", float]:
            """Map from each parameter to its best-fit nominal value.

            Java: ``public Map<Parameter,Double> getResults()``
            Returns a plain dict copy (R10: Java wraps in Collections.unmodifiableMap).
            """
            bpu: List[UncertainValue2] = self.getBestParametersU()
            return {
                self._mParameters[i]: bpu[i].doubleValue()
                for i in range(len(self._mParameters))
            }

        def tabulate(self) -> str:
            """Tab-separated table of parameter names, defaults, fit values, and uncertainties.

            Java: ``public String tabulate()``
            """
            lines: List[str] = ["Name\tDefault\tFit\tu(Fit)"]
            for p in self._mParameters:
                bfv: UncertainValue2 = self.getBestFitValue(p)
                lines.append(
                    f"{p._mName}\t{p._mDefaultValue}"
                    f"\t{bfv.doubleValue()}\t{bfv.uncertainty()}"
                )
            return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        self._mListener: Optional[Callable[[int], None]] = None

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def compute(
        self,
        f: "LevenbergMarquardtParameterized.Function",
        xVals: List[float],
        yData: List[float],
        sigma: List[float],
    ) -> "LevenbergMarquardtParameterized.ParameterizedFitResult":
        """Fit function *f* at ordinate *xVals* to data *yData* / uncertainties *sigma*.

        Builds a ``_ParameterizedFitFunction``, wraps it in a
        ``ConstrainedFitFunction`` (setting per-parameter constraints), runs
        ``LevenbergMarquardtConstrained``, and returns a
        :class:`ParameterizedFitResult`.

        Java: ``public ParameterizedFitResult compute(Function, double[], double[], double[])``
        """
        pff: LevenbergMarquardtParameterized._ParameterizedFitFunction = (
            LevenbergMarquardtParameterized._ParameterizedFitFunction(xVals, f)
        )
        cff: ConstrainedFitFunction = ConstrainedFitFunction(pff, pff.paramSize())
        for i in range(pff.paramSize()):
            cff.setConstraint(i, pff._mParameters[i]._mConstraint)
        lmq: LevenbergMarquardtConstrained = LevenbergMarquardtConstrained()
        if self._mListener is not None:
            lmq.addActionListener(self._mListener)
        n: int = len(yData)
        yM: JamaMatrix = JamaMatrix.zeros(n, 1)
        sM: JamaMatrix = JamaMatrix.zeros(n, 1)
        for i in range(n):
            yM.set(i, 0, float(yData[i]))
            sM.set(i, 0, float(sigma[i]))
        p0: JamaMatrix = JamaMatrix.zeros(pff.paramSize(), 1)
        for i in range(pff.paramSize()):
            p0.set(i, 0, pff._mParameters[i]._mDefaultValue)
        raw: LevenbergMarquardt2.FitResult = lmq.compute(cff, yM, sM, p0)
        return LevenbergMarquardtParameterized.ParameterizedFitResult(
            lmq, raw, pff._mParameters
        )

    def addActionListener(self, al: Callable[[int], None]) -> None:
        """Register a progress callback.

        Java: ``public void addActionListener(ActionListener al)``
        R10 deviation: ``ActionListener`` replaced by ``Callable[[int], None]``
        (same strategy as ``LevenbergMarquardt2``).
        """
        self._mListener = al


# Module-level aliases (R2)
Parameter = LevenbergMarquardtParameterized.Parameter
ParameterObject = LevenbergMarquardtParameterized.ParameterObject
Function = LevenbergMarquardtParameterized.Function
InvertableFunction = LevenbergMarquardtParameterized.InvertableFunction
FunctionImpl = LevenbergMarquardtParameterized.FunctionImpl
ParameterizedFitResult = LevenbergMarquardtParameterized.ParameterizedFitResult
