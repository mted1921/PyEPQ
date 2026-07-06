r"""
LevenbergMarquardtConstrained_ver2_1_0.py — Python port of
gov.nist.microanalysis.Utility.LevenbergMarquardtConstrained

Guide version : 2
Generation    : 1
Port-code fixes: 0

CHANGES
-------
- Extends LevenbergMarquardt2_ver2_1_0 (guide-version-2 sibling port).
- Java `Constraint.None` inner class is `Constraint.Unconstrained` in the port
  (R1/R10: `None` is a Python builtin; the Constraint port already renamed it).
- `ConstrainedFitFunction.realToConstrained` overload split (R4):
    realToConstrained(Matrix)  → realToConstrained_matrix
    realToConstrained(FitResult) → realToConstrained_result
- `constrainedToReal(Matrix)` → constrainedToReal_matrix (suffix for symmetry).
- `FitResult` inner-class instantiation: Java `fr.getModel().new FitResult(ff)`
  becomes `FitResult(fr.getModel(), ff)` using the LM2 port's explicit `model`
  parameter.
- Java `assert` statements in `realToConstrained_matrix` / `constrainedToReal_matrix`
  (`rParams.getColumnDimension() == 1`) dropped (disabled-by-default Java asserts).
- Reverse loops `for (i = n-1; i >= 0; --i)` preserved as `range(n-1, -1, -1)`
  for fidelity; loop order is immaterial for these independent per-element ops.
- Module-level alias `ConstrainedFitFunction` exported (R2).

------------------------------------------------------------------------
Original Java source (gov.nist.microanalysis.Utility.LevenbergMarquardtConstrained)
------------------------------------------------------------------------
/**
 * <p>
 * This class makes it realatively simple to perform constrained least squares
 * fitting. You implement the ConstrainedFitFunction class and specify the
 * constraints. The
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

from typing import List

try:
    from ._epq_compat import EPQException, F64Array, JamaMatrix, JavaRandom
except ImportError:
    try:
        from _epq_compat import EPQException, F64Array, JamaMatrix, JavaRandom  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility._epq_compat import EPQException, F64Array, JamaMatrix, JavaRandom  # type: ignore

try:
    from .LevenbergMarquardt2_ver2_1_0 import LevenbergMarquardt2, FitResult
except ImportError:
    try:
        from LevenbergMarquardt2_ver2_1_0 import LevenbergMarquardt2, FitResult  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility_ver2.LevenbergMarquardt2_ver2_1_0 import LevenbergMarquardt2, FitResult  # type: ignore

try:
    from .Constraint_ver2_1_3 import Constraint, _Unconstrained
except ImportError:
    try:
        from Constraint_ver2_1_3 import Constraint, _Unconstrained  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility_ver2.Constraint_ver2_1_3 import Constraint, _Unconstrained  # type: ignore

try:
    from .UncertainValue2_ver2_1_0 import UncertainValue2
except ImportError:
    try:
        from UncertainValue2_ver2_1_0 import UncertainValue2  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility_ver2.UncertainValue2_ver2_1_0 import UncertainValue2  # type: ignore

BUG_LEDGER: tuple = ()  # no bugs identified


class LevenbergMarquardtConstrained(LevenbergMarquardt2):
    """Python port of ``gov.nist.microanalysis.Utility.LevenbergMarquardtConstrained``.

    Extends :class:`LevenbergMarquardt2` with constrained-parameter support via
    :class:`ConstrainedFitFunction`.  Parameters are mapped from the full real
    line through :class:`Constraint` objects before being passed to the user
    fit function.
    """

    # ------------------------------------------------------------------
    # Inner class: ConstrainedFitFunction
    # ------------------------------------------------------------------

    class ConstrainedFitFunction(LevenbergMarquardt2.FitFunction):
        """Port of the Java static inner class ``ConstrainedFitFunction``.

        Wraps a domain-space :class:`~LevenbergMarquardt2.FitFunction` with
        per-parameter :class:`Constraint` objects that map the full real line
        onto the constrained domain.

        Overloaded Java ``realToConstrained`` is split by argument type (R4):
          ``realToConstrained(Matrix)``   → :meth:`realToConstrained_matrix`
          ``realToConstrained(FitResult)``→ :meth:`realToConstrained_result`
        """

        def __init__(self, ff: LevenbergMarquardt2.FitFunction, paramDim: int) -> None:
            # Fills _mConstraints with Constraint.Unconstrained() × paramDim.
            # Java `new Constraint.None()` → port `_Unconstrained()` (R1/R10).
            self._mConstraints: List[Constraint] = [
                _Unconstrained() for _ in range(paramDim)
            ]
            self._mFitFunction: LevenbergMarquardt2.FitFunction = ff

        def setConstraint(self, paramIdx: int, c: Constraint) -> None:
            """Specify the constraint for the ``paramIdx``-th parameter."""
            self._mConstraints[paramIdx] = c

        def realToConstrained_matrix(self, rParams: JamaMatrix) -> JamaMatrix:
            """Map each element of *rParams* from ℝ → constrained domain.

            Java: ``public Matrix realToConstrained(Matrix rParams)``
            R4 split: takes a Matrix argument.
            Java assert ``rParams.getColumnDimension() == 1`` dropped.
            """
            n: int = rParams.getRowDimension()
            params: JamaMatrix = JamaMatrix.zeros(n, 1)
            for i in range(n - 1, -1, -1):
                params.set(
                    i, 0,
                    self._mConstraints[i].realToConstrained(rParams.get(i, 0)),
                )
            return params

        def constrainedToReal_matrix(self, rParams: JamaMatrix) -> JamaMatrix:
            """Invert the per-element constraint map: constrained → ℝ.

            Java: ``public Matrix constrainedToReal(Matrix rParams)``
            R4 split: suffix ``_matrix`` kept for symmetry with
            ``realToConstrained_matrix``.
            Java assert ``rParams.getColumnDimension() == 1`` dropped.
            """
            n: int = rParams.getRowDimension()
            params: JamaMatrix = JamaMatrix.zeros(n, 1)
            for i in range(n - 1, -1, -1):
                params.set(
                    i, 0,
                    self._mConstraints[i].constrainedToReal(rParams.get(i, 0)),
                )
            return params

        def partials(self, rParams: JamaMatrix) -> JamaMatrix:
            """Jacobian in real space, chain-ruled through the constraint derivative.

            Java: ``@Override public Matrix partials(Matrix rParams)``
            Calls the wrapped function's partials at the constrained point, then
            scales each column by the constraint derivative w.r.t. the real parameter.
            Java assert ``tmp.getColumnDimension() == rParams.getRowDimension()`` dropped.
            """
            tmp: JamaMatrix = self._mFitFunction.partials(
                self.realToConstrained_matrix(rParams)
            )
            for c in range(tmp.getColumnDimension()):
                dp: float = self._mConstraints[c].derivative(rParams.get(c, 0))
                for r in range(tmp.getRowDimension()):
                    tmp.set(r, c, tmp.get(r, c) * dp)
            return tmp

        def compute(self, rParams: JamaMatrix) -> JamaMatrix:
            """Evaluate the wrapped function at the constrained parameters.

            Java: ``@Override public Matrix compute(Matrix rParams)``
            """
            return self._mFitFunction.compute(
                self.realToConstrained_matrix(rParams)
            )

        def realToConstrained_result(self, fr: FitResult) -> FitResult:
            """Transform a FitResult from ℝ → constrained domain.

            Java: ``public FitResult realToConstrained(FitResult fr)``
            R4 split: takes a FitResult argument.

            Builds a new FitResult (bound to ``fr.getModel()``) with:
            - ``_mBestParams`` propagated through each constraint's ``getResult``
            - ``_mCovariance`` scaled by the chain-rule factor dp_i * dp_j
            - all other fields copied verbatim
            """
            res: FitResult = FitResult(fr.getModel(), self._mFitFunction)
            n: int = len(self._mConstraints)
            res._mBestParams = [
                self._mConstraints[i].getResult(fr._mBestParams[i])
                for i in range(n)
            ]
            res._mBestY = list(fr._mBestY)  # clone
            res._mChiSq = fr._mChiSq
            # Covariance chain rule: covar[r,c] *= dp[r] * dp[c]
            covar: JamaMatrix = JamaMatrix(fr._mCovariance.getArrayCopy())
            dp: List[float] = [
                self._mConstraints[i].derivative(fr._mBestParams[i].doubleValue())
                for i in range(n)
            ]
            for c in range(covar.getColumnDimension()):
                for r in range(covar.getRowDimension()):
                    covar.set(r, c, fr._mCovariance.get(r, c) * dp[r] * dp[c])
            res._mCovariance = covar
            res._mImproveCount = fr._mImproveCount
            res._mIterCount = fr._mIterCount
            return res

    # ------------------------------------------------------------------
    # LevenbergMarquardtConstrained constructor
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        super().__init__()

    # ------------------------------------------------------------------
    # compute override
    # ------------------------------------------------------------------

    def compute(
        self,
        ff: LevenbergMarquardt2.FitFunction,
        yData: JamaMatrix,
        sigma: JamaMatrix,
        p0: JamaMatrix,
    ) -> FitResult:
        """Fit with optional constraint mapping.

        If *ff* is a :class:`ConstrainedFitFunction`, the start point *p0*
        (in constrained space) is converted to real space before passing to
        the parent solver, and the resulting :class:`FitResult` is mapped
        back to the constrained domain.  Any other :class:`FitFunction` is
        passed straight through to the parent.

        Java: ``@Override public FitResult compute(...)``
        """
        if isinstance(ff, LevenbergMarquardtConstrained.ConstrainedFitFunction):
            cff: LevenbergMarquardtConstrained.ConstrainedFitFunction = ff
            tmp: FitResult = super().compute(
                cff, yData, sigma, cff.constrainedToReal_matrix(p0)
            )
            return cff.realToConstrained_result(tmp)
        else:
            return super().compute(ff, yData, sigma, p0)


# Module-level aliases (R2)
ConstrainedFitFunction = LevenbergMarquardtConstrained.ConstrainedFitFunction
