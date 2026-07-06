r"""
LevenbergMarquardt2_ver2_1_0.py — Python port of
gov.nist.microanalysis.Utility.LevenbergMarquardt2

Guide version : 2
Generation    : 1
Port-code fixes: 0

CHANGES
-------
- `java.awt.event.ActionListener` / `ActionEvent` replaced by `Callable[[int], None]` (R10):
  there is no Python GUI event model; callers receive the percent-complete integer.
- `Jama.SingularValueDecomposition` replaced by `numpy.linalg.svd` (SCIPY-DEV-1):
  Jama's SVD is not wrapped by JamaMatrix; numpy SVD is the faithful equivalent.
  Jama `V` = numpy `Vt.T`.  No `_literal` twin — Java itself uses Jama's SVD.
- Java `assert` statements dropped (disabled by default in JVM, per Cross-Cutting rules).
- `FitResult` is a non-static Java inner class; ported with an explicit `model` parameter
  so `getModel()` can return the enclosing `LevenbergMarquardt2` instance.
- Inner classes (FitFunction, AutoPartialsFitFunction, FitResult) are defined inside the
  outer class body where inheritance permits; module-level aliases expose them (R2).

------------------------------------------------------------------------
Original Java source (gov.nist.microanalysis.Utility.LevenbergMarquardt2)
------------------------------------------------------------------------
/**
 * Fit m parameters to n data items using the non-linear Levenberg-Marquardt
 * iterative algorithm. Users must implement the FitFunction interface to
 * compute the partial derivatives (Jacobian) and the fit function.
 */
------------------------------------------------------------------------
"""
from __future__ import annotations

import abc
import math
from typing import Callable, List, Optional, Sequence

import numpy as np

try:
    from ._epq_compat import EPQException, F64Array, JamaMatrix, JavaRandom
except ImportError:
    try:
        from _epq_compat import EPQException, F64Array, JamaMatrix, JavaRandom  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility._epq_compat import EPQException, F64Array, JamaMatrix, JavaRandom  # type: ignore

try:
    from .UncertainValue2_ver2_1_0 import UncertainValue2
except ImportError:
    try:
        from UncertainValue2_ver2_1_0 import UncertainValue2  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility_ver2.UncertainValue2_ver2_1_0 import UncertainValue2  # type: ignore

try:
    from .Math2_ver8_1_5 import Math2
except ImportError:
    try:
        from Math2_ver8_1_5 import Math2  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility_ver2.Math2_ver8_1_5 import Math2  # type: ignore

BUG_LEDGER: tuple = (
    (
        "SCIPY-DEV-1",
        "_solve / compute",
        "Java uses `Jama.SingularValueDecomposition` via `a.svd()`.  Jama's SVD is not "
        "wrapped by JamaMatrix in _epq_compat.  Replaced by `numpy.linalg.svd` with "
        "`full_matrices=False`.  Jama's `V` matrix corresponds to numpy's `Vt.T`.  "
        "The pseudo-inverse of each singular value is computed identically to the Java "
        "`updateSingularValues` logic (invert if > 1e-10*max, else 0).  No `_literal` "
        "twin is provided — Java itself delegates this computation to Jama's native SVD.",
    ),
)


class LevenbergMarquardt2:
    """Python port of ``gov.nist.microanalysis.Utility.LevenbergMarquardt2``.

    Non-linear Levenberg–Marquardt least-squares fitting engine.  Callers
    implement :class:`FitFunction` and pass it to :meth:`compute`.
    """

    # ------------------------------------------------------------------
    # Inner interface: FitFunction
    # ------------------------------------------------------------------

    class FitFunction(abc.ABC):
        """Port of the Java ``FitFunction`` interface (inner interface of LM2)."""

        @abc.abstractmethod
        def partials(self, params: JamaMatrix) -> JamaMatrix:
            """Jacobian: n×m matrix of partial derivatives at *params* (m×1)."""

        @abc.abstractmethod
        def compute(self, params: JamaMatrix) -> JamaMatrix:
            """Fit function: n×1 column vector of predicted values at *params* (m×1)."""

    # ------------------------------------------------------------------
    # Inner abstract class: AutoPartialsFitFunction
    # ------------------------------------------------------------------

    class AutoPartialsFitFunction(FitFunction):
        """Estimates the Jacobian via finite differences (N_PARAMS+1 `compute` calls).

        Port of the Java ``static abstract class AutoPartialsFitFunction implements FitFunction``.
        Subclasses must implement :meth:`compute`; :meth:`partials` is provided here.
        """

        def __init__(self, delta: float = 1.0e-8) -> None:
            self._DELTA: float = float(delta)
            self._mDelta: Optional[List[float]] = None

        def partials(self, params: JamaMatrix) -> JamaMatrix:
            if self._mDelta is None:
                n: int = params.getRowDimension()
                self._mDelta = [self._DELTA] * n
            c: JamaMatrix = self.compute(params)
            res: JamaMatrix = JamaMatrix.zeros(c.getRowDimension(), params.getRowDimension())
            offset: JamaMatrix = JamaMatrix(params.getArrayCopy())
            for p in range(params.getRowDimension() - 1, -1, -1):
                v1: float = params.get(p, 0)
                v2: float = (self._mDelta[p] if v1 == 0.0 else v1 * (1.0 + self._mDelta[p]))
                offset.set(p, 0, v2)
                c2: JamaMatrix = self.compute(offset)
                for ch in range(c.getRowDimension() - 1, -1, -1):
                    res.set(ch, p, (c2.get(ch, 0) - c.get(ch, 0)) / (v2 - v1))
                offset.set(p, 0, v1)
            return res

        def setDelta(self, delta: Sequence[float]) -> None:
            self._mDelta = list(delta)  # clone

    # ------------------------------------------------------------------
    # Inner class: FitResult  (non-static — holds explicit model reference)
    # ------------------------------------------------------------------

    class FitResult:
        """Container for a completed LM fit.

        Port of the Java non-static inner class ``FitResult``.  In Python,
        the enclosing-instance reference is carried explicitly as *model*.
        """

        def __init__(
            self,
            model: "LevenbergMarquardt2",
            ff: "LevenbergMarquardt2.FitFunction",
        ) -> None:
            self._model: LevenbergMarquardt2 = model
            self._mFunction: LevenbergMarquardt2.FitFunction = ff
            self._mChiSq: float = 0.0
            self._mBestParams: List[UncertainValue2] = []
            self._mBestY: List[float] = []
            self._mCovariance: Optional[JamaMatrix] = None
            self._mIterCount: int = 0
            self._mImproveCount: int = 0

        def getBestParametersU(self) -> List[UncertainValue2]:
            return self._mBestParams

        def getBestParameters(self) -> List[float]:
            return [uv.doubleValue() for uv in self._mBestParams]

        def getBestFitValues(self) -> List[float]:
            return self._mBestY

        def getChiSquared(self) -> float:
            return self._mChiSq

        def getCovariance(self) -> Optional[JamaMatrix]:
            return self._mCovariance

        def getIterationCount(self) -> int:
            return self._mIterCount

        def getImproveCount(self) -> int:
            return self._mImproveCount

        def getModel(self) -> "LevenbergMarquardt2":
            return self._model

        def getFitFunction(self) -> "LevenbergMarquardt2.FitFunction":
            return self._mFunction

    # ------------------------------------------------------------------
    # LevenbergMarquardt2 constructor
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        self._mEps1: float = 1.0e-15
        self._mEps2: float = 1.0e-6
        self._mEps3: float = 1.0e-15
        self._mTau: float = 1.0e-3
        self._mKMax: int = 100
        self._mIteration: int = 0
        self._mListeners: List[Callable[[int], None]] = []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _jTj(self, j: JamaMatrix, sigma: JamaMatrix) -> JamaMatrix:
        """Symmetric J^T J weighted by 1/σ²; Java: ``jTj(j, sigma)``."""
        n: int = j.getRowDimension()
        m: int = j.getColumnDimension()
        a: JamaMatrix = JamaMatrix.zeros(m, m)
        for im1 in range(m):
            for im2 in range(im1, m):
                v: float = 0.0
                for in_ in range(n):
                    v += (j.get(in_, im1) * j.get(in_, im2)) / Math2.sqr(sigma.get(in_, 0))
                a.set(im1, im2, v)
                a.set(im2, im1, v)
        return a

    def _g(self, j: JamaMatrix, eps: JamaMatrix, sigma: JamaMatrix) -> JamaMatrix:
        """J^T * eps / sigma; Java: ``g(j, eps, sigma)``."""
        n: int = j.getRowDimension()
        m: int = j.getColumnDimension()
        g: JamaMatrix = JamaMatrix.zeros(m, 1)
        for i in range(m):
            v: float = 0.0
            for k in range(n):
                v += (j.get(k, i) * eps.get(k, 0)) / sigma.get(k, 0)
            g.set(i, 0, v)
        return g

    def _eps(self, yData: JamaMatrix, fp: JamaMatrix, sigma: JamaMatrix) -> JamaMatrix:
        """Weighted residual; Java: ``eps(yData, fp, sigma)``."""
        n: int = yData.getRowDimension()
        res: JamaMatrix = JamaMatrix.zeros(n, 1)
        for i in range(n):
            res.set(i, 0, (yData.get(i, 0) - fp.get(i, 0)) / sigma.get(i, 0))
        return res

    def _updateSingularValues(self, s: np.ndarray) -> np.ndarray:
        """Invert singular values above 1e-10*max; zero the rest.

        Java operated on the full diagonal Jama Matrix ``getS()``; here *s*
        is the 1-D numpy singular-value array from ``numpy.linalg.svd``.
        """
        max_s: float = float(s[0]) if len(s) > 0 else 0.0
        for i in range(1, len(s)):
            if s[i] > max_s:
                max_s = s[i]
        result: np.ndarray = np.empty_like(s)
        for i in range(len(s)):
            result[i] = 1.0 / s[i] if s[i] > 1.0e-10 * max_s else 0.0
        return result

    def _solve(self, a: JamaMatrix, mu: float, g: JamaMatrix) -> JamaMatrix:
        """Solve ``(A + mu·I)·deltaP = g`` via SVD pseudo-inverse.

        Mutates *a*'s diagonal in-place (mirrors Java).  Java uses
        ``a.svd()``; Python uses ``numpy.linalg.svd`` (SCIPY-DEV-1).
        """
        if math.isnan(mu):
            raise EPQException("mu is NaN in LevenbergMarquardt2.solve(a,mu,g)")
        for i in range(a.getRowDimension()):
            if math.isnan(a.get(i, i)):
                raise EPQException(
                    f"a({i},{i}) is NaN in LevenbergMarquardt2.solve(a,mu,g)"
                )
            a.set(i, i, a.get(i, i) + mu)
        # SCIPY-DEV-1: Jama SVD → numpy.linalg.svd
        U_arr, s, Vt_arr = np.linalg.svd(a.getArray(), full_matrices=False)
        w: np.ndarray = self._updateSingularValues(s)
        V: JamaMatrix = JamaMatrix(Vt_arr.T)        # Jama V = numpy Vt.T
        W: JamaMatrix = JamaMatrix(np.diag(w))
        U_mat: JamaMatrix = JamaMatrix(U_arr)
        return V.times(W).times(U_mat.transpose().times(g))

    def _maxDiagonal(self, a: JamaMatrix) -> float:
        """Return the maximum diagonal element of *a*."""
        max_val: float = a.get(0, 0)
        for i in range(1, a.getRowDimension()):
            if a.get(i, i) > max_val:
                max_val = a.get(i, i)
        return max_val

    def _chiSqr(self, eps: JamaMatrix) -> float:
        """Sum of squared residuals."""
        chiSq: float = 0.0
        for r in range(eps.getRowDimension()):
            chiSq += Math2.sqr(eps.get(r, 0))
        return chiSq

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def getIteration(self) -> int:
        return self._mIteration

    def addActionListener(self, al: Callable[[int], None]) -> None:
        """Register a progress callback.  Called each iteration with percent complete.

        R10 deviation: Java ``ActionListener.actionPerformed(ActionEvent)`` replaced by
        a ``Callable[[int], None]`` receiving ``(100 * iteration) // maxIterations``.
        """
        self._mListeners.append(al)

    def compute(
        self,
        ff: "LevenbergMarquardt2.FitFunction",
        yData: JamaMatrix,
        sigma: JamaMatrix,
        p0: JamaMatrix,
    ) -> "LevenbergMarquardt2.FitResult":
        """Fit parameters by Levenberg–Marquardt iteration.

        Parameters
        ----------
        ff     : FitFunction — provides ``compute`` and ``partials``
        yData  : n×1 matrix of observed data values
        sigma  : n×1 matrix of per-point error estimates (all > 0)
        p0     : m×1 matrix of initial parameter guesses

        Returns
        -------
        FitResult with ``getBestParameters()``, ``getChiSquared()``, ``getCovariance()``
        """
        m: int = p0.getRowDimension()
        n: int = yData.getRowDimension()
        # Java asserts (disabled by default) are dropped per conversion rules:
        #   assert n > m; assert yData.getColumnDimension()==1; etc.
        nu: float = 2.0
        p: JamaMatrix = p0
        j: JamaMatrix = ff.partials(p)
        a: JamaMatrix = self._jTj(j, sigma)
        fp: JamaMatrix = ff.compute(p)
        epsP: JamaMatrix = self._eps(yData, fp, sigma)
        chiSq: float = self._chiSqr(epsP)
        g: JamaMatrix = self._g(j, epsP, sigma)
        mu: float = self._mTau * self._maxDiagonal(a)
        self._mIteration = 0
        improveCx: int = 0
        stop1: bool = g.normInf() < self._mEps1
        stop2: bool = False
        stop3: bool = False
        while (not (stop1 or stop2 or stop3)) and (self._mIteration < self._mKMax):
            self._mIteration += 1
            rho: float = -1.0
            while True:
                deltaP: JamaMatrix = self._solve(a, mu, g)
                stop2 = deltaP.normF() <= (self._mEps2 * p.normF())
                if not (stop1 or stop2 or stop3):
                    pNew: JamaMatrix = p.plus(deltaP)
                    fp = ff.compute(pNew)
                    epsPnew: JamaMatrix = self._eps(yData, fp, sigma)
                    den: JamaMatrix = deltaP.transpose().times(deltaP.times(mu).plus(g))
                    newChiSq: float = self._chiSqr(epsPnew)
                    rho = (chiSq - newChiSq) / den.get(0, 0)
                    if rho > 0.0:
                        improveCx += 1
                        p = pNew
                        j = ff.partials(p)
                        a = self._jTj(j, sigma)
                        epsP = epsPnew
                        chiSq = newChiSq
                        g = self._g(j, epsP, sigma)
                        stop1 = g.normInf() <= self._mEps1
                        stop3 = self._chiSqr(epsP) <= self._mEps3
                        mu = self._mTau * self._maxDiagonal(a)
                        mu *= max(1.0 / 3.0, 1.0 - math.pow((2.0 * rho) - 1.0, 3.0))
                        nu = 2.0
                    elif rho == 0.0:
                        stop1 = True
                    else:
                        mu *= nu
                        nu *= 2.0
                for listener in reversed(self._mListeners):
                    listener((100 * self._mIteration) // self._mKMax)
                if not (rho < 0.0 and not (stop1 or stop2 or stop3)):
                    break

        res: LevenbergMarquardt2.FitResult = LevenbergMarquardt2.FitResult(self, ff)
        j = ff.partials(p)
        a = self._jTj(j, sigma)
        # SCIPY-DEV-1: Jama SVD → numpy.linalg.svd for covariance
        U_arr, s, Vt_arr = np.linalg.svd(a.getArray(), full_matrices=False)
        w_inv: np.ndarray = self._updateSingularValues(s)
        V: JamaMatrix = JamaMatrix(Vt_arr.T)
        W: JamaMatrix = JamaMatrix(np.diag(w_inv))
        U_mat: JamaMatrix = JamaMatrix(U_arr)
        covar: JamaMatrix = V.times(W).times(U_mat.transpose())
        res._mBestY = ff.compute(p).getArray().flatten(order="F").tolist()
        res._mChiSq = chiSq
        res._mIterCount = self._mIteration
        res._mImproveCount = improveCx
        res._mCovariance = covar
        res._mBestParams = []
        for i in range(p.getRowDimension()):
            c: float = covar.get(i, i)
            res._mBestParams.append(
                UncertainValue2(p.get(i, 0), "LM", math.sqrt(math.fabs(c)))
            )
        return res

    def getMaxIterations(self) -> int:
        return self._mKMax

    def setMaxIterations(self, max: int) -> None:
        self._mKMax = int(max)


# Module-level aliases (R2)
FitFunction = LevenbergMarquardt2.FitFunction
AutoPartialsFitFunction = LevenbergMarquardt2.AutoPartialsFitFunction
FitResult = LevenbergMarquardt2.FitResult
