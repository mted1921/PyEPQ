r"""
Simplex_ver1_1_0.py — Python port of gov.nist.microanalysis.Utility.Simplex

Guide version : 1
Generation    : 1
Port-code fixes: 0

------------------------------------------------------------------------
Original Java source (gov.nist.microanalysis.Utility.Simplex)
------------------------------------------------------------------------
/**
 * <p>
 * The Simplex algorithm is a method for minimizing non-linear functions (Use
 * LinearLeastSquares for linear optimization). In an N dimensional space, the
 * Simplex starts with N+1 user specified points and wanders these points around
 * the space looking to minimize 'function'. The Simplex algorithm stops when
 * all points are within the specified tolerance of a local minimum value.
 * </p>
 * <p>
 * Derive a class from Simplex and implement the method function(double[] args).
 * Simplex.perform will adjust the input parameters function to produce the
 * smallest possible return value.
 * </p>
 * <p>
 * Copyright: Pursuant to title 17 Section 105 of the United States Code this
 * software is not subject to copyright protection and is in the public domain
 * </p>
 * <p>
 * Company: National Institute of Standards and Technology
 * </p>
 *
 * @author Nicholas W. M. Ritchie
 * @version 1.0
 */
------------------------------------------------------------------------
"""

from __future__ import annotations

import abc
import math
import time
from typing import Any, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike
import scipy.optimize as _optimize

try:
    from ._epq_compat import EPQException, JavaRandom, F64Array
except ImportError:
    try:
        from _epq_compat import EPQException, JavaRandom, F64Array  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility._epq_compat import EPQException, JavaRandom, F64Array  # type: ignore

__all__ = ["Simplex"]


class Simplex(abc.ABC):

    BUG_LEDGER: tuple = (
        ("RNG-DEVIATION-1", "randomizedStartingPoints",
         "Java creates a new Random(System.currentTimeMillis()) each call. "
         "Python uses JavaRandom initialized with time.time_ns() unless a seed is provided. "
         "Matched seeds produce identical distributions, but Java's unseeded behaviour diverges.", False),
        ("SCIPY-DEV-1", "perform",
         "scipy.optimize.minimize does not mutate the initial simplex in-place with the final simplex states, "
         "only saving the best result to p[0]. Use perform_literal for exact Java mutation semantics.", False),
    )

    DEFAULT_TOLERANCE: float = 1.0e-8
    DEFAULT_EVALUATIONS: int = 5000
    _LARGEST_TOLERANCE: float = 0.01
    _MIN_EVALUATIONS: int = 100

    @staticmethod
    def _require_mutable_f64(arr: np.ndarray, name: str = "arr") -> None:
        """Type guard for in-place methods (CONVERSION_GUIDE R5)."""
        if not isinstance(arr, np.ndarray):
            raise TypeError(
                f"{name} must be a numpy ndarray (got {type(arr).__name__}); "
                "in-place helpers cannot mutate Python lists or tuples.")
        if arr.dtype != np.float64:
            raise TypeError(
                f"{name} must have dtype float64 (got {arr.dtype}); "
                "wrong-dtype arrays would silently copy.")
        if not arr.flags.writeable:
            raise TypeError(f"{name} must be writeable")

    def __init__(self, params: Optional[Sequence[Any]] = None) -> None:
        self.mPTry: Optional[F64Array] = None
        self.mTolerance: float = self.DEFAULT_TOLERANCE
        self.mMaxEvaluations: int = self.DEFAULT_EVALUATIONS
        self.mBest: Optional[F64Array] = None
        self.mResult: float = float('nan')
        self.mParameters: Optional[list[Any]] = list(params) if params is not None else None
        self.mEvaluationCount: int = 0

    def _ooze_literal(self, p: F64Array, y: F64Array, psum: F64Array, ihi: int, fac: float) -> float:
        nDim: int = p.shape[1]
        if self.mPTry is None or self.mPTry.shape[0] != nDim:
            self.mPTry = np.zeros(nDim, dtype=np.float64)
            
        fac1: float = (1.0 - fac) / nDim
        fac2: float = fac1 - fac
        
        for j in range(nDim):
            self.mPTry[j] = (psum[j] * fac1) - (p[ihi, j] * fac2)
            
        ytry: float = self._evaluateFunction_literal(self.mPTry)
        if ytry < y[ihi]:
            y[ihi] = ytry
            for j in range(nDim):
                psum[j] += self.mPTry[j] - p[ihi, j]
                p[ihi, j] = self.mPTry[j]
        return ytry

    def _evaluateFunction_literal(self, x: F64Array) -> float:
        res: float = float(self.function(x))
        if math.isnan(res):
            raise EPQException(f"The function in the Simplex routine returned NaN at {x}")
        if math.isinf(res):
            raise EPQException(f"The function in the Simplex routine is not finite at {x}")
        return res

    def getParameters(self) -> Optional[list[Any]]:
        return self.mParameters

    @abc.abstractmethod
    def function(self, x: F64Array) -> float:
        pass

    @staticmethod
    def randomizedStartingPoints(center: ArrayLike, scale: ArrayLike, seed: Optional[int] = None) -> F64Array:
        c = np.asarray(center, dtype=np.float64)
        s = np.asarray(scale, dtype=np.float64)
        assert c.shape[0] == s.shape[0]
        res = np.zeros((c.shape[0] + 1, c.shape[0]), dtype=np.float64)
        res[0, :] = c
        
        # RNG-DEVIATION-1
        r = JavaRandom(seed if seed is not None else time.time_ns())
        for i in range(1, res.shape[0]):
            for j in range(res.shape[1]):
                res[i, j] = c[j] + (s[j] * (1.0 - (2.0 * r.nextDouble())))
        return res

    @staticmethod
    def regularizedStartingPoints(center: ArrayLike, scale: ArrayLike) -> F64Array:
        c = np.asarray(center, dtype=np.float64)
        s = np.asarray(scale, dtype=np.float64)
        assert c.shape[0] == s.shape[0]
        res = np.zeros((c.shape[0] + 1, c.shape[0]), dtype=np.float64)
        for i in range(res.shape[0]):
            res[i, :] = c
        for i in range(1, res.shape[0]):
            res[i, i - 1] += s[i - 1]
        return res

    def perform_literal(self, p: F64Array) -> F64Array:
        self._require_mutable_f64(p, "p")
        nDim: int = p.shape[1]
        mpts: int = nDim + 1
        self.mEvaluationCount = nDim
        assert p.shape[0] == mpts
        
        psum = np.zeros(nDim, dtype=np.float64)
        y = np.zeros(mpts, dtype=np.float64)
        
        for i in range(mpts):
            y[i] = self._evaluateFunction_literal(p[i])
            
        for j in range(nDim):
            sum_val = 0.0
            for i in range(mpts):
                sum_val += p[i, j]
            psum[j] = sum_val
            
        while True:
            ilo: int = 0
            ihi: int = 0 if y[0] > y[1] else 1
            inhi: int = 1 - ihi
            
            for i in range(mpts):
                if y[i] <= y[ilo]:
                    ilo = i
                if y[i] > y[ihi]:
                    inhi = ihi
                    ihi = i
                elif (y[i] > y[inhi]) and (i != ihi):
                    inhi = i
                    
            tol: float = abs(y[ihi] - y[ilo])
            denom: float = abs(y[ihi]) + abs(y[ilo])
            rtol: float = (2.0 * tol) / denom if denom != 0.0 else float('nan')
            
            if (not math.isnan(rtol) and rtol < self.mTolerance) or (tol < self.mTolerance):
                self.mBest = p[ilo].copy()
                self.mResult = y[ilo]
                break
                
            if self.mEvaluationCount > self.mMaxEvaluations:
                self.mBest = p[ilo].copy()
                self.mResult = y[ilo]
                raise EPQException("Exceeded the maximum number of iterations in Simplex algorithm.")
                
            self.mEvaluationCount += 2
            yTry: float = self._ooze_literal(p, y, psum, ihi, -1.0)
            if yTry <= y[ilo]:
                yTry = self._ooze_literal(p, y, psum, ihi, 2.0)
            elif yTry >= y[inhi]:
                ySave: float = y[ihi]
                yTry = self._ooze_literal(p, y, psum, ihi, 0.5)
                if yTry >= ySave:
                    for i in range(mpts):
                        if i != ilo:
                            for j in range(nDim):
                                psum[j] = 0.5 * (p[i, j] + p[ilo, j])
                                p[i, j] = psum[j]
                            y[i] = self._evaluateFunction_literal(psum)
                    self.mEvaluationCount += nDim
                    for j in range(nDim):
                        sum_val = 0.0
                        for i in range(mpts):
                            sum_val += p[i, j]
                        psum[j] = sum_val
            else:
                self.mEvaluationCount -= 1
                
        assert self.mBest is not None
        return self.mBest

    def perform(self, p: F64Array) -> F64Array:
        """
        Primary implementation using scipy.optimize.minimize.
        
        SCIPY-DEV-1: scipy.optimize.minimize does not mutate the initial 
        simplex in-place in the exact same way as the literal Java port. It also 
        evaluates stopping criteria differently. The literal port is exact.
        """
        self._require_mutable_f64(p, "p")
        
        def wrapped_fun(x: np.ndarray) -> float:
            return self._evaluateFunction_literal(np.asarray(x, dtype=np.float64))

        x0 = p[0].copy()
        options = {
            'initial_simplex': p.copy(),
            'maxfev': self.mMaxEvaluations,
            'xatol': self.mTolerance,
            'fatol': self.mTolerance,
        }
        res = _optimize.minimize(wrapped_fun, x0, method='Nelder-Mead', options=options)
        
        self.mEvaluationCount = res.nfev
        self.mBest = np.asarray(res.x, dtype=np.float64)
        self.mResult = float(res.fun)
        
        if not res.success and res.status != 2:
            if res.nfev >= self.mMaxEvaluations:
                raise EPQException("Exceeded the maximum number of iterations in Simplex algorithm.")
                
        p[0] = self.mBest
        return self.mBest

    def getEvaluationCount(self) -> int:
        return int(self.mEvaluationCount)

    def getBestResult(self) -> float:
        return float(self.mResult)

    def setTolerance(self, t: float) -> None:
        t = abs(float(t))
        assert t < self._LARGEST_TOLERANCE
        self.mTolerance = t if t < self._LARGEST_TOLERANCE else self._LARGEST_TOLERANCE

    def getTolerance(self) -> float:
        return float(self.mTolerance)

    def setMaxEvaluations(self, n: int) -> None:
        self.mMaxEvaluations = max(int(n), self._MIN_EVALUATIONS)

    def getMaxEvaluations(self) -> int:
        return int(self.mMaxEvaluations)