r"""
LinearLeastSquares_ver2_1_1.py — Python port of
gov.nist.microanalysis.Utility.LinearLeastSquares

Guide version : 2
Generation    : 1
Port-code fixes: 1

CHANGES
-------
- `Jama.SingularValueDecomposition` replaced by `numpy.linalg.svd` (SCIPY-DEV-1):
  Jama's SVD is not wrapped by JamaMatrix.  `numpy.linalg.svd(A, full_matrices=False)`
  returns `U, s, Vt`.  Jama's `V` matrix = numpy `Vt.T`; all index uses updated
  accordingly.  No `_literal` twin for `_performFit` — Java itself delegates to Jama
  SVD; numpy is the faithful equivalent.
- `chiSqr` static method: primary uses `scipy.stats.chi2.ppf`; `chiSqr_literal` contains
  the Java bisection via `Math2.gammq`.  Marked `# SCIPY-DEV-2` (SCIPY-DEV-1 is the SVD).
- `synchronized (this)` lazy fit → `threading.Lock` (R10).
- Java `assert` statements dropped (disabled by default in JVM).
- `fitParamter` typo kept verbatim (R1).
- Abstract protected methods: `_fitFunctionCount`, `_fitFunction` (R1 `protected`→`_`,
  `abstract` does NOT add a second `_`).
- R4 splits: `setData_xysig` / `setData_xy`; `_chiSquared_d` / `_chiSquared_uv`;
  `fitQuality` / `fitQuality_fp`.
- `INTERVAL_MODE` enum nested, plus module-level alias (R2).
- FIX-1 (R4): Added `setData` dispatcher routing to `setData_xysig` / `setData_xy`
  by presence of `sig` argument.  Guide §R4 permits a unified dispatcher for
  source-style compatibility; Java callers use `setData(x,y,sig)` or `setData(x,y)`.

------------------------------------------------------------------------
Original Java source (gov.nist.microanalysis.Utility.LinearLeastSquares)
------------------------------------------------------------------------
/**
 * <p>
 * The object computes the linear least squares fit in the constructor and
 * provides methods to return the fit parameters, the covariances etc. The class
 * is abstract. Implement fitFunction to specify the m different functions that
 * are to be fit to the data. Implement error to assign an error measure to each
 * data point.
 * </p>
 * <p>
 * The implementation uses singular value decomposition (SVD) which is likely to
 * be use more memory and more CPU than the fastest algorithm but is extremely
 * robust. SVD is also very flexible and when many permutations of parameters
 * are considered it may actually turn out to be the most efficient.
 * </p>
 * <p>
 * This implementation is designed to be extended. It divides the computation of
 * the SVD from the computation of the fit parameters so that the weights on the
 * SVD can be modified without recomputing the whole SVD.
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
import enum
import math
import sys
import threading
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy import stats as _scipy_stats
    _HAVE_SCIPY: bool = True
except ImportError:
    _HAVE_SCIPY = False

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
        "_performFit",
        "Java uses `Jama.SingularValueDecomposition` via `a.svd()`.  JamaMatrix does not "
        "wrap SVD.  Replaced by `numpy.linalg.svd(A, full_matrices=False)` returning "
        "`U, s, Vt` where `A = U @ diag(s) @ Vt`.  Jama's `V` = numpy `Vt.T`; all V "
        "index accesses `v.get(j, i)` become `Vt[i, j]` in Python.  No `_literal` twin.",
    ),
    (
        "SCIPY-DEV-2",
        "chiSqr",
        "Java bisects `1 - Math2.gammq(dof/2, x/2)` to find the chi-squared quantile.  "
        "Primary `chiSqr()` uses `scipy.stats.chi2.ppf(prob, degsOfFree)` when scipy is "
        "available; falls back to `chiSqr_literal()` otherwise.  `chiSqr_literal()` "
        "contains the faithful bisection port.",
    ),
)

# ---- numpy SVD cache type alias ------------------------------------------------
_SVDCache = Tuple[np.ndarray, np.ndarray, np.ndarray]  # (U, s, Vt)


class LinearLeastSquares(abc.ABC):
    """Python port of ``gov.nist.microanalysis.Utility.LinearLeastSquares``.

    Abstract.  Subclasses implement :meth:`_fitFunctionCount` and
    :meth:`_fitFunction`.  The SVD-based fit is computed lazily on first access.
    """

    # ------------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------------

    _TOLERANCE: float = 1.0e-12  # protected static final
    _MAX_ERROR: float = sys.float_info.max  # private static final Double.MAX_VALUE

    # ------------------------------------------------------------------
    # Nested enum
    # ------------------------------------------------------------------

    class INTERVAL_MODE(enum.Enum):
        ONE_D_INTERVAL = 1
        JOINT_INTERVAL = 2

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    def __init__(
        self,
        x: Optional[Sequence[float]] = None,
        y: Optional[Sequence[float]] = None,
        sig: Optional[Sequence[float]] = None,
    ) -> None:
        self._mXCoordinate: Optional[F64Array] = None
        self._mData: Optional[F64Array] = None
        self._mSigma: Optional[F64Array] = None
        self._mFitCoefficients: Optional[List[UncertainValue2]] = None
        self._mCovariance: Optional[JamaMatrix] = None
        self._mSVD: Optional[_SVDCache] = None
        self._mZeroThese: Optional[List[bool]] = None
        self._mLock: threading.Lock = threading.Lock()
        if x is not None and y is not None:
            self.setData_xysig(x, y, sig)

    # ------------------------------------------------------------------
    # setData overloads (R4)
    # ------------------------------------------------------------------

    def setData_xysig(
        self,
        x: Sequence[float],
        y: Sequence[float],
        sig: Optional[Sequence[float]],
    ) -> None:
        """Java: ``public void setData(double[] x, double[] y, double[] sig)``"""
        x_arr: F64Array = np.asarray(x, dtype=np.float64)
        y_arr: F64Array = np.asarray(y, dtype=np.float64)
        sig_arr: Optional[F64Array] = (
            np.asarray(sig, dtype=np.float64) if sig is not None else None
        )
        # Java asserts (disabled) dropped: y.length==x.length, sig==null or sig.length==x.length
        cx: int = 0
        if sig_arr is None:
            cx = len(y_arr)
        else:
            for element in sig_arr:
                if element < 1.0e300:
                    cx += 1
        if cx < len(y_arr):
            self._mXCoordinate = np.empty(cx, dtype=np.float64)
            self._mData = np.empty(cx, dtype=np.float64)
            self._mSigma = np.empty(cx, dtype=np.float64)
            j: int = 0
            for i in range(len(sig_arr)):  # type: ignore[arg-type]
                if sig_arr[i] < 1.0e300:  # type: ignore[index]
                    self._mXCoordinate[j] = x_arr[i]
                    self._mData[j] = y_arr[i]
                    self._mSigma[j] = sig_arr[i]  # type: ignore[index]
                    j += 1
        else:
            self._mXCoordinate = x_arr
            self._mData = y_arr
            self._mSigma = sig_arr  # may be None when sig arg was None
        self._reevaluateAll()

    def setData_xy(self, x: Sequence[float], y: Sequence[float]) -> None:
        """Java: ``public void setData(double[] x, double[] y)``"""
        self.setData_xysig(x, y, None)

    def setData(  # FIX-1: R4 dispatcher — routes to setData_xysig / setData_xy by sig presence
        self,
        x: Sequence[float],
        y: Sequence[float],
        sig: Optional[Sequence[float]] = None,
    ) -> None:
        """R4 dispatcher for ``public void setData(...)`` overloads.

        Java: ``setData(x, y, sig)`` → ``setData_xysig``; ``setData(x, y)`` → ``setData_xy``.
        """
        if sig is not None:
            self.setData_xysig(x, y, sig)
        else:
            self.setData_xy(x, y)

    # ------------------------------------------------------------------
    # Private fit engine
    # ------------------------------------------------------------------

    def _performFit(self) -> None:
        """Lazy SVD-based fit.  Java: ``private void performFit()``."""
        if self._mFitCoefficients is None:
            with self._mLock:
                if self._mFitCoefficients is None:
                    if (
                        self._mXCoordinate is None
                        or self._mData is None
                        or self._mSigma is None
                    ):
                        raise IllegalArgumentError(
                            "No data specified for the linear least squares fit."
                        )
                    nTot: int = self._fitFunctionCount()
                    self._mFitCoefficients = [UncertainValue2.ZERO] * nTot
                    nFit: int = self.getNonZeroedCoefficientCount()
                    if nFit == 0:
                        return
                    nzIndex: List[int] = []
                    for j in range(nTot):
                        if not self.isZeroFitCoefficient(j):
                            nzIndex.append(j)
                    dataLen: int = len(self._mXCoordinate)
                    if self._mSVD is None:
                        # Build the design matrix
                        a_arr: np.ndarray = np.zeros((dataLen, nFit), dtype=np.float64)
                        afunc: F64Array = np.zeros(nTot, dtype=np.float64)
                        for i in range(dataLen):
                            # Java assert mSigma[i] >= 0 dropped
                            self._fitFunction(self._mXCoordinate[i], afunc)
                            for j in range(nFit):
                                val: float = float(afunc[nzIndex[j]]) / max(
                                    float(self._mSigma[i]), 1.0e-20
                                )
                                # Java assert !isNaN && !isInfinite dropped
                                a_arr[i, j] = val
                        # SCIPY-DEV-1: Jama SVD → numpy.linalg.svd
                        U_arr, s_arr, Vt_arr = np.linalg.svd(
                            a_arr, full_matrices=False
                        )
                        self._mSVD = (U_arr, s_arr, Vt_arr)
                    U_arr, s_arr, Vt_arr = self._mSVD
                    # Java: u = mSVD.getU(), s = mSVD.getS(), v = mSVD.getV()
                    # Jama V[j,i] = Vt_arr[i,j]
                    # Edit singular values
                    wi_full: List[float] = [0.0] * nTot
                    for i in range(nFit):
                        wi_full[nzIndex[i]] = float(s_arr[i])
                    wi_full = self._editSingularValues(wi_full)
                    w: List[float] = [0.0] * nFit
                    for i in range(nFit):
                        w[i] = wi_full[nzIndex[i]]
                    # Covariance matrix
                    self._mCovariance = JamaMatrix.zeros(nTot, nTot)
                    wti: List[float] = [
                        (1.0 / (w[i] * w[i]) if w[i] != 0.0 else 0.0)
                        for i in range(nFit)
                    ]
                    for j in range(nFit):
                        for k in range(j + 1):
                            s_cov: float = 0.0
                            for i in range(nFit):
                                # Jama: v.get(j,i)*v.get(k,i) → Vt[i,j]*Vt[i,k]
                                s_cov += float(Vt_arr[i, j]) * float(Vt_arr[i, k]) * wti[i]
                            self._mCovariance.set(nzIndex[j], nzIndex[k], s_cov)
                            self._mCovariance.set(nzIndex[k], nzIndex[j], s_cov)
                    # Fit coefficients
                    b: List[float] = [
                        float(self._mData[i]) / float(self._mSigma[i])
                        for i in range(dataLen)
                    ]
                    fcs: List[float] = [0.0] * nFit
                    for k in range(nFit):
                        fc: float = 0.0
                        for i in range(nFit):
                            if w[i] != 0.0:
                                dot: float = 0.0
                                for j in range(dataLen):
                                    # Jama: u.get(j,i) → U_arr[j,i]
                                    dot += float(U_arr[j, i]) * b[j]
                                # Jama: v.get(k,i) → Vt_arr[i,k]
                                fc += (dot / w[i]) * float(Vt_arr[i, k])
                        fcs[k] = fc
                    expU: List[float] = self.confidenceIntervals(
                        LinearLeastSquares.INTERVAL_MODE.ONE_D_INTERVAL,
                        0.683,
                        self._mCovariance,
                    )
                    for j in range(nFit):
                        self._mFitCoefficients[nzIndex[j]] = UncertainValue2(
                            fcs[j], "LLS", expU[nzIndex[j]]
                        )

    # ------------------------------------------------------------------
    # Protected extension hook
    # ------------------------------------------------------------------

    def _perform(self) -> None:
        """Protected hook; default delegates to `_performFit()`.

        Java: ``protected void perform() throws EPQException``
        Overridden by ``LinearLeastSquaresMS``.
        """
        self._performFit()

    # ------------------------------------------------------------------
    # editSingularValues (protected, overridable)
    # ------------------------------------------------------------------

    def _editSingularValues(self, wi: List[float]) -> List[float]:
        """Zero singular values below ``wMax * TOLERANCE``.

        Java: ``protected double[] editSingularValues(double[] wi)``
        """
        w: List[float] = list(wi)
        wMax: float = 0.0
        for element in w:
            if element > wMax:
                wMax = element
        thresh: float = wMax * self._TOLERANCE
        for j in range(len(w)):
            if w[j] < thresh:
                w[j] = 0.0
        return w

    # ------------------------------------------------------------------
    # reevaluate helpers (protected)
    # ------------------------------------------------------------------

    def _reevaluate(self) -> None:
        self._mFitCoefficients = None
        self._mCovariance = None

    def _reevaluateAll(self) -> None:
        self._mSVD = None
        self._reevaluate()

    # ------------------------------------------------------------------
    # Static chiSqr (primary + literal pair, R2)
    # ------------------------------------------------------------------

    @staticmethod
    def chiSqr_literal(degsOfFree: int, prob: float) -> float:
        """Bisection-based chi-squared quantile.  Java: ``public static double chiSqr(int, double)``."""
        min_val: float = 0.1
        max_val: float = 100.0
        minV: float = 1.0 - Math2.gammq(0.5 * degsOfFree, 0.5 * min_val)
        maxV: float = 1.0 - Math2.gammq(0.5 * degsOfFree, 0.5 * max_val)
        # Java asserts (minV < prob, maxV > prob) dropped
        while math.fabs(max_val - min_val) > 0.01:
            test: float = 0.5 * (max_val + min_val)
            testV: float = 1.0 - Math2.gammq(0.5 * degsOfFree, 0.5 * test)
            if testV > prob:
                max_val = test
                maxV = testV
            elif testV < prob:
                min_val = test
                minV = testV
        return min_val if math.fabs(minV - prob) < math.fabs(maxV - prob) else max_val

    @staticmethod
    def chiSqr(degsOfFree: int, prob: float) -> float:
        """Chi-squared quantile (SCIPY-DEV-2: uses scipy.stats.chi2.ppf when available)."""
        if _HAVE_SCIPY:
            return float(_scipy_stats.chi2.ppf(prob, degsOfFree))  # SCIPY-DEV-2
        return LinearLeastSquares.chiSqr_literal(degsOfFree, prob)

    # ------------------------------------------------------------------
    # confidenceIntervals
    # ------------------------------------------------------------------

    def confidenceIntervals(
        self,
        mode: "LinearLeastSquares.INTERVAL_MODE",
        prob: float,
        cov: JamaMatrix,
    ) -> List[float]:
        res: List[float] = [0.0] * cov.getRowDimension()
        if mode == LinearLeastSquares.INTERVAL_MODE.ONE_D_INTERVAL:
            k: float = 1.0 if prob == 0.683 else LinearLeastSquares.chiSqr(1, prob)
            for i in range(len(res)):
                res[i] = math.sqrt(k * cov.get(i, i))
        elif mode == LinearLeastSquares.INTERVAL_MODE.JOINT_INTERVAL:
            ci: JamaMatrix = cov.inverse()
            k = LinearLeastSquares.chiSqr(len(res), prob)
            d: float = ci.det()
            subDim: int = len(res) - 1
            sub: JamaMatrix = JamaMatrix.zeros(subDim, subDim)
            for i in range(len(res)):
                for r in range(subDim):
                    for c in range(subDim):
                        sub.set(
                            r,
                            c,
                            ci.get(r if r < i else r + 1, c if c < i else c + 1),
                        )
                res[i] = math.sqrt(math.fabs((k * sub.det()) / d))
        return res

    # ------------------------------------------------------------------
    # Public result getters
    # ------------------------------------------------------------------

    def fitParameters(self) -> List[float]:
        self._performFit()
        return [uv.doubleValue() for uv in self._mFitCoefficients]  # type: ignore[union-attr]

    def getResults(self) -> List[UncertainValue2]:
        self._performFit()
        return self._mFitCoefficients  # type: ignore[return-value]

    def fitParamter(self, i: int) -> float:  # Java spelling preserved (R1)
        self._performFit()
        return self._mFitCoefficients[i].doubleValue()  # type: ignore[index]

    def errors(self) -> List[float]:
        self._performFit()
        res: List[float] = [0.0] * self._mCovariance.getRowDimension()  # type: ignore[union-attr]
        for i in range(len(res)):
            res[i] = math.sqrt(self._mCovariance.get(i, i))  # type: ignore[union-attr]
        return res

    def covariance(self) -> JamaMatrix:
        self._performFit()
        return self._mCovariance  # type: ignore[return-value]

    def correlation(self) -> JamaMatrix:
        self._performFit()
        res: JamaMatrix = JamaMatrix(self._mCovariance.getArrayCopy())  # type: ignore[union-attr]
        for r in range(res.getRowDimension()):
            res.set(r, r, 1.0)
            for c in range(r + 1, res.getColumnDimension()):
                cv: float = self._mCovariance.get(r, c)  # type: ignore[union-attr]
                corr: float = (
                    cv / math.sqrt(self._mCovariance.get(r, r) * self._mCovariance.get(c, c))  # type: ignore[union-attr]
                    if cv != 0.0
                    else 0.0
                )
                # Java assert corr in [-1,1] dropped
                res.set(r, c, corr)
                res.set(c, r, corr)
        return res

    def chiSquared(self) -> float:
        return self._chiSquared_d(self.fitParameters())

    def reducedChiSquared(self, confidenceLevel: float) -> float:
        fp: List[float] = self.fitParameters()
        dof: int = 0
        for element in self._mSigma:  # type: ignore[union-attr]
            if element != self._MAX_ERROR:
                dof += 1
        for element in fp:
            if element != 0.0:
                dof -= 1
        # Java assert dof > 0 dropped
        return self._chiSquared_d(fp) / Math2.chiSquaredConfidenceLevel(
            confidenceLevel, dof
        )

    # ------------------------------------------------------------------
    # Protected chiSquared overloads (R4)
    # ------------------------------------------------------------------

    def _chiSquared_d(self, fitCoeff: List[float]) -> float:
        """Java: ``protected double chiSquared(double[] fitCoeff)``"""
        n: int = self._fitFunctionCount()
        # Java assert fitCoeff.length == n dropped
        ff: F64Array = np.zeros(n, dtype=np.float64)
        chi2: float = 0.0
        for ch in range(len(self._mXCoordinate)):  # type: ignore[arg-type]
            self._fitFunction(self._mXCoordinate[ch], ff)  # type: ignore[index]
            y: float = 0.0
            for j in range(n):
                y += fitCoeff[j] * float(ff[j])
            chi2 += Math2.sqr(
                (y - float(self._mData[ch])) / float(self._mSigma[ch])  # type: ignore[index]
            )
        return chi2

    def _chiSquared_uv(self, fitCoeff: List[UncertainValue2]) -> float:
        """Java: ``protected double chiSquared(UncertainValue2[] fitCoeff)``"""
        n: int = self._fitFunctionCount()
        # Java assert fitCoeff.length == n dropped
        ff: F64Array = np.zeros(n, dtype=np.float64)
        chi2: float = 0.0
        for ch in range(len(self._mXCoordinate)):  # type: ignore[arg-type]
            self._fitFunction(self._mXCoordinate[ch], ff)  # type: ignore[index]
            y: float = 0.0
            for j in range(n):
                y += fitCoeff[j].doubleValue() * float(ff[j])
            chi2 += Math2.sqr(
                (y - float(self._mData[ch])) / float(self._mSigma[ch])  # type: ignore[index]
            )
        return chi2

    # ------------------------------------------------------------------
    # Zeroed coefficients
    # ------------------------------------------------------------------

    def clearZeroedCoefficients(self) -> None:
        if self._mZeroThese is not None:
            self._mZeroThese = [False] * len(self._mZeroThese)

    def zeroFitCoefficient(self, i: int, b: bool) -> None:
        if (self._mZeroThese is None) and b:
            self._mZeroThese = [False] * self._fitFunctionCount()
        if (self._mZeroThese is not None) and (self._mZeroThese[i] != b):
            self._mZeroThese[i] = b
            self._reevaluateAll()

    def getNonZeroedCoefficientCount(self) -> int:
        if self._mZeroThese is None:
            return self._fitFunctionCount()
        return sum(1 for b in self._mZeroThese if not b)

    def isZeroFitCoefficient(self, i: int) -> bool:
        return (self._mZeroThese is not None) and self._mZeroThese[i]

    # ------------------------------------------------------------------
    # fitQuality overloads (R4)
    # ------------------------------------------------------------------

    def fitQuality(self) -> float:
        return self.fitQuality_fp(self.fitParameters())

    def fitQuality_fp(self, fp: List[float]) -> float:
        """Java: ``public double fitQuality(double[] fp)``"""
        dataPtCx: int = len(self._mXCoordinate)  # type: ignore[arg-type]
        paramCx: int = sum(1 for element in fp if element != 0.0)
        return Math2.gammq(
            0.5 * (dataPtCx - paramCx), 0.5 * self._chiSquared_d(fp)
        )

    # ------------------------------------------------------------------
    # Abstract methods (extension points)
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _fitFunctionCount(self) -> int:
        """Java: ``abstract protected int fitFunctionCount()``"""

    @abc.abstractmethod
    def _fitFunction(self, xi: float, afunc: F64Array) -> None:
        """Java: ``abstract protected void fitFunction(double xi, double[] afunc)``

        Fill *afunc* in place with the ``_fitFunctionCount()`` basis values at *xi*.
        """


# IllegalArgumentError alias for the Java IllegalArgumentException behaviour
IllegalArgumentError = ValueError

# Module-level aliases (R2)
INTERVAL_MODE = LinearLeastSquares.INTERVAL_MODE
