r"""
LinearLeastSquaresMS_ver2_1_1.py — Python port of
gov.nist.microanalysis.Utility.LinearLeastSquaresMS

Guide version : 2
Generation    : 1
Port-code fixes: 1

CHANGES
-------
- Extends LinearLeastSquares_ver2_1_1 (guide-version-2 sibling port).
- _INT_MIN = -2147483648 (Java Integer.MIN_VALUE sentinel for "not initialized").
- Three Java constructors collapsed into one Python __init__ with optional
  args (same dispatch pattern as the parent LinearLeastSquares port).
- MS instance fields (_mNParams, _mZero, _mAMax, _mMetric, _mOptimize)
  initialised BEFORE super().__init__() because the parent constructor calls
  setData → _reevaluateAll(), which is overridden here (ordering caveat from
  spec); guard is unnecessary because _reevaluateAll only SETS _mNParams.
- DEVIATION-LLS-HOOK (R10): The LLS port inverted the Java calling chain.
  Java: performFit() → perform() (hook).
  LLS port: _perform() → _performFit() (computation).
  fitParameters() calls _performFit() directly, bypassing _perform().
  This means overriding _perform() alone (as the spec suggests) would leave
  the model-selection setup unreachable. Resolution: override _performFit()
  instead of _perform(). _performFit() is not reentrant-safe (uses threading.Lock
  in the parent), so the LLSMS override runs setup OUTSIDE the lock and delegates
  the actual SVD computation to super()._performFit() WITHOUT holding any lock.
- Arrays.sort(dup) → dup = sorted(w); Arrays.binarySearch(dup, w[j]) < nDrop
  → bisect.bisect_left(dup, w[j]) < nDrop. Note: for tied singular values,
  bisect_left returns the leftmost index; Java binarySearch returns an arbitrary
  index. Results differ only when duplicate singular values straddle the nDrop
  boundary (R10 notation).
- Java assert wMax >= 0 in editSingularValues dropped (disabled by default).
- _mAMax initialised to sys.float_info.max (Java Double.MAX_VALUE).
- FIX-1 (API-mismatch): Java's _mZero mask in editSingularValues zeros SVD
  singular values but does NOT exclude the corresponding columns from the
  design matrix.  For non-orthogonal bases (e.g. [constant, linear]) zeroing
  only the smaller singular value still leaves a non-zero projection onto the
  negative-coefficient basis function, so the coefficient is NOT zeroed in the
  output.  Fix: after the first unconstrained fit, set self._mZeroThese (the
  base-class column-exclusion mask) for every negative coefficient, clear the
  SVD cache (self._mSVD = None) so it is rebuilt without the excluded columns,
  and track which entries LLSMS added in self._mZeroMS so that a subsequent
  _reevaluateAll() call can undo only the LLSMS-managed portion before the
  next first-unconstrained fit.
- FIX-2 (assert-error): Python raises ValueError for math.sqrt(d) when d < 0
  (Java returns NaN for Math.sqrt(negative)).  A floating-point slightly-negative
  covariance determinant caused a crash in _computeMetric during the optimize
  loop.  Fix: return float('nan') when d <= 0 to match Java's NaN sentinel
  behaviour (NaN never compares less than any metric, so the ill-conditioned
  fit is silently skipped in the selection loop).

------------------------------------------------------------------------
Original Java source (gov.nist.microanalysis.Utility.LinearLeastSquaresMS)
------------------------------------------------------------------------
/**
 * <p>
 * An implementation of the linear least squares fit that uses Bayesian model
 * selection to trim the number of fit paramters to an optimal number. MS stands
 * for 'model selection'.
 * </p>
 * <p>
 * Copyright: Pursuant to title 17 Section 105 of the United States Code this
 * software is not subject to copyright protection and is in the public domain
 * </p>
 * <p>
 * Institution: National Institute of Standards and Technology
 * </p>
 *
 * @author Nicholas
 * @version 1.0
 */
------------------------------------------------------------------------
"""
from __future__ import annotations

import bisect
import math
import sys
from typing import List, Optional, Sequence

import numpy as np

try:
    from ._epq_compat import EPQException, F64Array, JamaMatrix, JavaRandom
except ImportError:
    try:
        from _epq_compat import EPQException, F64Array, JamaMatrix, JavaRandom  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility._epq_compat import EPQException, F64Array, JamaMatrix, JavaRandom  # type: ignore

try:
    from .LinearLeastSquares_ver2_1_1 import LinearLeastSquares
except ImportError:
    try:
        from LinearLeastSquares_ver2_1_1 import LinearLeastSquares  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility_ver2.LinearLeastSquares_ver2_1_1 import LinearLeastSquares  # type: ignore

BUG_LEDGER: tuple = (
    (
        "DEVIATION-LLS-HOOK",
        "_performFit",
        "Java LLS.performFit() calls perform() (the overridable hook). "
        "The LLS Python port inverted this: _perform() calls _performFit(). "
        "fitParameters() calls _performFit() directly, so overriding only "
        "_perform() would leave the model-selection setup unreachable. "
        "LLSMS therefore overrides _performFit() instead of _perform(), "
        "running all setup outside the parent lock and delegating SVD "
        "computation to super()._performFit() without holding any lock.",
    ),
    (
        "FIX-1-ZEROMODE",
        "_editSingularValues / _performFit",
        "Java's mZero mask zeros entries in the wi[] array passed to "
        "editSingularValues, but that array is indexed by SVD order (not by "
        "coefficient index) for non-orthogonal design matrices.  Zeroing the "
        "SVD singular value does not zero the corresponding fit coefficient "
        "when the right singular vectors mix basis directions.  Python fix: "
        "set _mZeroThese (the LLS column-exclusion mask) for every negative "
        "coefficient and clear _mSVD to force a rebuild without those columns.",
    ),
    (
        "FIX-2-SQRT-NEG",
        "_computeMetric",
        "Python raises ValueError for math.sqrt(d<0); Java returns NaN. "
        "Return float('nan') when det(covariance) <= 0 so ill-conditioned "
        "fits are silently skipped in the model-selection loop.",
    ),
)

# Java Integer.MIN_VALUE — sentinel meaning "mNParams not yet initialised"
_INT_MIN: int = -2147483648


class LinearLeastSquaresMS(LinearLeastSquares):
    """Python port of ``gov.nist.microanalysis.Utility.LinearLeastSquaresMS``.

    Abstract.  Extends :class:`LinearLeastSquares` with Bayesian model
    selection to trim the number of non-zero fit parameters.  Subclasses
    still implement :meth:`_fitFunctionCount` and :meth:`_fitFunction`.

    When :meth:`setOptimize` is ``False`` (default) the class behaves like
    the parent but forces any negative fit coefficient to zero.

    When :meth:`setOptimize` is ``True`` a Bayesian metric (Sivia eqn 4.20)
    selects the number of non-zero parameters that minimises the metric.
    """

    def __init__(
        self,
        x: Optional[Sequence[float]] = None,
        y: Optional[Sequence[float]] = None,
        sig: Optional[Sequence[float]] = None,
    ) -> None:
        # Initialise MS fields BEFORE super().__init__() because the parent
        # constructor calls setData → _reevaluateAll() (overridden here).
        self._mNParams: int = _INT_MIN
        self._mZero: Optional[List[bool]] = None
        self._mAMax: float = sys.float_info.max  # Java Double.MAX_VALUE
        self._mMetric: Optional[List[float]] = None
        self._mOptimize: bool = False
        # FIX-1: tracks which _mZeroThese entries were added by LLSMS (not
        # user-set) so they can be undone before the next first-unconstrained fit.
        self._mZeroMS: Optional[List[bool]] = None
        super().__init__(x, y, sig)

    # ------------------------------------------------------------------
    # _reevaluateAll override
    # ------------------------------------------------------------------

    def _reevaluateAll(self) -> None:
        """Reset model-selection state, then delegate to parent.

        Java: ``@Override protected void reevaluateAll()``
        """
        self._mNParams = _INT_MIN
        super()._reevaluateAll()

    # ------------------------------------------------------------------
    # _editSingularValues override
    # ------------------------------------------------------------------

    def _editSingularValues(self, wi: List[float]) -> List[float]:
        """Retain the largest ``_mNParams`` singular values; apply ``_mZero`` mask.

        Java: ``@Override protected double[] editSingularValues(double[] wi)``

        Algorithm:
        1. Find wMax.
        2. Apply ``_mZero`` mask (force specific parameters to 0).
        3. Sort a copy; compute nDrop = fitFunctionCount() - _mNParams.
        4. Zero any entry whose sorted rank < nDrop OR whose value < wMax * TOLERANCE.

        ``Arrays.binarySearch`` → ``bisect.bisect_left`` (R10: for distinct singular
        values behaviour is identical; for ties, bisect_left picks the leftmost
        index which may differ from Java's arbitrary choice by ≤1).
        Java assert wMax >= 0 dropped (disabled by default).
        """
        w: List[float] = list(wi)
        # Find the largest weight
        wMax: float = -sys.float_info.max
        for element in w:
            if element > wMax:
                wMax = element
        # Apply forced-zero mask
        if self._mZero is not None:
            for j in range(len(w)):
                if self._mZero[j]:
                    w[j] = 0.0
        dup: List[float] = sorted(w)  # ascending, matches Arrays.sort
        nDrop: int = self._fitFunctionCount() - self._mNParams
        thresh: float = wMax * self._TOLERANCE
        for j in range(len(w)):
            if (bisect.bisect_left(dup, w[j]) < nDrop) or (w[j] < thresh):
                w[j] = 0.0
        return w

    # ------------------------------------------------------------------
    # _performFit override  (DEVIATION-LLS-HOOK — see CHANGES)
    # ------------------------------------------------------------------

    def _performFit(self) -> None:
        """Lazy model-selection fit.

        Replaces the parent lazy check with a two-phase approach:
        Phase 1 (first call only — _mNParams == _INT_MIN):
          a. Undo LLSMS-managed column exclusions from any prior fit (FIX-1).
          b. Set _mNParams = fitFunctionCount().
          c. (If optimize) Compute _mAMax from the basis at each data point.
          d. Run first fit (parent _performFit) to find which params go negative.
          e. Set _mZero mask from those results.
          f. (FIX-1) Set _mZeroThese for negative coefficients and clear _mSVD
             so the next fit rebuilds the design matrix without those columns.
          g. (If optimize) Model-selection loop over _mNParams values 1..n-zeroCx;
             pick the _mNParams that minimises _computeMetric().
          h. Clear cached coefficients so the final fit runs fresh.
        Phase 2 (every call with _mFitCoefficients is None):
          Delegate to super()._performFit() with correct _mNParams and _mZero.
        """
        if self._mFitCoefficients is not None:
            return  # Already computed (lazy guard)

        if self._mNParams == _INT_MIN:
            # ---- FIX-1: undo LLSMS-managed exclusions from a prior fit ----
            # so the first unconstrained fit sees the full column set.
            if self._mZeroMS is not None:
                for j, managed in enumerate(self._mZeroMS):
                    if managed and self._mZeroThese is not None and j < len(self._mZeroThese):
                        self._mZeroThese[j] = False  # FIX-1: undo LLSMS exclusion
                if self._mZeroThese is not None and not any(self._mZeroThese):
                    self._mZeroThese = None
                self._mZeroMS = None

            # ---- Phase 1: first-time model-selection setup ----
            self._mNParams = self._fitFunctionCount()

            if self._mOptimize:
                # Compute assumed prior for fit parameters (mAMax).
                n: int = self._mNParams
                maxCh: List[int] = [0] * n
                maxFF: List[float] = [0.0] * n
                ff_arr: F64Array = np.zeros(n, dtype=np.float64)
                self._fitFunction(self._mXCoordinate[0], ff_arr)
                for k in range(n):
                    maxFF[k] = float(ff_arr[k])
                ff2: F64Array = np.zeros(n, dtype=np.float64)
                for ch in range(1, len(self._mXCoordinate)):
                    self._fitFunction(self._mXCoordinate[ch], ff2)
                    for i in range(n):
                        if float(ff2[i]) > maxFF[i]:
                            maxCh[i] = ch
                            maxFF[i] = float(ff2[i])
                self._mAMax = float(self._mData[maxCh[0]]) / maxFF[0]
                for i in range(1, n):
                    candidate: float = float(self._mData[maxCh[i]]) / maxFF[i]
                    if candidate > self._mAMax:
                        self._mAMax = candidate

            self._mZero = None
            zeroCx: int = 0

            # First fit: find which fit parameters come out negative.
            # super()._performFit() acquires its own lock; we do NOT hold any lock here.
            super()._performFit()
            if self._mFitCoefficients is not None:
                self._mZero = [False] * len(self._mFitCoefficients)
                for j in range(len(self._mFitCoefficients)):
                    self._mZero[j] = (self._mFitCoefficients[j].doubleValue() < 0.0)
                    if self._mZero[j]:
                        zeroCx += 1

                # FIX-1: Exclude negative-coefficient columns from the design
                # matrix by setting _mZeroThese (the base-class exclusion mask)
                # and clearing _mSVD to force a rebuild without those columns.
                # SVD singular-value masking alone (via _mZero in
                # _editSingularValues) does not zero coefficients for
                # non-orthogonal bases because the right singular vectors mix
                # basis directions.
                if zeroCx > 0:
                    nTot: int = self._fitFunctionCount()
                    self._mZeroMS = [False] * nTot  # FIX-1: track LLSMS additions
                    if self._mZeroThese is None:
                        self._mZeroThese = [False] * nTot
                    for j in range(len(self._mZero)):
                        if self._mZero[j]:
                            self._mZeroThese[j] = True  # FIX-1: exclude column
                            self._mZeroMS[j] = True     # FIX-1: mark as LLSMS-managed
                    self._mSVD = None  # FIX-1: force SVD rebuild without excluded cols

            if self._mOptimize and zeroCx < self._fitFunctionCount():
                metric_len: int = self._fitFunctionCount() - zeroCx
                self._mMetric = [0.0] * metric_len
                minMetric: int = 0
                self._mMetric[0] = float("inf")
                for i in range(1, metric_len):
                    self._mNParams = i
                    self._reevaluate()
                    super()._performFit()
                    m: float = self._computeMetric()
                    if m < self._mMetric[minMetric]:
                        minMetric = i
                    self._mMetric[i] = m
                self._mNParams = minMetric

            # Clear so the unconditional final fit below runs fresh.
            self._reevaluate()

        # ---- Phase 2: final fit with current _mNParams and _mZeroThese ----
        super()._performFit()

    # ------------------------------------------------------------------
    # _computeMetric (protected)
    # ------------------------------------------------------------------

    def _computeMetric(self) -> float:
        """Bayesian model-selection metric (Sivia 1996, eqn 4.20).

        Java: ``protected double computeMetric() throws EPQException``
        FIX-2: Return float('nan') when det(covariance) <= 0 to match Java's
        Math.sqrt(negative) = NaN behaviour (NaN never wins the min comparison).
        """
        d: float = self.covariance().det()
        if d <= 0.0:
            return float('nan')  # FIX-2: match Java NaN for negative/zero det
        return (
            math.pow((4.0 * math.pi) / self._mAMax, self._mNParams)
            * math.exp(-0.5 * self.chiSquared())
        ) / math.sqrt(d)

    # ------------------------------------------------------------------
    # Public controls
    # ------------------------------------------------------------------

    def isOptimize(self) -> bool:
        """Return whether Bayesian model-selection optimisation is enabled.

        Java: ``public boolean isOptimize()``
        """
        return self._mOptimize

    def setOptimize(self, optimize: bool) -> None:
        """Enable or disable Bayesian model-selection optimisation.

        Toggling forces a full recomputation of all fit parameters.

        Java: ``public void setOptimize(boolean optimize)``
        """
        if self._mOptimize != bool(optimize):
            self._mOptimize = bool(optimize)
            self._reevaluateAll()
