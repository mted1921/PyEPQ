"""
Math2_ver8_1_5.py - Literal Python port of gov.nist.microanalysis.Utility.Math2

Guide version : 8
Generation    : 1
Port-code fixes: 5

REVISION HISTORY
----------------
Rev 1 (Math2_ver1 / Math2.py): Initial faithful port using scipy/numpy.

Rev 2 (Math2_ver1 / Math2.py): Addresses cross-module concerns identified in
       the EPQ-wide migration review (see CONVERSION_GUIDE.md).

CHANGES IN REV 2 (see CONVERSION_GUIDE.md rules)
-------------------------------------------------
* R3: EPQException, JavaRandom, JamaMatrix imported from _epq_compat
  (single source of truth -- never define local stand-ins).
* RNG is now JavaRandom (48-bit LCG) so seeds reproduce Java sequences
  bit-for-bit. This unblocks parity testing and reproduction of
  published Monte Carlo runs.
* R4: Java overloads split into type-suffixed functions
  (`plus_vv`, `plus_vs`, `negative_arr`, `negative_scalar`,
  `multiply_sv`, `multiply_vv`, `bound_double`, `bound_int`).
  The Java-named dispatchers remain for source compatibility but
  carry "ambiguity hazard" warnings.
* R5: Mutating helpers (`plusEquals`, `divideEquals`, `timesEquals`,
  `addInPlace`) call `_require_mutable_f64` on their target arg.
  Lists, tuples, and wrong-dtype arrays now raise TypeError instead
  of silently no-opping.
* R2: Numerical Recipes literal ports (`erf_literal`, `gammaln_literal`,
  etc.) are retained alongside the scipy substitutions for parity
  testing. Public APIs default to scipy.
* R6: Preserved Java bugs are tagged with `# JAVA-BUG-N` markers and
  catalogued in `Math2.BUG_LEDGER`. Where reasonable, `*_strict`
  variants fix the bug while leaving the buggy version in place for
  source compatibility.
* `createRowMatrix` now returns JamaMatrix (not raw ndarray), restoring
  Java callers' ability to chain `.times(...)`, `.solve(...)`, etc.
* `randomDir` now uses Math2.rgen instead of an independent module RNG.
  This is a DELIBERATE departure from Java -- see RNG-DEVIATION-1 below.

Rev 8.1 (Math2_ver8_1_5.py): Rewritten as strict literal translation from
       Java, complying with R1-R10. All array iterations, Numerical Recipes
       loops, and Java floating-point quirks are preserved explicitly without
       library substitutions except where bridging anonymous classes requires
       it.

CHANGES IN THIS REVISION (ver8.1.5)
-----------------------------------
* FIX-1 (Correctness): `findRoot` now has an early exit if one of the bracket
  endpoints is the root. The original algorithm failed to converge correctly
  when the lower endpoint was the root, a bug caught by the boundary-value
  test suite.
* FIX-2 (R1): Corrected visibility of `_literal` methods. `erf_literal`,
  `gammap_literal`, `gammaln_literal`, etc. are now public, fixing
  AttributeErrors in the parity harness.
* FIX-3 (Correctness): Moved `findRoot` early exit after the bracket check to
  ensure `test_raises_both_endpoints_zero` correctly raises EPQException.
* R2-COMPLIANCE: Added `chiSquaredConfidenceLevel_literal` (the faithful
  FindRoot-based Java port). Required by R2 — every Java member must appear.
* R2-COMPLIANCE: Restored zero-denominator guard in `angleBetween`. The guard
  was present in Java and in the earlier Math2.py port; ver8.1 dropped it,
  producing NaN instead of 0.0 for zero-magnitude inputs.
* R6-COMPLIANCE: Added `solveCubic_strict` fixing JAVA-BUG-7 (`q / A` instead
  of `q / a`). Updated JAVA-BUG-7 BUG_LEDGER entry has_strict_variant → True.

------------------------------------------------------------------------
Original Java source (gov.nist.microanalysis.Utility.Math2)
------------------------------------------------------------------------
/**
 * <p>
 * Useful math functions not provided in the standard libraries.
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

import math
import sys
from typing import Optional, Sequence, Union, Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import special as _sp_special
from scipy import stats as _sp_stats

try:
    from ._epq_compat import EPQException, JavaRandom, JamaMatrix, F64Array
except ImportError:
    try:
        from _epq_compat import EPQException, JavaRandom, JamaMatrix, F64Array  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility._epq_compat import EPQException, JavaRandom, JamaMatrix, F64Array  # type: ignore

try:
    from .FindRoot_ver1_1_1 import FindRoot as _FindRoot
except ImportError:
    try:
        from FindRoot_ver1_1_1 import FindRoot as _FindRoot  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility.FindRoot_ver1_1_1 import FindRoot as _FindRoot  # type: ignore


__all__ = ["Math2", "EPQException", "JavaRandom", "JamaMatrix", "F64Array"]


class Math2:

    # ==================================================================
    # Preserved-bug ledger (machine-readable)
    # ==================================================================
    # Each tuple: (id, method, description, has_strict_variant).
    # The parity harness reads this to skip strict-equality comparison
    # for documented behaviours that deliberately diverge.
    BUG_LEDGER: tuple = (
        ("JAVA-BUG-1", "abs",
         "Clamps negatives to zero rather than computing |x|. "
         "Use `abs_real()` for true element-wise absolute value.", True),
        ("JAVA-BUG-2", "ebeDivide",
         "Indexes divisor as b[i % len(b)] despite asserting equal length. "
         "Use `ebeDivide_strict()` for the assertion-respecting variant.", True),
        ("JAVA-BUG-3", "cubicSolver",
         "Branches on exact float equality h==0; preserved verbatim. "
         "Prefer `cubicSolver2` for production work.", False),
        ("JAVA-BUG-4", "toContinuedFraction",
         "Prints intermediate convergents to stdout (leftover debug). "
         "Pass `verbose=False` to suppress.", True),
        ("JAVA-BUG-5", "createRowMatrix",
         "Misleading name: returns a column matrix (Nx1), not a row "
         "matrix (1xN), because Jama's `new Matrix(vals, m=vals.length)` "
         "treats vals as column-packed. Discovered via parity testing. "
         "Use `createRowMatrix_strict` for an actual (1, N) row matrix.", True),
        ("JAVA-BUG-6", "solveCubic",
         "Three-real-roots branch uses `-2*q*cos(...)` where the correct "
         "Cardano formula is `2*sqrt(q)*cos(...)`. Off by a factor of "
         "sqrt(q). For e.g. x^3 - 6x^2 + 11x - 6 = 0, Java returns "
         "{~1.42, ~2.58, 2} instead of {1, 2, 3}. The faithful Python "
         "port reproduces this. Discovered via boundary testing.", False),
        ("JAVA-BUG-7", "solveCubic",
         "One-real-root branch computes `B = q / a` where `a` is the "
         "quadratic coefficient parameter; the correct Cardano identity "
         "requires `B = q / A` where `A` is the local cube-root variable. "
         "Java case-sensitivity confusion between parameter `a` and local `A`. "
         "The guard `a == 0.0` is also wrong (should be `A == 0.0`). "
         "Use `solveCubic_strict` for the corrected computation.", True),
        ("RNG-DEVIATION-1", "randomDir",
         "Java uses Math.random() (independent of `rgen`); Python port "
         "uses `rgen` so a single seed determinises everything. This "
         "MEANS Java and Python randomDir() will diverge even with "
         "matched seeds. Acceptable: published Monte Carlo runs cite "
         "specific seeds for the trajectory loop, not for randomDir.", False),
    )

    # ==================================================================
    # Constants  (read-only by convention; see CONVERSION_GUIDE R5)
    # ==================================================================
    ORIGIN_3D: F64Array = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    ONE: F64Array = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    X_AXIS: F64Array = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    Y_AXIS: F64Array = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    Z_AXIS: F64Array = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    MINUS_X_AXIS: F64Array = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
    MINUS_Y_AXIS: F64Array = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    MINUS_Z_AXIS: F64Array = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    SQRT_PI: float = math.sqrt(math.pi)

    # Java-bit-compatible RNG. Reseed via Math2.initializeRandom(seed).
    rgen: JavaRandom = JavaRandom()

    # ==================================================================
    # Internal guards
    # ==================================================================
    @staticmethod
    def _require_mutable_f64(arr, name: str = "arr") -> None:
        """Type guard for in-place methods (CONVERSION_GUIDE R5).

        Java's ``double[]`` is always a mutable double-precision buffer.
        Our in-place helpers can only honour that contract on numpy
        ndarrays with ``dtype=float64``. Lists, tuples, and
        wrong-dtype arrays would silently no-op or copy. Fail loud.
        """
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{name} must be a numpy ndarray")
        if arr.dtype != np.float64:
            raise TypeError(f"{name} must have dtype float64")
        if not arr.flags.writeable:
            raise TypeError(f"{name} must be writeable")

    # ==================================================================
    # RNG management
    # ==================================================================
    @staticmethod
    def initializeRandom(seed: Optional[int] = None) -> None:
        """Reseed the module-level JavaRandom.

        Pass an explicit int for reproducibility. Pass None for
        time-based seeding (matches Java's no-arg ``new Random()``).
        """
        Math2.rgen = JavaRandom(seed)

    # ==================================================================
    # Tiny helpers
    # ==================================================================
    @staticmethod
    def sqr(x: float) -> float:
        return x * x

    # ==================================================================
    # Numerical Recipes ports -- literal AND library-substituted
    # ==================================================================
    # Each public function delegates to scipy.special by default.
    # The `_literal` variants are the faithful Java ports, used by the
    # parity harness to assert that the substitutions match within
    # tolerance.
    # ------------------------------------------------------------------

    @staticmethod
    # FIX-2 (R1): Renamed from _gammaln_literal; this is a public API in Java.
    def gammaln_literal(xx: float) -> float:
        """Literal NR Lanczos port with the NR coefficients."""
        coeff = (76.18009172947146, -86.50532032941677, 24.01409824083091,
                 -1.231739572450155, 0.1208650973866179e-2,
                 -0.5395239384953e-5)
        y = xx
        tmp = xx + 5.5
        tmp -= (xx + 0.5) * math.log(tmp)
        s = 1.000000000190015
        for c in coeff:
            y += 1.0
            s += c / y
        return -tmp + math.log(2.5066282746310005 * s / xx)

    @staticmethod
    def _gser_literal(a: float, x: float) -> float:
        """P(a, x) by series expansion."""
        assert x >= 0.0
        ITMAX = 100
        EPS = 3.0e-7
        if x == 0.0:
            return 0.0
        ap = a
        s = 1.0 / a
        delta = s
        for _ in range(1, ITMAX + 1):
            ap += 1.0
            delta *= x / ap
            s += delta
            if abs(delta) < abs(s) * EPS:
                break
        return s * math.exp(-x + a * math.log(x) - Math2.gammaln_literal(a))

    @staticmethod
    def _gcf_literal(a: float, x: float) -> float:
        """Q(a, x) by continued fraction."""
        ITMAX = 100
        EPS = 3.0e-7
        FPMIN = 1.0e-30
        b = x + 1.0 - a
        c = 1.0 / FPMIN
        d = 1.0 / b
        h = d
        for i in range(1, ITMAX + 1):
            an = -i * (i - a)
            b += 2.0
            d = an * d + b
            if abs(d) < FPMIN:
                d = FPMIN
            c = b + an / c
            if abs(c) < FPMIN:
                c = FPMIN
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < EPS:
                break
        return math.exp(-x + a * math.log(x) - Math2.gammaln_literal(a)) * h

    @staticmethod
    # FIX-2 (R1): Renamed from _gammap_literal; public API in Java.
    def gammap_literal(a: float, x: float) -> float:
        assert x >= 0.0
        assert a > 0.0
        if x < a + 1.0:
            return Math2._gser_literal(a, x)
        return 1.0 - Math2._gcf_literal(a, x)

    @staticmethod
    # FIX-2 (R1): Renamed from _gammq_literal; public API in Java.
    def gammq_literal(a: float, x: float) -> float:
        assert x >= 0.0
        assert a > 0.0
        if x < a + 1.0:
            return 1.0 - Math2._gser_literal(a, x)
        return Math2._gcf_literal(a, x)

    @staticmethod
    # FIX-2 (R1): Renamed from _erf_literal; public API in Java.
    def erf_literal(x: float) -> float:
        if x < 0.0:
            return -Math2.gammap_literal(0.5, x * x)
        return Math2.gammap_literal(0.5, x * x)

    @staticmethod
    # FIX-2 (R1): Renamed from _erfc_literal; public API in Java.
    def erfc_literal(x: float) -> float:
        if x < 0.0:
            return 1.0 + Math2.gammap_literal(0.5, x * x)
        return Math2.gammq_literal(0.5, x * x)

    # ---- Public APIs (scipy-backed; same names as Java) ----

    @staticmethod
    def erf(x: float) -> float:
        return float(_sp_special.erf(x))

    @staticmethod
    def erfc(x: float) -> float:
        return float(_sp_special.erfc(x))

    @staticmethod
    def gammq(a: float, x: float) -> float:
        assert x >= 0.0
        assert a > 0.0
        return float(_sp_special.gammaincc(a, x))

    @staticmethod
    def gammap(a: float, x: float) -> float:
        assert x >= 0.0
        assert a > 0.0
        return float(_sp_special.gammainc(a, x))

    @staticmethod
    def chiSquaredConfidenceLevel_literal(confidence: float, degreesOfFreedom: int) -> float:
        """Literal port of Java chiSquaredConfidenceLevel using FindRoot.

        Mirrors Java exactly: anonymous FindRoot subclass with
            function(x) = gammap(dof/2, x/2) - confidence
        searched on [1.0, 2*dof+50] with eps=1e-3 and iMax=100.

        Raises ValueError if the search range does not straddle a zero
        (matches Java's IllegalArgumentException on those inputs).
        """
        dof: float = float(degreesOfFreedom)

        class _ChiSqFR(_FindRoot):
            def initialize(self, vars: list) -> None:
                self._dof: float = vars[0]
                self._conf: float = vars[1]

            def function(self, x0: float) -> float:
                return Math2.gammap(0.5 * self._dof, 0.5 * x0) - self._conf

        fr: _ChiSqFR = _ChiSqFR()
        fr.initialize([dof, confidence])
        return fr.perform(1.0, 2.0 * dof + 50.0, 1.0e-3, 100)

    @staticmethod
    def chiSquaredConfidenceLevel(confidence: float, degreesOfFreedom: int) -> float:
        assert 0.0 < confidence < 1.0, "Confidence must be in the range (0, 1)."
        assert degreesOfFreedom > 0, "Degrees of freedom must be 1 or larger."
        if not (0.0 < confidence < 1.0) or degreesOfFreedom <= 0:
            return float("nan")
        return float(_sp_stats.chi2.ppf(confidence, degreesOfFreedom))

    @staticmethod
    def gammaln(xx: float) -> float:
        return float(_sp_special.gammaln(xx))

    # ==================================================================
    # Random variates
    # ==================================================================
    @staticmethod
    def expRand(lambda_: float = 1.0) -> float:
        """Java has two overloads (no-arg with lambda=1; one-arg with
        explicit lambda). Collapsed via default arg."""
        return -math.log(Math2.rgen.nextDouble()) / lambda_

    @staticmethod
    def randomDir() -> F64Array:
        """Knop's algorithm.

        RNG-DEVIATION-1: Java uses ``Math.random()`` here (the JVM's
        internal RNG, independent of ``rgen``). Our port uses
        ``Math2.rgen`` so a single ``initializeRandom(seed)`` call
        determinises every RNG-dependent function in this module.

        Implication: matched seeds between Java and Python produce
        DIFFERENT randomDir output. This is acceptable for published
        Monte Carlo runs because those cite seeds for the trajectory
        loop, not for randomDir specifically. If you need bit-exact
        Java parity for randomDir, instantiate a private JavaRandom
        for ``randomDir`` calls and reseed it independently.
        """
        # RNG-DEVIATION-1: Utilizing Math2.rgen instead of Math.random()
        while True:
            x = 2.0 * (Math2.rgen.nextDouble() - 0.5)
            y = 2.0 * (Math2.rgen.nextDouble() - 0.5)
            s = (x * x) + (y * y)
            if s <= 1.0:
                break
        z = (2.0 * s) - 1.0
        s = math.sqrt((1.0 - (z * z)) / s)
        x *= s
        y *= s
        return np.array([x, y, z], dtype=np.float64)

    # ==================================================================
    # Vector geometry
    # ==================================================================
    @staticmethod
    def distance(p1: ArrayLike, p2: ArrayLike) -> float:
        a = np.asarray(p1, dtype=np.float64)
        b = np.asarray(p2, dtype=np.float64)
        assert a.shape[0] == b.shape[0]
        sum2 = 0.0
        for i in range(a.shape[0]):
            sum2 += Math2.sqr(b[i] - a[i])
        return math.sqrt(sum2)

    @staticmethod
    def distanceSqr(p1: ArrayLike, p2: ArrayLike) -> float:
        a = np.asarray(p1, dtype=np.float64)
        b = np.asarray(p2, dtype=np.float64)
        assert a.shape[0] == b.shape[0]
        sum2 = 0.0
        for i in range(a.shape[0]):
            sum2 += Math2.sqr(b[i] - a[i])
        return sum2

    @staticmethod
    def magnitude(p: ArrayLike) -> float:
        a = np.asarray(p, dtype=np.float64)
        sum2 = 0.0
        for element in a:
            sum2 += element * element
        return math.sqrt(sum2)

    @staticmethod
    def normalize(p: ArrayLike) -> F64Array:
        a = np.asarray(p, dtype=np.float64)
        return Math2.divide(a, Math2.magnitude(a))

    # ==================================================================
    # Reductions
    # ==================================================================
    @staticmethod
    def sum(da: ArrayLike) -> Union[float, int]:
        """Element sum. Java overloads for int[] and double[]; covered by
        dtype-conditional initial value."""
        a = np.asarray(da)
        res = 0.0 if a.dtype == np.float64 else 0
        for element in a:
            res += element
        return res.item() if hasattr(res, "item") else res

    # ==================================================================
    # add / addInPlace
    # ==================================================================
    @staticmethod
    def add(da: ArrayLike, db: ArrayLike) -> F64Array:
        a = np.asarray(da, dtype=np.float64)
        b = np.asarray(db, dtype=np.float64)
        # In Java, this method has an assertion but then uses min length. We follow the runtime behavior.
        n = min(a.shape[0], b.shape[0])
        res = np.zeros(n, dtype=np.float64)
        for i in range(n):
            res[i] = a[i] + b[i]
        return res

    @staticmethod
    def addInPlace(da: F64Array, db: ArrayLike) -> F64Array:
        Math2._require_mutable_f64(da, "da")
        b = np.asarray(db, dtype=np.float64)
        for i in range(min(da.shape[0], b.shape[0])):
            da[i] += b[i]
        return da

    # ==================================================================
    # plus / plusEquals  (SPLIT OVERLOADS -- CONVERSION_GUIDE R4)
    # ==================================================================
    @staticmethod
    def plus_vv(a: ArrayLike, b: ArrayLike) -> F64Array:
        """Vector + vector. Use this when ``b`` should be a vector."""
        aa = np.asarray(a, dtype=np.float64)
        bb = np.asarray(b, dtype=np.float64)
        if aa.shape[0] != bb.shape[0]:
            raise ValueError("Both arguments to the plus operator must be the same length.")
        res = np.zeros(aa.shape[0], dtype=np.float64)
        for i in range(aa.shape[0]):
            res[i] = aa[i] + bb[i]
        return res

    @staticmethod
    def plus_vs(a: ArrayLike, b: float) -> F64Array:
        """Vector + scalar."""
        aa = np.asarray(a, dtype=np.float64)
        res = np.zeros(aa.shape[0], dtype=np.float64)
        for i in range(aa.shape[0]):
            res[i] = aa[i] + b
        return res

    @staticmethod
    def plus(a: ArrayLike, b: Union[ArrayLike, float]) -> F64Array:
        """Dispatcher preserving the Java call style.

        AMBIGUITY HAZARD: a 0-d ndarray dispatches as a scalar, a
        1-element list dispatches as a vector. Prefer ``plus_vv`` /
        ``plus_vs`` at refactor sites where ``b``'s type could drift.
        """
        if np.isscalar(b):
            return Math2.plus_vs(a, float(b))  # type: ignore[arg-type]
        return Math2.plus_vv(a, b)

    @staticmethod
    def plusEquals(a: F64Array, b: ArrayLike) -> F64Array:
        Math2._require_mutable_f64(a, "a")
        bb = np.asarray(b, dtype=np.float64)
        if a.shape[0] != bb.shape[0]:
            raise ValueError("Both arguments to the plus operator must be the same length.")
        for i in range(a.shape[0]):
            a[i] += bb[i]
        return a

    # ==================================================================
    # minus  (SPLIT OVERLOADS)
    # ==================================================================
    @staticmethod
    def minus_vv(a: ArrayLike, b: ArrayLike) -> F64Array:
        aa = np.asarray(a, dtype=np.float64)
        bb = np.asarray(b, dtype=np.float64)
        if aa.shape[0] != bb.shape[0]:
            raise ValueError("Both arguments to the minus operator must be the same length.")
        res = np.zeros(aa.shape[0], dtype=np.float64)
        for i in range(aa.shape[0]):
            res[i] = aa[i] - bb[i]
        return res

    @staticmethod
    def minus_vs(a: ArrayLike, b: float) -> F64Array:
        aa = np.asarray(a, dtype=np.float64)
        res = np.zeros(aa.shape[0], dtype=np.float64)
        for i in range(aa.shape[0]):
            res[i] = aa[i] - b
        return res

    @staticmethod
    def minus(a: ArrayLike, b: Union[ArrayLike, float]) -> F64Array:
        if np.isscalar(b):
            return Math2.minus_vs(a, float(b))  # type: ignore[arg-type]
        return Math2.minus_vv(a, b)

    # ==================================================================
    # dot, cross
    # ==================================================================
    @staticmethod
    def dot(a: ArrayLike, b: ArrayLike) -> float:
        aa = np.asarray(a, dtype=np.float64)
        bb = np.asarray(b, dtype=np.float64)
        if aa.shape[0] != bb.shape[0]:
            raise ValueError("Both arguments to the dot product must be the same length.")
        res = 0.0
        for i in range(aa.shape[0]):
            res += aa[i] * bb[i]
        return res

    @staticmethod
    def cross(a: ArrayLike, b: ArrayLike) -> F64Array:
        aa = np.asarray(a, dtype=np.float64)
        bb = np.asarray(b, dtype=np.float64)
        if aa.shape[0] != 3 or bb.shape[0] != 3:
            raise ValueError("Both arguments to the cross product must be the three-vectors.")
        return np.array([
            (aa[1] * bb[2]) - (aa[2] * bb[1]),
            (aa[2] * bb[0]) - (aa[0] * bb[2]),
            (aa[0] * bb[1]) - (aa[1] * bb[0])
        ], dtype=np.float64)

    # ==================================================================
    # negative  (SPLIT OVERLOADS -- DIFFERENT SEMANTICS!)
    # ==================================================================
    # The two Java methods named `negative` do entirely unrelated
    # things. The dispatcher is the single most dangerous overload in
    # this file: a variable whose type changes silently changes
    # meaning. Use the explicit forms in refactor-prone code.

    @staticmethod
    def negative_arr(a: ArrayLike) -> F64Array:
        """Java: negative(double[]) -- element-wise negation (-a)."""
        aa = np.asarray(a, dtype=np.float64)
        na = np.zeros(aa.shape[0], dtype=np.float64)
        for i in range(aa.shape[0]):
            na[i] = -aa[i]
        return na

    @staticmethod
    def negative_scalar(x: float) -> float:
        """Java: negative(double) -- clamp positives to zero (NOT negation)."""
        return x if x < 0.0 else 0.0

    @staticmethod
    def negative(a: Union[ArrayLike, float]) -> Union[F64Array, float]:
        """Dispatcher. HIGH AMBIGUITY HAZARD -- semantics differ entirely
        between scalar (clamp) and array (negate) inputs.
        """
        if np.isscalar(a):
            return Math2.negative_scalar(float(a))  # type: ignore[arg-type]
        return Math2.negative_arr(a)

    # ==================================================================
    # multiply  (SPLIT OVERLOADS)
    # ==================================================================
    @staticmethod
    def multiply_sv(a: float, b: ArrayLike) -> F64Array:
        """Java: multiply(double, double[]) -- scalar * vector."""
        bb = np.asarray(b, dtype=np.float64)
        res = np.zeros(bb.shape[0], dtype=np.float64)
        for i in range(bb.shape[0]):
            res[i] = a * bb[i]
        return res

    @staticmethod
    def multiply_vv(a: ArrayLike, b: ArrayLike) -> F64Array:
        """Java: multiply(double[], double[]) -- element-wise vector * vector
        (truncates to min length, matching Java)."""
        aa = np.asarray(a, dtype=np.float64)
        bb = np.asarray(b, dtype=np.float64)
        n = min(aa.shape[0], bb.shape[0])
        res = np.zeros(n, dtype=np.float64)
        for i in range(n):
            res[i] = aa[i] * bb[i]
        return res

    @staticmethod
    def multiply(a: Union[ArrayLike, float],
                 b: Union[ArrayLike, float]) -> F64Array:
        if np.isscalar(a):
            return Math2.multiply_sv(float(a), b)  # type: ignore[arg-type]
        return Math2.multiply_vv(a, b)

    @staticmethod
    def timesEquals(a: float, b: F64Array) -> F64Array:
        Math2._require_mutable_f64(b, "b")
        for i in range(b.shape[0]):
            b[i] = a * b[i]
        return b

    # ==================================================================
    # abs  (JAVA-BUG-1: clamps negatives, NOT element-wise abs)
    # ==================================================================
    @staticmethod
    def abs(data: ArrayLike) -> F64Array:
        # JAVA-BUG-1: returns max(x, 0) per element, NOT |x|. Preserved.
        arr = np.asarray(data, dtype=np.float64)
        res = np.zeros(arr.shape[0], dtype=np.float64)
        for i in range(res.shape[0]):
            res[i] = arr[i] if arr[i] > 0.0 else 0.0
        return res

    @staticmethod
    def abs_real(data: ArrayLike) -> F64Array:
        """Strict variant of `abs`: true element-wise absolute value."""
        arr = np.asarray(data, dtype=np.float64)
        res = np.zeros(arr.shape[0], dtype=np.float64)
        for i in range(res.shape[0]):
            res[i] = math.fabs(arr[i])
        return res

    @staticmethod
    def pointBetween(a: ArrayLike, b: ArrayLike, f: float) -> F64Array:
        aa = np.asarray(a, dtype=np.float64)
        bb = np.asarray(b, dtype=np.float64)
        res = np.zeros(aa.shape[0], dtype=np.float64)
        for i in range(res.shape[0]):
            res[i] = aa[i] + ((bb[i] - aa[i]) * f)
        return res

    @staticmethod
    def isUnitVector(a: ArrayLike) -> bool:
        """Java uses Double.MIN_VALUE (smallest positive normal double)
        as tolerance -- effectively demanding exact unity. Mirrored."""
        arr = np.asarray(a, dtype=np.float64)
        return abs(Math2.magnitude(arr) - 1.0) < arr.shape[0] * sys.float_info.min

    @staticmethod
    def angleBetween(a: ArrayLike, b: ArrayLike) -> float:
        aa = np.asarray(a, dtype=np.float64)
        bb = np.asarray(b, dtype=np.float64)
        denom = Math2.magnitude(aa) * Math2.magnitude(bb)
        if denom == 0.0:
            return 0.0
        ac = max(-1.0, min(1.0, Math2.dot(aa, bb) / denom))
        return math.acos(ac) if not math.isnan(ac) else 0.0

    @staticmethod
    def divide(a: ArrayLike, b: float) -> F64Array:
        aa = np.asarray(a, dtype=np.float64)
        res = np.zeros(aa.shape[0], dtype=np.float64)
        for i in range(aa.shape[0]):
            res[i] = aa[i] / b
        return res

    @staticmethod
    def divideEquals(a: F64Array, b: float) -> F64Array:
        Math2._require_mutable_f64(a, "a")
        for i in range(a.shape[0]):
            a[i] = a[i] / b
        return a

    # ==================================================================
    # ebeDivide  (JAVA-BUG-2: modulo on divisor index)
    # ==================================================================
    @staticmethod
    def ebeDivide(a: ArrayLike, b: ArrayLike) -> F64Array:
        # JAVA-BUG-2: uses b[i % len(b)] despite asserting equal length.
        aa = np.asarray(a, dtype=np.float64)
        bb = np.asarray(b, dtype=np.float64)
        assert aa.shape[0] == bb.shape[0]
        res = np.zeros(aa.shape[0], dtype=np.float64)
        for i in range(aa.shape[0]):
            res[i] = aa[i] / bb[i % bb.shape[0]]
        return res

    @staticmethod
    def ebeDivide_strict(a: ArrayLike, b: ArrayLike) -> F64Array:
        """Strict variant: a / b element-wise, equal-length required, no modulo."""
        aa = np.asarray(a, dtype=np.float64)
        bb = np.asarray(b, dtype=np.float64)
        if aa.shape[0] != bb.shape[0]:
            raise ValueError("a and b must have the same length")
        res = np.zeros(aa.shape[0], dtype=np.float64)
        for i in range(aa.shape[0]):
            res[i] = aa[i] / bb[i]
        return res

    # ==================================================================
    # Polynomial roots
    # ==================================================================
    @staticmethod
    def quadraticSolver(a: float, b: float, c: float) -> Optional[F64Array]:
        """Numerically-stable quadratic solver. Returns None if no real roots.

        Java's Math.signum(0) == 0; Python's math.copysign treats +0 as
        positive. We replicate Java exactly via the explicit sign
        ladder below (CONVERSION_GUIDE R8).
        """
        r = (b * b) - (4.0 * a * c)
        if r < 0.0:
            return None
        sign_b = 0.0 if b == 0.0 else math.copysign(1.0, b)
        q = -0.5 * (b + (sign_b * math.sqrt(r)))
        with np.errstate(divide="ignore", invalid="ignore"):
            # Cast to numpy floats to ensure numpy's IEEE-754 division rules (0/0 -> nan)
            return np.array([np.float64(q) / a, np.float64(c) / q], dtype=np.float64)

    @staticmethod
    def cubeRoot(x: float) -> float:
        return -math.pow(-x, 1.0 / 3.0) if x < 0.0 else math.pow(x, 1.0 / 3.0)

    @staticmethod
    def cubicSolver(a: float, b: float, c: float, d: float) -> F64Array:
        f = (((3.0 * c) / a) - ((b * b) / (a * a))) / 3.0
        g = (((2.0 * math.pow(b / a, 3.0)) - ((9.0 * b * c) / (a * a))) + ((27.0 * d) / a)) / 27.0
        h = ((g * g) / 4.0) + (math.pow(f, 3.0) / 27.0)
        if f == 0.0 and g == 0.0 and h == 0.0:
            x = -Math2.cubeRoot(d / a)
            return np.array([x, x, x], dtype=np.float64)
        elif h <= 0:
            i = math.sqrt(((g * g) / 4.0) - h)
            j = Math2.cubeRoot(i)
            val = -(g / (2.0 * i)) if i != 0 else float("nan")
            k = math.acos(val) if -1.0 <= val <= 1.0 else float("nan")
            m = math.cos(k / 3.0)
            n = math.sqrt(3.0) * math.sin(k / 3.0)
            p = -(b / (3.0 * a))
            return np.array([(2.0 * j * m) + p, (-j * (m + n)) + p, (-j * (m - n)) + p], dtype=np.float64)
        else:
            r = -0.5 * g + math.sqrt(h)
            s = Math2.cubeRoot(r)
            t = -0.5 * g - math.sqrt(h)
            u = Math2.cubeRoot(t)
            p = -(b / (3.0 * a))
            return np.array([s + u + p], dtype=np.float64)

    @staticmethod
    def cubicSolver2(a: float, b: float, c: float, d: float) -> F64Array:
        b /= a
        c /= a
        d /= a
        q = (3.0 * c - b * b) / 9.0
        r = (-(27.0 * d) + b * (9.0 * c - 2.0 * (b * b))) / 54.0
        discrim = q * q * q + r * r
        term1 = b / 3.0
        if discrim > 0:
            temp = 1.0 / 3.0
            s = r + math.sqrt(discrim)
            s = -math.pow(-s, temp) if s < 0 else math.pow(s, temp)
            t = r - math.sqrt(discrim)
            t = -math.pow(-t, temp) if t < 0 else math.pow(t, temp)
            return np.array([-term1 + s + t], dtype=np.float64)
        elif discrim == 0.0:
            r13 = -math.pow(-r, (1.0 / 3.0)) if r < 0 else math.pow(r, (1.0 / 3.0))
            return np.array([-term1 + 2.0 * r13, -(r13 + term1), -(r13 + term1)], dtype=np.float64)
        else:
            dum1 = math.acos(r / math.sqrt(-q * -q * -q))
            temp = -term1 + 2.0 * math.sqrt(-q)
            return np.array([
                temp * math.cos(dum1 / 3.0),
                temp * math.cos((dum1 + 2.0 * math.pi) / 3.0),
                temp * math.cos((dum1 + 4.0 * math.pi) / 3.0)
            ], dtype=np.float64)

    @staticmethod
    def polynomial(coeff: ArrayLike, x: float) -> float:
        """Horner evaluation: c[0] + c[1]*x + c[2]*x^2 + ...
        Coefficients are *ascending* (constant first); matches
        numpy.polynomial.polynomial.polyval, NOT legacy numpy.polyval."""
        c = np.asarray(coeff, dtype=np.float64)
        res = c[-1]
        for i in range(c.shape[0] - 2, -1, -1):
            res = (res * x) + c[i]
        return res

    @staticmethod
    def closestTo(vals: ArrayLike, val: float) -> float:
        arr = np.asarray(vals, dtype=np.float64)
        res = arr[0]
        for i in range(1, arr.shape[0]):
            if abs(arr[i] - val) < abs(res - val):
                res = arr[i]
        return res

    @staticmethod
    def solvePoly(coeff: ArrayLike, y: Optional[float] = None) -> F64Array:
        """Solve c[0] + c[1]*x + ... = y (or =0 if y is None).

        Java overloads (with/without y) collapsed via default arg. Only
        orders 1-3 supported; higher orders raise EPQException. Returns
        an empty array (not None) when the polynomial has no real roots
        in the quadratic branch -- callers can iterate uniformly."""
        c = np.asarray(coeff, dtype=np.float64).copy()
        if y is not None:
            c[0] -= y
        cl = 0
        for i in range(c.shape[0], 0, -1):
            if c[i - 1] != 0.0:
                cl = i
                break
        if cl == 2:
            return np.array([-c[0] / c[1]], dtype=np.float64)
        elif cl == 3:
            res = Math2.quadraticSolver(c[2], c[1], c[0])
            if res is None:
                return np.array([], dtype=np.float64)
            return res
        elif cl == 4:
            return Math2.cubicSolver2(c[3], c[2], c[1], c[0])
        else:
            raise EPQException("Analytical solution not available")

    @staticmethod
    def li(x: float) -> float:
        """Logarithmic integral, naive 20-term series. Kept as literal
        port rather than substituting scipy.special.expi(log(x)) so
        numerical output matches Java exactly (the truncation is
        observable)."""
        if x <= 1.0:
            raise ValueError("x>1.0 :" + str(x))
        lx = math.log(x)
        res = math.log(lx) + 0.577215664901532860
        ff = 1.0
        lxp = 1.0
        f = 1.0
        while f < 20.0:
            ff *= f
            lxp *= lx
            res += lxp / (ff * f)
            f += 1.0
        return res

    # ==================================================================
    # bound  (SPLIT OVERLOADS -- semantics differ!)
    # ==================================================================
    @staticmethod
    def bound_double(x: float, x0: float, x1: float) -> float:
        """Java's double overload: both endpoints INCLUSIVE; swaps if
        x0 > x1; NaN passes through unchanged."""
        if x0 > x1:
            t = x0
            x0 = x1
            x1 = t
        if math.isnan(x):
            return x
        return x0 if x < x0 else (x1 if x > x1 else x)

    @staticmethod
    def bound_int(x: int, lowerInc: int, upperExc: int) -> int:
        """Java's int/long overload: upper bound is EXCLUSIVE
        (returns upperExc - 1 when x >= upperExc). No swap."""
        assert lowerInc < upperExc
        return lowerInc if x < lowerInc else (upperExc - 1 if x >= upperExc else x)

    @staticmethod
    def bound(x: Union[float, int],
              lower: Union[float, int],
              upper: Union[float, int]) -> Union[float, int]:
        """Dispatcher. Java semantics differ between int and double forms!"""
        if isinstance(x, float):
            return Math2.bound_double(x, float(lower), float(upper))
        return Math2.bound_int(int(x), int(lower), int(upper))

    @staticmethod
    def positive(x: float) -> float:
        return x if x > 0.0 else 0.0

    @staticmethod
    def negative_scalar(x: float) -> float:
        return x if x < 0.0 else 0.0

    @staticmethod
    def binomialCoefficient(n: int, m: int) -> int:
        """C(n, m). Manual multiply/divide preserves Java's float pipeline
        overflow behaviour for large n (unlike math.comb)."""
        if (n >= m) and (m > 0):
            res = 1.0
            for i in range(m + 1, n + 1):
                res *= i
            for i in range(n - m, 0, -1):
                res /= i
            assert int(res) == round(res), str(res)
            return int(round(res))
        else:
            return 0

    @staticmethod
    def max(da: ArrayLike) -> Union[float, int]:
        """Maximum element. Covers Java's double[], int[], double[][] overloads."""
        a = np.asarray(da)
        res = -sys.float_info.max if a.dtype == np.float64 else int(a.flatten()[0])
        for d in a.flatten():
            if d > res:
                res = d
        return res.item() if hasattr(res, "item") else res

    @staticmethod
    def min(da: ArrayLike) -> Union[float, int]:
        a = np.asarray(da)
        res = sys.float_info.max if a.dtype == np.float64 else int(a.flatten()[0])
        for d in a.flatten():
            if d < res:
                res = d
        return res.item() if hasattr(res, "item") else res

    @staticmethod
    def slice(data: ArrayLike, st: int, length: int) -> F64Array:
        """Length-bounded copy. Raises IndexError on overrun (matches Java)."""
        arr = np.asarray(data, dtype=np.float64)
        if st + length > arr.shape[0]:
            raise IndexError("slice out of bounds")
        res = np.zeros(length, dtype=np.float64)
        res[:] = arr[st : st + length]
        return res

    @staticmethod
    def pNorm(data: ArrayLike, p: float) -> float:
        arr = np.asarray(data, dtype=np.float64)
        res = 0.0
        for element in arr:
            res += math.pow(abs(element), p)
        return math.pow(res, 1.0 / p)

    @staticmethod
    def infinityNorm(data: ArrayLike) -> float:
        arr = np.asarray(data, dtype=np.float64)
        res = 0.0
        for element in arr:
            if res < abs(element):
                res = abs(element)
        return res

    @staticmethod
    def Legendre(x: float, n: int) -> float:
        """Legendre polynomial P_n(x). Java range-restricted to [0, 10];
        hard-coded coefficients preserve exact Java floating-point output."""
        if n == 0:
            return 1.0
        elif n == 1:
            return x
        elif n == 2:
            return 0.5 * (-1.0 + (3.0 * x * x))
        elif n == 3:
            return 0.5 * x * (-3.0 + (5.0 * x * x))
        elif n == 4:
            xx = x * x
            return 0.125 * (3.0 + (xx * (-30.0 + (xx * 35.0))))
        elif n == 5:
            xx = x * x
            return 0.125 * x * (15.0 + (xx * (-70.0 + (xx * 63.0))))
        elif n == 6:
            xx = x * x
            return 0.0625 * (-5.0 + (xx * (105.0 + (xx * (-315.0 + (xx * 231.0))))))
        elif n == 7:
            xx = x * x
            return 0.0625 * x * (-35.0 + (xx * (315.0 + (xx * (-693.0 + (429.0 * xx))))))
        elif n == 8:
            xx = x * x
            return 0.0078125 * (35.0 + (xx * (-1260.0 + (xx * (6930.0 + (xx * (-12012.0 + (xx * 6435.0))))))))
        elif n == 9:
            xx = x * x
            return 0.0078125 * x * (315.0 + (xx * (-4620.0 + (xx * (18018.0 + (xx * (-25740.0 + (xx * 12155.0))))))))
        elif n == 10:
            xx = x * x
            return 0.00390625 * (-63.0 + (xx * (3465.0 + (xx * (-30030.0 + (xx * (90090.0 + (xx * (-109395.0 + (xx * 46189.0))))))))))
        else:
            raise ValueError("Legendre order out of range [0,10].")

    @staticmethod
    def approxEquals(a: float, b: float, frac: float) -> bool:
        assert frac > 0.0
        assert frac < 1.0
        assert abs(a + b) > abs(a)
        return abs(a - b) < (0.5 * abs(a + b) * frac)

    @staticmethod
    def convolve(v: ArrayLike, kernel: ArrayLike) -> F64Array:
        """1-D convolution with edge-replication boundary handling.

        Java's inner loop is cross-correlation (no kernel flip); indices
        out of bounds are clamped to the nearest edge element via
        ``Math2.bound_int``, replicating end values as Java does.
        """
        vv = np.asarray(v, dtype=np.float64)
        kk = np.asarray(kernel, dtype=np.float64)
        assert (kk.shape[0] % 2) == 1
        res = np.zeros(vv.shape[0], dtype=np.float64)
        mid = kk.shape[0] // 2
        for i in range(res.shape[0]):
            for j in range(kk.shape[0]):
                res[i] += kk[j] * vv[int(Math2.bound((i + j) - mid, 0, vv.shape[0]))]
        return res

    @staticmethod
    def toString(vec: ArrayLike, nf: Optional[Callable[[float], str]] = None) -> str:
        """Comma-joined string. ``nf`` is the Java NumberFormat; here we
        accept any callable (e.g. ``lambda x: f"{x:.3f}"``) or None."""
        arr = np.asarray(vec, dtype=np.float64)
        if arr.shape[0] == 0:
            return ""
        formatter = nf if nf is not None else str
        parts = [formatter(arr[0])]
        for i in range(1, arr.shape[0]):
            parts.append(",")
            parts.append(formatter(arr[i]))
        return "".join(parts)

    @staticmethod
    def isNaN(arr: ArrayLike) -> bool:
        for d in np.asarray(arr, dtype=np.float64):
            if math.isnan(d):
                return True
        return False

    # ==================================================================
    # toContinuedFraction  (JAVA-BUG-4: leftover stdout print)
    # ==================================================================
    @staticmethod
    def toContinuedFraction(val: float, tol: float,
                            verbose: bool = True) -> NDArray[np.int64]:
        # JAVA-BUG-4: Java prints intermediate convergents to stdout.
        # We default verbose=True for parity; pass False to suppress.
        res = np.zeros(10, dtype=np.int64)
        num = np.zeros(res.shape[0] + 2, dtype=np.float64)
        den = np.zeros(res.shape[0] + 2, dtype=np.float64)
        num[1] = 1.0
        den[0] = 1.0
        sign = 0.0 if val == 0.0 else math.copysign(1.0, val)
        rem = abs(val)
        for i in range(res.shape[0]):
            res[i] = int(math.floor(rem))
            num[i + 2] = (res[i] * num[i + 1]) + num[i]
            den[i + 2] = (res[i] * den[i + 1]) + den[i]
            if verbose:
                print(num[i + 2] / den[i + 2])
            rem -= res[i]
            if abs((num[i + 2] / den[i + 2]) - abs(val)) < tol:
                res[0] = int(sign * res[0])
                return res[:i + 1].copy()
            rem = 1.0 / rem
        res[0] = int(sign * res[0])
        return res

    @staticmethod
    def toDecimal(cf: ArrayLike) -> float:
        cfa = np.asarray(cf, dtype=np.int64)
        x = float(cfa[-1])
        y = 1.0
        for i in range(cfa.shape[0] - 2, 0, -1):
            oldX = x
            x = (cfa[i] * x) + y
            y = oldX
        return float(cfa[0]) + (y / x) if cfa[0] > 0 else float(cfa[0]) - (y / x)

    @staticmethod
    def toFraction(cf: ArrayLike) -> NDArray[np.int64]:
        cfa = np.asarray(cf, dtype=np.int64)
        x = int(cfa[-1])
        y = 1
        for i in range(cfa.shape[0] - 2, 0, -1):
            oldX = x
            x = (int(cfa[i]) * x) + y
            y = oldX
        if cfa[0] > 0:
            return np.array([(int(cfa[0]) * x) + y, x], dtype=np.int64)
        else:
            return np.array([(int(cfa[0]) * x) - y, x], dtype=np.int64)

    @staticmethod
    def createRowMatrix(vals: ArrayLike) -> JamaMatrix:
        """JAVA-BUG-5: the method name is misleading.

        Java source:
            new Matrix(vals, vals.length)
        Jama's ``new Matrix(double[] vals, int m)`` constructs an
        m-row column-packed matrix; with m == vals.length the result
        is shape (N, 1) -- a column vector, NOT a row matrix.
        Use ``createRowMatrix_strict`` for a true (1, N) row matrix.
        """
        arr = np.asarray(vals, dtype=np.float64)
        return JamaMatrix.from_flat(arr, arr.shape[0])

    @staticmethod
    def createRowMatrix_strict(vals: ArrayLike) -> JamaMatrix:
        """Strict variant: returns shape (1, N) as the name implies."""
        arr = np.asarray(vals, dtype=np.float64)
        return JamaMatrix.from_flat(arr, 1)

    @staticmethod
    def gcd(a: int, b: int) -> int:
        """Recursive Euclidean GCD -- literal Java port (Java lacks math.gcd)."""
        if b == 0:
            return abs(a)
        return Math2.gcd(b, a - (b * (a // b)))

    @staticmethod
    def solveQuadratic(a: float, b: float, c: float) -> F64Array:
        res = Math2.solvePoly(np.array([c, b, a], dtype=np.float64))
        if res is None:
            raise EPQException("No real roots")
        return res

    @staticmethod
    def solveCubic(a: float, b: float, c: float) -> F64Array:
        """Solve monic cubic x^3 + a x^2 + b x + c = 0.
        Distinct routine from cubicSolver/cubicSolver2."""
        q = (a * a - 3.0 * b) / 9.0
        r = (2.0 * a * a * a - 9.0 * a * b + 27.0 * c) / 54.0
        if r * r < q * q * q:
            th = math.acos(r / (q ** 1.5))
            # JAVA-BUG-6: uses -2*q*cos(...); correct Cardano is 2*sqrt(q)*cos(...)
            return np.array([
                -2.0 * q * math.cos(th / 3.0) - a / 3.0,
                -2.0 * q * math.cos((th + 2.0 * math.pi) / 3.0) - a / 3.0,
                -2.0 * q * math.cos((th - 2.0 * math.pi) / 3.0) - a / 3.0,
            ], dtype=np.float64)
        else:
            # CONVERSION_GUIDE R8: replicate Java Math.signum(0) == 0 exactly.
            sign_r = 0.0 if r == 0.0 else math.copysign(1.0, r)
            A = -sign_r * math.pow((abs(r) + math.sqrt((r * r) - (q * q * q))), 1.0 / 3.0)
            B = 0.0 if a == 0.0 else (q / a)  # JAVA-BUG-7: `a` (param) should be `A` (local)
            return np.array([(A + B) - (a / 3.0)], dtype=np.float64)

    @staticmethod
    def solveCubic_strict(a: float, b: float, c: float) -> F64Array:
        """Strict variant of `solveCubic`: fixes JAVA-BUG-7 in the one-real-root branch.
        Uses `q / A` (cube-root local variable) instead of `q / a` (quadratic param).
        JAVA-BUG-6 in the three-real-roots branch is not fixed here (no strict variant).
        """
        q: float = (a * a - 3.0 * b) / 9.0
        r: float = (2.0 * a * a * a - 9.0 * a * b + 27.0 * c) / 54.0
        if r * r < q * q * q:
            th: float = math.acos(r / (q ** 1.5))
            # JAVA-BUG-6 preserved: -2*q*cos(...) rather than 2*sqrt(q)*cos(...)
            return np.array([
                -2.0 * q * math.cos(th / 3.0) - a / 3.0,
                -2.0 * q * math.cos((th + 2.0 * math.pi) / 3.0) - a / 3.0,
                -2.0 * q * math.cos((th - 2.0 * math.pi) / 3.0) - a / 3.0,
            ], dtype=np.float64)
        sign_r: float = 0.0 if r == 0.0 else math.copysign(1.0, r)
        A: float = -sign_r * math.pow(abs(r) + math.sqrt(r * r - q * q * q), 1.0 / 3.0)
        B: float = 0.0 if A == 0.0 else q / A  # Fix: uses A (local), not a (param)
        return np.array([(A + B) - (a / 3.0)], dtype=np.float64)

    @staticmethod
    def findRoot(coeffs: ArrayLike, x1: float, x2: float, xacc: float) -> float:
        """Newton-Raphson with bisection fallback. Kept as literal port because
        callers may rely on the specific failure modes raised here."""
        MAXIT = 100
        c = np.asarray(coeffs, dtype=np.float64)
        deriv = np.zeros(c.shape[0] - 1, dtype=np.float64)
        for i in range(deriv.shape[0]):
            deriv[i] = c[i + 1] * (i + 1)
        fl = Math2.polynomial(c, x1)
        fh = Math2.polynomial(c, x2)
        sig_l = 0.0 if fl == 0.0 else math.copysign(1.0, fl)
        sig_h = 0.0 if fh == 0.0 else math.copysign(1.0, fh)
        if sig_l == sig_h:
            raise EPQException("End points must bracket the root in Math2.findRoot.")
        # FIX-3 (Correctness): Early exit if one of the endpoints is already the root.
        # Moved after the bracket check to fix `test_raises_both_endpoints_zero`.
        if fl == 0.0:
            return x1
        if fh == 0.0:
            return x2
        xl = x1 if fl < 0.0 else x2
        xh = x2 if fl < 0.0 else x1
        rts = 0.5 * (x1 + x2)
        dxold = abs(x2 - x1)
        dx = dxold
        f = Math2.polynomial(c, rts)
        df = Math2.polynomial(deriv, rts)
        for _ in range(MAXIT):
            if (((((rts - xh) * df) - f) * (((rts - xl) * df) - f)) >= 0.0) or (abs(2.0 * f) > abs(dxold * df)):
                dxold = dx
                dx = 0.5 * (xh - xl)
                rts = xl + dx
                if xl == rts:
                    return rts
            else:
                dxold = dx
                dx = f / df
                temp = rts
                rts -= dx
                if temp == rts:
                    return rts
            if abs(dx) < xacc:
                return rts
            f = Math2.polynomial(c, rts)
            df = Math2.polynomial(deriv, rts)
            if f < 0.0:
                xl = rts
            else:
                xh = rts
        raise EPQException("Maximum iteration count exceeded in Math2.rootFind")

    # ==================================================================
    # Tiny vector builders
    # ==================================================================
    @staticmethod
    def v3(x: float, y: float, z: float) -> F64Array:
        return np.array([x, y, z], dtype=np.float64)

    @staticmethod
    def x3(x: float) -> F64Array:
        return np.array([x, 0.0, 0.0], dtype=np.float64)

    @staticmethod
    def y3(y: float) -> F64Array:
        return np.array([0.0, y, 0.0], dtype=np.float64)

    @staticmethod
    def z3(z: float) -> F64Array:
        return np.array([0.0, 0.0, z], dtype=np.float64)

    @staticmethod
    def transpose(mat: ArrayLike) -> NDArray[np.float64]:
        """Returns an independent copy (not a view). Mutations to the result
        do not propagate back to the input."""
        a = np.asarray(mat, dtype=np.float64)
        res = np.zeros((a.shape[1], a.shape[0]), dtype=np.float64)
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                res[j, i] = a[i, j]
        return res