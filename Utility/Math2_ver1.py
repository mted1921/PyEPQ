"""
Math2_ver1.py - Python port of gov.nist.microanalysis.Utility.Math2

REVISION HISTORY
----------------
Rev 1: Initial faithful port using scipy/numpy.
Rev 2: Addresses cross-module concerns identified in the EPQ-wide
       migration review (see CONVERSION_GUIDE.md).

CHANGES IN REV 2 (see CONVERSION_GUIDE.md rules)
------------------------------------------------
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
* R2: Numerical Recipes literal ports (`_erf_literal`, `_gammaln_literal`,
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
"""

from __future__ import annotations

import math
import sys
from typing import Optional, Sequence, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import special as _sp_special
from scipy import stats as _sp_stats

# Single source of truth -- no local stand-in classes.
# Three import paths, tried in order:
#   1. Relative import (works when Math2_ver1 is loaded as part of a package).
#   2. Bare-name import (works when the Utility directory is on sys.path
#      directly -- e.g. via the test file's sys.path manipulation).
#   3. Fully-qualified dotted path (works when the source tree is laid out
#      as a Python package with the appropriate __init__.py files).
try:
    from ._epq_compat import EPQException, JavaRandom, JamaMatrix, F64Array
except ImportError:
    try:
        from _epq_compat import EPQException, JavaRandom, JamaMatrix, F64Array  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility._epq_compat import EPQException, JavaRandom, JamaMatrix, F64Array  # type: ignore


# Explicit export list prevents `from Math2_ver1 import *` from shadowing
# Python builtins (sum, max, min, abs, slice). The intended call style
# is `Math2.sum(...)`, not bare `sum(...)`.
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
         "port reproduces this. Discovered via boundary testing. The "
         "one-real-root branch appears correct.", False),
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
    # `v3` is defined further down -- class body executes top-to-bottom,
    # so we materialise these via np.array directly.
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
            raise TypeError(
                f"{name} must be a numpy ndarray (got {type(arr).__name__}); "
                "in-place helpers cannot mutate Python lists or tuples.")
        if arr.dtype != np.float64:
            raise TypeError(
                f"{name} must have dtype float64 (got {arr.dtype}); "
                "wrong-dtype arrays would silently copy.")
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
    def _gammaln_literal(xx: float) -> float:
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
        return s * math.exp(-x + a * math.log(x) - Math2._gammaln_literal(a))

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
        return math.exp(-x + a * math.log(x) - Math2._gammaln_literal(a)) * h

    @staticmethod
    def _gammap_literal(a: float, x: float) -> float:
        assert x >= 0.0
        assert a > 0.0
        if x < a + 1.0:
            return Math2._gser_literal(a, x)
        return 1.0 - Math2._gcf_literal(a, x)

    @staticmethod
    def _gammq_literal(a: float, x: float) -> float:
        assert x >= 0.0
        assert a > 0.0
        if x < a + 1.0:
            return 1.0 - Math2._gser_literal(a, x)
        return Math2._gcf_literal(a, x)

    @staticmethod
    def _erf_literal(x: float) -> float:
        if x < 0.0:
            return -Math2._gammap_literal(0.5, x * x)
        return Math2._gammap_literal(0.5, x * x)

    @staticmethod
    def _erfc_literal(x: float) -> float:
        if x < 0.0:
            return 1.0 + Math2._gammap_literal(0.5, x * x)
        return Math2._gammq_literal(0.5, x * x)

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
        while True:
            x = 2.0 * (Math2.rgen.nextDouble() - 0.5)
            y = 2.0 * (Math2.rgen.nextDouble() - 0.5)
            s = x * x + y * y
            if s <= 1.0:
                break
        z = 2.0 * s - 1.0
        s = math.sqrt((1.0 - z * z) / s)
        return np.array([x * s, y * s, z], dtype=np.float64)

    # ==================================================================
    # Vector geometry
    # ==================================================================
    @staticmethod
    def distance(p1: ArrayLike, p2: ArrayLike) -> float:
        a = np.asarray(p1, dtype=np.float64)
        b = np.asarray(p2, dtype=np.float64)
        assert a.shape == b.shape
        return float(np.linalg.norm(a - b))

    @staticmethod
    def distanceSqr(p1: ArrayLike, p2: ArrayLike) -> float:
        a = np.asarray(p1, dtype=np.float64)
        b = np.asarray(p2, dtype=np.float64)
        assert a.shape == b.shape
        d = a - b
        return float(np.dot(d, d))

    @staticmethod
    def magnitude(p: ArrayLike) -> float:
        return float(np.linalg.norm(np.asarray(p, dtype=np.float64)))

    @staticmethod
    def normalize(p: ArrayLike) -> F64Array:
        a = np.asarray(p, dtype=np.float64)
        return Math2.divide(a, Math2.magnitude(a))

    # ==================================================================
    # Reductions
    # ==================================================================
    @staticmethod
    def sum(da: ArrayLike) -> Union[float, int]:
        """Element sum. Java overloads for int[] and double[]; numpy's
        ``.sum()`` is dtype-preserving so one definition covers both."""
        return np.asarray(da).sum().item()

    # ==================================================================
    # add / addInPlace
    # ==================================================================
    @staticmethod
    def add(da: ArrayLike, db: ArrayLike) -> F64Array:
        a = np.asarray(da, dtype=np.float64)
        b = np.asarray(db, dtype=np.float64)
        n = min(a.shape[0], b.shape[0])
        return a[:n] + b[:n]

    @staticmethod
    def addInPlace(da: F64Array, db: ArrayLike) -> F64Array:
        Math2._require_mutable_f64(da, "da")
        b = np.asarray(db, dtype=np.float64)
        n = min(da.shape[0], b.shape[0])
        da[:n] += b[:n]
        return da

    # ==================================================================
    # plus / plusEquals  (SPLIT OVERLOADS -- CONVERSION_GUIDE R4)
    # ==================================================================
    @staticmethod
    def plus_vv(a: ArrayLike, b: ArrayLike) -> F64Array:
        """Vector + vector. Use this when ``b`` should be a vector."""
        aa = np.asarray(a, dtype=np.float64)
        bb = np.asarray(b, dtype=np.float64)
        if aa.shape != bb.shape:
            raise ValueError(
                "Both arguments to the plus operator must be the same length.")
        return aa + bb

    @staticmethod
    def plus_vs(a: ArrayLike, b: float) -> F64Array:
        """Vector + scalar."""
        return np.asarray(a, dtype=np.float64) + float(b)

    @staticmethod
    def plus(a: ArrayLike, b: Union[ArrayLike, float]) -> F64Array:
        """Dispatcher preserving the Java call style.

        AMBIGUITY HAZARD: a 0-d ndarray dispatches as a scalar, a
        1-element list dispatches as a vector. Prefer ``plus_vv`` /
        ``plus_vs`` at refactor sites where ``b``'s type could drift.
        """
        return Math2.plus_vs(a, b) if np.isscalar(b) else Math2.plus_vv(a, b)

    @staticmethod
    def plusEquals(a: F64Array, b: ArrayLike) -> F64Array:
        Math2._require_mutable_f64(a, "a")
        bb = np.asarray(b, dtype=np.float64)
        if a.shape != bb.shape:
            raise ValueError(
                "Both arguments to the plus operator must be the same length.")
        a += bb
        return a

    # ==================================================================
    # minus  (SPLIT OVERLOADS)
    # ==================================================================
    @staticmethod
    def minus_vv(a: ArrayLike, b: ArrayLike) -> F64Array:
        aa = np.asarray(a, dtype=np.float64)
        bb = np.asarray(b, dtype=np.float64)
        if aa.shape != bb.shape:
            raise ValueError(
                "Both arguments to the plus operator must be the same length.")
        return aa - bb

    @staticmethod
    def minus_vs(a: ArrayLike, b: float) -> F64Array:
        return np.asarray(a, dtype=np.float64) - float(b)

    @staticmethod
    def minus(a: ArrayLike, b: Union[ArrayLike, float]) -> F64Array:
        return Math2.minus_vs(a, b) if np.isscalar(b) else Math2.minus_vv(a, b)

    # ==================================================================
    # dot, cross
    # ==================================================================
    @staticmethod
    def dot(a: ArrayLike, b: ArrayLike) -> float:
        aa = np.asarray(a, dtype=np.float64)
        bb = np.asarray(b, dtype=np.float64)
        if aa.shape != bb.shape:
            raise ValueError(
                "Both arguments to the dot product must be the same length.")
        return float(np.dot(aa, bb))

    @staticmethod
    def cross(a: ArrayLike, b: ArrayLike) -> F64Array:
        aa = np.asarray(a, dtype=np.float64)
        bb = np.asarray(b, dtype=np.float64)
        if aa.shape != (3,) or bb.shape != (3,):
            raise ValueError(
                "Both arguments to the cross product must be the three-vectors.")
        return np.cross(aa, bb)

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
        return -np.asarray(a, dtype=np.float64)

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
        return float(a) * np.asarray(b, dtype=np.float64)

    @staticmethod
    def multiply_vv(a: ArrayLike, b: ArrayLike) -> F64Array:
        """Java: multiply(double[], double[]) -- element-wise vector * vector
        (truncates to min length, matching Java)."""
        aa = np.asarray(a, dtype=np.float64)
        bb = np.asarray(b, dtype=np.float64)
        n = min(aa.shape[0], bb.shape[0])
        return aa[:n] * bb[:n]

    @staticmethod
    def multiply(a: Union[ArrayLike, float],
                 b: Union[ArrayLike, float]) -> F64Array:
        if np.isscalar(a):
            return Math2.multiply_sv(a, b)  # type: ignore[arg-type]
        return Math2.multiply_vv(a, b)

    @staticmethod
    def timesEquals(a: float, b: F64Array) -> F64Array:
        Math2._require_mutable_f64(b, "b")
        b *= a
        return b

    # ==================================================================
    # abs  (JAVA-BUG-1: clamps negatives, NOT element-wise abs)
    # ==================================================================
    @staticmethod
    def abs(data: ArrayLike) -> F64Array:
        # JAVA-BUG-1: returns max(x, 0) per element, NOT |x|. Preserved.
        arr = np.asarray(data, dtype=np.float64)
        return np.where(arr > 0.0, arr, 0.0)

    @staticmethod
    def abs_real(data: ArrayLike) -> F64Array:
        """Strict variant of `abs`: true element-wise absolute value."""
        return np.abs(np.asarray(data, dtype=np.float64))

    @staticmethod
    def pointBetween(a: ArrayLike, b: ArrayLike, f: float) -> F64Array:
        aa = np.asarray(a, dtype=np.float64)
        bb = np.asarray(b, dtype=np.float64)
        return aa + (bb - aa) * f

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
        ac = max(-1.0, min(1.0, float(np.dot(aa, bb) / denom)))
        return math.acos(ac) if not math.isnan(ac) else 0.0

    @staticmethod
    def divide(a: ArrayLike, b: float) -> F64Array:
        return np.asarray(a, dtype=np.float64) / float(b)

    @staticmethod
    def divideEquals(a: F64Array, b: float) -> F64Array:
        Math2._require_mutable_f64(a, "a")
        a /= float(b)
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
        idx = np.arange(aa.shape[0]) % bb.shape[0]
        return aa / bb[idx]

    @staticmethod
    def ebeDivide_strict(a: ArrayLike, b: ArrayLike) -> F64Array:
        """Strict variant: a / b element-wise, equal-length required, no modulo."""
        aa = np.asarray(a, dtype=np.float64)
        bb = np.asarray(b, dtype=np.float64)
        if aa.shape != bb.shape:
            raise ValueError("a and b must have the same length")
        return aa / bb

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
        r = b * b - 4.0 * a * c
        if r < 0.0:
            return None
        sign_b = 0.0 if b == 0.0 else math.copysign(1.0, b)
        q = -0.5 * (b + sign_b * math.sqrt(r))
        # numpy division so q==0 yields inf/nan rather than raising,
        # matching Java's IEEE-754 double division.
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.array([np.float64(q) / a, np.float64(c) / q],
                            dtype=np.float64)

    @staticmethod
    def cubeRoot(x: float) -> float:
        return math.copysign(abs(x) ** (1.0 / 3.0), x)

    @staticmethod
    def cubicSolver(a: float, b: float, c: float, d: float) -> F64Array:
        # JAVA-BUG-3: branches on exact-float h==0. Preserved.
        f = ((3.0 * c) / a - (b * b) / (a * a)) / 3.0
        g = ((2.0 * (b / a) ** 3 - (9.0 * b * c) / (a * a)) + (27.0 * d) / a) / 27.0
        h = (g * g) / 4.0 + (f ** 3) / 27.0
        if f == 0.0 and g == 0.0 and h == 0.0:
            x = -Math2.cubeRoot(d / a)
            return np.array([x, x, x], dtype=np.float64)
        elif h <= 0:
            i = math.sqrt((g * g) / 4.0 - h)
            # JAVA-BUG-3 amplification: for denormal-tiny inputs, g**2
            # can underflow to 0 making i==0; Java IEEE-754 then yields
            # 0/0 -> NaN -> acos(NaN) -> NaN-laden output. Python's
            # math.* raises on 0/0, so short-circuit to match Java.
            if i == 0.0:
                return np.array([np.nan, np.nan, np.nan], dtype=np.float64)
            j = Math2.cubeRoot(i)
            k = math.acos(-(g / (2.0 * i)))
            m = math.cos(k / 3.0)
            n = math.sqrt(3.0) * math.sin(k / 3.0)
            p = -(b / (3.0 * a))
            return np.array([2 * j * m + p,
                             -j * (m + n) + p,
                             -j * (m - n) + p], dtype=np.float64)
        else:
            r = -0.5 * g + math.sqrt(h)
            s = Math2.cubeRoot(r)
            t = -0.5 * g - math.sqrt(h)
            u = Math2.cubeRoot(t)
            p = -(b / (3.0 * a))
            return np.array([s + u + p], dtype=np.float64)

    @staticmethod
    def cubicSolver2(a: float, b: float, c: float, d: float) -> F64Array:
        """Numerically stabler cubic solver (Java's preferred variant)."""
        b /= a
        c /= a
        d /= a
        q = (3.0 * c - b * b) / 9.0
        r = (-(27.0 * d) + b * (9.0 * c - 2.0 * (b * b))) / 54.0
        discrim = q * q * q + r * r
        term1 = b / 3.0

        def _rcbrt(v: float) -> float:
            return -((-v) ** (1.0 / 3.0)) if v < 0 else v ** (1.0 / 3.0)

        if discrim > 0:
            s = _rcbrt(r + math.sqrt(discrim))
            t = _rcbrt(r - math.sqrt(discrim))
            return np.array([-term1 + s + t], dtype=np.float64)
        elif discrim == 0.0:
            r13 = _rcbrt(r)
            return np.array([-term1 + 2.0 * r13,
                             -(r13 + term1),
                             -(r13 + term1)], dtype=np.float64)
        else:
            dum1 = math.acos(r / math.sqrt(-q * -q * -q))
            temp = -term1 + 2.0 * math.sqrt(-q)
            return np.array([
                temp * math.cos(dum1 / 3.0),
                temp * math.cos((dum1 + 2.0 * math.pi) / 3.0),
                temp * math.cos((dum1 + 4.0 * math.pi) / 3.0),
            ], dtype=np.float64)

    @staticmethod
    def polynomial(coeff: ArrayLike, x: float) -> float:
        """Horner evaluation: c[0] + c[1]*x + c[2]*x^2 + ...
        Coefficients are *ascending* (constant first); matches
        numpy.polynomial.polynomial.polyval, NOT legacy numpy.polyval."""
        return float(np.polynomial.polynomial.polyval(
            x, np.asarray(coeff, dtype=np.float64)))

    @staticmethod
    def closestTo(vals: ArrayLike, val: float) -> float:
        arr = np.asarray(vals, dtype=np.float64)
        return float(arr[np.argmin(np.abs(arr - val))])

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
        for i in range(len(c), 0, -1):
            if c[i - 1] != 0.0:
                cl = i
                break
        if cl == 2:
            return np.array([-c[0] / c[1]], dtype=np.float64)
        elif cl == 3:
            res = Math2.quadraticSolver(c[2], c[1], c[0])
            return res if res is not None else np.empty(0, dtype=np.float64)
        elif cl == 4:
            return Math2.cubicSolver2(c[3], c[2], c[1], c[0])
        raise EPQException("Analytical solution not available")

    @staticmethod
    def li(x: float) -> float:
        """Logarithmic integral, naive 20-term series. Kept as literal
        port rather than substituting scipy.special.expi(log(x)) so
        numerical output matches Java exactly (the truncation is
        observable)."""
        if x <= 1.0:
            raise ValueError(f"x>1.0 : {x}")
        lx = math.log(x)
        # Euler-Mascheroni constant gamma.
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
            x0, x1 = x1, x0
        if math.isnan(x):
            return x
        return x0 if x < x0 else (x1 if x > x1 else x)

    @staticmethod
    def bound_int(x: int, lowerInc: int, upperExc: int) -> int:
        """Java's int/long overload: upper bound is EXCLUSIVE
        (returns upperExc - 1 when x >= upperExc). No swap."""
        assert lowerInc < upperExc
        return lowerInc if x < lowerInc else (
            upperExc - 1 if x >= upperExc else x)

    @staticmethod
    def bound(x: Union[float, int],
              lower: Union[float, int],
              upper: Union[float, int]) -> Union[float, int]:
        """Dispatcher. Java semantics differ between int and double forms!"""
        if isinstance(x, float):
            return Math2.bound_double(x, float(lower), float(upper))
        return Math2.bound_int(x, int(lower), int(upper))

    @staticmethod
    def positive(x: float) -> float:
        return x if x > 0.0 else 0.0

    # ``negative(double)`` lives in the combined ``negative()`` dispatcher
    # above as ``negative_scalar``. See that section.

    @staticmethod
    def binomialCoefficient(n: int, m: int) -> int:
        """C(n, m). math.comb is arbitrary-precision; Java's float
        pipeline can overflow for large n."""
        return math.comb(n, m) if n >= m and m > 0 else 0

    @staticmethod
    def max(da: ArrayLike) -> Union[float, int]:
        """Maximum element. Covers Java's double[], int[], double[][] overloads."""
        return np.asarray(da).max().item()

    @staticmethod
    def min(da: ArrayLike) -> Union[float, int]:
        return np.asarray(da).min().item()

    @staticmethod
    def slice(data: ArrayLike, st: int, length: int) -> F64Array:
        """Length-bounded copy. Raises IndexError on overrun (matches Java)."""
        arr = np.asarray(data, dtype=np.float64)
        if st + length > arr.shape[0]:
            raise IndexError("slice out of bounds")
        return arr[st:st + length].copy()

    @staticmethod
    def pNorm(data: ArrayLike, p: float) -> float:
        return float(np.linalg.norm(
            np.asarray(data, dtype=np.float64), ord=p))

    @staticmethod
    def infinityNorm(data: ArrayLike) -> float:
        return float(np.linalg.norm(
            np.asarray(data, dtype=np.float64), ord=np.inf))

    @staticmethod
    def Legendre(x: float, n: int) -> float:
        """Legendre polynomial P_n(x). Java range-restricted to [0, 10];
        we keep the same guard so callers see identical error behaviour."""
        if not 0 <= n <= 10:
            raise ValueError("Legendre order out of range [0,10].")
        return float(_sp_special.eval_legendre(n, x))

    @staticmethod
    def approxEquals(a: float, b: float, frac: float) -> bool:
        assert frac > 0.0
        assert frac < 1.0
        assert abs(a + b) > abs(a)
        return abs(a - b) < 0.5 * abs(a + b) * frac

    @staticmethod
    def convolve(v: ArrayLike, kernel: ArrayLike) -> F64Array:
        """1-D convolution with edge-replication boundary handling.

        ``numpy.convolve(mode='same')`` zero-pads at the boundaries;
        Java replicates the end-members. We pad manually with edge
        values and apply ``np.correlate`` (the Java inner loop is
        cross-correlation, not convolution).
        """
        v_arr = np.asarray(v, dtype=np.float64)
        k_arr = np.asarray(kernel, dtype=np.float64)
        assert k_arr.shape[0] % 2 == 1, "kernel length must be odd"
        mid = k_arr.shape[0] // 2
        padded = np.concatenate([
            np.full(mid, v_arr[0], dtype=np.float64),
            v_arr,
            np.full(mid, v_arr[-1], dtype=np.float64),
        ])
        return np.correlate(padded, k_arr, mode="valid")

    @staticmethod
    def toString(vec: ArrayLike, nf=None) -> str:
        """Comma-joined string. ``nf`` is the Java NumberFormat; here we
        accept any callable (e.g. ``lambda x: f"{x:.3f}"``) or None."""
        arr = np.asarray(vec)
        if arr.shape[0] == 0:
            return ""
        fmt = nf if callable(nf) else (lambda x: format(x))
        return ",".join(fmt(x) for x in arr)

    @staticmethod
    def isNaN(arr: ArrayLike) -> bool:
        return bool(np.any(np.isnan(np.asarray(arr, dtype=np.float64))))

    # ==================================================================
    # toContinuedFraction  (JAVA-BUG-4: leftover stdout print)
    # ==================================================================
    @staticmethod
    def toContinuedFraction(val: float, tol: float,
                            verbose: bool = True) -> NDArray[np.int64]:
        # JAVA-BUG-4: Java prints intermediate convergents to stdout.
        # We default verbose=True for parity; pass False to suppress.
        res = np.zeros(10, dtype=np.int64)
        num = np.zeros(len(res) + 2, dtype=np.float64)
        den = np.zeros(len(res) + 2, dtype=np.float64)
        num[1] = 1.0
        den[0] = 1.0
        sign = math.copysign(1.0, val) if val != 0.0 else 0.0
        rem = abs(val)
        for i in range(len(res)):
            res[i] = int(math.floor(rem))
            num[i + 2] = res[i] * num[i + 1] + num[i]
            den[i + 2] = res[i] * den[i + 1] + den[i]
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
    def toDecimal(cf: Sequence[int]) -> float:
        x = float(cf[-1])
        y = 1.0
        for i in range(len(cf) - 2, 0, -1):
            old_x = x
            x = cf[i] * x + y
            y = old_x
        return cf[0] + (y / x) if cf[0] > 0 else cf[0] - (y / x)

    @staticmethod
    def toFraction(cf: Sequence[int]) -> NDArray[np.int64]:
        x = int(cf[-1])
        y = 1
        for i in range(len(cf) - 2, 0, -1):
            old_x = x
            x = cf[i] * x + y
            y = old_x
        if cf[0] > 0:
            return np.array([cf[0] * x + y, x], dtype=np.int64)
        return np.array([cf[0] * x - y, x], dtype=np.int64)

    @staticmethod
    def createRowMatrix(vals: ArrayLike) -> JamaMatrix:
        """JAVA-BUG-5: the method name is misleading.

        Java source:
            new Matrix(vals, vals.length)
        Jama's ``new Matrix(double[] vals, int m)`` constructs an
        m-row column-packed matrix; with m == vals.length the result
        is shape (N, 1) -- a column vector, NOT a row matrix.

        The first revision of this port returned shape (1, N) per the
        name; that disagreed with the Java reference. Now we match
        Java exactly. Callers wanting a true row matrix should use
        ``createRowMatrix_strict`` below (named for what the method
        actually delivers when fixed).
        """
        arr = np.asarray(vals, dtype=np.float64).reshape(-1, 1)
        return JamaMatrix(arr)

    @staticmethod
    def createRowMatrix_strict(vals: ArrayLike) -> JamaMatrix:
        """Strict variant: returns shape (1, N) as the name implies.
        Use this when you actually want a row matrix; ``createRowMatrix``
        is preserved-with-bug for parity with the Java reference.
        """
        arr = np.asarray(vals, dtype=np.float64).reshape(1, -1)
        return JamaMatrix(arr)

    @staticmethod
    def gcd(a: int, b: int) -> int:
        """math.gcd: C-implemented, iterative, arbitrary precision."""
        return math.gcd(int(a), int(b))

    @staticmethod
    def solveQuadratic(a: float, b: float, c: float) -> F64Array:
        return Math2.solvePoly(np.array([c, b, a], dtype=np.float64))

    @staticmethod
    def solveCubic(a: float, b: float, c: float) -> F64Array:
        """Solve monic cubic x^3 + a x^2 + b x + c = 0.
        Distinct routine from cubicSolver/cubicSolver2."""
        q = (a * a - 3.0 * b) / 9.0
        r = (2.0 * a * a * a - 9.0 * a * b + 27.0 * c) / 54.0
        if r * r < q * q * q:
            th = math.acos(r / (q ** 1.5))
            return np.array([
                -2.0 * q * math.cos(th / 3.0) - a / 3.0,
                -2.0 * q * math.cos((th + 2.0 * math.pi) / 3.0) - a / 3.0,
                -2.0 * q * math.cos((th - 2.0 * math.pi) / 3.0) - a / 3.0,
            ], dtype=np.float64)
        # CONVERSION_GUIDE R8: replicate Java Math.signum(0) == 0 exactly.
        sign_r = 0.0 if r == 0.0 else math.copysign(1.0, r)
        A = -sign_r * (abs(r) + math.sqrt(r * r - q * q * q)) ** (1.0 / 3.0)
        B = 0.0 if a == 0.0 else q / a
        return np.array([A + B - a / 3.0], dtype=np.float64)

    @staticmethod
    def findRoot(coeffs: ArrayLike, x1: float, x2: float, xacc: float) -> float:
        """Newton-Raphson with bisection fallback. Could be a one-liner
        via scipy.optimize.brentq, but kept as literal port because
        callers may rely on the specific failure modes raised here."""
        MAXIT = 100
        coeffs = np.asarray(coeffs, dtype=np.float64)
        deriv = np.array([coeffs[i + 1] * (i + 1)
                          for i in range(len(coeffs) - 1)], dtype=np.float64)
        fl = Math2.polynomial(coeffs, x1)
        fh = Math2.polynomial(coeffs, x2)
        # CONVERSION_GUIDE R8: Java Math.signum(0) == 0. Two zero
        # endpoints therefore HAVE equal signum and DO trigger the
        # bracket error.
        sig_l = 0.0 if fl == 0.0 else math.copysign(1.0, fl)
        sig_h = 0.0 if fh == 0.0 else math.copysign(1.0, fh)
        if sig_l == sig_h:
            raise EPQException(
                "End points must bracket the root in Math2.findRoot.")
        xl = x1 if fl < 0.0 else x2
        xh = x2 if fl < 0.0 else x1
        rts = 0.5 * (x1 + x2)
        dxold = abs(x2 - x1)
        dx = dxold
        f = Math2.polynomial(coeffs, rts)
        df = Math2.polynomial(deriv, rts)
        for _ in range(MAXIT):
            if (((rts - xh) * df - f) * ((rts - xl) * df - f) >= 0.0 or
                    abs(2.0 * f) > abs(dxold * df)):
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
            f = Math2.polynomial(coeffs, rts)
            df = Math2.polynomial(deriv, rts)
            if f < 0.0:
                xl = rts
            else:
                xh = rts
        raise EPQException(
            "Maximum iteration count exceeded in Math2.rootFind")

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
        """numpy returns a VIEW; mutations propagate. Use .copy() if
        the caller needs an isolated buffer."""
        return np.asarray(mat, dtype=np.float64).T
