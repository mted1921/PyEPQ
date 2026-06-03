"""
_parity_lib.py -- shared infrastructure for parity-test harnesses.

Every `test_parity_<file>.py` in this directory imports from here. The
module provides everything that's REUSABLE across parity test files,
so per-file test code can focus on the file-specific test bodies
(boundary tables, tolerance overrides, method-specific quirks) rather
than re-implementing JVM startup, hypothesis profiles, comparators, etc.

What's exported
---------------
Configuration & gating (set once at import time):
  PARITY_ENABLED            True iff EPQ_PARITY=1 in the environment.
  needs_java                pytest.mark.skipif decorator; skips the
                            decorated test when parity prerequisites
                            (env var, jpype1, EPQ jar) aren't met.

Per-file setup (call once at module level):
  setup_parity(java_fqn) -> ParityContext
                            Starts the JVM if needed and returns a
                            context object whose .java_class attribute
                            holds the requested Java class via JClass.
                            Returns a disabled context when parity
                            isn't ready (tests will skip via @needs_java).
  jclass(fqn) -> JClass|None
                            Load additional Java classes after
                            setup_parity. Returns None if disabled.

Tolerance ladder (use the right one per function category):
  TOL_EXACT     0           bit-equal results expected (int ops, exact arithmetic)
  TOL_LITERAL   1e-14       literal port vs Java (~1 ULP); same algorithm
  TOL_LIB       1e-12       scipy/numpy substitution; same polynomial
  TOL_NR_LIB    1e-4        scipy substitution vs Java Numerical Recipes
  TOL_COMPOUND  1e-10       iterative / chained operations
  TOL_FINDROOT  1e-2        matches Java's FindRoot eps=1e-3
  TOL_REL       1e-12       relative tolerance for vector reductions

Hypothesis strategies (generic; combine or override per file):
  finite                    floats in [-1e6, 1e6], no NaN/Inf
  positive                  floats in [1e-3, 1e3]
  nr_arg                    floats in [1e-3, 50]  -- capped where NR
                            substitutions degrade; use for gammap/gammq-
                            like tests against Java NR ports.
  small                     floats in [-10, 10]
  vec3                      length-3 finite float lists
  vec_n                     length 2-20 finite float lists
  nonzero_vec_n             vec_n filtered to at least one |x| > 1e-6

Hypothesis profiles:
  slow        500 examples, derandomized (reproducible CI)
  slow_fuzz   10000 examples, random (nightly exploration)

JPype helpers (require JVM started):
  _jarr(xs)                 Python iterable -> Java double[]
  _to_pylist(jarr)          Java array -> Python list[float]

Comparators (NaN/Inf-aware):
  _close(j, p, atol, rtol)  scalar comparator (j ~ p within
                            atol + rtol*max(|j|,|p|))
  _arr_close(j, p, atol)    element-wise array comparator
  _roots_close(j, p, atol)  order-independent root-set comparator;
                            filters NaN entries, asserts NaN-count agrees.
  _bdry_close(actual, expected, atol, rtol)
                            boundary-value comparator; handles NaN/Inf in
                            `expected` correctly. Use in TestBoundaryValues.

Boundary constants (for hand-coded parametrize tables):
  _NAN, _INF                float("nan"), float("inf")

Side effect on import
---------------------
This module adds its parent directory (`PyEPQ/Utility/`) to sys.path so
test files can `from Math2_ver1 import ...` and `from _epq_compat
import ...` without their own sys.path setup. The parent dir is the
expected layout: parity tests sit in `PyEPQ/<Subpkg>/tests/`, ported
code lives at `PyEPQ/<Subpkg>/`.

Usage in a test file
--------------------
    from _parity_lib import (
        setup_parity, needs_java, slow, _jarr, _close,
        TOL_LITERAL, TOL_LIB, finite, positive, vec_n,
    )
    from Math2_ver1 import Math2 as PyMath2

    ctx = setup_parity("gov.nist.microanalysis.Utility.Math2")
    JavaMath2 = ctx.java_class  # None if parity disabled

    @needs_java
    @given(positive)
    @slow
    def test_erf_parity(x):
        assert _close(JavaMath2.erf(x), PyMath2.erf(x), TOL_LITERAL)
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from hypothesis import settings, strategies as st


# ============================================================================
# Side effect: add the port directory to sys.path so test files can import
# the ported modules and _epq_compat without their own boilerplate.
# Layout assumption: this file lives at PyEPQ/<Subpkg>/tests/_parity_lib.py
# and the port code lives at PyEPQ/<Subpkg>/<X>_ver1.py.
# ============================================================================
_HERE: Path = Path(__file__).resolve().parent
_PORT_DIR: Path = _HERE.parent
if str(_PORT_DIR) not in sys.path:
    sys.path.insert(0, str(_PORT_DIR))


# ============================================================================
# Parity gating
# ============================================================================

PARITY_ENABLED: bool = bool(os.environ.get("EPQ_PARITY"))

try:
    import jpype
    _JPYPE_OK: bool = True
except ImportError:
    _JPYPE_OK = False
    jpype = None  # type: ignore[assignment]

# JAR discovery: env var > PyEPQ/lib/epq.jar > PyEPQ/lib/EPQ.jar > repo lib/EPQ.jar.
# Path math: this file lives at PyEPQ/Utility/tests/, so parents[1] is
# PyEPQ/ and PyEPQ/lib/ is parents[1] / "lib".
_LIB_DIR: Path = _HERE.parents[1] / "lib"
_LOCAL_CANDIDATES: list[Path] = [_LIB_DIR / "epq.jar", _LIB_DIR / "EPQ.jar"]
_FALLBACK_JAR: Path = Path(__file__).resolve().parents[5] / "lib" / "EPQ.jar"
_DEFAULT_JAR: Path = next(
    (p for p in _LOCAL_CANDIDATES if p.is_file()), _FALLBACK_JAR,
)
_JAR_PATH: Path = Path(os.environ.get("EPQ_JAR", str(_DEFAULT_JAR)))
_JAR_OK: bool = _JAR_PATH.is_file()

# Every other *.jar in PyEPQ/lib/ joins the classpath (e.g.
# jama-1.0.3.jar dropped in alongside epq.jar).
_EXTRA_JARS: list[str] = [
    str(p) for p in sorted(_LIB_DIR.glob("*.jar"))
    if _LIB_DIR.is_dir() and p.resolve() != _JAR_PATH.resolve()
]

_PARITY_READY: bool = PARITY_ENABLED and _JPYPE_OK and _JAR_OK

_skip_reason: str = (
    "parity disabled (set EPQ_PARITY=1)" if not PARITY_ENABLED else
    "jpype1 not installed (pip install jpype1)" if not _JPYPE_OK else
    f"EPQ jar not found at {_JAR_PATH} (set EPQ_JAR=...)" if not _JAR_OK else
    "ok"
)
needs_java = pytest.mark.skipif(not _PARITY_READY, reason=_skip_reason)


# ============================================================================
# Tolerance ladder
# ============================================================================
# See CONVERSION_GUIDE.md "Tolerance Ladder" for the discovered rationale
# behind each value. Per-test overrides are normal -- these are starting
# points, not laws.

TOL_EXACT: float = 0.0
TOL_LITERAL: float = 1e-14
TOL_LIB: float = 1e-12
TOL_NR_LIB: float = 1e-4
TOL_COMPOUND: float = 1e-10
TOL_FINDROOT: float = 1e-2
TOL_REL: float = 1e-12


# ============================================================================
# Hypothesis strategies (generic)
# ============================================================================

finite = st.floats(
    allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6,
)
positive = st.floats(
    min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False,
)
# Narrower range for NR-substituted special functions; convergence of
# Java's 100-iteration series degrades sharply beyond a ~50.
nr_arg = st.floats(
    min_value=1e-3, max_value=50.0, allow_nan=False, allow_infinity=False,
)
small = st.floats(
    min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False,
)
vec3 = st.lists(finite, min_size=3, max_size=3)
vec_n = st.lists(finite, min_size=2, max_size=20)
nonzero_vec_n = vec_n.filter(lambda v: any(abs(x) > 1e-6 for x in v))


# ============================================================================
# Hypothesis profiles
# ============================================================================
# `slow` is the default. Derandomized so identical runs give identical
# results -- the harness becomes a regression suite, not a fuzzer.
# `slow_fuzz` is for nightly exploration; when it finds an edge, pin
# the input as a TestBoundaryValues entry so the deterministic suite
# catches it from then on.

slow = settings(max_examples=500, deadline=None, derandomize=True)
slow_fuzz = settings(max_examples=10000, deadline=None)


# ============================================================================
# Parity context + setup
# ============================================================================

@dataclass(frozen=True)
class ParityContext:
    """Returned by setup_parity(). Holds the requested Java class handle.

    `.enabled`     True iff parity prerequisites are satisfied.
    `.java_class`  The Java class loaded via JClass, or None if disabled.

    When `.enabled` is False, the @needs_java decorator skips any test
    that would call into Java, so `.java_class` being None is safe.
    """
    enabled: bool
    java_class: Any | None = None


def setup_parity(java_fqn: str) -> ParityContext:
    """Start the JVM if needed; return a ParityContext.

    Call ONCE at the top of each parity test file, immediately after
    importing the Python port:

        from Math2_ver1 import Math2 as PyMath2
        ctx = setup_parity("gov.nist.microanalysis.Utility.Math2")
        JavaMath2 = ctx.java_class  # None when parity disabled

    JVM startup uses `--enable-native-access=ALL-UNNAMED` (silences
    Java 25+ "restricted method" warning) and adds every *.jar in
    `PyEPQ/lib/` to the classpath.
    """
    if not _PARITY_READY:
        return ParityContext(enabled=False)
    if not jpype.isJVMStarted():
        jpype.startJVM(
            "--enable-native-access=ALL-UNNAMED",
            classpath=[str(_JAR_PATH), *_EXTRA_JARS],
        )
    return ParityContext(
        enabled=True,
        java_class=jpype.JClass(java_fqn),
    )


def jclass(fqn: str):
    """Convenience wrapper around `jpype.JClass`. Use to load additional
    Java classes after `setup_parity`. Returns None if parity is disabled.

        JavaRandomImpl = jclass("java.util.Random")
        DecimalFormat = jclass("java.text.DecimalFormat")
    """
    if not _PARITY_READY:
        return None
    return jpype.JClass(fqn)


# ============================================================================
# JPype helpers
# ============================================================================

def _jarr(xs):
    """Python iterable -> Java double[]. Requires JVM started (i.e. called
    from a @needs_java test, after setup_parity)."""
    return jpype.JArray(jpype.JDouble)(list(xs))


def _to_pylist(jarr) -> list[float]:
    """Java array -> Python list of floats."""
    return [float(x) for x in jarr]


# ============================================================================
# Comparators (NaN/Inf aware)
# ============================================================================

def _close(java_val, py_val, atol: float, rtol: float = 0.0) -> bool:
    """Scalar comparator: |j - p| <= atol + rtol * max(|j|, |p|).

    `rtol` scales with the magnitude of either operand, which matters
    when results are large enough that 1 ULP exceeds the absolute
    tolerance (e.g. magnitude/sum/dot of large vectors).
    """
    if java_val is None and py_val is None:
        return True
    if java_val is None or py_val is None:
        return False
    j, p = float(java_val), float(py_val)
    return abs(j - p) <= atol + rtol * max(abs(j), abs(p))


def _arr_close(java_arr, py_arr, atol: float) -> bool:
    """Element-wise array comparator. Tolerates None vs empty pairing."""
    if java_arr is None and (py_arr is None or len(py_arr) == 0):
        return True
    if java_arr is None or py_arr is None:
        return False
    j: list[float] = _to_pylist(java_arr)
    p: list[float] = list(map(float, py_arr))
    if len(j) != len(p):
        return False
    return all(abs(a - b) <= atol for a, b in zip(j, p))


def _roots_close(java_arr, py_arr, atol: float) -> bool:
    """Order-independent root-set comparator.

    Sorts the finite roots from each side and asserts pairwise closeness.
    NaN entries (preserved-Java IEEE-754 edge cases) are excluded from
    the value comparison but the NaN COUNT must agree across both sides.
    """
    if java_arr is None and (py_arr is None or len(py_arr) == 0):
        return True
    if java_arr is None or py_arr is None:
        return False
    j: list[float] = _to_pylist(java_arr)
    p: list[float] = list(map(float, py_arr))
    if len(j) != len(p):
        return False
    j_finite: list[float] = sorted(x for x in j if math.isfinite(x))
    p_finite: list[float] = sorted(x for x in p if math.isfinite(x))
    j_nans: int = sum(1 for x in j if math.isnan(x))
    p_nans: int = sum(1 for x in p if math.isnan(x))
    if j_nans != p_nans:
        return False
    if len(j_finite) != len(p_finite):
        return False
    return all(abs(a - b) <= atol for a, b in zip(j_finite, p_finite))


def _bdry_close(actual: float, expected: float,
                atol: float = 1e-14, rtol: float = 0.0) -> bool:
    """Boundary-value comparator with explicit NaN/Inf handling.

    Use in TestBoundaryValues tables where `expected` is a hard-coded
    reference (from scipy, an algebraic identity, or a Wolfram lookup).

    Semantics:
      * expected == NaN  -> requires actual is NaN.
      * expected == +Inf -> requires actual is +Inf (sign matters).
      * expected == -Inf -> requires actual is -Inf.
      * otherwise        -> requires actual finite and within atol+rtol.
    """
    if math.isnan(expected):
        return math.isnan(actual)
    if math.isinf(expected):
        return math.isinf(actual) and (actual > 0) == (expected > 0)
    if math.isnan(actual) or math.isinf(actual):
        return False
    return abs(actual - expected) <= atol + rtol * max(
        abs(actual), abs(expected),
    )


# ============================================================================
# Boundary constants
# ============================================================================
# Imported by test files for use in `@pytest.mark.parametrize` tables.

_NAN: float = float("nan")
_INF: float = float("inf")
