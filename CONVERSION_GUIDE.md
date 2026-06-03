# EPQ Java → Python Conversion Guide

Rulebook for porting `gov.nist.microanalysis` (NIST DTSA-II's EPQ library)
from Java to Python. Designed for AI-agent-driven refactoring.

## How to use

This guide pairs with [CONVERSION_PROMPTS.md](CONVERSION_PROMPTS.md),
which holds the reusable agent prompts. The two files are coupled:

- **CONVERSION_GUIDE.md** (this file) — the *rulebook*. Read by both
  humans and agents to learn how a conversion must be done.
- **CONVERSION_PROMPTS.md** — the *instructions*. Sent to the agent
  at task time. Each prompt references rules and sections in this guide.

Every conversion prompt should include the line:
**"Follow CONVERSION_GUIDE.md R1-R10 and produce the artifacts in Per-File Workflow."**

Files are named `<X>_ver1.py` so Java and Python coexist during
migration; the `_ver1` suffix is dropped in Phase 5.

---

## The Rulebook

### R1 — Naming: preserve Java identifiers verbatim

Keep `camelCase`, `Class.staticMethod(...)` patterns, field and constant
names. Even private helpers preserve their Java spelling — `_doRecompute`,
not `_do_recompute`.

```java
// Java
public class Element {
    public static final double AVOGADRO = 6.022e23;
    private int mAtomicNumber;
    public static Element byAtomicNumber(int z) { ... }
    private void invalidateCache() { ... }
}
```

```python
# Python — names preserved exactly
class Element:
    AVOGADRO: float = 6.022e23

    def __init__(self) -> None:
        self.mAtomicNumber: int = 0  # leading 'm' preserved

    @staticmethod
    def byAtomicNumber(z: int) -> "Element": ...

    def _invalidateCache(self) -> None: ...  # _ for Java 'private'
```

### R2 — Faithful AND complete port

The first pass is a complete literal translation. Two equal requirements:

**(a) Faithful** — no library substitutions, no bug fixes, no "cleaner"
algorithms. Substitute scipy/numpy only after the literal port passes
parity. Each substitution keeps its `_literal` companion.

**(b) Complete** — every Java member appears in the Python port. **Do
not skip "trivial" methods, drop unused-looking fields, or elide inner
classes.**

| Java | Python equivalent |
|---|---|
| `public static final` constant | Class-level annotated assignment |
| `private static final` constant | Class-level, `_`-prefixed |
| Instance field | `__init__` annotated assignment |
| `public` method | `@staticmethod` or instance method |
| `private` / `protected` method | `_underscored` method |
| Static initializer `static { ... }` | Class-body executable code or `_initialize()` |
| Inner class | Nested class |
| Anonymous inner class | Module-level helper or lambda |
| Each overloaded constructor | A `@classmethod` per signature (R4) |
| `equals` / `hashCode` / `toString` | `__eq__` / `__hash__` / `__str__` |

**Verification**: `grep` for `public `, `private `, `protected `,
`static ` in the Java file. The Python port must have the same total
declarations. Mismatched counts mean something was silently dropped.

### R3 — Use the shared compat module

All converted modules import `EPQException`, `JavaRandom`, `JamaMatrix`
from `_epq_compat`. Defining local fallbacks creates incompatible types
and breaks `except` clauses across module boundaries.

```python
# RIGHT
from ._epq_compat import EPQException, JavaRandom, JamaMatrix, F64Array

# WRONG — creates a NEW EPQException incompatible with everyone else's
class EPQException(Exception):
    pass
```

Use shared types in signatures too:

```python
def rotate(p: F64Array, theta: float) -> F64Array: ...
def matrix_solve(a: JamaMatrix, b: JamaMatrix) -> JamaMatrix: ...

def fail_if_negative(x: float) -> None:
    if x < 0:
        raise EPQException(f"x must be non-negative, got {x}")
```

### R4 — Split overloads

Java overloads → distinct type-suffixed functions. A unified dispatcher
MAY be provided for source-style compatibility but must flag the
ambiguity hazard.

| Suffix | Meaning |
|---|---|
| `_vv` | vector + vector |
| `_vs` | vector + scalar |
| `_sv` | scalar + vector |
| `_arr` / `_scalar` | array form / scalar form |
| `_int` / `_double` | integer- or double-keyed overload |

```python
# Java: plus(double[], double[]) AND plus(double[], double)

@staticmethod
def plus_vv(a: ArrayLike, b: ArrayLike) -> F64Array:
    aa: F64Array = np.asarray(a, dtype=np.float64)
    bb: F64Array = np.asarray(b, dtype=np.float64)
    if aa.shape != bb.shape:
        raise ValueError("plus_vv: shape mismatch")
    return aa + bb

@staticmethod
def plus_vs(a: ArrayLike, b: float) -> F64Array:
    return np.asarray(a, dtype=np.float64) + float(b)

@staticmethod
def plus(a: ArrayLike, b: Union[ArrayLike, float]) -> F64Array:
    """Dispatcher. AMBIGUITY HAZARD: 0-d ndarray dispatches as scalar."""
    return Math2.plus_vs(a, b) if np.isscalar(b) else Math2.plus_vv(a, b)
```

For overloads with **different semantics** (e.g. `Math2.negative` negates
arrays but clamps scalars), use fully descriptive suffixes — never
collapse them silently behind a dispatcher.

### R5 — Guard in-place mutations

Any function that mutates a buffer in place must call a guard on its
target argument that checks: `isinstance(arr, np.ndarray)`,
`arr.dtype == np.float64`, and `arr.flags.writeable`. Define it once per
class as `_require_mutable_f64`; see `Math2._require_mutable_f64` for
the reference implementation.

```python
@staticmethod
def addInPlace(target: F64Array, source: ArrayLike) -> F64Array:
    """Java: void addInPlace(double[] target, double[] source). Mutates target."""
    Math2._require_mutable_f64(target, "target")
    src: F64Array = np.asarray(source, dtype=np.float64)
    n: int = min(target.shape[0], src.shape[0])
    target[:n] += src[:n]
    return target
```

Without the guard, `addInPlace([1.0, 2.0, 3.0], ...)` succeeds silently
while mutating nothing.

### R6 — Maintain the bug ledger

Each preserved Java bug gets three artifacts:

1. A `# JAVA-BUG-N` marker at the offending line.
2. An entry in the class-level `BUG_LEDGER` tuple.
3. Where reasonable, a `*_strict` companion that fixes the bug.

**Fixing** a bug (vs preserving it) requires explicit prompt
authorization: *"Fix JAVA-BUG-N as part of this port."* Otherwise
port-then-preserve is the default.

```python
class Math2:
    BUG_LEDGER: tuple = (
        ("JAVA-BUG-1", "abs",
         "Clamps negatives to zero rather than computing |x|. "
         "Use `abs_real()` for true absolute value.", True),
    )

    @staticmethod
    def abs(data: ArrayLike) -> F64Array:
        # JAVA-BUG-1: returns max(x, 0) per element, NOT |x|. Preserved.
        arr: F64Array = np.asarray(data, dtype=np.float64)
        return np.where(arr > 0.0, arr, 0.0)

    @staticmethod
    def abs_real(data: ArrayLike) -> F64Array:
        """Strict variant of abs: true element-wise absolute value."""
        return np.abs(np.asarray(data, dtype=np.float64))
```

Tuple format: `("JAVA-BUG-N", "method_name", "description", has_strict_variant)`.

### R7 — Java integer division

Every Java `/` between integer types becomes Python `//`. This is the
#1 silent-bug source in Java → Python conversions.

```java
int mid = total / 2;       // 7 / 2 == 3
```

```python
# WRONG — Python `/` between ints returns a float
mid = total / 2            # 7 / 2 == 3.5

# RIGHT — `//` for integer floor division, with explicit int annotation
mid: int = total // 2      # 7 // 2 == 3
```

Annotating local variables (R9) makes the lint-rule version of this
check tractable.

### R8 — Floating point sign semantics

Java `Math.signum(0)` returns 0. Python's `math.copysign(x, 0.0)`
returns `+x`. They diverge at zero.

```python
# WRONG — diverges from Java when v == 0
sign = math.copysign(1.0, v)

# RIGHT — replicate Java exactly
sign: float = 0.0 if v == 0.0 else math.copysign(1.0, v)
```

Audit every `Math.signum` in the Java source; greppable.

### R9 — Type hints on EVERY declaration

Annotate:

1. Every function signature (parameters + return type).
2. Every class-level field and constant.
3. Every instance attribute in `__init__`.
4. Every module-level constant.
5. Every local variable whose type isn't obvious from the RHS literal.

Use the narrowest accurate type:
- `NDArray[np.float64]` (aliased as `F64Array`) — never bare `np.ndarray`.
- `int` for Java `int`/`long`.
- `float` for Java `double`.
- `ArrayLike` only at public boundaries that accept lists/tuples.
- `Optional[T]`, `Union[A, B]` (or `A | B`) as needed.

```python
# Module-level constants
F64Array = NDArray[np.float64]                                # type alias
SQRT_PI: float = math.sqrt(math.pi)
X_AXIS: F64Array = np.array([1.0, 0.0, 0.0], dtype=np.float64)

# Class fields
class Spectrum:
    DEFAULT_BIN_COUNT: int = 2048
    _registry: dict[str, "Spectrum"] = {}

    def __init__(self, name: str, bins: int = DEFAULT_BIN_COUNT) -> None:
        self.name: str = name
        self.bins: int = bins
        self.counts: F64Array = np.zeros(bins, dtype=np.float64)
        self._dirty: bool = False

# Function signatures + locals
@staticmethod
def normalize(p: ArrayLike) -> F64Array:
    a: F64Array = np.asarray(p, dtype=np.float64)
    mag: float = float(np.linalg.norm(a))
    if mag == 0.0:
        raise EPQException("cannot normalize zero vector")
    return a / mag
```

Type hints aren't enforced at runtime, but they make R7's lint check
tractable, catch errors in mypy/pyright, and materially improve LLM
extension accuracy.

### R10 — Document deviations explicitly

Any deliberate deviation from Java requires:

1. A comment at the call site.
2. An entry in the file's docstring `CHANGES` section.
3. An entry in `BUG_LEDGER` if observable (see R6 for format).

`Math2.randomDir` is the canonical example:

```python
@staticmethod
def randomDir() -> F64Array:
    """Knop's algorithm for a uniform-on-sphere 3-vector.

    RNG-DEVIATION-1: Java uses Math.random() (independent of rgen);
    Python uses Math2.rgen so a single initializeRandom(seed) call
    determinises every RNG-dependent function in this module. Matched
    seeds between Java and Python produce DIFFERENT outputs.
    """
    ...
```

---

## Per-File Workflow

For each Java file:

**Step 1 — Conversion specification** (planning artifact). Before writing
code, produce `<path>/File_ver1.spec.md` listing:
- Inbound dependencies (Java imports).
- Outbound dependents (`grep` for callers of every public method).
- Public API surface with signatures.
- Overloaded methods (split plan).
- Mutable-output methods (`*Equals`, `addInPlace`, ...).
- Touchpoints into Jama, javax.swing, java.awt, java.io.
- Suspected Java bugs with disposition (preserve / fix / flag).

This is the single most effective control — most port failures come
from decisions made without enumerating callers.

**Step 2 — Literal port.** Translate top to bottom. No library
substitutions. Apply R1, R7, R8. Split overloads (R4). Add mutation
guards (R5).

**Step 3 — Parity harness.** Generate `test_parity_<file>.py` per "The
Parity Harness" section.

**Step 4 — Library substitutions.** Replace hand-rolled algorithms with
scipy/numpy one at a time. Each substitution keeps its `_literal`
companion; the public name resolves to the library version with the
appropriate tolerance.

**Step 5 — Run parity harness + coverage.** All tests must pass. Run
branch coverage and either close any red lines or document why they're
unreachable.

---

## Tolerance Ladder

Use these as defaults; document per-call-site deviations.

| Constant | Value | Use case |
|---|---|---|
| `TOL_EXACT` | 0 | int ops, dtype-preserving sums, equality |
| `TOL_LITERAL` | 1e-14 | literal port vs Java (~1 ULP); same algorithm |
| `TOL_LIB` | 1e-12 | scipy/numpy substitution; same polynomial |
| `TOL_NR_LIB` | 1e-4 | scipy substitution vs Java Numerical Recipes |
| `TOL_COMPOUND` | 1e-10 | iterative / chained operations |
| `TOL_FINDROOT` | 1e-2 | matches Java FindRoot eps=1e-3 |
| `TOL_REL` | 1e-12 | relative tolerance for vector reductions |

`TOL_NR_LIB` is loose because Java's Numerical Recipes ports have
`EPS=3e-7` internally and convergence degrades at large args. The
`_literal` test still validates at `TOL_LITERAL` because both sides
use the same NR algorithm.

For reductions (`sum`, `dot`, `magnitude`, `pNorm`), use
`_close(j, p, TOL_LITERAL, rtol=TOL_REL)`. FLOP-order summation error
is `N² · ε`, which exceeds absolute tolerance for N > ~10.

---

## The Parity Harness

All shared parity-test infrastructure lives in
**`PyEPQ/Utility/tests/_parity_lib.py`**. Per-file test code imports
from it and contains only file-specific logic. Reference implementation:
[`test_parity_math2.py`](Utility/tests/test_parity_math2.py).

### What `_parity_lib.py` provides

| Category | Names |
|---|---|
| **Gating** | `PARITY_ENABLED`, `needs_java` |
| **Setup** | `setup_parity(java_fqn)`, `jclass(fqn)` |
| **Tolerances** | All seven `TOL_*` constants |
| **Strategies** | `finite`, `positive`, `nr_arg`, `small`, `vec3`, `vec_n`, `nonzero_vec_n` |
| **Profiles** | `slow` (500 examples, derandomized), `slow_fuzz` (10000, random) |
| **JPype helpers** | `_jarr`, `_to_pylist` |
| **Comparators** | `_close`, `_arr_close`, `_roots_close`, `_bdry_close` |
| **Boundary constants** | `_NAN`, `_INF` |

The library also fixes `sys.path` on import so the test file can
`from <X>_ver1 import ...` without its own setup.

### Per-file template

Copy verbatim; substitute `<NAME>` / `<Class>` / `<Package>`:

```python
r"""
test_parity_<NAME>.py — parity harness for <NAME>_ver1.py
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, strategies as st

from _parity_lib import (
    setup_parity, jclass, needs_java, PARITY_ENABLED,
    TOL_EXACT, TOL_LITERAL, TOL_LIB, TOL_NR_LIB,
    TOL_COMPOUND, TOL_FINDROOT, TOL_REL,
    finite, positive, nr_arg, small, vec3, vec_n, nonzero_vec_n,
    slow, slow_fuzz,
    _jarr, _to_pylist,
    _close, _arr_close, _roots_close, _bdry_close,
    _NAN, _INF,
)

from <NAME>_ver1 import <Class> as Py<Class>  # noqa: E402
from _epq_compat import EPQException, JavaRandom, JamaMatrix  # noqa: E402

ctx = setup_parity("gov.nist.microanalysis.<Package>.<Class>")
Java<Class> = ctx.java_class
# Add more here only if tests need additional Java classes:
# JavaRandomImpl = jclass("java.util.Random")
```

The gotchas these patterns address (JClass vs `from gov...import`,
`--enable-native-access`, jar discovery, raw docstrings) are
documented in Appendix A.

### Test body templates

**Standard parity** (function exists in both, same input):
```python
@needs_java
@given(positive, positive)
@slow
def test_gammap_parity(a, x):
    assert _close(JavaMath2.gammap(a, x), PyMath2.gammap(a, x), TOL_NR_LIB)
```

**Vector reduction with relative tolerance**:
```python
@needs_java
@given(vec_n)
@slow
def test_magnitude_parity(v):
    assert _close(JavaMath2.magnitude(_jarr(v)), PyMath2.magnitude(v),
                  TOL_LITERAL, rtol=TOL_REL)
```

**Boundary value table** (deterministic regression):
```python
class TestBoundaryScalars:
    @pytest.mark.parametrize("x, expected", [
        (0.0, 0.0),
        (1.0, 0.8427007929497149),
        (_INF, 1.0), (-_INF, -1.0), (_NAN, _NAN),
    ])
    def test_erf(self, x, expected):
        assert _bdry_close(PyMath2.erf(x), expected)
```

**Java exception handling** — wrap Java calls in try/except and skip
when Java raises (algorithmic limitation, not port bug):
```python
@needs_java
@given(st.floats(0.01, 0.99), st.integers(1, 20))
@slow
def test_chiSquaredConfidenceLevel(confidence, df):
    try:
        j = JavaMath2.chiSquaredConfidenceLevel(confidence, df)
    except Exception:
        return
    p = PyMath2.chiSquaredConfidenceLevel(confidence, df)
    assert _close(j, p, TOL_FINDROOT)
```

**Bug-aware tests** (for `BUG_LEDGER` entries with `has_strict_variant=True`):

1. `TestPreservedBugs` — parity test: buggy Java vs buggy Python.
2. `TestStrictVariants` — unit-test the `*_strict` companion (no Java comparison).

### Extending `_parity_lib.py`

Add to the library when a pattern repeats across **≥3 files**. For
one-off patterns, keep them in the per-file test. Over-abstraction
makes test failures harder to diagnose.

---

## Testing Disciplines

### Boundary value tables

Hypothesis explores randomly; boundary tables are hand-coded
`(input, expected)` pairs that catch values hypothesis is statistically
unlikely to sample but where bugs hide: IEEE-754 specials
(NaN, ±Inf, ±0), function singularities, known-exact algebraic values.

Expected-value sources, in order of preference:

1. **Exact arithmetic** — `binomialCoefficient(5, 2) == 10`, `cubeRoot(8) == 2`.
2. **Closed-form expressions** — `gammaln(0.5) == log(sqrt(pi))`.
3. **High-precision references** — `scipy.special` (~14 digits).

Add a boundary case **after every fuzz finding** (so the deterministic
suite catches it next time), **when porting a new function** (identify
boundaries before writing the test), and **when fixing a bug** (the
triggering input becomes a regression test).

### Hypothesis profiles

```python
slow      = settings(max_examples=500,   deadline=None, derandomize=True)
slow_fuzz = settings(max_examples=10000, deadline=None)
```

- `slow` is the default. `derandomize=True` pins the seed; identical
  runs give identical results.
- `slow_fuzz` is nightly / on-demand exploration. NOT derandomized.

Workflow:

1. Day-to-day / CI: `slow`.
2. Nightly: `slow_fuzz`. When it finds a new failure: pin the input
   as a boundary case; fix the bug or recalibrate the tolerance.
3. After a fix: re-run `slow` (confirm green) and `slow_fuzz` (look
   for related edges).

Optional pytest marker: `@pytest.mark.fuzz` + `@slow_fuzz` lets you
run `pytest -m fuzz` or `pytest -m "not fuzz"`. Hypothesis caches
counterexamples in `.hypothesis/examples`; don't commit it.

### Branch coverage

Run periodically, not every commit. Coverage measures Python code
touched — it does NOT verify Java parity.

```powershell
python -m pip install coverage

python -m coverage run --branch --include="*Math2_ver1.py" `
    -m pytest src\gov\nist\microanalysis\PyEPQ\Utility\tests\test_parity_math2.py
python -m coverage report
python -m coverage html  # browsable htmlcov/index.html
```

Sample output:
```
Name             Stmts   Miss Branch BrPart   Cover
Math2_ver1.py      412      3     94      2     99%
```

Target **>95% branch coverage**. <100% is acceptable when missing
branches are defensive guards (mark with `# pragma: no cover` and
document).

---

## Cross-Cutting Patterns

| Concept | Rule |
|---|---|
| RNG | Always `JavaRandom` from `_epq_compat`; never `random.Random`. Python's Mersenne Twister produces a different sequence for the same seed. |
| Matrix returns | Always `JamaMatrix` where Java returned `Jama.Matrix`. Raw ndarray breaks chained `.times(...).solve(...)`. Use `.getArray()` to access the underlying ndarray. |
| Exceptions | Always `EPQException` from `_epq_compat` (see R3). |
| Number formatting | Accept `Callable[[float], str]` where Java accepted `NumberFormat`. |
| Static init order | Cross-class static refs must be resolved lazily (property getters, factory functions, or per-module `_initialize()`). Audit during Step 1. |
| File I/O | Always specify `encoding="utf-8"`. |
| Concurrency | `synchronized` → `threading.Lock`; audit memory-model assumptions case by case. |

---

## Quick-Lookup: rule index

Most-common conversion concerns and which rule addresses each:

| Concern | Rule |
|---|---|
| Naming convention drift | R1 |
| Builtin shadowing | `__all__`; never `import *` |
| `EPQException` fragmentation | R3 + `_epq_compat` |
| Java integer division | R7 |
| `&&` vs `&` on booleans | linting |
| Mutating buffers | R5 + `_require_mutable_f64` |
| Overload collapse | R4 + suffixes |
| Java-compatible RNG | Cross-Cutting Patterns (use `JavaRandom`) |
| `Jama.Matrix` returns | Cross-Cutting Patterns (use `JamaMatrix`) |
| `Math.signum(0)` semantics | R8 |
| Bug ledger maintenance | R6 |

---

## Appendix A: JPype + JVM Setup (Windows)

### Minimum prerequisites

| Component | Minimum | Why |
|---|---|---|
| Python | 3.10+ | Type-hint syntax used by the port |
| JDK | Match epq.jar's class-file version | See matrix below |
| jpype1 | 1.7.x | Java-25-compatible loader |

### Class-file version matrix

Error `UnsupportedClassVersionError: class file version X` means the
JVM is too old:

| Bytecode | Built with | Requires JVM |
|---|---|---|
| 52 | Java 8 | ≥ 8 |
| 55 | Java 11 | ≥ 11 |
| 61 | Java 17 | ≥ 17 |
| 65 | Java 21 | ≥ 21 |
| 69 | Java 25 | ≥ 25 |

Current `epq.jar` is class-file 65 (Java 21). **Recommended install:
Java 25 LTS** — backwards-compatible to all older bytecode, current
LTS, harness baseline.

### Install steps

1. Install JDK 25 from `https://adoptium.net/temurin/releases/?version=25`.
   Check "Set JAVA_HOME variable" and "JavaSoft (Oracle) registry keys".
2. Open a fresh terminal so the new `JAVA_HOME` is inherited.
3. Verify: `java -version` → 25.x; `echo $env:JAVA_HOME` → jdk-25.x.x.
4. `python -m pip install jpype1`
5. Verify JPype sees the right JVM:
   `python -c "import jpype; print(jpype.getDefaultJVMPath())"` should
   print a path under `jdk-25.x.x\bin\server\jvm.dll`.

### Gotchas (and the patterns that address them)

**Stale JAVA_HOME in long-lived processes.** The Claude Code agent (or
any long-running tool) inherits env vars from process start. Override
in the current shell:

```powershell
$env:JAVA_HOME = "C:\Users\you\AppData\Local\Programs\Eclipse Adoptium\jdk-25.x.x-hotspot"
$env:Path = "$env:JAVA_HOME\bin;$env:Path"
```

**JPype 1.7 requires Java ≥ 9.** With Java 8 + JPype 1.7 you get a
Windows fatal access violation at `startJVM`. Install Java 9+
(recommended) or `pip install "jpype1<1.6"`.

**`from gov...import` doesn't work.** JPype's `from <pkg> import` hook
only auto-registers `java`, `javax`, `com`, `org`, `net`. The
`gov.nist.microanalysis.*` namespace raises `ImportError: Java package
'gov' not found` even with `jpype.imports.registerDomain("gov")`.
**Fix**: use `jpype.JClass("gov.nist.microanalysis.Utility.Math2")`.
The library's `setup_parity` and `jclass` helpers do this for you.

**Java 25 restricted-method warning.** Silenced by passing
`"--enable-native-access=ALL-UNNAMED"` to `startJVM` (the library
does this automatically).

**pytest assertion rewriter + JPype.** In some configurations,
pytest's rewriter triggers a JVM crash at startup. Loading classes via
`JClass` (not `from gov.* import`) avoids it.

### Where to drop the jar

Discovery order:

1. `$EPQ_JAR` env var.
2. `PyEPQ/lib/epq.jar` or `PyEPQ/lib/EPQ.jar`.
3. `<repo-root>/lib/EPQ.jar`.

Simplest setup: drop `epq.jar` (and any dependency jars like
`jama-1.0.3.jar`) into `PyEPQ/lib/`. All `*.jar` in that directory
join the classpath automatically.

### Where to find epq.jar

Ships with the [DTSA-II distribution](https://cstl.nist.gov/div837/837.02/epq/dtsa2/).
Look in `<DTSA-II install>/lib/epq.jar`. Building from source: no
Maven/Gradle file in the repo — import as Java project in IntelliJ or
Eclipse, export as JAR.

---

## Appendix B: Known limits of this methodology

Concerns the methodology does NOT fully solve. Listed so an agent
knows when to escalate rather than chase tolerance bumps.

| ID | Issue | Mitigation |
|---|---|---|
| **M1** | Numerical drift from library substitutions | Document slack per call site; use `TOL_NR_LIB` (1e-4) for NR substitutions, `TOL_COMPOUND` for chained ops |
| **M2** | Cross-class static init order | Resolve lazily; audit during Step 1 spec |
| **M3** | Swing / AWT GUI | Not parity-testable; rebuild against PyQt/Tkinter from spec |
| **H1** | Reproducibility of published outputs | Keep Java reference build operational; certify per-publication |
| **H2** | Concurrency beyond simple locks | Java memory-model guarantees on `volatile` / happens-before have no exact Python analogue |
| **H3** | Reflection on enums / annotations | `importlib` + `getattr` cover most cases; enums need bespoke handling |
| **H4** | Date/time arithmetic | Java `Calendar`/`Date` and Python `datetime` have different edge cases; audit DST/timezone case by case |
| **VH1** | Bit-exact FP under `-strictfp` | Usually beneath noise floor; matters only at convergence boundaries |
| **VH2** | Serialised Java objects on disk | Write a Java tool to re-emit as JSON/HDF5, or write a partial deserialiser in Python |
| **VH3** | JNI native code | Re-bridge via cffi/ctypes or replace |
| **U1** | Bug-for-bug exact Monte Carlo sequence | Retire output-pinned tests; use statistical-property tests (mean, variance, KS-distance) |
| **U2** | Third-party Java plugins against EPQ's API | Keep Java EPQ alive in parallel, write Python→Java bridge, or deprecate plugin API |
| **U3** | Performance parity with HotSpot JIT | Port hot loops to Cython/numba/Rust; accept 5-50× slowdown elsewhere |

---

## Appendix C: Current status

**Phase 1 — Foundation** (in progress)
- `_epq_compat` (EPQException, JavaRandom, JamaMatrix) — **DONE**
- `Math2_ver1` — **DONE** (209 tests green, ~99% branch coverage)
- `_parity_lib` — **DONE** at `PyEPQ/Utility/tests/_parity_lib.py`
- `DescriptiveStatistics`, `Histogram`, `FindRoot` — pending

**Phase 2 — EPQLibrary core**: `Element`, `Composition`, `Material`,
`AlgorithmClass`, `AlgorithmUser`, `XRayTransition`. Foundation
everything else depends on; audit static init carefully.

**Phase 3 — EPQLibrary algorithms**: `MassAbsorptionCoefficient`,
`EdgeEnergy`, `FluorescenceYield`, `BremsstrahlungAngularDistribution`,
`FluorescenceCorrection`, `Spectrum`. Most need M1-level tolerance
(1e-10 to 1e-12) due to compounding.

**Phase 4 — EPQTools / GUI**: choose framework (PyQt5 recommended).
Don't translate Swing line-for-line; spec and rebuild. GUI is not
parity-tested.

**Phase 5 — Pythonification**: snake_case sweep, Python idioms, retire
shims (`JamaMatrix` → ndarray; suffixed overloads → typed unions;
`BUG_LEDGER` → regression tests). Drop `_ver1` suffix; archive Java
sources.
