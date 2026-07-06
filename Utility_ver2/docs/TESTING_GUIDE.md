# EPQ Parity Test Harness Guide

Version 2 · 2026-06-24

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
from it and contains only file-specific logic. Reference implementation: `test_parity_math2_ver1_1_3.py`.

### What `_parity_lib.py` provides

| Category | Names |
|---|---|
| **Gating** | `PARITY_ENABLED`, `needs_java` |
| **Setup** | `setup_parity(java_fqn)`, `jclass(fqn)` |
| **Runner** | `run_tests(test_file)` — use in `__main__` block |
| **Tolerances** | All seven `TOL_*` constants |
| **Strategies** | `finite`, `positive`, `nr_arg`, `small`, `vec3`, `vec_n`, `nonzero_vec_n` |
| **Profiles** | `slow` (500 examples, derandomized), `slow_fuzz` (10000, random) |
| **JPype helpers** | `_jarr`, `_to_pylist` |
| **Comparators** | `_close`, `_arr_close`, `_roots_close`, `_bdry_close` |
| **Boundary constants** | `_NAN`, `_INF` |

The library also fixes `sys.path` on import so the test file can
`from <X>_ver1_1_0 import ...` without its own setup. `conftest.py`
in the same directory starts the JVM once at collection time, preventing
the Python 3.14 + JPype access-violation warning from interrupting test
discovery.

### Per-file template

```python
r"""
test_parity_<name>_ver{G}_{N}_{F}.py — parity harness for <Name>_ver{G}_{N}_{F}.py
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

from <Name>_ver{G}_{N}_{F} import <Class> as Py<Class>
from _epq_compat import EPQException, JavaRandom, JamaMatrix

ctx = setup_parity("gov.nist.microanalysis.<Package>.<Class>")
Java<Class> = ctx.java_class

if __name__ == "__main__":
    from _parity_lib import run_tests; run_tests(__file__)
```

### Required test sections

- **`TestConstruction`** — every constructor path, default field values.
  - *Abstract class*: `TestConstruction` is still required. Use a minimal concrete subclass fixture and verify base-class invariants (e.g. `_fitFunctionCount()` returns the expected count, construction does not raise). Do not leave `TestConstruction` absent because the class is abstract.
  - *Java interface with concrete inner classes*: the interface itself has no constructor. Write one `Test<Implementor>` class per concrete inner class. Each must exercise: (a) construction without raising, (b) a key mapping-method call confirming the formula, and (c) `getResult` (or the equivalent UV2-algebra method) — this last point is critical because `getResult` contains UV2 algebra that silently carries R4-violations when left untested.
- **`Test<Behaviour>`** — one class per logical behaviour group.
- **Edge cases** — zero, negative, empty input, NaN/±Inf where relevant.
- **`TestHypothesis`** — `@given` fuzz over the natural input domain.
- **`Test<Class>Parity`**:
  - *Abstract class*: `@pytest.mark.skip(reason="M4: ...")` — test via concrete Python subclasses with analytical ground truth instead.
  - *Concrete class*: `@needs_java` tests for every public method + a `@given` fuzz test.

### Test body templates

**Standard parity** (function exists in both, same input):
```python
@needs_java
@given(positive, positive)
@slow
def test_gammap_parity(self, a, x):
    assert _close(JavaMath2.gammap(a, x), PyMath2.gammap(a, x), TOL_NR_LIB)
```

**Vector reduction with relative tolerance**:
```python
@needs_java
@given(vec_n)
@slow
def test_magnitude_parity(self, v):
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

**Java exception handling** — wrap Java calls in try/except:
```python
@needs_java
@given(st.floats(0.01, 0.99), st.integers(1, 20))
@slow
def test_chiSquaredConfidenceLevel(self, confidence, df):
    try:
        j = JavaMath2.chiSquaredConfidenceLevel(confidence, df)
    except Exception:
        return
    try:
        p = PyMath2.chiSquaredConfidenceLevel(confidence, df)
    except Exception:
        return
    assert _close(j, p, TOL_FINDROOT)
```

**Bug-aware tests.** A `BUG_LEDGER` entry with `has_strict_variant=True` carries
a two-part test obligation (defined in BUG_GUIDE.md). The two test classes:

1. `TestPreservedBugs` — parity test: buggy Java vs buggy Python.
2. `TestStrictVariants` — unit-test the `*_strict` companion (no Java comparison).

> **Pitfall — dead code after a raising call.**
> `TestPreservedBugs` tests that `compute()` *raises*; `TestEdgeCases`/
> `TestStrictVariants` tests that `compute_strict()` *returns correctly*.
> These behaviors are mutually exclusive — never test both in the same
> method body. A common agent mistake is to append a `compute_strict()`
> call underneath the still-present `compute()` call:
>
> ```python
> # WRONG — compute() raises TypeError; the strict line is dead code.
> result = ig.compute(0)
> result = ig.compute_strict(0)   # never reached
> assert result is not None
> ```
>
> ```python
> # RIGHT — two separate tests, one per behavior.
> class TestEdgeCases:
>     def test_zero_samples(self):
>         assert ig.compute_strict(0) is not None
>
> class TestPreservedBugs:
>     def test_zero_samples_raises(self):
>         with pytest.raises(TypeError):
>             ig.compute(0)
> ```
>
> When reviewing an agent-generated repair for a `BUG_LEDGER` failure,
> check that no code appears after an exception-raising call in the same
> test body — it will never execute and the test will still fail.

**`_literal` vs library: two separate Part 2 test entries required.**
When a method has both a `_literal` port and a library-substituted version,
each needs its own `@needs_java` test at the appropriate tolerance:

```python
@given(small)
@slow
def test_erf_lib(self, x):
    # scipy vs Java NR: drift up to ~1e-8; use TOL_NR_LIB.
    assert _close(JavaMath2.erf(x), PyMath2.erf(x), TOL_NR_LIB)

@given(small)
@slow
def test_erf_literal(self, x):
    # Identical NR algorithm on both sides: expect near-exact match.
    assert _close(JavaMath2.erf(x), PyMath2.erf_literal(x), TOL_LITERAL)
```

Do NOT test both through a single entry. They exercise different code paths
at different tolerances. A method that has a `_literal` companion but only
a single parity test is silently undertested.

---

## Testing Disciplines

### Boundary value tables

Hypothesis explores randomly; boundary tables are hand-coded
`(input, expected)` pairs that catch values hypothesis is statistically
unlikely to sample: IEEE-754 specials (NaN, ±Inf, ±0), function
singularities, known-exact algebraic values.

Expected-value sources, in order of preference:

1. **Exact arithmetic** — `binomialCoefficient(5, 2) == 10`.
2. **Closed-form expressions** — `gammaln(0.5) == log(sqrt(pi))`.
3. **High-precision references** — `scipy.special` (~14 digits).

Add a boundary case **after every fuzz finding** (pin it so the
deterministic suite catches it next time) and **when porting a new
function** (identify boundaries before writing the test).

### Read the port before fixing the expected value

A **port-only test** — one with no Java comparison, e.g. a coverage test
that just exercises a method's branch — takes its expected value from
*the port's algorithm*, not from textbook math. Faithfully-ported
algorithms preserve Java's numerical degeneracies, so the mathematically
"obvious" answer can be wrong, and all three expected-value sources above
will hand you that wrong answer with full confidence.

Worked example: `solvePoly([-1.0, 0.0, 1.0])` (x² − 1 = 0) "should"
return ±1. But the port routes through the Numerical-Recipes stable
quadratic, whose `b == 0` branch computes `q = -0.5·(0 + 0·√r) = -0.0`
and returns `[-0.0, +inf]`. The port matches Java exactly — the ±1
assumption was the bug, in the test.

Before adding a port-only test:

1. **Read the method and everything it delegates to** before choosing
   inputs. Look for degenerate branches: `b == 0` in a stable quadratic,
   exact-float `==` guards, preserved bugs, IEEE-754 `0/0`/`x/0` paths.
2. Either pick an input that exercises the target branch *without*
   landing on a degeneracy (use x² − 3x + 2 = 0, `b ≠ 0`), **or** pin the
   degenerate output as the expected value with a comment explaining why
   it is correct.

This is distinct from the BUG_LEDGER discipline: a degeneracy like this
is inherent to the faithful algorithm, not a logged JAVA-BUG, so the
compliance checker will not flag it. The only defense is reading the port.

### Hypothesis profiles

```python
slow      = settings(max_examples=500,   deadline=None, derandomize=True)
slow_fuzz = settings(max_examples=10000, deadline=None)
```

- `slow` is the default. `derandomize=True` pins the seed; identical
  runs give identical results.
- `slow_fuzz` is nightly / on-demand exploration. NOT derandomized.

When `slow_fuzz` finds a new failure: pin the input as a boundary case;
fix the bug or recalibrate the tolerance; re-run `slow` to confirm green.

### `capsys` + hypothesis incompatibility

The pytest `capsys` fixture is incompatible with hypothesis `@given`.
Hypothesis treats every parameter named `capsys` as a strategy argument,
not a pytest fixture, causing a `TypeError` at collection time.

For methods with a JAVA-BUG-N whose effect is printing to stdout
(e.g. JAVA-BUG-4 `toContinuedFraction`), write **two separate tests**:

```python
# Part 1 — plain pytest test; capsys is safe here.
def test_toContinuedFraction_silent_mode(self, capsys):
    PyMath2.toContinuedFraction(3.14159, 1e-6, verbose=False)
    assert capsys.readouterr().out == ""

# Part 2 — hypothesis parity test; accepts Java stdout noise.
@given(st.floats(-100.0, 100.0, ...), st.floats(1e-9, 1e-3))
@slow
def test_toContinuedFraction_parity(self, val, tol):
    j = JavaMath2.toContinuedFraction(float(val), float(tol))
    p = PyMath2.toContinuedFraction(val, tol, verbose=False)
    assert [int(x) for x in j] == [int(x) for x in p]
```

The Java side will emit noise to stdout during the hypothesis test — this
is unavoidable without rerouting `java.lang.System.out`. Accept it.

### FIX-N boundary divergence: two-file discipline

When a port-code fix (FIX-N) changes Python behavior at a specific input
boundary relative to Java, a symmetric two-file update is required:

**Part 1 (boundary table)** — add the boundary input with the FIXED expected
value (not Java's), and comment the test with the FIX-N that produced it:

```python
def test_x1_is_exact_root(self):
    # fl == 0.0: FIX-1 returns x1 immediately; Java loops and converges slowly.
    r = PyMath2.findRoot([-1.0, 1.0], 1.0, 2.0, 1e-12)
    assert _bdry_close(r, 1.0, atol=1e-10)
```

**Part 2 (parity test)** — skip the boundary input with an explicit `return`
guard, citing the FIX-N in a comment:

```python
@given(...)
@slow
def test_findRoot_parity(self, coeffs, x1, x2):
    fl = PyMath2.polynomial(coeffs, x1)
    fh = PyMath2.polynomial(coeffs, x2)
    # Endpoint-is-root cases covered in TestBoundaryFindRoot (FIX-1/FIX-3).
    if fl == 0.0 or fh == 0.0:
        return
    ...
```

Neither file alone is sufficient: Part 1 without Part 2 leaves the
divergence untested in the hypothesis space; Part 2 without Part 1 leaves
the specific boundary uncovered by the deterministic suite.

**Crash-to-sentinel fixes** (ValueError / ZeroDivisionError → NaN / ±Inf):
When FIX-N changes Python from raising an exception to returning a sentinel
value (NaN or ±Inf) to match Java IEEE 754, Part 1 pins the sentinel with
a comment identifying the FIX-N. No Part 2 parity guard is required — Java
never raised on this input, so the divergence was Python-only.

```python
class TestLogBoundary:
    """Boundary pins for FIX-1: _java_log handles non-positive mRandVal."""

    def test_log_negative_sample_returns_nan(self) -> None:
        # _java_log(-1.0) must be NaN, not raise ValueError (FIX-1).
        r = PyUVMC.log(_mk(1.0, -1.0))
        assert math.isnan(r.doubleValue())

    def test_log_zero_sample_returns_neg_inf(self) -> None:
        # _java_log(0.0) must be -Inf, Java IEEE 754 (FIX-1).
        r = PyUVMC.log(_mk(1.0, 0.0))
        assert r.doubleValue() == float("-inf")
```

**`assume`-only fixes**: When FIX-N adds an `assume(cond)` to a hypothesis
test rather than changing port behaviour, a deterministic Part 1 pin is
required *only if the excluded values have a well-defined Python result*.
If the excluded values would raise on Python but compute a valid float on
Java due to intrinsic IEEE 754 / platform divergence (not a port bug),
document the exclusion with a comment in the hypothesis test and omit Part 1:

```python
# FIX-N: exclude subnormal-squared sigma — Python sqrt precision diverges
# from Java at subnormal boundaries. No deterministic pin: Python raises;
# Java returns a valid (but platform-dependent) float.
assume(math.sqrt(sigma * sigma) == sigma)
```

### Accessing port internals in tests

Tests sometimes need to verify internal state (workspace cleared after integration, sentinel fields reset, etc.). Use the **R1-prefixed Python name**, not the Java name:

| Java field | Python attribute | Correct test access |
|---|---|---|
| `private double mWs2` | `self._mWs2` | `assert obj._mWs2 is None` |
| `private double mSaveInterval` | `self._mSaveInterval` | `assert obj._mSaveInterval == math.inf` |
| `private int mNEvals` | `self._mNEvals` | `assert obj._mNEvals == 1` |

`obj.mWs2` (no underscore) raises `AttributeError` — there is no such attribute. The port is correct; the test reference is wrong. Apply this rule to every `private` or `protected` Java field accessed in a test body.

**Exception: `serialVersionUID`.** This field is public by convention (see CONVERSION_GUIDE R1 exception) and is accessed without underscore: `assert Cls.serialVersionUID == 0x1`.

---

## Appendix A: JPype known gotchas (Windows)

### Known gotchas

**Stale JAVA_HOME in long-lived processes.**
```powershell
$env:JAVA_HOME = "C:\Users\you\AppData\Local\Programs\Eclipse Adoptium\jdk-25.x.x-hotspot"
$env:Path = "$env:JAVA_HOME\bin;$env:Path"
```

**`from gov...import` doesn't work.** The `gov.nist.microanalysis.*`
namespace raises `ImportError` even with `registerDomain("gov")`. Use
`jpype.JClass("gov.nist.microanalysis.Utility.Math2")`. `setup_parity`
and `jclass` in `_parity_lib.py` do this for you.

**Java 25 restricted-method warning.** Silenced by passing
`"--enable-native-access=ALL-UNNAMED"` to `startJVM` (done automatically
by `conftest.py`).

**Python 3.14 + JPype access violation at `startJVM`.** Non-fatal on
Windows but interrupts pytest collection if raised during module import.
`conftest.py` starts the JVM once at `pytest_configure` time, before
collection, so all test files are discovered normally.

**JPype cannot extend Java abstract classes from Python (M4).** This
affects parity testing of any EPQ class declared `abstract`.

`@JImplements` only works for Java *interfaces*. Direct Python subclassing
raises `TypeError: Java classes cannot be extended in Python`.

Workarounds (in order of preference):
1. *Test indirectly* — use a concrete Java method that internally instantiates
   the abstract class (e.g. `Math2.chiSquaredConfidenceLevel` drives `FindRoot`).
2. *Compile a concrete Java helper* — write a small subclass, compile it, drop
   the `.jar` into `PyEPQ/lib/`, load via `jclass(...)`.

When neither is viable, mark the parity class with `@pytest.mark.skip`:
```python
@pytest.mark.skip(
    reason="M4: JPype cannot extend abstract Java classes from Python. "
           "Compile a concrete helper or test indirectly via Math2."
)
class TestAbstractClassParity:
    ...
```

**Accessing `private` Java fields via reflection.**
```python
field = JavaClass.class_.getDeclaredField("serialVersionUID")
field.setAccessible(True)
value = int(field.getLong(None))
```

### Where to drop the jar

Discovery order:
1. `$EPQ_JAR` env var.
2. `PyEPQ/lib/epq.jar` or `PyEPQ/lib/EPQ.jar`.
3. `<repo-root>/lib/EPQ.jar`.

All `*.jar` in `PyEPQ/lib/` join the classpath automatically.
