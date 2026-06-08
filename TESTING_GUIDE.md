# EPQ Parity Test Harness Guide

Infrastructure, templates, and disciplines for writing and running parity tests
against the Java EPQ library.

## How to use

This guide covers **parity-test harness construction only**. For conversion
rules (R1-R10, overloads, bug ledger, etc.) see **CONVERSION_GUIDE.md**. For
copy-paste agent prompts see **PROMPTS.md**.

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
unlikely to sample: IEEE-754 specials (NaN, ±Inf, ±0), function
singularities, known-exact algebraic values.

Expected-value sources, in order of preference:

1. **Exact arithmetic** — `binomialCoefficient(5, 2) == 10`.
2. **Closed-form expressions** — `gammaln(0.5) == log(sqrt(pi))`.
3. **High-precision references** — `scipy.special` (~14 digits).

Add a boundary case **after every fuzz finding** (pin it so the
deterministic suite catches it next time) and **when porting a new
function** (identify boundaries before writing the test).

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

### Branch coverage

Run periodically, not every commit.

```powershell
python -m pip install coverage

python -m coverage run --branch --include="*Math2_ver*.py" `
    -m pytest src\gov\nist\microanalysis\PyEPQ\Utility\tests\test_parity_math2.py
python -m coverage report
python -m coverage html  # browsable htmlcov/index.html
```

Target **>95% branch coverage**. Mark unreachable defensive guards with
`# pragma: no cover` and document why they are unreachable.

---

## Appendix A: JPype + JVM Setup (Windows)

### Minimum prerequisites

| Component | Minimum | Why |
|---|---|---|
| Python | 3.10+ | Type-hint syntax used by the port |
| JDK | Match epq.jar's class-file version | See matrix below |
| jpype1 | 1.7.x | Java-25-compatible loader |

### Class-file version matrix

| Bytecode | Built with | Requires JVM |
|---|---|---|
| 52 | Java 8 | ≥ 8 |
| 55 | Java 11 | ≥ 11 |
| 61 | Java 17 | ≥ 17 |
| 65 | Java 21 | ≥ 21 |
| 69 | Java 25 | ≥ 25 |

Current `epq.jar` is class-file 65 (Java 21). **Recommended: Java 25 LTS.**

### Install steps

1. Install JDK 25 from `https://adoptium.net/temurin/releases/?version=25`.
   Check "Set JAVA_HOME variable" and "JavaSoft (Oracle) registry keys".
2. Open a fresh terminal so the new `JAVA_HOME` is inherited.
3. Verify: `java -version` → 25.x; `echo $env:JAVA_HOME` → jdk-25.x.x.
4. `python -m pip install jpype1`
5. Verify: `python -c "import jpype; print(jpype.getDefaultJVMPath())"` should
   print a path under `jdk-25.x.x\bin\server\jvm.dll`.

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
