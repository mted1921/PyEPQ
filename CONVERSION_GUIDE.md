# EPQ Java -> Python Conversion Guide

This document is the rulebook for porting the `gov.nist.microanalysis`
EPQ library from Java to Python. It exists because per-file conversions
done in isolation -- however correct individually -- produce a library
that runs, looks right, and quietly disagrees with the Java reference.

The rules below trade some short-term inconvenience (writing parity
tests, splitting overloads, maintaining ledgers) for long-term
interoperability and verifiability.

## How to use this document

Future LLM prompts for converting any EPQ file should include the
sentence:

> "Follow the rules in CONVERSION_GUIDE.md, sections R1-R10, and
> produce the per-file artifacts described in 'Per-File Workflow'."

That alone resolves most of the recurring mistakes catalogued below.

For human reviewers: every PR adding a converted file should be
checked against the per-file workflow checklist before merging.

---

## The Rulebook

### R1 -- Naming: preserve Java identifiers verbatim

Keep `camelCase`, keep `Class.staticMethod(...)` call patterns. Java
`ParticleSizeAnalyzer.compute(...)` becomes Python
`ParticleSizeAnalyzer.compute(...)`, **not**
`particle_size_analyzer.compute()`.

Phase 5 of the migration re-spells everything to PEP 8 in a single
coordinated sweep; phases 1-4 stay verbatim so diffs against Java
stay legible and parity reviewers can grep both sources for the same
identifier.

### R2 -- Faithful port first

The first pass of any file is a literal translation. **No library
substitutions, no bug fixes, no signature improvements**. Only after
the parity harness is green for a literal port do you start
substituting scipy/numpy/etc. for hand-rolled algorithms.

Each substitution keeps its `_literal` companion alongside, so the
parity harness can continue to validate that the substitution agrees
with the Java algorithm within a documented tolerance.

### R3 -- Use the shared compat module

All converted modules import `EPQException`, `JavaRandom`, `JamaMatrix`
from `_epq_compat`. Defining local fallback classes -- even
temporarily -- creates non-interchangeable types and silently breaks
`except` clauses across module boundaries.

```python
# RIGHT:
from ._epq_compat import EPQException, JavaRandom, JamaMatrix

# WRONG:
try:
    from somewhere import EPQException
except ImportError:
    class EPQException(Exception): pass  # creates a non-interoperable type
```

### R4 -- Split overloads

Java method overloads must be split into distinct type-suffixed
functions (`plus_vv`, `plus_vs`, `bound_double`, `bound_int`, ...).
A unified dispatcher MAY also be provided for source-style
compatibility, but its docstring must flag the ambiguity hazard, and
refactor-prone sites should prefer the suffixed forms.

Suffix conventions:

| Suffix       | Meaning                          |
|--------------|----------------------------------|
| `_vv`        | vector + vector                  |
| `_vs`        | vector + scalar                  |
| `_sv`        | scalar + vector                  |
| `_arr`       | array form                       |
| `_scalar`    | scalar form                      |
| `_int`       | integer-keyed overload           |
| `_double`    | double-keyed overload            |

For overloads with **different semantics** (e.g. `Math2.negative`,
which negates as an array and clamps-to-zero as a scalar), use the
fully descriptive suffixes -- never collapse them silently behind a
dispatcher only.

### R5 -- Guard in-place mutations

Any function that mutates a buffer in place must call
`_require_mutable_f64(arr)` on its target argument. The guard checks:

- `isinstance(arr, np.ndarray)`
- `arr.dtype == np.float64`
- `arr.flags.writeable`

Without it, callers passing lists, tuples, or wrong-dtype arrays
silently get no-op behaviour while expecting mutation. Java's
`double[]` is always a mutable double buffer; the Python contract has
to be enforced at runtime.

### R6 -- Maintain the bug ledger

Each preserved Java bug gets three artifacts:

1. A `# JAVA-BUG-N` marker comment at the offending line.
2. An entry in the class-level `BUG_LEDGER` tuple.
3. Where reasonable, a `*_strict` companion that fixes the bug, so
   downstream code has an escape hatch.

Fixing a bug (rather than preserving it) requires **explicit
authorisation in the conversion prompt**:

> "Fix JAVA-BUG-N as part of this port."

Without that line, port-then-preserve is the default. This prevents
the agent from "improving" code in ways that diverge from the
reference output silently.

`BUG_LEDGER` format:

```python
BUG_LEDGER: tuple = (
    ("JAVA-BUG-N", "method_name", "description", has_strict_variant),
    ...
)
```

### R7 -- Java integer division

Every Java `/` between integer types becomes Python `//`. Floats
unchanged. This is **the #1 silent-bug source** in Java -> Python
conversions and is invisible to visual review. Treat it as a
lint-level concern:

```bash
grep -nE '\b[a-z_]+ */ *[a-z_]+\b' Module_ver1.py   # spot check
```

A future linter rule should flag any `/` where both operands are
type-hinted as `int`.

### R8 -- Floating point sign semantics

Java `Math.signum(0)` returns 0. Python's `math.copysign(x, 0.0)`
returns +x (treating +0 as positive). These diverge precisely when
the input is zero, which is exactly the boundary case Java's authors
often relied on.

Replicate Java exactly with:

```python
sign = 0.0 if v == 0.0 else math.copysign(1.0, v)
```

Audit every `Math.signum` in the Java source; greppable.

### R9 -- Type hints with intent

Type hints in ported code must use the narrowest accurate type:

- `NDArray[np.float64]` (aliased to `F64Array`), not `np.ndarray`.
- `int` for Java `int` / `long` (Python merges them; use `int`).
- `float` for Java `double`.
- `ArrayLike` only at public boundaries where input could be a list.

Hints are not enforced at runtime; the in-place guards (R5)
compensate at the points where it matters.

### R10 -- Document deviations explicitly

Any deliberate deviation from the Java source (semantics, return
type, side effects) requires:

1. A comment at the call site explaining the change.
2. An entry in the file's docstring `CHANGES` section.
3. An entry in `BUG_LEDGER` if it affects observable behaviour
   (Math2's `RNG-DEVIATION-1` for `randomDir` is the canonical
   example).

---

## Per-File Workflow

For each Java file:

### Step 1: Conversion specification (planning artifact)

Before writing any Python, produce a markdown spec listing:

- Inbound dependencies (Java imports).
- Outbound dependents (`grep` for callers of every public method).
- Public API surface (all `public` methods with signatures).
- Overloaded methods (split plan).
- Mutable-output methods (`*Equals`, `addInPlace`, ...).
- Touchpoints into Jama, javax.swing, java.awt, java.io.
- Suspected Java bugs (with disposition: preserve / fix / flag).

**The agent must produce this before writing code.** It is the single
most effective control because most port failures come from
decisions made without enumerating callers.

### Step 2: Literal port

Translate top to bottom. No library substitutions. Apply R1, R7, R8.
Split overloads (R4). Add mutation guards (R5).

### Step 3: Parity harness

Generate `test_parity_<file>.py` that, for every public function:

- Generates random inputs (hypothesis) AND has explicit boundary tables
  (see "Boundary Value Tables" below).
- Calls both the Java implementation (via JPype) and the Python port.
- Asserts equality within the tolerance ladder (see "Tolerance Ladder"
  below). Use `_close(a, b, atol, rtol=...)` so large-magnitude results
  use relative tolerance.

### Step 4: Library substitutions

Replace hand-rolled algorithms with scipy/numpy one at a time. Each
substitution:

- Keeps the literal port as `_<name>_literal`.
- Makes the public name resolve to the library version.
- Documents the appropriate tolerance bucket per the ladder below.

### Step 5: Run parity harness + coverage

All tests must pass before the file is considered converted. Run
branch coverage (see "Coverage Measurement") and either close any
red lines or document why they're unreachable.

### Tolerance Ladder

Discovered empirically during Math2 conversion. Use these as defaults;
document deviations per call site.

| Constant       | Value  | Use case                                          |
|----------------|--------|---------------------------------------------------|
| `TOL_EXACT`    | 0      | int ops, dtype-preserving sums, equality          |
| `TOL_LITERAL`  | 1e-14  | literal port vs Java (~1 ULP); same algorithm     |
| `TOL_LIB`      | 1e-12  | scipy/numpy substitution; same polynomial         |
| `TOL_NR_LIB`   | 1e-4   | scipy substitution vs Java Numerical Recipes      |
| `TOL_COMPOUND` | 1e-10  | iterative / chained operations                    |
| `TOL_FINDROOT` | 1e-2   | matches Java's FindRoot configured eps=1e-3       |
| `TOL_REL`      | 1e-12  | relative tolerance for vector reductions          |

`TOL_NR_LIB` is loose because Java's Numerical Recipes ports have
EPS=3e-7 internally and convergence degrades at large args. The
literal-port `_literal` test still validates at `TOL_LITERAL` because
both sides use the same NR algorithm.

For reductions (sum, dot, magnitude, pNorm), use `_close(j, p,
TOL_LITERAL, rtol=TOL_REL)`. Theoretical FLOP-order summation error is
`N^2 * eps`, which exceeds absolute tolerance for N > ~10.

---

## The Parity Harness

The reference implementation lives at
`PyEPQ/Utility/tests/test_parity_math2.py`. Future test files should
follow its three-layer structure (always-on unit tests, boundary
tables, JVM-backed parity tests) and its module-level setup
verbatim. The patterns below are the result of debugging actual
issues hit during Math2 conversion -- copy the structure rather than
recreating it.

### Module-level setup (copy verbatim)

```python
import os, sys
from pathlib import Path
import pytest

# Make the sibling port (e.g. Math2_ver1) importable without an
# installed package, so the test file works whether pytest runs from
# the repo root or the tests directory.
_UTILITY_DIR = Path(__file__).resolve().parent.parent
if str(_UTILITY_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILITY_DIR))

from Math2_ver1 import Math2 as PyMath2
from _epq_compat import EPQException, JavaRandom, JamaMatrix

# Parity gating.
PARITY_ENABLED = bool(os.environ.get("EPQ_PARITY"))
try:
    import jpype
    _JPYPE_OK = True
except ImportError:
    _JPYPE_OK = False

# JAR discovery: env var > tests/epq.jar > tests/EPQ.jar > repo lib/.
_HERE = Path(__file__).resolve().parent
_LOCAL_CANDIDATES = [_HERE / "epq.jar", _HERE / "EPQ.jar"]
_FALLBACK_JAR = Path(__file__).resolve().parents[5] / "lib" / "EPQ.jar"
_DEFAULT_JAR = next((p for p in _LOCAL_CANDIDATES if p.is_file()),
                    _FALLBACK_JAR)
_JAR_PATH = Path(os.environ.get("EPQ_JAR", str(_DEFAULT_JAR)))
_JAR_OK = _JAR_PATH.is_file()
# Any other *.jar in tests/ (Jama, commons-math, etc.) joins classpath.
_EXTRA_JARS = [str(p) for p in sorted(_HERE.glob("*.jar"))
               if p.resolve() != _JAR_PATH.resolve()]

_PARITY_READY = PARITY_ENABLED and _JPYPE_OK and _JAR_OK
needs_java = pytest.mark.skipif(not _PARITY_READY,
                                 reason="parity gating failed")

# JVM startup. JClass loads bypass JPype's `from <pkg> import` hook,
# which has quirks for non-standard top-level domains like "gov".
if _PARITY_READY:
    if not jpype.isJVMStarted():
        jpype.startJVM(
            "--enable-native-access=ALL-UNNAMED",   # silences Java 25 warning
            classpath=[str(_JAR_PATH), *_EXTRA_JARS],
        )
    JavaMath2 = jpype.JClass("gov.nist.microanalysis.Utility.Math2")
    JavaRandomImpl = jpype.JClass("java.util.Random")
    JDoubleArr = jpype.JArray(jpype.JDouble)
```

**Key patterns and why**:

1. **`JClass` over `from gov...import`**: JPype's import hook only
   auto-registers `java`, `javax`, `com`, `org`, `net`. Adding `gov`
   via `jpype.imports.registerDomain("gov")` did NOT work reliably in
   testing; `JClass()` is the documented stable API.

2. **`--enable-native-access=ALL-UNNAMED`**: silences a "restricted
   method has been called" warning emitted by Java 25+ when JPype's
   native loader runs. Cosmetic but noisy without it.

3. **Local-jar discovery**: dropping `epq.jar` into the tests
   directory is the lowest-friction setup. Avoid requiring callers to
   set `EPQ_JAR` for the common case.

4. **Extra-jar glob**: callers can drop dependency jars
   (`jama-1.0.3.jar`, `commons-math-2.2.jar`) next to `epq.jar` and
   they're picked up automatically.

5. **Hypothesis profiles**:

   ```python
   slow = settings(max_examples=500, deadline=None, derandomize=True)
   slow_fuzz = settings(max_examples=10000, deadline=None)
   ```

   See "Hypothesis Determinism & Fuzz Workflow" below.

### Test body template

```python
@given(st.floats(-100, 100, allow_nan=False),
       st.floats(-100, 100, allow_nan=False))
@slow
@needs_java
def test_dot_parity(a, b):
    j = JavaMath2.dot(JDoubleArr([a, b]), JDoubleArr([b, a]))
    p = PyMath2.dot([a, b], [b, a])
    assert _close(j, p, TOL_LITERAL, rtol=TOL_REL)
```

### Bug-aware testing

Tests for methods listed in `BUG_LEDGER` with `has_strict_variant=True`
should:

1. Parity test the BUGGY public form against Java (catches divergence
   from preserved behaviour).
2. Unit test the `*_strict` variant directly (catches regressions in
   the fixed implementation, no Java comparison).

See `TestPreservedBugs` and `TestStrictVariants` in the reference
harness for the canonical pattern.

### Java exception handling

If Java raises and Python doesn't (or vice versa) for some input,
that's a legitimate parity gap to document, but it's usually not a
port bug -- it's an algorithmic limitation in Java's implementation.
Wrap Java calls in try/except and skip when Java raises:

```python
try:
    j = JavaMath2.chiSquaredConfidenceLevel(confidence, df)
except Exception:
    return  # Java's FindRoot fails on some inputs; not a parity issue
p = PyMath2.chiSquaredConfidenceLevel(confidence, df)
assert _close(j, p, TOL_FINDROOT)
```

### NaN-aware comparators

Java IEEE-754 propagates NaN through degenerate inputs; Python's `math.*`
sometimes raises. The reference harness has `_roots_close` which
filters NaN entries and asserts only that the NaN-count matches.
Reuse it for any solver returning multi-element arrays.

### Performance

JPype startup is multi-second; happens once at module import. Each
JPype round-trip is ~10-100 us. With `max_examples=500`, full Math2
parity suite runs in ~30 s -- short enough for CI. Without
`derandomize=True` the runs aren't reproducible. The nightly
`slow_fuzz` profile (10000 examples) takes ~10-15 minutes.

---

## Boundary Value Tables

Hypothesis explores the input space randomly. Even with
`derandomize=True`, the sampled inputs are a finite subset of all
possible floats. **Boundary tables** are hand-coded `(input,
expected)` pairs that catch the inputs hypothesis is statistically
unlikely to sample but where bugs commonly hide: IEEE-754 special
values (NaN, +/-Inf, +/-0), mathematical boundary points (function
singularities, zeros of derivatives, sign changes), and known-exact
algebraic values.

### Pattern

```python
class TestBoundaryScalars:
    @pytest.mark.parametrize("x, expected", [
        (0.0, 0.0),
        (1.0, 0.8427007929497149),
        (float("inf"), 1.0),
        (-float("inf"), -1.0),
        (float("nan"), float("nan")),
    ])
    def test_erf(self, x, expected):
        assert _bdry_close(PyMath2.erf(x), expected)
```

### Expected-value sources

In order of preference:
1. **Exact arithmetic**: `binomialCoefficient(5, 2) == 10`,
   `gammaln(1) == 0`, `cubeRoot(8) == 2`.
2. **Closed-form expressions**: `gammaln(0.5) == log(sqrt(pi))`,
   `erfc(0) == 1`.
3. **High-precision references**: scipy.special values (~14 digits)
   for transcendentals.
4. **`_bdry_close` for NaN/Inf**: the helper short-circuits on
   `math.isnan(expected)` and `math.isinf(expected)`.

### When to add a boundary case

- After every fuzz finding: if `slow_fuzz` reveals a new edge,
  pin the input into a boundary table so the deterministic suite
  catches it on every run.
- When porting a new function: identify its boundary points before
  writing the test, not after.
- When a bug is fixed: add the input that triggered the bug as a
  regression test.

### Why this complements hypothesis

| Aspect           | Hypothesis              | Boundary tables           |
|------------------|-------------------------|---------------------------|
| Coverage         | Statistical, broad      | Targeted, narrow          |
| Reproducibility  | Random (or derandomized)| Always deterministic      |
| New-edge finding | Yes                     | No                        |
| Regression catch | Unreliable for known    | Guaranteed                |

Use both. Boundary tables are not a replacement for hypothesis; they
guarantee specific cases get tested every run.

---

## Hypothesis Determinism & Fuzz Workflow

Hypothesis is property-based: each run generates fresh random inputs
unless seeded. This makes test results vary between runs even with no
code changes -- exactly what you want for finding bugs, exactly what
you don't want for CI reproducibility. Resolve by maintaining two
profiles.

### The two-profile pattern

```python
slow = settings(max_examples=500, deadline=None, derandomize=True)
slow_fuzz = settings(max_examples=10000, deadline=None)
```

- **`slow`** is the default. `derandomize=True` pins the seed so
  identical runs give identical results. 500 examples covers the
  input space well without dragging CI past a couple minutes.
- **`slow_fuzz`** is for nightly / on-demand exploration. NOT
  derandomized, 20x the examples. Runs ~10-15 minutes on Math2.

### Tagging fuzz tests (optional)

If you want `slow_fuzz` tests to be opt-in via a pytest marker:

```python
@pytest.mark.fuzz
@slow_fuzz
@given(...)
def test_something_fuzz(...):
    ...
```

Then run with `pytest -m fuzz` for fuzz-only or `pytest -m "not fuzz"`
to skip them.

### Workflow

1. **Day-to-day / CI**: run with the default `slow` profile.
   Reproducible, fast enough.
2. **Nightly job**: run with `slow_fuzz`. If a new failure surfaces:
   - Pin the failing input as a boundary table case (see above).
   - Decide whether the failure is a real bug or a tolerance
     miscalibration; fix accordingly.
3. **After a fix**: re-run `slow` to confirm the deterministic suite
   stays green; re-run `slow_fuzz` to look for related edges.

### Cache directory

Hypothesis caches counterexamples in `.hypothesis/examples`. Don't
commit that directory; it's per-machine state. Once it found a
counterexample, hypothesis re-runs that exact input first on every
subsequent run until it passes (then it's purged from the cache).

---

## Coverage Measurement

Branch coverage tells you which code paths your tests actually
exercise. It doesn't catch bugs by itself, but it identifies untested
branches that might harbour them. Run periodically rather than every
commit.

### Setup (one-time)

```powershell
python -m pip install coverage
```

### Usage

```powershell
# Run tests under coverage instrumentation.
python -m coverage run --branch --include="*Math2_ver1.py" `
    -m pytest src\gov\nist\microanalysis\PyEPQ\Utility\tests\test_parity_math2.py

# Console summary.
python -m coverage report

# Browsable HTML report (htmlcov/index.html).
python -m coverage html
```

### Reading the report

```
Name                  Stmts   Miss Branch BrPart   Cover
Math2_ver1.py           412      3     94      2     99%
```

- `Stmts` / `Miss`: executable statements / unhit.
- `Branch` / `BrPart`: branch decision points / partially-hit.
- `Cover`: percentage including branches.

For each unhit line, open the HTML report, find the red highlight,
and either: (a) add a test (preferred for reachable code); (b) mark
with `# pragma: no cover` and document why (for genuinely
unreachable defensive guards).

### Target

Aim for >95% branch coverage. <100% is acceptable when the missing
branches are defensive guards (e.g. "should never happen" error paths
with documented rationale).

### Coverage is not parity

Coverage measures Python code touched. It does NOT verify that
touched code agrees with Java. Use it alongside the parity harness,
not as a replacement.

---

## Bug Ledger Format

Per-file class-level constant:

```python
BUG_LEDGER: tuple = (
    ("JAVA-BUG-N", "method_name", "description", has_strict_variant),
    ...
)
```

In-line marker at the buggy line:

```python
# JAVA-BUG-N: <one-line description>
```

Companion strict variant naming:

```python
@staticmethod
def methodName(...): ...        # buggy, preserved
@staticmethod
def methodName_strict(...): ... # fixed
```

Project-wide ledger: a top-level `BUG_LEDGER.md` aggregates per-file
ledgers for easy review and audit. Regenerate via a small script that
imports each `*_ver1` module and reads its `BUG_LEDGER` attribute.

### How bugs are discovered

Through Math2 conversion we discovered five preserved Java behaviors
and one outright Java bug. Source distribution:

| ID            | How found                                      |
|---------------|------------------------------------------------|
| JAVA-BUG-1    | Reading the Java source (obvious clamp-to-zero)|
| JAVA-BUG-2    | Reading the Java source (modulo despite assert)|
| JAVA-BUG-3    | Reading the Java source (exact-float compare)  |
| JAVA-BUG-4    | Reading the Java source (leftover stdout)      |
| JAVA-BUG-5    | **Parity test caught it** (createRowMatrix    |
|               |  returned wrong shape vs Java)                 |
| JAVA-BUG-6    | **Boundary test caught it** (solveCubic        |
|               |  returned wrong roots for known cubic)         |
| RNG-DEVIATION-1 | Deliberate design choice (randomDir uses rgen)|

Bugs 5 and 6 were not visible from reading either the Java or Python
source side by side -- the divergence only surfaced once the harness
forced a comparison against concrete inputs. This is the strongest
argument for running every converted file through the full harness
before declaring it done.

---

## Cross-Cutting Patterns

### RNG: use JavaRandom

Always `from _epq_compat import JavaRandom`. **Never** use
`random.Random` for any code path whose output is published, cited,
or test-pinned. Python's Mersenne Twister produces a different
sequence for the same seed; that breaks parity tests and reproduction
of historical runs.

### Matrix: use JamaMatrix

Anywhere the Java code returned `Jama.Matrix`, return `JamaMatrix`.
The underlying ndarray is accessible via `.getArray()` for native
interop. **Do not** return a raw ndarray where the Java signature
returned a Matrix -- it breaks chained `.times(...).solve(...)`
calls without any compile-time warning.

### Exceptions: use EPQException

Always `from _epq_compat import EPQException`. Never define a local
fallback; see R3.

### Number formatting: callable, not NumberFormat

Java `NumberFormat` has no clean Python equivalent. Accept a
callable (`Callable[[float], str]`) where Java accepted a
`NumberFormat`. Document the change in the method's docstring.

### Static initialisation order

Java initialises all static fields of a class before any instance
exists and respects cross-class static dependencies via the
classloader. Python class bodies execute top-to-bottom and have no
classloader; cross-class static references must be resolved lazily
(property getters, factory functions, or a per-module
`_initialize()` call invoked at import time).

Audit the conversion spec (Step 1) for inter-class static references
before writing code.

### File I/O

Java byte streams become Python `bytes`; Java character streams
become Python `str`. **Always specify `encoding=`** when opening
text files. Java's default encoding depends on platform; Python's
defaults to locale. Hard-code `encoding="utf-8"` unless the source
file documents otherwise.

### Concurrency

Java `synchronized` -> Python `threading.Lock`, but Java's memory
model and Python's GIL differ in nuance. Audit every `synchronized`
block for read-after-write hazards. For CPython-only deployments,
many synchronisations can be elided thanks to the GIL; do so only
with an explicit note in the code.

---

## Roadblocks Ranked by Difficulty

### Trivial (mechanical, one-shot fix)

- **T1** Naming convention drift -- R1 covers this.
- **T2** Builtin shadowing -- `__all__` lists; never `import *`.
- **T3** EPQException fragmentation -- R3 + `_epq_compat`.
- **T4** Java integer division -- R7 + linting.
- **T5** `&&` vs `&` on booleans -- linting.

### Easy (requires care; well-defined fix)

- **E1** Mutating buffers -- R5 + guards.
- **E2** Overload collapse -- R4 + suffixes.
- **E3** Java-compatible RNG -- `JavaRandom` done. The `randomDir`
  deviation (`RNG-DEVIATION-1`) is documented; pattern reusable.
- **E4** Jama -> numpy -- `JamaMatrix` shim covers the common subset.
- **E5** Bug ledger maintenance -- process; needs reviewer discipline.
- **E6** `Math.signum(0)` semantics -- R8 codified in helper pattern.

### Moderate (judgement calls; scope can grow)

- **M1** Numerical drift from library substitutions. Recommended
  tolerance policy: 1e-15 for literal ports, 1e-12 for substitutions
  unless documented. Compound calls (Monte Carlo, iterative fits)
  may need 1e-10 or weaker; document the slack per call site.
- **M2** Static initialisation order across class hierarchies.
  Audit during the Step 1 spec; use lazy properties or explicit
  `_initialize()` where needed.
- **M3** File I/O encoding -- always explicit, never platform default.
- **M4** Swing / AWT GUI code. No parity test possible; rebuild
  against PyQt or Tkinter from spec rather than translating
  line-for-line. Treat as a Phase 4 concern.

### Hard (subtle, broad blast radius)

- **H1** Reproducibility of published outputs. Drifts compound through
  Monte Carlo loops. Keep the Java reference build operational
  indefinitely and certify per-publication, not per-release.
- **H2** Concurrent code beyond simple locks. Java memory model
  guarantees on `volatile`, happens-before, etc. have no exact
  Python analogue. Audit case by case.
- **H3** Reflection. `Class.forName` + `Method.invoke` -> `importlib`
  + `getattr`, but enums and annotations need bespoke handling.
- **H4** Date/time arithmetic. Java `Calendar`/`Date` have notorious
  edge cases; Python `datetime` has different ones. Audit DST and
  timezone handling per call site.

### Very Hard

- **VH1** Bit-exact floating point under `-strictfp`. JVMs without
  `-strictfp` may use extended-precision intermediates; Python is
  pure IEEE-754 double. Usually beneath the noise floor; matters
  only at convergence boundaries of iterative fits.
- **VH2** Serialised Java objects on disk. `Serializable` is a full
  object-graph encoding (vtable, private state). No Python library
  reads it natively. Options:
  (a) write a Java tool that re-emits as JSON / HDF5 / Parquet;
  (b) write a partial deserialiser in Python for the specific
  classes you care about (much more work than it sounds).
- **VH3** JNI native code. Anything calling JNI must be re-bridged
  via cffi/ctypes or replaced entirely.

### Essentially unsolvable (or solvable only by keeping the JVM around)

- **U1** Bug-for-bug exact match for every Java floating point sequence.
  Sufficient drift will accumulate over millions of Monte Carlo
  iterations that "this seed produces exactly that output" tests
  cannot be satisfied. The pragmatic answer: retire output-pinned
  tests in favour of statistical-property tests (mean, variance,
  KS-distance vs reference distribution).
- **U2** Third-party Java plugins written against EPQ's Java API.
  They cannot run against the Python port. Solutions:
  (a) keep a Java EPQ build alive in parallel forever;
  (b) write a Python -> Java bridge so plugins still load;
  (c) deprecate the plugin API and migrate plugins one by one.
- **U3** Performance parity for code that JIT-compiles well in HotSpot
  but doesn't vectorise cleanly in numpy. Acceptable answer: port
  the hot inner loops to Cython, numba, or Rust via PyO3. Accept
  that a pure Python port will sometimes be 5-50x slower.

---

## Migration Phases

### Phase 1: Foundation

- `_epq_compat` (EPQException, JavaRandom, JamaMatrix). **DONE.**
- `Math2_ver1`. **DONE.** Full parity suite green: 207+ tests
  (boundary + property + parity), ~99% branch coverage.
- Parity harness infrastructure (JPype config, fixtures, env gating).
  **DONE** at `PyEPQ/Utility/tests/test_parity_math2.py`; use as
  reference template for all subsequent test files.
- `DescriptiveStatistics`, `Histogram`, `FindRoot`, simple utilities.
  Pending.

### Phase 2: EPQLibrary core

- `Element`, `Composition`, `Material`.
- `AlgorithmClass`, `AlgorithmUser`.
- `XRayTransition` family.

These are the foundation everything else depends on. Audit static
initialisation order carefully.

### Phase 3: EPQLibrary algorithms

- `MassAbsorptionCoefficient`, `EdgeEnergy`, `FluorescenceYield`.
- `BremsstrahlungAngularDistribution`, `FluorescenceCorrection`.
- `Spectrum` classes.

Each algorithm gets a parity test against the Java reference. Most
will need M1-level tolerance (1e-10 to 1e-12) due to compounding.

### Phase 4: EPQTools / GUI

- Decide framework. PyQt5 recommended for desktop parity with the
  current Swing UI. Spec, design, build -- do not translate
  Swing line-for-line. GUI is not parity-tested.

### Phase 5: Pythonification

- Convert camelCase to snake_case in a single coordinated sweep
  (script + manual review).
- Add Python idioms (iterators, context managers, dataclasses).
- Retire shim layers: `JamaMatrix` -> direct ndarray; suffixed
  overloads -> single well-typed function; `BUG_LEDGER` -> regression
  tests for fixed-only behaviour.

After Phase 5, the `_ver1` suffix is dropped and the Java sources
move to an archive subtree.

---

## Appendix A: Prompt Templates

### Conversion (Step 1 + Step 2)

For converting a new Java file:

> Convert `<path/to/File.java>` to Python at `<path/to/File_ver1.py>`,
> following the rules in CONVERSION_GUIDE.md (R1-R10).
>
> Before writing any code, produce the Step 1 conversion specification
> as a separate markdown file at `<path>/File_ver1.spec.md`.
>
> Then implement the literal port (Step 2). Split overloads, add
> mutation guards, replicate Java signum semantics, document
> deviations.
>
> If you identify suspected Java bugs, add them to `BUG_LEDGER`;
> default to preserving them. Only fix bugs if I explicitly authorize
> in this prompt.
>
> Import shared types from `_epq_compat` -- never define local
> stand-ins.
>
> Do NOT generate the parity harness yet; that's a follow-up step.

### Parity harness (Steps 3-5)

For generating the parity test file:

> Generate `test_parity_<file>.py` in
> `PyEPQ/Utility/tests/` (or analogous sub-package). Follow the
> reference layout at `tests/test_parity_math2.py` exactly:
>
> 1. Copy the module-level setup block verbatim from CONVERSION_GUIDE.md
>    section "The Parity Harness" -> "Module-level setup". Adjust the
>    JClass paths for the new file's class name.
> 2. Implement Part 1 (always-on) tests covering: strict-variant
>    correctness for every `*_strict` companion, self-consistency
>    properties, mutation guards, JavaRandom checks if the file uses it.
> 3. Implement TestBoundaryValues -- one parametrized table per public
>    function covering NaN, +/-Inf, +/-0, mathematical singularities,
>    and known-exact algebraic values. Use scipy reference values for
>    transcendentals.
> 4. Implement Part 2 (parity) tests using `@needs_java + @slow + @given`.
>    Choose tolerance from the ladder per function category. For
>    reductions, use `_close(..., rtol=TOL_REL)`.
> 5. Tag any preserved-bug tests with the corresponding JAVA-BUG-N
>    marker. For each bug with `has_strict_variant=True`, write one
>    parity test (buggy vs Java) and one unit test (strict variant
>    alone).

## Appendix B: Why `_ver1` suffixes

Files are named `<Java>_ver1.py` so the Java original and the
in-progress port can coexist in the same source tree during the
migration. Reviewers grep the same identifier and find both
implementations side-by-side. When Phase 5 lands, the `_ver1` suffix
is dropped in a single sweep.

---

## Appendix C: JPype + JVM Setup (Windows)

The parity harness needs Python <-> Java interop. This appendix
captures the install gotchas hit during Math2 setup so future ports
don't re-discover them.

### Minimum prerequisites

| Component  | Minimum             | Why                                  |
|------------|---------------------|--------------------------------------|
| Python     | 3.10+               | Type-hint syntax used in the port    |
| JDK        | Match epq.jar's     | Class-file version compat (see below)|
|            | class-file version  |                                      |
| jpype1     | 1.7.x               | Bundles a Java-25-compatible loader  |
| EPQ jar    | Same JDK as build   | Class-file version pin               |

### Class-file version matrix

The error `UnsupportedClassVersionError: class file version X` means
the JVM is too old for the jar. Quick reference:

| Bytecode | Built with | Requires JVM |
|----------|------------|--------------|
| 52       | Java 8     | >= 8         |
| 55       | Java 11    | >= 11        |
| 61       | Java 17    | >= 17        |
| 65       | Java 21    | >= 21        |
| 69       | Java 25    | >= 25        |

The current `epq.jar` (as of this writing) is class-file 65 (Java 21).
**Recommended install: Java 25 LTS** -- backwards compatible to all
older bytecode, latest LTS support window, used as the harness
baseline.

### Install steps

1. **JDK 25**: download the MSI from
   `https://adoptium.net/temurin/releases/?version=25`. During install:
   - Check "Set JAVA_HOME variable".
   - Check "JavaSoft (Oracle) registry keys".
2. **Open a fresh terminal** so the new `JAVA_HOME` is inherited.
3. **Verify**:
   ```powershell
   java -version            # should report 25.x
   echo $env:JAVA_HOME      # should point at the jdk-25.x.x directory
   ```
4. **Install jpype1**:
   ```powershell
   python -m pip install jpype1
   ```
5. **Verify JPype sees the right JVM**:
   ```powershell
   python -c "import jpype; print(jpype.getDefaultJVMPath())"
   ```
   Should print a path under `jdk-25.x.x\bin\server\jvm.dll`.

### Gotcha: stale JAVA_HOME in long-lived processes

The Claude Code agent (and any other long-lived tool) inherits
environment variables from when its process started. Installing a new
JDK doesn't update those processes. If `python -c "import jpype;
print(jpype.getDefaultJVMPath())"` reports the OLD JDK path:

```powershell
# Override JAVA_HOME for the current shell only.
$env:JAVA_HOME = "C:\Users\you\AppData\Local\Programs\Eclipse Adoptium\jdk-25.x.x-hotspot"
$env:Path = "$env:JAVA_HOME\bin;$env:Path"
```

### Gotcha: JPype 1.7 hard requirement on Java >= 9

JPype 1.6+ dropped Java 8 support. With Java 8 + JPype 1.7, you get a
Windows fatal access violation at `startJVM`. Either:
- Install Java 9+ (recommended); or
- Pin `pip install "jpype1<1.6"` (works with Java 8 but missing fixes).

### Gotcha: `from gov...import` doesn't work

JPype's `from <pkg> import <Class>` hook only auto-registers `java`,
`javax`, `com`, `org`, `net`. For the EPQ `gov.nist.microanalysis.*`
namespace it raises `ImportError: Java package 'gov' not found` even
with `jpype.imports.registerDomain("gov")` called.

**Fix**: use `jpype.JClass("gov.nist.microanalysis.Utility.Math2")`
instead. Documented stable API, no import-hook quirks. See the
"Module-level setup" code under "The Parity Harness".

### Gotcha: Java 25 restricted-method warning

Java 25 prints a "WARNING: A restricted method in java.lang.System
has been called" when JPype's native loader runs. Silence with:

```python
jpype.startJVM("--enable-native-access=ALL-UNNAMED",
               classpath=[...])
```

### Gotcha: pytest assertion rewriter + JPype

In some configurations, pytest's assertion rewriter triggers a hard
JVM crash at startup. The fix is to load Java classes via `JClass`
rather than `from gov.* import` -- the rewriter doesn't interact with
explicit factory calls.

### Where to drop the jar

The harness checks (in order):
1. `$EPQ_JAR` env var (highest priority; explicit override).
2. `PyEPQ/Utility/tests/epq.jar` (lowercase) or `EPQ.jar`.
3. `<repo-root>/lib/EPQ.jar`.

For the simplest setup, drop `epq.jar` directly into
`PyEPQ/Utility/tests/` and you're done -- no env var needed. Add
`jama-1.0.3.jar` (or any other dependency jar) to the same folder
and the harness picks them up automatically via the
`tests/*.jar` glob.

### Where to find epq.jar

Pre-built jars ship with the
[DTSA-II distribution](https://cstl.nist.gov/div837/837.02/epq/dtsa2/).
Look in `<DTSA-II install>/lib/epq.jar`. If you're building from
source, see the EPQ project's build instructions (no Maven/Gradle
file in the repo; use IntelliJ or Eclipse to import as a Java
project, then export as a JAR).
