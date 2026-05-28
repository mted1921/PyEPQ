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

- Generates random inputs (hypothesis or fixed corpus).
- Calls both the Java implementation (via JPype) and the Python port.
- Asserts equality within tolerance (1e-15 for the literal port).

### Step 4: Library substitutions

Replace hand-rolled algorithms with scipy/numpy one at a time. Each
substitution:

- Keeps the literal port as `_<name>_literal`.
- Makes the public name resolve to the library version.
- Relaxes the parity tolerance for the public name to match the
  library's documented precision (typically 1e-12).

### Step 5: Run parity harness

All tests must pass before the file is considered converted.

---

## The Parity Harness

Skeleton:

```python
# test_parity_math2.py
import os, pytest

if not os.environ.get("EPQ_PARITY"):
    pytest.skip("set EPQ_PARITY=1 to run JVM-backed parity tests",
                allow_module_level=True)

import jpype, jpype.imports
jpype.startJVM(classpath=["lib/EPQ.jar"])
from gov.nist.microanalysis.Utility import Math2 as JavaMath2  # noqa: E402

from Math2_ver1 import Math2 as PyMath2
from hypothesis import given, strategies as st

@given(st.floats(-100, 100, allow_nan=False),
       st.floats(-100, 100, allow_nan=False))
def test_dot_parity(a, b):
    java_out = JavaMath2.dot([a, b], [b, a])
    py_out = PyMath2.dot([a, b], [b, a])
    assert abs(java_out - py_out) < 1e-15

# ... one test per public method ...

# Literal-port parity is strict; library-substituted is looser:
@given(st.floats(0.01, 50.0))
def test_gammaln_literal_parity(x):
    java_out = JavaMath2.gammaln(x)
    assert abs(PyMath2._gammaln_literal(x) - java_out) < 1e-15
    # Library substitution agrees to ~1e-13 due to different Lanczos coeffs:
    assert abs(PyMath2.gammaln(x) - java_out) < 1e-12
```

**Performance**: JPype startup is multi-second. Gate parity tests
behind `EPQ_PARITY=1` so normal pytest runs stay fast.

**Documented bug exceptions**: tests for methods listed in
`BUG_LEDGER` with `has_strict_variant=True` should compare the buggy
Python form against the buggy Java form (parity test), and
separately exercise the strict variant (unit test, no Java
comparison).

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
- `Math2_ver1`. **DONE.**
- `DescriptiveStatistics`, `Histogram`, `FindRoot`, simple utilities.
- Parity harness infrastructure (JPype config, fixtures, env gating).

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

## Appendix A: Prompt Template

For converting a new Java file, the prompt should be roughly:

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

## Appendix B: Why `_ver1` suffixes

Files are named `<Java>_ver1.py` so the Java original and the
in-progress port can coexist in the same source tree during the
migration. Reviewers grep the same identifier and find both
implementations side-by-side. When Phase 5 lands, the `_ver1` suffix
is dropped in a single sweep.
