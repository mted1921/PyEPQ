# EPQ Java → Python Conversion Guide

Rulebook for porting `gov.nist.microanalysis` (NIST DTSA-II's EPQ library)
from Java to Python. Designed for AI-agent-driven refactoring.

## How to use

This guide covers **code conversion only** — the rules for translating Java
source to a correct, idiomatic Python port.

- **TESTING_GUIDE.md** covers parity-test harness construction, tolerances,
  hypothesis profiles, and JVM/JPype setup.
- **PROMPTS.md** holds the copy-paste agent prompts for both workflows.

Every conversion prompt should include the line:
**"Follow CONVERSION_GUIDE.md R1-R10 and produce the artifacts in Per-File Workflow."**

Files are named `<X>_ver{G}_{N}_{F}.py` during migration, where:
- `G` = guide version (increment when CONVERSION_GUIDE.md rules change materially)
- `N` = generation index (increment per independent agent run on the same guide version)
- `F` = port-code fix count (increment each time port code itself is patched; test-only changes do not increment)

---

## The Rulebook

### R1 — Naming: preserve Java identifiers verbatim

Keep `camelCase`, `Class.staticMethod(...)` patterns, field and constant
names. Even private helpers preserve their Java spelling — `_doRecompute`,
not `_do_recompute`.

The `_` prefix maps to Java **access modifiers only** — not to other keywords:

| Java declaration | Python name |
|---|---|
| `public void compute()` | `compute()` |
| `public abstract void compute()` | `compute()` — `abstract` does NOT add `_` |
| `protected void compute()` | `_compute()` |
| `private void compute()` | `_compute()` |

**Common mistake**: treating `abstract` as a privacy modifier and writing
`_compute()` for a `public abstract` method. This breaks every subclass
that overrides `compute()` with the correct name, causing
`TypeError: Can't instantiate abstract class … without an implementation
for abstract method '_compute'` on every test.

```java
// Java
public class Element {
    public static final double AVOGADRO = 6.022e23;
    private int mAtomicNumber;
    public static Element byAtomicNumber(int z) { ... }
    private void invalidateCache() { ... }
}
public abstract class LazyEvaluate<H> {
    public abstract H compute();   // public → no underscore
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

class LazyEvaluate(abc.ABC, Generic[H]):
    @abc.abstractmethod
    def compute(self) -> H: ...  # public abstract → compute(), NOT _compute()
```

### R2 — Faithful AND complete port

The first pass is a complete literal translation. Two equal requirements:

**(a) Faithful** — no library substitutions, no bug fixes, no "cleaner"
algorithms. Each scipy/numpy substitution keeps its `_literal` companion.
Under the combined Step 2 workflow, write both in the same pass; the
parity harness (see TESTING_GUIDE.md) is the verification gate.

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
| Inner class | Nested class, **plus a module-level alias** if the class is publicly accessible (see note below) |
| Anonymous inner class | Module-level helper or lambda |
| Each overloaded constructor | A `@classmethod` per signature (R4), **plus `__init__` dispatch** (see note below) |
| `equals` / `hashCode` / `toString` | `__eq__` / `__hash__` / `__str__` **AND** a named alias `equals()` / `hashCode()` / `toString()` for source compatibility |

**Verification**: `grep` for `public `, `private `, `protected `,
`static ` in the Java file. The Python port must have the same total
declarations. Mismatched counts mean something was silently dropped.

**Inner class module-level aliases.** A Java public inner/static class (e.g.
`MultiDHistogram.LinearBins`) is part of the public API surface and must be
importable directly from the module. After the outer class body, add:

```python
# Module-level aliases so `from MultiDHistogram_ver1_1_0 import LinearBins` works
LinearBins = MultiDHistogram.LinearBins
IBinning   = MultiDHistogram.IBinning
Bin        = MultiDHistogram.Bin
```

Also add these names to `__all__`. Omitting them causes `ImportError` in the
harness and any caller that imports the inner class by name.

**Constructor overload dispatch.** R4 says split Java overloads into
`@classmethod` forms. For constructors, Python callers will use `MyClass(arg)`
syntax — not `MyClass.from_something(arg)` — so `__init__` must also dispatch
on argument type. Keep the classmethods for the Java API surface; have `__init__`
delegate to them:

```python
def __init__(self, arg, minMembers: int = 1) -> None:
    if isinstance(arg, MultiDHistogram):
        # copy constructor path
        ...
    else:
        # IBinning path
        ...

@classmethod
def from_histogram(cls, src):      # keeps Java name in the API
    return cls(src, 1)
```

**Copy constructors that re-insert into a new container.** When a copy
constructor reconstructs a collection by iterating source elements and
reinserting them, internal sentinel fields on the elements — island indices,
cached-value flags, etc. — must be reset before reinsertion. The safe path
is to call the same public insertion method used at runtime (`add_bin()`)
rather than the private method directly (`_addBin()`). The public method owns
the invariant setup:

```python
# WRONG — from_bin preserves mIsland; _addBin requires mIsland == NO_ISLAND
for vox in arg.mData:
    self._addBin(Bin.from_bin(vox))   # AssertionError at runtime

# RIGHT — add_bin() resets mIsland before delegating to _addBin
for vox in arg.mData:
    self.add_bin(vox)
```

**M4 — abstract classes**: JPype cannot extend Java abstract classes from
Python. When a class is `abstract`, mark the parity test class with
`@pytest.mark.skip(reason="M4: ...")` and test correctness analytically
via concrete Python subclasses. See TESTING_GUIDE.md Appendix A for detail.

### R3 — Use the shared compat module

All converted modules import `EPQException`, `JavaRandom`, `JavaTreeSet`,
`JamaMatrix` from `_epq_compat`. Defining local fallbacks creates incompatible
types and breaks `except` clauses across module boundaries.

```python
# RIGHT
try:
    from ._epq_compat import EPQException, JavaRandom, JavaTreeSet, JamaMatrix, F64Array
except ImportError:
    try:
        from _epq_compat import EPQException, JavaRandom, JavaTreeSet, JamaMatrix, F64Array  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility._epq_compat import EPQException, JavaRandom, JavaTreeSet, JamaMatrix, F64Array  # type: ignore

# WRONG — creates a NEW EPQException incompatible with everyone else's
class EPQException(Exception):
    pass
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

### R6 — Maintain the bug ledger

Each preserved Java bug gets three artifacts:

1. A `# JAVA-BUG-N` marker at the offending line.
2. An entry in the class-level `BUG_LEDGER` tuple.
3. Where reasonable, a `*_strict` companion that fixes the bug.

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

**Source citation required.** Every BUG_LEDGER entry must quote the exact
Java source line that contains the bug (copy it as a comment). Do NOT infer
a bug from the method's behaviour or the ported code alone — only add an
entry when you can point to the offending Java line. Fabricated entries
mislead parity tests and can cause false `TestPreservedBugs` failures.

```python
# RIGHT — cites the actual Java line
# JAVA-BUG-1: Java source: `return data > 0 ? data : 0;` — clamps negatives.

# WRONG — inferred, not cited; likely to be wrong
# JAVA-BUG-1: uses reference identity instead of .equals()
```

If no bugs are found (or the Java source is not available), include an
explicit empty tuple:
```python
BUG_LEDGER: tuple = ()  # no bugs identified; Java source not attached
```

### R7 — Java integer division

Every Java `/` between integer types becomes Python `//`. This is the
#1 silent-bug source in Java → Python conversions.

```python
# WRONG — Python `/` between ints returns a float
mid = total / 2            # 7 / 2 == 3.5

# RIGHT
mid: int = total // 2      # 7 // 2 == 3
```

**`Math.round` is not Python's `round()`.** Java's `Math.round(x)` is
`floor(x + 0.5)`, always rounding ties toward +∞. Python's `round()` uses
HALF_EVEN (banker's rounding). Replace every `Math.round` with
`int(math.floor(x + 0.5))`.

### R8 — Floating point sign semantics

Java `Math.signum(0)` returns 0. Python's `math.copysign(x, 0.0)`
returns `+x`. They diverge at zero.

```python
# WRONG
sign = math.copysign(1.0, v)

# RIGHT
sign: float = 0.0 if v == 0.0 else math.copysign(1.0, v)
```

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

**Java implicit widening on return.** A method declared `public double`
that returns an `int` field must explicitly cast: `return float(self.mNEvals)`.
The `-> float` annotation alone does not coerce at runtime.

### R10 — Document deviations explicitly

Any deliberate deviation from Java requires:

1. A comment at the call site.
2. An entry in the file's docstring `CHANGES` section.
3. An entry in `BUG_LEDGER` if observable (see R6 for format).

---

## Per-File Workflow

For each Java file:

**Step 1 — Conversion specification** (planning artifact). Before writing
code, produce the spec file at `PyEPQ/<Package>/spec/<ClassName>.spec.md`.

The spec must list:
- Inbound dependencies (Java imports).
- Outbound dependents (`grep` for callers of every public method).
- Public API surface with signatures.
- Overloaded methods (split plan).
- Mutable-output methods (`*Equals`, `addInPlace`, ...).
- Touchpoints into Jama, javax.swing, java.awt, java.io.
- Suspected Java bugs with disposition (preserve / fix / flag).

**Step 2 — Literal port AND library substitution.** Begin by writing the
module docstring. **Preserve the original Java class-level Javadoc comment
verbatim** inside a clearly delimited section of the docstring:

```python
r"""
ClassName_ver{G}_{N}_{F}.py — Python port of gov.nist.microanalysis.Package.ClassName

Guide version : G
Generation    : N
Port-code fixes: F

------------------------------------------------------------------------
Original Java source (gov.nist.microanalysis.Package.ClassName)
------------------------------------------------------------------------
/**
 * ... verbatim Javadoc comment from the Java file ...
 */
------------------------------------------------------------------------
"""
```

Then translate the Java source top to bottom. Apply R1–R10. In the same pass,
add a scipy/numpy primary where a natural substitution exists:

- Name the faithful translation `<method>_literal` (e.g. `integrate_literal`).
- Name the library version `<method>` — the primary public name.
- Document each deviation in the scipy version with a `SCIPY-DEV-N` comment.

**Step 3 — Parity harness.** See **TESTING_GUIDE.md** for the full harness
specification and template. Use the prompt in **PROMPTS.md** to generate it
via an independent agent session.

**Step 4 — Run parity harness + coverage.** All tests must pass. See
TESTING_GUIDE.md for coverage targets and how to run branch coverage.

---

## Cross-Cutting Patterns

| Concept | Rule |
|---|---|
| RNG | Always `JavaRandom` from `_epq_compat`; never `random.Random`. Python's Mersenne Twister produces a different sequence for the same seed. |
| Sorted set (`java.util.TreeSet`) | Always `JavaTreeSet` from `_epq_compat`; never reimplement locally. Requires elements to have `__lt__` and `compareTo()` — the natural output of porting a `Comparable` class. Other Java collections: `ArrayList`/`LinkedList` → `list`; `HashMap`/`LinkedHashMap` → `dict`. |
| Nested container counts | A class with island → bins → items has methods measuring different levels: `getIslandSize()` may count bins while `getItemsInIsland()` sums items across bins. Never use a bin-count method to size a buffer that will be filled item-by-item, or vice versa — the silent mismatch produces `IndexError` or truncated output. Check the Java declaration comment, not just the method name. |
| Parametric helper with a fixed-`n` default | When `methodA()` delegates to `methodB(x, n)` with a fixed `n`, choose `n` from the documented semantics, not structural proximity. `adjacent_bin()` → `adjacent_bin_m(vox, 1)` gives face-sharing-only adjacency; passing `len(self.mIndex)` instead silently includes diagonal neighbours in 2D. Verify the correct value against the test expectations and the Java comment before choosing. |
| Matrix returns | Always `JamaMatrix` where Java returned `Jama.Matrix`. Raw ndarray breaks chained `.times(...).solve(...)`. Use `.getArray()` to access the underlying ndarray. |
| Exceptions | Always `EPQException` from `_epq_compat` (see R3). |
| Number formatting | Accept `Callable[[float], str]` where Java accepted `NumberFormat`. |
| Static init order | Cross-class static refs must be resolved lazily (property getters, factory functions, or per-module `_initialize()`). Audit during Step 1. |
| File I/O | Always specify `encoding="utf-8"`. |
| Concurrency | `synchronized` → `threading.Lock`; audit memory-model assumptions case by case. |

---

## Quick-Lookup: rule index

| Concern | Rule |
|---|---|
| Naming convention drift | R1 |
| `abstract` keyword triggering `_` prefix (wrong) | R1 — `_` is for access modifiers only |
| Missing named `equals()` / `hashCode()` alias | R2 member table |
| Public inner class not importable at module level | R2 — add module-level alias after outer class |
| Constructor overloads: only classmethods, no `__init__` dispatch | R2 / R4 — `__init__` must dispatch by type; classmethods kept for Java API surface |
| `java.util.TreeSet` reimplemented locally | R3 / Cross-Cutting — use `JavaTreeSet` from `_epq_compat` |
| Copy constructor breaks container invariants | R2 — reset sentinel fields (e.g. island index) before reinsertion; call the public insertion method, not the private one |
| Delegating `methodA()` with `len(dim)` instead of `1` | Cross-Cutting — verify the fixed-`n` value against documented semantics, not structural convenience |
| Bin count used where item count required | Cross-Cutting — check which level of the container hierarchy each `getSize`/`getCount` method measures |
| Fabricated BUG_LEDGER entries without Java source citation | R6 |
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
| Abstract class parity limit | M4 in Appendix B; TESTING_GUIDE.md Appendix A |

---

## Appendix B: Known limits of this methodology

| ID | Issue | Mitigation |
|---|---|---|
| **M1** | Numerical drift from library substitutions | Document slack per call site; use `TOL_NR_LIB` (1e-4) for NR substitutions, `TOL_COMPOUND` for chained ops |
| **M2** | Cross-class static init order | Resolve lazily; audit during Step 1 spec |
| **M3** | Swing / AWT GUI | Not parity-testable; rebuild against PyQt/Tkinter from spec |
| **M4** | Parity testing of abstract classes | JPype cannot extend Java abstract classes from Python. Test indirectly via concrete Java methods, or compile a thin concrete Java helper. See TESTING_GUIDE.md Appendix A. |
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
- `Math2` — **DONE** (209 tests green, ~99% branch coverage)
- `_parity_lib` — **DONE** at `PyEPQ/Utility/tests/_parity_lib.py`
- `FindRoot` — **DONE** (22 passed, 8 skipped; direct parity blocked by M4)
- `UtilException` — **DONE** (40 passed)
- `HalfUpFormat` — **DONE** (219 collected, 216 passed)
- `AdaptiveRungeKutta` — **DONE** (40 passed, 1 skipped; M4)
- `Simplex` — **DONE** (20 passed, 1 skipped; M4)
- `DescriptiveStatistics`, `Histogram` — pending

**Phase 2 — EPQLibrary core**: `Element`, `Composition`, `Material`,
`AlgorithmClass`, `AlgorithmUser`, `XRayTransition`.

**Phase 3 — EPQLibrary algorithms**: `MassAbsorptionCoefficient`,
`EdgeEnergy`, `FluorescenceYield`, `BremsstrahlungAngularDistribution`.

**Phase 4 — EPQTools / GUI**: choose framework (PyQt5 recommended).

**Phase 5 — Pythonification**: snake_case sweep, retire shims, drop
`_ver{G}_{N}_{F}` suffix.
