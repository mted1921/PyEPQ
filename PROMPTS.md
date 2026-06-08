# EPQ Agent Prompts

Copy-paste prompts for AI-agent-driven EPQ Java → Python workflows.
Each prompt is self-contained — attach the listed context files and send.

**Workflow order:**
1. Port agent (Prompt 1) → produces `<Class>_ver1_1_0.py`
2. Manual review against CONVERSION_GUIDE.md
3. Harness agent (Prompt 2) → produces `tests/test_parity_<class>_ver1_1_0.py`
4. Run tests: `python tests/test_parity_<class>_ver1_1_0.py`
5. If failures → Repair agent (Prompt 3) → attach `test_output.txt` + port + test + Java source
6. Repeat steps 4–5 until all tests pass

---

## Prompt 1 — Port Conversion

**Attach before sending:**
1. The Java source file to convert
2. `_epq_compat.py`
3. `CONVERSION_GUIDE.md`
4. `Math2.py` (style reference)

---

You are converting a single Java class to Python. Your only task is to produce
the Python port file. Do **not** write any test or harness files.

### Step 1 — Parse the Java source

Extract:
- `CLASS_NAME` — from `public [abstract] class <NAME>`
- `IS_ABSTRACT` — whether the class declaration includes `abstract`
- All `public` method signatures, constants, and fields
- Java bugs: dead branches, always-true conditions, off-by-one errors, sign errors

Output filename: `<CLASS_NAME>_ver1_1_0.py`

### Step 2 — Write the port

Use `Math2.py` as the style reference throughout.

**File header** — copy the Java `/** ... */` Javadoc verbatim:
```python
r"""
<CLASS_NAME>_ver1_1_0.py — Python port of gov.nist.microanalysis.Utility.<CLASS_NAME>

Guide version : 1
Generation    : 1
Port-code fixes: 0

------------------------------------------------------------------------
Original Java source (gov.nist.microanalysis.Utility.<CLASS_NAME>)
------------------------------------------------------------------------
/**
 * ... verbatim Javadoc from the Java file ...
 */
------------------------------------------------------------------------
"""
```

**Standard imports:**
```python
from __future__ import annotations
import abc, math
import numpy as np
from typing import Optional, Sequence, Union, Callable
try:
    from ._epq_compat import EPQException, F64Array, JavaRandom
except ImportError:
    try:
        from _epq_compat import EPQException, F64Array, JavaRandom  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility._epq_compat import EPQException, F64Array, JavaRandom  # type: ignore
```

**Rules (from CONVERSION_GUIDE.md):**

- **R1** — Preserve all Java identifiers verbatim. `_`-prefix maps to access modifiers only (`private`/`protected`). **`abstract` does NOT add `_`** — `public abstract void compute()` → `compute()`, never `_compute()`. Mapping `public abstract` to `_name` will break every subclass that correctly implements `name()`, causing `TypeError: Can't instantiate abstract class`.
- **R2** — For every public mathematical method `foo`: write `foo()` (scipy primary) and `foo_literal()` (line-for-line Java). Abstract methods: only the abstract declaration, no body. For `equals`/`hashCode`/`toString`: map to `__eq__`/`__hash__`/`__str__` **and also expose a named alias** `equals()`/`hashCode()`/`toString()` so Java-style call sites (`obj.equals(other)`) continue to work.
- **R3** — Import `EPQException`, `JavaRandom`, `JavaTreeSet`, `JamaMatrix`, `F64Array` from `_epq_compat` only — never define local replacements. Use `JavaTreeSet` wherever Java used `java.util.TreeSet`; never reimplement it locally.
- **R4** — Split Java overloads into type-suffixed functions (`_vv`, `_vs`, `_arr`, `_scalar`, `_int`, `_double`).
- **R5** — Call `_require_mutable_f64` before any in-place array mutation.
- **R6** — For every Java bug: quote the **exact Java source line** in the comment (`# JAVA-BUG-N: Java source: \`return x > 0 ? x : 0;\``), add an entry in `BUG_LEDGER`, and provide an optional `*_strict` companion. Do not infer bugs from behaviour without citing the Java line — fabricated entries cause false parity failures. If no bugs exist (or Java source is unavailable), write `BUG_LEDGER: tuple = ()  # no bugs identified`.
- **R7** — Every Java `/` between integer types → Python `//`. Every `Math.round(x)` → `int(math.floor(x + 0.5))`.
- **R8** — `Math.signum(0)` → `0.0 if v == 0.0 else math.copysign(1.0, v)`.
- **R9** — Annotate every parameter, return type, class field, and non-obvious local variable. Use `F64Array` for `double[]`. Explicitly cast `int` fields returned by `double`-declared methods: `return float(self.mField)`.
- **R10** — Document every deliberate deviation with a call-site comment, a `CHANGES` section in the docstring, and a `BUG_LEDGER` entry if observable.

### Step 3 — Checklist

- [ ] Every `public` Java method has a Python counterpart
- [ ] Every public non-abstract mathematical method has both `foo()` and `foo_literal()`
- [ ] `public abstract` methods use the **un-prefixed** Java name (e.g. `compute`, not `_compute`)
- [ ] `equals()`/`hashCode()`/`toString()` have both a dunder and a named alias
- [ ] `BUG_LEDGER` entries each cite the exact Java source line; empty tuple if none
- [ ] All parameters, returns, fields, and non-obvious locals annotated
- [ ] Abstract class → `abc.ABC`; abstract methods → `@abc.abstractmethod`
- [ ] Java Javadoc copied verbatim into module docstring
- [ ] Three-tier import fallback used for `_epq_compat`
- [ ] No test code written

---

## Prompt 2 — Parity Harness

**Attach before sending:**
1. The Java source file
2. The completed Python port (`<CLASS_NAME>_ver1_1_0.py`)
3. `_parity_lib.py`
4. `TESTING_GUIDE.md`

---

You are writing the parity test harness for an already-completed Python port.
Your only task is to produce the test file. Do **not** modify the port file or
any other file.

### Step 1 — Parse the inputs

Extract:
- `CLASS_NAME` — from the port file or Java source
- `IS_ABSTRACT` — whether the class is declared `abstract`
- All `public` method signatures and their documented behaviour
- Any `BUG_LEDGER` entries in the Python port

Output filename: `tests/test_parity_<class_name_lower>_ver1_1_0.py`

### Step 2 — Write the test harness

**File header and boilerplate:**
```python
r"""
test_parity_<class_name_lower>_ver1_1_0.py — parity harness for <CLASS_NAME>_ver1_1_0.py
"""
from __future__ import annotations
import math
import numpy as np
import pytest
from hypothesis import given, strategies as st
from _parity_lib import (
    setup_parity, needs_java, PARITY_ENABLED,
    TOL_EXACT, TOL_LITERAL, TOL_NR_LIB,
    finite, positive, slow,
    _close, _arr_close,
    _NAN, _INF,
)
from <CLASS_NAME>_ver1_1_0 import <CLASS_NAME> as Py<CLASS_NAME>
from _epq_compat import EPQException

ctx = setup_parity("gov.nist.microanalysis.Utility.<CLASS_NAME>")
Java<CLASS_NAME> = ctx.java_class

if __name__ == "__main__":
    from _parity_lib import run_tests; run_tests(__file__)
```

**Required sections:**
- `TestConstruction` — every constructor path, all default field values
- `Test<Behaviour>` — one class per logical behaviour group
- Edge cases — zero, negative, empty, NaN/±Inf where relevant
- `TestHypothesis` — `@given` fuzz over the natural input domain
- `Test<CLASS_NAME>Parity`:
  - *If IS_ABSTRACT*: `@pytest.mark.skip(reason="M4: JPype cannot extend Java abstract classes from Python. <abstract_method>() is abstract. Correctness validated analytically above.")` — create concrete Python subclasses validating against analytical ground truth instead.
  - *If concrete*: `@needs_java` tests for every public method + a `@given` hypothesis fuzz test.

**Tolerances** (from TESTING_GUIDE.md):
- `TOL_EXACT` (0) for integer ops and equality
- `TOL_LITERAL` (1e-14) for same-algorithm comparisons
- `TOL_NR_LIB` (1e-4) for scipy vs Java Numerical Recipes

**For `BUG_LEDGER` entries:**
- `TestPreservedBugs` — parity test: buggy Java output == buggy Python output
- `TestStrictVariants` — unit-test the `*_strict` Python companion (no Java call)

### Step 3 — Checklist

- [ ] `TestConstruction` covers every constructor path and all default values
- [ ] At least one `Test<Behaviour>` per distinct method category
- [ ] `TestHypothesis` with `@given` + `@slow` present
- [ ] Abstract → parity section is `@pytest.mark.skip(M4)` with concrete subclass validation
- [ ] Concrete → `@needs_java` parity tests cover all public methods
- [ ] `BUG_LEDGER` entries have both a preserved-bug parity test and a strict-variant test
- [ ] `if __name__ == "__main__":` block at bottom of file
- [ ] No port code written
- [ ] Item/argument types in every test call match the **Java method signature** — if Java requires `int item`, never use a string, list, or other object as the value
- [ ] `@needs_java` parity tests call Java constructors with the **actual Java signature** (look it up in the Java source), not the Python port's generalized form; add a comment showing the Java constructor when they differ

---

## Prompt 3 — Port Repair (fix failures from test_output.txt)

**Use when:** A port was generated, the parity tests were run, and failures appeared
in `test_output.txt`. Send this prompt to have the agent diagnose and repair the port.

**Attach before sending:**
1. The Java source file (the original being ported)
2. The current Python port (`<CLASS_NAME>_ver1_1_0.py`)
3. The current test file (`tests/test_parity_<class>_ver1_1_0.py`)
4. `test_output.txt` (the pytest failure log)
5. `CONVERSION_GUIDE.md`

---

You are repairing a Python port that is failing its parity test suite.
Your task is to read the test failures, identify the root cause in the port,
and produce a corrected port file. Do **not** modify the test file unless a
test itself is provably wrong (wrong expected value, wrong item type, etc.) —
if you must touch the test, state the reason explicitly and keep the change minimal.

### Step 1 — Read the failure log

Open `test_output.txt` and for every `FAILED` entry:

1. Copy the **full traceback** (not just the summary line).
2. Identify which assertion or exception fires, and on which line.
3. State whether the failure is in the **port** (wrong API name, wrong return type,
   wrong calculation) or in the **test** (wrong expected value, wrong argument type).

Classify each failure as one of:
- **R1-violation** — wrong method/field name (underscore prefix on public member, renamed method, etc.)
- **R2-omission** — public Java method missing from port, or `equals()`/`hashCode()`/`toString()` missing named alias
- **R4-violation** — overload split missing, wrong constructor dispatch
- **R9-violation** — wrong return type (e.g. float returned where bool expected)
- **API-mismatch** — constructor signature wrong (argument types/count), method renamed
- **Test-bug** — test passes wrong argument type (e.g. string where int required), or queries an item that was never added

### Step 2 — Identify root causes

For each failure class, trace back to the **single root cause** in the port:

- R1 mistakes: check whether a `_` prefix was applied to a `public abstract` method
  (the `abstract` keyword does NOT add `_`; only `private`/`protected` do).
- API mismatches: compare the constructor/method signature in the Java source against
  the port. The Java source is the authority. If the test uses a signature the Java
  source also uses, the port is wrong.
- Return-type failures: check if the Java method returns `boolean` vs `double`.
  A ratio that should be compared to a threshold and returned as bool is a common mistake.
- Missing module-level aliases: inner/nested Java classes (`MultiDHistogram.LinearBins`)
  that appear at the top level of Java's public API must be accessible via
  `from <module> import <ClassName>`. Add module-level aliases after the outer class.

### Step 3 — Repair the port

Make the minimum changes that fix all identified failures:

1. **Do not rewrite sections that are already correct.** Touch only what is broken.
2. For every change, add a `# FIX-N: <one-line description>` comment at the changed line
   and a `CHANGES` entry in the module docstring. Increment `Port-code fixes` in the header.
3. After each fix, mentally re-run the failing tests against the corrected code and
   confirm the fix resolves them without breaking passing tests.
4. If the Java source contradicts the test's expectation and the test is clearly wrong
   (e.g. passing a string where the Java signature requires `int`), fix the test instead
   and document it as a test fix (does NOT increment `Port-code fixes`).

### Step 4 — Checklist

- [ ] Every failure in `test_output.txt` is addressed (port fix or documented test fix)
- [ ] No passing test is broken by the repair
- [ ] `Port-code fixes` counter in the module docstring header is incremented
- [ ] `CHANGES` section lists each fix with its category (R1/R2/R4/etc.)
- [ ] Module-level aliases added for any inner class that must be directly importable
- [ ] No new functionality introduced beyond what fixes the failures

---

## When to add a new prompt

Add a prompt only when:
- The same task runs on ≥3 files (e.g. a new migration phase introduces a new artifact type).
- The prompt is non-trivial to reconstruct from memory.
- Correctness depends on details (specific tolerances, library APIs) a future contributor might not recall.

One-off tasks ("audit this file for missing JAVA-BUG-N markers") stay ad-hoc in conversation.
