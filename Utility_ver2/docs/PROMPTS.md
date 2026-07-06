# EPQ Agent Prompts

Version 2 · 2026-06-24

---




## Prompt 0 — Conversion Spec

**Attach before sending:**
1. The Java source file to analyse
2. `UTILITY_LEDGER.md` — to identify sibling port filenames for the dependency table

---

You are producing the conversion specification for a single Java class.
Your only task is to write the spec file. Do **not** write any Python code.

Output filename: `spec/<CLASS_NAME>.spec.md`

### Step 1 — Parse the Java source

Extract:
- `CLASS_NAME` — from `public [abstract] class <NAME>`
- `IS_ABSTRACT` — whether the class declaration includes `abstract`
- Every `public` and `protected` field, constant, and method signature
- Every Java `import` statement (inbound dependencies)
- Any `abstract` method declarations (extension points)

### Step 2 — Write the spec

```markdown
# <CLASS_NAME> Conversion Spec

## Java class
`gov.nist.microanalysis.Utility.<CLASS_NAME>`

Source: `src/gov/nist/microanalysis/Utility/<CLASS_NAME>.java`

---

## Inbound dependencies (Java imports)
- List each EPQ-internal import and what it provides.
- Note any Jama, javax.swing, java.awt, java.io imports separately.

---

## Outbound dependents (callers of public methods)
Summarise what other classes call this one (grep `<CLASS_NAME>` in `src/`).
If no grep is possible, state "not audited".

---

## Public API surface

| Java signature | Python signature | Notes |
|---|---|---|
| (one row per public constructor and method) | | |

For each abstract method, mark it `@abstractmethod` in the Notes column.
For each overloaded group, list all overloads.

---

## Private / protected members

| Java | Python |
|---|---|
| (fields and helpers worth noting) | |

---

## Overloaded methods (split plan)
For each group of Java overloads, state the Python translation:
- Same-type overloads with a default argument → single function with defaults.
- Different-type overloads → suffix-split per R4 (`_vv`, `_vs`, `_arr`, `_scalar`, etc.).
- Describe the dispatcher strategy if applicable.

---

## Mutable-output methods
List every method that mutates a caller-supplied array in place
(R5: `_require_mutable_f64` guard needed). Cite the Java line.

---

## Touchpoints into Jama / javax.swing / java.awt / java.io
If none: "None."

---

## Abstract class strategy
(Only if IS_ABSTRACT.)
State which abstract methods exist, what the extension contract is, and
that direct JPype parity is blocked by M4. Specify the concrete-subclass
strategy for the parity harness.

---

## Java-specific translation decisions

| Java pattern | Python translation | Rule |
|---|---|---|
| (one row per non-obvious mapping) | | |

---

## Suspected Java bugs
For each suspected bug:
- Quote the **exact Java source line**.
- State the disposition: Preserve (port the bug verbatim in `_literal`) /
  Fix (scipy/numpy primary corrects it) / Flag (needs human review).

If none: "None identified."

---

## Static init order
Note any `static {}` blocks or cross-class static references that require
lazy resolution. If none: "None."

---

## Thread safety
Copy the Java docstring note if present; otherwise assess from the source.
```

### Step 3 — Checklist

- [ ] Every `public` constructor and method appears in the API surface table
- [ ] Every Java `import` listed under inbound dependencies
- [ ] Overloads explicitly planned (default args vs. suffix-split)
- [ ] Mutable-output methods identified and R5 noted
- [ ] Abstract methods flagged in the table and M4 strategy stated
- [ ] Every suspected bug cites the exact Java source line
- [ ] No Python code written

---






## Prompt 1 — Port Conversion

**Attach before sending:**
1. The Java source file to convert
2. `spec/<CLASS_NAME>.spec.md` — **required** (produced by Prompt 0; reviewed and approved before this step)
3. `_epq_compat.py`
4. `CONVERSION_GUIDE.md`
5. `Math2_ver8_1_5.py` (style reference)
6. `UTILITY_LEDGER.md` — **required when the Java class imports other Utility classes**
   (tells you the exact current filename of each sibling port)

---

You are converting a single Java class to Python. Your only task is to produce
the Python port file. Do **not** write any test or harness files. Limit the context for each conversion to the ctx folder (src\gov\nist\microanalysis\PyEPQ\Utility_ver2\ctx), as well as the source Java (src\gov\nist\microanalysis\Utility), and corresponding spec (src\gov\nist\microanalysis\PyEPQ\Utility_ver2\spec). Perform each port conversion in a sequential manner, avoiding paralellization. 

### Step 1 — Parse the Java source

Extract:
- `CLASS_NAME` — from `public [abstract] class <NAME>`
- `IS_ABSTRACT` — whether the class declaration includes `abstract`
- All `public` method signatures, constants, and fields
- Java bugs: dead branches, always-true conditions, off-by-one errors, sign errors

Output filename: `<CLASS_NAME>_ver1_1_0.py`

### Step 2 — Write the port

Use `Math2_ver8_1_5.py` as the style reference throughout.

**File header** — copy the Java `/** ... */` Javadoc verbatim:
```python
r"""
<CLASS_NAME>_ver1_1_0.py — Python port of gov.nist.microanalysis.Utility.<CLASS_NAME>

Guide version : 2
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
- **R2** — For every public mathematical method `foo`: write `foo()` (scipy primary) and `foo_literal()` (line-for-line Java). **If no scipy substitute exists**, still write both: `foo()` calls `foo_literal()` and is marked `# SCIPY-NONE: no library substitute`. The `_literal` companion must always be present — test harnesses look for it unconditionally. Abstract methods: only the abstract declaration, no body. For `equals`/`hashCode`/`toString`: map to `__eq__`/`__hash__`/`__str__` **and also expose a named alias** `equals()`/`hashCode()`/`toString()` so Java-style call sites (`obj.equals(other)`) continue to work.
- **Java `assert`** — Java asserts are **disabled by default** (need `-ea`). The common `assert cond; field = clamp(x)` pattern means the clamp is the ONLY production behaviour. Port it as `field = clamp(x)` with no assert. Use `if not cond: raise AssertionError(...)` only when the spec explicitly marks the assertion as load-bearing. **Never** use bare Python `assert` to represent a Java `assert` — Python asserts are always active and will crash callers that depend on the silent clamp.
- **R3** — Import `EPQException`, `JavaRandom`, `JavaTreeSet`, `JamaMatrix`, `F64Array` from `_epq_compat` only — never define local replacements. Use `JavaTreeSet` wherever Java used `java.util.TreeSet`; never reimplement it locally.
- **R4** — Split Java overloads into type-suffixed functions (`_vv`, `_vs`, `_arr`, `_scalar`, `_int`, `_double`).
- **R5** — Call `_require_mutable_f64` before any in-place array mutation.
- **R6** — Maintain the `BUG_LEDGER` per **BUG_GUIDE.md**: quote the exact Java source line at a `# JAVA-BUG-N` marker, add a tuple entry, and provide an optional `*_strict` companion. Never infer a bug without citing the Java line. If no bugs exist, write `BUG_LEDGER: tuple = ()  # no bugs identified`.
- **R7** — Every Java `/` between integer types → Python `//`. Every `Math.round(x)` → `int(math.floor(x + 0.5))`.
- **R8** — `Math.signum(0)` → `0.0 if v == 0.0 else math.copysign(1.0, v)`.
- **R9** — Annotate every parameter, return type, class field, and non-obvious local variable. Use `F64Array` for `double[]`. Explicitly cast `int` fields returned by `double`-declared methods: `return float(self.mField)`.
- **R10** — Document every deliberate deviation with a call-site comment, a `CHANGES` section in the docstring, and a `BUG_LEDGER` entry if observable.

### Step 3 — Checklist

- Every `public` Java method has a Python counterpart
- Every public non-abstract mathematical method has both `foo()` and `foo_literal()` — when no scipy substitute exists, `foo()` calls `foo_literal()` and carries `# SCIPY-NONE`
- `public abstract` methods use the **un-prefixed** Java name (e.g. `compute`, not `_compute`)
- `equals()`/`hashCode()`/`toString()` have both a dunder and a named alias
- `BUG_LEDGER` maintained per BUG_GUIDE.md (exact Java line cited; empty tuple if none)
- All parameters, returns, fields, and non-obvious locals annotated
- Abstract class → `abc.ABC`; abstract methods → `@abc.abstractmethod`
- Java Javadoc copied verbatim into module docstring
- Three-tier import fallback used for `_epq_compat` **and any sibling port modules**
- Sibling module filenames taken from `UTILITY_LEDGER.md` "Port file" column — never guessed
- `serialVersionUID` exported as a public attribute (no `_` prefix) regardless of Java access modifier
- Every Java `assert cond; field = clamp()` ported as `field = clamp()` only — no Python `assert`
- If a subclass method shadows an inherited method of the same name, parent is aliased in the class body: `_alias = ParentClass.method` before the override is defined
- No test code written

---






## Prompt 2 — Parity Harness

**Attach before sending:**
1. The Java source file
2. `spec/<CLASS_NAME>.spec.md` — the approved conversion spec (produced by Prompt 0)
3. `_parity_lib.py`
4. `TESTING_GUIDE.md`

---

You are writing the parity test harness **before the Python port exists**.
Your inputs are the Java source and the approved conversion spec.
Your only task is to produce the test file. Do **not** write any port code or
modify any other file.

The port module (`<CLASS_NAME>_ver1_1_0.py`) does not exist yet — write imports
referencing its expected filename and the agent that writes the port will
produce a file that satisfies them.

### Step 1 — Parse the inputs

Extract from the **Java source and spec file**:
- `CLASS_NAME` — from the spec or Java `public [abstract] class <NAME>`
- `IS_ABSTRACT` — from the spec's "Abstract class strategy" section
- All `public` method signatures and their documented behaviour
- Any suspected Java bugs listed in the spec's "Suspected Java bugs" section
  (use these to plan `TestPreservedBugs` / `TestStrictVariants` — do NOT invent bugs
  not listed in the spec)

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

# NEVER use the full dotted path (gov.nist.microanalysis.PyEPQ.Utility.<CLASS_NAME>_ver...).
# _parity_lib.py adds PyEPQ/Utility/ to sys.path on import, so the bare module
# name above is the ONLY correct form. The full path causes "No module named 'gov'"
# because sys.path contains Utility/, not the repo root.
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

**For bugs listed in the spec's "Suspected Java bugs" section:**
- `TestPreservedBugs` — parity test: buggy Java output == buggy Python output
- `TestStrictVariants` — unit-test the `*_strict` Python companion (no Java call)

### Step 3 — Checklist

- [ ] `TestConstruction` covers every constructor path and all default values
  - Abstract class: use a concrete subclass fixture; verify base-class invariants (count, no-raise on construction). Do not omit `TestConstruction` because the class is abstract.
  - Java interface: write one `Test<Implementor>` class per concrete inner class. Each must call construction + the key mapping method + `getResult` (or UV2-algebra equivalent) — the last catches silent R4-violations.
- [ ] At least one `Test<Behaviour>` per distinct method category
- [ ] `TestHypothesis` with `@given` + `@slow` present
- [ ] Abstract → parity section is `@pytest.mark.skip(M4)` with concrete subclass validation
- [ ] Concrete → `@needs_java` parity tests cover all public methods
- [ ] Spec bugs (Suspected Java bugs section) have both a preserved-bug parity test and a strict-variant test; no bugs invented beyond what the spec lists
- [ ] `if __name__ == "__main__":` block at bottom of file
- [ ] No port code written — the port does not exist yet; the harness is pre-written
- [ ] Expected values derived from Java source or closed-form math, NOT from the port's algorithm (the port doesn't exist yet — see TESTING_GUIDE "Read the port before fixing the expected value" once the port exists)
- [ ] Item/argument types in every test call match the **Java method signature** — if Java requires `int item`, never use a string, list, or other object as the value
- [ ] `@needs_java` parity tests call Java constructors with the **actual Java signature** (look it up in the Java source), not any generalized Python form; add a comment showing the Java constructor when they differ
- [ ] Port module imported with the **bare module name** (`from <CLASS_NAME>_ver1_1_0 import ...`), never with the full dotted path (`gov.nist.microanalysis.PyEPQ.Utility.<CLASS_NAME>_ver1_1_0`). The three-tier import pattern in CONVERSION_GUIDE R3 is for port files importing `_epq_compat` — it does NOT apply here.
- [ ] Private and protected Java fields are referenced as `_mField` (R1 underscore prefix), never as `mField`. The test accesses the Python name, not the Java name.
- [ ] Any method listed in `spec/<CLASS_NAME>.spec.md` under inner-class API surfaces is tested by name from the spec — do not rename `adjacent()` to `adjacent_bin()`, `getDimensions()` to `dimensions()`, etc. unless the spec explicitly introduces the alias.
- [ ] The return value of every method on all code paths is explicitly confirmed with the spec before writing the assertion — `None`, a zero vector, and an empty list are all distinct; do not assume.

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

Review the compliance report (src\gov\nist\microanalysis\PyEPQ\tools\reports\compliance_Utility_ver2.md) and determine which ports need repair. You will find detailed test results for each port in the results archive (src\gov\nist\microanalysis\PyEPQ\Utility_ver2\tests\archive). Repair the ports in a sequential manner, without paralellization. Update the versioning scheme in accordance with the number of port repairs.

You are repairing a Python port that is failing its parity test suite.
Your task is to read the test failures, identify the root cause in the port,
and produce a corrected port file. Do **not** modify the test file unless a
test itself is provably wrong (wrong expected value, wrong item type, etc.) —
if you must touch the test, state the reason explicitly and keep the change minimal
while incrementing versioning respectively.

### Step 1 — Read the failure log

Open the output log (ex. `test_output.txt`) and for every `FAILED` entry:

1. Copy the **full traceback** (not just the summary line).
2. Identify which assertion or exception fires, and on which line.
3. State whether the failure is in the **port** (wrong API name, wrong return type,
   wrong calculation) or in the **test** (wrong expected value, wrong argument type).

Classify each failure as one of:
- **R1-violation** — wrong method/field name (underscore prefix on public member, renamed method, etc.)
- **R2-omission** — public Java method missing from port, `_literal` companion absent, or `equals()`/`hashCode()`/`toString()` missing named alias
- **R4-violation** — overload split missing, wrong constructor dispatch
- **R9-violation** — wrong return type (e.g. float returned where bool expected)
- **assert-error** — Python `assert` used to represent a Java `assert`; clamp-only path is blocked by the assertion
- **parent-alias-error** — subclass calls `self.foo_ode()` or similar name that was never defined; fix by binding `_alias = ParentClass.method` in the class body
- **API-mismatch** — constructor signature wrong (argument types/count), method renamed
- **Test-bug** — test passes wrong argument type (e.g. string where int required), queries an item that was never added, or accesses a Java private field as `mField` instead of `_mField`

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

- Every failure in `test_output.txt` is addressed (port fix or documented test fix)
- No passing test is broken by the repair
- `Port-code fixes` counter in the module docstring header is incremented
- `CHANGES` section lists each fix with its category (R1/R2/R4/etc.)
- Module-level aliases added for any inner class that must be directly importable
- No new functionality introduced beyond what fixes the failures
- If FIX-N changes behaviour at a specific input boundary (crash → sentinel, wrong value → correct value), apply the FIX-N two-file discipline from TESTING_GUIDE §"FIX-N boundary divergence": update the test harness with the boundary pin AND update the parity guard. Both file changes are required; neither alone is sufficient.
- If FIX-N adds an `assume` guard rather than a code change, add the exclusion comment per TESTING_GUIDE §"assume-only fixes". A deterministic pin is only required when the excluded values have a well-defined Python result.

