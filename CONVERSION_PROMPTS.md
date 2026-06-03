# Conversion Prompts

Reusable prompt templates for AI-agent-driven EPQ Java → Python
conversion. Both reference rules and patterns in
[CONVERSION_GUIDE.md](CONVERSION_GUIDE.md).

Substitute the `<placeholder>` tokens before sending each prompt. Run
them in order: **Conversion first** (Steps 1-2), then **Parity Harness**
(Steps 3-5) after the literal port is in place.

---

## 1. Conversion prompt (Steps 1-2)

For porting a Java file to Python.

> Convert `<path/to/File.java>` to Python at `<path/to/File_ver1.py>`,
> following CONVERSION_GUIDE.md R1-R10.
>
> First produce the Step 1 conversion specification at
> `<path>/File_ver1.spec.md`.
>
> Then implement the literal port: split overloads (R4), add mutation
> guards (R5), replicate Java signum (R8), preserve Java bugs in
> `BUG_LEDGER` (R6).
>
> Import shared types from `_epq_compat` (R3) — never define local
> stand-ins.
>
> Do NOT generate the parity harness yet.

---

## 2. Parity harness prompt (Steps 3-5)

For generating the differential test file.

> Generate `test_parity_<file>.py` in `PyEPQ/<Package>/tests/`,
> following the reference at `PyEPQ/Utility/tests/test_parity_math2.py`.
>
> 1. **Module setup**: copy the Per-file template from CONVERSION_GUIDE.md
>    "The Parity Harness". Substitute `<NAME>`, `<Class>`, `<Package>`.
>    Import all infrastructure from `_parity_lib` — do NOT re-implement
>    JVM startup, tolerances, hypothesis profiles, or comparators.
>
> 2. **Part 1 (always-on)**:
>    - `TestStrictVariants` — one per `*_strict` in BUG_LEDGER.
>    - `TestMutationGuards` — for each `*Equals`/`*InPlace`: lists,
>      wrong-dtype, and read-only inputs raise TypeError; valid input
>      mutates.
>    - `TestSelfConsistency` — mathematical identities
>      (`distance(a, a) == 0`, etc.).
>    - If the port uses `JavaRandom`: `TestJavaRandomFixedSequence`
>      with seed 42's first 5 nextDouble values as regression check.
>
> 3. **TestBoundaryValues** — one parametrized table per public
>    function. Cover NaN, ±Inf, ±0, singularities, known algebraic
>    values. Source from: exact arithmetic, closed-form expressions,
>    or `scipy.special`. Use `_bdry_close`.
>
> 4. **Part 2 (parity)** — `@needs_java + @slow + @given`. Choose
>    tolerance from the ladder. For reductions: `_close(..., rtol=TOL_REL)`.
>    For NR substitutions: use `nr_arg` not `positive`.
>
> 5. **Bug-aware tests** — for each `BUG_LEDGER` entry: parity test
>    buggy form vs Java. For `has_strict_variant=True`: also unit-test
>    the `*_strict` companion against a known expected value.
>
> Do NOT add to `_parity_lib.py` unless a pattern repeats across ≥3 files.

---

## When to add a new prompt here

Add a prompt only if it represents a **repeated agent task with stable
substitutions**. One-off tasks (e.g. "audit this file for missing
JAVA-BUG-N markers", "regenerate the dependency report") don't belong
here — keep them ad-hoc in the conversation.

A new prompt is justified when:
- The same task will be run on ≥3 files (e.g. when a new conversion
  phase introduces a new artifact like a benchmark file).
- The prompt is non-trivial to reconstruct from memory.
- The prompt's correctness depends on details a future contributor
  might not remember (specific tolerance choices, library APIs, etc.).
