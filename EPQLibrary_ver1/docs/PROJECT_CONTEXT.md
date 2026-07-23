# EPQ Project Context

Version 2 · 2026-06-24

Project-level context and orchestration notes for the EPQ Java → Python conversion.
Attach this file when an agent needs a broader view of the project structure. For
focused single-class tasks, the individual guides are sufficient without this file.

---

## Guide map

| Guide | Covers |
|---|---|
| **CONVERSION_GUIDE.md** | Code-conversion rules (R1-R10), naming, overloads, library substitution |
| **TESTING_GUIDE.md** | Parity-test harness construction, tolerances, test disciplines |
| **BUG_GUIDE.md** | Bug/deviation tracking convention (`BUG_LEDGER`, marker vocabulary) |
| **PROMPTS.md** | Copy-paste agent prompts for each workflow step |

The guides are independent: each covers its phase only. Ordering across phases is
controlled at the project level and is not prescribed by any individual guide.

---

## Prompt construction

Every conversion prompt should include the line:
**"Follow CONVERSION_GUIDE.md R1-R10 and produce the port per Producing the Port."**

Prompts are self-contained units — attach the context files listed in each prompt header
and send. Prompt numbers are reference labels, not an execution sequence.

### When to add a new prompt

Add a prompt to PROMPTS.md only when:
- The same task runs on ≥3 files (e.g. a new migration phase introduces a new artifact type).
- The prompt is non-trivial to reconstruct from memory.
- Correctness depends on details (specific tolerances, library APIs) a future contributor might not recall.

One-off tasks ("audit this file for missing JAVA-BUG-N markers") stay ad-hoc in conversation.

---

## R2 completeness verification

After generating a port, verify completeness against the Java source:

`grep` for `public `, `private `, `protected `, `static ` in the Java file. The Python
port must have the same total declarations. Mismatched counts mean something was silently
dropped.

---

## Full methodology limits

Extends CONVERSION_GUIDE.md Appendix A (which lists M1, M2, M4). Full table:

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

## Testing infrastructure maintenance

### Extending `_parity_lib.py`

Add to the library when a pattern repeats across **≥3 files**. For one-off patterns,
keep them in the per-file test. Over-abstraction makes test failures harder to diagnose.

### Branch coverage

Run periodically, not every commit.

```powershell
python -m pip install coverage

python -m coverage run --branch --include="*<Class>_ver*.py" `
    -m pytest src\gov\nist\microanalysis\PyEPQ\<Subpkg>\tests\test_parity_<class>_ver1_1_0.py
python -m coverage report
python -m coverage html  # browsable htmlcov/index.html
```

Target **>95% branch coverage**. Mark unreachable defensive guards with
`# pragma: no cover` and document why they are unreachable.

---

## JPype + JVM setup (Windows)

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

### Where to drop the jar

Discovery order:
1. `$EPQ_JAR` env var.
2. `PyEPQ/lib/epq.jar` or `PyEPQ/lib/EPQ.jar`.
3. `<repo-root>/lib/EPQ.jar`.

All `*.jar` in `PyEPQ/lib/` join the classpath automatically.
