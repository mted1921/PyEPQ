"""
check_compliance.py -- unified compliance + parity gate for the EPQ port.

For every Python port in PyEPQ/<Subpkg>/ it verifies, against the corresponding
Java source in src/gov/nist/microanalysis/<Subpkg>/:

  spec-missing        a spec file exists at PyEPQ/<Subpkg>/spec/<Class>.spec.md
  test-missing        a parity harness exists at tests/test_parity_<class>*.py
  method:<name>       every public Java method name is referenced in the harness
                      (suffix-tolerant: Java `plus` matches test text `plus_vv`;
                      comments and docstrings are stripped first, so a name
                      merely mentioned in prose does not count as covered)
  no-needs-java       concrete classes have at least one @needs_java parity test
  no-given-fuzz       the harness has at least one @given hypothesis test
  no-m4-skip          abstract classes carry the documented M4 skip marker

Unless --no-parity is given, it also RUNS each port's parity suite (pytest,
one subprocess per class) and records pass/fail/skip counts and pass%. Parity
failures or errors fail the gate alongside new compliance violations. A
suite that only skips (e.g. JVM unavailable) is not a failure.

Pre-written harnesses (test file exists, port not yet generated) are reported
as PENDING, never as failures, and are not run.

Supporting documents stay fresh automatically on a normal run:
  * UTILITY_LEDGER.md  auto-synced from the filesystem
  * BUG_LEDGER.md  per-subpackage registry, generated from each port's
                   class-level BUG_LEDGER tuple (see docs/BUG_GUIDE.md)
  * baseline       resolved entries auto-PRUNED (never auto-added)
  * report         markdown written to tools/reports/Compliance_report.md
--check-only suppresses all file mutation (ledger/baseline/registry) for pure
CI gating; the report is still written.

Baseline file (compliance_baseline.txt next to this script): one failure key
per line; matching failures downgrade to WARN (pre-existing debt). New
violations still fail. Accept new debt deliberately with --update-baseline.

Usage:
    python check_compliance.py                  # full gate (static + parity)
    python check_compliance.py --no-parity      # static checks only (fast)
    python check_compliance.py --check-only      # gate without mutating files
    python check_compliance.py --update-baseline # accept current debt
    python check_compliance.py --sync-ledger     # update UTILITY_LEDGER.md, then exit
    python check_compliance.py Utility EPQLibrary   # explicit subpackages
"""

from __future__ import annotations

import argparse
import ast
import io
import re
import subprocess
import sys
import tokenize
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

_HERE: Path = Path(__file__).resolve().parent
_PYEPQ: Path = _HERE.parent                       # .../PyEPQ
_MICROANALYSIS: Path = _PYEPQ.parent              # .../gov/nist/microanalysis
_BASELINE_PATH: Path = _HERE / "compliance_baseline.txt"
_MD_REPORT_PATH: Path = _HERE / "reports" / "Compliance_report.md"

_DEFAULT_SUBPKGS: tuple[str, ...] = ("Utility",)

# Java members whose Python counterparts are dunders or entry points; their
# absence from a harness by literal name is not a coverage gap.
_AUTO_EXEMPT: frozenset[str] = frozenset({"equals", "hashCode", "toString", "main"})

_JAVA_COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/", re.DOTALL)
_JAVA_STRING_RE = re.compile(r'"(?:\\.|[^"\\])*"')
# `public [modifiers] [<T>] ReturnType name(`  -- methods only; constructors
# (no return type) are matched separately per class name.
_PUBLIC_METHOD_RE = re.compile(
    r"public\s+(?:(?:static|final|abstract|synchronized|native|strictfp)\s+)*"
    r"(?:<[^<>]*>\s+)?"
    r"(?!class\b|interface\b|enum\b)"
    r"[\w$][\w$\[\]<>?.,\s]*?\s+"
    r"([\w$]+)\s*\("
)

# Matches _ver<digit> anywhere in a stem — distinguishes versioned from plain files.
_VER_SUFFIX_RE = re.compile(r"_ver\d")

# A brace whose header contains one of these opens a named *type* body; methods
# directly inside it are public API. Anything else (method body, anonymous class)
# is not. Used by _java_public_api's brace scan.
_TYPE_KW_RE = re.compile(r"\b(?:class|interface|enum)\b")

# Pytest summary parsing (parity run). The summary line looks like
# "1 failed, 231 passed in 19.68s"; counts are extracted from it.
_PYTEST_SUMMARY_LINE_RE = re.compile(
    r"\b\d+\s+(?:passed|failed|error|errors|skipped|deselected|xfailed|xpassed|no tests ran)"
)
_PYTEST_COUNT_RE = re.compile(r"(\d+)\s+(passed|failed|errors?|skipped)")

# Ledger table patterns (used by --sync-ledger).
_LEDGER_TIER_RE    = re.compile(r"^## Tier (\d+)\s*[-—]?\s*(.*)")
_LEDGER_GREY_RE    = re.compile(r"^## Grey", re.IGNORECASE)
_LEDGER_SUMMARY_RE = re.compile(r"^## Progress summary", re.IGNORECASE)
# 7-column class row: | `Name` | spec | sym | file | sym | file | deps |
_LEDGER_ROW7_RE    = re.compile(r"^\|\s*`(\w+)`\s*\|(?:[^|]*\|){5}[^|]*\|")
# 2-column grey row:  | `Name` | notes |
_LEDGER_ROW2_RE    = re.compile(r"^\|\s*`(\w+)`\s*\|[^|]*\|$")


@dataclass
class ClassReport:
    name: str
    failures: list[str] = field(default_factory=list)   # stable keys
    warnings: list[str] = field(default_factory=list)   # informational
    pending: bool = False
    # parity-run results (populated only when the suite is run)
    parity_ran: bool = False
    parity_passed: int = 0
    parity_failed: int = 0
    parity_errors: int = 0
    parity_skipped: int = 0
    parity_note: str = ""    # e.g. "pytest unavailable"

    @property
    def parity_failed_gate(self) -> bool:
        """True iff this suite's parity run should fail the gate."""
        return self.parity_ran and (self.parity_failed > 0 or self.parity_errors > 0)


@dataclass
class TierStats:
    label: str
    classes: int = 0
    specs: int = 0   # ✓ only
    ports: int = 0   # ✓ or ~
    tests: int = 0   # ✓ only


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _strip_python_noise(src: str) -> str:
    """Drop comments and string/docstring literals from Python source.

    Method-name coverage must reflect what the harness actually *exercises*, not
    what its module docstring or comments happen to mention. Tokenizing and
    discarding COMMENT/STRING (and f-string literal) tokens leaves only code.
    Falls back to the raw source if tokenization fails."""
    skip = {tokenize.COMMENT, tokenize.STRING}
    fstring_middle = getattr(tokenize, "FSTRING_MIDDLE", None)
    if fstring_middle is not None:
        skip.add(fstring_middle)
    try:
        return " ".join(
            tok.string
            for tok in tokenize.generate_tokens(io.StringIO(src).readline)
            if tok.type not in skip
        )
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return src


def _parse_pytest(out: str) -> dict[str, int]:
    """Extract pass/fail/error/skip counts from pytest's summary line."""
    counts = {"passed": 0, "failed": 0, "error": 0, "skipped": 0}
    summary = next(
        (ln for ln in reversed(out.splitlines())
         if _PYTEST_SUMMARY_LINE_RE.search(ln)),
        None,
    )
    if summary:
        for m in _PYTEST_COUNT_RE.finditer(summary):
            n, kind = int(m.group(1)), m.group(2)
            counts["error" if kind.startswith("error") else kind] = n
    return counts


def _run_parity(test_path: Path) -> dict[str, object]:
    """Run one parity suite as a subprocess and return its result counts.

    Uses the bare filename with cwd set to the tests directory so the harness's
    `from <Class>_ver... import ...` resolves (mirrors run_parity.py)."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", test_path.name,
             "-q", "--no-header", "-p", "no:cacheprovider"],
            capture_output=True, text=True, cwd=str(test_path.parent),
        )
    except (FileNotFoundError, OSError) as e:
        return {"ran": False, "note": f"pytest unavailable: {e}"}

    counts = _parse_pytest(proc.stdout + "\n" + proc.stderr)
    # rc 5 = no tests collected; 2/3/4 = interrupted/internal/usage error.
    if proc.returncode in (2, 3, 4) and counts["error"] == 0:
        counts["error"] = 1
    return {"ran": True, **counts}


def _java_public_api(java_src: str, class_name: str) -> tuple[set[str], bool]:
    """Return (public method names, is_abstract) from Java source text.

    Only methods declared directly in a named type body (class/interface/enum)
    count. Methods inside an anonymous class or a method body are skipped — e.g.
    an `@Override public double function(...)` on a `new FindRoot.Function(){...}`
    is not part of the enclosing class's public API. A brace scan classifies each
    open brace by whether its header contains a class/interface/enum keyword."""
    stripped = _JAVA_STRING_RE.sub('""', _JAVA_COMMENT_RE.sub("", java_src))

    # Ordered scan of braces, statement terminators, and method declarations.
    events: list[tuple[int, str, str]] = []
    for mm in re.finditer(r"[{};]", stripped):
        events.append((mm.start(), mm.group(), ""))
    for ms in _PUBLIC_METHOD_RE.finditer(stripped):
        events.append((ms.start(), "M", ms.group(1)))
    events.sort(key=lambda e: e[0])

    names: set[str] = set()
    stack: list[bool] = []     # per open brace: True if it opens a named type body
    seg_start = 0              # start of the current header segment
    for pos, kind, name in events:
        if kind == "{":
            header = stripped[seg_start:pos]
            stack.append(bool(_TYPE_KW_RE.search(header)))
            seg_start = pos + 1
        elif kind == "}":
            if stack:
                stack.pop()
            seg_start = pos + 1
        elif kind == ";":
            seg_start = pos + 1
        elif name != class_name and stack and stack[-1]:  # method in a type body
            names.add(name)

    # Modifier order varies in this codebase: both `public abstract class X`
    # and `abstract public class X` occur. Match `abstract` anywhere on the
    # class declaration line.
    is_abstract = bool(re.search(
        rf"\babstract\b[^;{{]*\bclass\s+{re.escape(class_name)}\b", stripped,
    ))
    return names, is_abstract


def _base_class_name(port_stem: str) -> str:
    """Math2_ver1_1_0 -> Math2; Math2 -> Math2. Dotted archive stems
    (Math2_ver8.1.2) also reduce to the part before `_ver`."""
    return port_stem.split("_ver")[0]


def _discover_ports(port_dir: Path) -> dict[str, Path]:
    """Map class name -> chosen port file. Skips _infra and archived dotted-
    version files; prefers the suffix-free (renamed) port when both forms exist."""
    ports: dict[str, Path] = {}
    for p in sorted(port_dir.glob("*.py")):
        if p.stem.startswith("_") or "." in p.stem:
            continue
        base = _base_class_name(p.stem)
        if base not in ports or p.stem == base:
            ports[base] = p
    return ports


def _find_test_file(tests_dir: Path, class_name: str) -> Path | None:
    lower = class_name.lower()
    candidates = sorted(tests_dir.glob(f"test_parity_{lower}*.py"))
    # Exact-stem matches first: test_parity_math2.py / test_parity_math2_ver*.py.
    for p in candidates:
        token = p.stem[len("test_parity_"):].split("_ver")[0]
        if token == lower:
            return p
    return None


def _ledger_spec(cls: str, spec_dir: Path) -> str:
    """Symbol for the Spec column: ✓ present / ✗ missing."""
    return "✓" if (spec_dir / f"{cls}.spec.md").is_file() else "✗"


def _ledger_port(cls: str, port_dir: Path) -> tuple[str, str]:
    """(symbol, display) for the Port column: ✓ versioned / ~ plain / ✗ missing."""
    versioned = [p for p in sorted(port_dir.glob(f"{cls}_ver*.py"))
                 if "." not in p.stem]
    if versioned:
        return "✓", f"`{versioned[-1].name}`"
    plain = port_dir / f"{cls}.py"
    if plain.is_file():
        return "~", f"`{plain.name}`"
    return "✗", "—"


def _ledger_test(cls: str, tests_dir: Path) -> tuple[str, str]:
    """(symbol, display) for the Test column: ✓ versioned / ~ plain / ✗ missing."""
    lower = cls.lower()
    for p in sorted(tests_dir.glob(f"test_parity_{lower}*.py")):
        if _VER_SUFFIX_RE.search(p.stem[len(f"test_parity_{lower}"):]):
            return "✓", f"`{p.name}`"
    candidates = sorted(tests_dir.glob(f"test_parity_{lower}*.py"))
    if candidates:
        return "~", f"`{candidates[0].name}`"
    return "✗", "—"


def _sync_ledger(subpkg: str) -> None:
    """Rewrite UTILITY_LEDGER.md status/filename columns from the current filesystem.

    Updates the Spec symbol, Port symbol, Port filename, Test symbol, and Test
    filename columns for every 7-column class row.  All hand-authored content
    (tier headers, dependency column, grey-node notes, next-milestone footnote)
    is preserved verbatim.  The Progress summary table is regenerated from the
    updated row data.

    Section headers (``## Tier N — Label``) control the tier-label text used in
    the summary; rename headers there to get shorter summary row labels.
    """
    ledger_path = _PYEPQ / subpkg / "spec" / "UTILITY_LEDGER.md"
    if not ledger_path.is_file():
        print(f"  no UTILITY_LEDGER.md at {ledger_path}; skipping ledger sync",
              file=sys.stderr)
        return

    port_dir  = _PYEPQ / subpkg
    tests_dir = port_dir / "tests"
    spec_dir  = port_dir / "spec"

    tier_stats: list[TierStats] = []
    grey = TierStats("Grey / deferred")
    current_tier: int | None = None
    in_grey = in_summary = False

    out: list[str] = []

    for line in _read(ledger_path).splitlines():
        # ---- section headers ------------------------------------------------
        m = _LEDGER_TIER_RE.match(line)
        if m:
            n    = int(m.group(1))
            rest = m.group(2).split("(")[0].strip()
            while len(tier_stats) <= n:
                tier_stats.append(TierStats(str(len(tier_stats))))
            tier_stats[n].label = f"{n} — {rest}" if rest else str(n)
            current_tier, in_grey, in_summary = n, False, False
            out.append(line)
            continue

        if _LEDGER_GREY_RE.match(line):
            in_grey, in_summary = True, False
            out.append(line)
            continue

        if _LEDGER_SUMMARY_RE.match(line):
            in_summary = True
            # Inject fresh summary immediately after the heading.
            cc = sum(t.classes for t in tier_stats)
            cs = sum(t.specs   for t in tier_stats)
            cp = sum(t.ports   for t in tier_stats)
            ct = sum(t.tests   for t in tier_stats)
            out.extend([
                line, "",
                "| Tier | Classes | Spec ✓ | Port (✓/~) | Test ✓ |",
                "|---|:---:|:---:|:---:|:---:|",
                *[f"| {t.label} | {t.classes} | {t.specs} | {t.ports} | {t.tests} |"
                  for t in tier_stats],
                f"| **Connected total** | **{cc}** | **{cs}** | **{cp}** | **{ct}** |",
                f"| {grey.label} | {grey.classes} | {grey.specs} | {grey.ports} | {grey.tests} |",
                f"| **Grand total** | **{cc+grey.classes}** "
                f"| **{cs+grey.specs}** | **{cp+grey.ports}** | **{ct+grey.tests}** |",
            ])
            continue

        # Inside the old summary block: drop table/blank lines; pass `>` footnotes.
        if in_summary:
            if line.startswith("|") or line.strip() == "":
                continue
            # First non-table non-blank line (e.g. `> **Next milestone:**`)
            in_summary = False
            out.append("")   # blank line before footnote

        # ---- 7-column class rows (main tiers) --------------------------------
        if not in_grey and _LEDGER_ROW7_RE.match(line):
            parts = [p.strip() for p in line.split("|")]
            cls  = parts[1].strip("`")
            deps = parts[7] if len(parts) > 7 else "—"
            ss        = _ledger_spec(cls, spec_dir)
            ps, pf    = _ledger_port(cls, port_dir)
            ts, tf    = _ledger_test(cls, tests_dir)
            if current_tier is not None:
                t = tier_stats[current_tier]
                t.classes += 1
                if ss == "✓":        t.specs += 1
                if ps in ("✓", "~"): t.ports += 1
                if ts == "✓":        t.tests += 1
            out.append(f"| `{cls}` | {ss} | {ps} | {pf} | {ts} | {tf} | {deps} |")
            continue

        # ---- 2-column grey rows ---------------------------------------------
        if in_grey and _LEDGER_ROW2_RE.match(line):
            grey.classes += 1
            out.append(line)
            continue

        out.append(line)

    ledger_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"ledger synced: {sum(t.classes for t in tier_stats)} connected "
          f"class(es), {grey.classes} grey → {ledger_path.name}")


def _extract_bug_ledger(port_src: str) -> list[tuple]:
    """Return the class-level BUG_LEDGER tuple entries from a port's source.

    Each entry is normalised to (marker, method, description, has_strict_variant).
    Handles both `BUG_LEDGER: tuple = (...)` (annotated) and `BUG_LEDGER = (...)`.
    Returns [] when absent, empty, or not a literal."""
    try:
        tree = ast.parse(port_src)
    except SyntaxError:
        return []
    for node in ast.walk(tree):
        target = value = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target, value = node.target.id, node.value
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    target, value = t.id, node.value
        if target != "BUG_LEDGER" or value is None:
            continue
        try:
            raw = ast.literal_eval(value)
        except (ValueError, SyntaxError, TypeError):
            return []
        return [tuple(e) for e in raw if isinstance(e, (tuple, list))]
    return []


def _sync_bug_ledger(subpkg: str) -> None:
    """Generate PyEPQ/<subpkg>/BUG_LEDGER.md from each port's BUG_LEDGER tuple.

    The per-class tuples (documented in docs/BUG_GUIDE.md) are the source of
    truth; this aggregates them into one table. Regenerated on every run, so it
    is never hand-edited."""
    port_dir = _PYEPQ / subpkg
    if not port_dir.is_dir():
        return

    rows: list[tuple[str, str, str, str, bool]] = []
    for cls, p in _discover_ports(port_dir).items():
        for entry in _extract_bug_ledger(_read(p)):
            marker = str(entry[0]) if len(entry) > 0 else ""
            method = str(entry[1]) if len(entry) > 1 else ""
            desc = str(entry[2]) if len(entry) > 2 else ""
            strict = bool(entry[3]) if len(entry) > 3 else False
            rows.append((cls, marker, method, desc, strict))
    rows.sort(key=lambda r: (r[0], r[1]))

    n_classes = len({r[0] for r in rows})
    out: list[str] = [
        f"# {subpkg} Bug Ledger",
        "",
        f"_Auto-generated by check_compliance.py on {datetime.now():%Y-%m-%d %H:%M}. "
        "Do not edit by hand — edit the `BUG_LEDGER` tuple in each port. "
        "Convention: docs/BUG_GUIDE.md._",
        "",
        f"{len(rows)} entr{'y' if len(rows) == 1 else 'ies'} "
        f"across {n_classes} class(es).",
        "",
    ]
    if rows:
        out += [
            "| Class | Marker | Method | Strict variant | Description |",
            "|---|---|---|:---:|---|",
        ]
        for cls, marker, method, desc, strict in rows:
            cell = desc.replace("|", r"\|").replace("\n", " ")
            sv = "✓" if strict else "—"
            out.append(f"| `{cls}` | {marker} | `{method}` | {sv} | {cell} |")
    else:
        out.append("_No bugs or deviations recorded._")

    (port_dir / "spec" / "BUG_LEDGER.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"bug ledger synced: {len(rows)} entr"
          f"{'y' if len(rows) == 1 else 'ies'} → {subpkg}/spec/BUG_LEDGER.md")


def check_subpackage(subpkg: str, run_parity: bool = False
                     ) -> tuple[list[ClassReport], list[str]]:
    """Check one subpackage. Returns (per-class reports, pending-test names).

    When run_parity is True, each ported class's harness is executed and the
    pass/fail/skip counts are recorded on its ClassReport."""
    java_dir = _MICROANALYSIS / subpkg
    port_dir = _PYEPQ / subpkg
    tests_dir = port_dir / "tests"
    spec_dir = port_dir / "spec"

    if not port_dir.is_dir():
        return [], []

    # ---- discover ports (skip _infra, archived dotted versions) ----------
    ports = _discover_ports(port_dir)

    reports: list[ClassReport] = []
    for cls, port_path in ports.items():
        rep = ClassReport(name=f"{subpkg}.{cls}")
        java_path = java_dir / f"{cls}.java"
        test_path = _find_test_file(tests_dir, cls) if tests_dir.is_dir() else None

        if not (spec_dir / f"{cls}.spec.md").is_file():
            rep.failures.append(f"{subpkg}.{cls}.spec-missing")

        if test_path is None:
            rep.failures.append(f"{subpkg}.{cls}.test-missing")
            reports.append(rep)
            continue

        test_src = _read(test_path)
        # Method-coverage search runs against code only — a name mentioned in a
        # comment or the module docstring must not count as exercised. Marker
        # checks (M4 lives in a skip-reason string; decorators are code) use the
        # raw source.
        test_code = _strip_python_noise(test_src)

        if java_path.is_file():
            methods, is_abstract = _java_public_api(_read(java_path), cls)
            for name in sorted(methods - _AUTO_EXEMPT):
                # Suffix-tolerant: Java `plus` covered by `plus_vv` (R4 split).
                if not re.search(rf"\b{re.escape(name)}", test_code):
                    key = f"{subpkg}.{cls}.method:{name}"
                    (rep.warnings if is_abstract else rep.failures).append(key)
        else:
            is_abstract = False
            rep.warnings.append(
                f"{subpkg}.{cls}: no Java source at {java_path.name}; "
                "API coverage not checked"
            )

        if is_abstract:
            if "M4" not in test_src:
                rep.failures.append(f"{subpkg}.{cls}.no-m4-skip")
        else:
            if "@needs_java" not in test_src:
                rep.failures.append(f"{subpkg}.{cls}.no-needs-java")
        if "@given" not in test_src:
            rep.failures.append(f"{subpkg}.{cls}.no-given-fuzz")

        if run_parity:
            res = _run_parity(test_path)
            if res.get("ran"):
                rep.parity_ran = True
                rep.parity_passed = int(res.get("passed", 0))
                rep.parity_failed = int(res.get("failed", 0))
                rep.parity_errors = int(res.get("error", 0))
                rep.parity_skipped = int(res.get("skipped", 0))
            else:
                rep.parity_note = str(res.get("note", "not run"))

        reports.append(rep)

    # ---- pre-written harnesses: test exists, port doesn't ----------------
    pending: list[str] = []
    if tests_dir.is_dir():
        ported_lower = {c.lower() for c in ports}
        for p in sorted(tests_dir.glob("test_parity_*.py")):
            token = p.stem[len("test_parity_"):].split("_ver")[0]
            if token not in ported_lower:
                pending.append(f"{subpkg}: {p.name}")

    return reports, pending


def _status(rep: ClassReport, baseline: set[str]) -> str:
    if any(k not in baseline for k in rep.failures):
        return "FAIL"
    if rep.failures:
        return "WARN (baselined)"
    if rep.warnings:
        return "WARN"
    return "OK"


def _parity_cell(rep: ClassReport) -> str:
    """Compact parity result for a class: ratio + pass% / skipped / error / —."""
    if not rep.parity_ran:
        return rep.parity_note or "—"
    ran = rep.parity_passed + rep.parity_failed + rep.parity_errors
    if rep.parity_errors:
        return f"ERROR ({rep.parity_errors})"
    if rep.parity_failed:
        return f"FAIL {rep.parity_failed}✗/{ran}"
    if ran == 0:
        return f"skipped ({rep.parity_skipped})" if rep.parity_skipped else "no tests"
    return f"{rep.parity_passed}/{ran} (100%)"


def _write_md_report(reports: list[ClassReport], pending: list[str],
                     baseline: set[str], new_failures: list[str],
                     baselined: list[str], path: Path) -> None:
    """Render the run as markdown. Written on EVERY run so the report file
    always reflects the latest check (consumed by reviewers and agents)."""
    parity_run = any(r.parity_ran for r in reports)
    parity_fails = [r for r in reports if r.parity_failed_gate]
    overall = "FAILED" if (new_failures or parity_fails) else "PASSED"
    parity_clause = (
        f", {len(parity_fails)} parity failure(s)" if parity_run else
        " (parity not run)"
    )
    lines: list[str] = [
        "# Parity compliance report",
        "",
        f"_Generated by `check_compliance.py` on "
        f"{datetime.now():%Y-%m-%d %H:%M}._",
        "",
        f"**Result: {overall}** — {len(reports)} ported class(es) checked; "
        f"{len(new_failures)} new failure(s), {len(baselined)} baselined, "
        f"{len(pending)} pending harness(es){parity_clause}.",
        "",
        "## Ported classes",
        "",
        "| Class | Status | Parity | Findings |",
        "|---|---|---|---|",
    ]
    for rep in reports:
        findings: list[str] = []
        for key in rep.failures:
            tag = "baselined" if key in baseline else "**FAIL**"
            findings.append(f"{tag}: `{key}`")
        findings += [f"warn: {msg}" for msg in rep.warnings]
        lines.append(
            f"| {rep.name} | {_status(rep, baseline)} | {_parity_cell(rep)} | "
            f"{'<br>'.join(findings) if findings else '—'} |"
        )
    if pending:
        lines += ["", "## Pending pre-written harnesses", "",
                  "Test files whose port module has not been generated yet "
                  "(skipped at collection by conftest.py):", ""]
        lines += [f"- `{name}`" for name in pending]
    if new_failures:
        lines += ["", "## New compliance violations", "",
                  "Fix these, or — if the debt is deliberate — run "
                  "`--update-baseline` and justify in the commit message:", ""]
        lines += [f"- `{key}`" for key in new_failures]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_baseline(keys: list[str]) -> None:
    """(Re)write the baseline file with a fixed header and the given keys."""
    header = (
        "# compliance_baseline.txt -- pre-existing compliance debt.\n"
        "# Each line is a failure key check_compliance.py downgrades to WARN.\n"
        "# Resolved entries are auto-pruned; do NOT add new ports here.\n"
    )
    _BASELINE_PATH.write_text(header + "\n".join(keys) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("subpackages", nargs="*", default=list(_DEFAULT_SUBPKGS))
    ap.add_argument("--no-parity", action="store_true",
                    help="skip running the parity suites (fast static-only check)")
    ap.add_argument("--check-only", action="store_true",
                    help="do not mutate any file (ledger/baseline); report still "
                         "written. For pure CI gating.")
    ap.add_argument("--update-baseline", action="store_true",
                    help="accept all current failures as pre-existing debt")
    ap.add_argument("--sync-ledger", action="store_true",
                    help="regenerate UTILITY_LEDGER.md and BUG_LEDGER.md from the "
                         "filesystem, then exit")
    args = ap.parse_args(argv)

    if args.sync_ledger:
        for subpkg in (args.subpackages or list(_DEFAULT_SUBPKGS)):
            _sync_ledger(subpkg)
            _sync_bug_ledger(subpkg)
        return 0

    baseline: set[str] = set()
    if _BASELINE_PATH.is_file():
        baseline = {
            line.strip() for line in _read(_BASELINE_PATH).splitlines()
            if line.strip() and not line.startswith("#")
        }

    subpkgs = args.subpackages or list(_DEFAULT_SUBPKGS)
    run_parity = not args.no_parity
    all_reports: list[ClassReport] = []
    all_pending: list[str] = []
    for subpkg in subpkgs:
        if run_parity:
            print(f"running parity suites for {subpkg} "
                  "(use --no-parity to skip)...")
        reports, pending = check_subpackage(subpkg, run_parity=run_parity)
        all_reports.extend(reports)
        all_pending.extend(pending)

    if args.update_baseline:
        keys = sorted({k for rep in all_reports for k in rep.failures})
        _write_baseline(keys)
        print(f"baseline updated: {len(keys)} key(s) -> {_BASELINE_PATH.name}")
        baseline = set(keys)  # report the run against the fresh baseline

    new_failures: list[str] = []
    baselined: list[str] = []
    for rep in all_reports:
        for key in rep.failures:
            (baselined if key in baseline else new_failures).append(key)

    # ---- console ----------------------------------------------------------
    print(f"check_compliance: {len(all_reports)} ported class(es) checked")
    for rep in all_reports:
        print(f"  [{_status(rep, baseline):>16}] {rep.name}"
              + (f"  parity: {_parity_cell(rep)}" if rep.parity_ran or rep.parity_note else ""))
        for key in rep.failures:
            tag = "baselined" if key in baseline else "FAIL"
            print(f"        {tag:>9}: {key}")
        for msg in rep.warnings:
            print(f"             warn: {msg}")
    for name in all_pending:
        print(f"  [         PENDING] {name} (pre-written harness; port not yet generated)")

    _write_md_report(all_reports, all_pending, baseline,
                     new_failures, baselined, _MD_REPORT_PATH)
    print(f"\nmarkdown report written to {_MD_REPORT_PATH}")

    # ---- keep supporting documents fresh (skipped under --check-only) ------
    stale = baseline - {k for rep in all_reports for k in rep.failures}
    if args.check_only:
        if stale:
            print(f"\nnote: {len(stale)} baseline entr(ies) no longer fail "
                  "(not pruned: --check-only).")
    else:
        for subpkg in subpkgs:
            _sync_ledger(subpkg)
            _sync_bug_ledger(subpkg)
        if stale and not args.update_baseline:
            _write_baseline(sorted(baseline - stale))
            baseline -= stale
            print(f"baseline pruned: removed {len(stale)} resolved entr(ies) "
                  f"-> {_BASELINE_PATH.name}")

    # ---- gate -------------------------------------------------------------
    parity_fails = [r for r in all_reports if r.parity_failed_gate]
    print(f"\nsummary: {len(new_failures)} new failure(s), "
          f"{len(baselined)} baselined, {len(all_pending)} pending harness(es), "
          f"{len(parity_fails)} parity failure(s)")
    if new_failures or parity_fails:
        print("FAILED -- gate not met:")
        for key in new_failures:
            print(f"    compliance: {key}")
        for r in parity_fails:
            print(f"    parity: {r.name} ({_parity_cell(r)})")
        if new_failures:
            print("  (accept deliberate compliance debt with --update-baseline)")
        return 1
    print("PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
