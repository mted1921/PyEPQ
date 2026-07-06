"""
run_parity.py — run AdaptiveRungeKutta parity suites against every candidate.

Tests each candidate file through three harnesses:
  direct   test_parity_adaptiverungekutta_ver1_1_0.py  (tests ARK directly)
  integr   test_parity_integrator_ver1_1_1.py          (Integrator depends on ARK)
  mcintgr  test_parity_mcintegrator_ver1_1_1.py        (MCIntegrator → Integrator → ARK)

Candidates are discovered automatically from agent subfolders in candidates/.
No file-moving or import-editing required — injection via --port-file in conftest.

Usage (from PyEPQ/ or agent_evaluations/):
    python agent_evaluations/run_parity.py
    python agent_evaluations/run_parity.py -v
    python agent_evaluations/run_parity.py --suite ark        # direct only
    python agent_evaluations/run_parity.py --suite integr     # indirect only

Results are written to agent_evaluations/reports/parity_results.md.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_HERE       = Path(__file__).resolve().parent                    # agent_evaluations/
_TESTS_DIR  = _HERE.parent / "Utility_ver1" / "tests"
_CANDIDATES = _HERE / "candidates"
_OUT_DIR    = _HERE / "reports"
_PORT_NAME  = "AdaptiveRungeKutta_ver1_1_2"

# (label, filename) — order matters for the output table columns
_SUITES: list[tuple[str, str]] = [
    ("ark",    "test_parity_adaptiverungekutta_ver1_1_0.py"),
    ("integr", "test_parity_integrator_ver1_1_1.py"),
]

_RESULT_RE = re.compile(r'(\d+)\s+(passed|failed|error|skipped)')


def _collect_candidates() -> list[tuple[str, str, Path]]:
    """Return (label, agent, path) for every .py in a subfolder of candidates/."""
    return sorted(
        (f"{d.name}/{f.name}", d.name, f)
        for d in sorted(_CANDIDATES.iterdir()) if d.is_dir()
        for f in sorted(d.glob("*.py")) if not f.name.startswith("_")
    )


def _parse_counts(output: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for m in _RESULT_RE.finditer(output):
        counts[m.group(2)] = int(m.group(1))
    return counts


def _run_suite(
    suite_file: Path,
    candidate: Path,
    verbose: bool,
) -> tuple[bool, dict[str, int], str]:
    """Run one suite against one candidate. Returns (passed, counts, output)."""
    cmd = [
        sys.executable, "-m", "pytest",
        suite_file.name,
        f"--port-file={candidate}",
        f"--port-name={_PORT_NAME}",
        "--tb=short" if verbose else "--tb=no",
        "--no-header", "-q",
        "-p", "no:cacheprovider",
    ]
    # Ensure Utility/ is on PYTHONPATH so candidates can resolve _epq_compat
    # before falling through to the gov.nist.* package path.
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_TESTS_DIR.parent) + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(_TESTS_DIR), env=env,
    )
    combined = r.stdout + r.stderr
    counts = _parse_counts(combined)
    passed = r.returncode == 0
    return passed, counts, combined


def _fmt_result(passed: bool, counts: dict[str, int]) -> str:
    p = counts.get("passed", 0)
    f = counts.get("failed", 0) + counts.get("error", 0)
    s = counts.get("skipped", 0)
    tag = "PASS" if passed else "FAIL"
    detail = f"{p}p/{f}f/{s}s"
    return f"{tag} ({detail})"


def _parity_pct(counts: dict[str, int]) -> float:
    """Percentage of executed tests that passed (skips excluded).

    passed / (passed + failed + error) * 100.  Returns 0.0 when no test
    executed — e.g. a collection error from an empty or unparseable
    candidate counts as a 0% port rather than being dropped.
    """
    p = counts.get("passed", 0)
    f = counts.get("failed", 0) + counts.get("error", 0)
    total = p + f
    return 100.0 * p / total if total else 0.0


class _Tee:
    def __init__(self, path: Path) -> None:
        self._fh = path.open("w", encoding="utf-8")

    def __call__(self, *args, **kwargs) -> None:
        kwargs.setdefault("flush", True)
        print(*args, **kwargs)
        print(*args, **{k: v for k, v in kwargs.items() if k != "flush"},
              file=self._fh, flush=True)

    def close(self) -> None:
        self._fh.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Run AdaptiveRungeKutta parity suites against all candidates.",
    )
    ap.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show short tracebacks for failures.",
    )
    ap.add_argument(
        "--suite", metavar="PATTERN", default=None,
        help="Run only suites whose label matches PATTERN (e.g. 'ark', 'integr').",
    )
    args = ap.parse_args(argv)

    # Filter suites
    suites = [
        (lbl, fname) for lbl, fname in _SUITES
        if args.suite is None or args.suite.lower() in lbl
    ]
    # Resolve and skip missing suite files
    suite_files = [(lbl, _TESTS_DIR / fname) for lbl, fname in suites]
    suite_files = [(lbl, p) for lbl, p in suite_files if p.is_file()]
    if not suite_files:
        print("No matching suite files found in", _TESTS_DIR)
        return 1

    candidates = _collect_candidates()
    if not candidates:
        print("No candidate files found in subfolders of", _CANDIDATES)
        return 1

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUT_DIR / "parity_results.md"
    p = _Tee(out_path)

    suite_labels = [lbl for lbl, _ in suite_files]
    n_agents = len({agent for _, agent, _ in candidates})
    n_runs = len(candidates) * len(suite_files)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    p("# AdaptiveRungeKutta parity report\n")
    p(f"_Generated by run_parity.py on {ts}._\n")
    p(f"- Port name: `{_PORT_NAME}`")
    p(f"- Candidates: {len(candidates)} file(s) across {n_agents} agent(s); "
      f"{n_runs} suite run(s).")
    p(f"- Suites: {', '.join(f'`{lbl}`' for lbl in suite_labels)}")
    p("- Result key: **PASS**/**FAIL** followed by "
      "`Xp/Yf/Zs` (passed / failed+error / skipped).\n")

    # ── Per-candidate results table ────────────────────────────────────
    p("## Per-candidate results\n")
    p("Each candidate port run through every suite.\n")
    p("| Candidate | " + " | ".join(suite_labels) + " |")
    p("|---|" + "|".join([":---:"] * len(suite_labels)) + "|")

    # Results[candidate_label][suite_label] = (passed, counts)
    results: dict[str, dict] = {}
    failures: list[tuple[str, str, str]] = []  # (candidate, suite, output)

    for label, agent, cand_path in candidates:
        results[label] = {}
        cells = []
        for suite_lbl, suite_path in suite_files:
            passed, counts, output = _run_suite(suite_path, cand_path, args.verbose)
            results[label][suite_lbl] = (passed, counts)
            cells.append(_fmt_result(passed, counts))
            if not passed:
                failures.append((label, suite_lbl, output))
        p(f"| `{label}` | " + " | ".join(cells) + " |")

    # ── Summary: mean parity score per candidate ───────────────────────
    # A "candidate" is an agent (model), not an individual port. Each agent
    # contributes several ports (typically 4), and every port is run through
    # every suite. The candidate score is the mean of the per-run parity
    # percentages across all of that agent's ports and suites.
    agent_pcts: dict[str, list[float]] = {}
    agent_ports: dict[str, set[str]] = {}
    for label, agent, _ in candidates:
        agent_ports.setdefault(agent, set()).add(label)
        for suite_lbl in suite_labels:
            if suite_lbl in results[label]:
                _passed, counts = results[label][suite_lbl]
                agent_pcts.setdefault(agent, []).append(_parity_pct(counts))

    ranked = sorted(
        (
            (agent, (sum(pcts) / len(pcts) if pcts else 0.0), len(agent_ports[agent]))
            for agent, pcts in agent_pcts.items()
        ),
        key=lambda r: r[1],
        reverse=True,
    )

    p("\n## Summary — mean parity per candidate\n")
    p("A \"candidate\" is an agent (model), not an individual port. Each score "
      "is the mean of that agent's per-run parity percentages across all of its "
      "ports and suites (skipped tests excluded), ranked high to low.\n")
    p("| Rank | Candidate | Parity | Ports |")
    p("|---:|---|---:|---:|")
    for rank, (agent, score, n_ports) in enumerate(ranked, 1):
        p(f"| {rank} | `{agent}` | {score:.1f}% | {n_ports} |")

    n_total = len(candidates) * len(suite_files)
    n_fail = len(failures)
    n_pass = n_total - n_fail

    p(f"\n**{n_pass}/{n_total} suite runs passed, {n_fail} failed.**")

    # ── Failure details ────────────────────────────────────────────────
    if failures:
        p("\n## Failure details\n")
        for cand_lbl, suite_lbl, output in failures:
            p(f"### [{suite_lbl}] `{cand_lbl}`\n")
            p("```")
            for line in output.strip().splitlines():
                p(line)
            p("```\n")

    # Terminal-only notice — kept out of the Markdown body.
    print(f"\nReport written: {out_path}")
    p.close()
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
