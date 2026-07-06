"""
run_parity.py — run all parity harnesses and report only failures.

Output is written to both the terminal and run_parity.txt in this directory.

Usage:
    python run_parity.py                  # all test_parity_*.py
    python run_parity.py math2            # files whose name contains 'math2'
    python run_parity.py multid integrator  # multiple filters (OR)
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
_OUTPUT   = _TESTS_DIR / "run_parity.txt"
_WIDTH    = 64

# Matches pytest result lines: "3 passed", "1 failed", "2 passed, 1 skipped in 0.4s"
_RESULT_RE = re.compile(r'\d+\s+(passed|failed|error)')


def _run_file(path: Path) -> tuple[bool, str, str]:
    """Run one test file. Returns (passed, stdout, stderr)."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", path.name,
         "--tb=short", "-q", "--no-header", "--color=no"],
        capture_output=True, text=True,
        cwd=str(_TESTS_DIR),
    )
    return proc.returncode == 0, proc.stdout.strip(), proc.stderr.strip()


def _summary_line(stdout: str) -> str:
    """Extract pytest result line from stdout (not stderr, to avoid conftest noise)."""
    for line in reversed(stdout.splitlines()):
        if line.strip() and _RESULT_RE.search(line):
            return line.strip()
    return "(no summary)"


class _Tee:
    """Writes to both the terminal and the output file."""

    def __init__(self, path: Path) -> None:
        self._fh = path.open("w", encoding="utf-8")

    def __call__(self, *args, **kwargs) -> None:
        kwargs["flush"] = True
        print(*args, **kwargs)
        print(*args, **{k: v for k, v in kwargs.items() if k != "flush"},
              file=self._fh, flush=True)

    def close(self) -> None:
        self._fh.close()


def main() -> int:
    filters = [a.lower() for a in sys.argv[1:]]
    files = sorted(_TESTS_DIR.glob("test_parity_*.py"))
    if filters:
        files = [f for f in files if any(x in f.name.lower() for x in filters)]

    if not files:
        print("No matching test files found.", flush=True)
        return 1

    p = _Tee(_OUTPUT)

    p(f"\n{'='*_WIDTH}")
    p(f"  Parity run — {len(files)} suite(s)  →  {_OUTPUT.name}")
    p(f"{'='*_WIDTH}\n")

    passed: list[tuple[str, str]] = []
    failed: list[tuple[str, str]] = []

    for i, f in enumerate(files, 1):
        p(f"  [{i:2}/{len(files)}] {f.name} ...", end=" ")
        ok, stdout, stderr = _run_file(f)
        p("PASSED" if ok else "FAILED")
        if ok:
            passed.append((f.name, stdout))
        else:
            # Include stderr (conftest messages, import errors) only in failures
            combined = "\n".join(x for x in (stdout, stderr) if x)
            failed.append((f.name, combined))

    if failed:
        p(f"\n{'─'*_WIDTH}")
        p(f"  FAILURES  ({len(failed)})")
        p(f"{'─'*_WIDTH}")
        for name, output in failed:
            p(f"\n  >>> {name}")
            for line in output.splitlines():
                p(f"  {line}")

    p(f"\n{'─'*_WIDTH}")
    p(f"  PASSED  ({len(passed)})")
    p(f"{'─'*_WIDTH}")
    for name, stdout in passed:
        p(f"  {name:<52}  {_summary_line(stdout)}")

    p(f"\n{'='*_WIDTH}")
    p(f"  {len(passed)} passed, {len(failed)} failed")
    p(f"{'='*_WIDTH}\n")

    p.close()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
