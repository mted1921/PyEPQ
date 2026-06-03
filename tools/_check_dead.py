"""
_check_dead.py -- one-off: validate "dead-code candidates" against the
entire gov.nist.microanalysis source tree.

For each class flagged is_dead in EPQLibrary_nodes.csv, search every
.java file OUTSIDE the EPQLibrary directory for:
  * Full-FQN import:   import gov.nist.microanalysis.EPQLibrary.<Name>;
  * Bare class usage:  word-boundary match on the simple name (after
                       comment/string stripping to avoid false positives).

Reports which candidates are truly dead vs which have cross-package callers.
"""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd

ROOT: Path = Path(__file__).resolve().parents[6]
MICRO: Path = ROOT / "src" / "gov" / "nist" / "microanalysis"
EPQLIB: Path = MICRO / "EPQLibrary"
# After the restructure, generated CSVs live in tools/reports/ next to
# this script (script lives at PyEPQ/tools/_check_dead.py).
NODES_CSV: Path = Path(__file__).resolve().parent / "reports" / "EPQLibrary_nodes.csv"

_RE_LINE_COMMENT = re.compile(r"//[^\n]*")
_RE_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_RE_STRING_LIT = re.compile(r'"(?:\\.|[^"\\])*"', re.DOTALL)


def strip_noise(src: str) -> str:
    src = _RE_BLOCK_COMMENT.sub("", src)
    src = _RE_LINE_COMMENT.sub("", src)
    src = _RE_STRING_LIT.sub('""', src)
    return src


def main() -> None:
    nodes_df: pd.DataFrame = pd.read_csv(NODES_CSV)
    dead: list[str] = nodes_df[nodes_df["is_dead"]]["class_name"].tolist()

    # Pre-compile per-class patterns.
    import_pat: dict[str, re.Pattern[str]] = {
        c: re.compile(
            rf"^\s*import\s+gov\.nist\.microanalysis\.EPQLibrary\.{re.escape(c)}\s*;",
            re.MULTILINE,
        )
        for c in dead
    }
    usage_pat: dict[str, re.Pattern[str]] = {
        c: re.compile(rf"\b{re.escape(c)}\b")
        for c in dead
    }

    # Scan every .java file outside EPQLibrary.
    files: list[Path] = [
        p for p in MICRO.rglob("*.java")
        if EPQLIB not in p.parents
    ]
    print(f"Scanning {len(files)} .java files outside EPQLibrary "
          f"(under {MICRO.relative_to(ROOT)})")

    # findings[class_name] = list of (kind, package, file_relpath)
    findings: dict[str, list[tuple[str, str, str]]] = {c: [] for c in dead}

    for f in files:
        try:
            raw: str = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        clean: str = strip_noise(raw)
        rel: str = str(f.relative_to(MICRO)).replace("\\", "/")
        pkg: str = rel.rsplit("/", 1)[0] if "/" in rel else ""

        for c in dead:
            kinds: list[str] = []
            if import_pat[c].search(clean):
                kinds.append("import")
            elif usage_pat[c].search(clean):
                # No import but the name appears -- could be same-package
                # (impossible here, EPQLibrary is excluded) or false positive.
                kinds.append("name-only")
            for k in kinds:
                findings[c].append((k, pkg, rel))

    truly_dead: list[str] = [c for c, fs in findings.items() if not fs]
    used: list[tuple[str, list[tuple[str, str, str]]]] = [
        (c, fs) for c, fs in findings.items() if fs
    ]

    print(f"\n{'=' * 78}")
    print(f"TRULY DEAD across microanalysis ({len(truly_dead)} of {len(dead)})")
    print('=' * 78)
    for c in truly_dead:
        print(f"  {c}")

    print(f"\n{'=' * 78}")
    print(f"USED FROM OUTSIDE EPQLIBRARY ({len(used)} of {len(dead)})")
    print('=' * 78)
    for c, fs in sorted(used, key=lambda kv: -len(kv[1])):
        pkgs: set[str] = {pkg for _, pkg, _ in fs}
        n_imp: int = sum(1 for k, _, _ in fs if k == "import")
        n_use: int = sum(1 for k, _, _ in fs if k == "name-only")
        print(f"  {c:<35} {n_imp:>3} imports + "
              f"{n_use:>3} name-only refs across "
              f"{len(pkgs)} package(s): {', '.join(sorted(pkgs))}")


if __name__ == "__main__":
    main()
