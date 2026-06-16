"""
compare_variants.py -- measure variance between AI-generated code conversions.

The PyEPQ port is produced by a prompt-driven AI workflow (docs/PROMPTS.md). To
gauge how reproducible that workflow is, drop several AI-generated conversions of
the SAME class (responses to one prompt) into a folder and run this tool. It
scores every unordered pair of files with two similarity metrics:

  CodeBLEU  code-aware similarity = weighted sum of four components:
              * n-gram match            (token BLEU)
              * weighted n-gram match   (BLEU with Python keywords up-weighted)
              * syntax match            (AST subtree overlap)
              * dataflow match          (variable def-use edge overlap)
            Scored SYMMETRICALLY for pairwise variance; DIRECTIONALLY when
            a canonical reference is provided (canonical = reference).
  ROUGE-N   token-n-gram overlap on a code-aware token stream (comments and
            layout stripped). The F1 figure is symmetric and is the headline
            ROUGE score; precision/recall are also recorded.

Agent layout
------------
If the input directory contains subdirectories, each subdirectory is treated as
one AI agent's responses (e.g. candidates/Claude/, candidates/ChatSRS/). Labels
in the report become "Agent/file.py". The aggregate section then breaks down
variance within each agent AND across agents.

Canonical comparison
--------------------
Pass --canonical PATH to score every candidate against an accepted reference port
(e.g. the committed AdaptiveRungeKutta_ver1_1_2.py). This is directional: the
canonical is the reference and each candidate is the hypothesis, so the score
answers "how closely did this AI output match the accepted port?"

CodeBLEU engine
---------------
Because every candidate is Python, the syntax/dataflow components are computed
from the standard-library `ast` module -- no `tree-sitter`, no compiler, no
install required (the "ast" engine, default). If the optional `codebleu` PyPI
package is installed it can be used instead (`--engine codebleu`).

Output (default: agent_evaluations/reports/):
  variance_<inputdir>.json   full breakdown (pairs + canonical + aggregates)
  variance_<inputdir>.md     human-readable report

Usage:
    python compare_variants.py                            # candidates/ (agent layout)
    python compare_variants.py path/to/folder             # flat or agent layout
    python compare_variants.py --canonical path/to/ref.py # score vs accepted port
    python compare_variants.py --rouge-n 1 2 3
    python compare_variants.py --weights .25 .25 .25 .25
    python compare_variants.py --engine codebleu
    python compare_variants.py --out path/to/reports
"""

from __future__ import annotations

import argparse
import ast
import io
import itertools
import json
import keyword
import math
import re
import statistics
import subprocess
import sys
import tokenize
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Sequence

_HERE: Path = Path(__file__).resolve().parent     # agent_evaluations/
_PYEPQ: Path = _HERE.parent                        # .../PyEPQ
_DEFAULT_INPUT: Path = _HERE / "candidates"
_DEFAULT_OUT: Path = _HERE / "reports"
_PARITY_TEST: Path = _HERE / "test_parity_adaptiverungekutta_ver1_1_0.py"

_PYTEST_COUNT_RE = re.compile(r'(\d+)\s+(passed|failed|error|skipped)')

_NOISE_TYPES: set[int] = {
    tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE,
    tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING, tokenize.ENDMARKER,
}

_KEYWORDS: frozenset[str] = frozenset(
    list(keyword.kwlist) + list(getattr(keyword, "softkwlist", []))
)
_KEYWORD_WEIGHT: float = 4.0

_CB_KEYS: tuple[str, ...] = (
    "codebleu", "ngram_match_score", "weighted_ngram_match_score",
    "syntax_match_score", "dataflow_match_score",
)


# --------------------------------------------------------------------------- #
# Parity runner
# --------------------------------------------------------------------------- #
def _parse_pytest_counts(output: str) -> dict:
    counts: dict[str, int] = {}
    for m in _PYTEST_COUNT_RE.finditer(output):
        counts[m.group(2)] = int(m.group(1))
    return counts


def _run_parity_for_candidates(
    entries: list[tuple[str, str, Path]],
    test_file: Path,
    verbose: bool = False,
) -> list[dict]:
    """Run parity harness for each candidate via --port-file injection."""
    results: list[dict] = []
    for label, agent, path in entries:
        cmd = [
            sys.executable, "-m", "pytest",
            str(test_file),
            f"--port-file={path}",
            "--tb=no",
            "--no-header",
            "-q",
        ]
        r = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=str(test_file.parent),
        )
        counts = _parse_pytest_counts(r.stdout + r.stderr)
        passed = r.returncode == 0
        results.append({
            "candidate": label,
            "agent": agent,
            "passed": passed,
            "returncode": r.returncode,
            "counts": counts,
        })
        if verbose:
            status = "PASS" if passed else "FAIL"
            sys.stderr.write(f"  parity {label}: {status} {counts}\n")
    return results


# --------------------------------------------------------------------------- #
# File collection — flat layout or per-agent subdirectory layout
# --------------------------------------------------------------------------- #
def _collect_files(input_dir: Path) -> tuple[list[tuple[str, str, Path]], bool]:
    """Return (label, agent, path) triples and a flag indicating agent layout.

    Agent layout is detected when the input dir contains subdirectories that
    themselves contain .py files. In that case each subdirectory name is the
    agent label and .py files directly in input_dir are ignored (they are
    typically canonical/reference copies, not candidates).

    Flat layout: all .py files directly in input_dir; agent = ""."""
    subdirs = sorted(d for d in input_dir.iterdir() if d.is_dir())
    agent_files: list[tuple[str, str, Path]] = []
    for d in subdirs:
        for f in sorted(d.glob("*.py")):
            agent_files.append((f"{d.name}/{f.name}", d.name, f))

    if agent_files:
        return agent_files, True

    flat = [(p.name, "", p) for p in sorted(input_dir.glob("*.py")) if p.is_file()]
    return flat, False


# --------------------------------------------------------------------------- #
# I/O + tokenisation
# --------------------------------------------------------------------------- #
def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _code_tokens(src: str) -> list[str]:
    """Tokenise Python source into a clean code-token stream.

    Mirrors the robust tokenizer in tools/check_compliance.py."""
    try:
        return [
            tok.string
            for tok in tokenize.generate_tokens(io.StringIO(src).readline)
            if tok.type not in _NOISE_TYPES and tok.string != ""
        ]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return src.split()


def _ngram_counts(tokens: Sequence[str], n: int) -> Counter:
    if n <= 0 or len(tokens) < n:
        return Counter()
    return Counter(zip(*(tokens[i:] for i in range(n))))


# --------------------------------------------------------------------------- #
# ROUGE-N
# --------------------------------------------------------------------------- #
def rouge_n(tokens_a: Sequence[str], tokens_b: Sequence[str], n: int) -> dict:
    """ROUGE-N precision/recall/F1. Recall = overlap/|A|, precision = overlap/|B|."""
    counts_a = _ngram_counts(tokens_a, n)
    counts_b = _ngram_counts(tokens_b, n)
    total_a = sum(counts_a.values())
    total_b = sum(counts_b.values())
    overlap = sum((counts_a & counts_b).values())
    recall = overlap / total_a if total_a else 0.0
    precision = overlap / total_b if total_b else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


# --------------------------------------------------------------------------- #
# Built-in ast-based CodeBLEU (no tree-sitter required)
# --------------------------------------------------------------------------- #
def _kw_weight(tok: str) -> float:
    return _KEYWORD_WEIGHT if tok in _KEYWORDS else 1.0


def _bleu(ref: Sequence[str], cand: Sequence[str],
          wfn: Optional[Callable[[str], float]] = None, max_n: int = 4) -> float:
    """Sentence BLEU of *cand* against *ref*, optionally with per-token weights."""
    if not cand:
        return 0.0
    precisions: list[float] = []
    for n in range(1, max_n + 1):
        cand_ng = _ngram_counts(cand, n)
        if not cand_ng:
            precisions.append(0.0)
            continue
        ref_ng = _ngram_counts(ref, n)
        clipped = total = 0.0
        for gram, count in cand_ng.items():
            w = sum(wfn(t) for t in gram) if wfn else 1.0
            total += count * w
            clipped += min(count, ref_ng.get(gram, 0)) * w
        precisions.append(clipped / total if total else 0.0)
    eps = 1e-12
    log_mean = sum((1.0 / max_n) * math.log(p if p > 0 else eps) for p in precisions)
    c, r = len(cand), len(ref)
    bp = 1.0 if c > r else math.exp(1.0 - r / c)
    return bp * math.exp(log_mean)


def _ast_sigs(code: str) -> Counter:
    """Multiset of AST subtree signatures (node type + child node types)."""
    sigs: Counter = Counter()
    for node in ast.walk(ast.parse(code)):
        kids = tuple(type(k).__name__ for k in ast.iter_child_nodes(node))
        sigs[(type(node).__name__, kids)] += 1
    return sigs


def _dotted(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return None


def _target_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        name = _dotted(node)
        return [name] if name else []
    if isinstance(node, ast.Subscript):
        return _target_names(node.value)
    if isinstance(node, (ast.Tuple, ast.List)):
        out: list[str] = []
        for elt in node.elts:
            out += _target_names(elt)
        return out
    if isinstance(node, ast.Starred):
        return _target_names(node.value)
    return []


def _dfg_edges(code: str) -> Counter:
    """Variable def-use edges including attribute assignments (self.mX = ...)."""
    edges: Counter = Counter()
    for node in ast.walk(ast.parse(code)):
        value = None
        targets: list[str] = []
        if isinstance(node, ast.Assign):
            value = node.value
            for t in node.targets:
                targets += _target_names(t)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            value = node.value
            targets = _target_names(node.target)
        elif isinstance(node, ast.AugAssign):
            value = node.value
            targets = _target_names(node.target)
            for t in targets:
                edges[(t, t)] += 1
        if value is None or not targets:
            continue
        sources = [n.id for n in ast.walk(value) if isinstance(n, ast.Name)]
        for t in targets:
            for s in sources:
                edges[(t, s)] += 1
    return edges


def _counter_precision(ref: Counter, cand: Counter) -> float:
    if not cand:
        return 0.0
    matched = sum(min(c, ref.get(g, 0)) for g, c in cand.items())
    return matched / sum(cand.values())


def _dataflow_dir(ref: Counter, cand: Counter) -> float:
    if not cand and not ref:
        return 1.0
    return _counter_precision(ref, cand)


def _directional_scores(ref: dict, cand: dict) -> tuple[float, float, float, float]:
    """One-directional component scores: (ngram, wngram, syntax, dataflow)."""
    ngram = _bleu(ref["tokens"], cand["tokens"])
    wngram = _bleu(ref["tokens"], cand["tokens"], _kw_weight)
    if ref["parse_ok"] and cand["parse_ok"]:
        syntax = _counter_precision(ref["sigs"], cand["sigs"])
        dataflow = _dataflow_dir(ref["edges"], cand["edges"])
    else:
        syntax = dataflow = 0.0
    return ngram, wngram, syntax, dataflow


def _combine(components: tuple[float, ...], weights: tuple[float, ...]) -> dict:
    ngram, wngram, syntax, dataflow = components
    total = sum(weights) or 1.0
    w0, w1, w2, w3 = (w / total for w in weights)
    return {
        "codebleu": w0 * ngram + w1 * wngram + w2 * syntax + w3 * dataflow,
        "ngram_match_score": ngram,
        "weighted_ngram_match_score": wngram,
        "syntax_match_score": syntax,
        "dataflow_match_score": dataflow,
    }


def _ast_codebleu_symmetric(a: dict, b: dict, weights: tuple[float, ...]) -> dict:
    """Symmetric CodeBLEU: average of both directional scores."""
    ab = _directional_scores(a, b)
    ba = _directional_scores(b, a)
    avg = tuple((ab[i] + ba[i]) / 2.0 for i in range(4))
    return _combine(avg, weights)


def _ast_codebleu_directional(ref: dict, cand: dict, weights: tuple[float, ...]) -> dict:
    """Directional CodeBLEU: ref is the reference, cand is the hypothesis."""
    return _combine(_directional_scores(ref, cand), weights)


# --------------------------------------------------------------------------- #
# Optional canonical codebleu package engine
# --------------------------------------------------------------------------- #
def _pkg_codebleu_symmetric(code_a: str, code_b: str, weights: tuple[float, ...]) -> dict:
    from codebleu import calc_codebleu
    fwd = calc_codebleu([code_a], [code_b], lang="python", weights=weights)
    rev = calc_codebleu([code_b], [code_a], lang="python", weights=weights)
    return {k: (fwd.get(k, 0.0) + rev.get(k, 0.0)) / 2.0 for k in _CB_KEYS}


def _pkg_codebleu_directional(ref_src: str, cand_src: str,
                               weights: tuple[float, ...]) -> dict:
    from codebleu import calc_codebleu
    return {k: v for k, v in
            calc_codebleu([ref_src], [cand_src], lang="python", weights=weights).items()
            if k in _CB_KEYS}


def _resolve_engine(name: str) -> str:
    if name == "ast":
        return "ast"
    try:
        import codebleu  # noqa: F401
        return "codebleu"
    except ImportError:
        if name == "codebleu":
            sys.stderr.write(
                "error: --engine codebleu requested but 'codebleu' is not installed.\n"
                "       It needs tree-sitter (requires a C compiler on Python 3.14).\n"
                "       Use the default built-in engine instead: --engine ast\n"
            )
            raise SystemExit(2)
        return "ast"


# --------------------------------------------------------------------------- #
# Aggregation helpers
# --------------------------------------------------------------------------- #
def _agg(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "stdev": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    return {
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def _load_file(path: Path) -> dict:
    src = _read(path)
    rec: dict = {"path": path, "src": src, "tokens": _code_tokens(src)}
    try:
        rec["sigs"] = _ast_sigs(src)
        rec["edges"] = _dfg_edges(src)
        rec["parse_ok"] = True
    except SyntaxError:
        rec["sigs"] = Counter()
        rec["edges"] = Counter()
        rec["parse_ok"] = False
    return rec


# --------------------------------------------------------------------------- #
# Main comparison driver
# --------------------------------------------------------------------------- #
def compare(input_dir: Path, canonical_path: Optional[Path],
            rouge_orders: list[int], weights: tuple[float, ...],
            engine: str, verbose: bool = False,
            parity_test: Optional[Path] = None) -> dict:

    entries, agent_layout = _collect_files(input_dir)
    if len(entries) < 2:
        raise SystemExit(
            f"error: need at least 2 .py files in {input_dir}; found {len(entries)}."
        )

    data: dict[str, dict] = {}
    for label, _agent, path in entries:
        data[label] = _load_file(path)
        data[label]["agent"] = _agent

    unparsable = [label for label, rec in data.items() if not rec["parse_ok"]]
    if unparsable and verbose:
        sys.stderr.write(
            f"  warning: could not parse (syntax/dataflow=0): "
            f"{', '.join(unparsable)}\n"
        )

    # ---- pairwise variance ------------------------------------------------
    pairs: list[dict] = []
    labels = [e[0] for e in entries]
    for la, lb in itertools.combinations(labels, 2):
        if verbose:
            sys.stderr.write(f"  comparing {la} <-> {lb}\n")
        da, db = data[la], data[lb]
        if engine == "codebleu":
            cb = _pkg_codebleu_symmetric(da["src"], db["src"], weights)
        else:
            cb = _ast_codebleu_symmetric(da, db, weights)
        rouge = {str(n): rouge_n(da["tokens"], db["tokens"], n) for n in rouge_orders}
        pairs.append({
            "a": la, "b": lb, "agent_a": da["agent"], "agent_b": db["agent"],
            "codebleu": cb, "rouge": rouge,
        })

    def _pair_agg(subset: list[dict]) -> dict:
        agg: dict = {"codebleu": _agg([p["codebleu"]["codebleu"] for p in subset])}
        for n in rouge_orders:
            agg[f"rouge{n}_f1"] = _agg([p["rouge"][str(n)]["f1"] for p in subset])
        return agg

    aggregate: dict = {"overall": _pair_agg(pairs)}
    if agent_layout:
        agents = sorted({e[1] for e in entries})
        for ag in agents:
            intra = [p for p in pairs
                     if p["agent_a"] == ag and p["agent_b"] == ag]
            aggregate[f"agent_{ag}"] = _pair_agg(intra) if intra else {}
        cross = [p for p in pairs if p["agent_a"] != p["agent_b"]]
        aggregate["cross_agent"] = _pair_agg(cross) if cross else {}

    # ---- canonical comparison ---------------------------------------------
    canonical_scores: list[dict] = []
    canonical_label: Optional[str] = None
    if canonical_path is not None:
        canonical_label = canonical_path.name
        if verbose:
            sys.stderr.write(f"  loading canonical: {canonical_label}\n")
        canon = _load_file(canonical_path)
        for label in labels:
            cand = data[label]
            if verbose:
                sys.stderr.write(f"  canonical vs {label}\n")
            if engine == "codebleu":
                cb = _pkg_codebleu_directional(canon["src"], cand["src"], weights)
            else:
                cb = _ast_codebleu_directional(canon, cand, weights)
            rouge = {str(n): rouge_n(canon["tokens"], cand["tokens"], n)
                     for n in rouge_orders}
            canonical_scores.append({
                "candidate": label,
                "agent": cand["agent"],
                "codebleu": cb,
                "rouge": rouge,
            })
        canonical_scores.sort(key=lambda x: x["codebleu"]["codebleu"], reverse=True)

    # ---- parity correctness -----------------------------------------------
    parity_results: Optional[list[dict]] = None
    if parity_test is not None and parity_test.is_file():
        if verbose:
            sys.stderr.write(f"  running parity harness for {len(entries)} candidate(s)...\n")
        parity_results = _run_parity_for_candidates(entries, parity_test, verbose)

    return {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_dir": str(input_dir),
        "agent_layout": agent_layout,
        "candidates": labels,
        "codebleu_engine": engine,
        "rouge_orders": rouge_orders,
        "codebleu_weights": list(weights),
        "unparsable": unparsable,
        "canonical": canonical_label,
        "canonical_scores": canonical_scores,
        "pairs": pairs,
        "aggregate": aggregate,
        "parity": parity_results,
        "parity_test": parity_test.name if parity_test is not None else None,
    }


# --------------------------------------------------------------------------- #
# Markdown renderer
# --------------------------------------------------------------------------- #
def _fmt(x: float) -> str:
    return f"{x:.4f}"


def _agg_row(label: str, agg: dict) -> str:
    a = agg
    return (f"| {label} | {_fmt(a['mean'])} | {_fmt(a['stdev'])} | "
            f"{_fmt(a['min'])} | {_fmt(a['max'])} | {a['n']} |")


def _render_markdown(result: dict) -> str:
    rouge_orders = result["rouge_orders"]
    engine_label = ("built-in ast (tree-sitter-free)"
                    if result["codebleu_engine"] == "ast" else "codebleu package")
    n_cands = len(result["candidates"])
    n_pairs = len(result["pairs"])

    lines: list[str] = [
        "# AI conversion variance report",
        "",
        f"_Generated by compare_variants.py on {result['generated']}._",
        "",
        f"- Input folder: `{result['input_dir']}`",
        f"- Layout: **{'per-agent subfolders' if result['agent_layout'] else 'flat'}**",
        f"- Candidates: {n_cands} file(s) across "
        + (f"{len({c.split('/')[0] for c in result['candidates']})} agent(s)"
           if result["agent_layout"] else "1 group")
        + f"; {n_pairs} pair(s) compared.",
        f"- CodeBLEU engine: **{engine_label}**, "
        f"weights = {result['codebleu_weights']} (n-gram, w-ngram, syntax, dataflow).",
    ]
    if result["canonical"]:
        lines.append(
            f"- Canonical reference: `{result['canonical']}` "
            f"(directional — canonical is the reference, candidates are hypotheses)."
        )
    if result["unparsable"]:
        lines.append(
            "- ⚠ Unparsable (syntax/dataflow scored 0): "
            + ", ".join(f"`{c}`" for c in result["unparsable"])
        )

    # ---- metric glossary --------------------------------------------------
    lines += [
        "",
        "## Metric glossary",
        "",
        "All scores are in **[0, 1]** — higher means more similar.",
        "",
        "### CodeBLEU",
        "A composite code-similarity score designed for machine-generated code, "
        "computed as a weighted average of four components:",
        "",
        "| Component | What it measures |",
        "|---|---|",
        "| **n-gram match** | Overlap of raw code token sequences (like BLEU in NLP). "
        "Rewards sharing the same identifiers, operators, and literals. |",
        "| **weighted n-gram match** | Same, but language keywords (`def`, `return`, "
        "`for`, `class`, etc.) are up-weighted 4×, so structural divergence is penalised "
        "more heavily than naming differences. |",
        "| **syntax match** | Overlap of AST node-type patterns — whether two files use "
        "the same control structures (loops vs recursion, comprehensions vs explicit loops, "
        "class hierarchy shape). Equivalent code written differently scores lower here. |",
        "| **dataflow match** | Overlap of variable definition-use edges — whether the "
        "same computations are wired together the same way "
        "(e.g. `self.mResult = f(self.mInput)` appears in both). |",
        "",
        "In the **pairwise variance** section CodeBLEU is symmetric (both directions "
        "averaged, since neither output is a reference). In the **canonical comparison** "
        "section it is directional (canonical = reference, candidate = hypothesis).",
        "",
        "### ROUGE-N F1",
        "ROUGE-N measures n-gram overlap on a code-aware token stream (comments and "
        "layout tokens stripped). F1 is the harmonic mean of precision and recall.",
        "",
        "- **ROUGE-1** counts individual tokens — high score means files share the "
        "same vocabulary of identifiers, keywords, and literals.",
        "- **ROUGE-2** counts consecutive token pairs — high score means local token "
        "sequences also match, a stronger signal of structural similarity.",
        "",
        "ROUGE-N is surface-level and cannot distinguish equivalent code written "
        "differently. Read it alongside CodeBLEU's syntax and dataflow components.",
        "",
        "### Reading variance statistics",
        "Statistics are computed across all pairwise scores within the stated group.",
        "",
        "| Column | Meaning |",
        "|---|---|",
        "| **Mean** | Average similarity. Higher = outputs agree more on average. |",
        "| **Stdev** | Spread of similarity — the primary variance signal. "
        "Low stdev = consistently similar; high stdev = some pairs much closer "
        "than others (bimodal or clustered). |",
        "| **Min / Max** | Least and most similar pair — find them in the pairwise "
        "table to identify which responses diverged or converged most. |",
        "| **N** | Number of pairs contributing to the statistics. |",
    ]

    # ---- canonical comparison section ------------------------------------
    if result["canonical_scores"]:
        lines += [
            "",
            "## Canonical comparison",
            "",
            f"Each candidate scored against `{result['canonical']}` "
            f"(the accepted reference port). Higher = closer to canonical. "
            f"Ranked by CodeBLEU.",
            "",
        ]
        rouge_h = "".join(f" ROUGE-{n} F1 |" for n in rouge_orders)
        rouge_a = "".join(" ---: |" for _ in rouge_orders)
        header = "Agent | " if result["agent_layout"] else ""
        align = "---|" if result["agent_layout"] else ""
        lines += [
            f"| {header}Candidate | CodeBLEU | n-gram | w-ngram | syntax | dataflow |{rouge_h}",
            f"| {align}---|---:|---:|---:|---:|---:|{rouge_a}",
        ]
        for cs in result["canonical_scores"]:
            cb = cs["codebleu"]
            rouge_cells = "".join(
                f" {_fmt(cs['rouge'][str(n)]['f1'])} |" for n in rouge_orders
            )
            agent_cell = f"`{cs['agent']}` | " if result["agent_layout"] else ""
            lines.append(
                f"| {agent_cell}`{cs['candidate']}` | "
                f"{_fmt(cb['codebleu'])} | "
                f"{_fmt(cb.get('ngram_match_score', 0.0))} | "
                f"{_fmt(cb.get('weighted_ngram_match_score', 0.0))} | "
                f"{_fmt(cb.get('syntax_match_score', 0.0))} | "
                f"{_fmt(cb.get('dataflow_match_score', 0.0))} |"
                f"{rouge_cells}"
            )

    # ---- aggregate sections -----------------------------------------------
    agg = result["aggregate"]
    if result["agent_layout"]:
        agents = sorted({c.split("/")[0] for c in result["candidates"]})

        lines += ["", "### Within-agent variance (intra-agent pairs)", ""]
        for ag in agents:
            key = f"agent_{ag}"
            intra = agg.get(key, {})
            if not intra:
                lines.append(f"**{ag}**: only one file — no intra-agent pairs.")
                continue
            lines += [
                f"**{ag}** ({intra['codebleu']['n']} pair(s)):",
                "",
                "| Metric | Mean | Stdev | Min | Max | N |",
                "|---|---:|---:|---:|---:|---:|",
                _agg_row("CodeBLEU", intra["codebleu"]),
            ]
            for n in rouge_orders:
                lines.append(_agg_row(f"ROUGE-{n} F1", intra[f"rouge{n}_f1"]))
            lines.append("")


    # ---- pairwise table (combined scores + components) --------------------
    rouge_h = "".join(f" ROUGE-{n} F1 |" for n in rouge_orders)
    rouge_a = "".join(" ---: |" for _ in rouge_orders)
    lines += [
        "",
        "## Pairwise scores",
        "",
        f"| File A | File B | CodeBLEU | n-gram | w-ngram | syntax | dataflow |{rouge_h}",
        f"|---|---|---:|---:|---:|---:|---:|{rouge_a}",
    ]
    for p in result["pairs"]:
        cb = p["codebleu"]
        rouge_cells = "".join(f" {_fmt(p['rouge'][str(n)]['f1'])} |" for n in rouge_orders)
        lines.append(
            f"| `{p['a']}` | `{p['b']}` | "
            f"{_fmt(cb['codebleu'])} | "
            f"{_fmt(cb.get('ngram_match_score', 0.0))} | "
            f"{_fmt(cb.get('weighted_ngram_match_score', 0.0))} | "
            f"{_fmt(cb.get('syntax_match_score', 0.0))} | "
            f"{_fmt(cb.get('dataflow_match_score', 0.0))} |"
            f"{rouge_cells}"
        )

    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Measure CodeBLEU + ROUGE-N variance between AI-generated "
                    "code conversions.",
    )
    ap.add_argument(
        "input", nargs="?", type=Path, default=_DEFAULT_INPUT,
        help="folder of candidate .py files or per-agent subfolders "
             "(default: agent_evaluations/candidates/).",
    )
    ap.add_argument(
        "--canonical", type=Path, default=None, metavar="PATH",
        help="path to the accepted reference port file; each candidate is scored "
             "against it directionally (canonical = reference).",
    )
    ap.add_argument(
        "--out", type=Path, default=_DEFAULT_OUT,
        help="output directory (default: agent_evaluations/reports/).",
    )
    ap.add_argument(
        "--engine", choices=["auto", "ast", "codebleu"], default="auto",
        help="CodeBLEU engine (default: auto = codebleu package if installed, else ast).",
    )
    ap.add_argument(
        "--rouge-n", type=int, nargs="+", default=[1, 2], metavar="N",
        help="ROUGE n-gram orders (default: 1 2).",
    )
    ap.add_argument(
        "--weights", type=float, nargs=4, default=[0.25, 0.25, 0.25, 0.25],
        metavar=("NGRAM", "WNGRAM", "SYNTAX", "DATAFLOW"),
        help="CodeBLEU component weights (default: 0.25 each).",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    input_dir: Path = args.input
    if not input_dir.is_dir():
        sys.stderr.write(f"error: input folder not found: {input_dir}\n")
        return 2

    # Auto-detect canonical: in agent layout, any .py file directly in the
    # input dir root (not in a subfolder) is treated as the reference port.
    # Override with --canonical if needed.
    canonical_path: Optional[Path] = args.canonical
    if canonical_path is not None and not canonical_path.is_file():
        sys.stderr.write(f"error: canonical file not found: {canonical_path}\n")
        return 2
    if canonical_path is None:
        root_py = sorted(p for p in input_dir.glob("*.py") if p.is_file())
        if root_py:
            canonical_path = root_py[0]
            if len(root_py) > 1:
                sys.stderr.write(
                    f"warning: multiple .py files at candidates root; "
                    f"using {canonical_path.name} as canonical. "
                    f"Move extras into an agent subfolder or use --canonical.\n"
                )
            print(f"canonical (auto-detected): {canonical_path.name}")

    engine = _resolve_engine(args.engine)
    result = compare(
        input_dir,
        canonical_path=canonical_path,
        rouge_orders=sorted(set(args.rouge_n)),
        weights=tuple(args.weights),
        engine=engine,
        verbose=args.verbose,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"variance_{input_dir.name}"
    json_path = args.out / f"{stem}.json"
    md_path = args.out / f"{stem}.md"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_render_markdown(result), encoding="utf-8")

    agg = result["aggregate"]["overall"]["codebleu"]
    layout = "agent" if result["agent_layout"] else "flat"
    print(
        f"engine={engine}, layout={layout}; "
        f"{len(result['candidates'])} candidate(s), {len(result['pairs'])} pair(s):"
    )
    if result["unparsable"]:
        print(f"  warning: unparsable: {', '.join(result['unparsable'])}")
    print(
        f"  CodeBLEU  mean={_fmt(agg['mean'])}  stdev={_fmt(agg['stdev'])}  "
        f"min={_fmt(agg['min'])}  max={_fmt(agg['max'])}"
    )
    for n in result["rouge_orders"]:
        a = result["aggregate"]["overall"][f"rouge{n}_f1"]
        print(
            f"  ROUGE-{n}   mean={_fmt(a['mean'])}  stdev={_fmt(a['stdev'])}  "
            f"min={_fmt(a['min'])}  max={_fmt(a['max'])}"
        )
    if result["canonical_scores"]:
        print(f"  Canonical ({result['canonical']}) top match: "
              f"{result['canonical_scores'][0]['candidate']} "
              f"CodeBLEU={_fmt(result['canonical_scores'][0]['codebleu']['codebleu'])}")
    print(f"reports written:\n  {json_path}\n  {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
