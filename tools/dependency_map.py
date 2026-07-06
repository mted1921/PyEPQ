"""
dependency_map.py -- Java -> Python dependency analyzer for EPQ.

Combines and supersedes analyze_dependencies.py and find_components.py.
Implements industry-standard dependency metrics so the conversion project
can prioritize porting work objectively.

What it produces
----------------
For each analyzed package:
  * nodes_df        one row per class with all metrics
  * edges_df        one row per dependency (import / extends / implements)
  * wccs_df         weakly connected components (isolated subgraphs)
  * sccs_df         strongly connected components (cycle clusters)
  * cycles_df       enumerated elementary cycles
  * port_order_df   recommended Java->Python conversion sequence

Metrics computed per class
--------------------------
  in_degree            classes inside the package that depend on this one
  out_degree           classes inside the package that this one depends on
  internal_degree      in + out
  instability          Robert Martin's I = out / (in + out). 0 = stable,
                       1 = unstable. Stable classes should rarely change.
  depth                topological depth from leaves. 0 = no internal
                       dependencies (foundation). Computed on SCC condensation
                       so cyclic graphs work; all cycle members share depth.
  reachable            transitive fan-out size (descendants count). "Blast
                       radius" if this class breaks.
  pagerank             centrality via random walk; high = many things
                       (transitively) depend on this class.
  betweenness          fraction of shortest paths going through this node;
                       high = chokepoint / refactoring hot-spot.
  in_cycle             True if part of an SCC of size > 1 (or self-loop).
  is_leaf              True if out_degree == 0 (no internal deps -> port-first
                       candidate).
  is_dead              True if in_degree == out_degree == 0 (no internal
                       coupling at all; might be an entry point, test
                       fixture, or genuinely orphaned).

Conversion-order recommendation
-------------------------------
Topological order on the SCC condensation, leaves first. Within an SCC
(cyclic cluster) members are sorted by in_degree descending. Use this
list as the default "port these next" sequence; it minimizes the chance
of a Python port referencing a class that hasn't been ported yet.

Output formats
--------------
The default (``all``) is tuned for human reading: it writes only the
Markdown reports and the Graphviz graphs. The tabular dumps (CSV, xlsx)
carry the same data as flat tables and are opt-in for when you need the
raw numbers.

  --format md      Markdown only — per-package dependency TIERS, plus a
                   cross-package INTERCONNECTION report for each linked pair.
  --format dot     Graphviz only — per-package graph (tier-ranked) and a
                   bipartite cross-package graph. Render: dot -Tsvg x.dot > x.svg
  --format all     default: the Markdown reports AND the DOT graphs.
  --format csv     raw data dump — one CSV per dataframe per package.
  --format xlsx    raw data dump — one .xlsx workbook per package (all sheets).

Usage
-----
  python dependency_map.py                          # default: EPQLibrary Utility, all formats
  python dependency_map.py EPQLibrary               # one package
  python dependency_map.py EPQLibrary Utility NISTMonte
  python dependency_map.py --format xlsx --out reports/
  python dependency_map.py --no-cycles              # skip elementary-cycle enumeration
                                                    # (large packages; can be slow)

Importable API
--------------
  from dependency_map import analyze_package, write_xlsx
  analysis = analyze_package(target_dir, "gov.nist.microanalysis.EPQLibrary")
  write_xlsx(analysis, Path("EPQLibrary.xlsx"))
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import networkx as nx
import pandas as pd

log = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

NIST_PREFIX: str = "gov.nist.microanalysis"

NIST_PACKAGES: frozenset[str] = frozenset({
    "EPQDatabase", "EPQImage", "EPQLibrary", "EPQTools",
    "JythonGUI", "NISTMonte", "Utility", "EPQTests",
})


# ============================================================================
# Regex patterns
# ============================================================================
# Strip these from source before structural parsing so regexes don't match
# inside comments or string literals.
_RE_LINE_COMMENT: re.Pattern[str] = re.compile(r"//[^\n]*")
_RE_BLOCK_COMMENT: re.Pattern[str] = re.compile(r"/\*.*?\*/", re.DOTALL)
_RE_STRING_LIT: re.Pattern[str] = re.compile(r'"(?:\\.|[^"\\])*"', re.DOTALL)

_RE_PACKAGE: re.Pattern[str] = re.compile(
    r"^\s*package\s+([\w.]+)\s*;", re.MULTILINE,
)
_RE_IMPORT: re.Pattern[str] = re.compile(
    r"^\s*import\s+(?:static\s+)?([\w.*]+)\s*;", re.MULTILINE,
)
_RE_CLASS_HDR: re.Pattern[str] = re.compile(
    r"""
    (?:^|\s)
    (?P<modifiers>(?:(?:public|protected|private|abstract|final|static)\s+)*)
    (?P<kind>class|interface|enum|@interface)
    \s+(?P<name>\w+)
    (?:\s*<[^{]*?>)?
    (?:\s+extends\s+(?P<extends>[\w,.\s<>]+?))?
    (?:\s+implements\s+(?P<implements>[\w,.\s<>]+?))?
    \s*\{
    """,
    re.MULTILINE | re.VERBOSE,
)


# ============================================================================
# Dataclasses
# ============================================================================

@dataclass(frozen=True)
class JavaClassInfo:
    """One row per top-level type declaration in a .java file."""
    file: str
    package: str
    class_name: str
    kind: str               # "class" | "interface" | "enum" | "@interface"
    modifiers: str          # "public", "abstract final", etc.
    extends: tuple[str, ...]
    implements: tuple[str, ...]
    imports: tuple[str, ...]
    body: str               # comment/string-stripped source; used for implicit-dep scan


@dataclass(frozen=True)
class DependencyMetrics:
    """Per-class metrics computed from the intra-package dependency graph."""
    class_name: str
    file: str
    in_degree: int
    out_degree: int
    instability: float       # Martin's I in [0, 1]
    depth: int               # topological depth from leaves
    reachable: int           # |descendants|
    pagerank: float
    betweenness: float
    in_cycle: bool
    is_leaf: bool
    is_dead: bool


@dataclass
class PackageAnalysis:
    """Complete analysis of one package."""
    package_fqn: str
    target_dir: Path
    classes: list[JavaClassInfo]
    graph: nx.DiGraph
    metrics: dict[str, DependencyMetrics]
    wccs: list[set[str]]               # weakly connected components
    sccs: list[set[str]]               # strongly connected components
    cycles: list[list[str]]            # enumerated simple cycles
    conversion_order: list[str]        # recommended port sequence


@dataclass(frozen=True)
class CrossPackageEdge:
    """One deduplicated import edge from a class in one package to a top-level
    class in another NIST package.

    *dep_fqn* is the original import FQN and may reference an inner class
    (e.g. ``...LevenbergMarquardt2.FitFunction``); *target_class* is always
    the enclosing top-level class (``LevenbergMarquardt2``).
    """
    source_class: str
    source_file: str
    target_class: str   # simple top-level class name in the target package
    dep_fqn: str        # original import FQN


# ============================================================================
# Parsing
# ============================================================================

def _strip_noise(src: str) -> str:
    """Remove comments and string literals so regexes don't match inside them."""
    src = _RE_BLOCK_COMMENT.sub("", src)
    src = _RE_LINE_COMMENT.sub("", src)
    src = _RE_STRING_LIT.sub('""', src)
    return src


def _split_type_list(raw: str) -> tuple[str, ...]:
    """Split `extends`/`implements` clauses into individual type names,
    stripping generic parameters."""
    if not raw:
        return ()
    clean: str = re.sub(r"<[^<>]*>", "", raw)
    return tuple(t.strip() for t in clean.split(",") if t.strip())


def parse_java_file(path: Path) -> JavaClassInfo:
    """Parse a single .java file. Returns the FIRST top-level type found.

    Files with multiple top-level types (one public, others
    package-private) are not unusual in some codebases, but rare in
    EPQ. The current parser tracks only the first to keep node
    accounting simple and matches prior tool behaviour.
    """
    raw: str = path.read_text(encoding="utf-8", errors="replace")
    src: str = _strip_noise(raw)

    pkg_match = _RE_PACKAGE.search(src)
    package: str = pkg_match.group(1) if pkg_match else ""

    imports: tuple[str, ...] = tuple(
        m.group(1) for m in _RE_IMPORT.finditer(src)
    )

    class_name: str = ""
    kind: str = ""
    modifiers: str = ""
    extends: tuple[str, ...] = ()
    implements: tuple[str, ...] = ()

    class_match = _RE_CLASS_HDR.search(src)
    if class_match:
        class_name = class_match.group("name")
        kind = class_match.group("kind")
        modifiers = class_match.group("modifiers").strip()
        extends = _split_type_list(class_match.group("extends") or "")
        implements = _split_type_list(class_match.group("implements") or "")

    return JavaClassInfo(
        file=path.name,
        package=package,
        class_name=class_name,
        kind=kind,
        modifiers=modifiers,
        extends=extends,
        implements=implements,
        imports=imports,
        body=src,
    )


def classify_import(fqn: str, package_fqn: str) -> str:
    """Categorize an import FQN. Categories drive count columns and
    edge classification."""
    if fqn.startswith("java."):
        return "java_stdlib"
    if fqn.startswith("javax."):
        return "javax_stdlib"
    if fqn.startswith(package_fqn):
        return "internal"
    if fqn.startswith(NIST_PREFIX):
        first: str = fqn[len(NIST_PREFIX) + 1:].split(".")[0]
        if first in NIST_PACKAGES:
            return f"nist_{first.lower()}"
        return "nist_other"
    return "external"


# ============================================================================
# Graph construction
# ============================================================================

def build_dependency_graph(
    classes: list[JavaClassInfo],
    package_fqn: str,
) -> nx.DiGraph:
    """Build the intra-package directed dependency graph.

    Node: simple class name (e.g. "Element").
    Edge u -> v: u depends on v (imports / extends / implements).
    Self-edges and edges to types outside the package are excluded.
    """
    G: nx.DiGraph = nx.DiGraph()
    internal_names: set[str] = {c.class_name for c in classes if c.class_name}

    for c in classes:
        if not c.class_name:
            continue
        G.add_node(c.class_name, file=c.file, kind=c.kind,
                   modifiers=c.modifiers)

    for c in classes:
        if not c.class_name:
            continue

        for fqn in c.imports:
            if classify_import(fqn, package_fqn) != "internal":
                continue
            simple: str = fqn.rsplit(".", 1)[-1]
            if simple == "*" or simple == c.class_name:
                continue
            if simple in internal_names:
                G.add_edge(c.class_name, simple, kind="import")
            else:
                # Inner-class import: e.g. gov.nist...LM2.FitFunction where
                # FitFunction is not a top-level class.  The actual dependency
                # is on the enclosing top-level class; find it as the first
                # segment after stripping the package prefix.
                suffix: str = fqn[len(package_fqn) + 1:]   # e.g. "LM2.FitFunction"
                outer: str = suffix.split(".")[0]
                if outer and outer in internal_names and outer != c.class_name:
                    G.add_edge(c.class_name, outer, kind="import")

        for parent in c.extends:
            if parent in internal_names and parent != c.class_name:
                G.add_edge(c.class_name, parent, kind="extends")
        for iface in c.implements:
            if iface in internal_names and iface != c.class_name:
                G.add_edge(c.class_name, iface, kind="implements")

    # Implicit same-package dependencies: Java classes in the same package
    # are visible without any import statement, so a class can reference a
    # peer by bare name and the dependency won't appear in the import list.
    # Scan each class's stripped body for whole-word occurrences of every
    # other class name in the package and add edges that aren't already
    # covered by an explicit import/extends/implements edge.
    if internal_names:
        _implicit_pat = re.compile(
            r'\b(' + '|'.join(re.escape(n) for n in sorted(internal_names)) + r')\b'
        )
        for c in classes:
            if not c.class_name:
                continue
            for _m in _implicit_pat.finditer(c.body):
                name = _m.group(1)
                if name == c.class_name or G.has_edge(c.class_name, name):
                    continue
                G.add_edge(c.class_name, name, kind="implicit")

    return G


# ============================================================================
# Metrics
# ============================================================================

def _compute_depths(G: nx.DiGraph) -> dict[str, int]:
    """Depth from leaves: 0 = no internal out-edges (a foundation).

    Computed on the SCC condensation (a DAG) so cyclic graphs work.
    All members of an SCC share the SCC's depth.

    For port-order purposes, depth-0 nodes are pure leaves (depend on
    nothing internal -> port first), depth-N nodes sit atop dependency
    chains of length N.
    """
    if G.number_of_nodes() == 0:
        return {}
    condensation: nx.DiGraph = nx.condensation(G)
    levels: dict[int, int] = {}
    # Reverse topological order = sinks first = leaves first.
    for scc in reversed(list(nx.topological_sort(condensation))):
        succs: list[int] = list(condensation.successors(scc))
        levels[scc] = max((levels[s] for s in succs), default=-1) + 1
    mapping: dict[str, int] = condensation.graph["mapping"]
    return {orig: levels[scc_idx] for orig, scc_idx in mapping.items()}


def compute_metrics(
    G: nx.DiGraph,
    classes: list[JavaClassInfo],
) -> dict[str, DependencyMetrics]:
    """Compute the per-node metrics block. Empty graph -> empty dict."""
    if G.number_of_nodes() == 0:
        return {}

    pr: dict[str, float] = nx.pagerank(G)
    bc: dict[str, float] = nx.betweenness_centrality(G)
    depths: dict[str, int] = _compute_depths(G)

    sccs: list[set[str]] = list(nx.strongly_connected_components(G))
    cycle_members: set[str] = set()
    for scc in sccs:
        if len(scc) > 1 or any(G.has_edge(n, n) for n in scc):
            cycle_members.update(scc)

    file_of: dict[str, str] = {
        c.class_name: c.file for c in classes if c.class_name
    }

    metrics: dict[str, DependencyMetrics] = {}
    for node in G.nodes:
        in_d: int = G.in_degree(node)
        out_d: int = G.out_degree(node)
        total: int = in_d + out_d
        instability: float = (out_d / total) if total > 0 else 0.0
        reachable: int = len(nx.descendants(G, node))
        metrics[node] = DependencyMetrics(
            class_name=node,
            file=file_of.get(node, ""),
            in_degree=in_d,
            out_degree=out_d,
            instability=instability,
            depth=depths.get(node, 0),
            reachable=reachable,
            pagerank=pr.get(node, 0.0),
            betweenness=bc.get(node, 0.0),
            in_cycle=node in cycle_members,
            is_leaf=(out_d == 0),
            is_dead=(in_d == 0 and out_d == 0),
        )
    return metrics


# ============================================================================
# Component / cycle analysis
# ============================================================================

def find_components(
    G: nx.DiGraph,
) -> tuple[list[set[str]], list[set[str]]]:
    """Return (weakly_connected, strongly_connected) components, largest first."""
    wccs: list[set[str]] = sorted(
        nx.weakly_connected_components(G), key=len, reverse=True,
    )
    sccs: list[set[str]] = sorted(
        nx.strongly_connected_components(G), key=len, reverse=True,
    )
    return wccs, sccs


def find_simple_cycles(
    G: nx.DiGraph,
    max_cycles: int = 1000,
) -> list[list[str]]:
    """Enumerate up to max_cycles elementary cycles.

    Johnson's algorithm; can be expensive on dense cyclic graphs.
    Pass max_cycles=0 to disable enumeration entirely.
    """
    if max_cycles <= 0:
        return []
    cycles: list[list[str]] = []
    for cycle in nx.simple_cycles(G):
        cycles.append(list(cycle))
        if len(cycles) >= max_cycles:
            break
    return cycles


# ============================================================================
# Conversion-order recommendation
# ============================================================================

def recommend_port_order(
    G: nx.DiGraph,
    metrics: dict[str, DependencyMetrics],
) -> list[str]:
    """Suggest a Java -> Python port order.

    Strategy:
      1. Iterate the SCC condensation in REVERSE topological order
         (leaves / foundations first).
      2. Within each SCC, sort members by in_degree DESC (port the
         most-depended-upon first so callers find them ready).

    Members of a non-trivial SCC must be ported as a group (the cycle
    means none can stand alone). The returned list preserves cycle
    members contiguously.
    """
    if G.number_of_nodes() == 0:
        return []
    condensation: nx.DiGraph = nx.condensation(G)
    order: list[str] = []
    for scc_idx in reversed(list(nx.topological_sort(condensation))):
        members: list[str] = sorted(
            condensation.nodes[scc_idx]["members"],
            key=lambda n: (-metrics[n].in_degree, n),
        )
        order.extend(members)
    return order


# ============================================================================
# Top-level
# ============================================================================

def analyze_package(
    target_dir: Path,
    package_fqn: str,
    *,
    cycle_limit: int = 1000,
) -> PackageAnalysis:
    """Analyze every .java file under target_dir and return a PackageAnalysis.

    Raises FileNotFoundError if no .java files are found.
    """
    java_files: list[Path] = sorted(target_dir.rglob("*.java"))
    if not java_files:
        raise FileNotFoundError(f"No .java files under {target_dir}")

    classes: list[JavaClassInfo] = [parse_java_file(f) for f in java_files]
    G: nx.DiGraph = build_dependency_graph(classes, package_fqn)
    metrics: dict[str, DependencyMetrics] = compute_metrics(G, classes)
    wccs, sccs = find_components(G)
    cycles: list[list[str]] = find_simple_cycles(G, cycle_limit)
    conv_order: list[str] = recommend_port_order(G, metrics)

    return PackageAnalysis(
        package_fqn=package_fqn,
        target_dir=target_dir,
        classes=classes,
        graph=G,
        metrics=metrics,
        wccs=wccs,
        sccs=sccs,
        cycles=cycles,
        conversion_order=conv_order,
    )


def build_cross_package_edges(
    src: PackageAnalysis,
    tgt_pkg_fqn: str,
) -> list[CrossPackageEdge]:
    """Return one CrossPackageEdge per (source_class, target_class) pair
    where a class in *src* imports from *tgt_pkg_fqn*.

    Edges are deduplicated: if a source class imports both
    ``Utility.Math2`` and ``Utility.Math2.SomeInner``, only one edge is
    emitted for ``Math2``.  Wildcard imports (``Utility.*``) are skipped
    because the referenced class cannot be determined statically.
    """
    tgt_prefix: str = tgt_pkg_fqn + "."
    edges: list[CrossPackageEdge] = []
    for c in src.classes:
        if not c.class_name:
            continue
        seen: set[str] = set()
        for fqn in c.imports:
            if not fqn.startswith(tgt_prefix):
                continue
            remainder: str = fqn[len(tgt_prefix):]     # "Math2" or "LM2.FitFunction"
            tgt_class: str = remainder.split(".")[0]    # "Math2" or "LM2"
            if not tgt_class or tgt_class == "*" or tgt_class in seen:
                continue
            seen.add(tgt_class)
            edges.append(CrossPackageEdge(
                source_class=c.class_name,
                source_file=c.file,
                target_class=tgt_class,
                dep_fqn=fqn,
            ))
    return edges


# ============================================================================
# DataFrame builders
# ============================================================================

def nodes_dataframe(a: PackageAnalysis) -> pd.DataFrame:
    """One row per class with all metrics and import counts."""
    rows: list[dict[str, object]] = []
    for c in a.classes:
        if not c.class_name:
            continue
        m: DependencyMetrics | None = a.metrics.get(c.class_name)
        if m is None:
            continue
        counts: dict[str, int] = {}
        for fqn in c.imports:
            cat: str = classify_import(fqn, a.package_fqn)
            counts[cat] = counts.get(cat, 0) + 1

        row: dict[str, object] = {
            "class_name": c.class_name,
            "file": c.file,
            "package": c.package,
            "kind": c.kind,
            "modifiers": c.modifiers,
            "extends": ", ".join(c.extends),
            "implements": ", ".join(c.implements),
            "in_degree": m.in_degree,
            "out_degree": m.out_degree,
            "internal_degree": m.in_degree + m.out_degree,
            "instability": round(m.instability, 4),
            "depth": m.depth,
            "reachable": m.reachable,
            "pagerank": round(m.pagerank, 6),
            "betweenness": round(m.betweenness, 6),
            "in_cycle": m.in_cycle,
            "is_leaf": m.is_leaf,
            "is_dead": m.is_dead,
            "total_imports": len(c.imports),
        }
        # Stable count columns (always present, even if zero).
        for cat in ("internal", "java_stdlib", "javax_stdlib", "external",
                    "nist_other"):
            row[f"n_{cat}"] = counts.get(cat, 0)
        for pkg in sorted(NIST_PACKAGES):
            row[f"n_nist_{pkg.lower()}"] = counts.get(f"nist_{pkg.lower()}", 0)
        rows.append(row)
    return pd.DataFrame(rows)


def edges_dataframe(a: PackageAnalysis) -> pd.DataFrame:
    """One row per dependency (import / extends / implements / implicit)."""
    rows: list[dict[str, object]] = []
    internal_names: set[str] = set(a.metrics.keys())
    file_of: dict[str, str] = {
        c.class_name: c.file for c in a.classes if c.class_name
    }
    for c in a.classes:
        if not c.class_name:
            continue
        for fqn in c.imports:
            simple: str = fqn.rsplit(".", 1)[-1]
            cat: str = classify_import(fqn, a.package_fqn)
            # An import is an internal target if the simple name is a known
            # top-level class, OR if it's an inner-class import whose outer
            # class is a known top-level class (e.g. LM2.FitFunction -> LM2).
            _is_internal: bool = False
            if cat == "internal" and simple != "*":
                if simple in internal_names:
                    _is_internal = True
                else:
                    _outer = fqn[len(a.package_fqn) + 1:].split(".")[0]
                    _is_internal = bool(_outer and _outer in internal_names)
            rows.append({
                "source_class": c.class_name,
                "source_file": c.file,
                "dep_fqn": fqn,
                "dep_simple": simple,
                "dep_category": cat,
                "rel_type": "import",
                "is_internal_target": _is_internal,
            })
        for parent in c.extends:
            rows.append({
                "source_class": c.class_name,
                "source_file": c.file,
                "dep_fqn": parent,
                "dep_simple": parent,
                "dep_category": "internal" if parent in internal_names else "unknown",
                "rel_type": "extends",
                "is_internal_target": parent in internal_names,
            })
        for iface in c.implements:
            rows.append({
                "source_class": c.class_name,
                "source_file": c.file,
                "dep_fqn": iface,
                "dep_simple": iface,
                "dep_category": "internal" if iface in internal_names else "unknown",
                "rel_type": "implements",
                "is_internal_target": iface in internal_names,
            })
    # Implicit same-package edges detected by body scan (no import statement).
    for u, v, data in a.graph.edges(data=True):
        if data.get("kind") == "implicit":
            rows.append({
                "source_class": u,
                "source_file": file_of.get(u, ""),
                "dep_fqn": f"{a.package_fqn}.{v}",
                "dep_simple": v,
                "dep_category": "internal",
                "rel_type": "implicit",
                "is_internal_target": True,
            })
    return pd.DataFrame(rows)


def components_dataframe(
    a: PackageAnalysis,
    *,
    strong: bool = False,
) -> pd.DataFrame:
    """Components ranked by size. strong=True for SCCs, else WCCs."""
    components: list[set[str]] = a.sccs if strong else a.wccs
    rows: list[dict[str, object]] = []
    for i, comp in enumerate(components, 1):
        for cls in sorted(comp):
            rows.append({"component": i, "size": len(comp), "class_name": cls})
    return pd.DataFrame(rows)


def cycles_dataframe(a: PackageAnalysis) -> pd.DataFrame:
    """One row per (cycle, position) pair so cycles preserve their order."""
    rows: list[dict[str, object]] = []
    for i, cyc in enumerate(a.cycles, 1):
        for pos, cls in enumerate(cyc):
            rows.append({
                "cycle": i, "length": len(cyc),
                "position": pos, "class_name": cls,
            })
    return pd.DataFrame(rows)


def port_order_dataframe(a: PackageAnalysis) -> pd.DataFrame:
    """Recommended port sequence with the metrics that justified the rank."""
    rows: list[dict[str, object]] = []
    for i, cls in enumerate(a.conversion_order, 1):
        m: DependencyMetrics = a.metrics[cls]
        rows.append({
            "port_order": i,
            "class_name": cls,
            "depth": m.depth,
            "in_degree": m.in_degree,
            "out_degree": m.out_degree,
            "instability": round(m.instability, 4),
            "in_cycle": m.in_cycle,
        })
    return pd.DataFrame(rows)


def cross_package_dataframe(edges: list[CrossPackageEdge]) -> pd.DataFrame:
    """One row per (source_class, target_class) cross-package import pair."""
    if not edges:
        return pd.DataFrame(
            columns=["source_class", "source_file", "target_class", "dep_fqn"],
        )
    return pd.DataFrame([
        {
            "source_class": e.source_class,
            "source_file": e.source_file,
            "target_class": e.target_class,
            "dep_fqn": e.dep_fqn,
        }
        for e in edges
    ])


# ============================================================================
# Reporters
# ============================================================================

def write_csvs(a: PackageAnalysis, out_dir: Path, prefix: str) -> list[Path]:
    """Write all dataframes as CSVs. Returns the list of written paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, df in (
        ("nodes",      nodes_dataframe(a)),
        ("edges",      edges_dataframe(a)),
        ("wccs",       components_dataframe(a)),
        ("sccs",       components_dataframe(a, strong=True)),
        ("cycles",     cycles_dataframe(a)),
        ("port_order", port_order_dataframe(a)),
    ):
        p: Path = out_dir / f"{prefix}_{name}.csv"
        df.to_csv(p, index=False)
        written.append(p)
    return written


def write_xlsx(a: PackageAnalysis, path: Path) -> Path:
    """Write all dataframes to a single workbook (one sheet each)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        nodes_dataframe(a).to_excel(w, sheet_name="nodes", index=False)
        edges_dataframe(a).to_excel(w, sheet_name="edges", index=False)
        components_dataframe(a).to_excel(w, sheet_name="wccs", index=False)
        components_dataframe(a, strong=True).to_excel(
            w, sheet_name="sccs", index=False)
        cycles_dataframe(a).to_excel(w, sheet_name="cycles", index=False)
        port_order_dataframe(a).to_excel(
            w, sheet_name="port_order", index=False)
    return path


def write_dot(a: PackageAnalysis, path: Path) -> Path:
    """Write Graphviz DOT. Render with:  dot -Tsvg out.dot > out.svg

    Coloring:
      * salmon       node is in a cycle (architectural smell)
      * lightgrey    dead-code candidate (zero internal coupling)
      * ellipse      depth-0 with callers (foundation; port first)
      * lightblue    everything else
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = [
        f'digraph "{a.package_fqn}" {{',
        '  rankdir=LR;',
        '  node [shape=box, style=filled, fillcolor=lightblue];',
    ]
    for n, m in a.metrics.items():
        attrs: list[str] = []
        if m.in_cycle:
            attrs.append('fillcolor=salmon')
        if m.is_dead:
            attrs.append('fillcolor=lightgrey')
        if m.depth == 0 and m.in_degree > 0:
            attrs.append('shape=ellipse')
        attr_str: str = f' [{", ".join(attrs)}]' if attrs else ''
        lines.append(f'  "{n}"{attr_str};')
    for u, v in a.graph.edges:
        lines.append(f'  "{u}" -> "{v}";')
    # Pin each tier (topological depth) to the same rank so the rendered
    # graph reads as horizontal layers: foundations on one side, the most
    # dependent classes on the other. Dead/isolated classes (no edges) have
    # no real tier — leave them out of the rank groups so graphviz floats
    # them freely instead of pinning them into the foundation rank.
    by_depth: dict[int, list[str]] = {}
    for n, m in a.metrics.items():
        if m.is_dead:
            continue
        by_depth.setdefault(m.depth, []).append(n)
    for depth in sorted(by_depth):
        group: str = " ".join(f'"{n}";' for n in sorted(by_depth[depth]))
        lines.append(f'  {{ rank=same; {group} }}')
    lines.append('}')
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_tiers_report(a: PackageAnalysis, path: Path) -> Path:
    """Human-readable dependency-tier report for one package.

    Classes are grouped into tiers by their depth in the internal dependency
    graph: tier 0 depends on nothing else in the package (foundations), and a
    class in tier N depends only on classes in lower tiers. Members of a
    dependency cycle can't be separated, so they share a tier and are flagged;
    the cyclic clusters are also listed on their own at the end.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    short: str = a.package_fqn.rsplit(".", 1)[-1]
    G: nx.DiGraph = a.graph

    tiers: dict[int, list[str]] = {}
    for name, m in a.metrics.items():
        tiers.setdefault(m.depth, []).append(name)
    max_tier: int = max(tiers) if tiers else 0
    clusters: list[set[str]] = [s for s in a.sccs if len(s) > 1]

    lines: list[str] = [
        f"# Dependency Tiers: {short}",
        "",
        f"Classes grouped by dependency tier. **Tier 0** depends on nothing "
        f"else inside {short} (the foundations); a class in tier _N_ depends "
        "only on classes in lower tiers. Classes bound together in a "
        "dependency cycle share a tier and are marked _(cycle)_.",
        "",
        f"- **Classes**: {G.number_of_nodes()}",
        f"- **Tiers**: {max_tier + 1}",
        f"- **Internal edges**: {G.number_of_edges()}",
        f"- **Cyclic clusters**: {len(clusters)}",
        "",
    ]

    for tier in range(max_tier + 1):
        members: list[str] = sorted(tiers.get(tier, []))
        if not members:
            continue
        suffix: str = " — foundations" if tier == 0 else ""
        lines.append(f"## Tier {tier}{suffix} ({len(members)} classes)")
        lines.append("")
        for cls in members:
            mark: str = " _(cycle)_" if a.metrics[cls].in_cycle else ""
            lines.append(f"- {cls}{mark}")
        lines.append("")

    if clusters:
        lines += [
            "## Cyclic clusters",
            "",
            "Each group imports itself (directly or transitively) and must be "
            "ported together — no member stands alone.",
            "",
        ]
        for i, cl in enumerate(sorted(clusters, key=len, reverse=True), 1):
            lines.append(f"{i}. ({len(cl)}) " + " ↔ ".join(sorted(cl)))
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ============================================================================
# Cross-package reporters
# ============================================================================

def write_cross_package_csv(
    edges: list[CrossPackageEdge],
    path: Path,
) -> Path:
    """Write cross-package edges to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cross_package_dataframe(edges).to_csv(path, index=False)
    return path


def write_cross_package_dot(
    edges: list[CrossPackageEdge],
    src_pkg_fqn: str,
    tgt_pkg_fqn: str,
    path: Path,
) -> Path:
    """Write a bipartite Graphviz DOT showing cross-package import edges.

    Left cluster: source-package classes that import from the target.
    Right cluster: target-package classes that are imported.
    Fan-in count (how many source classes reference a target class) is
    appended to each right-side node label.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    src_short: str = src_pkg_fqn.rsplit(".", 1)[-1]
    tgt_short: str = tgt_pkg_fqn.rsplit(".", 1)[-1]

    src_nodes: set[str] = {e.source_class for e in edges}
    tgt_nodes: set[str] = {e.target_class for e in edges}

    tgt_fan_in: dict[str, int] = {}
    for e in edges:
        tgt_fan_in[e.target_class] = tgt_fan_in.get(e.target_class, 0) + 1

    lines: list[str] = [
        f'digraph "{src_short}_uses_{tgt_short}" {{',
        "  rankdir=LR;",
        "  node [style=filled];",
        "",
        f"  subgraph cluster_{src_short} {{",
        f'    label="{src_short}";',
        "    node [fillcolor=lightblue, shape=box];",
    ]
    for n in sorted(src_nodes):
        lines.append(f'    "{n}";')
    lines += [
        "  }",
        "",
        f"  subgraph cluster_{tgt_short} {{",
        f'    label="{tgt_short}";',
        "    node [fillcolor=lightyellow, shape=ellipse];",
    ]
    for n in sorted(tgt_nodes):
        fi: int = tgt_fan_in.get(n, 0)
        lines.append(f'    "{n}" [label="{n}\\n({fi})"];')
    lines.append("  }")
    lines.append("")
    for e in sorted(edges, key=lambda x: (x.source_class, x.target_class)):
        lines.append(f'  "{e.source_class}" -> "{e.target_class}";')
    lines.append("}")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_cross_package_markdown(
    edges: list[CrossPackageEdge],
    src_pkg_fqn: str,
    tgt_pkg_fqn: str,
    path: Path,
) -> Path:
    """Human-readable interconnection report between two packages.

    Lists every point where *src* reaches into *tgt*, grouped by the target
    class (the shared connection point) and, most-shared first, the source
    classes that depend on it.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    src_short: str = src_pkg_fqn.rsplit(".", 1)[-1]
    tgt_short: str = tgt_pkg_fqn.rsplit(".", 1)[-1]

    by_target: dict[str, set[str]] = {}
    for e in edges:
        by_target.setdefault(e.target_class, set()).add(e.source_class)

    n_src: int = len({e.source_class for e in edges})
    n_tgt: int = len(by_target)

    md_lines: list[str] = [
        f"# Interconnection: {src_short} → {tgt_short}",
        "",
        f"Every point where **{src_short}** depends on **{tgt_short}**. Each "
        f"row is a {tgt_short} connection point and the {src_short} classes "
        "that reach into it (most-shared first).",
        "",
        f"- **{tgt_short} connection points**: {n_tgt}",
        f"- **{src_short} classes that cross over**: {n_src}",
        f"- **Interconnection edges**: {len(edges)}",
        "",
        f"| {tgt_short} class | Used by | {src_short} classes |",
        "|---|---:|---|",
    ]
    for tgt_cls, srcs in sorted(
        by_target.items(), key=lambda kv: (-len(kv[1]), kv[0]),
    ):
        src_list: str = ", ".join(f"`{s}`" for s in sorted(srcs))
        md_lines.append(
            f"| `{tgt_cls}` | {len(srcs)} | {src_list} |"
        )

    path.write_text("\n".join(md_lines), encoding="utf-8")
    return path


def _print_cross_package_summary(
    src_short: str,
    tgt_short: str,
    edges: list[CrossPackageEdge],
) -> None:
    n_src: int = len({e.source_class for e in edges})
    n_tgt: int = len({e.target_class for e in edges})
    print(
        f"{src_short}→{tgt_short}: {len(edges)} cross-package edges "
        f"({n_src} {src_short} classes → {n_tgt} {tgt_short} classes)"
    )
    tgt_counts: dict[str, int] = {}
    for e in edges:
        tgt_counts[e.target_class] = tgt_counts.get(e.target_class, 0) + 1
    top: list[tuple[str, int]] = sorted(
        tgt_counts.items(), key=lambda kv: kv[1], reverse=True,
    )[:5]
    print(f"  Top {tgt_short} classes by fan-in:")
    for name, cnt in top:
        print(f"    {name:<40} referenced by {cnt} {src_short} class(es)")


# ============================================================================
# CLI
# ============================================================================

def _resolve_target(name: str, src_root: Path) -> tuple[Path, str]:
    """Map a short package name (e.g. "EPQLibrary") to (target_dir, fqn)."""
    target: Path = src_root / "src" / "gov" / "nist" / "microanalysis" / name
    if not target.is_dir():
        raise FileNotFoundError(f"package directory not found: {target}")
    return target, f"{NIST_PREFIX}.{name}"


def _print_summary(pkg_short: str, a: PackageAnalysis) -> None:
    n_scc_clusters: int = sum(1 for s in a.sccs if len(s) > 1)
    print(
        f"{pkg_short}: {len(a.classes)} files | "
        f"{a.graph.number_of_edges()} internal edges | "
        f"{len(a.wccs)} wccs | {n_scc_clusters} cycle clusters | "
        f"{len(a.cycles)} simple cycles"
    )
    top: list[tuple[str, DependencyMetrics]] = sorted(
        a.metrics.items(),
        key=lambda kv: kv[1].in_degree + kv[1].out_degree,
        reverse=True,
    )[:5]
    print("  Top 5 by internal degree:")
    for name, m in top:
        print(
            f"    {name:<40} in={m.in_degree:<3} out={m.out_degree:<3} "
            f"pagerank={m.pagerank:.4f} instability={m.instability:.3f}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Dependency analyzer for gov.nist.microanalysis Java packages.",
    )
    parser.add_argument(
        "packages", nargs="*",
        default=["EPQLibrary", "Utility"],
        help="Short package names to analyze (default: EPQLibrary Utility).",
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).resolve().parent / "reports",
        help="Output directory (default: <script_dir>/reports/).",
    )
    parser.add_argument(
        "--format",
        choices=["csv", "xlsx", "dot", "md", "all"],
        default="all",
        help=("Output format(s) to write (default: all). 'all' writes only "
              "the human-readable artifacts: the per-package TIERS report and "
              "cross-package INTERCONNECTION report (md) plus the graphs "
              "(dot). 'csv'/'xlsx' are opt-in raw data dumps."),
    )
    # This script lives at .../src/gov/nist/microanalysis/PyEPQ/tools/, so
    # parents[6] is the repo root (the directory that contains src/).
    parser.add_argument(
        "--src-root", type=Path,
        default=Path(__file__).resolve().parents[6],
        help=("Repository root (the directory containing "
              "src/gov/nist/microanalysis/). Defaults to a path derived "
              "from this script's location."),
    )
    parser.add_argument(
        "--no-cycles", action="store_true",
        help=("Skip enumerating simple cycles. Cycle enumeration uses "
              "Johnson's algorithm and can be slow on dense graphs."),
    )
    parser.add_argument(
        "--cycle-limit", type=int, default=1000,
        help="Cap on enumerated elementary cycles (default: 1000).",
    )
    parser.add_argument(
        "--no-cross-package", action="store_true",
        help=("Skip cross-package dependency analysis when multiple packages "
              "are specified."),
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable INFO-level logging.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    args.out.mkdir(parents=True, exist_ok=True)
    cycle_limit: int = 0 if args.no_cycles else args.cycle_limit

    exit_code: int = 0
    analyses: dict[str, PackageAnalysis] = {}
    for pkg_short in args.packages:
        try:
            target, pkg_fqn = _resolve_target(pkg_short, args.src_root)
        except FileNotFoundError as e:
            log.error(str(e))
            print(f"ERROR: {e}", file=sys.stderr)
            exit_code = 1
            continue

        log.info("Analyzing %s", pkg_fqn)
        try:
            analysis: PackageAnalysis = analyze_package(
                target, pkg_fqn, cycle_limit=cycle_limit,
            )
        except FileNotFoundError as e:
            log.error(str(e))
            print(f"ERROR: {e}", file=sys.stderr)
            exit_code = 1
            continue

        if args.format == "xlsx":
            p: Path = write_xlsx(analysis, args.out / f"{pkg_short}.xlsx")
            log.info("  wrote %s", p)
        if args.format == "csv":
            for p in write_csvs(analysis, args.out, pkg_short):
                log.info("  wrote %s", p)
        if args.format in ("dot", "all"):
            p = write_dot(analysis, args.out / f"{pkg_short}.dot")
            log.info("  wrote %s  (render: dot -Tsvg %s > %s.svg)",
                     p, p.name, pkg_short)
        if args.format in ("md", "all"):
            p = write_tiers_report(
                analysis, args.out / f"{pkg_short}_tiers.md",
            )
            log.info("  wrote %s", p)

        _print_summary(pkg_short, analysis)
        analyses[pkg_short] = analysis

    # Cross-package analysis: every ordered (src, tgt) pair where edges exist.
    if not args.no_cross_package and len(analyses) > 1:
        for src_short, src_a in analyses.items():
            for tgt_short, tgt_a in analyses.items():
                if src_short == tgt_short:
                    continue
                xedges: list[CrossPackageEdge] = build_cross_package_edges(
                    src_a, tgt_a.package_fqn,
                )
                if not xedges:
                    continue
                prefix: str = f"{src_short}_uses_{tgt_short}"
                _print_cross_package_summary(src_short, tgt_short, xedges)
                if args.format == "csv":
                    xp: Path = write_cross_package_csv(
                        xedges, args.out / f"{prefix}_edges.csv",
                    )
                    log.info("  wrote %s", xp)
                if args.format in ("dot", "all"):
                    xp = write_cross_package_dot(
                        xedges, src_a.package_fqn, tgt_a.package_fqn,
                        args.out / f"{prefix}.dot",
                    )
                    log.info(
                        "  wrote %s  (render: dot -Tsvg %s > %s.svg)",
                        xp, xp.name, prefix,
                    )
                if args.format in ("md", "all"):
                    xp = write_cross_package_markdown(
                        xedges, src_a.package_fqn, tgt_a.package_fqn,
                        args.out / f"{prefix}.md",
                    )
                    log.info("  wrote %s", xp)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
