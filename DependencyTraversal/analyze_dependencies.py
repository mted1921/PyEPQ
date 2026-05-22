"""
Dependency analyzer for gov.nist.microanalysis Java packages.

Parses every .java file in a target directory, extracts import statements,
extends/implements relationships, and class metadata, then returns:

  nodes_df      — one row per .java file with import counts and
                  interconnectedness metrics (in/out degree within package)
  edges_df      — one row per dependency (import / extends / implements)
  components_df — connected components ranked by size

Import as a module and call analyze_package(), or run directly to process
EPQLibrary and write CSVs to the same directory as this script.
"""

import re
import sys
from pathlib import Path

import networkx as nx
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NIST_PREFIX = "gov.nist.microanalysis"

NIST_PACKAGES = {
    "EPQDatabase", "EPQImage", "EPQLibrary", "EPQTools",
    "JythonGUI", "NISTMonte", "Utility", "EPQTests",
}

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

RE_PACKAGE = re.compile(r"^\s*package\s+([\w.]+)\s*;", re.MULTILINE)
RE_IMPORT  = re.compile(r"^\s*import\s+(static\s+)?([\w.*]+)\s*;", re.MULTILINE)
RE_CLASS_HDR = re.compile(
    r"""
    (?:^|\s)
    (?P<modifiers>(?:(?:public|protected|private|abstract|final|static)\s+)*)
    (?P<kind>class|interface|enum|@interface)
    \s+(?P<name>\w+)
    (?:\s*<[^{]*?>)?
    (?:\s+extends\s+(?P<extends>[\w,\s<>]+?))?
    (?:\s+implements\s+(?P<implements>[\w,\s<>]+?))?
    \s*\{
    """,
    re.MULTILINE | re.VERBOSE,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def classify_import(fqn: str, package_fqn: str) -> str:
    if fqn.startswith("java.") or fqn.startswith("javax."):
        return "java_stdlib"
    if fqn.startswith(package_fqn):
        return "internal"
    if fqn.startswith(NIST_PREFIX):
        pkg = fqn[len(NIST_PREFIX) + 1:].split(".")[0]
        if pkg in NIST_PACKAGES:
            return f"nist_{pkg.lower()}"
        return "nist_other"
    return "external"


def split_type_list(raw: str) -> list[str]:
    if not raw:
        return []
    clean = re.sub(r"<[^<>]*>", "", raw)
    return [t.strip() for t in clean.split(",") if t.strip()]


def parse_java_file(path: Path) -> dict:
    src = path.read_text(encoding="utf-8", errors="replace")
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL)

    package = ""
    m = RE_PACKAGE.search(src)
    if m:
        package = m.group(1)

    imports = [m.group(2) for m in RE_IMPORT.finditer(src)]

    class_name = kind = modifiers = extends_raw = implements_raw = ""
    m = RE_CLASS_HDR.search(src)
    if m:
        class_name     = m.group("name")
        kind           = m.group("kind")
        modifiers      = m.group("modifiers").strip()
        extends_raw    = (m.group("extends") or "").strip()
        implements_raw = (m.group("implements") or "").strip()

    return {
        "file":       path.name,
        "package":    package,
        "class_name": class_name,
        "kind":       kind,
        "modifiers":  modifiers,
        "extends":    split_type_list(extends_raw),
        "implements": split_type_list(implements_raw),
        "imports":    imports,
    }

# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyze_package(target_dir: Path, package_fqn: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Analyze all .java files under target_dir.

    Returns (nodes_df, edges_df, components_df).
    """
    java_files = sorted(target_dir.rglob("*.java"))
    if not java_files:
        sys.exit(f"No .java files found in {target_dir}")

    parsed = []
    pkg_classes: set[str] = set()
    for f in java_files:
        d = parse_java_file(f)
        parsed.append(d)
        if d["class_name"]:
            pkg_classes.add(d["class_name"])

    # ------------------------------------------------------------------
    # edges_df
    # ------------------------------------------------------------------
    edge_rows = []
    for d in parsed:
        for fqn in d["imports"]:
            simple = fqn.rsplit(".", 1)[-1]
            cat = classify_import(fqn, package_fqn)
            edge_rows.append({
                "source_file":        d["file"],
                "source_class":       d["class_name"],
                "dep_fqn":            fqn,
                "dep_simple":         simple,
                "dep_category":       cat,
                "rel_type":           "import",
                "is_internal_target": simple in pkg_classes,
            })
        for name in d["extends"]:
            edge_rows.append({
                "source_file":        d["file"],
                "source_class":       d["class_name"],
                "dep_fqn":            name,
                "dep_simple":         name,
                "dep_category":       "internal" if name in pkg_classes else "unknown",
                "rel_type":           "extends",
                "is_internal_target": name in pkg_classes,
            })
        for name in d["implements"]:
            edge_rows.append({
                "source_file":        d["file"],
                "source_class":       d["class_name"],
                "dep_fqn":            name,
                "dep_simple":         name,
                "dep_category":       "internal" if name in pkg_classes else "unknown",
                "rel_type":           "implements",
                "is_internal_target": name in pkg_classes,
            })

    edges_df = pd.DataFrame(edge_rows)

    # ------------------------------------------------------------------
    # Compute in/out degree within the package
    # ------------------------------------------------------------------
    internal_edges = edges_df[edges_df["is_internal_target"] == True]

    out_degree = internal_edges.groupby("source_class")["dep_simple"].count().rename("out_degree")
    in_degree  = internal_edges[internal_edges["dep_simple"] != "*"] \
                     .groupby("dep_simple")["source_class"].count().rename("in_degree")

    # ------------------------------------------------------------------
    # nodes_df
    # ------------------------------------------------------------------
    node_rows = []
    for d in parsed:
        counts = {}
        for fqn in d["imports"]:
            cat = classify_import(fqn, package_fqn)
            counts[cat] = counts.get(cat, 0) + 1

        n_internal = counts.get("internal", 0)
        n_external = sum(v for k, v in counts.items() if k != "internal")

        node_rows.append({
            "file":             d["file"],
            "package":          d["package"],
            "class_name":       d["class_name"],
            "kind":             d["kind"],
            "modifiers":        d["modifiers"],
            "extends":          ", ".join(d["extends"]),
            "implements":       ", ".join(d["implements"]),
            "total_imports":    len(d["imports"]),
            "n_internal":       n_internal,
            "n_java_stdlib":    counts.get("java_stdlib", 0),
            "n_nist_utility":   counts.get("nist_utility", 0),
            "n_nist_nistmonte": counts.get("nist_nistmonte", 0),
            "n_nist_epqlib":    counts.get("nist_epqlibrary", 0),
            "n_nist_other":     counts.get("nist_other", 0),
            "n_external":       counts.get("external", 0),
            "external_imports": n_external,
            "imports_raw":      "; ".join(d["imports"]),
        })

    nodes_df = pd.DataFrame(node_rows)
    nodes_df = nodes_df.join(out_degree, on="class_name").join(in_degree, on="class_name")
    nodes_df["out_degree"]      = nodes_df["out_degree"].fillna(0).astype(int)
    nodes_df["in_degree"]       = nodes_df["in_degree"].fillna(0).astype(int)
    nodes_df["internal_degree"] = nodes_df["out_degree"] + nodes_df["in_degree"]

    # ------------------------------------------------------------------
    # components_df
    # ------------------------------------------------------------------
    G = nx.DiGraph()
    G.add_edges_from(zip(internal_edges["source_class"], internal_edges["dep_simple"]))
    for cls in nodes_df["class_name"].dropna():
        G.add_node(cls)

    components = sorted(nx.weakly_connected_components(G), key=len, reverse=True)
    comp_rows = []
    for i, comp in enumerate(components, 1):
        for cls in sorted(comp):
            comp_rows.append({"component": i, "size": len(comp), "class_name": cls})
    components_df = pd.DataFrame(comp_rows)

    return nodes_df, edges_df, components_df


# ---------------------------------------------------------------------------
# CLI entry point (EPQLibrary only)
# ---------------------------------------------------------------------------

def main():
    root      = Path(__file__).resolve().parent.parent
    out_dir   = Path(__file__).resolve().parent
    target    = root / "src" / "gov" / "nist" / "microanalysis" / "EPQLibrary"
    pkg_fqn   = "gov.nist.microanalysis.EPQLibrary"

    print(f"Scanning: {target}")
    nodes_df, edges_df, components_df = analyze_package(target, pkg_fqn)

    nodes_df.to_csv(out_dir / "epqlib_nodes.csv", index=False)
    edges_df.to_csv(out_dir / "epqlib_edges.csv", index=False)
    components_df.to_csv(out_dir / "epqlib_components.csv", index=False)

    print(f"\n{len(nodes_df)} files | {len(edges_df)} edges | "
          f"{components_df['component'].max()} components")
    print("\nTop 10 by internal_degree:")
    print(nodes_df[["class_name", "in_degree", "out_degree", "internal_degree", "external_imports"]]
          .sort_values("internal_degree", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
