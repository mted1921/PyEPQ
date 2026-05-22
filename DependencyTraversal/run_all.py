"""
Runs dependency analysis on both EPQLibrary and Utility packages.

Writes one xlsx per package, each with three sheets:
  nodes      — class metadata + interconnectedness metrics
  edges      — dependency edges (import / extends / implements)
  components — connected components ranked by size

Output files:
  EPQLibrary.xlsx
  Utility.xlsx
"""

import pandas as pd
from pathlib import Path
from analyze_dependencies import analyze_package

ROOT    = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent
NIST    = ROOT / "src" / "gov" / "nist" / "microanalysis"

PACKAGES = [
    ("EPQLibrary", "gov.nist.microanalysis.EPQLibrary"),
    ("Utility",    "gov.nist.microanalysis.Utility"),
]

for pkg_name, pkg_fqn in PACKAGES:
    target    = NIST / pkg_name
    xlsx_path = OUT_DIR / f"{pkg_name}.xlsx"

    print(f"Scanning {pkg_name} ...")
    nodes_df, edges_df, components_df = analyze_package(target, pkg_fqn)

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        nodes_df.to_excel(writer,      sheet_name="nodes",      index=False)
        edges_df.to_excel(writer,      sheet_name="edges",       index=False)
        components_df.to_excel(writer, sheet_name="components",  index=False)

    print(f"  {len(nodes_df)} files | {len(edges_df)} edges | "
          f"{components_df['component'].max()} components")
    print(f"  Top 5 by internal_degree:")
    top = (nodes_df[["class_name", "in_degree", "out_degree",
                      "internal_degree", "external_imports"]]
           .sort_values("internal_degree", ascending=False).head(5))
    for _, r in top.iterrows():
        print(f"    {r['class_name']:<40} "
              f"in={r['in_degree']}  out={r['out_degree']}  "
              f"external={r['external_imports']}")
    print(f"  Written to {xlsx_path}\n")
