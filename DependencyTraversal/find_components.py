"""
Identifies isolated dependency clusters within EPQLibrary using weakly
connected components on the intra-library dependency graph.

Reads epqlib_edges.csv (produced by analyze_dependencies.py) and prints
each component ranked by size, largest first.
"""

import pandas as pd
import networkx as nx
from pathlib import Path

EDGES_CSV = Path(__file__).resolve().parent / "epqlib_edges.csv"

edges_df = pd.read_csv(EDGES_CSV)

# Keep only edges where both ends are EPQLibrary classes
internal = edges_df[edges_df["is_epqlib_target"] == True].copy()

G = nx.DiGraph()
G.add_edges_from(zip(internal["source_class"], internal["dep_simple"]))

# Also add nodes that appear only as sources with no internal deps
for cls in edges_df["source_class"].dropna().unique():
    G.add_node(cls)

components = sorted(
    nx.weakly_connected_components(G),
    key=len,
    reverse=True,
)

print(f"Found {len(components)} isolated component(s)\n")
print(f"{'#':<4} {'Size':<6}  Classes")
print("-" * 72)

for i, comp in enumerate(components, 1):
    members = sorted(comp)
    print(f"{i:<4} {len(comp):<6}  {members[0]}")
    for name in members[1:]:
        print(f"{'':10}  {name}")
    print()

# Write results to CSV
rows = []
for i, comp in enumerate(components, 1):
    for cls in sorted(comp):
        rows.append({"component": i, "size": len(comp), "class_name": cls})

out = Path(__file__).resolve().parent / "epqlib_components.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print(f"Written to {out}")
