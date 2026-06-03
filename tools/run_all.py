"""
run_all.py -- convenience runner that analyzes EPQLibrary and Utility.

Thin wrapper around `dependency_map.main` for the most common case (analyze
both packages, write all output formats next to this script). Equivalent
to running:

    python dependency_map.py EPQLibrary Utility --format all --verbose

Kept as a separate script so existing automation that invokes `run_all.py`
keeps working.
"""

from __future__ import annotations

import sys

from dependency_map import main

if __name__ == "__main__":
    sys.exit(main(["EPQLibrary", "Utility", "--format", "all", "--verbose"]))
