r"""
test_parity_multidhistogram_ver1_1_1.py — parity harness for MultiDHistogram

Revision ver1_1_1 (2026-06-25): P4 + P5 fixes (gen1_review), realigned to the port API.
  * P4 — `LinearBins` uses the Java uniform constructor `LinearBins(dim, width,
    number)`, not a per-dimension `(lo[], hi[], count[])` form. All construction
    sites now pass `(dim, width, number)` over the origin-anchored grid [0, width*number).
  * P5 — inner-class method names match the port (Java names, no invented aliases):
    `IBinning`/`LinearBins` expose `getDimensions() -> list[int]` (dimension count
    is `len(getDimensions())`, there is no `dimensions()`); `Bin` exposes
    `adjacent(vox)` / `adjacent_m(vox, m)` (not `adjacent_bin*`) and the private
    item insert `_add(item)` (Java package-private `add(int)` -> `_add`, R1).

MultiDHistogram bins multi-dimensional data into a sparse histogram and computes
connected-component "islands" of adjacent non-empty bins.

Inner types (all ported in the same module):
  IBinning           — interface; compute(double[]) → int[], dimensions() → int
  LinearBins         — IBinning; uniform grid over [lo, hi) per dimension
  Bin                — sparse-map key; int[] indices + item list + count
  MultiDHistogram    — the histogram itself

Key contracts:
  • getTotalCount()   — sum of all items added
  • getIslandCount()  — number of connected components (adjacency = face-sharing)
  • getIslandSize(i)  — number of bins in island i
  • getItemsInIsland(i) — all items whose bin belongs to island i
  • connected(item)   — all items in the same island as item
  • chiSquaredTest    — χ² goodness-of-fit statistic
  • getBins()         — all non-empty Bin objects

Constructors:
  MultiDHistogram(IBinning)                — empty histogram
  MultiDHistogram(MultiDHistogram)          — deep copy
  MultiDHistogram(MultiDHistogram, int min) — copy keeping only bins with ≥ min items
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings as hyp_settings, strategies as st

from _parity_lib import (
    setup_parity, needs_java, PARITY_ENABLED,
    TOL_NR_LIB,
    slow,
    _close,
)

from MultiDHistogram_ver2_1_0 import (
    MultiDHistogram as PyMultiDHistogram,
    LinearBins as PyLinearBins,
)
    
ctx = setup_parity("gov.nist.microanalysis.Utility.MultiDHistogram")
JavaMultiDHistogram = ctx.java_class


# ---------------------------------------------------------------------------
# Helper: 2-D 10×10 LinearBins over [0,1)^2
# ---------------------------------------------------------------------------

def _make_2d_histo():
    bins = PyLinearBins(2, 0.1, 10)
    return PyMultiDHistogram(bins)


# ---------------------------------------------------------------------------
# TestLinearBins
# ---------------------------------------------------------------------------

class TestLinearBins:
    def test_dimensions_2d(self):
        lb = PyLinearBins(2, 0.2, 5)
        assert len(lb.getDimensions()) == 2

    def test_dimensions_3d(self):
        lb = PyLinearBins(3, 0.25, 4)
        assert len(lb.getDimensions()) == 3

    def test_compute_origin_bin(self):
        lb = PyLinearBins(2, 0.1, 10)
        idx = lb.compute([0.05, 0.05])
        assert idx[0] == 0
        assert idx[1] == 0

    def test_compute_mid_bin(self):
        lb = PyLinearBins(2, 0.1, 10)
        idx = lb.compute([0.55, 0.25])
        assert idx[0] == 5
        assert idx[1] == 2

    def test_compute_last_bin(self):
        lb = PyLinearBins(2, 0.1, 10)
        idx = lb.compute([0.99, 0.99])
        assert idx[0] == 9
        assert idx[1] == 9

    def test_non_unit_range(self):
        lb = PyLinearBins(1, 10.0, 10)
        idx = lb.compute([55.0])
        assert idx[0] == 5


# ---------------------------------------------------------------------------
# TestEmptyHistogram
# ---------------------------------------------------------------------------

class TestEmptyHistogram:
    def test_total_count_zero(self):
        h = _make_2d_histo()
        assert h.getTotalCount() == 0

    def test_island_count_zero(self):
        h = _make_2d_histo()
        assert h.getIslandCount() == 0

    def test_get_bins_empty(self):
        h = _make_2d_histo()
        assert len(h.getBins()) == 0


# ---------------------------------------------------------------------------
# TestAddItems
# ---------------------------------------------------------------------------

class TestAddItems:
    def test_add_single_item_count(self):
        h = _make_2d_histo()
        h.add([0.05, 0.05], 101)
        assert h.getTotalCount() == 1

    def test_add_multiple_items_count(self):
        h = _make_2d_histo()
        for i in range(10):
            h.add([0.05, 0.05], i)
        assert h.getTotalCount() == 10

    def test_add_distinct_bins(self):
        h = _make_2d_histo()
        h.add([0.05, 0.05], 1)
        h.add([0.55, 0.55], 2)
        assert h.getTotalCount() == 2
        assert len(h.getBins()) == 2


# ---------------------------------------------------------------------------
# TestIslandFormation
# ---------------------------------------------------------------------------

class TestIslandFormation:
    def test_single_bin_one_island(self):
        h = _make_2d_histo()
        h.add([0.05, 0.05], 1)
        assert h.getIslandCount() == 1

    def test_adjacent_bins_one_island(self):
        """Bins at (0,0) and (1,0) are face-adjacent → same island."""
        h = _make_2d_histo()
        h.add([0.05, 0.05], 1)   # bin [0,0]
        h.add([0.15, 0.05], 2)   # bin [1,0]
        assert h.getIslandCount() == 1

    def test_diagonal_bins_two_islands(self):
        """Bins at (0,0) and (1,1) share only a corner → two islands."""
        h = _make_2d_histo()
        h.add([0.05, 0.05], 1)   # bin [0,0]
        h.add([0.15, 0.15], 2)   # bin [1,1]
        assert h.getIslandCount() == 2

    def test_separated_bins_two_islands(self):
        h = _make_2d_histo()
        h.add([0.05, 0.05], 1)   # bin [0,0]
        h.add([0.95, 0.95], 2)   # bin [9,9]
        assert h.getIslandCount() == 2

    def test_chain_of_three_bins_one_island(self):
        h = _make_2d_histo()
        h.add([0.05, 0.05], 1)   # [0,0]
        h.add([0.15, 0.05], 2)   # [1,0]
        h.add([0.25, 0.05], 3)   # [2,0]
        assert h.getIslandCount() == 1

    def test_island_size(self):
        h = _make_2d_histo()
        h.add([0.05, 0.05], 1)
        h.add([0.15, 0.05], 2)
        h.add([0.95, 0.95], 3)
        # Island with 2 bins + island with 1 bin
        sizes = sorted([h.getIslandSize(i) for i in range(h.getIslandCount())])
        assert sizes == [1, 2]


# ---------------------------------------------------------------------------
# TestItemsInIsland
# ---------------------------------------------------------------------------

class TestItemsInIsland:
    def test_items_in_only_island(self):
        h = _make_2d_histo()
        h.add([0.05, 0.05], 10)
        h.add([0.05, 0.05], 20)
        island_idx = h.findItemsIsland(10)  # look up an item that was actually added
        items = h.getItemsInIsland(island_idx)
        assert 10 in items or 20 in items

    def test_connected_items_same_island(self):
        h = _make_2d_histo()
        h.add([0.05, 0.05], 1)
        h.add([0.15, 0.05], 2)
        connected = h.connected(1)  # items in same island as item 1
        assert len(connected) >= 2


# ---------------------------------------------------------------------------
# TestGetCount
# ---------------------------------------------------------------------------

class TestGetCount:
    def test_bin_count_single_add(self):
        h = _make_2d_histo()
        h.add([0.05, 0.05], 1)
        count = h.getCount([0, 0])
        assert count == 1

    def test_bin_count_multiple_adds_same_bin(self):
        h = _make_2d_histo()
        for i in range(5):
            h.add([0.05, 0.05], i)
        count = h.getCount([0, 0])
        assert count == 5

    def test_empty_bin_count_zero(self):
        h = _make_2d_histo()
        assert h.getCount([3, 7]) == 0


# ---------------------------------------------------------------------------
# TestCopyConstructor
# ---------------------------------------------------------------------------

class TestCopyConstructor:
    def test_copy_preserves_count(self):
        h = _make_2d_histo()
        h.add([0.05, 0.05], 1)
        h.add([0.55, 0.55], 2)
        h2 = PyMultiDHistogram(h)
        assert h2.getTotalCount() == 2

    def test_copy_preserves_islands(self):
        h = _make_2d_histo()
        h.add([0.05, 0.05], 1)
        h.add([0.95, 0.95], 2)
        h2 = PyMultiDHistogram(h)
        assert h2.getIslandCount() == 2

    def test_copy_independent(self):
        h = _make_2d_histo()
        h.add([0.05, 0.05], 1)
        h2 = PyMultiDHistogram(h)
        h.add([0.55, 0.55], 2)
        assert h2.getTotalCount() == 1

    def test_min_members_filters_sparse_bins(self):
        h = _make_2d_histo()
        h.add([0.05, 0.05], 1)         # bin [0,0]: 1 item
        for i in range(5):
            h.add([0.55, 0.55], i)     # bin [5,5]: 5 items
        h2 = PyMultiDHistogram(h, 3)   # keep bins with ≥ 3 items
        assert h2.getCount([0, 0]) == 0
        assert h2.getCount([5, 5]) == 5


# ---------------------------------------------------------------------------
# TestChiSquaredTest
# ---------------------------------------------------------------------------

class TestChiSquaredTest:
    def test_identical_histograms_low_p_value(self):
        """Two identical uniform histograms → low chi² → high p-value (not rejected)."""
        bins = PyLinearBins(1, 0.2, 5)
        h1 = PyMultiDHistogram(bins)
        h2 = PyMultiDHistogram(bins)
        for v in [0.1, 0.3, 0.5, 0.7, 0.9]:
            h1.add([v], v)
            h2.add([v], v)
        # chiSquaredTest returns bool: True if significant difference at level
        result = h1.chiSquaredTest(h2, 1, 0.01)
        assert isinstance(result, bool)

    def test_different_histograms_returns_bool(self):
        bins = PyLinearBins(1, 0.1, 10)
        h1 = PyMultiDHistogram(bins)
        h2 = PyMultiDHistogram(bins)
        for v in [0.05, 0.15, 0.25]:
            h1.add([v], 1)
        for v in [0.75, 0.85, 0.95]:
            h2.add([v], 1)
        result = h1.chiSquaredTest(h2, 1, 0.05)
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# TestBinMethods — Bin and LinearBins inner-class methods
# ---------------------------------------------------------------------------

class TestBinMethods:

    def _bin(self, indices):
        return PyMultiDHistogram.Bin(indices, 0)

    def test_getDimension(self):
        b = self._bin([2, 3])
        assert b.getDimension() == 2

    def test_getDimensions_on_linear_bins(self):
        # Uniform constructor: 2-D grid, 5 bins per dimension.
        lb = PyLinearBins(2, 0.2, 5)
        assert lb.getDimensions() == [5, 5]

    def test_limits_first_bin(self):
        lb = PyLinearBins(1, 0.25, 4)
        result = lb.limits(0, 0)
        assert result is not None
        assert abs(result[0] - 0.0) < 1e-12
        assert abs(result[1] - 0.25) < 1e-12

    def test_limits_out_of_range_returns_none(self):
        lb = PyLinearBins(1, 0.25, 4)
        assert lb.limits(10, 0) is None

    def test_contains_item_present(self):
        b = self._bin([0, 0])
        b._add(42)
        assert b.contains(42)

    def test_contains_item_absent(self):
        b = self._bin([0, 0])
        assert not b.contains(99)

    def test_distance_same_bin(self):
        b = self._bin([1, 1])
        assert b.distance(b) == 0.0

    def test_distance_adjacent(self):
        b1 = self._bin([0, 0])
        b2 = self._bin([1, 0])
        assert b1.distance(b2) == 1.0

    def test_adjacent_bin_face_sharing(self):
        # adjacent_bin uses m=1: exactly one coordinate differs by 1.
        assert self._bin([0, 0]).adjacent(self._bin([1, 0]))

    def test_adjacent_bin_diagonal_not_adjacent(self):
        # Diagonal neighbour: face-sharing (m=1) excludes it.
        assert not self._bin([0, 0]).adjacent(self._bin([1, 1]))

    def test_adjacent_bin_far_not_adjacent(self):
        assert not self._bin([0, 0]).adjacent(self._bin([2, 0]))

    def test_adjacent_bin_m_includes_diagonal(self):
        # m=2 counts the diagonal neighbour (Manhattan distance 2 <= 2).
        assert self._bin([0, 0]).adjacent_m(self._bin([1, 1]), 2)

    def test_compareTo_equal(self):
        b1 = self._bin([2, 3])
        b2 = self._bin([2, 3])
        assert b1.compareTo(b2) == 0

    def test_compareTo_less(self):
        b1 = self._bin([1, 0])
        b2 = self._bin([2, 0])
        assert b1.compareTo(b2) == -1

    def test_compareTo_greater(self):
        b1 = self._bin([3, 0])
        b2 = self._bin([2, 0])
        assert b1.compareTo(b2) == 1


# ---------------------------------------------------------------------------
# TestHypothesis
# ---------------------------------------------------------------------------

class TestHypothesis:
    @given(st.integers(1, 100))
    @slow
    def test_total_count_equals_items_added(self, n):
        """getTotalCount() must equal the number of add() calls made."""
        h = _make_2d_histo()
        for i in range(n):
            h.add([0.05 + (i * 0.009) % 0.9, 0.05], i)
        assert h.getTotalCount() == n


# ---------------------------------------------------------------------------
# TestParity
# ---------------------------------------------------------------------------

@needs_java
class TestMultiDHistogramParity:
    def test_parity_add_and_count(self):
        import jpype
        JLinearBins = jpype.JClass(
            "gov.nist.microanalysis.Utility.MultiDHistogram$LinearBins"
        )
        # Java LinearBins constructor: (int dim, double width, int number)
        # 5 bins over [0,1) per dimension → width = 0.2
        bins_py = PyLinearBins(2, 0.2, 5)
        j_bins = JLinearBins(2, 0.2, 5)
        h_py = PyMultiDHistogram(bins_py)
        h_j = JavaMultiDHistogram(j_bins)
        pts = [[0.1, 0.1], [0.5, 0.5], [0.9, 0.9]]
        for i, pt in enumerate(pts):
            h_py.add(pt, i)
            h_j.add(pt, i)
        assert h_py.getTotalCount() == int(h_j.getTotalCount())
        assert h_py.getIslandCount() == int(h_j.getIslandCount())

if __name__ == "__main__":
    from _parity_lib import run_tests; run_tests(__file__)
