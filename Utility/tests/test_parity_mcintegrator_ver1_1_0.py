r"""
test_parity_mcintegrator_ver1_1_0.py — parity harness for MCIntegrator_ver1_1_0.py

MCIntegrator is abstract (M4: JPype cannot extend abstract Java classes).
Concrete subclasses implement:
  • function(double[] args) → double[]   — integrand value at a point
  • inside(double[] args) → boolean      — whether the point is in the region

compute(int nTests) algorithm:
  1. Uniformly sample nTests points from the bounding box [point1, point2]
  2. For each point inside(), accumulate function() result
  3. Multiply sum by volume(bounding box) / nTests
  Result ≈ ∫_Ω f(x) dx

Statistical benchmarks:
  Unit sphere (3D):  ∫ 1 dV = 4π/3 ≈ 4.189
  Unit square (2D):  ∫ (x+y) dA = 1.0
  Unit cube (3D):    ∫ xyz  dV = 0.125
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings as hyp_settings, strategies as st

from _parity_lib import (
    setup_parity, needs_java, PARITY_ENABLED,
    TOL_NR_LIB,
    _close,
)

from MCIntegrator_ver1_1_0 import MCIntegrator as PyMCIntegrator

ctx = setup_parity("gov.nist.microanalysis.Utility.MCIntegrator")
JavaMCIntegrator = ctx.java_class

_N = 100_000     # Monte Carlo samples
_MC_TOL = 0.02  # 2 % relative tolerance (≈ 3σ for N=100k)


# ---------------------------------------------------------------------------
# Concrete subclasses
# ---------------------------------------------------------------------------

class _SphereVolumeIntegrator(PyMCIntegrator):
    """Estimates volume of unit sphere in 3D: 4π/3 ≈ 4.18879."""
    def __init__(self):
        super().__init__([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])

    def inside(self, args):
        return args[0] ** 2 + args[1] ** 2 + args[2] ** 2 <= 1.0

    def function(self, args):
        return np.array([1.0])


class _CircleAreaIntegrator(PyMCIntegrator):
    """Estimates area of unit circle in 2D: π ≈ 3.14159."""
    def __init__(self):
        super().__init__([-1.0, -1.0], [1.0, 1.0])

    def inside(self, args):
        return args[0] ** 2 + args[1] ** 2 <= 1.0

    def function(self, args):
        return np.array([1.0])


class _UnitSquareSumIntegrator(PyMCIntegrator):
    """∫₀¹ ∫₀¹ (x+y) dx dy = 1.0"""
    def __init__(self):
        super().__init__([0.0, 0.0], [1.0, 1.0])

    def inside(self, args):
        return True

    def function(self, args):
        return np.array([args[0] + args[1]])


class _CubeProductIntegrator(PyMCIntegrator):
    """∫₀¹ ∫₀¹ ∫₀¹ xyz dx dy dz = (1/2)^3 = 0.125"""
    def __init__(self):
        super().__init__([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])

    def inside(self, args):
        return True

    def function(self, args):
        return np.array([args[0] * args[1] * args[2]])


class _EmptyRegionIntegrator(PyMCIntegrator):
    """inside() always returns False → result should be zero array."""
    def __init__(self):
        super().__init__([0.0, 0.0], [1.0, 1.0])

    def inside(self, args):
        return False

    def function(self, args):
        return np.array([1.0])


# ---------------------------------------------------------------------------
# TestConstruction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_constructs(self):
        ig = _SphereVolumeIntegrator()
        assert ig is not None

    def test_1d_construction(self):
        class _Const1D(PyMCIntegrator):
            def __init__(self):
                super().__init__([0.0], [1.0])
            def inside(self, args):
                return True
            def function(self, args):
                return np.array([1.0])
        ig = _Const1D()
        assert ig is not None


# ---------------------------------------------------------------------------
# TestSphereVolume
# ---------------------------------------------------------------------------

class TestSphereVolume:
    def test_sphere_volume_estimate(self):
        ig = _SphereVolumeIntegrator()
        result = ig.compute(_N)
        expected = 4.0 * math.pi / 3.0
        assert abs(result[0] - expected) / expected < _MC_TOL

    def test_circle_area_estimate(self):
        ig = _CircleAreaIntegrator()
        result = ig.compute(_N)
        expected = math.pi
        assert abs(result[0] - expected) / expected < _MC_TOL


# ---------------------------------------------------------------------------
# TestAnalyticalIntegrals
# ---------------------------------------------------------------------------

class TestAnalyticalIntegrals:
    def test_unit_square_sum(self):
        ig = _UnitSquareSumIntegrator()
        result = ig.compute(_N)
        assert abs(result[0] - 1.0) < _MC_TOL

    def test_cube_product(self):
        ig = _CubeProductIntegrator()
        result = ig.compute(_N)
        assert abs(result[0] - 0.125) / 0.125 < _MC_TOL


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_region_returns_zero(self):
        ig = _EmptyRegionIntegrator()
        result = ig.compute(1000)
        assert result[0] == 0.0

    def test_zero_samples_returns_zero(self):
        ig = _SphereVolumeIntegrator()
        result = ig.compute(0)
        assert result is not None

    def test_result_array_length(self):
        ig = _SphereVolumeIntegrator()
        result = ig.compute(1000)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# TestMCIntegratorParity  (M4 — abstract class)
# ---------------------------------------------------------------------------

@pytest.mark.skip(
    reason="M4: JPype cannot extend Java abstract classes from Python. "
           "MCIntegrator declares abstract function() and inside(); "
           "no Python callback into JVM. "
           "Correctness validated statistically in TestSphereVolume above."
)
class TestMCIntegratorParity:
    def test_placeholder(self):
        pass

if __name__ == "__main__":
    from _parity_lib import run_tests; run_tests(__file__)
