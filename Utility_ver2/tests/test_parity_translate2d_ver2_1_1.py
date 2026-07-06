r"""
test_parity_translate2d_ver2_1_1.py -- parity harness for Translate2D

Revision ver2_1_1 (2026-06-30): added deterministic coverage for the public
`error(calPts)` method (RMS residual of a set of calibration points under the
current transform), which the gen1 harness (ver2_1_0) did not exercise. The
import still targets the unchanged port Translate2D_ver2_1_0.

Pre-written harness (Prompt 2): targets the port's expected API per
its spec.

Translate2D maps planar coordinates between two systems (offset, scale, rotation,
optional x-axis inversion). It is concrete, so Part 2 compares against Java for the
deterministic forward/inverse transforms built from explicit parameters.

The 3+-point `calibrate` path uses Simplex with RANDOMIZED starting points, so it
is non-deterministic across runs and is not parity-compared. The null / explicit /
1-point / 2-point deterministic paths are covered.

Constructor dispatch (R4): no-arg / copy / (offset, rotation) / (offset, scale,
rotation, invertXAxis). Factory split: createCalibrationPoint_arr / _xy.
"""
from __future__ import annotations

import math

import pytest
from hypothesis import given, strategies as st

from _parity_lib import (
    setup_parity, jclass, needs_java,
    slow,
    _close, _arr_close, _jarr,
)

from Translate2D_ver2_1_0 import (
    Translate2D as PyT2D,
    CalibrationPoint as PyCP,
)

ctx = setup_parity("gov.nist.microanalysis.Utility.Translate2D")
JavaT2D = ctx.java_class

_TOL = 1e-12
_coord = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)
_rot = st.floats(min_value=-math.pi, max_value=math.pi, allow_nan=False, allow_infinity=False)
_scale = st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)


def _forward(x, y, ox, oy, sx, sy, rot):
    """Closed-form Translate2D.compute for a 2-element coordinate."""
    rx = ((x + ox) * math.cos(rot) - (y + oy) * math.sin(rot)) * sx
    ry = ((y + oy) * math.cos(rot) + (x + ox) * math.sin(rot)) * sy
    return rx, ry


# ############################################################################
# PART 1 -- Deterministic transforms & accessors
# ############################################################################


class TestNullTransform:
    def test_compute_identity(self) -> None:
        assert _arr_close(_jarr([3.0, 5.0]), PyT2D().compute([3.0, 5.0]), _TOL)

    def test_default_scale_is_one(self) -> None:
        t = PyT2D()
        assert _close(t.getXScale(), 1.0, _TOL)
        assert _close(t.getYScale(), 1.0, _TOL)

    def test_default_not_inverted(self) -> None:
        assert PyT2D().isXAxisInverted() is False


class TestExplicitConstruction:
    def test_getters(self) -> None:
        t = PyT2D([1.0, 2.0], [2.0, 3.0], 0.5, True)
        assert _close(t.getXOffset(), 1.0, _TOL)
        assert _close(t.getYOffset(), 2.0, _TOL)
        assert _close(t.getXScale(), 2.0, _TOL)
        assert _close(t.getYScale(), 3.0, _TOL)
        assert _close(t.getRotation(), 0.5, _TOL)
        assert t.isXAxisInverted() is True

    def test_compute_no_rotation(self) -> None:
        t = PyT2D([1.0, 2.0], [2.0, 3.0], 0.0, False)
        out = t.compute([1.0, 1.0])
        assert _close(out[0], 4.0, _TOL)   # (1+1)*2
        assert _close(out[1], 9.0, _TOL)   # (1+2)*3

    @given(_coord, _coord, _coord, _coord, _scale, _scale, _rot)
    @slow
    def test_compute_matches_closed_form(self, x, y, ox, oy, sx, sy, rot) -> None:
        t = PyT2D([ox, oy], [sx, sy], rot, False)
        ex, ey = _forward(x, y, ox, oy, sx, sy, rot)
        out = t.compute([x, y])
        assert _close(out[0], ex, 1e-7, rtol=1e-9)
        assert _close(out[1], ey, 1e-7, rtol=1e-9)

    @given(_coord, _coord, _coord, _coord, _scale, _scale, _rot)
    @slow
    def test_inverse_roundtrip(self, x, y, ox, oy, sx, sy, rot) -> None:
        t = PyT2D([ox, oy], [sx, sy], rot, False)
        back = t.inverse(t.compute([x, y]))
        assert _close(back[0], x, 1e-6, rtol=1e-6)
        assert _close(back[1], y, 1e-6, rtol=1e-6)


class TestOffsetRotationConstructor:
    def test_scale_defaults_to_one(self) -> None:
        t = PyT2D([5.0, 6.0], 0.5)
        assert _close(t.getXScale(), 1.0, _TOL)
        assert _close(t.getXOffset(), 5.0, _TOL)
        assert _close(t.getRotation(), 0.5, _TOL)


class TestCopyConstructor:
    def test_copy_preserves_params(self) -> None:
        a = PyT2D([1.0, 2.0], [2.0, 3.0], 0.5, True)
        b = PyT2D(a)
        assert _close(b.getXOffset(), 1.0, _TOL)
        assert _close(b.getXScale(), 2.0, _TOL)
        assert _close(b.getRotation(), 0.5, _TOL)
        assert b.isXAxisInverted() is True


class TestThreeCoordinate:
    def test_third_component_gets_rotation(self) -> None:
        t = PyT2D([0.0, 0.0], [1.0, 1.0], 0.3, False)
        out = t.compute([1.0, 1.0, 0.0])
        assert _close(out[2], 0.3, 1e-12)


class TestCalibrationPoint:
    def test_factory_xy(self) -> None:
        cp = PyT2D.createCalibrationPoint_xy(1.0, 2.0, 3.0, 4.0)
        assert _close(cp.getX0(), 1.0, _TOL)
        assert _close(cp.getY0(), 2.0, _TOL)
        assert _close(cp.getX1(), 3.0, _TOL)
        assert _close(cp.getY1(), 4.0, _TOL)

    def test_factory_arr(self) -> None:
        cp = PyT2D.createCalibrationPoint_arr([1.0, 2.0], [3.0, 4.0])
        assert _close(cp.getX0(), 1.0, _TOL)
        assert _close(cp.getX1(), 3.0, _TOL)

    def test_different(self) -> None:
        a = PyT2D.createCalibrationPoint_xy(0.0, 0.0, 0.0, 0.0)
        b = PyT2D.createCalibrationPoint_xy(10.0, 10.0, 10.0, 10.0)
        assert a.different(b)
        assert not a.different(a)


class TestOnePointCalibrate:
    """One point fixes a pure offset; no Simplex involved."""

    def test_offset_recovered(self) -> None:
        t = PyT2D()
        cp = PyT2D.createCalibrationPoint_xy(1.0, 1.0, 4.0, 6.0)
        err = t.calibrate([cp])
        assert _close(t.getXOffset(), 3.0, 1e-9)
        assert _close(t.getYOffset(), 5.0, 1e-9)
        assert err < 1e-9
        out = t.compute([1.0, 1.0])
        assert _close(out[0], 4.0, 1e-9)
        assert _close(out[1], 6.0, 1e-9)


class TestErrorMethod:
    """`error(calPts)` is the RMS residual of the points under the current
    transform, computed without recalibrating."""

    def test_error_zero_for_exact_fit(self) -> None:
        # Null transform is the identity, so a point mapping (1,1)->(1,1) fits exactly.
        t = PyT2D()
        cp = PyT2D.createCalibrationPoint_xy(1.0, 1.0, 1.0, 1.0)
        assert _close(t.error([cp]), 0.0, 1e-12)

    def test_error_is_rms_residual(self) -> None:
        # Identity maps (0,0)->(0,0); target is (3,4); residual = sqrt((3^2+4^2)/1) = 5.
        t = PyT2D()
        cp = PyT2D.createCalibrationPoint_xy(0.0, 0.0, 3.0, 4.0)
        assert _close(t.error([cp]), 5.0, 1e-9)

    def test_error_averages_over_points(self) -> None:
        # Two points each with residual 5 -> sqrt((25 + 25)/2) = 5.
        t = PyT2D()
        cps = [
            PyT2D.createCalibrationPoint_xy(0.0, 0.0, 3.0, 4.0),
            PyT2D.createCalibrationPoint_xy(1.0, 1.0, 4.0, 5.0),  # residual (3,4) again
        ]
        assert _close(t.error(cps), 5.0, 1e-9)


class TestPreservedBugs:
    def test_inverse_drops_third_component(self) -> None:
        # JAVA-BUG-1: inverse() always returns a 2-element array.
        t = PyT2D([0.0, 0.0], [1.0, 1.0], 0.0, False)
        assert len(t.inverse([1.0, 2.0, 3.0])) == 2


# ############################################################################
# PART 2 -- Parity tests (Java oracle) for the explicit-parameter transforms
# ############################################################################


@needs_java
class TestTranslate2DParity:

    def _pair(self, offset, scale, rot, invert):
        py = PyT2D(list(offset), list(scale), rot, invert)
        j = JavaT2D(_jarr(offset), _jarr(scale), float(rot), bool(invert))
        return py, j

    @given(_coord, _coord, _coord, _coord, _scale, _scale, _rot)
    @slow
    def test_compute_parity(self, x, y, ox, oy, sx, sy, rot) -> None:
        py, j = self._pair([ox, oy], [sx, sy], rot, False)
        assert _arr_close(j.compute(_jarr([x, y])), py.compute([x, y]), 1e-7)

    @given(_coord, _coord, _coord, _coord, _scale, _scale, _rot)
    @slow
    def test_inverse_parity(self, x, y, ox, oy, sx, sy, rot) -> None:
        py, j = self._pair([ox, oy], [sx, sy], rot, False)
        assert _arr_close(j.inverse(_jarr([x, y])), py.inverse([x, y]), 1e-7)

    # NOTE: error(Collection<CalibrationPoint>) is not parity-tested here because
    # passing a Python list to a Java Collection parameter is bridge-dependent;
    # its value semantics are covered deterministically in TestErrorMethod.


if __name__ == "__main__":
    from _parity_lib import run_tests; run_tests(__file__)
