r"""
test_parity_stagerelocation_ver2_1_0.py -- parity harness for StageRelocation

Pre-written harness (Prompt 2): targets the port's expected API per
its spec.

StageRelocation maps points between two stage coordinate systems (origin shift,
scale, rotation, optional mirror). It is concrete, so Part 2 compares against Java.

Coverage scope
--------------
The null transform and the 1-point fit (origin-only) are exercised here: they are
exact and side-effect free. The 2-point and 3-point fits are intentionally NOT
asserted for specific outputs:
  * the 2-point closed form rejects non-tiny relocations via a distance check whose
    threshold is `0.001 * |x2p - x1p|` (documented JAVA-BUG-1), throwing EPQException;
  * the 3-point path runs a Simplex (Nelder-Mead) minimisation.
Those paths are better validated once Simplex parity is in place; this harness
covers the deterministic transforms and leaves a documented gap for the fits.
"""
from __future__ import annotations

import math

import pytest
from hypothesis import given, strategies as st

from _parity_lib import (
    setup_parity, jclass, needs_java,
    slow,
    _close, _arr_close, _jarr, _to_pylist,
)

from StageRelocation_ver2_1_0 import (
    StageRelocation as PySR,
    RelocatedPoint as PyRP,
)

ctx = setup_parity("gov.nist.microanalysis.Utility.StageRelocation")
JavaSR = ctx.java_class

_TOL = 1e-12
_coord = st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)


# ############################################################################
# PART 1 -- Deterministic transforms
# ############################################################################


class TestNullTransform:
    """Default constructor: origin (0,0), scale (1,1), rotation 0 -> identity."""

    def test_apply_is_identity(self) -> None:
        assert _arr_close(_jarr([3.0, 5.0]), PySR().apply([3.0, 5.0]), _TOL)

    def test_inverse_is_identity(self) -> None:
        assert _arr_close(_jarr([3.0, 5.0]), PySR().inverse([3.0, 5.0]), _TOL)

    def test_not_mirrored(self) -> None:
        assert PySR().isMirrored() is False

    def test_residual_error_zero(self) -> None:
        assert PySR().getResidualError() == 0.0


class TestRelocatedPoint:
    def test_accessors(self) -> None:
        rp = PyRP([1.0, 2.0], [3.0, 4.0])
        assert list(rp.getNativePoint()) == [1.0, 2.0]
        assert list(rp.getRelocatedPoint()) == [3.0, 4.0]

    def test_defensive_copy(self) -> None:
        native = [1.0, 2.0]
        rp = PyRP(native, [3.0, 4.0])
        native[0] = 99.0
        assert list(rp.getNativePoint()) == [1.0, 2.0]   # unaffected by mutation


class TestOnePointFit:
    """A single point fixes a pure translation: origin = relocated - native."""

    def test_maps_native_to_relocated(self) -> None:
        sr = PySR([PyRP([1.0, 1.0], [4.0, 6.0])], False)
        out = sr.apply([1.0, 1.0])
        assert _close(out[0], 4.0, 1e-9)
        assert _close(out[1], 6.0, 1e-9)

    def test_translation_of_other_point(self) -> None:
        # origin = (4-1, 6-1) = (3, 5); apply(pt) = pt + origin.
        sr = PySR([PyRP([1.0, 1.0], [4.0, 6.0])], False)
        out = sr.apply([0.0, 0.0])
        assert _close(out[0], 3.0, 1e-9)
        assert _close(out[1], 5.0, 1e-9)

    @given(_coord, _coord, _coord, _coord, _coord, _coord)
    @slow
    def test_apply_inverse_roundtrip(self, nx, ny, rx, ry, px, py) -> None:
        sr = PySR([PyRP([nx, ny], [rx, ry])], False)
        back = sr.inverse(sr.apply([px, py]))
        assert _close(back[0], px, 1e-7, rtol=1e-7)
        assert _close(back[1], py, 1e-7, rtol=1e-7)


# ############################################################################
# PART 2 -- Parity tests (Java oracle) for the deterministic paths
# ############################################################################


def _java_one_point(native, relocated):
    JRP = jclass("gov.nist.microanalysis.Utility.StageRelocation$RelocatedPoint")
    JArrayList = jclass("java.util.ArrayList")
    lst = JArrayList()
    lst.add(JRP(_jarr(native), _jarr(relocated)))
    return JavaSR(lst, False)


@needs_java
class TestStageRelocationParity:

    @given(_coord, _coord)
    @slow
    def test_null_apply_parity(self, x, y) -> None:
        j = JavaSR().apply(_jarr([x, y]))
        p = PySR().apply([x, y])
        assert _arr_close(j, p, 1e-9)

    @given(_coord, _coord, _coord, _coord, _coord, _coord)
    @slow
    def test_one_point_apply_parity(self, nx, ny, rx, ry, px, py) -> None:
        jsr = _java_one_point([nx, ny], [rx, ry])
        psr = PySR([PyRP([nx, ny], [rx, ry])], False)
        assert _arr_close(jsr.apply(_jarr([px, py])), psr.apply([px, py]), 1e-7)

    @given(_coord, _coord, _coord, _coord, _coord, _coord)
    @slow
    def test_one_point_inverse_parity(self, nx, ny, rx, ry, px, py) -> None:
        jsr = _java_one_point([nx, ny], [rx, ry])
        psr = PySR([PyRP([nx, ny], [rx, ry])], False)
        assert _arr_close(jsr.inverse(_jarr([px, py])), psr.inverse([px, py]), 1e-7)


if __name__ == "__main__":
    from _parity_lib import run_tests; run_tests(__file__)
