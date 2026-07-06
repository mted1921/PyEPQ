r"""
test_parity_progressevent_ver1_1_0.py — parity harness for ProgressEvent

Java:  ProgressEvent extends java.awt.event.ActionEvent
Python port cannot extend ActionEvent (AWT / GUI framework absent).  The port
stores src, id, and progress; the ActionEvent-specific command string is
computed as  str(Math2.bound(progress, 0, 101)) + "%"  to match Java's
super(src, id, Integer.toString(Math2.bound(progress,0,101)) + "%").

Clamping contract: Math2.bound(x, lo, hi) returns lo ≤ result < hi, so
  Math2.bound(progress, 0, 101) ∈ [0, 100].

Java parity for getProgress() tested with @needs_java.
ActionEvent-specific methods (getSource, getID, getActionCommand) are tested
where they exist in the Python port; skip assertions that don't apply.
"""
from __future__ import annotations

import pytest
from hypothesis import given, strategies as st

from _parity_lib import (
    setup_parity, needs_java, PARITY_ENABLED,
    TOL_EXACT,
    _close,
)

from ProgressEvent_ver2_1_0 import ProgressEvent as PyProgressEvent

ctx = setup_parity("gov.nist.microanalysis.Utility.ProgressEvent")
JavaProgressEvent = ctx.java_class


# ---------------------------------------------------------------------------
# TestConstruction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_basic_construction(self):
        ev = PyProgressEvent("test_source", 1, 50)
        assert ev is not None

    def test_construction_progress_0(self):
        ev = PyProgressEvent("test_source", 0, 0)
        assert ev.getProgress() == 0

    def test_construction_progress_100(self):
        ev = PyProgressEvent("test_source", 0, 100)
        assert ev.getProgress() == 100


# ---------------------------------------------------------------------------
# TestGetProgress
# ---------------------------------------------------------------------------

class TestGetProgress:
    def test_progress_mid_value(self):
        ev = PyProgressEvent("test_source", 1, 50)
        assert ev.getProgress() == 50

    def test_progress_1(self):
        ev = PyProgressEvent("test_source", 1, 1)
        assert ev.getProgress() == 1

    def test_progress_99(self):
        ev = PyProgressEvent("test_source", 1, 99)
        assert ev.getProgress() == 99


# ---------------------------------------------------------------------------
# TestClamping
# ---------------------------------------------------------------------------

class TestClamping:
    """Math2.bound(x, 0, 101) clamps to [0, 100]."""

    def test_negative_clamped_to_0(self):
        ev = PyProgressEvent("test_source", 1, -1)
        assert ev.getProgress() == 0

    def test_very_negative_clamped(self):
        ev = PyProgressEvent("test_source", 1, -999)
        assert ev.getProgress() == 0

    def test_101_clamped_to_100(self):
        ev = PyProgressEvent("test_source", 1, 101)
        assert ev.getProgress() == 100

    def test_500_clamped_to_100(self):
        ev = PyProgressEvent("test_source", 1, 500)
        assert ev.getProgress() == 100

    def test_100_not_clamped(self):
        ev = PyProgressEvent("test_source", 1, 100)
        assert ev.getProgress() == 100

    def test_0_not_clamped(self):
        ev = PyProgressEvent("test_source", 1, 0)
        assert ev.getProgress() == 0


# ---------------------------------------------------------------------------
# TestCommandString
# ---------------------------------------------------------------------------

class TestCommandString:
    """The ActionEvent command string should be '<progress>%'."""

    def test_command_contains_percent(self):
        ev = PyProgressEvent("test_source", 1, 50)
        if hasattr(ev, "getActionCommand"):
            assert "%" in ev.getActionCommand()

    def test_command_contains_value(self):
        ev = PyProgressEvent("test_source", 1, 75)
        if hasattr(ev, "getActionCommand"):
            cmd = ev.getActionCommand()
            assert "75" in cmd

    def test_command_clamped_value(self):
        ev = PyProgressEvent("test_source", 1, 200)
        if hasattr(ev, "getActionCommand"):
            cmd = ev.getActionCommand()
            assert "100" in cmd


# ---------------------------------------------------------------------------
# TestHypothesis
# ---------------------------------------------------------------------------

class TestHypothesis:
    @given(st.integers(-1000, 1000))
    def test_progress_always_in_0_100(self, p):
        ev = PyProgressEvent("test_source", 1, p)
        assert 0 <= ev.getProgress() <= 100

    @given(st.integers(0, 100))
    def test_valid_progress_unmodified(self, p):
        ev = PyProgressEvent("test_source", 1, p)
        assert ev.getProgress() == p


# ---------------------------------------------------------------------------
# TestParity
# ---------------------------------------------------------------------------

@needs_java
class TestProgressEventParity:
    """Java parity for getProgress() — ActionEvent superclass not checked."""

    def test_parity_progress_50(self):
        src = "test_source"
        py_ev = PyProgressEvent(src, 1, 50)
        j_ev = JavaProgressEvent(src, 1, 50)
        assert py_ev.getProgress() == int(j_ev.getProgress())

    def test_parity_progress_0(self):
        src = "test_source"
        py_ev = PyProgressEvent(src, 1, 0)
        j_ev = JavaProgressEvent(src, 1, 0)
        assert py_ev.getProgress() == int(j_ev.getProgress())

    def test_parity_progress_100(self):
        src = "test_source"
        py_ev = PyProgressEvent(src, 1, 100)
        j_ev = JavaProgressEvent(src, 1, 100)
        assert py_ev.getProgress() == int(j_ev.getProgress())

    def test_parity_clamp_negative(self):
        src = "test_source"
        py_ev = PyProgressEvent(src, 1, -5)
        j_ev = JavaProgressEvent(src, 1, -5)
        assert py_ev.getProgress() == int(j_ev.getProgress())

    def test_parity_clamp_high(self):
        src = "test_source"
        py_ev = PyProgressEvent(src, 1, 200)
        j_ev = JavaProgressEvent(src, 1, 200)
        assert py_ev.getProgress() == int(j_ev.getProgress())

    @given(st.integers(-200, 200))
    def test_parity_hypothesis(self, p):
        if not PARITY_ENABLED:
            pytest.skip("Java not available")
        src = "test_source"
        py_ev = PyProgressEvent(src, 1, p)
        j_ev = JavaProgressEvent(src, 1, p)
        assert py_ev.getProgress() == int(j_ev.getProgress())

if __name__ == "__main__":
    from _parity_lib import run_tests; run_tests(__file__)
