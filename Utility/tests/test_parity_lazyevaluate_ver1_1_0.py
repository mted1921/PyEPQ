r"""
test_parity_lazyevaluate_ver1_1_0.py — parity harness for LazyEvaluate_ver1_1_0.py

LazyEvaluate<H> is a generic abstract class (M4: JPype cannot extend abstract Java classes).
Part 1: behavioural correctness via Python concrete subclasses.
Part 2: Java parity — @pytest.mark.skip(M4).

Key invariants:
  - compute() is called at most once between resets
  - evaluated() reflects whether the cache is populated
  - reset() clears the cache (next get() re-invokes compute())
  - equals() delegates to the cached value's equals()
"""
from __future__ import annotations

import threading
from typing import Generic, TypeVar

import pytest
from hypothesis import given, strategies as st

from _parity_lib import (
    setup_parity, needs_java, PARITY_ENABLED,
    TOL_EXACT,
    slow,
    _close,
)

from LazyEvaluate_ver1_1_2 import LazyEvaluate as PyLazyEvaluate

ctx = setup_parity("gov.nist.microanalysis.Utility.LazyEvaluate")
JavaLazyEvaluate = ctx.java_class


# ---------------------------------------------------------------------------
# Concrete subclasses for testing
# ---------------------------------------------------------------------------

class _ConstantLazy(PyLazyEvaluate):
    """Returns a fixed value; counts calls to compute()."""
    def __init__(self, value):
        super().__init__()
        self._value = value
        self.call_count = 0

    def compute(self):
        self.call_count += 1
        return self._value


class _MutableStateLazy(PyLazyEvaluate):
    """Each compute() returns the next integer from a counter."""
    def __init__(self):
        super().__init__()
        self._counter = 0

    def compute(self):
        self._counter += 1
        return self._counter


class _ListLazy(PyLazyEvaluate):
    """Returns a list — tests that equality check uses list.__eq__."""
    def __init__(self, lst):
        super().__init__()
        self._lst = lst

    def compute(self):
        return self._lst


# ---------------------------------------------------------------------------
# TestInitialState
# ---------------------------------------------------------------------------

class TestInitialState:
    def test_not_evaluated_initially(self):
        lz = _ConstantLazy(42)
        assert not lz.evaluated()

    def test_compute_not_called_on_construction(self):
        lz = _ConstantLazy(42)
        assert lz.call_count == 0


# ---------------------------------------------------------------------------
# TestGet
# ---------------------------------------------------------------------------

class TestGet:
    def test_get_returns_computed_value(self):
        lz = _ConstantLazy(99)
        assert lz.get() == 99

    def test_get_calls_compute_once(self):
        lz = _ConstantLazy(7)
        lz.get()
        lz.get()
        lz.get()
        assert lz.call_count == 1

    def test_evaluated_true_after_get(self):
        lz = _ConstantLazy(1)
        lz.get()
        assert lz.evaluated()

    def test_float_value(self):
        lz = _ConstantLazy(3.14)
        assert lz.get() == 3.14

    def test_string_value(self):
        lz = _ConstantLazy("hello")
        assert lz.get() == "hello"


# ---------------------------------------------------------------------------
# TestReset
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_evaluated(self):
        lz = _ConstantLazy(5)
        lz.get()
        lz.reset()
        assert not lz.evaluated()

    def test_get_after_reset_calls_compute_again(self):
        lz = _ConstantLazy(5)
        lz.get()
        lz.reset()
        lz.get()
        assert lz.call_count == 2

    def test_mutable_state_increments_on_reset(self):
        lz = _MutableStateLazy()
        assert lz.get() == 1
        lz.reset()
        assert lz.get() == 2

    def test_double_reset_safe(self):
        lz = _ConstantLazy(0)
        lz.reset()
        lz.reset()
        assert not lz.evaluated()

    def test_reset_then_get_then_reset_cycle(self):
        lz = _ConstantLazy("x")
        for _ in range(5):
            assert lz.get() == "x"
            lz.reset()
        assert lz.call_count == 5


# ---------------------------------------------------------------------------
# TestEquals
# ---------------------------------------------------------------------------

class TestEquals:
    def test_equals_same_instance(self):
        lz = _ConstantLazy(42)
        assert lz.equals(lz)

    def test_equals_uses_cached_value(self):
        lz1 = _ConstantLazy([1, 2, 3])
        lz2 = _ListLazy([1, 2, 3])
        lz1.get()
        lz2.get()
        # Both caches hold equal lists
        assert lz1.equals(lz2)

    def test_equals_different_values(self):
        lz1 = _ConstantLazy(1)
        lz2 = _ConstantLazy(2)
        lz1.get(); lz2.get()
        assert not lz1.equals(lz2)

    def test_equals_unevaluated_vs_evaluated(self):
        lz1 = _ConstantLazy(0)
        lz2 = _ConstantLazy(0)
        lz2.get()
        # Java semantics: unevaluated compared to evaluated — value vs null
        # Port must handle gracefully (either False or True, not crash)
        lz1.equals(lz2)  # no exception


# ---------------------------------------------------------------------------
# TestConcurrency
# ---------------------------------------------------------------------------

class TestConcurrency:
    def test_compute_called_once_under_concurrent_get(self):
        """Double-checked locking must ensure compute() runs exactly once."""
        lz = _ConstantLazy(100)
        results = []

        def _getter():
            results.append(lz.get())

        threads = [threading.Thread(target=_getter) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(r == 100 for r in results)
        assert lz.call_count == 1


# ---------------------------------------------------------------------------
# TestHypothesis
# ---------------------------------------------------------------------------

class TestHypothesis:
    @given(st.integers(1, 50))
    @slow
    def test_repeated_get_always_returns_same_value(self, n):
        """get() must return the same cached value regardless of call count."""
        lz = _ConstantLazy(99)
        results = [lz.get() for _ in range(n)]
        assert all(r == 99 for r in results)
        assert lz.call_count == 1

    @given(st.integers(1, 20))
    @slow
    def test_reset_cycles_recompute_exactly_once_each(self, n_cycles):
        """Each reset-then-get cycle must invoke compute() exactly once."""
        lz = _ConstantLazy(7)
        for _ in range(n_cycles):
            lz.get()
            lz.reset()
        assert lz.call_count == n_cycles


# ---------------------------------------------------------------------------
# TestLazyEvaluateParity  (M4 — abstract class)
# ---------------------------------------------------------------------------

@pytest.mark.skip(
    reason="M4: JPype cannot extend Java abstract classes from Python. "
           "LazyEvaluate.compute() is abstract; no Python callback into JVM. "
           "Behavioural correctness validated analytically above."
)
class TestLazyEvaluateParity:
    def test_placeholder(self):
        pass

if __name__ == "__main__":
    from _parity_lib import run_tests; run_tests(__file__)
