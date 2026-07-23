r"""
LazyEvaluate_ver2_1_2.py — Python port of gov.nist.microanalysis.Utility.LazyEvaluate

Guide version : 2
Generation    : 1
Port-code fixes: 2

------------------------------------------------------------------------
Original Java source (gov.nist.microanalysis.Utility.LazyEvaluate)
------------------------------------------------------------------------
/**
 * <p>
 * Used to delay the computation of a computationally expensive object until and
 * when necessary. Derived classes compute the object in the function compute().
 * </p>
 *
 * @author Nicholas
 * @version 1.0
 */
------------------------------------------------------------------------

CHANGES:
* `synchronized (this)` -> a per-instance `threading.Lock`; the double-checked
  locking in get() is preserved so _compute() runs at most once.
* The Java `assert mValue != null;` post-condition in get() is omitted (Java
  asserts are disabled by default; not load-bearing).
* `transient` has no Python equivalent; _mValue is a plain attribute.

CHANGES:
* FIX-1 (API-mismatch): renamed abstract method `_compute` → `compute`. The spec
  said `protected` → `_` prefix, but the parity harness defines concrete subclasses
  implementing `compute` (no underscore). The test is authoritative; the abstract
  method name is public so subclasses don't need the `_` guard. `get()` updated
  to call `self.compute()` accordingly.
* FIX-2 (assert-error / API-mismatch): `equals()` now short-circuits on same-instance
  (`self is val`) and compares cached values when `val` is another LazyEvaluate.
  Old implementation used `__eq__` which does Java reference-identity (`is`) on
  the raw argument rather than value equality between caches. Tests are authoritative.
"""

from __future__ import annotations

import abc
import threading
from typing import Generic, Optional, TypeVar

H = TypeVar("H")

__all__ = ["LazyEvaluate"]


class LazyEvaluate(abc.ABC, Generic[H]):

    BUG_LEDGER: tuple = ()  # no bugs identified

    def __init__(self) -> None:
        self._mValue: Optional[H] = None
        self._mLock: threading.Lock = threading.Lock()

    def reset(self) -> None:
        """Clear the cached value; the next get() recomputes via compute()."""
        with self._mLock:
            self._mValue = None

    def evaluated(self) -> bool:
        return self._mValue is not None

    def __eq__(self, val: object) -> bool:
        # Java: return evaluated() && (mValue == val);  -- `==` is reference
        # identity on objects, so use `is`.
        return self.evaluated() and (self._mValue is val)

    def equals(self, val: object) -> bool:  # FIX-2: compare cached values, not raw argument identity
        if self is val:
            return True
        if isinstance(val, LazyEvaluate):
            return self.evaluated() and val.evaluated() and (self._mValue == val._mValue)
        return self.evaluated() and (self._mValue == val)

    def __hash__(self) -> int:
        # Java does not override hashCode -> identity hash. Restore identity
        # hashing (defining __eq__ would otherwise make instances unhashable).
        return object.__hash__(self)

    def get(self) -> H:
        """Return the cached value, computing it once on first access."""
        res: Optional[H] = self._mValue
        if res is None:
            with self._mLock:
                if self._mValue is None:
                    self._mValue = self.compute()  # FIX-1: was self._compute()
                # Java: assert mValue != null;  (omitted — disabled by default)
                res = self._mValue
        return res  # type: ignore[return-value]

    @abc.abstractmethod
    def compute(self) -> H:  # FIX-1: was _compute; test subclasses implement compute()
        """Compute the value returned by get(). Implement in a subclass."""
        ...
