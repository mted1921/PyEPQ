r"""
LazyEvaluate_ver1_1_2.py — Python port of gov.nist.microanalysis.Utility.LazyEvaluate

Guide version : 1
Generation    : 1
Port-code fixes: 2

CHANGES:
* FIX-1 (R1): Renamed abstract method _compute → compute. Java declares it
  `public abstract H compute()`, which is a public member and must not receive
  the `_` prefix reserved for private/protected members.
* FIX-2: Added named `equals(val)` method alongside `__eq__` so callers can
  use the Java-style `lz.equals(other)` syntax. Semantics: instance-identity
  short-circuit, then cached-value equality for two LazyEvaluate instances.
* Concurrency: Java `synchronized (this)` replaced with an explicit
  instance-level `threading.Lock()` (`self._lock`), per CONVERSION_GUIDE R10
  cross-cutting patterns.
* NOTE: `None` cannot be used as a cached value — `None` is the "not yet
  computed" sentinel (faithfully mirrors Java's `null` sentinel). Calling
  `get()` on a subclass whose `compute()` returns `None` will raise
  AssertionError, matching Java's infinite-recomputation behavior for a
  `compute()` that returns `null`.

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
"""

from __future__ import annotations
import abc
import threading
from typing import Generic, Optional, TypeVar

try:
    from ._epq_compat import EPQException, F64Array, JavaRandom
except ImportError:
    try:
        from _epq_compat import EPQException, F64Array, JavaRandom  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility._epq_compat import EPQException, F64Array, JavaRandom  # type: ignore

__all__ = ["LazyEvaluate"]

H = TypeVar('H')


class LazyEvaluate(abc.ABC, Generic[H]):

    BUG_LEDGER: tuple = ()  # no Java bugs identified in this class

    def __init__(self) -> None:
        self.mValue: Optional[H] = None
        self._lock: threading.Lock = threading.Lock()

    def reset(self) -> None:
        """Clears the cached value; next get() will re-invoke compute()."""
        with self._lock:
            self.mValue = None

    def evaluated(self) -> bool:
        """Returns True if the cached value has been computed."""
        return self.mValue is not None

    def equals(self, val: object) -> bool:
        """
        Mirrors Java Object.equals() contract.
        Same instance always returns True. Two LazyEvaluate instances are equal
        when both are evaluated and their cached values compare equal. A
        non-LazyEvaluate argument is compared directly to the cached value.
        """
        if val is self:
            return True
        if isinstance(val, LazyEvaluate):
            if not self.evaluated() or not val.evaluated():
                return False
            return bool(self.mValue == val.mValue)
        return self.evaluated() and (self.mValue == val)

    def __eq__(self, val: object) -> bool:
        return self.equals(val)

    def get(self) -> H:
        """
        Returns the cached value, computing it on first call via compute().
        Thread-safe: compute() is called at most once (double-checked locking).
        """
        res: Optional[H] = self.mValue
        if res is None:
            with self._lock:
                if self.mValue is None:
                    self.mValue = self.compute()
                assert self.mValue is not None, (
                    "compute() returned None — None cannot be used as a cached value "
                    "(None is the 'not yet computed' sentinel)"
                )
                res = self.mValue
        return res

    @abc.abstractmethod
    def compute(self) -> H:
        """
        Implement this method to produce the value that get() will cache and return.
        Must not return None (None is the internal 'not computed' sentinel).
        """
        ...
