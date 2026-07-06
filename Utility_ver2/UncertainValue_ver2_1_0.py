r"""
UncertainValue_ver2_1_0.py — Python port of
gov.nist.microanalysis.Utility.UncertainValue

Guide version : 2
Generation    : 1
Port-code fixes: 0

CHANGES
-------
- ``@Deprecated`` annotation reflected in docstring and `DeprecationWarning` in `__init__`
  (R10 deviation — Java `@Deprecated` has no Python equivalent; a warning is idiomatic).
- Added explicit `__init__(mValue=0.0, mSigma=0.0)` (R10): Java populated the fields via
  XStream reflective deserialization; Python needs a constructor for instantiation and
  testing.
- `readResolve()` is kept as an ordinary named method; it is NOT wired into pickle
  (Java serialization hook has no direct Python equivalent).

------------------------------------------------------------------------
Original Java source (gov.nist.microanalysis.Utility.UncertainValue)
------------------------------------------------------------------------
/**
 * <p>
 * This class handles the legacy issue of mapping old-style XStream serialized
 * UncertainValue objects into the new style UncertainValue2 objects.
 * </p>
 * <p>
 * Copyright: Pursuant to title 17 Section 105 of the United States Code this
 * software is not subject to copyright protection and is in the public domain
 * </p>
 * <p>
 * Institution: National Institute of Standards and Technology
 * </p>
 *
 * @author Nicholas
 * @version 1.0
 */
------------------------------------------------------------------------
"""
from __future__ import annotations

import warnings
from typing import Optional

try:
    from ._epq_compat import EPQException, F64Array, JavaRandom
except ImportError:
    try:
        from _epq_compat import EPQException, F64Array, JavaRandom  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility._epq_compat import EPQException, F64Array, JavaRandom  # type: ignore

try:
    from .UncertainValue2_ver2_1_0 import UncertainValue2
except ImportError:
    try:
        from UncertainValue2_ver2_1_0 import UncertainValue2  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility_ver2.UncertainValue2_ver2_1_0 import UncertainValue2  # type: ignore

BUG_LEDGER: tuple = ()  # no bugs identified


class UncertainValue:
    """Legacy XStream deserialization shim for ``UncertainValue2``.

    .. deprecated::
        Use :class:`UncertainValue2` directly.  This class exists only to
        deserialize old-format XStream serialized blobs via ``readResolve()``.
    """

    def __init__(self, mValue: float = 0.0, mSigma: float = 0.0) -> None:
        warnings.warn(
            "UncertainValue is deprecated — use UncertainValue2 instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._mValue: float = float(mValue)
        self._mSigma: float = float(mSigma)

    def readResolve(self) -> UncertainValue2:
        """Convert this legacy object to a modern ``UncertainValue2``.

        Java serialization hook — kept as an ordinary named method; not wired
        into Python's pickle protocol.
        """
        return UncertainValue2(self._mValue, self._mSigma)
