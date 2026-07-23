r"""
ProgressEvent_ver2_1_0.py — Python port of gov.nist.microanalysis.Utility.ProgressEvent

Guide version : 2
Generation    : 1
Port-code fixes: 0

------------------------------------------------------------------------
Original Java source (gov.nist.microanalysis.Utility.ProgressEvent)
------------------------------------------------------------------------
/**
 * <p>
 * An event representing percentage progress towards a goal.
 * </p>
 * <p>
 * Copyright: Pursuant to title 17 Section 105 of the United States Code this
 * software is not subject to copyright protection and is in the public domain
 * </p>
 * <p>
 * Institution: National Institute of Standards and Technology
 * </p>
 *
 * @author nritchie
 * @version 1.0
 */
------------------------------------------------------------------------

CHANGES (from Java):
  R10-1 — ProgressEvent extends java.awt.event.ActionEvent in Java; no AWT
           equivalent exists in Python.  Ported as a standalone class.
           Simulated ActionEvent fields (_src, _id, _command) are stored as
           instance attributes; getActionCommand() exposes _command for
           API-surface completeness.

JAVA-NOTE-1: Math2.bound(progress, 0, 101) clamps to [0, 100] (upperExc=101).
             Preserved verbatim — may be intentional ("101 = done-plus-one").
"""
from __future__ import annotations

try:
    from .Math2_ver8_1_5 import Math2
except ImportError:
    try:
        from Math2_ver8_1_5 import Math2  # type: ignore[no-redef]
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility_ver2.Math2_ver8_1_5 import Math2  # type: ignore[no-redef]


BUG_LEDGER: tuple = ()  # no bugs identified


class ProgressEvent:
    """Port of gov.nist.microanalysis.Utility.ProgressEvent.

    An event representing percentage progress towards a goal.
    Extends ActionEvent in Java; ported as a standalone class (R10-1 deviation).
    """

    def __init__(self, src: object, id: int, progress: int) -> None:
        clamped: int = Math2.bound(progress, 0, 101)  # JAVA-NOTE-1: upperExc=101
        self._src: object = src
        self._id: int = id
        self._command: str = str(clamped) + "%"
        self._mProgress: int = clamped

    def getProgress(self) -> int:
        return self._mProgress

    def getActionCommand(self) -> str:
        return self._command

    def getSource(self) -> object:
        return self._src

    def getID(self) -> int:
        return self._id
