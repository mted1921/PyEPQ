r"""
ProgressEvent_ver1_1_0.py — Python port of gov.nist.microanalysis.Utility.ProgressEvent

Guide version : 1
Generation    : 1
Port-code fixes: 0

CHANGES:
* DEVIATION-1: java.awt.event.ActionEvent is not available in Python. ProgressEvent 
  inherits from object instead and provides simulated methods for the base class constructor.

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
"""

from __future__ import annotations
import abc, math
import numpy as np
from typing import Any, Optional, Sequence, Union, Callable

try:
    from ._epq_compat import EPQException, F64Array, JavaRandom
except ImportError:
    try:
        from _epq_compat import EPQException, F64Array, JavaRandom  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility._epq_compat import EPQException, F64Array, JavaRandom  # type: ignore

try:
    from .Math2_ver8_1_5 import Math2
except ImportError:
    try:
        from Math2_ver8_1_5 import Math2  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility.Math2_ver8_1_5 import Math2  # type: ignore

__all__ = ["ProgressEvent"]


class ProgressEvent:
    # No Java bugs found in this class.
    BUG_LEDGER: tuple = (
        ("DEVIATION-1", "__init__",
         "java.awt.event.ActionEvent is not available in Python. "
         "ProgressEvent inherits from object instead and simulates the base class fields.", False),
    )

    _serialVersionUID: int = 7157720139752608198

    def __init__(self, src: Any, id: int, progress: int) -> None:
        # DEVIATION-1: Simulate ActionEvent fields since we cannot extend it directly.
        self.source: Any = src
        self.id: int = int(id)
        
        bounded_progress: int = Math2.bound_int(progress, 0, 101)
        self.command: str = str(bounded_progress) + "%"
        
        self.mProgress: int = bounded_progress

    def getProgress(self) -> int:
        return int(self.mProgress)

    # -------------------------------------------------------------------------
    # Simulated java.awt.event.ActionEvent methods
    # -------------------------------------------------------------------------

    def getSource(self) -> Any:
        return self.source

    def getID(self) -> int:
        return int(self.id)

    def getActionCommand(self) -> str:
        return str(self.command)