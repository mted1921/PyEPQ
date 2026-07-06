r"""
UtilException_ver2_1_1.py — Python port of gov.nist.microanalysis.Utility.UtilException

Guide version : 2
Generation    : 1
Port-code fixes: 1

------------------------------------------------------------------------
Original Java source (gov.nist.microanalysis.Utility.UtilException)
------------------------------------------------------------------------
/**
 * <p>
 * Defines a class for non-fatal exceptions occuring in the utility library.
 * </p>
 * <p>
 * Copyright: Pursuant to title 17 Section 105 of the United States Code this
 * software is not subject to copyright protection and is in the public domain
 * </p>
 * <p>
 * Company: National Institute of Standards and Technology
 * </p>
 *
 * @author Nicholas W. M. Ritchie
 * @version 1.0
 */
------------------------------------------------------------------------

CHANGES:
* `serialVersionUID` is kept as a PUBLIC class attribute (R1 exception), not
  discarded — it participates in the Java serialization protocol by name.
* The four Java constructors collapse into one `__init__` dispatched by argument
  type, following the guide's exception-constructor mapping:
    UtilException()                -> super().__init__("")      (args == ("",))
    UtilException(msg)             -> super().__init__(msg)
    UtilException(msg, cause)      -> super().__init__(msg); __cause__ = cause
    UtilException(cause)           -> super().__init__(str(cause)); __cause__ = cause

CHANGES:
* FIX-1 (R4-violation): added named classmethods `from_string`, `from_string_throwable`,
  `from_throwable` that the parity harness calls for constructor disambiguation. The
  unified `__init__` dispatch is preserved; classmethods delegate to it.
"""

from __future__ import annotations

from typing import Optional, Union

__all__ = ["UtilException"]


class UtilException(Exception):

    BUG_LEDGER: tuple = ()  # no bugs identified

    serialVersionUID: int = 0x1   # R1 exception: public, no underscore

    def __init__(self, message: Union[str, BaseException, None] = None,
                 cause: Optional[BaseException] = None) -> None:
        if isinstance(message, BaseException):
            # UtilException(Throwable): the cause's toString() is the message.
            cause = message
            super().__init__(str(cause))
            self.__cause__ = cause
        elif message is None:
            # UtilException(): Java super() leaves a null message; use "" so
            # str(exc) and exc.args are consistent (("",)).
            super().__init__("")
        else:
            # UtilException(String [, Throwable])
            super().__init__(message)
            if cause is not None:
                self.__cause__ = cause

    # ---- named constructor classmethods (R4 overload-split) ----

    @classmethod
    def from_string(cls, message: str) -> "UtilException":  # FIX-1
        """UtilException(String) — message stored verbatim, no cause."""
        return cls(message)

    @classmethod
    def from_string_throwable(cls, message: str,
                              cause: BaseException) -> "UtilException":  # FIX-1
        """UtilException(String, Throwable) — message + chained cause."""
        return cls(message, cause)

    @classmethod
    def from_throwable(cls, cause: BaseException) -> "UtilException":  # FIX-1
        """UtilException(Throwable) — message is str(cause), cause chained."""
        return cls(cause)
