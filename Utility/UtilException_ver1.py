"""
UtilException_ver1.py -- Faithful Python port of
gov.nist.microanalysis.Utility.UtilException.

CHANGES
-------
- Constructor overloads split into classmethods (R4):
  from_string, from_string_throwable, from_throwable.
- `serialVersionUID` retained as a class-level constant (R2).
"""
from __future__ import annotations


class UtilException(Exception):
    """Port of gov.nist.microanalysis.Utility.UtilException.

    Non-fatal exception for errors occurring in the Utility library.
    Use `EPQException` (from `_epq_compat`) for EPQLibrary errors; this
    type is intentionally distinct so callers can narrow `except` clauses
    to the Utility layer.
    """

    serialVersionUID: int = 0x1

    def __init__(self, message: str = "") -> None:
        """Java: UtilException() / UtilException(String string) base path."""
        super().__init__(message)

    @classmethod
    def from_string(cls, string: str) -> "UtilException":
        """Java: UtilException(String string)."""
        return cls(string)

    @classmethod
    def from_string_throwable(cls, string: str, throwable: BaseException) -> "UtilException":
        """Java: UtilException(String string, Throwable throwable)."""
        inst: UtilException = cls(string)
        inst.__cause__ = throwable
        return inst

    @classmethod
    def from_throwable(cls, throwable: BaseException) -> "UtilException":
        """Java: UtilException(Throwable throwable).

        Java's Exception(Throwable) uses throwable.toString() as the
        message; Python str(exception) is the equivalent.
        """
        inst: UtilException = cls(str(throwable))
        inst.__cause__ = throwable
        return inst
