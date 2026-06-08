r"""
test_parity_utilexception.py -- parity harness for UtilException_ver1.py

Structure
---------
PART 1  (always-on)
  TestConstruction      Construction semantics for all four overload paths:
                          * no-arg: empty message, no cause
                          * from_string: message stored verbatim
                          * from_string_throwable: message + __cause__
                          * from_throwable: message == str(cause), __cause__ set

  TestInheritance       Exception hierarchy and class-level constants:
                          * UtilException is a subclass of Exception
                          * UtilException is distinct from EPQException
                          * serialVersionUID == 0x1

  TestRaiseable         Can be raised and caught through the exception hierarchy.

  TestHypothesis        from_string roundtrip under arbitrary text strings.

PART 2  (parity, requires EPQ_PARITY=1 + jpype1 + EPQ jar)
  TestUtilExceptionParity
                        UtilException is a concrete class; JPype can instantiate
                        it directly (no abstract-class workaround needed).
                        Tests compare Java getMessage() against Python str().

                        NOTE: Java's no-arg constructor calls super() which
                        stores null as the message (getMessage() returns null).
                        Python's __init__ always calls super().__init__(message)
                        with the default "" so str(exc) returns "".  This
                        null-vs-empty-string difference is preserved behaviour,
                        NOT a port bug; the test documents and asserts it.
"""
from __future__ import annotations

import sys

import pytest
from hypothesis import given, strategies as st

from _parity_lib import (
    setup_parity, needs_java, PARITY_ENABLED,
    slow,
)

from UtilException import UtilException as PyUtilException
from _epq_compat import EPQException

ctx = setup_parity("gov.nist.microanalysis.Utility.UtilException")
JavaUtilException = ctx.java_class


# ############################################################################
# PART 1 -- Always-on tests
# ############################################################################


class TestConstruction:
    """All four constructor overload paths."""

    # ------------------------------------------------------------------
    # no-arg / __init__
    # ------------------------------------------------------------------

    def test_no_arg_message_is_empty(self) -> None:
        exc = PyUtilException()
        assert str(exc) == ""

    def test_no_arg_args_tuple(self) -> None:
        exc = PyUtilException()
        assert exc.args == ("",)

    def test_no_arg_cause_is_none(self) -> None:
        exc = PyUtilException()
        assert exc.__cause__ is None

    # ------------------------------------------------------------------
    # from_string
    # ------------------------------------------------------------------

    def test_from_string_stores_message(self) -> None:
        exc = PyUtilException.from_string("hello world")
        assert str(exc) == "hello world"

    def test_from_string_empty_string(self) -> None:
        exc = PyUtilException.from_string("")
        assert str(exc) == ""

    def test_from_string_no_cause(self) -> None:
        exc = PyUtilException.from_string("msg")
        assert exc.__cause__ is None

    def test_from_string_returns_utilexception(self) -> None:
        exc = PyUtilException.from_string("msg")
        assert isinstance(exc, PyUtilException)

    # ------------------------------------------------------------------
    # from_string_throwable
    # ------------------------------------------------------------------

    def test_from_string_throwable_message(self) -> None:
        cause = ValueError("original error")
        exc = PyUtilException.from_string_throwable("wrapped", cause)
        assert str(exc) == "wrapped"

    def test_from_string_throwable_sets_cause(self) -> None:
        cause = ValueError("original error")
        exc = PyUtilException.from_string_throwable("wrapped", cause)
        assert exc.__cause__ is cause

    def test_from_string_throwable_returns_utilexception(self) -> None:
        exc = PyUtilException.from_string_throwable("msg", RuntimeError("x"))
        assert isinstance(exc, PyUtilException)

    def test_from_string_throwable_cause_can_be_any_base_exception(self) -> None:
        cause = KeyboardInterrupt("interrupted")
        exc = PyUtilException.from_string_throwable("msg", cause)
        assert exc.__cause__ is cause

    # ------------------------------------------------------------------
    # from_throwable
    # ------------------------------------------------------------------

    def test_from_throwable_message_is_str_of_cause(self) -> None:
        cause = ValueError("the original")
        exc = PyUtilException.from_throwable(cause)
        assert str(exc) == str(cause)

    def test_from_throwable_sets_cause(self) -> None:
        cause = RuntimeError("boom")
        exc = PyUtilException.from_throwable(cause)
        assert exc.__cause__ is cause

    def test_from_throwable_returns_utilexception(self) -> None:
        exc = PyUtilException.from_throwable(OSError("oops"))
        assert isinstance(exc, PyUtilException)

    def test_from_throwable_with_empty_message_cause(self) -> None:
        cause = ValueError("")
        exc = PyUtilException.from_throwable(cause)
        assert str(exc) == ""

    def test_from_throwable_message_matches_str_of_exception(self) -> None:
        cause = TypeError("type mismatch: expected int")
        exc = PyUtilException.from_throwable(cause)
        assert str(exc) == "type mismatch: expected int"


class TestInheritance:
    """Exception hierarchy and class-level constants."""

    def test_is_exception_subclass(self) -> None:
        assert issubclass(PyUtilException, Exception)

    def test_instance_is_exception(self) -> None:
        exc = PyUtilException("test")
        assert isinstance(exc, Exception)

    def test_is_base_exception(self) -> None:
        exc = PyUtilException("test")
        assert isinstance(exc, BaseException)

    def test_is_distinct_from_epq_exception(self) -> None:
        assert not issubclass(PyUtilException, EPQException)
        assert not issubclass(EPQException, PyUtilException)

    def test_serial_version_uid(self) -> None:
        assert PyUtilException.serialVersionUID == 0x1

    def test_serial_version_uid_is_int(self) -> None:
        assert isinstance(PyUtilException.serialVersionUID, int)


class TestRaiseable:
    """Exception can be raised and caught through the hierarchy."""

    def test_raise_and_catch_as_utilexception(self) -> None:
        with pytest.raises(PyUtilException):
            raise PyUtilException("error")

    def test_raise_and_catch_as_exception(self) -> None:
        with pytest.raises(Exception):
            raise PyUtilException("error")

    def test_raise_from_string_and_catch(self) -> None:
        with pytest.raises(PyUtilException, match="specific message"):
            raise PyUtilException.from_string("specific message")

    def test_raise_from_string_throwable_cause_accessible(self) -> None:
        cause = ValueError("root cause")
        try:
            raise PyUtilException.from_string_throwable("outer", cause)
        except PyUtilException as exc:
            assert exc.__cause__ is cause

    def test_raise_from_throwable_cause_accessible(self) -> None:
        cause = OSError("io failure")
        try:
            raise PyUtilException.from_throwable(cause)
        except PyUtilException as exc:
            assert exc.__cause__ is cause

    def test_from_string_throwable_not_caught_as_value_error(self) -> None:
        """UtilException wrapping a ValueError is NOT itself a ValueError."""
        with pytest.raises(PyUtilException):
            try:
                raise PyUtilException.from_string_throwable(
                    "wrapped", ValueError("inner")
                )
            except ValueError:
                pass  # should NOT be caught here


class TestHypothesis:
    """Property-based roundtrip under arbitrary text messages."""

    @given(st.text(max_size=200))
    @slow
    def test_from_string_message_roundtrip(self, msg: str) -> None:
        exc = PyUtilException.from_string(msg)
        assert str(exc) == msg

    @given(st.text(max_size=200))
    @slow
    def test_init_message_roundtrip(self, msg: str) -> None:
        exc = PyUtilException(msg)
        assert str(exc) == msg

    @given(st.text(max_size=200), st.text(max_size=200))
    @slow
    def test_from_string_throwable_message_roundtrip(
        self, msg: str, cause_msg: str
    ) -> None:
        cause = RuntimeError(cause_msg)
        exc = PyUtilException.from_string_throwable(msg, cause)
        assert str(exc) == msg
        assert exc.__cause__ is cause

    @given(st.text(max_size=200))
    @slow
    def test_from_throwable_message_is_str_of_cause(
        self, cause_msg: str
    ) -> None:
        cause = ValueError(cause_msg)
        exc = PyUtilException.from_throwable(cause)
        assert str(exc) == cause_msg


# ############################################################################
# PART 2 -- Parity tests (require JVM + EPQ.jar + EPQ_PARITY=1)
# ############################################################################

@needs_java
class TestUtilExceptionParity:
    """Compare Java UtilException against Python UtilException_ver1.

    UtilException is a concrete class so JPype can instantiate it directly;
    no abstract-class workaround is needed.

    Null-vs-empty divergence: Java's no-arg constructor inherits
    Exception() which stores null internally -- getMessage() returns null.
    Python's __init__ calls super().__init__("") so str() returns "".
    This is preserved behaviour; see test_no_arg_message_divergence.
    """

    def test_no_arg_message_divergence(self) -> None:
        """Java no-arg getMessage() is null; Python no-arg str() is "".
        This is a documented preserved divergence, not a bug."""
        j = JavaUtilException()
        assert j.getMessage() is None          # Java stores null
        p = PyUtilException()
        assert str(p) == ""                    # Python stores empty string

    @given(st.text(min_size=1, max_size=200,
                   alphabet=st.characters(blacklist_categories=("Cs",))))
    @slow
    def test_string_message_parity(self, msg: str) -> None:
        """Java getMessage() matches Python str() for non-empty messages."""
        j = JavaUtilException(msg)
        p = PyUtilException.from_string(msg)
        assert str(j.getMessage()) == str(p)

    def test_string_message_known_values(self) -> None:
        """Deterministic regression for known message strings."""
        for msg in ("hello", "error in computation", "value out of range"):
            j = JavaUtilException(msg)
            p = PyUtilException.from_string(msg)
            assert str(j.getMessage()) == str(p)

    def test_java_class_is_throwable(self) -> None:
        """Java UtilException is a java.lang.Throwable."""
        import jpype
        Throwable = jpype.JClass("java.lang.Throwable")
        j = JavaUtilException("test")
        assert isinstance(j, Throwable)

    def test_java_class_is_exception(self) -> None:
        """Java UtilException is a java.lang.Exception."""
        import jpype
        JavaException = jpype.JClass("java.lang.Exception")
        j = JavaUtilException("test")
        assert isinstance(j, JavaException)

    def test_python_not_java_type(self) -> None:
        """Python UtilException is NOT a JPype Java type."""
        import jpype
        Throwable = jpype.JClass("java.lang.Throwable")
        p = PyUtilException("test")
        assert not isinstance(p, Throwable)

    def test_serial_version_uid_matches(self) -> None:
        """serialVersionUID constant matches between Java and Python."""
        field = JavaUtilException.class_.getDeclaredField("serialVersionUID")
        field.setAccessible(True)
        j_uid = int(field.getLong(None))
        assert j_uid == PyUtilException.serialVersionUID

    def test_java_string_throwable_message(self) -> None:
        """Java UtilException(String, Throwable).getMessage() == string."""
        inner = JavaUtilException("inner cause")
        outer = JavaUtilException("outer message", inner)
        assert str(outer.getMessage()) == "outer message"
        p = PyUtilException.from_string_throwable(
            "outer message", RuntimeError("inner cause")
        )
        assert str(p) == "outer message"


# ############################################################################
# Entry point: tee pytest output to test_output.txt
# ############################################################################

if __name__ == "__main__":
    from _parity_lib import run_tests; run_tests(__file__)
