"""
conftest.py — pytest session configuration for PyEPQ Utility parity tests.

Starts the JVM exactly once, at pytest_configure time (before any test file
is collected), when parity prerequisites are met (jpype1 installed, EPQ jar
present).

Without this, each test file calls setup_parity() at module-import time
during pytest's collection phase.  On Python 3.14 + JPype the JVM startup
emits a Windows fatal-exception warning that interrupts collection and
prevents later files from being discovered.

By starting the JVM here — before collection — the warning fires once in
isolation and all test files then find isJVMStarted()==True, skipping the
startJVM call entirely.

Also tees all terminal output to test_output.txt (ANSI codes stripped) so
every test run produces an artefact that can be inspected without a terminal.
No __main__ block is needed in individual test files for this to work.
"""
import re as _re
import sys
from pathlib import Path

import pytest

_ANSI_RE = _re.compile(r'\x1b\[[0-9;]*[mK]')


class _TxtTee:
    """File-like wrapper that writes to *orig* (stdout) and *fh* (txt file).

    ANSI colour codes are stripped before writing to *fh* so the output file
    is clean plain text.  Replaces TerminalReporter._tw._file at session start.
    """

    def __init__(self, orig, fh):
        self._orig = orig
        self._fh = fh

    def write(self, s):
        self._orig.write(s)
        clean = _ANSI_RE.sub('', s) if isinstance(s, str) else ''
        self._fh.write(clean)
        self._fh.flush()

    def flush(self):
        self._orig.flush()
        self._fh.flush()

    def isatty(self):
        return False

    def __getattr__(self, name):
        return getattr(self._orig, name)


def pytest_configure():
    try:
        from _parity_lib import _PARITY_READY, _JAR_PATH, _EXTRA_JARS
    except ImportError:
        return
    if not _PARITY_READY:
        return
    try:
        import jpype
        if not jpype.isJVMStarted():
            jpype.startJVM(
                "--enable-native-access=ALL-UNNAMED",
                classpath=[str(_JAR_PATH), *_EXTRA_JARS],
            )
    except Exception as exc:
        print(
            f"\nconftest.py: JVM startup failed — all @needs_java tests will skip.\n"
            f"  Cause: {exc}\n",
            file=sys.stderr,
        )


@pytest.hookimpl(trylast=True)
def pytest_sessionstart(session):
    """Tee TerminalReporter output to test_output.txt alongside this conftest."""
    out_path = Path(__file__).parent / "test_output.txt"
    try:
        tr = session.config.pluginmanager.get_plugin("terminalreporter")
        if tr is None or not hasattr(tr, '_tw') or not hasattr(tr._tw, '_file'):
            return
        fh = open(out_path, "w", encoding="utf-8")
        session.config._parity_txt_fh = fh
        tr._tw._file = _TxtTee(tr._tw._file, fh)
    except Exception:
        pass  # never abort the test run over txt capture


def pytest_unconfigure(config):
    fh = getattr(config, '_parity_txt_fh', None)
    if fh is not None:
        try:
            fh.close()
        except Exception:
            pass
