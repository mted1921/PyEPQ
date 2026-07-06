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
import importlib.util
import re as _re
import sys
from pathlib import Path

import pytest

_ANSI_RE = _re.compile(r'\x1b\[[0-9;]*[mK]')

# Top-level `import X` / `from X import ...` statements in a test file.
_IMPORT_RE = _re.compile(r'^(?:from|import)\s+(\w+)', _re.MULTILINE)


def pytest_addoption(parser):
    """--port-file / --port-name: inject a candidate file as a port module.

    Allows any .py file to be tested through a parity harness without moving
    files or editing imports. The candidate is loaded and registered in
    sys.modules under *port_name* before collection, so every
    `from <port_name> import …` in every test file and every port module that
    those tests import resolves to the candidate (direct AND indirect testing).

    Example — direct test:
        pytest test_parity_adaptiverungekutta_ver1_1_0.py \\
            --port-file=../../agent_evaluations/candidates/Claude/EPQ_CC1.py \\
            --port-name=AdaptiveRungeKutta_ver1_1_2

    Example — indirect test (Integrator depends on ARK):
        pytest test_parity_integrator_ver1_1_1.py \\
            --port-file=../../agent_evaluations/candidates/Claude/EPQ_CC1.py \\
            --port-name=AdaptiveRungeKutta_ver1_1_2
    """
    parser.addoption(
        "--port-file",
        metavar="PATH",
        default=None,
        help="Candidate .py file to inject as the port module under test.",
    )
    parser.addoption(
        "--port-name",
        metavar="NAME",
        default=None,
        help="Module name to register the candidate under "
             "(default: stem of --port-file).",
    )


def pytest_ignore_collect(collection_path, config):
    """Skip pre-written parity tests whose port module doesn't exist yet.

    Harnesses are sometimes written BEFORE the port is generated (the
    test file is part of the conversion spec). Until the port module
    lands, importing the test file raises ImportError at collection,
    aborting the whole-suite run. Instead: detect the missing port
    module and ignore the file, reporting it as pending on stderr.
    """
    p = Path(str(collection_path))
    if not (p.name.startswith("test_parity_") and p.suffix == ".py"):
        return None
    try:
        src = p.read_text(encoding="utf-8")
    except OSError:
        return None
    port_dir = Path(__file__).resolve().parent.parent
    missing: list[str] = []
    for name in dict.fromkeys(_IMPORT_RE.findall(src)):
        if name.startswith("_"):       # _parity_lib, _epq_compat, __future__
            continue
        if name in sys.modules:        # already injected via --port-file
            continue
        if (port_dir / f"{name}.py").is_file():
            continue
        try:
            if importlib.util.find_spec(name) is not None:
                continue               # stdlib or installed package
        except (ImportError, ValueError):
            pass
        missing.append(name)
    if missing:
        sys.stderr.write(
            f"conftest: SKIPPING {p.name} — pending pre-written harness; "
            f"port module(s) not yet generated: {', '.join(missing)}\n"
        )
        return True
    return None


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


def pytest_configure(config):
    # Ensure _parity_lib is importable when pytest is invoked from the repo
    # root rather than from inside the tests/ directory.
    _tests_dir = str(Path(__file__).resolve().parent)
    if _tests_dir not in sys.path:
        sys.path.insert(0, _tests_dir)

    # Inject candidate file into sys.modules before collection so that every
    # `from <port_name> import …` in test files AND in port modules that those
    # tests import resolves to the candidate (enables direct + indirect testing).
    port_file = config.getoption("--port-file", default=None)
    if port_file:
        p = Path(port_file).resolve()
        name = config.getoption("--port-name", default=None) or p.stem
        spec = importlib.util.spec_from_file_location(name, p)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception as exc:
            sys.stderr.write(
                f"\nconftest: --port-file '{p.name}' failed to load — "
                f"tests will error.\n  Cause: {exc}\n"
            )

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
