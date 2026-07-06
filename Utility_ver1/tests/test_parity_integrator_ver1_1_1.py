r"""
test_parity_integrator_ver1_1_1.py — parity harness for Integrator_ver1_1_2.py

Guide version : 1
Generation    : 1
Port-code fixes: 1

CHANGES IN THIS REVISION (ver1_1_1)
-----------------------------------
* Renamed from test_parity_integrator_ver1_1_0.py; header now references the
  actual port under test (Integrator_ver1_1_2.py) per the {G}_{N}_{F} scheme.
* INDIRECT-ARK COVERAGE: the previous suite exercised only Integrator.integrate()
  (which delegates to scipy.integrate.quad and never touches AdaptiveRungeKutta).
  Added test classes that drive Integrator.integrate_literal() and
  integrate_strict(), both of which call super().integrate_literal(...) — i.e.
  the AdaptiveRungeKutta Cash-Karp _baseStep/_qcStep engine — so a fault in the
  ARK dependency now surfaces here through a real consumer:
    - TestIntegrateLiteral   ARK scalar forward stepping vs closed-form integrals.
    - TestIntegrateStrict    ARK forward stepping + signed-limit handling.
    - TestJavaBug1Preserved  JAVA-BUG-1 (reversed limits → 0.0) is preserved in
                             integrate_literal but corrected in integrate /
                             integrate_strict.
    - TestScipyVsLiteral     Cross-validates scipy.quad against the ARK stepper
                             through the consumer (TOL_NR_LIB).
* CANDIDATE INJECTION: set EPQ_ARK_PORT=/path/to/candidate.py (e.g. an
  agent_evaluations EPQ_CC*.py) to load that file's AdaptiveRungeKutta under the
  module name Integrator imports, so this whole suite runs against the
  candidate's stepper instead of the canonical AdaptiveRungeKutta_ver1_1_2.
  Unset (default) → runs against the committed canonical, exactly as before.

COVERAGE NOTE (what this CANNOT reach)
--------------------------------------
Integrator is intrinsically single-variable (super().__init__(1)) and forward-
only (integrate_strict swaps limits before stepping), and it never sets a save
interval. So this suite exercises ARK's scalar, forward, no-save Cash-Karp path
only. Multi-variable stepping, the save-interval machinery, backward integration
(negative h via _sign), and the mMaxSteps/mMinStepSize exception paths remain the
responsibility of test_parity_adaptiverungekutta_ver1_1_0.py.

Integrator is doubly-abstract:
  • extends AdaptiveRungeKutta (abstract)
  • declares abstract getValue(double x)

M4 applies twice over.  Behavioural correctness is validated analytically by
creating Python concrete subclasses.

Analytical benchmarks:
  ∫₀¹ 1   dx  = 1.0
  ∫₀¹ x   dx  = 0.5
  ∫₀ᵖ sin dx  = 2.0   (p = math.pi)
  ∫₀¹ e^x dx  = e − 1 ≈ 1.71828
"""
from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from hypothesis import assume, given, strategies as st

from _parity_lib import (
    setup_parity, needs_java, PARITY_ENABLED,
    TOL_LITERAL, TOL_NR_LIB,
    slow,
    _close,
)


# ---------------------------------------------------------------------------
# Optional candidate injection (must run BEFORE importing Integrator)
# ---------------------------------------------------------------------------
def _maybe_inject_ark_port() -> str:
    """Swap the AdaptiveRungeKutta implementation Integrator imports.

    Integrator_ver1_1_2 resolves its dependency with a bare-name fallback
    (`from AdaptiveRungeKutta_ver1_1_2 import AdaptiveRungeKutta`).  When this
    test file is collected as a top-level module the relative form fails and
    that bare import wins, so pre-seeding sys.modules with a module of that
    name redirects Integrator to whatever file EPQ_ARK_PORT points at.

    Returns a label describing which ARK is under test (for diagnostics).
    """
    port = os.environ.get("EPQ_ARK_PORT")
    if not port:
        return "AdaptiveRungeKutta_ver1_1_2 (canonical)"
    path = Path(port).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"EPQ_ARK_PORT does not exist: {path}")
    spec = importlib.util.spec_from_file_location(
        "AdaptiveRungeKutta_ver1_1_2", str(path)
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load EPQ_ARK_PORT as a module: {path}")
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so any self-referential import resolves to this module.
    sys.modules["AdaptiveRungeKutta_ver1_1_2"] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, "AdaptiveRungeKutta"):
        raise AttributeError(
            f"EPQ_ARK_PORT module defines no AdaptiveRungeKutta class: {path}"
        )
    return f"{path.name} (EPQ_ARK_PORT={path})"


ARK_UNDER_TEST: str = _maybe_inject_ark_port()

from Integrator_ver1_1_2 import Integrator as PyIntegrator  # noqa: E402

ctx = setup_parity("gov.nist.microanalysis.Utility.Integrator")
JavaIntegrator = ctx.java_class

_ITOL = 1e-6   # default ARK tolerance carried over by Integrator()


# ---------------------------------------------------------------------------
# Concrete subclasses
# ---------------------------------------------------------------------------

class _ConstantIntegrator(PyIntegrator):
    """getValue(x) = 1.0  →  ∫ = (high − low)"""
    def getValue(self, x: float) -> float:
        return 1.0


class _LinearIntegrator(PyIntegrator):
    """getValue(x) = x  →  ∫₀¹ = 0.5"""
    def getValue(self, x: float) -> float:
        return x


class _SinIntegrator(PyIntegrator):
    """getValue(x) = sin(x)  →  ∫₀ᵖ = 2.0"""
    def getValue(self, x: float) -> float:
        return math.sin(x)


class _ExpIntegrator(PyIntegrator):
    """getValue(x) = exp(x)  →  ∫₀¹ = e − 1"""
    def getValue(self, x: float) -> float:
        return math.exp(x)


class _ConstantTolIntegrator(PyIntegrator):
    """Constructed with explicit tolerance."""
    def __init__(self, tol: float):
        super().__init__(tol)

    def getValue(self, x: float) -> float:
        return 1.0


# ---------------------------------------------------------------------------
# TestConstruction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_constructor(self):
        ig = _ConstantIntegrator()
        assert ig is not None

    def test_tolerance_constructor(self):
        ig = _ConstantTolIntegrator(1e-8)
        assert ig is not None

    def test_ark_dependency_wired(self):
        """Integrator must actually be an AdaptiveRungeKutta subclass.

        Guards the indirect-coverage premise: if the inheritance ever breaks
        (or a candidate fails to define the class), integrate_literal would not
        reach the ARK stepper and the tests below would be testing nothing.
        """
        from AdaptiveRungeKutta_ver1_1_2 import AdaptiveRungeKutta
        assert issubclass(PyIntegrator, AdaptiveRungeKutta)


# ---------------------------------------------------------------------------
# TestIntegrateConstant  (scipy.quad path — unchanged)
# ---------------------------------------------------------------------------

class TestIntegrateConstant:
    def test_unit_interval(self):
        ig = _ConstantIntegrator()
        result = ig.integrate(0.0, 1.0)
        assert _close(result, 1.0, TOL_NR_LIB)

    def test_wider_interval(self):
        ig = _ConstantIntegrator()
        result = ig.integrate(0.0, 5.0)
        assert _close(result, 5.0, TOL_NR_LIB)

    def test_reversed_limits(self):
        ig = _ConstantIntegrator()
        result = ig.integrate(1.0, 0.0)
        assert _close(abs(result), 1.0, TOL_NR_LIB)

    def test_zero_width_interval(self):
        ig = _ConstantIntegrator()
        result = ig.integrate(3.0, 3.0)
        assert _close(result, 0.0, TOL_NR_LIB)


# ---------------------------------------------------------------------------
# TestIntegrateLinear  (scipy.quad path — unchanged)
# ---------------------------------------------------------------------------

class TestIntegrateLinear:
    def test_zero_to_one(self):
        ig = _LinearIntegrator()
        result = ig.integrate(0.0, 1.0)
        assert _close(result, 0.5, TOL_NR_LIB)

    def test_zero_to_two(self):
        ig = _LinearIntegrator()
        result = ig.integrate(0.0, 2.0)
        assert _close(result, 2.0, TOL_NR_LIB)

    def test_one_to_three(self):
        ig = _LinearIntegrator()
        result = ig.integrate(1.0, 3.0)
        # ∫₁³ x dx = [x²/2]₁³ = 9/2 − 1/2 = 4.0
        assert _close(result, 4.0, TOL_NR_LIB)


# ---------------------------------------------------------------------------
# TestIntegrateSin  (scipy.quad path — unchanged)
# ---------------------------------------------------------------------------

class TestIntegrateSin:
    def test_zero_to_pi(self):
        ig = _SinIntegrator()
        result = ig.integrate(0.0, math.pi)
        assert _close(result, 2.0, TOL_NR_LIB)

    def test_zero_to_half_pi(self):
        ig = _SinIntegrator()
        result = ig.integrate(0.0, math.pi / 2.0)
        assert _close(result, 1.0, TOL_NR_LIB)


# ---------------------------------------------------------------------------
# TestIntegrateExp  (scipy.quad path — unchanged)
# ---------------------------------------------------------------------------

class TestIntegrateExp:
    def test_zero_to_one(self):
        ig = _ExpIntegrator()
        result = ig.integrate(0.0, 1.0)
        assert _close(result, math.e - 1.0, TOL_NR_LIB)


# ---------------------------------------------------------------------------
# TestDerivatives
# ---------------------------------------------------------------------------

class TestDerivatives:
    def test_derivatives_sets_dydx(self):
        """derivatives() must write getValue(x) into dydx[0]."""
        ig = _ConstantIntegrator()
        y = np.zeros(1, dtype=np.float64)
        dydx = np.zeros(1, dtype=np.float64)
        ig.derivatives(0.5, y, dydx)
        assert dydx[0] == 1.0

    def test_derivatives_linear(self):
        ig = _LinearIntegrator()
        y = np.zeros(1, dtype=np.float64)
        dydx = np.zeros(1, dtype=np.float64)
        ig.derivatives(3.0, y, dydx)
        assert dydx[0] == 3.0


# ---------------------------------------------------------------------------
# TestIntegrateLiteral  (exercises the AdaptiveRungeKutta Cash-Karp stepper)
# ---------------------------------------------------------------------------

class TestIntegrateLiteral:
    """integrate_literal() delegates to super().integrate_literal(), i.e. the
    ARK _baseStep/_qcStep engine. These are the suite's only assertions that
    actually run the dependency under test.

    Tolerance: TOL_NR_LIB (1e-4) — the hand-rolled Numerical Recipes stepper at
    the default eps=1e-6, compared against the closed-form integral.
    """

    def test_constant_unit_interval(self):
        ig = _ConstantIntegrator()
        assert _close(ig.integrate_literal(0.0, 1.0), 1.0, TOL_NR_LIB)

    def test_constant_wider_interval(self):
        ig = _ConstantIntegrator()
        assert _close(ig.integrate_literal(0.0, 5.0), 5.0, TOL_NR_LIB)

    def test_linear_zero_to_one(self):
        ig = _LinearIntegrator()
        assert _close(ig.integrate_literal(0.0, 1.0), 0.5, TOL_NR_LIB)

    def test_linear_one_to_three(self):
        ig = _LinearIntegrator()
        assert _close(ig.integrate_literal(1.0, 3.0), 4.0, TOL_NR_LIB)

    def test_sin_zero_to_pi(self):
        ig = _SinIntegrator()
        assert _close(ig.integrate_literal(0.0, math.pi), 2.0, TOL_NR_LIB)

    def test_exp_zero_to_one(self):
        ig = _ExpIntegrator()
        assert _close(ig.integrate_literal(0.0, 1.0), math.e - 1.0, TOL_NR_LIB)

    def test_zero_width_returns_zero(self):
        """high == low: the high>low guard is false → 0.0 (no stepping)."""
        ig = _ConstantIntegrator()
        assert ig.integrate_literal(3.0, 3.0) == 0.0

    def test_no_nan_on_success(self):
        """integrate_literal returns NaN if ARK raises UtilException; a finite
        result confirms the stepper completed without underflow/too-many-steps."""
        ig = _ExpIntegrator()
        assert math.isfinite(ig.integrate_literal(0.0, 2.0))


# ---------------------------------------------------------------------------
# TestIntegrateStrict  (ARK stepper + corrected signed-limit handling)
# ---------------------------------------------------------------------------

class TestIntegrateStrict:
    """integrate_strict() also routes forward stepping through ARK, and
    additionally fixes JAVA-BUG-1 by negating the reversed-limit result."""

    def test_forward_constant(self):
        ig = _ConstantIntegrator()
        assert _close(ig.integrate_strict(0.0, 2.0), 2.0, TOL_NR_LIB)

    def test_forward_linear(self):
        ig = _LinearIntegrator()
        assert _close(ig.integrate_strict(1.0, 3.0), 4.0, TOL_NR_LIB)

    def test_reversed_constant_is_negative(self):
        ig = _ConstantIntegrator()
        assert _close(ig.integrate_strict(1.0, 0.0), -1.0, TOL_NR_LIB)

    def test_reversed_linear_is_negative(self):
        ig = _LinearIntegrator()
        # ∫₃¹ x dx = −(∫₁³ x dx) = −4.0
        assert _close(ig.integrate_strict(3.0, 1.0), -4.0, TOL_NR_LIB)

    def test_zero_width_returns_zero(self):
        ig = _ConstantIntegrator()
        assert ig.integrate_strict(4.0, 4.0) == 0.0


# ---------------------------------------------------------------------------
# TestJavaBug1Preserved  (literal keeps the bug; scipy/strict correct it)
# ---------------------------------------------------------------------------

class TestJavaBug1Preserved:
    """BUG_LEDGER JAVA-BUG-1: Java returns 0.0 for reversed limits instead of
    the negative integral. The literal port preserves this; the scipy primary
    and integrate_strict correct it."""

    def test_literal_reversed_returns_zero(self):
        """Preserved bug: integrate_literal(high<low) → 0.0, not −integral."""
        ig = _ConstantIntegrator()
        assert ig.integrate_literal(1.0, 0.0) == 0.0

    def test_scipy_reversed_is_signed(self):
        """scipy.quad computes the signed integral (bug corrected)."""
        ig = _ConstantIntegrator()
        assert _close(ig.integrate(1.0, 0.0), -1.0, TOL_NR_LIB)

    def test_strict_reversed_is_signed(self):
        """integrate_strict negates the reversed-limit result (bug corrected)."""
        ig = _ConstantIntegrator()
        assert _close(ig.integrate_strict(1.0, 0.0), -1.0, TOL_NR_LIB)


# ---------------------------------------------------------------------------
# TestScipyVsLiteral  (cross-validate scipy.quad against the ARK stepper)
# ---------------------------------------------------------------------------

class TestScipyVsLiteral:
    """The two independent integration paths — scipy.quad (integrate) and the
    hand-rolled ARK Cash-Karp stepper (integrate_literal) — must agree on
    forward integrals to TOL_NR_LIB. Divergence localises a fault to the ARK
    dependency (the literal side)."""

    @pytest.mark.parametrize(
        "factory,lo,hi",
        [
            (_ConstantIntegrator, 0.0, 1.0),
            (_ConstantIntegrator, 0.0, 5.0),
            (_LinearIntegrator, 0.0, 2.0),
            (_LinearIntegrator, 1.0, 3.0),
            (_SinIntegrator, 0.0, math.pi),
            (_ExpIntegrator, 0.0, 1.0),
        ],
    )
    def test_paths_agree(self, factory, lo, hi):
        sci = factory().integrate(lo, hi)
        lit = factory().integrate_literal(lo, hi)
        assert _close(sci, lit, TOL_NR_LIB)


# ---------------------------------------------------------------------------
# TestHypothesis
# ---------------------------------------------------------------------------

class TestHypothesis:
    @given(st.floats(0.0, 10.0), st.floats(0.0, 10.0))
    @slow
    def test_constant_integral_equals_interval_length(self, a, b):
        ig = _ConstantIntegrator()
        result = ig.integrate(a, b)
        assert _close(result, b - a, 1e-4)  # FIX-1: signed integral; abs() was wrong

    @given(st.floats(0.0, 10.0), st.floats(0.0, 10.0))
    @slow
    def test_literal_constant_forward(self, a, b):
        """Forward-only (a<b) so the JAVA-BUG-1 guard is not taken; exercises
        the ARK stepper across random forward intervals."""
        lo, hi = (a, b) if a < b else (b, a)
        assume(hi - lo > 0.1)
        ig = _ConstantIntegrator()
        assert _close(ig.integrate_literal(lo, hi), hi - lo, 1e-4)

    @given(st.floats(0.1, 5.0), st.floats(0.1, 5.0))
    @slow
    def test_scipy_vs_literal_linear(self, a, b):
        """scipy and the ARK stepper agree on random forward linear integrals."""
        lo, hi = (a, b) if a < b else (b, a)
        assume(hi - lo > 0.1)
        assert _close(
            _LinearIntegrator().integrate(lo, hi),
            _LinearIntegrator().integrate_literal(lo, hi),
            TOL_NR_LIB,
        )


# ---------------------------------------------------------------------------
# TestIntegratorParity  (M4 — doubly abstract)
# ---------------------------------------------------------------------------

@pytest.mark.skip(
    reason="M4: Integrator extends AdaptiveRungeKutta (abstract) and itself "
           "declares abstract getValue(). JPype cannot extend abstract Java "
           "classes from Python. Correctness validated analytically above."
)
class TestIntegratorParity:
    def test_placeholder(self):
        pass


if __name__ == "__main__":
    from _parity_lib import run_tests; run_tests(__file__)
