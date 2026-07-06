r"""
test_parity_uncertainvalue_ver2_1_1.py -- parity harness for UncertainValue

Pre-written harness (Prompt 2): the port may not exist yet; this file targets its
expected API per UncertainValue_ver1_1_0.spec.md.

UncertainValue is a `@Deprecated` legacy XStream-deserialization shim. The Java
class has two private fields (`mValue`, `mSigma`), no public constructor (fields
were injected by the deserializer), and a single hook `readResolve()` that maps
the legacy object onto a modern `UncertainValue2(mValue, mSigma)`. The port adds an
explicit `__init__(mValue=0.0, mSigma=0.0)` (R10 deviation) so it can be constructed
and tested in Python.

Structure
---------
PART 1  (always-on)
  TestConstruction   default and explicit field values.
  TestReadResolve    readResolve() returns an UncertainValue2 carrying the same
                     value and uncertainty.

PART 2  (parity) -- SKIPPED.
  Java `UncertainValue` exposes no public constructor (fields are set only by the
  XStream deserializer) and no public accessors, so a Java instance cannot be
  built from Python via JPype. Correctness is validated analytically in Part 1
  against the UncertainValue2 mapping contract.
"""
from __future__ import annotations

import math
import pytest
from hypothesis import assume, given, strategies as st

from _parity_lib import needs_java, slow

from UncertainValue_ver2_1_0 import UncertainValue as PyUncertainValue
from UncertainValue2_ver2_1_0 import UncertainValue2 as PyUncertainValue2


_finite = st.floats(allow_nan=False, allow_infinity=False,
                    min_value=-1e6, max_value=1e6)
_nonneg = st.floats(allow_nan=False, allow_infinity=False,
                    min_value=0.0, max_value=1e6)


# ############################################################################
# PART 1 -- Always-on tests
# ############################################################################


class TestConstruction:
    """The Python-only constructor (R10 deviation) populates the two fields."""

    def test_default_fields_zero(self) -> None:
        uv = PyUncertainValue()
        assert uv._mValue == 0.0
        assert uv._mSigma == 0.0

    def test_explicit_fields(self) -> None:
        uv = PyUncertainValue(2.5, 0.4)
        assert uv._mValue == 2.5
        assert uv._mSigma == 0.4


class TestReadResolve:
    """readResolve() maps the legacy object onto UncertainValue2(mValue, mSigma)."""

    def test_returns_uncertain_value2(self) -> None:
        uv = PyUncertainValue(1.0, 0.1)
        res = uv.readResolve()
        assert isinstance(res, PyUncertainValue2)

    def test_value_preserved(self) -> None:
        uv = PyUncertainValue(3.0, 0.5)
        res = uv.readResolve()
        assert res.doubleValue() == 3.0

    def test_uncertainty_preserved(self) -> None:
        uv = PyUncertainValue(3.0, 0.5)
        res = uv.readResolve()
        assert res.uncertainty() == 0.5

    def test_zero_uncertainty(self) -> None:
        uv = PyUncertainValue(7.0, 0.0)
        res = uv.readResolve()
        assert res.doubleValue() == 7.0
        assert res.uncertainty() == 0.0

    @given(_finite, _nonneg)
    @slow
    def test_readResolve_roundtrip(self, value: float, sigma: float) -> None:
        # Test bug fix: uncertainty() = sqrt(sigma^2). For small sigma the roundtrip
        # sqrt(sigma^2) != sigma due to subnormal precision loss (Java is identical).
        # Only test sigma values where the roundtrip is exact.
        assume(sigma == 0.0 or math.sqrt(sigma * sigma) == sigma)
        uv = PyUncertainValue(value, sigma)
        res = uv.readResolve()
        assert res.doubleValue() == value
        assert res.uncertainty() == sigma


# ############################################################################
# PART 2 -- Parity tests
# ############################################################################

@needs_java
@pytest.mark.skip(
    reason="Java UncertainValue has no public constructor (fields set only by the "
           "XStream deserializer) and no public accessors, so a Java instance "
           "cannot be created from Python via JPype. The readResolve() -> "
           "UncertainValue2 mapping is validated analytically in Part 1."
)
class TestUncertainValueParity:
    def test_placeholder(self) -> None:
        pass


if __name__ == "__main__":
    from _parity_lib import run_tests; run_tests(__file__)
