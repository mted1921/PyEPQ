r"""
test_parity_poissondeviate_ver1_1_0.py — parity harness for PoissonDeviate_ver1_1_0.py

PoissonDeviate is a concrete class with its own RNG stream.
  • Constructor: PoissonDeviate(long seed)
  • Method:      randomDeviate(double mean) → long

The implementation has two branches:
  mean < 12  : direct Knuth algorithm (product of uniforms)
  mean >= 12 : rejection-sampling with logarithms / Math2.gammaln

Statistical properties of Poisson(λ):
  E[X]   = λ
  Var[X] = λ
  All values ≥ 0

Reproducibility: two instances with the same seed produce the same sequence.

Java parity: seeded-sequence comparison.
"""
from __future__ import annotations

import math
import statistics

import pytest
from hypothesis import given, settings as hyp_settings, strategies as st

from _parity_lib import (
    setup_parity, needs_java, PARITY_ENABLED,
    TOL_LITERAL,
    _close,
)

from PoissonDeviate import PoissonDeviate as PyPoissonDeviate

ctx = setup_parity("gov.nist.microanalysis.Utility.PoissonDeviate")
JavaPoissonDeviate = ctx.java_class

_N_SAMPLES = 10_000
_STAT_TOL = 0.05   # 5 % relative tolerance on mean/variance (Monte Carlo)
_SEED = 42


# ---------------------------------------------------------------------------
# TestConstruction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_constructs(self):
        pd = PyPoissonDeviate(_SEED)
        assert pd is not None

    def test_different_seeds_exist(self):
        pd1 = PyPoissonDeviate(1)
        pd2 = PyPoissonDeviate(2)
        seq1 = [pd1.randomDeviate(5.0) for _ in range(20)]
        seq2 = [pd2.randomDeviate(5.0) for _ in range(20)]
        assert seq1 != seq2


# ---------------------------------------------------------------------------
# TestNonNegative
# ---------------------------------------------------------------------------

class TestNonNegative:
    def test_small_mean_non_negative(self):
        pd = PyPoissonDeviate(_SEED)
        samples = [pd.randomDeviate(3.0) for _ in range(200)]
        assert all(s >= 0 for s in samples)

    def test_large_mean_non_negative(self):
        pd = PyPoissonDeviate(_SEED)
        samples = [pd.randomDeviate(50.0) for _ in range(200)]
        assert all(s >= 0 for s in samples)

    def test_mean_one_non_negative(self):
        pd = PyPoissonDeviate(_SEED)
        samples = [pd.randomDeviate(1.0) for _ in range(200)]
        assert all(s >= 0 for s in samples)


# ---------------------------------------------------------------------------
# TestStatistical (small mean branch, mean < 12)
# ---------------------------------------------------------------------------

class TestStatisticalSmallMean:
    def _stats(self, lam):
        pd = PyPoissonDeviate(_SEED)
        samples = [pd.randomDeviate(lam) for _ in range(_N_SAMPLES)]
        return statistics.mean(samples), statistics.variance(samples)

    def test_mean_lambda_1(self):
        mu, var = self._stats(1.0)
        assert abs(mu - 1.0) / 1.0 < _STAT_TOL

    def test_mean_lambda_5(self):
        mu, var = self._stats(5.0)
        assert abs(mu - 5.0) / 5.0 < _STAT_TOL

    def test_variance_lambda_1(self):
        mu, var = self._stats(1.0)
        assert abs(var - 1.0) / 1.0 < _STAT_TOL * 2

    def test_variance_lambda_5(self):
        mu, var = self._stats(5.0)
        assert abs(var - 5.0) / 5.0 < _STAT_TOL * 2

    def test_mean_lambda_10(self):
        mu, var = self._stats(10.0)
        assert abs(mu - 10.0) / 10.0 < _STAT_TOL


# ---------------------------------------------------------------------------
# TestStatistical (large mean branch, mean >= 12)
# ---------------------------------------------------------------------------

class TestStatisticalLargeMean:
    def _stats(self, lam):
        pd = PyPoissonDeviate(_SEED)
        samples = [pd.randomDeviate(lam) for _ in range(_N_SAMPLES)]
        return statistics.mean(samples), statistics.variance(samples)

    def test_mean_lambda_12(self):
        mu, _ = self._stats(12.0)
        assert abs(mu - 12.0) / 12.0 < _STAT_TOL

    def test_mean_lambda_50(self):
        mu, _ = self._stats(50.0)
        assert abs(mu - 50.0) / 50.0 < _STAT_TOL

    def test_variance_lambda_20(self):
        mu, var = self._stats(20.0)
        assert abs(var - 20.0) / 20.0 < _STAT_TOL * 2

    def test_mean_lambda_100(self):
        mu, _ = self._stats(100.0)
        assert abs(mu - 100.0) / 100.0 < _STAT_TOL


# ---------------------------------------------------------------------------
# TestReproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_sequence(self):
        pd1 = PyPoissonDeviate(_SEED)
        pd2 = PyPoissonDeviate(_SEED)
        seq1 = [pd1.randomDeviate(7.0) for _ in range(50)]
        seq2 = [pd2.randomDeviate(7.0) for _ in range(50)]
        assert seq1 == seq2

    def test_same_seed_large_mean(self):
        pd1 = PyPoissonDeviate(999)
        pd2 = PyPoissonDeviate(999)
        seq1 = [pd1.randomDeviate(30.0) for _ in range(50)]
        seq2 = [pd2.randomDeviate(30.0) for _ in range(50)]
        assert seq1 == seq2

    def test_mixed_mean_sequence(self):
        pd1 = PyPoissonDeviate(7)
        pd2 = PyPoissonDeviate(7)
        means = [1.0, 5.0, 12.0, 50.0, 3.0, 20.0]
        seq1 = [pd1.randomDeviate(m) for m in means]
        seq2 = [pd2.randomDeviate(m) for m in means]
        assert seq1 == seq2


# ---------------------------------------------------------------------------
# TestParity
# ---------------------------------------------------------------------------

@needs_java
class TestPoissonDeviateParity:
    """Java parity: seeded sequences match exactly."""

    def test_parity_small_mean_sequence(self):
        py_pd = PyPoissonDeviate(_SEED)
        j_pd = JavaPoissonDeviate(_SEED)
        for _ in range(100):
            py_v = py_pd.randomDeviate(5.0)
            j_v = int(j_pd.randomDeviate(5.0))
            assert py_v == j_v

    def test_parity_large_mean_sequence(self):
        py_pd = PyPoissonDeviate(_SEED)
        j_pd = JavaPoissonDeviate(_SEED)
        for _ in range(100):
            py_v = py_pd.randomDeviate(30.0)
            j_v = int(j_pd.randomDeviate(30.0))
            assert py_v == j_v

    def test_parity_mixed_means(self):
        py_pd = PyPoissonDeviate(12345)
        j_pd = JavaPoissonDeviate(12345)
        means = [1.0, 3.0, 6.0, 11.9, 12.0, 12.1, 25.0, 100.0]
        for m in means:
            for _ in range(10):
                assert py_pd.randomDeviate(m) == int(j_pd.randomDeviate(m))

if __name__ == "__main__":
    from _parity_lib import run_tests; run_tests(__file__)
