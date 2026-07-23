r"""
PoissonDeviate_ver2_1_0.py — Python port of gov.nist.microanalysis.Utility.PoissonDeviate

Guide version : 2
Generation    : 1
Port-code fixes: 0

------------------------------------------------------------------------
Original Java source (gov.nist.microanalysis.Utility.PoissonDeviate)
------------------------------------------------------------------------
/**
 * <p>
 * Calculates a random deviate from the Poisson distribution with a specified
 * mean. This is based on the algorithm in Press et al. Numerical Recipes in C,
 * second edition.
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

CHANGES (from Java):
  R3  — java.util.Random replaced by JavaRandom from _epq_compat; produces the
         same bit-exact sequence as Java's LCG for the same seed.
         (Python's random.Random is statistically equivalent but diverges
         from Java for the same seed — unsuitable for parity tests.)
  R2  — randomDeviate() is the scipy-primary entry point; no scipy substitute
         exists for this algorithm, so it delegates to randomDeviate_literal()
         (# SCIPY-NONE).  randomDeviate_literal() is the line-for-line Java
         translation.
  R7  — Java do-while loops translated as while-True / break (body executes at
         least once).  Math.floor() returns double in Java; float(math.floor())
         preserves that semantics in the large-mean branch.

BUG_LEDGER: tuple = ()  # no bugs identified
"""
from __future__ import annotations

import math

try:
    from ._epq_compat import JavaRandom
except ImportError:
    try:
        from _epq_compat import JavaRandom  # type: ignore[no-redef]
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility_ver2._epq_compat import JavaRandom  # type: ignore[no-redef]

try:
    from .Math2_ver8_1_5 import Math2
except ImportError:
    try:
        from Math2_ver8_1_5 import Math2  # type: ignore[no-redef]
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility_ver2.Math2_ver8_1_5 import Math2  # type: ignore[no-redef]


BUG_LEDGER: tuple = ()  # no bugs identified


class PoissonDeviate:
    """Port of gov.nist.microanalysis.Utility.PoissonDeviate.

    Generates random deviates from the Poisson distribution.  Two branches:
      mean < 12  — Knuth product-of-uniforms method
      mean >= 12 — Lorentzian rejection-sampling (Numerical Recipes)
    """

    def __init__(self, seed: int) -> None:
        self._mRandom: JavaRandom = JavaRandom(seed)
        self._mPrevMean: float = -1.0
        self._mG: float = 0.0
        self._mSqr: float = 0.0
        self._mLogMean: float = 0.0

    def randomDeviate(self, mean: float) -> float:
        # SCIPY-NONE: no library substitute for Java-parity Poisson sampling
        return self.randomDeviate_literal(mean)

    def randomDeviate_literal(self, mean: float) -> float:
        """Line-for-line translation of PoissonDeviate.randomDeviate()."""
        if mean < 12.0:
            if mean != self._mPrevMean:
                self._mPrevMean = mean
                self._mG = math.exp(-mean)
            em: int = -1
            t: float = 1.0
            while True:
                em += 1
                t *= self._mRandom.nextDouble()
                if not (t > self._mG):
                    break
            return float(em)
        else:
            if mean != self._mPrevMean:
                self._mPrevMean = mean
                self._mSqr = math.sqrt(2.0 * mean)
                self._mLogMean = math.log(mean)
                self._mG = (mean * self._mLogMean) - Math2.gammaln(mean + 1.0)
            y: float = 0.0
            em_f: float = 0.0
            t_f: float = 0.0
            while True:
                while True:
                    y = math.tan(math.pi * self._mRandom.nextDouble())
                    em_f = (self._mSqr * y) + mean
                    if not (em_f < 0.0):
                        break
                em_f = float(math.floor(em_f))  # Java Math.floor returns double
                t_f = 0.9 * (1.0 + (y * y)) * math.exp(
                    (em_f * self._mLogMean) - Math2.gammaln(em_f + 1.0) - self._mG
                )
                if not (self._mRandom.nextDouble() > t_f):
                    break
            return em_f
