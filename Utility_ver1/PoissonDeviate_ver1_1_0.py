r"""
PoissonDeviate_ver1_1_0.py — Python port of gov.nist.microanalysis.Utility.PoissonDeviate

Guide version : 1
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
"""

from __future__ import annotations
import abc, math
import numpy as np
from typing import Optional, Sequence, Union, Callable

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

__all__ = ["PoissonDeviate"]


class PoissonDeviate:

    BUG_LEDGER: tuple = ()  # no bugs identified; exact NR algorithm.

    def __init__(self, seed: int) -> None:
        """
        PoissonDeviate - Create a new instance of the PoissonDeviate class. Each
        instance includes its own separate random number generator stream.
        """
        self.mRandom: JavaRandom = JavaRandom()
        self.mRandom.setSeed(int(seed))
        self.mPrevMean: float = -1.0
        self.mG: float = 0.0
        self.mSqr: float = 0.0
        self.mLogMean: float = 0.0

    def randomDeviate(self, mean: float) -> float:
        """
        randomDeviate - Calculate a random deviate taken from the Poisson
        distribution with the specified mean.
        """
        assert mean > 0.0, f"The mean of a random Poisson deviate must be greater than zero. {mean}"
        
        if mean < 12.0:
            if mean != self.mPrevMean:
                self.mPrevMean = mean
                self.mG = math.exp(-mean)
            em: int = -1
            t: float = 1.0
            while True:
                em += 1
                t *= self.mRandom.nextDouble()
                if t <= self.mG:
                    break
            assert em >= 0
            return float(em)
        else:
            if mean != self.mPrevMean:
                self.mPrevMean = mean
                self.mSqr = math.sqrt(2.0 * mean)
                self.mLogMean = math.log(mean)
                self.mG = (mean * self.mLogMean) - Math2.gammaln_literal(mean + 1.0)
            
            y: float = 0.0
            em_f: float = 0.0
            t_val: float = 0.0
            while True:
                while True:
                    y = math.tan(math.pi * self.mRandom.nextDouble())
                    em_f = (self.mSqr * y) + mean
                    if em_f >= 0.0:
                        break
                em_f = float(math.floor(em_f))
                t_val = 0.9 * (1.0 + (y * y)) * math.exp((em_f * self.mLogMean) - Math2.gammaln_literal(em_f + 1.0) - self.mG)
                if self.mRandom.nextDouble() <= t_val:
                    break
            assert em_f >= 0.0
            return float(em_f)