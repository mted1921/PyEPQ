r"""
MCUncertaintyEngine_ver2_1_0.py — Python port of
gov.nist.microanalysis.Utility.MCUncertaintyEngine

Guide version : 2
Generation    : 1
Port-code fixes: 0

CHANGES
-------
- `abstract public UncertainValueMC compute(UncertainValueMC[])` → un-prefixed
  `@abc.abstractmethod def compute(...)` (R1: `public abstract` does NOT add `_`).
- `new TreeMap<String, Double>()` per-iteration deviates map → `{}` (`dict`).
  Python dicts are insertion-ordered; sorted-key semantics are not relied upon
  here (keys are correlation component tags).
- `mResults.iterator().next()` → `self._mResults[0]` (IndexError on empty list,
  matching Java NoSuchElementException — preserve the exception; do not swallow).
- `Number[] arguments` → `list` (accepts `float`, `int`, `UncertainValue2`).
- `DescriptiveStatistics.compute(Collection)` → `compute_collection` (R4 split
  name in the DescriptiveStatistics port).
- `UncertainValue2.asUncertainValue2(Number)` → `UncertainValue2.asUncertainValue2`
  (static method in the UncertainValue2 port; accepts float/int/UncertainValue2).
- Constructor immediately runs `doIterations(iterations)`; this is faithful —
  the Java constructor also runs iterations at construction time.

------------------------------------------------------------------------
Original Java source (gov.nist.microanalysis.Utility.MCUncertaintyEngine)
------------------------------------------------------------------------
/**
 * (no Javadoc in the Java source)
 */
------------------------------------------------------------------------
"""
from __future__ import annotations

import abc
from typing import List

try:
    from ._epq_compat import EPQException, F64Array, JavaRandom
except ImportError:
    try:
        from _epq_compat import EPQException, F64Array, JavaRandom  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility._epq_compat import EPQException, F64Array, JavaRandom  # type: ignore

try:
    from .UncertainValue2_ver2_1_0 import UncertainValue2
except ImportError:
    try:
        from UncertainValue2_ver2_1_0 import UncertainValue2  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility_ver2.UncertainValue2_ver2_1_0 import UncertainValue2  # type: ignore

try:
    from .UncertainValueMC_ver2_1_1 import UncertainValueMC
except ImportError:
    try:
        from UncertainValueMC_ver2_1_1 import UncertainValueMC  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility_ver2.UncertainValueMC_ver2_1_1 import UncertainValueMC  # type: ignore

try:
    from .DescriptiveStatistics_ver2_1_2 import DescriptiveStatistics
except ImportError:
    try:
        from DescriptiveStatistics_ver2_1_2 import DescriptiveStatistics  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility_ver2.DescriptiveStatistics_ver2_1_2 import DescriptiveStatistics  # type: ignore

BUG_LEDGER: tuple = ()  # no bugs identified


class MCUncertaintyEngine(abc.ABC):
    """Python port of ``gov.nist.microanalysis.Utility.MCUncertaintyEngine``.

    Abstract Monte Carlo uncertainty engine.  Subclasses implement
    :meth:`compute` to evaluate the model once per random draw.  The
    constructor immediately runs *iterations* trials by calling
    :meth:`doIterations`, matching the Java behaviour.

    Edge cases
    ----------
    - ``nominalValue()`` raises ``IndexError`` when called with zero iterations
      (Java raises ``NoSuchElementException``); do not swallow.
    - ``getResult()`` / ``getStatistics()`` with zero iterations delegates to
      ``DescriptiveStatistics.compute_collection([])``.
    """

    def __init__(self, iterations: int, arguments: list) -> None:
        self._mResults: List[UncertainValueMC] = []
        self._mArguments: list = arguments  # stored by reference (Java semantics)
        self.doIterations(iterations)

    def doIterations(self, iterations: int) -> None:
        """Run *iterations* additional Monte Carlo trials.

        Each iteration shares one ``rd`` deviates dict across all arguments
        so common uncertainty components are correlated.

        Java: ``public void doIterations(int iterations)``
        """
        for _ in range(iterations):
            rd: dict = {}  # Java: new TreeMap<String, Double>() per iteration
            args: List[UncertainValueMC] = [
                UncertainValueMC(UncertainValue2.asUncertainValue2(self._mArguments[j]), rd)
                for j in range(len(self._mArguments))
            ]
            self._mResults.append(self.compute(args))

    def getResults(self) -> List[UncertainValueMC]:
        """Return the live results list.

        Java: ``public List<UncertainValueMC> getResults()``
        Java returns the field directly (reference semantics); preserved here.
        """
        return self._mResults

    def getStatistics(self) -> DescriptiveStatistics:
        """Return a DescriptiveStatistics over all result samples.

        Java: ``public DescriptiveStatistics getStatistics()``
        Uses ``compute_collection`` (R4 name for Java ``compute(Collection)``).
        """
        return DescriptiveStatistics.compute_collection(self._mResults)

    def nominalValue(self) -> float:
        """Return the nominal value of the first result.

        Java: ``public double nominalValue()`` via ``mResults.iterator().next()``.
        Raises ``IndexError`` when ``_mResults`` is empty (Java raises
        ``NoSuchElementException``).
        """
        return self._mResults[0].nominalValue()

    def getResult(self) -> UncertainValue2:
        """Return mean ± standard-deviation as an UncertainValue2.

        Java: ``public UncertainValue2 getResult()``
        """
        res: DescriptiveStatistics = self.getStatistics()
        return UncertainValue2(res.average(), res.standardDeviation())

    @abc.abstractmethod
    def compute(self, arguments: List[UncertainValueMC]) -> UncertainValueMC:
        """Evaluate the model for one Monte Carlo draw.

        Java: ``abstract public UncertainValueMC compute(UncertainValueMC[] arguments)``
        R1: `public abstract` — method is un-prefixed.
        """
