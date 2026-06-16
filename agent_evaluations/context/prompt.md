# EPQ Code Conversion Variance Testing: ver1

You are converting a single Java class to Python. Your only task is to produce the Python port file. Do not write any test or harness files.

Step 1 — Parse the Java source
Extract:

CLASS_NAME — from public [abstract] class <NAME>
IS_ABSTRACT — whether the class declaration includes abstract
All public method signatures, constants, and fields
Java bugs: dead branches, always-true conditions, off-by-one errors, sign errors
Output filename: <CLASS_NAME>_ver1_1_0.py

Step 2 — Write the port
Use Math2_ver8_1_5.py as the style reference throughout.

File header — copy the Java /** ... */ Javadoc verbatim:

r"""
<CLASS_NAME>_ver1_1_0.py — Python port of gov.nist.microanalysis.Utility.<CLASS_NAME>

Guide version : 1
Generation    : 1
Port-code fixes: 0

------------------------------------------------------------------------
Original Java source (gov.nist.microanalysis.Utility.<CLASS_NAME>)
------------------------------------------------------------------------
/**
 * ... verbatim Javadoc from the Java file ...
 */
------------------------------------------------------------------------
"""
Standard imports:

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
Rules (from CONVERSION_GUIDE.md):

R1 — Preserve all Java identifiers verbatim. _-prefix maps to access modifiers only (private/protected). abstract does NOT add _ — public abstract void compute() → compute(), never _compute(). Mapping public abstract to _name will break every subclass that correctly implements name(), causing TypeError: Can't instantiate abstract class.
R2 — For every public mathematical method foo: write foo() (scipy primary) and foo_literal() (line-for-line Java). Abstract methods: only the abstract declaration, no body. For equals/hashCode/toString: map to __eq__/__hash__/__str__ and also expose a named alias equals()/hashCode()/toString() so Java-style call sites (obj.equals(other)) continue to work.
R3 — Import EPQException, JavaRandom, JavaTreeSet, JamaMatrix, F64Array from _epq_compat only — never define local replacements. Use JavaTreeSet wherever Java used java.util.TreeSet; never reimplement it locally.
R4 — Split Java overloads into type-suffixed functions (_vv, _vs, _arr, _scalar, _int, _double).
R5 — Call _require_mutable_f64 before any in-place array mutation.
R6 — Maintain the BUG_LEDGER per BUG_GUIDE.md: quote the exact Java source line at a # JAVA-BUG-N marker, add a tuple entry, and provide an optional *_strict companion. Never infer a bug without citing the Java line. If no bugs exist, write BUG_LEDGER: tuple = () # no bugs identified.
R7 — Every Java / between integer types → Python //. Every Math.round(x) → int(math.floor(x + 0.5)).
R8 — Math.signum(0) → 0.0 if v == 0.0 else math.copysign(1.0, v).
R9 — Annotate every parameter, return type, class field, and non-obvious local variable. Use F64Array for double[]. Explicitly cast int fields returned by double-declared methods: return float(self.mField).
R10 — Document every deliberate deviation with a call-site comment, a CHANGES section in the docstring, and a BUG_LEDGER entry if observable.
Step 3 — Checklist
[ ] Every public Java method has a Python counterpart
[ ] Every public non-abstract mathematical method has both foo() and foo_literal()
[ ] public abstract methods use the un-prefixed Java name (e.g. compute, not _compute)
[ ] equals()/hashCode()/toString() have both a dunder and a named alias
[ ] BUG_LEDGER maintained per BUG_GUIDE.md (exact Java line cited; empty tuple if none)
[ ] All parameters, returns, fields, and non-obvious locals annotated
[ ] Abstract class → abc.ABC; abstract methods → @abc.abstractmethod
[ ] Java Javadoc copied verbatim into module docstring
[ ] Three-tier import fallback used for _epq_compat and any sibling port modules
[ ] Sibling module filenames taken from UTILITY_LEDGER.md "Port file" column — never guessed
[ ] No test code written