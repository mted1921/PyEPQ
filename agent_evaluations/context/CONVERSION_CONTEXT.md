# AdaptiveRungeKutta.java

```java
package gov.nist.microanalysis.Utility;

/**
 * <p>
 * An adaptive step size Runge-Kutta algorithm for numerically evaluating
 * differential equations. This implementation can optionally save intermediate
 * points along the ODE trajectory at a user specified interval. Using this
 * option may limit the step size and thus
 * </p>
 * <p>
 * See Press, Teulolsky, Vetterling &amp; Flannery, Numerical Recipes in C,
 * Second Edition pp 714-722
 * </p>
 * <p>
 * Example:<br>
 * </p>
 * 
 * <pre>
 * AdaptiveRungeKutta trial = new AdaptiveRungeKutta(2) {
 *    void derivatives(double x, double[] y, double[] dydx) {
 *       dydx[0] = -Math.sin(x);
 *       dydx[1] = Math.cos(x);
 *    }
 * };
 * </pre>
 * 
 * <pre>
 * try {
 *    double[] yst = {1.0, 0.0};
 *    trial.setSaveInterval(Math.PI / 16.0);
 *    trial.integrate(0.0, 2.0 * Math.PI, yst, 1.0e-6, 0.01);
 * } catch (UtilException ex) {
 *    System.err.println(ex.toString());
 * }
 * for (int i = 0; i &lt; trial.getNSaved(); ++i)
 *    System.out.println(trial.getX(i) + &quot;\t&quot; + trial.getY(i)[0] + &quot;\t&quot; + trial.getY(i)[1]);
 * </pre>
 * <p>
 * NOTE: This algorithm is not thread-safe. Use each instance in one and only
 * one thread.
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

abstract public class AdaptiveRungeKutta {
   private final int mNVariables; // The number of differential equations
   private double mHDid; // Actual step size accomplished in last call to
   // qcStep
   private double mHNext; // Next step size to try when calling qcStep
   private double mSaveInterval = Double.MAX_VALUE;
   private double mMinStepSize = 0.0;
   private double[] mXSave;
   private double[][] mYSave;
   private int mNSaved = 0;
   private int mMaxSteps = 10000;
   private int mNOk; // Number of ok steps
   private int mNBad; // Number of repeated steps
   // Temporary work space used by baseStep
   private double[] mWs2, mWs3, mWs4, mWs5, mWs6, mYTemp;
   // Temporary work space used by qcStep
   private double[] mYErr, mQcYTemp;

   private double sign(double magnitude, double sign) {
      return sign >= 0.0 ? Math.abs(magnitude) : -Math.abs(magnitude);
   }

   /**
    * baseStep - Take a single Cash-Karp Runge-Kutta step. Given the
    * n=mNDimensions values y[0..n-1] and their derivatives dydx[0..n-1] know at
    * x, use a fifth order Cash-Karp Runge-Kutta method to advance the solution
    * over an interval h. The resulting y value is returned in yout. An estimate
    * of the truncation error is returned in yerr.
    * 
    * @param x
    *           double
    * @param y
    *           double[]
    * @param dydx
    *           double[]
    * @param h
    *           double
    * @param yout
    *           double[]
    */
   private void baseStep(double x, double[] y, double[] dydx, double h, double[] yout, double[] yerr) {
      final double a2 = 0.2, a3 = 0.3, a4 = 0.6, a5 = 1.0, a6 = 0.875;
      final double b21 = 0.2;
      final double b31 = 3.0 / 40.0, b32 = 9.0 / 40.0;
      final double b41 = 0.3, b42 = -0.9, b43 = 1.2;
      final double b51 = -11.0 / 54.0, b52 = 2.5, b53 = -70.0 / 27.0, b54 = 35.0 / 27.0;
      final double b61 = 1631.0 / 55296.0, b62 = 175.0 / 512.0, b63 = 575.0 / 13824.0, b64 = 44275.0 / 110592.0, b65 = 253.0 / 4096.0;
      final double c1 = 37.0 / 378.0, c3 = 250.0 / 621.0, c4 = 125.0 / 594.0, c6 = 512.0 / 1771.0;
      final double dc1 = c1 - (2825.0 / 27648.0), dc3 = c3 - (18575.0 / 48384.0), dc4 = c4 - (13525.0 / 55296.0), dc5 = -277.0 / 14336.0,
            dc6 = c6 - 0.25;
      // Workspace
      if (mWs2 == null) {
         mWs2 = new double[mNVariables];
         mWs3 = new double[mNVariables];
         mWs4 = new double[mNVariables];
         mWs5 = new double[mNVariables];
         mWs6 = new double[mNVariables];
         mYTemp = new double[mNVariables];
      }
      // First step
      for (int i = 0; i < mNVariables; ++i)
         mYTemp[i] = y[i] + (b21 * h * dydx[i]);
      // Second step
      derivatives(x + (a2 * h), mYTemp, mWs2);
      for (int i = 0; i < mNVariables; ++i)
         mYTemp[i] = y[i] + (h * ((b31 * dydx[i]) + (b32 * mWs2[i])));
      // Third step
      derivatives(x + (a3 * h), mYTemp, mWs3);
      for (int i = 0; i < mNVariables; ++i)
         mYTemp[i] = y[i] + (h * ((b41 * dydx[i]) + (b42 * mWs2[i]) + (b43 * mWs3[i])));
      // Fourth step
      derivatives(x + (a4 * h), mYTemp, mWs4);
      for (int i = 0; i < mNVariables; ++i)
         mYTemp[i] = y[i] + (h * ((b51 * dydx[i]) + (b52 * mWs2[i]) + (b53 * mWs3[i]) + (b54 * mWs4[i])));
      // Fifth step
      derivatives(x + (a5 * h), mYTemp, mWs5);
      for (int i = 0; i < mNVariables; ++i)
         mYTemp[i] = y[i] + (h * ((b61 * dydx[i]) + (b62 * mWs2[i]) + (b63 * mWs3[i]) + (b64 * mWs4[i]) + (b65 * mWs5[i])));
      // Sixth step
      derivatives(x + (a6 * h), mYTemp, mWs6);
      for (int i = 0; i < mNVariables; ++i)
         yout[i] = y[i] + (h * ((c1 * dydx[i]) + (c3 * mWs3[i]) + (c4 * mWs4[i]) + (c6 * mWs6[i])));
      // Estimate the error
      for (int i = 0; i < mNVariables; ++i)
         yerr[i] = h * ((dc1 * dydx[i]) + (dc3 * mWs3[i]) + (dc4 * mWs4[i]) + (dc5 * mWs5[i]) + (dc6 * mWs6[i]));
   }

   /**
    * qcStep - Take a fifth order Runge-Kutta step with monitoring of local
    * truncation error. Input are the dependent variable y[0..mNDimensions-1]
    * and its derivatives dydx[0..mNDimensions-1] at the starting value of the
    * independent variable x. Also input is the attempted step size htry, the
    * required accuracy eps and the vector yscal against which the errors are
    * scaled. Upon return, y is replaced with the new values, x is returned and
    * mHDid and mHNext are set to the actual step size and the size of the next
    * step to try.
    * 
    * @param x
    *           double - (In) independent variable
    * @param y
    *           double[] - (In,Out) dependent variable
    * @param dydx
    *           double[] - (In) derivative at x
    * @param htry
    *           double - The step size to attempt
    * @param eps
    *           double - Desired accuracy
    * @param yscal
    *           double[] - (In) Error scaling vector
    * @throws UtilException
    *            - When the step size becomes too small
    * @return double - The new value of x
    */
   private double qcStep(double x, double[] y, double[] dydx, double htry, double eps, double[] yscal) throws UtilException {
      final double safety = 0.9;
      final double pgrow = -0.2;
      final double pshrnk = -0.25;
      final double errcon = 1.89e-4;
      if (mYErr == null) {
         mYErr = new double[mNVariables];
         mQcYTemp = new double[mNVariables];
      }
      double errmax, h = htry;
      do {
         baseStep(x, y, dydx, h, mQcYTemp, mYErr);
         errmax = 0.0;
         for (int i = 0; i < mNVariables; ++i)
            errmax = Math.max(errmax, Math.abs(mYErr[i] / yscal[i]));
         errmax /= eps;
         if (errmax > 1.0) {
            final double htemp = safety * h * Math.pow(errmax, pshrnk);
            h = (h >= 0 ? Math.max(htemp, 0.1 * h) : Math.min(htemp, 0.1 * h));
            // Check for step size underflow
            final double xnew = x + h;
            if (xnew == x)
               throw new UtilException("Step size underflow in AdaptiveRungeKutta.qcStep.");
         }
      } while (errmax > 1.0);
      mHNext = (errmax > errcon ? safety * h * Math.pow(errmax, pgrow) : 5.0 * h);
      mHDid = h;
      x += h;
      System.arraycopy(mQcYTemp, 0, y, 0, mNVariables);
      return x;
   }

   /**
    * clearWorkspace - null all temporary space to free memory
    */
   private void clearWorkspace() {
      mWs2 = null;
      mWs3 = null;
      mWs4 = null;
      mWs5 = null;
      mWs6 = null;
      mYTemp = null;
      mYErr = null;
      mQcYTemp = null;
   }

   /**
    * AdaptiveRungeKutta - Construct an AdaptiveRungeKutta object to solve a
    * differential equation of nVars variables. The implementation of
    * derivatives should return nVars derivative values for each x &amp; y.
    * 
    * @param nVars
    *           int
    */
   public AdaptiveRungeKutta(int nVars) {
      super();
      mNVariables = nVars;
   }

   /**
    * setSaveInterval - Set the interval on which to save intermediate points on
    * the integrated trajectory. (Use clearSaveInterval to not save any
    * intermediate points.) Note: The default is not to save any intermediate
    * points.
    * 
    * @param interval
    *           double
    */
   public void setSaveInterval(double interval) {
      mSaveInterval = Math.abs(interval);
   }

   /**
    * clearSaveInterval - Return to the default of not saving any intermediate
    * points.
    */
   public void clearSaveInterval() {
      mSaveInterval = Double.MAX_VALUE;
   }

   /**
    * getNSaved - Returns the number of saved values.
    * 
    * @return int
    */
   public int getNSaved() {
      return mNSaved;
   }

   /**
    * getX - Returns the x-coordinate of the i-th saved value
    * 
    * @param i
    *           int - Where i&lt;getNSaved()
    * @return double
    */
   public double getX(int i) {
      return mXSave[i];
   }

   /**
    * getY - returns the getNVariable x y-coordinates of the i-th saved values.
    * 
    * @param i
    *           int - Where i&lt;getNSaved()
    * @return double[] - Of dimension getNVariables
    */
   public double[] getY(int i) {
      return mYSave[i];
   }

   /**
    * setMaxSteps - Set the maximum number of ODE steps to allow. Default is
    * 10000.
    * 
    * @param maxSteps
    *           int
    */
   public void setMaxSteps(int maxSteps) {
      mMaxSteps = maxSteps;
   }

   /**
    * setMinStepSize - Sets the minimum permissible step size. Default is 0.0.
    * 
    * @param minStep
    *           double
    */
   public void setMinStepSize(double minStep) {
      mMinStepSize = Math.abs(minStep);
   }

   /**
    * getNVariables - Returns the number of variables as set in the constructor.
    * 
    * @return int
    */
   public int getNVariables() {
      return mNVariables;
   }

   /**
    * getStepCount - Get the total number of steps required to perform the
    * previous integrate operation.
    * 
    * @return int
    */
   public int getStepCount() {
      return mNOk + mNBad;
   }

   /**
    * getGoodStepCount - Get the number of steps leading to results of the
    * desired accuracy.
    * 
    * @return int
    */
   public int getGoodStepCount() {
      return mNOk;
   }

   /**
    * getBadStepCount - Get the number of steps that were needed to be
    * subdivided to attain results of the desired accuracy.
    * 
    * @return int
    */
   public int getBadStepCount() {
      return mNBad;
   }

   /**
    * integrate - Integrate the ODE specified by derivatives using the adaptive
    * step size Runge-Kutta algorithm over the independent variable interval x1
    * to x2. ystart contains the initial y values. eps is measure of the
    * permissible error. h is the initial step size.
    * 
    * @param x1
    *           double - Start of the integration range
    * @param x2
    *           double - End of the integration range
    * @param ystart
    *           double[] - (In &amp; out) The initial y value
    * @param eps
    *           double - The permissible relative error
    * @param h1
    *           double - The initial step size
    * @return The final y values as an array of length getNVariables().
    * @throws UtilException
    *            - Upon too many steps or too small a step
    */
   public double[] integrate(double x1, double x2, double[] ystart, double eps, double h1) throws UtilException {
      final double tiny = 1.0e-10 * eps;
      final double[] yscal = new double[mNVariables];
      final double[] dydx = new double[mNVariables];
      final double[] y = new double[mNVariables];
      double x = x1;
      double h = sign(h1, x2 - x1);
      double xsav = 0.0;
      double saveInt = Double.MAX_VALUE;
      int kMax = 0;
      mNSaved = 0;
      mNOk = 0;
      mNBad = 0;
      System.arraycopy(ystart, 0, y, 0, mNVariables);
      if (mSaveInterval != Double.MAX_VALUE) {
         kMax = (int) Math.round((Math.abs(x2 - x1) + mSaveInterval) / mSaveInterval);
         saveInt = sign(mSaveInterval, x2 - x1);
         xsav = x - (2.0 * saveInt); // to ensure that the first step is
                                     // saved...
         mXSave = new double[kMax];
         mYSave = new double[kMax][mNVariables];
      }
      for (int step = 0; step < mMaxSteps; ++step) {
         // Save the necessary points
         if ((kMax != 0) && (mNSaved < kMax) && (Math.abs(x - xsav) >= (0.9999 * mSaveInterval))) {
            mXSave[mNSaved] = x;
            System.arraycopy(y, 0, mYSave[mNSaved], 0, mNVariables);
            xsav = x;
            ++mNSaved;
         }
         derivatives(x, y, dydx);
         // Rescale h to ensure we hit desired points
         final double hMax = Math.abs((xsav + saveInt) - x);
         if (Math.abs(h) > hMax)
            h = sign(hMax, h);
         // Scaling to monitor accuracy...
         for (int i = 0; i < mNVariables; ++i)
            yscal[i] = Math.abs(y[i]) + Math.abs(dydx[i] * h) + tiny;
         if ((((x + h) - x2) * ((x + h) - x1)) > 0.0)
            h = x2 - x;
         x = qcStep(x, y, dydx, h, eps, yscal);
         if (mHDid == h)
            ++mNOk;
         else
            ++mNBad;
         if (((x - x2) * (x2 - x1)) >= 0.0) {
            System.arraycopy(y, 0, ystart, 0, mNVariables);
            if (kMax != 0) {
               mNSaved = Math.min(mNSaved, kMax - 1);
               mXSave[mNSaved] = x;
               System.arraycopy(y, 0, mYSave[mNSaved], 0, mNVariables);
               ++mNSaved;
            }
            clearWorkspace();
            return y;
         }
         if (Math.abs(mHNext) <= mMinStepSize)
            throw new UtilException("Step size too small in AdaptiveRungeKutta.integrate");
         h = mHNext;
      }
      throw new UtilException("Too many steps in AdaptiveRungeKutta.integrate");
   }

   /**
    * derivatives - The derived class provides an implementation of the
    * derivatives function. x &amp; y[] are input and the user provided
    * implementation of derivatives is resposible for returning the derivatives
    * in the array dydx. The lengths of y and dydx are equal to mNDimensions.
    * 
    * @param x
    *           double - In
    * @param y
    *           double[] - In (of dimension mNDimensions)
    * @param dydx
    *           double[] - Out (of dimension mNDimensions)
    */
   abstract public void derivatives(double x, double[] y, double[] dydx);

}
```

# _epq_compat.py

```python
"""
_epq_compat.py
==============

Single source of truth for cross-module types in the EPQ Python port.

Every converted module MUST import these from here. Defining local
stand-in classes (as the initial Math2_ver1 revision did) creates
incompatible types: ``except EPQException`` written against module A's
stand-in does not catch ``raise EPQException()`` from module B.

Exported types
--------------
EPQException  : Exception subclass, port of the Java EPQException.
JavaRandom    : Bit-exact reimplementation of java.util.Random.
JavaTreeSet   : Sorted-set replacement for java.util.TreeSet.
JamaMatrix    : Minimal numpy-backed shim for Jama.Matrix.
F64Array      : Type alias for NDArray[np.float64].

This file has no project-internal dependencies; it is safe to import
from any other converted module without circular-import risk.
"""

from __future__ import annotations

import bisect
import math
import time
from typing import Any, Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray


__all__ = ["EPQException", "JavaRandom", "JavaTreeSet", "JamaMatrix", "F64Array"]


F64Array = NDArray[np.float64]


# ======================================================================
# EPQException
# ======================================================================

class EPQException(Exception):
    """Port of gov.nist.microanalysis.EPQLibrary.EPQException.

    This is the single source of truth for the EPQException type during
    the migration. When the dedicated EPQException port lands, replace
    the body of this class but keep the import path stable so call sites
    do not need to change.
    """
    pass


# ======================================================================
# JavaRandom
# ======================================================================

class JavaRandom:
    """Bit-exact reimplementation of java.util.Random.

    A given seed produces the same sequence as ``new Random(seed)`` on
    the JVM. Required for:
      * Parity testing the Python port against reference Java outputs.
      * Reproducing Monte Carlo runs cited in publications.
      * Any test that asserts specific RNG-dependent values.

    Python's ``random.Random`` (Mersenne Twister) is statistically
    excellent but produces a *different* stream for the same seed --
    unsuitable as a drop-in for parity work.

    Algorithm (per JDK source):
        scramble:  s' = (seed ^ MULT) & MASK
        step:      s' = (s * MULT + INCR) & MASK
        next(b):   top b bits of s', sign-extended when b == 32.

    Public method names follow Java conventions so call sites translate
    verbatim; ``random()`` is provided as a Python-convention alias for
    ``nextDouble()``.
    """

    _MULTIPLIER: int = 0x5DEECE66D
    _INCREMENT: int = 0xB
    _MASK: int = (1 << 48) - 1

    def __init__(self, seed: Optional[int] = None) -> None:
        if seed is None:
            # Java default: nanoTime XOR'd with a "uniquifier". For
            # parity-critical work, always pass an explicit seed.
            seed = time.time_ns()
        self._seed: int = (int(seed) ^ self._MULTIPLIER) & self._MASK
        self._have_next_gaussian: bool = False
        self._next_gaussian: float = 0.0

    def setSeed(self, seed: int) -> None:
        self._seed = (int(seed) ^ self._MULTIPLIER) & self._MASK
        self._have_next_gaussian = False

    def _next(self, bits: int) -> int:
        """Internal LCG step returning the top `bits` bits.

        Java's protected ``next(int)`` returns a *signed* 32-bit int.
        We return the unsigned value and let public methods sign-extend
        as needed (see ``nextInt`` no-arg form).
        """
        self._seed = (self._seed * self._MULTIPLIER + self._INCREMENT) & self._MASK
        return self._seed >> (48 - bits)

    def nextDouble(self) -> float:
        # Java: ((next(26) << 27) + next(27)) / 2^53
        return ((self._next(26) << 27) + self._next(27)) / (1 << 53)

    def random(self) -> float:
        """Python-convention alias for nextDouble()."""
        return self.nextDouble()

    def nextFloat(self) -> float:
        return self._next(24) / (1 << 24)

    def nextInt(self, bound: Optional[int] = None) -> int:
        """One-arg form: uniform in [0, bound).
        No-arg form: uniform signed 32-bit int.

        The bounded form implements Java's exact rejection-sampling
        algorithm including the power-of-two fast path -- this affects
        which seeds map to which outputs.
        """
        if bound is None:
            r = self._next(32)
            return r - (1 << 32) if r >= (1 << 31) else r  # sign-extend
        if bound <= 0:
            raise ValueError("bound must be positive")
        m = bound - 1
        if (bound & m) == 0:  # power of two
            return (bound * self._next(31)) >> 31
        while True:
            r = self._next(31)
            v = r % bound
            if r - v + m >= 0:
                return v

    def nextLong(self) -> int:
        r = (self._next(32) << 32) + self._next(32)
        return r - (1 << 64) if r >= (1 << 63) else r  # sign-extend

    def nextBoolean(self) -> bool:
        return self._next(1) != 0

    def nextGaussian(self) -> float:
        """Marsaglia polar method, matching Java's java.util.Random."""
        if self._have_next_gaussian:
            self._have_next_gaussian = False
            return self._next_gaussian
        while True:
            v1 = 2.0 * self.nextDouble() - 1.0
            v2 = 2.0 * self.nextDouble() - 1.0
            s = v1 * v1 + v2 * v2
            if 0.0 < s < 1.0:
                break
        mult = math.sqrt(-2.0 * math.log(s) / s)
        self._next_gaussian = v2 * mult
        self._have_next_gaussian = True
        return v1 * mult


# ======================================================================
# JavaTreeSet
# ======================================================================

class JavaTreeSet:
    """Replacement for java.util.TreeSet<T> where T implements Comparable.

    Provides the exact subset of TreeSet's API used by EPQ converted classes:
    - ``add(item)``   — insert maintaining sorted order; returns False if duplicate
    - ``floor(item)`` — greatest element ≤ item, or None
    - ``__iter__``    — iterate in ascending order
    - ``__len__``     — element count

    **Requirements on stored elements:**
    Elements must implement both:
    - ``__lt__`` (used by bisect for insertion position)
    - ``compareTo(other) -> int`` (returns -1 / 0 / +1, used for equality in add/floor)

    These are the Python equivalents of Java's ``Comparable<T>`` contract. Any
    correctly ported Java class that ``implements Comparable`` will satisfy them
    via the R2 ``compareTo`` → ``__lt__``/``compareTo()`` mapping.

    Unlike Python's ``SortedList`` (sortedcontainers), this class requires no
    third-party dependency and implements floor() directly, matching Java's
    NavigableSet.floor() semantics.
    """

    def __init__(self) -> None:
        self._list: list[Any] = []

    def add(self, item: Any) -> bool:
        """Insert item in sorted order. Returns False (no-op) if already present."""
        idx: int = bisect.bisect_left(self._list, item)
        if idx < len(self._list) and self._list[idx].compareTo(item) == 0:
            return False
        self._list.insert(idx, item)
        return True

    def floor(self, item: Any) -> Optional[Any]:
        """Return the greatest element ≤ item, or None if no such element exists."""
        idx: int = bisect.bisect_right(self._list, item)
        if idx == 0:
            return None
        return self._list[idx - 1]

    def __iter__(self):
        return iter(self._list)

    def __len__(self) -> int:
        return len(self._list)

    def __repr__(self) -> str:
        return f"JavaTreeSet({self._list!r})"


# ======================================================================
# JamaMatrix
# ======================================================================

class JamaMatrix:
    """Minimal Jama-compatible matrix wrapper around numpy.

    Implements the subset of Jama's API actually used by EPQ. Exotic
    decompositions (SVD, eig, Cholesky) intentionally are NOT wrapped --
    use scipy.linalg directly; the result types are well-documented and
    there is no value in re-boxing them.

    Storage is numpy float64. ``getArray()`` exposes the underlying
    ndarray for native interop (caller mutations propagate to the
    matrix). Use ``getArrayCopy()`` if isolation is required.
    """

    def __init__(self, data: ArrayLike) -> None:
        arr = np.asarray(data, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            raise ValueError("JamaMatrix requires 1-D or 2-D input")
        self._A: NDArray[np.float64] = arr

    # ---- factory methods (Jama's alternate constructors) ----

    @classmethod
    def from_flat(cls, vals: ArrayLike, m: int) -> "JamaMatrix":
        """Java: ``new Matrix(double[] vals, int m)`` -- column-packed flat array."""
        arr = np.asarray(vals, dtype=np.float64)
        n = arr.shape[0] // m
        return cls(arr.reshape(n, m).T)

    @classmethod
    def zeros(cls, m: int, n: int) -> "JamaMatrix":
        return cls(np.zeros((m, n), dtype=np.float64))

    @classmethod
    def filled(cls, m: int, n: int, s: float) -> "JamaMatrix":
        return cls(np.full((m, n), float(s), dtype=np.float64))

    @classmethod
    def identity(cls, m: int, n: Optional[int] = None) -> "JamaMatrix":
        return cls(np.eye(m, n if n is not None else m, dtype=np.float64))

    @classmethod
    def random(cls, m: int, n: int,
               rng: Optional["JavaRandom"] = None) -> "JamaMatrix":
        """Jama's random matrix. Pass a JavaRandom for seed parity."""
        if rng is None:
            return cls(np.random.rand(m, n))
        out = np.empty((m, n), dtype=np.float64)
        for i in range(m):
            for j in range(n):
                out[i, j] = rng.nextDouble()
        return cls(out)

    # ---- size and access ----

    def getRowDimension(self) -> int:
        return self._A.shape[0]

    def getColumnDimension(self) -> int:
        return self._A.shape[1]

    def get(self, i: int, j: int) -> float:
        return float(self._A[i, j])

    def set(self, i: int, j: int, val: float) -> None:
        self._A[i, j] = val

    def getArray(self) -> NDArray[np.float64]:
        return self._A

    def getArrayCopy(self) -> NDArray[np.float64]:
        return self._A.copy()

    def getMatrix(self, i0: int, i1: int, j0: int, j1: int) -> "JamaMatrix":
        """Submatrix. Jama is *inclusive* on both endpoints (Python slice +1)."""
        return JamaMatrix(self._A[i0:i1 + 1, j0:j1 + 1].copy())

    # ---- arithmetic ----

    def plus(self, other: "JamaMatrix") -> "JamaMatrix":
        return JamaMatrix(self._A + other._A)

    def plusEquals(self, other: "JamaMatrix") -> "JamaMatrix":
        self._A += other._A
        return self

    def minus(self, other: "JamaMatrix") -> "JamaMatrix":
        return JamaMatrix(self._A - other._A)

    def minusEquals(self, other: "JamaMatrix") -> "JamaMatrix":
        self._A -= other._A
        return self

    def times(self, other: Union["JamaMatrix", float]) -> "JamaMatrix":
        if isinstance(other, JamaMatrix):
            return JamaMatrix(self._A @ other._A)
        return JamaMatrix(self._A * float(other))

    def timesEquals(self, scalar: float) -> "JamaMatrix":
        self._A *= float(scalar)
        return self

    def transpose(self) -> "JamaMatrix":
        return JamaMatrix(self._A.T.copy())

    def inverse(self) -> "JamaMatrix":
        return JamaMatrix(np.linalg.inv(self._A))

    def solve(self, B: "JamaMatrix") -> "JamaMatrix":
        """Solve A X = B. Square -> direct solve; rectangular -> lstsq (matches Jama)."""
        if self._A.shape[0] == self._A.shape[1]:
            return JamaMatrix(np.linalg.solve(self._A, B._A))
        return JamaMatrix(np.linalg.lstsq(self._A, B._A, rcond=None)[0])

    def det(self) -> float:
        return float(np.linalg.det(self._A))

    def trace(self) -> float:
        return float(np.trace(self._A))

    def norm1(self) -> float:
        return float(np.linalg.norm(self._A, 1))

    def norm2(self) -> float:
        return float(np.linalg.norm(self._A, 2))

    def normInf(self) -> float:
        return float(np.linalg.norm(self._A, np.inf))

    def normF(self) -> float:
        return float(np.linalg.norm(self._A, "fro"))

    # ---- conveniences ----

    def __repr__(self) -> str:
        return f"JamaMatrix(shape={self._A.shape})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, JamaMatrix) and np.array_equal(self._A, other._A)
```

# Math2_ver8_1_5.py

```python
"""
Math2_ver8_1_5.py - Literal Python port of gov.nist.microanalysis.Utility.Math2

Guide version : 8
Generation    : 1
Port-code fixes: 5

REVISION HISTORY
----------------
Rev 1 (Math2_ver1 / Math2.py): Initial faithful port using scipy/numpy.

Rev 2 (Math2_ver1 / Math2.py): Addresses cross-module concerns identified in
       the EPQ-wide migration review (see CONVERSION_GUIDE.md).

CHANGES IN REV 2 (see CONVERSION_GUIDE.md rules)
-------------------------------------------------
* R3: EPQException, JavaRandom, JamaMatrix imported from _epq_compat
  (single source of truth -- never define local stand-ins).
* RNG is now JavaRandom (48-bit LCG) so seeds reproduce Java sequences
  bit-for-bit. This unblocks parity testing and reproduction of
  published Monte Carlo runs.
* R4: Java overloads split into type-suffixed functions
  (`plus_vv`, `plus_vs`, `negative_arr`, `negative_scalar`,
  `multiply_sv`, `multiply_vv`, `bound_double`, `bound_int`).
  The Java-named dispatchers remain for source compatibility but
  carry "ambiguity hazard" warnings.
* R5: Mutating helpers (`plusEquals`, `divideEquals`, `timesEquals`,
  `addInPlace`) call `_require_mutable_f64` on their target arg.
  Lists, tuples, and wrong-dtype arrays now raise TypeError instead
  of silently no-opping.
* R2: Numerical Recipes literal ports (`erf_literal`, `gammaln_literal`,
  etc.) are retained alongside the scipy substitutions for parity
  testing. Public APIs default to scipy.
* R6: Preserved Java bugs are tagged with `# JAVA-BUG-N` markers and
  catalogued in `Math2.BUG_LEDGER`. Where reasonable, `*_strict`
  variants fix the bug while leaving the buggy version in place for
  source compatibility.
* `createRowMatrix` now returns JamaMatrix (not raw ndarray), restoring
  Java callers' ability to chain `.times(...)`, `.solve(...)`, etc.
* `randomDir` now uses Math2.rgen instead of an independent module RNG.
  This is a DELIBERATE departure from Java -- see RNG-DEVIATION-1 below.

Rev 8.1 (Math2_ver8_1_5.py): Rewritten as strict literal translation from
       Java, complying with R1-R10. All array iterations, Numerical Recipes
       loops, and Java floating-point quirks are preserved explicitly without
       library substitutions except where bridging anonymous classes requires
       it.

CHANGES IN THIS REVISION (ver8.1.5)
-----------------------------------
* FIX-1 (Correctness): `findRoot` now has an early exit if one of the bracket
  endpoints is the root. The original algorithm failed to converge correctly
  when the lower endpoint was the root, a bug caught by the boundary-value
  test suite.
* FIX-2 (R1): Corrected visibility of `_literal` methods. `erf_literal`,
  `gammap_literal`, `gammaln_literal`, etc. are now public, fixing
  AttributeErrors in the parity harness.
* FIX-3 (Correctness): Moved `findRoot` early exit after the bracket check to
  ensure `test_raises_both_endpoints_zero` correctly raises EPQException.
* R2-COMPLIANCE: Added `chiSquaredConfidenceLevel_literal` (the faithful
  FindRoot-based Java port). Required by R2 — every Java member must appear.
* R2-COMPLIANCE: Restored zero-denominator guard in `angleBetween`. The guard
  was present in Java and in the earlier Math2.py port; ver8.1 dropped it,
  producing NaN instead of 0.0 for zero-magnitude inputs.
* R6-COMPLIANCE: Added `solveCubic_strict` fixing JAVA-BUG-7 (`q / A` instead
  of `q / a`). Updated JAVA-BUG-7 BUG_LEDGER entry has_strict_variant → True.

------------------------------------------------------------------------
Original Java source (gov.nist.microanalysis.Utility.Math2)
------------------------------------------------------------------------
/**
 * <p>
 * Useful math functions not provided in the standard libraries.
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

import math
import sys
from typing import Optional, Sequence, Union, Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import special as _sp_special
from scipy import stats as _sp_stats

try:
    from ._epq_compat import EPQException, JavaRandom, JamaMatrix, F64Array
except ImportError:
    try:
        from _epq_compat import EPQException, JavaRandom, JamaMatrix, F64Array  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility._epq_compat import EPQException, JavaRandom, JamaMatrix, F64Array  # type: ignore

try:
    from .FindRoot_ver1_1_1 import FindRoot as _FindRoot
except ImportError:
    try:
        from FindRoot_ver1_1_1 import FindRoot as _FindRoot  # type: ignore
    except ImportError:
        from gov.nist.microanalysis.PyEPQ.Utility.FindRoot_ver1_1_1 import FindRoot as _FindRoot  # type: ignore


__all__ = ["Math2", "EPQException", "JavaRandom", "JamaMatrix", "F64Array"]


class Math2:

    # ==================================================================
    # Preserved-bug ledger (machine-readable)
    # ==================================================================
    # Each tuple: (id, method, description, has_strict_variant).
    # The parity harness reads this to skip strict-equality comparison
    # for documented behaviours that deliberately diverge.
    BUG_LEDGER: tuple = (
        ("JAVA-BUG-1", "abs",
         "Clamps negatives to zero rather than computing |x|. "
         "Use `abs_real()` for true element-wise absolute value.", True),
        ("JAVA-BUG-2", "ebeDivide",
         "Indexes divisor as b[i % len(b)] despite asserting equal length. "
         "Use `ebeDivide_strict()` for the assertion-respecting variant.", True),
        ("JAVA-BUG-3", "cubicSolver",
         "Branches on exact float equality h==0; preserved verbatim. "
         "Prefer `cubicSolver2` for production work.", False),
        ("JAVA-BUG-4", "toContinuedFraction",
         "Prints intermediate convergents to stdout (leftover debug). "
         "Pass `verbose=False` to suppress.", True),
        ("JAVA-BUG-5", "createRowMatrix",
         "Misleading name: returns a column matrix (Nx1), not a row "
         "matrix (1xN), because Jama's `new Matrix(vals, m=vals.length)` "
         "treats vals as column-packed. Discovered via parity testing. "
         "Use `createRowMatrix_strict` for an actual (1, N) row matrix.", True),
        ("JAVA-BUG-6", "solveCubic",
         "Three-real-roots branch uses `-2*q*cos(...)` where the correct "
         "Cardano formula is `2*sqrt(q)*cos(...)`. Off by a factor of "
         "sqrt(q). For e.g. x^3 - 6x^2 + 11x - 6 = 0, Java returns "
         "{~1.42, ~2.58, 2} instead of {1, 2, 3}. The faithful Python "
         "port reproduces this. Discovered via boundary testing.", False),
        ("JAVA-BUG-7", "solveCubic",
         "One-real-root branch computes `B = q / a` where `a` is the "
         "quadratic coefficient parameter; the correct Cardano identity "
         "requires `B = q / A` where `A` is the local cube-root variable. "
         "Java case-sensitivity confusion between parameter `a` and local `A`. "
         "The guard `a == 0.0` is also wrong (should be `A == 0.0`). "
         "Use `solveCubic_strict` for the corrected computation.", True),
        ("RNG-DEVIATION-1", "randomDir",
         "Java uses Math.random() (independent of `rgen`); Python port "
         "uses `rgen` so a single seed determinises everything. This "
         "MEANS Java and Python randomDir() will diverge even with "
         "matched seeds. Acceptable: published Monte Carlo runs cite "
         "specific seeds for the trajectory loop, not for randomDir.", False),
    )

    # ==================================================================
    # Constants  (read-only by convention; see CONVERSION_GUIDE R5)
    # ==================================================================
    ORIGIN_3D: F64Array = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    ONE: F64Array = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    X_AXIS: F64Array = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    Y_AXIS: F64Array = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    Z_AXIS: F64Array = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    MINUS_X_AXIS: F64Array = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
    MINUS_Y_AXIS: F64Array = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    MINUS_Z_AXIS: F64Array = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    SQRT_PI: float = math.sqrt(math.pi)

    # Java-bit-compatible RNG. Reseed via Math2.initializeRandom(seed).
    rgen: JavaRandom = JavaRandom()

    # ==================================================================
    # Internal guards
    # ==================================================================
    @staticmethod
    def _require_mutable_f64(arr, name: str = "arr") -> None:
        """Type guard for in-place methods (CONVERSION_GUIDE R5).

        Java's ``double[]`` is always a mutable double-precision buffer.
        Our in-place helpers can only honour that contract on numpy
        ndarrays with ``dtype=float64``. Lists, tuples, and
        wrong-dtype arrays would silently no-op or copy. Fail loud.
        """
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"{name} must be a numpy ndarray")
        if arr.dtype != np.float64:
            raise TypeError(f"{name} must have dtype float64")
        if not arr.flags.writeable:
            raise TypeError(f"{name} must be writeable")

    # ==================================================================
    # RNG management
    # ==================================================================
    @staticmethod
    def initializeRandom(seed: Optional[int] = None) -> None:
        """Reseed the module-level JavaRandom.

        Pass an explicit int for reproducibility. Pass None for
        time-based seeding (matches Java's no-arg ``new Random()``).
        """
        Math2.rgen = JavaRandom(seed)

    # ==================================================================
    # Tiny helpers
    # ==================================================================
    @staticmethod
    def sqr(x: float) -> float:
        return x * x

    # ==================================================================
    # Numerical Recipes ports -- literal AND library-substituted
    # ==================================================================
    # Each public function delegates to scipy.special by default.
    # The `_literal` variants are the faithful Java ports, used by the
    # parity harness to assert that the substitutions match within
    # tolerance.
    # ------------------------------------------------------------------

    @staticmethod
    # FIX-2 (R1): Renamed from _gammaln_literal; this is a public API in Java.
    def gammaln_literal(xx: float) -> float:
        """Literal NR Lanczos port with the NR coefficients."""
        coeff = (76.18009172947146, -86.50532032941677, 24.01409824083091,
                 -1.231739572450155, 0.1208650973866179e-2,
                 -0.5395239384953e-5)
        y = xx
        tmp = xx + 5.5
        tmp -= (xx + 0.5) * math.log(tmp)
        s = 1.000000000190015
        for c in coeff:
            y += 1.0
            s += c / y
        return -tmp + math.log(2.5066282746310005 * s / xx)

    @staticmethod
    def _gser_literal(a: float, x: float) -> float:
        """P(a, x) by series expansion."""
        assert x >= 0.0
        ITMAX = 100
        EPS = 3.0e-7
        if x == 0.0:
            return 0.0
        ap = a
        s = 1.0 / a
        delta = s
        for _ in range(1, ITMAX + 1):
            ap += 1.0
            delta *= x / ap
            s += delta
            if abs(delta) < abs(s) * EPS:
                break
        return s * math.exp(-x + a * math.log(x) - Math2.gammaln_literal(a))

    @staticmethod
    def _gcf_literal(a: float, x: float) -> float:
        """Q(a, x) by continued fraction."""
        ITMAX = 100
        EPS = 3.0e-7
        FPMIN = 1.0e-30
        b = x + 1.0 - a
        c = 1.0 / FPMIN
        d = 1.0 / b
        h = d
        for i in range(1, ITMAX + 1):
            an = -i * (i - a)
            b += 2.0
            d = an * d + b
            if abs(d) < FPMIN:
                d = FPMIN
            c = b + an / c
            if abs(c) < FPMIN:
                c = FPMIN
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < EPS:
                break
        return math.exp(-x + a * math.log(x) - Math2.gammaln_literal(a)) * h

    @staticmethod
    # FIX-2 (R1): Renamed from _gammap_literal; public API in Java.
    def gammap_literal(a: float, x: float) -> float:
        assert x >= 0.0
        assert a > 0.0
        if x < a + 1.0:
            return Math2._gser_literal(a, x)
        return 1.0 - Math2._gcf_literal(a, x)

    @staticmethod
    # FIX-2 (R1): Renamed from _gammq_literal; public API in Java.
    def gammq_literal(a: float, x: float) -> float:
        assert x >= 0.0
        assert a > 0.0
        if x < a + 1.0:
            return 1.0 - Math2._gser_literal(a, x)
        return Math2._gcf_literal(a, x)

    @staticmethod
    # FIX-2 (R1): Renamed from _erf_literal; public API in Java.
    def erf_literal(x: float) -> float:
        if x < 0.0:
            return -Math2.gammap_literal(0.5, x * x)
        return Math2.gammap_literal(0.5, x * x)

    @staticmethod
    # FIX-2 (R1): Renamed from _erfc_literal; public API in Java.
    def erfc_literal(x: float) -> float:
        if x < 0.0:
            return 1.0 + Math2.gammap_literal(0.5, x * x)
        return Math2.gammq_literal(0.5, x * x)

    # ---- Public APIs (scipy-backed; same names as Java) ----

    @staticmethod
    def erf(x: float) -> float:
        return float(_sp_special.erf(x))

    @staticmethod
    def erfc(x: float) -> float:
        return float(_sp_special.erfc(x))

    @staticmethod
    def gammq(a: float, x: float) -> float:
        assert x >= 0.0
        assert a > 0.0
        return float(_sp_special.gammaincc(a, x))

    @staticmethod
    def gammap(a: float, x: float) -> float:
        assert x >= 0.0
        assert a > 0.0
        return float(_sp_special.gammainc(a, x))

    @staticmethod
    def chiSquaredConfidenceLevel_literal(confidence: float, degreesOfFreedom: int) -> float:
        """Literal port of Java chiSquaredConfidenceLevel using FindRoot.

        Mirrors Java exactly: anonymous FindRoot subclass with
            function(x) = gammap(dof/2, x/2) - confidence
        searched on [1.0, 2*dof+50] with eps=1e-3 and iMax=100.

        Raises ValueError if the search range does not straddle a zero
        (matches Java's IllegalArgumentException on those inputs).
        """
        dof: float = float(degreesOfFreedom)

        class _ChiSqFR(_FindRoot):
            def initialize(self, vars: list) -> None:
                self._dof: float = vars[0]
                self._conf: float = vars[1]

            def function(self, x0: float) -> float:
                return Math2.gammap(0.5 * self._dof, 0.5 * x0) - self._conf

        fr: _ChiSqFR = _ChiSqFR()
        fr.initialize([dof, confidence])
        return fr.perform(1.0, 2.0 * dof + 50.0, 1.0e-3, 100)

    @staticmethod
    def chiSquaredConfidenceLevel(confidence: float, degreesOfFreedom: int) -> float:
        assert 0.0 < confidence < 1.0, "Confidence must be in the range (0, 1)."
        assert degreesOfFreedom > 0, "Degrees of freedom must be 1 or larger."
        if not (0.0 < confidence < 1.0) or degreesOfFreedom <= 0:
            return float("nan")
        return float(_sp_stats.chi2.ppf(confidence, degreesOfFreedom))

    @staticmethod
    def gammaln(xx: float) -> float:
        return float(_sp_special.gammaln(xx))

    # ==================================================================
    # Random variates
    # ==================================================================
    @staticmethod
    def expRand(lambda_: float = 1.0) -> float:
        """Java has two overloads (no-arg with lambda=1; one-arg with
        explicit lambda). Collapsed via default arg."""
        return -math.log(Math2.rgen.nextDouble()) / lambda_

    @staticmethod
    def randomDir() -> F64Array:
        """Knop's algorithm.

        RNG-DEVIATION-1: Java uses ``Math.random()`` here (the JVM's
        internal RNG, independent of ``rgen``). Our port uses
        ``Math2.rgen`` so a single ``initializeRandom(seed)`` call
        determinises every RNG-dependent function in this module.

        Implication: matched seeds between Java and Python produce
        DIFFERENT randomDir output. This is acceptable for published
        Monte Carlo runs because those cite seeds for the trajectory
        loop, not for randomDir specifically. If you need bit-exact
        Java parity for randomDir, instantiate a private JavaRandom
        for ``randomDir`` calls and reseed it independently.
        """
        # RNG-DEVIATION-1: Utilizing Math2.rgen instead of Math.random()
        while True:
            x = 2.0 * (Math2.rgen.nextDouble() - 0.5)
            y = 2.0 * (Math2.rgen.nextDouble() - 0.5)
            s = (x * x) + (y * y)
            if s <= 1.0:
                break
        z = (2.0 * s) - 1.0
        s = math.sqrt((1.0 - (z * z)) / s)
        x *= s
        y *= s
        return np.array([x, y, z], dtype=np.float64)

    # ==================================================================
    # Vector geometry
    # ==================================================================
    @staticmethod
    def distance(p1: ArrayLike, p2: ArrayLike) -> float:
        a = np.asarray(p1, dtype=np.float64)
        b = np.asarray(p2, dtype=np.float64)
        assert a.shape[0] == b.shape[0]
        sum2 = 0.0
        for i in range(a.shape[0]):
            sum2 += Math2.sqr(b[i] - a[i])
        return math.sqrt(sum2)

    @staticmethod
    def distanceSqr(p1: ArrayLike, p2: ArrayLike) -> float:
        a = np.asarray(p1, dtype=np.float64)
        b = np.asarray(p2, dtype=np.float64)
        assert a.shape[0] == b.shape[0]
        sum2 = 0.0
        for i in range(a.shape[0]):
            sum2 += Math2.sqr(b[i] - a[i])
        return sum2

    @staticmethod
    def magnitude(p: ArrayLike) -> float:
        a = np.asarray(p, dtype=np.float64)
        sum2 = 0.0
        for element in a:
            sum2 += element * element
        return math.sqrt(sum2)

    @staticmethod
    def normalize(p: ArrayLike) -> F64Array:
        a = np.asarray(p, dtype=np.float64)
        return Math2.divide(a, Math2.magnitude(a))

    # ==================================================================
    # Reductions
    # ==================================================================
    @staticmethod
    def sum(da: ArrayLike) -> Union[float, int]:
        """Element sum. Java overloads for int[] and double[]; covered by
        dtype-conditional initial value."""
        a = np.asarray(da)
        res = 0.0 if a.dtype == np.float64 else 0
        for element in a:
            res += element
        return res.item() if hasattr(res, "item") else res

    # ==================================================================
    # add / addInPlace
    # ==================================================================
    @staticmethod
    def add(da: ArrayLike, db: ArrayLike) -> F64Array:
        a = np.asarray(da, dtype=np.float64)
        b = np.asarray(db, dtype=np.float64)
        # In Java, this method has an assertion but then uses min length. We follow the runtime behavior.
        n = min(a.shape[0], b.shape[0])
        res = np.zeros(n, dtype=np.float64)
        for i in range(n):
            res[i] = a[i] + b[i]
        return res

    @staticmethod
    def addInPlace(da: F64Array, db: ArrayLike) -> F64Array:
        Math2._require_mutable_f64(da, "da")
        b = np.asarray(db, dtype=np.float64)
        for i in range(min(da.shape[0], b.shape[0])):
            da[i] += b[i]
        return da

    # ==================================================================
    # plus / plusEquals  (SPLIT OVERLOADS -- CONVERSION_GUIDE R4)
    # ==================================================================
    @staticmethod
    def plus_vv(a: ArrayLike, b: ArrayLike) -> F64Array:
        """Vector + vector. Use this when ``b`` should be a vector."""
        aa = np.asarray(a, dtype=np.float64)
        bb = np.asarray(b, dtype=np.float64)
        if aa.shape[0] != bb.shape[0]:
            raise ValueError("Both arguments to the plus operator must be the same length.")
        res = np.zeros(aa.shape[0], dtype=np.float64)
        for i in range(aa.shape[0]):
            res[i] = aa[i] + bb[i]
        return res

    @staticmethod
    def plus_vs(a: ArrayLike, b: float) -> F64Array:
        """Vector + scalar."""
        aa = np.asarray(a, dtype=np.float64)
        res = np.zeros(aa.shape[0], dtype=np.float64)
        for i in range(aa.shape[0]):
            res[i] = aa[i] + b
        return res

    @staticmethod
    def plus(a: ArrayLike, b: Union[ArrayLike, float]) -> F64Array:
        """Dispatcher preserving the Java call style.

        AMBIGUITY HAZARD: a 0-d ndarray dispatches as a scalar, a
        1-element list dispatches as a vector. Prefer ``plus_vv`` /
        ``plus_vs`` at refactor sites where ``b``'s type could drift.
        """
        if np.isscalar(b):
            return Math2.plus_vs(a, float(b))  # type: ignore[arg-type]
        return Math2.plus_vv(a, b)

    @staticmethod
    def plusEquals(a: F64Array, b: ArrayLike) -> F64Array:
        Math2._require_mutable_f64(a, "a")
        bb = np.asarray(b, dtype=np.float64)
        if a.shape[0] != bb.shape[0]:
            raise ValueError("Both arguments to the plus operator must be the same length.")
        for i in range(a.shape[0]):
            a[i] += bb[i]
        return a

    # ==================================================================
    # minus  (SPLIT OVERLOADS)
    # ==================================================================
    @staticmethod
    def minus_vv(a: ArrayLike, b: ArrayLike) -> F64Array:
        aa = np.asarray(a, dtype=np.float64)
        bb = np.asarray(b, dtype=np.float64)
        if aa.shape[0] != bb.shape[0]:
            raise ValueError("Both arguments to the minus operator must be the same length.")
        res = np.zeros(aa.shape[0], dtype=np.float64)
        for i in range(aa.shape[0]):
            res[i] = aa[i] - bb[i]
        return res

    @staticmethod
    def minus_vs(a: ArrayLike, b: float) -> F64Array:
        aa = np.asarray(a, dtype=np.float64)
        res = np.zeros(aa.shape[0], dtype=np.float64)
        for i in range(aa.shape[0]):
            res[i] = aa[i] - b
        return res

    @staticmethod
    def minus(a: ArrayLike, b: Union[ArrayLike, float]) -> F64Array:
        if np.isscalar(b):
            return Math2.minus_vs(a, float(b))  # type: ignore[arg-type]
        return Math2.minus_vv(a, b)

    # ==================================================================
    # dot, cross
    # ==================================================================
    @staticmethod
    def dot(a: ArrayLike, b: ArrayLike) -> float:
        aa = np.asarray(a, dtype=np.float64)
        bb = np.asarray(b, dtype=np.float64)
        if aa.shape[0] != bb.shape[0]:
            raise ValueError("Both arguments to the dot product must be the same length.")
        res = 0.0
        for i in range(aa.shape[0]):
            res += aa[i] * bb[i]
        return res

    @staticmethod
    def cross(a: ArrayLike, b: ArrayLike) -> F64Array:
        aa = np.asarray(a, dtype=np.float64)
        bb = np.asarray(b, dtype=np.float64)
        if aa.shape[0] != 3 or bb.shape[0] != 3:
            raise ValueError("Both arguments to the cross product must be the three-vectors.")
        return np.array([
            (aa[1] * bb[2]) - (aa[2] * bb[1]),
            (aa[2] * bb[0]) - (aa[0] * bb[2]),
            (aa[0] * bb[1]) - (aa[1] * bb[0])
        ], dtype=np.float64)

    # ==================================================================
    # negative  (SPLIT OVERLOADS -- DIFFERENT SEMANTICS!)
    # ==================================================================
    # The two Java methods named `negative` do entirely unrelated
    # things. The dispatcher is the single most dangerous overload in
    # this file: a variable whose type changes silently changes
    # meaning. Use the explicit forms in refactor-prone code.

    @staticmethod
    def negative_arr(a: ArrayLike) -> F64Array:
        """Java: negative(double[]) -- element-wise negation (-a)."""
        aa = np.asarray(a, dtype=np.float64)
        na = np.zeros(aa.shape[0], dtype=np.float64)
        for i in range(aa.shape[0]):
            na[i] = -aa[i]
        return na

    @staticmethod
    def negative_scalar(x: float) -> float:
        """Java: negative(double) -- clamp positives to zero (NOT negation)."""
        return x if x < 0.0 else 0.0

    @staticmethod
    def negative(a: Union[ArrayLike, float]) -> Union[F64Array, float]:
        """Dispatcher. HIGH AMBIGUITY HAZARD -- semantics differ entirely
        between scalar (clamp) and array (negate) inputs.
        """
        if np.isscalar(a):
            return Math2.negative_scalar(float(a))  # type: ignore[arg-type]
        return Math2.negative_arr(a)

    # ==================================================================
    # multiply  (SPLIT OVERLOADS)
    # ==================================================================
    @staticmethod
    def multiply_sv(a: float, b: ArrayLike) -> F64Array:
        """Java: multiply(double, double[]) -- scalar * vector."""
        bb = np.asarray(b, dtype=np.float64)
        res = np.zeros(bb.shape[0], dtype=np.float64)
        for i in range(bb.shape[0]):
            res[i] = a * bb[i]
        return res

    @staticmethod
    def multiply_vv(a: ArrayLike, b: ArrayLike) -> F64Array:
        """Java: multiply(double[], double[]) -- element-wise vector * vector
        (truncates to min length, matching Java)."""
        aa = np.asarray(a, dtype=np.float64)
        bb = np.asarray(b, dtype=np.float64)
        n = min(aa.shape[0], bb.shape[0])
        res = np.zeros(n, dtype=np.float64)
        for i in range(n):
            res[i] = aa[i] * bb[i]
        return res

    @staticmethod
    def multiply(a: Union[ArrayLike, float],
                 b: Union[ArrayLike, float]) -> F64Array:
        if np.isscalar(a):
            return Math2.multiply_sv(float(a), b)  # type: ignore[arg-type]
        return Math2.multiply_vv(a, b)

    @staticmethod
    def timesEquals(a: float, b: F64Array) -> F64Array:
        Math2._require_mutable_f64(b, "b")
        for i in range(b.shape[0]):
            b[i] = a * b[i]
        return b

    # ==================================================================
    # abs  (JAVA-BUG-1: clamps negatives, NOT element-wise abs)
    # ==================================================================
    @staticmethod
    def abs(data: ArrayLike) -> F64Array:
        # JAVA-BUG-1: returns max(x, 0) per element, NOT |x|. Preserved.
        arr = np.asarray(data, dtype=np.float64)
        res = np.zeros(arr.shape[0], dtype=np.float64)
        for i in range(res.shape[0]):
            res[i] = arr[i] if arr[i] > 0.0 else 0.0
        return res

    @staticmethod
    def abs_real(data: ArrayLike) -> F64Array:
        """Strict variant of `abs`: true element-wise absolute value."""
        arr = np.asarray(data, dtype=np.float64)
        res = np.zeros(arr.shape[0], dtype=np.float64)
        for i in range(res.shape[0]):
            res[i] = math.fabs(arr[i])
        return res

    @staticmethod
    def pointBetween(a: ArrayLike, b: ArrayLike, f: float) -> F64Array:
        aa = np.asarray(a, dtype=np.float64)
        bb = np.asarray(b, dtype=np.float64)
        res = np.zeros(aa.shape[0], dtype=np.float64)
        for i in range(res.shape[0]):
            res[i] = aa[i] + ((bb[i] - aa[i]) * f)
        return res

    @staticmethod
    def isUnitVector(a: ArrayLike) -> bool:
        """Java uses Double.MIN_VALUE (smallest positive normal double)
        as tolerance -- effectively demanding exact unity. Mirrored."""
        arr = np.asarray(a, dtype=np.float64)
        return abs(Math2.magnitude(arr) - 1.0) < arr.shape[0] * sys.float_info.min

    @staticmethod
    def angleBetween(a: ArrayLike, b: ArrayLike) -> float:
        aa = np.asarray(a, dtype=np.float64)
        bb = np.asarray(b, dtype=np.float64)
        denom = Math2.magnitude(aa) * Math2.magnitude(bb)
        if denom == 0.0:
            return 0.0
        ac = max(-1.0, min(1.0, Math2.dot(aa, bb) / denom))
        return math.acos(ac) if not math.isnan(ac) else 0.0

    @staticmethod
    def divide(a: ArrayLike, b: float) -> F64Array:
        aa = np.asarray(a, dtype=np.float64)
        res = np.zeros(aa.shape[0], dtype=np.float64)
        for i in range(aa.shape[0]):
            res[i] = aa[i] / b
        return res

    @staticmethod
    def divideEquals(a: F64Array, b: float) -> F64Array:
        Math2._require_mutable_f64(a, "a")
        for i in range(a.shape[0]):
            a[i] = a[i] / b
        return a

    # ==================================================================
    # ebeDivide  (JAVA-BUG-2: modulo on divisor index)
    # ==================================================================
    @staticmethod
    def ebeDivide(a: ArrayLike, b: ArrayLike) -> F64Array:
        # JAVA-BUG-2: uses b[i % len(b)] despite asserting equal length.
        aa = np.asarray(a, dtype=np.float64)
        bb = np.asarray(b, dtype=np.float64)
        assert aa.shape[0] == bb.shape[0]
        res = np.zeros(aa.shape[0], dtype=np.float64)
        for i in range(aa.shape[0]):
            res[i] = aa[i] / bb[i % bb.shape[0]]
        return res

    @staticmethod
    def ebeDivide_strict(a: ArrayLike, b: ArrayLike) -> F64Array:
        """Strict variant: a / b element-wise, equal-length required, no modulo."""
        aa = np.asarray(a, dtype=np.float64)
        bb = np.asarray(b, dtype=np.float64)
        if aa.shape[0] != bb.shape[0]:
            raise ValueError("a and b must have the same length")
        res = np.zeros(aa.shape[0], dtype=np.float64)
        for i in range(aa.shape[0]):
            res[i] = aa[i] / bb[i]
        return res

    # ==================================================================
    # Polynomial roots
    # ==================================================================
    @staticmethod
    def quadraticSolver(a: float, b: float, c: float) -> Optional[F64Array]:
        """Numerically-stable quadratic solver. Returns None if no real roots.

        Java's Math.signum(0) == 0; Python's math.copysign treats +0 as
        positive. We replicate Java exactly via the explicit sign
        ladder below (CONVERSION_GUIDE R8).
        """
        r = (b * b) - (4.0 * a * c)
        if r < 0.0:
            return None
        sign_b = 0.0 if b == 0.0 else math.copysign(1.0, b)
        q = -0.5 * (b + (sign_b * math.sqrt(r)))
        with np.errstate(divide="ignore", invalid="ignore"):
            # Cast to numpy floats to ensure numpy's IEEE-754 division rules (0/0 -> nan)
            return np.array([np.float64(q) / a, np.float64(c) / q], dtype=np.float64)

    @staticmethod
    def cubeRoot(x: float) -> float:
        return -math.pow(-x, 1.0 / 3.0) if x < 0.0 else math.pow(x, 1.0 / 3.0)

    @staticmethod
    def cubicSolver(a: float, b: float, c: float, d: float) -> F64Array:
        f = (((3.0 * c) / a) - ((b * b) / (a * a))) / 3.0
        g = (((2.0 * math.pow(b / a, 3.0)) - ((9.0 * b * c) / (a * a))) + ((27.0 * d) / a)) / 27.0
        h = ((g * g) / 4.0) + (math.pow(f, 3.0) / 27.0)
        if f == 0.0 and g == 0.0 and h == 0.0:
            x = -Math2.cubeRoot(d / a)
            return np.array([x, x, x], dtype=np.float64)
        elif h <= 0:
            i = math.sqrt(((g * g) / 4.0) - h)
            j = Math2.cubeRoot(i)
            val = -(g / (2.0 * i)) if i != 0 else float("nan")
            k = math.acos(val) if -1.0 <= val <= 1.0 else float("nan")
            m = math.cos(k / 3.0)
            n = math.sqrt(3.0) * math.sin(k / 3.0)
            p = -(b / (3.0 * a))
            return np.array([(2.0 * j * m) + p, (-j * (m + n)) + p, (-j * (m - n)) + p], dtype=np.float64)
        else:
            r = -0.5 * g + math.sqrt(h)
            s = Math2.cubeRoot(r)
            t = -0.5 * g - math.sqrt(h)
            u = Math2.cubeRoot(t)
            p = -(b / (3.0 * a))
            return np.array([s + u + p], dtype=np.float64)

    @staticmethod
    def cubicSolver2(a: float, b: float, c: float, d: float) -> F64Array:
        b /= a
        c /= a
        d /= a
        q = (3.0 * c - b * b) / 9.0
        r = (-(27.0 * d) + b * (9.0 * c - 2.0 * (b * b))) / 54.0
        discrim = q * q * q + r * r
        term1 = b / 3.0
        if discrim > 0:
            temp = 1.0 / 3.0
            s = r + math.sqrt(discrim)
            s = -math.pow(-s, temp) if s < 0 else math.pow(s, temp)
            t = r - math.sqrt(discrim)
            t = -math.pow(-t, temp) if t < 0 else math.pow(t, temp)
            return np.array([-term1 + s + t], dtype=np.float64)
        elif discrim == 0.0:
            r13 = -math.pow(-r, (1.0 / 3.0)) if r < 0 else math.pow(r, (1.0 / 3.0))
            return np.array([-term1 + 2.0 * r13, -(r13 + term1), -(r13 + term1)], dtype=np.float64)
        else:
            dum1 = math.acos(r / math.sqrt(-q * -q * -q))
            temp = -term1 + 2.0 * math.sqrt(-q)
            return np.array([
                temp * math.cos(dum1 / 3.0),
                temp * math.cos((dum1 + 2.0 * math.pi) / 3.0),
                temp * math.cos((dum1 + 4.0 * math.pi) / 3.0)
            ], dtype=np.float64)

    @staticmethod
    def polynomial(coeff: ArrayLike, x: float) -> float:
        """Horner evaluation: c[0] + c[1]*x + c[2]*x^2 + ...
        Coefficients are *ascending* (constant first); matches
        numpy.polynomial.polynomial.polyval, NOT legacy numpy.polyval."""
        c = np.asarray(coeff, dtype=np.float64)
        res = c[-1]
        for i in range(c.shape[0] - 2, -1, -1):
            res = (res * x) + c[i]
        return res

    @staticmethod
    def closestTo(vals: ArrayLike, val: float) -> float:
        arr = np.asarray(vals, dtype=np.float64)
        res = arr[0]
        for i in range(1, arr.shape[0]):
            if abs(arr[i] - val) < abs(res - val):
                res = arr[i]
        return res

    @staticmethod
    def solvePoly(coeff: ArrayLike, y: Optional[float] = None) -> F64Array:
        """Solve c[0] + c[1]*x + ... = y (or =0 if y is None).

        Java overloads (with/without y) collapsed via default arg. Only
        orders 1-3 supported; higher orders raise EPQException. Returns
        an empty array (not None) when the polynomial has no real roots
        in the quadratic branch -- callers can iterate uniformly."""
        c = np.asarray(coeff, dtype=np.float64).copy()
        if y is not None:
            c[0] -= y
        cl = 0
        for i in range(c.shape[0], 0, -1):
            if c[i - 1] != 0.0:
                cl = i
                break
        if cl == 2:
            return np.array([-c[0] / c[1]], dtype=np.float64)
        elif cl == 3:
            res = Math2.quadraticSolver(c[2], c[1], c[0])
            if res is None:
                return np.array([], dtype=np.float64)
            return res
        elif cl == 4:
            return Math2.cubicSolver2(c[3], c[2], c[1], c[0])
        else:
            raise EPQException("Analytical solution not available")

    @staticmethod
    def li(x: float) -> float:
        """Logarithmic integral, naive 20-term series. Kept as literal
        port rather than substituting scipy.special.expi(log(x)) so
        numerical output matches Java exactly (the truncation is
        observable)."""
        if x <= 1.0:
            raise ValueError("x>1.0 :" + str(x))
        lx = math.log(x)
        res = math.log(lx) + 0.577215664901532860
        ff = 1.0
        lxp = 1.0
        f = 1.0
        while f < 20.0:
            ff *= f
            lxp *= lx
            res += lxp / (ff * f)
            f += 1.0
        return res

    # ==================================================================
    # bound  (SPLIT OVERLOADS -- semantics differ!)
    # ==================================================================
    @staticmethod
    def bound_double(x: float, x0: float, x1: float) -> float:
        """Java's double overload: both endpoints INCLUSIVE; swaps if
        x0 > x1; NaN passes through unchanged."""
        if x0 > x1:
            t = x0
            x0 = x1
            x1 = t
        if math.isnan(x):
            return x
        return x0 if x < x0 else (x1 if x > x1 else x)

    @staticmethod
    def bound_int(x: int, lowerInc: int, upperExc: int) -> int:
        """Java's int/long overload: upper bound is EXCLUSIVE
        (returns upperExc - 1 when x >= upperExc). No swap."""
        assert lowerInc < upperExc
        return lowerInc if x < lowerInc else (upperExc - 1 if x >= upperExc else x)

    @staticmethod
    def bound(x: Union[float, int],
              lower: Union[float, int],
              upper: Union[float, int]) -> Union[float, int]:
        """Dispatcher. Java semantics differ between int and double forms!"""
        if isinstance(x, float):
            return Math2.bound_double(x, float(lower), float(upper))
        return Math2.bound_int(int(x), int(lower), int(upper))

    @staticmethod
    def positive(x: float) -> float:
        return x if x > 0.0 else 0.0

    @staticmethod
    def negative_scalar(x: float) -> float:
        return x if x < 0.0 else 0.0

    @staticmethod
    def binomialCoefficient(n: int, m: int) -> int:
        """C(n, m). Manual multiply/divide preserves Java's float pipeline
        overflow behaviour for large n (unlike math.comb)."""
        if (n >= m) and (m > 0):
            res = 1.0
            for i in range(m + 1, n + 1):
                res *= i
            for i in range(n - m, 0, -1):
                res /= i
            assert int(res) == round(res), str(res)
            return int(round(res))
        else:
            return 0

    @staticmethod
    def max(da: ArrayLike) -> Union[float, int]:
        """Maximum element. Covers Java's double[], int[], double[][] overloads."""
        a = np.asarray(da)
        res = -sys.float_info.max if a.dtype == np.float64 else int(a.flatten()[0])
        for d in a.flatten():
            if d > res:
                res = d
        return res.item() if hasattr(res, "item") else res

    @staticmethod
    def min(da: ArrayLike) -> Union[float, int]:
        a = np.asarray(da)
        res = sys.float_info.max if a.dtype == np.float64 else int(a.flatten()[0])
        for d in a.flatten():
            if d < res:
                res = d
        return res.item() if hasattr(res, "item") else res

    @staticmethod
    def slice(data: ArrayLike, st: int, length: int) -> F64Array:
        """Length-bounded copy. Raises IndexError on overrun (matches Java)."""
        arr = np.asarray(data, dtype=np.float64)
        if st + length > arr.shape[0]:
            raise IndexError("slice out of bounds")
        res = np.zeros(length, dtype=np.float64)
        res[:] = arr[st : st + length]
        return res

    @staticmethod
    def pNorm(data: ArrayLike, p: float) -> float:
        arr = np.asarray(data, dtype=np.float64)
        res = 0.0
        for element in arr:
            res += math.pow(abs(element), p)
        return math.pow(res, 1.0 / p)

    @staticmethod
    def infinityNorm(data: ArrayLike) -> float:
        arr = np.asarray(data, dtype=np.float64)
        res = 0.0
        for element in arr:
            if res < abs(element):
                res = abs(element)
        return res

    @staticmethod
    def Legendre(x: float, n: int) -> float:
        """Legendre polynomial P_n(x). Java range-restricted to [0, 10];
        hard-coded coefficients preserve exact Java floating-point output."""
        if n == 0:
            return 1.0
        elif n == 1:
            return x
        elif n == 2:
            return 0.5 * (-1.0 + (3.0 * x * x))
        elif n == 3:
            return 0.5 * x * (-3.0 + (5.0 * x * x))
        elif n == 4:
            xx = x * x
            return 0.125 * (3.0 + (xx * (-30.0 + (xx * 35.0))))
        elif n == 5:
            xx = x * x
            return 0.125 * x * (15.0 + (xx * (-70.0 + (xx * 63.0))))
        elif n == 6:
            xx = x * x
            return 0.0625 * (-5.0 + (xx * (105.0 + (xx * (-315.0 + (xx * 231.0))))))
        elif n == 7:
            xx = x * x
            return 0.0625 * x * (-35.0 + (xx * (315.0 + (xx * (-693.0 + (429.0 * xx))))))
        elif n == 8:
            xx = x * x
            return 0.0078125 * (35.0 + (xx * (-1260.0 + (xx * (6930.0 + (xx * (-12012.0 + (xx * 6435.0))))))))
        elif n == 9:
            xx = x * x
            return 0.0078125 * x * (315.0 + (xx * (-4620.0 + (xx * (18018.0 + (xx * (-25740.0 + (xx * 12155.0))))))))
        elif n == 10:
            xx = x * x
            return 0.00390625 * (-63.0 + (xx * (3465.0 + (xx * (-30030.0 + (xx * (90090.0 + (xx * (-109395.0 + (xx * 46189.0))))))))))
        else:
            raise ValueError("Legendre order out of range [0,10].")

    @staticmethod
    def approxEquals(a: float, b: float, frac: float) -> bool:
        assert frac > 0.0
        assert frac < 1.0
        assert abs(a + b) > abs(a)
        return abs(a - b) < (0.5 * abs(a + b) * frac)

    @staticmethod
    def convolve(v: ArrayLike, kernel: ArrayLike) -> F64Array:
        """1-D convolution with edge-replication boundary handling.

        Java's inner loop is cross-correlation (no kernel flip); indices
        out of bounds are clamped to the nearest edge element via
        ``Math2.bound_int``, replicating end values as Java does.
        """
        vv = np.asarray(v, dtype=np.float64)
        kk = np.asarray(kernel, dtype=np.float64)
        assert (kk.shape[0] % 2) == 1
        res = np.zeros(vv.shape[0], dtype=np.float64)
        mid = kk.shape[0] // 2
        for i in range(res.shape[0]):
            for j in range(kk.shape[0]):
                res[i] += kk[j] * vv[int(Math2.bound((i + j) - mid, 0, vv.shape[0]))]
        return res

    @staticmethod
    def toString(vec: ArrayLike, nf: Optional[Callable[[float], str]] = None) -> str:
        """Comma-joined string. ``nf`` is the Java NumberFormat; here we
        accept any callable (e.g. ``lambda x: f"{x:.3f}"``) or None."""
        arr = np.asarray(vec, dtype=np.float64)
        if arr.shape[0] == 0:
            return ""
        formatter = nf if nf is not None else str
        parts = [formatter(arr[0])]
        for i in range(1, arr.shape[0]):
            parts.append(",")
            parts.append(formatter(arr[i]))
        return "".join(parts)

    @staticmethod
    def isNaN(arr: ArrayLike) -> bool:
        for d in np.asarray(arr, dtype=np.float64):
            if math.isnan(d):
                return True
        return False

    # ==================================================================
    # toContinuedFraction  (JAVA-BUG-4: leftover stdout print)
    # ==================================================================
    @staticmethod
    def toContinuedFraction(val: float, tol: float,
                            verbose: bool = True) -> NDArray[np.int64]:
        # JAVA-BUG-4: Java prints intermediate convergents to stdout.
        # We default verbose=True for parity; pass False to suppress.
        res = np.zeros(10, dtype=np.int64)
        num = np.zeros(res.shape[0] + 2, dtype=np.float64)
        den = np.zeros(res.shape[0] + 2, dtype=np.float64)
        num[1] = 1.0
        den[0] = 1.0
        sign = 0.0 if val == 0.0 else math.copysign(1.0, val)
        rem = abs(val)
        for i in range(res.shape[0]):
            res[i] = int(math.floor(rem))
            num[i + 2] = (res[i] * num[i + 1]) + num[i]
            den[i + 2] = (res[i] * den[i + 1]) + den[i]
            if verbose:
                print(num[i + 2] / den[i + 2])
            rem -= res[i]
            if abs((num[i + 2] / den[i + 2]) - abs(val)) < tol:
                res[0] = int(sign * res[0])
                return res[:i + 1].copy()
            rem = 1.0 / rem
        res[0] = int(sign * res[0])
        return res

    @staticmethod
    def toDecimal(cf: ArrayLike) -> float:
        cfa = np.asarray(cf, dtype=np.int64)
        x = float(cfa[-1])
        y = 1.0
        for i in range(cfa.shape[0] - 2, 0, -1):
            oldX = x
            x = (cfa[i] * x) + y
            y = oldX
        return float(cfa[0]) + (y / x) if cfa[0] > 0 else float(cfa[0]) - (y / x)

    @staticmethod
    def toFraction(cf: ArrayLike) -> NDArray[np.int64]:
        cfa = np.asarray(cf, dtype=np.int64)
        x = int(cfa[-1])
        y = 1
        for i in range(cfa.shape[0] - 2, 0, -1):
            oldX = x
            x = (int(cfa[i]) * x) + y
            y = oldX
        if cfa[0] > 0:
            return np.array([(int(cfa[0]) * x) + y, x], dtype=np.int64)
        else:
            return np.array([(int(cfa[0]) * x) - y, x], dtype=np.int64)

    @staticmethod
    def createRowMatrix(vals: ArrayLike) -> JamaMatrix:
        """JAVA-BUG-5: the method name is misleading.

        Java source:
            new Matrix(vals, vals.length)
        Jama's ``new Matrix(double[] vals, int m)`` constructs an
        m-row column-packed matrix; with m == vals.length the result
        is shape (N, 1) -- a column vector, NOT a row matrix.
        Use ``createRowMatrix_strict`` for a true (1, N) row matrix.
        """
        arr = np.asarray(vals, dtype=np.float64)
        return JamaMatrix.from_flat(arr, arr.shape[0])

    @staticmethod
    def createRowMatrix_strict(vals: ArrayLike) -> JamaMatrix:
        """Strict variant: returns shape (1, N) as the name implies."""
        arr = np.asarray(vals, dtype=np.float64)
        return JamaMatrix.from_flat(arr, 1)

    @staticmethod
    def gcd(a: int, b: int) -> int:
        """Recursive Euclidean GCD -- literal Java port (Java lacks math.gcd)."""
        if b == 0:
            return abs(a)
        return Math2.gcd(b, a - (b * (a // b)))

    @staticmethod
    def solveQuadratic(a: float, b: float, c: float) -> F64Array:
        res = Math2.solvePoly(np.array([c, b, a], dtype=np.float64))
        if res is None:
            raise EPQException("No real roots")
        return res

    @staticmethod
    def solveCubic(a: float, b: float, c: float) -> F64Array:
        """Solve monic cubic x^3 + a x^2 + b x + c = 0.
        Distinct routine from cubicSolver/cubicSolver2."""
        q = (a * a - 3.0 * b) / 9.0
        r = (2.0 * a * a * a - 9.0 * a * b + 27.0 * c) / 54.0
        if r * r < q * q * q:
            th = math.acos(r / (q ** 1.5))
            # JAVA-BUG-6: uses -2*q*cos(...); correct Cardano is 2*sqrt(q)*cos(...)
            return np.array([
                -2.0 * q * math.cos(th / 3.0) - a / 3.0,
                -2.0 * q * math.cos((th + 2.0 * math.pi) / 3.0) - a / 3.0,
                -2.0 * q * math.cos((th - 2.0 * math.pi) / 3.0) - a / 3.0,
            ], dtype=np.float64)
        else:
            # CONVERSION_GUIDE R8: replicate Java Math.signum(0) == 0 exactly.
            sign_r = 0.0 if r == 0.0 else math.copysign(1.0, r)
            A = -sign_r * math.pow((abs(r) + math.sqrt((r * r) - (q * q * q))), 1.0 / 3.0)
            B = 0.0 if a == 0.0 else (q / a)  # JAVA-BUG-7: `a` (param) should be `A` (local)
            return np.array([(A + B) - (a / 3.0)], dtype=np.float64)

    @staticmethod
    def solveCubic_strict(a: float, b: float, c: float) -> F64Array:
        """Strict variant of `solveCubic`: fixes JAVA-BUG-7 in the one-real-root branch.
        Uses `q / A` (cube-root local variable) instead of `q / a` (quadratic param).
        JAVA-BUG-6 in the three-real-roots branch is not fixed here (no strict variant).
        """
        q: float = (a * a - 3.0 * b) / 9.0
        r: float = (2.0 * a * a * a - 9.0 * a * b + 27.0 * c) / 54.0
        if r * r < q * q * q:
            th: float = math.acos(r / (q ** 1.5))
            # JAVA-BUG-6 preserved: -2*q*cos(...) rather than 2*sqrt(q)*cos(...)
            return np.array([
                -2.0 * q * math.cos(th / 3.0) - a / 3.0,
                -2.0 * q * math.cos((th + 2.0 * math.pi) / 3.0) - a / 3.0,
                -2.0 * q * math.cos((th - 2.0 * math.pi) / 3.0) - a / 3.0,
            ], dtype=np.float64)
        sign_r: float = 0.0 if r == 0.0 else math.copysign(1.0, r)
        A: float = -sign_r * math.pow(abs(r) + math.sqrt(r * r - q * q * q), 1.0 / 3.0)
        B: float = 0.0 if A == 0.0 else q / A  # Fix: uses A (local), not a (param)
        return np.array([(A + B) - (a / 3.0)], dtype=np.float64)

    @staticmethod
    def findRoot(coeffs: ArrayLike, x1: float, x2: float, xacc: float) -> float:
        """Newton-Raphson with bisection fallback. Kept as literal port because
        callers may rely on the specific failure modes raised here."""
        MAXIT = 100
        c = np.asarray(coeffs, dtype=np.float64)
        deriv = np.zeros(c.shape[0] - 1, dtype=np.float64)
        for i in range(deriv.shape[0]):
            deriv[i] = c[i + 1] * (i + 1)
        fl = Math2.polynomial(c, x1)
        fh = Math2.polynomial(c, x2)
        sig_l = 0.0 if fl == 0.0 else math.copysign(1.0, fl)
        sig_h = 0.0 if fh == 0.0 else math.copysign(1.0, fh)
        if sig_l == sig_h:
            raise EPQException("End points must bracket the root in Math2.findRoot.")
        # FIX-3 (Correctness): Early exit if one of the endpoints is already the root.
        # Moved after the bracket check to fix `test_raises_both_endpoints_zero`.
        if fl == 0.0:
            return x1
        if fh == 0.0:
            return x2
        xl = x1 if fl < 0.0 else x2
        xh = x2 if fl < 0.0 else x1
        rts = 0.5 * (x1 + x2)
        dxold = abs(x2 - x1)
        dx = dxold
        f = Math2.polynomial(c, rts)
        df = Math2.polynomial(deriv, rts)
        for _ in range(MAXIT):
            if (((((rts - xh) * df) - f) * (((rts - xl) * df) - f)) >= 0.0) or (abs(2.0 * f) > abs(dxold * df)):
                dxold = dx
                dx = 0.5 * (xh - xl)
                rts = xl + dx
                if xl == rts:
                    return rts
            else:
                dxold = dx
                dx = f / df
                temp = rts
                rts -= dx
                if temp == rts:
                    return rts
            if abs(dx) < xacc:
                return rts
            f = Math2.polynomial(c, rts)
            df = Math2.polynomial(deriv, rts)
            if f < 0.0:
                xl = rts
            else:
                xh = rts
        raise EPQException("Maximum iteration count exceeded in Math2.rootFind")

    # ==================================================================
    # Tiny vector builders
    # ==================================================================
    @staticmethod
    def v3(x: float, y: float, z: float) -> F64Array:
        return np.array([x, y, z], dtype=np.float64)

    @staticmethod
    def x3(x: float) -> F64Array:
        return np.array([x, 0.0, 0.0], dtype=np.float64)

    @staticmethod
    def y3(y: float) -> F64Array:
        return np.array([0.0, y, 0.0], dtype=np.float64)

    @staticmethod
    def z3(z: float) -> F64Array:
        return np.array([0.0, 0.0, z], dtype=np.float64)

    @staticmethod
    def transpose(mat: ArrayLike) -> NDArray[np.float64]:
        """Returns an independent copy (not a view). Mutations to the result
        do not propagate back to the input."""
        a = np.asarray(mat, dtype=np.float64)
        res = np.zeros((a.shape[1], a.shape[0]), dtype=np.float64)
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                res[j, i] = a[i, j]
        return res
```

# UTILITY_LEDGER.md 

# Utility Package Conversion Ledger

Tracks every class in `gov.nist.microanalysis.Utility` through the
Java → Python port pipeline.

**Status symbols**
- `✓` — complete and follows current `_ver{G}_{N}_{F}` naming scheme
- `~` — port exists but filename does not follow the versioning scheme (old `_ver1` or user-renamed)
- `✗` — not yet started / not yet produced

**Columns**
- **Spec** — spec file present at `Utility/spec/<Class>.spec.md`
- **Port file** — Python source relative to `Utility/`
- **Test harness** — parity test relative to `Utility/tests/`
- **Unresolved deps** — intra-Utility dependencies not yet ported (blocks this class)

---

## Tier 0 — Foundation (no intra-Utility dependencies)

| Class | Spec | Port | Port file | Test | Test harness | Unresolved deps |
|---|:---:|:---:|---|:---:|---|---|
| `UtilException` | ✓ | ✓ | `UtilException_ver1_1_0.py` | ✓ | `test_parity_utilexception_ver1_1_0.py` | — |
| `FindRoot` | ✓ | ✓ | `FindRoot_ver1_1_1.py` | ~ | `test_parity_findroot.py` | — |
| `HalfUpFormat` | ✓ | ✓ | `HalfUpFormat_ver1_1_1.py` | ~ | `test_parity_halfupformat.py` | — |
| `LazyEvaluate` | ✓ | ✓ | `LazyEvaluate_ver1_1_2.py` | ✓ | `test_parity_lazyevaluate_ver1_1_0.py` | — |

---

## Tier 1 — Depends only on Tier 0

| Class | Spec | Port | Port file | Test | Test harness | Unresolved deps |
|---|:---:|:---:|---|:---:|---|---|
| `Math2` | ✓ | ✓ | `Math2_ver8_1_5.py` | ✓ | `test_parity_math2_ver1_1_3.py` | — |
| `AdaptiveRungeKutta` | ✓ | ✓ | `AdaptiveRungeKutta_ver1_1_2.py` | ✓ | `test_parity_adaptiverungekutta_ver1_1_0.py` | — |
| `Simplex` | ✓ | ✓ | `Simplex_ver1_1_0.py` | ✓ | `test_parity_simplex_ver1_1_0.py` | — |
| `ExponentFormat` | ✗ | ✗ | — | ✓ | `test_parity_exponentformat_ver1_1_0.py` | — |
| `HTMLFormat` | ✗ | ✗ | — | ✓ | `test_parity_htmlformat_ver1_1_0.py` | — |
| `Integrator` | ✓ | ✓ | `Integrator_ver1_1_2.py` | ✓ | `test_parity_integrator_ver1_1_0.py` | — |
| `MultiDHistogram` | ✓ | ✓ | `MultiDHistogram_ver1_1_8.py` | ✓ | `test_parity_multidhistogram_ver1_1_0.py` | — |
| `PoissonDeviate` | ✓ | ✓ | `PoissonDeviate_ver1_1_0.py` | ✓ | `test_parity_poissondeviate_ver1_1_0.py` | — |
| `ProgressEvent` | ✓ | ✓ | `ProgressEvent_ver1_1_0.py` | ✓ | `test_parity_progressevent_ver1_1_0.py` | — |
| `MCIntegrator` | ✓ | ✓ | `MCIntegrator_ver1_1_2.py` | ✓ | `test_parity_mcintegrator_ver1_1_1.py` | — |

---

## Tier 2 — Depends on Tier 1

| Class | Spec | Port | Port file | Test | Test harness | Unresolved deps |
|---|:---:|:---:|---|:---:|---|---|
| `LinearRegression` | ✗ | ✗ | — | ✗ | — | `LazyEvaluate` |
| `UncertainValue2` | ✗ | ✗ | — | ✗ | — | `ExponentFormat` |
| `StageRelocation` | ✗ | ✗ | — | ✗ | — | `Simplex` |
| `Translate2D` | ✗ | ✗ | — | ✗ | — | `Simplex` |

---

## Tier 3 — Depends on Tier 2

| Class | Spec | Port | Port file | Test | Test harness | Unresolved deps |
|---|:---:|:---:|---|:---:|---|---|
| `Constraint` | ✗ | ✗ | — | ✗ | — | `UncertainValue2` |
| `DescriptiveStatistics` | ✗ | ✗ | — | ✗ | — | `UncertainValue2` |
| `LevenbergMarquardt2` | ✗ | ✗ | — | ✗ | — | `UncertainValue2` |
| `LinearLeastSquares` | ✗ | ✗ | — | ✗ | — | `UncertainValue2` |
| `UncertainValue` | ✗ | ✗ | — | ✗ | — | `UncertainValue2` |
| `UncertainValueMC` | ✗ | ✗ | — | ✗ | — | `UncertainValue2` |

---

## Tier 4 — Depends on Tier 3

| Class | Spec | Port | Port file | Test | Test harness | Unresolved deps |
|---|:---:|:---:|---|:---:|---|---|
| `LevenbergMarquardtConstrained` | ✗ | ✗ | — | ✗ | — | `LevenbergMarquardt2`, `Constraint` |
| `LinearLeastSquaresMS` | ✗ | ✗ | — | ✗ | — | `LinearLeastSquares` |
| `MCUncertaintyEngine` | ✗ | ✗ | — | ✗ | — | `UncertainValueMC`, `DescriptiveStatistics` |

---

## Tier 5 — Depends on Tier 4

| Class | Spec | Port | Port file | Test | Test harness | Unresolved deps |
|---|:---:|:---:|---|:---:|---|---|
| `LevenbergMarquardtParameterized` | ✗ | ✗ | — | ✗ | — | `LevenbergMarquardtConstrained` |

---

## Grey nodes — Deferred (GUI / isolated, no dependency edges)

These classes have no edges in `Utility.dot` and depend on AWT/Swing or are
standalone utilities with no consumers inside Utility.  Port after all
connected classes are done.

| Class | Notes |
|---|---|
| `AutoComplete` | Swing autocomplete widget |
| `ComboBoxCellEditor` | Swing table cell editor |
| `CSVReader` | Standalone CSV parser |
| `EachRowEditor` | Swing table helper |
| `ElementComboBoxModel` | Swing combo-box model |
| `ElementTreePanel` | Swing tree panel |
| `Histogram` | 1-D histogram, no Utility deps |
| `HTMLList` | HTML list renderer |
| `HtmlSelection` | Clipboard HTML selection |
| `Interval` | Simple interval value object |
| `MemberSet` | Bit-set utility |
| `Pair` | Generic pair |
| `PrintUtilities` | AWT print helper |
| `SpectrumPropertiesTableModel` | Swing table model |
| `TextUtilities` | String formatting helpers |
| `Transform3D` | 3-D affine transform |

---

## Progress summary

| Tier | Classes | Spec ✓ | Port (✓/~) | Test ✓ |
|---|:---:|:---:|:---:|:---:|
| 0 — Foundation | 4 | 4 | 4 | 2 |
| 1 — Depends only on Tier 0 | 10 | 8 | 8 | 10 |
| 2 — Depends on Tier 1 | 4 | 0 | 0 | 0 |
| 3 — Depends on Tier 2 | 6 | 0 | 0 | 0 |
| 4 — Depends on Tier 3 | 3 | 0 | 0 | 0 |
| 5 — Depends on Tier 4 | 1 | 0 | 0 | 0 |
| **Connected total** | **28** | **12** | **12** | **12** |
| Grey / deferred | 16 | 0 | 0 | 0 |
| **Grand total** | **44** | **12** | **12** | **12** |

> **Next milestone:** port remaining Tier 1 classes (`ExponentFormat`, `HTMLFormat`) to unlock Tier 2.
> `UncertainValue2` is the highest-leverage single class: it unblocks 6 Tier 3 dependents.
