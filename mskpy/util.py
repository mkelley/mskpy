# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
util --- Short and sweet functions, generic algorithms
======================================================

.. autosummary::
   :toctree: generated/

   Mathmatical
   -----------

   archav
   cartesian
   davint
   Gaussian
   Gaussian2d
   deriv
   hav
   rotmat

   Searching, sorting
   ------------------

   between
   groupby
   nearest
   numalpha
   whist

   Statistics
   ----------

   kuiper
   kuiperprob
   mean2minmax
   meanclip
   midstep
   minmax
   nanmedian
   nanminmax
   randpl
   wmean

   "Special" functions
   -------------------

   bandpass
   deresolve
   Planck
   redden
   pcurve
   savitzky_golay

"""

import numpy as np
import astropy.units as u

__all__ = [
    'archav',
    'cartesian',
    'davint',
    'deriv',
    'Gaussian',
    'Gaussian2d',
    'hav',
    'rotmat',

    'between',
    'groupby',
    'nearest',
    'numalpha',
    'takefrom',
    'whist',

    'kuiper',
    'kuiperprob',
    'mean2minmax',
    'meanclip',
    'midstep',
    'minmax',
    'nanmedian',
    'nanminmax',
    'randpl',
    'uclip',
    'wmean',

    'bandpass',
    'deresolve',
    'Planck',
    'redden',
    'pcurve',
    'savitzky_golay'
]

def archav(y):
    """Inverse haversine.

    Haversine is (1 - cos(th)) / 2 = sin**2(th/2)

    Parameters
    ----------

    y : float or array like
      The angle.

    Returns
    -------

    th : astropy Quantity
      The inverse haversine.

    """
    return 2.0 * np.arcsin(np.sqrt(y)) * u.rad

def cartesian(*arrays):
    """Cartesian product of the input arrays.

    Parameters
    ----------

    arrays : array-like
      The arrays on which to operate.

    Returns
    -------

    result : ndarray
      The Cartesian product of (array[0] and array[1]) and array[2],
      etc.

    Examples
    --------

    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])
    """
    from itertools import product
    return np.array(list(product(*arrays)))

def davint(x, y, x0, x1):
    """Integrate a function using overlapping parabolas.
    
    Interface to davint.f from SLATEC at netlib.org.

    DAVINT integrates a function tabulated at arbitrarily spaced
    abscissas.  The limits of integration need not coincide with the
    tabulated abscissas.

    A method of overlapping parabolas fitted to the data is used
    provided that there are at least 3 abscissas between the limits of
    integration.  DAVINT also handles two special cases.  If the
    limits of integration are equal, DAVINT returns a result of zero
    regardless of the number of tabulated values.  If there are only
    two function values, DAVINT uses the trapezoid rule.

    Parameters
    ----------

    x : ndarray
      Abscissas, must be in increasing order.

    y : ndarray
      Function values.

    x0 : float
      Lower limit of integration.

    x1 : float
      Upper limit of integration.

    Returns
    -------

    r : float
      The result.

    """
    err = dict()
    err[2] = 'x1 was less than x0'
    err[3] = 'the number of x between x0 and x1 (inclusive) was less than 3 and neither of the two special cases described in the abstract occurred.  No integration was performed.'
    err[4] = 'the restriction x(i+1) > x(i) was violated.'
    err[5] = 'the number of function values was < 2'
    r, ierr = _davint(x, y, len(x), x0, x1)
    if ierr != 1:
        raise RuntimeError("DAVINT integration error: {}".format(err[ierr]))
    return r

def deriv(y, x=None):
    """The numerical derivative using 3-point Lagrangian interpolation.

    Parameters
    ----------

    y : array
      Variable to be differentiated, there must be at least 3 points

    x : array, optional
      Variable to differentiate with respect to; if equal to None,
      then use unit spacing

    Returns
    -------

    d : ndarray
      dy/dx

    Notes
    -----

    Based on deriv.pro from RSI/IDL, which is based on Hildebrand,
    1956, Introduction to Numerical Analysis.

    """

    if y.shape[0] < 3:
        raise ValueError("y must have at least 3 elements")

    if x is None:
        dydx = (np.roll(y, -1) - np.roll(y, 1)) / 2.0
        dydx[0] = (-3.0 * y[0] + 4.0 * y[1] - y[2]) / 2.0
        dydx[-1] = (3.0 * y[-1] - 4.0 * y[-2] + y[-3]) / 2.0
        return dydx

    if x.shape != y.shape:
        raise ValueError("y and x must have the same number of elements")
        return None

    xx = x.astype(float)
    x12 = xx - np.roll(xx, -1)           # x1 - x2
    x01 = np.roll(xx, 1) - xx            # x0 - x1
    x02 = np.roll(xx, 1) - np.roll(xx, -1) # x0 - x2

    # mid points
    dydx = (np.roll(y, 1) * (x12 / (x01 * x02)) +
            y * (1.0 / x12 - 1.0 / x01) -
            np.roll(y,-1) * (x01 / (x02 * x12)))

    # end points
    dydx[0] = (y[0] * (x01[1] + x02[1]) / (x01[1] * x02[1]) -
               y[1] *           x02[1]  / (x01[1] * x12[1]) +
               y[2] *           x01[1]  / (x02[1] * x12[1]))

    dydx[-1] = (-y[-3] *            x12[-2]  / (x01[-2] * x02[-2]) +
                 y[-2] *            x02[-2]  / (x01[-2] * x12[-2]) -
                 y[-1] * (x02[-2] + x12[-2]) / (x02[-2] * x12[-2]))

    return dydx

def Gaussian(x, mu, sigma):
    """A normalized Gaussian curve.

    Parameters
    ----------

    x : array
      Dependent variable.

    mu : float
      Position of the peak.

    sigma : float
      Width of the curve (sqrt(variance)).

    Returns
    -------

    G : ndarray
      The Gaussian function.

    """
    return (np.exp(-(x - mu)**2 / 2.0 / sigma**2) /
            np.sqrt(2.0 * pi) / sigma)

def Gaussian2d(shape, sigma, theta=0):
    """A normalized 2-D Gaussian function.

    Take care to make sure the result is normalized, if needed.

    Parameters
    ----------

    shape : tuple
      The shape of the resultant array.  The Gaussian will be centered
      at y = (shape[0] - 1) / 2, x = (shape[1] - 1) / 2.

    sigma : float or array
      Width of the Gaussian (sqrt(variance)).  If sigma is a
      two-element array, the first element will be the width along the
      first axis, and the second along the second axis.

    theta : float
      The angle for an elliptical Gaussian.  [degrees]

    Returns
    -------

    G : ndarray
      The 2D Gaussian function.

    """
    if not np.iterable(sigma):
        sy = sigma
        sx = sigma
    else:
        sy = sigma[0]
        sx = sigma[1]

    thr = np.radians(theta)
    a = np.cos(thr)**2 / 2.0 / sx**2 + np.sin(thr)**2 / 2.0 / sy**2
    b = np.sin(2 * thr) / 4.0 / sx**2 + np.sin(2 * thr) / 4.0 / sy**2
    c = np.sin(thr)**2 / 2.0 / sx**2 + np.cos(thr)**2 / 2.0 / sy**2

    y, x = np.indices(shape)
    y -= (shape[0] - 1) / 2.0
    x -= (shape[1] - 1) / 2.0

    G = np.exp(-(a * x**2 + 2 * b * x * y + c * y**2))
    G /= 2.0 * pi * sx * sy
    return G

def hav(th):
    """Haversine of an angle.

    Haversine is (1 - cos(th)) / 2 = sin**2(th/2)

    Parameters
    ----------

    th : float, astropy Quantity, or array
      The angle. [radians]

    Returns
    -------

    y : float or array like
      The haversine.

    """

    if isinstance(th, astropy.units.quantity.Quantity):
        return hav(th.to(u.rad))

    return np.sin(th / 2.0)**2

def rotmat(th):
    """Returns a rotation matrix.

    The matrix rotates the vector [x, y] by the amount a.

    Parameters
    ----------

    th : float or astropy Quantity
      The amount to rotate. [radians]

    Returns
    -------

    r : np.matrix
      Rotation matrix.

    Examples
    --------

    import numpy as np
    from mskpy import rotmat
    print np.array([1, 0]) * rotmat(radians(90.0))
    --> matrix([[  6.12323400e-17,   1.00000000e+00]])
    print ap.array([0, 1]) * rotmat(pi)
    --> matrix([[ -1.00000000e+00,   6.12323400e-17]])

    """
    if isinstance(th, astropy.units.quantity.Quantity):
        return rotmat(th.to(u.rad))
    c = np.cos(th)
    s = np.sin(th)
    return np.matrix([[c, s], [-s, c]])
