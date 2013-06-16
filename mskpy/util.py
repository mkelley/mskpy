# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
util --- Short and sweet functions, generic algorithms
======================================================

.. autosummary::
   :toctree: generated/

   Classes
   -------

   LongLat

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
   cmp_numalpha
   groupby
   nearest
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

   Variable manipulation
   ---------------------

   asAngle
   asQuantity

"""

import numpy as np
import astropy.units as u
from astropy.units import Quantity
import astropy.coordinates as coords

__all__ = [
    'archav',
    'cartesian',
    'davint',
    'deriv',
    'Gaussian',
    'Gaussian2d',
    'hav',
    'rotmat',
    'spherical_coord_rotate',
    'vector_rotate',

    'between',
    'cmp_numalpha',
    'groupby',
    'nearest',
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
    'sigma',
    'spearman',
    'uclip',
    'wmean',

    'bandpass',
    'deresolve',
    'Planck',
    'redden',
    'polcurve',
    'savitzky_golay',

    'asAngle'
    'asQuantity'
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
    th : Quantity
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
    """Integrate an array using overlapping parabolas.
    
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
    from lib import davint as _davint
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
    th : float, Quantity, or array
      The angle. [radians]

    Returns
    -------
    y : float or array like
      The haversine.

    """

    th = asQuantity(th, u.rad)
    return np.sin(th / 2.0)**2

def rotmat(th):
    """Returns a rotation matrix.

    The matrix rotates the vector [x, y] by the amount a.

    Parameters
    ----------
    th : float or Quantity
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
    th = asQuantity(th, u.rad)
    c = np.cos(th)
    s = np.sin(th)
    return np.matrix([[c, s], [-s, c]])

def spherical_coord_rotate(lon0, lat0, lon1, lat1, lon, lat):
    """Rotate about an axis defined by two reference points.

    Given two reference points (lon0, lat0), and (lon1, lat1), rotate
    (lon, lat) in the same manner that (lon0, lat0) needs to be
    rotated to match (lon1, lat1).

    Parameters
    -----------
    lon0, lat0 : float
      The reference point.  [degrees]

    lon1, lat1 : float
      A second reference point that defines the rotation axis and
      direction.  [degrees]
    lon, lat : float or array-like
      The point(s) to rotate [degrees]

    Returns
    -------
    lon_new, lat_new : float or array-like
      lon, lat rotated in the sense as lon0, lat0 must be rotated to
      produce lon1, lat1.  [degrees]

    Notes
    -----

    Based on the IDL routine spherical_coord_rotate.pro written by
    J.D. Smith, and distributed with CUBISM.

    """

    if (lon0 == lon1) and (lat0 == lat1):
        return (lon, lat)

    def rd2cartesian(lon, lat):
        # convert to cartesian coords
        clat = np.cos(lat)
        return np.array([clat * np.cos(lon),
                            clat * np.sin(lon),
                            np.sin(lat)])
    v0 = rd2cartesian(np.radians(lon0), np.radians(lat0))
    v1 = rd2cartesian(np.radians(lon1), np.radians(lat1))
    v  = rd2cartesian(np.radians(lon), np.radians(lat))

    # construct coordinate frame with x -> ref point and z -> rotation
    # axis
    x = v0
    z = np.cross(v1, v0)  # rotate about this axis
    z = z / np.sqrt((z**2).sum())  # normalize
    y = np.cross(z, x)
    y = y / np.sqrt((y**2).sum())

    # construct a new coordinate frame (x along new direction)
    x2 = v1
    y2 = np.cross(z, x2)
    y2 = y2 / np.sqrt((y2**2).sum())

    # project onto the inital frame, the re-express in the rotated one
    if len(v.shape) == 1:
        v = (v * x).sum() * x2 + (v * y).sum() * y2 + (v * z).sum() * z
    else:
        vx = np.dot(v.T, x)
        vy = np.dot(v.T, y)
        vz = np.dot(v.T, z)
        v  = vx * np.repeat(x2, v.shape[1]).reshape(v.shape)
        v += vy * np.repeat(y2, v.shape[1]).reshape(v.shape)
        v += vz * np.repeat(z,  v.shape[1]).reshape(v.shape)

    lat_new = np.degrees(np.arcsin(v[2]))
    lon_new = np.degrees(np.arctan2(v[1], v[0]))

    lon_new = lon_new % 360.0

    return (lon_new, lat_new)

def vector_rotate(r, n, th):
    """Rotate vector `r` an angle `th` CCW about `n`.

    Parameters
    ----------
    r : array
      The vector to rotate [x, y, z].
    n : array
      The vector to rotate about.
    th : float, array, Angle, or Quantity
      The CCW angle to rotate by. [float/array in degrees]

    Returns
    -------
    rp : array
      The rotated vector [x, y, z].

    Notes
    -----
    Described in Goldstein p165, 2nd ed. Note that Goldstein presents
    the formula for clockwise rotation.

    """

    from astropy.coordinates import Angle

    if isinstance(th, Angle):
        theta = Angle.radians
    elif isinstance(th, Quantity):
        theta = th.to(u.rad).value
    else:
        theta = np.radians(th)

    nhat = n / np.sqrt((n**2).sum())

    def rot(r, nhat, theta):
        return (r * np.cos(-theta) +
                nhat * (nhat * r).sum() * (1.0 - np.cos(-theta)) +
                np.cross(r, nhat) * np.sin(-theta))

    if theta.size == 1:
        return rot(r, nhat, theta)
    else:
        return np.array([rot(r, nhat, t) for t in theta])

def between(a, limits, closed=True):
    """Return True for elements within the given limits.

    Parameters
    ----------
    a : array
      Array to test.
    limits : array
      A 2-element array of the lower- and upper-limits, or an Nx2
      element array of lower- and upper-limits where limits[i] is a
      set of upper- and lower-limits.
    closed : bool, optional
      Set to True and the interval will be closed (i.e., use <= and >=
      at the limits).

    Returns
    -------
    i : ndarray
      True where a is between each set of limits.

    """

    b = np.array(a)
    lim = np.array(limits)

    if len(lim.shape) == 1:
        if closed:
            i = (a >= lim[0]) * (a <= lim[1])
        else:
            i = (a > lim[0]) * (a < lim[1])
    else:
        i = np.zeros(b.shape)
        for j in xrange(lim.shape[0]):
            i += between(a, lim[j,:])

    return i.astype(bool)

def cmp_numalpha(x, y):
    """Compare two strings, considering leading multidigit integers.

    A normal string comparision will compare the strings character by
    character, e.g., "101P" is less than "1P" because "0" < "P".
    `cmp_numalpha` will instead consider the leading multidigit
    integer, e.g., "101P" > "1P" because 101 > 1.

    Parameters
    ----------
    x, y : strings
      The strings to compare.

    Returns
    -------
    cmp : integer
      -1, 0, 1 if x < y, x == y, x > y.

    """

    na = re.compile('^([0-9]*)(.*)')
    mx = na.findall(x)[0]
    my = na.findall(y)[0]

    if (len(mx[0]) == 0) and (len(my[0]) == 0):
        if x > y:
            return 1
        elif x < y:
            return -1
        else:
            return 0
    elif len(mx[0]) == 0:
        return 1
    elif len(my[0]) == 0:
        return -1
    else:
        xx = int(mx[0])
        yy = int(my[0])
        if xx < yy:
            return -1
        if xx > yy:
            return 1
        else:
            return cmp_numalpha(mx[1], my[1])

def groupby(key, *lists):
    """Sort elements of `lists` by `unique(key)`.

    Note: this is not the same as `itertools.groupby`.

    Parameters
    ----------
    key : array
      A set of keys that indicate how to group the elements of each
      list.
    lists : array
      Lists to sort.

    Returns
    -------
    groups : dictionary
      A dictionary, where the keys are `unqiue(key)`, and the values
      are tuples of `list` corresponding to sorted entries from
      `lists`.  Does that make sense?

    Examples
    --------
    >>> import numpy as np
    >>> from mskpy.util import groupby
    >>> keys = (np.random.rand(26) * 3).astype(int)
    >>> print keys
    [1 2 2 0 1 1 1 1 1 1 2 1 2 1 0 0 0 1 2 2]
    >>> lists = (list('abcdefghijklmnopqrstuvwxyz'), range(26))
    >>> groupby(keys, *lists)
    {0: (['d', 'o', 'p', 'q'], [3, 14, 15, 16]),
     1: (['a', 'e', 'f', 'g', 'h', 'i', 'j', 'l', 'n', 'r'],
         [0, 4, 5, 6, 7, 8, 9, 11, 13, 17]),
     2: (['b', 'c', 'k', 'm', 's', 't'], [1, 2, 10, 12, 18, 19])}

    """

    groups = dict()
    key = np.asarray(key)
    for k in np.unique(key):
        i = np.flatnonzero(key == k)
        groups[k] = ()
        for l in lists:
            groups[k] += (list(np.asarray(l)[i]),)
    return groups

def nearest(array, v):
    """Return the index of `array` where the value is nearest `v`.

    Parameters
    ----------
    array : array
      An array.
    v : scalar
      The requested value.

    Returns
    -------
    result : int
      The index.

    """
    return np.abs(np.array(a) - v).argmin()

def takefrom(arrays, indices):
    """Return elements from each array at the given indices.

    Parameters
    ----------
    arrays : tuple of arrays
      The arrays to index.
    indices : array
      The indices to return from each array in `a`.

    Returns
    -------
    r : tuple of arrays
      a[0][indices], a[1][indices], etc.

    """

    r = ()
    for a in arrays:
        r += (a[indices],)
    return r

def whist(x, y, w, errors=True, **keywords):
    """A weighted histogram binned by an independent variable.

    Parameters
    ----------
    x : array-like
      The independent variable.
    y : array-like
      The parameter to average.
    w : array-like
      The weights for each `y`.  If `errors` is `True`, then `x` will
      be weighted by `1 / w**2`.
    errors : bool, optional
      Set to `True` if `w` is an array of uncertainties on `x`, and
      not the actual weights.
    **keywords : optional
      Any `numpy.histogram` keyword, except `weights`.

    Returns
    -------
    h : ndarray
      The weighted mean of `y`, binned by `x`.
    err : ndarray
      When `errors` is `True`, `err` will be the uncertainty on `h`,
      otherwise it will be `None`.
    n : ndarray
      The number of `x`'s in each bin.
    edges: ndarray
      The bin edges.

    """

    if keywords.has_key('weights'):
        raise RuntimeError('weights not allowed in keywords')

    _x = np.array(x)
    _y = np.array(y)
    _w = np.array(w)

    if errors:
        _w = 1.0 / _w**2

    n, edges = np.histogram(x, **keywords)
    n = n.astype(float)

    num = np.histogram(x, weights=_y * _w, **keywords)[0]
    den = np.histogram(x, weights=_w, **keywords)[0].astype(float)
    m = num / den

    if errors:
        err = 1.0 / np.sqrt(den)
    else:
        err = None

    return m, err, n, edges

def kuiper(x, y):
    """Compute Kuiper's statistic and probablity.

    Parameters
    ----------
    x, y : array-like
    The two distributions to compare.

    Returns
    -------
    V : float
      Kuiper's statistic.
    p : float
      The probability that `V` > observed may occur for uncorrelated
      data sets.

    Notes
    -----

    Based on p. 627 of Press et al. (1992, Numerical Recipies in C,
    2nd Ed.), and scipy.stats.ks_2samp.

    """

    data1, data2 = map(np.asarray, (x, y))
    n1 = data1.shape[0]
    n2 = data2.shape[0]
    data1 = np.sort(data1)
    data2 = np.sort(data2)
    data_all = np.sort(np.concatenate([data1, data2]))
    cdf1 = np.searchsorted(data1, data_all, side='right') / float(n1)
    cdf2 = np.searchsorted(data2, data_all, side='right') / float(n2)
    V = np.ptp(cdf1 - cdf2)
    Ne = n1 * n2 / float(n1 + n2)
    return V, kuiperprob(V, Ne)

def kuiperprob(V, Ne):
    """The probability of a false positive in Kuiper's test.

    Parameters
    ----------
    V : float
      The Kuiper statistic.
    Ne : int
      Effective sample size (i.e., `n1 * n2 / (n1 + n2)`).

    Returns
    -------
    p : float
      The probability of a false positive.

    Notes
    -----
    Based on prob_kuiper.pro from Astro IDL library.

    """

    # Numerical Recipes algorithm:
    lam = (np.sqrt(Ne) + 0.155 + 0.24 / np.sqrt(Ne)) * V
    if lam <= 0.4:
        # good to 7 sig. figs.
        return 1.0

    EPS1 = 0.001
    EPS2 = 1e-8
    p = 0.0
    termbf = 0.0
    a2 = -2 * lam**2
    for j in range(1, 101):
        a2j2 = a2 * j**2
        term = 2 * (-2 * a2j2 - 1) * np.exp(a2j2)
        p += term
        if (abs(term) <= (EPS1 * termbf)) or (abs(term) <= (EPS2 * p)):
            return p
        termbf = abs(term)
    return 1.0  # did not converge        

def mean2minmax(a):
    """The distance from the mean to the min and max of `a`.

    This function is suitable for computing asymetric errorbars for
    matplotlib.errorbar (the result will need to be reshaped to a 2x1
    array).

    Parameters
    ----------
    a : array

    Returns
    -------
    result : ndarray
      A two-element `ndarray`, the first element is `mean(a) -
      min(a)`, the second is `max(a) - mean(a)`.

    """
    return np.abs(minmax(a) - np.array(a).mean())

def meanclip(x, axis=None, lsig=3.0, hsig=3.0, maxiter=5, minfrac=0.02,
             full_output=False):
    """Average `x` after iteratively removing outlying points.

    Clipping is performed about the median.  NaNs are ignored.

    Parameters
    ----------
    x : array
    axis : int, optional
      Set to `None` to clip the entire array, or an integer to clip
      over that axis.
    lsig : float or tuple, optional
      The lower-sigma-rejection limit.  If `lsig` is a `tuple`, then
      the contents will be placed into the keyword parameters (for
      compatibility with functions like np.apply_along_axis()).
    hsig : float, optional
      The upper-sigma-rejection limit
    maxiter : int, optional
      The maximum number of clipping iterations.
    minfrac : float, optional
      Stop iterating if less than or equal to `minfrac` of the data
      points are rejected.
    full_output : bool, optional
      If `True`, also return the standard deviation of the clipped
      data, their indicies, and the number of iterations.

    Returns
    -------
    mean : float
      The mean of the clipped data.
    sigma : float, optional
      The standard deviation of the clipped data.
    good : ndarray, optional
      The indices of the good data.
    iter : int, optional
      The number of clipping iterations used.

    .. Todo::
      Look into using scipy.stats.tmean, tstd for meanclip.

    """

    if axis is not None:
        if axis < len(x.shape):
            x2 = np.rollaxis(x, axis)
            y = np.zeros(x2.shape[0])
            ys = np.zeros(x2.shape[0])
            yind = ()
            yiter = np.zeros(x2.shape[0])
            for i in xrange(x2.shape[0]):
                mc = meanclip(x2[i], axis=None, lsig=lsig, hsig=hsig,
                              maxiter=maxiter, minfrac=minfrac,
                              full_output=True)
                y[i], ys[i], yiter[i] = mc[0], mc[1], mc[3]
                yind += (mc[2],)
            if full_output:
                return y.mean(), ys, yind, yiter
            else:
                return y.mean()
        else:
            raise ValueError("There is no axis {0} in the input array".format(axis))

    if isinstance(lsig, tuple):
        lsig = list(lsig)
        if len(lsig) == 5:
            full_output = lsig.pop()
        if len(lsig) >= 4:
            minfrac = lsig.pop()
        if len(lsig) >= 3:
            maxiter = lsig.pop()
        if len(lsig) >= 2:
            hsig = lsig.pop()
        if len(lsig) >= 1:
            lsig = lsig.pop()

    good = np.flatnonzero(np.isfinite(x))
    if good.size == 0:
        # no good data
        if full_output:
            return np.nan, np.nan, (), 0
        else:
            return np.nan  

    for i in range(maxiter):
        y = x.flatten()[good]
        medval = np.median(y)
        sig = y.std()

        keep = (y > (medval - lsig * sig)) * (y < (medval + hsig * sig))
        cutfrac = float(abs(good.size - keep.sum())) / good.size

        if keep.sum() > 0:
            good = good[keep]
        else:
            break  # somehow all the data were clipped

        if cutfrac <= minfrac:
            break

    y = x.flatten()[good]
    if full_output:
        return y.mean(), y.std(), good, i+1
    else:
        return y.mean()

def midstep(a):
    """Compute the midpoints of each step in `a`.

    Parameters
    ----------
    a : array

    Returns
    -------
    b : ndarray
      The midsteps of `a`, i.e., `b = (a[1:] + a[:-1]) / 2.0`.

    """
    return (np.array(a)[1:] + np.array(a)[:-1]) / 2.0

def minmax(a):
    """Compute the minimum and the maximum of an array.

    Parameters
    ----------
    a : array

    Returns
    -------
    result : ndarray
      A two-element array, the first element is `min(a)`, the second
      is `max(a)`.

    """
    return np.array([np.min(a), np.max(a)])

def nanmedian(a, axis=None):
    """Median of `a`, ignoring NaNs.

    Parameters
    ----------
    a : array

    Returns
    -------
    m : ndarray
      The median, or `nan` if all of `a` is `nan`.

    """
    if axis is not None:
        return np.apply_along_axis(nanmedian, axis, a)

    a = np.array(a)
    i = ~np.isnan(a)
    if np.any(i):
        return np.median(a[i])
    else:
        return np.nan

def nanminmax(a):
    """Compute the minimum and the maximum of an array, ignoring NaNs.

    Parameters
    ----------
    a : array

    Returns
    -------
    result : ndarray
      A two-element array, the first element is `nanmin(a)`, the
      second is `nanmax(a)`.

    """
    return np.array([np.nanmin(a), np.nanmax(a)])

def randpl(x0, x1, k, n=1):
    """Pick random deviates from a power-law distribution.

    This returns:
      .. math:: dn/dx \propto x**k
    For:
      .. math:: dn/dlog(x) \propto x**alpha
    set `k = alpha - 1`.

    Parameters
    ----------
    x0 : float
      The minimum value to pick.
    x1 : float
      The maximum value to pick.
    k : float
      The logarithmic slope of the distribution.
    n : int, optional
      The number to pick.

    Returns
    -------
    y : float or ndarray
      The random number(s).

    Notes
    -----
    Algorithm from Weisstein, Eric W. "Random Number." From
    MathWorld--A Wolfram Web Resource.
    http://mathworld.wolfram.com/RandomNumber.html

    """

    y = np.random.rand(n)
    return ((x1**(k + 1) - x0**(k + 1)) * y + x0**(k + 1))**(1.0 / (k + 1))

def sigma(s):
    """The probablity a normal variate will be `<s` sigma from the mean.

    Parameters
    ----------
    s : float
      The number of sigma from the mean.

    Returns
    -------
    p : float
      The probability that a value within +/-s would occur.

    """
    from scipy.special import erf
    return 0.5 * (erf(s / np.sqrt(2.0)) - erf(-s / np.sqrt(2.0)))

def spearman(x, y, nmc=None, xerr=None, yerr=None):
    """Perform a Spearman "rho" test on two or more data sets.

    Parameters
    ----------
    x, y : array
      The parameters being tested.
    nmc : int
      The number of Monte Carlo tests to perform.
    xerr, yerr : array, optional
      If Monte Carlo tests are requested, use these 1 sigma
      uncertainties for each value of x and/or y, assumed to be
      normally distributed.  Set to None for no errors.

    Returns
    -------
    r : float or ndarray
      The Spearman correlation coefficient between x and y.
    p : float or ndarray
      The probability that a value greater than r may occur in
      uncorrelated data sets.  According to scipy.stats.spearmanr p
      may not be reliable for datasets smaller 500.
    Z : float or ndarray
      The significance of r expressed in units of standard deviations
      based on the expectation value and variance of the null
      hypothesis that x and y are uncorrelated.
    meanZ : float or ndarray, optional
      The average Z measured in the Monte Carlo tests.
    n : float or ndarray, optional
      The number of Monte Carlo runs for which Z was greater than 3
      sigma.

    """

    def spearmanZ(x, y):
        N = len(x)
        rankx = stats.rankdata(x)
        ranky = stats.rankdata(y)

        # find the corrections for ties
        ties = stats.mstats.count_tied_groups(x)
        sx = sum((k**3 - k) * v for k, v in ties.iteritems())
        ties = stats.mstats.count_tied_groups(y)
        sy = sum((k**3 - k) * v for k, v in ties.iteritems())

        D = sum((rankx - ranky)**2)
        meanD = (N**3 - N) / 6.0 - (sx + sy) / 12.0
        varD = (N - 1) * N**2 * (N + 1)**2 / 36.0
        varD *= (1 - sx / (N**3 - N)) * (1 - sy / (N**3 - N))
        return abs(D - meanD) / np.sqrt(varD)
 
    N = len(x)

    rp = stats.mstats.spearmanr(x, y, use_ties=True)
    r = rp[0]
    p = rp[1].data[()]
    Z = spearmanZ(x, y)

    if nmc is not None:
        if xerr is None:
            xerr = np.zeros(N)
        if yerr is None:
            yerr = np.zeros(y.shape)

        mcZ = np.zeros(nmc)
        for i in xrange(nmc):
            dx = np.random.randn(N) * xerr
            dy = np.random.rand(N) * yerr
            mcZ[i] = spearmanZ(x + dx, y + dy)
        meanZ = mcZ.mean()
        n = sum(mcZ > 3.0)
        return r, p, Z, meanZ, n

    return r, p, Z

def uclip(x, ufunc, full_output=False, **keywords):
    """Sigma clip data and apply the function ufunc.

    Clipping is done by `meanclip`.

    Parameters
    ----------
    x : array
    ufunc : function
      A function to apply to the sigma clipped `x`.
    **keywords
      Any `meanclip` keyword.

    Returns
    -------
    y : 
      The result.
    ind : ndarray, optional
      The array indices of the good data in `x.flatten()`.
    iter : int, optional
      The number of clipping iterations used.

    """

    mc = meanclip(x, full_output=True, **keywords)
    if full_output:
        return ufunc(x.flatten()[mc[2]]), mc[2], mc[3]
    else:
        return ufunc(x.flatten()[mc[2]])

def wmean(x, w, errors=True, axis=None):
    """The weighted mean.

    Parameters
    ----------
    x : array
      The parameter to average.
    w : array
      The weights for each `x`.  If `errors` is `True`, then `x` will
      be weighted by `1 / w**2`.
    errors : bool, optional
      Set to `True` if `w` is an array of uncertainties on `x`, and
      not the actual weights.
    axis : int, optional
      Set to the axis over which to average.

    Returns
    -------
    m : float
      The weighted mean.
    merr : float
      The uncertainty on `m` (when `errors` is `True`).

    """

    _x = np.array(x)
    _w = np.array(w)

    if errors:
        _w = 1.0 / _w**2

    m = np.sum(_x * _w, axis=axis) / np.sum(_w, axis=axis)

    if errors:
        merr = np.sqrt(1.0 / np.sum(_w, axis=axis))
        return m, merr
    else:
        return m

def bandpass(sw, sf, se, fw=None, ft=None, filter=None, filterdir=None,
             s=None):
    """Filters a spectrum given a transimission function.

    If the filter has a greater spectreal dispersion than the
    spectrum, the spectrum is interpolated onto the filter's
    wavelengths.  Otherwise, the filter is interpoalted onto the
    spectrum's wavelengths.

    Either fw+ft or filter must be given.

    Parameters
    ----------
    sw : array
      Spectrum wavelengths.
    sf : array
      Spectrum flux per unit wavelength.
    se : array
      Error in `sf`.
    fw : array, optional
      Filter transmission profile wavelengths, same units as `sw`.
    ft : array, optional
      Filter transmission profile.
    filter : string, optional
      The name of a filter (see `calib.filtertrans`).
    filterdir : string, optional
      The directory containing the filter transmission files
      (see `calib.filtertrans`).
    s : float, optional
      Interpolation smoothing.  See scipy.interpolate.splrep().

    Returns
    -------
    wave, flux, err : ndarray
      The effective wavelength, flux density, and error of the
      filtered spectrum.

    """
    from . import calib

    # local copies
    _sw = sw.copy()
    _sf = sf.copy()
    _se = se.copy()

    if (fw is not None) and (ft != None):
        _fw = fw.copy()
        _ft = ft.copy()
    elif filter is not None:
        _fw, _ft = calib.filtertrans(filter)
    else:
        raise ValueError("Neither fw+ft nor filter was supplied.")

    # We need a scale for the errorbars since 1/err^2 can be fairly large
    errscale = _se.mean()
    _se = _se / errscale

    # determine if the spectrum or filter has the greater dispersion
    if np.median(_fw / deriv(_fw)) > np.median(_sw / deriv(_sw)):
        # interpolate the spectrum onto the filter wavelengths
        # the spectrum may be incomplete
        i = (_fw >= min(_sw)) * (_fw <= max(_sw))
        _fw = _fw[i]
        _ft = _ft[i]

        _w = _fw
        #_sf = interpolate.interp1d(_sw, _sf, kind=kind)(_w)
        #_se2 = interpolate.interp1d(_sw, _se**2, kind=kind)(_w)
        spl = interpolate.splrep(_sw, _sf)
        _sf = interpolate.splev(_w, spl)
        spl = interpolate.splrep(_sw, _se**2)
        _se2 = interpolate.splev(_w, spl)
        _ft = _ft
    else:
        # the spectrum or filter transmission may be incomplete
        # interpolate the filter onto the spectrum wavelengths
        i = (_sw >= min(_fw)) * (_sw <= max(_fw))
        _sw = _sw[i]
        _sf = _sf[i]
        _se = _se[i]

        _w = _sw
        #_ft = interpolate.interp1d(_fw, _ft, kind=kind)(_w)
        spl = interpolate.splrep(_fw, _ft)
        _ft = interpolate.splev(_w, spl)
        _sf = _sf
        _se2 = _se**2

    # weighted mean to get the effective wavelength
    wave = integrate.trapz(_w * _ft * _sf, _w) / \
        integrate.trapz(_ft * _sf, _w)

    # flux is the weighted average using the transmission and
    # errorbars as weights
    weights = _ft / _se2
    flux = integrate.trapz(_sf * weights, _w) / integrate.trapz(weights, _w)
    err = np.sqrt(integrate.trapz(weights, _w) /
                     integrate.trapz(1.0 / _se2, _w))
    err *= errscale

    return wave, flux, err

def deresolve(func, wave, flux, err=None):
    """De-resolve a spectrum using the supplied instrument profile.

    Parameters
    ----------
    func : function or string
      The instrument profile/weighting function.  The function only
      takes one parameter: delta-wavelength (distance from the center
      of the filter) in the same units as `wave`.  Some shortcut
      strings are allowed (case insensitive):
        "Gaussian(sigma)" - specifiy sigma in the same units as `wave`
        "uniform(fwhm)" - specifiy fwhm in the same units as `wave`
    wave : ndarray
      The wavelengths of the spectrum.
    flux : ndarray
      The spectral flux.
    err : ndarray, optional
      The uncertainties on `flux`.  If provided, the fluxes will be
      weighted by `1/err**2` before deresolving.

    Results
    -------
    f : ndarray
      The de-resolved fluxes.

    """

    if type(func) is str:
        if 'gaussian' in func.lower():
            sigma = float(re.findall('gaussian\(([^)]+)\)', func)[0])
            def func(dw):
                return Gaussian(dw, 0, sigma)
        elif 'uniform' in func.lower():
            hwhm = float(re.findall('uniform\(([^)]+)\)', func)[0]) / 2.0
            def func(dw):
                f = np.zeros_like(dw)
                i = (dw > hwhm) * (dw <= hwhm)
                if any(i):
                    f[i] = 1.0
                return f
        else:
            raise ValueError("Function '{}' not recognized.".format(func))

    if err is not None:
        weights = err**-2
        sumWeights = 1.0 / np.sqrt(deresolve(func, wave, weights))
    else:
        weights = 1.0
        sumWeights = 1.0

    wflux = flux * weights
    fluxout = np.zeros_like(wflux)
    
    for i in range(len(wave)):
        dw = wave - wave[i]
        f = func(dw)
        f /= f.sum()
        fluxout[i] = np.sum(f * wflux) / sumWeights

    return fluxout

def Planck(wave, T, unit=u.Unit('MJy/sr'), deriv=None):
    """The Planck function.

    Parameters
    ----------
    wave : array or Quantity
      The wavelength(s) to evaluate the Planck function. [micron]
    T : float, array, Quantity
      The temperature(s) of the Planck function. [Kelvin]
    unit : u.Unit
      The output units.
    deriv : string
      Set to 'T' to return the first derivative with respect to
      temperature.

    Returns
    -------
    B : Quantity

    """

    from astropy import constants as const

    # prevent over/underflow warnings
    oldseterr = np.seterr(all='ignore')

    wave = asQuantity(wave, u.um)
    T = asQuantity(T, u.K)

    c1 = 2.0 * const.si.h * const.si.c
    c2 = const.si.h * const.si.c / const.si.k_B
    a = np.exp(c2 / wave.si / T.to(u.K))
    B = c1 / ((wave.si)**3 * (a - 1.0)) / u.sr

    if deriv is not None:
        if deriv.lower() == 't':
            B *= c2 / T.to(u.K)**2 / wave.si * a / (a - 1.0)
            unit /= u.K

    equiv = u.equivalencies.spectral_density(u.m, wave.si)
    B = B.to(unit, equivalencies=equiv)

    # restore seterr
    np.seterr(**oldseterr)

    return B

def redden(wave, S, wave0=0.55):
    """Redden a spectrum with the slope S.

    Parameters
    ----------
    wave : array
      An array of wavelengths.
    S : float or array
      Redden the spectrum by the fraction `S` per unit wavelength.
      `S` should be defined for each wavelength `wave`, or be a single
      value for all wavelengths.
    wave0 : float, optional
      The wavelength to hold constant.

    Returns
    -------
    spec : ndarray
      The scale factors to produce the reddened spectrum.

    Examples
    --------
    Comet dust slopes are typically described as % per 0.1 um

    >>> import numpy as np
    >>> from mskpy.util import redden
    >>> wave = np.array([0.4, 0.45, 0.5, 0.55, 0.65, 1.55])
    >>> S = 12. * 0.01 / 0.1  # 12% / (0.1 um)
    >>> print redden(wave, S)
    [ 0.83527021  0.88692044  0.94176453  1.          1.12749685  3.32011692]

    """

    from scipy.integrate import quad
    from scipy.interpolate import interp1d

    if not np.iterable(wave):
        wave = np.array(wave).reshape(1)

    if not np.iterable(S):
        S = np.ones_like(wave) * S

    slope = interp1d(np.r_[0, wave, np.inf], np.r_[S[0], S, S[-1]],
                     kind='linear')

    spec = np.zeros_like(wave)
    for i in xrange(len(wave)):
        # integrate S*dwave from wave0 to wave[i]
        intS = quad(slope, wave0, wave[i], epsabs=1e-3, epsrel=1e-3)[0]
        spec[i] = np.exp(intS)

    return spec

def polcurve(th, p, a, b, th0):
    """The comet polarization versus phase angle curve.

    Levasseur-Regourd et al. 1996:
      .. math:: P(th) = p * sin(th)^a  * cos(th / 2)^b * sin(th - th0)

    Parameters
    ----------
    th : float, array, or Quantity
      The phase angle.  [degrees]
    p, a, b : float
      The parameters of the function.
    th0 : float or Quantity
      The negative to positive branch turnover angle. [degrees]

    Returns
    -------
    P : float or array
      The polarization at phase angle `th`.

    """
    th = asQuantity(th, u.deg).to(u.rad)
    th0 = asQuantity(th0, u.deg).to(u.rad)
    return (p * np.sin(th.value)**a * np.cos(th.value / 2.)**b *
            np.sin(th.value - th0.value))

def savitzky_golay(x, kernel=11, order=4):
    """Smooth with the Savitzky-Golay filter.

    Parameters
    ----------
    x : array
    kernel : int, optional
      A positive odd integer giving the kernel size.  `kernel > 2 + order`.
    order : int, optional
      Order of the polynomal.

    Returns
    -------
    smoothed : ndarray
      The smoothed `x`.

    Notes
    -----

    From the SciPy Cookbook,
    http://www.scipy.org/Cookbook/SavitzkyGolay, 01 Dec 2009

    """

    if (kernel % 2) != 1 or kernel < 1:
        raise ValueError("kernel size must be a positive odd number, was:{}".format(kernel))
    if kernel < order + 2:
        raise ValueError("kernel is to small for the polynomals\nshould be > order + 2")

    # a second order polynomal has 3 coefficients
    order_range = range(order + 1)
    half_window = (kernel - 1) // 2
    b = np.mat([[k**i for i in order_range]
                for k in range(-half_window, half_window+1)])

    # since we don't want the derivative, else choose [1] or [2], respectively
    m = np.linalg.pinv(b).A[0]
    window_size = len(m)
    half_window = (window_size - 1) // 2

    # precompute the offset values for better performance
    offsets = range(-half_window, half_window + 1)
    offset_data = zip(offsets, m)

    # temporary data, extended with a mirror image to the left and right
    firstval = data[0]
    lastval = data[len(data) - 1]
    # left extension: f(x0-x) = f(x0)-(f(x)-f(x0)) = 2f(x0)-f(x)
    # right extension: f(xl+x) = f(xl)+(f(xl)-f(xl-x)) = 2f(xl)-f(xl-x)
    leftpad = np.zeros(half_window) + 2 * firstval
    rightpad = np.zeros(half_window) + 2 * lastval
    leftchunk = data[1:(1 + half_window)]
    leftpad = leftpad-leftchunk[::-1]
    rightchunk = data[len(data) - half_window - 1:len(data) - 1]
    rightpad = rightpad - rightchunk[::-1]
    data = np.concatenate((leftpad, data))
    data = np.concatenate((data, rightpad))

    smooth_data = list()
    for i in range(half_window, len(data) - half_window):
        value = 0.0
        for offset, weight in offset_data:
            value += weight * data[i + offset]
        smooth_data.append(value)

    return np.array(smooth_data)

def asAngle(x, unit=None):
    """Make `x` an astropy `Angle`.

    Parameters
    ----------
    x : float, array, Quantity, Angle
    unit : astropy.units.Unit, optional
      The units of `x`.  Required if `x` is not a `Quantity` or `Angle`.

    Returns
    -------
    a : Angle

    """
    from astropy.coordinates import Angle

    if not isinstance(x, Angle):
        if isinstance(x, Quantity):
            a = Angle(x.value, x.unit)
        else:
            a = Angle(x, unit)

    return a

def asQuantity(x, unit, **keywords):
    """Make `x` a Quantity with units `unit`.

    Parameters
    ----------
    x : float, array, Quantity
    unit : astropy.units.Unit
    **keywords
      Additional keywords are passed to `Quantity.to`.

    Returns
    -------
    q : Quantity
      `x` in units `unit`.

    """

    if not isinstance(x, Quantity):
        q = x * unit

    return q.to(unit, **keywords)
