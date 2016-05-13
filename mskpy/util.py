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
   gaussian
   gaussian2d
   deriv
   hav
   rotmat

   FITS images and WCS
   -------------------
   basicwcs
   fitslog
   getrot

   Optimizations
   -------------
   gaussfit
   glfit
   linefit
   planckfit

   Searching, sorting
   ------------------
   between
   clusters
   groupby
   leading_num_key
   nearest
   takefrom
   stat_avg
   whist

   Spherical/Celestial/vectorial geometry
   --------------------------------------
   ec2eq
   lb2xyz
   projected_vector_angle
   spherical_coord_rotate
   state2orbit
   vector_rotate
   xyz2lb

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
   sigma
   spearman
   uclip

   "Special" functions
   -------------------
   bandpass
   deresolve
   phase_integral
   planck
   #redden
   polcurve
   savitzky_golay

   Time
   ----
   cal2doy
   cal2iso
   cal2time
   date_len
   date2time
   dh2hms
   doy2md
   hms2dh
   jd2doy
   jd2time
   timestamp
   tz2utc

   Other
   -----
   asAngle
   asQuantity
   asValue
   autodoc
   file2list
   horizons_csv
   spectral_density_sb
   timesten
   write_table

"""

from functools import singledispatch
import datetime
import numpy as np
import astropy.time

__all__ = [
    'archav',
    'cartesian',
    'davint',
    'deriv',
    'gaussian',
    'gaussian2d',
    'hav',
    'rotmat',

    'basicwcs',
    'fitslog',
    'getrot',

    'gaussfit',
    'glfit',
    'linefit',
    'planckfit',

    'between',
    'clusters',
    'groupby',
    'leading_num_key',
    'nearest',
    'stat_avg',
    'takefrom',
    'whist',

    'ec2eq',
    'lb2xyz',
    'projected_vector_angle',
    'spherical_coord_rotate',
    'state2orbit',
    'vector_rotate',
    'xyz2lb',

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

    'bandpass',
    'deresolve',
    'phase_integral',
    'planck',
#    'redden',
    'polcurve',
    'savitzky_golay',

    'cal2doy',
    'cal2iso',
    'cal2time',
    'date_len',
    'date2time',
    'dh2hms',
    'doy2md',
    'hms2dh',
    'jd2doy',
    'jd2time',
    'timestamp',
    'tz2utc',

    'asAngle',
    'asQuantity',
    'asValue',
    'autodoc',
    'file2list',
    'spectral_density_sb',
    'timesten',
    'write_table'
]

def archav(y):
    """Inverse haversine.

    Haversine is (1 - cos(th)) / 2 = sin**2(th/2)

    Parameters
    ----------
    y : float or array
      The value.

    Returns
    -------
    th : float or ndarray
      The inverse haversine. [radians]

    """
    return 2.0 * np.arcsin(np.sqrt(y))

def cartesian(*arrays):
    """Cartesian product of the input arrays.

    Parameters
    ----------
    arrays : array
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

_davint_err = dict()
_davint_err[2] = 'x1 was less than x0'
_davint_err[3] = 'the number of x between x0 and x1 (inclusive) was less than 3 and neither of the two special cases described in the abstract occurred.  No integration was performed.'
_davint_err[4] = 'the restriction x(i+1) > x(i) was violated.'
_davint_err[5] = 'the number of function values was < 2'

def davint(x, y, x0, x1, axis=0):
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
    axis : int
      If `y` is a 2D array, then integrate over axis `axis` for each
      element of the other axis.

    Returns
    -------
    float
      The result.

    """
    from .lib import davint as _davint

    y = np.array(y)
    if y.ndim == 1:
        r, ierr = _davint(x, y, len(x), x0, x1)
        if ierr != 1:
            raise RuntimeError("DAVINT integration error: {}".format(err[ierr]))
    elif y.ndim == 2:
        r = np.zeros(y.shape[axis])
        for i, yy in enumerate(np.rollaxis(y, axis)):
            r[i] = davint(x, yy, x0, x1)
    else:
        raise ValueError("y must have 1 or 2 dimensions.")

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

def gaussian(x, mu, sigma):
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
            np.sqrt(2.0 * np.pi) / sigma)

def gaussian2d(shape, sigma, theta=0):
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
    G /= 2.0 * np.pi * sx * sy
    return G

def hav(th):
    """Haversine of an angle.

    Haversine is (1 - cos(th)) / 2 = sin**2(th/2)

    Parameters
    ----------
    th : float or array
      The angle. [radians]

    Returns
    -------
    y : float or ndarray
      The haversine.

    """
    return np.sin(th / 2.0)**2

def rotmat(th):
    """Returns a rotation matrix.

    The matrix rotates the vector [x, y] by the amount a.

    Parameters
    ----------
    th : float
      The amount to rotate. [radians]

    Returns
    -------
    r : np.matrix
      Rotation matrix.

    Examples
    --------
    import numpy as np
    from mskpy import rotmat
    print(np.array([1, 0]) * rotmat(np.radians(90.0)))
    --> matrix([[  6.12323400e-17,   1.00000000e+00]])
    print(np.array([0, 1]) * rotmat(np.pi))
    --> matrix([[ -1.00000000e+00,   6.12323400e-17]])

    """
    c = np.cos(th)
    s = np.sin(th)
    return np.matrix([[c, s], [-s, c]])

def basicwcs(crpix, crval, cdelt, pa, projection='TAN'):
    """A basic world coordinate system (WCS) object.

    Parameters
    ----------
    crpix : array
      The center of the WCS projection: [x, y].
    crval : array
      The coordinates at CRPIX: [ra, dec]. [degrees]
    cdelt : double or array
      The image scale in arcsecons per pixel.  If cdelt is a scalar
      value then the WCS CDELT will be [-1, 1] * cdelt.
    pa : double
      The position angle of N from the y-axis. [degrees]

    Returns
    -------
    wcs : astropy wcs
      Your new WCS.

    """

    import astropy.wcs

    wcs = astropy.wcs.wcs.WCS()
    wcs.wcs.crpix = crpix
    wcs.wcs.crval = crval
    if np.iterable(cdelt):
        wcs.wcs.cdelt = cdelt / 3600.0
    else:
        wcs.wcs.cdelt = np.array([-1, 1]) * cdelt / 3600.0
    par = np.radians(pa)
    wcs.wcs.pc = np.array([[-np.cos(par), -np.sin(par)],
                            [-np.sin(par),  np.cos(par)]])
    return wcs

def fitslog(keywords, files=None, path='.', format=None, csv=True):
    """One-line descriptions of a list of FITS files.

    By default, `fitslog` will summarize *.fit{,s} files in the
    current directory.

    Parameters
    ----------
    keywords : array or str
      A list of FITS keywords to extract from each header.  Keywords
      may also be the name of a template: Bigdog, Guidedog, MIRSI.
    files : array, optional
      A list of files to summarize.  Overrides path.
    path : str, optional
      Summarize all FITS files in this location.
    format : str, optional
      The output format string.  A newline character will be appended.
    csv : bool, optional
      Set to `True` to separate output fields with commas.  Ignored
      for user defined formats.

    Returns
    -------
    log : str
      The summary of the FITS files as a string.

    """

    from astropy.io import fits

    if files is None:
        files = glob("{0}/*.fit".format(path))
        files.extend(glob("{0}/*.fits".format(path)))
        files.sort()

    if type(keywords) is str:
        if keywords.lower() == 'bigdog':
            keywords = ['TIME_OBS', 'ITIME', 'CO_ADDS', 'CYCLES',
                        'AIRMASS', 'GRAT', 'OBJECT']
            format = ["{0:16}", "{1:18}", "{2:6.2f}", "{3:4d}"
                      "{4:4d}", "{5:7.3f}", "{6:<12}", "{7:<25}"]
        elif keywords.lower() == 'guidedog':
            keywords = ['TIME_OBS', 'ITIME', 'CO_ADDS', 'CYCLES',
                        'AIRMASS', 'GFLT', 'OBJECT']
            format = ["{0:16}", "{1:18}", "{2:6.2f}", "{3:4d}"
                      "{4:4d}", "{5:7.3f}", "{6:<12}", "{7:<25}"]
        elif keywords.lower() == 'mirsi':
            keywords = ['UTC_TIME', 'OBS-MODE', 'EXPTIME', 'FRAME-T',
                        'NCOADS', 'AIRMASS', 'WHEEL1', 'WHEEL2', 'WHEEL3',
                        'OBJECT']
            format = ['{0:12}', '{1:13}', '{2:7.3f}', '{3:>7}', '{4:3d}',
                      '{5:>6}', '{6:>20}', '{7:>20}', '{8:>20}',
                      '{9}']
        else:
            print("{0} not a recognized template".format(keywords))
            return None

        if csv:
            format = ", ".join(format)
        else:
            format = " ".join(format)
    else:
        if format is None:
            format = []
            for i in range(len(keywords)):
                format.append("{{{0}}}".format(i))
            if csv:
                format = ", ".join(format)
            else:
                format = " ".join(format)

    log = ""
    s = max([len(x.replace('.fits', '').replace('.fit', '').split('/')[-1])
             for x in files])
    for f in files:
        log += '{0:{1}}'.format(
            f.replace('.fits', '').replace('.fit', '').split('/')[-1], s)
        if csv:
            log += ','
        log += ' '
        h = fits.getheader(f)
        values = ()
        for k in keywords:
            values += (h[k], )
        log += format.format(*values)
        log += "\n"
    
    return log

def getrot(h):
    """Image rotation and pixel scale from a FITS header.

    Based on the IDL Astronomy routine getrot.pro (W. Landsman).

    Parameters
    ----------
    h : astropy.io.fits header or string
      A FITS header or the name of a file with a defined world
      coordinate system.  The file name will be passed to
      `fits.getheader`.

    Returns
    -------
    cdelt : ndarray
      Two-element array of the pixel scale (x, y).  [arcseconds/pixel]
    rot : float
      The image orientation (position angle of north).  [degrees]

    """

    from astropy.io import fits

    if isinstance(h, str):
        h = fits.getheader(h)

    # Does CDELTx exist?
    cdelt = np.zeros(2)
    cdeltDefined = False
    if (('CDELT1' in h) and ('CDELT2' in h)):
        # these keywords take precedence over the CD matrix
        cdeltDefined = True
        cdelt = np.array([h['CDELT1'], h['CDELT2']])

    # Transformation matrix?
    tmDefined = False
    if (('CD1_1' in h) and ('CD1_2' in h) and
        ('CD2_1' in h) and ('CD2_2' in h)):
        tmDefined = True
        cd = np.array(((h['CD1_1'], h['CD1_2']), (h['CD2_1'], h['CD2_2'])))

    if (('PC1_1' in h) and ('PC1_2' in h) and
        ('PC2_1' in h) and ('PC2_2' in h)):
        tmDefined = True
        cd = np.array(((h['PC1_1'], h['PC1_2']), (h['PC2_1'], h['PC2_2'])))

    if not tmDefined:
        # if CDELT is defined but the transformation matrix isn't,
        # then CROT should be defined
        if cdeltDefined and ('CROTA2' in h):
            rot = h['CROTA2']
            return cdelt, rot

        raise ValueError("WCS has CDELTx but is missing CROTA2,"
                         " and CDi_j or PCi_j")

    if (h['CTYPE1'].find('DEC-') >= 0) or (h['CTYPE1'].find('LAT') >= 0):
        newcd = cd.copy()
        newcd[0,:] = cd[1,:]
        newcd[1,:] = cd[0,:]
        cd = newcd.copy()

    if np.linalg.det(cd) < 0:
        sgn = -1.0
    else:
        sgn = 1.0

    if (cd[1, 0] == 0) and (cd[0, 1] == 0):
        # unrotated coordinate system
        rot1 = 0
        rot2 = 0

        if not cdeltDefined:
            cdelt[0] = cd[0, 0]
            cdelt[1] = cd[1, 1]
    else:
        rot1 = np.arctan2(sgn * np.radians(cd[0, 1]),
                             sgn * np.radians(cd[0, 0]))
        rot2 = np.arctan2(np.radians(-cd[1, 0]),
                             np.radians(cd[1, 1]))

        if not cdeltDefined:
            cdelt[0] = sgn * np.sqrt(cd[0, 0]**2 + cd[0, 1]**2)
            cdelt[1] = np.sqrt(cd[1, 1]**2 + cd[1, 0]**2)

    return cdelt * 3600.0, np.degrees(rot1)

def gaussfit(x, y, err, guess, covar=False, **kwargs):
    """A quick Gaussian fitting function, optionally including a line.

    Parameters
    ----------
    x, y : array
      The independent and dependent variables.
    err : array
      `y` errors, set to `None` for unweighted fitting.
    guess : tuple
      Initial guess.  The length of the guess determines the fitting
      function:
        `(amplitude, mu, sigma)` - pure Gaussian
        `(amplitude, mu, sigma, b)` - Gaussian + constant offset `b`
        `(amplitude, mu, sigma, m, b)` - Gaussian + linear term `m x + b`
    covar : bool, optional
      Set to `True` to return the covariance matrix rather than the
      error.
    **kwargs
      Keyword arguments to pass to `scipy.optimize.leastsq`.

    Returns
    -------
    fit : tuple
      Best-fit parameters.
    err or cov : tuple or ndarray
      Errors on the fit or the covariance matrix of the fit (see
      `covar` keyword).

    """

    from scipy.optimize import leastsq

    def gauss_chi(p, x, y, err):
        A, mu, sigma = p
        model = A * gaussian(x, mu, sigma)
        chi = (np.array(y) - model) / np.array(err)
        return chi

    def gauss_offset_chi(p, x, y, err):
        A, mu, sigma, b = p
        model = A * gaussian(x, mu, sigma) + b
        chi = (np.array(y) - model) / np.array(err)
        return chi

    def gauss_line_chi(p, x, y, err):
        A, mu, sigma, m, b = p
        model = A * gaussian(x, mu, sigma) + m * x + b
        chi = (np.array(y) - model) / np.array(err)
        return chi

    if err is None:
        err = np.ones(len(y))

    assert len(guess) in (3, 4, 5), "guess must have length of 3, 4, or 5."

    opts = dict(args=(x, y, err), full_output=True, epsfcn=1e-4,
                xtol=1e-4, ftol=1e-4)
    opts.update(**kwargs)
    if len(guess) == 3:
        output = leastsq(gauss_chi, guess, **opts)
    elif len(guess) == 4:
        output = leastsq(gauss_offset_chi, guess, **opts)
    elif len(guess) == 5:
        output = leastsq(gauss_line_chi, guess, **opts)

    fit = output[0]
    cov = output[1]
    if cov is None:
        print(output[3])
        err = None
    else:
        err = np.sqrt(np.diag(cov))

    if covar:
        return fit, cov
    else:
        return fit, err

def glfit(x, y, err, guess, covar=False):
    """A quick Gaussian + line fitting function.

    Parameters
    ----------
    x, y : array
      The independent and dependent variables.
    err : array
      `y` errors, set to `None` for unweighted fitting.
    guess : tuple
      Initial guess: `(amplitude, mu, sigma, m, b)`.
    covar : bool, optional
      Set to `True` to return the covariance matrix rather than the
      error.

    Returns
    -------
    fit : tuple
      Best-fit parameters.
    err or cov : tuple or ndarray
      Errors on the fit or the covariance matrix of the fit (see
      `covar` keyword).

    """

    from scipy.optimize import leastsq

    def chi(p, x, y, err):
        A, mu, sigma, m, b = p
        model = A * gaussian(x, mu, sigma) + m * x + b
        chi = (np.array(y) - model) / np.array(err)
        return chi

    if err is None:
        err = np.ones(len(y))

    output = leastsq(chi, guess, args=(x, y, err), full_output=True,
                     epsfcn=1e-4)
    fit = output[0]
    cov = output[1]
    err = np.sqrt(np.diag(cov))

    if covar:
        return fit, cov
    else:
        return fit, err

def linefit(x, y, err, guess, covar=False):
    """A quick line fitting function.

    Parameters
    ----------
    x, y : array
      The independent and dependent variables.
    err : array
      `y` errors, set to `None` for unweighted fitting.
    guess : tuple (double, double)
      `(m, b)` a guess for the slope, `m`, and y-axis intercept `b`.
    covar : bool, optional
      Set to `True` to return the covariance matrix rather than the
      error.

    Returns
    -------
    fit : tuple (double, double)
      `(m, b)` the best-fit slope, `m`, and y-axis intercept `b`.
    err or cov : tuple (double, double) or ndarray
      Errors on the fit or the covariance matrix of the fit (see
      `covar` keyword).

    """

    from scipy.optimize import leastsq

    def chi(p, x, y, err):
        m, b = p
        model = m * np.array(x) + b
        chi = (np.array(y) - model) / np.array(err)
        return chi

    if err is None:
        err = np.ones(len(y))

    output = leastsq(chi, guess, args=(x, y, err), full_output=True,
                     epsfcn=1e-3)
    fit = output[0]
    cov = output[1]
    err = np.sqrt(np.diag(cov))

    if covar:
        return fit, cov
    else:
        return fit, err

def planckfit(wave, fluxd, err, guess, covar=False):
    """A quick scaled Planck fitting function.

    The scale factor includes a factor of pi for the conversion from
    specific surface brightness to flux density.

    Parameters
    ----------
    wave, fluxd : Quantity
      The wavelength and flux density.
    err : Quantity
      Flux density uncertainties; set to `None` for unweighted fitting.
    guess : tuple (double, double)
      `(scale, T)` a guess for the temperature, `T`, and scale factor.
    covar : bool, optional
      Set to `True` to return the covariance matrix rather than the
      error.

    Returns
    -------
    fit : tuple (double, double)
      `(scale, T)` the best-fit parameters.
    err or cov : tuple (double, double) or ndarray
      Errors on the fit or the covariance matrix of the fit (see
      `covar` keyword).

    """

    from scipy.optimize import leastsq

    def chi(p, wave, fluxd, err):
        import astropy.units as u
        scale, T = p
        model = scale * planck(wave, T, unit=fluxd.unit / u.sr) * u.sr
        chi = (fluxd - model) / err
        return chi.decompose().value

    if err is None:
        err = np.ones(len(y)) * fluxd.unit

    output = leastsq(chi, guess, args=(wave, fluxd, err), full_output=True,
                     epsfcn=1e-3)
    print(output[-2])
    fit = output[0]
    cov = output[1]

    if covar:
        return fit, cov
    else:
        if cov is None:
            return fit, None
        else:
            return fit, np.sqrt(np.diag(cov))

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
        for j in range(lim.shape[0]):
            i += between(a, lim[j,:])

    return i.astype(bool)

def clusters(test):
    """Define array slices based on a test value.

    Parameters
    ----------
    test : array
      The test result.

    Returns
    -------
    objects : tuple of slices
      An array of slices that return each cluster of `True` values in
      `test`.

    """

    import scipy.ndimage as nd

    labels, n = nd.label(test)
    print("{} clusters found".format(n))
    return nd.find_objects(labels)

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
    >>> print(keys)
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

def leading_num_key(s):
    """Keys for sorting strings, based on leading multidigit numbers.

    A normal string comparision will compare the strings character by
    character, e.g., "101P" is less than "1P" because "0" < "P".
    `leading_num_key` will generate keys so that `str.sort` can
    consider the leading multidigit integer, e.g., "101P" > "1P"
    because 101 > 1.

    Parameters
    ----------
    s : string

    Returns
    -------
    keys : tuple
      They keys to sort by for this string: `keys[0]` is the leading
      number, `keys[1]` is the rest of the string.

    """

    pfx = ''
    for i in range(len(s)):
        if not s[i].isdigit():
            break
        pfx += s[i]
    sfx = s[i:]

    if len(pfx) > 0:
        pfx = int(pfx)
    return pfx, sfx

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
    return np.abs(np.array(array) - v).argmin()

def stat_avg(x, y, u, N):
    """Bin an array, weighted by measurement errors.

    Parameters
    ----------
    x : array
      The independent variable.
    y : array
      The parameter to average.
    u : array
      The uncertainties on y. weights for each `y`.
    N : int
      The number of points to bin.  The right-most bin may contain
      fewer than `N` points.

    Returns
    -------
    bx, by, bu : ndarray
      The binned data.  The `x` data is straight averaged (unweighted).
    n : ndarray
      The number of points in each bin.

    """

    nbins = x.size // N
    remainder = x.size % nbins
    shape = (nbins, N)

    w = (1.0 / np.array(u)**2)
    _w = w[:-remainder].reshape(shape)
    _x = np.array(x)[:-remainder].reshape(shape)
    _y = np.array(y)[:-remainder].reshape(shape)

    _x = _x.mean(1)
    _y = (_y * _w).sum(1) / _w.sum(1)
    _u = np.sqrt(1.0 / _w.sum(1))

    n = np.ones(len(_x)) * N
    if remainder > 0:
        _x = np.r_[_x, np.mean(x[-remainder:])]
        _y = np.r_[_y, (np.array(y[-remainder:]) / w[-remainder:]).sum()]
        _u = np.r_[_u, np.sqrt(1.0 / w[-remainder:].sum())]
        n = np.r_[n, remainder]

    return _x, _y, _u, n

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
        newa = np.array(a)[indices]
        if not isinstance(newa, type(a)):
            newa = type(a)(newa)
        r += (newa,)
    return r

def whist(x, y, w, errors=True, **keywords):
    """A weighted histogram binned by an independent variable.

    Parameters
    ----------
    x : array
      The independent variable.
    y : array
      The parameter to average.
    w : array
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

    if 'weights' in keywords:
        raise RuntimeError('weights not allowed in keywords')

    _x = np.array(x)
    _y = np.array(y)
    _w = np.array(w)

    if errors:
        _w = 1.0 / _w**2

    n, edges = np.histogram(x, **keywords)
    n = n.astype(float)

    num = np.histogram(x, weights=_y * _w, **keywords)[0]
    den = np.histogram(x, weights=_w, **keywords)[0]
    m = num / den

    if errors:
        err = 1.0 / np.sqrt(den)
    else:
        err = None

    return m, err, n, edges

def ec2eq(lam, bet):
    """Ecliptic coordinates to equatorial (J2000.0) coordinates.

    Parameters
    ----------
    lam, bet : float or array
      Ecliptic longitude and latitude. [degrees]

    Returns
    -------
    ra, dec : float or ndarray
      Equatorial (J2000.0) longitude and latitude. [degrees]

    Notes
    -----
    Based on euler.pro in the IDL Astro library (W. Landsman).

    """

    # using the mean obliquity of the ecliptic at the J2000.0 epoch
    # eps = 23.439291111 degrees (Astronomical Almanac 2008)
    ceps = 0.91748206207 # cos(eps)
    seps = 0.39777715593 # sin(eps)

    # convert to radians
    lam = np.radians(lam)
    bet = np.radians(bet)
    
    cbet = np.cos(bet)
    sbet = np.sin(bet)
    clam = np.cos(lam)
    slam = np.sin(lam)

    ra = np.arctan2(ceps * cbet * slam - seps * sbet, cbet * clam)
    sdec = seps * cbet * slam + ceps * sbet

    if np.iterable(sdec):
        sdec[sdec > 1.0] = 1.0
    else:
        if sdec > 1.0:
            sdec = 1.0
    dec = np.arcsin(sdec)

    # make sure 0 <= ra < 2pi
    ra = (ra + 4.0 * np.pi) % (2.0 * np.pi)

    return np.degrees(ra), np.degrees(dec)

def lb2xyz(lam, bet=None):
    """Transform longitude and latitude to a unit vector.

    Parameters
    ----------
    lam : float, array, or 2xN array
      The longitude(s), or an array of longitudes and
      latitudes. [degrees]
    bet : float or array, optional
      The latitude(s). [degrees]

    Returns
    -------
    xyz : array or 3xN array
      The unit vectors.

    """
    _lam = np.array(lam).squeeze()
    if bet is None:
        return lb2xyz(_lam[0], _lam[1])

    lamr = np.radians(_lam)
    betr = np.radians(np.array(bet).squeeze())
    return np.array((np.cos(betr) * np.cos(lamr),
                     np.cos(betr) * np.sin(lamr),
                     np.sin(betr)))

def projected_vector_angle(r, rot, ra, dec):
    """Position angle of a vector projected onto the observing plane.

    Parameters
    ----------
    r : array
      The vector to project, in heliocentric ecliptic
      coordinates. [km]
    rot : array
      The observer-target vector. [km]
    ra, dec : float
      The right ascention and declination of the target, as seen by
      the observer. [deg]

    Returns
    -------
    angle : float
      The position angle w.r.t. to equatorial north. [deg]

    """
    r0 = np.sqrt((r**2).sum())  # magnitude of r
    dv = rot + r / r0  # delta vector

    # find the projected vectors in RA, Dec
    lam2 = np.degrees(np.arctan2(dv[1], dv[0]))
    bet2 = np.degrees(np.arctan2(dv[2], np.sqrt(dv[0]**2 + dv[1]**2)))

    ra2, dec2 = ec2eq(lam2, bet2)

    x2 = (ra2 - ra) * np.cos(np.radians(dec2))
    y2 = (dec2 - dec)

    th = np.degrees(np.arctan2(y2, x2))
    pa = 90.0 - th
    
    return pa

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

def state2orbit(R, V):
    """Convert a small body's state vector into osculating orbital elements.

    CURRENTLY INCOMPLETE!  Only a, ec, q, Tp, P, f, E, and M are
    computed, and even fewer are computed for near-parabolic orbits.

    Two-body osculating solution. For details, see Murry and Dermott,
    Solar System Dynamics, Chapter 2.

    Parameters
    ----------
    R : array
      The x, y, z heliocentric ecliptic coordinates. [km]
    V : array
      The vx, vy, vz heliocentric ecliptic speeds. [km/s]

    Returns
    -------
    orbit : dict
      A dictionary {a, ec, in, node, peri, Tp, P, f, E, M}, where:
        a = semi-major axis [km]
        ec = eccentricity
        in = inclination [radians]
        node = longitude of the acending node, Omega [radians]
        peri = argument of pericenter [radians]
        Tp = time of perihelion passage [days]
        q = perihelion distance
        P = orbital period [days]
        f = true anomaly at date [radians]
        E = eccentric anomaly at date [radians]
        M = mean anomaly at date [radians]

    """

    import astropy.units as u

    mu = 1.32712440018e11  # km3/s2
    AU = u.au.to(u.kilometer)

    # some usefull things
    r = np.sqrt((R**2).sum())  # heliocentric distance [km]
    v = np.sqrt((V**2).sum())  # velocity [km/s]

    H = np.cross(R, V)         # specific angular momentum vector [km2/s]
    h = np.sqrt((H**2).sum())  # specific angular momentum [km2/s]

    s = np.dot(R, V)
    drdt = np.sign(s) * np.sqrt(v**2 - h**2 / r**2)

    a = 1.0 / (2.0 / r - v**2 / mu)  # [km]
    ec = np.sqrt(1.0 - h**2 / mu / a)  # eccentricity
    q = a * (1.0 - ec) / AU  # perihelion distance [AU]

    if ec < 0.98:
        sinf = h / mu / ec * drdt
        cosf = (h**2 / mu / r - 1.0) / ec
        f = np.arctan2(sinf, cosf)  # true anomaly [radians]
    elif ec < 1.1:
        # punt!
        return dict(a=a, ec=ec, q=q, Tp=None, P=None, f=None, E=None,
                    H=None, M=None)
    else:
        raise ValueError("eccentricity is too high")

    # eccentric anomaly [radians]
    if ec < 1.0:
        E = 2.0 * np.arctan2(np.sqrt(1.0 - ec) * np.sin(f / 2.0),
                             np.sqrt(1.0 + ec) * np.cos(f / 2.0))
        M = E - ec * np.sin(E)  # mean anomaly [radians]
    else:
        # hyperbolic eccentric anomaly [radians]
        E = 2.0 * np.arccosh((ec + cosf) / (1 + ec + cosf))
        M = -E + ec * np.sinh(E)  # mean anomaly [radians]

    # date of perihelion [Julian date]
    if a < 0:
        n = np.sqrt(mu / -a**3) / 86400.0  # mean motion
        Tp = -M * np.sqrt(-a**3 / mu) / 86400.0
        P = None
    else:
        Tp = -M * np.sqrt(a**3 / mu) / 86400.0
        P = 2.0 * np.pi * np.sqrt(a**3 / mu) / 86400.0  # orbital period [days]

    return dict(a=a, ec=ec, q=q, Tp=Tp, P=P, f=f, E=E, M=M)

def vector_rotate(r, n, th):
    """Rotate vector `r` an angle `th` CCW about `n`.

    Parameters
    ----------
    r : array (3)
      The vector to rotate [x, y, z].
    n : array (3)
      The vector to rotate about.
    th : float or array
      The CCW angle to rotate by. [radians]

    Returns
    -------
    rp : ndarray
      The rotated vector [x, y, z].

    Notes
    -----
    Described in Goldstein p165, 2nd ed. Note that Goldstein presents
    the formula for clockwise rotation.

    """

    nhat = n / np.sqrt((n**2).sum())

    def rot(r, nhat, theta):
        return (r * np.cos(-theta) +
                nhat * (nhat * r).sum() * (1.0 - np.cos(-theta)) +
                np.cross(r, nhat) * np.sin(-theta))

    if np.size(th) == 1:
        return rot(r, nhat, th)
    else:
        return np.array([rot(r, nhat, t) for t in th])

def xyz2lb(r):
    """Transform a vector to angles.

    Parameters
    ----------
    r : array
      The vector, shape = (3,) or (n, 3).

    Returns
    -------
    lam : float or array
      Longitude. [degrees]
    bet : float or array
      Latitude. [degrees]

    """

    r = np.array(r)
    if r.ndim == 1:
        lam = np.arctan2(r[1], r[0])
        bet = np.arctan2(r[2], np.sqrt(r[0]**2 + r[1]**2))
    else:
        # assume it is an array of vectors
        lam = np.arctan2(r[:, 1], r[:, 0])
        bet = np.arctan2(r[:, 2], np.sqrt(r[:, 0]**2 + r[:, 1]**2))

    return np.degrees(lam), np.degrees(bet)

def kuiper(x, y):
    """Compute Kuiper's statistic and probablity.

    Parameters
    ----------
    x, y : array
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

    data1, data2 = list(map(np.asarray, (x, y)))
    n1 = data1.shape[0]
    n2 = data2.shape[0]
    data1 = np.sort(data1)
    data2 = np.sort(data2)
    data_all = np.sort(np.concatenate([data1, data2]))
    cdf1 = np.searchsorted(data1, data_all, side='right') / n1
    cdf2 = np.searchsorted(data2, data_all, side='right') / n2
    V = np.ptp(cdf1 - cdf2)
    Ne = n1 * n2 / (n1 + n2)
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

def meanclip(x, axis=None, lsig=3.0, hsig=3.0, maxiter=5, minfrac=0.001,
             full_output=False, dtype=np.float64):
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
            for i in range(x2.shape[0]):
                mc = meanclip(x2[i], axis=None, lsig=lsig, hsig=hsig,
                              maxiter=maxiter, minfrac=minfrac,
                              full_output=True)
                y[i], ys[i], yiter[i] = mc[0], mc[1], mc[3]
                yind += (mc[2],)
            if full_output:
                return y.mean(dtype=dtype), ys, yind, yiter
            else:
                return y.mean(dtype=dtype)
        else:
            raise ValueError("There is no axis {0} in the input"
                             " array".format(axis))

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
        sig = y.std(dtype=dtype)

        keep = (y > (medval - lsig * sig)) * (y < (medval + hsig * sig))
        cutfrac = abs(good.size - keep.sum()) / good.size

        if keep.sum() > 0:
            good = good[keep]
        else:
            break  # somehow all the data were clipped

        if cutfrac <= minfrac:
            break

    y = x.flatten()[good]
    if full_output:
        return y.mean(dtype=dtype), y.std(dtype=dtype), good, i+1
    else:
        return y.mean(dtype=dtype)

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
    from scipy import stats

    def spearmanZ(x, y):
        N = len(x)
        rankx = stats.rankdata(x)
        ranky = stats.rankdata(y)

        # find the corrections for ties
        ties = stats.mstats.count_tied_groups(x)
        sx = sum((k**3 - k) * v for k, v in ties.items())
        ties = stats.mstats.count_tied_groups(y)
        sy = sum((k**3 - k) * v for k, v in ties.items())

        D = sum((rankx - ranky)**2)
        meanD = (N**3 - N) / 6.0 - (sx + sy) / 12.0
        varD = (N - 1) * N**2 * (N + 1)**2 / 36.0
        varD *= (1 - sx / (N**3 - N)) * (1 - sy / (N**3 - N))
        return abs(D - meanD) / np.sqrt(varD)
 
    N = len(x)

    rp = stats.mstats.spearmanr(x, y, use_ties=True)
    r = rp[0]
    p = rp[1]
    Z = spearmanZ(x, y)

    if nmc is not None:
        if xerr is None:
            xerr = np.zeros(N)
        if yerr is None:
            yerr = np.zeros(y.shape)

        mcZ = np.zeros(nmc)
        for i in range(nmc):
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

def bandpass(sw, sf, se=None, fw=None, ft=None, filter=None, filterdir=None,
             k=3, s=None):
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
    se : array, optional
      Weight the fluxes with these uncertainties.
    fw : array, optional
      Filter transmission profile wavelengths, same units as `sw`.
    ft : array, optional
      Filter transmission profile.
    filter : string, optional
      The name of a filter (see `calib.filter_trans`).  The wavelength
      units will be micrometers.
    filterdir : string, optional
      The directory containing the filter transmission files
      (see `calib.filter_trans`).
    k : int, optional
      Order of the spline fit for interpolation.  See
      `scipy.interpolate.splrep`.
    s : float, optional
      Interpolation smoothing.  See `scipy.interpolate.splrep`.

    Returns
    -------
    wave, flux : ndarray
      The effective wavelength and flux density of the filtered spectrum.
    err : ndarray, optional
      The uncertaintiy on the filtered spectrum.  Returned if `se` is
      not `None`.

    """

    from scipy import interpolate
    import astropy.units as u
    from . import calib

    # local copies
    _sw = np.array(sw)
    _sf = np.array(sf)
    if se is None:
        _se = np.ones_like(_sf)
    else:
        _se = np.array(se)

    if (fw is not None) and (ft != None):
        _fw = np.array(fw)
        _ft = np.array(ft)
    elif filter is not None:
        _fw, _ft = calib.filter_trans(filter)
        _fw = _fw.to(u.um).value
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
        spl = interpolate.splrep(_sw, _sf, k=k, s=s)
        _sf = interpolate.splev(_w, spl)
        spl = interpolate.splrep(_sw, _se**2, k=k, s=s)
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
        spl = interpolate.splrep(_fw, _ft, k=k, s=s)
        _ft = interpolate.splev(_w, spl)
        _sf = _sf
        _se2 = _se**2

    # weighted mean to get the effective wavelength
    wrange = minmax(_w)
    weights = _ft * _sf / _se2
    wave = (davint(_w, _w * weights, *wrange) / davint(_w, weights, *wrange))

    # weighted mean for the flux
    weights = _ft / _se2
    flux = davint(_w, _sf * weights, *wrange) / davint(_w, weights, *wrange)
    err = davint(_w, weights, *wrange) / davint(_w, 1.0 / _se2, *wrange)
    err = np.sqrt(err) * errscale

    if se is None:
        return wave, flux
    else:
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
        "gaussian(sigma)" - specifiy sigma in the same units as `wave`
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

    import re

    if type(func) is str:
        if 'gaussian' in func.lower():
            sigma = float(re.findall('gaussian\(([^)]+)\)', func.lower())[0])
            def func(dw):
                return gaussian(dw, 0, sigma)
        elif 'uniform' in func.lower():
            hwhm = (float(re.findall('uniform\(([^)]+)\)', func.lower())[0])
                    / 2.0)
            def func(dw):
                f = np.zeros_like(dw)
                i = (dw > -hwhm) * (dw <= hwhm)
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

def phase_integral(phasef, range=[0, 180]):
    """The phase integral of a phase function.

    Parameters
    ----------
    phasef : function
      The phase function, takes one parameter, `phase`, in units of
      degrees.
    range : array, optional
      The integration limits.  [degrees]

    Returns
    -------
    pint : float

    """
    from scipy.integrate import quad
    range = np.radians(range)
    pint = 2.0 * quad(lambda x: phasef(np.degrees(x)) * np.sin(x),
                      min(range), max(range))[0]
    return pint

def planck(wave, T, unit=None, deriv=None):
    """The Planck function.

    Parameters
    ----------
    wave : array or Quantity
      The wavelength(s) to evaluate the Planck function. [micron]
    T : float or array
      The temperature(s) of the Planck function. [Kelvin]
    unit : u.Unit
      The output units.  Set to `None` to return a float in the
      default units.
    deriv : string
      Set to 'T' to return the first derivative with respect to
      temperature.

    Returns
    -------
    B : float or Quantity
      If `unit is None`, a `float` will be returned in units of
      W/m2/sr/Hz.

    Raises
    ------
    ValueError when deriv isn't an allowed value.

    """

    import astropy.units as u
    from astropy.units import Quantity

    # prevent over/underflow warnings
    oldseterr = np.seterr(all='ignore')

    # wave in m
    if isinstance(wave, Quantity):
        wave = wave.si.value
    else:
        wave = wave * 1e-6

    #from astropy import constants as const
    #c1 = 2.0 * const.si.h * const.si.c / u.s / u.Hz
    #c2 = const.si.h * const.si.c / const.si.k_B
    #a = np.exp(c2 / wave.si / T.to(u.K))
    #B = c1 / ((wave.si)**3 * (a - 1.0)) / u.sr

    c1 = 3.9728913665386057e-25  # J m
    c2 = 0.0143877695998  # K m
    a = np.exp(c2 / wave / T)
    B = c1 / (wave**3 * (a - 1.0))

    if unit is not None:
        Bunit = u.Unit('W / (m2 sr Hz)')

    if deriv is not None:
        if deriv.lower() == 't':
            B *= c2 / T**2 / wave * a / (a - 1.0)
            if unit is not None:
                Bunit /= u.K
        else:
            raise ValueError("deriv parameter not allowed: {}".format(
                    deriv))

    # restore seterr
    np.seterr(**oldseterr)

    if unit is not None:
        B *= Bunit
        if unit != Bunit:
            B = B.to(unit, equivalencies=spectral_density_sb(wave * u.m))

    return B

def _redden(wave, S, wave0=0.55):
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
    >>> print(redden(wave, S))
    [ 0.83527021  0.88692044  0.94176453  1.          1.12749685  3.32011692]

    """

    from scipy.integrate import quad
    from scipy.interpolate import interp1d

    if not np.iterable(wave):
        wave = np.array(wave).reshape(1)

    if not np.iterable(S):
        S = np.ones_like(wave) * S
    elif len(S) == 1:
        S = np.ones_like(wave) * S[0]

    slope = interp1d(np.r_[0, wave, np.inf], np.r_[S[0], S, S[-1]],
                     kind='linear')

    spec = np.zeros_like(wave)
    for i in range(len(wave)):
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
    th : float or array
      The phase angle.  [degrees]
    p, a, b : float
      The parameters of the function.
    th0 : float
      The negative to positive branch turnover angle. [degrees]

    Returns
    -------
    P : float or ndarray
      The polarization at phase angle `th`.

    """
    thr = np.radians(th)
    return (p * np.sin(thr)**a * np.cos(thr / 2.)**b
            * np.sin(thr - np.radians(th0)))

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

    half_window = (kernel - 1) // 2
    b = np.mat([[k**i for i in range(order + 1)]
                for k in range(-half_window, half_window+1)])

    # since we don't want the derivative, else choose [1] or [2], respectively
    m = np.linalg.pinv(b).A[0]
    window_size = len(m)
    half_window = (window_size - 1) // 2

    # precompute the offset values for better performance
    offsets = zip(range(-half_window, half_window + 1), m)

    # temporary data, extended with a mirror image to the left and right
    # left extension: f(x0-x) = f(x0)-(f(x)-f(x0)) = 2f(x0)-f(x)
    # right extension: f(xl+x) = f(xl)+(f(xl)-f(xl-x)) = 2f(xl)-f(xl-x)
    leftpad = np.zeros(half_window) + 2 * x[0]
    rightpad = np.zeros(half_window) + 2 * x[-1]
    leftchunk = x[1:(1 + half_window)]
    leftpad = leftpad-leftchunk[::-1]
    rightchunk = x[len(x) - half_window - 1:len(x) - 1]
    rightpad = rightpad - rightchunk[::-1]
    data = np.concatenate((leftpad, x))
    data = np.concatenate((data, rightpad))

    smooth_data = list()
    for i in range(half_window, len(data) - half_window):
        value = 0.0
        for offset, weight in offsets:
            value += weight * data[i + offset]
        smooth_data.append(value)

    return np.array(smooth_data)

def cal2doy(cal, scale='utc'):
    """Calendar date to day of year.

    Parameters
    ----------
    cal : string or array
      Calendar date.  See `cal2iso` for details.
    scale : string, optional
      See `astropy.time.Time`.

    Returns
    -------
    doy : astropy Time
      Day of year.

    """
    from astropy.time import Time
    t = cal2time(cal, scale=scale)
    if len(t) > 1:
        return [int(x.yday.split(':')[1]) for x in t]
    else:
        return int(t.yday.split(':')[1])

def cal2iso(cal):
    """Calendar date to ISO format.

    Parameters
    ----------
    cal : string or array
      Calendar date.  Format: YYYY-MM-DD HH:MM:SS.SSS.  May be
      shortened, for example, to YYYY or YYYY-MM.  DD == 0 is not
      allowed and is forced to 1.  MM may be a three character
      abbreviation.  Fractional values are allowed for days and
      smaller units.

    Returns
    -------
    iso : string or list
      `cal`, ISO formatted.

    """

    if isinstance(cal, (list, tuple, np.ndarray)):
        return [cal2iso(x) for x in cal]

    # mapping function to remove nondigits from the date string
    def a2space(c):
        return c if (c.isdigit() or c == ".") else " "

    # if the month is an abbreviation, replace it with a number
    cal = cal.lower()
    cal = cal.replace('jan', '01')
    cal = cal.replace('feb', '02')
    cal = cal.replace('mar', '03')
    cal = cal.replace('apr', '04')
    cal = cal.replace('may', '05')
    cal = cal.replace('jun', '06')
    cal = cal.replace('jul', '07')
    cal = cal.replace('aug', '08')
    cal = cal.replace('sep', '09')
    cal = cal.replace('oct', '10')
    cal = cal.replace('nov', '11')
    cal = cal.replace('dec', '12')

    d = (''.join(map(a2space, cal))).split(" ")
    d = d[:6] # truncate at seconds
    d = [float(t) for t in d] + [0] * (6 - len(d))
    if d[1] == 0.0:
        d = d[:1] + [1.0] + d[2:]
    if d[2] == 0.0:
        d = d[:2] + [1.0] + d[3:]
    dt = datetime.timedelta(days=d[2] - 1.0, hours=d[3], minutes=d[4],
                            seconds=d[5])
    d = datetime.datetime(int(d[0]), int(d[1]), 1) + dt
    return d.isoformat()

def cal2time(cal, scale='utc'):
    """Calendar date to astropy `Time`.

    Parameters
    ----------
    cal : string or array
      Calendar date.  See `cal2iso` for details.
    scale : string, optional
      See `astropy.time.Time`.

    Returns
    -------
    doy : int or list
      Day of year.

    """
    from astropy.time import Time
    return Time(cal2iso(cal), format='isot', scale=scale)

def date_len(date):
    """Length of the date, or 0 if it is a scalar.

    Useful for routines that use `date2time`.

    Parameters
    ----------
    date : string, float, astropy Time, datetime, array, None
      Some time-like thingy, or `None`.

    Returns
    -------
    n : int
      The length of the array, or 0 if it is a scalar.

    """

    from astropy.time import Time
    if isinstance(date, (list, tuple, np.ndarray)):
        return len(date)
    elif isinstance(date, Time):
        if date.isscalar:
            return 0
        else:
            return len(date)
    elif date is None:
        return 0
    elif np.isscalar(date):
        return 0
    else:
        return len(date)

@singledispatch
def date2time(date, scale='utc'):
    """Lazy date to astropy `Time`.

    Parameters
    ----------
    date : string, float, astropy Time, datetime, or array
      Some time-like thingy, or `None` to return the current date (UTC).
    scale : string, optional
      See `astropy.time.Time`.

    Returns
    -------
    date : astropy Time

    """
    if (date is not None):
        raise ValueError("Bad date: {} ({})".format(date, type(date)))
    return astropy.time.Time(datetime.datetime.utcnow(), scale=scale,
                             format='datetime')

@date2time.register(astropy.time.Time)
def _(date, scale='utc'):
    return astropy.time.Time(date, scale=scale)

@date2time.register(int)
@date2time.register(float)
def _(date, scale='utc'):
    return jd2time(date, scale=scale)

@date2time.register(str)
def _(date, scale='utc'):
    return cal2time(date, scale=scale)

@date2time.register(datetime.datetime)
def _(date, scale='utc'):
    return astropy.time.Time(date, scale=scale)

@date2time.register(list)
@date2time.register(tuple)
@date2time.register(np.ndarray)
def _(date, scale='utc'):
    date = [date2time(d, scale=scale) for d in date]
    return astropy.time.Time(date)

def dh2hms(dh, format="{:02d}:{:02d}:{:06.3f}"):
    """Decimal hours as HH:MM:SS.SSS, or similar.

    Will work for degrees, too.

    Parameters
    ----------
    dh : float
    format : string, optional
      Use this format, e.g., for [+/-]HH:MM, use "{:+02d}:{:02d}".

    Returns
    -------
    hms : string

    """

    sign = -1 if dh < 0 else 1
    dh = abs(dh)
    hh = int(dh)
    mm = int((dh - hh) * 60.0)
    ss = ((dh - hh) * 60.0 - mm) * 60.0
    if ss >= 60:
        ss -= 60
        mm += 1
    if mm >= 60:
        mm -= 60
        hh += 1
    return format.format(sign * hh, mm, ss)

def doy2md(doy, year):
    """Day of year in MM-DD format.

    Parameters
    ----------
    doy : int or array
      Day(s) of year.
    year : int
      The year in question.

    Returns
    -------
    md : string or list
      MM-DD for each `doy`.

    """

    jd0 = s2jd('{0}-12-31'.format(year - 1))
    if isinstance(doy, (tuple, list, numpy.ndarray)):
        md = []
        for i in range(len(doy)):
            md.append(jd2dt(jd0 + doy[i]).strftime('%m-%d'))
    else:
        md = jd2dt(jd0 + doy).strftime('%m-%d')
    return md

def hms2dh(hms):
    """HH:MM:SS to decimal hours.

    This function may also be used to format degrees.

    Parameters
    ----------
    hms : string or array
      A string of the form "HH:MM:SS" (: may be any non-digit except
      ., +, or -).  Alternatively, `hms` may take the form [hh, mm,
      ss].  If any element is < 0, then the result will be < 0.
      Caution: The numeric value -0 is not < 0, but this function will
      treat the string value "-0" as < 0.

    Returns
    -------
    dh : float or list
      Decimal hours.

    """
    if (isinstance(hms, (list, tuple, np.ndarray))
        and isinstance(hms[0], (list, tuple, np.ndarray, str))):
        return [hms2dh(x) for x in hms]

    def a2space(c):
        if c.isdigit() or c in ['.', '+', '-']:
            return c
        else:
            return " "

    if isinstance(hms, str):
        s = -1 if hms.find('-') >= 0 else 1
        hms = ''.join(map(a2space, hms)).split()
        hms = s * np.array(hms, dtype=float)
    else:
        hms = np.array(hms, dtype=float)

    if len(hms) > 3:
        raise ValueError("hms has more than 3 parts.")

    # If any value is < 0, the final result should be < 0
    s = -1 if (np.sign(hms) < 0).any() else 1
    hms = np.abs(hms)

    dh = hms[0]
    if len(hms) > 1: dh += hms[1] / 60.0
    if len(hms) > 2: dh += hms[2] / 3600.0
    return s * dh

def jd2doy(jd, jd2=None, scale='utc'):
    """Julian date to day of year.

    Parameters
    ----------
    jd : float or array
      Julian date.
    jd2 : float or array, optional
      Second part of `jd`, to preserve precision, if needed.  Must
      have the same number of elements as `jd`.
    scale : string, optional
      See `astropy.time.Time`.

    Returns
    -------
    doy : int or list
      Day of year.

    """
    from astropy.time import Time
    t = Time(jd, val2=jd2, format='jd', scale=scale)
    if len(t) > 1:
        return [int(x.yday.split(':')[1]) for x in t]
    else:
        return int(t.yday.split(':')[1])

def jd2time(jd, jd2=None, scale='utc'):
    """Julian date to astropy `Time`.

    Parameters
    ----------
    jd : float or array
      Julian date.
    jd2 : float or array, optional
      Second part of `jd`, to preserve precision, if needed.  Must
      have the same number of elements as `jd`.
    scale : string, optional
      See `astropy.time.Time`.

    Returns
    -------
    t : astropy Time

    """
    from astropy.time import Time
    return Time(jd, val2=jd2, format='jd', scale=scale)

def timestamp(format='%Y%m%d'):
    """The current date/time as a string.

    Parameters
    ----------
    format : string
      The time format.

    """
    from datetime import datetime
    return datetime.utcnow().strftime(format)

def tz2utc(date, tz):
    """Offset between local time and UTC.

    Parameters
    ----------
    date : various
      The local time, in any format acceptable to `date2time`.
    tz : string
      date will be processed via `pytz`.

    Returns
    -------
    offset : datetime.timedelta
      The UTC offset.

    """

    from pytz import timezone
    return timezone(tz).utcoffset(date2time(date).datetime)

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

    from astropy.units import Quantity
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

    from astropy.units import Quantity
    if not isinstance(x, Quantity):
        q = x * unit
    else: 
        q = x

    return q.to(unit, **keywords)

def asValue(x, unit_in, unit_out):
    """Return the value of `x` in units of `unit_out`.

    Parameters
    ----------
    x : float, array, Quantity, astropy Angle
      The parameter to consider.
    unit_in : astropy Unit
      If `x` is a float or array, assume it is in these units.
    unit_out : astropy Unit
      `x` will be converted into these output units.

    Returns
    -------
    y : float or ndarray

    Raises
    ------
    ValueError when a `x` cannot be converted to `unit_out`.

    """

    import astropy.units as u
    from astropy.units import Quantity
    from astropy.coordinates import Angle

#    if isinstance(x, Angle):
#        if unit_out == u.deg:
#            y = x.degrees
#        elif unit_out == u.rad:
#            y = x.radians
#        else:
#            raise ValueError("Cannot convert Angle to units of {}".format(
#                    unit_out))
#    elif isinstance(x, Quantity):
    if isinstance(x, (Angle, Quantity)):
        y = x.to(unit_out).value
    else:
        y = (x * unit_in).to(unit_out).value

    return y

def autodoc(glbs, width=15, truncate=True):
    """Update a module's docstring with a summary of its functions.

    The docstring of the module is searched for the names of functions
    and classes (one per line), which are appended with their one-line
    summaries.

    Parameters
    ----------
    glbs : dict
      The `globals` dictionary from a module.  __doc__ will be
      updated.
    width : int, optional
      The width of the function table cell.
    truncate : bool, optional
      If `True`, truncate the newly generated lines at 80 characters.

    """

    try:
        docstring = glbs['__doc__'].splitlines()
    except AttributeError:
        return

    newdoc = ""
    for i in range(len(docstring)):
        s = docstring[i]
        x = s.strip()
        if x in glbs:
            if callable(glbs[x]):
                try:
                    topline = glbs[x].__doc__.splitlines()[0].strip()
                    summary = "{:{width}s} - {:}".format(
                        x, topline, width=width)
                    s = s.replace(x, summary)
                    if truncate:
                        s = s[:80]
                except AttributeError:
                    pass
        newdoc += s + "\n"

    glbs['__doc__'] = newdoc

def file2list(f, strip=True):
    """A list from strings from a file.

    Parameters
    ----------
    f : string
      The name of the file to read.
    strip : bool, optional
      Set to `True` to strip whitespace from each line.

    Returns
    -------
    lines : list
      The contents of the file.

    """

    lines = []
    with open(f, 'r') as inf:
        for line in inf.readlines():
            lines.append(line.strip() if strip else line)
    return lines

def horizons_csv(table):
    """Read a JPL/HORIZONS CSV file into a Table.

    May not be feature complete: need to test all input sources.

    Parameters
    ----------
    table : str, file-like, list
      Input table as a file name, file-like object, list of strings,
      or single newline-separated string.

    Returns
    -------
    astropy.table.Table

    """

    from astropy.extern import six
    from astropy.io import ascii

    def split(line):
        import re
        return re.split('\s*,\s*', line.strip())

    if isinstance(table, six.string_types):
        inf = open(table, 'r')
    else:
        inf = table

    header = []
    for line in inf:
        if line.startswith('$$SOE'):
            break
        header.append(line)

    colnames = split(header[-2].strip())
    for i in range(len(colnames)):
        if colnames[i] == '':
            colnames[i] = 'col{}'.format(i)
        
    data = ''
    for line in inf:
        if line.startswith('$$EOE'):
            break
        data += line

    tab = ascii.read(data, names=colnames)

    footer = ''
    for line in inf:
        footer += line

    tab.meta['header'] = ''.join(header)
    tab.meta['footer'] = footer
    
    return tab

def spectral_density_sb(s):

    """Equivalence pairs for spectra density surface brightness.

    For use with `astropy.units`.

    Parameters
    ----------
    s : Quantity
      The spectral unit and value.

    Returns
    -------
    equiv : list
      A list of equivalence pairs.

    Notes
    -----
    Basically a copy of `u.spectral_density`, but per steradian.

    """

    import astropy.constants as const
    import astropy.units as u

    c_Aps = const.c.si.value * 10**10

    fla = u.erg / u.angstrom / u.cm**2 / u.s / u.sr
    fnu = u.erg / u.Hz / u.cm**2 / u.s / u.sr
    nufnu = u.erg / u.cm**2 / u.s / u.sr
    lafla = nufnu

    sunit = s.decompose().unit
    sfactor = s.decompose().value

    def converter(x):
        return x * (sunit.to(u.AA, sfactor, u.spectral())**2 / c_Aps)

    def iconverter(x):
        return x / (sunit.to(u.AA, sfactor, u.spectral())**2 / c_Aps)

    def converter_fnu_nufnu(x):
        return x * sunit.to(u.Hz, sfactor, u.spectral())

    def iconverter_fnu_nufnu(x):
        return x / sunit.to(u.Hz, sfactor, u.spectral())

    def converter_fla_lafla(x):
        return x * sunit.to(u.AA, sfactor, u.spectral())

    def iconverter_fla_lafla(x):
        return x / sunit.to(u.AA, sfactor, u.spectral())

    return [
        (u.AA, fnu, converter, iconverter),
        (fla, fnu, converter, iconverter),
        (u.AA, u.Hz, converter, iconverter),
        (fla, u.Hz, converter, iconverter),
        (fnu, nufnu, converter_fnu_nufnu, iconverter_fnu_nufnu),
        (fla, lafla, converter_fla_lafla, iconverter_fla_lafla),
    ]

def timesten(v, sigfigs):
    """Format a number in LaTeX style scientific notation: $A\times10^{B}$.

    Parameters
    ----------
    v : float, int, or array
      The number(s) for format.
    sigfigs : int
      The number of significant figures.

    """

    if np.iterable(v):
        s = []
        for i in range(len(v)):
            s.append(timesten(v[i], sigfigs))
        return s

    s = "{0:.{1:d}e}".format(v, sigfigs - 1).split('e')
    s = r"${0}\times10^{{{1:d}}}$".format(s[0], int(s[1]))
    return s

def write_table(fn, tab, header, comments=[], **kwargs):
    """Write an astropy Table with a simple header.

    Parameters
    ----------
    fn : string
      The name of the file to write to.
    tab : astropy Table
      The table to write.
    header : dict
      A dictionary of keywords to save or `None`.  Use an
      `OrderedDict` to preserve header keyword order.
    comments : list
      A list of comments to add to the top of the file.  Each line
      will be prepended with a comment character.
    **kwargs
      Keyword arguments for `tab.write()`.  Default format is
      'ascii.fixed_width_two_line'.

    """

    format = kwargs.pop('format', 'ascii.fixed_width_two_line')
    with open(fn, 'w') as outf:
        outf.write("# {}\n#\n".format(date2time(None).iso))

        for c in comments:
            outf.write("# {}\n".format(c))

        for k, v in header.items():
            outf.write("# {} = {}\n".format(k, str(v)))

        outf.write('#\n')

        tab.write(outf, format=format, **kwargs)

# summarize the module
autodoc(globals())
