# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
image.analysis --- Analyze (astronomical) images.
=================================================

.. autosummary::
   :toctree: generated/

   anphot
   apphot
   azavg
   bgfit
   centroid
   imstat
   linecut
   polyfit2d
   radprof
   trace

.. todo:: Re-write anphot to generate pixel weights via rarray, rather
   than sub-sampling the images.  Update apphot and azavg, if needed.

.. todo:: Re-write linecut to generate pixel weights via xarray?

"""

from __future__ import print_function
import numpy as np
from . import core

__all__ = [
    'anphot',
    'apphot',
    'azavg',
    'bgfit',
    'centroid',
    'imstat',
    'linecut',
    'polyfit2d',
    'radprof',
    'trace'
]

def anphot(im, yx, rap, subsample=4):
    """Simple annular aperture photometry.

    Pixels may be sub-sampled (drizzled).

    `anphot` is not optimized and is best for extended objects in
    small images.

    Parameters
    ----------
    im : array or array of arrays
      An image, cube, or array of images on which to measure
      photometry.  For data cubes, the first axis iterates over the
      images.  All images must have the same shape.
    yx : array
      The `y, x` center of the aperture(s).  Only one center is
      allowed. [pixels]
    rap : float or array
      Aperture radii.  [pixels]
    subsample : int, optional
      The sub-pixel sampling factor.  Set to `<= 1` for no sampling.
      This will sub-sample the entire image.

    Returns
    -------
    n : ndarray
      The number of pixels per annular bin.
    f : ndarray
      The annular photometry.  If `im` is a set of images of depth
      `N`, `f` will have shape `N x len(rap)`

    """

    _im = np.array(im)
    ndim = _im.ndim  # later, we will flatten the images

    assert ndim in [2, 3], ("Only images, data cubes, or tuples/lists"
                            " of images are allowed.")

    yx = np.array(yx, float)
    assert yx.shape == (2,), "yx has incorrect shape."

    if subsample > 1:
        if ndim == 3:
            _im = np.array([core.rebin(x, subsample, flux=True) for x in _im])
        else:
            _im = core.rebin(_im, subsample, flux=True)
        
        yx = yx * subsample + (subsample - 1) / 2.0

    sz = _im.shape[-2:]

    r = core.rarray(sz, yx=yx, subsample=10) / float(subsample)

    # annular photometry via histograms, histogram via digitize() to
    # save CPU time when mulitiple images are passed

    # flatten all arrays for digitize
    r = r.flatten()
    if ndim == 3:
        N = _im.shape[0]
        M = _im.shape[1] * _im.shape[2]
        _im = _im.flatten().reshape((N, M))
    else:
        _im = _im.flatten()

    bins = np.digitize(r, rap)
    n = np.zeros(len(rap))
    if ndim == 3:
        f = np.zeros((len(_im), len(rap)))
        for i in range(len(rap)):
            j = np.flatnonzero(bins == i)
            f[:, i] = np.sum(_im[:, j], 1)
            n[i] = len(j)
    else:
        f = np.zeros(len(rap))
        for i in range(len(rap)):
            j = np.flatnonzero(bins == i)
            f[i] = np.sum(_im[j])
            n[i] = len(j)

    n /= float(subsample**2)
    return n, f

def apphot(im, yx, rap, subsample=4):
    """Simple aperture photometry.

    Pixels may be sub-sampled (drizzled).

    `apphot` is not optimized and is best for single objects in small
    images.

    Parameters
    ----------
    im : array or array of arrays
      An image, cube, or tuple of images on which to measure
      photometry.  For data cubes, the first axis iterates over the
      images.  All images must have the same shape.
    yx : array
      The `y, x` center of the aperture(s). Only one center is
      allowed. [pixels]
    rap : float or array
      Aperture radii.  [pixels]
    subsample : int, optional
      The sub-pixel sampling factor.  Set to `<= 1` for no sampling.
      This will sub-sample the entire image.

    Returns
    -------
    n : ndarray
      The number of pixels per aperture.
    f : ndarray
      The aperture photometry.  If `im` is a set of images of depth
      `N`, `f` will have shape `N x len(rap)`.
      
    """
    if not np.iterable(rap):
        rap = np.array([rap])
    n, f = anphot(im, xy, rap, subsample=subsample)
    return n.cumsum(-1), f.cumsum(-1)

def azavg(im, yx, raps=None, subsample=4, **kwargs):
    """Create an aziumthally averaged image.

    A radial profile is generated and interpolated back onto the final
    image.

    Parameters
    ----------
    im : array
      The image to process.
    yx : array
      `y, x` point around which to determine the azimuthal average.
    raps : int or array
      The number of radial steps or the edges of radial bins used to
      derive the profile.  The default bins will be logarithmicly
      stepped (base 2) from 1 to the largest radial distance in the
      image.
    subsample : int, optional
      The subsampling factor used for annular photometry.
    **kwargs :
      Keyword arguments to pass to `scipy.interpolate.interp1d` for
      creating the final image.  Default parameters are `kind='zero'`,
      `bounds_error=False`.

    Returns
    -------
    aa : ndarray
      The azimthual average image.

    Notes
    -----

    Apertures without any pixels are removed from the azimuthal
    average before interpolation.

    """

    from scipy.interpolate import interp1d
    from ..util import takefrom 

    kind = kwargs.pop('kind', 'zero')
    bounds_error = kwargs.pop('bounds_error', False)

    r = core.rarray(im.shape, yx, subsample=10)

    if raps is None:
        maxr = int(r.max()) + 1
        raps = np.logspace(0, np.log2(maxr), int(np.log2(maxr) * 2), base=2)

    n, f = anphot(im, yx, raps, subsample=subsample)
    n, f, raps = takefrom((n, f, raps), n != 0)

    f /= n
    f = np.r_[f, f[-1]]
    raps = np.r_[0, raps]

    aa = interp1d(raps, f, kind=kind, bounds_error=bounds_error,
                  **kwargs)
    aa = aa(r).reshape(im.shape)

    return aa

def bgfit(im, unc=None, order=1, mask=True):
    """Fit an image background.

    Parameters
    ----------
    im : array
      The image.
    unc : array, optional
      Image uncertainties as a 2D array.
    order : int
      The polynomial order (in one dimension) of the fitting function.
    mask : array, optional
      A pixel mask, where `True` indicates background pixels.  NaNs
      are always ignored.

    Returns
    -------
    bg : ndarray
      An image of the best-fit background.

    """

    mask *= np.isfinite(im)

    # yy and xx are the coordinates for the original image, y and x
    # are the coordinates in the flux array
    yy, xx = np.indices(im.shape)
    x = xx[mask]
    y = yy[mask]

    # the flux array: u
    u = im[mask]

    # the uncertanty array: v
    if unc is None:
        v = np.ones_like(u)
    else:
        v = unc[mask]

    cx, cy, cov = core.polyfit2d(u, y, x, unc=v, order=order)
    bg = np.polyval(cy, yy) + np.polyval(cx, xx)

    return bg

def centroid(im, yx=None, box=None, niter=1, shrink=True, silent=True):
    """Centroid (center of mass) of an image.

    Parameters
    ----------
    im : ndarray
      A 2D image on which to centroid.
    yx : float array, optional
      `(y, x)` guess for the centroid.  The default guess is the image
      peak.
    box : array, optional
      Specify the size of the box over which to compute the centroid.
      This may be an integer, or an array (width, height).  The
      default is to use the whole image.
    niter : int, optional
      When box is not None, iterate niter times.
    shrink : bool, optional
      When iterating, decrease the box size by sqrt(2) each time.
    silent : bool, optional
      Suppress any print commands.

    Returns
    -------
    cyx : ndarray
      The computed center of mass.  The lower-left corner of a pixel
      is -0.5, -0.5.

    """

    if yx is None:
        yx = np.unravel_index(np.nanargmax(im), im.shape)

    if box is None:
        box = np.array((im.shape[0], im.shape[1]))
    elif np.size(box) == 1:
        box = np.array((box, box)).reshape((2,))

    y0 = max(yx[0] - box[0] / 2, 0)
    y1 = min(yx[0] + box[0] / 2 + 1, im.shape[0])
    x0 = max(yx[1] - box[1] / 2, 0)
    x1 = min(yx[1] + box[1] / 2 + 1, im.shape[1])
    subim = im[y0:y1, x0:x1].copy()
    subim -= subim.min()

    y, x = np.indices(subim.shape)
    cyx = np.array([(subim * y).sum() / subim.sum() + y0,
                    (subim * x).sum() / subim.sum() + x0], float)

    if niter > 1:
        if shrink:
            box = (box / np.sqrt(2)).round().astype(int)
            box = box + (box % 2 - 1)  # keep it odd

        if not silent:
            print("y, x = {0[0]:.1f}, {0[1]:.1f}, next box size = {1}".format(
                    cyx, str(box)))

        if np.any(box < 2):
            if not silent:
                print(" - Box size too small -")
            return cyx

        cyx = centroid(im, yx=cyx, box=box, niter=niter-1,
                        shrink=shrink, silent=silent)
    else:
        if not silent:
            print("x, y = {0:.1f}, {1:.1f}".format(cx, cy))

    return cyx

def imstat(im, **kwargs):
    """Get some basic statistics from an array.

    NaNs are ignored.

    Parameters
    ----------
    im : array
      The image to operate on.

    **kwargs
      Additional keywords to pass on to meanclip.

    Returns
    -------
    stats : dict
      Statistics commonly used in astronomical interpretations of
      images: min, max, mean, median, mode, stdev, sigclip mean,
      sigclip median, sigclip stdev (3-sigma clipping).  The mode is
      estimated assuming a unimodal data set that isn't too
      asymmetric: mode = 3 * median - 2 * mean, where median and mean
      are the sigma-clipped estimates.

    """

    from ..util import meanclip

    mc = meanclip(im, full_output=True, **keywords)
    scmean, scstdev = mc[:2]
    scmedian = np.median(im.flatten()[mc[2]])

    return dict(min = np.nanmin(im),
                max = np.nanmax(im),
                mean = scipy.stats.nanmean(im.ravel()),
                median = math.nanmedian(im),
                mode = 3.0 * scmedian - 2.0 * scmean,
                stdev = scipy.stats.nanstd(im.ravel()),
                scmean = scmean,
                scmedian = scmedian,
                scstdev = scstdev)

def linecut(im, yx, width, length, pa, subsample=4):
    """Photometry along a line.

    Pixels may be sub-sampled (drizzled).

    Parameters
    ----------
    im : array or array of arrays
      An image, cube, or tuple of images from which to measure
      photometry.  For data cubes, the first axis iterates over the
      images.  All images must have the same shape.
    yx : array-like
      The y, x center of the extraction.
    width : float
    length: float or array
      If `length` is a float, the line cut will be boxes of size
      `width` x `width` along position angle `pa`, spanning a total
      length of `length`.  If `length` is an array, it specifies the
      bin edges along `pa`, each bin having a width of `width`.
    pa : float
      Position angle measured counter-clockwise from the
      x-axis. [degrees]
    subsample : int, optional
      The sub-pixel sampling factor.  Set to <= 1 for no sampling.

    Returns
    -------
    x : ndarray
      The centers of the bins, measured along the line cut.
    n : ndarray
      The number of pixels per bin.
    f : ndarray
      The line cut photometry.
      
    """

    _im = np.array(im)
    ndim = _im.ndim  # later, we will flatten the images
    assert ndim in [2, 3], ("Only images, data cubes, or tuples/lists"
                            " of images are allowed.")

    yx = np.array(yx, float)
    assert yx.shape == (2,), "yx has incorrect shape."

    if subsample > 1:
        if ndim == 3:
            _im = np.array([core.rebin(x, subsample, flux=True) for x in _im])
        else:
            _im = core.rebin(_im, subsample, flux=True)
        
        yx = yx * subsample + (subsample - 1) / 2.0

    sz = _im.shape[-2:]

    # x is parallel to length, y is perpendicular to it
    a = np.radians(pa)

    x = xarray(sz, yx, rot=a, dtype=float) / subsample
    y = np.abs(yarray(sz, yx, rot=a, dtype=float) / subsample)

    if np.iterable(length):
        Nbins = len(length) - 1
        xap = np.array(length)
    else:
        # carefully set up the bins along x, the first bin is thrown away
        Nbins = int(np.floor(length / float(width)))
        xap = width * (np.arange(Nbins + 1) - Nbins / 2.0)

    # line cut photometry via histograms, histogram via digitize() to
    # save CPU time when mulitiple images are passed

    # flatten all arrays for digitize
    x = x.flatten()
    y = y.flatten()
    if ndim == 3:
        N = _im.shape[0]
        M = _im.shape[1] * _im.shape[2]
        _im = _im.flatten().reshape((N, M))
    else:
        _im = _im.flatten()

    # Bin data with digitize().  Note that bin 0 is to the left of our
    # line cut.  We will discard this bin below.
    bins = np.digitize(x, xap)
    n = np.zeros(Nbins)
    if ndim == 3:
        f = np.zeros((len(_im), Nbins))
        for i in range(Nbins):
            ap = (bins == (i + 1)) * (y < width / 2.0)
            j = np.flatnonzero(ap)
            n[i] = len(j)
            f[:, i] = np.sum(_im[:, j], 1)
    else:
        f = np.zeros(Nbins)
        for i in range(Nbins):
            ap = (bins == (i + 1)) * (y < width / 2.0)
            j = np.flatnonzero(ap)
            n[i] = len(j)
            f[i] = np.sum(_im[j])

    n /= float(sample**2)
    return math.midstep(xap), n, f

def polyfit2d(f, y, x, unc=None, order=1):
    """Fit a polynomial surface to 2D data.

    Assumes the axes are independent of each other.

    Evaluate the fit via: np.polyval(polyy, y) + np.polyval(polyx, x).

    Parameters
    ----------
    f : array
     The array to fit.
    y, x : array
      The coordinates for each value of `f`.
    unc : array, optional
      The uncertainties on each value of `f`, or a single value for
      all points.  If `None`, then 1 is assumed.
    order : int, optional
      The polynomial order (in one dimension) of the fit.

    Returns
    -------
    polyx, polyy : ndarray
      The polynomial coefficients, in the same format as from
      `np.polyfit`.

    cov : ndarray
      The covariance matrix.

    v1.0.0 Written by Michael S. Kelley, UMD, Mar 2009
    """

    from scipy.optimize import leastsq as lsq

    # the fitting function
    def chi(p, y, x, f, unc, order):
        cy = p[:1+order]
        cx = p[1+order:]
        model = np.zeros(f.shape) + np.polyval(cy, y) + np.polyval(cx, x)
        chi = (f - model) / unc
        return chi

    if unc is None:
        unc = 1.0

    # run the fit
    guess = np.zeros((order + 1) * 2)
    result = lsq(chi, guess, args=(y, x, f, unc, order), full_output=True)
    fit = result[0]
    cov = result[1]
    cy = fit[:1+order]
    cx = fit[1+order:]

    return cx, cy, cov

def radprof(im, yx, bins=10, range=None, subsample=4):
    """Radial surface brightness profile of an image.

    Profile is generated via `anphot`.

    Parameters
    ----------
    im : array
      The image to examine.
    yx : array
      The `y, x` center of the radial profile.
    bins : int or array, optional
      The number of radial bins, or the radial bin edges.
    range : array, optional
      If bins is a single number, set the bin range to this, otherwise
      set the range to from 0 to the maximal radius.
    subsample : int, optional
      Sub-sample the image at this scale factor.

    Returns
    -------
    r : ndarray
      The mean radius of each point in the surface brightness profile.
    sb : ndarray
      The surface brightness at each `r`.
    n : ndarray
      The number of pixels in each bin.

    """

    from ..util import takefrom

    if range is None:
        rmax = np.sqrt(max(
                sum(yx**2),
                sum((yx - np.r_[0, im.shape[1]])**2),
                sum((yx - np.r_[im.shape[0], 0])**2),
                sum((yx - np.r_[im.shape])**2)))
        range = [0, int(rmax) + 1]

    if np.iterable(bins):
        rap = bins
    else:
        rap = np.linspace(range[0], range[1], bins + 1)

    n, f = anphot(im, yx, rap, subsample=subsample)
    r = core.rarray(im.shape, yx=yx, subsample=10)
    r = anphot(r, yx, rap, subsample=subsample)

    i = n != 0
    f[i] = f[i] / n[i]
    return r, f, n

def trace(im, err, guess):
    """Trace the peak pixels along the second axis of an image.

    Parameters
    ----------
    im : array
      The image to trace.
    err : array
      The image uncertainties.  Set to None for unweighted fitting.
    guess : array
      The initial guesses for Gaussian fitting the y-value at x=0:
      (mu, sigma, height, [m, b]) = position (mu), width (sigma),
      height, and linear background m*x + b.

    Returns
    -------
    y : ndarray
      y-axis positions of the peak, along the second axis.

    """

    from scipy.optimize import leastsq as lsq
    from ..util import gaussian

    def chi(p, x, y, err):
        if len(p) > 3:
            mu, sigma, height, m, b = p
        else:
            mu, sigma, height = p
            m, b = 0, 0
        model = gaussian(x, mu, sigma) * height + m * x + b
        chi = (y - model) / err
        return chi

    y = np.zeros(im.shape[1])
    y0 = np.arange(im.shape[0])
    last = guess

    for i in xrange(im.shape[1]):
        if err is None:
            err = np.ones_like(y)
        fit, err = lsq(chi, last, (np.array(x), np.array(y), np.array(err)))
        y[i] = fit[0]
        last = fit

    return y


# update module docstring
from ..util import autodoc
autodoc(globals())
del autodoc
