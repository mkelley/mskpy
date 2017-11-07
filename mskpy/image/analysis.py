# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
image.analysis --- Analyze (astronomical) images.
=================================================

.. autosummary::
   :toctree: generated/

   anphot
   apphot
   apphot_by_wcs
   azavg
   bgfit
   bgphot
   centroid
   find
   fwhm
   gcentroid
   imstat
   linecut
   polyfit2d
   radprof
   spextract
   trace

.. todo:: Re-write anphot to generate pixel weights via rarray, rather
   than sub-sampling the images?  Update apphot and azavg, if needed.

.. todo:: Re-write linecut to generate pixel weights via xarray?

"""

import numpy as np
from . import core

__all__ = [
    'anphot',
    'apphot',
    'apphot_by_wcs',
    'azavg',
    'bgfit',
    'bgphot',
    'centroid',
    'gcentroid',
    'find',
    'fwhm',
    'imstat',
    'linecut',
    'polyfit2d',
    'radprof',
    'spextract',
    'trace'
]

class UnableToCenter(Exception):
    pass

class LostTraceWarning(Warning):
    pass

class UnableToTrace(Exception):
    pass

class NoSourcesFound(Exception):
    pass


def anphot(im, yx, rap, subsample=4, squeeze=True):
    """Simple annular aperture photometry.

    Pixels may be sub-sampled, and sub-sampling may be CPU and memory
    intensive.

    Parameters
    ----------
    im : array or array of arrays
      An image, cube, or array of images on which to measure
      photometry.  For data cubes, the first axis iterates over the
      images.  All images must have the same shape.
    yx : array
      The `y, x` center of the aperture(s), or an Nx2 length array of
      centers. [pixels]
    rap : float or array
      Aperture radii.  The inner-most aperture will be the annulus 0
      to `min(rap)`.  [pixels]
    subsample : int, optional
      The sub-pixel sampling factor.  Set to `<= 1` for no sampling.
      This will sub-sample the entire image.
    squeeze : bool, optional
      Set to `True` to sqeeze single length dimensions out of the
      results.

    Returns
    -------
    n : ndarray
      The number of pixels per annular bin, either shape `(len(rap),)`
      or `(len(yx), len(rap))`.
    f : ndarray
      The annular photometry.  The shape will be one of:
        `(len(rap),)`
        `(len(yx), len(rap))`
        `(len(im), len(yx), len(rap))`

    """

    _im = np.array(im)
    assert _im.ndim in [2, 3], ("Only images, data cubes, or tuples/lists"
                                " of images are allowed.")
    if _im.ndim == 2:
        _im = _im.reshape((1,) + _im.shape)

    yx = np.array(yx, float)
    assert yx.ndim in [1, 2], "yx must be one or two dimensional."
    if yx.ndim == 1:
        assert yx.shape[0] == 2, "yx must have length 2."
        yx = yx.reshape((1, 2))
    else:
        assert yx.shape[1] == 2, "Second axis of yx must have length 2."

    if not np.iterable(rap):
        rap = np.array([rap])

    if subsample > 1:
        _im = np.array([core.rebin(x, subsample, flux=True) for x in _im])
        yx = yx * subsample + (subsample - 1) / 2.0

    sz = _im.shape[-2:]

    # flatten all arrays for digitize
    N = _im.shape[0]
    M = _im.shape[1] * _im.shape[2]
    _im = _im.flatten().reshape((N, M))

    n = np.zeros((len(yx), len(rap)))
    f = np.zeros((N, len(yx), len(rap)))

    # annular photometry via histograms, histogram via digitize() to
    # save CPU time when mulitiple images are passed
    for i in range(len(yx)):
        r = core.rarray(sz, yx=yx[i], subsample=10) / float(subsample)
        bins = np.digitize(r.flatten(), rap)
        for j in range(len(rap)):
            ap = np.flatnonzero(bins == j)
            f[:, i, j] = np.sum(_im[:, ap], 1)
            n[i, j] = len(ap)

    n /= float(subsample**2)
    if squeeze:
        return n.squeeze(), f.squeeze()
    else:
        return n, f

def apphot(im, yx, rap, subsample=4, **kwargs):
    """Simple aperture photometry.

    Pixels may be sub-sampled, and sub-sampling may be CPU and memory
    intensive.

    Parameters
    ----------
    im : array or array of arrays
      An image, cube, or tuple of images on which to measure
      photometry.  For data cubes, the first axis iterates over the
      images.  All images must have the same shape.
    yx : array
      The `y, x` center of the aperture(s), or an Nx2 length array of
      centers. [pixels]
    rap : float or array
      Aperture radii.  [pixels]
    subsample : int, optional
      The sub-pixel sampling factor.  Set to `<= 1` for no sampling.
      This will sub-sample the entire image.
    **kwargs
      Any `anphot` keyword argument.

    Returns
    -------
    n : ndarray
      The number of pixels per aperture, either shape `(len(rap),)` or
      `(len(yx), len(rap))`.
    f : ndarray
      The annular photometry.  The shape will be one of:
        `(len(rap),)`
        `(len(yx), len(rap))`
        `(len(im), len(yx), len(rap))`  (for multiple images)

    """
    n, f = anphot(im, yx, rap, subsample=subsample, **kwargs)
    if np.size(rap) > 1:
        return n.cumsum(-1), f.cumsum(-1)
    else:
        return n, f

def apphot_by_wcs(im, coords, wcs, rap, centroid=False,
                  cfunc=None, ckwargs={}, **kwargs):
    """Simple aperture photometry, using a world coordinate system.

    The WCS only affects source locations, and does not affect
    aperture areas.

    Pixels may be sub-sampled, and sub-sampling may be CPU and memory
    intensive.

    Parameters
    ----------
    im : array
      An image, cube, or tuple of images on which to measure
      photometry.  For data cubes, the first axis iterates over the
      images.  All images must have the same shape.
    coords : astropy SkyCoord
      The coordinates (e.g., RA, Dec) of the targets.  Only sources
      interior to the image and at least `2 * max(rap)` from the edges
      are considered.
    wcs : astropy WCS
      The world coordinate system transformation.
    rap : float or array
      Aperture radii.  [pixels]
    centroid : bool, optional
      Set to `True` to centroid on each source with `cfunc`.  When
      multiple images are provided, the centroid is based on the first
      image.
    cfunc : function, optional
      The centroiding function to use, or `None` for `gcentroid`.
    ckwargs: dict
      Any `cfunc` keyword arguments.
    **kwargs:
      Any `apphot` keyword arguments.

    Returns
    -------
    yx : ndarray
      Pixel positions of all sources, `Nx2`.
    n : ndarray
      The number of pixels per aperture, either shape `(len(rap),)` or
      `(len(yx), len(rap))`.
    f : ndarray
      The annular photometry.  The shape will be one of:
        `(len(rap),)`
        `(len(yx), len(rap))`
        `(len(im), len(yx), len(rap))`  (for multiple images)

    """

    from ..util import between

    squeeze = kwargs.pop('squeeze', True)
    shape = np.array(im).shape

    # When SIP is included in WCS, coords.to_pixel can fail when the
    # coordinate is outside the image.  Use a two-pass system to
    # prevent crashing.
    x, y = coords.to_pixel(wcs, mode='wcs')
    i = between(y, [0, shape[0] + 1]) * between(x, [0, shape[1] + 1])
    if not np.any(i):
        raise NoSourcesFound

    x[i], y[i] = coords[i].to_pixel(wcs, mode='all')
    yx = np.c_[y, x]

    n = np.zeros((len(yx), np.size(rap)))
    if len(shape) == 3:
        shape = shape[1:]
        f = np.zeros((shape[0], ) + n.shape)
    else:
        f = np.zeros((1, ) + n.shape)

    max_rap = np.max(rap)
    sources = np.flatnonzero((x >= 2 * max_rap)
                             * (x <= (shape[1] - 2 * max_rap))
                             * (y >= 2 * max_rap)
                             * (y <= (shape[0] - 2 * max_rap)))

    if centroid:
        if cfunc is None:
            cfunc = gcentroid

        for i in sources:
            try:
                if len(shape) == 3:
                    yx[i] = cfunc(im[0], yx[i], **ckwargs)
                else:
                    yx[i] = cfunc(im, yx[i], **ckwargs)
            except UnableToCenter:
                pass

    _n, _f = apphot(im, yx[sources], rap, squeeze=False, **kwargs)
    n[sources] = _n
    f[:, sources] = _f
    if squeeze:
        return yx, n.squeeze(), f.squeeze()
    else:
        return yx, n, f

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
    elif isinstance(raps, int):
        maxr = int(r.max()) + 1
        raps = np.logspace(0, np.log2(maxr), raps, base=2)

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

    cx, cy, cov = polyfit2d(u, y, x, unc=v, order=order)
    bg = np.polyval(cy, yy) + np.polyval(cx, xx)

    return bg

def bgphot(im, yx, rap, ufunc=np.mean, squeeze=True, **kwargs):
    """Background photometry and error analysis in an annulus.

    Pixels are not sub-sampled.  The annulus is processed via
    `util.uclip`.

    Parameters
    ----------
    im : array or array of arrays
      An image, cube, or array of images on which to measure
      photometry.  For data cubes, the first axis iterates over the
      images.  All images must have the same shape.
    yx : array
      The `y, x` center of the aperture, or an `Nx2` array of centers.
    rap : array
      Inner and outer radii of the annulus.  [pixels]
    ufunc : function, optional
      The function with which to determine the background average.
    squeeze : bool, optional
      Set to `True` to sqeeze single length dimensions out of the
      results.
    **kwargs :
      Keyword arguments passed to `util.uclip`.

    Returns
    -------
    n : ndarray
      The number of pixels for each aperture.
    bg : ndarray
      The background level in the annulus, shape `(len(yx),)` or
      `(len(im), len(yx))`.
    sig : ndarray
      The standard deviation of the background pixels.  Same shape
      comment as for `bg`.

    """

    from ..util import uclip

    _im = np.array(im)
    assert _im.ndim in [2, 3], ("Only images, data cubes, or tuples/lists"
                                " of images are allowed.")
    if _im.ndim == 2:
        _im = _im.reshape((1,) + _im.shape)

    yx = np.array(yx, float)
    assert yx.ndim in [1, 2], "yx must be one or two dimensional."
    if yx.ndim == 1:
        assert yx.shape[0] == 2, "yx must have length 2."
        yx = yx.reshape((1, 2))

    assert yx.shape[1] == 2, "Second axis of yx must have length 2."

    rap = np.array(rap, float)
    assert rap.shape == (2,), "rap has incorrect shape."

    sz = _im.shape[-2:]

    n = np.zeros(len(yx))
    bg, sig = np.zeros((2, len(_im), len(yx)))

    for i in range(len(yx)):
        r = core.rarray(sz, yx=yx[i], subsample=10)
        annulus = (r >= rap.min()) * (r <= rap.max())
        for j in range(len(_im)):
            f = _im[j][annulus]
            bg[j, i], k, niter = uclip(f, ufunc, full_output=True, **kwargs)
            n[i] = len(k)
            sig[j, i] = np.std(f[k])

    if squeeze:
        return n.squeeze(), bg.squeeze(), sig.squeeze()
    else:
        return n, bg, sig

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

    y0 = max(yx[0] - box[0] // 2, 0)
    y1 = min(yx[0] + box[0] // 2 + 1, im.shape[0])
    x0 = max(yx[1] - box[1] // 2, 0)
    x1 = min(yx[1] + box[1] // 2 + 1, im.shape[1])
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
            print("y, x = {0[0]:.1f}, {0[1]:.1f}".format(cyx))

    return cyx

def find(im, sigma=None, thresh=2, centroid=None, fwhm=2, **kwargs):
    """Find sources in an image.

    Generally designed for point-ish sources.

    Parameters
    ----------
    im : array
      The image to search.
    sigma : float, optional
      The 1-sigma uncertainty in the background, or `None` to estimate
      the uncertainty with the sigma-clipped mean and standard
      deviation of `meanclip`.  If provided, then the image should be
      background subtracted.
    thresh : float, optional
      The detection threshold in sigma.  If a pixel is detected above
      `sigma * thresh`, it is an initial source candidate.
    centroid : function, optional
      The centroiding function, or `None` to use `gcentroid`.  The
      function is passed a subsection of the image.  All other
      parameters are provided via `kwargs`.
    fwhm : int, optional
      A rough estimate of the FWHM of a source, used for binary
      morphology operations.
    **kwargs
      Any keyword arguments for `centroid`.

    Returns
    -------
    cat : ndarray
      A catalog of `(y, x)` source positions.
    f : ndarray
      An array of approximate source fluxes (a background estimate is
      removed if `sigma` is `None`).

    """
    
    import scipy.ndimage as nd
    from ..util import meanclip

    assert isinstance(fwhm, int), 'FWHM must be integer'

    _im = im.copy()

    if sigma is None:
        stats = meanclip(_im, full_output=True)[:2]
        _im -= stats[0]
        sigma = stats[1]

    if centroid is None:
        centroid = gcentroid

    det = _im > thresh * sigma
    det = nd.binary_erosion(det, iterations=fwhm)  # remove small objects
    det = nd.binary_dilation(det, iterations=fwhm * 2 + 1)  # grow aperture size
    label, n = nd.label(det)

    yx = []
    f = []
    bad = 0
    for i in nd.find_objects(label):
        star = _im[i]

        if not np.isfinite(star.sum()):
            bad += 1
            continue

        try:
            cen = centroid(star, **kwargs)
        except:
            bad += 1
            continue

        if any(np.array(cen) < -0.5):
            bad += 1
            continue

        if any((cen[0] >= star.shape[0] - 0.5, cen[1] >= star.shape[1] - 0.5)):
            bad += 1
            continue

        cen += np.array((i[0].start, i[1].start))

        if not any(np.isfinite(cen)):
            bad += 1
            continue

        yx.append(cen)
        f.append(star.sum())

    print('[find] {} good, {} bad sources'.format(len(yx), bad))
    return np.array(yx), np.array(f)

def fwhm(im, yx, unc=None, guess=None, kind='radial', width=1, length=21,
         **kwargs):
    """Compute the FWHM of an image.

    Least-squares fit, optionally weighted by uncertainties.

    Parameters
    ----------
    im : array
      The image to fit.
    yx : array
      The center on which to fit.
    unc : array, optional
      Image uncertainties.
    guess : tuple, optional
      Initial guess.  See `util.gaussfit`.  If `guess` is `None`, the
      default fit is a Gaussian + constant background.  For radial
      fits, the center (`mu`) must be zero.
    kind : string, optional
      'radial', 'x', or 'y'.
    width : float, optional
      Extraction width for line cuts.
    length : float, optional
      Extraction length for line cuts, from -length/2 to length/2, 1 pixel
      steps.
    **kwargs
      Any `radprof` or `linecut` keyword.

    Returns
    -------
    fwhm : float
      The FHWM of the radial profile.

    """

    from scipy.optimize import leastsq as lsq
    from ..util import gaussfit

    length = np.arange(-length / 2.0, length / 2.0 + 1)

    assert kind in ['radial', 'x', 'y'], "Invalid kind."
    if kind == 'radial':
        R, I, n = radprof(im, yx, **kwargs)[:3]
        if guess is None:
            r = R[I < (I.ptp() / 2.0 + I.min())][0]  # guess for Gaussian sigma
            guess = (I.ptp(), 0.0, r / 2.35, I.min())
        args = (R, I)
    elif kind == 'x':
        x, n, I = linecut(im, yx, width, length, 0, **kwargs)
        if guess is None:
            r = x[I > (I.ptp() / 2.0 + I.min())][0]  # guess for Gaussian sigma
            guess = (I.max(), 0.0, r / 2.35, I.min())
        args = (x, I)
    elif kind == 'y':
        y, n, I = linecut(im, yx, width, length, 0, **kwargs)
        if guess is None:
            r = y[I > (I.ptp() / 2.0 + I.min())][0]  # guess for Gaussian sigma
            guess = (I.max(), 0.0, r / 2.35, I.min())
        args = (y, I)

    if unc is None:
        unc = args[1] / args[1]

    fit, err = gaussfit(args[0], args[1], unc, guess)

    return abs(fit[2]) * 2.35

def gcentroid(im, yx=None, box=None, niter=1, dim=2, shrink=True, silent=True):
    """Centroid (x-/y-cut Gaussian fit) of an image.

    The best-fit should be bounded within `box`.

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
    dim : int, optional
      Set to 1 to fit two 1D Gaussians, or 2 to fit a 2D Gaussian.
    shrink : bool, optional
      When iterating, decrease the box size by sqrt(2) each time.
    silent : bool, optional
      Suppress any print commands.

    Returns
    -------
    cyx : ndarray
      The computed center.  The lower-left corner of a pixel is -0.5,
      -0.5.

    """

    from photutils.centroids import centroid_1dg, centroid_2dg

    assert dim in [1, 2], '`dim` must be one of [1, 2]'
    if dim == 1:
        centroid_func = centroid_1dg
    else:
        centroid_func = centroid_2dg

    if yx is None:
        yx = np.array(np.unravel_index(np.nanargmax(im), im.shape), float)

    # the array index location of yx
    iyx = np.round(yx).astype(int)

    if box is None:
        box = np.array((im.shape[0], im.shape[1]))
    elif np.size(box) == 1:
        box = np.array((box, box)).reshape((2,))
    else:
        box = np.array(box)

    halfbox = box // 2

    yr = [max(iyx[0] - halfbox[0], 0),
          min(iyx[0] + halfbox[0] + 1, im.shape[0] - 1)]
    xr = [max(iyx[1] - halfbox[1], 0),
          min(iyx[1] + halfbox[1] + 1, im.shape[1] - 1)]
    ap = (slice(*yr), slice(*xr))

    try:
        cyx = centroid_func(im[ap])[::-1]
    except ValueError as e:
        raise UnableToCenter(str(e))

    # convert from aperture coords to image coords
    cyx = cyx + np.r_[yr[0], xr[0]]

    if niter > 1:
        if shrink:
            box = (halfbox * 2 / np.sqrt(2)).round().astype(int)
            box = box + (box % 2 - 1)  # keep it odd

        if not silent:
            print("y, x = {0[0]:.1f}, {0[1]:.1f}, next box size = {1}".format(
                cyx, str(box)))

        if max(box) < 2:
            if not silent:
                print(" - Box size too small -")
            return cyx

        return gcentroid(im, yx=cyx, box=box, niter=niter-1, dim=dim,
                         shrink=shrink, silent=silent)
    else:
        if not silent:
            print("y, x = {0[0]:.1f}, {0[1]:.1f}".format(cyx))

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
      sigclip median, sigclip stdev (3-sigma clipping), sum.  The mode
      is estimated assuming a unimodal data set that isn't too
      asymmetric: mode = 3 * median - 2 * mean, where median and mean
      are the sigma-clipped estimates.

    """

    from .. import util

    mc = util.meanclip(im, full_output=True, **kwargs)
    scmean, scstdev = mc[:2]
    scmedian = np.median(im.flatten()[mc[2]])

    return dict(min = np.nanmin(im),
                max = np.nanmax(im),
                mean = np.nanmean(im.ravel()),
                median = util.nanmedian(im),
                mode = 3.0 * scmedian - 2.0 * scmean,
                stdev = np.nanstd(im.ravel()),
                scmean = scmean,
                scmedian = scmedian,
                scstdev = scstdev,
                sum = np.nansum(im))

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

    from ..util import midstep

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

    x = core.xarray(sz, yx, rot=a, dtype=float) / subsample
    y = np.abs(core.yarray(sz, yx, rot=a, dtype=float) / subsample)

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

    n /= float(subsample**2)
    return midstep(xap), n, f

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
    rc : ndarray
      The center of each radial bin.
    sb : ndarray
      The surface brightness at each `r`.
    n : ndarray
      The number of pixels in each bin.
    rmean : ndarray
      The mean radius of the points in each radial bin.

    """

    from ..util import midstep

    if range is None:
        yx = np.array(yx)
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

    r = core.rarray(im.shape, yx=yx, subsample=10)
    n, f = anphot(im, yx, rap, subsample=subsample)
    rmean = anphot(r, yx, rap, subsample=subsample)[1]

    # flux -> surface brightness
    i = n != 0
    f[i] /= n[i]
    rmean[i] /= n[i]

    # first rap is really an inner edge, but anphot doesn't know that
    n = n[1:]
    f = f[1:]
    rmean = rmean[1:]
    rc = midstep(rap)

    return rc, f, n, rmean

def spextract(im, cen, rap, axis=0, trace=None, mean=False,
              bgap=None, bgorder=0, subsample=5):
    """Extract a spectrum, or similar, from an image.

    To account for potential pixel aliasing and missing values, each
    element is scaled to the same aperture size, defined as `2 * rap`.

    Parameters
    ----------
    im : ndarray or MaskedArray
      The 2D spectral image or similar.  If `im` is not a
      `MaskedArray`, then a bad value mask will be created from all
      non-finite elements.
    cen : int, float or array thereof
      The center(s) of the extraction.  The extraction will be along axis
      `axis`.
    rap : int or float
      Aperture radius.
    axis : int, optional
      The axis of the extraction.
    trace : array, optional
      An array of polynomial coefficients to use to adjust the
      aperture center based on axis index.  The first element is the
      coefficient of the highest order.  `cen` will be added to the
      last element.  If `cen` is an array, then `trace` may be an
      array of polynomial sets for each `cen`.  The first axis
      iterates over each set.
    mean : bool, optional
      Set to `True` to return the average value within the aperture,
      rather than the sum.
    bgap : array, optional
      Inner and outer radii for a background aperture.  If defined,
      the background will be removed, and the `nbg`, `mbg`, `bgvar`
      arrays will be returned.
    bgorder : int, optional
      Fit the background with a `bgorder` polynomial.  Currently
      limited to 0.
    subsample : int, optional
      The image is linearly subsampled to define the aperture edges.
      Set to `None` for no subsampling.

    Returns
    -------
    n : MaskedArray
      The aperture size in pixels for each element in `spec`.
    spec : MaskedArray
      The extracted spectrum, background removed when `bgap` is
      defined.
    nbg : MaskedArray, optional
      The background area in pixels for each element in `bg`.
    mbg : MaskedArray, optional
      The mean background spectrum.
    bgvar : MaskedArray, optional
      The background variance spectrum.

    """

    from numpy.ma import MaskedArray
    from . import yarray
    
    assert bgorder == 0, "bgorder must be 0"

    if axis == 1:
        return spectract(im.T, cen, rap, axis=0, trace=trace, mean=mean,
                         bgap=bgap, bgorder=bgorder, subsample=subsample)

    # parameter normalization
    if not np.iterable(cen):
        cen = [cen]
    N_aper = len(cen)

    if trace is not None:
        trace = np.array(trace)
        if trace.ndim != 2:
            order = len(trace)
            trace = np.tile(trace, N_aper).reshape((N_aper, order))
    else:
        trace = np.zeros(N_aper).reshape((N_aper, 1))

    # setup arrays for results
    n, spec = MaskedArray(np.zeros((2, N_aper, im.shape[1])))
    if bgap is not None:
        nbg, mbg, bgvar = MaskedArray(np.zeros((3, N_aper, im.shape[1])))

    y = yarray((im.shape[0] * subsample, im.shape[1])) / subsample
    x = np.arange(im.shape[1])

    for i in range(N_aper):
        p = trace[i].copy()
        p[-1] += cen[i]
        tr = np.polyval(p, x)

        aper = _spextract_mask(y, tr, rap, subsample)
        n[i] = np.sum(aper * ~im.mask, 0)
        spec[i] = np.sum(aper * im, 0) / n[i]
        if not mean:
            spec[i] *= 2 * rap

        if bgap is not None:
            c = np.mean(bgap)
            r = np.ptp(bgap)
            w = (_spextract_mask(y, tr + c, r, subsample)
                 + _spextract_mask(y, tr - c, r, subsample))
            nbg[i] = np.sum(w, 0)
            mbg[i] = np.sum(w * im, 0) / nbg[i]
            bgvar[i] = (np.sum(w * im**2, 0) - mbg[i]) / nbg[i]
            if mean:
                spec[i] -= mbg[i]
            else:
                spec[i] -= 2 * rap * mbg[i]
        
    if bgap is None:
        return n, spec
    else:
        return n, spec, nbg, mbg, bgvar

def _spextract_mask(y, trace, rap, subsample):
    """Create an photometry mask for `spextract`."""
    aper = (y >= trace - rap) * (y < trace + rap)
    aper = aper.reshape(y.shape[0] // subsample, subsample, y.shape[1])
    aper = aper.sum(1) / subsample
    return aper
        
def trace(im, err, guess, rap=5, axis=1, polyfit=False, order=2, plot=False,
          **imshow_kwargs):
    """Trace the peak pixels along an axis of a 2D image.

    Parameters
    ----------
    im : ndarray or MaskedArray
      The image to trace.  If a `MaskedArray`, masked values will be
      ignored.
    err : ndaarray
      The image uncertainties.  Set to None for unweighted fitting.
    guess : array
      The initial guesses for Gaussian fitting.  Fitting begins at the
      lowest index that is not masked.  Subsequent fits use the
      previous fit as a guess: (height, mu, sigma, [m, b]) = height,
      position (mu), width (sigma), and linear background m*x + b.
    rap : int, optional
      The size of the aperture (radius) to use around the peak guess.
    axis : int, optional
      The axis along which to measure the peaks.
    polyfit : bool, optional
      Set to `True` to fit the resulting array with a polynomial.  The
      polynomial coefficients will be added to the output and are
      suitable for the `np.polyval` function.  After an initial fit,
      the residuals will be inspected and outliers rejected before
      performing an additional fit.
    order : bool, optional
      The degree of the polynomial fit.
    plot : bool, optional
      Set to `True` to plot the result.
    **imshow_kwargs
      Any `matplotlib.imshow` keyword arguments for the plot.

    Returns
    -------
    peaks : ndarray
      Positions of the peaks.
    p : ndarray, optional
      Polynomial coefficients of the trace when the `fit` parameter is
      enabled.

    """

    import warnings
    from ..util import between, gaussfit, meanclip

    if axis == 0:
        if err is None:
            _err = None
        else:
            err = err.T
        return trace(im.T, _err, guess, axis=1, polyfit=polyfit, order=order)

    # The remainder is coded for dispersion along axis 1.
    if isinstance(im, np.ma.MaskedArray):
        mask = im.mask
        if im.mask.shape == ():
            mask = np.zeros(im.shape, bool)
            mask[:, :] = im.mask
    else:
        mask = np.zeros(im.shape, bool)

    peaks = np.ma.MaskedArray(np.zeros(im.shape[1]),
                              np.zeros(im.shape[1], bool))
    x = np.arange(im.shape[1], dtype=float)

    if err is None:
        err = np.ones_like(im)

    for i in range(im.shape[1]):
        c = int(round(guess[1]))
        aper = (max(0, c - rap), min(c + rap, im.shape[0]))
        j = slice(*aper)
        if np.all(mask[j, i]):
            peaks.mask[i] = True
            continue

        fit = gaussfit(x[j], im[j, i], err[j, i], guess)[0]
        if not between(fit[1], aper):
            peaks.mask[i] = True
            warnings.warn("Best-fit peak outside of aperture.",
                          LostTraceWarning)
            continue

        peaks[i] = fit[1]
        guess = fit

    if np.all(peaks.mask):
        raise UnableToTrace("No peaks found.")

    if polyfit:
        i = ~peaks.mask
        y = np.arange(im.shape[1])
        p = np.polyfit(y[i], peaks[i], order)
        good = meanclip((peaks - np.polyval(p, y))[i], full_output=True)[2]

        p = np.polyfit(y[i][good], peaks[i][good], order)
        return peaks, p
    else:
        return peaks


# update module docstring
from ..util import autodoc
autodoc(globals())
del autodoc
