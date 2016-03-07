# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
image.process --- Process (astronomical) images.
================================================

.. autosummary::
   :toctree: generated/

   align_by_centroid
   align_by_offset
   align_by_wcs
   columnpull
   combine
   crclean
   fixpix
   mkflat
   psfmatch
   stripes
   subim

"""

import numpy as np
from . import core, analysis

__all__ = [
    'align_by_centroid',
    'align_by_offset',
    'align_by_wcs',
    'columnpull',
    'combine',
    'crclean',
    'fixpix',
    'mkflat',
    'psfmatch',
    'stripes',
    'subim',
]

def align_by_centroid(data, yx, cfunc=None, ckwargs=dict(box=5),
                      **kwargs):
    """Align a set of images by centroid of a single source.

    Parameters
    ----------
    data : list or array
      The list of FITS files, or stack of images to align.  If the
      first element is a string, then a file list is assumed.
    yx : array
      The approximate (y, x) coordinate of the source.
    cfunc : function, optional
      The centroiding function or `None` to use `gcentroid`.
    ckwargs : dict
      Keyword arguments for `cfunc`.
    **kwargs
      Keyword arguments for `imshift`.

    Results
    -------
    stack : ndarray
      The aligned images.
    dyx : ndarray
      The offsets.  Suitable for input into `align_by_offset`.

    """

    import astropy.units as u
    from astropy.io import fits
    from .analysis import gcentroid

    if cfunc is None:
        cfunc = gcentroid

    if isinstance(data[0], str):
        im = fits.getdata(data[0])
        stack = np.zeros((len(data), ) + im.shape)
        stack[0] = im
        del im
        for i in range(1, len(data)):
            stack[i] = fits.getdata(data[i])
    else:
        stack = data.copy()

    y0, x0 = cfunc(stack[0], yx, **ckwargs)

    dyx = np.zeros((len(stack), 2))
    for i in range(1, len(stack)):
        y, x = cfunc(stack[i], yx, **ckwargs)
        dyx[i] = y0 - y, x0 - x
        stack[i] = core.imshift(stack[i], dyx[i], **kwargs)
        if int(dyx[i, 0]) != 0:
            if int(dyx[i, 0]) < 0:
                stack[i, int(dyx[i, 0]):] = np.nan
            else:
                stack[i, :int(dyx[i, 0])] = np.nan
        if int(dyx[i, 1]) != 0:
            if int(dyx[i, 1]) < 0:
                stack[i, :, int(dyx[i, 1]):] = np.nan
            else:
                stack[i, :, :int(dyx[i, 1])] = np.nan

    return stack, dyx

def align_by_offset(data, dyx, **kwargs):
    """Align a set of images by a given list of offsets.

    Parameters
    ----------
    data : list or array
      The list of FITS files, or stack of images to align.  If the
      first element is a string, then a file list is assumed.
    dyx : array
      The offsets.
    **kwargs
      Keyword arguments for `imshift`.

    Results
    -------
    stack : ndarray
      The aligned images.

    """

    import astropy.units as u
    from astropy.io import fits

    if isinstance(data[0], str):
        im = fits.getdata(data[0])
        stack = np.zeros((len(data), ) + im.shape)
        stack[0] = im
        del im
        for i in range(1, len(data)):
            stack[i] = fits.getdata(data[i])
    else:
        stack = data.copy()

    for i in range(1, len(stack)):
        stack[i] = core.imshift(stack[i], dyx[i], **kwargs)
        if int(dyx[i, 0]) != 0:
            if int(dyx[i, 0]) < 0:
                stack[i, int(dyx[i, 0]):] = np.nan
            else:
                stack[i, :int(dyx[i, 0])] = np.nan
        if int(dyx[i, 1]) != 0:
            if int(dyx[i, 1]) < 0:
                stack[i, :, int(dyx[i, 1]):] = np.nan
            else:
                stack[i, :, :int(dyx[i, 1])] = np.nan

    return stack

def align_by_wcs(files, target=None, observer=None, time_key='DATE-OBS',
                 **kwargs):
    """Align a set of images using their world coordinate systems.

    Parameters
    ----------
    files : list
      The list of FITS files to align.
    target : SolarSysObject
      Align in the reference frame of this object.
    observer : SolarSysObject
      Observe `target` with this observer.
    time_key : string
      The header keyword for the observation time.
    **kwargs
      Keyword arguments for `imshift`.

    Results
    -------
    stack : ndarray
      The aligned images.
    dyx : ndarray
      The offsets.  Suitable for input into `align_by_offset`.

    """

    import astropy.units as u
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.coordinates import Angle

    im, h0 = fits.getdata(files[0], header=True)
    wcs0 = WCS(h0)
    stack = np.zeros((len(files), ) + im.shape)
    stack[0] = im

    y0, x0 = np.array(im.shape) / 2.0
    if target is not None:
        assert observer is not None, "observer required"
        g0 = observer.observe(target, h0[time_key])
        xt, yt = wcs0.wcs_world2pix(np.c_[g0.ra, g0.dec], 0)[0]

    ra0, dec0 = Angle(wcs0.wcs_pix2world(np.c_[x0, y0], 0)[0] * u.deg)
    dra = 0 * u.deg
    ddec = 0 * u.deg

    dyx = np.zeros((len(files), 2))
    for i in range(1, len(files)):
        im, h = fits.getdata(files[i], header=True)
        wcs = WCS(h)
        if target is not None:
            g = observer.observe(target, h[time_key])
            dra = g.ra - g0.ra
            ddec = g.dec - g0.dec

        x, y = wcs.wcs_world2pix(np.c_[ra0 + dra, dec0 + ddec], 0)[0]
        dyx[i] = y0 - y, x0 - x
        stack[i] = core.imshift(im, dyx[i], **kwargs)
        if int(dyx[i, 0]) != 0:
            if int(dyx[i, 0]) < 0:
                stack[i, int(dyx[i, 0]):] = np.nan
            else:
                stack[i, :int(dyx[i, 0])] = np.nan
        if int(dyx[i, 1]) != 0:
            if int(dyx[i, 1]) < 0:
                stack[i, :, int(dyx[i, 1]):] = np.nan
            else:
                stack[i, :, :int(dyx[i, 1])] = np.nan

    return stack, dyx

def columnpull(column, index, bg, stdev):
    """Define a column pull detector artifact.

    Parameters
    ----------
    column : array
      The column from a detector.
    index : int
      The index at which the column pull may have started, e.g., the
      location of a bright star.
    bg : float
      The background level of the image.
    stdev : float
      The background standard deviation.

    Returns
    -------
    pull : ndarray
      The shape of the column pull.

    """

    if (index < 0) or (index >= column.shape[0]):
        return

    m1 = np.median(column[:index]) - bg
    m2 = np.median(column[index:]) - bg

    pull = np.zeros_like(column)
    if (np.abs(m1 - m2) / stdev) > 1.0:
        pull[:index] = m1
        pull[index:] = m2

    return pull


def combine(images, axis=0, func=np.mean, niter=0, lsig=3, hsig=3):
    """Combine a set of images, clipping as necessary.

    Parameters
    ----------
    images : array
      The set of images.  May be a `MaskedArray`.
    axis : int
      If images is an n-dimensional array, this is the axis which
      iterates over each image.
    func : function
      The function to use to combine the clipped data.  Must be able
      to accept a `MaskedArray`, and must be able to accept the `axis`
      keyword.
    niter : int
      Number of clipping iterations.
    lsig, hsig : float
      Lower- and upper-sigma clipping limits.

    Returns
    -------
    comb : ndarray
      The combined data set.

    """

    print('[Combine] Remaining iterations: ', niter)
    if not isinstance(images, np.ma.MaskedArray):
        images = np.ma.MaskedArray(images)

    print('  Median')
    m = np.median(images, axis=axis)
    print('  Standard deviation')
    s = np.std(images, axis=axis)
    d = (images - m) / s
    images.mask += (d < -lsig) + (d > hsig)
    if niter <= 0:
        print('  Combining')
        return func(images, axis=axis)
    else:
        return combine(images, axis=axis, func=func, niter=niter-1,
                       lsig=lsig, hsig=hsig)

def crclean(im, thresh, niter=1, unc=None, gain=1.0, rn=0.0, fwhm=2.0):
    """Clean cosmic rays from an image.

    Based on LACOSMIC, as described by van Dokkum 2001, PASP, 113,
    789.  Each iteration runs 3 to 4 median filters (one 3x3, one to
    two 5x5, and one 7x7) and a 3x3 discrete convolution (the
    Laplacian).

    Parameters
    ----------
    im : array
      The image to filter.
    thresh : float
      The rejection threshold in number of sigma.
    niter : int
      Number of iterations to apply the filter.
    unc : array, optional
      The image uncertainties, pixel-by-pixel, or `None` to use gain &
      read noise (the latter assume the image has not been background
      subtracted).  [DN]
    gain : float, optional
      The image gain factor.  [e-/DN]
    rn : float, optional
      The instrument's read noise.  [e-]
    fwhm : float, optional
      The FWHM of point sources used in determining the rejection
      limit based on the fine-structure image.  If point sources are
      narrower than FWHM, they may be rejected.  [pixels]

    Returns
    -------
    clean : ndarray
      An image where identified cosmic rays have been replaced with
      the median of the surrounding pixels.

    """

    from scipy.ndimage import convolve, median_filter

    if niter > 1:
        print('[crclean] Iteration {}'.format(niter))
        im = crclean(im, thresh, niter=niter-1, unc=unc, gain=gain, rn=rn)

    # subsample the image by a factor of 2 to avoid contamination from
    # neighboring high pixels
    im2 = core.rebin(im, 2)

    # Take the Laplacian of the image
    Laplacian = 0.25 * np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    Lim2 = convolve(im2, Laplacian)

    # Remove negative cross patterns
    Lim2[Lim2 < 0] = 0

    # Back to the original resolution
    Lim = 0.25 * core.rebin(Lim2, -2)

    if unc is None:
        # Determine the noise in the original image
        unc = np.sqrt(median_filter(im, 5) * gain + rn**2) / gain

    S = Lim / 2.0 / unc

    # Remove smooth structures
    S = S - median_filter(S, 5)

    # Generate a fine-structure image.  Point sources have greater
    # symmetry than cosmic rays, so they should be brighter in this
    # image.
    mim3 = median_filter(im, 3)
    F = mim3 - median_filter(mim3, 7)

    # flim based on FWHM and Fig. 4 of van Dokkum 2001
    flim = 10.16 * fwhm**-2.76 + 1.0 - 0.5

    mask = (S > thresh) * ((Lim / F) > flim) * (np.isfinite(Lim / F))

    clean = im.copy()
    clean[mask] = mim3[mask]

    return clean

def fixpix(im, mask, max_area=10):
    """Replace masked values replaced with a linear interpolation.

    Probably only good for isolated bad pixels.

    Parameters
    ----------
    im : array
      The image.
    mask : array
      `True` where `im` contains bad pixels.
    max_area : int
      Only fix areas smaller or equal to this value.

    Returns
    -------
    cleaned : ndarray

    """

    from scipy.interpolate import interp2d
    from scipy.ndimage import binary_dilation, label, find_objects

    # create domains around masked pixels
    dilated = binary_dilation(mask)
    domains, n = label(dilated)

    # loop through each domain, replace bad pixels with the average
    # from nearest neigboors
    cleaned = im.copy()
    for i in find_objects(domains):
        submask = mask[i]
        if submask.sum() > max_area:
            continue
        subim = im[i].copy()
        subgood = (submask == False) * dilated[i]
        subim[submask] = subim[subgood].mean()
        cleaned[i] = subim

    return cleaned

def mkflat(flat, **kwargs):
    """Flat field correction and bad pixel mask from an image.

    Parameters
    ----------
    im : array
      The image of sky, screen, dome, etc.  May be a `MaskedArray`.
    kwargs : dict
      Any `util.meanclip` keyword, except `full_output`.

    Returns
    -------
    flat : numpy MaskedArray
      The flat-field correction and bad pixel mask.

    """

    from ..util import meanclip

    if not isinstance(flat, np.ma.MaskedArray):
        flat = np.ma.MaskedArray(flat)

    mc = meanclip(flat, full_output=True, **kwargs)
    mask = np.ones(flat.shape, dtype=bool).ravel()
    mask[mc[2]] = False
    mask = mask.reshape(flat.shape)

    flat /= mc[0]
    flat.mask += mask

    return flat

def psfmatch(psf, psfr, ps=1, psr=1, smooth=None, mask=None):
    """Generate a convolution kernel to match the PSFs of two images.

    Parameters
    ----------
    psf, psfr : array
      An image of the input point spread function (psf) and reference
      point spread function (psfr).  Each should be centered on the
      PSF, and square.
    ps, psr : float
      The pixel scale of the input and reference PSFs.  If they
      differ, `scipy.ndimage.zoom` will be used to match the `psfr`
      scale with `psf`.
    smooth : float, optional
      If not `None`, smooth the resulting kernel with a `smooth`-width
      Gaussian kernel.
    mask : float, optional
      If not `None`, set all pixels at a distance greater than `mask`
      from the center of the kernel to 0 (i.e., mask the
      high-frequency components, which are usually dominated by
      noise).

    Returns
    -------
    K : ndarray
      The convolution kernel to change the PSF of an image to match
      the reference PSF.

    """

    from scipy.ndimage import zoom, gaussian_filter
    from numpy import fft

    assert psf.shape[0] == psf.shape[1], "psf should have a square shape"
    assert psfr.shape[0] == psfr.shape[1], "psfr should have a square shape"

    _psf = psf

    # rebin to match pixel scales?
    if ps != psr:
        _psfr = zoom(psfr, psr / ps) # change the reference PSF
    else:
        _psfr = psfr

    # trim psfr?
    d = _psfr.shape[0] - _psf.shape[0]
    if d != 0:
        sl = slice(np.floor(d / 2.), -np.ceil(d / 2.))
        _psfr = rebin(_psfr, 2)
        _psfr = rebin(_psfr[d:-d, d:-d], -2)

    # normalize to 1.
    _psfr /= _psfr.sum()
    _psf /= _psf.sum()
    R = fft.fft2(_psfr)
    I = fft.fft2(_psf)
    K = fft.fft2(R / I).real
    K = np.roll(np.roll(K, K.shape[0] / 2, 0), K.shape[1] / 2, 1)

    if smooth is not None:
        K = gaussian_filter(K, smooth)

    if mask is not None:
        r = core.rarray(K.shape)
        K[r > mask] = 0

    return K / K.sum()

def stripes(im, axis=0, stat=np.median, **keywords):
    """Find and compute column/row stripe artifacts in an image.

    Parameters
    ----------
    im : array
      The image with stripe artifacts.  The image is first sigma
      clipped with `util.meanclip`.
    axis : int
      The axis parallel to the stripes.
    stat : function
      The statistic to derive the background stripes (usually
      `np.mean` or `np.median`).  The function must take the `axis`
      keyword, and must be able to handle a `MaskedArray`.
    **keywords
      Any `util.meanclip` keyword except `full_output`.

    Returns
    -------
    s : ndarray
      A 1D array of the stripes.

    """

    from ..util import meanclip

    m, sig, good = meanclip(im, full_output=True, **keywords)[:3]

    #print("mean/sig/% masked = {0}/{1}/{2}".format(
    #    m, sig, 1 - len(good) / np.prod(im.shape).astype(float)))

    mask = np.ones_like(im).astype(bool)
    mask.ravel()[good] = False
    if type(im) is np.ma.MaskedArray:
        mask += im.mask

    _im = np.ma.MaskedArray(im, mask=mask)
    s = stat(_im, axis=axis)
    
    return s

def subim(im, yx, half_box):
    """Extract a sub-image.

    If any part of the sub-image is beyond the image edge, it will be
    truncated.

    Parameters
    ----------
    im : array
      The full image.
    yx : array
      The center of the sub-image.  Floating point values will be
      rounded.
    half_box : int
      The desired half-length of a side of the sub-image.  The image
      shape will be `half_box * 2 + 1` for each dimension, unless
      truncated by an image edge.

    Returns
    -------
    subim : ndarray

    """

    y0 = int(np.around(yx[0]))
    x0 = int(np.around(yx[0]))
    s = np.s_[max(y0 - half_box, 0) : min(y0 + half_box + 1, im.shape[0]),
              max(x0 - half_box, 0) : min(x0 + half_box + 1, im.shape[1])]
    return im[s]

# update module docstring
from ..util import autodoc
autodoc(globals())
del autodoc
