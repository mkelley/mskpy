# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
image.process --- Process (astronomical) images.
================================================

.. autosummary::
   :toctree: generated/

   columnpull - Define a column pull detector artifact.
   crclean - Clean cosmic rays (LACOSMIC).
   fixpix - Replace masked pixels.
   fwhmfit - Measure the FWHM of a image.
   mkflat - Create a flat field.
   psfmatch - Generate a PSF matching kernel.
   stripes - Define jailbar/stripe artifacts.

.. todo:: Re-write anphot to generate pixel weights via rarray, rather
   than sub-sampling the images.  Update apphot and azavg, if needed.

.. todo:: Re-write linecut to generate pixel weights via xarray?

"""

from __future__ import print_function
import numpy as np
from . import core, analysis

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
      The rejection threshold in number of sigma.  [DN]
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
    Lim = 0.25 * rebin(Lim2, -2)

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

def fixpix(im, mask):
    """Replace masked values replaced with a linear interpolation.

    Probably only good for isolated badpixels.

    Parameters
    ----------
    im : array
      The image.
    mask : array
      `True` where `im` contains bad pixels.

    Returns
    -------
    cleaned : ndarray

    """

    from scipy.interpolate import interp2d
    from scipy.ndimage import binary_dilation, label

    # create domains around masked pixels
    dilated = binary_dilation(mask)
    domains, n = label(dilated)

    # loop through each domain, replace bad pixels with the average
    # from nearest neigboors
    x = xarray(im.shape)
    y = yarray(im.shape)
    cleaned = im.copy()
    for d in (np.arange(n) + 1):
        i = (domains == d)  # find the current domain

        # extract the sub-image
        x0, x1 = x[i].min(), x[i].max() + 1
        y0, y1 = y[i].min(), y[i].max() + 1
        subim = im[y0:y1, x0:x1]
        submask = mask[y0:y1, x0:x1]
        subgood = (submask == False)

        cleaned[i * mask] = subim[subgood].mean()

    return cleaned

def fwhmfit(im, yx, bg=True, **kwargs):
    """Compute the FWHM of an image.

    Parameters
    ----------
    im : array
      The image to fit.
    yx : array
      The center on which to fit.
    bg : bool, optional
      Set to `True` if there is a constant background to be
      considered.
    **kwargs
      Any `radprof` keyword, e.g., `range` or `bins`.

    Returns
    -------
    fwhm : float
      The FHWM of the radial profile.

    """

    from scipy.optimize import leastsq as lsq
    from ..util import gaussian

    def fitfunc(p, R, I):
        return I - gaussian(R, 0, p[0]) * p[1] + p[2]

    R, I, n = analysis.radprof(im, yx, **keywords)
    r = R[I < (I.max() / 2.0)][0]  # guess for Gaussian sigma
    fit = lsq(fitfunc, (r, I.max(), I.min()), args=(R, I))[0]
    fit = abs(fit)

    return fit[0] * 2.35


def mkflat(images, bias, func=np.mean, lsig=3., hsig=3., **kwargs):
    """Flat field correction and bad pixel mask from a set of images.

    Parameters
    ----------
    images : array
      Images to combine into a flat field.  The first axis defines the
      set of images.
    bias : array
      Bias frame(s) to subtract from each image.
    func : function, optional
      The combining function.
    lsig, hsig : float
      The lower- and upper-sigma limits used to define bad pixels.
    **kwargs, optional
      Any `core.uclip` keyword.

    Returns
    -------
    flat : ndarray
      The flat field correction.
    bpm : ndarray
      `True` for bad pixels.

    """

    from ..util import uclip

    def clip(a):
        return uclip(a, func, **kwargs)

    flat = np.apply_over_axes(clip, 0, images)

    lhsig = dict(lsig=lsig, hsig=hsig)

    stat = imstat(flat, **lhsig)
    flat /= stat['scmean']
    bpm = flat.copy()

    stat = imstat(bpm, **lhsig)
    bpm[bpm < (1 - stat['scstdev'] * lsig)] = 0
    bpm[bpm > (1 + stat['scstdev'] * hsig)] = 0
    bpm = bpm == 0

    stat = imstat(flat[~bpm], **lhsig)
    flat /= stat['scmean']

    return flat, bpm

def psfmatch(psf, psfr, ps=1, psr=1):
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
    smooth : float
      If not `None`, smooth the resulting kernel with a `smooth`-width
      Gaussian kernel.
    mask : float
      If not `None`, set all pixels at a distance greater than `mask`
      from the center of the kernel to 0 (i.e., mask the
      high-frequency components, which are usually dominated by
      noise).

    Returns
    -------
    k : ndarray
      The convolution kernel to change the PSF of an image to match
      the reference PSF.

    """

    from scipy.ndimage import zoom, gaussian_filter

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

    print("mean/sig/% masked = {0}/{1}/{2}".format(
        m, sig, 1 - good.size / np.prod(im.shape).astype(float)))

    mask = np.ones_like(im).astype(bool)
    mask.ravel()[good] = False
    if type(im) is np.ma.MaskedArray:
        mask += im.mask

    _im = np.ma.MaskedArray(im, mask=mask)
    s = stat(_im, axis=axis)
    
    return s
