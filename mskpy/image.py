# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
image --- For working with images, maybe spectra.
=================================================


.. autosummary::
   :toctree: generated/

   Classes
   -------
   Image

#   stack2grid

   General
   -------
   rarray
   rebin
   tarray
   xarray
   yarray

   Analysis
   --------
   anphot
   apphot
   azavg
   bgfit
   centroid
   imstat
   phot
   polyfit2d
   radprof
   trace

   Processing
   ----------
   columnpull
   combine
   crclean
   fixpix
   fwhmfit
   imshift
   jailbar
   imcombine
   mkflat
   psfmatch
   stripes
   unwrap

   FITS or WCS
   -----------
   basicwcs
   getrot

"""

#__all__ = [
#    ''
#]

import numpy as np

def rebin(a, factor, flux=False, trim=False):
    """Rebin a 1, 2, or 3 dimensional array by integer amounts.

    Parameters
    ----------
    a : ndarray
      Image to rebin.
    factor : int
      Rebin factor.  Set factor < 0 to shrink the image.  When
      shrinking, all axes of a must be an integer multiple of factor
      (see trim).
    flux : bool
      Set to True to preserve flux, otherwise, surface brightness is
      preserved.
    trim : bool
      Set to True to automatically trim the shape of the input array
      to be an integer multiple of factor.

    Returns
    -------
    b : ndarray
      The rebinned array.

    Notes
    -----
    By default, the function preserves surface brightness, not flux.

    """

    def mini(a, factor):
        b = a[::-factor]
        for i in range(-factor - 1):
            b += a[(i + 1)::-factor]
        if not flux:
            b /= float(-factor)
        return b

    def magni(a, factor):
        s = np.array(a.shape)
        s[0] *= factor
        b = np.zeros(s)
        for i in range(factor):
            b[i::factor] = a
        if flux:
            b /= float(factor)
        return b

    _a = a.copy()
    if factor < 0:
        for i in range(_a.ndim):
            if trim:
                r = _a.shape[i] % abs(factor)
                if r != 0:
                    _a = np.rollaxis(np.rollaxis(_a, i)[:-r], 0, i + 1)

            assert (_a.shape[i] % factor) == 0, (
                "Axis {0} must be an integer multiple of "
                "the minification factor.".format(i))
        f = mini
    else:
        f = magni

    b = f(_a, factor)
    for i in range(len(_a.shape) - 1):
        c = f(np.rollaxis(b, i + 1), factor)
        b = np.rollaxis(c, 0, i + 2)

    return b

def rarray(shape, center=None, subsample=False, dtype=float):
    """Array of distances from a point.

    Parameters
    ----------
    shape : array
      The shape of the resulting array `(y, x)`.
    center : array, optional
      The center of the array `(yc, xc)`.  If set to None, then the
      center is `shape / 2.0 - 1.0` (floating point arithmetic).
      Integer values refer to the center of the pixel.
    subsample : bool, optional
      Set to True to sub-pixel sample the array.
    dtype : np.dtype or similar, optional
      Set to the data type of the resulting array.

    Returns
    -------
    r : ndarray
      The array of radial values.

    """

    if center is None:
        c = (np.array(shape, dtype) - 1.0) / 2.0
    else:
        c = np.array(center, dtype)

    if subsample == True:
        r = rarray(shape, center=center, subsample=False, dtype=dtype)

        # for pixels at >5. pixels from the center, subsamping will
        # probably be a <1% effect
        # c = c - np.rint(c) + np.array((5, 5)) * 10 - 5.5
        cprime = (c[-2:] - np.rint(c[-2:])) * 10.0
        cprime += np.array([54.5, 54.5])
        y = yarray(np.array((11, 11), dtype=dtype) * 10) - cprime[0]
        x = xarray(np.array((11, 11), dtype=dtype) * 10) - cprime[1]

        # *0.1 for the subpixel sampling
        rprime = rebin(np.sqrt(x**2 + y**2), -10, flux=False) * 0.1
        rprime = rprime.astype(dtype)

        yi, xi = np.indices((11, 11)) - 5
        yi += np.rint(c[-2]).astype(int)
        xi += np.rint(c[-1]).astype(int)
        i = (yi >= 0) * (xi >= 0) * (yi < shape[0]) * (xi < shape[1])
        r[yi[i], xi[i]] = rprime[i]
        return r

    y = yarray(shape, dtype=dtype) - c[-2]
    x = xarray(shape, dtype=dtype) - c[-1]
    return (np.sqrt(x**2 + y**2)).astype(dtype)

def xarray(shape, cen=[0, 0], rot=0, dtype=int):
    """Array of x values.

    Parameters
    ----------
    shape : tuple of int
      The shape of the resulting array, e.g., (y, x).
    cen : tuple of float
      Offset the array to align with this y, x center.
    rot : float, optional
      Rotate the array by rot, measured from the x-axis. [radians]
    dtype : numpy.dtype, optional
      The data type of the result.

    Returns
    -------
    x : ndarray
      An array of x values.

    """

    y, x = np.indices(shape, dtype)[-2:]
    y -= cen[0]
    x -= cen[1]
    if rot == 0:
        return x
    R = math.rotmat(rot)
    return x * R[0, 0] + y * R[0, 1]

def yarray(shape, cen=[0, 0], rot=0, dtype=int):
    """Array of y values.

    Parameters
    ----------
    shape : tuple of int
      The shape of the resulting array, e.g., (y, x).
    cen : tuple of float
      Offset the array to align with this y, x center.
    rot : float, optional
      Rotate the array by rot, measured from the x-axis. [radians]
    dtype : numpy.dtype, optional
      The data type of the result.

    Returns
    -------
    y : ndarray
      An array of y values.

    """

    y, x = np.indices(shape, dtype)[-2:]
    y -= cen[0]
    x -= cen[1]
    if rot == 0:
        return y
    R = math.rotmat(rot)
    return x * R[1, 0] + y * R[1, 1]

def centroid(im, guess=None, box=None, niter=1, shrink=True, silent=True):
    """Centroid (center of mass) of an image.

    Parameters
    ----------
    im : ndarray
      A 2D image on which to centroid.
    guess : float array, optional
      (x, y) guess for the centroid.  The default guess is the image
      peak.
    box : int array, optional
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
    cx, cy : floats
      The computed center of mass.  The lower-left corner of a pixel
      is -0.5, -0.5.

    """

    if guess is None:
        i = np.isfinite(im)
        y, x = np.indices(im.shape)
        peak = im[i].argmax()
        guess = (x[i][peak], y[i][peak])

    if box is None:
        box = np.array((im.shape[1], im.shape[0]))
        
    if np.size(box) == 1:
        box = np.array((box, box))

    x0 = max(guess[0] - box[0] / 2, 0)
    x1 = min(guess[0] + box[0] / 2 + 1, im.shape[1])
    y0 = max(guess[1] - box[1] / 2, 0)
    y1 = min(guess[1] + box[1] / 2 + 1, im.shape[0])
    subim = im[y0:y1,x0:x1].copy()
    subim -= subim.min()

    y, x = np.indices(subim.shape)
    cx = (subim * x).sum() / subim.sum() + x0
    cy = (subim * y).sum() / subim.sum() + y0

    if niter > 1:
        if shrink:
            box = (box / 1.414).astype(int)
            # keep it odd
            box = box + (box % 2 - 1)

        if not silent:
            print "x, y = {0:.1f}, {1:.1f}, next box size = {2}".format(
                cx, cy, str(box))

        if np.any(box < 2):
            if not silent:
                print " - Box size too small -"
            return float(cx), float(cy)

        return centroid(im, guess=[float(cx), float(cy)], box=box,
                        niter=niter - 1, shrink=shrink, silent=silent)
    else:
        if not silent:
            print "x, y = {0:.1f}, {1:.1f}".format(cx, cy)
        return float(cx), float(cy)

def imshift(im, xo, yo, sample=4):
    """Shift an image, allowing for sub-pixel offsets (drizzle method).

    Parameters
    ----------
    im : ndarray
      The image to shift.
    xo, yo : floats
      x, y offsets. [unsampled pixels]
    sample : int, optional
      The subsampling factor.

    Returns
    -------
    sim : ndarray
      The shifted image (at the original pixel scale).

    """

    if sample <= 1:
        raise ValueError("sample must be > 1.")

    # convert xo, yo into whole sampled pixels
    sxo = int(round(xo * sample))
    syo = int(round(yo * sample))

    sim = rebin(im, sample, flux=True)
    sim = np.roll(sim, syo, 0)
    sim = np.roll(sim, sxo, 1)
    return rebin(sim, -sample, flux=True)
