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
   zarray

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
