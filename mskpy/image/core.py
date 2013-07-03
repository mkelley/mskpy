# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
image.core --- General image operations.
========================================

.. autosummary::
   :toctree: generated/

   imshift
   rarray
   rebin
   stack2grid
   tarray
   unwrap
   xarray
   yarray

"""

import numpy as np

def imshift(im, xo, yo, subsample=4):
    """Shift an image, allowing for sub-pixel offsets (drizzle method).

    Parameters
    ----------
    im : ndarray
      The image to shift.
    xo, yo : floats
      x, y offsets. [unsampled pixels]
    subsample : int, optional
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

def rarray(shape, yx=None, subsample=0, dtype=float):
    """Array of distances from a point.

    Parameters
    ----------
    shape : array
      The shape of the resulting array `(y, x)`.
    yx : array, optional
      The center of the array `(y, x)`.  If set to None, then the
      center is `shape / 2.0 - 1.0` (floating point arithmetic).
      Integer values refer to the center of the pixel.
    subsample : int, optional
      Set to `>1` to sub-pixel sample the core of the array.
    dtype : np.dtype or similar, optional
      Set to the data type of the resulting array.

    Returns
    -------
    r : ndarray
      The array of radial values.

    """

    if yx is None:
        yx = (np.array(shape, dtype) - 1.0) / 2.0
    else:
        yx = np.array(yx, dtype)

    y = yarray(shape, dtype=dtype) - yx[-2]
    x = xarray(shape, dtype=dtype) - yx[-1]
    r = (np.sqrt(x**2 + y**2)).astype(dtype)

    if subsample == True:
        # for pixels at >5. pixels from the center, subsamping will
        # probably be a <1% effect
        # c = c - np.rint(c) + np.array((5, 5)) * 10 - 5.5
        cprime = (yx[-2:] - np.rint(yx[-2:])) * 10.0
        cprime += np.array([54.5, 54.5])
        y = yarray(np.array((11, 11), dtype=dtype) * 10) - cprime[0]
        x = xarray(np.array((11, 11), dtype=dtype) * 10) - cprime[1]

        # x0.1 for the subpixel sampling
        rprime = rebin(np.sqrt(x**2 + y**2), -10, flux=False) * 0.1
        rprime = rprime.astype(dtype)

        yi, xi = np.indices((11, 11)) - 5
        yi += np.rint(yx[-2]).astype(int)
        xi += np.rint(yx[-1]).astype(int)
        i = (yi >= 0) * (xi >= 0) * (yi < shape[0]) * (xi < shape[1])
        r[yi[i], xi[i]] = rprime[i]

    return r

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

def stack2grid(stack):
    """Reshape a stack of images into an NxN (2D) grid.

    Parameters
    ----------
    stack : ndarray
      The data cube to be reshaped.  The first axis iterates over the
      images.  If the cube is NxMxL, the result will be a grid of
      sqrt(N) x sqrt(N) images.

    Returns
    -------
    grid : ndarray
      A sqrt(N)*M by sqrt(N)*L grid of images.

    Notes
    -----
    Based on
    http://stackoverflow.com/questions/13990465/3d-numpy-array-to-2d.

    """
    n = int(np.sqrt(stack.shape[0]))
    if n**2 != stack.shape[0]:
        raise RuntimeError("Cannot make a square grid from a stack of "
                           "depth {}".format(stack.shape[0]))
    grid = stack.reshape(n, n, stack.shape[1], stack.shape[2])
    grid = grid.swapaxes(1, 2)
    grid = grid.reshape(n * stack.shape[1], -1)
    return grid

def tarray(shape, yx=None, subsample=0, dtype=float):
    """Array of azimuthal angles values.

    Parameters
    ----------
    shape : array
      The shape of the resulting array, (y, x).
    yx : array, optional
      The center of the array `(y, x)`.  If set to `None`, then the
      center is `shape / 2.0 - 1.0` (floating point arithmetic).
    subsample : int, optional
      Set to `>1` to sub-pixel sample the core of the array.
    dtype : object, optional
      Set to the data type of the resulting array (all math will use
      this dtype).

    Returns
    -------
    th : ndarray
      An array of angles, starting with 0 along the x-axis.  [radians]

    """

    if yx is None:
        yx = (np.array(shape, dtype) - 1.0) / 2.0
    else:
        yx = np.array(yx, dtype)

    y = yarray(shape, dtype=dtype) - yx[-2]
    x = xarray(shape, dtype=dtype) - yx[-1]
    th = np.arctan2(y, x)

    if subsample > 0:
        th = refine_center(tarray, th, yx, 5, 10, dtype=dtype)

    return th

def refine_center(func, im, yx, N, subsample, scale=0, **kwargs):
    """Subsample an array generating function near the center.

    Parameters
    ----------
    func : function
      The array generating function, e.g., `rarray`.  The first
      argument of the function is the shape of the result.  Function
      keywords must include `yx` and `subsample`.  See `rarray` for an
      example.
    im : array
      The initially generated array, the center pixels of which will
      be replaced.
    yx : array
      The function's origin.
    N : int
      The center `(N, N)` pixels will be refined.  Best is an odd
      value.
    subsample : int
      The sub-pixel sampling factor.
    scale : float
      Scale the refined center by `subsample**scale`.  The refined
      area will be generated at high resolution, and rebinned
      (averaging) to a resolution of 1 pixel.  When a function has
      dimensions of length, the rebinned array needs to be scaled by
      `subsample**-1`.
    **kwargs
      Keyword arguments for `func`.

    Returns
    -------
    refined : ndarray
      The refined array.

    """

    refined = im.copy()

    # where is the center of the NxN region?
    yx_N = (np.ones(2) * N - 1.0) / 2.0
    yx_N += (yx - np.array([round(x) for x in yx]))

    # subsample the NxN region, where is the center of that?
    shape_c = np.ones(2) * N * subsample
    xy_c = (np.ones(2) * N * subsample - 1) / 2.0
    xy_c += (xy_N - np.array([round(x) for x in xy_N])) * subsample

    # generate the subsampled center
    refined_c = func(shape_c, yx=yx_c, subsample=0, **kwargs)
    refined_c = image.rebin(th_c, -subsample, flux=False)
    refined_c *= subsample**scale

    # The region to be refined: xi, yi
    yi, xi = np.indices((N, N)) - N / 2
    xi += int(round(xy[0]))
    yi += int(round(xy[1]))

    # insert into the result
    i = (yi >= 0) * (xi >= 0) * (yi < im.shape[0]) * (xi < im.shape[1])
    if any(i):
        refined[yi[i], xi[i]] = refined_c[i]

    return refined

def unwrap(im, cyx, dtdr=0, scale=None, bins=100, range=None,
           subsample=False, dtype=float):
    """Transformation image from rectangular to cylindrical projection.

    Parameters
    ----------
    im : array
      The image on which to operate.
    cyx : array
      The center of the array (y, x).
    dtdr : float, optional
      The change of theta with radius.
    scale : int
      Scale the image via rebin() before computing the radial profile.
      The output r will use the original pixel size.
    bins : int or tuple of ints, optional
      The number of bins for the final image or a 2 element array:
      (radial-bins, theta-bins).
    range : array-like, shape(2, 2), optional
      The ranges in radius and theta to consdier:
        [[rmin, rmax], [thmin, thmax]].  [pixels and radians]
    subsample : bool, optional
      Set to True to sub-pixel sample the array.
    dtype : object, optional
      Set to the data type of the resulting array (all math will use
      this dtype).

    Returns
    -------

    rt : ndarray

      The image transformed to place radius along the first dimension,
      and theta along the second dimension.

    rbins : ndarray

      The radial bins (from np.histogram2d). [pixels]

    thbins : ndarray

      The azimuthal bins (from np.histogram2d). [radians]

    Notes
    -----

    v1.0.0 Written by Michael S. Kelley, UMD, July 2011
    """
    if scale is not None:
        image = rebin(im, scale)
        if scale < 0:
            scale = 1.0 / float(scale)
        cen = np.array(center) * scale
    else:
        image = im
        scale = 1.0
        cen = np.array(center)

    r = rarray(image.shape, center=cen, subsample=subsample,
               dtype=dtype) / scale
    th = tarray(image.shape, center=cen, subsample=subsample,
                dtype=dtype) + np.pi
    th = (th + dtdr * r) % (2 * np.pi)

    rt = np.histogram2d(r.flatten(), th.flatten(), bins=bins,
                           range=range, weights=image.flatten())
    rbin = np.histogram2d(r.flatten(), th.flatten(), bins=bins,
                             range=range, weights=r.flatten())
    thbin = np.histogram2d(r.flatten(), th.flatten(), bins=bins,
                              range=range, weights=th.flatten())
    n = np.histogram2d(r.flatten(), th.flatten(), bins=bins,
                           range=range)[0].astype(float)
    return rt[0] / n, rbin[0] / n, thbin[0] / n, n
