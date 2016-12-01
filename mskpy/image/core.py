# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
image.core --- General image operations, generators.
====================================================

.. autosummary::
   :toctree: generated/

   imshift
   rarray
   rebin
   stack2grid
   tarray
   yx2rt
   xarray
   yarray

"""

__all__ = [
    'imshift',
    'rarray',
    'rebin',
    'stack2grid',
    'tarray',
    'yx2rt',
    'xarray',
    'yarray'
]

import numpy as np

def imshift(im, yx, subsample=4):
    """Shift an image, allowing for sub-pixel offsets.

    Parameters
    ----------
    im : ndarray
      The image to shift.
    yx : floats
      `y, x` offsets.  Positive values move pixels to the
      up/right. [unsampled pixels]
    subsample : int, optional
      The sub-sampling factor.  If <=1, then the image is only shifted
      whole pixels.

    Returns
    -------
    sim : ndarray
      The shifted image (at the original pixel scale).

    """

    if subsample <= 1:
        subsample = 1

    sy = int(round(yx[0] * subsample)) # whole sampled pixels
    sx = int(round(yx[1] * subsample))

    sim = rebin(im, subsample, flux=True)
    sim = np.roll(sim, sy, 0)
    sim = np.roll(sim, sx, 1)

    return rebin(sim, -subsample, flux=True)

def rarray(shape, yx=None, subsample=0, dtype=float):
    """Array of distances from a point.

    Parameters
    ----------
    shape : array
      The shape of the resulting array `(y, x)`.
    yx : array, optional
      The center of the array `(y, x)`.  If set to None, then the
      center is `(shape - 1.0) / 2.0` (floating point arithmetic).
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
        yx = (np.array(shape) - 1.0) / 2.0
    else:
        yx = np.array(yx)

    y = yarray(shape, dtype=dtype) - yx[-2]
    x = xarray(shape, dtype=dtype) - yx[-1]
    r = (np.sqrt(x**2 + y**2)).astype(dtype)

    if subsample > 0:
        r = refine_center(rarray, r, yx, 11, subsample, scale=-1, dtype=dtype)

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

    if factor == 1:
        # done!
        return a

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
    if n**2 < stack.shape[0]:
        n += 1
    _stack = np.zeros((n**2, stack.shape[1], stack.shape[2]))
    _stack[:stack.shape[0]] = stack
    grid = _stack.reshape(n, n, _stack.shape[1], _stack.shape[2])
    grid = grid.swapaxes(1, 2)
    grid = grid.reshape(n * _stack.shape[1], -1)
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
      Set to the data type of the resulting array.

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
        th = refine_center(tarray, th, yx, 5, subsample, dtype=dtype)

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
      (averaging) to a resolution of 1 pixel.  When, for example, a
      function has dimensions of length, the rebinned array needs to
      be scaled by `subsample**-1`.
    **kwargs
      Additional keyword arguments for `func`.

    Returns
    -------
    refined : ndarray
      The refined array.

    Notes
    -----
    Numpy rounds to even numbers, which causes an error when working
    with an origin at half-pixel steps (i.e., 50.5 rounds to 50, but
    should be 51).  `refine_center` adds a small value to `yx` to
    mitigate this issue.  If better than 1e-4 pixel precision is
    needed, this function may not work for you.

    """

    refined = im.copy()

    # where is the center of the NxN region?
    yx_N = (np.ones(2) * N - 1.0) / 2.0
    yx_N += (yx - np.array([round(x) for x in yx]))

    # subsample the NxN region, where is the center of that?
    shape_c = np.ones(2) * N * subsample
    yx_c = (np.ones(2) * N * subsample - 1) / 2.0
    yx_c += (yx_N - np.array([round(x) for x in yx_N])) * subsample

    # generate the subsampled center
    refined_c = func(shape_c, yx=yx_c, subsample=0, **kwargs)
    refined_c = rebin(refined_c, -subsample, flux=False)
    refined_c *= subsample**scale

    # The region to be refined: xi, yi
    yi, xi = np.indices((N, N)) - N // 2

    # numpy only rounds to even numbers, adding a small value fixes
    # this.  Hopefully 1e-5 pixel precision is never needed!
    yi += int(np.around(yx[0] + 1e-5))
    xi += int(np.around(yx[1] + 1e-5))

    # insert into the result
    i = (yi >= 0) * (xi >= 0) * (yi < im.shape[0]) * (xi < im.shape[1])
    if np.any(i):
        refined[yi[i], xi[i]] = refined_c[i]

    return refined

def yx2rt(im, yx, dtdr=0, scale=None, bins=100, range=None,
          dtype=float):
    """Cartesian to polar transformation.

    Parameters
    ----------
    im : array
      The image on which to operate.
    yx : array
      The center of the array `(y, x)`.
    dtdr : float, optional
      The change of theta with radius.
    scale : int
      Scale the image via `rebin` before computing the radial profile.
      The outputs will be in units of the original pixels.  Following
      `rebin`, use negative scale factors for minification.
    bins : int or tuple, optional
      The number of bins for the final image or a 2-element array:
      `(radial-bins, theta-bins)`.
    range : array, optional
      The ranges in radius and theta to consdier:
        `[[rmin, rmax], [thmin, thmax]]`.  [pixels and radians]
    dtype : object, optional
      Set to the data type of the resulting array.

    Returns
    -------
    rt : ndarray
      The image transformed to place radius along the first dimension,
      and theta along the second dimension.
    rbins : ndarray
      The radial bins (from `np.histogram2d`). [pixels]
    thbins : ndarray
      The azimuthal bins (from `np.histogram2d`). [radians]
    n : ndarray
      The number of (sub)pixels that were placed into each bin.

    """

    if scale is not None:
        image = rebin(im, scale)
        if scale < 0:
            scale = 1.0 / scale
        yx = np.array(yx, float) * scale + (scale - 1) / 2.0
    else:
        image = im
        scale = 1
        yx = np.array(yx)

    r = rarray(image.shape, yx=yx, subsample=10, dtype=dtype) / scale
    th = tarray(image.shape, yx=yx, subsample=10, dtype=dtype) + np.pi
    th = (th + dtdr * r) % (2 * np.pi)

    r = r.flatten()
    th = th.flatten()
    image = image.flatten()

    rt = np.histogram2d(r, th, bins=bins, range=range, weights=image)
    rbin = np.histogram2d(r, th, bins=bins, range=range, weights=r)
    thbin = np.histogram2d(r, th, bins=bins, range=range, weights=th)
    n = np.histogram2d(r, th, bins=bins, range=range)[0]

    nn = n.astype(float)
    nn[n == 0] = 1
    return rt[0] / nn, rbin[0] / nn, thbin[0] / nn, n

def xarray(shape, yx=[0, 0], rot=0, dtype=int):
    """Array of x values.

    Parameters
    ----------
    shape : tuple of int
      The shape of the resulting array, e.g., (y, x).
    yx : tuple of float
      Offset the array to align with this y, x center.
    rot : float, optional
      Place the x-axis along this position angle, measured
      counterclockwise from the original x-axis. [radians]
    dtype : numpy.dtype, optional
      The data type of the result.

    Returns
    -------
    x : ndarray
      An array of x values.

    """

    from ..util import rotmat

    y, x = np.indices(shape, dtype)[-2:]
    y -= yx[0]
    x -= yx[1]

    if rot != 0:
        R = rotmat(rot)
        x = x * R[0, 0] + y * R[0, 1]

    return x

def yarray(shape, yx=[0, 0], rot=0, dtype=int):
    """Array of y values.

    Parameters
    ----------
    shape : tuple of int
      The shape of the resulting array, e.g., (y, x).
    yx : tuple of float
      Offset the array to align with this y, x center.
    rot : float, optional
      Place the y-axis along this position angle, measured
      counterclockwise from the original y-axis. [radians]
    dtype : numpy.dtype, optional
      The data type of the result.

    Returns
    -------
    y : ndarray
      An array of y values.

    """

    from ..util import rotmat

    y, x = np.indices(shape, dtype)[-2:]
    y -= yx[0]
    x -= yx[1]

    if rot != 0:
        R = rotmat(rot)
        y = x * R[1, 0] + y * R[1, 1]

    return y

# update module docstring
from ..util import autodoc
autodoc(globals())
del autodoc
