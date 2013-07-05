# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
image --- For working with images, maybe spectra.
=================================================


.. autosummary::
   :toctree: generated/

   Classes
   -------
   Image - An ndarray with builtin analysis functions.

   Core
   ----
   imshift - Shift an image.
   rarray - Create an array of distances to a point.
   rebin - Scale an array by an integer amount.
   stack2grid - Tile a stack of images.
   tarray - Create an array of angles with the same vertex.
   yx2rt - Cartesian to polar transformation.
   xarray - Create an array of distances along a line.
   yarray - Create an array of distances along a line.

   Analysis
   --------
   anphot - Annular photometry.
   apphot - Aperture photometry.
   azavg - Azimuthal averaging.
   bgfit - 2D background fitting.
   centroid - Image centroid (center of mass).
   imstat - A suite of array statistics.
   linecut - Linear photometry.
   polyfit2d - 2D polynomial fitting.
   radprof - Radial profiling.
   trace - Peak fitting along an axis.

   Processing
   ----------
   columnpull - Define a column pull detector artifact.
   crclean - Clean cosmic rays (LACOSMIC).
   fixpix - Replace masked pixels.
   fwhmfit - Measure the FWHM of a image.
   mkflat - Create a flat field.
   psfmatch - Generate a PSF matching kernel.
   stripes - Define jailbar/stripe artifacts.

"""

import numpy as np

from . import core
from . import analysis
from . import process

from .core import *
from .analysis import *
from .process import *

__all__ = core.__all__ + analysis.__all__ + process.__all__ + ['Image']

class Image(np.ndarray):
    """A simple Image class for astronomy.

    A way package of some image analysis routines into an ndarray.

    Parameters
    ----------
    arr : array
      The image data.  Must be 2 dimensional.
    yx : array, optional
      The center of an object of interest. [pixels]

    Attributes
    ----------
    yx : ndarray
      The center of an object of interest. [pixels]

    All numpy.ndarray attributes.

    Methods
    -------
    anphot - Annular photometry.
    apphot - Aperture photometry.
    azavg - Create an azimuthally averaged image.
    bgfit - 2D polynomial fit to the background.
    centroid - Simple center of mass centroiding.
    linecut - Photometry along a line.
    radprof - Radial profiling.
    rebin - Image scaling.
    shift - Shift an image (drizzle).
    stat - A suite of image statistics.
    yx2rt - Transform from a rectangular to azimuthal projection.

    """

    def __new__(cls, im, yx=(0, 0)):
        obj = np.asarray(im).view(cls)
        obj.yx = np.array(yx)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.yx = np.array(getattr(obj, 'yx', (0, 0)))

    def __array_wrap__(self, out_arr, context=None):
        if out_arr.ndim == 2:
            return np.ndarray.__array_wrap__(self, out_arr, context)
        else:
            return np.ndarray.__array_wrap__(out_arr, out_arr, context)

    def anphot(self, rap, yx=None, subsample=4):
        """Annular photometry.

        Parameters
        ----------
        rap : float or array
          Aperture radii.  [pixels]
        yx : array, optional
          The center of the apertures.  If `yx` is None, `self.yx`
          will be used.
        subsample : int, optional
          The sub-pixel sampling factor.  Set to `<= 1` for no
          sampling.  This will sub-sample the entire image.

        Returns
        -------
        n : ndarray
          The number of pixels per aperture.
        f : ndarray
          The photometry.  If `im` is a set of images of depth `N`,
          `f` will have shape `N x len(rap)`.

        """
        yx = self.yx if yx is None else yx
        return anphot(self, yx, rap, subsample=subsample)

    def apphot(self, rap, yx=None, subsample=4):
        """Simple aperture photometry.

        Parameters
        ----------
        rap : float or array
          Aperture radii.  [pixels]
        yx : array, optional
          The center of the apertures.  If `yx` is None, `self.yx`
          will be used.
        subsample : int, optional
          The sub-pixel sampling factor.  Set to `<= 1` for no
          sampling.  This will sub-sample the entire image.

        Returns
        -------
        n : ndarray
          The number of pixels per aperture.
        f : ndarray
          The photometry.  If `im` is a set of images of depth `N`,
          `f` will have shape `N x len(rap)`.

        """
        yx = self.yx if yx is None else yx
        return apphot(self, yx, rap, subsample=subsample)

    def azavg(self, yx=None, **kwargs):
        """Compute an azimuthally averaged image.

        Parameters
        ----------
        yx : array, optional
          The center of the averaging.  If `yx` is None, `self.yx`
          will be used.
        **kwargs
          Any `analysis.azavg` keywords.

        Returns
        -------
        aa : ndarray

        """
        yx = self.yx if yx is None else yx
        return azavg(self, yx, **kwargs)

    def bgfit(self, **kwargs):
        """Fit the background.

        Parameters
        ----------
        **kwargs
          Any `analysis.bgfit` keywords.

        Returns
        -------
        bg : ndarray
          An image of the best-fit background.

        """
        return bgfit(self, **kwargs)

    def centroid(self, yx=None, **kwargs):
        """Simple center of mass centroiding.

        Parameters
        ----------
        yx : array, optional
          The initial guess, or `None` to use `self.yx`.
        **kwargs
          Any `analysis.centroid` keywords.

        Returns
        -------
        cyx : ndarray
          The computed center of mass.  The lower-left corner of a
          pixel is -0.5, -0.5.

        """
        yx = self.yx if yx is None else yx
        return centroid(self, yx=yx, **kwargs)

    def linecut(self, width, length, pa, yx=None, subsample=4):
        """Photometry along a line.

        Parameters
        ----------
        width : float
        length: float or array
          If `length` is a float, the line cut will be boxes of size
          `width` x `width` along position angle `pa`, spanning a
          total length of `length`.  If `length` is an array, it
          specifies the bin edges along `pa`, each bin having a width
          of `width`.
        pa : float
          Position angle measured counter-clockwise from the
          x-axis. [degrees]
        yx : array, optional
          The `y, x` center of the extraction.  If `None`, `self.yx`
          will be used.
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
        yx = self.yx if yx is None else yx
        return linecut(self, yx, width, length, pa, subsample=subsample)

    def radprof(self, yx=None, **kwargs):
        """Radial profile.

        Parameters
        ----------
        yx : array
          The center of the profile.  If `yx` is `None`, `self.yx`
          will be used.

        Returns
        -------
        r : ndarray
          The mean radius of each point in the surface brightness profile.
        sb : ndarray
          The surface brightness at each `r`.
        n : ndarray
          The number of pixels in each bin.

        """
        yx = self.yx if yx is None else yx
        return radprof(self, yx, **kwargs)

    def rebin(self, factor, **kwargs):
        """Rebin by integer amounts.

        Parameters
        ----------
        factor : int
          Rebin factor.
        **kwargs
          Any valid `core.rebin` keywords.

        Returns
        -------
        rim : ndarray

        """
        return rebin(self, factor, **kwargs)

    def shift(self, yx, subsample=4):
        """Shift the image, allowing for sub-pixel offsets (drizzle).

        Parameters
        ----------
        im : ndarray
          The image to shift.
        yx : floats
          `y, x` offsets.  Positive values move pixels to the
          up/right. [unsampled pixels]
        subsample : int, optional
          The sub-sampling factor.

        Returns
        -------
        sim : ndarray
          The shifted image (at the original pixel scale).

        """
        return imshift(self, yx, subsample=subsample)

    def stat(self, **kwargs):
        """Some quick image statistics.

        Parameters
        ----------
        **kwargs
          Any valid `analysis.imstat` keywords.

        Returns
        -------
        stats : dict
          See `analysis.imstat` for details.

        """
        return imstat(self, **kwargs)

    def yx2rt(self, yx=None, **kwargs):
        """Cartesian to polar transformation.

        Parameters
        ----------
        yx : array, optional
          The center of the transformation.  If yx is None, then
          self.yx will be used.
        **kwargs
          Any `core.unwrap` keyowrd.

        Returns
        -------
        rt : ndarray

        """
        yx = self.yx if yx is None else yx
        return yx2rt(self, yx, **kwargs)
