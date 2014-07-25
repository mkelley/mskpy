# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
image --- For working with images, maybe spectra.
=================================================

.. autosummary::
   :toctree: generated/

   Classes
   -------
   Image

   Core
   ----
   imshift
   rarray
   rebin
   stack2grid
   tarray
   yx2rt
   xarray
   yarray

   Analysis
   --------
   anphot
   apphot
   azavg
   bgfit
   centroid
   gcentroid
   imstat
   linecut
   polyfit2d
   radprof
   trace


   Processing
   ----------
   columnpull
   crclean
   fixpix
   fwhmfit
   mkflat
   psfmatch
   stripes

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
    bgphot - Background photometry in an annulus.
    centroid - Simple center of mass centroiding.
    gcentroid - Simple centroiding by Gaussian fits.
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

    def bgphot(self, rap, yx=None, **kwargs):
        """Background photometry and error analysis in an annulus.

        Pixels are not sub-sampled.  The annulus is processed via
        `util.uclip`.

        Parameters
        ----------
        rap : array
          Inner and outer radii of the annulus.  [pixels]
        yx : array, optional
          The `y, x` center of the aperture.  Only one center is
          allowed. If `yx` is None, `self.yx` will be used. [pixels]
        **kwargs :
          Any `analysis.bgphot` keywords.

        Returns
        -------
        n : ndarray
          The number of pixels per annular bin.  Same shape comment as for
          `bg`.
        bg : ndarray
          The background level in the annulus.  If `im` is a set of images
          of depth `N`, `bg` will have shape `N x len(rap)`
        sig : ndarray
          The standard deviation of the background pixels.  Same shape
          comment as for `bg`.

        """

        yx = self.yx if yx is None else yx
        return bgphot(self, yx, rap, **kwargs)

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

    def gcentroid(self, yx=None, **kwargs):
        """Simple centroiding by Gaussian fits.

        Parameters
        ----------
        yx : array, optional
          The initial guess, or `None` to use `self.yx`.
        **kwargs
          Any `analysis.gcentroid` keywords.

        Returns
        -------
        cyx : ndarray
          The computed center.  The lower-left corner of a pixel is
          -0.5, -0.5.

        """
        yx = self.yx if yx is None else yx
        return gcentroid(self, yx=yx, **kwargs)

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
        rc : ndarray
          The center of each radial bin.
        sb : ndarray
          The surface brightness at each `r`.
        n : ndarray
          The number of pixels in each bin.
        rmean : ndarray
          The mean radius of the points in each radial bin.

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

# update module docstring
from ..util import autodoc
autodoc(globals())
del autodoc

