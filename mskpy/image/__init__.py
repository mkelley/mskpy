# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
image --- For working with images, maybe spectra.
=================================================


.. autosummary::
   :toctree: generated/

   Classes
   -------
   Image

   General
   -------
   imshift
   rarray
   rebin
   stack2grid
   tarray
   unwrap
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
   jailbar
   imcombine
   mkflat
   psfmatch
   stripes

   FITS or WCS
   -----------
   basicwcs
   getrot

"""

#__all__ = [
#    ''
#]

import numpy as np

from .core import *

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
