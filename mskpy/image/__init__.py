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
   rebin - Grow or shrink an array by an integer amount.
   stack2grid - Tile a stack of images.
   tarray - Create an array of angles with the same vertex.
   unwrap - Transform an array from a rectangular to azimuthal projection.
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

