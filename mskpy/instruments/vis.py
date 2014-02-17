# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
vis --- Visual-band instruments.
================================

   Classes
   -------
   OptiPol

"""

import numpy as np
import astropy.units as u

try:
    from ..ephem import Earth
except ImportError:
    Earth = None

from .instrument import Instrument, Camera

__all__ = ['OptiPol']

class OptiPol(Camera):
    """University of Minnesota's optical imaging polarimeter.

    Effective wavelengths are for an A0V star.

    Methods
    -------
    read : Read an Optipol image from a file.

    """

    top = slice(500, 1000)
    bot = slice(None, 500)
    x = slice(-1000, None)

    def __init__(self):
        # UBVRI
        w = [0.37, 0.44, 0.54, 0.64, 0.80] * u.um

        shape = (500, 1000)
        ps = 0.2 * u.arcsec
        location = Earth
        Camera.__init__(self, w, shape, ps, location=location)

    def read(self, filename, dark=0, flats=[1, 1], header=False):
        """Read an OptiPol image from a file.

        Parameters
        ----------
        filename : string
          The name of the file.
        dark : array, optional
          A full-sized dark frame to subtract from the image.
        flats : array, optional
          The flat-field corrections: `[top_flat, bottom_flat]`.
        header : bool, optional
          Set to `True` to return the file's FITS header.

        Returns
        -------
        top, bot : Image
          The top and bottom frames, split from the original, and,
          optionally, dark subtracted and flat-field corrected.
        header : astropy FITS header, optional
          The image's FITS header.

        """

        from astropy.io import fits
        from ..image import Image

        im = fits.getdata(filename) - dark
        top = Image(im[self.top, self.x] / flats[0])
        bot = Image(im[self.bot, self.x] / flats[1])

        if header:
            return top, bot, fits.getheader(filename)
        else:
            return top, bot

    def readQU(self, qfile, ufile, dark=0, qflats=[1, 1], uflats=[1, 1]):
        """Read a QU sequence from files.

        Parameters
        ----------
        qfile, ufile : string
          The names of the files.
        dark : array, optional
          A full-sized dark frame to subtract from the image.
        qflats, uflats : array, optional
          The flat-field corrections for the Q and U frames:
          `[top_flat, bottom_flat]`.

        Returns
        -------
        pol : HalfWavePlate
          The images as a polarization object.

        """

        from astropy.io import fits
        from ..polarimetry import HalfWavePlate

        qtb = self.read(qfile, dark=dark, flats=qflats)
        utb = self.read(ufile, dark=dark, flats=uflats)

        return HalfWavePlate((qtb[0], utb[0], qtb[1], utb[1]))
        

# update module docstring
from ..util import autodoc
autodoc(globals())
del autodoc
