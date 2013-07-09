# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
midir --- Mid-infrared instruments.
===================================

   Classes
   -------
   MIRSI

"""

import numpy as np
import astropy.units as u

try:
    from ..ephem import Earth
except ImportError:
    Earth = None

from .instrument import Instrument, Camera, LongSlitSpectrometer

__all__ = ['MIRSI']

class MIRSI(Instrument):
    """Mid-Infrared Spectrometer and Imager.

    Attributes
    ----------
    imager : `Camera` for imaging mode.
    sp10r200 : `LongSlitSpectrometer` for 10-micron spectroscopy.
    sp20r100 : `LongSlitSpectrometer` for 20-micron spectroscopy.
    mode : The current MIRSI mode (see examples).

    Examples
    --------

    """

    shape = (240, 320)
    ps = 0.265 * u.arcsec
    location = Earth

    def __init__(self):
        w = [4.9, 7.7, 8.7, 9.8, 10.6, 11.6, 12.7, 20.6, 24.4] * u.um
        self.imager = Camera(w, self.shape, self.ps, location=self.location)

        self.sp10r200 = LongSlitSpectrometer(10.5 * u.um, self.shape, self.ps,
                                             2.25, 0.022 * u.um, R=200,
                                             location=self.location)

        self.sp20r100 = LongSlitSpectrometer(21.5 * u.um, self.shape, self.ps,
                                             4.5, 0.028 * u.um, R=100,
                                             location=self.location)

        self._mode = 'imager'

    @property
    def mode(self):
        if self._mode in ['imager', 'sp10r200', 'sp20r100']:
            return self.__dict__[self._mode]
        else:
            raise KeyError("Invalid mode: {:}".format(self._mode))

    def sed(self, *args, **kwargs):
        """Spectral energy distribution of a target.

        Parameters
        ----------
        *args
        **kwargs
          Arguments and keywords depend on the current MIRSI mode.

        Returns
        -------
        sed : ndarray

        """
        return self.mode.sed(*args, **kwargs)

    def lightcurve(self, *args, **kwargs):
        """Secular lightcurve of a target.

        Parameters
        ----------
        *args
        **kwargs
          Arguments and keywords depend on the current MIRSI mode.

        Returns
        -------
        lc : astropy Table

        """
        return self.mode.lightcurve(*args, **kwargs)


# update module docstring
from ..util import autodoc
autodoc(globals())
del autodoc
