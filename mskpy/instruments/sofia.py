# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
spitzer --- Spitzer instruments.
================================

   Classes
   -------
   IRAC

"""

import numpy as np
import astropy.units as u

try:
    from ..ephem import Earth
except ImportError:
    Earth = None

from .instrument import Camera, LongSlitSpectrometer

__all__ = ['FLITECAM']

class FLITECAM(object):
    """FLITECAM.

    Attributes
    ----------
    camera : Direct imaging mode.
    grism : Dictionary of spectroscopy modes.

    """

    shape = (1024, 1024)
    ps = 0.475 * u.arcsec
    location = Earth
    readnoise = 40 * u.electron
    welldepth = 80e3 * u.electron

    def __init__(self):
        w = [1.25, 1.64, 2.12, 3.05, 3.55, 3.61, 4.81, 4.87] * u.um
        self.camera = Camera(w, self.shape, self.ps, location=self.location)

        R = 1700
        modes = {
            'A1LM': (4.395, 5.533),
            'A2KL': (2.211, 2.722),
            'A3Hw': (1.493, 1.828),
            'B1LM': (3.303, 4.074),
            'B2Hw': (1.675, 2.053),
            'B3J':  (1.142, 1.385),
            'C2LM': (2.758, 3.399),
            'C3Kw': (1.854, 2.276),
            'C4Hw': (1.408, 1.718)}

        self.grism = dict()
        for k, v in modes.items():
            dlam = ((v[1] - v[0]) / self.shape[1]) * u.um
            self.grism[k] = LongSlitSpectrometer(
                np.mean(v) * u.um, self.shape, self.ps, 2.1, dlam,
                self.location)

# update module docstring
from ..util import autodoc
autodoc(globals())
del autodoc
