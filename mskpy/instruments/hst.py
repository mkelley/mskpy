# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
hst --- Hubble instruments.
================================

   Classes
   -------
   WFC3

"""

import numpy as np
import astropy.units as u

try:
    from ..ephem import Earth
except ImportError:
    Earth = None

from .instrument import Instrument, Camera, LongSlitSpectrometer

__all__ = ['WFC3UVIS']

class WFC3UVIS(Camera):
    """Wide Field Camera 3 UV Visible

    Attributes
    ----------

    Examples
    --------

    """

    def __init__(self):
        w = [0.438, 0.606, 0.775] * u.um
        shape = (4096, 2051)
        ps = 0.0395 * u.arcsec
        location = Earth
        Camera.__init__(self, w, shape, ps, location=location)

    def diffusion(self, wave, im):
        """Simulate UVIS's charge diffusion.

        Parameters
        ----------
        wave : float
          The effective wavelength of the image. [micron]
        im : ndarray
          The image to process.

        Returns
        -------
        newim : ndarray
          The processed image.

        Notes
        -----
        Based on diffusion parameters from STScI Instrument Science Report
        2008-014.

        """

        from astropy.convolution import convolve

        w0 = [0.250, 0.810]  # micron
        k0 = np.array([[[0.027, 0.111, 0.027],
                        [0.111, 0.432, 0.111],
                        [0.027, 0.111, 0.027]],
                       [[0.002, 0.037, 0.002],
                        [0.037, 0.844, 0.037],
                        [0.002, 0.037, 0.002]]])
        dk = (k0[1] - k0[0]) / (w0[1] - w0[0])
        k = k0[0] + dk * (wave - w0[0])
        k /= k.sum()
        return convolve(im, k, boundary='extend')


# update module docstring
from ..util import autodoc
autodoc(globals())
del autodoc
