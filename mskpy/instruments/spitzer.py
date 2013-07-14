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
    from ..ephem import Spitzer
except ImportError:
    Spitzer = None

from .instrument import Instrument, Camera, LongSlitSpectrometer

__all__ = ['IRAC']

class IRAC(Camera):
    """InfraRed Array Camera

    Attributes
    ----------

    Examples
    --------

    """

    def __init__(self):
        w = [3.550, 4.493, 5.731, 7.872] * u.um
        shape = (256, 256)
        ps = 1.22 * u.um
        location = Spitzer
        Camera.__init__(self, w, shape, ps, location=location)

    def ccorrection(self, sf):
        """IRAC color correction.

        Parameters
        ----------
        sf : function
          A function that generates source flux density as a Quantity
          given wavelength as a Quantity.

        Returns
        -------
        K : ndarray
          Color correction factor, where `Fcc = F / K`.

        """

        from scipy import interpolate
        import astropy.constants as const
        from ..calib import filter_trans
        from ..util import davint

        nu0 = (const.c.si / self.wave).Hertz
        K = np.zeros(4)
        for i in range(4):
            fw, ft = filter_trans('IRAC CH{:}'.format(i + 1))

            _sf = sf(fw).to(u.Jy, equivalencies=u.spectral_density(fw.unit, fw))
            _sf /= sf(self.wave[i])

            nu = (const.c / fw).Hertz
            j = nu.argsort()
            K[i] = (davint(nu[j], (_sf * ft * nu0[i] / nu)[j], nu[-1], nu[0])
                    / davint(nu[j], (ft * (nu / nu0[i])**2)[j], nu[-1], nu[0]))

        return K

    def ccorrection_tab(self, sw, sf):
        """IRAC color correction of a tabulated spectrum.

        Parameters
        ----------
        sw : Quantity
          Source wavelength.
        sf : Quantity
          Source flux density.

        Returns
        -------
        K : ndarray
          Color correction: `Fcc = F / K`.

        """

        from scipy import interpolate
        import astropy.constants as const
        from ..calib import filter_trans
        from ..util import davint

        nu0 = (const.c.si / self.wave).Hertz
        K = np.zeros(4)
        for i in range(4):
            fw, ft = filter_trans('IRAC CH{:}'.format(i + 1))

            s = interpolate.splrep(sw, sf.value)
            _sf = interpolate.splev(fw, s, ext=1)
            _sf /= interpolate.splev(self.wave[i], s, ext=1)

            equiv = u.spectral_density(fw.unit, fw)
            _sf *= sf.unit.to(u.Jy, equivalencies=equiv)

            nu = (const.c / fw).Hertz
            j = nu.argsort()
            K[i] = (davint(nu[j], (_sf * ft * nu0[i] / nu)[j], nu[-1], nu[0])
                    / davint(nu[j], (ft * (nu / nu0[i])**2)[j], nu[-1], nu[0]))

        return K

# update module docstring
from ..util import autodoc
autodoc(globals())
del autodoc
