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
        ps = 1.22 * u.arcsec
        location = Spitzer
        Camera.__init__(self, w, shape, ps, location=location)

    def ccorrection(self, sf, channels=[1, 2, 3, 4]):
        """IRAC color correction.

        Seems to agree within 1% of the IRAC Instrument Handbook.
        Thier quoted values are good to ~1%.

        Parameters
        ----------
        sf : function
          A function that generates source flux density as a Quantity
          given wavelength as a Quantity.
        channels : list, optional
          A list of the IRAC channels for which to compute the color
          correction, e.g., `[1, 2]` for 3.6 and 4.5 um.

        Returns
        -------
        K : ndarray
          Color correction factor, where `Fcc = F / K`.

        """

        from scipy import interpolate
        import astropy.constants as const
        from ..calib import filter_trans
        from ..util import davint, takefrom

        nu0 = (const.c.si / self.wave).to(u.teraHertz).value
        K = np.zeros(len(channels))
        for ch in channels:
            tw, tr = filter_trans('IRAC CH{:}'.format(ch))
            nu = (const.c / tw).to(u.teraHertz).value

            sfnu = sf(tw).to(u.Jy, u.spectral_density(tw)).value

            i = ch - 1  # self.wave index
            sfnu /= sf(self.wave[i]).to(u.Jy, u.spectral_density(self.wave[i])).value

            sfnu, tr, nu = takefrom((sfnu, tr, nu), nu.argsort())
            K[i] = (davint(nu, sfnu * tr * nu0[i] / nu, nu[0], nu[-1])
                    / davint(nu, tr * (nu0[i] / nu)**2, nu[0], nu[-1]))

        return K

#    def ccorrection_tab(self, sw, sf):
#        """IRAC color correction of a tabulated spectrum.
#
#        Parameters
#        ----------
#        sw : Quantity
#          Source wavelength.
#        sf : Quantity
#          Source flux density.
#
#        Returns
#        -------
#        K : ndarray
#          Color correction: `Fcc = F / K`.
#
#        """
#
#        from scipy import interpolate
#        import astropy.constants as const
#        from ..calib import filter_trans
#        from ..util import davint, takefrom
#
#        nu0 = (const.c.si / self.wave).to(u.teraHertz).value
#        K = np.zeros(4)
#        for i in range(4):
#            tw, tr = filter_trans('IRAC CH{:}'.format(i + 1))
#            nu = (const.c / tw).to(u.teraHertz).value
#
#            # interpolate the filter transmission to a higher
#            # resolution
#            t
#
#            s = interpolate.splrep(sw.value, sf.value)
#            _sf = interpolate.splev(fw.value, s, ext=1)
#            _sf /= interpolate.splev(self.wave[i].value, s, ext=1)
#
#            _sf *= sf.unit.to(u.Jy, u.spectral_density(fw))
#
#            _sf, ft, nu = takefrom((_sf, ft, nu), nu.argsort())
#            K[i] = (davint(nu, _sf * ft * nu0[i] / nu, nu[0], nu[-1])
#                    / davint(nu, ft * (nu0[i] / nu)**2, nu[0], nu[-1]))
#        return K

# update module docstring
from ..util import autodoc
autodoc(globals())
del autodoc
