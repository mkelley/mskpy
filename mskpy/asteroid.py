# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
asteroid --- Asteroids!
=======================


.. autosummary::
   :toctree: generated/

   Classes
   -------
   Asteroid

"""

__all__ = [
    'Asteroid'
]


import numpy as np
import astropy.units as u
from astropy.time import Time

from .ephem import SpiceObject
from .models import DAv, NEATM

class Asteroid(SpiceObject):
    """An asteroid.

    Parameters
    ----------
    obj : string or int
      The name, NAIF ID, or integer designation of the object.
    D : Quantity
      Diameter.
    Ap : float
      Geometric albedo.
    reflected : SurfaceEmission, optional
      Will be initialized using **kwargs.
    thermal : SurfaceEmission, optional
      Will be initialized using **kwargs.
    kernel : string, optional
      The name of an ephemeris kernel in which to find the ephemeris
      for `obj`.
    **kwargs
      Additional keywords are passed to `reflected` and `thermal`.

    Methods
    -------
    fluxd : Total flux density as seen by an observer.

    """

    def __init__(self, obj, D, Ap, reflected=DAv, thermal=NEATM,
                 kernel=None, **kwargs):
        self.obj = obj
        self.D = D
        self.Ap = Ap
        self.reflected = reflected(self.D, self.Ap, **kwargs)
        self.thermal = thermal(self.D, self.Ap, **kwargs)
        self.kernel = kernel
        SpiceObject.__init__(self, obj, kernel=kernel)

    def fluxd(self, observer, date, wave, reflected=True, thermal=True,
              ltt=False, unit=u.Unit('W / (m2 um)')):
        """Total flux density as seen by an observer.

        Parameters
        ----------
        observer : SolarSysObject
          The observer.
        date : string, float, astropy Time, datetime
          The time of the observation in any format acceptable to
          `observer`.
        wave : Quantity
          The wavelengths to compute `fluxd`.
        reflected : bool, optional
          If `True` include the reflected light model.
        thermal : bool, optional
          If `True` include the thermal emission model.
        ltt : bool, optional
          Set to `True` to correct the asteroid's position for light
          travel time.
        unit : astropy Unit
          The return unit, must be flux density.
        
        Returns
        -------
        fluxd : Quantity

        """

        if self.D <= 0:
            return np.zeros(len(wave)) * unit

        g = observer.observe(self, date, ltt=ltt)

        if reflected:
            reflected = self.reflected.fluxd(g, wave, unit=unit)
        else:
            reflected = 0 * unit

        if thermal:
            thermal = self.thermal.fluxd(g, wave, unit=unit)
        else:
            thermal = 0 * unit

        return reflected + thermal

