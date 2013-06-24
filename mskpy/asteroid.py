# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
asteroid --- Asteroids!
=======================


.. autosummary::
   :toctree: generated/


"""

__all__ = []


import numpy as np
import astropy.units as u
from astropy.time import Time

from .ephem import SpiceObject
from .models import Dpv, NEATM

class Asteroid(SpiceObject):
    """An asteroid.

    Parameters
    ----------
    obj : string or int
      The name, NAIF ID, or integer designation of the object.
    D : Quantity
      Diameter.
    pv : float
      Geometric albedo.
    reflected : SurfaceEmission
      Will be initialized using **kwargs.
    thermal : SurfaceEmission
      Will be initialized using **kwargs.
    kernel : string
      The name of an ephemeris kernel in which to find the ephemeris
      for `obj`.
    **kwargs
      Additional keywords are passed to `reflected` and `thermal`.

    Methods
    -------
    fluxd : Total flux density as seen by an observer.

    """

    def __init__(self, obj, D, pv, reflected=Dpv, thermal=NEATM,
                 kernel=None, **kwargs):
        self.obj = obj
        self.D = D
        self.pv = pv
        self.reflected = reflected(self.D, self.pv, **kwargs)
        self.thermal = thermal(self.D, self.pv, **kwargs)
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

