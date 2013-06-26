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

from .ephem import SolarSysObject, SpiceObject
from .models import SurfaceEmission, DAv, NEATM

class Asteroid(SolarSysObject):
    """An asteroid.

    Parameters
    ----------
    obj : string, int, or SolarSysObject
      The name, NAIF ID, or integer designation of the object, or a
      `SolarSysObject`.
    D : Quantity
      Diameter.
    Ap : float
      Geometric albedo.
    reflected : dict or SurfaceEmission, optional
      A model of the reflected light.  If `None` a `DAp` model will be
      initialized (including `**kwargs`).
    thermal : dict or SurfaceEmission, optional
      A model of the thermal emission.  If `None` a `NEATM` model will
      be initialized (including `**kwargs`).
    kernel : string, optional
      The name of an ephemeris kernel in which to find the ephemeris
      for `obj`.
    **kwargs
      Additional keywords for the default `reflected` and `thermal`
      models.

    Methods
    -------
    fluxd : Total flux density as seen by an observer.

    """

    def __init__(self, obj, D, Ap, reflected=None, thermal=None,
                 kernel=None, **kwargs):
        if isinstance(obj, SolarSysObject):
            self.obj = obj
        else:
            self.obj = SpiceObject(obj, kernel=kernel)

        self.D = D
        self.Ap = Ap

        if isinstance(reflected, SurfaceEmission):
            self.reflected = reflected
        else:
            self.reflected = DAp(self.D, self.Ap, **reflected)

        if isinstance(thermal, SurfaceEmisssion):
            self.thermal = thermal
        else:
            self.thermal = NEATM(self.D, self.Ap, **thermal)

        self.kernel = kernel
        SpiceObject.__init__(self, obj, kernel=kernel)

    def r(self, date):
        return self.obj.r(date)
    r.__doc__ = self.obj.r.__doc__

    def v(self, date):
        return self.obj.v(date)
    v.__doc__ = self.obj.v.__doc__

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
          Set to `True` to correct the object's position for light
          travel time.
        unit : astropy Unit
          The return unit, must be flux density.
        
        Returns
        -------
        fluxd : Quantity

        """

        fluxd = np.zeros(len(wave)) * unit
        if self.D <= 0:
            return fluxd

        g = observer.observe(self.obj, date, ltt=ltt)

        if reflected:
            fluxd += self.reflected.fluxd(g, wave, unit=unit)
        if thermal:
            fluxd += self.thermal.fluxd(g, wave, unit=unit)

        return fluxd
