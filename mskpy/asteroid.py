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

from .ephem import SolarSysObject, State
from .models import SurfaceRadiation, DAp, NEATM

class Asteroid(SolarSysObject):
    """An asteroid.

    Parameters
    ----------
    state : State
      The location of the asteroid.
    D : Quantity
      Diameter.
    Ap : float
      Geometric albedo.
    reflected : dict or SurfaceRadiation, optional
      A model of the reflected light.  If `None` a `DAp` model will be
      initialized (including `**kwargs`).
    thermal : dict or SurfaceRadiation, optional
      A model of the thermal emission.  If `None` a `NEATM` model will
      be initialized (including `**kwargs`).
    **kwargs
      Additional keywords for the default `reflected` and `thermal`
      models.

    Methods
    -------
    fluxd : Total flux density as seen by an observer.

    """

    def __init__(self, state, D, Ap, reflected=None, thermal=None, **kwargs):
        assert isinstance(state, State), "state must be a State."
        assert isinstance(D, u.Quantity), "D must be a Quantity."
        self.state = state
        self.D = D
        self.Ap = Ap

        if isinstance(reflected, SurfaceRadiation):
            self.reflected = reflected
        elif reflected is None:
            self.reflected = DAp(self.D, self.Ap)
        else:
            self.reflected = DAp(self.D, self.Ap, **reflected)

        if isinstance(thermal, SurfaceRadiation):
            self.thermal = thermal
        elif thermal is None:
            self.thermal = NEATM(self.D, self.Ap)
        else:
            self.thermal = NEATM(self.D, self.Ap, **thermal)

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

        assert isinstance(observer, SolarSysObject), "observer must be a SolarSysObject"
        assert isinstance(wave, u.Quantity), "wave must be a Quantity"

        fluxd = np.zeros(np.size(wave.value)) * unit
        if self.D.value <= 0:
            return fluxd

        g = observer.observe(self, date, ltt=ltt)

        if reflected:
            fluxd += self.reflected.fluxd(g, wave, unit=unit)
        if thermal:
            fluxd += self.thermal.fluxd(g, wave, unit=unit)

        return fluxd

# update module docstring
from .util import autodoc
autodoc(globals())
del autodoc

