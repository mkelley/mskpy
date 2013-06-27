# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
comet --- Comets!
=================

.. autosummary::
   :toctree: generated/

   Classes
   -------
   Comet - All(?) things comet.

"""

__all__ = [
    'Coma',
    'Comet'
]


import numpy as np
import astropy.units as u
from astropy.time import Time

from .ephem import SolarSysObject, SpiceObject
from .asteroid import Asteroid
from .models import DustEmission, ScatteredSun, SingleBB

class Coma(SolarSysObject):
    """A comet coma.

    Parameters
    ----------
    obj : string, int, or SolarSysObject
      The name, NAIF ID, or integer designation of the object, or a
      `SolarSysObject`.
    Afrho : Quantity
      Afrho of the coma.
    reflected : dict or AfrhoEmission, optional
      A model for light scattered by dust or a dictionary of keywords
      to pass to `AfrhoScattered`.
    thermal : dict or AfrhoEmission, optional
      A model for thermal emission from dust or a dictionary of
      keywords to pass to `AfrhoThermal`.
    kernel : string, optional
      The name of an ephemeris kernel in which to find the ephemeris
      for `obj`.

    """
    def __init__(self, obj, Afrho, reflected=dict(), thermal=dict(),
                 kernel=None):
        if isinstance(obj, SolarSysObject):
            self.obj = obj
        else:
            self.obj = SpiceObject(obj, kernel=kernel)

        self.Afrho = Afrho

        if isinstance(reflected, AfrhoEmission):
            self.reflected = reflected
        else:
            self.reflected = AfrhoScattered(
                1 * self.Afrho.unit, **reflected)

        if isinstance(thermal, AfrhoEmission):
            self.thermal = thermal
        else:
            self.thermal = AfrhoThermal(
                1 * self.Afrho.unit, self.Afrho, **thermal)

    def r(self, date):
        return self.obj.r(date)
    r.__doc__ = self.obj.r.__doc__

    def v(self, date):
        return self.obj.v(date)
    v.__doc__ = self.obj.v.__doc__

    def fluxd(self, observer, date, wave, rap, reflected=True, thermal=True,
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
        rap : Quantity
          The aperture radius, angular or projected distance at the
          comet.
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

        g = observer.observe(self, date, ltt=ltt)
        fluxd = np.zeros(len(wave)) * unit

        if reflected:
            f = self.reflected.fluxd(g, wave, rap, unit=unit)
            fluxd += f * self.Afrho
        if thermal:
            f = self.thermal.fluxd(g, wave, rap, unit=unit)
            fluxd += f * self.Afrho

        return fluxd

class Comet(SolarSysObject):
    """A comet.

    Parameters
    ----------
    obj : string, int, or SolarSysObject
      The name, NAIF ID, or integer designation of the object, or a
      `SolarSysObject`.
    nucleus : dict or Asteroid
      The nucleus of the comet as an `Asteroid` instance, or a
      dictionary of keywords to initialize a new `Asteroid`, including
      `R` or `D` and `Ap`.  Yes, `R` is allowed, this is a comet
      nucleus, afterall.
    coma : dict or Coma
      The coma of the comet as a `Coma` instance or a dictionary of
      keywords to initialize a new `Coma`, including `Afrho`.
    kernel : string, optional
      The name of an ephemeris kernel in which to find the ephemeris
      for `obj`.

    Attributes
    ----------
    nucleus : The nucleus as an `Asteroid` instance.
    coma : The coma as a `Coma` instance.

    Methods
    -------
    fluxd : Total flux density as seen by an observer.

    Examples
    --------
    Create a comet with `Asteroid` and `Coma` objects.

    >>> import astropy.units as u
    >>> from mskpy import Asteroid, Coma, Comet
    >>> nucleus = Asteroid(0.6 * u.km, 0.04, eta=1.0, epsilon=0.95)
    >>> coma = Coma(302 * u.cm, S=0.0, A=0.37, Tscale=1.18)
    >>> comet = Comet('hartley 2', nucleus=nucleus, coma=coma)

    Create a comet with keyword arguments.

    >>> import astropy.units as u
    >>> from mskpy import Comet
    >>> nucleus = dict(R=0.6 * u.km, Ap=0.04, eta=1.0, epsilon=0.95)
    >>> coma = dict(Afrho=302 * u.cm, S=0.0, A=0.37, Tscale=1.18)
    >>> comet = Comet('hartley 2', nucleus=nucleus, coma=coma)

    """

    def __init__(self, obj, nucleus=dict(), coma=dict(), kernel=None):
        if isinstance(obj, SolarSysObject):
            self.obj = obj
        else:
            self.obj = SpiceObject(obj, kernel=kernel)

        if isinstance(nucleus, dict):
            try:
                D = nucleus.pop('R') * 2
            except KeyError:
                D = nucleus.pop('D')
            Ap = nucleus.pop(Ap)
            self.nucleus = Asteroid(self, D, Ap, **nucleus)
        else:
            self.nucleus = nucleus

        if isinstance(coma, dict):
            Afrho = nucleus.pop('Afrho')
            self.coma = Coma(self, Afrho, **coma)
        else:
            self.coma = coma

    def r(self, date):
        return self.obj.r(date)
    r.__doc__ = self.obj.r.__doc__

    def v(self, date):
        return self.obj.v(date)
    v.__doc__ = self.obj.v.__doc__

    def fluxd(self, observer, date, wave, rap, reflected=True, thermal=True,
              nucleus=True, coma=True, ltt=False, unit=u.Unit('W / (m2 um)')):
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
        rap : Quantity
          The aperture radius, angular or projected distance at the
          comet.
        reflected : bool, optional
          If `True` include the reflected light model.
        thermal : bool, optional
          If `True` include the thermal emission model.
        nucleus : bool, optional
          If `True` include the nucleus.
        coma : bool, optional
          If `True` include the coma.
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

        if nucleus:
            fluxd += self.nucleus.fluxd(observer, date, wave,
                                        reflected=reflected,
                                        thermal=thermal, unit=unit)
        if coma:
            fluxd += self.coma.fluxd(observer, date, wave, rap,
                                     reflected=reflected,
                                     thermal=thermal, unit=unit)

        return fluxd
