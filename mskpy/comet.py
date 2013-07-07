# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
comet --- Comets!
=================

.. autosummary::
   :toctree: generated/

   Classes
   -------
   Coma
   Comet

"""

__all__ = [
    'Coma',
    'Comet'
]

import numpy as np
import astropy.units as u
from astropy.time import Time

from .ephem import SolarSysObject, State
from .asteroid import Asteroid
from .models import AfrhoRadiation, AfrhoScattered, AfrhoThermal

class Coma(SolarSysObject):
    """A comet coma.

    Parameters
    ----------
    state : State
      The location of the coma.
    Afrho1 : Quantity
      Afrho of the coma at 1 AU.
    k : float, optional
      The logarithmic slope of the dust production rate's dependence
      on rh.
    reflected : dict or AfrhoRadiation, optional
      A model for light scattered by dust or a dictionary of keywords
      to pass to `AfrhoScattered`.
    thermal : dict or AfrhoRadiation, optional
      A model for thermal emission from dust or a dictionary of
      keywords to pass to `AfrhoThermal`.

    """
    def __init__(self, state, Afrho1, k=-2, reflected=dict(), thermal=dict()):
        assert isinstance(state, State), "state must be a State"
        assert isinstance(Afrho1, u.Quantity), "Afrho1 must be a Quantity"

        self.state = state
        self.Afrho1 = Afrho1
        self.k = k

        if isinstance(reflected, AfrhoRadiation):
            self.reflected = reflected
        else:
            self.reflected = AfrhoScattered(
                1 * self.Afrho1.unit, **reflected)

        if isinstance(thermal, AfrhoRadiation):
            self.thermal = thermal
        else:
            self.thermal = AfrhoThermal(
                1 * self.Afrho1.unit, **thermal)

    def fluxd(self, observer, date, wave, rap=1.0 * u.arcsec,
              reflected=True, thermal=True, ltt=False,
              unit=u.Unit('W / (m2 um)')):
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
        rap : Quantity, optional
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

        assert isinstance(observer, SolarSysObject), "observer must be a SolarSysObject"
        assert isinstance(wave, u.Quantity), "wave must be a Quantity"

        fluxd = np.zeros(np.size(wave.value)) * unit
        if self.Afrho1.value <= 0:
            return fluxd

        g = observer.observe(self, date, ltt=ltt)

        if reflected:
            f = self.reflected.fluxd(g, wave, rap, unit=unit)
            fluxd += f * self.Afrho1.value * g['rh'].au**self.k
        if thermal:
            f = self.thermal.fluxd(g, wave, rap, unit=unit)
            fluxd += f * self.Afrho1.value * g['rh'].au**self.k

        return fluxd

class Comet(SolarSysObject):
    """A comet.

    Parameters
    ----------
    state : State
      The location of the comet.
    nucleus : dict or Asteroid
      The nucleus of the comet as an `Asteroid` instance, or a
      dictionary of keywords to initialize a new `Asteroid`, including
      `R` or `D` and `Ap`.  Yes, `R` is allowed, this is a comet
      nucleus, afterall.
    coma : dict or Coma
      The coma of the comet as a `Coma` instance or a dictionary of
      keywords to initialize a new `Coma`, including `Afrho`.

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
    >>> from mskpy import Asteroid, Coma, Comet, SpiceState
    >>> state = SpiceState('hartley 2')
    >>> nucleus = Asteroid(state, 0.6 * u.km, 0.04, eta=1.0, epsilon=0.95)
    >>> coma = Coma(state, 302 * u.cm, S=0.0, A=0.37, Tscale=1.18)
    >>> comet = Comet(state, nucleus=nucleus, coma=coma)

    Create a comet with keyword arguments.

    >>> import astropy.units as u
    >>> from mskpy import Comet, SpiceState
    >>> nucleus = dict(R=0.6 * u.km, Ap=0.04, eta=1.0, epsilon=0.95)
    >>> coma = dict(Afrho1=302 * u.cm, S=0.0, A=0.37, Tscale=1.18)
    >>> comet = Comet(SpiceState('hartley 2'), nucleus=nucleus, coma=coma)

    """

    def __init__(self, state, nucleus=dict(), coma=dict()):
        assert isinstance(state, State), "state must be a State."
        self.state = state

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
            Afrho1 = nucleus.pop('Afrho1')
            self.coma = Coma(self, Afrho1, **coma)
        else:
            self.coma = coma

    def fluxd(self, observer, date, wave, rap=1.0 * u.arcsec,
              reflected=True, thermal=True, nucleus=True, coma=True,
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
        rap : Quantity, optional
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

# update module docstring
from .util import autodoc
autodoc(globals())
del autodoc

