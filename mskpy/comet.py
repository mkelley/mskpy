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
    reflected : AfrhoRadiation, optional
      A model for light scattered by dust or `None` to pass `kwargs`
      to `AfrhoScattered`.
    thermal : AfrhoRadiation, optional
      A model for thermal emission from dust or `None` to pass
      `kwargs` `AfrhoThermal`.
    **kwargs
      Keywords for `AfrhoRadiation` and/or `AfrhoThermal`, when
      `reflected` and/or `thermal` are `None`, respectively.

    """
    _Afrho1 = None

    def __init__(self, state, Afrho1, k=-2, reflected=None, thermal=None,
                 **kwargs):
        assert isinstance(state, State), "state must be a State"
        assert isinstance(Afrho1, u.Quantity), "Afrho1 must be a Quantity"

        self.state = state
        self.k = k

        if reflected is None:
            self.reflected = AfrhoScattered(1 * Afrho1.unit, **kwargs)
        else:
            self.reflected = reflected
        assert isinstance(self.reflected, AfrhoRadiation)

        if thermal is None:
            self.thermal = AfrhoThermal(1 * Afrho1.unit, **kwargs)
        else:
            self.thermal = thermal
        assert isinstance(self.thermal, AfrhoRadiation)

        # Afrho1 property will also update self.thermal, self.scattered
        self.Afrho1 = Afrho1

    @property
    def Afrho1(self):
        return self._Afrho1

    @Afrho1.setter
    def Afrho1(self, a):
        assert isinstance(a, u.Quantity)
        self._Afrho1 = a
        self.thermal.Afrho = 1 * a.unit
        self.reflected.Afrho = 1 * a.unit

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

        assert isinstance(observer, SolarSysObject)
        assert isinstance(wave, u.Quantity)

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
    Afrho1 : Quantity
      Afrho at 1 AU.
    R : Quantity
      Nucleus radius.
    Ap : float, optional
      Geometric albedo of the nucleus.
    nucleus : dict or Asteroid
      The nucleus of the comet as an `Asteroid` instance, or a
      dictionary of keywords to initialize a new `Asteroid`.
    coma : dict or Coma
      The coma of the comet as a `Coma` instance or a dictionary of
      keywords to initialize a new `Coma`.

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
    >>> Afrho1 = 302 * u.cm
    >>> R = 0.6 * u.km
    >>> Ap = 0.04
    >>> nucleus = Asteroid(state, R * 2, Ap, eta=1.0, epsilon=0.95)
    >>> coma = Coma(state, Afrho1, S=0.0, A=0.37, Tscale=1.18)
    >>> comet = Comet(state, Afrho1, R, Ap, nucleus=nucleus, coma=coma)

    Create a comet with keyword arguments.

    >>> import astropy.units as u
    >>> from mskpy import Comet, SpiceState
    >>> Afrho1 = 302 * u.cm
    >>> R = 0.6 * u.km
    >>> nucleus = dict(eta=1.0, epsilon=0.95)
    >>> coma = dict(S=0.0, A=0.37, Tscale=1.18)
    >>> comet = Comet(SpiceState('hartley 2'), Afrho1, R, nucleus=nucleus, coma=coma)

    """

    def __init__(self, state, Afrho1, R, Ap=0.04, nucleus=dict(), coma=dict()):
        assert isinstance(state, State)
        self.state = state

        if isinstance(nucleus, dict):
            self.nucleus = Asteroid(self.state, 2 * R, Ap, **nucleus)
        else:
            self.nucleus = nucleus

        if isinstance(coma, dict):
            self.coma = Coma(self.state, Afrho1, **coma)
        else:
            self.coma = coma

        # these properties will update nucleus and coma
        self.Afrho1 = Afrho1
        self.R = R
        self.Ap = Ap

    @property
    def R(self):
        return self.nucleus.D / 2.0

    @R.setter
    def R(self, r):
        self.nucleus.D = 2 * r

    @property
    def Ap(self):
        return self.nucleus.Ap

    @Ap.setter
    def Ap(self, p):
        self.nucleus.Ap = p

    @property
    def Afrho1(self):
        return self.coma.Afrho1

    @Afrho1.setter
    def Afrho1(self, a):
        self.coma.Afrho1 = a

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

