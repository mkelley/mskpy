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
from .ephem import SolarSysObject

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
    reflected : SurfaceRadiation, optional
      A model of the reflected light.  If `None` a `DAp` model will be
      initialized (including `**kwargs`).
    thermal : SurfaceRadiation, optional
      A model of the thermal emission.  If `None` a `NEATM` model will
      be initialized (including `**kwargs`).
    name : string, optional
      A name for this object.
    **kwargs
      Additional keywords for the default `reflected` and `thermal`
      models.

    Methods
    -------
    fluxd : Total flux density as seen by an observer.
    fit : Least-squares fit to a spectrum.

    """

    _D = None
    _Ap = None

    def __init__(self, state, D, Ap, reflected=None, thermal=None,
                 name=None, **kwargs):
        from .ephem import State
        from .models import SurfaceRadiation, DAp, NEATM

        SolarSysObject.__init__(self, state, name=name)

        assert isinstance(D, u.Quantity), "D must be a Quantity."

        if reflected is None:
            self.reflected = DAp(D, Ap, **kwargs)
        else:
            self.reflected = reflected
        assert isinstance(self.reflected, SurfaceRadiation)

        if thermal is None:
            self.thermal = NEATM(D, Ap, **kwargs)
        else:
            self.thermal = thermal
        assert isinstance(self.thermal, SurfaceRadiation)

        self.D = D
        self.Ap = Ap

    @property
    def Ap(self):
        return self._Ap

    @Ap.setter
    def Ap(self, p):
        self._Ap = p
        if self._Ap < 0:
            self._Ap = 0
        self.reflected.Ap = self._Ap
        self.thermal.Ap = self._Ap

    @property
    def D(self):
        return self._D

    @D.setter
    def D(self, d):
        assert isinstance(d, u.Quantity)
        self._D = d
        self.reflected.D = d
        self.thermal.D = d

    def fit(self, observer, date, wave, fluxd, unc, free=['D', 'Ap'],
            **kwargs):
        """Least-squares fit to a spectrum.

        Parameters
        ----------
        observer : SolarSysObject
          The observer.
        date : string, float, Astropy Time, datetime
          The epoch of the spectrum, in any format acceptable to
          `observer`.
        wave : Quantity
          The wavelengths.
        flxud : Quantity
          The spectral data (flux density) to fit.
        unc : Quantity
          The uncertainty on `fluxd`.
        free : list, optional
          The names of the free parameters.  `fit` does not handle
          units, except with `D`.
        **kwargs
          Any `scipy.optimize.leastsq` keyword.

        Returns
        -------
        fit : Asteroid
          Best-fit parameters.
        fiterr : dict
          Uncertainties.
        result : tuple
          The full output from `scipy.optimize.leastsq`.

        """

        from copy import copy
        from scipy.optimize import leastsq

        def chi(p, free, asteroid, observer, date, wave, fluxd, unc):
            for i in range(len(p)):
                if free[i] == 'D':
                    asteroid.D = p[i] * u.km
                elif free[i] == 'Ap':
                    asteroid.Ap = p[i]
                elif free[i] in asteroid.reflected.__dict__.keys():
                    asteroid.reflected.__dict__[free[i]] = p[i]
                elif free[i] in asteroid.thermal.__dict__.keys():
                    asteroid.thermal.__dict__[free[i]] = p[i]
            model = asteroid.fluxd(observer, date, wave, unit=fluxd.unit).value
            chi = (model - fluxd.value) / unc.value
            rchisq = (chi**2).sum() / (len(wave) - len(p))
            print model, fluxd, p
            return chi

        asteroid = copy(self)
        kwargs['epsfcn'] = kwargs.get('epsfcn', 1e-5)

        p = []
        for i in range(len(free)):
            if free[i] == 'D':
                p.append(asteroid.D.to(u.km).value)
            elif free[i] == 'Ap':
                p.append(asteroid.Ap)
            elif free[i] in asteroid.reflected.__dict__.keys():
                p.append(asteroid.reflected.__dict__[free[i]])
            elif free[i] in asteroid.thermal.__dict__.keys():
                p.append(asteroid.thermal.__dict__[free[i]])

        kwargs['full_output'] = True
        args = (free, asteroid, observer, date, wave, fluxd, unc)
        result = leastsq(chi, p, args, **kwargs)
    
        for i in range(len(free)):
            if free[i] == 'D':
                asteroid.D = result[0][i] * u.km
            elif free[i] == 'Ap':
                asteroid.Ap = result[0][i]
            elif free[i] in asteroid.reflected.__dict__.keys():
                asteroid.reflected.__dict__[free[i]] = result[0][i]
            elif free[i] in asteroid.thermal.__dict__.keys():
                asteroid.thermal.__dict__[free[i]] = result[0][i]

        cov = result[1]
        if cov is None:
            err = None
        else:
            err = np.sqrt(np.diagonal(cov))

        return asteroid, err, result

    def fluxd(self, observer, date, wave, reflected=True, thermal=True,
              ltt=False, unit=u.Unit('W / (m2 um)'), **kwargs):

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

