# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
models --- MSK's model library
==============================

.. autosummary::
   :toctree: generated/

   Surface Models
   --------------

   SurfaceEmission
   NEATM


"""

__all__ = [
    'Surface',
    'NEATM'
]

import numpy as np
import astropy.units as u
from astropy.units import Quantity

class SurfaceEmission(object):
    """An abstract class for surface emission in the Solar System.

    Methods
    -------
    fluxd : Total fluxd from the object.

    Notes
    -----
    Inheriting classes should override `fluxd`.

    """

    def __init__(self):
        pass

    def fluxd(self, geom, wave, unit=None):
        pass

class NEATM(SurfaceEmission):
    """The Near Earth Asteroid Thermal Model.

    If you use this model, please reference Harris (1998, Icarus, 131,
    291-301).

    If unknown, use `eta=1.0` `epsilon=0.95` for a comet (Fernandez et
    al., Icarus, submitted), and `eta=0.96` `epsilon=0.9` (or
    `eta=0.91` with `epsilon=0.95`) for an asteroid (Mainzer et
    al. 2011, ApJ 736, 100).

    Parameters
    ----------
    radius : float or Quantity
      The radius of the asteroid (no, NOT diameter!).  [float: km]
    albedo : float
      The geometric albedo.
    eta : float
      The IR-beaming parameter.
    epsilon : float, optional
      The mean IR emissivity.
    G : float, optional
      The slope parameter of the Bowell H, G magnitude system, used to
      estimate the phase integral.
    tol : float, optional
      The relative error tolerance in the result.

    Methods
    -------
    fluxd : Total flux density from the asteroid.

    """

    def __init__(self, radius, albedo, eta, epsilon=0.95, G=0.15, tol=1e-3):

        from util import asQuantity

        self.radius = asQuantity(radius, u.km)
        self.albedo = albedo
        self.eta = eta
        self.epsilon = epsilon
        self.G = G
        self.tol = tol

    def fluxd(self, geom, wave, unit=u.Jy):
        """Flux density.

        Parameters
        ----------
        geom : dict of floats or Quantities
          A dictionary-like object with the keys 'rh' (heliocentric
          distance), 'delta' (observer-target distance), and 'phase'
          (phase angle). [floats: AU, AU, and deg]
        wave : float, array, or Quantity
          The wavelengths at which to compute the emission. [float:
          micron]
        unit : astropy Units, optional
          The return units.  Must be spectral flux density.

        Returns
        -------
        fluxd : Quantity
          The flux density from the whole asteroid.

        """

        from numpy import pi
        from scipy.integrate import quad
        from astropy.constants import sigma_sb, L_sun

        rh = asQuantity(geom['rh'], u.AU)
        delta = asQuantity(geom['delta'], u.AU).to(u.m)
        phase = asQuantity(geom['phase'], u.deg).to(u.rad)
        wave = asQuantity(wave, u.um)
        if not np.iterable(wave):
            wave = Quantity([wave], wave.unit)

        # Bond albedo = A = geometric albedo * phase integral = p * q
        #  p = 0.04 (default)
        #  G = slope parameter = 0.15 (mean val.)
        #  -> q = 0.290 + 0.684 * G = 0.3926
        #  -> A = 0.0157
        A = albedo * (0.290 + 0.684 * G)

        # insolation is greatest at the sub-solar point, call this
        # temperature T0
        Fsun = const.L_sun / u.AU.to(u.m)**2 / 4 / np.pi * u.AU**2
        T0 = (((1.0 - A) * constants.Fsun) / rh**2 / abs(eta) / epsilon /
              sigma_sb.si)**0.25

        D = 2 * self.radius.to(u.m)  # diameter

        # Integrate theta from -pi/2 to pi/2: emission is emitted from
        # the daylit hemisphere: theta = (phase - pi/2) to (phase +
        # pi/2), therfore the theta limits become [-pi/2, pi/2 -
        # phase]
        #
        # Integrate phi from -pi/2 to pi/2 (or 2 * integral from 0 to
        # pi/2)
        fluxd = np.zeros(len(wave))
        for i in range(len(wave)):
            integral = quad(_latitude_emission,
                            -pi / 2.0 + phase, pi / 2.0,
                            args=(wave[i], T0, phase),
                            epsrel=self.tol)
            fluxd[i] = (self.epsilon * (D / delta)**2 *
                        integral[0] / pi / 2.0).value # W/m^2/um

        fluxd = fluxd * u.Unit('W / (m2 um)')
        equiv = u.spectral_density(wave.unit, wave)
        fluxd = fluxd.to(unit, equivalencies=equiv)
        if len(fluxd) == 1:
            return fluxd[0]
        else:
            return fluxd

    def _point_emission(self, phi, theta, wave, T0):
        """The emission from a single point.

        phi, theta : float  [radians]

        """

        from util import Planck

        T = T0 * np.cos(phi)**0.25 * np.cos(theta)**0.25
        B = Planck(wave, T, units=u.Unit('W/(m2 sr um)'))
        return (B * pi * np.cos(phi)**2).value

    def _latitute_emission(self, theta, wave, T0, phase):
        """The emission from a single latitude.

        The function does not consider day vs. night, so make sure the
        integration limits are correctly set.

        theta : float [radians]

        """

        from scipy.integrate import quad
        from numpy import pi

        if not np.iterable(theta):
            theta = np.array([theta])

        # Quantities do not support item assignment, so fluxd needs to
        # be unitless
        fluxd = np.zeros_like(theta)
        for i in range(len(theta)):
            integral = quad(_point_emission, 0.0, pi / 2.0,
                            args=(theta[i], wave, T0),
                            epsrel=self.tol / 10.0)
            fluxd[i] = (integral[0] * np.cos(theta[i] - phase))

        i = np.isnan(fluxd)
        if any(i):
            fluxd[i] = 0.0

        return fluxd

