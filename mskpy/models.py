# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
models --- MSK's model library
==============================

.. autosummary::
   :toctree: generated/

   Surface Models
   --------------

   SurfaceEmission
   HG
   NEATM


"""

__all__ = [
    'SurfaceEmission'
    'HG',
    'NEATM',
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
    pv : float
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

    Attributes
    ----------
    A : float
      Bond albedo.
    D : Quantity
      Diameter.

    Methods
    -------
    T0 : Sub-solar point temperature.
    fluxd : Total flux density from the asteroid.

    """

    def __init__(self, radius, pv, eta, epsilon=0.95, G=0.15, tol=1e-3):
        from util import asQuantity

        self.radius = asQuantity(radius, u.km)
        self.pv = pv
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

        from .util import asQuantity

        rh = asQuantity(geom['rh'], u.AU).value
        delta = asQuantity(geom['delta'], u.AU).to(u.m).value
        phase = asQuantity(geom['phase'], u.deg).to(u.rad).value
        wave = asQuantity(wave, u.um).value
        if not np.iterable(wave):
            wave = np.array([wave])
        T0 = self.T0(rh).value

        # Integrate theta from -pi/2 to pi/2: emission is emitted from
        # the daylit hemisphere: theta = (phase - pi/2) to (phase +
        # pi/2), therfore the theta limits become [-pi/2, pi/2 -
        # phase]
        #
        # Integrate phi from -pi/2 to pi/2 (or 2 * integral from 0 to
        # pi/2)
        fluxd = np.zeros(len(wave))
        for i in range(len(wave)):
            integral = quad(self._latitude_emission,
                            -pi / 2.0 + phase, pi / 2.0,
                            args=(wave[i], T0, phase),
                            epsrel=self.tol)
            fluxd[i] = (self.epsilon * (self.D.value * 1e3 / delta)**2 *
                        integral[0] / pi / 2.0) # W/m^2/Hz

        fluxd = fluxd * u.Unit('W / (m2 Hz)')
        equiv = u.spectral_density(u.um, wave)
        fluxd = fluxd.to(unit, equivalencies=equiv)
        if len(fluxd) == 1:
            return fluxd[0]
        else:
            return fluxd

    @property
    def A(self):
        """Bond albedo.

        A = geometric albedo * phase integral = p * q
        p = 0.04 (default)
        G = slope parameter = 0.15 (mean val.)
        -> q = 0.290 + 0.684 * G = 0.3926
        -> A = 0.0157

        """
        return self.pv * (0.290 + 0.684 * self.G)

    @property
    def D(self):
        """Diameter."""
        return 2 * self.radius  # diameter

    def T0(self, rh):
        """Sub-solar point temperature.

        Parameters
        ----------
        rh : float or Quantity
          Heliocentric distance. [float: AU]

        Returns
        -------
        T0 : Quantity
          Temperature.

        """

        from .util import asQuantity

        #Fsun = L_sun / u.AU.to(u.m)**2 / 4 / np.pi * u.AU**2
        #Fsun = 1367.567 * u.Unit('W AU2 / m2')

        if isinstance(rh, Quantity):
            rh = rh.to(u.AU).value

        Fsun = 1367.567 # W AU2 / m2
        sigma = 5.670373e-08 # W / (K4 m2)
        T0 = (((1.0 - self.A) * Fsun) / rh**2 / abs(self.eta)
              / self.epsilon / sigma)**0.25
        return T0 * u.K

    def _point_emission(self, phi, theta, wave, T0):
        """The emission from a single point.

        phi, theta : float  [radians]

        """

        from numpy import pi
        from util import Planck

        T = T0 * np.cos(phi)**0.25 * np.cos(theta)**0.25
        B = Planck(wave, T, unit=None) # W / (m2 sr Hz)
        return (B * pi * np.cos(phi)**2) # W / (m2 Hz)

    def _latitude_emission(self, theta, wave, T0, phase):
        """The emission from a single latitude.

        The function does not consider day vs. night, so make sure the
        integration limits are correctly set.

        theta : float [radians]

        """

        from scipy.integrate import quad
        from numpy import pi

        if not np.iterable(theta):
            theta = np.array([theta])

        fluxd = np.zeros_like(theta)
        for i in range(len(theta)):
            integral = quad(self._point_emission, 0.0, pi / 2.0,
                            args=(theta[i], wave, T0),
                            epsrel=self.tol / 10.0)
            fluxd[i] = (integral[0] * np.cos(theta[i] - phase))

        i = np.isnan(fluxd)
        if any(i):
            fluxd[i] = 0.0
        return fluxd

class HG(SurfaceEmission):
    """The IAU HG system for reflected light from asteroids.

    Parameters
    ----------
    H : float
      Absolute magnitude.
    G : float
      The slope parameter.
    mzp : float or Quantity, optional
      Flux density of magnitude 0. [float: W/m2/um]

    Attributes
    ----------

    Methods
    -------
    R : Radius.
    D : Diameter.
    Phi : Phase function.
    fluxd : Total flux density.

    """

    def __init__(self, H, G, mzp=3.51e-8):
        self.H = H
        self.G = G
        if isinstance(mzp, Quantity):
            self.mzp = mzp
        else:
            self.mzp = mzp * u.Unit('W/(m2 um)')

    def _Phi_i(self, i, phase):
        # i = integer, phase = float, radians
        A = [3.332, 1.862]
        B = [0.631, 1.218]
        C = [0.986, 0.238]
        Phi_S = 1.0 - C[i] * np.sin(phase) / \
            (0.119 + 1.341 * np.sin(phase) - 0.754 * np.sin(phase)**2)
        Phi_L = np.exp(-A[i] * np.tan(0.5 * phase)**B[i])
        W = np.exp(-90.56 * np.tan(0.5 * phase)**2)
        return W * Phi_S + (1.0 - W) * Phi_L

    def Phi(self, phase):
        """Phase function.

        Parameters
        ----------
        phase : float
          Phase angle. [radians]

        Returns
        -------
        phi : float

        """
        return ((1.0 - self.G) * self._Phi_i(0, phase)
                + self.G * self._Phi_i(1, phase))

    def fluxd(self, geom, wave, unit=u.Unit('W / (m2 um)')):
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

        from .util import asQuantity
        from .calib import solar_flux

        rh = asQuantity(geom['rh'], u.AU).value
        delta = asQuantity(geom['delta'], u.AU).value
        phase = abs(asQuantity(geom['phase'], u.deg).to(u.rad).value)
        wave = asQuantity(wave, u.um).value
        if not np.iterable(wave):
            wave = np.array([wave])

        mv = (self.H + 5.0 * np.log10(rh * delta)
              - 2.5 * np.log10(self.Phi(phase)))

        wave_v = np.linspace(0.5, 0.6)
        fsun_v = solar_flux(wave_v, unit=unit).value.mean()

        fluxd = (self.mzp * 10**(-0.4 * mv) * solar_flux(wave, unit=unit)
                 / fsun_v)

        if len(fluxd) == 1:
            return fluxd[0]
        else:
            return fluxd

    def D(self, pv, Msun=-26.75):
        """Diameter.

        Parameters
        ----------
        pv : float
          Geometric albedo.
        Msun : float, optional
          Absolute magnitude of the Sun.

        Returns
        -------
        D : Quantity
          Diameter of the asteroid.
        
        """
        import astropy.constants as const
        return (2 * const.au.to(u.km).value / np.sqrt(pv)
                * 10**(0.2 * (Msun - self.H)))

    def R(self, *args, **kwargs):
        """Radius via D()."""
        return self.D(*args, **kwargs) / 2.0
