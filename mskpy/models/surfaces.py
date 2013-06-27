# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
surfaces --- Models for surfaces
================================

.. autosummary::
   :toctree: generated/

   Surface Models
   --------------

   SurfaceRaditation - Base class for all surfaces.
   DAp - Reflected light based on diameter and geometric albedo.
   HG - Reflected light based on IAU H-G magnitude system.
   NEATM - Near-Earth asteroid thermal model.

   Phase functions
   ---------------
   hg_phi - IAU H-G phase function
   lambertian - Lambertian sphere.

"""

from __future__ import print_function
import numpy as np
import astropy.units as u
from astropy.units import Quantity

__all__ = [
    'SurfaceRaditation',
    'DAp',
    'HG',
    'NEATM',

    'hg_phi',
    'lambertian'
]

class SurfaceRaditation(object):
    """An abstract class for light from a surface in the Solar System.

    Methods
    -------
    fluxd : Total flux density from the object.

    Notes
    -----
    Inheriting classes should override `fluxd`, and `__init__`
    functions, and should only take D and Ap as arguments (if
    possible), remaining parameters should be keywords.

    As much as possible, share the same keyword arguments between
    reflected and thermal models.

    """

    def __init__(self):
        pass

    def fluxd(self, geom, wave, unit=None):
        pass

class NEATM(SurfaceRaditation):
    """The Near Earth Asteroid Thermal Model.

    If you use this model, please reference Harris (1998, Icarus, 131,
    291-301).

    If unknown, use `eta=1.0` `epsilon=0.95` for a comet (Fernandez et
    al., Icarus, submitted), and `eta=0.96` `epsilon=0.9` (or
    `eta=0.91` with `epsilon=0.95`) for an asteroid (Mainzer et
    al. 2011, ApJ 736, 100).

    Parameters
    ----------
    D : Quantity
      The diameter of the asteroid.
    Ap : float
      The geometric albedo.
    eta : float, optional
      The IR-beaming parameter.
    epsilon : float, optional
      The mean IR emissivity.
    G : float, optional
      The slope parameter of the Bowell H, G magnitude system, used to
      estimate the phase integral when `phaseint` is `None`.
    phaseint : float, optional
      Use this phase integral instead of that from the HG system.
    tol : float, optional
      The relative error tolerance in the result.

    Attributes
    ----------
    A : float
      Bond albedo.
    R : Quantity
      Radius.

    Methods
    -------
    T0 : Sub-solar point temperature.
    fluxd : Total flux density from the asteroid.

    """

    def __init__(self, D, Ap, eta=1.0, epsilon=0.95, G=0.15,
                 phaseint=None, tol=1e-3):
        self.D = D.to(u.km)
        self.Ap = Ap
        self.eta = eta
        self.epsilon = epsilon
        self.G = G
        self.phaseint = phaseint
        self.tol = tol

    def fluxd(self, geom, wave, unit=u.Jy):
        """Flux density.

        Parameters
        ----------
        geom : dict of Quantities
          A dictionary-like object with the keys 'rh' (heliocentric
          distance), 'delta' (observer-target distance), and 'phase'
          (phase angle).
        wave : Quantity
          The wavelengths at which to compute the emission.
        unit : astropy Units, optional
          The return units.  Must be spectral flux density.

        Returns
        -------
        fluxd : Quantity
          The flux density from the whole asteroid.

        """

        from numpy import pi
        from scipy.integrate import quad

        phase = geom['phase']
        if not np.iterable(wave):
            wave = np.array([wave.value]) * wave.unit
        T0 = self.T0(geom['rh']).Kelvin
        fluxd = np.zeros(len(wave))

        # Integrate theta from -pi/2 to pi/2: emission is emitted from
        # the daylit hemisphere: theta = (phase - pi/2) to (phase +
        # pi/2), therfore the theta limits become [-pi/2, pi/2 -
        # phase]
        #
        # Integrate phi from -pi/2 to pi/2 (or 2 * integral from 0 to
        # pi/2)
        #
        # Drop some units for efficiency
        phase_r = np.abs(phase.radian)
        wave_um = wave.micrometer
        for i in range(len(wave_um)):
            fluxd[i] = quad(self._latitude_emission,
                            -pi / 2.0 + phase_r, pi / 2.0,
                            args=(wave_um[i], T0, phase_r),
                            epsrel=self.tol)[0]

        fluxd *= (self.epsilon * (self.D / geom['delta'])**2
                  / pi / 2.0).decompose() # W/m^2/Hz

        fluxd = fluxd * u.Unit('W / (m2 Hz)')
        equiv = u.spectral_density(u.um, wave.micrometer)
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
        if self.phaseint is None:
            A = self.Ap * (0.290 + 0.684 * self.G)
        else:
            A = self.Ap * self.phaseint
        return A

    @property
    def R(self):
        """Radius."""
        return self.D / 2.0

    def T0(self, rh):
        """Sub-solar point temperature.

        Parameters
        ----------
        rh : Quantity
          Heliocentric distance.

        Returns
        -------
        T0 : Quantity
          Temperature.

        """

        Fsun = 1367.567 / rh.au**2  # W / m2
        sigma = 5.670373e-08  # W / (K4 m2)
        T0 = (((1.0 - self.A) * Fsun) / abs(self.eta) / self.epsilon
              / sigma)**0.25
        return T0 * u.K

    def _point_emission(self, phi, theta, wave, T0):
        """The emission from a single point.

        phi, theta : float  [radians]
        wave : float [um]

        """

        from numpy import pi
        from ..util import planck

        T = T0 * np.cos(phi)**0.25 * np.cos(theta)**0.25
        B = planck(wave, T, unit=None) # W / (m2 sr Hz)
        return (B * pi * np.cos(phi)**2) # W / (m2 Hz)

    def _latitude_emission(self, theta, wave, T0, phase):
        """The emission from a single latitude.

        The function does not consider day vs. night, so make sure the
        integration limits are correctly set.

        theta : float [radians]
        wave : float [um]

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

class HG(SurfaceRaditation):
    """The IAU HG system for reflected light from asteroids.

    Parameters
    ----------
    H : float
      Absolute magnitude.
    G : float
      The slope parameter.
    mzp : Quantity, optional
      Flux density of magnitude 0.

    Attributes
    ----------

    Methods
    -------
    R : Radius.
    D : Diameter.
    fluxd : Total flux density.

    """

    def __init__(self, H, G, mzp=3.51e-8 * u.Unit('W / (m2 um)')):
        self.H = H
        self.G = G
        self.mzp = mzp

    def fluxd(self, geom, wave, unit=u.Unit('W / (m2 um)')):
        """Flux density.

        Parameters
        ----------
        geom : dict of Quantities
          A dictionary-like object with the keys 'rh' (heliocentric
          distance), 'delta' (observer-target distance), and 'phase'
          (phase angle).
        wave : Quantity
          The wavelengths at which to compute the emission.
        unit : astropy Units, optional
          The return units.  Must be spectral flux density.

        Returns
        -------
        fluxd : Quantity
          The flux density from the whole asteroid.

        """

        from ..calib import solar_flux

        if not np.iterable(wave):
            wave = np.array([wave.value]) * wave.unit

        rhdelta = geom['rh'].au * geom['delta'].au
        phase = geom['phase']

        mv = (self.H + 5.0 * np.log10(rhdelta)
              - 2.5 * np.log10(hg_phi(np.abs(phase.degree), self.G)))

        wave_v = np.linspace(0.5, 0.6) * u.um
        fsun_v = solar_flux(wave_v, unit=unit).value.mean()
        fsun = solar_flux(wave, unit=unit)

        fluxd = self.mzp * 10**(-0.4 * mv) * fsun / fsun_v

        if len(fluxd) == 1:
            return fluxd[0]
        else:
            return fluxd

    def D(self, Ap, Msun=-26.75):
        """Diameter.

        Parameters
        ----------
        Ap : float
          Geometric albedo.
        Msun : float, optional
          Absolute magnitude of the Sun.

        Returns
        -------
        D : Quantity
          Diameter of the asteroid.
        
        """
        D = 2 / np.sqrt(Ap) * 10**(0.2 * (Msun - self.H)) * u.au
        return D.to(u.km)

    def R(self, *args, **kwargs):
        """Radius via D()."""
        return self.D(*args, **kwargs) / 2.0

class DAv(SurfaceRaditation):
    """Reflected light from asteroids given D, Ap.

    Parameters
    ----------
    D : Quantity
      Diameter.
    Ap : float
      Geometric albedo.
    G : float, optional
      If `phasef` is None, generate an IAU HG system phase function.
    phasef : function, optional
      Phase function.  It must only take one parameter, phase angle,
      in units of degrees.

    Attributes
    ----------
    R : radius

    Methods
    -------
    H : Absolute magnitude

    """

    def __init__(self, D, Ap, G=0.15, phasef=None):
        self.D = D
        self.Ap = Ap

        if phasef is None:
            def phi_g(phase):
                return hg_phi(phase, G)
            self.phasef = phi_g
        else:        
            self.phasef = phasef

    def fluxd(self, geom, wave, unit=u.Unit('W / (m2 um)')):
        """Flux density.

        Parameters
        ----------
        geom : dict of Quantities
          A dictionary-like object with the keys 'rh' (heliocentric
          distance), 'delta' (observer-target distance), and 'phase'
          (phase angle).
        wave : Quantity
          The wavelengths at which to compute the emission.
        unit : astropy Units, optional
          The return units.  Must be spectral flux density.

        Returns
        -------
        fluxd : Quantity
          The flux density from the whole asteroid.

        """

        from numpy import pi
        from ..calib import solar_flux

        if not np.iterable(wave):
            wave = np.array([wave.value]) * wave.unit

        delta = geom['delta']
        phase = geom['phase']
        fsun = solar_flux(wave, unit=unit) / geom['rh'].au**2

        #fsca = fsun * Ap * phasef(phase) * pi * R**2 / pi / delta**2
        fsca = (fsun * self.Ap * self.phasef(np.abs(phase.degree))
                * (self.R / delta).decompose()**2)

        if unit != fsca.unit:
            fsca = fsca.to(unit, equivalencies=u.spectral_density(u.um, wave))

        return fsca

    def H(self, Msun=-26.75):
        """Absolute (V) magnitude.

        Parameters
        ----------
        Msun : float, optional
          Absolute magnitude of the Sun.

        Returns
        -------
        H : float

        """

        return 5 * np.log10(self.R.au * np.sqrt(self.Ap)) - Msun

    @property
    def R(self):
        """Radius."""
        return self.D / 2.0


def _hg_phi_i(i, phase):
    """Helper function for hg_phi.

    i: integer
    phase : float, radians

    """
    A = [3.332, 1.862]
    B = [0.631, 1.218]
    C = [0.986, 0.238]
    Phi_S = 1.0 - C[i] * np.sin(phase) / \
        (0.119 + 1.341 * np.sin(phase) - 0.754 * np.sin(phase)**2)
    Phi_L = np.exp(-A[i] * np.tan(0.5 * phase)**B[i])
    W = np.exp(-90.56 * np.tan(0.5 * phase)**2)
    return W * Phi_S + (1.0 - W) * Phi_L

def hg_phi(phase, G):
    """IAU HG system phase function.

    Parameters
    ----------
    phase : float
      Phase angle. [deg]

    Returns
    -------
      phi : float

    """
    phase = np.radians(phase)
    return ((1.0 - G) * _hg_phi_i(0, phase) + G * _hg_phi_i(1, phase))

def lambertian(phase):
    """Return the phase function from an Lambert disc computed at a
    specific phase.

    Parameters
    ----------
    phase : float or array
      The phase or phases in question. [degrees]

    Returns
    -------
    phi : float or array_like
      The ratio of light observed at the requested phase to that
      observed at phase = 0 degrees (full disc).

    Notes
    -----
    Uses the analytic form found in Brown 2004, ApJ 610, 1079.

    """
    phase = np.radians(np.abs(phase))
    return (np.sin(phase) + (pi - phase) * np.cos(phase)) / pi
