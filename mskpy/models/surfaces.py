# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
surfaces --- Models for surfaces
================================

.. autosummary::
   :toctree: generated/

   Surface Models
   --------------
   SurfaceRadiation
   DAp
   DApColor
   HG
   NEATM

   Phase functions
   ---------------
   phaseHG
   lambertian

   Convenience functions
   ---------------------
   neatm

"""

import numpy as np
import astropy.units as u
from astropy.units import Quantity

__all__ = [
    'SurfaceRadiation',
    'DAp',
    'DApColor',
    'HG',
    'NEATM',

    'phaseHG',
    'lambertian',

    'neatm'
]

class SurfaceRadiation(object):
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

    def __init__(self, **kwargs):
        pass

    def fluxd(self, geom, wave, unit=None):
        pass

class NEATM(SurfaceRadiation):
    """The Near Earth Asteroid Thermal Model.

    If you use this model, please reference Harris (1998, Icarus, 131,
    291-301).

    If unknown, use `eta=1.03` `epsilon=0.95` for a comet (Fernandez
    et al., 2013, Icarus 226, 1138-1170), and `eta=0.96` `epsilon=0.9`
    (or `eta=0.91` with `epsilon=0.95`) for an asteroid (Mainzer et
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
                 phaseint=None, tol=1e-3, **kwargs):
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
        T0 = self.T0(geom['rh']).to(u.Kelvin).value
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
        phase_r = np.abs(phase.to(u.rad).value)
        wave_um = wave.to(u.um).value
        for i in range(len(wave_um)):
            fluxd[i] = quad(self._latitude_emission,
                            -pi / 2.0 + phase_r, pi / 2.0,
                            args=(wave_um[i], T0, phase_r),
                            epsrel=self.tol)[0]

        fluxd *= (self.epsilon * (self.D / geom['delta'])**2
                  / pi / 2.0).decompose().value # W/m^2/Hz

        fluxd = fluxd * u.Unit('W / (m2 Hz)')
        equiv = u.spectral_density(u.um, wave.to(u.um).value)
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

        Fsun = 1367.567 / rh.to(u.au).value**2  # W / m2
        sigma = 5.670373e-08  # W / (K4 m2)
        T0 = (((1.0 - self.A) * Fsun) / abs(self.eta) / self.epsilon
              / sigma)**0.25
        return T0 * u.K

    def fit(self, g, wave, fluxd, unc, **kwargs):
        """Least-squares fit to a spectrum, varying `D` and `eta`.

        Uses the object's current state as the initial parameter set.

        Parameters
        ----------
        g : dict-like
          A dictionary-like object with the keys 'rh' (heliocentric
          distance), 'delta' (observer-target distance), and 'phase'
          (phase angle) as Quantities.
        wave : Quantity
          The spectrum wavelengths.
        fluxd : Quantity
          The spectrum flux density.
        unc : Quantity
          The uncertainties on `fluxd`.
        **kwargs
          Any keyword arguments for `scipy.optimize.leastsq`.

        Returns
        -------
        fit : NEATM
          Best-fit parameters.
        fiterr : tuple
          `(D, eta)` fit errors (assuming independent variables) or
          `None` if they cannot be computed.
        result : tuple
          The full output from `scipy.optimize.leastsq`.

        """

        from copy import copy
        from scipy.optimize import leastsq

        def chi(p, neatm, g, wave, fluxd, unc):
            neatm.D = u.Quantity(abs(p[0]), u.km)
            neatm.eta = abs(p[1])
            model = neatm.fluxd(g, wave, unit=fluxd.unit).value
            chi = (model - fluxd.value) / unc.value
            rchisq = (chi**2).sum() / (len(wave) - 2.0)
            print(neatm.D, neatm.eta, rchisq)
            return chi

        neatm = copy(self)
        args = (neatm, g, wave, fluxd, unc)
        kwargs['epsfcn'] = kwargs.get('epsfcn', 1e-5)

        kwargs['full_output'] = True
        result = leastsq(chi, (self.D.value, self.eta), args, **kwargs)

        neatm.D = u.Quantity(result[0][0], u.km)
        neatm.eta = result[0][1]
        cov = result[1]
        if cov is None:
            err = None
        else:
            err = np.sqrt(np.diagonal(cov))

        return neatm, err, result

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

class HG(SurfaceRadiation):
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

    def __init__(self, H, G, mzp=3.51e-8 * u.Unit('W / (m2 um)'), **kwargs):
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

        rhdelta = geom['rh'].to(u.au).value * geom['delta'].to(u.au).value
        phase = geom['phase']

        mv = (self.H + 5.0 * np.log10(rhdelta)
              - 2.5 * np.log10(phaseHG(np.abs(phase.to(u.deg).value), self.G)))

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

class DAp(SurfaceRadiation):
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

    def __init__(self, D, Ap, G=0.15, phasef=None, **kwargs):
        self.D = D
        self.Ap = Ap

        if phasef is None:
            def phi_g(phase):
                return phaseHG(phase, G)
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
        fsun = solar_flux(wave, unit=unit) / geom['rh'].to(u.au).value**2

        #fsca = fsun * Ap * phasef(phase) * pi * R**2 / pi / delta**2
        fsca = (fsun * self.Ap * self.phasef(np.abs(phase.to(u.deg).value))
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

        return 5 * np.log10(self.R.to(u.au).value * np.sqrt(self.Ap)) - Msun

    @property
    def R(self):
        """Radius."""
        return self.D / 2.0

class DApColor(DAp):
    """Reflected light from asteroids given D, Ap, and a color.

    Parameters
    ----------
    D : Quantity
      Diameter.
    Ap : float
      Geometric albedo at 0.55 um.
    S : float
      Spectral slope for reflected light:
        `refl = 1 + (lambda - lambda0) * S / 10`
      where `S` has units of % per 0.1 um, `lambda` has units of um,
      and `lambda0` is 0.55 um.  `R` is limited to `0 <= refl <=
      refl_max`.
    refl_max : float
      Use this value as the maximum reflectance.
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

    def __init__(self, D, Ap, S, refl_max=2.5, **kwargs):
        self.S = S
        self.refl_max = refl_max
        DAp.__init__(self, D, Ap, **kwargs)

    def fluxd(self, geom, wave, unit=u.Unit('W / (m2 um)')):
        from numpy import pi
        from ..calib import solar_flux

        if not np.iterable(wave):
            wave = np.array([wave.value]) * wave.unit

        delta = geom['delta']
        phase = geom['phase']
        fsun = solar_flux(wave, unit=unit) / geom['rh'].to(u.au).value**2

        refl = 1 + (wave - 0.55 * u.um).value * self.S / 10.
        if np.any(refl > self.refl_max):
            refl[refl > self.refl_max] = self.refl_max
        if np.any(refl < 0.0):
            refl[refl < 0.0] = 0.0

        #fsca = fsun * Ap * phasef(phase) * pi * R**2 / pi / delta**2
        fsca = (fsun * self.Ap * refl
                * self.phasef(np.abs(phase.to(u.deg).value))
                * (self.R / delta).decompose()**2)

        if unit != fsca.unit:
            fsca = fsca.to(unit, equivalencies=u.spectral_density(u.um, wave))

        return fsca

    fluxd.__doc__ = DAp.__doc__

def _phaseHG_i(i, phase):
    """Helper function for phaseHG.

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

def phaseHG(phase, G):
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
    return ((1.0 - G) * _phaseHG_i(0, phase) + G * _phaseHG_i(1, phase))

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

def neatm(D, Ap, geom, wave, unit=u.Jy, **kwargs):
    """Convenience function for NEATM.

    Parameters
    ----------
    D : Quantity
      Diameter.
    Ap : float
      Geometric albedo.
    geom : dict of Quantities
      Geometry of observation: rh, Delta, phase.
    wave : Quantity
      Wavelengths at which to evaluate the model.
    unit : astropy Units, optional
      The return units.  Must be spectral flux density.
    **kwargs
      Any `models.NEATM` keyword argument.

    Returns
    -------
    fluxd : Quantity
      The flux density from the whole asteroid.

    """

    return NEATM(D, Ap, **kwargs).fluxd(geom, wave, unit=unit)

# update module docstring
from ..util import autodoc
autodoc(globals())
del autodoc

