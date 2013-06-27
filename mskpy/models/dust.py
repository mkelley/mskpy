# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
dust --- Models for dust
========================

.. autosummary::
   :toctree: generated/

   Dust Models
   -----------

   AfrhoRadiation - Base class for radiation from a comet based on Afrho.
   AfrhoScattered - Scattered light from a comet.
   AfrhoThermal - Thermal emission from a comet.

   Phase functions
   ---------------
   phaseK - Phase function based on Kolokolova et al. (2004, Comets II).
   phaseH - Halley phase function based on Schleicher et al. (1998).
   phaseHM - (Halley-Marcus) phase function based on Schleicher et al. (2011).

"""

from __future__ import print_function
import numpy as np
import astropy.units as u
from astropy.units import Quantity

__all__ = [
    'AfrhoRadiation',
    'AfrhoScattered',
    'AfrhoThermal',

    'phaseK',
    'phaseH',
    'phaseHM'
]

class AfrhoRadiation(object):
    """An abstract class for light from a coma in the Solar System.

    Methods
    -------
    fluxd : Total flux density from the object.

    Notes
    -----
    Inheriting classes should override `fluxd`, and `__init__`
    functions.  `__init__` should take a single argument, `Afrho`.

    As much as possible, share the same keyword arguments between
    reflected and thermal models.

    """

    def __init__(self):
        pass

    def fluxd(self, geom, wave, unit=None):
        pass

class AfrhoScattered(SurfaceEmission):
    """Scattered light from a coma parameterized by Afrho.

    If you use this model, please reference A'Hearn et al. (1984, AJ
    89, 579-591) as the source of the Afrho parameter.

    Parameters
    ----------
    Afrho : Quantity
      The product of albedo at zero phase, A, dust filling factor, f,
      and observer's aperture radius, rho.
    phasef : function, optional
      The phase function of the coma.

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

    def __init__(self, Afrho, phasef=phaseK):
        self.D = Afrho
        self.phasef = phasef

    def fluxd(self, geom, wave, rap, unit=u.unit('W / (m2 um)')):
        """Flux density.

        Parameters
        ----------
        geom : dict of Quantities
          A dictionary-like object with the keys 'rh' (heliocentric
          distance), 'delta' (observer-target distance), and 'phase'
          (phase angle).
        wave : Quantity
          The wavelengths at which to compute the emission.
        rap : Quantity
          The aperture radius, angular or projected distance at the
          comet.
        unit : astropy Units, optional
          The return units.  Must be spectral flux density.

        Returns
        -------
        fluxd : Quantity
          The flux density from the coma.

        Raises
        ------
        ValueError : If `rap` has incorrect units.

        """

        from .calib import solar_flux

        if rap.unit.is_equivalent(u.cm):
            rho = rap.to(self.Afrho.unit)
        elif rap.unit.is_equivalent(u.arcsec):
            rho = rap.radian * g['delta'].to(self.Afrho.unit)
        else:
            raise ValueError("rap must have angular or length units.")

        fsun = solar_flux(wave, unit=unit) / g['rh'].au**2
        fluxd = (self.Afrho * phasef(np.abs(g['phase'].degree)) / 
                 4.0 / g['delta'].to(self.Afrho.unit)**2 * rho * fsun)

        return fluxd

def phaseK(phase, mean=None):
    """Phase function derived from Kolokolova et al. (2004, Comets II).

    The phase function of K04 is scaled to phasef(0) = 1.0.

    Parameters
    ----------
    phase : float or array
        Phase angle. [degrees]
    mean : float, optional
      Normalize `phaseK`, weighted by `sin(phase)`, to this average
      value.

    Returns
    -------
    phi : float or ndarray
        The phase function.

    Notes
    -----
    To estimate the phase function, I fit a polynomial function to the
    solid line of Kolokolova et al. (2004, Comets II):

      a = array([0.27, 0.21, 0.17, 0.15, 0.14, 0.135, 0.135, 0.135, 0.15,
           0.175, 0.225, 0.3, 0.43, 0.62, 0.775])
      b = array([0.0,  10,   20,   30,   40,   60,    70,   80,   100,
           110,   120,   130, 140,  150,  156])
      fit = poly1d(polyfit(b, a / 0.27, 4))
      plot(b, a, 'o')
      plot(b, fit(b) * min(a), 'r-')

    """

    from scipy.integrate import quad

    phasef = np.poly1d([  3.14105489e-08,  -7.84714255e-06,   7.34255521e-04,
                         -3.09608957e-02,   1.00920684e+00])
    
    if mean is not None:
        scale = quad(lambda x: phasef(x) * np.sin(np.radians(x)),
                     0, 180)[0] / (360 / np.pi) / mean
    else:
        scale = 1.0

    return phasef(np.abs(phase)) / scale
