# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
dust --- Models for dust
========================

.. autosummary::
   :toctree: generated/

   Dust Models
   -----------
   AfrhoRadiation
   AfrhoScattered
   AfrhoThermal

   Phase functions
   ---------------
   phaseK
   phaseH
   phaseHM

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
    """Light from a coma in the Solar System parameterized by Afrho.

    Methods
    -------
    fluxd : Total flux density from the object.

    Notes
    -----
    Afrho should refer to the value at zero phase angle.

    Inheriting classes should override `fluxd`, and `__init__`
    functions.  `__init__` should take a single argument, `Afrho`, as
    a Quantity.

    As much as possible, share the same keyword arguments between
    reflected and thermal models.

    """

    def __init__(self, Afrho):
        pass

    def fluxd(self, geom, wave, unit=None):
        pass

class AfrhoScattered(AfrhoRadiation):
    """Scattered light from a coma parameterized by Afrho.

    If you use this model, please reference A'Hearn et al. (1984, AJ
    89, 579-591) as the source of the Afrho parameter.

    Parameters
    ----------
    Afrho : Quantity
      The product of albedo at zero phase, A, dust filling factor, f,
      and observer's aperture radius, rho.
    phasef : function, optional
      The phase function of the coma.  Set to `None` to use `phaseK`.

    Methods
    -------
    fluxd : Total flux density from the coma.

    """

    def __init__(self, Afrho, phasef=None):
        self.D = Afrho
        if phasef is None:
            self.phasef = phaseK
        else:
            self.phasef = phasef

    def fluxd(self, geom, wave, rap, unit=u.Unit('W / (m2 um)')):
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

        from ..calib import solar_flux

        if rap.unit.is_equivalent(u.cm):
            rho = rap.to(self.Afrho.unit)
        elif rap.unit.is_equivalent(u.arcsec):
            rho = rap.radian * geom['delta'].to(self.Afrho.unit)
        else:
            raise ValueError("rap must have angular or length units.")

        fsun = solar_flux(wave, unit=unit) / geom['rh'].au**2
        fluxd = (self.Afrho * phasef(np.abs(geom['phase'].degree)) / 
                 4.0 / geom['delta'].to(self.Afrho.unit)**2 * rho * fsun)

        return fluxd

class AfrhoThermal(AfrhoRadiation):
    """Thermal emisson from a coma parameterized by Afrho.

    If you use this model, please reference Kelley and Wooden (2009,
    Planetary and Space Science 57, 1133-1145) for the conversion from
    `Afrho` to thermal emission.

    Here, the `A` in `Afrho` can be thought of as the albedo in the
    visual wavelength range, or the mean albedo weighted by the solar
    spectrum.

    Parameters
    ----------
    Afrho : Quantity
      The product of albedo at zero phase, A, dust filling factor, f,
      and observer's aperture radius, rho.
    phasef : function, optional
      The phase function of the coma.  Set to `None` to use `phaseK`.
      The phase integral of this function will be computed.
    A : float, optional
      The mean bolometric albedo of the dust.  Gerhz and Ney (1992,
      Icarus 100, 162-186) find ~0.32 for several Oort cloud comets.
    Tscale : float, optional
      The isothermal blackbody sphere temperature scale factor to
      determine the spectral shape of the thermal emission.

    Methods
    -------
    fluxd : Total flux density from the coma.

    """

    def __init__(self, Afrho, phasef=None, A=0.32, Tscale=1.1):
        self.D = Afrho
        if phasef is None:
            self.phasef = phaseK
        else:
            self.phasef = phasef
        self.A = A
        self.Tscale = Tscale

    def fluxd(self, geom, wave, rap, unit=u.Unit('W / (m2 um)')):
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

        from ..util import phase_integal, planck

        if rap.unit.is_equivalent(u.cm):
            rho = rap.to(self.Afrho.unit)
        elif rap.unit.is_equivalent(u.arcsec):
            rho = rap.radian * geom['delta'].to(self.Afrho.unit)
        else:
            raise ValueError("rap must have angular or length units.")

        # A0 = A(phase=0)
        q = phase_integral(phasef)
        A0 = self.phasef(geom['phase'].degree) / q * self.A

        T = self.Tscale * 278 / np.sqrt(g['rh'].au) * u.K
        B = planck(wave, T, unit=unit / u.sr).value

        fluxd = ((1 - self.A) / A0 * np.pi * B * rho.centimeter
                 * self.Afrho.centimeter / geom['delta'].centimeter**2)

        return fluxd * unit

def phaseK(phase):
    """Phase function derived from Kolokolova et al. (2004, Comets II).

    The phase function of K04 is scaled to phasef(0) = 1.0.

    Parameters
    ----------
    phase : float or array
        Phase angle. [degrees]

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

    phasef = np.poly1d([  3.14105489e-08,  -7.84714255e-06,   7.34255521e-04,
                         -3.09608957e-02,   1.00920684e+00])
    return phasef(np.abs(phase))

def phaseH(phase):
    """Halley phase function from Schleicher et al. (1998).

    The Halley phase function is from Schleicher et al. (1998, Icarus
    132, 397-417).  The Comet Halley observations were at phases less
    than 70 degrees.

    Parameters
    ----------
    phase : float or array
        Phase angle. [degrees]

    Returns
    -------
    phi : float or ndarray
        The phase function.

    """

    from util import phase_integral

    phasef = np.poly1d([0.000177, -0.01807, 1])
    return phasef(np.abs(phase))

def phaseHM(phase):
    """Halley-Marcus phase function from Schleicher et al. (2011).

    The Halley phase function is first published in Schleicher and
    Bair (2011, AJ 141, 117), but only described in detail by
    Schleicher and Marcus (May 2010) at:

      http://asteroid.lowell.edu/comet/dustphase.html

      "To distinguish this curve from others, we designate this as the
      HM phase function, for the sources of the two components: Halley
      and Marcus, where the Halley curve for smaller phase angles
      comes from our previous work (Schleicher et al. 1998) while Joe
      Marcus has fit a Henyey-Greenstein function to a variety of mid-
      and large-phase angle data sets (Marcus 2007); see here for
      details. Note that we do not consider our composite curve to be
      a definitive result, but rather appropriate for performing
      first-order adjustments to dust measurements for changing phase
      angle."

    Parameters
    ----------
    phase : float or array
        Phase angle. [degrees]

    Returns
    -------
    phi : float or ndarray
        The phase function.

    """

    from scipy.interpolate import splrep, splev

    th = np.arange(181)
    ph = np.array(
        [  1.0000e+00,   9.5960e-01,   9.2170e-01,   8.8590e-01,
           8.5220e-01,   8.2050e-01,   7.9060e-01,   7.6240e-01,
           7.3580e-01,   7.1070e-01,   6.8710e-01,   6.6470e-01,
           6.4360e-01,   6.2370e-01,   6.0490e-01,   5.8720e-01,
           5.7040e-01,   5.5460e-01,   5.3960e-01,   5.2550e-01,
           5.1220e-01,   4.9960e-01,   4.8770e-01,   4.7650e-01,
           4.6590e-01,   4.5590e-01,   4.4650e-01,   4.3770e-01,
           4.2930e-01,   4.2150e-01,   4.1420e-01,   4.0730e-01,
           4.0090e-01,   3.9490e-01,   3.8930e-01,   3.8400e-01,
           3.7920e-01,   3.7470e-01,   3.7060e-01,   3.6680e-01,
           3.6340e-01,   3.6030e-01,   3.5750e-01,   3.5400e-01,
           3.5090e-01,   3.4820e-01,   3.4580e-01,   3.4380e-01,
           3.4210e-01,   3.4070e-01,   3.3970e-01,   3.3890e-01,
           3.3850e-01,   3.3830e-01,   3.3850e-01,   3.3890e-01,
           3.3960e-01,   3.4050e-01,   3.4180e-01,   3.4320e-01,
           3.4500e-01,   3.4700e-01,   3.4930e-01,   3.5180e-01,
           3.5460e-01,   3.5760e-01,   3.6090e-01,   3.6450e-01,
           3.6830e-01,   3.7240e-01,   3.7680e-01,   3.8150e-01,
           3.8650e-01,   3.9170e-01,   3.9730e-01,   4.0320e-01,
           4.0940e-01,   4.1590e-01,   4.2280e-01,   4.3000e-01,
           4.3760e-01,   4.4560e-01,   4.5400e-01,   4.6270e-01,
           4.7200e-01,   4.8160e-01,   4.9180e-01,   5.0240e-01,
           5.1360e-01,   5.2530e-01,   5.3750e-01,   5.5040e-01,
           5.6380e-01,   5.7800e-01,   5.9280e-01,   6.0840e-01,
           6.2470e-01,   6.4190e-01,   6.5990e-01,   6.7880e-01,
           6.9870e-01,   7.1960e-01,   7.4160e-01,   7.6480e-01,
           7.8920e-01,   8.1490e-01,   8.4200e-01,   8.7060e-01,
           9.0080e-01,   9.3270e-01,   9.6640e-01,   1.0021e+00,
           1.0399e+00,   1.0799e+00,   1.1223e+00,   1.1673e+00,
           1.2151e+00,   1.2659e+00,   1.3200e+00,   1.3776e+00,
           1.4389e+00,   1.5045e+00,   1.5744e+00,   1.6493e+00,
           1.7294e+00,   1.8153e+00,   1.9075e+00,   2.0066e+00,
           2.1132e+00,   2.2281e+00,   2.3521e+00,   2.4861e+00,
           2.6312e+00,   2.7884e+00,   2.9592e+00,   3.1450e+00,
           3.3474e+00,   3.5685e+00,   3.8104e+00,   4.0755e+00,
           4.3669e+00,   4.6877e+00,   5.0418e+00,   5.4336e+00,
           5.8682e+00,   6.3518e+00,   6.8912e+00,   7.4948e+00,
           8.1724e+00,   8.9355e+00,   9.7981e+00,   1.0777e+01,
           1.1891e+01,   1.3166e+01,   1.4631e+01,   1.6322e+01,
           1.8283e+01,   2.0570e+01,   2.3252e+01,   2.6418e+01,
           3.0177e+01,   3.4672e+01,   4.0086e+01,   4.6659e+01,
           5.4704e+01,   6.4637e+01,   7.7015e+01,   9.2587e+01,
           1.1237e+02,   1.3775e+02,   1.7060e+02,   2.1348e+02,
           2.6973e+02,   3.4359e+02,   4.3989e+02,   5.6292e+02,
           7.1363e+02,   8.8448e+02,   1.0533e+03,   1.1822e+03,
           1.2312e+03])

    C = splrep(th, ph)
    return splev(np.abs(phase), C)

# update module docstring
from ..util import autodoc
autodoc(globals())
del autodoc

