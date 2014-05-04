# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
modeling --- Models for fitting data.
=====================================

.. autosummary::
   :toctree: generated/

   BlackbodyEmission
   LinearReflectance
   ScatteredSunlight

"""

from __future__ import print_function
import numpy as np
import astropy.units as u
from astropy.modeling import Parameter, FittableModel

__all__ = [
    #'ComaSED'
]

# Need astropy v0.4 for models.  BlackbodyEmission is updated, the
# other are not.

class BlackbodyEmission(FittableModel):
    """Single temperature, blackbody emission.

    Parameters
    ----------
    scale : float
      Planck function scale factor.  The Planck function is first
      normalized to its peak value.
    T : float
      Temperature (K).
    unit : astropy unit, optional
      The output flux density units.

    """
    scale = Parameter()
    T = Parameter()
    fit_deriv = None

    def __init__(self, scale, T, unit=u.Unit('W/(m2 um)'), param_dim=1,
                 **constraints):
        super(BlackbodyEmission, self).__init__(
            scale=scale, T=T, param_dim=param_dim, **constraints)

        self.unit = unit

    @staticmethod
    def eval(self, wave, scale, T):
        from ..util import planck
        f = params[0] * planck(wave, params[1], unit=self.unit)
        w0 = 0.29e4 / params[1]
        f /= np.pi * planck(w0, params[1], unit=self.unit / self.sr).value
        return f

    @format_input
    def __call__(self, wave):
        return self.eval(wave, *self.param_sets)

class LinearReflectance(ParametricModel):
    """Reflectance parameterized by a linear slope with wavelength.

    Parameters
    ----------
    R0 : float
    slope : float
      `R = R0 + slope * 10 * (wave - 0.55 um)`, where slope has
      units % per 0.1 um.

    """
    param_names = ['R0', 'slope']
    deriv = None  # compute numerical derivatives

    def __init__(self, scale, slope, param_dim=1):
        self._scale = Parameter(name='scale', val=scale, mclass=self,
                                param_dim=param_dim)
        self._slope = Parameter(name='slope', val=scale, mclass=self,
                                param_dim=param_dim)

        ParametricModel.__init__(self, self.param_names, n_inputs=1,
                                 n_outputs=1, param_dim=param_dim)
        self.linear = True

    def eval(self, wave, params):
        import astropy.units as u

        w = u.Quantity(wave, u.um)
        return params[0] + params[1] * 10 * (wave - 0.55 * u.um)

    def __call__(self, wave):
        from astropy.modeling import _convert_input, _convert_output
        wave, format = _convert_input(wave, self.param_dim)
        result = self.eval(wave, self.param_sets)
        return _convert_output(result, format)

class ScatteredSunlight(ParametricModel):
    """Scattered sunlight.

    The solar spectrum is the smoothed E490 from `calib.solar_flux`.

    Parameters
    ----------
    scale : float
      Spectrum scale factor, equal to `(rh / 1 AU)**2 * (delta / 1 m)**2`,
      where `rh` is the sun-target distance, `delta` is the
      target-observer distance.
    unit : astropy unit, optional
      The output flux density units.

    """
    param_names = ['scale']
    deriv = None  # compute numerical derivatives

    def __init__(self, scale, unit=u.Unit('W/(m2 um)'), param_dim=1):
        from scipy.interpolate import interp1d
        import astropy.units as u
        from .calib import _e490_sm

        self._scale = Parameter(name='scale', val=scale, mclass=self,
                                param_dim=param_dim)
        self.unit = unit

        ParametricModel.__init__(self, self.param_names, n_inputs=1,
                                 n_outputs=1, param_dim=param_dim)
        self.linear = True

        self._wave, self._flux = np.loadtxt(_e490_sm).T
        self._wave *= u.um
        self._flux *= u.W / u.m**2 / u.um
        if self._flux.unit != self.unit:
            self._flux = self._flux.to(self.unit,
                                       u.spectral_density(self._wave))
        self.solar_interp = interp1d(self._wave.value, self._flux.value)

    def eval(self, wave, params):
        if not np.iterable(wave):
            f = self.solar_interp([wave])[0]
        else:
            f = self.solar_interp(wave)
        return f * params[0] * self._flux.unit

    def __call__(self, wave):
        from astropy.modeling import _convert_input, _convert_output
        wave, format = _convert_input(wave, self.param_dim)
        result = self.eval(wave, self.param_sets)
        return _convert_output(result, format)

class ComaSED(ParametricModel):
    """A simple, semi-empirical coma SED, based on Afrho.

    Parameters
    ----------
    geom : dict of Quantity, or ephem.Geom
      The observing geometry via keywords rh, delta, and phase.
    rap : Quantity
      Aperture radius in length or angular units.
    Afrho : Quantity
      In units of length.
    ef2af : float, optional
      Conversion from episilon-f_therm to A-f_sca.
    Tscale : float, optional
      Temperature scale factor.
    unit : astropy Unit, optional
      The output flux density units.

    Examples
    --------

    Fit IRAC photometry of 67P/Churyumov-Gerasimenko, ignoring any
    possible gas contribution at 4.5 um.

    >>> import astropy.units as u
    >>> from astropy.modeling import fitting
    >>> from mskpy.modeling import ComaSED
    >>> wave = np.array([3.55, 4.49, 5.73, 7.87])
    >>> flux = array([1.49e-15, 3.31e-15, 1.08e-14, 2.92e-14]) # W/m2/um
    >>> geom = dict(rh=1.490 * u.au, delta=1.669 * u.au, phase=35.7 * u.deg)
    >>> rap = 7.32 * u.arcsec
    >>> coma = ComaSED(geom, rap, 1000 * u.cm)
    >>> fit = fitting.NonLinearLSQFitter(coma)
    >>> fit(wave, flux)
    >>> print (fit.model(wave) - flux) / flux
    [ 0.04580127 -0.06261374  0.00882257 -0.00058844]

    """

    param_names = ['Afrho', 'ef2af', 'Tscale']
    deriv = None  # compute numerical derivatives

    def __init__(self, geom, rap, Afrho, phasef=None, ef2af=2.0, Tscale=1.1,
                 unit=u.Unit('W/(m2 um)'), param_dim=1):
        from .models import dust

        assert isinstance(Afrho, u.Quantity)
        assert isinstance(rap, u.Quantity)

        if phasef is None:
            phasef = dust.phaseK
        self.phasef = phasef
        assert hasattr(self.phasef, '__call__')

        self._Afrho = Parameter(name='Afrho', val=Afrho.centimeter,
                                mclass=self, param_dim=param_dim)
        self._ef2af = Parameter(name='ef2af', val=ef2af, mclass=self,
                                param_dim=param_dim)
        self._Tscale = Parameter(name='Tscale', val=Tscale, mclass=self,
                                 param_dim=param_dim)
        self.geom = geom
        self.rap = rap
        self.unit = unit

        ParametricModel.__init__(self, self.param_names, n_inputs=1,
                                 n_outputs=1, param_dim=param_dim)
        self.linear = False

        self.reflected = dust.AfrhoScattered(Afrho, phasef=phasef)
        self.thermal = dust.AfrhoThermal(Afrho, ef2af=ef2af, Tscale=Tscale)

    def eval(self, wave, params):
        self.reflected.Afrho = params[0] * u.cm
        self.reflected.phasef = self.phasef
        self.thermal.Afrho = params[0] * u.cm
        self.thermal.ef2af = params[1]
        self.thermal.Tscale = params[2]
        f = self.reflected.fluxd(self.geom, wave * u.um, self.rap,
                                 unit=self.unit).value
        f += self.thermal.fluxd(self.geom, wave * u.um, self.rap,
                                unit=self.unit).value
        return f

    def __call__(self, wave):
        from astropy.modeling import _convert_input, _convert_output
        wave, format = _convert_input(wave, self.param_dim)
        result = self.eval(wave, self.param_sets)
        return _convert_output(result, format)

# update module docstring
from .util import autodoc
autodoc(globals())
del autodoc

