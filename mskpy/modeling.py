# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
modeling --- Models for fitting data.
=====================================

.. autosummary::
   :toctree: generated/

   ComaSED

"""

from __future__ import print_function
import numpy as np
import astropy.units as u
from astropy.modeling import Parameter, ParametricModel

__all__ = [
    'ComaSED'
]

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

