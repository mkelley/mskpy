# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This package contains utilities to run the test suite.
"""

import numpy as np
import mskpy

class TestModeling():
    def test_BlackbodyEmission(self):
        import astropy.units as u
        from astropy.modeling import fitting
        from mskpy.modeling import BlackbodyEmission
        from mskpy.util import planck

        wave = np.logspace(-0.5, 2) * u.um
        flux = planck(wave, 300) * np.pi * u.sr

        bb = BlackbodyEmission(1.0, 300.)
        lsq = fitting.NonLinearLSQFitter()
        fit = lsq(bb, wave, flux)
        
        assert np.allclose(flux, fit(wave))

#    def test_Dpv(self):
#        from mskpy.models import Dpv


#        phaethon = NEATM(5.13 * u.km, 0.11, 1.6, epsilon=0.9)
#        geom = dict(rh=1.131 * u.au, delta=0.246 * u.au, phase=48.3 * u.deg)
#        model = phaethon.fluxd(geom, wave, unit=u.Unit('W / (m2 um)')).value
#
#        # results from mskpy1 are within 1%, they are probably so
#        # large because of the formulae using different units,
#        # constants
#        #mskpy1_model = np.array(
#        #    [  3.95568129e-15,   5.97434772e-15,   2.40411341e-14,
#        #       9.93098754e-14,   1.00969494e-13,   9.95610099e-14,
#        #       9.83586508e-14,   9.26712814e-14,   8.62076514e-14,
#        #       4.11747414e-14])
#        standard  = np.array(
#            [  3.97957168e-15,   6.00855442e-15,   2.41505664e-14,
#               9.95628623e-14,   1.01202048e-13,   9.97780671e-14,
#               9.85675688e-14,   9.28528197e-14,   8.63657659e-14,
#               4.12279675e-14])
#
#        assert np.allclose(standard / model, np.ones_like(model))
