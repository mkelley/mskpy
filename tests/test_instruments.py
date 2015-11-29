# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This package contains utilities to run the test suite.
"""

import numpy as np
import mskpy

class TestInstruments():
    def test_irac(self, test=True):
        import astropy.units as u
        from mskpy.util import planck
        from mskpy.instruments import IRAC

        irac = IRAC()

        # Color correction standard values from the IRAC Instrument
        # Handbook.  Those calculations are good to ~1%.
        templates = dict(
            nu_2 = (lambda w: w.value**2 * u.Jy,
                    [1.0037, 1.0040, 1.0052, 1.0111], 0.015),
            nu_1 = (lambda w: w.value**1 * u.Jy,
                    [1.0, 1.0, 1.0, 1.0], 0.015),
            nu0 = (lambda w: w.value**0 * u.Jy,
                   [1.0, 1.0, 1.0, 1.0], 0.015),
            nu1 = (lambda w: w.value**-1 * u.Jy,
                   [1.0037, 1.0040, 1.0052, 1.0113], 0.015),
            nu2 = (lambda w: w.value**-2 * u.Jy,
                   [1.0111, 1.0121, 1.0155, 1.0337], 0.015),
            bb5000 = (lambda x: planck(x, 5000., unit=u.Jy / u.sr) * u.sr,
                      [1.0063, 1.0080, 1.0114, 1.0269], 0.015),
            bb2000 = (lambda x: planck(x, 2000., unit=u.Jy / u.sr) * u.sr,
                      [0.9990, 1.0015, 1.0048, 1.0163], 0.015),
            bb1500 = (lambda x: planck(x, 1500., unit=u.Jy / u.sr) * u.sr,
                      [0.9959, 0.9983, 1.0012, 1.0112], 0.015),
            bb1000 = (lambda x: planck(x, 1000., unit=u.Jy / u.sr) * u.sr,
                      [0.9933, 0.9938, 0.9952, 1.0001], 0.015),
            bb800 = (lambda x: planck(x, 800., unit=u.Jy / u.sr) * u.sr,
                     [0.9953, 0.9927, 0.9921, 0.9928], 0.015),
            bb600 = (lambda x: planck(x, 600., unit=u.Jy / u.sr) * u.sr,
                     [1.0068, 0.9961, 0.9907, 0.9839], 0.015),
            bb400 = (lambda x: planck(x, 400., unit=u.Jy / u.sr) * u.sr,
                     [1.0614, 1.0240, 1.0042, 0.9818], 0.015),
            bb200 = (lambda x: planck(x, 200., unit=u.Jy / u.sr) * u.sr,
                     [1.5138, 1.2929, 1.1717, 1.1215], 0.03)
            )

        for k, v in templates.items():
            f, K0, rtol = templates[k]
            K = irac.ccorrection(f)
            print(k, (K - K0) / K0)
            if test:
                assert np.allclose(K, K0, rtol=rtol)

