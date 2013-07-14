# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This package contains utilities to run the test suite.
"""

import numpy as np
import mskpy

class TestInstruments():
    def test_irac(self):
        import astropy.units as u
        from mskpy.util import planck
        from mskpy.instruments import IRAC

        irac = IRAC()
        K = irac.ccorrection(lambda x: planck(x, 5000, unit=u.Jy / u.sr) * u.sr)

        assert np.allclose(K, [1.0063, 1.0080, 1.0114, 1.0269])

