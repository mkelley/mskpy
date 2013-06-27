# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This package contains utilities to run the test suite.
"""

import numpy as np
import astropy.units as u
import mskpy

from .. import util

def test_phase_integral():
    from ..models.surfaces import phaseHG
    def phasef(phase):
        return phaseHG(phase, 0.15)

    pint = util.phase_integral(phasef)
    # According to Muinonen et al. (2010, Icarus 209, 542), the
    # approximation pint(G) == (0.290 + 0.684 * G) is good to 5%.  For
    # G = 0.15, this is 0.3926.
    assert np.allclose(pint, 0.384039206983)
