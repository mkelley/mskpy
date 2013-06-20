# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This package contains utilities to run the test suite.
"""

import numpy as np
import astropy.units as u
import mskpy

class TestModels():
    def test_neatm(self):
        from mskpy.models import NEATM

        # photometry of phaeton from Green et al. (1985, MNRaS 214,
        # 29-36) and modeled in the Harris (1998) NEATM paper.

        wave = [3.53, 3.73, 4.73, 8.7, 9.7,
                10.3, 10.6, 11.6, 12.5, 19.2]
        wave *= u.um
        fluxd = 10**np.array([-14.288, -14.148, -13.584, -12.993, -12.995,
                              -13.027, -13.012, -13.019, -13.070, -13.359])
        fluxd *= u.Unit('W / (m2 um)')
        fderr = np.array([0.05, 0.07, 0.08, 0.04, 0.06,
                          0.06, 0.06, 0.05, 0.06, 0.06])
        fderr *= fluxd / 1.0857

        phaethon = NEATM(5.13 / 2.0, 0.11, 1.6, epsilon=0.9)
        geom = dict(rh=1.131, delta=0.246, phase=48.3)
        model = phaethon.fluxd(geom, wave, unit=u.Unit('W / (m2 um)'))

        rchisq = sum(((fluxd - model) / fderr).value**2) / (10 - 3)
        stop
