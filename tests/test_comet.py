# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This package contains utilities to run the test suite.
"""

from __future__ import print_function
import numpy as np
from numpy import pi
import astropy.units as u
from mskpy import Coma, Comet, Earth, SpiceState
from mskpy.models import dust

class TestComet():
    # No real testing.  Just exercise the routines for now.
    def test_coma(self):
        mars = SpiceState('mars', kernel='planets.bsp')
        c = Coma(mars, 1 * u.cm, k=-2)

        r = dict(phasef=dust.phaseH)
        t = dict(phasef=dust.phaseH, A=0.25, Tscale=1.05)
        c = Coma(mars, 1 * u.cm, reflected=r)
        c = Coma(mars, 1 * u.cm, thermal=t)
        c = Coma(mars, 1 * u.cm, reflected=r, thermal=t)

        r = dust.AfrhoScattered(1 * u.cm)
        t = dust.AfrhoThermal(1 * u.cm)
        c = Coma(mars, 1 * u.cm, reflected=r)
        c = Coma(mars, 1 * u.cm, thermal=t)
        c = Coma(mars, 1 * u.cm, reflected=r, thermal=t)

    def test_comet(self):
        from mskpy import Asteroid

        mars = SpiceState('mars', kernel='planets.bsp')
        nucleus = Asteroid(mars, 0.6 * u.km, 0.04, eta=1.0, epsilon=0.95)
        coma = Coma(mars, 302 * u.cm, S=0.0, A=0.37, Tscale=1.18)
        comet = Comet(mars, nucleus=nucleus, coma=coma)

        nucleus = dict(R=0.6 * u.km, Ap=0.04, eta=1.0, epsilon=0.95)
        coma = dict(Afrho1=302 * u.cm, S=0.0, A=0.37, Tscale=1.18)
        comet = Comet(mars, nucleus=nucleus, coma=coma)
