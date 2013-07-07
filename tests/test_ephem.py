# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This package contains utilities to run the test suite.
"""

from __future__ import print_function
import numpy as np
from numpy import pi
from mskpy import ephem
from mskpy.ephem import core

class TestEphemCore():
    # No real testing.  Just exercise the routines for now.
    def test_load_kernel(self):
        krn = core.find_kernel('planets')
        core.load_kernel(krn)

    def test_cal2et(self):
        core.cal2et('2000-1-1')

    def test_date2et(self):
        from astropy.time import Time
        core.date2et('2000-1-1')
        core.date2et(24500000.4)
        core.date2et(Time('2000-1-1', scale='utc'))
        core.date2et(('2000-1-1', 24500000.4, Time('2000-1-1', scale='utc')))

    def test_jd2et(self):
        core.jd2et(24500000.4)

    def test_time2et(self):
        from astropy.time import Time
        core.time2et(Time('2000-1-1', scale='utc'))

class TestEphemSolarSysObject():
    def test_r(self):
        assert np.allclose(ephem.Sun.r('2000-1-1'), [0., 0., 0.])

    def test_v(self):
        assert np.allclose(ephem.Sun.v('2000-1-1'), [0., 0., 0.])

    def test_observe(self):
        g = ephem.Earth.observe(ephem.Mars, '2000-1-1')
        print(g.bet)
        print(g.date)
        print(g.dec)
        print(g.delta)
        print(g.lam)
        print(g.lambet)
        print(g.lelong)
        print(g.obsrh)
        print(g.phase)
        print(g.ra)
        print(g.radec)
        print(g.rh)
        print(g.ro)
        print(g.rt)
        print(g.sangle)
        print(g.selong)
        print(g.signedphase)
        print(g.so)
        print(g.st)
        print(g.vangle)
        print(g.vo)
        print(g.vt)

    def test_ephemeris(self):
        eph = ephem.Earth.ephemeris(ephem.Mars, ['2000-1-1', '2010-1-1'],
                                    num=10)

    def test_orbit(self):
        print(ephem.Earth.orbit('2000-1-1'))
