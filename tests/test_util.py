# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This package contains utilities to run the test suite.
"""

import numpy as np
from numpy import pi
from mskpy import util

class TestUtilMathmatical():
    def test_haversine(self):
        th = np.arange(-pi, pi)
        assert np.allclose(util.archav(util.hav(th)), np.abs(th))

    def test_cartesian(self):
        a = [0, 1, 2]
        b = [-1, 0]
        c = np.array([[0, -1], [0, 0], [1, -1], [1, 0], [2, -1], [2, 0]])
        assert np.alltrue(util.cartesian(a, b) == c)

    def test_davint(self):
        x = np.arange(0, 2 * pi)
        y = np.sin(x)
        assert util.davint(x, y, 0, 2 * pi) == 0

    def test_deriv(self):
        a = np.arange(10)
        assert np.allclose(util.deriv(a), np.ones(10))

    def test_gaussian(self):
        assert util.gaussian(0, 0, 1) == 1 / np.sqrt(2 * pi)
        assert (util.gaussian(2 * np.sqrt(2 * np.log(2)) / 2.0, 0, 1)
                == util.gaussian(0, 0, 2))
        assert util.gaussian(1, 1, 1) == 1 / np.sqrt(2 * pi)

    def test_rotmat(self):
        r = np.array([1, 0]) * util.rotmat(pi / 2).squeeze()
        assert np.allclose(r, [0, 1])
        r = np.array([1, 0]) * util.rotmat(-pi / 2).squeeze()
        assert np.allclose(r, [0, -1])
        r = np.array([1, 0]) * util.rotmat(pi).squeeze()
        assert np.allclose(r, [-1, 0])
        r = np.array([1, 0]) * util.rotmat(2 * pi).squeeze()
        assert np.allclose(r, [1, 0])

class TestSearchingSorting():
    def test_between(self):
        a = np.arange(100, int)
        assert util.between(a, [5, 8]) == np.array([5, 6, 7, 8])
        assert util.between(a, [5, 8], closed=False) == np.array([6, 7])

    def test_cmp_leading_num(self):
        a = '1P 103P 88P asdf'.split()
        a.sort(util.cmp_leading_num)
        assert a == '1P 88P 103P asdf'.split()

    def test_groupby(self):
        lists = (list('abcdef'), range(6))
        key = [3, 3, 100, 0, 0, 100]
        grouped = util.groupby(key, *lists)
        assert key[100] == (['c', 'f'], [2, 5])

    def test_nearest(self):
        a = np.logspace(-1, 2, 7)
        n = util.nearest(a, np.log(20))
        assert n == 2

    def test_takefrom(self):
        r = util.takefrom((list('asdf'), list('jkl;')), [3, 2])
        assert r == (['f', 'd'], [';', 'l'])

    def test_whist(self):
        x = np.arange(5) + 1
        y = x**2
        w = 1.0 / x
        h = util.whist(x, y, w, errors=False, bins=[0, 2.5, 5.0])
        assert np.allclose(h[0], np.array([0.5, 3.0]))
        assert h[1] is None
        assert h[2] == np.array([2, 3])
        assert h[3] == np.array([0, 2.5, 5.0])

def test_phase_integral():
    from mskpy.models.surfaces import phaseHG
    def phasef(phase):
        return phaseHG(phase, 0.15)

    pint = util.phase_integral(phasef)
    # According to Muinonen et al. (2010, Icarus 209, 542), the
    # approximation pint(G) == (0.290 + 0.684 * G) is good to 5%.  For
    # G = 0.15, this is 0.3926.
    assert np.allclose(pint, 0.384039206983)
