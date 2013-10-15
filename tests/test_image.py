# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This package contains utilities to run the test suite.
"""

import numpy as np
from numpy import pi
from mskpy import image

class TestImageCore():
    def test_imshift(self):
        a = np.arange(5.0).reshape((5, 1))
        b = image.imshift(a, [0.5, 0], subsample=2)
        assert b[0] == 2.0

    def test_rarray(self):
        r = image.rarray((10, 10), subsample=10)
        assert all(r[4:6, 4] == r[4:6, 5])

    def test_rebin(self):
        a = np.ones((5, 5))
        b = image.rebin(a, 5, flux=False)
        assert b.shape == (25, 25)
        assert b[0, 0] == 1.0
        b = image.rebin(a, 5, flux=True)
        assert b[0, 0] == 1 / 25.
        b = image.rebin(a, -5, flux=True)
        assert b.shape == (1, 1)
        assert b[0, 0] == 25.
        b = image.rebin(a, -4, flux=True, trim=True)
        assert b.shape == (1, 1)
        assert b[0, 0] == 16.

    def test_stack2grid(self):
        a = np.arange(16).reshape((4, 2, 2))
        b = image.stack2grid(a)
        assert all(b[2] == np.array([8, 9, 12, 13]))

    def test_tarray(self):
        t = image.tarray((101, 101), yx=(0, 0))
        assert t[0, 100] == 0.0
        assert t[100, 0] == pi / 2
        assert t[100, 100] == pi  / 4
        t = image.tarray((101, 101), yx=(0, 0), subsample=10)
        assert t[0, 100] == 0.0
        assert t[100, 0] == pi / 2
        assert t[100, 100] == pi  / 4

    def unwrap(self):
        im = image.rarray((101, 101))
        rt, r, th, n = image.unwrap(im, [50, 50], bins=10)
        assert np.allclose(rt[3, :5], rt[3, 5:])

    def xarray(self):
        x = image.xarray((5, 5))
        assert x[0, 0] == 0
        assert x[0, 4] == 4
        x = image.xarray((5, 5), yx=(2, 2))
        assert x[0, 0] == -2
        assert x[0, 4] == 2
        x = image.xarray((5, 5), yx=(2, 2), rot=pi / 4)
        assert np.allclose(x[0, 0], -2 * np.sqrt(2))
        assert np.allclose(x[0, 4], 0)

    def yarray(self):
        y = image.yarray((5, 5))
        assert y[0, 0] == 0
        assert y[4, 0] == 4
        y = image.yarray((5, 5), yx=(2, 2))
        assert y[0, 0] == -2
        assert y[4, 0] == 2
        y = image.yarray((5, 5), yx=(2, 2), rot=pi / 4)
        assert np.allclose(y[4, 0], 2 * np.sqrt(2))
        assert np.allclose(y[0, 0], 0)

