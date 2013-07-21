# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This package contains utilities to run the test suite.
"""

import numpy as np
from numpy import pi, allclose
from mskpy import util

class TestUtilMathmatical():
    def test_haversine(self):
        th = np.arange(-pi, pi)
        assert allclose(util.archav(util.hav(th)), np.abs(th))

    def test_cartesian(self):
        a = [0, 1, 2]
        b = [-1, 0]
        c = np.array([[0, -1], [0, 0], [1, -1], [1, 0], [2, -1], [2, 0]])
        assert np.alltrue(util.cartesian(a, b) == c)

    def test_davint(self):
        x = np.linspace(0, 2 * pi)
        y = np.sin(x)
        assert allclose(util.davint(x, y, 0, 2 * pi), 0)

    def test_deriv(self):
        a = np.arange(10)
        assert allclose(util.deriv(a), np.ones(10))

    def test_gaussian(self):
        assert util.gaussian(0, 0, 1) == 1 / np.sqrt(2 * pi)
        assert (util.gaussian(2 * np.sqrt(2 * np.log(2)) / 2.0, 0, 1)
                == util.gaussian(0, 0, 2))
        assert util.gaussian(1, 1, 1) == 1 / np.sqrt(2 * pi)

    def test_rotmat(self):
        r = np.array([1, 0]) * util.rotmat(pi / 2).squeeze()
        assert allclose(r, [0, 1])
        r = np.array([1, 0]) * util.rotmat(-pi / 2).squeeze()
        assert allclose(r, [0, -1])
        r = np.array([1, 0]) * util.rotmat(pi).squeeze()
        assert allclose(r, [-1, 0])
        r = np.array([1, 0]) * util.rotmat(2 * pi).squeeze()
        assert allclose(r, [1, 0])

class TestFITSWCS():
    def test_basicwcs(self):
        wcs = util.basicwcs([256, 256], [0., 45], 1, 60)

    #fitslog
    #getrot

class TestSearchingSorting():
    def test_between(self):
        a = np.arange(20.)
        assert util.between(a, [4, 9]).sum() == 6
        assert util.between(a, [4, 9], closed=False).sum() == 4
        assert util.between(a, [[4, 9], [13, 14]]).sum() == 8

    def test_groupby(self):
        keys = [1, 2, 2, 0, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 0, 0, 0, 1, 2, 2]
        lists = (list('abcdefghijklmnopqrstuvwxyz'), range(26))
        grouped = util.groupby(keys, *lists)
        answer = {0: (['d', 'o', 'p', 'q'], [3, 14, 15, 16]),
                  1: (['a', 'e', 'f', 'g', 'h', 'i', 'j', 'l', 'n', 'r'],
                      [0, 4, 5, 6, 7, 8, 9, 11, 13, 17]),
                  2: (['b', 'c', 'k', 'm', 's', 't'], [1, 2, 10, 12, 18, 19])}
        assert grouped == answer

    def test_leading_num_key(self):
        a = ['1P', '101P', '2P', 'C/2001 Q4']
        b = ['1P', '2P', '101P', 'C/2001 Q4']
        assert sorted(a, key=util.leading_num_key) == b

    def test_nearest(self):
        a = np.arange(20.)
        assert util.nearest(a, 2.5) == 2
        assert util.nearest(a, 2.9) == 3
        assert util.nearest(a, 3) == 3
        assert util.nearest(a, 3.1) == 3
        assert util.nearest(a, 3.5) == 3

    def test_takefrom(self):
        a = list(range(20))
        b = list(range(20))
        c = [1, 5, 10]
        assert util.takefrom((a, b), c) == (c, c)

    def test_whist(self):
        x = np.arange(1, 5.)
        y = x**2
        w = x**-1
        wh = util.whist(x, y, w, errors=False, bins=[0, 5])
        assert allclose(wh[0], 4.8)

class TestSpecial():
    def test_ec2eq(self):
        # from IDL Astro Library, euler.pro
        assert allclose(util.ec2eq(45, -60), [62.139614, -40.838363])

    def test_projected_vector_angle(self):
        a = util.projected_vector_angle([0, 1, 0], [0, 0, 1000], 0, 0)
        assert allclose(a, 0)

    def test_spherical_coord_rotate(self):
        ll = util.spherical_coord_rotate(0, 90, 0, 0, 0, 0)
        assert allclose(ll, [0, -90])

    def test_state2orbit(self):
        from astropy.constants import au
        vcirc = 1.32712440018e11 / au.kilometer
        orbit = util.state2orbit([0, 1, 0], [vcirc, 0, 0])
        assert allclose([orbit['a'], orbit['ec'], orbit['in']],
                           [au.kilometer, 0.0, 0.0])

    def test_vector_rotate(self):
        r = util.vector_rotate([0, 0, 1], [1, 0, 1], 180)
        assert allclose(r, [1, 0, 0])

class TestStatistics():
    def test_kuiper(self):
        from scipy.stats import ks_2samp
        x = np.random.randn(1000)
        y = np.random.randn(1000) * 2
        v, pk = util.kuiper(x, y)
        d, pks = ks_2samp(x, y)
        assert v > d
        assert pk < pks

    # kuiper_prob

    def test_mean2minmax(self):
        x = util.mean2minmax([0, 1, 2])
        assert all(x == np.array([1., 1.]))

    def test_meanclip(self):
        x = np.arange(10)
        assert util.meanclip(x) == 4.5
        x[-1] = 100
        assert util.meanclip(x) == 4.

    def test_midstep(self):
        assert all(util.midstep([0, 1, 2]) == np.array([0.5, 1.5]))

    def test_minmax(self):
        assert all(util.minmax(np.arange(10)) == np.array([0, 9]))

    def test_nanmedian(self):
        x = np.arange(10.)
        x[4] = np.nan
        assert util.nanmedian(x) == 5.

    def test_nanminmax(self):
        x = np.arange(10.)
        x[4] = np.nan
        assert all(util.nanminmax(x) == np.array([0., 9.]))

    def test_randpl(self):
        p1 = util.randpl(1, 100, 1, 1000)
        p2 = util.randpl(1, 100, 2, 1000)
        assert allclose(sum(p1 > 10), sum(p2 > 10) / 10.)

    def test_sigma(self):
        assert allclose(util.sigma(1), 0.682689492)

    def test_spearman(self):
        x = np.arange(1000)
        y = 2 * x
        r, p, z = util.spearman(x, y)
        assert r == 1.

    def test_uclip(self):
        x = np.arange(10)
        assert util.uclip(x, np.median) == 5
        x[-1] = 100
        assert util.uclip(x, np.median) == 4        

class TestSpecial():
    def test_bandpass(self):
        wave = np.arange(1, 101)
        flux = util.gaussian(wave, 50, 3)
        fw = wave
        ft = np.zeros(100)
        ft[40:60] = 1
        w, f = util.bandpass(wave, flux, fw=fw, ft=ft)
        flux_analytical = util.sigma(10 / 3.) * 0.05
        assert allclose(f, flux_analytical, 1e-4)

        w, f, e = util.bandpass(wave, flux, np.ones(100), fw=fw, ft=ft)
        assert allclose(f, flux_analytical, 1e-4)

        fw = [1, 39.9, 40, 59, 59.1, 100]
        ft = [0,    0,  1,  1,    0,   0]
        w, f = util.bandpass(wave, flux, fw=fw, ft=ft, k=1)
        assert allclose(f, flux_analytical, 1e-4)

        flux = np.zeros(100)
        flux[50] = 1.0
        fw = wave
        ft = np.zeros(100)
        ft[40:60] = 1
        w, f = util.bandpass(wave, flux, fw=fw, ft=ft)
        assert allclose(f, 0.05)

    def test_deresolve(self):
        wave = np.arange(1, 101)
        flux = np.zeros(wave.shape)
        flux[50] = 1
        f = util.deresolve("Gaussian(3.0)", wave, flux)
        assert allclose(flux.sum(), f.sum())

        f = util.deresolve("uniform(3.0)", wave, flux)
        assert allclose(flux.sum(), f.sum())
        assert allclose(f.max(), 0.33)

        f = util.deresolve(lambda w: np.sin(w / 10. / np.pi), wave, flux)
        assert allclose(flux.sum(), f.sum())

    def test_planck(self):
        import astropy.units as u
        import astropy.constants as const

        wave = np.logspace(-1, 3, 10000) * u.um
        I = util.planck(wave, 300, unit=u.Unit('W/(m2 um sr)'))
        assert allclose(wave[I.value.argmax()], 2.8977685e3 / 300, rtol=1e-3)

        F = (util.davint(wave.value, I.value, wave.value[0], wave.value[-1])
             * np.pi)
        assert allclose(F, const.sigma_sb.si.value * 300.**4, rtol=1e-5)

    def test_phase_integral(self):
        from mskpy.models.surfaces import phaseHG
        def phasef(phase):
            return phaseHG(phase, 0.15)

        pint = util.phase_integral(phasef)
        # According to Muinonen et al. (2010, Icarus 209, 542), the
        # approximation pint(G) == (0.290 + 0.684 * G) is good to 5%.
        # For G = 0.15, this is 0.3926.
        assert allclose(pint, 0.384039206983)

    def test_polcurve(self):
        th = np.arange(180.)
        p = util.polcurve(th, 48.8, 1.26, 1.53, 21.1)
        assert th[p.argmax()] == 83
        assert th[p.argmin()] == 12

    def test_savitzky_golay(self):
        a = np.random.randn(1000)
        b = util.savitzky_golay(a, 7)
        assert allclose(a.sum(), b.sum(), 1e-2)
