# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This package contains utilities to run the test suite.
"""

import numpy as np
import astropy.units as u
from mskpy import photometry as P
from mskpy.photometry import hb

class TestHB():
    """Test photometry.hb.

    From Dave Schleicher 2017 Mar 24:

      As an example, a set of values for Garradd had the following:
  
      cont. mag (u, b, g):   13.126, 12.014, 11.412
      flux(erg/cm/cm/s/A):  4.39e-14, 9.72e-14, 9.85e-14
  
      color excess (u-b, b-g):  0.010, 0.096
      cont. redden (u-b, b-g):  0.096, 1.086
      normal color (u-b, b-g):  0.010, 0.118
  
    A'Hearn et al. (1984, AJ 89, 579):

      A color excess of 0.3 mag [from 3675 to 5240 Å] corresponds to a
      reflectivity gradient of 18% per 1000 Å.

    """

    def test_continuum_color(self):
        color = hb.continuum_color('UC', 13.126, 0, 'BC', 12.014, 0)[0]
        assert np.isclose(color.to('10 % / um').value, 0.010, atol=0.001)

    def test_estimate_continuum_short_short_to_long(self):
        m = dict(UC=13.126, BC=12.014)
        fluxd = hb.estimate_continuum('BC', m)

        # test constants all from Farnham et al. 2000
        R_UCBC = 0.998 * ((m['UC'] - m['BC']) - 1.101)
        dm = m['BC'] - 0.507 + (4.453 - 5.259) * R_UCBC
        test = 3.616e-9 * 10**(-0.4 * dm)

        assert np.isclose(fluxd['GC'].to('erg/(s cm2 AA)').value, test,
                          atol=1e-16)

    def test_estimate_continuum_short_long_to_long(self):
        m = dict(UC=13.126, BC=12.014, GC=11.412)
        fluxd = hb.estimate_continuum('BC', m)
        assert np.isclose(fluxd['GC'].to('erg/(s cm2 AA)').value, 9.85e-14,
                          atol=1e-16)

    def test_estimate_continuum_long_short_to_short(self):
        m = dict(UC=13.126, BC=12.014, GC=11.412)
        fluxd = hb.estimate_continuum('BC', m)
        assert np.isclose(fluxd['UC'].to('erg/(s cm2 AA)').value, 4.39e-14,
                          atol=1e-16)

    def test_estimate_continuum_long_long_to_short(self):
        m = dict(BC=12.014, GC=11.412)
        fluxd = hb.estimate_continuum('BC', m)

        # test constants all from Farnham et al. 2000
        R_BCGC = 1.235 * ((m['BC'] - m['GC']) - 0.507)
        dm = m['BC'] + 1.101 + (4.453 - 3.449) * R_BCGC
        test = 7.802e-9 * 10**(-0.4 * dm)

        assert np.isclose(fluxd['UC'].to('erg/(s cm2 AA)').value, test,
                          atol=1e-16)

    def test_Rm2S(self):
        Rm = 0.3 * u.mag / (5240 * u.AA - 3675 * u.AA)
        S = hb.Rm2S(3675 * u.AA, 5240 * u.AA, Rm)

        alpha = 10**(0.4 * 0.3)
        test = (alpha - 1) / (alpha + 1) * 2 / (5.240 - 3.675) * 100
        
        assert np.isclose(S.to('10 % / um').value, test)

    def test_S2Rm(self):
        alpha = 10**(0.4 * 0.3)
        S = (alpha - 1) / (alpha + 1) * 2 / (5.240 - 3.675) * 100
        Rm = hb.S2Rm(3675 * u.AA, 5240 * u.AA, S * u.Unit('10 % / um'))

        test = 0.3 / (5240 * u.AA - 3675 * u.AA).to('0.1 um').value
        assert np.isclose(Rm.to('10 mag / um').value, test)
