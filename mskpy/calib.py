# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
calib --- Tools for photometric and spectroscopic calibrations
==============================================================

.. autosummary::
   :toctree: generated/

   cohen_standard
   dw_atran
   e490
   filter_trans
   solar_flux
   wehrli
   sun

"""

from .util import autodoc
__all__ = [
    'cohen_standard',
    'dw_atran',
    'e490',
    'filter_trans',
    'solar_flux',
    'wehrli',
    'sun_w18'
]

import os
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.units import Quantity
from astropy.table import Table
import synphot
from sbpy.calib import Sun

from . import __path__ as __mskpy_path__
from .config import config

# Solar spectra downloaded from
#   http://rredc.nrel.gov/solar/spectra/am0/
#
# Wehrli 1985 and ASTM E-490-00
_wehrli = __mskpy_path__[0] + "/data/wehrli85.txt"
_e490 = __mskpy_path__[0] + "/data/E490_00a_AM0.txt"
_e490_sm = __mskpy_path__[0] + "/data/e490-lowres.txt"

# the filter transmission files
_filterdir = __mskpy_path__[0] + '/data/filters'

# The location of mid-IR calibration data.  cohenstandard() will
# search all directories that match cohen* in _midirdir.  Many
# templates are available from Gemini:
# http://www.gemini.edu/sciops/instruments/mid-ir-resources/spectroscopic-calibrations
_midirdir = '/home/msk/data/mid-ir'


def e490(smooth=False, unit=u.Unit('W/(m2 um)')):
    """The ASTM (2000) E490-00 solar spectrum (at 1 AU).

    Parameters
    ----------
    smooth : bool
      Return a lower-resolution (histogrammed) spectrum (see Notes).
    unit : astropy Unit
      Return flux in these units (must be spectral flux density).

    Returns
    -------
    w : Quantity
      Spectrum wavelength.
    f : Quantity
      Spectrum flux density.

    Notes
    -----
    The smoothed spectrum, up to 10 um, is the original E490 table,
    rebinned.  At > 10 um, the original resolution is retained.

    """
    if smooth:
        w, f = np.loadtxt(_e490_sm).T
    else:
        w, f = np.loadtxt(_e490).T

    w = w * u.um
    f = f * u.W / u.m**2 / u.um
    if f.unit != unit:
        equiv = u.spectral_density(w.unit, w.value)
        f = f.to(unit, equivalencies=equiv)

    return w, f


def wehrli(smooth=True, unit=u.Unit('W/(m2 um)')):
    """Wehrli (1985) solar spectrum (at 1 AU).

    Parameters
    ----------
    smooth : bool, optional
      Set to `True` to smooth the solar spectrum (see Notes).
    unit : astropy Unit
      Return flux in these units (must be spectral flux density).

    Returns
    -------
    w : Quantity
        Spectrum wavelength.
    f : Quantity
        Spectrum flux density.

    Notes
    -----
    The saved smoothed spectrum is generated via:
      util.deresolve('gaussian(0.005)', w, f)

    """
    if smooth:
        # smoothed version already in W/cm2/um
        w, f = np.loadtxt(_wehrli.replace('.txt', '_smoothed0.005.txt')).T
    else:
        w, f = np.loadtxt(_wehrli).T[:2]
        w *= 0.001  # nm -> micron
        f *= 0.1    # W/m2/nm -> 1e-4 m2/cm2 * 1e3 nm/um = W/cm2/um

    w = w * u.um
    f = f * u.W / u.cm**2 / u.um
    if f.unit != unit:
        equiv = u.spectral_density(w.unit, w.value)
        f = f.to(unit, equivalencies=equiv)

    return w, f


def solar_flux(wave, smooth=True, unit=u.Unit('W/(m2 um)')):
    """Spectrum of the Sun.

    `e490` is linearly interpolated to `wave`.

    Parameters
    ----------
    wave : float, array or Quantity
        Wavelength. [float/array: microns]
    smooth : bool, optional
        Set to `True` to use `e490`'s smoothed spectrum.
    unit : astropy Unit
      Return flux in these units (must be spectral flux density).

    Returns
    -------
    f : Quantity
      The solar flux density at `wave`.

    """

    from scipy.interpolate import interp1d
    from .util import asQuantity

    wave = asQuantity(wave, u.um).value

    solarw, solarf = e490(smooth=smooth, unit=unit)
    solarInterp = interp1d(solarw.value, solarf.value)

    if not np.iterable(wave):
        return solarInterp([wave])[0] * solarf.unit
    else:
        return solarInterp(wave) * solarf.unit


def filter_trans(name):
    """Wavelength and filter transmission for a requested filter.

    UBV are Johnson, RI are Cousins, from STScI's CDBS.

    Parameters
    ----------
    name : str
      One of the following (case insensitive):
        * 2MASS J
        * 2MASS H
        * 2MASS KS
        * MKO J
        * MKO H
        * MKO KS
        * MKO K
        * MKO KP
        * MKO LP
        * MKO MP
        * IRAC CH1
        * IRAC CH2
        * IRAC CH3
        * IRAC CH4
        * MIPS 24
        * MIPS 70
        * MIPS 160
        * IRS Red
        * IRS Blue
        * PS1 open
        * PS1 g
        * PS1 r
        * PS1 i
        * PS1 z
        * PS1 y
        * PS1 w
        * U
        * B
        * V
        * R
        * I
        * LSST u
        * LSST g
        * LSST r
        * LSST i
        * LSST z
        * LSST y
        * FOR 5.4
        * FOR 6.4
        * FOR 6.6
        * FOR 7.7
        * FOR 8.6
        * FOR 11.1
        * FOR 11.3
        * FOR 20
        * FOR 24
        * FOR 32
        * FOR 34
        * FOR 35
        * FOR 37
        * WISE W1
        * WISE W2
        * WISE W3
        * WISE W4
        * SDSS u'
        * SDSS g'
        * SDSS r'
        * SDSS i'
        * SDSS z'


    Returns
    -------
    bp : `~synphot.SpectralElement`


    Raises
    ------
    KeyError when the requested filter is invalid.

    """

    # file name, [wavelength column, transmission column], wavelength units
    filters = {
        '2mass j': ('/2mass/jrsr.tbl', [1, 2], u.um, {}),
        '2mass h': ('/2mass/hrsr.tbl', [1, 2], u.um, {}),
        '2mass ks': ('/2mass/krsr.tbl', [1, 2], u.um, {}),
        'mko j': ('/mko/nsfcam_jmk_trans.dat', [0, 1], u.um, {}),
        'mko h': ('/mko/nsfcam_hmk_trans.dat', [0, 1], u.um, {}),
        'mko ks': ('/mko/nsfcam_ksmk_trans.dat', [0, 1], u.um, {}),
        'mko k': ('/mko/nsfcam_kmk_trans.dat', [0, 1], u.um, {}),
        'mko kp': ('/mko/nsfcam_kpmk_trans.dat', [0, 1], u.um, {}),
        'mko lp': ('/mko/nsfcam_lpmk_trans.dat', [0, 1], u.um, {}),
        'mko mp': ('/mko/nsfcam_mpmk_trans.dat', [0, 1], u.um, {}),
        'irac ch1': ('/spitzer/080924ch1trans_full.txt', [0, 1], u.um, {}),
        'irac ch2': ('/spitzer/080924ch2trans_full.txt', [0, 1], u.um, {}),
        'irac ch3': ('/spitzer/080924ch3trans_full.txt', [0, 1], u.um, {}),
        'irac ch4': ('/spitzer/080924ch4trans_full.txt', [0, 1], u.um, {}),
        'mips 24': ('/spitzer/mips24.txt', [0, 1], u.um, {}),
        'mips 70': ('/spitzer/mips70.txt', [0, 1], u.um, {}),
        'mips 160': ('/spitzer/mips160.txt', [0, 1], u.um, {}),
        'ps1 open': ('/panstarrs/tonry12-transmission.txt', [0, 1], u.nm,
                     {'skiprows': 26}),
        'ps1 g': ('/panstarrs/tonry12-transmission.txt', [0, 2], u.nm,
                  {'skiprows': 26}),
        'ps1 r': ('/panstarrs/tonry12-transmission.txt', [0, 3], u.nm,
                  {'skiprows': 26}),
        'ps1 i': ('/panstarrs/tonry12-transmission.txt', [0, 4], u.nm,
                  {'skiprows': 26}),
        'ps1 z': ('/panstarrs/tonry12-transmission.txt', [0, 5], u.nm,
                  {'skiprows': 26}),
        'ps1 y': ('/panstarrs/tonry12-transmission.txt', [0, 6], u.nm,
                  {'skiprows': 26}),
        'ps1 w': ('/panstarrs/tonry12-transmission.txt', [0, 7], u.nm,
                  {'skiprows': 26}),
        'irs red': ('/spitzer/redPUtrans.txt', [0, 1], u.um, {}),
        'irs blue': ('/spitzer/bluePUtrans.txt', [0, 1], u.um, {}),
        'u': ('johnson/johnson_u_004_syn.fits',
              ('WAVELENGTH', 'THROUGHPUT'), u.AA, {}),
        'b': ('johnson/johnson_b_004_syn.fits',
              ('WAVELENGTH', 'THROUGHPUT'), u.AA, {}),
        'v': ('johnson/johnson_v_004_syn.fits',
              ('WAVELENGTH', 'THROUGHPUT'), u.AA, {}),
        'r': ('cousins/cousins_r_004_syn.fits',
              ('WAVELENGTH', 'THROUGHPUT'), u.AA, {}),
        'i': ('cousins/cousins_i_004_syn.fits',
              ('WAVELENGTH', 'THROUGHPUT'), u.AA, {}),
        'lsst u': ('lsst/total_u.dat', [0, 1], u.nm, {'skiprows': 7}),
        'lsst g': ('lsst/total_g.dat', [0, 1], u.nm, {'skiprows': 7}),
        'lsst r': ('lsst/total_r.dat', [0, 1], u.nm, {'skiprows': 7}),
        'lsst i': ('lsst/total_i.dat', [0, 1], u.nm, {'skiprows': 7}),
        'lsst z': ('lsst/total_z.dat', [0, 1], u.nm, {'skiprows': 7}),
        'lsst y': ('lsst/total_y.dat', [0, 1], u.nm, {'skiprows': 7}),
        'for 5.4': ('/sofia/OCLI_NO5352-8_2.txt', [1, 2], u.um, {}),
        'for 6.4': ('/sofia/OCLI_N06276-9_2.txt', [1, 2], u.um, {}),
        'for 6.6': ('/sofia/N06611.txt', [1, 2], u.um, {}),
        'for 7.7': ('/sofia/OCLI_N07688-9A_1.txt', [1, 2], u.um, {}),
        'for 8.6': ('/sofia/OCLI_N08606-9_1.txt', [1, 2], u.um, {}),
        'for 11.1': ('/sofia/OCLI_N11035-9A.txt', [1, 2], u.um, {}),
        'for 11.3': ('/sofia/OCLI_N11282-9_1.txt', [1, 2], u.um, {}),
        'for 20': ('/sofia/FOR-20um-542-090-091.txt', [1, 2], u.um, {}),
        'for 24': ('/sofia/Lakeshore_24um_5000_18-28um_double.txt',
                   [1, 2], u.um, {}),
        'for 32': ('/sofia/FOR-30um-542-84-85.txt', [1, 2], u.um, {}),
        'for 34': ('/sofia/Lakeshore_33um_4587_28-40um_double.txt',
                   [1, 2], u.um, {}),
        'for 35': ('/sofia/Lakeshore_34um_5007_28-40um_double.txt',
                   [1, 2], u.um, {}),
        'for 37': ('/sofia/Lakeshore_38um_5130_5144_double.txt',
                   [1, 2], u.um, {}),
        'wise w1': ('/wise/RSR-W1.txt', [0, 1], u.um, {}),
        'wise w2': ('/wise/RSR-W2.txt', [0, 1], u.um, {}),
        'wise w3': ('/wise/RSR-W3.txt', [0, 1], u.um, {}),
        'wise w4': ('/wise/RSR-W4.txt', [0, 1], u.um, {}),
        "sdss u'": ('/usno40/usno_u.res', [0, 3], u.AA, {'comments': '\\'}),
        "sdss g'": ('/usno40/usno_g.res', [0, 3], u.AA, {'comments': '\\'}),
        "sdss r'": ('/usno40/usno_r.res', [0, 3], u.AA, {'comments': '\\'}),
        "sdss i'": ('/usno40/usno_i.res', [0, 3], u.AA, {'comments': '\\'}),
        "sdss z'": ('/usno40/usno_z.res', [0, 3], u.AA, {'comments': '\\'}),
    }

    try:
        fil = filters[name.lower()]
    except KeyError:
        raise KeyError("filter {} cannot be found.".format(name.lower()))

    fn = _filterdir + '/' + fil[0]
    if fn.endswith('.fits'):
        table = Table(fits.getdata(fn))
    else:
        table = np.loadtxt(fn, **fil[3]).T

    cols = fil[1]
    w = table[cols[0]] * fil[2]
    tr = table[cols[1]]

    bp = synphot.SpectralElement(synphot.Empirical1D, points=w,
                                 lookup_table=tr)
    return bp


def cohen_standard(star, unit=u.Unit('W/(m2 um)')):
    """Cohen spectral templates.

    Parameters
    ----------
    star : string
      The name of a star.  This must match the filename of a template.
      For example, use HD6112 or alpha-lyr.  The suffix .tem is
      appended.
    unit : astropy Unit
      Return these units, must be spectral flux density.

    Returns
    -------
    wave : Quantity
      The wavelengths.
    flux : Quantity
      The fluxes.

    Notes
    -----
    The path to the templates is defined in `calib._midirdir`.

    """
    import os
    import re

    templatefile = "{0}/cohen/{1}.tem".format(_midirdir, star)
    if not os.path.exists(templatefile):
        raise ValueError("{0} not found.".format(templatefile))

    # Cohen template format:
    #  1-11   E11.4  um        Lambda      Wavelength
    # 12-22   E11.4  W/cm2/um  F_Lambda    Monochromatic specific intensity
    # 23-33   E11.4  W/cm2/um  e_F_Lambda *Total uncertainty in F_Lambda
    # 34-44   E11.4  %         Local       Local bias
    # 45-55   E11.4  %         Global      Global bias

    # many of the template files have a header
    tableheader = re.compile("Wavelength.*Irradiance.*Total")
    with open(templatefile, 'r') as inf:
        lines = inf.readlines()
        for i, line in enumerate(lines):
            if len(tableheader.findall(line)) > 0:
                break

    if i == (len(lines) - 1):
        # no header, just read it in
        skiprows = 0
    else:
        skiprows = i + 2

    wave, fd, efd = np.loadtxt(templatefile, skiprows=skiprows, unpack=True,
                               usecols=(0, 1, 2))

    wave = wave * u.um
    fd = (fd * u.Unit('W/(cm2 um)')).to(unit, u.spectral_density(wave))

    return wave, fd


def dw_atran(airmass, fw, ft, pw='2.5'):
    """Use the Diane Wooden method to compute the transmission of the
    atmosphere in a filter.

    Parameters
    ----------
    am : float
      The airmass at which to compute the transmission.
    fw : array
      Filter wavelengths.
    ft : array
      Filter transmission.
    pw : str, optional
      The precipitable water in mm (either 2.5 or 3.3).

    Returns
    -------
    tr : float or ndarray
      The computed transmission of the sky.

    """

    from glob import glob
    from .util import bandpass

    f = glob('{0}/atmosphere/tr_1??00ft_{1}mm_7*.txt'.format(
        _midirdir, pw))[0]
    tw10, tb10, tc10 = np.loadtxt(f).T
    if pw == '3.3':
        f = glob('{0}/atmosphere/tr_1??00ft_3.4mm_15*.txt'.format(
            _midirdir))[0]

    else:
        f = glob('{0}/atmosphere/tr_1??00ft_{1}mm_15*.txt'.format(
            _midirdir, pw))[0]
    tw20, tb20, tc20 = np.loadtxt(f, usecols=(0, 2, 3)).T

    tw = np.r_[tw10, tw20]
    tt = np.r_[np.exp(-tb10 * np.sqrt(airmass) - tc10 * airmass),
               np.exp(-tb20 * np.sqrt(airmass) - tc20 * airmass)]

    return bandpass(tw, tt, fw=fw, ft=ft)[1]


try:
    with fits.open(os.path.join(config.get('calib', 'solar_spectra_path'), 'willmer18-sun_composite.fits')) as hdu:
        sun_w18 = Sun.from_array(
            hdu[1].data['WAVE'] * u.AA,
            hdu[1].data['FLUX'] * u.erg / u.cm**2 / u.s / u.AA,
            description='Willmer (2018) composite solar spectrum',
            bibcode='2018ApJS..236...47W')
except FileNotFoundError:
    pass

del Sun


# update module docstring
autodoc(globals())
del autodoc
