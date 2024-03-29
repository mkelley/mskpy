# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
spitzer --- Spitzer instruments.
================================

   Functions
   ---------
   irsclean
   irsclean_files
   irs_summary
   moving_wcs_fix

   Classes
   -------
   IRAC
   IRS
   IRSCombine

"""

import os.path
from collections import OrderedDict, defaultdict
import numpy as np
from scipy.interpolate import splev, splrep
import scipy.ndimage as nd
import astropy.units as u
from astropy.table import Table
from astropy.io import fits, ascii
import astropy.constants as const
from astropy.coordinates import SkyCoord, Angle

try:
    from ..ephem import Spitzer
except ImportError:
    Spitzer = None

from ..util import (autodoc, davint, deriv, meanclip, minmax, nearest,
                    cal2time, between, linefit, spherical_coord_rotate)
from ..config import config
from ..models import NEATM
from .instrument import Instrument, Camera, LongSlitSpectrometer
from ..calib import filter_trans
from ..image import apphot, bgphot, fixpix
from ..ephem import getgeom

__all__ = ['irsclean', 'irsclean_files', 'irs_summary',
           'IRAC', 'IRS', 'IRSCombine']

campaign2rogue = {
    'IRSX002500': 'IRS1',
    'IRSX002600': 'IRS2',
    'IRSX002700': 'IRS3',
    'IRSX002800': 'IRS4',
    'IRSX002900': 'IRS5',
    'IRSX003000': 'IRS6',
    'IRSX003100': 'IRS7',
    'IRSX003300': 'IRS8',
    'IRSX003400': 'IRS9',
    'IRSX003500': 'IRS10',
    'IRSX003600': 'IRS11',
    'IRSX003700': 'IRS12',
    'IRSX003800': 'IRS13',
    'IRSX003900': 'IRS14',
    'IRSX004000': 'IRS15',
    'IRSX004100': 'IRS16',
    'IRSX004300': 'IRS17',
    'IRSX004500': 'IRS18',
    'IRSX004600': 'IRS19',
    'IRSX004800': 'IRS20',
    'IRSX005000': 'IRS21.1',
    'IRSX007100': 'IRS21.2',
    'IRSX006900': 'IRS21.3',
    'IRSX007000': 'IRS21.4',
    'IRSX005200': 'IRS22',
    'IRSX005300': 'IRS23.1',
    'IRSX007300': 'IRS23.2',
    'IRSX005500': 'IRS24',
    'IRSX005700': 'IRS25',
    'IRSX005800': 'IRS26',
    'IRSX006000': 'IRS27',
    'IRSX006100': 'IRS28',
    'IRSX006300': 'IRS29',
    'IRSX006500': 'IRS30',
    'IRSX006700': 'IRS31',
    'IRSX006800': 'IRS32',
    'IRSX007200': 'IRS33',
    'IRSX007400': 'IRS34',
    'IRSX007500': 'IRS35',
    'IRSX007600': 'IRS36',
    'IRSX007700': 'IRS37',
    'IRSX007800': 'IRS38',
    'IRSX008000': 'IRS39',
    'IRSX008100': 'IRS40',
    'IRSX008200': 'IRS41',
    'IRSX008300': 'IRS42',
    'IRSX008400': 'IRS43',
    'IRSX009800': 'IRS44',
    'IRSX009900': 'IRS45',
    'IRSX010000': 'IRS46',
    'IRSX010100': 'IRS47',
    'IRSX008900': 'IRS48',
    'IRSX010200': 'IRS49',
    'IRSX010300': 'IRS50',
    'IRSX010400': 'IRS51.1',
    'IRSX011600': 'IRS51.2',
    'IRSX011400': 'IRS52',
    'IRSX009400': 'IRS53',
    'IRSX009500': 'IRS54',
    'IRSX010600': 'IRS55',
    'IRSX010700': 'IRS56',
    'IRSX010800': 'IRS57.1',
    'IRSX011700': 'IRS57.2',
    'IRSX010900': 'IRS58.1',
    'IRSX011800': 'IRS58.2',
    'IRSX011900': 'IRS58.3',
    'IRSX011000': 'IRS59.1',
    'IRSX012000': 'IRS59.2',
    'IRSX011100': 'IRS60',
    'IRSX012200': 'IRS61.1',
    'IRSX011200': 'IRS61.2'
}

module2channel = {'sl': 0, 'sh': 1, 'll': 2, 'lh': 3}


class IRAC(Camera):
    """Spitzer's Infrared Array Camera

    Attributes
    ----------

    Examples
    --------

    """

    def __init__(self):
        w = [3.550, 4.493, 5.731, 7.872] * u.um
        shape = (256, 256)
        ps = 1.22 * u.arcsec
        location = Spitzer
        Camera.__init__(self, w, shape, ps, location=location)

    def ccorrection(self, sf, channels=[1, 2, 3, 4]):
        """IRAC color correction.

        Seems to agree within 1% of the IRAC Instrument Handbook.
        Thier quoted values are good to ~1%.

        Parameters
        ----------
        sf : function
          A function that generates source flux density as a Quantity
          given wavelength as a Quantity.
        channels : list, optional
          A list of the IRAC channels for which to compute the color
          correction, e.g., `[1, 2]` for 3.6 and 4.5 um.

        Returns
        -------
        K : ndarray
          Color correction factor, where `Fcc = F / K`.

        """

        nu0 = (const.c.si / self.wave).to(u.teraHertz).value
        K = []
        for ch in channels:
            bp = filter_trans('IRAC CH{:}'.format(ch))
            tw = bp.waveset
            tr = bp(tw)
            nu = (const.c / tw).to(u.teraHertz).value

            sfnu = sf(tw).to(u.Jy, u.spectral_density(tw)).value

            i = ch - 1  # self.wave index
            sfnu /= sf(self.wave[i]).to(u.Jy,
                                        u.spectral_density(self.wave[i])).value
            j = nu.argsort()
            sfnu, tr, nu = [x[j] for x in (sfnu, tr, nu)]
            K.append((davint(nu, sfnu * tr * nu0[i] / nu, nu[0], nu[-1])
                      / davint(nu, tr * (nu0[i] / nu)**2, nu[0], nu[-1])))

        return np.array(K)


def warm_aperture_correction(rap, bgan):
    """Compute an aperture correction for IRAC Warm Mission data.

    Parameters
    ----------
    rap : float
      The radius of the photometric aperture.
    bgan : 2-element array-like
      The inner and outer radii of the background annulus, or `None`
      if there is no background annulus.

    Result
    ------
    c : float
      The aperture correction as a multiplicative factor: `F_true =
      F_measured * c`.

    Notes
    -----
    Requires I1_hdr_warm_psf.fits and I2_hdr_warm_psf.fits from July 2013:
    http://irsa.ipac.caltech.edu/data/SPITZER/docs/irac/calibrationfiles/psfprf/

    The default aperture flux was measured via:

    psf = (fits.getdata('I1_hdr_warm_psf.fits'),
           fits.getdata('I2_hdr_warm_psf.fits'))
    n, f = apphot(psf, (640., 640.), 10 / 0.24, subsample=1)
    bg = bgphot(psf, (640., 640.), r_[12, 20] / 0.24, ufunc=np.mean)[1]
    f -= n * bg
    Out[42]: array([  2.02430430e+08,   1.29336376e+08])

    """

    f0 = np.array([2.02430430e+08,   1.29336376e+08])

    path = config.get('irac', 'psf_path')
    psf = (fits.getdata(os.path.join(path, 'I1_hdr_warm_psf.fits')),
           fits.getdata(os.path.join(path, 'I2_hdr_warm_psf.fits')))

    n, f = apphot(psf, (640., 640.), rap / 0.24, subsample=1)
    if bgan is None:
        bg = 0
    else:
        bg = bgphot(psf, (640., 640.), np.array(bgan) / 0.24, ufunc=np.mean)[1]
    f -= n * bg

    return f0 / f

#    def ccorrection_tab(self, sw, sf):
#    """IRAC color correction of a tabulated spectrum.
#
#        Parameters
#        ----------
#        sw : Quantity
#          Source wavelength.
#        sf : Quantity
#          Source flux density.
#
#        Returns
#        -------
#        K : ndarray
#          Color correction: `Fcc = F / K`.
#
#        """
#
#        from scipy import interpolate
#        import astropy.constants as const
#        from ..calib import filter_trans
#        from ..util import davint, takefrom
#
#        nu0 = (const.c.si / self.wave).to(u.teraHertz).value
#        K = np.zeros(4)
#        for i in range(4):
#            tw, tr = filter_trans('IRAC CH{:}'.format(i + 1))
#            nu = (const.c / tw).to(u.teraHertz).value
#
#            # interpolate the filter transmission to a higher
#            # resolution
#            t
#
#            s = interpolate.splrep(sw.value, sf.value)
#            _sf = interpolate.splev(fw.value, s, ext=1)
#            _sf /= interpolate.splev(self.wave[i].value, s, ext=1)
#
#            _sf *= sf.unit.to(u.Jy, u.spectral_density(fw))
#
#            _sf, ft, nu = takefrom((_sf, ft, nu), nu.argsort())
#            K[i] = (davint(nu, _sf * ft * nu0[i] / nu, nu[0], nu[-1])
#                    / davint(nu, ft * (nu0[i] / nu)**2, nu[0], nu[-1]))
#        return K


class IRS(Instrument):
    """Spitzer's Infrared Spectrometer.

    Attributes
    ----------
    module : The current IRS module: SL1, SL2, Blue, Red, etc.  SH, LH, SL3, LL3 not yet implemented.

    Examples
    --------

    """

    modes = ['sl1', 'sl2', 'll1', 'll2', 'blue', 'red']

    def __init__(self):
        self.sl2 = LongSlitSpectrometer(
            6.37 * u.um,
            [32, 128],
            1.8 * u.arcsec,
            2.0,
            0.073 * u.um,
            R=90,
            location=Spitzer)
        self.sl1 = LongSlitSpectrometer(
            10.88 * u.um,
            [32, 128],
            1.8 * u.arcsec,
            2.06,
            0.12 * u.um,
            R=90,
            location=Spitzer)
        self.ll2 = LongSlitSpectrometer(
            17.59 * u.um,
            [33, 128],
            5.1 * u.arcsec,
            2.1,
            0.21 * u.um,
            R=90,
            location=Spitzer)
        self.ll1 = LongSlitSpectrometer(
            29.91 * u.um,
            [33, 128],
            5.1 * u.arcsec,
            2.1,
            0.35 * u.um,
            R=85,
            location=Spitzer)
        self.blue = Camera(
            15.8 * u.um,
            [31, 44],
            1.8 * u.arcsec,
            location=Spitzer)
        self.red = Camera(
            22.3 * u.um,
            [32, 43],
            1.8 * u.arcsec,
            location=Spitzer)

        self._mode = 'sl1'

    @property
    def mode(self):
        if self._mode in self.modes:
            return self.__dict__[self._mode]
        else:
            raise KeyError("Invalid mode: {}".format(self._mode))

    @mode.setter
    def mode(self, m):
        if m.lower() in self.modes:
            self._mode = m.lower()
        else:
            raise KeyError("Invalid mode: {}".format(m.lower()))

    def sed(self, *args, **kwargs):
        """Spectral energy distribution of a target.

        Parameters
        ----------
        *args
        **kwargs
          Arguments and keywords depend on the current IRS mode.

        Returns
        -------
        sed : ndarray

        """
        return self.mode.sed(*args, **kwargs)

    def lightcurve(self, *args, **kwargs):
        """Secular lightcurve of a target.

        Parameters
        ----------
        *args
        **kwargs
          Arguments and keywords depend on the current IRS mode.

        Returns
        -------
        lc : astropy Table

        """
        return self.mode.lightcurve(*args, **kwargs)


class IRSCombine:
    """Combine extracted and calibrated IRS data into a single spectrum.

    Only SL and LL currently supported.

    Parameters
    ----------
    files : array-like, optional
      Files to load.
    minimum_uncertainty : float, optional
      Force this minimum uncertainty on all results. (Jy)
    **kwargs
      Passed to `IRSCombine.read`.

    Attributes
    ----------
    aploss_corrected : dict
      The so-called aperture-loss corrected spectra for each module.
    coadded : dict
      The combined spectra for each module.
    coma : dict
      The nucleus subtracted spectra for each module.
    meta : dict
      A list of processing meta data.
    file_scales : dict
      A list of scale factors for each file.
    headers : dict
      The header from each file.
    nucleus : Table
      The nucleus model.
    modules : dict
      A list of files for each module name.
    order_scaled : dict
      Spectra including order-to-order scale factors.
    order_scales : dict
      Order-to-order scale factors.
    raw : dict
      A spectrum from each file.
    spectra : dict
      Always returns the most current reduction state.
    trimmed : dict
      The wavelength-trimmed spectra for each module.

    Examples
    --------

    files = sorted(glob('extract/*/*/*spect.tbl'))
    tab = irs_summary(files)

    combine = IRSCombine(files=files, sl=dict(column=[2], row=[4]))
    combine.scale_spectra()
    combine.coadd()
    combine.trim()
    combine.zap(ll2=[15.34, 15.42])
    combine.subtract_nucleus(2.23 * u.km, 0.04, eta=1.03, epsilon=0.95)
    # combine.aploss_correct()  # only for full-width extractions
    combine.slitloss_correct()  # uses IRS pipeline SLCF
    # combine.shape_correct(correction)  # Other shape corrections?
    combine.scale_orders('ll2')
    combine.write('comet-irs.txt')

    fig = plt.figure(1)
    plt.clf()
    plt.minorticks_on()
    combine.plot('raw')
    plt.setp(plt.gca(), ylim=(0, 0.8))
    plt.draw()

    fig = plt.figure(2)
    plt.clf()
    plt.minorticks_on()
    combine.plot('coadded')
    combine.plot('nucleus', ls='--', label='nucleus')
    mskpy.nicelegend(loc='lower right')
    plt.draw()

    fig = plt.figure(3)
    plt.clf()
    plt.minorticks_on()
    combine.plot_spectra()
    plt.draw()

    """

    def __init__(self, files=[], minimum_uncertainty=0.0001, **kwargs):
        self.minimum_uncertainty = minimum_uncertainty

        self.raw = None
        self.trimmed = None
        self.coadded = None
        self.nucleus = None
        self.coma = None
        self.aploss_corrected = None
        self.slitloss_corrected = None
        self.shape_corrected = None
        self.order_scaled = None

        self.meta = OrderedDict()
        self.meta['read_files'] = []
        self.meta['scale_spectra'] = []
        self.meta['coadd'] = []
        self.meta['zap'] = []
        self.meta['trim'] = []
        self.meta['subtract_nucleus'] = []
        self.meta['aploss_correct'] = []
        self.meta['slitloss_correct'] = []
        self.meta['shape_correct'] = []
        self.meta['scale_orders'] = []
        self.meta['scale_spectra'] = []

        self.read(files, **kwargs)

    @property
    def spectra(self):
        for k in ['order_scaled', 'shape_corrected', 'slitloss_corrected',
                  'aploss_corrected', 'coma', 'trimmed', 'coadded',
                  'trimmed']:
            if getattr(self, k) is not None:
                return getattr(self, k)

        return self.raw

    def aploss_correct(self):
        path = config.get('irs', 'spice_path')
        h = list(self.headers.values())[0]
        calset = h['CAL_SET'].strip("'").strip('.A')

        coeffs = ['a5', 'a4', 'a3', 'a2', 'a1', 'a0']
        self.aploss_corrected = dict()
        for k in self.coma.keys():
            fn = 'b{}_aploss_fluxcon.tbl'.format(module2channel[k[:2]])
            aploss = ascii.read(os.path.join(path, 'cal', calset, fn))

            fn = 'b{}_fluxcon.tbl'.format(module2channel[k[:2]])
            fluxcon = ascii.read(os.path.join(path, 'cal', calset, fn))

            i = int(k[-1]) - 1
            a = tuple(aploss[coeffs][i])
            b = tuple(fluxcon[coeffs][i])
            wa = aploss[i]['key_wavelength']
            wb = aploss[i]['key_wavelength']
            polya = np.polyval(a, wa - self.coma[k]['wave'])
            polyb = np.polyval(b, wb - self.coma[k]['wave'])
            alcf = aploss[i]['fluxcon'] * polya / \
                (fluxcon[i]['fluxcon'] * polyb)

            self.aploss_corrected[k] = dict()
            for kk, vv in self.coma[k].items():
                self.aploss_corrected[k][kk] = vv
            self.aploss_corrected[k]['fluxd'] *= alcf
            e = np.maximum(self.aploss_corrected[k]['err'] * alcf,
                           self.minimum_uncertainty)
            self.aploss_corrected[k]['err'] = e

        self.meta['aploss_correct'] = ['Aperture loss corrected.']

    def coadd(self, scales=dict(), sig=2.5, repeatability=0.018,
              combine='mean'):
        """Combine by module.

        Scale factors derived by `scale_spectra()` are applied by
        default.

        Run `coadd()` even when there is only one spectrum per module.

        Parameters
        ----------
        scales : dict or None
          Use these scale factors for each spectrum in
          `scales.keys()`.  Scale factors not in `scales` will be
          taken from `self.scales`.  Set to `None` to prevent any
          scaling.

        sig : float, optional
          If the number of spectra for a module is greater than 2,
          then `mskpy.meanclip` is used to clip the data.

        repeatability : float, optional
          Add this fraction of the flux density to the uncertainty, in
          quadrature.

        combine : string, optional
          Combination function: mean or median.  When combining two samples,
          the mean will always be used.

        """

        assert scales is None or isinstance(scales, dict)
        if scales is None:
            _scales = dict(zip(self.raw.keys(), np.ones(len(self.raw))))
        else:
            _scales = dict(**self.scales)
            _scales.update(scales)

        self.coadded = dict()
        self.meta['coadd'] = []
        for module, files in self.modules.items():
            for order in np.unique(self.raw[files[0]]['order']):
                k = self.raw[files[0]]['order'] == order

                w = np.array(sorted(self.raw[files[0]]['wavelength'][k]))
                dw = deriv(w) / 2.0
                bins = np.zeros(len(w) + 1)
                bins[:-1] = w - dw / 2.0
                bins[-1] = w[-1] + dw[-1] / 2.0
                shape = (len(files), len(bins) - 1)
                wave = np.ma.MaskedArray(np.zeros(shape), mask=np.zeros(shape))
                fluxd = np.ma.MaskedArray(
                    np.zeros(shape), mask=np.zeros(shape))
                err2 = np.ma.MaskedArray(np.zeros(shape), mask=np.zeros(shape))
                for i, fn in enumerate(files):
                    spec = self.raw[fn]
                    good = np.isfinite(spec['flux_density'][k]
                                       * spec['error'][k])
                    w = spec['wavelength'][k][good]
                    f = spec['flux_density'][k][good]
                    e = spec['error'][k][good]
                    n = np.histogram(w, bins)[0]

                    wave[i] = np.histogram(w, bins, weights=w)[0]
                    fluxd[i] = np.histogram(w, bins, weights=f)[0]
                    err2[i] = np.histogram(w, bins, weights=e**2)[0]

                    j = n > 0
                    wave[i, j] /= n[j]  # just in case 2 indices fell in 1 bin
                    fluxd[i, j] /= n[j]
                    err2[i, j] /= n[j]

                    # some bins end up empty
                    j = n == 0
                    if np.any(j):
                        wave.mask[i, j] = True
                        fluxd.mask[i, j] = True
                        err2.mask[i, j] = True

                    try:
                        j = spec['bit-flag'][k] > 0
                    except KeyError:
                        j = spec['bit_flag'][k] > 0
                    fluxd.mask[i, j] = True
                    err2.mask[i, j] = True

                    fluxd[i] *= _scales[fn]
                    err2[i] *= _scales[fn]**2

                w = wave.mean(0)  # mitigates empty bins
                f = np.ma.MaskedArray(np.zeros(len(w)), mask=np.zeros(len(w)))
                e = np.ma.MaskedArray(np.zeros(len(w)), mask=np.zeros(len(w)))
                for i in range(len(w)):
                    if fluxd.shape[0] > 2:
                        mc = meanclip(fluxd[:, i], lsig=sig, hsig=sig,
                                      full_output=True)
                        if combine == 'mean':
                            f[i] = mc[0]
                            e[i] = np.sqrt(sum(err2[mc[2], i])) / len(mc[2])
                        elif combine == 'median':
                            f[i] = np.median(fluxd[:, i][mc[2]])
                            # https://en.wikipedia.org/wiki/Median#Efficiency
                            e[i] = np.sqrt(
                                np.pi / 2 * sum(err2[mc[2], i])) / len(mc[2])
                    else:
                        # two or one samples, just use mean
                        f[i] = np.mean(fluxd[:, i])
                        e[i] = np.sqrt(np.sum(err2[:, i])) / fluxd.shape[0]

                f.mask[e == 0] == True
                e.mask[e == 0] == True

                # add in repeatability factor (cf., Section 9.3.2 of
                # IRS Instrument Handbook)
                e = np.sqrt((f * repeatability)**2 + e**2)
                e = np.maximum(e, self.minimum_uncertainty)

                i = ~(w.mask + f.mask + e.mask)
                self.coadded[module[:2] + str(order)] = dict(
                    wave=w[i], fluxd=f[i], err=e[i])

                if fluxd.shape[0] > 2:
                    self.meta['coadd'].append("{} {}{} spectra coadded with {}clip(sig={}).".format(
                        fluxd.shape[0], module[:-1], order, combine, sig))
                elif fluxd.shape[0] == 2:
                    self.meta['coadd'].append("{} {}{} spectra {}ed together.".format(
                        fluxd.shape[0], module[:-1], order, combine))
                else:
                    self.meta['coadd'].append("{} {}{} spectrum included.".format(
                        fluxd.shape[0], module[:-1], order))

    def plot(self, name='spectra', ax=None, errorbar=True,
             label=str.upper, unit=u.Jy, **kwargs):
        """Plot spectra.

        The plot is not cleared, and the x and y labels are changed.

        Parameters
        ----------
        name : string
          The name of the spectra to plot.  The names correspond to
          the `IRSCombine` spectra attributes, e.g., 'trimmed',
          'coadded', 'coma'.  Default is the spectrum with the
          highest processing level ('spectra').  If `name` is 'raw' or
          'nucleus' then the appropriate methods are called (see
          `plot_raw`, `plot_nucleus`).
        ax : matplotlib Axes, optional
          Plot to this axis.
        label : function or string, optional
          A label generator.  Accepts a single parameter, the order
          being plotted.
        unit : astropy Unit, optional
          Convert to these units first.
        **kwargs
          Additional keyword arguments are passed to `matplotlib`'s
          `errorbar`.

        Returns
        -------
        lines : list
          A list of all lines added to the plot.

        """

        import matplotlib.pyplot as plt

        if name == 'raw':
            return self.plot_raw(ax=ax, label=label, unit=unit, **kwargs)
        elif name == 'nucleus':
            return self.plot_nucleus(ax=ax, label=label, unit=unit, **kwargs)

        if ax is None:
            ax = plt.gca()
        rax = ax.twinx()

        if not callable(label):
            def _label(k): return label
        else:
            _label = label

        lines = []
        spectra = getattr(self, name, None)
        assert spectra is not None, '{} does not exist.'.format(name)
        w = []
        s = []
        for k, spec in spectra.items():
            c = u.Jy.to(unit, 1, u.spectral_density(spec['wave'] * u.um))
            w.append(spec['wave'])
            s.append(c * spec['fluxd'])

            if errorbar:
                line = ax.errorbar(spec['wave'], c * spec['fluxd'],
                                   c * spec['err'], label=_label(k),
                                   **kwargs)[0]
            else:
                line = ax.plot(spec['wave'], c * spec['fluxd'],
                               label=_label(k), **kwargs)
            lines.append(line)
        plt.setp(ax, xlabel='Wavelength (μm)')

        w = np.concatenate(w)
        s = np.concatenate(s)
        if unit.is_equivalent('W/(m2 um)'):
            plt.setp(ax, ylabel=r'$F_\lambda$ ({})'.format(unit))
            scale = w / w.mean()
            label = '~λF_λ'
        else:
            plt.setp(ax, ylabel=r'$F_\nu$ ({})'.format(unit))
            scale = w.mean() / w
            label = '~νF_ν'

        if name == 'spectra':
            rax.scatter(w, s * scale, label=label,
                        marker='.', color='0.5', alpha=0.66)

        return lines

    def plot_order_scaling(self, ax=None, **kwargs):
        """Diagnostic plot for order scaling.

        Parameters
        ----------
        ax : matplotlib Axes
          Plot to this axis.
        **kwargs
          Additional keyword arguments are passed to `matplotlib`'s
          `errorbar`.

        """

        import matplotlib.pyplot as plt

        assert self.order_scales is not None, "scale_orders must first be run."

        if ax is None:
            ax = plt.gca()

        for k in ['aploss_corrected', 'coma', 'coadded']:
            if getattr(self, k) is not None:
                lines = self.plot(k)
                break
        else:
            raise RuntimeError(
                'scale_orders has been run, but the prior processing steps are missing.')

        orders = list(self._scale_order_lines.keys())
        for (k1, k2) in zip(orders[:-1], orders[1:]):
            wm = self._scale_order_wave[k1]
            ax.axvline(wm, ls=':', color='k')

            for k, edge in zip((k1, k2), ['long', 'short']):
                mm = minmax(self.spectra[k]['wave'])
                w = np.linspace(mm[0] - 3, mm[1] + 3)
                m, b = self._scale_order_lines[k][edge]
                ax.plot(w, w * m + b, ls='-', color='k', alpha=0.5)
                ax.scatter(wm, wm * m + b, color='k', marker='o')

    def plot_nucleus(self, ax=None, unit=u.Jy, lambda_scale=False, **kwargs):
        """Plot the nucleus spectrum.

        The plot is not cleared, and the x and y labels are changed.

        Parameters
        ----------
        ax : matplotlib Axes
          Plot to this axis.
        unit : astropy Unit, optional
          Convert to these units first.
        lambda_scale : bool, optional
          Multiply by wavelength before plotting.
        **kwargs
          Additional keyword arguments are passed to `matplotlib`'s
          `errorbar`.

        Returns
        -------
        line : list
          The matplotlib line for the nucleus

        """

        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        c = u.Jy.to(unit, 1, u.spectral_density(self.nucleus['wave'] * u.um))
        if lambda_scale:
            assert unit.is_equivalent(
                'W/(m2 um)'), "lambda_scale only makes sense for units of flux per wavelength"
            c *= self.nucleus['wave']

        line = ax.plot(self.nucleus['wave'], c * self.nucleus['fluxd'],
                       **kwargs)

        if u.Jy.is_equivalent(unit):
            ylabel = r'$F_\nu$ ({})'.format(str(unit))
        else:
            ylabel = r'$F_\lambda$ ({})'.format(str(unit))

        plt.setp(ax, xlabel='Wavelength (μm)', ylabel=ylabel)

        return line

    def plot_raw(self, ax=None, label=lambda f: ' '.join(f.split('_')[3:5]),
                 unit=u.Jy, lambda_scale=False, **kwargs):
        """Plot all raw spectra.

        The plot is not cleared, and the x and y labels are changed.

        Parameters
        ----------
        ax : matplotlib Axes, optional
          Plot to this axis.
        label : function or string, optional
          A label generator.  Accepts a single paramter, the file
          name.
        unit : astropy Unit, optional
          Convert to these units first.
        lambda_scale : bool, optional
          Multiply by wavelength before plotting.
        **kwargs
          Additional keyword arguments are passed to `matplotlib`'s
          `errorbar`.

        Returns
        -------
        lines : list
          A list of all lines added to the plot.

        """

        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        if lambda_scale:
            assert unit.is_equivalent(
                'W/(m2 um)'), "lambda_scale only makes sense for units of flux per wavelength"

        lines = []
        for f, spec in sorted(self.raw.items()):
            c = u.Jy.to(unit, 1, u.spectral_density(spec['wavelength'] * u.um))
            if lambda_scale:
                c *= spec['wavelength']
            line = ax.errorbar(spec['wavelength'], c * spec['flux_density'],
                               c * spec['error'], label=label(f), **kwargs)[0]
            lines.append(line)

        if u.Jy.is_equivalent(unit):
            ylabel = r'$F_\nu$ ({})'.format(str(unit))
        else:
            ylabel = r'$F_\lambda$ ({})'.format(str(unit))

        plt.setp(ax, xlabel='Wavelength (μm)', ylabel=ylabel)

        return lines

    def read(self, files, cube=False, **kwargs):
        """Read all data and headers.

        Parameters
        ----------
        files : list
          The file names.

        cube : bool, optional
          Set to true if this was a spectral cube.  This will enable
          alternate exposure info calculations, based on the spectral
          mapping keywords.

        sl, ll : dict, optional
          Constraints for which files to keep.  For example, use
          `sl=dict(column=[2], row=[3, 4, 5])` to keep those particular
          exposures from an IRS map.

        """
        from .. import spice

        self.headers = dict()
        self.raw = dict()
        self.modules = dict()
        for f in files:
            spec, header, module = spice_read(f)

            # test for column, row constraints
            keep = False
            if module in kwargs:
                m = module
            elif module[:2] in kwargs:
                m = module[:2]
            else:
                m = None

            if m is None:
                keep = True
            else:
                keep_column = False
                keep_row = False
                if 'column' in kwargs[m]:
                    if int(header['COLUMN']) in kwargs[m]['column']:
                        keep_column = True
                if 'row' in kwargs[m]:
                    if int(header['ROW']) in kwargs[m]['row']:
                        keep_row = True

                keep = keep_column and keep_row

            if not keep:
                continue

            self.raw[f] = spec
            self.headers[f] = header

            if module not in self.modules:
                self.modules[module] = []

            self.modules[module].append(f)
            self.meta['read_files'].append(f)

        print('IRSCombine read {} files.'.format(len(self.raw)))

        m = self.modules.keys()
        print('IRSCombine found {} supported IRS modules: {}.'.format(
            len(m), ' '.join(m)))

        headers = list(self.headers.values())
        times = [h['DATE_OBS'] for h in headers]
        first = times.index(min(times))
        last = times.index(max(times))

        start_time = times[first]
        dt = float(headers[last]['RAMPTIME']) + \
            float(headers[last]['DEADTIME'])
        stop_time = (cal2time(times[last]) + dt * u.s).isot

        self.header = OrderedDict()
        self.header['object'] = headers[first]['OBJECT']
        self.header['naif id'] = headers[first].get('NAIFID')
        if self.header['naif id'] is None:
            self.header['naif name'] = None
        else:
            self.header['naif name'] = spice.bodc2s(
                int(self.header['naif id']))
        self.header['observer'] = headers[first]['OBSRVR']
        self.header['program id'] = headers[first]['PROGID']
        self.header['start time'] = start_time
        self.header['stop time'] = stop_time

        self.header['exposures'] = {
            'comment': ('Exposure information from input files. '
                        ' Some orders may be missing from final'
                        ' spectrum (see: trim).  itime is total'
                        ' time collecting photons in seconds.'),
            'number': {},
            'itime': {}
        }
        position_angles = defaultdict(list)
        for module, files in self.modules.items():
            k = module.upper()

            n = len(files)
            itime = sum([float(self.headers[f]['RAMPTIME']) for f in files])
            if cube:
                f = files[0]
                self.header['exposures']['cube'] = True
                scale = (int(self.headers[f]['STEPSPAR'])
                         * int(self.headers[f]['STEPSPER'])
                         * int(self.headers[f]['NCYCLES']))
            else:
                scale = 1
            self.header['exposures']['number'][k] = n * scale
            self.header['exposures']['itime'][k] = round(itime * scale, 2)

            for f in files:
                position_angles[module].append(
                    float(self.headers[f]['PA_SLT']))

        if self.header['naif id'] is None:
            g = {}
            self.geom = {'date': cal2time(self.header['start time'])}
        else:
            g = getgeom(self.header['naif id'], Spitzer,
                        self.header['start time'])
            self.geom = g
            self.header['rh'] = '{:.3f}'.format(g.rh)
            self.header['Delta'] = '{:.3f}'.format(g.delta)
            self.header['phase'] = '{:.1f}'.format(g.phase)
            self.header['Sun angle'] = ['{:.1f}'.format(
                g.sangle), 'Projected Sun angle (E of N)']
            self.header['Velocity angle'] = ['{:.1f}'.format(
                g.vangle), 'Projected target velocity angle (E of N)']

        # double str because single str returns numpy.str_
        self.header['RA'] = [
            str(str(Angle(float(headers[first]['RA_SLT']), 'deg'))),
            'Initial slit center RA']
        self.header['Dec'] = [
            str(str(Angle(float(headers[first]['DEC_SLT']), 'deg'))),
            'Initial slit center Dec']

        self.header['position angle'] = {
            'comment': 'Slit position angle (E of N)'
        }
        for module, pa in position_angles.items():
            self.header['position angle'][module.upper()] = (
                '{:.2f}'.format(float(np.mean(pa)) * u.deg))

        c = SkyCoord(self.header['RA'][0],
                     self.header['Dec'][0], 1 * u.Mpc, frame='icrs')
        self.header['lambda'] = [
            str(str(c.heliocentrictrueecliptic.lon)), 'Ecliptic longitude']
        self.header['beta'] = [
            str(str(c.heliocentrictrueecliptic.lat)), 'Ecliptic latitude']

        # self.header['R_Spitzer'] = ([float(headers[first]['SPTZR_' + x]) for x in 'XYZ'] * u.km, 'Observatory heliocentric rectangular coordinates')
        xyz = [float(headers[first]['SPTZR_' + x]) * u.km for x in 'XYZ']
        self.header['R_Spitzer'] = dict(
            [(k, str(v)) for k, v in zip('xyz', xyz)])

    def delete_nucleus(self):
        """Delete the model nucleus."""
        self.nucleus = None
        self.coma = None
        self.aploss_corrected = None
        self.meta['subtract_nucleus'] = []
        print('IRSCombine: Model nucleus removed.')

    def scale_orders(self, fixed, dlam=0.1):
        """Order-to-order scaling with linear interpolation.

        Within this context, SH and LH are treated as single orders.

        Parameters
        ----------
        fixed : string
          The order to keep fixed, e.g., 'll2'.
        dlam : float
          A scale factor to determine the number of wavelengths to use
          for linear fitting at the edge of each order, e.g., for
          `dlam=0.1` at 13.2 μm, `scale_orders` would use 11.9–13.2 μm
          (1.3 μm).

        """

        assert self.coadded is not None, "Spectra must first be coadded (even if there is only one spectrum per module)."

        stitching_order = ['sl2', 'sl3', 'sl1',
                           'sh', 'll2', 'll3', 'll1', 'lh']
        self.order_scales = dict()
        self._scale_order_lines = OrderedDict()
        self._scale_order_edges = dict()
        self._scale_order_wave = dict()
        self.order_scaled = None  # reset for self.spectra property
        for k in stitching_order:
            if k in self.spectra.keys():
                self._scale_order_lines[k] = dict()
                self._scale_order_edges[k] = dict()
                self.order_scales[k] = 1.0

                w = self.spectra[k]['wave']
                f = self.spectra[k]['fluxd']
                e = self.spectra[k]['err']

                # short wavlength edge
                wrange = (w.min(), w.min() * (1 + dlam))
                i = between(w, wrange) * np.isfinite(f)
                self._scale_order_lines[k]['short'] = linefit(
                    w[i], f[i], e[i], (1.0, 0))[0]
                self._scale_order_edges[k]['short'] = w[i].min()

                # long wavlength edge
                wrange = (w.max() * (1 - dlam), w.max())
                i = between(w, wrange) * np.isfinite(f)
                self._scale_order_lines[k]['long'] = linefit(
                    w[i], f[i], e[i], (1.0, 0))[0]
                self._scale_order_edges[k]['long'] = w[i].max()

        # first pass on scale factors
        orders = list(self._scale_order_lines.keys())
        for (k1, k2) in zip(orders[:-1], orders[1:]):
            m1, b1 = self._scale_order_lines[k1]['long']
            m2, b2 = self._scale_order_lines[k2]['short']
            wm = (self._scale_order_edges[k1]['long']
                  + self._scale_order_edges[k2]['short']) / 2
            self._scale_order_wave[k1] = wm
            self.order_scales[k1] = (m2 * wm + b2) / (m1 * wm + b1)

        # second pass: incorporate scale factors into the shortest
        # wavelength order, then next shortest, and so on
        for i, k in enumerate(orders):
            for j in range(i + 1, len(orders)):
                self.order_scales[k] *= self.order_scales[orders[j]]

        # final pass: scale everything to user requested order, and
        # actually apply the scale factors to the data
        spectra = self.spectra
        self.order_scaled = dict()
        fixed_scale = self.order_scales[fixed]
        for k in orders:
            self.order_scaled[k] = dict()
            for kk, vv in spectra[k].items():
                self.order_scaled[k][kk] = vv.copy()

            self.order_scales[k] /= fixed_scale
            self.order_scaled[k]['fluxd'] *= self.order_scales[k]
            self.order_scaled[k]['err'] *= self.order_scales[k]

            self.meta['scale_orders'].append('{} scaled by {}'.format(
                k, self.order_scales[k]))

    def scale_spectra(self, sl1=(9, 11), sl2=(6, 7), ll1=(25, 30),
                      ll2=(16, 18)):
        """Generate scale factors for combining multiple exposures.

        Parameters
        ----------
        sl2, sl1, ll2, ll1 : two-element tuple, optional
          Wavelength range with which to derive the scale factors.

        """

        ranges = dict(sl1=sl1, sl2=sl2, ll1=ll1, ll2=ll2)

        self.scales = dict()

        self.meta['scale_spectra'] = {}
        for module, files in self.modules.items():
            bandfluxd = dict()
            for f in files:
                i = between(self.raw[f]['wavelength'], ranges[module])
                bandfluxd[f] = meanclip(self.raw[f]['flux_density'][i])
            mfluxd = np.nanmedian(list(bandfluxd.values()))
            for f in files:
                self.scales[f] = float(mfluxd / bandfluxd[f])

            self.meta['scale_spectra']['sl1'] = self.scales

        print('IRSCombine generated {} scale factors.'.format(len(self.scales)))

    def subtract_nucleus(self, R, Ap, target=None, **kwargs):
        """Generate a model NEATM to subtract from the spectrum.

        Delete this model with `self.delete_nucleus`.

        Parameters
        ----------
        R : Quantity
          The radius.
        Ap : float
          The geometric albedo.
        target : string, optional
          Use this target name, instead of whatever was gleaned from
          the IRS file.
        **kwargs
          `mskpy.models.NEATM` keyword arguments.

        """

        if self.nucleus is not None:
            self.delete_nucleus()

        target = self.header['naif id'] if target is None else target

        model = NEATM(R * 2, Ap, **kwargs)
        wave = np.logspace(np.log10(5), np.log10(40), 100) * u.um
        fluxd = model.fluxd(self.geom, wave, unit=u.Jy)

        self.nucleus = Table((wave, fluxd), names=['wave', 'fluxd'])
        self.nucleus['wave'].meta['unit'] = 'um'
        self.nucleus['fluxd'].meta['unit'] = 'Jy'
        self.nucleus.meta['R'] = R
        self.nucleus.meta['Ap'] = Ap
        for k, v in kwargs.items():
            self.nucleus.meta[k] = v

        self.meta['subtract_nucleus'] = dict(R=str(R), Ap=Ap, **kwargs)

        model = splrep(self.nucleus['wave'], self.nucleus['fluxd'])
        spec = self.spectra
        self.coma = dict()
        for k, v in spec.items():
            self.coma[k] = {}
            for kk, vv in spec[k].items():
                self.coma[k][kk] = vv.copy()
            f = splev(self.coma[k]['wave'], model)
            self.coma[k]['fluxd'] -= f

        print('IRSCombine generated and subtracted a model nucleus.')

    def subtract_other_nucleus(self, wave, fluxd, comment=None):
        """Subtract the provided model nucleus.

        Delete this model with `self.delete_nucleus`.

        Parameters
        ----------
        wave : Quantity
            Model wavelengths.

        fluxd : Quantity
            Model flux density.

        """

        if self.nucleus is not None:
            self.delete_nucleus()

        self.nucleus = Table((wave, fluxd), names=['wave', 'fluxd'])
        self.nucleus['wave'].meta['unit'] = 'um'
        self.nucleus['fluxd'].meta['unit'] = 'Jy'
        self.nucleus.meta['comment'] = comment

        self.meta['subtract_nucleus'] = 'other'

        model = splrep(self.nucleus['wave'].data, self.nucleus['fluxd'].data)
        spec = self.spectra
        self.coma = dict()
        for k, v in spec.items():
            self.coma[k] = {}
            for kk, vv in spec[k].items():
                self.coma[k][kk] = vv.copy()
            f = splev(self.coma[k]['wave'], model)
            self.coma[k]['fluxd'] -= f

        print('IRSCombine subtracted an externally provided model nucleus.')

    def slitloss_correct(self, modules=['sl1', 'sl2', 'sl3', 'll1', 'll2', 'll3']):
        path = config.get('irs', 'spice_path')
        h = list(self.headers.values())[0]
        calset = h['CAL_SET'].strip("'").strip('.A')

        if self.aploss_corrected is not None:
            spec = self.aploss_corrected
        else:
            spec = self.spectra

        self.slitloss_corrected = dict()
        corrected_modules = []
        for k in spec.keys():
            if k in modules:
                corrected_modules.append(k)
                fn = 'b{}_slitloss_convert.tbl'.format(module2channel[k[:2]])
                tab = ascii.read(os.path.join(path, 'cal', calset, fn))
                slcf = np.interp(spec[k]['wave'], tab['wavelength'],
                                 tab['correction'])

                self.slitloss_corrected[k] = dict()
                for kk, vv in spec[k].items():
                    self.slitloss_corrected[k][kk] = vv
                self.slitloss_corrected[k]['fluxd'] *= slcf
                e = np.maximum(self.slitloss_corrected[k]['err'] * slcf,
                               self.minimum_uncertainty)
                self.slitloss_corrected[k]['err'] = e
            else:
                self.slitloss_corrected[k] = spec[k]

        self.meta['slitloss_correct'] = [
            'Slit loss corrected for uniform sources using IRS pipeline correction: {}'
            .format(','.join(corrected_modules))]

    def shape_correct(self, correction):
        """Correct the shape of the coma spectra.

        This module should be executed after `subtract_nucleus`,
        `aploss_correct`, and `slitloss_correct` (if using).

        Parameters
        ----------
        correction : dict of functions
          Keys are order names, e.g., 'sl2', and values are functions
          that return multiplicative correction factors given
          wavelength, e.g., a `scipy.interpolate.UnivariateSpline`
          instance.

        """

        assert self.coma is not None
        if self.slitloss_corrected is not None:
            spec = self.slitloss_corrected
        elif self.aploss_corrected is not None:
            spec = self.aploss_corrected
        else:
            spec = self.coma

        self.slitloss_corrected = dict()
        for k in spec.keys():
            self.slitloss_corrected[k] = dict()
            for kk, vv in spec[k].items():
                self.slitloss_corrected[k][kk] = vv
            w = self.slitloss_corrected[k]['wave']
            self.slitloss_corrected[k]['fluxd'] *= correction[k](w)
            e = np.maximum(
                self.slitloss_corrected[k]['err'] * correction[k](w),
                self.minimum_uncertainty)
            self.aploss_corrected[k]['err'] = e

        self.meta['shape_correct'] = [
            'Shape corrected with user provided data.']

    def trim(self, **kwargs):
        """Trim orders at given limits.

        Parameters
        ----------
        sl3, sl2, sl1, ll3, ll2, ll1 : two-element tuples, optional
          Keep wavelengths within these ranges.

        """

        assert self.coadded is not None, 'Must run `coadd()` first.'

        tr = dict(sl1=[0, 13.5], sl2=[0, 100], sl3=[0, 100],
                  ll2=[0, 19.55], ll3=[19.55, 100], ll1=[21.51, 35])
        tr.update(kwargs)

        self.trimmed = self.coadded.copy()
        self.meta['trim'] = {}
        delete_keys = []
        for k in self.trimmed.keys():
            wrange = tr[k]
            self.meta['trim'][k] = wrange

            if wrange[0] >= wrange[1]:
                # delete this order
                delete_keys.append(k)
                continue

            i = between(self.trimmed[k]['wave'], wrange)
            for j in ('wave', 'fluxd', 'err'):
                self.trimmed[k][j] = self.trimmed[k][j][i]

        for k in delete_keys:
            del self.trimmed[k]
            print('IRSCombine trim: deleting', k)

    def write(self, filename, meta={}, overwrite=False):
        """Write the spectra to a single file.

        Parameters
        ----------
        filename : string
          The name of the file to which to save the data.
        meta : dict, optional
          Additional meta data to write to the file.
        overwrite : bool, optional
          Overwrite existing files.

        """

        assert self.coadded is not None, "Spectra must be processed at least through `coadd()`."

        header = self.header.copy()
        header.update(meta)

        i = np.argsort([s['wave'].min() for s in self.spectra.values()])
        keys = np.array(list(self.spectra.keys()))[i]

        if self.nucleus is not None:
            nucleus_interp = splrep(
                self.nucleus['wave'], self.nucleus['fluxd'])

        wave = []
        fluxd = []
        err = []
        orders = []
        scales = []
        nucleus = []
        for k in keys:
            i = np.isfinite(self.spectra[k]['fluxd'])
            wave.extend(self.spectra[k]['wave'][i])
            fluxd.extend(self.spectra[k]['fluxd'][i])
            err.extend(self.spectra[k]['err'][i])
            orders.extend([k.upper()] * i.sum())
            scales.extend(np.ones(i.sum()) * self.order_scales[k])
            if self.nucleus is None:
                nucleus.extend(np.zeros(i.sum()))
            else:
                nf = splev(self.spectra[k]['wave'][i], nucleus_interp)
                nucleus.extend(nf)

        tab = Table((wave, fluxd, err, orders, scales, nucleus),
                    names=['wave', 'fluxd', 'err', 'order',
                           'scales', 'nucleus'])
        tab['wave'].meta['unit'] = 'um'
        tab['fluxd'].meta['unit'] = 'Jy'
        tab['err'].meta['unit'] = 'Jy'
        tab['nucleus'].meta['unit'] = 'Jy'
        for k in ['wave', 'fluxd', 'err', 'nucleus', 'scales']:
            tab[k].format = "{:#.5g}"

        tab.meta.update(self.meta)
        tab.meta.update(header)
        tab.write(filename, format='ascii.ecsv', overwrite=overwrite)

    def zap(self, **kwargs):
        """Remove data points from the coadded spectra.

        Parameters
        ----------
        kwargs
          The orders and wavelengths to remove, e.g., `ll2=[15.34, 15.42]`.

        """

        for order, waves in kwargs.items():
            assert order in self.coadded
            for w in waves:
                i = nearest(self.coadded[order]['wave'], w)
                j = list(range(len(self.coadded[order]['wave'])))
                del j[i]
                w = self.coadded[order]['wave'][i]
                self.coadded[order]['wave'] = self.coadded[order]['wave'][j]
                self.coadded[order]['fluxd'] = self.coadded[order]['fluxd'][j]
                self.coadded[order]['err'] = self.coadded[order]['err'][j]
                print('IRSCombine zapped {} from {}'.format(w, order))
                self.meta['zap'].append('Zapped {} from {}'.format(
                    w, order))


def irsclean(im, h, bmask=None, maskval=28672,
             rmask=None, func=None, nan=True, sigma=None, box=3,
             **fargs):
    """Clean bad pixels from an IRS image.

    Parameters
    ----------
    im : ndarray
      The image.
    h : dict-like
      FITS header keywords from the original data file.
    bmask : ndarray, optional
      The SSC pipeline BMASK array.
    maskval : int, optional
      Bitmask value applied to BMASK array to generate a bad pixel
      map.  These values will be removed from `bmask` and returned.
    rmask : ndarray, optional
      The rogue mask array.
    func : function, optional
      Use this function to clean the image: first argument is the
      image to clean, the second is the mask (`True` for each bad
      pixel).  The default is `image.fixpix`.
    nan : bool, optional
      Set to `True` to also clean any pixels set to NaN.
    sigma : float, optional
      Set to sigma clip the image using a filter of width `box` and
      clipping at `sigma`-sigma outliers.
    box : int, optional
      The size of the filter for sigma clipping.
    **fargs
      Additional keyword arguments are pass to `func`.

    Returns
    -------
    cleaned : ndarray
      The cleaned data.
    h : dict-like
      The updated header.
    new_mask : ndarray, optional
      The cleaned mask.

    """

    cleaner = fixpix if func is None else func

    mask = np.zeros_like(im, bool)
    if nan:
        mask += ~np.isfinite(im)
    if bmask is not None:
        mask += (bmask & maskval).astype(bool)
        new_mask = bmask & (32767 | maskval)
    if rmask is not None:
        mask += rmask.astype(bool)

    if sigma is not None:
        stdev = nd.generic_filter(im, np.std, size=box)
        m = nd.median_filter(im, size=box)
        mask += ((im - m) / stdev) > sigma

    h.add_history('Cleaned with mskpy.instruments.spitzer.irsclean.')
    h.add_history('irsclean: function={}, arguments={}'.format(
        str(cleaner), str(fargs)))

    cleaned = cleaner(im, mask, **fargs)
    if bmask is None:
        return cleaned, h
    else:
        return cleaned, h, new_mask


def irsclean_files(files, outfiles, uncs=True, bmasks=True,
                   maskval=16384, rmasks=True, func=None, nan=True,
                   sigma=None, box=3, **fargs):
    """Clean bad pixels from a list of IRS files.

    For automatic rogue mask file name gleaning, this function
    requires that irs.rogue_masks_path is set in mskpy.cfg to the
    location of the rogue masks files from the Spitzer Science Center.

    Parameters
    ----------
    files : array of strings
      A list of image names to clean.
    outfiles : array-like
      Save cleaned images to these files.  Existing files will be
      overwritten.  Uncertainty file names will be based on `outfiles`.
    uncs : list or bool, optional
      Also clean these uncertainty arrays.  Set to `True` and the
      uncertainty filename will be guessed.  Otherwise set to `False`.
    bmasks : ndarray, optional
      The SSC pipeline BMASK arrays.  Same format as `uncs`.
    maskval : int, optional
      Bitmask value applied to BMASK array to generate bad pixel
      maps.
    rmasks : ndarray, optional
      The rogue mask arrays.  Same format as `uncs`.  When `True` but
      the IRS campaign is not in the `capaign2rogue` array (e.g., for
      early release observations), then no rogue mask will be used.
    func : function, optional
      Use this function to clean the images: first argument is the
      image to clean, the second is the mask (`True` for each bad
      pixel).  The default is `image.fixpix`.
    nan : bool, optional
      Set to `True` to also clean any pixels set to NaN.
    sigma : float, optional
      Set to sigma clip the image using a filter of width `box` and
      clipping at `sigma`-sigma outliers.
    box : int, optional
      The size of the filter for sigma clipping.
    **fargs
      Additional keyword arguments are pass to `func`.

    """

    def file_generator(in_list, optional_list, replace_string):
        for i in range(len(in_list)):
            if optional_list is True:
                f = in_list[i].replace('_bcd', replace_string)
            elif np.iterable(optional_list):
                f = optional_list[i]
            else:
                f = None

            if f is None:
                yield None, None
            else:
                yield f, fits.getdata(f)

    def rmask_file_generator(in_list, optional_list):
        path = config.get('irs', 'rogue_masks_path')
        for i in range(len(in_list)):
            if optional_list is True:
                h = fits.getheader(in_list[i])
                if h['CAMPAIGN'] in campaign2rogue:
                    f = 'b{}_rmask_{}.fits'.format(
                        h['CHNLNUM'], campaign2rogue[h['CAMPAIGN']])
                    f = os.path.join(path, f)
                else:
                    f = None
            elif np.iterable(optional_list):
                f = optional_list[i]
            else:
                f = None

            if f is None:
                yield None, None,
            else:
                yield f, fits.getdata(f)

    unc_files = file_generator(files, uncs, '_func')
    bmask_files = file_generator(files, bmasks, '_bmask')
    rmask_files = rmask_file_generator(files, rmasks)

    for i in range(len(files)):
        unc_file, unc = next(unc_files)
        bmask_file, bmask = next(bmask_files)
        rmask_file, rmask = next(rmask_files)

        im, h = fits.getdata(files[i], header=True)
        cleaned = irsclean(im, h, bmask=bmask, maskval=maskval,
                           rmask=rmask, func=func, sigma=sigma,
                           box=box, **fargs)
        fits.writeto(outfiles[i], cleaned[0], cleaned[1], clobber=True)

        if len(cleaned) == 3:
            # bmask was updated, save it
            bmask = cleaned[2]  # update array for use with unc cleaning
            h_bmask = fits.getheader(bmask_file)
            h_bmask.add_history(
                'Updated with mskpy.instruments.spitzer.irsclean')
            fits.writeto(bmask_file, bmask, h_bmask, clobber=True)

        if unc is not None:
            if '_bcd' in outfiles[i]:
                outf = outfiles[i].replace('_bcd', '_func')
            elif outfiles[i].endswith('.fits'):
                outf = outfiles[i].replace('.fits', '_func.fits')
            else:
                outf = outfiles[i] + '_func.fits'

            h = fits.getheader(unc_file)

            # do not use sigma clipping with unc array!
            cleaned = irsclean(unc, h, bmask=bmask, maskval=maskval,
                               rmask=rmask, func=func, sigma=None, **fargs)

            fits.writeto(outf, cleaned[0], cleaned[1], clobber=True)


def moving_wcs_fix(files, ref=None):
    """Correct IRS FITS WCS for the motion of the targeted moving object.

    Parameters
    ----------
    files : array of strings
      A list of files to update.  The files are updated in place.
    ref : tuple
      The "reference" RA and Dec of the target expressed as a tuple:
      `(ra_ref, dec_ref)`.  This is usually the position of the moving
      target at the start of the IRS observation.  The difference
      between ra_ref, dec_ref and the RA_REF, DEC_REF in the FITS
      headers is the motion of the target.  Set ref to `None` to use
      RA_REF and DEC_REF from the first file in the file list as the
      initial position.  [units: degrees]

    """

    assert np.iterable(files), "files must be an array of file names"

    ra_ref0, dec_ref0 = ref

    for f in files:
        im, h = fits.getdata(f, header=True)
        ra_ref1 = h["RA_REF"]
        dec_ref1 = h["DEC_REF"]

        # I found CRVALx missing in some LH files
        if h.get("CRVAL1") is not None:
            crval1, crval2 = spherical_coord_rotate(
                ra_ref1, dec_ref1, ra_ref0, dec_ref0,
                h["CRVAL1"], h["CRVAL2"])
        rarqst, decrqst = spherical_coord_rotate(
            ra_ref1, dec_ref1, ra_ref0, dec_ref0,
            h["RA_RQST"], h["DEC_RQST"])

        raslt, decslt = spherical_coord_rotate(
            ra_ref1, dec_ref1, ra_ref0, dec_ref0,
            h["RA_SLT"], h["DEC_SLT"])

        print("{} moved {:.3f} {:.3f}".format(f, (ra_ref1 - ra_ref0) * 3600.,
                                              (dec_ref1 - dec_ref0) * 3600.))

        if h.get("CRVAL1") is not None:
            h["CRVAL1"] = crval1
            h["CRVAL2"] = crval2
        h["RA_RQST"] = rarqst
        h["RA_SLT"] = raslt
        h["DEC_RQST"] = decrqst
        h["DEC_SLT"] = decslt
        h.add_history(
            "WCS updated for moving target motion with mskpy.instrum3ents.spitzer.moving_wcs_fix")
        fits.update(f, im, h)


def spice_read(filename, drop_fainter_order=True):
    """Read in an IRS spectrum and header from a SPICE file.

    Parameters
    ----------
    filename : string
      The name of the file to read.
    drop_fainter_order : bool, optional
      For AORs observing with the Center field of view, drop the
      fainter order.

    Returns
    -------
    spec : astropy Table
      The data.
    h : dictionary
      The header.
    module : string
      The module name of the primary field of view, e.g., sl1.

    """

    IGNORE_HEADER_PREFIXES = ('\\char HISTORY',
                              '\\char COMMENT')

    h = dict()
    with open(filename, 'r') as inf:
        for line in inf:
            if not line.startswith('\\char '):
                continue
            elif line.startswith(IGNORE_HEADER_PREFIXES):
                continue
            elif '=' not in line:
                continue

            line = line[6:]
            k, vc = line.partition('=')[::2]
            v, c = vc.partition('/')[::2]
            h[k.strip()] = v.strip(' \'')

    spec = ascii.read(filename)

    module = 'unknown'
    if 'Short-Lo' in h['FOVNAME']:
        module = 'sl'
    elif 'Long-Lo' in h['FOVNAME']:
        module = 'll'
    else:
        raise UserWarning('Only SL and LL are presently supported.')

    if '1st_Order' in h['FOVNAME']:
        module += '1'
    elif '2nd_Order' in h['FOVNAME']:
        module += '2'
    else:
        # AOR used slit center, try to guess which was the primary order
        ranges = dict(sl1=(9, 11), sl2=(6, 7), ll1=(25, 30), ll2=(16, 18))
        f1 = np.median(spec['flux_density'][between(
            spec['wavelength'], ranges[module + '1'])])
        f2 = np.median(spec['flux_density'][between(
            spec['wavelength'], ranges[module + '2'])])
        if f1 > f2:
            module += '1'
            if drop_fainter_order:
                spec = spec[spec['order'] == 1]
        else:
            module += '2'
            if drop_fainter_order:
                spec = spec[spec['order'] > 1]

    return spec, h, module


def irs_summary(files):
    """Summarize a set of IRS spectra produced with SPICE.

    Primarily for selecting DCEs to use in IRSCombine.

    Parameters
    ----------
    files : list
      The list of files to check.

    """

    tab = Table(names=['file', 'date', 'module', 'expid', 'dce',
                       'column', 'row', 'fluxd'],
                dtype=['S256', 'S32', 'S4', int, int, int, int, float])

    ranges = dict(sl1=(9, 11), sl2=(6, 7), ll1=(25, 30), ll2=(16, 18))

    for f in files:
        spec, h, module = spice_read(f)
        i = between(spec['wavelength'], ranges[module])
        try:
            tab.add_row([f, h['DATE_OBS'], module, h['EXPID'], h['DCENUM'],
                         h['COLUMN'] if 'COLUMN' in h else -1,
                         h['ROW'] if 'ROW' in h else -1,
                         np.median(spec['flux_density'][i])])
        except Exception as e:
            print('Error processing {}'.format(f))
            raise e

    tab.sort('date')
    tab.pprint(max_lines=-1, max_width=-1)
    return tab


def main():
    import argparse
    from glob import glob
    import json
    import matplotlib.pyplot as plt
    from ..graphics import nicelegend
    from mskpy import hms2dh

    parser = argparse.ArgumentParser(
        description='Reduce Spitzer/IRS comet data.')
    parser.add_argument('config', help='configuration file name')
    parser.add_argument('-i', action='store_true',
                        help='show plots (interactive mode)')
    parser.add_argument('--debug', action='store_true',
                        help='enter python debugger after execution')
    parser.add_argument('--no-skip', action='store_false', dest='skip',
                        help='Ignore "skip" configuration parameter.')
    parser.add_argument('-n', type=int,
                        help='reduce config item N, 0-based')
    parser.add_argument('--reduced-by', help='Name of a scientist.')
    parser.add_argument('--summary', action='store_true',
                        help='only show the data summary')

    args = parser.parse_args()

    lines = [line for line in open(args.config, 'r')
             if not line.startswith('#')]
    config_sets = json.loads(''.join(lines))['rx']

    for n, config in enumerate(config_sets):
        if config.get('skip', False) and args.skip:
            continue

        if (args.n is not None) and (n != args.n):
            continue

        if isinstance(config['files'], list):
            files = sum([glob(g) for g in config['files']], [])
        else:
            files = glob(config['files'])
        if len(files) == 0:
            raise ValueError("No data to read.")

        rx = IRSCombine(files, **config.get('IRSCombine', {}))

        dh = '{:.2f}'.format(hms2dh(rx.geom['date'].iso[11:]) / 24)[1:]
        date = rx.geom['date'].iso[:10].replace('-', '') + dh
        if rx.header['naif id'] is None:
            pfx = config.get('prefix').format(date=date)
        else:
            if np.dot(rx.geom.rt, rx.geom.vt) < 0:
                pre_post = 'pre'
            else:
                pre_post = 'post'
            rh = "{pre_post}{g[rh].value:.3f}".format(
                pre_post=pre_post, g=rx.geom)
            pfx = config.get('prefix').format(g=rx.geom, rh=rh, date=date)

        tab = irs_summary(files)
        tab.write(pfx + '-summary.csv', format="ascii.ecsv", overwrite=True)
        if args.summary:
            continue

        irsc_opts = config.get('IRSCombine', {})
        scale_orders_opts = config.get('scale_orders', {})

        meta = {}
        if args.reduced_by is not None:
            meta['Reduced by'] = args.reduced_by

        rx.scale_spectra(**config.get('scale_spectra', {}))
        rx.coadd(**config.get('coadd', {}))

        opts = config.get('zap', {})
        if len(opts) > 0:
            rx.zap(**opts)

        rx.trim(**config.get('trim', {}))

        if "slitloss_correct" in config:
            rx.slitloss_correct(**config.get('slitloss_correct', {}))

        if config.get('subtract_nucleus') is not None and config.get('subtract_other_nucleus') is not None:
            raise ValueError(
                'Only one of subtract_nucleus and subtract_other_nucleus may be specified.')

        opts = config.get('subtract_nucleus', {})
        try:
            R = u.Quantity(opts.pop('R'))
            assert R > 0 * u.km
            Ap = opts.pop('Ap')
            rx.subtract_nucleus(R, Ap, **opts)
        except (AssertionError, KeyError):
            pass

        opts = config.get('subtract_other_nucleus')
        if opts is not None:
            if not all([k in opts.keys() for k in ['wave', 'fluxd']]):
                raise KeyError(
                    'subtract_other_nucleus requires wave (μm) and fluxd (Jy)')
            wave = u.Quantity(opts.pop('wave'), 'um')
            fluxd = u.Quantity(opts.pop('fluxd'), 'Jy')
            rx.subtract_other_nucleus(wave, fluxd, **opts)

        opts = config.get('scale_orders', {})
        if len(opts) > 0:
            anchor = opts.pop('anchor')
            rx.scale_orders(anchor, **opts)

        rx.write(pfx + '.csv', meta=meta, overwrite=True)

        opts = config.get('plot', {})
        figtype = opts.pop('figtype', 'png')

        fig = plt.figure(1)
        plt.clf()
        plt.minorticks_on()
        rx.plot('raw')
        plt.setp(plt.gca(), **opts)
        plt.draw()
        plt.savefig(pfx + '-raw.' + figtype)

        fig = plt.figure(2)
        plt.clf()
        plt.minorticks_on()
        rx.plot('coadded')
        if rx.nucleus is not None:
            rx.plot('nucleus', ls='--', label='nucleus')

        nicelegend(loc='lower right')
        plt.setp(plt.gca(), **opts)
        plt.draw()
        plt.savefig(pfx + '-coadded.' + figtype)

        fig = plt.figure(3)
        plt.clf()
        plt.minorticks_on()
        rx.plot_order_scaling()
        plt.setp(plt.gca(), **opts)
        plt.draw()
        plt.savefig(pfx + '-scaling.' + figtype)

        fig = plt.figure(4)
        plt.clf()
        plt.minorticks_on()
        rx.plot()
        plt.setp(plt.gca(), **opts)
        plt.draw()
        plt.savefig(pfx + '.' + figtype)

        if args.debug:
            import pdb
            pdb.set_trace()

        if args.i:
            plt.show()


if __name__ == "__main__":
    main()


# update module docstring
autodoc(globals())
del autodoc
