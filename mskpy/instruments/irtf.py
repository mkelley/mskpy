# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
irtf --- NASA IRTF instruments.
===============================

   Classes
   -------
   BASS
   MIRSI
   SpeX
   SpeXPrism60

"""

import numpy as np
import astropy.units as u

try:
    from ..ephem import Earth
except ImportError:
    Earth = None

from .instrument import Instrument, Camera
from .instrument import CircularApertureSpectrometer, LongSlitSpectrometer

__all__ = [
    'BASS',
    'MIRSI',
    'SpeX',
    'SpeXPrism60',
]

class BASS(CircularApertureSpectrometer):
    """Broadband Array Spectrograph System.
    """

    def __init__(self):
        waves = [
            3.02961,   3.13797,   3.24272,   3.34419,   3.63162,   3.7225 ,
            3.89791,   3.98272,   4.06576,   4.53217,   4.74822,   4.81809,
            4.88695,   4.95486,   5.02185,   5.28133,   5.34423,   7.27842,
            7.46219,   7.64154,   7.98818,   8.15597,   8.32038,   8.48161,
            8.63982,   8.79519,   8.94787,   9.09798,   9.24565,   9.39101,
            9.53414,   9.67516,   9.81416,   9.95121,  10.0864 ,  10.2198 ,
            10.4815 ,  10.6099 ,  10.7368 ,  10.8623 ,  10.9862 ,  11.1089 ,
            11.2301 ,  11.3501 ,  11.5863 ,  11.7026 ,  11.8178 ,  11.9318 ,
            12.0448 ,  12.1567 ,  12.2677 ,  12.3776 ,  12.4865 ,  12.5945 ,
            12.7016 ,  12.8078 ,  12.9131 ,  13.0176 ,  13.1212 ,  13.224  
        ] * u.um
        CircularApertureSpectrometer.__init__(
            self, waves, 2.0 * u.arcsec, Earth)

class MIRSI(Instrument):
    """Mid-Infrared Spectrometer and Imager.

    Attributes
    ----------
    imager : `Camera` for imaging mode.
    sp10r200 : `LongSlitSpectrometer` for 10-micron spectroscopy.
    sp20r100 : `LongSlitSpectrometer` for 20-micron spectroscopy.
    mode : The current MIRSI mode (see examples).

    Methods
    -------
    standard_fluxd : Flux density of a standard star in a MIRSI filter.
    fluxd : Flux density of a spectrum through a MIRSI filter.

    Examples
    --------

    """

    shape = (240, 320)
    ps = 0.265 * u.arcsec
    location = Earth

    # Central wavelengths
    filters = np.r_[4.9, 7.7, 8.7, 9.8, 10.6, 11.6, 12.3,
                    18.4, 20.6, 24.4] * u.um
    # Width of the filters (in percent)
    width_per = np.r_[21.0, 9.0, 8.9, 9.4, 46.0, 9.9, 9.6,
                      8.0, 37.4, 7.9]
    # Half width of the filters
    hwidth = (filters * width_per * 0.01) / 2.

    def __init__(self):
        self.imager = Camera(self.filters, self.shape, self.ps,
                             location=self.location)

        self.sp10r200 = LongSlitSpectrometer(10.5 * u.um, self.shape, self.ps,
                                             2.25, 0.022 * u.um, R=200,
                                             location=self.location)

        self.sp20r100 = LongSlitSpectrometer(21.5 * u.um, self.shape, self.ps,
                                             4.5, 0.028 * u.um, R=100,
                                             location=self.location)

        self._mode = 'imager'

    @property
    def mode(self):
        if self._mode in ['imager', 'sp10r200', 'sp20r100']:
            return self.__dict__[self._mode]
        else:
            raise KeyError("Invalid mode: {:}".format(self._mode))

    def sed(self, *args, **kwargs):
        """Spectral energy distribution of a target.

        Parameters
        ----------
        *args
        **kwargs
          Arguments and keywords depend on the current MIRSI mode.

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
          Arguments and keywords depend on the current MIRSI mode.

        Returns
        -------
        lc : astropy Table

        """
        return self.mode.lightcurve(*args, **kwargs)

    def filter_atran(self, wave, airmass, pw='2.5'):
        """Atmospheric transmission through a filter.

        Diane Wooden method.

        Parameters
        ----------
        wave : float or array
          Filter central wavelengths (see `self.filters`).
        airmass : float
          Airmass to compute.
        pw : string, optional
          Precipitable water vapor.  Must match a saved file.

        Returns
        -------
        tr : float or array
          The filter transmissions.

        """

        from .. import util
        from ..calib import dw_atran

        _w = np.r_[wave]
        tr = np.zeros_like(_w)

        for i in range(len(_w)):
            j = self.filters.value == _w[i]
            bp = np.r_[self.filters.value[j] - self.hwidth.value[j],
                       self.filters.value[j] + self.hwidth.value[j]]

            fw = np.linspace(bp[0] - 1, bp[1] + 1, 10000)
            ft = fw * 0.0
            ft[util.between(fw, bp)] = 1.0

            tr[i] = dw_atran(airmass, fw, ft, pw=pw)

        return tr

    def fluxd(self, sw, sf, wave):
        """Flux density of a spectrum through a filter.

        Parameters
        ----------
        sw : Quantity
          The wavelenths of the spectrum.
        sf : Quantity
          The spectrum (flux per unit wavelength).
        wave : float or array
          The central wavelength of the filters for which the flux should
          be computed.

        Returns
        -------
        flux : Quantity
          The computed flux density of the spectrum through each filter.

        """
        from .. import calib
        from .. import util

        _w = np.r_[wave]
        flux = u.Quantity(np.zeros_like(_w), sf.unit)

        for i in range(len(_w)):
            j = self.filters.value == _w[i]
            bp = np.r_[self.filters.value[j] - self.hwidth.value[j],
                       self.filters.value[j] + self.hwidth.value[j]]

            fw = np.linspace(bp[0] - 1, bp[1] + 1, 1000)
            ft = fw * 0.0
            ft[util.between(fw, bp)] = 1.0

            result = util.bandpass(sw.to(u.um).value,
                                   sf.value,
                                   fw=fw, ft=ft, s=0)
            flux[i] = result[1] * sf.unit

        return flux

    def standard_fluxd(self, star, wave, unit=u.Unit('W/(m2 um)')):
        """Flux density of a standard star in a MIRSI filter.

        Parameters
        ----------
        star : str
          The name of a star, passed on to `calib.cohenstandard()`.
        wave : float or array
          The central wavelength of the filters for which the flux should
          be computed.
        units : str, optional
          The units of the output.  See `cohenstandard()`.

        Returns
        -------
        flux : Quantity
          The computed flux density of the star in each filter.

        """
        from .. import calib
        from .. import util

        sw,  sf = calib.cohen_standard(star, unit=unit)
        return self.fluxd(sw, sf, wave)

class SpeX(LongSlitSpectrometer):
    """SpeX.

    Attributes
    ----------
    guidedog : SpeX's guide `Camera`.
    prism : `LongSlitSpectrometer` for 1- to 2.5-micron spectroscopy.
    mode : The current SpeX mode (see examples).

    Examples
    --------

    """

    shape = dict(guidedog=(512, 512), bigdog=(1024, 1024))
    ps = dict(guidedog=0.12 * u.arcsec, bigdog=0.15 * u.arcsec)
    location = Earth

    def __init__(self):
        w = [1.25, 1.64, 2.12, 3.75, 4.70] * u.um
        self.imager = Camera(w, self.shape['guidedog'], self.ps['guidedog'],
                             location=self.location)

        self.prism = LongSlitSpectrometer(
            1.65 * u.um, self.shape['bigdog'], self.ps['bigdog'],
            2.0, 0.034 * u.um, R=250, location=self.location)

        self._mode = 'guidedog'

    @property
    def mode(self):
        if self._mode in ['guidedog', 'prism']:
            return self.__dict__[self._mode]
        else:
            raise KeyError("Invalid mode: {:}".format(self._mode))

    def getheader(self, filename, ext=0):
        """Get a header from a SpeX FITS file.

        SpeX headers tend to be missing quotes around strings.  The
        header will be silently fixed.

        """
        from astropy.io import fits
        inf = fits.open(filename)
        inf[0].verify('silentfix')
        h = inf[0].header.copy()
        inf.close()
        return h
    
    def sed(self, *args, **kwargs):
        """Spectral energy distribution of a target.

        Parameters
        ----------
        *args
        **kwargs
          Arguments and keywords depend on the current MIRSI mode.

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
          Arguments and keywords depend on the current MIRSI mode.

        Returns
        -------
        lc : astropy Table

        """
        return self.mode.lightcurve(*args, **kwargs)

    def spec_correct(self, ftarget, ftelluric, dtt=0.0, dat=0.0, ext=0.0):
        """Correct a spectrum processed with xspextool.

        Take the reduced and telluric spectra, and return a final
        calibrated spectrum.

        Parameters
        ----------
        ftarget : string
          The file name of the target spectrum (FITS format).
        ftelluric : string, optional
          The file name of the telluric spectrum (FITS format).
        dtt : float, optional
          The shift in wavelength to align the telluric spectrum with
          the target. [micron]
        dat : float, optional
          The shift in wavelength to align the ATRAN spectrum with the
          target. [micron]
        ext : float, optional
          Correct the final spectrum using this amount of extinction
          and an ATRAN model.

        Returns
        -------
        wave : ndarray
        flux : ndarray
        err : ndarray
          The final wavelength, flux, and uncertainty.

        """

        from os import path
        from astropy.io import fits
        from ..config import config

        raw_w, raw_f, raw_e = fits.getdata(ftarget)
        tel_w, tel_f, tel_e = fits.getdata(ftelluric)

        x = np.arange(len(tel_w))
        tc = np.interp(x, x + dtt, tel_f)
        tar_f = raw_f * tc
        tar_e = raw_e * tc
        tar_w = raw_w

        atf = path.sep.join([config.get('spex', 'spextool_path'), 'data',
                             'atrans.fits'])
        atran = fits.getdata(atf)
        bw =  np.diff(tar_w) / 2.0
        bins = np.r_[tar_w[0] - bw[0], tar_w[1:] - bw, tar_w[-1] + bw[-1]]
        n = np.histogram(atran[0] + dat, bins=bins)[0].astype(float)
        h = np.histogram(atran[0] + dat, bins=bins, weights=atran[1])
        at = h[0] / n
        try:
            atc = 1 + (1 - at) * ext
        except ZeroDivisionError:
            atc = np.ones_like(tar_w)
        atc[~np.isfinite(atc)] = 1.0

        tar_f /= atc
        tar_e /= atc
        return tar_w, tar_f, tar_e

class SpeXPrism60(SpeX):
    """Reduce uSpeX 60" prism data."""

    config = { # Based on Spextool v4.1
        'lincor max': 35000,
        'y range': [625, 1225],  # generous boundaries
        'x range': [1040, 1901],
        'step': 5,
        'bottom': 624,  # based on _edges()
        'top': 1223,
        'readnoise': 12,  # per single read
        'gain': 1.5
    }

    def __init__(self, *args, **kwargs):
        from astropy.io import ascii, fits
        from ..config import config as C

        calpath = C.get('spex', 'spextool_path') + '/instruments/uspex/data/'
        self.bpm = fits.getdata(calpath + 'uSpeX_bdpxmk.fits')
        
        self.lincoeff = fits.getdata(calpath + 'uSpeX_lincorr.fits')
        
        self.bias = fits.getdata(calpath + 'uSpeX_bias.fits')
        self.bias = (fits.getdata(calpath + 'uSpeX_bias.fits')
                     / self.getheader(calpath + 'uSpeX_bias.fits')['DIVISOR'])
        
        self.linecal = fits.getdata(calpath + 'Prism_LineCal.fits')
        # Prism_LineCal header is bad and will throw a warning
        self.linecal_header = self.getheader(calpath + 'Prism_LineCal.fits')
        
        self.lines = ascii.read(calpath + 'lines.dat', names=('wave', 'type'))
        
        self.mask = ~self.bpm.astype(bool)
        self.mask[:, :self.config['x range'][0]] = 1
        self.mask[:, self.config['x range'][1]:] = 1
        self.mask[:self.config['bottom']] = 1
        self.mask[self.config['top']:] = 1
        
        self.flat = None
        self.flat_var = None
        self.flat_h = None
        self.wavecal = None
        self.wavecal_h = None

        self.wave = None
        self.spec = None
        self.var = None
        self.rap = None
        self.bgap = None
        self.bgorder = None
        
        SpeX.__init__(self, *args, **kwargs)
    
    def _edges(self, im, order=2, plot=False):
        """Find the edges of the spectrum using a flat.

        Parameters
        ----------
        im : ndarray
          A flat field.
        order : int, optional
          The order of the polynomial fit to determine the edge.
        plot : bool, optional
          If `True`, show the image and edges in a matplotlib window.

        Returns
        -------
        b, t : ndarray
          The polynomial coefficients of the bottom and top edges,
          e.g., `bedge = np.polyval(b, x)`

        Notes
        -----
        Based on mc_findorders in Spextool v4.1 (M. Cushing).

        """

        from .. import image
        import scipy.ndimage as nd

        y = image.yarray(im.shape)
        x = image.xarray(im.shape)

        binf = 4
        rebin = lambda im: np.mean(
            im.reshape((im.shape[0], binf, im.shape[1] / binf)),
            1).astype(int)
        
        rim = rebin(im)
        ry = rebin(y)
        rx = rebin(x)

        # find where signal falls to 0.75 x center        
        i = int(np.mean(self.config['y range']))
        fcen = rim[i]

        bguess, tguess = np.zeros((2, rim.shape[1]), int)
        for j in range(rim.shape[1]):
            bguess[j] = np.min(ry[:i, j][rim[:i, j] > 0.75 * fcen[j]])
            tguess[j] = np.max(ry[i:, j][rim[i:, j] > 0.75 * fcen[j]])

        # scale back up to 2048
        bguess = np.repeat(bguess, binf)
        tguess = np.repeat(tguess, binf)
        
        # find actual edge by centroiding (center of mass) on Sobel
        # filtered image
        def center(yg, y, sim):
            s = slice(yg - 5, yg + 6)
            yy = y[s]
            f = sim[s]
            return nd.center_of_mass(f) + yy[0]

        sim = np.abs(nd.sobel(im * 1000 / im.max(), 0))
        bcen, tcen = np.zeros((2, im.shape[1]))
        for i in range(*self.config['x range']):
            bcen[i] = center(bguess[i], y[:, i], sim[:, i])
            tcen[i] = center(tguess[i], y[:, i], sim[:, i])

        def fit(x, centers):
            A = np.vstack((x**2, x, np.ones(len(x)))).T
            return np.linalg.lstsq(A, centers)[0]

        xx = np.arange(im.shape[1])
        s = slice(*self.config['x range'])
        b = fit(xx[s], bcen[s])
        t = fit(xx[s], tcen[s])

        if plot:
            import matplotlib.pyplot as plt
            fig = plt.figure(figsize=(8, 8))
            ax = plt.gca()
            ax.imshow(im, cmap=plt.cm.gray)
            ax.plot(xx[::binf], bcen[::binf], 'gx')
            ax.plot(xx[::binf], tcen[::binf], 'gx')
            ax.plot(xx, np.polyval(b, xx), 'r-')
            ax.plot(xx, np.polyval(t, xx), 'r-')
            plt.draw()

        return b, t

    def _ampcor(self, im):
        """Correct an image for amplifier noise.

        The median of the reference pixels is subtracted from each
        amplifier column.

        Notes
        -----
        Based on mc_findorders in Spextool v4.1 (M. Cushing).

        """

        amps = np.rollaxis(im[2044:].reshape((4, 64, 32)), 1).reshape(64, 128)
        m = np.median(amps, 1)
        return im - np.tile(np.repeat(m, 32), 2048).reshape(im.shape)

    def _lincor(self, im):
        """Linearity correction for an image.

        Notes
        -----
        Following mc_imgpoly from Spextool v4.1 (M. Cushing).

        """
        y = 0
        for i in range(len(self.lincoeff)):
            y = y * im + self.lincoeff[-(i + 1)]
        return y

    def read(self, files, pair=False, ampcor=True, lincor=True, flatcor=True,
             abba_test=True):
        """Read uSpeX files.

        Parameters
        ----------
        files : string or list
          A file name or list thereof.
        pair : bool, optional
          Assume the observations are taken in ABBA mode and return
          A-B for each pair.
        ampcor : bool optional
          Set to `True` to apply the amplifcation noise correction.
        lincor : bool, optional
          Set to `True` to apply the linearity correction.
        flatcor : bool, optional
          Set to `True` to apply flat field correction.
        abba_test : bool, optional
          Set to `True` to test for ABBA ordering when `pair` is
          `True`.  If `abba_test` is `False`, then the file order is
          not checked.

        Returns
        -------
        stack : MaskedArray
          The resultant image(s).  [counts / s]
        var : MaskedArray
          The variance.  [total DN]
        headers : list or astropy FITS header
          If `pair` is `True`, the headers will be a list of lists,
          where each element is a list containing the A and B headers.

        """

        from numpy.ma import MaskedArray
        from astropy.io import fits

        if isinstance(files, (list, tuple)):
            print('Loading {} files.'.format(len(files)))
            stack = MaskedArray(np.empty((len(files), 2048, 2048)))
            var = np.empty((len(files), 2048, 2048))
            headers = []
            for i in range(len(files)):
                kwargs = dict(pair=False, ampcor=ampcor, lincor=lincor,
                              flatcor=flatcor)
                stack[i], var[i], h = self.read(files[i], **kwargs)
                headers.append(h)

            if pair:
                print('\nABBA pairing and subtracting.')
                if abba_test:
                    # Require ABBA ordering
                    msg = 'Files not in an ABBA sequence'
                    assert all([h['BEAM'] == 'A' for h in headers[::4]]), msg
                    assert all([h['BEAM'] == 'B' for h in headers[1::4]]), msg
                    assert all([h['BEAM'] == 'B' for h in headers[2::4]]), msg
                    assert all([h['BEAM'] == 'A' for h in headers[3::4]]), msg
                
                # fancy slicing, stacking, and reshaping to get:
                #   [0 - 1, 3 - 2, 4 - 5, 7 - 6, ...]
                stack_A = stack[::4] - stack[1::4]  # A - B
                var_A = var[::4] + var[1::4]
                headers_A = [[a, b] for a, b in zip(headers[::4], headers[1::4])]
                if len(files) > 2:
                    stack_B = stack[3::4] - stack[2::4] # -(B - A)
                    stack = np.ma.vstack((stack_A, stack_B))

                    var_B = var[2::4] + var[3::4]
                    var = np.ma.vstack((var_A, var_B))

                    headers_B = [[a, b] for a, b in zip(headers[3::4], headers[2::4])]
                    headers = [None] * (len(headers_A) + len(headers_B))
                    headers[::2] = headers_A
                    headers[1::2] = headers_B
                else:
                    stack = stack_A[0]
                    var = var_A[0]
                    headers = headers_A[0]

            return stack, var, headers

        print('Reading {}'.format(files))
        data = fits.open(files)
        data[0].verify('silentfix')
        h = data[0].header.copy()
        read_var = (2 * self.config['readnoise']**2
                    / h['NDR']
                    / h['CO_ADDS']
                    / h['ITIME']**2
                    / self.config['gain']**2)

        # TABLE_SE is read time, not sure what crtn is.
        crtn = (1 - h['TABLE_SE'] * (h['NDR'] - 1)
                / 3.0 / h['ITIME'] / h['NDR'])
        t_exp = h['ITIME'] * h['CO_ADDS']
        
        im_p = data[1].data / h['DIVISOR']
        im_s = data[2].data / h['DIVISOR']
        data.close()

        mask_p = im_p < (self.bias - self.config['lincor max'])
        mask_s = im_s < (self.bias - self.config['lincor max'])
        mask = mask_p + mask_s
        h.add_history('Masked saturated pixels.')

        im = MaskedArray(im_p - im_s, mask)

        if ampcor:
            im = self._ampcor(im)
            h.add_history('Corrected for amplifier noise.')

        if lincor:
            cor = self._lincor(im)
            cor[mask] = 1.0
            cor[:4] = 1.0
            cor[:, :4] = 1.0
            cor[2044:] = 1.0
            cor[:, 2044:] = 1.0
            im /= cor
            h.add_history('Applied linearity correction.')

        if flatcor:
            assert self.flat is not None, "Flat correction requested but flat not loaded."
            im /= self.flat
            h.add_history('Flat corrected.')
            
        # total DN
        var = (np.abs(im * h['DIVISOR'])
               * crtn
               / h['CO_ADDS']**2
               / h['ITIME']**2
               / self.config['gain']
               + read_var) # / h['DIVISOR']**2 / h['ITIME']**2
        # counts / s
        im = im / h['ITIME']
        im.mask += self.mask
        return im, var, h

    def process_cal(self, files, path=None, overwrite=True):
        """Generate flat field and wavelength calibration files.

        Parameters
        ----------
        files : list
          A list of images created by SpeX calibration macros.  Only
          60" prism data will be considered.
        path : string, optional
          Save files to this directory.  If `None`, they will be saved
          to "cal-YYYYMMDD".
        overwrite : bool, optional
          Set to `True` to overwrite previous calibration files.

        """
        
        import re
        import os
        import os.path
        from astropy.io import fits

        # flats
        flats = sorted(filter(lambda f: os.path.split(f)[1].startswith('flat'),
                              files))
        fset = []
        n_sets = 0
        first_n = -1
        last_n = -1
        for i in range(len(flats)):
            h = self.getheader(flats[i])
            test = (h['OBJECT'] != 'Inc lamp'
                    or h['GRAT'] != 'Prism'
                    or 'x60' not in h['SLIT'])
            if test:
                fset = []
                continue

            m = re.findall('flat-([0-9]+).a.fits$', flats[i])
            assert len(m) == 1, 'Cannot parse file name: {}'.format(flats[i])
            n = int(m[0])

            fset.append(flats[i])
            print('Found {}  ({})'.format(flats[i], len(fset)))

            if len(fset) == 1:
                first_n = n
            elif len(fset) > 2:
                step = n - last_n
                last_n = n
                if step != 1:
                    print('  Bad image sequence.')
                    fset = []
                    continue
            else:
                last_n = n
            
            if len(fset) == 5:
                if path is None:
                    _path = 'cal-' + h['DATE_OBS'].replace('-', '')
                else:
                    _path = path

                try:
                    os.mkdir(_path)
                except FileExistsError:
                    pass

                fn = '{}/flat-{:05d}-{:05d}.fits'.format(
                    _path, first_n, last_n)
                self.load_flat(fset)
                outf = fits.HDUList()
                outf.append(fits.PrimaryHDU(self.flat, self.flat_h))
                outf.append(fits.ImageHDU(self.flat_var, name='var'))
                outf.writeto(fn, output_verify='silentfix', clobber=overwrite)

                fset = []

        # done with flats

        # arcs
        arcs = sorted(filter(lambda f: os.path.split(f)[1].startswith('arc'),
                             files))
        for i in range(len(arcs)):
            h = self.getheader(arcs[i])
            test = (h['OBJECT'] != 'Argon lamp'
                    or h['GRAT'] != 'Prism'
                    or 'x60' not in h['SLIT'])
            if test:
                continue

            m = re.findall('arc-([0-9]+).a.fits$', arcs[i])
            assert len(m) == 1, 'Cannot parse file name: {}'.format(arcs[i])
            n = int(m[0])

            # only one arc lamp observation per Prism 60" cal
            if path is None:
                _path = 'cal-' + h['DATE_OBS'].replace('-', '')
            else:
                _path = path

            try:
                os.mkdir(_path)
            except FileExistsError:
                pass

            fn = '{}/wavecal-{:05d}.fits'.format(_path, n)
            self.load_wavecal(arcs[i])
            fits.writeto(fn, self.wavecal, self.wavecal_h,
                         output_verify='silentfix', clobber=overwrite)
        # done with arcs
    
    def load_flat(self, files):
        """Generate or read in a flat.

        Parameters
        ----------
        files : list or string
          A list of file names of data taken with the SpeX cal macro,
          or the name of an already prepared flat.
        save : bool, optional
          If `True`, save the new flat and variance data as a FITS
          file.  The name will be generated from the file list
          assuming they originated from a SpeX calibration macro.

        """

        from numpy.ma import MaskedArray
        import scipy.ndimage as nd
        from scipy.interpolate import splrep, splev
        from astropy.io import fits

        if isinstance(files, str):
            self.flat, self.flat_h = fits.getdata(files, header=True)
            self.flat_var = fits.getdata(files, ext=1)
            return

        stack, headers = self.read(files, flatcor=False)[::2]
        h = headers[0]
        scale = np.array([np.ma.median(im) for im in stack])
        scale /= np.mean(scale)
        for i in range(len(stack)):
            stack[i] /= scale[i]

        flat = np.median(stack, 0)
        var = np.var(stack, 0) / len(stack)

        c = np.zeros(flat.shape)
        x = np.arange(flat.shape[1])
        for i in range(flat.shape[1]):
            if np.all(flat[i].mask):
                continue
            j = ~flat[i].mask
            y = nd.median_filter(flat[i][j], 7)
            c[i] = splev(x, splrep(x[j], y))

        h.add_history('Flat generated from: ' + ' '.join(files))
        h.add_history('Images were scaled to the median flux value, then median combined.  The variance was computed then normalized by the number of images.')

        self.flat = (flat / c).data
        self.flat_var = (var / c).data
        self.flat_h = h

    def load_wavecal(self, fn, plot=False, debug=False):
        """Load or generate a wavelength calibration.

        Parameters
        ----------
        fn : string
          The name of an arc file taken with the SpeX cal macro or an
          already prepared wavelength calibration.
        plot : bool, optional
          Set to `True` to plot representative wavelength solutions.

        Notes
        -----
        Based on Spextool v4.1 (M. Cushing).

        """

        import scipy.ndimage as nd
        from astropy.io import fits
        from .. import util, image

        h = self.getheader(fn)
        if 'wavecal' in h:
            assert h['wavecal'] == 'T', "WAVECAL keyword present in FITS header, but is not 'T'."
            print('Loading stored wavelength solution.')
            wavecal = fits.getdata(fn)
            mask = ~np.isfinite(wavecal)
            self.wavecal = np.ma.MaskedArray(wavecal, mask=mask)
            self.wavecal_h = h
            return
        
        arc = self.read(fn, flatcor=False)[0]

        slit = h['SLIT']
        slitw = float(slit[slit.find('x') + 1:])

        flux_anchor = self.linecal[1]
        wave_anchor = self.linecal[0]
        offset = np.arange(len(wave_anchor)) - int(len(wave_anchor) / 2.0)
        self.wavecal = image.xarray(arc.shape, dtype=float)

        disp_deg = self.linecal_header['DISPDEG']
        w2p = []
        p2w = []
        for i in range(disp_deg + 1):
            w2p.append(self.linecal_header['W2P01_A{}'.format(i)])
            p2w.append(self.linecal_header['P2W_A0{}'.format(i)])

        xr = slice(*self.config['x range'])
        xcor_offset = np.zeros(2048)
        for i in range(self.config['bottom'], self.config['top']):
            spec = arc[i, xr]

            xcor = nd.correlate(spec, flux_anchor, mode='constant')
            j = np.argmax(xcor)
            s = slice(j - 7, j + 8)
            guess = (np.max(xcor), offset[j], 5, 0.0)

            xx = offset[s]
            yy = xcor[s]
            j = np.isfinite(xx * yy)
            fit = util.gaussfit(xx[j], yy[j], None, guess)
            xcor_offset[i] = fit[0][1]

        # smooth out bad fits
        i = xcor_offset != 0
        y = np.arange(2048)
        p = np.polyfit(y[i], xcor_offset[i], 2)
        r = np.abs(xcor_offset - np.polyval(p, y))

        i = (xcor_offset != 0) * (r < 1)
        p = np.polyfit(y[i], xcor_offset[i], 2)
        
        # update wave cal with solution
        for i in range(self.config['bottom'], self.config['top']):
            x = self.wavecal[i, xr] - np.polyval(p, y[i])
            self.wavecal[i, xr] = np.polyval(p2w[::-1], x)

        h['wavecal'] = 'T'
        h['bunit'] = 'um'

        self.wavecal[self.mask] = np.nan
        self.wavecal_h = h
        self.arc = arc

        if plot:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            plt.clf()
            for y in (700, 920, 1150):
                plt.plot(self.wavecal[y, xr], arc[y, xr])
            plt.draw()

        if debug:
            return offset, xcor_offset

    def peak(self, im, mode='AB', rap=5, smooth=0, plot=True,
             ex_rap=None, bgap=None):
        """Find approximate locations of profile peaks in an image.

        The strongest peaks are found via centroid on the profile
        min/max.

        Parameters
        ----------
        im : ndarray or MaskedArray
          An image.
        mode : string, optional
          'AB' if there is both a positive and a negative peak.  Else,
          set to 'A' for a single positive peak.
        rap : int, optional
          Radius of the fitting aperture.
        smooth : float, optional
          Smooth the profile with a `smooth`-width Gaussian before
          searching for the peak.
        plot : bool, optional
          Plot results.
        ex_rap : float
          Show this extraction aperture radius in the plot.
        bgap : array of two floats
          Show this extraction background aperture in the plot.

        Result
        ------
        self.peaks : ndarray
          The peaks.  For a stack: NxM array where N is the number of
          images, and M is the number of peaks.

        """

        import scipy.ndimage as nd
        from ..util import between, gaussfit

        profile = np.ma.median(im, 1)
        if smooth > 0:
            profile = nd.gaussian_filter(profile, smooth)

        self.peaks = []
        x = np.arange(im.shape[1])

        i = between(x, profile.argmax() + np.r_[-rap, rap])
        c = nd.center_of_mass(profile[i]) + x[i][0]
        self.peaks.append(c[0])
        
        if mode.upper() == 'AB':
            i = between(x, profile.argmin() + np.r_[-rap, rap])
            c = nd.center_of_mass(-profile[i]) + x[i][0]
            self.peaks.append(c[0])

        self.peaks = np.array(self.peaks)

        if plot:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            fig.clear()
            ax = fig.add_subplot(111)
            ax.plot(profile, color='k')
            for p in self.peaks:
                ax.axvline(p, color='r')

                if ex_rap is not None:
                    i = between(x, p + np.r_[-1, 1] * ex_rap)
                    ax.plot(x[i], profile[i], color='b', lw=3)
                                  
                if bgap is not None:
                    for s in [-1, 1]:
                        i = between(x, np.sort(p + s * np.r_[bgap]))
                        ax.plot(x[i], profile[i], color='c', lw=3)
                                  
            fig.canvas.draw()
            fig.show()

    def trace(self, im, plot=True):
        """Trace the peak(s) of an object.

        Best executed with standard stars.

        Initial peak guesses taken from `self.peaks`.  If there are
        multiple, even-indexed peaks are assumed to be the positive,
        odd-indexed peaks are assumed to be the negative beam.

        Parameters
        ----------
        im : MaskedArray
          The 2D spectrum to trace.
        plot : bool, optional
          Plot results.

        Result
        ------
        self.traces : list of ndarray
          The traces of each peak.
        self.trace_fits : list of ndarray
          The best-fit polynomical coefficients of the traces.

        """

        from .. import image

        profile = np.ma.median(im, 1)
        self.traces = []
        self.trace_fits = []
        for i in range(len(self.peaks)):
            s = (-1)**i
            guess = ((s * profile).max(), self.peaks[i], 2.)
            trace, fit = image.trace(s * im, None, guess, rap=10,
                                     polyfit=True, order=7)
            fit = np.r_[fit[:-1], fit[-1] - self.peaks[i]]
            self.traces.append(trace)
            self.trace_fits.append(fit)

        if plot:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            fig.clear()
            ax = fig.add_subplot(111)

            for i in range(len(self.traces)):
                j = ~self.traces[i].mask
                x = np.arange(len(self.traces[i]))
                ax.plot(x, self.traces[i], color='k', marker='+', ls='none')
                fit = np.r_[self.trace_fits[i][:-1],
                            self.trace_fits[i][-1] + self.peaks[i]]
                ax.plot(x[j], np.polyval(fit, x[j]), color='r')

            fig.canvas.draw()
            fig.show()

    def _aper(self, y, trace, rap, subsample):
        """Create an aperture array for `extract`."""
        aper = (y >= trace - rap) * (y <= trace + rap)
        aper = aper.reshape(y.shape[0] // subsample, subsample, y.shape[1])
        aper = aper.sum(1) / subsample
        return aper

    def extract(self, im, h, rap, bgap=None, bgorder=0, traces=True,
                abcombine=True, append=False):
        """Extract a spectrum from an image.

        Extraction positions are from `self.peaks`.

        See `image.spextract` for implementation details.

        Parameters
        ----------
        im : MaskedArray
          The 2D spectral image.
        h : astropy FITS header
          The header for im.
        rap : float
          Aperture radius.
        bgap : array, optional
          Inner and outer radii for the background aperture, or `None`
          for no background subtraction.
        bgorder : int, optional
          Fit the background with a `bgorder` polynomial.
        traces : bool, optional
          Use `self.traces` for each peak.
        abcombine : bool, optional
          Combine (sum) apertures as if they were AB pairs.  The
          B-beam will be linearly interpolated onto A's wavelengths.
        append : bool, optional
          Append results to arrays, rather than creating new ones.

        Results
        -------
        self.wave : list of ndarray
          The wavelengths.
        self.spec : list of ndarray
          The spectra.
        self.var : list of ndarray
          The variance on the spectrum due to background.

        """

        from .. import image

        N_peaks = len(self.peaks)
        if abcombine:
            assert N_peaks == 2, "There must be two peaks when abcombine is requested."
            print('Combining (sum) AB apertures.')

        if traces:
            trace = self.trace_fits
        else:
            trace = None

        if bgap is None:
            spec = image.spextract(im, self.peaks, rap, trace=trace,
                                   subsample=5)[1]
            var = np.ma.MaskedArray(np.zeros(spec.shape))
        else:
            n, spec, nbg, mbg, bgvar = image.spextract(
                im, self.peaks, rap, trace=trace, bgap=bgap,
                bgorder=bgorder, subsample=5)
            var = 2 * rap * bgvar * (2 * rap / nbg)

        if self.wavecal is None:
            # dummy wavelengths
            wave = np.tile(np.arange(im.shape[1], dtype=float), N_peaks)
            wave = wave.reshape((N_peaks, im.shape[1]))
        else:
            wave = image.spextract(self.wavecal, self.peaks, rap, mean=True,
                                   trace=trace, subsample=5)[1]

        if abcombine:
            w = wave[::2]
            s = spec[::2]
            v = var[::2]
            h_other = h[1::2]
            h = h[::2]
            for i in range(len(s)):
                h[i].add_history('AB beam combined (sum)')
                h[i]['ITOT'] = h[i]['ITIME'] + h_other[i]['ITIME']

                b = i * 2 + 1

                mask = s[i].mask + wave[b].mask + spec[b].mask + var[b].mask

                x = np.interp(w[i], wave[b, ~mask], spec[b, ~mask])
                s[i] -= np.ma.MaskedArray(x, mask=mask)

                x = np.interp(w[i], wave[b, ~mask], var[b, ~mask])
                v[i] += np.ma.MaskedArray(x, mask=mask)

            wave = w
            spec = s
            var = v

        # for spextool compatability
        for i in range(len(h)):
            ps = h[i]['PLATE_SC']
            appos = (self.peaks - self.config['bottom']) * ps
            h[i]['APPOSO01'] = str(list(appos))[1:-1], 'Aperture positions (arcsec) for order 01'
            h[i]['AP01RAD'] = rap * ps, 'Aperture radius in arcseconds'
            h[i]['BGORDER'] = bgorder, 'Background polynomial fit degree'
            if bgap is None:
                h[i]['BGSTART'] = 0
                h[i]['BGWIDTH'] = 0
            else:
                h[i]['BGSTART'] = bgap[0] * ps, 'Background start radius in arcseconds'
                h[i]['BGWIDTH'] = np.ptp(bgap) * ps, 'Background width in arcseconds'
            h[i]['MODENAME'] = 'Prism', 'Spectroscopy mode'
            h[i]['NORDERS'] = 1, 'Number of orders'
            h[i]['ORDERS'] = '1', 'Order numbers'

            slit = [float(x) for x in h[i]['SLIT'].split('x')]
            h[i]['SLTW_ARC'] = slit[0]
            h[i]['SLTH_ARC'] = slit[1]
            h[i]['SLTW_PIX'] = slit[0] / ps
            h[i]['SLTH_PIX'] = slit[1] / ps

            h[i]['RP'] = int(82 * slit[0] / 0.8), 'Resovling power'
            h[i]['DISP001'] = 0.00243624, 'Dispersion (um pixel-1) for order 01'
            h[i]['XUNITS'] = 'um', 'Units of the X axis'
            h[i]['YUNITS'] = 'DN / s', 'Units of the Y axis'
            h[i]['XTITLE'] = '!7k!5 (!7l!5m)', 'IDL X title'
            h[i]['YTITLE'] = 'f (!5DN s!u-1!N)', 'IDL Y title'
            
        if (self.spec is None) or (self.spec is not None and not append):
            self.wave = wave
            self.spec = spec
            self.var = var
            self.h = h
        else:
            self.wave = np.ma.concatenate((self.wave, wave))
            self.spec = np.ma.concatenate((self.spec, spec))
            self.var = np.ma.concatenate((self.var, var))
            self.h.extend(h)

    def save_spec(self, fnformat='spec-{n}.fits', **kwargs):
        """Write extracted spectra to FITS files.

        The file columns are wavelength, DN/s, and uncertainty.

        The files should be compatible with xspextool.

        Parameters
        ----------
        fnformat : string, optional
          The format string for file names.  Use '{n}' for the
          spectrum number which will be gleaned from the FITS headers.
        **kwargs
          `astropy.io.fits.writeto` keyword arguments.

        """

        import re
        from astropy.io import fits

        assert self.spec is not None, "No spectra have been extracted"
        kwargs['output_verify'] = kwargs.get('output_verify', 'silentfix')

        for i in range(len(self.spec)):
            n = re.findall('.*-([0-9]+).[ab].fits', self.h[i]['IRAFNAME'],
                           re.IGNORECASE)[0]
            fn = fnformat.format(n=n)
    
            x = np.c_[self.wave[i].filled(np.nan),
                      self.spec[i].filled(np.nan),
                      np.sqrt(self.var[i]).filled(np.nan)].T
            j = np.flatnonzero(np.isfinite(np.prod(x, 0)))
            fits.writeto(fn, x[:, j], self.h[i], **kwargs)

# update module docstring
from ..util import autodoc
autodoc(globals())
del autodoc
