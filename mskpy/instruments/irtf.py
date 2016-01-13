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
        'y range': np.array((625, 1225)),  # generous boundaries
        'x range': np.array((1050, 1860)),
        'step': 5,
        'bottom': 624,  # based on _edges()
        'top': 1223,
        'readnoise': 12,  # per single read
        'gain': 1.5
    }

    def __init__(self, *args, **kwargs):
        from astropy.io import fits
        from ..config import config as C
        
        calpath = C.get('spex', 'spextool_path') + '/instruments/uspex/data/'
        self.bpm = fits.getdata(calpath + 'uSpeX_bdpxmk.fits')
        self.lincoeff = fits.getdata(calpath + 'uSpeX_lincorr.fits')
        self.bias = fits.getdata(calpath + 'uSpeX_bias.fits')
        self.bias /= fits.getheader(calpath + 'uSpeX_bias.fits')['DIVISOR']
        self.mask = ~self.bpm.astype(bool)
        self.mask[:, :self.config['x range'][0]] = 1
        self.mask[:, self.config['x range'][1]:] = 1
        self.mask[:self.config['bottom']] = 1
        self.mask[self.config['top']:] = 1
        self.flat_im = None
        self.flat_var = None
        self.flat_h = None
        
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
        i = int(self.config['y range'].mean())
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

    def getheader(self, filename, ext=0):
        """Get a header from a SpeX FITS file.

        The header will be silently fixed.

        """
        from astropy.io import fits
        inf = fits.open(filename)
        inf[0].verify('silentfix')
        h = inf[0].header.copy()
        inf.close()
        return h
    
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
                    stack = np.ma.hstack((stack_A, stack_B))
                    stack = stack.reshape((len(files) / 2, 2048, 2048))

                    var_B = var[2::4] + var[3::4]
                    var = np.hstack((var_A, var_B))
                    var = var.reshape((len(files) / 2, 2048, 2048))

                    headers_B = [[a, b] for a, b in zip(headers[3::4], headers[2::4])]
                    headers = []
                    for a, b in zip(headers_A, headers_B):
                        headers.extend([a, b])

            return stack, var, headers

        print('Reading {}'.format(files))
        data = fits.open(files)
        h = data[0].verify('silentfix')
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
        h.add_history('Masked saturated pixels')

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
            assert self.flat_im is not None, "Flat correction requested but flat not loaded."
            im /= self.flat_im
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
        return im, var, h

    def process_cal(self, files, path=None):
        """Generate flat field and wavelength calibration files.

        Parameters
        ----------
        files : list
          A list of images created by SpeX calibration macros.  Only
          60" prism data will be considered.
        path : string
          Save files to this directory.  If `None`, they will be saved
          to "cal-YYYYMMDD".

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

                fn = '{}/flat{:05d}-{:05d}.fits'.format(_path, first_n, last_n)
                self.flat(fset)
                outf = fits.HDUList()
                outf.append(fits.PrimaryHDU(self.flat_im, self.flat_h))
                outf.append(fits.ImageHDU(self.flat_var, name='var'))
                outf.writeto(fn, output_verify='silentfix')

                fset = []

        # done with flats

        # arcs
        arcs = sorted(filter(lambda f: os.path.split(f)[1].startswith('arc'),
                             files))
    
    def flat(self, files):
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

        if isinstance(files, str):
            self.flat_im = fits.getdata(files)
            self.flat_var = fits.getdata(files, ext=1)
        else:
            stack, headers = self.read(files, flatcor=False)[::2]
            h = headers[0]
            scale = np.array([np.median(im) for im in stack])
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
                y = nd.gaussian_filter(flat[i][j], 3.0)
                c[i] = splev(x, splrep(x[j], y))

            h.add_history('Flat generated from: ' + ' '.join(files))
            h.add_history('Images were scaled to the median flux value, then median combined.  The variance was computed then normalized by the number of images.')
                
            self.flat_im = (flat / c).data
            self.flat_var = (var / c).data
            self.flat_h = h

    def wavecal(self, f):
        """Generate or read in a wavelength calibration.

        Parameters
        ----------
        f : list or string
          A list of file names of data taken with the SpeX cal macro,
          or the name of an already prepared wavecal.

        Notes
        -----
        Based on Spextool v4.1 (M. Cushing).

        """
        raise NotImplemented
            
# update module docstring
from ..util import autodoc
autodoc(globals())
del autodoc
