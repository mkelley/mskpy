# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
spitzer --- Spitzer instruments.
================================

   Functions
   ---------
   irsclean
   irsclean_files
   moving_wcs_fix

   Classes
   -------
   IRAC
   IRS

"""

import numpy as np
import astropy.units as u

try:
    from ..ephem import Spitzer
except ImportError:
    Spitzer = None

from .instrument import Instrument, Camera, LongSlitSpectrometer

__all__ = ['irsclean', 'irsclean_files', 'IRAC', 'IRS']

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

        from scipy import interpolate
        import astropy.constants as const
        from ..calib import filter_trans
        from ..util import davint, takefrom

        nu0 = (const.c.si / self.wave).to(u.teraHertz).value
        K = np.zeros(len(channels))
        for ch in channels:
            tw, tr = filter_trans('IRAC CH{:}'.format(ch))
            nu = (const.c / tw).to(u.teraHertz).value

            sfnu = sf(tw).to(u.Jy, u.spectral_density(tw)).value

            i = ch - 1  # self.wave index
            sfnu /= sf(self.wave[i]).to(u.Jy, u.spectral_density(self.wave[i])).value

            sfnu, tr, nu = takefrom((sfnu, tr, nu), nu.argsort())
            K[i] = (davint(nu, sfnu * tr * nu0[i] / nu, nu[0], nu[-1])
                    / davint(nu, tr * (nu0[i] / nu)**2, nu[0], nu[-1]))

        return K

#    def ccorrection_tab(self, sw, sf):
#        """IRAC color correction of a tabulated spectrum.
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

def irsclean(im, h, bmask=None, maskval=16384,
             rmask=None, func=None, nan=True, **fargs):
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
      map.
    rmask : ndarray, optional
      The rogue mask array.
    func : function, optional
      Use this function to clean the image: first argument is the
      image to clean, the second is the mask (`True` for each bad
      pixel).  The default is `image.fixpix`.
    nan : bool, optional
      Set to `True` to also clean any pixels set to NaN.
    **fargs
      Additional keyword arguments are pass to `func`.

    Returns
    -------
    cleaned : ndarray
      The cleaned array.
    h : dict-like
      The updated header.

    """

    from ..image import fixpix

    cleaner = fixpix if func is None else func

    mask = 0
    if nan:
        mask += ~np.isfinite(im)
    if bmask is not None:
        mask += np.bitwise_and(bmask, maskval)
    if rmask is not None:
        mask += rmask
    mask = mask.astype(bool)
    
    h.add_history('Cleaned with mskpy.instruments.spitzer.irsclean.')
    h.add_history('irsclean: function={}, arguments={}'.format(
        str(cleaner), str(fargs)))
    
    cleaned = cleaner(im, mask, **fargs)
    return cleaned, h

def irsclean_files(files, outfiles, uncs=True, bmasks=True,
                   maskval=16384, rmasks=True, func=None, nan=True,
                   **fargs):
    """Clean bad pixels from a list of IRS files.

    For automatic rogue mask file name gleaning, this function
    requires that irs.rogue_masks_path is set in mskpy.cfg to the
    location of the rogue masks files from the Spitzer Science Center.

    Parameters
    ----------
    files : array of strings or string
      A list of image names to clean, or the filename of a list of files.
    outfiles : array-like or filename
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
    **fargs
      Additional keyword arguments are pass to `func`.

    """

    from astropy.io import fits
    
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
        import os.path
        from ..config import config
        path = config.get('irs', 'rogue_masks_path')
        for i in range(len(in_list)):
            if optional_list is True:
                h = fits.getheader(in_list[i])
                if h['CAMPAIGN'] in campaign2rogue:
                    f = 'b{}_rmask_{}.fits'.format(
                        h['CHNLNUM'], campaign2rogue[h['CAMPAIGN']])
                    f = os.path.join((path, f))
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
    bmask_files = file_generator(files, uncs, '_bmask')
    rmask_files = rmask_file_generator(files, rmasks)

    for i in range(len(files)):
        unc_file, unc = next(unc_files)
        bmask_file, bmask = next(bmask_files)
        rmask_file, rmask = next(rmask_files)

        im, h = fits.getdata(files[i], header=True)
        cleaned = irsclean(im, h, bmask=bmask, maskval=maskval,
                           rmask=rmask, func=func, **fargs)
        fits.writeto(outfiles[i], cleaned[0], cleaned[1], clobber=True)
        
        if unc is not None:
            if '_bcd' in outfiles[i]:
                outf = outfiles[i].replace('_bcd', '_func')
            elif outfiles[i].endswith('.fits'):
                outf = outfiles[i].replace('.fits', '_func.fits')
            else:
                outf = outfiles[i] + '_func.fits'

            h = fits.getheader(unc_file)
            cleaned = irsclean(unc, h, bmask=bmask, maskval=maskval,
                               rmask=rmask, func=func, **fargs)
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
                h["CRVAL1"], h.get["CRVAL2"])
        rarqst, decrqst = spherical_coord_rotate(
            ra_ref1, dec_ref1, ra_ref0, dec_ref0,
            h["RA_RQST"], h["DEC_RQST"])

        raslt, decslt = spherical_coord_rotate(
            ra_ref1, dec_ref1, ra_ref0, dec_ref0,
            h["RA_SLT"], h["DEC_SLT"])

        print("{} moved {{:.3f} {:.3f}".format(f, (ra_ref1 - ra_ref0) * 3600.,
                                               (dec_ref1 - dec_ref0) * 3600.))

        if h.get("CRVAL1") is not None:
            h.update("CRVAL1", crval1)
            h.update("CRVAL2", crval2)
        h.update("RA_RQST", rarqst)
        h.update("RA_SLT", raslt)
        h.update("DEC_RQST", decrqst)
        h.update("DEC_SLT", decslt)
        h.add_history("WCS updated for moving target motion with mskpy.instrum3ents.spitzer.moving_wcs_fix")
        fits.update(f, im, h)

# update module docstring
from ..util import autodoc
autodoc(globals())
del autodoc
