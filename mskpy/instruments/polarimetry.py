# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
polarimetry --- Polarimeters.
=============================

   Classes
   -------
   OptiPol

"""

import numpy as np
import astropy.units as u

try:
    from ..ephem import Earth
except ImportError:
    Earth = None

#from .instrument import Instrument, Camera

__all__ = ['OptiPol']

class LinPol(object):
    """Describing linear polarization, parameterized with I, QI, and UI.

    pol = LinPol(I=a, QI=b, UI=c)
    pol = LinPol(I=a, sig_I=sig_a, QI=b, sig_QI=sig_b, UI=c, sig_UI=sig_c)

    Parameters
    ----------
    I : float or array, optional
      Total intensity.
    QI, UI : float or array, optional
      The linear polarization parameters normalized by intensity.
    sig_I, sig_QI, sig_UI : float or array, optional
      `I`, `QI` and `UI` uncertainties.
    correct : bool, optional
      Set to `False` to prevent the Ricean correction from being
      applied.

    Attributes
    ----------
    p : float or array
      Total linear polarization.
    sig_p : float or array
      Uncertainty on `p`.
    theta : astropy Angle
      Linear polarizaton position angle.
    sig_theta : astropy Angle
      Uncertainty on `theta`.

    Methods
    -------
    rotate - This linear polarization vector, rotated.

    """

    def __init__(self, **kwargs):
        self.I = kwargs.pop('I', None)
        self.QI = kwargs.pop('QI', None)
        self.UI = kwargs.pop('UI', None)

        self.sig_I = kwargs.pop('sig_I', None)
        self.sig_QI = kwargs.pop('sig_QI', None)
        self.sig_UI = kwargs.pop('sig_UI', None)

        self.correct = kwargs.pop('correct', True)

    @property
    def _pth(self):
        args = self.QI, self.UI, self.sig_QI, self.sig_UI
        p = linear_pol(*args)
        th = linear_pol_angle(*args)
        if self.correct:
            return ricean_correction(p + th)
        else:
            return p + th

    @property
    def p(self):
        return self._pth[0]

    @property
    def sig_p(self):
        return self._pth[1]

    @property
    def theta(self):
        return self._pth[2]

    @property
    def sig_theta(self):
        return self._pth[3]

class HalfWavePlate(LinPol):
    """Linear polarimetry with a 1/2-wave plate.

    pol = HalfWavePlate(I)
    pol = HalfWavePlate(I, sig_I)

    Parameters
    ----------
    I : array
      Intensities from each polarization angle: `I0`, `I45`, `I90`,
      `I135`.  Each intensity may in turn be an array.
    sig_I : array
      Intensity uncertainties, same form as `I`.

    Attributes
    ----------

    """

    def __init__(self, **kwargs):
        

class OptiPol(HalfWavePlate):
    """Univ. Minnesota's optical imaging polarimeter.

    Attributes
    ----------


    Examples
    --------

    """

    shape = (240, 320)
    ps = 0.265 * u.arcsec
    location = Earth

    def __init__(self):
        w = [4.9, 7.7, 8.7, 9.8, 10.6, 11.6, 12.7, 20.6, 24.4] * u.um
        self.imager = Camera(w, self.shape, self.ps, location=self.location)

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

def linear_pol(QI, UI, sig_QI=None, sig_UI=None):
    """Compute total linear polarization.

    The Ricean correction is not applied.

    Parameters
    ----------
    QI, UI : float or ndarray
      Stokes parameters Q/I and U/I.
    sig_QI, sig_UI : float or ndarray, optional
      Uncertainties in Q/I and U/I.

    Returns
    -------
    p, sig_p : float or ndarray
      Total linear polarization and uncertainty.  `sig_p` will be None
      if either of `sig_QI` or `sig_UI` is `None`.

    """

    p = np.sqrt(QI**2 + UI**2)
    if any([sig_QI is None, sig_UI is None]):
        sig_p = None
    else:
        sig_p = np.sqrt((sig_QI * QI)**2 + (sig_UI * UI)**2) / p
    return p, sig_p

def linear_pol_angle(QI, UI, sig_QI, sig_UI):
    """Compute linear polarization position angle.

    Parameters
    ----------
    QI, UI : float or ndarray
      Stokes parameters Q/I and U/I.
    sig_QI, sig_UI : float or ndarray, optional
      Uncertainties in Q/I and U/I.

    Returns
    -------
    theta, sig_theta : float or ndarray
      Position angle and uncertainty.  `sig_theta` will be `None` if
      either of `sig_QI` or `sig_UI` is `None`. [degrees]

    """

    p, sig_p = linear_pol(QI, UI, sig_QI, sig_UI)
    theta = np.degrees(np.arctan2(UI, QI) / 2.0)
    theta = (theta + 720) % 180.0
    if sig_p is None:
        sig_theta = None
    else:
        sig_theta = np.maximum(0, np.degrees(sig_p / p) / 2.0)
        sig_theta = np.minimum(sig_theta, 180)
    return theta, sig_theta

def ricean_correction(p, sig_p, theta, sig_theta):
    """Apply Ricean correction to the polarization and angle.

    The Ricean correction, as implemented, is described in Wardle &
    Kronberg 1974, ApJ, 194, 249-255.

    Parameters
    ----------
    p, theta : float or array
      Total linear polarization and position angle. [degrees]
    sig_p, sig_theta : float or array
      Uncertainties in `p` and `theta`.

    Returns
    -------
    p, sig_p, theta, sig_th : float or ndarray
      Corrected values.  Formally, `theta` and `sig_theta` do not need
      to be updated, but will be set to 0 and 180 when the
      polarization cannot be computed.

    Notes
    -----
    Thanks to Dan Clemens (BU) for assistance with the polarization
    correction equations.

    """

    if sig_p is None:
        # nothing to do
        return p, sig_p, theta, sig_theta

    if np.iterable(p):
        i = (1 - (sig_p / p)**2) < 0
        p = (p * np.sqrt(1 - (sig_p / p)**2))
        if np.any(i):
            p[i] = 0.0
            theta[i] = 0.0
            sig_theta[i] = 180.0
    else:
        p = (p * np.sqrt(1 - (sig_p / p)**2))

    return p, sig_p, theta, sig_theta

# update module docstring
from ..util import autodoc
autodoc(globals())
del autodoc
