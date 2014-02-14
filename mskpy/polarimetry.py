# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
polarimetry --- Polarization tools.
===================================

   Classes
   -------
   LinPol
   HalfWavePlate

   Functions
   ---------
   linear_pol
   linear_pol_angle
   ricean_correction

"""

import numpy as np
import astropy.units as u

__all__ = [
    'LinPol'
    'HalfWavePlate',
]
        
class LinPol(object):
    """Describing linear polarization, parameterized with I, QI, and UI.

    Parameters
    ----------
    I : float or array
      Total intensity.
    QI, UI : float or array
      The linear polarization parameters normalized by intensity:
      `Q/I`, `U/I`.
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

    def __init__(self, I, QI, UI, sig_I=None, sig_QI=None, sig_UI=None,
                 correct=True):
        self.I = I
        self.QI = QI
        self.UI = UI

        self.sig_I = sig_I
        self.sig_QI = sig_QI
        self.sig_UI = sig_UI

        self.correct = correct

    @property
    def _pth(self):
        from astropy.coordinates import Angle
        args = self.QI, self.UI, self.sig_QI, self.sig_UI
        p = linear_pol(*args)
        th = linear_pol_angle(*args)
        th = (Angle(th[0] * u.deg),
              None if th[1] is None else Angle(th[1] * u.deg))
        if self.correct:
            return ricean_correction(*(p + th))
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

    def rotate(self, a):
        """Rotate the polarization vector.

        The vector is rotated by the amount `a`, e.g.::
          >>> print pol.theta
          <Angle 0 deg>
          >>> print pol.rotate(90 * u.deg).theta
          <Angle 90 deg>

        Parameters
        ----------
        a : astropy Angle or Quantity
          The amount to rotate.

        Returns
        -------
        rpol : LinPol
          The rotated vector.

        """

        c = np.cos(2 * a)
        s = np.sin(2 * a)

        # We want uncorrected vectors
        correct = self.correct
        self.correct = False

        QI =  self.QI * c + self.UI * s
        UI = -self.QI * s + self.UI * c
        if (self.sig_QI is None) or (self.sig_UI is None):
            sig_QI = None
            sig_UI = None
        else:
            sig_QI =  self.sig_QI * c + self.sig_UI * s
            sig_UI = -self.sig_QI * s + self.sig_UI * c

        self.correct = correct
        return LinPol(I, QI, UI, sig_I=sig_I, sig_QI=sig_QI, sig_UI=sig_UI,
                      correct=correct)

class HalfWavePlate(LinPol):
    """Linear polarimetry with a 1/2-wave plate.

    Parameters
    ----------
    I : array
      Intensities from each polarization angle: `[I0, I45, I90,
      I135]`.  Each intensity may in turn be an array.
    sig_I : array, optional
      Intensity uncertainties, same form as `I`.
    correct : bool, optional
      Set to `False` to prevent the Ricean correction from being
      applied.
    flipU : bool, optional
      Set to `True` to reverse the sense of the `U` parameter.

    Attributes
    ----------
    I0, I45, I90, I135 : float or array
      Intensities.
    sig_I0, sig_I45, sig_I90, sig_I135 : float or array
      Uncertainties on intensities, or `None` if not provided.
    I, QI, UI : float or array
      Total intensity and normalized Q and U parameters.
    sig_I, sig_QI, sig_UI : float or array
      Uncertainties on total intensity and normalized Q and U
      parameters, or `None` if not provided.
    p, sig_p : float or array
      Total linear polarization.  Uncertainty is `None` if it cannot
      be computed.
    theta, sig_theta
      Polarization position angle.  Uncertainty is `None` if it cannot
      be computed.

    """

    def __init__(self, I, sig_I=None, correct=True, flipU=False):
        self.I0 = I[0]
        self.I45 = I[1]
        self.I90 = I[2]
        self.I135 = I[3]

        if sig_I is None:
            self.sig_I0 = None
            self.sig_I45 = None
            self.sig_I90 = None
            self.sig_I135 = None
        else:
            self.sig_I0 = sig_I[0]
            self.sig_I45 = sig_I[1]
            self.sig_I90 = sig_I[2]
            self.sig_I135 = sig_I[3]

        self.correct = correct
        self.flipU = flipU

    def __getitem__(self, k):
        if k in ['I0', 'I45', 'I90', 'I135', 'sig_I0',
                 'sig_I45', 'sig_I90', 'sig_I135']:
            return self.__dict__[k]
        else:
            raise KeyError

    def __setitem__(self, k, v):
        if k in ['I0', 'I45', 'I90', 'I135', 'sig_I0',
                 'sig_I45', 'sig_I90', 'sig_I135']:
            self.__dict__[k] = v
        else:
            raise KeyError

    @property
    def I(self):
        return (self.I0 + self.I45 + self.I90 + self.I135) / 2.0

    @property
    def sig_I(self):
        if any((self.sig_I0 is None, self.sig_I45 is None,
                self.sig_I90 is None, self.sig_I135 is None)):
            return None
        else:
            return np.sqrt(self.sig_I0**2 + self.sig_I45**2 + self.sig_I90**2
                           + self.sig_I135**2)

    @property
    def QI(self):
        return (self.I0 - self.I90) / (self.I0 + self.I90)

    @property
    def sig_QI(self):
        if any((self.sig_I0 is None, self.sig_I90 is None)):
            return None
        else:
            return (np.sqrt(self.sig_I0**2 + self.sig_I90**2)
                    / (self.I0 + self.I90))

    @property
    def UI(self):
        scale = -1 if self.flipU else 1
        return scale * (self.I45 - self.I135) / (self.I45 + self.I135)

    @property
    def sig_UI(self):
        if any((self.sig_I45 is None, self.sig_I135 is None)):
            return None
        else:
            return (np.sqrt(self.sig_I45**2 + self.sig_I135**2)
                    / (self.I45 + self.I135))

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
from .util import autodoc
autodoc(globals())
del autodoc
