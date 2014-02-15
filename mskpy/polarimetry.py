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
    p_eff : float, optional
      Polarization efficiency correction.
    rc_correct : bool, optional
      If `True`, enable the Ricean correction.
    eff_correct : bool, optional
      If `True`, enable the polarization efficiency correction.

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
    rotate - This linear polarization object, rotated.
    table - A table of results.

    """

    def __init__(self, I, QI, UI, sig_I=None, sig_QI=None, sig_UI=None,
                 p_eff=1.0, sig_p_eff=0.0, rc_correct=True,
                 p_eff_correct=True):
        from astropy.coordinates import Angle

        self.I = I
        self.QI = QI
        self.UI = UI

        self.sig_I = sig_I
        self.sig_QI = sig_QI
        self.sig_UI = sig_UI

        self.p_eff = p_eff
        self.sig_p_eff = sig_p_eff

        self.p_eff_correct = p_eff_correct
        self.rc_correct = rc_correct

        self.rotated = False
        self.dtheta = Angle(0.0 * u.deg)
        self.sig_dtheta = Angle(0.0 * u.deg)

    @property
    def _pth(self):
        if self.p_eff_correct:
            kwargs = dict(p_eff=self.p_eff, sig_p_eff=self.sig_p_eff)
        else:
            kwargs = dict(p_eff=1.0, sig_p_eff=0.0)

        pth = linear_pol(self.QI, self.UI, self.sig_QI, self.sig_UI,
                         **kwargs)

        if self.rc_correct:
            pth = ricean_correction(*pth)

        return pth
            
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

    def rotate(self, dth, sig_dth=None):
        """Rotate the polarization vector.

        The vector is rotated by the given angle, e.g.::
          >>> print pol.theta
          0.0
          >>> print pol.rotate(90 * u.deg).theta
          90.0

        Parameters
        ----------
        dth : astropy Angle or Quantity
          The amount to rotate.
        sig_dth : astropy Angle or Quantity, optional
          The uncertainty in the rotation.

        Returns
        -------
        rpol : LinPol
          The rotated vector.

        """
        from astropy.coordinates import Angle

        if sig_dth is None:
            sig_dth = 0 * u.deg

        dth = Angle(dth)
        sig_dth = Angle(sig_dth)

        c = np.cos(2 * dth).value
        s = np.sin(2 * dth).value

        # Rotate uncorrected QI, UI.
        rc_correct = self.rc_correct
        self.rc_correct = False

        QI =  self.QI * c + self.UI * s
        UI = -self.QI * s + self.UI * c
        if (self.sig_QI is None) or (self.sig_UI is None):
            sig_QI = None
            sig_UI = None
        else:
            sig_QI = np.sqrt((self.sig_QI * c)**2 + (self.sig_UI * s)**2
                             + (self.QI * 2.0 * sig_dth.radian * c)**2
                             + (self.UI * 2.0 * sig_dth.radian * s)**2)
            sig_UI = np.sqrt((self.sig_QI * s)**2 + (self.sig_UI * c)**2
                             + (self.QI * 2.0 * sig_dth.radian * s)**2
                             + (self.UI * 2.0 * sig_dth.radian * c)**2)

        self.rc_correct = rc_correct
        newp =  LinPol(self.I, QI, UI, sig_I=self.sig_I,
                       sig_QI=sig_QI, sig_UI=sig_UI,
                       p_eff=self.p_eff, sig_p_eff=self.sig_p_eff,
                       rc_correct=self.rc_correct,
                       p_eff_correct=self.p_eff_correct)
        newp.rotated = True
        newp.dtheta = self.dtheta + dth
        newp.sig_dtheta = np.sqrt(self.sig_dtheta**2 + sig_dth**2)

    def table(self):
        """The data, formatted as a table.

        Best for 1D vectors of intensity/polarization.

        Returns
        -------
        tab : astropy.table.Table
          A table of I, Q, U, P, and th (and, potentially,
          uncertainties).

        """
        from astropy.table import Table
        data = (self.I, self.sig_I,
                self.QI, self.sig_QI,
                self.UI, self.sig_UI,
                self.p, self.sig_p,
                self.theta, self.sig_theta)
        names = ['I', 'sig_I',
                 'QI', 'sig_QI',
                 'UI', 'sig_UI',
                 'P', 'sig_P',
                 'th', 'sig_th']
        if self.sig_I is None:
            tab = Table(data[::2], names=names[::2])
        else:
            tab = Table(data, names=names)
        tab.meta['rc_applied'] = self.rc_correct
        tab.meta['p_eff_applied'] = self.p_eff_correct
        tab.meta['p_eff'] = self.p_eff
        tab.meta['sig_p_eff'] = self.sig_p_eff
        tab.meta['rotated'] = self.rotated
        if self.rotated:
            tab.meta['dtheta'] = self.dtheta
            tab.meta['sig_dtheta'] = self.sig_dtheta
        return tab

class HalfWavePlate(LinPol):
    """Linear polarimetry with a 1/2-wave plate.

    Parameters
    ----------
    I : array
      Intensities from each polarization angle: `[I0, I45, I90,
      I135]`.  Each intensity may in turn be an array.
    sig_I : array, optional
      Intensity uncertainties, same form as `I`.
    p_eff, sig_p_eff : float, optional
      Polarization efficiency and uncertainty.
    rc_correct : bool, optional
    eff_correct : bool, optional
      Ricean and polarization efficiency correction flags.  See `LinPol`.
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

    def __init__(self, I, sig_I=None, p_eff=1.0, sig_p_eff=0.0,
                 rc_correct=True, p_eff_correct=True, flipU=False):
        from astropy.coordinates import Angle

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

        self.p_eff = p_eff
        self.sig_p_eff = sig_p_eff
            
        self.rc_correct = rc_correct
        self.p_eff_correct = p_eff_correct
        self.flipU = flipU

        self.rotated = False
        self.dtheta = Angle(0.0 * u.deg)
        self.sig_dtheta = Angle(0.0 * u.deg)

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

def linear_pol(QI, UI, sig_QI=None, sig_UI=None, p_eff=1.0, sig_p_eff=0.0):
    """Compute total linear polarization and angle.

    The Ricean correction is not applied.

    Parameters
    ----------
    QI, UI : float or ndarray
      Stokes parameters Q/I and U/I.
    sig_QI, sig_UI : float or ndarray, optional
      Uncertainties in Q/I and U/I.
    p_eff : float, optional
      The polarization efficiency.
    sig_p_eff : float, optional
      The polarization efficiency uncertainty.

    Returns
    -------
    p, sig_p : float or ndarray
      Total linear polarization and uncertainty.  `sig_p` will be None
      if either of `sig_QI` or `sig_UI` is `None`.
    theta, sig_theta : float or ndarray
      Position angle and uncertainty.  `sig_theta` will be `None` if
      either of `sig_QI` or `sig_UI` is `None`. [degrees]

    """

    p = np.sqrt(QI**2 + UI**2) / p_eff
    theta = np.degrees(np.arctan2(UI, QI) / 2.0)
    theta = (theta + 720) % 180.0

    if any([sig_QI is None, sig_UI is None]):
        sig_p = None
        sig_theta = None
    else:
        sig_p = np.sqrt((sig_QI * QI)**2 + (sig_UI * UI)**2
                        + 2 * QI**2 * UI**2 * sig_p_eff**2) / p_eff / p
        sig_theta = np.maximum(0, np.degrees(sig_p / p) / 2.0)
        sig_theta = np.minimum(sig_theta, 180)

    return p, sig_p, theta, sig_theta

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
