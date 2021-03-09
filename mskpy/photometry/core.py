# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
core --- Core code for photometry.
==================================

.. autosummary::
   :toctree: generated/

   Functions
   ---------
   airmass_app
   airmass_loc
   cal_airmass
   cal_color_airmass

"""

from ..util import autodoc
import numpy as np
import astropy.units as u

__all__ = [
    'airmass_app',
    'airmass_loc',
    'cal_airmass',
    'cal_color',
    'cal_color_airmass',
]


def airmass_app(z_true, h):
    """Apparent airmass.

    Hardie's 1962 formula, with a correction for atmospheric
    refraction.  Used by Farnham and Schleicher 2000.

    For OH, use `airmass_loc`.

    Parameters
    ----------
    z_true : Angle or Quantity
      The object's true zenith angle.
    h : Quantity
      The observer's elevation.

    """

    tan = np.tan(z_true)
    exp = np.exp(-h.to(u.km).value / 8.0)
    z_app = z_true - (60.4 * tan - 0.0668 * tan**3) / 3600. * exp * u.deg
    sec = 1.0 / np.cos(z_app)
    X = (sec - 0.0018167 * (sec - 1) - 0.002875 * (sec - 1)**2
         - 0.0008083 * (sec - 1)**3)
    return X.value


def airmass_loc(z_true):
    """Airmass based on local zenith angle.

    Use for OH extinction.

    Parameters
    ----------
    z : Angle or Quantity
      The object's true zenith angle.

    """

    R = 6378.
    H = 22.
    X = (R + H) / np.sqrt((R + H)**2 - (R * np.sin(z_true))**2)
    return X.value


def cal_airmass(m, munc, M, X, guess=(25., -0.1),
                covar=False):
    """Calibraton coefficients, based on airmass.

    Parameters
    ----------
    m : array
      Instrumental (uncalibrated) magnitude.
    munc : array
      Uncertainties on m.
    M : array
      Calibrated magnitude.
    X : array
      Airmass.
    guess : array, optional
      An intial guess for the fitting algorithm.
    covar : bool, optional
      Set to `True` to return the covariance matrix.

    Results
    -------
    A : ndarray
      The photometric zero point, and airmass correction slope. [mag,
      mag/airmass]
    unc or cov : ndarray
      Uncertainties on each parameter, based on least-squares fitting,
      or the covariance matrix, if `covar` is `True`.

    """

    from scipy.optimize import leastsq

    def chi(A, m, munc, M, X):
        model = M - A[0] + A[1] * X
        chi = (np.array(m) - model) / np.array(munc)
        return chi

    output = leastsq(chi, guess, args=(m, munc, M, X),
                     full_output=True, epsfcn=1e-3)
    fit = output[0]
    cov = output[1]
    err = np.sqrt(np.diag(cov))

    if covar:
        return fit, cov
    else:
        return fit, err


def cal_color_airmass(m, munc, M, color, X, guess=(25., -0.1, -0.01),
                      covar=False):
    """Calibraton coefficients, based on airmass and color index.

    Parameters
    ----------
    m : array
      Instrumental (uncalibrated) magnitude.
    munc : array
      Uncertainties on m.
    M : array
      Calibrated magnitude.
    color : array
      Calibrated color index, e.g., V - R.
    X : array
      Airmass.
    guess : array, optional
      An initial guess for the fitting algorithm.
    covar : bool, optional
      Set to `True` to return the covariance matrix.

    Results
    -------
    A : ndarray
      The photometric zero point, airmass correction slope, and color
      correction slope. [mag, mag/airmass, mag/color index]
    unc or cov : ndarray
      Uncertainties on each parameter, based on least-squares fitting,
      or the covariance matrix, if `covar` is `True`.

    """

    from scipy.optimize import leastsq

    def chi(A, m, munc, M, color, X):
        model = M - A[0] + A[1] * X + A[2] * color
        chi = (np.array(m) - model) / np.array(munc)
        return chi

    output = leastsq(chi, guess, args=(m, munc, M, color, X),
                     full_output=True, epsfcn=1e-3)
    fit = output[0]
    cov = output[1]
    err = np.sqrt(np.diag(cov))

    if covar:
        return fit, cov
    else:
        return fit, err


def cal_color(m, munc, M, color, guess=(25., -0.01),
              covar=False):
    """Calibraton coefficients, based on color index.

    Parameters
    ----------
    m : array
      Instrumental (uncalibrated) magnitude.
    munc : array
      Uncertainties on m.
    M : array
      Calibrated magnitude.
    color : array
      Calibrated color index, e.g., V - R.
    guess : array, optional
      An initial guess for the fitting algorithm.
    covar : bool, optional
      Set to `True` to return the covariance matrix.

    Results
    -------
    A : ndarray
      The photometric zero point and color correction slope. [mag,
      mag/color index]

    unc or cov : ndarray
      Uncertainties on each parameter, based on least-squares fitting,
      or the covariance matrix, if `covar` is `True`.

    """

    from scipy.optimize import leastsq

    def chi(A, m, munc, M, color):
        model = M - A[0] + A[1] * color
        chi = (np.array(m) - model) / np.array(munc)
        return chi

    output = leastsq(chi, guess, args=(m, munc, M, color),
                     full_output=True, epsfcn=1e-3)
    fit = output[0]
    cov = output[1]
    err = np.sqrt(np.diag(cov))

    if covar:
        return fit, cov
    else:
        return fit, err


# update module docstring
autodoc(globals())
del autodoc
