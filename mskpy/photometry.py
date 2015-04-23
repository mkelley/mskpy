# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
photometry --- Tools for working with photometry.
=================================================

.. autosummary::
   :toctree: generated/

   airmass_app
   airmass_loc
   cal_airmass
   cal_color_airmass
   cal_oh
   ext_aerosol_bc
   ext_aerosol_oh
   ext_total_oh
   ext_ozone_oh
   ext_rayleigh_oh

"""

import numpy as np
import astropy.units as u

__all__ = [
    'airmass_app',
    'airmass_loc',
    'cal_airmass',
    'cal_color_airmass',
    'cal_oh',
    'ext_aerosol_bc',
    'ext_aerosol_oh',
    'ext_total_oh',
    'ext_ozone_oh',
    'ext_rayleigh_oh',
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
      An intial guess for the fitting algorithm.
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

def cal_oh(oh, oh_unc, OH, z_true, b, c, E_bc, h, guess=(20, 0.15),
           covar=False):
    """OH calibraton coefficients.

    Considers Rayleigh and ozone components for the OH filter.

    Solves Eq. 10 of Farnham & Schleicher 2000.

    Parameters
    ----------
    oh, oh_unc : array
      Instrumental (uncalibrated) OH magnitude and uncertainty.
    OH : array
      Calibrated OH magnitude.
    z_true : Angle or Quantity
      True zenith angle.
    b : array
      Coefficient sets to use for `ext_rayleigh_oh`.
    c : array
      Coefficient sets to use for `ext_ozone_oh`.
    E_bc : float
      BC airmass extinction. [mag/airmass]
    h : Quantity
      The observer's elevation.
    guess : array, optional
      An intial guess for the fitting algorithm: OH zero point, BC zero point, 
    covar : bool, optional
      Set to `True` to return the covariance matrix.

    Results
    -------
    A : ndarray
      The OH magnitude zero point, ozone amount.
    unc or cov : ndarray
      Uncertainties on each parameter, based on least-squares fitting,
      or the covariance matrix, if `covar` is `True`.

    """

    from scipy.optimize import leastsq

    assert np.size(E_bc) == 1

    def chi(A, oh, oh_unc, OH, z_true, b, c, E_bc, h):
        zp, toz = A
        toz = np.abs(toz)
        model = OH - zp + ext_total_oh(toz, z_true, b, c, E_bc, h)
        chi = (oh - model) / oh_unc
        return chi

    args = (np.array(oh), np.array(oh_unc), np.array(OH),
            z_true, b, c, E_bc, h)
    output = leastsq(chi, guess, args=args, full_output=True, epsfcn=1e-5)

    fit = output[0]
    fit[1] = np.abs(fit[1])  # toz is positive
    cov = output[1]
    err = np.sqrt(np.diag(cov))

    if covar:
        return fit, cov
    else:
        return fit, err

def ext_aerosol_bc(E_bc, h):
    """Aerosol extinction for BC filter.

    E_A_BC, Eq. 13 of Farnham & Schleicher 2000.

    Parameters
    ----------
    E_bc : float
      The total linear extinction in the BC filter.
    h : Quantity
      The observer's elevation.

    Returns
    -------
    E : array
      The extinction in magnitudes.

    """

    return E_bc - 0.2532 * np.exp(-h.to(u.km).value / 7.5)

def ext_aerosol_oh(E_bc, h):
    """Aerosol extinction for OH filter.

    E_A_OH, Eq. 14 of Farnham & Schleicher 2000.

    Parameters
    ----------
    E_bc : float
      The total linear extinction in the BC filter.
    h : Quantity
      The observer's elevation.

    Returns
    -------
    E : array
      The extinction in magnitudes.

    """

    #return (3097 / 4453.)**-0.8 * E_aerosol_bc(E_bc, h)
    return 1.33712 * ext_aerosol_bc(E_bc, h)

def ext_ozone_oh(z_true, toz, c):
    """Ozone extinction for OH filter.

    G_O_OH, Eq. 15 of Farnham & Schleicher 2000.

    Parameters
    ----------
    z_true : Angle or Quantity
      The source true zenith angle.
    toz : float
      The amount of ozone.  0.15 is a good starting guess.
    c : string or 4x4 array
      The ozone c_ij coefficients.  Use 'B', 'G', 'OH', or '25%' for
      the corresponding coefficients from Table VIII of Farhnam &
      Schleicher (2000).

    Returns
    -------
    G : array
      The extinction in magnitudes.

    """

    if isinstance(c, str):
        assert c.upper() in ['B', 'G', 'OH', '25%'], 'Invalid c name'
        if c.upper() == 'B':
            c = np.array([[ 1.323e-2, -1.605e-1,  4.258e-1,  9.099e-1],
                          [-1.731e-2,  3.273,    -2.815e-1, -2.221],
                          [ 5.349e-3, -5.031e-2, -4.182e-1,  1.649],
                          [ 2.810e-4, -1.195e-2,  1.063e-1, -3.877e-1]])
        elif c.upper() == 'G':
            c = np.array([[ 2.880e-2, -3.912e-1,  1.597,    -7.460e-1],
                          [-5.284e-2,  3.753,    -2.632,     8.912e-1],
                          [ 2.634e-2, -3.380e-1,  8.699e-1,  4.253e-2],
                          [-3.141e-3,  3.359e-2, -9.235e-2, -1.677e-1]])
        elif c.upper() == 'OH':
            c = np.zeros((4, 4))
            c[1, 0] = -1.669e-3
            c[1, 1] =  3.365
            c[1, 2] = -2.973e-2
            c[2, 0] =  3.521e-4
            c[2, 1] = -6.662e-3
            c[2, 2] = -5.335e-2
        elif c.upper() == '25%':
            c = np.array([[-2.523e-2,  4.382e-1, -2.415,     4.993],
                          [ 4.276e-2,  2.538,     4.109,    -8.518],
                          [-2.510e-2,  4.148e-1, -2.405,     4.666],
                          [ 5.070e-3, -8.168e-2,  4.272e-1, -8.618e-1]])
    else:
        c = np.array(c)
        assert c.shape == (4, 4), "c should have shape (4, 4)"

    a = np.matrix(c) * np.matrix([[1, toz, toz**2, toz**3]]).T
    a = np.squeeze(np.array(a))

    X = airmass_loc(z_true)
    if np.size(X) == 1:
        G = np.dot(a, np.r_[1, X, X**2, X**3])
    else:
        G = np.r_[[np.dot(a, np.r_[1, x, x**2, x**3]) for x in X]]

    return G

def ext_rayleigh_oh(z_true, h, b):
    """Rayliegh extinction for OH filter.

    G_R_OH, Eq. 11 of Farnham & Schleicher 2000.

    Parameters
    ----------
    z_true : Angle or Quantity
      The source true zenith angle.
    h : Quantity
      The observer's elevation.
    b : string or 2-element array
      The Rayleigh b_i coefficients.  Use 'B', 'G', 'OH', or '25%' for
      the corresponding coefficients from Table VIII of Farhnam &
      Schleicher (2000).

    Returns
    -------
    G : array
      The extinction in magnitudes.

    """

    if isinstance(b, str):
        b = b.upper()
        assert b in ['B', 'G', 'OH', '25%'], 'Invalid b name'
        if b == 'B':
            b = np.r_[1.159, -4.433e-4]
        elif b == 'G':
            b = np.r_[1.158, -5.359e-4]
        elif b == 'OH':
            b = np.r_[1.170, 0]
        elif b == '25%':
            b = np.r_[1.168, -1.918e-4]
    else:
        b = np.array(b)
        assert b.shape == (2,), "b should have shape (2,)"

    X = airmass_app(z_true, h)
    e = np.exp(-h.to(u.km).value / 7.5)
    if np.size(X) == 1:
        G = np.dot(b, [X, X**2]) * e
    else:
        G = np.r_[[np.dot(b, [x, x**2]) for x in X]] * e

    return G

def ext_total_oh(toz, z_true, b, c, E_bc, h):
    """Total OH extinction.

    G_R_OH + E_A_OH + G_O_OH, Eq. 10 of Farnham and Schleicher 2000.

    Parameters
    ----------
    toz : float
      Amount of ozone.
    z_true : Angle or Quantity
      True zenith angle.
    b : array
      Coefficient sets to use for `ext_rayleigh_oh`.
    c : array
      Coefficient sets to use for `ext_ozone_oh`.
    E_bc : float
      BC airmass extinction. [mag/airmass]
    h : Quantity
      The observer's elevation.
    guess : array, optional
      An intial guess for the fitting algorithm: OH zero point, BC zero point, 
    covar : bool, optional
      Set to `True` to return the covariance matrix.

    """

    return (ext_rayleigh_oh(z_true, h, b)
            + ext_aerosol_oh(E_bc, h) * airmass_app(z_true, h)
            + ext_ozone_oh(z_true, toz, c))

# update module docstring
from .util import autodoc
autodoc(globals())
del autodoc

