# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
hb --- Hale-Bopp filter set photometric calibration
===================================================

.. autosummary::
   :toctree: generated/

   cal_oh
   continuum_color
   continuum_fluxd
   ext_aerosol_bc
   ext_aerosol_oh
   ext_total_oh
   ext_ozone_oh
   ext_rayleigh_oh

"""

import numpy as np
import astropy.units as u

from .core import *

__all__ = [
    'cal_oh',
    'continuum_color',
    'continuum_fluxd',
    'ext_aerosol_bc',
    'ext_aerosol_oh',
    'ext_total_oh',
    'ext_ozone_oh',
    'ext_rayleigh_oh',
    'remove_continuum'
]

def cal_oh(oh, oh_unc, OH, z_true, b, c, E_bc, h, guess=(20, 0.15),
           covar=False):
    """OH calibraton coefficients.

    Considers Rayleigh and ozone components for the OH filter.

    Solves Eq. 10 of Farnham et al. 2000.

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

def continuum_color(w0, f0, f0_unc, w1, f1, f1_unc, s0=None, s1=None):
    """Comet continuum color.

    Parameters
    ----------
    w0 : string or Quantity
      The shorter wavelength filter name (UC, BC, GC, RC), or the
      effective wavelength.
    f0 : Quantity
      The observed flux density through the shorter wavelength filter.
    f0_unc : Quantity
      The uncertainty in `f0`.
    w1, f1, f1_unc : various
      The same as above, but for the longer wavelength filter.
    s0, s1 : Quanitity, optional
      The solar flux desnities.  If `None`, `calib.solar_flux` or the
      values in Farnham et al. (2000) will be used.

    Returns
    -------
    Rm : Quantity
      The color in mangitudes per 0.1 um.
    Rp : Quantity
      The color in percent per 0.1 um.

    """

    from ..calib import solar_flux

    def Rp(Rm):
        return 10**(-0.4 * Rm.value) * Rm.unit * u.percent / u.mag

    if isinstance(w0, str):
        w0 = w0.upper()
    if isinstance(w1, str):
        w1 = w1.upper()

    if w0 == 'UC':
        if w1 == 'BC':
            Rm = 0.998 * (2.5 * np.log10(f1 / f0) - 1.101)
            return Rm, Rp(Rm)
        else:
            w0 = 0.3448 * u.um
    elif w0 == 'BC':
        if w1 == 'GC':
            Rm = 1.235 * (2.5 * np.log10(f1 / f0) + 0.507)
            return Rm, Rp(Rm)
        else:
            w0 = 0.4450 * u.um
    elif w0 == 'GC':
        if w1 == 'RC':
            Rm = 0.535 * (2.5 * np.log10(f1 / f0) + 0.769)
            return Rm, Rp(Rm)
        else:
            w0 = 0.5260 * u.um
    elif w0 == 'RC':
        w0 = 0.7128 * u.um

    assert isinstance(w0, u.Quantity)
    assert w0.unit.is_equivalent(u.m)

    if w1 == 'UC':
        w1 = 0.3448 * u.um
    elif w1 == 'BC':
        w1 = 0.4450 * u.um
    elif w1 == 'GC':
        w1 = 0.5260 * u.um
    elif w1 == 'RC':
        w1 = 0.7128 * u.um

    if s0 is None:
        s0 = solar_flux(w0)
    if s1 is None:
        s1 = solar_flux(w1)

    assert isinstance(w1, u.Quantity)
    assert w1.unit.is_equivalent(u.m)
    assert w0 < w1
    assert f0.unit.is_equivalent(f1.unit)
    assert s0.unit.is_equivalent(s1.unit)

    dw = (w1 - w0).to(0.1 * u.um)
    Rm = -2.5 * np.log10(f1.to(f0.unit) / f0 * s0.to(s1.unit) / s1) * u.mag / dw
    return Rm, Rp(Rm)

def continuum_fluxd(m_bc, Rm, filt):
    """Extrapolate BC continuum to another filter.

    Table VI, Eqs. 34-40 and 42 of Farhnam et al. 2000.

    Parameters
    ----------
    m_bc : float
      Observed BC magnitude.
    Rm : float or Quantity
      Observed color, in units of magnitudes per 1000 A.
    filt : string
      Name of a filter: OH, NH, CN, C3, CO+, C2.

    """

    if isinstance(Rm, u.Quantity):
        Rm = u.Quantity(Rm, '10 mag / um')
    else:
        Rm = Rm * u.Unit('10 mag / um')

    filt = filt.upper()
    if filt == 'OH':
        fc = 10**(-0.4 * m_bc) * 10**(-0.4 * 1.791) * 10**(-0.5440 * Rm.value)
        # 10.560e-9 erg/cm2/s/A = 1.0560e-7 W/m2/um
        fc *= 1.056e-7 * u.Unit('W/(m2 um)')
    else:
        raise ValueError('{} not yet implemented.'.format(filt))

    return fc

def ext_aerosol_bc(E_bc, h):

    """Aerosol extinction for BC filter.

    E_A_BC, Eq. 13 of Farnham et al. 2000.

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

    E_A_OH, Eq. 14 of Farnham et al. 2000.

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

    G_O_OH, Eq. 15 of Farnham et al. 2000.

    Parameters
    ----------
    z_true : Angle or Quantity
      The source true zenith angle.
    toz : float
      The amount of ozone.  0.15 is a good starting guess.
    c : string or 4x4 array
      The ozone c_ij coefficients.  Use 'B', 'G', 'OH', or '25%' for
      the corresponding coefficients from Table VIII of Farhnam et
      al. (2000).

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

    G_R_OH, Eq. 11 of Farnham et al. 2000.

    Parameters
    ----------
    z_true : Angle or Quantity
      The source true zenith angle.
    h : Quantity
      The observer's elevation.
    b : string or 2-element array
      The Rayleigh b_i coefficients.  Use 'B', 'G', 'OH', or '25%' for
      the corresponding coefficients from Table VIII of Farhnam et
      al. (2000).

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

def remove_continuum(f, fc, filt):
    """Remove the dust from a gas filter.

    Table VI and Eqs. 45-51 of Farnham et al. 2000.

    Parameters
    ----------
    f : Quantity
      The observed flux density through the gas filter.
    fc : Quantity
      The estimated continuum flux density through the gas filter.
    filt : str
      The name of the gas filter.

    Returns
    -------
    fgas : Quantity

    """

    fc = u.Quantity(fc, f.unit)

    filt = filt.upper()
    if filt == 'OH':
        fgas = (f - fc) / 1.698e-2
    else:
        raise ValueError('{} not yet implemented.'.format(filt))

    return fgas

# update module docstring
from ..util import autodoc
autodoc(globals())
del autodoc
