# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
hb --- Hale-Bopp filter set photometric calibration
===================================================

.. autosummary::
   :toctree: generated/

   cal_oh
   continuum_color
   ext_aerosol_bc
   ext_aerosol_oh
   ext_total_oh
   ext_ozone_oh
   ext_rayleigh_oh
   fluxd_continuum
   fluxd_oh
   remove_continuum

"""

import numpy as np
import astropy.units as u

from .core import *

__all__ = [
    'cal_oh',
    'continuum_color',
    'ext_aerosol_bc',
    'ext_aerosol_oh',
    'ext_total_oh',
    'ext_ozone_oh',
    'ext_rayleigh_oh',
    'fluxd_continuum',
    'fluxd_oh',
    'remove_continuum',
]

# Table I of Farnham et al. 2000
filters = set(['OH', 'NH', 'CN', 'C3', 'CO+', 'C2', 'H2O+', 'UC',
               'BC', 'GC', 'RC'])

cw = {  # center wavelengths
      'OH': u.Quantity(0.3097, 'um'),
      'NH': u.Quantity(0.3361, 'um'),
      'UC': u.Quantity(0.3449, 'um'),
      'CN': u.Quantity(0.3869, 'um'),
      'C3': u.Quantity(0.4063, 'um'),
     'CO+': u.Quantity(0.4266, 'um'),
      'BC': u.Quantity(0.4453, 'um'),
      'C2': u.Quantity(0.5135, 'um'),
      'GC': u.Quantity(0.5259, 'um'),
    'H2O+': u.Quantity(0.7028, 'um'),
      'RC': u.Quantity(0.7133, 'um'),
       'R': u.Quantity(0.641, 'um')  # Bessell 1998
}

cw_50 = {  # 50% power width
      'OH': u.Quantity( 58, 'AA'),
      'NH': u.Quantity( 54, 'AA'),
      'UC': u.Quantity( 79, 'AA'),
      'CN': u.Quantity( 56, 'AA'),
      'C3': u.Quantity( 58, 'AA'),
     'CO+': u.Quantity( 64, 'AA'),
      'BC': u.Quantity( 61, 'AA'),
      'C2': u.Quantity(119, 'AA'),
      'GC': u.Quantity( 56, 'AA'),
    'H2O+': u.Quantity(164, 'AA'),
      'RC': u.Quantity( 58, 'AA')
}


# Table VI of Farnham et al. 2000
F_0 = {  # Zero magnitude flux density
      'OH': u.Quantity(1.0560e-7, 'W/(m2 um)'),
      'NH': u.Quantity(8.420e-8, 'W/(m2 um)'),
      'CN': u.Quantity(8.6e-8,   'W/(m2 um)'),
      'C3': u.Quantity(8.160e-8, 'W/(m2 um)'),
     'CO+': u.Quantity(7.323e-8, 'W/(m2 um)'),
      'C2': u.Quantity(3.887e-8, 'W/(m2 um)'),
    'H2O+': u.Quantity(1.380e-8, 'W/(m2 um)'),
      'UC': u.Quantity(7.802e-8, 'W/(m2 um)'),
      'BC': u.Quantity(6.210e-8, 'W/(m2 um)'),
      'GC': u.Quantity(3.616e-8, 'W/(m2 um)'),
      'RC': u.Quantity(1.316e-8, 'W/(m2 um)'),
       'R': u.Quantity(2.177e-8, 'W/(m2 um)')  # Bessell 1998
}

MmBC_sun = {  # M - BC for the Sun
      'OH':  1.791,
      'NH':  1.188,
      'CN':  1.031,
      'C3':  0.497,
     'CO+':  0.338,
      'C2': -0.423,
    'H2O+': -1.249,
      'UC':  1.101,
      'BC':  0.000,
      'GC': -0.507,
      'RC': -1.276,
       'R': -0.90  # From solar mags, below
}

gamma_XX = {
      'OH': 1.698e-2,
      'NH': 1.907e-2,
      'CN': 1.812e-2,
      'C3': 3.352e-2,
     'CO+': 1.549e-2,
      'C2': 5.433e-3,
    'H2O+': 5.424e-3
}

gamma_prime_XX = {
      'OH': 0.98,
      'NH': 0.99,
      'CN': 0.99,
      'C3': 0.19,
     'CO+': 0.99,
      'C2': 0.66,
    'H2O+': 1.00
}

Msun = {  # apparent magnitude of the Sun, based on Appendix D.
    'UC': -25.17,
    'BC': -26.23,
    'GC': -26.77,
    'RC': -27.44,
     'R': -27.13,  # Bessell 1998
}

S0 = {  # Solar flux density at 1 AU, based on Appendix D.
    'UC': u.Quantity(908.9, 'W/(m2 um)'),
    'BC': u.Quantity(1934, 'W/(m2 um)'),
    'GC': u.Quantity(1841, 'W/(m2 um)'),
    'RC': u.Quantity(1250, 'W/(m2 um)'),
     'R': u.Quantity(1534, 'W/(m2 um)')  # Bessell 1998
}

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

def continuum_color(w0, m0, m0_unc, w1, m1, m1_unc, s0=None, s1=None):
    """Comet continuum color.

    The color in percent per 0.1 um = 10**(-0.4 * Rm)

    Parameters
    ----------
    w0 : string or Quantity
      The shorter wavelength filter name, or the effective wavelength.
      May be any of UC, BC, GC, RC, or R.
    m0 : float
      The apparant magnitude through the shorter wavelength filter.
    m0_unc : Quantity
      The uncertainty in `f0`.
    w1, m1, m1_unc : various
      The same as above, but for the longer wavelength filter.
    s0, s1 : float, optional
      The magnitude of the Sun.

    Returns
    -------
    Rm, Rm_unc : Quantity
      The color in mangitudes per 0.1 um, and uncertainty.

    """

    from ..calib import solar_flux

    if isinstance(w0, str):
        w0 = w0.upper()
    if isinstance(w1, str):
        w1 = w1.upper()

    if w0 in filters and w1 in filters:
        assert cw[w1] > cw[w0], 'w0 must be the shorter wavelength bandpass'
        dw = (cw[w1] - cw[w0]).to(0.1 * u.um)
        ci = MmBC_sun[w0] - MmBC_sun[w1]
        Rm = (m0 - m1 - ci) * u.mag / dw
        Rm_unc = np.sqrt(m0_unc**2 + m1_unc**2) * u.mag / dw
        return Rm, Rm_unc

    if w0 in Msun:
        s0 = Msun[w0]
        w0 = cw[w0]
    if w1 in Msun:
        s1 = Msun[w1]
        w1 = cw[w1]

    assert isinstance(w0, u.Quantity)
    assert w0.unit.is_equivalent(u.m)

    assert s0 is not None        
    assert s1 is not None

    assert isinstance(w1, u.Quantity)
    assert w1.unit.is_equivalent(u.m)
    assert w0 < w1

    dw = (w1 - w0).to(0.1 * u.um)
    ci = s0 - s1
    Rm = (m0 - m1 - ci) * u.mag / dw
    Rm_unc = np.sqrt(m0_unc**2 + m1_unc**2) * u.mag / dw
    return Rm, Rm_unc

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

def fluxd_continuum(bc, bc_unc, Rm, Rm_unc, filt):
    """Extrapolate BC continuum to another filter.

    Table VI, Eqs. 34-40 and 42 of Farhnam et al. 2000.

    Parameters
    ----------
    bc, bc_unc : float
      Observed BC magnitude and uncertainty.
    Rm, Rm_unc : float or Quantity
      Observed color, in units of magnitudes per 1000 A, and uncertainty.
    filt : string
      Name of the other filter: OH, NH, CN, C3, CO+, C2.

    Returns
    -------
    fc, fc_unc : Quantity
      Continuum flux density and uncertainty.

    """

    Rm = u.Quantity(Rm, '10 mag / um').value
    Rm_unc = u.Quantity(Rm_unc, '10 mag / um').value

    filt = filt.upper()
    if filt == 'OH':
        fc = (10**(-0.4 * bc) * 10**(-0.4 * MmBC_sun[filt])
              * 10**(-0.5440 * Rm))
        m_unc = np.sqrt(bc_unc**2 + (0.544 / 0.4 * Rm_unc)**2)
        fc_unc = fc * m_unc / 1.0857
    else:
        raise ValueError('{} not yet implemented.'.format(filt))

    fc *= F_0[filt]
    fc_unc *= F_0[filt]
    return fc, fc_unc

def fluxd_oh(oh, oh_unc, bc, bc_unc, Rm, Rm_unc, zp, toz, z_true, E_bc, h):
    """Flux from OH.

    Appendix A and D of Farnham et al. 2000.

    Parameters
    ----------
    oh, oh_unc : float
      OH instrumental magnitude and uncertainty.
    bc, bc_unc : float
      BC instrumental magnitude and uncertainty.
    Rm, Rm_unc : float or Quantity
      Continuum color in units of magnitudes per 0.1 um, and
      uncertainty.
    zp : float
      OH magnitude zero point.
    toz : float
      Amount of ozone.
    z_true : Angle or Quantity
      True zenith angle.
    E_bc : float
      BC airmass extinction. [mag/airmass]
    h : Quantity
      The observer's elevation.

    Returns
    -------
    E_tot, E_unc : float
      Total OH extinction and uncertainty.
    f_oh, f_oh_unc : Quantity
      Total band flux from OH and uncertainty.

    Notes
    -----
    1) Compute extinction for pure OH and 25% continuum cases.

    2) Use pure OH extinction to compute band flux.

    3) From step 2, compute percent continuum contribution, and
       linearly interpolate between the two cases of step 1 to
       determine actual extinction.

    4) Given extinction from step 3, re-compute band flux.

    """

    E_0 = ext_total_oh(toz, z_true, 'OH', 'OH', E_bc, h)
    E_25 = ext_total_oh(toz, z_true, '25%', '25%', E_bc, h)
    fc, fc_unc = fluxd_continuum(bc, bc_unc, Rm, Rm_unc, 'OH')
    f = 10**(-0.4 * (oh + zp - E_0)) * F_0['OH']
    frac = (1 - (f - fc) / f)  # fraction that is continuum
    frac_unc = frac * np.sqrt(oh_unc**2 + bc_unc**2) / 1.0857
    assert frac < 0.25, "Continuum = {:%}, more than 25% of observed OH band flux density.".format(frac)
    E_tot = 4 * ((0.25 - frac) * E_0 + frac * E_25)
    E_unc = 1.0857 * frac
    f = 10**(-0.4 * (oh + zp - E_tot)) * F_0['OH']
    f_unc = f * frac_unc
    f_oh, f_oh_unc = remove_continuum(f, f_unc, fc, fc_unc, 'OH')
    return E_tot, E_unc, f_oh, f_oh_unc
    
def remove_continuum(f, f_unc, fc, fc_unc, filt):
    """Remove the dust from a gas filter.

    Table VI and Eqs. 45-51 of Farnham et al. 2000.

    Parameters
    ----------
    f, f_unc : float or Quantity
      The observed flux density through the gas filter and
      uncertainty.
    fc, fc_unc : float or Quantity
      The estimated continuum flux density through the gas filter and
      uncertainty.
    filt : str
      The name of the gas filter.

    Returns
    -------
    fgas, fgas_unc : Quantity

    """

    filt = filt.upper()
    assert filt in filters

    if isinstance(f, u.Quantity):
        f_unc = u.Quantity(f_unc, f.unit)
        fc = u.Quantity(fc, f.unit)
        fc_unc = u.Quantity(fc_unc, f.unit)
    elif isinstance(fc, u.Quantity):
        fc_unc = u.Quantity(fc_unc, f.unit)
        f = u.Quantity(f, f.unit)
        f_unc = u.Quantity(f_unc, f.unit)

    if filt in ['OH', 'C3', 'C2', 'H2O+']:
        fgas = (f - fc) / gamma_XX[filt]
        fgas_unc = np.sqrt(f_unc**2 + fc_unc**2) / gamma_XX[filt]
    else:
        raise ValueError('{} not yet implemented.'.format(filt))

    return fgas, fgas_unc

# update module docstring
from ..util import autodoc
autodoc(globals())
del autodoc
