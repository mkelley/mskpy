# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
hb --- Hale-Bopp filter set photometric calibration
===================================================

.. autosummary::
   :toctree: generated/

   cal_oh
   continuum_color
   continuum_colors
   estimate_continuum
   ext_aerosol_bc
   ext_aerosol_oh
   ext_total_oh
   ext_ozone_oh
   ext_rayleigh_oh
   flux_gas
   flux_oh
   Rm2S

   todo: Need more uncertainty propagations.

"""

import numpy as np
import astropy.units as u

from .core import *

__all__ = [
    'cal_oh',
    'continuum_color',
    'continuum_colors',
    'estimate_continuum',
    'ext_aerosol_bc',
    'ext_aerosol_oh',
    'ext_total_oh',
    'ext_ozone_oh',
    'ext_rayleigh_oh',
    'flux_gas',
    'flux_oh',
    'fluxd_continuum',
    'Rm2S'
]

# Table I of Farnham et al. 2000
filters = set(['OH', 'NH', 'CN', 'C3', 'CO+', 'C2', 'H2O+', 'UC',
               'BC', 'GC', 'RC'])

# Including V, R, etc.
all_filters = filters | set(['R', 'V', 'SDSS-R', 'SDSSR'])

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
       'R': u.Quantity(0.641,  'um'),  # Bessell 1998
       'V': u.Quantity(0.545,  'um'),  # Bessell 1998
  'SDSS-R': u.Quantity(0.6222, 'um'),  # Smith et al. 2002
   'SDSSR': u.Quantity(0.6222, 'um'),
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
      'OH': u.Quantity(10.560e-8, 'W/(m2 um)'),
      'NH': u.Quantity( 8.420e-8, 'W/(m2 um)'),
      'CN': u.Quantity( 8.6e-8,   'W/(m2 um)'),
      'C3': u.Quantity( 8.160e-8, 'W/(m2 um)'),
     'CO+': u.Quantity( 7.323e-8, 'W/(m2 um)'),
      'C2': u.Quantity( 3.887e-8, 'W/(m2 um)'),
    'H2O+': u.Quantity( 1.380e-8, 'W/(m2 um)'),
      'UC': u.Quantity( 7.802e-8, 'W/(m2 um)'),
      'BC': u.Quantity( 6.210e-8, 'W/(m2 um)'),
      'GC': u.Quantity( 3.616e-8, 'W/(m2 um)'),
      'RC': u.Quantity( 1.316e-8, 'W/(m2 um)'),
       'R': u.Quantity( 2.177e-8, 'W/(m2 um)'), # Bessell 1998
       'V': u.Quantity( 3.631e-8, 'W/(m2 um)'), # Bessell 1998
  'SDSS-R': u.Quantity( 2.812e-8, 'W/(m2 um)'), # Smith et al. 2002 for zeropoint in Jy and effective wavelength 6222 Å.
   'SDSSR': u.Quantity(2.812e-8, 'W/(m2 um)'),
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
       'V': -0.53,  # From solar mags, below
       'R': -0.90,  # From solar mags, below
  'SDSS-R': -0.70,  # From solar mags, below
   'SDSSR': -0.70,
}

gamma_XX_XX = {
      'OH': u.Quantity(1.698e-2, '1/AA'),
      'NH': u.Quantity(1.907e-2, '1/AA'),
      'CN': u.Quantity(1.812e-2, '1/AA'),
      'C3': u.Quantity(3.352e-2, '1/AA'),
     'CO+': u.Quantity(1.549e-2, '1/AA'),
      'C2': u.Quantity(5.433e-3, '1/AA'),
    'H2O+': u.Quantity(5.424e-3, '1/AA')
}

gamma_prime_XX_XX = {
      'OH': 0.98,
      'NH': 0.99,
      'CN': 0.99,
      'C3': 0.19,
     'CO+': 0.99,
      'C2': 0.66,
    'H2O+': 1.00
}

# apparent magnitude of the Sun, based on Appendix Table VI
# and Appendix D text near Eq. 44.
#
# For Msun at r', use Rsun, (V-R)sun = 0.370, and transformation from
# Smith et al. 2002: r' = V - 0.81 (V - R) + 0.13.  This is close to
# the -26.95 used by Ivezić et al. 2001.
Msun = {  
    'OH': -24.443,
    'NH': -25.046,
    'CN': -25.203,
    'C3': -25.737,
   'CO+': -25.896,
    'C2': -26.657,
  'H2O+': -27.483,
    'UC': -25.133,
    'BC': -26.234,  # -2.5 * log10(2.4685e19 / 1.276e17 / 6.210e-9)
    'GC': -26.741,
    'RC': -27.510,
     'V': -26.76,  # Bessell 1998
     'R': -27.13,  # Bessell 1998, (V-R)sun=0.370 (Colina et al. 1996)
'SDSS-R': -26.93,  # R to r' via Smith et al. 2002 + Bessel 1998
 'SDSSR': -26.93,
}

S0 = {  # Solar flux density at 1 AU, F_0 * 10**(-0.4 * Msun)
    'OH': u.Quantity( 632.2, 'W/(m2 um)'),
    'NH': u.Quantity( 878.4, 'W/(m2 um)'),
    'CN': u.Quantity(1036.8, 'W/(m2 um)'),
    'C3': u.Quantity(1608.8, 'W/(m2 um)'),
   'CO+': u.Quantity(1671.4, 'W/(m2 um)'),
    'C2': u.Quantity(1788.2, 'W/(m2 um)'),
  'H2O+': u.Quantity(1358.6, 'W/(m2 um)'),
    'UC': u.Quantity( 881.9, 'W/(m2 um)'),
    'BC': u.Quantity(1935.0, 'W/(m2 um)'),
    'GC': u.Quantity(1797.3, 'W/(m2 um)'),
    'RC': u.Quantity(1328.2, 'W/(m2 um)'),
     'V': u.Quantity(1836.4, 'W/(m2 um)'),
     'R': u.Quantity(1548.3, 'W/(m2 um)'),
'SDSS-R': u.Quantity(1663.5, 'W/(m2 um)'),
 'SDSSR': u.Quantity(1663.5, 'W/(m2 um)')
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

def Rm2S(w0, w1, Rm, Rm_unc=None):
    """Convert continuum color, `Rm`, to spectral slope, `S`.

    `Rm` defined by Farnham et al. 2000, `S` defined, e.g., by Jewitt
    and Meech 1986.

      S = (f2 - f1) * 2 / (f2 + f1) / Δλ

    Parameters
    ----------
    w0, w1 : string or Quantity
      The wavelengths used to determine `Rm`.
    Rm : Quantity
      The color in mangitudes per 0.1 um, and uncertainty.
    Rm_unc : Quantity or float, optional
      `Rm` uncertainty.

    Returns
    -------
    S : Quantity
      Spectral slope.
    S_unc : Quantity, optional
      Uncertainty on `S`.

    """

    assert Rm.decompose().unit == u.Unit("dex /m")

    if isinstance(w0, str):
        w0 = cw[w0.upper()]
    if isinstance(w1, str):
        w1 = cw[w1.upper()]

    dw = (w1 - w0).to(0.1 * u.um)
    r = 10**(0.4 * (Rm * dw).to(u.mag).value)
    S = 2 * (r - 1) / (r + 1) / dw * 100 * u.percent

    if Rm_unc is not None:
        S_unc = S * Rm_unc / Rm
        return S, S_unc

    return S

def continuum_color(w0, m0, m0_unc, w1, m1, m1_unc):
    """Comet continuum color.
    
    Color is with respect to the mean flux between the two filters,
    assuming the reflectance spectrum is linear with wavelength.
    Eq. 1 of A'Hearn et al. (1984, AJ 89, 579):

              R(lambda1) - R(lambda0)  2    α - 1  2
      slope = ----------------------- --- = ----- ---
              R(lambda1) + R(lambda0) Δλ    α + 1 Δλ

    where

      α = 10**(0.4 * (Δm - C_sun))

    and Δm and C_sun are color indices based on the two filters in
    question, and λ is measured in units of 0.1 μm.  This is
    equivalent to Eqs. 1 and 3 of Jewitt and Meech (1986, ApJ 310,
    937).


    Parameters
    ----------
    w0 : string
      The shorter wavelength filter name.
    m0 : float
      The apparant magnitude through the shorter wavelength filter.
    m0_unc : float
      The uncertainty in `m0`.
    w1, m1, m1_unc : various
      The same as above, but for the longer wavelength filter.

    Returns
    -------
    R, R_unc : Quantity
      The color in percent per 0.1 μm, and uncertainty.

    """

    from ..calib import solar_flux

    assert isinstance(w0, str)
    w0 = w0.upper()

    assert isinstance(w1, str)
    w1 = w1.upper()

    assert cw[w0] < cw[w1], 'w0 must be the shorter wavelength bandpass'

    dw = (cw[w1] - cw[w0]).to(0.1 * u.um)
    dm = m0 - m1 - (Msun[w0] - Msun[w1])
    alpha = 10**(0.4 * dm)
    
    R = (alpha - 1) / (alpha + 1) * 2 / dw * u.percent
    dm_unc = np.sqrt(m0_unc**2 + m1_unc**2)
    # R_unc = R * 2 * 0.4 * np.log(10) * alpha / (alpha + 1) / (alpha - 1)
    R_unc = R * 1.8421 * alpha / (alpha**2 - 1)

    return R, R_unc

def continuum_colors(m, unc=None):
    """Convert observed HB magntiudes into continnum colors.

    See `continuum_color` for notes.

    Parameters
    ----------
    m : dictionary
      Dictionaries of HB continuum filters and apparent magnitude
      pairs.  Any additional filters are ignored.
    unc : dictionary, optional
      Same as `m`, but for uncertainties.

    Returns
    -------
    color : dictionary
      Reflectance colors in units of magnitudes per 0.1 μm.
    color_unc : dictionary, optional
      Uncertainties on `color`, if `unc` was provided.

    """

    from collections import OrderedDict
    
    if unc is None:
        unc = dict.fromkeys(m, 0)
        return continuum_colors(m, unc=unc)[0]
    
    continuum_filters = [f for f in m if f in ['UC', 'BC', 'GC', 'RC']]
    continuum_filters.sort(key=cw.get)
    color = OrderedDict()
    color_unc = OrderedDict()
    for i in range(len(continuum_filters) - 1):
        left = continuum_filters[i]
        right = continuum_filters[i + 1]
        c = continuum_color(left, m[left], unc[left],
                            right, m[right], unc[right])
        color['-'.join((left, right))] = c[0]
        color_unc['-'.join((left, right))] = c[1]

    return color, color_unc

def estimate_continuum(base_filter, m, unc=None, color=None):
    """Continuum flux density, based on the measured continuum.

    All filters must be in present in `hb.cw`, `hb.F_0`, and
    `hb.Msun`.

    Note that this will return a continuum estimate for all filters in
    `m`.  Correlated errors are not considered and the uncertainties
    will be unecessarily high for some filters.  For example, the
    program can use BC and GC to compute a color, then extrapolate BC
    to GC using that color.  The result will be an uncerainty on GC's
    flux density that includes BC's uncertainty.  If you want a better
    uncertainty estimate for GCor any other filter closer to GC than
    BC in this case, use GC as the base filter.

    Parameters
    ----------
    base_filter : string
      Use this filter as the flux density basis for the continuum.
    m : dictionary
      Dictionaries of HB continuum filters and apparent magnitude
      pairs.  Any additional filters are ignored.
    unc : dictionary, optional
      Same as `m`, but for uncertainties.
    color : Quantity, optional
      Assume this spectral gradient centered at the basis filter.  If
      `None` and only one filter is provided, a spectral color of 0
      mag/0.1 μm will be used.

    Returns
    -------
    fluxd : dictionary
      The flux density estimated at each filter.
    fluxd_unc : dictionary, optional
      The uncertianties on `fluxd`, if `unc` was provided.

    """
    
    from collections import OrderedDict
    from operator import itemgetter
    import logging

    if unc is None:
        unc = dict.fromkeys(m, 0)
        return estimate_continuum(base_filter, m, unc=unc, color=color)[0]

    if color is None:
        if len(m) == 1:
            color = 0 * u.mag / u.Unit('0.1 um')
            color_unc = 0 * u.mag / u.Unit('0.1 um')
        else:
            # compute colors with respect to the basis filter
            color = {}
            color_unc = {}
            fxx = dict([(f, 10**(-0.4 * m[f])) for f in m])
            for f in m:
                if f == base_filter:
                    continue

                dw = (cw[f] - cw[base_filter]).to('0.1 um')
                dm = m[base_filter] - m[f] - (Msun[base_filter] - Msun[f])
                color[f] = dm / dw * u.mag
                color_unc[f] = np.sqrt(
                    unc[base_filter]**2 * fxx[base_filter]**2
                    + unc[f]**2 * fxx[f]**2) / fxx[base_filter] * color[f].unit
    else:
        assert color.unit.is_equivalent(u.mag / (0.1 * u.um))
        color = dict.fromkeys(m, color)
        color_unc = dict.fromkeys(m, 0 * color.unit)

    # Farnham et al. recommends UC-BC for all filters, except BC-GC
    # for C2 and GC-RC for H2O+.  Here, we are always going to use the
    # basis filter.  Select the other filter based on Δλ to each
    # filter in question.
    fluxd = OrderedDict()
    fluxd_unc = OrderedDict()
    for f, _cw in sorted(cw.items(), key=itemgetter(1)):
        mxx = m[base_filter]
        mxx_unc2 = unc[base_filter]**2
        if f != base_filter:
            mxx += Msun[f] - Msun[base_filter]

            # find Δλ to all filters used to compute the color
            dw = dict([(k, np.abs(cw[k] - cw[f])) for k in color.keys()])
            # find the nearest one
            k = sorted(dw.items(), key=itemgetter(1))[0][0]
            # use it
            dw = cw[base_filter] - cw[f]
            c = (dw * color[k]).to(u.mag).value
            mxx += c

            dc = (dw * color_unc[k]).to(u.mag).value
            mxx_unc2 = ((mxx_unc2 * 10**(-0.4 * mxx))**2
                        + (dc * 10**(-0.4 * c))**2)

        fluxd[f] = F_0[f] * 10**(-0.4 * mxx)
        fluxd_unc[f] = np.sqrt(mxx_unc2) * fluxd[f] / 1.0857

    return fluxd, fluxd_unc
  
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
      The amount of ozone.  0.15 is a good starting
      guess.
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

def flux_oh(oh, oh_unc, m, unc, zp, toz, z_true, E_bc, h, color=None):
    """Flux from OH.

    Appendix A and D of Farnham et al. 2000.

    Parameters
    ----------
    oh, oh_unc : float
      OH instrumental magnitude and uncertainty.
    m, unc : dictionary
      Apparent magnitudes for continuum filters.  'BC' must be one of
      the filters.
    zp : float
      OH magnitude zero point.
    toz : float
      Amount of ozone.
    z_true : Angle or Quantity
      True zenith angle.
    E_bc : float
      Extinction per airmass at BC. [mag/airmass]
    h : Quantity
      The observer's elevation.
    base_filter : string
      The name of the basis filter to use for extrapolating the
      continuum.  Must be present in `m` and `unc`.
    color : Quantity
      Use this color at BC, units of mag/0.1 μm.

    Returns
    -------
    E_tot, E_unc : float
      Total OH extinction and uncertainty.
    m_oh, m_oh_unc : float
      Calibrated OH apparent magnitude and uncertainty (including
      continuum).
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

    assert 'BC' in m and 'BC' in unc, 'BC apparent magnitude is required'

    E_0 = ext_total_oh(toz, z_true, 'OH', 'OH', E_bc, h)
    E_25 = ext_total_oh(toz, z_true, '25%', '25%', E_bc, h)
    E_100 = ext_total_oh(toz, z_true, 'G', 'G', E_bc, h)
    fluxd = estimate_continuum('BC', m, unc=unc, color=color)
    fc = fluxd['OH']
    fc_unc = fluxd['OH']

    f = 10**(-0.4 * (oh + zp - E_0)) * F_0['OH']
    f_unc = oh_unc * f / 1.0857  # first estimate

    frac = fc['OH'] / f  # fraction that is continuum
    frac_unc = np.sqrt(f_unc**2 * fc['OH']**2 + fc_unc['OH']**2 * f**2) / f**2
    #E_unc = frac_unc / frac * 1.0857
    #E_unc = np.sqrt(oh_unc**2 + ((fc_unc / fc) * 1.0857)**2)

    if frac <= 0.25:
        E_tot = 4 * ((0.25 - frac) * E_0 + frac * E_25)
        E_unc = 4 * (E_0 + E_25) * frac_unc
    else:
        #assert frac < 0.25, "Continuum = {:%}, more than 25% of observed OH band flux density.  Untested code.".format(frac)
        #print("Etot", 4 * ((0.25 - frac) * E_0 + frac * E_25), E_tot)
        # the following yields the same as above at the <0.001 mag level
        E_tot = ((1 - frac) * E_25 + (frac - 0.25) * E_100) / 0.75
        E_unc = np.abs(E_100 - E_25) * frac_unc / 0.75

    m = oh + zp - E_tot
    m_unc = np.sqrt(oh_unc**2 + E_unc**2)

    f = F_0['OH'] * 10**(-0.4 * m)
    f_unc = m_unc * f / 1.0857

    f_oh, f_oh_unc = flux_gas(f, f_unc, fc['OH'], fc_unc['OH'], 'OH')
    return E_tot, E_unc, m, m_unc, f_oh, f_oh_unc

def flux_gas(f, f_unc, fc, fc_unc, filt):
    """Gas emission band total flux.

    Does not consider contamination from other species.

    Table VI of Farnham et al. 2000.

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
      Estimated total gas flux, including contaimintion from any other
      species.

    """

    filt = filt.upper()

    if isinstance(f, u.Quantity):
        f_unc = u.Quantity(f_unc, f.unit)
        fc = u.Quantity(fc, f.unit)
        fc_unc = u.Quantity(fc_unc, f.unit)
    elif isinstance(fc, u.Quantity):
        fc_unc = u.Quantity(fc_unc, f.unit)
        f = u.Quantity(f, f.unit)
        f_unc = u.Quantity(f_unc, f.unit)

    fgas = (f - fc) / gamma_XX_XX[filt]
    fgas_unc = np.sqrt(f_unc**2 + fc_unc**2) / gamma_XX_XX[filt]

    return fgas.decompose([u.W, u.m]), fgas_unc.decompose([u.W, u.m])

# update module docstring
from ..util import autodoc
autodoc(globals())
del autodoc
