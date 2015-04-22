# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
photometry --- Tools for working with photometry.
=================================================

.. autosummary::
   :toctree: generated/

   cal_color_airmass

"""

import numpy as np

__all__ = [
    'cal_color_airmass',
]


def cal_color_airmass(m, munc, M, C, am, guess=(25., -0.1, -0.01),
                      covar=False):
    """Determine calibraton coefficients, based on airmass and color.

    Parameters
    ----------
    m : array
      Instrumental (uncalibrated) magnitude.
    munc : array
      Uncertainties on m.
    M : array
      Calibrated magnitude.
    C : array
      Calibrated color index, e.g., V - R.
    am : array
      Airmass.
    guess : array, optional
      An intial guess for the fitting algorithm.
    covar : bool, optional
      Set to `True` to return the covariance matrix.

    Results
    -------
    zKA : ndarray
      The photometric zero point, airmass correction slope, and color
      correction slope. [mag, mag/airmass, mag/color index]
    unc or cov : ndarray
      Uncertainties on each parameter, based on least-squares fitting,
      or the covariance matrix, if `covar` is `True`.

    """

    from scipy.optimize import leastsq
    
    def chi(A, m, munc, M, C, am):
        model = M + A[0] + A[1] * am + A[2] * C
        chi = (np.array(m) - model) / np.array(munc)
        return chi

    output = leastsq(chi, guess, args=(m, munc, M, C, am), full_output=True,
                     epsfcn=1e-3)
    fit = output[0]
    cov = output[1]
    err = np.sqrt(np.diag(cov))

    if covar:
        return fit, cov
    else:
        return fit, err
