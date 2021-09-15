# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
outbursts --- Lightcurve and outburst analysis
==============================================

"""

__all__ = [
    'CometaryTrends'
]

from collections import namedtuple
import logging
import numpy as np
from scipy.cluster import hierarchy
import astropy.units as u
from astropy.time import Time
from astropy.stats import sigma_clip
from ..util import linefit

dmdtFit = namedtuple(
    'dmdtFit', ['m0', 'dmdt', 'm0_unc', 'dmdt_unc', 'rms', 'rchisq', 'k']
)
dHdtFit = namedtuple(
    'dHdtFit', ['H0', 'dHdt', 'H0_unc', 'dHdt_unc', 'rms', 'rchisq']
)
mrhFit = namedtuple(
    'mrhFit', ['m0', 'k', 'm0_unc', 'k_unc', 'rms', 'rchisq']
)
HrhFit = namedtuple(
    'HrhFit', ['m0', 'k', 'm0_unc', 'k_unc', 'rms', 'rchisq']
)

Color = namedtuple(
    'Color', ['t', 'clusters', 'c', 'c_unc', 'avg', 'avg_unc']
)
Color.__doc__ = 'Color estimate.'
Color.t.__doc__ = 'Average observation date for each color estimate. [astropy Time]'
Color.clusters.__doc__ = 'Observation clusters used to define color; 0 for unused.'
Color.c.__doc__ = 'Individual colors.  [ndarray]'
Color.c_unc.__doc__ = 'Uncertainty on c.  [ndarray]'
Color.avg.__doc__ = 'Weighted average color.'
Color.avg_unc.__doc__ = 'Uncertainty on avg.'


class CometaryTrends:
    """Define lightcurve trends designed for identifying cometary outbursts.


    Parameters
    ----------
    eph : sbpy Ephem
        Ephemeris of the target.  Field requirements depend on the trend
        fitting methods to be used.  Generally provide date, rh, delta, phase.

    m, m_unc : Quantity
        Photometry and uncertainty in magnitudes.

    filt : array, optional
        Filters for each ``m``.

    fit_mask : array, optional
        ``True`` for elements to ignore when fitting (e.g., outbursts).

    logger : Logger, optional
        Use this logger for messaging.

    **kwargs
        Any ``CometaryTrends`` property.


    Properties
    ----------
    m_original : Quantity
        Unmodified (input) photometry.

    m : Quantity
        Apparent magnitude, possibly limited to one filter (see ``fit_filter``)
        or filter transformed (see ``color_transform``).

    colors : dict of Quantity
        Use these colors when transforming between filters.  Key by filter
        tuple in wavelength order, e.g., to set g-r use:

            `{('g', 'r'): 0.5 * u.mag}`

        ``colors`` is also set when ``self.color`` is used.

    fit_filter : str or None
        Set to a filter in ``self.filt`` to limit fitting to this filter.

    color_transform : bool
        Set to ``True`` to transform observations to that specified in
        ``fit_filter`` via ``colors``.

    """

    def __init__(self, eph, m, m_unc, filt=None, fit_mask=None, logger=None,
                 **kwargs):
        # store parameters and properties
        self.eph = eph
        self.m = m
        self.m_unc = m_unc
        self.filt = np.array(filt)
        self.fit_mask = (
            np.zeros(len(m), bool) if fit_mask is None
            else np.array(fit_mask)
        )
        self.colors = kwargs.get('colors', {})
        self.fit_filter = kwargs.get('fit_filter')
        self.color_transform = kwargs.get('color_transform', False)

        if logger is None:
            self.logger = logging.getLogger('CometaryTrends')
        else:
            self.logger = logger

        # parameter check
        if not all((isinstance(m, u.Quantity), isinstance(m_unc, u.Quantity))):
            raise ValueError(
                'm, m_unc must be Quantity in units of magnitude.')

        n = [len(x) for x in (eph, m, m_unc, self.fit_mask)]
        if filt is not None:
            n += [len(filt)]
        if len(np.unique(n)) != 1:
            raise ValueError('all arrays must have the same length')

    @property
    def m_original(self):
        return self._m

    @property
    def m(self):
        """Apparent magnitude.

        Possibly limited to one filter (see ``fit_filter``) or filter
        transformed (see ``color_transform``).

        """

        m = np.ma.MaskedArray(self._m.copy(),
                              mask=np.zeros(len(self._m), bool))
        if (self.filt is not None) and (self.fit_filter is not None):
            for i in range(len(m)):
                if self.filt[i] != self.fit_filter:
                    if self.color_transform:
                        # try to color transform
                        color = (self.filt[i], self.fit_filter)
                        if color in self.colors:
                            m[i] -= self.colors[color]
                        elif color[::-1] in self.colors:
                            m[i] += self.colors[color[::-1]]
                        else:
                            # not possible
                            m.mask[i] = True
                    else:
                        # not color transforming this filter
                        m.mask[i] = True
        return m

    @m.setter
    def m(self, _m):
        self._m = _m

    @property
    def fit_m(self):
        """Magnitude array masked for fitting."""
        m = self.m
        m.mask += self.fit_mask
        return m

    @property
    def fit_filter(self):
        """Filter to fit.

        Set to ``None`` to fit all data(without color transformations).

        """

        return self._fit_filter

    @fit_filter.setter
    def fit_filter(self, filt):
        if not isinstance(filt, (str, type(None))):
            raise ValueError('fit filter must be a string or ``None``')

        self._fit_filter = filt

    @property
    def color_transform(self):
        """Color transformation flag.

        If fitting only one filter, set to ``True`` to allow
        color transformations via ``self.color``.

        """

        return self._color_transform

    @color_transform.setter
    def color_transform(self, flag):
        self._color_transform = bool(flag)

    def color(self, blue, red, max_dt=16 / 24, max_unc=0.25 * u.mag):
        """Estimate the color, blue - red, using weighted averages.

        ``eph`` requires ``'date'``.

        Masked data is excluded.

        Data is not nucleus subtracted.


        Parameters
        ----------
        blue: string
            The name of the bluer filter.

        red: string
            The name of the redder filter.

        max_dt: float, optional
            Maximum time difference to consider when clustering observations.

        max_unc: Quantity, optional
            Ignore results with uncertainty > ``max_unc``.


        Returns
        -------
        color: Color
            The color results or ``None`` if it cannot be calculated.

        """

        if len(self.filt) < 2:
            self.logger.info('Not enough filters.')
            return None

        b = self.filt == blue
        r = self.filt == red

        clusters = hierarchy.fclusterdata(
            self.eph['date'].mjd[:, np.newaxis],
            max_dt, criterion='distance'
        )
        self.logger.info(f'{clusters.max()} clusters found.')

        mjd = []
        bmr = []
        bmr_unc = []
        for cluster in np.unique(clusters):
            i = (clusters == cluster) * ~self.fit_mask

            # require both filters in this cluster
            if (not np.any(b[i])) or (not np.any(r[i])):
                clusters[i] = 0
                continue

            # estimate weighted averages and compute color
            wb, sw = np.average(self.m_original[b * i],
                                weights=self.m_unc[b * i]**-2,
                                returned=True)
            wb_unc = sw**-0.5

            wr, sw = np.average(self.m_original[r * i],
                                weights=self.m_unc[r * i]**-2,
                                returned=True)
            wr_unc = sw**-0.5

            if np.hypot(wb_unc, wr_unc) > max_unc:
                continue

            mjd.append(self.eph['date'].mjd[i].mean())
            bmr.append(wb - wr)
            bmr_unc.append(np.hypot(wb_unc, wr_unc))

        if len(bmr) == 0:
            self.logger.info('No colors measured.')
            return None

        bmr = u.Quantity(bmr)
        bmr_unc = u.Quantity(bmr_unc)
        avg, sw = np.average(bmr, weights=bmr_unc**-2, returned=True)
        avg_unc = sw**-0.5

        self.colors[(blue, red)] = avg

        return Color(Time(mjd, format='mjd'), clusters, bmr, bmr_unc, avg, avg_unc)

    @staticmethod
    def linear_add(a, b):
        """The sum a+b computed in linear space."""
        return -np.log(np.exp(-a.value) + np.exp(-b.to_value(a.unit))) * a.unit

    @staticmethod
    def linear_subtract(a, b):
        """The difference a-b computed in linear space."""
        return -np.log(np.exp(-a.value) - np.exp(-b.to_value(a.unit))) * a.unit

    def H(self, fixed_angular_size=False, Phi=None, nucleus=None):
        """Absolute magnitude.


        Parameters
        ----------
        fixed_angular_size: bool
            ``True`` if the photometric aperture is measured with a fixed
            angular size.  If so, the target-observer distance(Δ) correction
            will be Δ**-1.

        Phi: function, optional
            Phase function.

        nucleus : Quantity
            Subtract this nucleus before scaling.

        """

        m = self.m.copy()
        unit = m.data.unit
        if nucleus is not None:
            m = np.ma.MaskedArray(self.linear_subtract(m.data, nucleus),
                                  mask=m.mask)

        d = 2.5 if fixed_angular_size else 5
        H = (m - 5 * np.log10(self.eph['rh'].to_value('au')) * unit
             - d * np.log10(self.eph['delta'].to_value('au')) * unit)
        if Phi is not None:
            H += 2.5 * np.log10(Phi(self.eph['phase'])) * unit

        return H

    def ostat(self, k=4, dt=14, sigma=2, **kwargs):
        """Compute the outburst statistic for each photometry point.

        ostat is calculated for each masked point, but the masked points are
        not included in the photometric baseline calculation.


        Parameters
        ----------
        k : float, optional
            Heliocentric distance slope on apparent magnitude for the baseline
            estimate.

        dt : float, optional
            Number of days of history to use for the baseline estimate.

        sigma : float, optional
            Number of sigmas to clip the data.

        **kwargs
            Additional keyword arguments are passed to ``H()``.


        Returns
        -------
        o : array
            The outburst statistic.

        """

        Hy = (
            self.H(**kwargs)
            - 2.5 * (k - 2) * np.log10(self.eph['rh'].to_value('au')) * u.mag
        )

        o = np.ma.zeros(len(Hy))
        for i in range(len(Hy)):
            j = (
                (self.eph['date'] < self.eph['date'][i])
                * (self.eph['date'] > (self.eph['date'][i] - dt * u.day))
            )
            if j.sum() < 1:
                o[i] = np.ma.masked
                continue

            # reject outliers, calculate weighted mean
            good = j * ~Hy.mask * np.isfinite(Hy.data)
            if np.sum(good) > 2:
                m = sigma_clip(Hy[good].data, sigma=sigma)
            else:
                m = Hy[good]
            m -= Hy[i]  # normalize to data point being tested
            m_unc = self.m_unc[good]

            baseline, sw = np.ma.average(m, weights=m_unc**-2,
                                         returned=True)
            baseline_unc = sw**-0.5
            unc = max(np.sqrt(baseline_unc**2 + self.m_unc[i]**2).value, 0.1)
            o[i] = np.round(baseline.value / unc, 1)

        return o

    def dmdt(self, nucleus=None, k=1):
        """Fit apparent magnitude constant slope versus time.

        ``eph`` requires ``'date'``.


        Parameters
        ----------
        nucleus : Quantity
            Subtract this nucleus before fitting, assumed to be in the same
            filter as ``self.m``.

        k : float, optional
            Scale time by ``t^k``.


        Returns
        -------
        dt: np.array

        trend: np.array
            Including the nucleus.

        fit_mask: np.array
            Data points used in the fit.

        fit: DmDt
            Fit results.

        """

        dt = self.eph['date'].mjd * u.day
        dt -= dt.min()

        m = self.fit_m
        unit = m.data.unit
        mask = m.mask
        if nucleus is not None:
            m = np.ma.MaskedArray(
                self.linear_subtract(m.data, nucleus),
                mask=m.mask
            )
            # subtraction may introduce nans
            mask += ~np.isfinite(m)

        r = linefit(dt.value[~mask]**k, m.data.value[~mask],
                    self.m_unc.value[~mask], (0.05, 15))
        trend = (r[0][1] + r[0][0] * dt.value**k) * unit

        # restore nucleus?
        if nucleus is not None:
            trend = self.linear_add(trend, nucleus)

        residuals = self.m - trend

        fit = dmdtFit(r[0][1] * unit, r[0][0] * unit / u.day**k,
                      r[1][1] * unit, r[1][0] * unit / u.day**k,
                      np.std(residuals[~mask].data),
                      np.sum((residuals[~mask].data / self.m_unc[~mask])**2)
                      / np.sum(~mask),
                      k)

        return dt, trend, ~mask, fit

    # def dHdt(self, fixed_angular_size, filt=None, color_transform=True, Phi=phase_HalleyMarcus):
    #     """Fit absolute magnitude constant slope versus time.

    #     ``eph`` requires date, rh, delta, phase.

    #     Parameters
    #     ----------
    #     fixed_angular_size: bool
    #         Aperture is fixed in angular size.  See ``self.H()``.

    #     filt: str, optional
    #         Fit only this filter.

    #     color_transform: bool, optional
    #         If fitting only one filter, set to ``True`` to allow
    #         color transformations via ``self.color``.

    #     Phi: function, optional
    #         Use this phase function.

    #     Returns
    #     -------
    #     dt: np.array

    #     trend: np.array

    #     fit_mask: np.array
    #         Data points used in the fit.

    #     fit: DmDt

    #     """

    #     dt = self.eph['date'].mjd * u.day
    #     dt -= dt.min()

    #     if filt is None:
    #         H = self.H(fixed_angular_size, Phi=Phi)
    #     elif color_transform:
    #         H = self.H(fixed_angular_size, filt=filt,
    #                    color_transform=color_transform, Phi=Phi)
    #     else:
    #         H = self.H(fixed_angular_size, Phi=Phi) * (self.filt == filt)
    #         H[self.filt != filt] = np.nan

    #     i = ~self.fit_mask * np.isfinite(H)

    #     r = linefit(dt[i].value, H[i].value, self.m_unc[i].value,
    #                 (0.05, 15))

    #     dm = self.H(fixed_angular_size, Phi=Phi) - self.m
    #     trend = (r[0][1] + r[0][0] * dt.value) * H.unit - dm
    #     residuals = self.m - trend

    #     fit = dHdtFit(r[0][1] * H.unit, r[0][0] * H.unit / u.day,
    #                   r[1][1] * H.unit, r[1][0] * H.unit / u.day,
    #                   np.std(residuals[i]),
    #                   np.sum((residuals[i] / self.m_unc[i])**2) / np.sum(i))

    #     return dt, trend, i, fit

    # def mrh(self, fixed_angular_size, filt=None, color_transform=True,
    #         Phi=phase_HalleyMarcus):
    #     """Fit magnitude as a function of rh.

    #     ``eph`` requires rh, delta, phase.

    #     m = M - k log10(rh) - d log10(Delta) + 2.5 log10(Phi(phase))

    #     d = 2.5 for fixed_angular_size == True, 5 otherwise.

    #     Parameters
    #     ----------
    #     fixed_angular_size: bool
    #         Aperture is fixed in angular size.

    #     filt: str, optional
    #         Fit only this filter.

    #     color_transformation: bool, optional
    #         If fitting only one filter, set to ``True`` to allow
    #         color transformations via ``self.color``.

    #     Phi: function, optional
    #         Use this phase function.

    #     Returns
    #     -------
    #     trend: np.array

    #     fit_mask: np.array
    #         Data points used in the fit.

    #     fit: mrhFit

    #     """

    #     m = self.coma(filt)
    #     if filt is not None and not color_transform:
    #         m[self.filt != filt] = np.nan

    #     if fixed_angular_size:
    #         d = 2.5
    #     else:
    #         d = 5

    #     dm = (-d * np.log10(self.eph['delta'].to_value('au'))
    #           + 2.5 * np.log10(Phi(self.eph['phase']))) * u.mag

    #     i = ~self.fit_mask * np.isfinite(m)

    #     r = linefit(self.eph['rh'][i].value, (m - dm)[i].value,
    #                 self.m_unc[i].value, (0.05, 15))

    #     trend = (r[0][1] + r[0][0] * self.eph['rh'].value) * m.unit + dm
    #     residuals = m - trend

    #     # restore nucleus?
    #     if self.nucleus is not None:
    #         trend = -np.log(np.exp(-trend.value) +
    #                         np.exp(-self.nucleus.value)) * u.mag

    #     fit = mrhFit(r[0][1] * m.unit, r[0][0] * m.unit / u.day,
    #                  r[1][1] * m.unit, r[1][0] * m.unit / u.day,
    #                  np.std(residuals[i]),
    #                  np.sum((residuals[i] / self.m_unc[i])**2) / np.sum(i))

    #     return trend, i, fit
