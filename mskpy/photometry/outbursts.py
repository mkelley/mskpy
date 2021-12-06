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
from scipy.optimize import leastsq
import astropy.units as u
from astropy.time import Time
from astropy.stats import sigma_clip
from sbpy.activity import Afrho
from ..util import linefit

dmdtFit = namedtuple(
    'dmdtFit', ['m0', 'dmdt', 'm0_unc', 'dmdt_unc', 'rms', 'rchisq']
)
ExpFit = namedtuple(
    'ExpFit', ['dm', 'tau', 'dm_unc', 'tau_unc', 'rms', 'rchisq']
)
AfrhoRhFit = namedtuple(
    'AfrhoRhFit', ['afrho1', 'k', 'afrho1_unc', 'k_unc', 'rms', 'rchisq']
)

Color = namedtuple(
    'Color', ['t', 'clusters', 'm_filter', 'm',
              'm_unc', 'c', 'c_unc', 'avg', 'avg_unc']
)
Color.__doc__ = 'Color estimate.'
Color.t.__doc__ = 'Average observation date for each color estimate. [astropy Time]'
Color.clusters.__doc__ = 'Observation clusters used to define color; 0 for unused.'
Color.m_filter.__doc__ = 'Filter for m.'
Color.m.__doc__ = 'Apparent mag for each date in given filter. [mag]'
Color.m_unc.__doc__ = 'Uncertainty on m.  [mag]'
Color.c.__doc__ = 'Individual colors.  [mag]'
Color.c_unc.__doc__ = 'Uncertainty on c.  [mag]'
Color.avg.__doc__ = 'Weighted average color.  [mag]'
Color.avg_unc.__doc__ = 'Uncertainty on avg.  [mag]'


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

    aper : Quantity, optional
        Photometric aperture radii.

    fit_mask : array, optional
        ``True`` for elements to ignore when fitting (e.g., outbursts).

    logger : Logger, optional
        Use this logger for messaging.

    **kwargs
        Any ``CometaryTrends`` property.


    Attributes
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

    def __init__(self, eph, m, m_unc, filt=None, aper=None, fit_mask=None,
                 logger=None, **kwargs):
        # store parameters and properties
        self.eph = eph
        self.m = m
        self.m_unc = m_unc
        self.filt = np.array(filt)
        self.aper = aper
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
                            m[i] = ((m[i].value - self.colors[color].value)
                                    * m.data.unit)
                        elif color[::-1] in self.colors:
                            m[i] = ((m[i].value + self.colors[color[::-1]].value)
                                    * m.data.unit)
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

    def color(self, blue, red, max_dt=16 / 24, max_unc=0.25 * u.mag,
              m_filter=None):
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

        m_filter : string, optional
            Report mean apparent magnitude in this filter.  Default is the
            redder filter.


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
        if m_filter is None:
            m_filter = red
        elif m_filter not in [blue, red]:
            raise ValueError("m_filter must be one of blue or red")

        clusters = hierarchy.fclusterdata(
            self.eph['date'].mjd[:, np.newaxis],
            max_dt, criterion='distance'
        )
        self.logger.info(f'{clusters.max()} clusters found.')

        mjd = []
        m_mean = []
        m_mean_unc = []
        bmr = []
        bmr_unc = []
        for cluster in np.unique(clusters):
            i = (clusters == cluster) * ~self.fit_mask

            # require both filters in this cluster
            if (not np.any(b[i])) or (not np.any(r[i])):
                clusters[i] = 0
                continue

            # estimate weighted averages and compute color
            wb, sw = np.average(self.m_original[b * i].value,
                                weights=self.m_unc.value[b * i]**-2,
                                returned=True)
            wb_unc = sw**-0.5

            wr, sw = np.average(self.m_original[r * i].value,
                                weights=self.m_unc.value[r * i]**-2,
                                returned=True)
            wr_unc = sw**-0.5

            if np.hypot(wb_unc, wr_unc) > max_unc.value:
                continue

            mjd.append(self.eph['date'].mjd[i].mean())
            if m_filter == 'blue':
                m_mean.append(wb)
                m_mean_unc.append(wb_unc)
            else:
                m_mean.append(wr)
                m_mean_unc.append(wr_unc)

            bmr.append(wb - wr)
            bmr_unc.append(np.hypot(wb_unc, wr_unc))

        if len(bmr) == 0:
            self.logger.info('No colors measured.')
            return None

        unit = self.m_original.unit
        m_mean = m_mean * unit
        m_mean_unc = m_mean_unc * unit
        bmr = bmr * unit
        bmr_unc = bmr_unc * unit
        avg, sw = np.average(bmr.value, weights=bmr_unc.value**-2,
                             returned=True)
        avg_unc = sw**-0.5 * unit

        self.colors[(blue, red)] = avg * unit

        return Color(Time(mjd, format='mjd'), clusters, m_filter,
                     m_mean, m_mean_unc, bmr, bmr_unc, avg, avg_unc)

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

    def afrho(self, Phi=None):
        """Compute the coma dust quantity Afρ.

        Requires ``'rh'``, ``'phase'`` in ``eph``, ``delta`` additionally
        required if ``aper`` is in angular units, ``filt`` (must be understood
        by sbpy's calibration system), and ``aper``.

        Uses ``fit_filter`` and ``color_transform``, if defined.


        Parameters
        ----------
        Phi : function, optional
            Use this phase function and return A(0°)fρ.


        Returns
        -------
        afrho : masked array
            Afρ in units of cm.

        """

        if self.fit_filter:
            filt = self.fit_filter
        else:
            filt = self.filt

        afrho = Afrho.from_fluxd(filt, self.m * self.m_original.unit,
                                 self.aper, self.eph, Phi=Phi,
                                 phasecor=Phi is not None)
        return np.ma.MaskedArray(afrho.to_value('cm'), mask=self.m.mask)

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

    def _fit_setup(self, nucleus=None, absolute=False, **kwargs):
        dt = self.eph['date'].mjd * u.day
        dt -= dt.min()

        if absolute:
            m = self.H(nucleus=nucleus, **kwargs)
            m.mask = self.fit_m.mask
        else:
            m = self.fit_m
            if nucleus is not None:
                m = np.ma.MaskedArray(
                    self.linear_subtract(m.data, nucleus),
                    mask=m.mask
                )
                # subtraction may introduce nans
                m.mask += ~np.isfinite(m)

        return dt, m

    def dmdt(self, nucleus=None, guess=None, k=1, absolute=False, **kwargs):
        """Fit magnitude versus time as a function of ``t**k``.

        ``eph`` requires ``'date'``.

        ``absolute`` requires ``'rh'``, ``'delta'``, and ``'phase'`` in
        ``eph``.


        Parameters
        ----------
        nucleus : Quantity
            Subtract this nucleus before fitting, assumed to be in the same
            filter as ``self.m``.

        guess : tuple of floats
            Initial fit guess: (m0, slope).

        k : float, optional
            Scale time by ``t^k``.

        absolute : boo, optional
            Fix absolute magnitude via ``self.H()``.

        **kwargs
            Additional keyword arguments pass to ``self.H()``.


        Returns
        -------
        dt: np.array

        trend: np.array
            Including the nucleus.

        fit_mask: np.array
            Data points used in the fit.

        fit: dmdtFit
            Fit results.

        """

        dt, m = self._fit_setup(nucleus=nucleus, absolute=absolute, **kwargs)
        unit = m.data.unit
        mask = m.mask

        guess = (0.05, 15) if guess is None else guess
        r = linefit(dt.value[~mask]**k, m.data.value[~mask],
                    self.m_unc.value[~mask], guess)
        trend = (r[0][1] + r[0][0] * dt.value**k) * unit
        fit_unc = r[1] if r[1] is not None else (0, 0)

        # restore nucleus?
        if nucleus is not None:
            trend = self.linear_add(trend, nucleus)

        residuals = m - trend

        fit = dmdtFit(r[0][1] * unit, r[0][0] * unit / u.day**k,
                      fit_unc[1] * unit, fit_unc[0] * unit / u.day**k,
                      np.std(residuals[~mask].data),
                      np.sum((residuals[~mask].data / self.m_unc[~mask])**2)
                      / np.sum(~mask))

        return dt, trend, ~mask, fit

    def exp(self, baseline, absolute=False, **kwargs):
        """Fit magnitude versus time as a function of ``e**(k*t)``.

        ``eph`` requires ``'date'``.

        ``absolute`` requires ``'rh'``, ``'delta'``, and ``'phase'`` in
        ``eph``.


        Parameters
        ----------
        baseline : Quantity
            Fit the exponential with respect to this baseline trend (may
            include the nucleus).  Must be absolute magnitude if ``absolute``
            is true.

        absolute : boo, optional
            Fix absolute magnitude via ``self.H()``.

        **kwargs
            Additional keyword arguments pass to ``self.H()``.


        Returns
        -------
        dt: np.array

        trend: np.array
            Including the nucleus.

        fit_mask: np.array
            Data points used in the fit.

        fit: ExpFit
            Fit results.

        """

        dt, m = self._fit_setup(absolute=absolute, **kwargs)
        dm = m - baseline
        unit = m.data.unit
        mask = m.mask
        print(m)

        def model(dt, peak, tau):
            lc = peak * np.exp(-dt / tau)
            lc[dt < 0] = 0
            return lc

        def chi(p, dt, dm, m_unc):
            m = model(dt, *p)
            return (dm - m) / m_unc

        args = (dt.value[~mask], dm.data.value[~mask], self.m_unc.value[~mask])
        guess = (dm.compressed().min().value, 10)
        r = leastsq(chi, guess, args=args, full_output=True)
        fit_unc = np.sqrt(np.diag(r[1]))
        trend = model(dt.value, *r[0]) * unit

        # restore baseline
        trend = trend + baseline

        residuals = m - trend

        fit = ExpFit(r[0][0] * unit, r[0][1] * u.day,
                     fit_unc[0] * unit, fit_unc[1] * u.day,
                     np.std(residuals[~mask].data),
                     np.sum((residuals[~mask].data / self.m_unc[~mask])**2)
                     / np.sum(~mask))

        return dt, trend, ~mask, fit

    def afrho_rh(self, nucleus=None, guess=None, **kwargs):
        """Fit Afρ as a function of ``rh**k``.


        Parameters
        ----------
        guess : tuple of floats
            Initial fit guess: (Afrho1, k) = (Afrho at 1 au, rh power-law
            slope).

        **kwargs
            Additional keyword arguments pass to ``self.Afrho()``.


        Returns
        -------
        trend: np.array

        mtrend: np.array

        fit_mask: np.array
            Data points used in the fit.

        fit: AfrhoRhFit
            Fit results.

        """

        guess = (-4, 3) if guess is None else guess
        afrho = self.afrho(**kwargs)
        afrho.mask += self.fit_mask
        log10afrho = np.log10(afrho)
        log10afrho_unc = (self.m_unc.value / 1.0857 / np.log(10))
        log10rh = np.log10(self.eph['rh'].to_value('au'))
        mask = afrho.mask
        r = linefit(log10rh[~mask], log10afrho[~mask], log10afrho_unc[~mask],
                    guess)

        trend = 10**(r[0][1] + r[0][0] * log10rh)
        fit_unc = r[1] if r[1] is not None else (0, 0)

        if self.fit_filter:
            filt = self.fit_filter
        else:
            filt = self.filt
        mtrend = Afrho(trend * u.cm).to_fluxd(
            filt, self.aper, self.eph, unit=u.ABmag, Phi=kwargs.get('Phi'),
            phasecor='Phi' in kwargs)

        residuals = np.ma.MaskedArray(
            self.m.data.value - mtrend.value,
            mask=mask)

        fit = AfrhoRhFit(10**r[0][1] * u.cm, r[0][0],
                         fit_unc[1] * u.cm, fit_unc[0],
                         np.std(residuals[~mask].data),
                         np.sum((residuals[~mask].data
                                 / self.m_unc[~mask].value)**2)
                         / np.sum(~mask))

        return trend, mtrend, ~mask, fit

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
