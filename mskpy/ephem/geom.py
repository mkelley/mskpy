# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
geom --- Solar System geometry.
===============================

Class
-----
Geom

"""

__all__ = ['Geom']

from datetime import datetime

import numpy as np
from astropy.time import Time
import astropy.units as u

class Geom(object):
    """Observing geometry parameters for Solar System objects.

    Coordinates are all in the heliocentric ecliptic J2000 frame.

    Parameters
    ----------
    ro : Quantity
      The observer's coordinates, shape must be (3,) or (N, 3).
    rt : Quantity
      The target's coordinates.
    vo : Quantity, optional
      The observer's velocity, shape must be the same as `ro`.
    vt : Quantity, optional
      The target's velocity, shape must be the same as `rt`.
    date : astropy Time, optional
      The date of the observation.

    Attributes
    ----------
    rh : Quantity
      Target's heliocentric distance.
    delta : Quantity
      Observer-target distance.
    phase : Quantity
      Phase angle (Sun-target-observer).
    signedphase : Quantity
      Phase angle, <0 for pre-opposition, >0 for post-opposition.
    obsrh : Quantity
      The observer's heliocentric distance.
    so : Quantity
      The observer's speed.
    st : Quantity
      The target's speed.
    lambet : tuple of Quantity
      Ecliptic longitude and latitude.
    lam : Quantity
      Ecliptic longitude.
    bet : Quantity
      Ecliptic latitude.
    radec : tuple of Quantity
      Right ascension and declination.
    ra : Quantity
      Right ascension.
    dec : Quantity
      Declination.
    sangle : Quantity
      Projected Sun angle.
    vangle : Quantity
      Projected velocity angle.
    selong : Quantity
      Solar elongation.
    lelong : Quantity
      Lunar elongation.

    Methods
    -------
    argmax : Index of the maximum of each parameter as a `dict`.
    argmin : Index of the minimum of each parameter as a `dict`.
    max : Maximum of each parameter as a `dict`.
    mean : Average geometry as a `dict`.
    min : Minimum of each parameter as a `dict`.
    reduce : Apply a function to each vector.
    summary : Return a pretty summary of the geometry.

    """

    _ro = None
    _rt = None
    _vo = None
    _vt = None
    _keys = ['ro', 'rt', 'vo', 'vt', 'date', 'rh', 'delta', 'phase',
             'signedphase', 'obsrh', 'so', 'st', 'lambet', 'lam', 'bet',
             'radec', 'ra', 'dec', 'sangle', 'vangle', 'selong', 'lelong']

    def __init__(self, ro, rt, vo=None, vt=None, date=None):
        from astropy.units import Quantity
        from .. import util

        self._ro = ro.to(u.km).value
        self._rt = rt.to(u.km).value

        if (self._ro.shape[-1] != 3) or (self._ro.ndim > 2):
            raise ValueError("Incorrect shape for ro.  Must be (3,) or (N, 3).")

        if self._rt.shape != self._ro.shape:
            raise ValueError("The shapes of ro and ro must agree.")

        if self._ro.ndim == 1:
            self._len = 1
        else:
            self._len = self._ro.shape[0]

        if vo is not None:
            self._vo = vo.to(u.km / u.s).value
            if self._vo.shape != self._ro.shape:
                raise ValueError("The shape of vo and ro must agree.")

        if vt is not None:
            self._vt = vt.to(u.km / u.s).value
            if self._vt.shape != self._rt.shape:
                raise ValueError("The shape of vt and rt must agree.")

        if date is not None:
            self.date = date
            N = util.date_len(self.date)
            if N == 0:
                N += 1
            if self._len != N:
                raise ValueError("Given ro, the length of date "
                                 " must be {}.".format(self._len))

    def __len__(self):
        if self._ro.ndim == 1:
            self._len = 1
        else:
            self._len = self._ro.shape[0]
        return self._len

    def __getitem__(self, key):
        from .. import util
        # are we slicing?
        if isinstance(key, (int, slice, list, np.ndarray)):
            if self._ro.ndim == 1:
                raise IndexError("Attempting to subscript a 1D Geom object.")

            ro = self.ro
            vo = self.vo
            if self._ro.ndim == 2:
                ro = ro[key]
                if vo is not None:
                    vo = vo[key]

            rt = self.rt
            vt = self.vt
            if self._rt.ndim == 2:
                rt = rt[key]
                if vt is not None:
                    vt = vt[key]

            if self.date is None:
                date = None
            elif util.date_len(self.date) <= 1:
                date = self.date
            else:
                date = self.date[key]

            return Geom(ro, rt, vo=vo, vt=vt, date=date)
        else:
            return self.__getattribute__(key)

    def __str__(self):
        keys = ['date', 'rh', 'delta', 'phase', 'obsrh', 'so', 'st',
                'lam', 'bet', 'ra', 'dec', 'sangle', 'vangle', 'selong',
                'lelong']
        s = ""
        for k in keys:
            s += "{:>6s} : ".format(k)
            if len(self) == 1:
                s += "{:}\n".format(self[k])
            else:
                s += "{:}\n".format(self[k])
        return s[:-1]

    @property
    def _rot(self):
        return self._rt - self._ro

    @property
    def ro(self):
        return self._ro * u.km

    @property
    def rt(self):
        return self._rt * u.km

    @property
    def vo(self):
        if self._vo is None:
            return None
        else:
            return self._vo * u.km / u.s

    @property
    def vt(self):
        if self._vt is None:
            return None
        else:
            return self._vt * u.km / u.s

    @property
    def rh(self):
        return np.sqrt(np.sum(self._rt**2, -1)) / 1.495978707e8 * u.au

    @property
    def delta(self):
        return np.sqrt(np.sum(self._rot**2, -1)) / 1.495978707e8 * u.au

    @property
    def phase(self):
        phase = np.arccos((self.rh**2 + self.delta**2 - self.obsrh**2) /
                          2.0 / self.rh / self.delta)
        return np.degrees(phase)  # quantity magic will put this in deg.

    @property
    def signedphase(self):
        """Signed phase angle, based on pre- or post-opposition.

        For ho, the angular momentum of the observer's orbit (ro X
        vo), the sign is + when (rt X rot) * h > 0.

        """
        if self._vt is None:
            return None
        dot = np.sum((np.cross(self._rt, self._rot)
                      * np.cross(self._ro, self._vo)), -1)
        sign = np.sign(dot)
        phase = self.phase
        return (sign * self.phase.value) * u.deg

    @property
    def obsrh(self):
        """The observer's heliocentric distance."""
        return np.sqrt(np.sum(self._ro**2, -1)) / 1.495978707e8 * u.au

    @property
    def so(self):
        """The observer's speed."""
        if self._vo is None:
            return None
        return np.sqrt(np.sum(self._vo**2, -1)) * u.km / u.s

    @property
    def st(self):
        """The target's speed."""
        if self._vt is None:
            return None
        return np.sqrt(np.sum(self._vt**2, -1)) * u.km / u.s

    @property
    def lambet(self):
        """Ecliptic longitude and latitude."""
        lam = np.arctan2(self._rot.T[1], self._rot.T[0])
        bet = np.arctan2(self._rot.T[2],
                         np.sqrt(self._rot.T[0]**2 + self._rot.T[1]**2))
        return np.degrees(lam) * u.deg, np.degrees(bet) * u.deg

    @property
    def lam(self):
        """Ecliptic longitude."""
        return self.lambet[0]

    @property
    def bet(self):
        """Ecliptic latitude."""
        return self.lambet[1]

    @property
    def radec(self):
        """Right ascension and declination."""
        from ..util import ec2eq
        lam, bet = self.lambet
        ra, dec = ec2eq(lam.to(u.deg).value, bet.to(u.deg).value)
        return ra * u.deg, dec * u.deg

    @property
    def ra(self):
        """Right ascension."""
        return self.radec[0]

    @property
    def dec(self):
        """Declination."""
        return self.radec[1]

    @property
    def sangle(self):
        """Projected Sun angle."""

        from ..util import projected_vector_angle as pva

        ra, dec = self.radec
        if len(self) > 1:
            sangle = np.zeros(len(self))
            for i in range(len(self)):
                sangle[i] = pva(-self._rt[i], self._rot[i],
                                 ra[i].to(u.deg).value,
                                dec[i].to(u.deg).value)
        else:
            sangle = pva(-self._rt, self._rot, ra.to(u.deg).value,
                          dec.to(u.deg).value)
            
        return sangle * u.deg

    @property
    def vangle(self):
        """Projected velocity angle."""

        from ..util import projected_vector_angle as pva

        ra, dec = self.radec
        if len(self) > 1:
            vangle = np.zeros(len(self))
            for i in range(len(self)):
                vangle[i] = pva(self._vt[i], self._rot[i],
                                ra[i].to(u.deg).value,
                                dec[i].to(u.deg).value)
        else:
            vangle = pva(self._vt, self._rot, ra.to(u.deg).value,
                         dec.to(u.deg).value)

        return vangle * u.deg

    @property
    def selong(self):
        """Solar elongation."""
        selong = np.arccos(np.sum(-self._ro * self._rot, -1)
                           / self.obsrh.to(u.km).value
                           / self.delta.to(u.km).value)
        return np.degrees(selong) * u.deg

    @property
    def lelong(self):
        """Lunar elongation."""
        from . import Moon
        if self.date is None:
            return None
        rm = Moon.r(self.date)
        rom = rm - self._ro
        deltam = np.sqrt(np.sum(rom**2, -1))
        lelong = np.arccos(np.sum(rom * self._rot, -1)
                           / deltam / self.delta.to(u.km).value)
        return np.degrees(lelong) * u.deg

    def reduce(self, func, units=False):
        """Apply a function to each vector.

        Parameters
        ----------
        func : function
          The function to apply; accepts an ndarray as its first
          argument, and an optional axis to iterate over as its
          second.
        units : bool, optional
          Set to `True` to keep the units of each parameter in the
          output dictionary.  Keeping track of the units may not make
          sense for some functions, e.g., `np.argmin`.

        Returns
        -------
        g : dict

        """
        g = dict()

        for k in ['ro', 'rt', 'vo', 'vt']:
            v = self[k]
            if v is None:
                g[k] = None
            else:
                if v.value.ndim == 2:
                    g[k] = func(v.value, 0)
                else:
                    g[k] = v.value  # nothing to do
                if units:
                    g[k] *= v.unit
            
        if self['date'] is None:
            g['date'] = None
        else:
            g['date'] = func(self.date.utc.jd)
            if units:
                g['date'] = Time(g['date'], scale='utc', format='jd')

        for k in ['rh', 'delta', 'phase', 'signedphase', 'obsrh', 'so', 'st',
                  'lam', 'bet', 'ra', 'dec', 'sangle', 'vangle', 'selong',
                  'lelong']:
            v = self[k]
            if v is None:
                g[k] = None
            else:
                g[k] = func(v.value)
                if units:
                    g[k] *= v.unit

        g['lambet'] = g['lam'], g['bet']
        g['radec'] = g['ra'], g['dec']

        return g

    def mean(self):
        """Mean of each attribute.

        Parameters
        ----------
        None

        Returns
        -------
        g : dict

        """
        return self.reduce(np.mean, units=True)

    def min(self):
        """Minimum of each attribute.

        Note that vectors like `ro` will now be `[min(x), min(y),
        min(z)]`, and likely not a real vector from the original.

        Parameters
        ----------
        None

        Returns
        -------
        g : dict

        """
        return self.reduce(np.min, units=True)

    def max(self):
        """Maximum of each attribute.

        Note that vectors like `ro` will now be `[max(x), max(y),
        max(z)]`, and likely not a real vector from the original.

        Parameters
        ----------
        None

        Returns
        -------
        g : dict

        """
        return self.reduce(np.max, units=True)

    def argmin(self):
        """Index of the minimum of each attribute.

        Parameters
        ----------
        None

        Returns
        -------
        g : dict

        """
        return self.reduce(np.argmin)

    def argmax(self):
        """Index of the maximum of each attribute.

        Parameters
        ----------
        None

        Returns
        -------
        g : dict

        """
        return self.reduce(np.argmax)

    def summary(self):
        """A pretty summary of the object.

        If `Geom` is an array, then mean values will be printed.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """

        from astropy.coordinates import Angle
        from ..util import jd2time

        opts = dict(sep=':', precision=1, pad=True)

        if len(self) > 1:
            g = self.mean()
            gmax = self.max()
            gmin = self.min()

            datemin, timemin = jd2time(gmin['date']).iso.split()
            timemin = timemin.split('.')[0]

            datemax, timemax = jd2time(gmax['date']).iso.split()
            timemax = timemax.split('.')[0]

            minmaxtime = '     [{:}, {:}]'.format(timemin, timemax)
            minmaxdate = '   [{:}, {:}]'.format(datemin, datemax)

            ramin = Angle(gmin['ra'].value, u.deg).format(
                'hour', **opts)
            decmin = Angle(gmin['dec'].value, u.deg).format(
                'deg', alwayssign=True, **opts)
            ramax = Angle(gmax['ra'].value, u.deg).format(
                'hour', **opts)
            decmax = Angle(gmax['dec'].value, u.deg).format(
                'deg', alwayssign=True, **opts)

            raminmax = '  [ {:},  {:}]'.format(ramin, ramax)
            decminmax = '  [{:}, {:}]'.format(decmin, decmax)

            jdminmax = '   [{:.2f}, {:.2f}]'.format(gmin['date'].jd, gmax['date'].jd)

            def minmax(p, f):
                return '     [{:{f}}, {:{f}}]'.format(
                    gmin[p].value, gmax[p].value, f=f)
        else:
            g = self
            gmax = self
            gmin = self
            minmaxdate = ''
            minmaxtime = ''
            raminmax = ''
            decminmax = ''
            jdminmax = ''
            def minmax(p, f):
                return ''

        date, time = jd2time(g['date']).iso.split()
        time = time.split('.')[0]

        ra = Angle(g['ra'].value, u.deg).format('hour', **opts)
        dec = Angle(g['dec'].value, u.deg).format('deg', alwayssign=True,
                                                  **opts)

        print ("""
{:>34s} {:s}{:}
{:>34s} {:s}{:}
{:>34s} {:.2f}{:}

{:>34s} {:8.3f}{:}
{:>34s} {:8.3f}{:}
{:>34s} {:8.3f}{:}

{:>34s} {:8.3f}{:}
{:>34s} {:8.3f}{:}

{:>34s}  {:}{:}
{:>34s} {:}{:}

{:>34s} {:8.3f}{:}
{:>34s} {:8.3f}{:}
""".format("Date:", date, minmaxdate,
           "Time (UT):", time, minmaxtime,
           "Julian day:", g['date'].jd, jdminmax,
           "Heliocentric distance (AU):", g['rh'].value,
           minmax('rh', '8.3f'),
           "Target-Observer distance (AU):", g['delta'].value,
           minmax('delta', '8.3f'),
           "Sun-Object-Observer angle (deg):", g['phase'].value,
           minmax('phase', '8.3f'),
           "Sun-Observer-Target angle (deg):", g['selong'].value,
           minmax('selong', '8.3f'),
           "Moon-Observer-Target angle (deg):", g['lelong'].value,
           minmax('lelong', '8.3f'),
           "RA (hr):", ra, raminmax,
           "Dec (deg):", dec, decminmax,
           "Projected sun vector (deg):", g['sangle'].value,
           minmax('sangle', '8.3f'),
           "Projected velocity vector (deg):", g['vangle'].value,
           minmax('vangle', '8.3f')))

# update module docstring
from ..util import autodoc
autodoc(globals())
del autodoc
