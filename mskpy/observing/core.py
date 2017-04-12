# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
observing.core --- Core observing functions
===========================================

   Functions
   ---------
   airmass
   ct2lst
   ct2lst0
   hadec2altaz
   rts

"""

import numpy as np

__all__ = [
    'airmass',
    'ct2lst',
    'ct2lst0',
    'hadec2altaz',
    'rts'
]

def airmass(ra, dec, date, lon, lat, tz):
    """Target airmass.

    Uses Eq. 3 from Kasten and Young, 1989, Applied Optics, 28, 4735.
    If the zenith angle is greater than 89 degrees, then airmass will
    be se to NaN.

    Parameters
    ----------
    ra, dec : float
      Right ascension and declination of the target. [deg]
    date : string, float, astropy Time, datetime, or array
      The current date (civil time), passed to `util.date2time`.
    lon, lat : float
      The (east) longitude, and latitude of the Earth-bound
      observer. [deg]
    tz : float or string
      float: The UTC offset of the observer. [hr]
      string: A timezone name processed with `pytz` (e.g., US/Arizona).

    Returns
    -------
    am : float or array
      The object's airmass for each date.

    """

    lst = ct2lst(date, lon, tz) * 15.0
    ha = lst - ra
    alt = hadec2altaz(ha, dec, lat)[0]

    old = np.seterr(invalid='ignore')
    secz = 1.0 / np.cos(np.radians(90.0 - alt))
    am = 1.0 / (np.sin(np.radians(alt)) + 0.1500 * (alt + 3.885)**-1.253)
    np.seterr(**old)

    if np.iterable(am):
        am[alt < 1.0] = np.nan
    else:
        if alt < 1.0:
            am = np.nan
    return am

def ct2lst(date, lon, tz):
    """Convert civil time to local sidereal time.

    See Meeus, Astronomical Algorithms.

    Parameters
    ----------
    date : string, float, astropy Time, datetime, or array
      The current date (civil time), passed to `util.date2time`.
    lon : float
      The East longitude of the observer. [deg]
    tz : float or string
      float: The UTC offset of the observer. [hr]
      string: A timezone name processed with `pytz` (e.g., US/Arizona).

    Returns
    -------
    lst : float
      The local sidereal time.  [hr]

    """

    from ..util import date2time, tz2utc

    d = date2time(date)
    jd = d.jd
    if isinstance(tz, (float, int)):
        tzoff = tz
    else:
        tzoff = tz2utc(d, tz).total_seconds() / 3600.0

    jd0 = np.round(jd - tzoff / 24.0 - 1.0) + 0.5  # JD for 0h UT
    T = (jd - 2451545.0) / 36525  # JD2000 = 2451545
    th0 = 280.46061837 + 360.98564736629 * (jd - 2451545.0) + \
        0.000387933 * T**2 - T**3 / 38710000.0
    th0 = th0 % 360.0
    lst = ((th0 + lon) / 15.0  - tzoff) % 24.0
    return lst

def ct2lst0(date, lon, tz):
    """Convert civil time to local sidereal time at nearest midnight.

    See Meeus, Astronomical Algorithms.

    Parameters
    ----------
    date : string, float, astropy Time, datetime, or array
      The current date (civil time), passed to `util.date2time`.
    lon : float
      The East longitude of the observer. [deg]
    tz : float or string
      float: The UTC offset of the observer. [hr]
      string: A timezone name processed with `pytz` (e.g., US/Arizona).

    Returns
    -------
    lst0 : float
      The local sidereal time at nearest midnight.  [hr]

    """

    from ..util import date2time

    d = date2time(date)
    jd = round(d.jd - 0.5) + 0.5
    return ct2lst(jd, lon, tz)

def hadec2altaz(ha, dec, lat):
    """Convert hour angle and declination to altitude and azimuth.

    Parameters
    ----------
    ha : float or array
      Hour angle. [deg]
    dec : float
      Target declination. [deg]
    lat : float
      The latitude of the observer. [deg]

    Returns
    -------
    alt, az : float
      The altitude and azimuth of the object.  The outputs may be NxM
      arrays, where N is the length of `dec`, and M is the length of
      `ha`. [deg]

    Notes
    -----
    Based on the IDL Astron hadec2altaz procedure by Chris O'Dell
    (UW-Maddison).

    """

    sha = np.sin(np.radians(np.array(ha)))
    cha = np.cos(np.radians(np.array(ha)))
    sdec = np.sin(np.radians(np.array(dec)))
    cdec = np.cos(np.radians(np.array(dec)))
    slat = np.sin(np.radians(lat))
    clat = np.cos(np.radians(lat))

    x = -cha * cdec * slat + sdec * clat
    y = -sha * cdec
    z = cha * cdec * clat + sdec * slat
    r = np.sqrt(x**2 + y**2)

    alt = np.degrees(np.arctan2(z, r))
    az = np.degrees(np.arctan2(y, x)) % 360.0

    return alt, az

def rts(ra, dec, date, lon, lat, tz, limit=20, precision=1440):
    """Rise, transit, set times for an object.

    Rise and set may be at the horizon, or elsewhere.

    Parameters
    ----------
    ra, dec : float
      Right ascension and declination of the target. [deg]
    date : string, float, astropy Time, datetime, or array
      The current date (civil time), passed to `util.date2time`.
    lon, lat : float
      The (east) longitude, and latitude of the Earth-bound
      observer. [deg]
    tz : float or string
      float: The UTC offset of the observer. [hr]
      string: A timezone name processed with `pytz` (e.g., US/Arizona).
    limit : float
      The altitude at which the object should be considered
      risen/set. [deg]
    precision : int
      Number of steps to take per day, affecting the r/t/s precision.

    Returns
    -------
    r, t, s : float
      Rise, transit, set times for `date`.  If the object does not
      set, `r` and `s` will be `None`.  If the object is always lower
      than `limit`, `t` will be `None`. [hr]

    """

    from ..util import date2time, nearest, deriv

    # truncate the date
    lst0 = ct2lst0(date, lon, tz) * 15.0 # deg

    ha = np.linspace(-180, 180, precision, endpoint=False)
    ct = np.linspace(-12, 12, precision, endpoint=False)  # civil time
    ct[ct < 0] += 24

    # roll ha, so that we get the correct hour angle at midnight
    ha0 = (lst0 - ra) % 360.0  # HA at midnight
    if ha0 > 180:
        ha0 -= 360.0

    i = nearest(ha, ha0)
    ha = np.roll(ha, int(precision / 2 - i))

    alt = hadec2altaz(ha, dec, lat)[0]

    # make a branch cut at alt.min()
    cut = alt.argmin()
    alt = np.roll(alt, -cut)
    ct = np.roll(ct, -cut)
    dalt = deriv(alt)

    transit = alt.argmax()
    if not any(alt < limit):
        r, t, s = None, ct[transit], None
    elif all(alt < limit):
        r, t, s = None, None, None
    else:
        t = ct[transit]
        i = nearest(alt[:transit], limit)
        j = transit + nearest(alt[transit:], limit)
        if dalt[i] > 0:
            r = ct[i]
            s = ct[j]
        else:
            r = ct[j]
            s = ct[i]

    return r, t, s

