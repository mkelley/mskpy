# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
observing.core --- Core observing functions
===========================================

   Functions
   ---------
   airmass
   ct2lst
   hadec2altaz


"""

import numpy as np

__all__ = [
    'airmass',
    'ct2lst',
    'hadec2altaz'
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
    lon : float
      The (east) longitude, and latitude of the Earth-bound
      observer. [deg]
    tz : float
      The UTC offset of the observer. [hr]

    Returns
    -------
    am : float or array
      The object's airmass for each date.

    """

    lst = ct2lst(date, lon, tz) * 15.0
    ha = lst - ra
    alt = hadec2altaz(ha, dec, lat)[0]
    secz = 1.0 / np.cos(np.radians(90.0 - alt))
    am = 1.0 / (np.sin(np.radians(alt)) + 0.1500 * (alt + 3.885)**-1.253)
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
    tz : float
      The UTC offset of the observer. [hr]

    Returns
    -------
    lst : float
      The local sidereal time.  [hr]

    """

    from ..util import date2time

    jd = date2time(date).jd
    jd0 = np.round(jd - tz / 24.0 - 1.0) + 0.5  # JD for 0h UT
    T = (jd - 2451545.0) / 36525  # JD2000 = 2451545
    th0 = 280.46061837 + 360.98564736629 * (jd - 2451545.0) + \
        0.000387933 * T**2 - T**3 / 38710000.0
    th0 = th0 % 360.0
    lst = ((th0 + lon) / 15.0  - tz) % 24.0
    return lst

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
