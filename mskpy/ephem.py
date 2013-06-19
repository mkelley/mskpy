# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
ephem --- Ephemeris tools
=========================

Requres PySPICE.

About dates
-----------

Most functions accept multiple kinds of dates: calendar strings,
Julian dates, `Time`, or `datetime`.  If the scale is not defined (as
it is for `Time` instances), we assume the scale is UTC.

.. autosummary::
   :toctree: generated/

   Classes
   -------
   Geom
   MovingObject
   SpiceObject

   Functions
   ---------
   cal2et
   date2et
   jd2et
   time2et

   ec2eq
   find_kernel
   getgeom
   getxyz
   projected_vector_angle

   Built-in MovingObjects
   ----------------------
   Sun
   Mercury
   Venus
   Mars
   Earth
   Moon
   Mars
   Jupiter
   Saturn
   Uranus
   Neptune
   Pluto
   Spitzer
   DeepImpact

   Exceptions
   ----------
   ObjectError

"""

__all__ = [
    'Geom',
    'MovingObject',
    'SpiceObject',

    'cal2et',
    'date2et',
    'jd2et',
    'time2et',

    'ec2eq',
    'find_kernel',
    'getgeom',
    'getxyz',
    'projected_vector_angle',
    'summarizegeom',

    'Sun',
    'Mercury',
    'Venus',
    'Earth',
    'Moon',
    'Mars',
    'Jupiter',
    'Saturn',
    'Uranus',
    'Neptune',
    'Pluto'
]

import exceptions
from datetime import datetime

import numpy as np
from astropy.time import Time
import spice

_kernel_path = '/home/msk/data/kernels'
_kernel_setup = False

class ObjectError(Exception):
    pass

class Geom(dict):
    """Contains observing geometry parameters for solar system objects.

    Not very robust and needs more error checking.

    Key list and definitions:
      rh = r_h [AU]
      delta = Delta [AU]
      phase = phase >0 for before opposition, <0 after [degrees]
      ra = RA [degrees]
      dec = Dec [degrees]
      sangle = sun angle [degrees]
      vangle = velocity angle [degrees]
      selong = solar elongation [degrees]
      lelong = lunar elongation [degrees]
      date = the input date
      obsxyz = the observer's position [km]
      tarxyz = the target's position [km]
      obsrh = the observer's heliocentric distance

    Parameters
    ----------
    g : dict or Geom, optional
      The parameters.

    Notes
    -----
    There is no sanity checking on the geometry.

    Dates are converted to Julian Date before adding, subtracting,
    etc.

    """

    allowedKeys = ('date', 'rh', 'delta', 'phase', 'ra', 'dec', 'lelong',
                   'selong', 'sangle', 'vangle', 'obsxyz', 'tarxyz',
                   'obsrh')

    def __init__(self, g=None):
        if g is None:
            for k in self.allowedKeys:
                self[k] = None
        else:
            for k, v in g.items():
                self[k] = v

    def __getitem__(self, key):
        # are we slicing or getting a key?
        if type(key) in (int, slice, list, np.ndarray):
            r = Geom()
            for k, v in self.items():
                r[k] = v[key]
            return r
        else:
            if key not in self.allowedKeys:
                raise AttributeError("{:} is not an allowed key".format(key))
            return dict.__getitem__(self, key)

    def __len__(self):
        if self['rh'] is None:
            return 0
        elif np.iterable(self['rh']):
            return len(self['rh'])
        else:
            return 1

    def __add__(self, other):
        if type(self) != type(other):
            return NotImplemented
        r = Geom()
        for k in self.keys():
            if k == 'date':
                r[k] = self._date2jd(self[k]) + self._date2jd(other[k])
            else:
                r[k] = self[k] + other[k]
        return r

    def __sub__(self, other):
        if type(self) != type(other):
            return NotImplemented
        r = Geom()
        for k in self.keys():
            if k == 'date':
                r[k] = self._date2jd(self[k]) - self._date2jd(other[k])
            else:
                r[k] = self[k] - other[k]
        return r

    def __mul__(self, other):
        if type(self) == type(other):
            return NotImplemented
        r = Geom()
        for k in self.keys():
            if k == 'date':
                r[k] = self._date2jd(self[k]) * other
            else:
                r[k] = self[k] * other
        return r

    def __div__(self, other):
        if type(self) == type(other):
            return NotImplemented
        r = Geom()
        for k in self.keys():
            if k == 'date':
                r[k] = self._date2jd(self[k]) / other
            else:
                r[k] = self[k] / other
        return r

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def append(self, g):
        for k, v in g.items():
            if self[k] is None:
                self[k] = v
            elif isinstance(self[k], np.ndarray):
                if v.shape == ():
                    self[k] = np.concatenate((self[k], [v]))
                elif 'xyz' in k:
                    self[k] = np.vstack((self[k], v))
                else:
                    self[k] = np.concatenate((self[k], v))
            else:
                self[k] = np.concatenate(([self[k]], [v]))

    def mean(self):
        r = Geom()
        for k, v in self.items():
            if k == 'date':
                r[k] = np.mean(self._date2jd(v))
            elif 'xyz' in k:
                if np.rank(v) > 1:
                    r[k] = np.mean(v, 0)
                else:
                    r[k] = v
            else:
                r[k] = np.mean(v)
        return r

    def _date2jd(self, date):
        if isinstance(date, str):
            return cal2time(cal2iso(date), scale='utc').jd
        elif isinstance(date, float):
            return date
        elif isinstance(date, Time):
            return date.jd
        elif isinstance(date, datetime):
            return Time(datetime, scale='utc').jd
        elif isinstance(date, (list, tuple, np.ndarray)):
            return np.ndarray([self._date2jd(d) for d in date])
        else:
            raise ValueError("Invalid date: {}".format(date))

    def summary(self):
        """Print a pretty summary of the object.

        If `Geom` is an array, then mean values will be printed.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """

        from astropy.coordinates import Angle
        import astropy.units as u
        from util import jd2time

        g = self.mean()

        date, time = jd2time(g['date']).iso.split()
        time = time.split('.')[0]

        opts = dict(sep=':', precision=1, pad=True)
        ra = Angle(g['ra'], u.deg).format('hour', **opts)
        dec = Angle(g['dec'], u.deg).format('deg', alwayssign=True, **opts)

        print ("""
{:>34s} {:s}
{:>34s} {:s}
{:>34s} {:f}

{:>34s} {:8.3f}
{:>34s} {:8.3f}
{:>34s} {:+8.3f}

{:>34s} {:8.3f}
{:>34s} {:8.3f}

{:>34s}  {:}
{:>34s} {:}

{:>34s} {:8.3f}
{:>34s} {:8.3f}
""".format("Date:", date,
           "Time (UT):", time,
           "Julian day:", g['date'],
           "Heliocentric distance (AU):", g['rh'],
           "Target-Observer distance (AU):", g['delta'],
           "Sun-Object-Observer angle (deg):", g['phase'],
           "Sun-Observer-Target angle (deg):", g['selong'],
           "Moon-Observer-Target angle (deg):", g['lelong'],
           "RA (hr):", ra,
           "Dec (deg):", dec,
           "Projected sun vector (deg):", g['sangle'],
           "Projected velocity vector (deg):", g['vangle']))

class MovingObject(object):
    """An abstract class for an object moving in space.

    Methods
    -------
    observe : Distance, phase angle, etc. to another object.
    r : Position vector
    v : Velocity vector

    Notes
    -----
    Inheriting classes should override `r`, `v`.

    """

    def __init__(self):
        pass

    def r(self, date):
        pass

    def v(self, date):
        pass

    def observe(self, target, date, ltt=False):
        """Distance, phase angle, etc. to another object.

        Parameters
        ----------
        target : MovingObject
          The target to observe.
        date : string, float, astropy Time, datetime, or array
          Strings are parsed with `util.cal2iso`.  Floats are assumed
          to be Julian dates.  All dates except instances of `Time`
          are assumed to be UTC.  Use `None` for now.
        ltt : bool, optional
          Account for light travel time when `True`.

        Returns
        -------
        geom : Geom
          The geometric parameters of the observation.

        """

        import astropy.units as u
        from astropy.time import TimeDelta
        import astropy.constants as const

        from util import cal2time, jd2time

        global Moon

        if (isinstance(date, (list, tuple, np.ndarray))
            or (isinstance(date, Time) and len(date) > 1)):
            geom = Geom()
            for i in xrange(len(date)):
                geom.append(self.observe(target, date[i], ltt=ltt))
            return geom

        # reformat date as a Time object
        if date is None:
            date = Time(datetime.now(), scale='utc')
        elif isinstance(date, float):
            date = jd2time(date)
        elif isinstance(date, str):
            date = cal2time(date)
        elif isinstance(date, datetime):
            date = Time(date, scale='utc')

        rt = target.r(date)  # target postion
        vt = target.v(date)  # target velocity
        ro = self.r(date)    # observer position
        vo = self.v(date)    # observer velocity

        rht = np.sqrt(sum(rt**2))  # target heliocentric distance
        rho = np.sqrt(sum(ro**2))  # observer heliocentric distance
        rot = rt - ro  # observer-target vector
        delta = np.sqrt(sum((rt - ro)**2))

        # The "sign" on the phase angle.  The convention is >0 for
        # before opposition, and <0 for after opposition.  If -rh is
        # the target-Sun vector, -rot is the target-observer vector,
        # and h is the angular momentum then use >0 for (-rh X rto) *
        # h > 0.  h = rh cross vh.
        phase = np.degrees(np.arccos((rht**2 + delta**2 - rho**2) /
                                     2.0 / rht / delta))
        phase *= np.sign((np.cross(-rt, -rot) * np.cross(rt, vt)).sum())

        if ltt:
            date -= TimeDelta(delta / const.c.si.value, format='sec')
            return self.observe(target, date, ltt=False)

        # sky coordinates
        lam = np.degrees(np.arctan2(rot[1], rot[0]))
        bet = np.degrees(np.arctan2(rot[2], np.sqrt(rot[0]**2 + rot[1]**2)))
        ra, dec = ec2eq(lam, bet)

        # projected sun and velocity vectors
        sangle = projected_vector_angle(-rt, rot, ra, dec)
        vangle = projected_vector_angle(vt, rot, ra, dec)

        # solar and lunar elongations
        selong = np.degrees(np.arccos(sum(-ro * rot) / rho / delta))
        if target == Moon:
            lelong = 0
        else:
            rom = Moon.r(date) - ro
            deltam = sum(np.sqrt(rom**2))
            lelong = np.degrees(np.arccos(sum(rom * rot) / deltam / delta))

        return Geom(dict(rh = rht * u.km.to(u.AU),
                         delta = delta * u.km.to(u.AU),
                         phase = phase,
                         ra = ra,
                         dec = dec,
                         sangle = sangle,
                         vangle = vangle,
                         selong = selong,
                         lelong = lelong,
                         date = date,
                         obsxyz = ro,
                         tarxyz = rt,
                         obsrh = rho))

class SpiceObject(MovingObject):
    """A moving object saved in a SPICE planetary ephemeris kernel.

    Parameters
    ----------
    obj : string or int
      The object's name or NAIF ID, as found in the relevant SPICE
      kernel.
    kernel : string, optional
      The name of a specific SPICE planetary ephemeris kernel (SPK) to
      use for this object, or `None` to automatically search for a
      kernel through `find_kernel`.

    Methods
    -------
    observe : Distance, phase angle, etc. to another object.
    r : Position vector
    v : Velocity vector

    """

    def __init__(self, obj, kernel=None):
        if kernel is None:
            kernel = find_kernel(obj)
        _load_kernel(kernel)
        self.kernel = kernel

        if isinstance(obj, int):
            obj = str(obj)
        naifid = spice.bods2c(obj)
        if naifid is None:
            raise ObjectError(("NAIF ID of {} cannot be found"
                               " in kernel {}.").format(obj, kernel))
        self.obj = obj
        self.naifid = naifid

    def r(self, date):
        """Position vector.

        Parameters
        ----------
        date : string, float, astropy Time, datetime, or array
          Strings are parsed with `util.cal2iso`.  Floats are assumed
          to be Julian dates.  All dates except instances of `Time`
          are assumed to be UTC.

        Returns
        -------
        r : ndarray
          Position vector (3-element or Nx3 element array). [km]
       
        """

        et = date2et(date)
        if np.iterable(et):
            return np.array([self.r(t) for t in et])

        # no light corrections, sun = 10
        state, lt = spice.spkez(self.naifid, et, "ECLIPJ2000", "NONE", 10)
        return np.array(state[:3])

    def v(self, date):
        """Velocity vector.

        Parameters
        ----------
        date : string, float, astropy Time, datetime, or array
          Strings are parsed with `util.cal2iso`.  Floats are assumed
          to be Julian dates.  All dates except instances of `Time`
          are assumed to be UTC.

        Returns
        -------
        v : ndarray
          Velocity vector (3-element or Nx3 element array). [km/s]
       
        """

        et = date2et(date)
        if np.iterable(et):
            return np.array([self.v(t) for t in et])

        # no light corrections, sun = 10
        state, lt = spice.spkez(self.naifid, et, "ECLIPJ2000", "NONE", 10)
        return np.array(state[3:])

class FixedObject(MovingObject):
    """A fixed point in space.

    Parameters
    ----------
    xyz : 3-element array
      The heliocentric rectangular ecliptic coordinates of the
      point. [km]

    Methods
    -------
    observe : Distance, phase angle, etc. to another object.
    r : Position vector
    v : Velocity vector

    Raises
    ------
    ValueError if xyz.shape != (3,).

    """

    def __init__(self, xyz):
        self.xyz = np.array(xyz)
        if self.xyz.shape != (3,):
            raise ValueError("xyz must be a 3-element vector.")

    def r(self, date=None):
        """Position vector.

        Parameters
        ----------
        date : string, float, astropy Time, datetime, or array, optional
          Mosty ignored, since the position is fixed, but if an array,
          then r will be an array of positions.

        Returns
        -------
        r : ndarray
          Position vector (3-element or Nx3 element array). [km]
       
        """

        if (isinstance(date, (list, tuple, np.ndarray))
            or (isinstance(date, Time) and len(date) > 1)):
            N = len(date)
            return np.tile(self.xyz, N).reshape(N, 3)
        else:
            return self.xyz

    def v(self, date=None):
        """Velocity vector.

        Parameters
        ----------
        date : string, float, astropy Time, datetime, or array, optional
          Mosty ignored, since the position is fixed, but if an array,
          then r will be an array of positions.

        Returns
        -------
        v : ndarray
          Velocity vector (3-element or Nx3 element array). [km/s]
       
        """

        if (isinstance(date, (list, tuple, np.ndarray))
            or (isinstance(date, Time) and len(date) > 1)):
            shape = (len(date), 3)
        else:
            shape = (3,)

        return np.zeros(shape)

def _load_kernel(filename):
    """Load the named kernel into memory.

    The kernel must be in the current directory, or in `_kernel_path`.

    No-op if the kernel is already loaded.

    Parameters
    ----------
    filename : string

    Returns
    -------
    None

    Raises
    ------
    `OSError` if the file is not found.

    """

    import os.path
    global _kernel_path

    if not os.path.exists(filename):
        filename = os.path.join(_kernel_path, filename)
        if not os.path.exists(filename):
            raise OSError("naif.tls not found")

    if spice.kinfo(filename) is None:
        spice.furnsh(filename)

def _load_pck():
    """Load the planetary constants kernel into memory.

    The kernel must be named "pck.tpc" and be in the current
    directory, or in `_kernel_path`.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    `OSError` if the file is not found.

    """
    _load_kernel("pck.tpc")

def _load_tls():
    """Load the leap seconds kernel into memory.

    The kernel must be named "niaf.tls" and be in the current
    directory, or in `_kernel_path`.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    `OSError` if the file is not found.

    """
    _load_kernel('naif.tls')


def cal2et(date):
    """Convert calendar date to SPICE ephemeris time.

    Ephemeris time is seconds past 2000.0.

    UTC is assumed.

    Parameters
    ----------
    date : string or array
      The date.

    Returns
    -------
    et : float or array
      Ephemeris time.

    """

    from util import cal2iso
    global _kernel_setup

    if not _kernel_setup:
        _load_tls()
        _load_pck()
        _kernel_setup = True

    if isinstance(date, (list, tuple, np.ndarray)):
        return [cal2et(x) for x in t]

    return spice.utc2et(cal2iso(date))

def date2et(date):
    """Variety of date formats to ephemeris time.

    SPICE ephemeris time is seconds past 2000.

    Parameters
    ----------
    date : string, float, astropy Time, datetime, or array
      Strings are parsed with `util.cal2iso`.  Floats are assumed to
      be Julian dates.

    Returns
    -------
    et : float or ndarray
      SPICE ephemeris time.

    """

    if date is None:
        date = Time(datetime.now(), scale='utc')

    if isinstance(date, float):
        et = jd2et(date)
    elif isinstance(date, str):
        et = cal2et(date)
    elif isinstance(date, Time):
        et = time2et(date)
    elif isinstance(date, (list, tuple, np.ndarray)):
        et = np.array([date2et(t) for t in date])
    else:
        raise ValueError("Invalid date: {}".format(date))

    return et

def jd2et(jd):
    """Convert Julian date to SPICE ephemeris time.

    Ephemeris time is seconds past 2000.0.

    UTC is assumed.

    Parameters
    ----------
    jd : string, float, or array
      The Julian date.  Strings will have higher precisions than
      floats.

    Returns
    -------
    et : float or array
      Ephemeris time.

    """

    global _kernel_setup

    if not _kernel_setup:
        _load_tls()
        _load_pck()
        _kernel_setup = True

    if isinstance(jd, (list, tuple, np.ndarray)):
        return [jd2et(x) for x in jd]

    if isinstance(jd, float):
        jd = "{:17.9f} JD".format(jd)

    return spice.utc2et(jd)

def time2et(t):
    """Convert astropy `Time` to SPICE ephemeris time.

    Ephemeris time is seconds past 2000.0.

    Parameters
    ----------
    t : astropy Time
      The time.  Must be convertable to the UTC scale.

    Returns
    -------
    et : float or array
      Ephemeris time.

    """

    global _kernel_setup

    if not _kernel_setup:
        _load_tls()
        _load_pck()
        _kernel_setup = True

    if isinstance(t, (list, tuple, np.ndarray)) or len(t) > 1:
        return [time2et(x) for x in t]

    return spice.utc2et(t.utc.iso)

def ec2eq(lam, bet):
    """Ecliptic coordinates to equatorial (J2000.0) coordinates.

    Parameters
    ----------
    lam, bet : float or array
      Ecliptic longitude and latitude. [degrees]

    Returns
    -------
    ra, dec : float or ndarray
      Equatorial (J2000.0) longitude and latitude. [degrees]

    Notes
    -----
    Based on euler.pro in the IDL Astro library (W. Landsman).

    """

    # using the mean obliquity of the ecliptic at the J2000.0 epoch
    # eps = 23.439291111 degrees (Astronomical Almanac 2008)
    ceps = 0.91748206207 # cos(eps)
    seps = 0.39777715593 # sin(eps)

    # convert to radians
    lam = np.radians(lam)
    bet = np.radians(bet)
    
    cbet = np.cos(bet)
    sbet = np.sin(bet)
    clam = np.cos(lam)
    slam = np.sin(lam)

    ra = np.arctan2(ceps * cbet * slam - seps * sbet, cbet * clam)
    sdec = seps * cbet * slam + ceps * sbet

    if np.iterable(sdec):
        sdec[sdec > 1.0] = 1.0
    else:
        if sdec > 1.0:
            sdec = 1.0
    dec = np.arcsin(sdec)

    # make sure 0 <= ra < 2pi
    ra = (ra + 4.0 * np.pi) % (2.0 * np.pi)

    return np.degrees(ra), np.degrees(dec)

def find_kernel(obj):
    """Find a planetary ephemeris kernel, based on object name.

    Searches the current directory first, then `_kernel_path`.

    Parameters
    ----------
    obj : string or int
      The object's name or NAIF ID.  Object names are converted to
      lower case, and all non-alphanumeric characters are removed.
      The suffix '.bsp' is appended.

    Returns
    -------
    kernel : string
      The path to the kernel file.

    Raises
    ------
    ValueError if the file cannot be found.

    """

    from os import path
    global _kernel_path

    kernel = filter(lambda s: s.isalnum(), str(obj)) + '.bsp'
    if not path.isfile(kernel):
        if path.isfile(path.join(_kernel_path, kernel)):
            kernel = path.join(_kernel_path, kernel)
        else:
            raise ValueError("Cannot find kernel (" + kernel + ")")

    return kernel

def getgeom(target, observer, date=None, ltt=False, kernel=None):
    """Moving target geometry parameters for an observer and date.

    Parameters
    ----------
    target : string, MovingObject
      The object's name or NAIF ID, as found in the relevant SPICE
      kernel, or a `MovingObject`.
    observer : string, array, MovingObject
      A valid built-in observer name, set of heliocentric rectangular
      ecliptic J2000 coordinates, or a `MovingObject`.  See the
      `ephem` package documentation for built-in moving objects that
      can be used as observers.
    date : string, float or array, optional
      The date(s) for which to compute the target's geometry.
    ltt : bool, optional
      Set to true to correct parameters for light travel time
      (currently, only one ltt iteration is implemented).
    kernel : string, optional
      If the target is a string, use this kernel.

    Returns
    -------
    geom : Geom
      The geometric parameters of the observation.

    Raises
    ------
    ValueError on invalid input.

    """

    global _loaded_objects

    if isinstance(target, str):
        target = SpiceObject(target, kernel=kernel)
    elif not isinstance(target, MovingObject):
        raise ValueError("target must be a string or MovingObject")

    if isinstance(observer, str):
        if observer.lower() in _loaded_objects:
            observer = _loaded_objects[observer.lower()]
        else:
            ValueError("{} is not in the built-in list: {}".format(
                    observer.lower(), _loaded_objects.keys()))
    elif np.isiterable(observer):
        observer = FixedObject(target)
    elif not isinstance(target, MovingObject):
        raise ValueError("target must be a string or MovingObject")

    return observer.observe(target, date, ltt=ltt)

def getxyz(obj, date=None, kernel=None):
    """Coordinates and velocity from an ephemeris kernel.

    Coordinates are heliocentric rectangular ecliptic J2000.

    Parameters
    ----------
    obj : string or int
      The object's name or NAIF ID, as found in the relevant SPICE
      kernel.
    date : string, float, astropy Time, datetime, or array, optional
      Strings are parsed with `util.cal2iso`.  Floats are assumed to
      be Julian dates.
    kernel : string, optional
      The name of a specific SPICE planetary ephemeris kernel (SPK) to
      use for this object, or `None` to automatically search for a
      kernel through `find_kernel`.

    Returns
    -------

    r, v: array
      The position and veloctiy vectors as 3-element arrays, or, if
      multiple dates are requested, Nx3-element arrays. [km and km/s]

    Raises
    ------
    ValueError for invalid `date`.
    ObjectError for `obj` not in `kernel`.

    """

    obj = SpiceObject(obj, kernel=kernel)
    return obj.r(date), obj.v(date)

def projected_vector_angle(r, rot, ra, dec):
    """Position angle of a vector projected onto the observing plane.

    Parameters
    ----------
    r : array
      The vector to project, in heliocentric ecliptic
      coordinates. [km]
    rot : array
      The observer-target vector. [km]
    ra, dec : float
      The right ascention and declination of the target, as seen by
      the observer. [deg]

    Returns
    -------
    angle : float
      The position angle w.r.t. to equatorial north. [deg]

    """

    rh = np.sqrt((r**2).sum())
    dv = rot + rh / r / 1000.  # delta vector

    # find the projected vectors in RA, Dec
    lam2 = np.degrees(np.arctan2(dv[1], dv[0]))
    bet2 = np.degrees(np.arctan2(dv[2], np.sqrt(dv[0]*dv[0] + dv[1]*dv[1])))

    ra2, dec2 = ec2eq(lam2, bet2)

    x2 = (ra2 - ra) * np.cos(np.radians(dec2)) * 3600.0
    y2 = (dec2 - dec) * 3600.0

    th = np.degrees(np.arctan2(y2, x2))
    pa = 90.0 - th
    
    return pa

def summarizegeom(*args, **kwargs):
    """Pretty print a summary of the observing geometry.

    See `getgeom` for a description of the parameters.

    """

    getgeom(*args, **kwargs).summary()

# load up a few objects
Sun = SpiceObject('sun', kernel='planets.bsp')
Mercury = SpiceObject('mercury', kernel='planets.bsp')
Venus = SpiceObject('venus', kernel='planets.bsp')
Earth = SpiceObject('earth', kernel='planets.bsp')
Moon = SpiceObject('moon', kernel='planets.bsp')
Mars = SpiceObject('mars', kernel='planets.bsp')
Jupiter = SpiceObject('jupiter', kernel='planets.bsp')
Saturn = SpiceObject('saturn', kernel='planets.bsp')
Uranus = SpiceObject('uranus', kernel='planets.bsp')
Neptune = SpiceObject('neptune', kernel='planets.bsp')
Pluto = SpiceObject('pluto', kernel='planets.bsp')
Spitzer = SpiceObject('-79', kernel='spitzer.bsp')
DeepImpact = SpiceObject('-140', kernel='deepimpact.txt')
_loaded_objects = dict(sun=Sun, mercury=Mercury, venus=Venus, earth=Earth,
                       moon=Moon, mars=Mars, jupiter=Jupiter, saturn=Saturn,
                       uranus=Uranus, neptune=Neptune, pluto=Pluto,
                       spitzer=Spitzer, deepimpact=DeepImpact)
