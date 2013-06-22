# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
ephem --- Ephemeris tools
=========================

Requres PySPICE.


About kernels
-------------

See `find_kernel` for a description of how `ephem` tries to determine
kernel file names from object names.

Three SPICE kernels are required:

- naif.tls : a leap seconds kernel,

- pck.tpc : a planetary constants kernel,

- planets.bsp : a planetary ephemeris kernel, e.g., de421.

There are three optional kernels:

- spitzer.bsp : an ephemeris kernel for the Spitzer Space Telescope,

- deepimpact.txt : an ephemeris meta-kernel for Deep Impact Flyby,

- naif-names.txt : your own body to ID code mappings.


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
   find_kernel
   getgeom
   getxyz
   summarizegeom

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
   Spitzer (optional)
   DeepImpact (optional)

   Exceptions
   ----------
   ObjectError

"""

__all__ = [
    'Geom',
    'MovingObject',
    'SpiceObject',

    'find_kernel',
    'getgeom',
    'getxyz',
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
import astropy.units as u
from astropy.time import Time
import spice

_kernel_path = '/home/msk/data/kernels'
_spice_setup = False

class ObjectError(Exception):
    pass

class Geom(dict):
    """Contains observing geometry parameters for solar system objects.

    Not very robust and needs more error checking.

    Keys
    ----
    rh : float or Quantity
      Heliocentric distance.  [float: AU]
    delta = float or Quantity
      Observer-target distance.  [float: AU]
    phase : float, Quantity, or Angle
      Phase angle.  May be signed.  [float: degrees]
    ra : float, Quantity, or Angle
      Right Ascension.  [float: degrees]
    dec : float, Quantity, or Angle
      Declination.  [float: degrees]
    sangle : float, Quantity, or Angle
      Projected Sun angle.  [float: degrees]
    vangle : float, Quantity, or Angle
      Projected velocity angle.  [float: degrees]
    selong : float, Quantity, or Angle
      Solar elongation.  [float: degrees]
    lelong : float, Quantity, or Angle
      Lunar elongation [float: degrees]
    date : string, astropy Time, datetime
      The input date.  Except for `Time` objects, UTC is assumed.
    obsxyz : float or Quantity
      The observer's position.  [float: km]
    tarxyz : float or Quantity
      The target's position.  [float: km]
    obsrh : float or Quantity
      The observer's heliocentric distance.  [float: AU]

    Parameters
    ----------
    g : dict or Geom, optional
      The parameters.

    Notes
    -----
    `g[key]` returns a Quantity, but `g.items()`, `g.values()`,
    `g.itervalues()` returns the underlying values (floats or
    ndarrays).

    The only operation defined for `Geom['date']` is `-`; `+`, `*`,
    `/` all return `None`.

    There is no sanity checking on the geometry.

    Dates are converted to Julian Date before adding, subtracting,
    etc.

    """

    allowedKeys = ('date', 'rh', 'delta', 'phase', 'ra', 'dec', 'lelong',
                   'selong', 'sangle', 'vangle', 'obsxyz', 'tarxyz',
                   'obsrh')

    units = dict(date=None, rh=u.au, delta=u.au, phase=u.deg,
                 ra=u.deg, dec=u.deg, lelong=u.deg, selong=u.deg,
                 sangle=u.deg, vangle=u.deg, obsxyz=u.km,
                 tarxyz=u.km, obsrh=u.au)

    def __init__(self, g=None):
        from astropy.units import Quantity
        from astropy.coordinates import Angle

        if g is None:
            for k in self.allowedKeys:
                self[k] = None
        else:
            for k, v in g.items():
                if self.units[k] is None:
                    self[k] = v
                elif isinstance(v, Quantity):
                    self[k] = v.to(self.units[k]).value
                elif isinstance(v, Angle):
                    self[k] = v.degrees
                else:
                    self[k] = v

    def __getitem__(self, key):
        # are we slicing or getting a key?
        if isinstance(key, (int, slice, list, np.ndarray)):
            r = Geom()
            for k, v in self.items():
                r[k] = v[key]
            return r
        else:
            if key not in self.allowedKeys:
                raise AttributeError("{:} is not a Geom key".format(key))
            v = dict.__getitem__(self, key)
            if (v is not None) and (self.units[key] is not None):
                v *= self.units[key]
            return v

    def __len__(self):
        if self['rh'] is None:
            return 0
        elif np.iterable(self['rh']):
            return len(self['rh'])
        else:
            return 1

    def __add__(self, other):
        if not isinstance(other, Geom):
            raise NotImplemented("other must be a Geometry object")
        r = Geom()
        for k in self.keys():
            if k == 'date':
                r[k] = None
            else:
                r[k] = self[k] + other[k]
        return r

    def __sub__(self, other):
        if not isinstance(other, Geom):
            raise NotImplemented("other must be a Geometry object")
        r = Geom()
        for k in self.keys():
            if k == 'date':
                r[k] = self._date2time(self[k]) - self._date2time(other[k])
            else:
                r[k] = self[k] - other[k]
        return r

    def __mul__(self, other):
        if not isinstance(other, Geom):
            raise NotImplemented("other must be a Geometry object")
        r = Geom()
        for k in self.keys():
            if k == 'date':
                r[k] = None
            else:
                r[k] = self[k] * other
        return r

    def __div__(self, other):
        if not isinstance(other, Geom):
            raise NotImplemented("other must be a Geometry object")
        r = Geom()
        for k in self.keys():
            if k == 'date':
                r[k] = None
            else:
                r[k] = self[k] / other
        return r

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def append(self, g):
        if not isinstance(g, Geom):
            g = Geom(g)
        for k in g.keys():
            if self[k] is None:
                self[k] = g[k]
            elif np.iterable(self[k]):
                this = self[k].value
                that = np.array(g[k].value)
                if that.shape == ():
                    self[k] = np.concatenate((self[k], [that]))
                elif 'xyz' in k:
                    self[k] = np.vstack((self[k], that))
                else:
                    self[k] = np.concatenate((self[k], that))
            else:
                self[k] = np.concatenate(([self[k]], [that]))

    def mean(self):
        """Averaged geometry.

        Parameters
        ----------
        None

        Returns
        -------
        g : Geom

        """

        r = Geom()
        for k, v in self.items():
            if k == 'date':
                r[k] = np.mean(self[k].jd)
            elif 'xyz' in k:
                if np.rank(v) > 1:
                    r[k] = np.mean(v, 0)
                else:
                    r[k] = v
            else:
                r[k] = np.mean(v)
        return r

    def _date2time(self, date):
        from .util import cal2time, jd2time
        if isinstance(date, str):
            return cal2time(cal2iso(date), scale='utc')
        elif isinstance(date, float):
            return jd2time(date)
        elif isinstance(date, Time):
            return Time
        elif isinstance(date, datetime):
            return Time(datetime, scale='utc')
        elif isinstance(date, (list, tuple, np.ndarray)):
            return np.ndarray([self._date2time(d) for d in date])
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
        from util import jd2time

        g = self.mean()

        date, time = jd2time(g['date']).iso.split()
        time = time.split('.')[0]

        opts = dict(sep=':', precision=1, pad=True)
        ra = Angle(g['ra'].value, u.deg).format('hour', **opts)
        dec = Angle(g['dec'].value, u.deg).format('deg', alwayssign=True,
                                                  **opts)

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
           "Heliocentric distance (AU):", g['rh'].value,
           "Target-Observer distance (AU):", g['delta'].value,
           "Sun-Object-Observer angle (deg):", g['phase'].value,
           "Sun-Observer-Target angle (deg):", g['selong'].value,
           "Moon-Observer-Target angle (deg):", g['lelong'].value,
           "RA (hr):", ra,
           "Dec (deg):", dec,
           "Projected sun vector (deg):", g['sangle'].value,
           "Projected velocity vector (deg):", g['vangle'].value))

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

        from astropy.time import TimeDelta
        import astropy.constants as const

        from util import cal2time, jd2time, ec2eq, projected_vector_angle

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
        global _spice_setup

        if not _spice_setup:
            _setup_spice()

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
            raise OSError("{} not found".format(filename))

    if spice.kinfo(filename) is None:
        spice.furnsh(filename)

def _setup_spice():
    """Load some kernels into memory.

    Three SPICE kernels are required:

    - naif.tls : a leap seconds kernel,

    - pck.tpc : a planetary constants kernel,

    - planets.bsp : a planetary ephemeris kernel, e.g., de421.

    Additionally, naif-names.txt, your own body to ID code mappings,
    is loaded if it exists.
    
    Parameters
    ----------
    None

    Returns
    -------
    None


    """

    global _spice_setup

    _load_kernel("naif.tls")
    _load_kernel("pck.tpc")
    _load_kernel("planets.bsp")
    try:
        _load_kernel("naif-names.txt")
    except OSError:
        pass
    _spice_setup = True

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
    global _spice_setup

    if not _spice_setup:
        _setup_spice()

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

    global _spice_setup

    if not _spice_setup:
        _setup_spice()

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

    global _spice_setup

    if not _spice_setup:
        _setup_spice()

    if isinstance(t, (list, tuple, np.ndarray)) or len(t) > 1:
        return [time2et(x) for x in t]

    return spice.utc2et(t.utc.iso)

def find_kernel(obj):
    """Find a planetary ephemeris kernel, based on object name.

    Searches the current directory first, then `_kernel_path`.
    
    Three steps are taken to find the appropriate kernel:

    1) Try the object name with the suffix '.bsp'.

    2) Convert `obj` to lower case, and remove all non-alphanumeric
       characters.  The suffix '.bsp' is appended.

    3) If `obj` is an integer and < 1000000, it is assumed to be an
       asteroid designation.  Try once more, with `obj + 2000000`.

    Parameters
    ----------
    obj : string or int
      The object's name or NAIF ID.

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

    kernel = str(obj) + '.bsp'
    if path.isfile(obj):
        return kernel
    elif path.isfile(path.join(_kernel_path, kernel)):
        return path.join(_kernel_path, kernel)

    kernel = filter(lambda s: s.isalnum(), str(obj)).lower() + '.bsp'
    if path.isfile(kernel):
        return kernel
    elif path.isfile(path.join(_kernel_path, kernel)):
        return path.join(_kernel_path, kernel)

    if isinstance(obj, int):
        if obj < 1000000:
            # looks like an asteroid designation, try the
            # asteroid's NAIFID
            return find_kernel(obj + 2000000)

    raise ValueError("Cannot find kernel (" + kernel + ")")

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
    elif np.iterable(observer):
        observer = FixedObject(observer)
    elif not isinstance(observer, MovingObject):
        raise ValueError("observer must be a string or MovingObject")

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
_loaded_objects = dict(sun=Sun, mercury=Mercury, venus=Venus, earth=Earth,
                       moon=Moon, mars=Mars, jupiter=Jupiter, saturn=Saturn,
                       uranus=Uranus, neptune=Neptune, pluto=Pluto)

# load 'em if you got 'em
try:
    Spitzer = SpiceObject('-79', kernel='spitzer.bsp')
    _loaded_objects['spitzer'] = Spitzer
    __all__.append('Spitzer')
except OSError:
    pass

try:
    DeepImpact = SpiceObject('-140', kernel='deepimpact.txt')
    _loaded_objects['deepimpact'] = DeepImpact
    __all__.append('DeepImpact')
except OSError:
    pass
    
