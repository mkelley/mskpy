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
   SolarSysObject
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

   Built-in SolarSysObjects
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
    'SolarSysObject',
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
from astropy.coordinates import Angle
import spice

_kernel_path = '/home/msk/data/kernels'
_spice_setup = False

class ObjectError(Exception):
    pass

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
    mean : Return the average geometry as a `FrozenGeom`.

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
            if len(self.date) != self._len:
                raise ValueError("Given ro, the length of date "
                                 " must be {}.".format(self._len))

    def __len__(self):
        if self._ro.ndim == 1:
            self._len = 1
        else:
            self._len = self._ro.shape[0]
        return self._len

    def __getitem__(self, key):
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
            elif len(self.date) == 1:
                date = self.date
            else:
                date = self.date[key]

            return Geom(ro, rt, vo=vo, vt=vt, date=date)
        else:
            return self.__getattribute__(key)

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
        return np.degrees(phase) * u.deg

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
        from .util import ec2eq
        lam, bet = self.lambet
        ra, dec = ec2eq(lam.degree, bet.degree)
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

        from .util import projected_vector_angle as pva

        ra, dec = self.radec
        if len(self) > 1:
            sangle = np.zeros(len(self))
            for i in range(len(self)):
                sangle[i] = pva(-self._rt[i], self._rot[i], ra[i].degree,
                                 dec[i].degree)
        else:
            sangle = pva(-self._rt, self._rot, ra.degree, dec.degree)
            
        return sangle * u.deg

    @property
    def vangle(self):
        """Projected velocity angle."""

        from .util import projected_vector_angle as pva

        ra, dec = self.radec
        if len(self) > 1:
            vangle = np.zeros(len(self))
            for i in range(len(self)):
                vangle[i] = pva(-self._vt[i], self._rot[i], ra[i].degree,
                                 dec[i].degree)
        else:
            vangle = pva(-self._vt, self._rot, ra.degree, dec.degree)

        return vangle * u.deg

    @property
    def selong(self):
        """Solar elongation."""
        selong = np.arccos(np.sum(-self._ro * self._rot, -1)
                           / self.obsrh.kilometer / self.delta.kilometer)
        return np.degrees(selong) * u.deg

    @property
    def lelong(self):
        """Lunar elongation."""
        from .ephem import Moon
        if self.date is None:
            return None
        rom = Moon.r(self.date)
        deltam = np.sqrt(np.sum(rom**2, -1))
        lelong = np.arccos(np.sum(rom * self._rot, -1)
                           / deltam / self.delta.kilometer)
        return np.degrees(lelong) * u.deg

    def reduce(self, func, units=False):
        """Apply `func` to each vector.

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


class SolarSysObject(object):
    """An abstract class for an object in the Solar System.

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

    def _date2time(self, date):
        # reformat date as a Time object
        from .util import cal2time, jd2time

        if date is None:
            date = Time(datetime.now(), scale='utc')
        elif isinstance(date, float):
            date = jd2time(date)
        elif isinstance(date, str):
            date = cal2time(date)
        elif isinstance(date, datetime):
            date = Time(date, scale='utc')
        elif isinstance(date, (list, tuple, np.ndarray)):
            date = [self._date2time(d) for d in date]
            date = Time(date)
        else:
            raise ValueError("Bad date: {}".format(date))
        return date

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
        pass

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
        pass

    def observe(self, target, date, ltt=False):
        """Distance, phase angle, etc. to another object.

        Parameters
        ----------
        target : SolarSysObject
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

        date = self._date2time(date)

        rt = target.r(date) # target postion
        ro = self.r(date)   # observer position
        vt = target.v(date) # target velocity
        vo = self.v(date)   # observer velocity

        g = Geom(ro * u.km, rt * u.km,
                 vo=vo * u.km / u.s, vt=vt * u.km / u.s,
                 date=date)

        if ltt:
            date -= TimeDelta(delta / const.c.si.value, format='sec')
            g = self.observe(target, date, ltt=False)

        return g

class SpiceObject(SolarSysObject):
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

        if isinstance(date, (list, tuple, np.ndarray)):
            return np.array([self.r(d) for d in date])
        if isinstance(date, Time) and len(date) > 1:
            return np.array([self.r(d) for d in date])

        et = date2et(date)

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
        if isinstance(date, (list, tuple, np.ndarray)):
            return np.array([self.v(d) for d in date])
        if isinstance(date, Time) and len(date) > 1:
            return np.array([self.v(d) for d in date])

        et = date2et(date)

        # no light corrections, sun = 10
        state, lt = spice.spkez(self.naifid, et, "ECLIPJ2000", "NONE", 10)
        return np.array(state[3:])

class FixedObject(SolarSysObject):
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
    if path.isfile(kernel):
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
    target : string, SolarSysObject
      The object's name or NAIF ID, as found in the relevant SPICE
      kernel, or a `SolarSysObject`.
    observer : string, array, SolarSysObject
      A valid built-in observer name, set of heliocentric rectangular
      ecliptic J2000 coordinates, or a `SolarSysObject`.  See the
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
    elif not isinstance(target, SolarSysObject):
        raise ValueError("target must be a string or SolarSysObject")

    if isinstance(observer, str):
        if observer.lower() in _loaded_objects:
            observer = _loaded_objects[observer.lower()]
        else:
            ValueError("{} is not in the built-in list: {}".format(
                    observer.lower(), _loaded_objects.keys()))
    elif np.iterable(observer):
        observer = FixedObject(observer)
    elif not isinstance(observer, SolarSysObject):
        raise ValueError("observer must be a string or SolarSysObject")

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
    
