# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
ephem --- Ephemeris tools
=========================

Requres PySPICE.

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
   date2time
   jd2et
   time2et

   ec2eq
   find_kernel
   getgeom
   getxyz
   projected_vector_angle

   Exceptions
   ----------
   ObjectError

"""

__all__ = [
    'Geom',
    'MovingObject',
    'SpiceObject',

    'cal2et',
    'date2time',
    'jd2et',
    'time2et',

    'ec2eq',
    'find_kernel',
    'getgeom',
    'getxyz',
    'projected_vector_angle',

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

import numpy as np
import astropy.units as u
from astropy.units import Quantity
import spice

_kernel_path = '/home/msk/data/kernels'
_kernel_setup = False

# planets + Pluto + Sun names to NAIF IDs
_planets = dict(mercury='199', venus='299', earth='399', mars='499',
                jupiter='5', saturn='6', uranus='7', neptune='9',
                pluto='9', sun='10')

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
                r[k] = time.s2jd(self[k]) + time.s2jd(other[k])
            else:
                r[k] = self[k] + other[k]
        return r

    def __sub__(self, other):
        if type(self) != type(other):
            return NotImplemented
        r = Geom()
        for k in self.keys():
            if k == 'date':
                r[k] = time.s2jd(self[k]) - time.s2jd(other[k])
            else:
                r[k] = self[k] - other[k]
        return r

    def __mul__(self, other):
        if type(self) == type(other):
            return NotImplemented
        r = Geom()
        for k in self.keys():
            if k == 'date':
                r[k] = time.s2jd(self[k]) * other
            else:
                r[k] = self[k] * other
        return r

    def __div__(self, other):
        if type(self) == type(other):
            return NotImplemented
        r = Geom()
        for k in self.keys():
            if k == 'date':
                r[k] = time.s2jd(self[k]) / other
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
                r[k] = np.mean(time.s2jd(v))
            elif 'xyz' in k:
                r[k] = np.mean(v, 0)
            else:
                r[k] = np.mean(v)
        return r

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
          are assumed to be UTC.
        ltt : bool, optional
          Account for light travel time when `True`.

        Returns
        -------
        geom : Geom
          The geometric parameters of the observation.

        """

        from datetime import datetime

        from astropy.time import Time, TimeDelta
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
        if isinstance(date, float):
            date = jd2time(date)
        elif isinstance(date, str):
            date = cal2time(date)
        elif isinstance(date, datetime):
            date = Time(date, scale='utc').jd

        rt = target.r(date)  # target postion
        vt = target.r(date)  # target velocity
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

    from datetime import datetime
    from astropy.time import Time

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

#def find_names(kernel):
#    """Return the names of objects in a planetary ephemeris kernel.
#
#    Parameters
#    ----------
#    kernel : string
#      The name of a kernel file.
#
#    Returns
#    -------
#    names : list
#      A list of the objects in `kernel`.
#
#    """
# need a working spice.spkobj

#def getgeom(target, observer, date=None, ltt=False, kernel=None):
#    """Moving target geometry parameters for an observer and date.
#
#    Parameters
#    ----------
#    target : string
#      A valid target name (see spice.getxyz()).
#    observer : string or array
#      A valid observer name or set of heliocentric rectangular
#      ecliptic J2000 coordinates.
#    date : string, float or array, optional
#      The date(s) for which to compute the target's geometry.
#    ltt : bool, optional
#      Set to true to correct parameters for light travel time
#      (currently, only one ltt iteration is implemented).
#    kernel : string, optional
#      Use this kernel for the target (see `getxyz`).
#
#    Returns
#    -------
#    geom : Geom
#      The geometric parameters of the observation.
#
#    Notes
#    -----
#      
#    v1.0.0 Written by Michael S. Kelley, UCF, Jun 2007
#
#    v1.0.1 coords.Position is not returning the correct RA and Dec
#           values; instead, we now use ec2eq(), MSK, 24 Aug
#           2007
#
#    v1.1.0 Added a sign to phase to indicate if the observation is
#           before or after opposition using the convention described
#           to me by Thomas Mueller; RA and Dec still not right, so I
#           am masking their output, MSK, 22 Sep 2008
#
#    v1.1.1 Small tweaks while trying to find the RA/Dec issue... there
#           is no issue.  RA and Dec computed with ec2eq is
#           good, but comes out with a negative value that confuses
#           some later step.  RA and Dec output is re-enabled.  MSK,
#           UMD, 24 Jun 2009
#
#    v1.2.0 Renamed as getgeom() from get_comet_astrom(); calls to
#           get_comet_xyz() renamed to use getxyz(), MSK, 15 Jul 2009
#
#    v1.3.0 Added selong and lelong, MSK, 24 Jul 2009
#
#    v1.4.0 Allows date to be a list of dates, MSK, 2 Jun 2010
#
#    v1.5.0 Now checks to see if the observer is in the supplied
#           kernel.  MSK, 08 Nov 2011
#
#    v1.6.0 Now returns a Geom object.  MSK, 18 Oct 2012.
#
#    v1.6.1 Now, really really returns a Geom object.  Adding obsxyz
#           and tarxyz.  MSK, 1 Feb 2013.
#    """
#
#    if date is not None:
#        if type(date) in (tuple, list, np.ndarray):
#            geom = Geom()
#            for i in range(len(date)):
#                geom.append(getgeom(target, observer, date=date[i],
#                                    ltt=ltt, kernel=kernel))
#            return geom
#
#    # get the target's position and velocity
#    RHt, VHt = spice.getxyz(target, date=date, kernel=kernel)
#    rht = np.sqrt(sum(RHt**2))
#    vht = np.sqrt(sum(VHt**2))
#
#    # get the observer's position, pass the kernel along in case the
#    # observer contained within it
#    RHo = np.array(spice.get_observer_xyz(observer, date, kernel=kernel),
#                   np.float64)
#    # return None on error
#    if RHo is None: return None
#    rho = np.sqrt(sum(RHo**2))
#
#    # target-observer distance and phase angle
#    delta = np.sqrt(((RHt - RHo)**2).sum())
#    phase = np.degrees(np.arccos((rht**2 + delta**2 - rho**2) / 
#                                 2.0 / rht / delta))
#
#    # What is the "sign" on phase angle?
#    # The convention is + for before opposition, and - for after opposition.
#    # If -r_h is the target-Sun vector, r_to is the target-observer
#    # vector, and h is the angular momentum then use + for (-r_h X
#    # r_to) * h > 0, and - if < 0.  h = r_h X v_h
#    phase *= np.sign((np.cross(-RHt, (RHo - RHt)) *
#                         np.cross(RHt, VHt)).sum())
#
#    # correct for light travel time, if requested
#    # 1 AU / c = 499.004783806 +/- 0.00000001 s
#    # c = 173.14463 AU/day
#    if ltt:
#        jd = time.date2jd(date) - delta / AU / 173.14463
#        return getgeom(target, observer, date=jd, kernel=kernel)
#
#    # RA, Dec (observer->target J2000)
#    ROt = RHt - RHo
#    lambdaOT = np.arctan2(ROt[1], ROt[0]) * 180.0 / pi
#    betaOT = np.arctan2(ROt[2], np.sqrt(ROt[0]**2 + ROt[1]**2)) * 180.0 / pi
#
#    # ecliptic to equatorial
#    raOT, decOT = ec2eq(lambdaOT, betaOT)
#
#    # # project v and r onto the observing plane
#    # ro = ROt - RHt / rht / 1000.
#    # vo = ROt + VHt / vht / 1000.
#    # 
#    # # find the projected vectors in RA, Dec
#    # lambdaOR = np.arctan2(ro[1], ro[0]) * 180.0 / pi
#    # lambdaOV = np.arctan2(vo[1], vo[0]) * 180.0 / pi
#    # betaOR = np.arctan2(ro[2], np.sqrt(ro[0]*ro[0] + ro[1]*ro[1])) * 180.0 / pi
#    # betaOV = np.arctan2(vo[2], np.sqrt(vo[0]*vo[0] + vo[1]*vo[1])) * 180.0 / pi
#    # 
#    # raOR, decOR = ec2eq(lambdaOR, betaOR)
#    # raOV, decOV = ec2eq(lambdaOV, betaOV)
#    # 
#    # xro = (raOR - raOT) * np.cos(decOR * pi / 180.0) * 3600.0
#    # xvo = (raOV - raOT) * np.cos(decOV * pi / 180.0) * 3600.0
#    # yro = (decOR - decOT) * 3600.0
#    # yvo = (decOV - decOT) * 3600.0
#    # 
#    # thOR = np.arctan2(yro, xro) * 180.0 / pi
#    # thOV = np.arctan2(yvo, xvo) * 180.0 / pi
#    # 
#    # sangle = 90.0 - thOR
#    # vangle = 90.0 - thOV
#
#    sangle = projected_vector_angle(-RHt, ROt, raOT, decOT)
#    vangle = projected_vector_angle(VHt, ROt, raOT, decOT)
#
#    # compute the solar and lunar elongations
#    ROs = -RHo
#    selong = np.degrees(np.arccos((ROs * ROt).sum() / 
#                                        np.sqrt((ROs**2).sum()) /
#                                        np.sqrt((ROt**2).sum())))
#
#    RHm = spice.getxyz('301', date=date, kernel='planets.bsp')[0]
#    ROm = RHm - RHo
#    lelong = np.degrees(np.arccos((ROm * ROt).sum() / 
#                                        np.sqrt((ROm**2).sum()) /
#                                        np.sqrt((ROt**2).sum())))
#
#    return Geom({'rh': rht / AU, 'delta': delta / AU, 'phase': phase,
#                 'ra': raOT, 'dec': decOT, 'sangle': sangle, 'vangle': vangle,
#                 'selong': selong, 'lelong': lelong, 'date': date,
#                 'obsxyz': RHo, 'tarxyz': RHt})

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

    et = date2et(date)
    if isinstance(date, np.ndarray):
        rv = np.array([getxyz(obj, t, kernel=kernel) for t in date])
        return rv[:, 0], rv[:, 1]

    if kernel is None:
        kernel = find_kernel(obj)
    _load_kernel(kernel)

    if isinstance(obj, int):
        obj = str(obj)
    naifid = spice.bods2c(obj)
    if naifid is None:
        raise ObjectError("NAIF ID of {} cannot be found in kernel {}.".format(
                obj, kernel))

    # no light corrections, sun = 10
    state, lt = spice.spkez(naifid, et, "ECLIPJ2000", "NONE", 10)
    return state[:3], state[3:]

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
