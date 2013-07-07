# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
ssobj --- Solar System objects.
===============================

   Classes
   -------
   SolarSysObject

   Functions
   ---------
   getgeom
   getxyz
   summarizegeom

"""

__all__ = [
    'FixedObject',
    'SolarSysObject',
    'SpiceObject',
    'getgeom',
    'getxyz',
    'summarizegeom',
    'ObjectError'
]

import numpy as np
import astropy.units as u
from astropy.time import Time

from . import core

class ObjectError(Exception):
    pass

class SolarSysObject(object):
    """An abstract class for an object in the Solar System.

    Methods
    -------
    ephemeris : Ephemeris for a target.
    observe : Distance, phase angle, etc. to another object.
    orbit : Osculating orbital parameters at date.
    r : Position vector
    v : Velocity vector

    Notes
    -----
    Inheriting classes should override `r`, `v`.

    """

    def __init__(self):
        pass

    def r(self, date):
        """Position vector.

        Parameters
        ----------
        date : string, float, astropy Time, datetime, or array
          Processed via `util.date2time`.

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
          Processed via `util.date2time`.

        Returns
        -------
        v : ndarray
          Velocity vector (3-element or Nx3 element array). [km/s]
       
        """
        pass

    def ephemeris(self, target, dates, num=None, columns=None,
                  cformats=None, date_format=None, ltt=False):
        """Ephemeris for a target.

        Parameters
        ----------
        target : SolarSysObject
          The target to observe.
        dates : array
          If `num` is `None`, set `dates` to a list of exact times for
          the ephemeris, otherwise, let `dates` be a start and stop
          time, and `num` be the number of dates to generate.  `dates`
          may be in any format that `observe` accepts.
        num : int, optional
          If not `None`, generate this many time steps between
          `min(dates)` and `max(dates)`.
        columns : array, optional
          A list of `Geom` keywords to use as table columns, or `None`
          for the default list.
        cformats : dict or list, optional
          A list of format strings or functions with which to convert
          the columns into strings, or a dictionary of formats with
          keys corresponding with `columns`.
        date_format : function
          A function to format the `date` column before creating the
          table.
        ltt : bool, optional
          Set to `True` to account for light travel time.

        Returns
        -------
        eph : astropy Table

        Notes
        -----
        `date_format` should be removed when astropy `Table` can
        handle Time objects.

        """

        from astropy.table import Table, Column
        from astropy.units import Quantity
        from ..util import dh2hms, date2time

        dates = date2time(dates)
        if num is not None:
            if num <= 0:
                time = []
            elif num == 1:
                time = [dates[0], dates[-1]]
            else:
                step = (dates[-1] - dates[0]) / float(num - 1)
                time = []
                for i in xrange(num):
                    time += [dates[0] + step * i]

        g = self.observe(target, time, ltt=ltt)

        if columns is None:
            columns = ['date', 'ra', 'dec', 'rh', 'delta', 'phase', 'selong']

        if cformats is None:
            cformats = dict(
                date = '{:s}',
                ra = lambda x: dh2hms(x / 15.0)[:-7],
                dec = lambda x: dh2hms(x)[:-7],
                lam = '{:.0f}',
                bet = '{:+.0f}',
                rh = '{:.3f}',
                delta = '{:.3f}',
                phase = '{:.0f}',
                selong = '{:.0f}',
                lelong = '{:.0f}')

        if date_format is None:
            date_format = lambda d: d.iso[:-7]

        eph = Table()
        eph.meta['ltt'] = ltt
        #eph.meta['observer'] = str(self)
        #eph.meta['target'] = str(target)

        for c in columns:
            if c == 'date':
                data = [date_format(d) for d in g[c]]
            else:
                data = g[c].value

            if c in cformats:
                cf = cformats[c]
            else:
                cf = None

            eph.add_column(Column(data=data, name=c, format=cf))

        return eph

    def observe(self, target, date, ltt=False):
        """Distance, phase angle, etc. to another object.

        Parameters
        ----------
        target : SolarSysObject
          The target to observe.
        date : string, float, astropy Time, datetime, or array
          Processed via `util.date2time`.
        ltt : bool, optional
          Account for light travel time when `True`.

        Returns
        -------
        geom : Geom
          The geometric parameters of the observation.

        """

        from astropy.time import TimeDelta
        import astropy.constants as const
        from . import Geom
        from ..util import date2time

        date = date2time(date)

        rt = target.r(date) # target postion
        ro = self.r(date)   # observer position
        vt = target.v(date) # target velocity
        vo = self.v(date)   # observer velocity

        g = Geom(ro * u.km, rt * u.km,
                 vo=vo * u.km / u.s, vt=vt * u.km / u.s,
                 date=date)

        if ltt:
            date -= TimeDelta(g['delta'] / const.c.si.value, format='sec')
            g = self.observe(target, date, ltt=False)

        return g

    def orbit(self, date):
        """Osculating orbital elements.

        Parameters
        ----------
        date : string, float, astropy Time, datetime, or array
          Processed via `util.date2time`.

        Returns
        -------
        orbit : dict
          See `util.state2orbit`.

        """

        from ..util import state2orbit, date2time
        r = self.r(date)
        v = self.v(date)
        jd = date2time(date).jd
        return state2orbit(r, v)

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
        import spice

        if not core._spice_setup:
            core._setup_spice()

        if kernel is None:
            kernel = core.find_kernel(obj)
        core.load_kernel(kernel)
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
          Processed via `util.date2time`.

        Returns
        -------
        r : ndarray
          Position vector (3-element or Nx3 element array). [km]
       
        """

        import spice

        if isinstance(date, (list, tuple, np.ndarray)):
            return np.array([self.r(d) for d in date])
        if isinstance(date, Time) and len(date) > 1:
            return np.array([self.r(d) for d in date])

        et = core.date2et(date)

        # no light corrections, sun = 10
        state, lt = spice.spkez(self.naifid, et, "ECLIPJ2000", "NONE", 10)
        return np.array(state[:3])

    def v(self, date):
        """Velocity vector.

        Parameters
        ----------
        date : string, float, astropy Time, datetime, or array
          Processed via `util.date2time`.

        Returns
        -------
        v : ndarray
          Velocity vector (3-element or Nx3 element array). [km/s]
       
        """

        import spice

        if isinstance(date, (list, tuple, np.ndarray)):
            return np.array([self.v(d) for d in date])
        if isinstance(date, Time) and len(date) > 1:
            return np.array([self.v(d) for d in date])

        et = core.date2et(date)

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
      Processed via `util.date2time`.
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

# update module docstring
from ..util import autodoc
autodoc(globals())
del autodoc
