# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
state --- Position and velocity vector classes.
===============================================

   Classes
   -------
   State
   FixedState
   KeplerState
   SpiceState

   Exceptions
   ----------
   ObjectError

"""

__all__ = [
    'State',
    'FixedState',
    'KeplerState',
    'SpiceState',

    'ObjectError'
]

from abc import ABC, abstractmethod
from datetime import datetime
import numpy as np
from astropy.time import Time
import spiceypy as spice

from . import core

class ObjectError(Exception):
    pass

class State(ABC):
    """An abstract base class for computing state vectors.

    Methods
    -------
    r : Position vector.
    v : Velocity vector.
    rv : Both vectors.
    h : Angular momentum integral.

    """

    def __init__(self, name=None):
        self.name = name

    @abstractmethod
    def rv(self, date):
        """Position and velocity vectors.

        Parameters
        ----------
        date : string, float, astropy Time, datetime
          Processed via `util.date2time`.

        Returns
        -------
        r : ndarray
          Position vector. [km]
        v : ndarray
          Velocity vector. [km/s]
       
        """
        pass

    def r(self, date, **kwargs):
        """Position vector.

        Parameters
        ----------
        date : string, float, astropy Time, datetime, or array
          Processed via `util.date2time`.
        kwargs :
          Keyword arguments passed to `rv`.

        Returns
        -------
        r : ndarray
          Position vector (3-element or Nx3 element array). [km]
       
        """
        from .. import util
        N = util.date_len(date)
        if N > 0:
            return np.array([self.rv(d, **kwargs)[0] for d in date])
        return self.rv(date, **kwargs)[0]

    def v(self, date, **kwargs):
        """Velocity vector.

        Parameters
        ----------
        date : string, float, astropy Time, datetime, or array
          Processed via `util.date2time`.
        kwargs :
          Keyword arguments passed to `rv`.

        Returns
        -------
        v : ndarray
          Velocity vector (3-element or Nx3 element array). [km/s]
       
        """
        from .. import util
        N = util.date_len(date)
        if N > 0:
            return np.array([self.rv(d, **kwargs)[1] for d in date])
        return self.rv(date, **kwargs)[1]

    def h(self, date, **kwargs):
        """Angular momentum integral, `r × v`.

        Parameters
        ----------
        date : string, float, astropy Time, datetime, or array
          Processed via `util.date2time`.
        kwargs :
          Keyword arguments passed to `rv`.

        Returns
        -------
        h : ndarray
          3-element or Nx3 element array.  [km2/s]

        """
        return np.cross(*self.rv(date, **kwargs))    

class FixedState(State):
    """A fixed point in space.

    Parameters
    ----------
    xyz : 3-element array
      The heliocentric rectangular ecliptic coordinates of the
      point. [km]
    name : string, optional
      The object's name.

    Methods
    -------
    r : Position vector.
    v : Velocity vector.
    rv : Position and velocity vectors.

    Raises
    ------
    ValueError if xyz.shape != (3,).

    """

    def __init__(self, xyz, **kwargs):
        State.__init__(self, **kwargs)
        self.xyz = np.array(xyz)
        if self.xyz.shape != (3,):
            raise ValueError("xyz must be a 3-element vector.")

    def rv(self, date=None):
        """Position and velocity vectors.

        Parameters
        ----------
        date : string, float, astropy Time, datetime, optional
          Ignored, since the position is fixed.

        Returns
        -------
        r : ndarray
          Position vector. [km]
        v : ndarray
          Velocity vector. [km/s]
       
        """
        return self.xyz, np.zeros(3)

class KeplerState(State):
    """A moving object state based on a two-body solution.

    obj = KeplerState(r_i, v_i, date)
    obj = KeplerState(state, date)

    Parameters
    ----------
    r_i : array
      The initial position vector. [km]
    v_i : array
      The initial velocity vector. [km/s]
    state : State or simlar
      Create a KeplerState based on another solution.  `state` must
      have callable `r` and `v` methods.
    date : various
      The date at which `r_i` and `v_i` are valid, processed with
      `util.date2time`.
    GM : float, optional
      Gravity of the central mass. [km**3/s**2]
    name : string, optional
      The object's name.

    Attributes
    ----------
    jd : `date` as a Julian date.

    Methods
    -------
    r : Position vector.
    v : Velocity vector.
    rv : Both vectors.

    """

    def __init__(self, *args, **kwargs):
        from ..util import date2time

        State.__init__(self, **kwargs)

        if hasattr(args[0], 'r') and hasattr(args[0], 'v'):
            self.date = date2time(args[1])
            self.r_i = args[0].r(self.date)
            self.v_i = args[0].v(self.date)
            if (self.name is None) and hasattr(args[0], 'name'):
                self.name = args[0].name
        else:
            self.r_i = np.array(args[0])
            self.v_i = np.array(args[1])
            self.date = date2time(args[2])

        self.GM = kwargs.pop('GM', 132749351440.0)  # km**3/s**2
        self.jd = self.date.jd
        self.rv_i = np.r_[self.r_i, self.v_i]  # km, km/s

    def __str__(self):
        if self.name is None:
            return '<KeplerState>'
        else:
            return '<KeplerState name="{}">'.format(self.name)

    def rv(self, date):
        """Position and velocity vectors.

        Parameters
        ----------
        date : string, float, astropy Time, or datetime
          Processed via `util.date2time`.

        Returns
        -------
        r : ndarray
          Position vector. [km]
        v : ndarray
          Velocity vector. [km/s]

        """
        from .. import util
        jd = util.date2time(date).jd
        dt = (jd - self.jd) * 86400.0
        rv = np.array(spice.prop2b(self.GM, self.rv_i, dt))
        return rv[:3], rv[3:]

class SpiceState(State):
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
    r : Position vector. [km]
    v : Velocity vector. [km/s]
    rv : Both vectors.
    h : Angular momentum integral. [km2/s]
    oscelt : Conic orbital elements.

    Attributes
    ----------
    obj, kernel : from Parameters
    naifid : the NAIF ID of the object

    """

    def __init__(self, obj, kernel=None):
        State.__init__(self, name=obj)
        
        if not core._spice_setup:
            core._setup_spice()

        if kernel is None:
            kernel = core.find_kernel(obj)
        core.load_kernel(kernel)
        self.kernel = kernel

        if isinstance(obj, int):
            obj = str(obj)

        naifid = spice.bods2c(obj)

        self.obj = obj
        self.naifid = naifid

    def rv(self, date, frame="ECLIPJ2000", corr="NONE", observer=10):
        """Position and velocity vectors.

        Parameters
        ----------
        date : string, float, astropy Time, datetime
          Processed via `util.date2time`.
        frame : string
          The name of a SPICE reference frame.
        corr : string
          The SPICE abberation correction.
        observer : integer
          The NAIF ID of the observer when computing vectors.

        Returns
        -------
        r : ndarray
          Position vector. [km]
        v : ndarray
          Velocity vector. [km/s]
       
        """

        from .. import util

        N = util.date_len(date)
        if N > 0:
            rv = tuple(zip(*(self.rv(d) for d in date)))
            return np.array(rv[0]), np.array(rv[1])

        et = core.date2et(date)
        # no light corrections, sun = 10
        state, lt = spice.spkez(self.naifid, et, frame, corr, observer)
        return np.array(state[:3]), np.array(state[3:])

    def oscelt(self, date, frame="ECLIPJ2000", mu=None):
        """Concic osculating orbital elements.

        Returns the orbit from oscelt in the SPICE toolkit.  The
        results are unreliable for eccentricities very close to 1.0,
        specific angular momentum near zero, and inclinations near 0
        or 180 degrees.  See the SPICE toolkit for notes.

        Parameters
        ----------
        date : string, float, astropy Time, datetime
          Processed via `util.date2time`.
        frame : string
          The name of a SPICE reference frame.
        mu : float, optional
          `G M` of the primary body, or `None` to use the Sun.
        
        Returns
        -------
        orbit : dict
          Orbital parameters as a dictionary.

        """

        import astropy.units as u
        from mskpy.ephem import GM_sun
        from mskpy.util import jd2time

        et = core.date2et(date)
        state, lt = spice.spkez(self.naifid, et, frame, "NONE", 10)

        if mu is None:
            mu = GM_sun.to('km3 / s2').value

        o = spice.oscelt(state, et, mu)
        orbit = {}
        orbit['q'] = (o[0] * u.km).to(u.au)
        orbit['e'] = o[1]
        orbit['i'] = (o[2] * u.rad).to(u.deg)
        orbit['node'] = (o[3] * u.rad).to(u.deg)
        orbit['peri'] = (o[4] * u.rad).to(u.deg)
        orbit['n'] = (o[5] * u.rad).to(u.deg) / u.s
        orbit['t'] = jd2time(float(core.et2jd(o[6])))

        return orbit
