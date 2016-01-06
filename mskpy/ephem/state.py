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

from datetime import datetime

import numpy as np
from astropy.time import Time
import spiceypy.wrapper as spice

from . import core

class ObjectError(Exception):
    pass

class State(object):
    """An abstract class for computing state vectors.

    Methods
    -------
    r : Position vector.
    v : Velocity vector.
    rv : Both vectors.

    Notes
    -----
    Inheriting classes should override `rv`.

    """

    def __init__(self):
        pass

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
        from .. import util
        N = util.date_len(date)
        if N > 0:
            return np.array([self.rv(d)[0] for d in date])
        return self.rv(date)[0]

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
        from .. import util
        N = util.date_len(date)
        if N > 0:
            return np.array([self.rv(d)[1] for d in date])
        return self.rv(date)[1]

class FixedState(State):
    """A fixed point in space.

    Parameters
    ----------
    xyz : 3-element array
      The heliocentric rectangular ecliptic coordinates of the
      point. [km]

    Methods
    -------
    r : Position vector.
    v : Velocity vector.
    rv : Position and velocity vectors.

    Raises
    ------
    ValueError if xyz.shape != (3,).

    """

    def __init__(self, xyz):
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

        self.name = kwargs.pop('name', None)
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
    r : Position vector.
    v : Velocity vector.
    rv : Both vectors.

    Attributes
    ----------
    obj, kernel : from Parameters
    naifid : the NAIF ID of the object

    """

    def __init__(self, obj, kernel=None):
        if not core._spice_setup:
            core._setup_spice()

        if kernel is None:
            kernel = core.find_kernel(obj)
        core.load_kernel(kernel)
        self.kernel = kernel

        if isinstance(obj, int):
            obj = str(obj)
        naifid, success = spice.bods2c(obj)
        if not success:
            s = ("NAIF ID of {} cannot be found in kernel {}"
                 " or kernel pool.").format(obj, kernel)
            raise ObjectError(s)

        self.obj = obj
        self.naifid = naifid

    def r(self, date, **kwargs):
        """Position vector.

        Parameters
        ----------
        date : string, float, astropy Time, datetime, or array
          Processed via `util.date2time`.
        kwargs :
          Keyword arguments passed to `SpiceState.rv`.

        Returns
        -------
        r : ndarray
          Position vector (3-element or Nx3 element array). [km]
       
        """
        from .. import util
        N = util.date_len(date)
        if N > 0:
            return np.array([self.rv(d)[0] for d in date])
        return self.rv(date, **kwargs)[0]

    def v(self, date, **kwargs):
        """Velocity vector.

        Parameters
        ----------
        date : string, float, astropy Time, datetime, or array
          Processed via `util.date2time`.
        kwargs :
          Keyword arguments passed to `SpiceState.rv`.

        Returns
        -------
        v : ndarray
          Velocity vector (3-element or Nx3 element array). [km/s]
       
        """
        from .. import util
        N = util.date_len(date)
        if N > 0:
            return np.array([self.rv(d)[1] for d in date])
        return self.rv(date, **kwargs)[1]

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
        et = core.date2et(date)
        # no light corrections, sun = 10
        state, lt = spice.spkez(self.naifid, et, frame, corr, observer)
        return np.array(state[:3]), np.array(state[3:])
