# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
state --- Position and velocity vector classes.
===============================================

   Classes
   -------
   State
   FixedState
   SpiceState

   Exceptions
   ----------
   ObjectError

"""

__all__ = [
    'State',
    'FixedState',
    'SpiceState',

    'ObjectError'
]

from datetime import datetime

import numpy as np
from astropy.time import Time
import spice

from . import core

class ObjectError(Exception):
    pass

class State(object):
    """An abstract class for computing state vectors.

    Methods
    -------
    r : Position vector.
    v : Velocity vector.

    Notes
    -----
    Inheriting classes should override `r` and `v`.

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

        N = util.date_len(date)
        if N != 0:
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

        N = util.date_len(date)
        if N != 0:
            shape = (N, 3)
        else:
            shape = (3,)

        return np.zeros(shape)

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

        N = util.date_len(date)
        if N > 0:
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

        N = util.date_len(date)
        if N > 0:
            return np.array([self.v(d) for d in date])

        et = core.date2et(date)

        # no light corrections, sun = 10
        state, lt = spice.spkez(self.naifid, et, "ECLIPJ2000", "NONE", 10)
        return np.array(state[3:])

