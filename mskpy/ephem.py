# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
ephem --- Ephemeris tools
=========================

Requres PySPICE.

.. autosummary::
   :toctree: generated/

   MovingTarget

   getxyz

"""

__all__ = [
    'MovingObject',
    'getxyz'
]

import numpy as np
import astropy.units as u
from astropy.units import Quantity
import spice

class MovingObject():
    pass

def getxyz():
    """Coordinates and velocity from an ephemeris kernel.

    Coordinates are heliocentric rectangular ecliptic.

    Parameters
    ----------
    obj : string or int
      The object's name or NAIFID, as found in the relevant SPICE
      kernel.
    date : string, astropy Time, or datetime, optional
      Strings are assumed to be UTC: YYYY-MM-DD HH:MM:SS.SSS, only
      YYYY is required, e.g., try '2013', or '2013-06-17'.
    kernel : string, optional
      The name of a specific SPICE planetary ephemeris kernel (SPK) to
      use for this object, or None to automatically search for a
      kernel through `find_kernel`.

    Returns
    -------
    
    """
    pass

def s2dt():
    """

    """
