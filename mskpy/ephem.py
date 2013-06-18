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

_kernel_path = '/home/msk/data/kernels'

# planets + Pluto + Sun names to NAIF IDs
_planets = dict(mercury='199', venus='299', earth='399', mars='499',
                jupiter='5', saturn='6', uranus='7', neptune='9',
                pluto='9', sun='10')

class MovingObject():
    pass

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

def getxyz():
    """Coordinates and velocity from an ephemeris kernel.

    Coordinates are heliocentric rectangular ecliptic.

    Parameters
    ----------
    obj : string or int
      The object's name or NAIF ID, as found in the relevant SPICE
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
