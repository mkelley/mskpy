"""
core --- Basic functions, mostly time, for ephem.
=================================================

   Functions
   ---------
   cal2et
   date2et
   jd2et
   time2et
   find_kernel
   load_kernel

"""

from datetime import datetime
from os.path import expanduser

import numpy as np
from astropy.time import Time
import spiceypy.wrapper as spice
from ..config import config

_kernel_path = config.get('ephem.core', 'kernel_path')
_spice_setup = False

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

    load_kernel("naif.tls")
    load_kernel("pck.tpc")
    load_kernel("planets.bsp")
    try:
        load_kernel("naif-names.txt")
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

    from ..util import cal2iso
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
    from .. import util

    if not _spice_setup:
        _setup_spice()

    if util.date_len(t) > 0:
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

def load_kernel(filename):
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

    test = spice.kinfo(filename, 512, 512)
    if test[3]:
        spice.furnsh(filename)

# update module docstring
from ..util import autodoc
autodoc(globals())
del autodoc
