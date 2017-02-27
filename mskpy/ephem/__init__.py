# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
ephem --- Ephemeris tools
=========================

Requres SpiceyPy.

   Classes
   -------
   Geom
   SolarSysObject
   SpiceObject

   Functions
   ---------
   getgeom
   getxyz
   summarizegeom

   Built-in SolarSysObjects
   ------------------------
   Sun, Mercury, Venus, Earth, Moon, Mars, Jupiter, Saturn, Uranus,
   Neptune, PlutoSys.  Optional: Spitzer, DeepImpact, Kepler.

   Exceptions
   ----------
   ObjectError


About kernels
-------------

See `find_kernel` for a description of how `ephem` tries to determine
kernel file names from object names.

Three SPICE kernels are required:
  - naif.tls : a leap seconds kernel,
  - pck.tpc : a planetary constants kernel,
  - planets.bsp : a planetary ephemeris kernel, e.g., de421.

There are five optional kernels:
  - L2.bsp : an ephemeris kernel for the second Lagrange point in the Earth-Sun system, https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/lagrange_point/
  - spitzer.bsp : an ephemeris kernel for the Spitzer Space Telescope, ftp://naif.jpl.nasa.gov/pub/naif/SIRTF/kernels/spk/
  - kepler.bsp : an ephemeris kernel for the Kepler Telescope, https://archive.stsci.edu/pub/k2/spice/
  - deepimpact.txt : an ephemeris meta-kernel for Deep Impact Flyby, ftp://naif.jpl.nasa.gov/pub/naif/
  - naif-names.txt : your own body to ID code mappings.

About dates
-----------

Most functions accept multiple types of dates: calendar strings,
Julian dates, `Time`, or `datetime`.  If the scale is not defined (as
it is for `Time` instances), we assume the scale is UTC.

"""

import astropy.units as u

from . import core
from . import geom
from . import state
from . import ssobj

from .geom import *
from .state import *
from .ssobj import *

__all__ = geom.__all__ + state.__all__ + ssobj.__all__

# load up a few objects

# G * Masses are from Standish, E.M. (1998) "JPL Planetary and Lunar
# Ephemerides, DE405/LE405", JPL IOM 312.F-98-048.
GM_sun = 1.32712440017987e20 * u.m**3 / u.s**2
GM_planets = [22032.080, 324858.599, 398600.433, 42828.314,
              126712767.863, 37940626.063, 5794549.007,
              6836534.064, 981.601]
GM_ast = [62.375, 13.271, 17.253]  # Ceres, Pallas, Vesta
GM_moon = 4902.801
GM_earth_sys = 403503.233

Sun = getspiceobj('Sun', kernel='planets.bsp', GM=GM_sun)
Mercury = getspiceobj('Mercury', kernel='planets.bsp', GM=GM_planets[0])
Venus = getspiceobj('Venus', kernel='planets.bsp', GM=GM_planets[1])
EarthSys = getspiceobj('3', kernel='planets.bsp', GM=GM_earth_sys)
Earth = getspiceobj('399', kernel='planets.bsp', GM=GM_planets[2])
Moon = getspiceobj('301', kernel='planets.bsp', GM=GM_moon)
Mars = getspiceobj('4', name='Mars', kernel='planets.bsp', GM=GM_planets[3])
Jupiter = getspiceobj('5', name='Jupiter', kernel='planets.bsp', GM=GM_planets[4])
Saturn = getspiceobj('6', name='Saturn', kernel='planets.bsp', GM=GM_planets[5])
Uranus = getspiceobj('7', name='Uranus', kernel='planets.bsp', GM=GM_planets[6])
Neptune = getspiceobj('8', name='Neptune', kernel='planets.bsp', GM=GM_planets[7])
PlutoSys = getspiceobj('9', name='PlutoSys', kernel='planets.bsp', GM=GM_planets[8])
_loaded_objects = dict(sun=Sun, mercury=Mercury, venus=Venus, earth=Earth,
                       moon=Moon, mars=Mars, jupiter=Jupiter, saturn=Saturn,
                       uranus=Uranus, neptune=Neptune, pluto=PlutoSys)
__all__.extend(['Sun', 'Earth', 'Moon'])

# load 'em if you got 'em
try:
    Earth_L2 = getspiceobj('392', kernel='L2.bsp', name='Earth L2')
    _loaded_objects['earth_l2'] = Earth_L2
    __all__.append('Earth_L2')
except OSError:
    pass

try:
    Spitzer = getspiceobj('-79', kernel='spitzer.bsp', name='Spitzer')
    _loaded_objects['spitzer'] = Spitzer
    __all__.append('Spitzer')
except OSError:
    pass

try:
    DeepImpact = getspiceobj('-140', kernel='deepimpact.txt',
                             name='Deep Impact')
    _loaded_objects['deepimpact'] = DeepImpact
    __all__.append('DeepImpact')
except OSError:
    pass

try:
    Kepler = getspiceobj('-227', kernel='kepler.bsp', name='Kepler')
    _loaded_objects['kepler'] = Kepler
    __all__.append('Kepler')
except OSError:
    pass

# update module docstring
from ..util import autodoc
autodoc(globals())
del autodoc
