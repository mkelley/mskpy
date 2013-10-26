# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
ephem --- Ephemeris tools
=========================

Requres PySPICE.

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
   Neptune, Pluto, Spitzer (optional), DeepImpact (optional)

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

There are three optional kernels:
  - spitzer.bsp : an ephemeris kernel for the Spitzer Space Telescope,
  - deepimpact.txt : an ephemeris meta-kernel for Deep Impact Flyby,
  - naif-names.txt : your own body to ID code mappings.

About dates
-----------

Most functions accept multiple types of dates: calendar strings,
Julian dates, `Time`, or `datetime`.  If the scale is not defined (as
it is for `Time` instances), we assume the scale is UTC.

"""

from . import core
from . import geom
from . import state
from . import ssobj

from .geom import *
from .state import *
from .ssobj import *

__all__ = geom.__all__ + state.__all__ + ssobj.__all__

# load up a few objects
Sun = getspiceobj('Sun', kernel='planets.bsp')
Mercury = getspiceobj('Mercury', kernel='planets.bsp')
Venus = getspiceobj('Venus', kernel='planets.bsp')
Earth = getspiceobj('Earth', kernel='planets.bsp')
Moon = getspiceobj('Moon', kernel='planets.bsp')
Mars = getspiceobj('Mars', kernel='planets.bsp')
Jupiter = getspiceobj('Jupiter', kernel='planets.bsp')
Saturn = getspiceobj('Saturn', kernel='planets.bsp')
Uranus = getspiceobj('Uranus', kernel='planets.bsp')
Neptune = getspiceobj('Neptune', kernel='planets.bsp')
Pluto = getspiceobj('Pluto', kernel='planets.bsp')
_loaded_objects = dict(sun=Sun, mercury=Mercury, venus=Venus, earth=Earth,
                       moon=Moon, mars=Mars, jupiter=Jupiter, saturn=Saturn,
                       uranus=Uranus, neptune=Neptune, pluto=Pluto)
__all__.extend(['Sun', 'Earth', 'Moon'])

# load 'em if you got 'em
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
    
# update module docstring
from ..util import autodoc
autodoc(globals())
del autodoc
