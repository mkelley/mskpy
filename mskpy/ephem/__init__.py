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
Sun = SolarSysObject(SpiceState('sun', kernel='planets.bsp'))
Mercury = SolarSysObject(SpiceState('mercury', kernel='planets.bsp'))
Venus = SolarSysObject(SpiceState('venus', kernel='planets.bsp'))
Earth = SolarSysObject(SpiceState('earth', kernel='planets.bsp'))
Moon = SolarSysObject(SpiceState('moon', kernel='planets.bsp'))
Mars = SolarSysObject(SpiceState('mars', kernel='planets.bsp'))
Jupiter = SolarSysObject(SpiceState('jupiter', kernel='planets.bsp'))
Saturn = SolarSysObject(SpiceState('saturn', kernel='planets.bsp'))
Uranus = SolarSysObject(SpiceState('uranus', kernel='planets.bsp'))
Neptune = SolarSysObject(SpiceState('neptune', kernel='planets.bsp'))
Pluto = SolarSysObject(SpiceState('pluto', kernel='planets.bsp'))
_loaded_objects = dict(sun=Sun, mercury=Mercury, venus=Venus, earth=Earth,
                       moon=Moon, mars=Mars, jupiter=Jupiter, saturn=Saturn,
                       uranus=Uranus, neptune=Neptune, pluto=Pluto)
__all__.extend(_loaded_objects.keys())

# load 'em if you got 'em
try:
    Spitzer = SolarSysObject(SpiceState('-79', kernel='spitzer.bsp'))
    _loaded_objects['spitzer'] = Spitzer
    __all__.append('Spitzer')
except OSError:
    pass

try:
    DeepImpact = SolarSysObject(SpiceState('-140', kernel='deepimpact.txt'))
    _loaded_objects['deepimpact'] = DeepImpact
    __all__.append('DeepImpact')
except OSError:
    pass
    
# update module docstring
from ..util import autodoc
autodoc(globals())
del autodoc
