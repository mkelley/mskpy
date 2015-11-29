# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
instruments --- Cameras, spectrometers, etc. for astronomy.
===========================================================

Instruments can be used observe a `SolarSysObject`.

   Classes
   -------
   Instrument
   Camera
   CircularApertureSpectrometer
   LongSlitSpectrometer

"""

from . import instrument
from . import hst
from . import irtf
from . import spitzer
from . import vis

from .instrument import *
from .hst import *
from .irtf import *
from .spitzer import *
from .vis import *

# update module docstring
from ..util import autodoc
autodoc(globals())
del autodoc
