# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
instruments --- Cameras, spectrometers, etc. for astronomy.
===========================================================

Instruments can be used observe a `SolarSysObject`.

   Classes
   -------
   Instrument
   Camera
   LongSlitSpectrometer

"""

from . import instrument
from . import midir

from instrument import *
from midir import *

# update module docstring
from ..util import autodoc
autodoc(globals())
del autodoc
