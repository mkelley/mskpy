# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
mskpy --- MSK's personal library for astronomy and stuff.
=========================================================

.. autosummary::
   :toctree: generated/

   Modules
   -------
   asteroid    - Defines an asteroid for observing, flux estimates.
   calib       - Photometric calibration.
   comet       - Defines a comet for observing, flux estimates.
   ephem       - Solar System object ephemerides (requires PySPICE).
   graphics    - Helper functions for making plots.
   image       - Image generators, analysis, and processing.
   instruments - Cameras, spectrometers, etc. for astronomy.
   modeling    - For fitting models to data.
   models      - Surface and dust models.
   observing   - Tools for observing preparations.
   util        - Grab bag of utility functions.

"""

from . import config

from . import image
from . import util

from .util import *
from .image import *

from . import calib
from . import instruments
from . import models
from . import modeling
from . import observing

# the following block depends on PySPICE
try:
    import spice
except ImportError:
    spice = None

if spice is not None:
    from . import ephem
    from . import asteroid
    from . import comet

    from .ephem import *
    from .comet import *
    from .asteroid import *
# end dependency on PySPICE

# depends on matplotlib
try:
    from . import graphics
    from .graphics import *
except ImportError:
    pass
# end matplotlib
