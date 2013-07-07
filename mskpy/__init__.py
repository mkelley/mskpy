# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
mskpy --- MSK's personal library for astronomy and stuff
========================================================

.. autosummary::
   :toctree: generated/

   Modules
   -------
   asteroid - Defines an asteroid for observing, flux estimates.
   calib    - Photometric calibration.
   comet    - Defines a comet for observing, flux estimates.
   ephem    - Solar System object ephemerides (requires PySPICE).
   graphics - Helper functions for making plots.
   image    - Image generators, analysis, and processing.
   models   - Surface and dust models.
   util     - Grab bag of utility functions.

"""

from . import calib
from . import image
from . import models
from . import util

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
    from comet import *
    from asteroid import *

# depends on matplotlib
try:
    from . import graphics
except ImportError:
    pass
