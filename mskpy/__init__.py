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
   config      - mskpy configuration parameters.
   ephem       - Solar System object ephemerides (requires SpiceyPy).
   graphics    - Helper functions for making plots (requires matplotlib).
   image       - Image generators, analysis, and processing.
   instruments - Cameras, spectrometers, etc. for astronomy.
   modeling    - For fitting models to data.
   models      - Surface and dust models.
   observing   - Tools for observing preparations.
   photometry  - Tools for photometry.
   polarimetry - Classes and functions for polarimetry.
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

# depends on SpiceyPy
try:
    import spiceypy.wrapper as spice
    _spiceypy = True
except ImportError:
    _spiceypy = False
    raise UserWarning("MSKPY: spiceypy not available.  ephem, asteroid, and comet modules will not be loaded.")

if _spiceypy:
    from . import ephem
    from . import asteroid
    from . import comet

    from .ephem import *
    from .comet import *
    from .asteroid import *

# depends on matplotlib
try:
    import matplotlib
    _matplotlib = True
except ImportError:
    _matplotlib = False
    raise UserWarning("MSKPY: matplotlib not available.  Graphics module will not be loaded.")

if _matplotlib:
    from . import graphics
    from .graphics import *

del _spiceypy, _matplotlib
