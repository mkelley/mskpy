# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
mskpy --- MSK's personal library for astronomy and stuff
========================================================

.. autosummary::
   :toctree: generated/

"""

from . import calib
from . import util
from . import models

# the following block depends on PySPICE
try:
    from . import ephem
    from . import asteroid
    from . import comet

    from .ephem import Sun, Earth
    from comet import *
    from asteroid import *


    try:
        from .ephem import Spitzer
    except ImportError:
        pass

    try:
        from .ephem import DeepImpact
    except ImportError:
        pass

except ImportError:
    pass
