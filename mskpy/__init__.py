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
from . import asteroid

try:
    from . import ephem
    from .ephem import Sun, Earth

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
