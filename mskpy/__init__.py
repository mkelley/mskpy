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

try:
    from . import ephem
except ImportError:
    pass
