# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
mskpy --- MSK's personal library for astronomy and stuff
========================================================
"""

import calib
import util

try:
    import ephem
except ImportError:
    pass
