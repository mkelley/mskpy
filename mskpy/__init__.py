# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
mskpy --- MSK's personal library for astronomy and stuff
========================================================
"""

for module in 'calib util'.split():
    eval('import ' + module)

try:
    import ephem
except ImportError:
    pass
