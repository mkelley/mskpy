# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
photometry --- Tools for working with photometry.
=================================================

.. autosummary::
   :toctree: generated/

   Modules
   -------
   hb - Hale-Bopp filter set calibration.

   Functions
   ---------
   airmass_app
   airmass_loc
   cal_airmass
   cal_color_airmass

"""

import numpy as np
import astropy.units as u

from .core import *
from .outbursts import *
from . import hb

__all__ = [
    'airmass_app',
    'airmass_loc',
    'cal_airmass',
    'cal_color_airmass',
    'hb'
]

# update module docstring
from ..util import autodoc
autodoc(globals())
del autodoc
