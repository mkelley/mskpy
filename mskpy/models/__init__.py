# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
A model library
===============

.. autosummary::
   :toctree: generated/

   Surface Models
   --------------
   SurfaceRadiation
   DAp
   HG
   NEATM

   Dust Models
   -----------
   AfrhoRadiation
   AfrhoScattered
   AfrhoThermal

   Phase functions
   ---------------
   phaseK
   phaseH
   phaseHG
   phaseHM
   lambertian

"""

__all__ = [
    'surfaces',
    'dust'
]

from . import surfaces
from . import dust

from .surfaces import *
from .dust import *

# update module docstring
from ..util import autodoc
autodoc(globals())
del autodoc

