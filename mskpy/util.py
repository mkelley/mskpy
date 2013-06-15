# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
#import matplotlib.pyplot as plt
#from astropy.io import fits
#from astropy.io import ascii
#from astropy.table import Table

"""
util --- Short and sweet functions, generic algorithms
======================================================

.. autosummary::
   :toctree: generated/

   Mathmatical
   -----------

   archav
   cartesian
   davint
   Gaussian
   Gaussian2d
   deriv
   hav
   rotmat

   Searching, sorting
   ------------------

   between
   groupby
   nearest
   numalpha
   whist

   Statistics
   ----------

   kuiper
   kuiperprob
   mean2minmax
   meanclip
   midstep
   minmax
   nanmedian
   nanminmax
   randpl
   wmean

   "Special" functions
   -------------------

   bandpass
   deresolve
   Planck
   redden
   pcurve
   savitzky_golay

"""

import numpy as np

__all__ = [
    'archav',
    'davint',
    'cartesian',
    'Gaussian',
    'Gaussian2d',
    'deriv',
    'hav',
    'rotmat',

    'between',
    'groupby',
    'nearest',
    'numalpha',
    'takefrom',
    'whist',

    'kuiper',
    'kuiperprob',
    'mean2minmax',
    'meanclip',
    'midstep',
    'minmax',
    'nanmedian',
    'nanminmax',
    'randpl',
    'uclip',
    'wmean',

    'bandpass',
    'deresolve',
    'Planck',
    'redden',
    'pcurve',
    'savitzky_golay'
]

#from . import davint.davint as _davint
