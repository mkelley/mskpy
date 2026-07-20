"""Use defaults for nlambda (which will be 9 for medium band, 20 for wide).
Size to 201 pixels.  Use the source spectrum.

Mike Kelley
University of Maryland
2026 April 8

Licensed as part of mskpy with the BSD 3-Clause license.

"""

import os
import numpy as np
from astropy.io import ascii, fits
import astropy.units as u
import stpsf
import synphot

spec = ascii.read("3i-jwst-nirspec-post2.4au-rap0.3-offset_+0.0+0.0-00001.ecsv")
spec = spec[np.isfinite(spec["spec"])]
wave = spec["wave"].quantity
flux = spec["spec"].quantity

spectrum = synphot.SourceSpectrum(
    synphot.Empirical1D,
    points=wave,
    lookup_table=flux,
    keep_neg=True,
    meta={"name": "3I"},
)

tab = ascii.read("centers-v7.csv")

for row in tab:
    fn = "psf-v6/" + row["file"]
    if os.path.exists(fn):
        continue

    dir = os.path.dirname(fn)
    if not os.path.exists(dir):
        os.mkdir(dir)

    nircam = stpsf.setup_sim_to_match_file("data/" + row["file"])
    nircam.detector_position = (row["x"], row["y"])

    psf = nircam.calc_psf(oversample=8, source=spectrum, fov_pixels=201)
    psf.writeto(fn)
