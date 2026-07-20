"""
This version v2a actually uses the source spectrum!

Mike Kelley
University of Maryland
2026 March 25

Licensed as part of mskpy with the BSD 3-Clause license.

"""

import os
from astropy.io import ascii
import synphot
import stpsf

spectrum = synphot.SourceSpectrum(synphot.BlackBody1D, 211)

tab = ascii.read("centers-v6.csv")

for row in tab:
    fn = "psf-v2a/" + row["file"]
    if os.path.exists(fn):
        continue

    dir = os.path.dirname(fn)
    if not os.path.exists(dir):
        os.mkdir(dir)

    miri = stpsf.setup_sim_to_match_file("data/" + row["file"])
    miri.detector_position = (row["x"], row["y"])

    psf = miri.calc_psf(oversample=8, source=spectrum, fov_pixels=201)
    psf.writeto(fn)
