"""
This version for the nucleus.

Mike Kelley
University of Maryland
2026 April 8

Licensed as part of mskpy with the BSD 3-Clause license.

"""

import os
import numpy as np
from astropy.io import ascii
import astropy.units as u
import synphot
import stpsf
from mskpy.models.surfaces import neatm

wave = np.linspace(1, 6, 300) * u.um
flux = neatm(
    1 * u.km,
    0.05,
    {"rh": 2.37 * u.au, "delta": 1.798 * u.au, "phase": 22.7 * u.deg},
    wave,
    unit="mJy",
)

spectrum = synphot.SourceSpectrum(
    synphot.Empirical1D,
    points=wave,
    lookup_table=flux,
    keep_neg=True,
    meta={"name": "3I"},
)

tab = ascii.read("centers-v6.csv")

for row in tab:
    fn = "psf-neatm/" + row["file"]
    if os.path.exists(fn):
        continue

    dir = os.path.dirname(fn)
    if not os.path.exists(dir):
        os.mkdir(dir)

    miri = stpsf.setup_sim_to_match_file("data/" + row["file"])
    miri.detector_position = (row["x"], row["y"])

    psf = miri.calc_psf(oversample=8, source=spectrum, fov_pixels=201)
    psf.writeto(fn)
