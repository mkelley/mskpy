"""Send data to DS9."""

import os
import uuid
from tempfile import mkdtemp

import numpy as np
from astropy.io.fits.hdu.base import DTYPE2BITPIX

import astropy_samp_ds9.launcher


class DS9(astropy_samp_ds9.launcher.DS9):
    def view(self, data):
        if data.ndim not in [2, 3]:
            raise ValueError("Requires 2 or 3 dimensional data.")

        filename = os.path.join(mkdtemp(), uuid.uuid4().hex + ".fits")
        fp = np.memmap(filename, dtype=data.dtype, mode="w+", shape=data.shape)
        fp[:] = data[:]
        fp.flush()

        if data.ndim == 2:
            dim = f"xdim={data.shape[1]},ydim={data.shape[0]}"
        if data.ndim == 3:
            dim = f"xdim={data.shape[2]},ydim={data.shape[1]},zdim={data.shape[0]}"

        bitpix = DTYPE2BITPIX[str(data.dtype)]

        self.set("array " + filename + f"[{dim},bitpix={bitpix}]")
