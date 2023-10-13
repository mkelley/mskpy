from typing import NamedTuple
import enum
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.table import Table
from astropy.io import ascii, fits
from astropy.wcs import WCS
from photutils import RectangularAperture, CircularAperture, aperture_photometry


class Shape(enum.Enum):
    CIRCLE = "circle"
    SQUARE = "square"


class JWSTSpectrum(NamedTuple):
    wave: u.Quantity
    spec: u.Quantity
    unc: u.Quantity

    def save(self, fn, meta, overwrite=True, **kwargs):
        tab = Table(
            (self.wave, self.spec, self.unc), names=["wave", "spec", "unc"]
        )
        tab.meta.update(meta)
        tab.write(fn, overwrite=overwrite, **kwargs)

    @classmethod
    def read(cls, fn):
        tab = ascii.read(fn)
        data = cls(
            tab["wave"].quantity, tab["spec"].quantity, tab["unc"].quantity
        )
        data.meta = tab.meta
        return data

    @classmethod
    def from_cube(cls, fn, x, y, shape="circle", size=3, unit=u.mJy):
        """Simple spectral extraction from a data cube.


        Parameters
        ----------
        fn : str
            FITS file of spectral data cube.

        x, y : float
            Aperture center.

        shape : Shape or str
            Circle or square.

        size : int
            Circle radius or square side length.

        unit : astropy.units.Unit
            Spectral data unit.

        """

        hdu = fits.open(fn)
        N = hdu["SCI"].data.shape[0]

        shape = Shape(shape)

        if shape == Shape.CIRCLE:
            aper = CircularAperture((x, y), size)
        else:
            aper = RectangularAperture((x, y), size, size)

        wcs = WCS(hdu["SCI"])
        wave = wcs.all_pix2world(x, y, np.arange(N), 0)[2] * 1e6 * u.um
        omega = wcs.proj_plane_pixel_area()
        conv = (1 * u.MJy / u.sr * omega).to(
            unit, u.spectral_density(wave)
        )
        if conv.size == 1:
            conv = conv * np.ones(N)

        spec = []
        unc = []
        for i in range(N):
            if not np.any(np.isfinite(hdu["SCI"].data)):
                spec.append(0)
                unc.append(0)
                continue

            phot = aperture_photometry(
                hdu["SCI"].data[i] * conv[i],
                aper,
                error=hdu["ERR"].data[i] * conv[i],
            )
            spec.append(phot["aperture_sum"][0])
            unc.append(phot["aperture_sum_err"][0])

        return cls(wave, spec * unit, unc * unit)

    def plot(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        ds = kwargs.get("drawstyle", kwargs.get("ds", "steps-mid"))
        ax.errorbar(self.wave, self.spec, self.unc, ds=ds, **kwargs)
