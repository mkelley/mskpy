import warnings
import enum
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.table import Table
from astropy.io import ascii, fits
from astropy.wcs import WCS
from photutils.aperture import (
    RectangularAperture,
    CircularAperture,
    aperture_photometry,
)
from ..image.analysis import gcentroid, UnableToCenter


class Shape(enum.Enum):
    CIRCLE = "circle"
    SQUARE = "square"


class JWSTSpectrum:
    def __init__(self, wave, spec, unc, x=None, y=None):
        self.wave = wave
        self.spec = spec
        self.unc = unc
        self.x = x
        self.y = y

    def save(self, fn, meta, overwrite=True, **kwargs):
        data = (self.wave, self.spec, self.unc)
        names = ("wave", "spec", "unc")
        if self.x is not None:
            data += (self.x, self.y)
            names += ("x", "y")
        tab = Table(data, names=names)
        tab.meta.update(meta)
        tab.write(fn, overwrite=overwrite, **kwargs)

    @classmethod
    def read(cls, fn):
        tab = ascii.read(fn)
        data = cls(
            tab["wave"].quantity,
            tab["spec"].quantity,
            tab["unc"].quantity,
            tab["x"].data,
            tab["y"].data,
        )
        data.meta = tab.meta
        return data

    @classmethod
    def from_cube(
        cls,
        fn,
        x,
        y,
        shape="circle",
        size=3,
        unit=u.mJy,
        centroid_by_wavelength=False,
        centroid_window=5,
        centroid_options=None,
    ):
        """Simple spectral extraction from a data cube.


        Parameters
        ----------
        fn : str
            FITS file of spectral data cube.

        x, y : float
            Aperture center.

        shape : Shape or str, optional
            Circle or square.

        size : float, optional
            Circle radius or square side length.

        unit : astropy.units.Unit, optional
            Spectral data unit.

        centroid_by_wavelength : bool, optional
            If `True`, then refine the centroid for each wavelength.  If a
            centroid fails, the spectrum will be NaN for that wavelength.

        centroid_window : int, optional
            Median this number of wavelengths together before centroiding.  Best
            to use an odd value.

        centroid_options : dict
            Additional keyword arguments for `mskpy.image.gcentroid`.  Default:
            {"box": 5}.

        """

        _centroid_options = {"box": 5}
        if centroid_options is not None:
            _centroid_options.update(centroid_options)

        hdu = fits.open(fn)
        cube = hdu["SCI"].data
        cube_unc = hdu["ERR"].data
        N = cube.shape[0]

        shape = Shape(shape)

        if shape == Shape.CIRCLE:
            aper = CircularAperture((x, y), size)
        else:
            aper = RectangularAperture((x, y), size, size)

        wcs = WCS(hdu["SCI"])
        wave = wcs.all_pix2world(x, y, np.arange(N), 0)[2] * 1e6 * u.um
        omega = wcs.proj_plane_pixel_area()
        conv = (1 * u.MJy / u.sr * omega).to_value(unit, u.spectral_density(wave))
        if conv.size == 1:
            conv = conv * np.ones(N)

        spec = []
        unc = []
        cx = []
        cy = []
        for i in range(N):
            if not np.any(np.isfinite(cube[i])):
                spec.append(0)
                unc.append(0)
                cx.append(0)
                cy.append(0)
                continue

            if centroid_by_wavelength:
                hw = centroid_window // 2
                s = np.s_[max(0, i - hw) : min(N, i + hw + 1)]

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    im = np.nanmedian(cube[s], 0)

                try:
                    yx = gcentroid(im, aper.positions[::-1], **_centroid_options)
                except UnableToCenter:
                    spec.append(np.nan)
                    unc.append(np.nan)
                    cx.append(np.nan)
                    cy.append(np.nan)
                    continue
                aper.positions = yx[::-1]

            phot = aperture_photometry(
                cube[i] * conv[i],
                aper,
                error=cube_unc[i] * conv[i],
            )
            spec.append(phot["aperture_sum"][0])
            unc.append(phot["aperture_sum_err"][0])
            cx.append(aper.positions[0])
            cy.append(aper.positions[1])

        return cls(wave, spec * unit, unc * unit, cx, cy)

    def plot(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        ds = kwargs.get("drawstyle", kwargs.get("ds", "steps-mid"))
        ax.errorbar(self.wave, self.spec, self.unc, ds=ds, **kwargs)
