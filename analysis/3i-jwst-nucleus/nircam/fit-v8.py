"""
Mike Kelley
University of Maryland
2026 April 8

Licensed as part of mskpy with the BSD 3-Clause license.


v8:

Allow a different PSF to be used for coma and nucleus.

v7:

Updates based on tests/test-fit-v6.py

  * Don't smooth before calling wrap, which now has the smoothing code (not that
    it is a large affect anyway).

  * Don'r rely on fixpix, but use median filter for pixel replacement

  * slightly different approach to the inner pixel value (use the neighbor x 2),
    but not a big effect in the end

Also:

    * Apply pixel area map

v6:
  * subtract nucleus model before wedge fitting
  * bugfix: fit_wedge did not respect convolve parameter, but v5 behavior did
    convolve by default
  * do not limit convolution kernel use, but use full PSF size

"""

import os
import sys
import numpy as np
from numpy import pi
import scipy.ndimage as nd
from scipy.optimize import nnls
from scipy.interpolate import make_smoothing_spline, make_interp_spline
import matplotlib.pyplot as plt
from astropy.stats import sigma_clipped_stats
from astropy.io import fits, ascii
from astropy.table import Table
from astropy.convolution import convolve_fft
from mskpy import (
    fixpix,
    gcentroid,
    linefit,
    niceplot,
    radprof,
    rarray,
    rebin,
    subim,
    tarray,
    unwrap,
    wrap,
    yarray,
)


class ComaFit:
    def __init__(self, dest="results"):
        self.subsample = None
        self.kernel = {}
        self.dest = dest

    def __repr__(self):
        return "<ComaFit\n  subsample={subsamp}\n  dx={dx}\n  dy={dy}\n  rmin={rmin}\n  rmax={rmax}\n  theta_steps={nth}\n  mean slope={mean_slope}\n  nucleus={nucleus}>".format(
            subsamp=getattr(self, "subsample", "None"),
            dx=getattr(self, "dyx", ["None"] * 2)[1],
            dy=getattr(self, "dyx", ["None"] * 2)[0],
            rmin=getattr(self, "r_min", "None"),
            rmax=getattr(self, "r_max", "None"),
            nth=len(getattr(self, "theta_bins", [])),
            mean_slope=np.mean(getattr(self, "best_slopes", [0])),
            nucleus=getattr(self, "best_nucleus_scale", "None"),
        )

    def load_row(
        self,
        row,
        size,
        coma_psf_dir,
        nucleus_psf_dir,
        use_background=True,
        **kwargs,
    ):
        """From centers.csv table row."""
        self.load_data(
            row["file"],
            (row["y"], row["x"]),
            (row["dy"], row["dx"]),
            row["bg"] if use_background else 0,
            size,
            coma_psf_dir,
            nucleus_psf_dir,
            **kwargs,
        )

    def load_psfs(self, coma_fn, nucleus_fn, jitter=None):
        subsample_c, self.kernel["coma"] = self.get_psf(coma_fn, jitter=jitter)
        subsample_n, self.kernel["nucleus"] = self.get_psf(nucleus_fn, jitter=jitter)
        assert subsample_c == subsample_n
        self.subsample = subsample_c

    @staticmethod
    def get_psf(fn, jitter=None):
        with fits.open(fn) as hdul:
            # sampling factor
            det_scale = hdul["DET_DIST"].header["PIXELSCL"]
            over_scale = hdul["OVERDIST"].header["PIXELSCL"]
            subsample = int(det_scale / over_scale)
            psf = hdul["OVERDIST"].data + 0

        # smooth it
        if jitter is not None:
            if jitter.unit.is_equivalent("arcsec"):
                size = jitter.to_value("arcsec") / over_scale
            else:
                size = jitter.to_value("pix") * subsample

            psf = nd.gaussian_filter(psf, size)

        return subsample, psf[:-1, :-1]  # must be odd?

    def load_data(
        self,
        fn,
        gyx,
        dyx,
        bg,
        size,
        coma_psf_dir,
        nucleus_psf_dir,
        annulus=[14, 24],
        jitter=None,
    ):
        self.fn = fn
        self.gyx = gyx
        self.dyx = np.array(dyx)
        self.size = size
        self.shape = np.ones(2, int) * size
        self.annulus = annulus

        self.load_psfs(
            f"{coma_psf_dir}/{self.fn}", f"{nucleus_psf_dir}/{self.fn}", jitter=jitter
        )
        self.overshape = self.shape * self.subsample

        self.results_dir = self.dest + "/" + fn.split("/")[0]
        if not os.path.exists(self.results_dir):
            os.system("mkdir -p " + self.results_dir)

        with fits.open("data/" + self.fn) as hdul:
            comet = hdul["SCI"].data - bg
            comet_var = hdul["ERR"].data ** 2
            area = hdul["AREA"].data
            comet /= area
            comet_var /= area

        # interpolate over nans or else median_filter will spread them out
        i = ~np.isfinite(comet * comet_var)
        comet = fixpix(comet, i, max_area=100)
        # then replace them
        m = nd.median_filter(comet, 3)
        comet[i] = m[i]
        comet_var[i] = np.nanmax(comet_var) * 10

        h = self.size // 2
        comet = subim(comet, gyx, half_box=h)
        comet_var = subim(comet_var, gyx, half_box=h)

        # clean up some bright spots
        r = rarray(self.shape, (h, h))
        m = nd.median_filter(comet, 3)
        _, med, stdev = sigma_clipped_stats(comet - m)
        det = (m > (med + 3 * stdev)) * (r > 10)
        comet[det] = m[det]

        # save results to self
        self.comet = comet
        self.comet_var = comet_var
        self.cyx = np.round(gcentroid(comet, (h, h), box=5)).astype(int)

        # Model center at oversampled pixel scale.
        self.oyx = (
            np.round((self.cyx - 0.5) * self.subsample).astype(int)
            + self.subsample // 2
            + self.dyx
        )

        # Model center at native pixel scale.
        self.nyx = self.cyx + self.dyx / self.subsample

        h = fits.Header()
        h["gx"] = self.gyx[1]
        h["gy"] = self.gyx[0]
        h["cx"] = self.cyx[1]
        h["cy"] = self.cyx[0]
        h["dx"] = self.dyx[1]
        h["dy"] = self.dyx[0]
        h["x"] = self.nyx[1]
        h["y"] = self.nyx[0]
        fits.writeto(
            f"{self.results_dir}/comet.fits",
            self.comet,
            h,
            overwrite=True,
        )

    @property
    def oy(self):
        return self.oyx[0]

    @property
    def ox(self):
        return self.oyx[1]

    def generate_nucleus(self, convolve=False):
        """cyx is native pixels, dyx is shift in oversampled pixels"""
        nucleus = np.zeros(self.overshape)
        nucleus[self.oy, self.ox] = 1
        if convolve is not None:
            nucleus = convolve_fft(
                nucleus, self.kernel["nucleus"], normalize_kernel=False, allow_huge=True
            )
        return nucleus

    @staticmethod
    def generate_polar_image(f, shape):
        """Generate an image in polar coordinates given a function of theta.

        shape = polar coordinate image shape
        axis 0 is radial (0 to rmax)
        axis 1 is theta (0 to 360)

        """

        th = np.linspace(0, 360, shape[1] + 1)[:-1]
        r_max = shape[0]
        im = np.tile(f(th), r_max).reshape((r_max, len(th)))

        return im

    def generate_coma(self, slope, scale, convolve=False):
        """scales and slopes are functions of theta in degrees"""

        r_max = (
            int(
                round(
                    max(
                        np.hypot(self.oyx[0], self.oyx[1]),
                        np.hypot(
                            self.overshape[0] - self.oyx[0],
                            self.overshape[1] - self.oyx[1],
                        ),
                    )
                )
            )
            + 1
        )
        polar_shape = [r_max * self.subsample, 360 * self.subsample]
        slope_image = self.generate_polar_image(slope, polar_shape)
        scale_image = self.generate_polar_image(scale, polar_shape)
        r = yarray(polar_shape)
        r[0] = 1

        # normalize the polar coma at annulus
        polar_coma = scale_image * (r / np.mean(self.annulus)) ** slope_image
        polar_coma[0] = polar_coma[1] * 2

        rect_coma = wrap(polar_coma, self.oyx, self.overshape, smooth=True)

        if convolve:
            rect_coma = convolve_fft(
                rect_coma, self.kernel["coma"], normalize_kernel=False, allow_huge=True
            )

        return polar_coma, rect_coma

    @staticmethod
    def azimuthal_smoothing(im, yx, annulus):
        """Returns a smoothing spline as a function of theta.

        360 pixel circumference at r=57 pixels
        180 @ 28
        120 @ 19
        90 @ 14

        """

        uim = unwrap(im, yx, max(annulus), 180)
        th = np.linspace(0, 360, uim.shape[1] + 1)[:-1]

        az_profile = uim[min(annulus) :].mean(0)

        scale = make_smoothing_spline(th, az_profile, lam=1e4)

        return scale

    @staticmethod
    def interpolate_coma_parameters(theta, slopes, scales):
        """Generate interpolating splines for coma slope and scale.

        theta = azimuthal angles for the coma control points in degrees
        slopes, scales = values at each theta

        """

        slope = make_interp_spline(
            np.r_[theta, 360 + theta[0]], np.r_[slopes, slopes[0]], bc_type="periodic"
        )
        if scales is None:

            def scale(x):
                return np.ones_like(x)

        else:
            scale = make_interp_spline(
                np.r_[theta, 360 + theta[0]],
                np.r_[scales, scales[0]],
                bc_type="periodic",
            )

        return slope, scale

    def model_comet_images(
        self, theta, coma_slopes, coma_scales, nucleus_scale, convolve=True
    ):
        """Model comet image as a coma + nucleus.

        theta = azimuthal angles for the coma control points
        coma_slopes, coma_scales = coma control point values
        nucleus_scale = scale factor for nucleus
        shape = image shape in native pixels
        cyx = center pixel
        dyx = model offset from center pixel in units of subsampled pixels
        kernel is the kernel

        Returns = model images at subsampled resolution

        """

        slope, scale = self.interpolate_coma_parameters(theta, coma_slopes, coma_scales)
        coma = (
            self.generate_coma(slope, scale, convolve=convolve)[1] / self.subsample**2
        )
        nucleus = nucleus_scale * self.generate_nucleus(convolve=convolve)

        return coma, nucleus

    def fit_wedges(
        self,
        nucleus_scale,
        r_min,
        r_max,
        theta_bins,
        iterations=2,
        convolve=False,
        fixed=None,
    ):
        """Fit the radial profile from r_min to r_max in each theta bin.

        theta is between 0 and 360.

        First iteration fits the comet in each wedge between r_min and r_max.
        The annulus should not contain values below zero.

        Then a model comet is generated, optionally convolved with the PSF.

        The model is compared to the comet, and corrections for the slope and
        scale are fit.

        The nucleus is subtracted before fitting in both cases.

        """

        r = rarray(self.shape, self.nyx, subsample=2)
        t = np.degrees(tarray(self.shape, self.nyx))
        t[t < 0] += 360
        weights = np.sqrt(self.comet_var) / self.comet / np.log(10)

        # comet image with nucleus estimate removed, respecting convolve
        # parameter
        nucleus = rebin(
            self.generate_nucleus(convolve=convolve), -self.subsample, flux=True
        )
        comet_coma = self.comet - nucleus

        # first pass
        print(".", flush=True, end="")
        slopes1, scales1 = np.zeros((2, len(theta_bins) - 1))
        i = (r >= r_min) * (r <= r_max) * np.isfinite(comet_coma) * (comet_coma > 0)
        for step, (theta_min, theta_max) in enumerate(
            zip(theta_bins[:-1], theta_bins[1:])
        ):
            j = i * (t >= theta_min) * (t <= theta_max)
            if fixed:
                fit = [-1, np.log10(comet_coma[j] * r[j]).mean()]
            else:
                fit, _ = linefit(
                    np.log10(r[j]), np.log10(comet_coma[j]), weights[j], (-1, 10)
                )

            slopes1[step] = fit[0]
            scales1[step] = 10 ** fit[1]

        # remaining iterations: fit the ratio of comet to model coma for each
        # wedge to refine the slope and scale
        slopes = slopes1
        scales = scales1
        for iter in range(2, iterations + 1):
            print(".", flush=True, end="")
            # model coma
            theta = (theta_bins[:-1] + theta_bins[1:]) / 2
            coma, nucleus = self.model_comet_images(
                theta, slopes, scales, nucleus_scale, convolve=convolve
            )
            nucleus = rebin(nucleus, -self.subsample, flux=True)
            model = rebin(coma, -self.subsample, flux=True)

            ratio = (self.comet - nucleus) / model
            i = (r >= r_min) * (r <= r_max) * np.isfinite(ratio) * (ratio > 0)
            for step, (theta_min, theta_max) in enumerate(
                zip(theta_bins[:-1], theta_bins[1:])
            ):
                j = i * (t >= theta_min) * (t <= theta_max)
                if fixed:
                    fit = [-1, np.log10(ratio[j] * r[j]).mean()]
                else:
                    fit, _ = linefit(
                        np.log10(r[j]), np.log10(ratio[j]), weights[j], (-0.1, 1)
                    )

                slopes[step] += fit[0]
                scales[step] *= 10 ** fit[1]

        return slopes, scales

    def profiles(self, save=False):
        """azimuthally averaged radial profiles"""

        self.profile_bins = np.linspace(0, self.r_max, self.r_max + 1)
        rc, self.comet_profile, n, _ = radprof(self.comet, self.nyx, self.profile_bins)
        self.profile_centers = rc
        self.profile_count = n
        _, self.best_coma_profile, _, _ = radprof(
            self.best_coma, self.nyx, self.profile_bins
        )
        _, self.best_nucleus_profile, _, _ = radprof(
            self.best_nucleus, self.nyx, self.profile_bins
        )

        def inverse_rho(x):
            return -np.ones_like(x)

        _, im = self.generate_coma(inverse_rho, np.ones_like, convolve=True)
        self.nominal_coma = rebin(im, -self.subsample, flux=True)
        _, self.nominal_coma_profile, _, _ = radprof(
            self.nominal_coma, self.nyx, self.profile_bins
        )

        if not save:
            return

        tab = Table(
            (
                self.profile_bins[:-1],
                self.profile_bins[1:],
                self.profile_centers,
                self.nominal_coma_profile,
                self.comet_profile,
                self.best_coma_profile,
                self.best_nucleus_profile,
                self.comet_profile - self.best_coma_profile - self.best_nucleus_profile,
            ),
            names=(
                "left",
                "right",
                "center",
                "1/rho",
                "comet",
                "coma",
                "nucleus",
                "residuals",
            ),
        )
        tab.write(
            f"{self.results_dir}/radial-profiles.txt",
            format="ascii.fixed_width_two_line",
            overwrite=True,
        )

    def plot_profiles(self):
        # fig, ax = plt.subplots(num=1, clear=True)
        fig = plt.figure()
        tax, bax = fig.subplots(2, 1, height_ratios=[3, 1], sharex=True)

        c = self.comet_profile[3] / self.nominal_coma_profile[3]
        tax.plot(
            self.profile_centers,
            c * self.nominal_coma_profile,
            color="k",
            ls="--",
            ds="steps-mid",
            label="1/ρ",
        )
        tax.plot(
            self.profile_centers, self.comet_profile, ds="steps-mid", label="comet"
        )
        tax.plot(
            self.profile_centers,
            self.best_coma_profile,
            ds="steps-mid",
            label="best coma",
        )
        tax.plot(
            self.profile_centers,
            self.best_nucleus_profile,
            ds="steps-mid",
            label="best nucleus",
        )
        tax.plot(
            self.profile_centers,
            self.best_coma_profile + self.best_nucleus_profile,
            ds="steps-mid",
            label="coma+nucleus",
        )
        bax.plot(
            self.profile_centers,
            self.comet_profile - self.best_coma_profile - self.best_nucleus_profile,
            ds="steps-mid",
            label="residuals",
            color="tab:purple",
        )
        for ax in (tax, bax):
            for r in (self.r_min, self.r_max):
                ax.axvline(r, lw=1, color="k", ls=":")

        tax.legend()
        bax.legend()

        ylim = (
            10 ** np.floor(np.log10(min(self.comet_profile))),
            10 ** np.ceil(np.log10(max(self.comet_profile))),
        )
        plt.setp(
            tax,
            yscale="log",
            ylim=ylim,
            ylabel="Surface brightness (MJy/sr)",
            xticklabels=[],
        )
        plt.setp(
            bax,
            xscale="log",
            xlabel="$ρ$ (pix)",
        )
        niceplot()

    def save_images(self):
        h = fits.Header()
        h["gx"] = self.gyx[1]
        h["gy"] = self.gyx[0]
        h["cx"] = self.cyx[1]
        h["cy"] = self.cyx[0]
        h["dx"] = self.dyx[1]
        h["dy"] = self.dyx[0]
        h["x"] = self.nyx[1]
        h["y"] = self.nyx[0]
        h["r_min"] = self.r_min, "coma fit inner annulus radius"
        h["r_max"] = self.r_max, "coma fit outer annulus radius"

        h["mslope"] = np.mean(self.best_slopes), "mean coma slope"
        h["mscale"] = np.mean(self.best_scales), "mean coma scale"
        h["nucleus"] = self.best_nucleus_scale

        hdul = fits.HDUList()
        hdul.append(fits.PrimaryHDU(self.best_coma, h))
        hdul.append(fits.ImageHDU(self.best_slopes, name="slopes"))
        hdul.append(fits.ImageHDU(self.best_scales, name="scales"))
        hdul.append(fits.ImageHDU(theta_bins, name="thetabin"))
        hdul.writeto(f"{self.results_dir}/best-coma.fits", overwrite=True)

        fits.writeto(
            f"{self.results_dir}/best-nucleus.fits",
            self.best_nucleus,
            h,
            overwrite=True,
        )

        fits.writeto(
            f"{self.results_dir}/residuals.fits",
            self.comet - self.best_coma - self.best_nucleus,
            h,
            overwrite=True,
        )

    @staticmethod
    def plot_images(path, **kwargs):
        fig, axes = plt.subplots(2, 2, clear=True, figsize=(8, 8))
        axes = axes.ravel()

        im = {}
        for i, k in enumerate(["comet", "best-coma", "best-nucleus", "residuals"]):
            im[k] = fits.getdata(f"{path}/{k}.fits")

        axes[0].imshow(im["comet"], **kwargs)
        axes[1].imshow(im["best-coma"], **kwargs)
        axes[2].imshow(im["comet"] - im["best-coma"], **kwargs)
        axes[3].imshow(im["residuals"], **kwargs)

        plt.tight_layout(pad=0.2)

    def fit(self, r_min, r_max, theta_bins, iterations=3, fixed=None):
        r = rarray(self.shape, self.nyx, subsample=4)
        theta = (theta_bins[:-1] + theta_bins[1:]) / 2

        self.r_min = r_min
        self.r_max = r_max
        self.theta = theta
        self.theta_bins = theta_bins

        # fit and refit the coma and nucleus to the center
        nucleus = 1  # start small
        for _ in range(iterations):
            print(".", flush=True, end="")

            # first fit the wedges
            slopes, scales = self.fit_wedges(
                nucleus,
                r_min,
                r_max,
                theta_bins,
                fixed=fixed,
                iterations=iterations,
                convolve=True,
            )

            im_coma, im_nucleus = self.model_comet_images(
                self.theta, slopes, scales, nucleus
            )
            im_coma = rebin(im_coma, -self.subsample, flux=True)
            im_nucleus = rebin(im_nucleus, -self.subsample, flux=True)
            i = r < max(r_max, 4)
            model = [im_coma[i], im_nucleus[i]]

            model = np.array(model).T
            fit, rnorm = nnls(model, self.comet[i])

            scales *= fit[0]
            nucleus *= fit[1]

        print("/", end="", flush=True)
        self.best_slopes = slopes
        self.best_scales = scales
        self.best_nucleus_scale = nucleus
        im_coma, im_nucleus = self.model_comet_images(
            self.theta, slopes, scales, nucleus
        )
        self.best_coma = rebin(im_coma, -self.subsample, flux=True)
        self.best_nucleus = rebin(im_nucleus, -self.subsample, flux=True)

        print("/", end="", flush=True)
        self.profiles(save=True)
        self.plot_profiles()
        plt.savefig(f"{self.results_dir}/profiles.png", dpi=300)

        print("/", end="", flush=True)
        self.save_images()
        self.plot_images(self.results_dir)
        plt.savefig(f"{self.results_dir}/images.png", dpi=300)

        print()


version = os.path.splitext(os.path.basename(sys.argv[0]))[0].split("-")[-1]
coma_psfs = "psf-v6/"
nucleus_psfs = "psf-neatm/"

tab = ascii.read("centers-v7.csv")
if len(sys.argv) > 1:
    i = int(sys.argv[1])
    j = i + 1 if len(sys.argv) == 2 else int(sys.argv[2])
    rows = tab[i:j]
else:
    rows = tab

r_min = 1
theta_bins = np.linspace(0, 360, 20 + 1)
sfx = ""

for row in rows:
    plt.close("all")
    print(row["file"], end="")
    r_min = 1
    r_max = 50

    fitter = ComaFit(f"results-{version}-{r_min}_{r_max}-{len(theta_bins)}th{sfx}")
    fitter.load_row(
        row, 121, coma_psfs, nucleus_psfs if "blong" in row["file"] else coma_psfs
    )
    fitter.fit(r_min, r_max, theta_bins)

    print(fitter)
