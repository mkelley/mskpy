2.1.8
-----

Bug fixes: `Geom` crash, `KeplerState` comet name from `SpiceState`,
scripts/ephemeris to use correct end date.

2.1.7
-----

- New `util.lb2xyz`.

- New `ephem.state.KeplerState`.

- Optimize `ephem.state.State` with `rv` method.


2.1.6
-----

- `util.date_len` bug fixes.

- `ephem.ssobj.getxyz` fix: wasn't running at all.

2.1.5
-----

- New `image.combine`, more efficient than `util.meanclip` for 2D
  arrays.

- Re-write `image.mkflat` to only do the normalization.

2.1.4
-----

- `image.gcentroid`

  - Fixes for new `astropy.modeling` API.

  - Contrain fits to within the box.

- Bug fix in `anphot` for single apertures.

2.1.3
-----

- `observing` airmass charts now use different line styles, as well as
  different colors.

2.1.2
-----

- Fix `observing` crash for JD to `astropy` `Time` conversion.

2.1.1
-----

Critical Fixes
^^^^^^^^^^^^^^

- `image.fwhm` renamed from `fwhmfit` and now actually respects the
  `bg` keyword.

New Features
^^^^^^^^^^^^

- `image.bgphot` function.


2.1.0
-----

New Features
^^^^^^^^^^^^

- New `observing` module, updated from `mskpy1`.


2.0.0
-----

Critical Fixes
^^^^^^^^^^^^^^

- Converting Afrho to thermal emission in `mskpy1` resulted in fluxes
  a factor of 4 too high (`comet.fluxest`).  This has been corrected
  by implementing an Afrho to efrho conversion factor (`ef2af`) in
  `dust.AfrhoThermal`.

New Features
^^^^^^^^^^^^

- New `ephem` module.

  - `SolarSysObject` for object ephemerides and, possibly, flux
    estimates.

  - `SpiceState` to retrieve positions and velocities from SPICE
    kernels.  `ephem` includes a set of default `SolarSysObject`s,
    e.g., `Sun`, `Earth`, `Spitzer` (if the kernels are available).

  - Use `getspiceobj` to easily create a `SolarSysObject` with a
    `SpiceState`.

- `comet` and `asteroid` modules define the `Asteroid`, `Coma`, and
  `Comet` `SolarSysObject`s for flux estimates of comets and
  asteroids.

- `Geom` is completely rewritten, and should be much more useful.

- `models` module, including `surfaces` and `dust`.

  - `NEATM`, `DAp`, and `HG` for thermal and reflected light from
    surfaces.

  - `AfrhoScattered` and `AfrhoThermal` for comet comae described with
    the Afrho parameter.

  - Various phase functions for dust and surfaces: `phaseHG`,
    `lambertian`, `phaseK`, `phaseH`, `phaseHM`.

- New `modeling` module (mirroring `astropy.modeling`) for fitting
  models to data.

- `Asteroid`, `Coma`, and `Comet` objects for easy estimates of their
  fluxes.  These objects package together `SpiceObject` and `models`.

- A few key functions are now `astropy` `Quantity` aware.  E.g.,
  `util.Planck`, `calib.solar_flux`.

- New time functions in `util`:

  - `cal2iso` to ISO format your lazy calendar dates.

  - `cal2doy` and `jd2doy` for time to day of year conversions.

  - `cal2time` and `jd2time` to lazily generate `astropy.time.Time`
    objects.

- New `instruments` module.  It can currently be used to estimate
  fluxes from comets and asteroids, but may have other uses in the
  future.  Includes `midir` sub-module with `MIRSI`, and `spitzer`
  sub-module with `IRAC`.

- New `scripts` directory for command-line scripts.  Currently
  includes an ephemeris generator.

Changes From mskpy v1.7.0
^^^^^^^^^^^^^^^^^^^^^^^^^

- `math` renamed `util` and sorted:

  - `archav` and `Planck` return Quantities!

  - `nanmedian` now considers `inf` as a real value.

  - `numalpha` replaced with `leading_num_key`.

  - `dminmax` renamed `mean2minmax`.

  - `powerlaw` renamed `randpl`.

  - `pcurve` renamed `polcurve`

  - Added `projected_vector_angle` and `vector_rotate`.

  - Rather than returning ndarrays, `takefrom` now returns lists,
    tuples, etc., based on the input arrays' type.

  - `spectral_density_sb` for `astropy.unit` surface brightness
    conversions.

  - `autodoc` to automatically update a module's docstring.

- `calib`:

  - `cohenstandard` renamed `cohen_standard`.

  - `filtertrans` renamed `filter_trans`

  - `solarflux` renamed `solar_flux`

- `spice` renamed `ephem`:

  - Removed `get_observer_xyz`, `get_planet_xyz`, `get_spitzer_xyz`,
    `get_herschel_xyz`, `get_comet_xyz`.

  - `getgeom` code incorporated into `Geom`.

  - `summarizegeom` code incorporated into `Geom`.

- `Geom`, `getgeom`, and `summarizegeom` moved from `observing` to
  `ephem`.

- `time` functions moved into `util`:

  - `date2X`, `jd2dt`, `s2dt`, `s2jd` removed in favor of `cal2time`,
    `jd2time`, or `date2time`.

  - `jd2dt` removed in favor of `jd2time`.

  - `dms2dd` renamed `hms2dh`.  Accepts `format`.

  - `doy2md` now requires year.

- `orbit.state2orbit` moved into `util`.

- `image` reorganized.  FITS and WCS functions moved to `util`.

  - `combine`, `imcombine`, `jailbar`, `phot`, `zarray` didn't make it.

  - Argument names made more consistent between all functions.  For
    example, `center` and `cen` renamed `yx`, `sample` renamed
    `subsample`.  Functions which previously took two coordinates, `y`
    and `x` now take one `yx`.

  - New `refine_center` to handle refining `rarray` and `tarray`
    subsampling.

  - `rarray` and `tarray` subsample parameters changed from bool to
    int so the exact subsampling factor can be specified.

  - Re-write `azavg` and `radprof` to use `anphot`.

  - New `gcentroid`.

  - `bgfit` arguments renamed.  Only 2D uncertainty maps are allowed.

  - `mkflat` re-written since `imcombine` was removed.

Bug fixes
^^^^^^^^^

- `hms2dh` checks for rounding errors (e.g., 1000 ms, should be 1 s
  and 0 ms).
