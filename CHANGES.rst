2.2.4
-----

Critical fixes
^^^^^^^^^^^^^^

- `eph.State.v` for an array of dates returned `r`, now returns `v`.

New features
^^^^^^^^^^^^

- New `util.planckfit`.

- New `comet` functions:
  - `Q2flux` to convert gas production rates into fluxes.
  - `afrho2flux` to convert Afrho into flux density.
  - `m2qh2o` to convert absolute magnitude into water production rate,
    based on Jorda et al. (2008) relationship.
  - Renamed `m2afrho1` to `M2afrho1`.
  - New `m2afrho` to convert apparent magnitude into Afrho.  This is
    an EXPERIMENTAL relationship that WILL CHANGE.

- New `SolarSysObject.ephemeris` functionality:
  - Filter output given solar elongation limits.
  - Allow observers other than Earth.

- New `ephem.proper_motion`.

- New instrument: `BASS`.

Other improvements
^^^^^^^^^^^^^^^^^^

- `image.gcentroid` now ignores nans, infs.

- Fix time bug when milliseconds are passed to
  `SolarSysObject.ephemeris`.

- The ephemeris script in `scripts/` now displays help when no
  parameters are given.

2.2.3
-----

New features
^^^^^^^^^^^^

- `image.radprof` now returns centers of the radius bins, in addition
  to average of the radii within each bin.  This change breaks old
  code.

- New `instruments`:
  - Added `FLITECAM` to `sofia`.
  - Moved `MIRSI` to new `irtf`.
  - Added `SpeX` to `irtf`.

Critical fixes
^^^^^^^^^^^^^^

- Fix `SolarSysObject.lightcurve` call to `Column`.

- Fix `Asteroid` crashes due to missing name parameter and
  `astropy.time.Time`.

Other improvements
^^^^^^^^^^^^^^^^^^

- Modeling commented out and awaiting finalized astropy modeling API.


2.2.2
-----

New features
^^^^^^^^^^^^

- Maximum liftable grain radius: `models.dust.acrit`.

Critical fixes
^^^^^^^^^^^^^^

- Crash fixes:
  - `util.state2orbit`
  - `graphics.circle`

- Timezone (pytz) fixes for `Observer`.

Other improvements
^^^^^^^^^^^^^^^^^^

- `ephem`
  - Add mass to `SolarSysObject`.
  - Add masses to planets in `ephem`.

- `graphics`
  - Add `ax` keyword to `circle`.
  - Change default font size for `niceplot`.

- Add La Palma (`lapalma`) to `observing.

- `comets.Coma`
  - Initializes via `SolarSysObject` (still need to change other
    classes).
  - Improved `Afrho1` parameter checks.

- Update `astropy.units` usage in `instruments.spitzer.IRAC`.


2.2.1
-----

- Critical fix to meanclip: use higher precision float64 by default.

2.2.0
-----

- New `polarimetry` module.
- Removed `graphics.ds9`.  The XPA interface in `pyds9` isn't working
  well on my setup.


2.1.0 to 2.1.14
---------------

New features
^^^^^^^^^^^^

- `catalogs`, currently limited to spatially matching lists of sources
  together.
- `graphics.ds9`: if pyds9 is installed, `graphics.ds9` is a class
  with a `view` method for more lazy display calls.
- `observing` module, updated from `mskpy1`.
- `image`
  - `combine`, more efficient than `util.meanclip` for 2D arrays.
  - `bgphot` for background photometry.
- Instruments: `hst.wfc3uvis`, `vis.OptiPol`.
- `util`
  - `linefit` for simple line fitting with uncertainties.
  - `timestamp` string generator.
- New `util.lb2xyz`.
- New `ephem.state.KeplerState`.
  - `KeplerState` gets comet name from `SpiceState`.

Critical fixes
^^^^^^^^^^^^^^

- `image`
  - Fix `linecut` fatal crash.
  - Fix `crclean` fatal crash.  I'm not sure algorithm is working
    properly, though.
  - `fwhm` renamed from `fwhmfit` and now actually respects the `bg`
    keyword.
  - Bug fix in `anphot` for single apertures.
- `ephem`
  - `Geom` crash fix.
  - `ssobj.getxyz` fix: wasn't running at all.
- scripts/ephemeris now uses correct end date.
- `util`
  - Fix `gaussian` crash.
  - Fix `hms2dh` crash given any input.
  - `date_len` bug fixes.

Other improvements
^^^^^^^^^^^^^^^^^^

- `graphics`
  - Fix exception handling (e.g., when `matplotlib` does not exist)
    during `graphics` importing.
  - `nicelegend` now handles font properties via `prop` keyword.
- Fix `spitzer.irac.ps` units.
- `image`
  - Let `stack2grid` work for any number of images.
  - `gcentroid`:
    - Uses `scipy.optimize`.
    - Contrain fits to within the box.
  - Re-write `mkflat` to only do the normalization.
- `ephem`:
  - Fix some planet NAIF IDs.
  - Optimize `state.State` with `rv` method.


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
