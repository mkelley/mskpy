2.0.0 (unreleased)
------------------

New Features
^^^^^^^^^^^^

- `ephem.SpiceObject` for your SPICE ephemeris needs, and a set of
  default objects, e.g., `Sun`, `Earth`, `Spitzer` (if the kernels are
  available).

- `Geom` is completely rewritten, and should be much more useful.

- `models` module, including `surfaces` and `dust`.

  - `NEATM` and `DAp` for thermal and reflected light from surfaces.

  - `AfrhoScattered` and `AfrhoThermal` for comet comae described with
    the Afrho parameter.

  - Various phase functions for dust and surfaces: `phaseHG`,
    `lambertian`, `phaseK`, `phaseH`, `phaseHM`.

- `Asteroid`, `Coma`, and `Comet` objects for easy estimates of their
  fluxes.  These objects package together `SpiceObject` and `models`.

- A few key functions are now `astropy` `Quantity` aware.  E.g.,
  `util.Planck`, `calib.solar_flux`.

- New time functions in `util`:

  - `cal2iso` to ISO format your lazy calendar dates.

  - `cal2doy` and `jd2doy` for time to day of year conversions.

  - `cal2time` and `jd2time` to lazily generate `astropy.time.Time`
    objects.

Changes From mskpy v1.7.0
^^^^^^^^^^^^^^^^^^^^^^^^^

- `math` renamed `util` and sorted:

  - `archav` and `Planck` return Quantities!

  - `nanmedian` now considers `inf` as a real value.

  - `numalpha` renamed `cmp_leading_num`.

  - `dminmax` renamed `mean2minmax`.

  - `numalpha` renamed `cmp_numalpha`.

  - `powerlaw` renamed `randpl`.

  - `pcurve` renamed `polcurve`

  - Added `projected_vector_angle` and `vector_rotate`.

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
    or `jd2time`.

  - `jd2dt` removed in favor of `jd2time`.

  - `dms2dd` renamed `hms2dh`.  Accepts `format`.

  - `doy2md` now requires year.

- `orbit.state2orbit` moved into `util`.

- `image` reorganized.  FITS and WCS functions moved to `util`.

  - `imshift` parameter sample renamed subsample for consistency with
    other functions.

  - Parameters named `center` renamed `yx` for clarity.

  - New `refine_center` to handle refining `rarray` and `tarray`
    subsampling.

  - `rarray` and `tarray` subsample parameters changed from bool to
    int so the exact subsampling factor can be specified.

Bug fixes
^^^^^^^^^

- `hms2dh` checks for rounding errors (e.g., 1000 ms, should be 1 s
  and 0 ms).
