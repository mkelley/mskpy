2.0.0 (unreleased)
------------------

New Features
^^^^^^^^^^^^

- Many functions are now `astropy` `Quantity` aware.  E.g.,
  `util.Planck`, `calib.solar_flux`.

- `ephem.SpiceObject` for your SPICE ephemeris needs.

- New time functions:

  - `cal2iso` to ISO format your lazy calendar dates.

  - `cal2doy` and `jd2doy` for time to day of year conversions.

  - `cal2et`, `date2et`, `time2et`, and `date2time` helper functions
    within `ephem`.

Changes From mskpy v1.7.0
^^^^^^^^^^^^^^^^^^^^^^^^^

- `math` renamed `util` and sorted:

  - `archav` and `Planck` return Quantities!

  - `nanmedian` now considers `inf` as a real value.

  - `numalpha` renamed `cmp_numalpha`.

  - `dminmax` renamed `mean2minmax`.

  - `numalpha` renamed `cmp_numalpha`.

  - `powerlaw` renamed `randpl`.

  - `pcurve` renamed `polcurve`

  - `ec2eq` moved to `ephem`.

- `calib`:

  - `cohenstandard` renamed `cohen_standard`.

  - `filtertrans` renamed `filter_trans`

  - `solarflux` renamed `solar_flux`

- `spice` renamed `ephem`:

  - Removed `get_observer_xyz`, `get_planet_xyz`, `get_spitzer_xyz`,
    `get_herschel_xyz`, `get_comet_xyz`.

  - `getgeom` rewritten to use `MovingObject`.

  - Most of the `summarizegeom` code incorporated into `Geom`.

- `Geom`, `getgeom`, and `summarizegeom` moved from `observing` to
  `ephem`.

- `time` functions moved into `util`:

  - `date2X`, `jd2dt`, `s2dt`, `s2jd` removed in favor of `cal2time`,
    or `jd2time`.

  - `jd2dt` removed in favor of `jd2time`.

  - `dms2dd` renamed `hms2dh`.  Accepts `format`.

  - `doy2md` now requires year.

Bug fixes
^^^^^^^^^

- `hms2dh` checks for rounding errors (e.g., 1000 ms, should be 1 s
  and 0 ms).
