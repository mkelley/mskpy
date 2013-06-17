2.0.0 (unreleased)
------------------

New Features
^^^^^^^^^^^^

- Many functions are now `astropy` `Quantity` aware.  E.g.,
  `util.Planck`, `calib.solar_flux`.

- `ephem.MovingObject` for your ephemeris needs.

Changes From mskpy v1.7.0
^^^^^^^^^^^^^^^^^^^^^^^^^

- `math` renamed `util` and sorted.

  - `archav` and `Planck` return Quantities!

  - `nanmedian` now considers `inf` as a real value.

  - `numalpha` renamed `cmp_numalpha`.

  - `dminmax` renamed `mean2minmax`.

  - `numalpha` renamed `cmp_numalpha`.

  - `powerlaw` renamed `randpl`.

  - `pcurve` renamed `polcurve`

- `calib`

  - `cohenstandard` renamed `cohen_standard`.

  - `filtertrans` renamed `filter_trans`

  - `solarflux` renamed `solar_flux`

- `spice` renamed `ephem` and many functions removed.

- `time` functions moved into `util`.

  - `date2X` removed.

  - `dms2dd` renamed `hms2dh`.  Accepts `format`.

  - `doy2md` now requires year.

  ` New `s2doy`.

Bug fixes
^^^^^^^^^

- `hms2dh` checks for rounding errors (e.g., 1000 ms, should be 1 s
  and 0 ms).
