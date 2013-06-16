2.0.0 (unreleased)
------------------

New Features
^^^^^^^^^^^^

- Many functions are now `astropy` `Quantity` aware.  E.g.,
  `util.Planck`, `calib.solar_flux`, `calib.cohen_standard`.

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

