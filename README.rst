=====
mskpy
=====

MSK's personal Python library, mostly for astronomy work.

Requires: numpy, scipy, astropy (v0.3 development version).

Recommended: pyspice, matplotlib.


Caution
=======

I hope you find mskpy useful, but use at your own risk.

Configuration
=============

After installation, the file $HOME/.config/mskpy/mskpy.cfg should be
created.  If not simply execute ``python -c 'import mskpy.config'``.
This file currently contains paths to your SPICE kernels and Cohen
mid-IR standards (both are not required to run mskpy).


Examples
========

Solar System observing geometry
-------------------------------

Download 2P/Encke SPICE kernel from JPL HORIZONS; save as
'encke.bsp'::

  >>> from mskpy import getspiceobj
  >>> encke = getspiceobj('encke')
  >>> Earth.observe(encke, '2013-11-01').summary()
  
                               Date: 2013-11-01
                          Time (UT): 00:00:00
                         Julian day: 2456597.50
  
         Heliocentric distance (AU):    0.618
      Target-Observer distance (AU):    0.600
    Sun-Object-Observer angle (deg):  109.105
  
    Sun-Observer-Target angle (deg):   36.038
   Moon-Observer-Target angle (deg):   14.920
  
                            RA (hr):  12:35:04.3
                          Dec (deg): +09:20:03.7
  
         Projected sun vector (deg):  130.751
    Projected velocity vector (deg):  164.120

Which is pretty close to what JPL/HORIZONS reports (note the 15 arcsec
difference in the Equatorial coordinates)::

  rh:        0.618
  Delta:     0.600
  phase:     109.116
  s_elong:   36.041
  l_elong:   14.9
  RA:        12:35:04.1
  Dec:       09:20:20.0
  PsAng-180: 130.755
  PsAMV-180: 344.127


Ephemerides
-----------

  >>> from mskpy import Earth, Moon
  >>> print Moon.ephemeris(Earth, ['2013-1-1', '2013-12-31'], num=365)
        date         ra   dec     rh  delta phase selong
  ---------------- ----- ------ ----- ----- ----- ------
  2013-01-01 00:00 09:25  09:46 0.985 0.003    40    140
  2013-01-02 00:00 10:13  05:35 0.985 0.003    52    128
  2013-01-03 00:00 11:01  01:05 0.984 0.003    63    116
  2013-01-04 00:00 11:50  -3:33 0.984 0.003    75    104
  2013-01-05 00:00 12:40  -8:07 0.983 0.003    88     92
               ...   ...    ...   ...   ...   ...    ...
  2013-12-27 00:00 13:19  -9:42 0.983 0.003   107     73
  2013-12-28 00:00 14:12 -13:21 0.982 0.003   119     61
  2013-12-29 00:00 15:08 -16:23 0.982 0.002   132     48
  2013-12-30 00:00 16:08 -18:32 0.981 0.002   145     35
  2013-12-31 00:00 17:11 -19:29 0.981 0.002   159     21


Flux estimates
--------------

Asteroid
^^^^^^^^

Download (24) Themis SPICE kernel from JPL HORIZONS; save as
'2000024.bsp'::

  >>> import astropy.units as u
  >>> from mskpy import Asteroid, SpiceState, Earth
  >>> themis = Asteroid(SpiceState(2000024), 198 * u.km, 0.067, G=0.19, eta=1.0)
  >>> print themis.fluxd(Earth, '2013-10-15', [0.55, 3.0, 10] * u.um, unit=u.Jy)
  [ 0.03166609  0.01328637  6.19537744] Jy


Comet coma
^^^^^^^^^^

Download 2P/Encke SPICE kernel from JPL HORIZONS; save as 'encke.bsp'.
Download *Spitzer Space Telescope* kernel from JPL NAIF; save as
'spitzer.bsp'::

  >>> import astropy.units as u
  >>> from mskpy import Coma, SpiceState, Spitzer
  >>> Afrho1 = 8.9 * u.cm * 2.53**2
  >>> encke = Coma(SpiceState('encke'), Afrho1, ef2af=3.5, Tscale=1.1)
  >>> print encke.fluxd(Spitzer, '2004-06-20 18:35', 23.7 * u.um,
                        rap=12.5 * u.arcsec, unit=u.Jy)
  [ 0.02589534] Jy


Observing
---------

Airmass charts
^^^^^^^^^^^^^^

Create a file with your list of targets [#]_::

  Rubin 149 B,          07:24:18h, -00:33:06d
  C/2013 R1 (Lovejoy),    7 19 hr,   2 32 deg
  SA 101-316,           09h54m52s, -00d18m35s
  C/2012 S1 (ISON),     [[1003203]]

.. [#] In order for the last entry to work, the SPICE kernel for
       comet C/2012 S1 (ISON) must be downloaded and saved as
       '1003203.bsp' in your kernel directory.

Then, execute the following::

  >>> import astropy.units as u
  >>> from mskpy import observing
  >>> targets = observing.file2targets('targets.txt')
  >>> telescope = observing.Observer(-110.791667 * u.deg, 32.441667 * u.deg, -7, None)
  >>> observing.am_plot(targets, telescope)

.. image:: doc/images/am_plot.png

