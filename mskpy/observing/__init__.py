# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
observing --- Observing stuff
=============================

   Classes
   -------
   Target
   MovingTarget
   Observer


   Functions
   ---------
   am_plot
   file2targets

"""
from __future__ import print_function
import numpy as np
import astropy.units as u
from astropy.coordinates import Angle

from . import core
from .core import *

__all__ = core.__all__ + [
    'Observer',
    'Target',

    'am_plot',
    'file2targets'
    ]

class Target(object):
    """A target to be observed.

    Parameters
    ----------
    ra, dec : Angle
      The position of the target.
    label : string, optional
      The name of the target.

    """

    def __init__(self, ra, dec, label=None):
        from .. import util
        self.ra = ra
        self.dec = dec
        self.label = label

    def __repr__(self):
        opts = dict(sep=':', precision=0, pad=True)
        return '<Target ra={} dec={} label="{}">'.format(
            self.ra.to_string(**opts),
            self.dec.to_string(alwayssign=True, **opts), self.label)

class MovingTarget(Target):
    """A moving target to be observed.

    ..todo:: Merge with ephem.State?

    Parameters
    ----------
    ssobj : SolarSysObject
      The moving target.
    ltt : bool
      Set to `True` to account for light travel time.
    label : string, optional
      The name of the target.

    """

    def __init__(self, ssobj, ltt=False, label=None):
        self.target = ssobj
        self.ltt = ltt
        self.label = label

    def __repr__(self):
        return '<MovingTarget label={}>'.format(self.label)

    def ra(self, observer, date):
        return Angle(observer.observe(self.target, date, ltt=self.ltt)['ra'])

    def dec(self, observer, date):
        return Angle(observer.observe(self.target, date, ltt=self.ltt)['dec'])

class Observer(object):
    """An Earth-based observer.

    Parameters
    ----------
    lon, lat : Angle
      The longitude and (East) latitude of the observer.
    tz : float
      The time zone of the observer.
    date : string, float, astropy Time, datetime, or array
      The current date (civil time), passed to `util.date2time`.  If
      `None`, `date` will be set to now.

    Properties
    ----------
    lst : Angle
      Local sideral time.

    Methods
    -------
    airmass
    altaz
    lst
    lst0
    plot_am
    rts

    """
    from ..ephem import Earth

    def __init__(self, lon, lat, tz, date):
        from .. import util
        self.lon = lon
        self.lat = lat
        self.tz = tz
        if date is None:
            self.date = util.date2time(None)
            if isinstance(tz, float):
                self.date += tz * u.hr
            else:
                self.date += util.tz2utc(self.date, tz).total_seconds() * u.s
        else:
            self.date = util.date2time(date)

    @property
    def lst(self):
        """Local sidereal time."""
        return Angle(core.ct2lst(self.date, self.lon.degree, self.tz),
                     unit=u.hr)

    @property
    def lst0(self):
        """Local sidereal time at nearest midnight."""
        return Angle(core.ct2lst0(self.date, self.lon.degree, self.tz),
                     unit=u.hr)

    def _radec(self, target, date):
        if isinstance(target, MovingTarget):
            ra = target.ra(self.Earth, date)
            dec = target.dec(self.Earth, date)
        else:
            ra = target.ra
            dec = target.dec
        return ra, dec

    def airmass(self, target):
        """Target airmass.

        See `core.airmass` for method.

        Parameters
        ----------
        target : Target
          The target to observe.

        Returns
        -------
        am : float or array
          Target airmass.

        """

        ra, dec = self._radec(target, self.date)
        return core.airmass(ra.degree, dec.degree,
                            self.date, self.lon.degree, self.lat.degree,
                            self.tz)

    def altaz(self, target):
        """Altitude and azimuth of a target.

        Parameters
        ----------
        target : Target
          The target to observe.

        Returns
        -------
        alt, az : Angle
          Altitude and azimuth of the target.

        """

        ra, dec = self._radec(target, self.date)
        ha = self.lst - ra
        alt, az = core.hadec2altaz(ha.degree, dec.degree,
                                   self.lat.degree)
        return Angle(alt, unit=u.deg), Angle(az, unit=u.deg)

    def plot_am(self, target, N=100, ax=None, **kwargs):
        """Plot the airmass of this target, centered on the current date.

        Parameters
        ----------
        target : Target
          The target to observe.
        N : int
          Number of points to plot.
        ax : matplotlib.axes
          The axis to which to plot, or `None` for the current axis.
        **kwargs
          Any `matplotlib.pyplot.plot` keywords.

        Returns
        -------
        line : list
          Return value from `matplotlib.pyplot.plot`.

        """

        from datetime import timedelta
        import matplotlib.pyplot as plt
        from astropy.time import Time

        if ax is None:
            ax = plt.gca()
        label = kwargs.pop('label', target.label)

        # round to nearest day
        time = self.date.datetime.time()
        dt = time.hour * u.hr
        dt = dt + time.minute * u.min
        dt += time.second * u.second
        dt += time.microsecond * u.microsecond
        date = self.date - dt

        am = np.zeros(N)
        dt = np.linspace(-12, 12, N) * u.hr
        for i in range(N):
            ra, dec = self._radec(target, date + dt[i])
            am[i] = core.airmass(ra.degree, dec.degree,
                                 date + dt[i],
                                 self.lon.degree, self.lat.degree,
                                 self.tz)

        return ax.plot(dt.value, am, label=label, **kwargs)

    def rts(self, target, limit=20):
        """Rise, transit, set times for targets.

        Parameters
        ----------
        target : Target or array
          The target(s) to observe.
        limit : float
          The altitude at which the target is considered risen/set.

        Returns
        -------
        r, t, s : Quantity
          Rise, transit, set times.  `rise` and `set` will be `None`
          if the target never sets.  `transit` will be `None` if the
          target never rises.

        """

        from ..util import dh2hms

        if isinstance(target, (list, tuple)):
            times = ()
            for t in target:
                times += (self.rts(t, limit=limit), )
            return times

        ra, dec = self._radec(target, self.date)
        r, t, s = rts(ra.degree, dec.degree, self.date, self.lon.degree,
                      self.lat.degree, self.tz, limit)
        if t is not None:
            t = t * u.hr

        if r is not None:
            r = r * u.hr
            s = s * u.hr

        rr, tt, ss = None, None, None
        if r is not None:
            rr = dh2hms(r.value, '{:02d}:{:02d}')
        if t is not None:
            tt = dh2hms(t.value, '{:02d}:{:02d}')
        if s is not None:
            ss = dh2hms(s.value, '{:02d}:{:02d}')
        print("{:32s} {} {} {}".format(target.label, rr, tt, ss))

        return r, t, s

def am_plot(targets, observer, fig=None, **kwargs):
    """Generate a letter-sized, pretty airmass plot for a night.

    Parameters
    ----------
    targets : array of Target
      A list of targets to plot.
    observer : Observer
      The observer.
    fig : matplotlib Figure or None
      Figure: The matplotlib figure (number) to use.
      None: Use current figure.
    **kwargs
      Keyword arguments for `Observer.plot_am`.

    Returns
    -------
    None

    """

    import matplotlib.pyplot as plt
    from .. import graphics
    from ..util import dh2hms
    from .. import ephem

    if fig is None:
        fig = plt.gcf()
        fig.clear()
        fig = fig.set_size_inches(11, 8.5, forward=True)

    sun = MovingTarget(ephem.Sun, label='Sun')
    astro_twilight = MovingTarget(ephem.Sun, label='Astro twilight')
    civil_twilight = MovingTarget(ephem.Sun, label='Civil twilight')
    moon = MovingTarget(ephem.Moon, label='Moon')

    for target in targets:
        observer.plot_am(target, **kwargs)
        observer.rts(target)

    print()
    for target in [sun, moon]:
        observer.plot_am(target, **kwargs)
        observer.rts(target, limit=0)

    observer.rts(astro_twilight, limit=-18)
    observer.rts(civil_twilight, limit=-6)

    ax = plt.gca()
    plt.minorticks_on()
    plt.setp(ax, xlim=[-8, 8], ylim=[3, 1], yscale='log',
             ylabel='Airmass', xlabel='Time (CT)')
    plt.setp(ax, yticks=[3, 2.5, 2, 1.5, 1.2, 1.0],
             yticklabels=['3.0', '2.5', '2.0', '1.5', '1.2', '1.0'])

    xts = np.array(ax.get_xticks())
    if any(xts < 0):
        xts[xts < 0] += 24.0
    ax.set_xticklabels([dh2hms(t, '{:02d}:{:02d}') for t in xts])

    graphics.niceplot()
    graphics.nicelegend(frameon=False, loc='upper left')

def file2targets(filename):
    """Create a list of targets from a file.

    Parameters
    ----------
    filename : string
      The name of the file.

    Returns
    -------
    targets : list
      A list of the targets from the file.

    Notes
    -----

    File format: One object per line, consisting of an object label,
    right ascension and declination, separated by commas.  The angles
    can be in any string format acceptable to
    `astropy.coordinates.Angle`.  For example the following objects
    are equivalent:

      # this is a comment
      HD 106965,               12:17:57.5 hr, +01:34:31.1 deg
      HD 106965 [K=7.3],       12:17:57.5 hr, +01:34:31.1 deg
      HD 106965 [K=7.3],       12 17 57.5 hr, +01 34 31.1 deg

    Alternatively, you may request a moving target.  Specify the
    object's name and, optionally, kernel file in a set of double
    square brackets, e.g., [[object, kernel]].

      2P/Encke,                [[encke]]
      Jupiter,                 [[5, planets.bsp]]

    """

    import re
    from ..ephem import getspiceobj

    fixed = re.compile('(.+),\s*(.+),\s*(.+)')
    moving = re.compile('(.+),\s*\[\[([^,]+)(,\s*(.+))?]]')

    targets = []
    skipped = 0
    for line in open(filename, 'r').readlines():
        mmov = moving.findall(line)
        mfix = fixed.findall(line)

        if len(line.strip()) == 0:
            pass
        elif line.strip()[0] == '#':
            pass
        elif len(mmov) > 0:
            label, name, dummy, kernel = mmov[0]
            if len(kernel) == 0:
                kernel = None
            ssobj = getspiceobj(name, kernel=kernel)
            targets.append(MovingTarget(ssobj, label=label))
        elif len(mfix) > 0:
            label, ra, dec = mfix[0]
            targets.append(Target(Angle(ra), Angle(dec), label=label))
        else:
            skipped += 1

    if skipped > 0:
        print("Skipped {} possible targets".format(skipped))
    return targets

#class MovingObserver(Observer):
#    def __init__(self, obj):
#        self.observer = obj

mlof = Observer(Angle(-110.791667, u.deg), Angle(32.441667, u.deg), -7.0, None)

# update module docstring
from ..util import autodoc
autodoc(globals())
del autodoc
