# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
observing --- Observing stuff
=============================

   Classes
   -------
   Target
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
    name : string, optional
      The name of the target.

    """

    def __init__(self, ra, dec, name=None):
        from .. import util
        self.ra = ra
        self.dec = dec
        self.name = name

    def __repr__(self):
        opts = dict(sep=':', precision=0, pad=True)
        return '<Target ra={} dec={} name="{}">'.format(
            self.ra.to_string(**opts),
            self.dec.to_string(alwayssign=True, **opts), self.name)

class Observer(object):
    """An Earth-based observer.

    Parameters
    ----------
    lon, lat : astropy Angle or Quantity
      The longitude and (East) latitude of the observer.
    tz : float
      The time zone of the observer.
    date : string, float, astropy Time, datetime, or array
      The current date (civil time), passed to `util.date2time`.  If
      `None`, `date` will be set to now.
    name : string
      The name of the observer/observatory.

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

    def __init__(self, lon, lat, tz, date, name=None):
        from .. import util
        self.lon = Angle(lon)
        self.lat = Angle(lat)
        self.tz = tz
        if date is None:
            self.date = util.date2time(None)
        else:
            self.date = util.date2time(date)
        self.name = name

    @property
    def date(self):
        """Observation date"""
        return self._date

    @date.setter
    def date(self, d):
        """Observation date"""
        from .. import util
        self._date = util.date2time(d)
        if isinstance(self.tz, float):
            self._date += self.tz * u.hr
        else:
            self._date += util.tz2utc(self.date, self.tz).total_seconds() * u.s

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

    def __repr__(self):
        if self.name is not None:
            return '<Observer ({}): {}, {}>'.format(
                self.name, self.lon.degree, self.lat.degree)
        else:
            return '<Observer: {}, {}>'.format(
                self.lon.degree, self.lat.degree)


    def _radec(self, target, date):
        from ..ephem import Earth, SolarSysObject
        if isinstance(target, SolarSysObject):
            g = Earth.observe(target, date, ltt=True)
            ra = g['ra']
            dec = g['dec']
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
        from ..util import jd2time

        if ax is None:
            ax = plt.gca()
        label = kwargs.pop('label', target.name)

        # round to nearest day
        date = jd2time(round(self.date.jd - 0.5) + 0.5)

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
        print("{:32s} {} {} {}".format(target.name, rr, tt, ss))

        return r, t, s

def am_plot(targets, observer, fig=None, ylim=[2.5, 1], **kwargs):
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
    ylim : array
      Y-axis limits (airmass).
    **kwargs
      Keyword arguments for `Observer.plot_am`.

    Returns
    -------
    None

    """

    import itertools
    import matplotlib.pyplot as plt
    from .. import graphics
    from ..util import dh2hms
    from .. import ephem

    linestyles = itertools.product(['-', ':', '--', '-.'], 'bgrcmyk')

    if fig is None:
        fig = plt.gcf()
        fig.clear()
        fig.set_size_inches(11, 8.5, forward=True)
        fig.subplots_adjust(left=0.06, right=0.94, bottom=0.1, top=0.9)

    ax = fig1.gca()
    plt.minorticks_on()

    astro_twilight = ephem.getspiceobj('Sun', kernel='planets.bsp',
                                       name='Astro twilight')
    civil_twilight = ephem.getspiceobj('Sun', kernel='planets.bsp',
                                       name='Civil twilight')

    for target in targets:
        ls, color = linestyles.next()
        observer.plot_am(target, color=color, ls=ls, **kwargs)
        observer.rts(target, limit=25)

    print()
    for target, ls in zip((ephem.Sun, ephem.Moon), ('y--', 'k:')):
        observer.plot_am(target, color=ls[0], ls=ls[1:], **kwargs)
        observer.rts(target, limit=0)

    at_rts = observer.rts(astro_twilight, limit=-18)
    ct_rts = observer.rts(civil_twilight, limit=-6)

    y = [ylim[0], ylim[0], ylim[1], ylim[1]]
    c = (0.29, 0.64, 1.0, 0.1)
    for twilight in (at_rts, ct_rts):
        x = [-12, twilight[2].value - 24, twilight[2].value - 24, -12]
        ax.fill(x, y, color=c)

        x = [12, twilight[0].value, twilight[0].value, 12]
        ax.fill(x, y, color=c)

    plt.setp(ax, xlim=[-8, 8], ylim=ylim, ylabel='Airmass',
             xlabel='Time (CT)')

    # civil time labels
    xts = np.array(ax.get_xticks())
    if any(xts < 0):
        xts[xts < 0] += 24.0
    ax.set_xticklabels([dh2hms(t, '{:02d}:{:02d}') for t in xts])

    # LST labels
    xts = np.array(ax.get_xticks()) + observer.lst0.hour
    if any(xts < 0):
        xts[xts < 0] += 24.0
    tax = ax.twiny()
    tax.set_xticklabels([dh2hms(t, '{:02d}:{:02d}') for t in xts])
    plt.minorticks_on()
    plt.setp(tax, xlim=ax.get_xlim(), xlabel='LST')

    plt.sca(ax)
    graphics.niceplot(lw=1.6, tightlayout=False)
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
            name, naifname, dummy, kernel = mmov[0]
            if len(kernel) == 0:
                kernel = None
            targets.append(getspiceobj(naifname, kernel=kernel, name=name))
        elif len(mfix) > 0:
            name, ra, dec = mfix[0]
            targets.append(Target(Angle(ra), Angle(dec), name=name))
        else:
            skipped += 1

    if skipped > 0:
        print("Skipped {} possible targets".format(skipped))
    return targets

#class MovingObserver(Observer):
#    def __init__(self, obj):
#        self.observer = obj

mlof = Observer(Angle(-110.791667, u.deg),
                Angle(32.441667, u.deg),
                -7.0, None, name='MLOF')
lowell = Observer(Angle(-111.5358, u.deg),
                  Angle(35.0969, u.deg),
                  -7.0, None, name='Lowell')
mko = Observer(Angle('-155 28 19', u.deg),
               Angle('19 49 34', u.deg),
               -11.0, None, name='MKO')

# update module docstring
from ..util import autodoc
autodoc(globals())
del autodoc
