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
   plot_transit_time

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
    'file2targets',
    'plot_transit_time'
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
    finding_chart
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
            return '<Observer ({}): {}, {}, {}>'.format(
                self.name, self.lon.degree, self.lat.degree, self.date.iso[:10])
        else:
            return '<Observer: {}, {}, {}>'.format(
                self.lon.degree, self.lat.degree, self.date.iso[:10])


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

    def finding_chart(self, target, ds9, trange=[-6, 6] * u.hr, ticks=1 * u.hr,
                      fov=1 * u.arcmin, frame=1, dss=True):
        """Plot a DS9 finding chart for a moving target.

        Parameters
        ----------
        target : SolarSysObject
          The target to observe.
        ds9 : pysao.ds9
          Plot to this DS9 instance.
        trange : Quantity, optional
          Plot the target's path over this time span, centered on the
          observer's date (`self.date`).
        ticks : Quantity, optional
          Plot tick marks using this interval.  The first tick is a circle.
        fov : Quantity, optional
          Angular size of a box or rectangle to draw, indicating your
          instrument's FOV, or `None`.
        frame : int, optional
          DS9 frame number to display image.
        dss : bool, optional
          Set to `True` to retrieve a DSS image.

        """

        import matplotlib.pyplot as plt
        import pysao
        from ..ephem import Earth, SolarSysObject

        assert isinstance(target, SolarSysObject), "target must be a SolarSysObject"
        trange = u.Quantity(trange, u.hr)
        ticks = u.Quantity(ticks, u.hr)
        ds9 = ds9 if ds9 is not None else pysao.ds9()

        # DSS
        g = Earth.observe(target, self.date, ltt=True)
        ds9.set('frame {}'.format(frame))
        ds9.set('dsssao frame current')
        ds9.set('dsssao size 60 60')
        ds9.set('dsssao coord {} {}'.format(
            g['ra'].to_string(u.hr, sep=':'),
            g['dec'].to_string(u.deg, sep=':')))
        ds9.set('dsssao close')
        ds9.set('cmap b')
        ds9.set('align')
        
        # FOV
        if fov is not None:
            if fov.size == 1:
                fov = [fov, fov]
            fov_deg = u.Quantity(fov, u.deg).value
            reg = 'fk5; box {} {} {} {} 0'.format(
                g['ra'].to_string(u.hr, sep=':'),
                g['dec'].to_string(u.deg, sep=':'),
                fov_deg[0], fov_deg[1])
            ds9.set('regions', reg)

        # path
        dt = np.linspace(trange[0], trange[1], 31)
        g = Earth.observe(target, self.date + dt, ltt=True)
        for i in range(len(g) - 1):
            ds9.set('regions', 'fk5; line {} {} {} {}'.format(
                g[i]['ra'].to_string(u.hr, sep=':'),
                g[i]['dec'].to_string(u.deg, sep=':'),
                g[i+1]['ra'].to_string(u.hr, sep=':'),
                g[i+1]['dec'].to_string(u.deg, sep=':')))

        # ticks
        dt1 = np.arange(0, trange[0].value, -ticks.value)
        if dt1[-1] != trange[0].value:
            dt1 = np.concatenate((dt1, [trange[0].value]))
        dt2 = np.arange(ticks.value, trange[1].value, ticks.value)
        if dt2[-1] != trange[0].value:
            dt2 = np.concatenate((dt2, [trange[1].value]))
        dt = np.concatenate((dt1[::-1], dt2)) * u.hr
        del dt1, dt2
        g = Earth.observe(target, self.date + dt, ltt=True)
        for i in range(len(g)):
            s = 'fk5; point({},{}) # point=cross'.format(
                g[i]['ra'].to_string(u.hr, sep=':'),
                g[i]['dec'].to_string(u.deg, sep=':'))
            if i == 0:
                s = s.replace('cross', 'circle')
            ds9.set('regions', s)

        g = Earth.observe(target, self.date, ltt=True)
        ds9.set('regions', 'fk5; point({},{}) # point=x'.format(
            g['ra'].to_string(u.hr, sep=':'),
            g['dec'].to_string(u.deg, sep=':')))

        return ds9

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
    rts : Table
      A table of rise, transit, and set times.

    Notes
    -----
    To change the x-axis limits, use:
      `plt.setp(fig.axes, xlim=[min, max])`

    """

    import itertools
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    from astropy.table import Table
    from .. import graphics
    from ..util import dh2hms
    from .. import ephem

    linestyles = itertools.product(['-', ':', '--', '-.'], 'bgrcmyk')

    if fig is None:
        fig = plt.gcf()
        fig.clear()
        fig.set_size_inches(11, 8.5, forward=True)
        fig.subplots_adjust(left=0.06, right=0.94, bottom=0.1, top=0.9)

    ax = fig.gca()
    plt.minorticks_on()

    astro_twilight = ephem.getspiceobj('Sun', kernel='planets.bsp',
                                       name='Astro twilight')
    civil_twilight = ephem.getspiceobj('Sun', kernel='planets.bsp',
                                       name='Civil twilight')

    names, rise, transit, set_ = [], [], [], []
    for target in targets:
        names.append(target.name)
        ls, color = linestyles.next()
        observer.plot_am(target, color=color, ls=ls, **kwargs)
        rts = observer.rts(target, limit=25)
        rise.append(rts[0].value)
        transit.append(rts[1].value)
        set_.append(rts[2].value)

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

    plt.setp(ax, ylim=ylim, ylabel='Airmass',
             xlabel='Time (Civil Time)')

    # civil time labels
    def ctformatter(x, pos):
        return dh2hms(x % 24.0, '{:02d}:{:02d}')
    ax.xaxis.set_major_formatter(FuncFormatter(ctformatter))

    # LST labels
    def lstformatter(x, pos, lst0=observer.lst0.hour):
        return dh2hms(((x + lst0) % 24.0), '{:02d}:{:02d}')

    tax = plt.twiny()
    tax.xaxis.set_major_formatter(FuncFormatter(lstformatter))
    plt.minorticks_on()
    plt.setp(tax, xlabel='LST ' + str(observer))

    plt.sca(ax)
    graphics.niceplot(lw=1.6, tightlayout=False)

    fontprop = dict(family='monospace')
    graphics.nicelegend(frameon=False, loc='upper left', prop=fontprop)

    plt.draw()

    return Table([names, rise, transit, set_],
                 names=['target', 'rise', 'transit', 'set'])

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

def plot_transit_time(target, g_sun, observer=None, ax=None, **kwargs):
    """Plot the transit time of a target.

    Parameters
    ----------
    target : SolarSysObject
    g_sun : Geom
      The Sun's observing geometry for the dates of interest.
    observer : SolarSysObject
      The observer, or `None` for Earth.
    ax : matplotlib axis, optional
      The axis to which to plot, or `None` for current.
    **kwargs
      Any plot keyword.

    """

    import matplotlib.pyplot as plt
    from ..ephem import Earth

    if observer is None:
        observer = Earth

    if ax is None:
        ax = plt.gca()
    
    g = observer.observe(target, g_sun.date)
    tt = (g.ra - g_sun.ra - 12 * u.hourangle).wrap_at(180 * u.deg).hourangle

    cut = np.concatenate((np.abs(np.diff(tt)) > 12, [True]))
    i = 0
    name = target.name
    for j in np.flatnonzero(cut):
        line = ax.plot(tt[i:j], g.date[i:j].datetime, label=name,
                       **kwargs)[0]
        name = None
        i = j + 1

    # mark perihelion
    i = g.rh.argmin()
    if (i > 0) and (i < (len(g) - 1)):
        ax.plot([tt[i]], [g.date[i].datetime], '.', color=line.get_color())
        ax.annotate(' q={:.1f}'.format(g.rh[i].value),
                    [tt[i], g.date[i].datetime], color=line.get_color(),
                    fontsize=8)

    # pick 12 points and plot rh
    x = np.random.randint(0, len(g) / 10)
    for i in np.linspace(0 + x, len(g) - x - 1, 12).astype(int):
        ax.plot([tt[i]], [g.date[i].datetime], '.', color=line.get_color())
        ax.annotate(' {:.1f}'.format(g.rh[i].value),
                    [tt[i], g.date[i].datetime], color=line.get_color(),
                    fontsize=8)

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
               -10.0, None, name='MKO')
lapalma = Observer(Angle('-17 53 31', u.deg),
                   Angle('28 45 24', u.deg),
                   'Europe/London', None, name='La Palma')

# update module docstring
from ..util import autodoc
autodoc(globals())
del autodoc
