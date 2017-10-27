# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
ssobj --- Solar System objects.
===============================

   Classes
   -------
   SolarSysObject

   Functions
   ---------
   getgeom
   getspiceobj
   getxyz
   summarizegeom

"""

from __future__ import print_function

import numpy as np
import astropy.units as u
from astropy.time import Time

from . import core

__all__ = [
    'SolarSysObject',
    'getgeom',
    'getspiceobj',
    'getxyz',
    'summarizegeom',
]

class SolarSysObject(object):
    """A star, planet, comet, etc. in the Solar System.

    Parameters
    ----------
    state : State
      The object from which to retrieve positions and velocities.
    M : float or Quantity, optional
      Mass of the object.  [float: kg]
    GM : float or Quantity, optional
      Gravitational constant times the mass of the object. [float:
      km**3/s**2]
    name : string, optional
      The name of the object.

    Methods
    -------
    ephemeris : Ephemeris for an observer.
    fluxd : Total flux density as seen by an observer.
    lightcurve : An ephemeris table with fluxes.
    observe : Distance, phase angle, etc. to another object.
    orbit : Osculating orbital parameters at date.
    r : Position vector
    v : Velocity vector

    Notes
    -----
    Inheriting classes should override `fluxd`.

    """

    def __init__(self, state, M=None, GM=None, name=None):
        from .state import State
        assert isinstance(state, State)
        self.state = state
        self.name = name

        if M is None and GM is None:
            self.M = 0
        elif M is not None:
            self.M = M
        else:
            self.GM = GM

    @property
    def M(self):
        """Mass.  [kg]"""
        return self._GM / 6.67384e-20

    @M.setter
    def M(self, m):
        m = u.Quantity(m, u.kg)
        self._GM = 6.67384e-20 * m.value

    @property
    def GM(self):
        """Gravitational constant times mass. [kg**3 / s**2]"""
        return self._GM

    @GM.setter
    def GM(self, gm):
        gm = u.Quantity(gm, u.km**3 / u.s**2)
        self._GM = gm.value

    def __repr__(self):
        return '<SolarSysObject name="{}" M="{:.3g} kg">'.format(
            self.name, self.M)

    def r(self, date):
        """Position vector.

        Parameters
        ----------
        date : string, float, astropy Time, datetime, or array
          Processed via `util.date2time`.

        Returns
        -------
        r : ndarray
          Position vector (3-element or Nx3 element array). [km]
       
        """
        return self.state.r(date)

    def v(self, date):
        """Velocity vector.

        Parameters
        ----------
        date : string, float, astropy Time, datetime, or array
          Processed via `util.date2time`.

        Returns
        -------
        v : ndarray
          Velocity vector (3-element or Nx3 element array). [km/s]
       
        """
        return self.state.v(date)

    def rv(self, date):
        """Position and velocity vectors.

        Parameters
        ----------
        date : string, float, astropy Time, datetime, or array
          Processed via `util.date2time`.

        Returns
        -------
        r : ndarray
          Position vector (3-element or Nx3 element array). [km]
        v : ndarray
          Velocity vector (3-element or Nx3 element array). [km/s]

        """
        return self.state.rv(date)

    def ephemeris(self, observer, dates, num=None, columns=None,
                  cformats=None, ra_unit='hourangle', date_format=None,
                  ltt=False, **kwargs):
        """Ephemeris for an observer.

        Parameters
        ----------
        observer : SolarSysObject
          The observer.
        dates : array
          If `num` is `None`, set `dates` to a list of exact times for
          the ephemeris, otherwise, let `dates` be a start and stop
          time, and `num` be the number of dates to generate.  `dates`
          may be in any format that `observe` accepts.
        num : int, optional
          If not `None`, generate this many time steps between
          `min(dates)` and `max(dates)`.
        columns : array, optional
          A list of `Geom` keywords to use as table columns, or `None`
          for the default list.
        cformats : dict or list, optional
          A dictionary of formats with keys corresponding to
          `columns`.
        ra_unit : str or astropy Unit, optional
          The unit of Right Ascention output, e.g., hourangle or deg.
        date_format : function
          A function to format the `date` column before creating the
          table.
        ltt : bool, optional
          Set to `True` to account for light travel time.

        Returns
        -------
        eph : astropy Table

        Notes
        -----
        `date_format` should be removed when astropy `Table` can
        handle Time objects.

        Override `_add_lc_columns` to add additional columns to
        lightcurve output.

        """

        from astropy.table import Table, Column
        from astropy.units import Quantity
        from ..util import date2time, dh2hms

        dates = date2time(dates)
        if num is not None:
            if num <= 0:
                time = []
            elif num == 1:
                time = [dates[0], dates[-1]]
            else:
                step = (dates[-1] - dates[0]) / float(num - 1)
                time = []
                for i in range(num):
                    time += [dates[0] + step * i]
        else:
            time = dates

        g = observer.observe(self, time, ltt=ltt)

        if columns is None:
            columns = ['date', 'ra', 'dec', 'rh', 'delta', 'phase', 'selong']

        if cformats is None:
            cformats = dict()

        _cformats = dict(
            date = '{:s}',
            ra = lambda x: dh2hms(x, "{:2d}:{:02d}"),
            dec = lambda x: dh2hms(x, "{:2d}:{:02d}"),
            lam = '{:.0f}',
            bet = '{:+.0f}',
            rh = '{:.3f}',
            delta = '{:.3f}',
            phase = '{:.0f}',
            selong = '{:.0f}',
            lelong = '{:.0f}')

        for k, v in cformats:
            _cformats[k] = v

        if date_format is None:
            date_format = lambda d: d.iso[:-7]

        eph = Table()
        eph.meta['ltt'] = ltt
        #eph.meta['observer'] = str(self)
        #eph.meta['target'] = str(target)

        for c in columns:
            if c == 'date':
                data = [date_format(d) for d in g[c]]
            elif c == 'ra':
                data = g[c].to(ra_unit)
            else:
                data = g[c]

            if c in _cformats:
                cf = _cformats[c]
            else:
                cf = None

            eph.add_column(Column(data=data, name=c, format=cf))

        return eph

    def fluxd(self, observer, date, wave, ltt=False,
              unit=u.Unit('W / (m2 um)'), **kwargs):
        """Total flux density as seen by an observer.

        Parameters
        ----------
        observer : SolarSysObject
          The observer.
        date : string, float, astropy Time, datetime
          The time of the observation in any format acceptable to
          `observer`.
        wave : Quantity
          The wavelengths at which to compute `fluxd`.
        ltt : bool, optional
          Set to `True` to correct the object's position for light
          travel time.
        unit : astropy Unit
          The return unit, must be flux density.
        
        Returns
        -------
        fluxd : Quantity

        """
        raise NotImplemented('This class has not implemented fluxd.')

    def lightcurve(self, observer, dates, wave, wformat='f{:.1f}',
                   verbose=True, **kwargs):
        """An ephemeris table with fluxes.

        Parameters
        ----------
        observer : SolarSysObject
          The observer.
        dates : string, float, astropy Time, datetime
          The dates of the observation in any format acceptable to
          `SolarSysObservable.ephemeris`.
        wave : Quantity
          The wavelengths at which to compute `fluxd`.
        wformat : string, optional
          The flux density columns will are labed with their
          wavelengths using this format string.
        verbose : bool, optional
          If `True`, give some visual feedback.
        **kwargs
          Additional `fluxd` and `ephemeris` keywords.

        Returns
        -------
        lc : astropy Table

        Notes
        -----
        `date` must be in the table returned by `ephemeris`.

        The `cformat` keyword `fluxd` will be used to format all flux
        density columns.

        """

        from astropy.table import Column

        lc = self.ephemeris(observer, dates, **kwargs)
        if 'date' not in lc.columns:
            raise KeyError("Ephemeris must return a date column."
                           "  Update columns.")

        if not np.iterable(wave):
            wave = [wave.value] * wave.unit

        if verbose:
            print('Computing date:')

        fluxd = np.zeros((len(lc), len(wave)))
        for i, d in enumerate(lc['date']):
            if verbose:
                print(d)
            f = self.fluxd(observer, d, wave, **kwargs)
            fluxd[i] = f.value
            unit = f.unit

        if verbose:
            print()

        lc = self._add_lc_columns(lc)

        cf = kwargs.get('cformat', dict()).get('fluxd', '{:9.3g}')
        unit = str(unit)
        for i in range(fluxd.shape[1]):
            lc.add_column(Column(data=fluxd[:, i],
                                 name=wformat.format(wave.value[i]),
                                 format=cf, unit=unit))
        return lc

    def _add_lc_columns(self, lc):
        """Add additional columns to a lightcurve table.

        Parameters
        ----------
        lc : Table
          The current lightcurve table.

        """
        return lc

    def observe(self, target, date, ltt=False):
        """Distance, phase angle, etc. to another object.

        Parameters
        ----------
        target : SolarSysObject
          The target to observe.
        date : string, float, astropy Time, datetime, or array
          Processed via `util.date2time`.
        ltt : bool, optional
          Account for light travel time when `True`.

        Returns
        -------
        geom : Geom
          The geometric parameters of the observation.

        """

        from astropy.time import TimeDelta
        import astropy.constants as const
        from . import Geom
        from ..util import date2time

        date = date2time(date)

        rt = target.r(date) # target postion
        ro = self.r(date)   # observer position
        vt = target.v(date) # target velocity
        vo = self.v(date)   # observer velocity

        g = Geom(ro * u.km, rt * u.km,
                 vo=vo * u.km / u.s, vt=vt * u.km / u.s,
                 date=date)

        if ltt:
            dt = (g['delta'] / const.c.si).decompose().value
            date -= TimeDelta(dt, format='sec')
            g = self.observe(target, date, ltt=False)

        return g

    def orbit(self, date):
        """Osculating orbital elements.

        Parameters
        ----------
        date : string, float, astropy Time, datetime, or array
          Processed via `util.date2time`.

        Returns
        -------
        orbit : dict
          See `util.state2orbit`.

        """

        from ..util import state2orbit, date2time
        r = self.r(date)
        v = self.v(date)
        jd = date2time(date).jd
        return state2orbit(r, v)

def getgeom(target, observer, date=None, ltt=False, kernel=None):
    """Moving target geometry parameters for an observer and date.

    Parameters
    ----------
    target : string, SolarSysObject
      The object's name or NAIF ID, as found in the relevant SPICE
      kernel, or a `SolarSysObject`.
    observer : string, array, SolarSysObject
      A valid built-in observer name, set of heliocentric rectangular
      ecliptic J2000 coordinates, or a `SolarSysObject`.  See the
      `ephem` package documentation for built-in moving objects that
      can be used as observers.
    date : string, float or array, optional
      The date(s) for which to compute the target's geometry.
    ltt : bool, optional
      Set to true to correct parameters for light travel time
      (currently, only one ltt iteration is implemented).
    kernel : string, optional
      If the target or observer is a string, use this kernel.

    Returns
    -------
    geom : Geom
      The geometric parameters of the observation.

    Raises
    ------
    ValueError on invalid input.

    """

    from . import _loaded_objects
    from .state import SpiceState

    if isinstance(target, str):
        target = SolarSysObject(SpiceState(target, kernel=kernel))
    elif not isinstance(target, SolarSysObject):
        raise ValueError("target must be a string or SolarSysObject.")

    if isinstance(observer, str):
        if observer.lower() in _loaded_objects:
            observer = _loaded_objects[observer.lower()]
        else:
            observer = SolarSysObject(SpiceState(target, kernel=kernel))
    elif np.iterable(observer):
        observer = FixedObject(observer)
    elif not isinstance(observer, SolarSysObject):
        raise ValueError("observer must be a string, array, or SolarSysObject")

    return observer.observe(target, date, ltt=ltt)

def getspiceobj(obj, kernel=None, name=None, **kwarg):
    """Create a new SolarSysObject with a SPICE kernel, for your convenience.

    Parameters
    ----------
    obj : string or int
      The object's name or NAIF ID, as found in the relevant SPICE
      kernel.
    kernel : string, optional
      The name of a specific SPICE planetary ephemeris kernel (SPK) to
      use for this object, or `None` to automatically search for a
      kernel through `find_kernel`.
    name : string
      The name of the object, or `None` to use `obj`.
    **kwarg
      Any other `SolarSysObject` keyword argument.

    Returns
    -------
    ssobj : SolarSysObject
      A `SolarSysObject` loaded with the requested SPICE ephemeris
      file.

    """

    from .state import SpiceState
    name = str(obj) if name is None else name
    return SolarSysObject(SpiceState(obj, kernel=kernel), name=name, **kwarg)

def getxyz(obj, date=None, kernel=None):
    """Coordinates and velocity from an ephemeris kernel.

    Coordinates are heliocentric rectangular ecliptic J2000.

    Parameters
    ----------
    obj : string or int
      The object's name or NAIF ID, as found in the relevant SPICE
      kernel.
    date : string, float, astropy Time, datetime, or array, optional
      Processed via `util.date2time`.
    kernel : string, optional
      The name of a specific SPICE planetary ephemeris kernel (SPK) to
      use for this object, or `None` to automatically search for a
      kernel through `find_kernel`.

    Returns
    -------
    r, v: array
      The position and veloctiy vectors as 3-element arrays, or, if
      multiple dates are requested, Nx3-element arrays. [km and km/s]

    Raises
    ------
    ValueError for invalid `date`.
    ObjectError for `obj` not in `kernel`.

    """

    from .state import SpiceState
    obj = SpiceState(obj, kernel=kernel)
    return obj.r(date), obj.v(date)

def summarizegeom(*args, **kwargs):
    """Pretty print a summary of the observing geometry.

    See `getgeom` for a description of the parameters.

    """

    getgeom(*args, **kwargs).summary()

# update module docstring
from ..util import autodoc
autodoc(globals())
del autodoc
