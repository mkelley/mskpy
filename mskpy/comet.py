# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
comet --- Comets!
=================

.. autosummary::
   :toctree: generated/

   Classes
   -------
   Coma
   Comet

   Functions
   ---------
   afrho2fluxd
   afrho2Q
   flux2Q
   fluxd2afrho
   fluxd2efrho
   m2afrho
   M2afrho1
   m2qh2o
   Q2flux

"""

__all__ = [
    'Coma',
    'Comet',
    'afrho2fluxd',
    'afrho2Q',
    'flux2Q',
    'fluxd2afrho',
    'fluxd2efrho',
    'm2afrho',
    'M2afrho1',
    'm2qh2o',
    'Q2flux'
]

import numpy as np
import astropy.units as u
from astropy.time import Time

from .ephem import SolarSysObject, State
from .asteroid import Asteroid
from .models import AfrhoRadiation, AfrhoScattered, AfrhoThermal

class Coma(SolarSysObject):
    """A comet coma.

    Parameters
    ----------
    state : State
      The location of the coma.
    Afrho1 : Quantity
      Afrho of the coma at 1 AU.
    k : float, optional
      The logarithmic slope of the dust production rate's dependence
      on rh.
    reflected : AfrhoRadiation, optional
      A model for light scattered by dust or `None` to pass `kwargs`
      to `AfrhoScattered`.
    thermal : AfrhoRadiation, optional
      A model for thermal emission from dust or `None` to pass
      `kwargs` `AfrhoThermal`.
    name : string, optional
      A name for this coma.
    **kwargs
      Keywords for `AfrhoRadiation` and/or `AfrhoThermal`, when
      `reflected` and/or `thermal` are `None`, respectively.

    """
    _Afrho1 = None

    def __init__(self, state, Afrho1, k=-2, reflected=None, thermal=None,
                 name=None, **kwargs):
        assert u.cm.is_equivalent(Afrho1), "Afrho1 must have units of length"

        SolarSysObject.__init__(self, state, name=name)
        self.k = k

        if reflected is None:
            self.reflected = AfrhoScattered(1 * Afrho1.unit, **kwargs)
        else:
            self.reflected = reflected
        assert isinstance(self.reflected, AfrhoRadiation)

        if thermal is None:
            self.thermal = AfrhoThermal(1 * Afrho1.unit, **kwargs)
        else:
            self.thermal = thermal
        assert isinstance(self.thermal, AfrhoRadiation)

        # Afrho1 property will also update self.thermal, self.scattered
        self.Afrho1 = Afrho1

    @property
    def Afrho1(self):
        return self._Afrho1

    @Afrho1.setter
    def Afrho1(self, a):
        assert u.cm.is_equivalent(a), "Afrho1 must have units of length"
        self._Afrho1 = a
        self.thermal.Afrho = 1 * a.unit
        self.reflected.Afrho = 1 * a.unit

    def fluxd(self, observer, date, wave, rap=1.0 * u.arcsec,
              reflected=True, thermal=True, ltt=False,
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
          The wavelengths to compute `fluxd`.
        rap : Quantity, optional
          The aperture radius, angular or projected distance at the
          comet.
        reflected : bool, optional
          If `True` include the reflected light model.
        thermal : bool, optional
          If `True` include the thermal emission model.
        ltt : bool, optional
          Set to `True` to correct the object's position for light
          travel time.
        unit : astropy Unit
          The return unit, must be flux density.
        
        Returns
        -------
        fluxd : Quantity

        """

        if not np.iterable(wave):
            wave = [wave.value] * wave.unit

        fluxd = np.zeros(len(wave)) * unit
        if self.Afrho1.value <= 0:
            return fluxd

        g = observer.observe(self, date, ltt=ltt)

        if reflected:
            f = self.reflected.fluxd(g, wave, rap, unit=unit)
            fluxd += f * self.Afrho1.value * g['rh'].to(u.au).value**self.k

        if thermal:
            f = self.thermal.fluxd(g, wave, rap, unit=unit)
            fluxd += f * self.Afrho1.value * g['rh'].to(u.au).value**self.k

        return fluxd

    def _add_lc_columns(self, lc):
        from astropy.table import Column
        afrho = self.Afrho1 * lc['rh'].data**self.k
        lc.add_column(Column(afrho, name='Afrho', format='{:.4g}'))
        return lc

class Comet(SolarSysObject):
    """A comet.

    Parameters
    ----------
    state : State
      The location of the comet.
    Afrho1 : Quantity
      Afrho at 1 AU.
    R : Quantity
      Nucleus radius.
    Ap : float, optional
      Geometric albedo of the nucleus.
    nucleus : dict or Asteroid
      The nucleus of the comet as an `Asteroid` instance, or a
      dictionary of keywords to initialize a new `Asteroid`.
    coma : dict or Coma
      The coma of the comet as a `Coma` instance or a dictionary of
      keywords to initialize a new `Coma`.
    name : string, optional
      A name for this comet.

    Attributes
    ----------
    nucleus : The nucleus as an `Asteroid` instance.
    coma : The coma as a `Coma` instance.

    Methods
    -------
    fluxd : Total flux density as seen by an observer.

    Examples
    --------
    Create a comet with `Asteroid` and `Coma` objects.

    >>> import astropy.units as u
    >>> from mskpy import Asteroid, Coma, Comet, SpiceState
    >>> state = SpiceState('hartley 2')
    >>> Afrho1 = 302 * u.cm
    >>> R = 0.6 * u.km
    >>> Ap = 0.04
    >>> nucleus = Asteroid(state, R * 2, Ap, eta=1.0, epsilon=0.95)
    >>> coma = Coma(state, Afrho1, S=0.0, A=0.37, Tscale=1.18)
    >>> comet = Comet(state, Afrho1, R, Ap, nucleus=nucleus, coma=coma)

    Create a comet with keyword arguments.

    >>> import astropy.units as u
    >>> from mskpy import Comet, SpiceState
    >>> Afrho1 = 302 * u.cm
    >>> R = 0.6 * u.km
    >>> nucleus = dict(eta=1.0, epsilon=0.95)
    >>> coma = dict(S=0.0, A=0.37, Tscale=1.18)
    >>> comet = Comet(SpiceState('hartley 2'), Afrho1, R, nucleus=nucleus, coma=coma)

    """

    def __init__(self, state, Afrho1, R, Ap=0.04, nucleus=dict(), coma=dict(),
                 name=None):
        SolarSysObject.__init__(self, state, name=name)

        if isinstance(nucleus, dict):
            self.nucleus = Asteroid(self.state, 2 * R, Ap, **nucleus)
        else:
            self.nucleus = nucleus

        if isinstance(coma, dict):
            self.coma = Coma(self.state, Afrho1, **coma)
        else:
            self.coma = coma

        # these properties will update nucleus and coma
        self.Afrho1 = Afrho1
        self.R = R
        self.Ap = Ap

    @property
    def R(self):
        return self.nucleus.D / 2.0

    @R.setter
    def R(self, r):
        self.nucleus.D = 2 * r

    @property
    def Ap(self):
        return self.nucleus.Ap

    @Ap.setter
    def Ap(self, p):
        self.nucleus.Ap = p

    @property
    def Afrho1(self):
        return self.coma.Afrho1

    @Afrho1.setter
    def Afrho1(self, a):
        self.coma.Afrho1 = a

    def fluxd(self, observer, date, wave, rap=1.0 * u.arcsec,
              reflected=True, thermal=True, nucleus=True, coma=True,
              ltt=False, unit=u.Unit('W / (m2 um)'), **kwargs):
        """Total flux density as seen by an observer.

        Parameters
        ----------
        observer : SolarSysObject
          The observer.
        date : string, float, astropy Time, datetime
          The time of the observation in any format acceptable to
          `observer`.
        wave : Quantity
          The wavelengths to compute `fluxd`.
        rap : Quantity, optional
          The aperture radius, angular or projected distance at the
          comet.
        reflected : bool, optional
          If `True` include the reflected light model.
        thermal : bool, optional
          If `True` include the thermal emission model.
        nucleus : bool, optional
          If `True` include the nucleus.
        coma : bool, optional
          If `True` include the coma.
        ltt : bool, optional
          Set to `True` to correct the object's position for light
          travel time.
        unit : astropy Unit
          The return unit, must be flux density.
        
        Returns
        -------
        fluxd : Quantity

        """

        fluxd = np.zeros(np.size(wave.value)) * unit

        if nucleus:
            fluxd += self.nucleus.fluxd(observer, date, wave,
                                        reflected=reflected,
                                        thermal=thermal, unit=unit)
        if coma:
            fluxd += self.coma.fluxd(observer, date, wave, rap=rap,
                                     reflected=reflected,
                                     thermal=thermal, unit=unit)

        return fluxd

    def _add_lc_columns(self, lc):
        return self.coma._add_lc_columns(lc)

def afrho2fluxd(wave, afrho, rap, geom, sun=None, unit=u.Unit('W/(m2 um)'),
                bandpass=None):
    """Convert A(th)frho to flux density.

    See A'Hearn et al. (1984) for the definition of Afrho.

    Parameters
    ----------
    wave : Quantity
      The wavelength of the measurement.
    afrho : Quantity
      The Afrho parameter.
    rap : Quanitity
      The aperture radius.  May be angular size or projected linear
      size at the distance of the comet.
    geom : dictionary of Quantities or ephem.Geom
      The observing geometry via keywords `rh`, `delta`.
    sun : Quantity, optional
      Use this value for the solar flux density at 1 AU, or `None` to
      use `calib.solar_flux`.
    unit : Unit, optional
      Unit to pass to `calib.solar_flux`.
    bandpass : dict, optional
      Instead of using `sun`, set to a dictionary of keywords to pass,
      along with the solar spectrum, to `util.bandpass`.

    Returns
    -------
    fluxd : Quantity
      The flux density of the comet.

    Notes
    -----
    Farnham, Schleicher, and A'Hearn (2000), Hale-Bopp
    filter set:

      UC = 0.3449 * u.um, qUC = 2.716e17 -> 908.9 * u.Unit('W/m2/um')
      BC = 0.4453 * u.um, qBC = 1.276e17 -> 1934 * u.Unit('W/m2/um')
      GC = 0.5259 * u.um, qGC = 1.341e17 -> 1841 * u.Unit('W/m2/um')
      RC = 0.7133 * u.um, qRC = 1.975e17 -> 1250 * u.Unit('W/m2/um')

    """

    from . import util
    from . import calib

    # parameter check
    assert wave.unit.is_equivalent(u.um)
    assert afrho.unit.is_equivalent(u.um)
    assert geom['rh'].unit.is_equivalent(u.um)
    assert geom['delta'].unit.is_equivalent(u.um)

    if rap.unit.is_equivalent(u.cm):
        rho = rap.to(afrho.unit)
    elif rap.unit.is_equivalent(u.arcsec):
        rho = geom['delta'].to(afrho.unit) * rap.to(u.rad).value
    else:
        raise ValueError("rap must have angular or length units.")

    if sun is None:
        assert unit.is_equivalent('W/(m2 um)', u.spectral_density(wave))

        if bandpass is None:
            sun = calib.solar_flux(wave, unit=unit)
        else:
            sw, sf = calib.e490(smooth=True, unit=unit)
            sun = util.bandpass(sw.to(u.um).value, sf.value, **bandpass)[1]
            sun *= unit
    else:
        assert sun.unit.is_equivalent('W/(m2 um)', u.spectral_density(wave))

    fluxd = (afrho * rho * sun * (1 * u.au / geom['rh'])**2
             / 4. / geom['delta']**2)
    return fluxd.to(unit)

def flux2Q(fgas, wave, geom, g, rap, v):
    """Convert gas emission to Q.

    Use when rho << gas photodissociation lengthscale.

    Q = 2 N v / (pi rho)

    Parameters
    ----------
    fgas : Quantity
      The total line/band emission from the gas in units of flux.
    wave : Quantity
      The wavelength of the line/band.
    geom : dict of Quantity, or ephem.Geom
      The observing circumstances (rh and delta).
    g : Quantity
      The g-factor of the line/band at 1 AU.
    rap : Quantity
      The radius of the aperture used to measure the flux.  May be an
      angle or length.
    v : Quantity
      The expansion velocity of the gas.

    Returns
    -------
    Q : Quantity
      The gas production rate.

    """

    from numpy import pi

    assert isinstance(fgas, u.Quantity)
    assert isinstance(wave, u.Quantity)
    assert isinstance(geom['delta'], u.Quantity)
    assert isinstance(geom['rh'], u.Quantity)
    assert isinstance(g, u.Quantity)
    assert isinstance(rap, u.Quantity)
    assert isinstance(v, u.Quantity)

    hc = 1.9864457e-25 * u.J * u.m

    if rap.unit.is_equivalent(u.radian):
        rho = (rap * geom['delta'].to(u.au).value
               * 725 * u.km / u.arcsec).decompose()
    elif rap.unit.is_equivalent(u.meter):
        rho = rap
    else:
        raise ValueError("rap must have angular or length units.")

    N = (4 * pi * geom['delta']**2 * fgas * wave / hc
         * geom['rh'].to(u.au).value**2 / g)

    return (2 * N * v / pi / rho).decompose()

def afrho2Q(Afrho, rap, geom, k, v1, u1=-0.5, u2=-1.0, Ap=0.05,
            rho_g=1 * u.Unit('g/cm3'), arange=[0.1, 1e4] * u.um):
    """Convert Afrho to dust production rate.

    The conversion assumes Afrho is measured within the 1/rho regime,
    and allows for a size-dependent expansion speed, and power-law
    size distributions.

      Q = 2 / (rho pi) \int_a0^a1 n(a) m(a) v(a) da

    where rho is the projected linear aperture radius at the distance
    of the comet, the particle radii range from a0 to a1, n(a) is the
    differential size distribution, m(a) is the mass of a grain with
    radius a, v(a) is the expansion speed of a grain with radius a.

    Parameters
    ----------
    Afrho : Quantity
      The Afrho value (at a phase angle of 0 deg).
    rap : Quantity
      The angular or linear raidus of the projected aperture within
      which `Afrho` was measured.
    geom : dictionary of Quantity
      The observation geometry via keywords `rh`, `delta`.
    k : float
      The power-law slope of the differential size distribution.
    v1 : Quantity
      The expansion speed of 1 micron radius grains ejected at 1 AU
      from the sun.
    u1, u2 : float, optional
      Defines the relationship between expansion speed, grain radius,
      and heliocentric distance: v = v1 a^{u1} rh^{u2}.
    Ap : float, optional
      The geometric albedo of the dust at the same wavelength as
      `Afrho` is measured.
    rho_g : Quantity, optional
      The dust grain density.
    arange : Quanitity array, optional
      The minimum and maximum grain sizes in the coma.

    Returns
    -------
    Q : Quantity
      The dust mass production rate.

    """

    from scipy.integrate import quad
    from numpy import pi

    Afrho = u.Quantity(Afrho, u.m)
    rh = u.Quantity(geom['rh'], u.au)
    delta = u.Quantity(geom['delta'], u.au)

    try:
        rho = u.Quantity(rap, u.m)
    except u.UnitsError:
        try:
            rho = (u.Quantity(rap, u.arcsec) * 725e3 * u.m / u.arcsec / u.au
                   * delta)
        except u.UnitsError:
            print('rap must have units of length or angluar size.')
            raise

    v1 = u.Quantity(v1, u.m / u.s)
    rho_g = u.Quantity(rho_g, u.kg / u.m**3)
    arange = u.Quantity(arange, u.m)

    A = 4 * Ap
    cs = pi * quad(lambda a: a**(2 + k), *arange.value)[0] * u.m**2
    N = (Afrho * pi * rho / A / cs).decompose().value
    q = (4 * pi / 3 * rho_g * v1 * (rh / (1 * u.au))**u2 * 1e6**u1
         * quad(lambda a: a**(k + 3 + u1), *arange.value)[0] * u.m**3)
    Q = (N * 2 / pi / rho * q).to(u.kg / u.s)

    return Q

def fluxd2afrho(wave, fluxd, rho, geom, sun=None, bandpass=None):
    """Convert flux density to A(th)frho.

    See A'Hearn et al. (1984) for the definition of Afrho.

    Parameters
    ----------
    wave : float or array
      The wavelength of flux.  [micron]
    fluxd : float or array
      The flux density of the comet.  [W/m2/um, or same units as sun
      keyword]
    rho : float
      The aperture radius.  [arcsec]
    geom : dictionary or ephem.Geom
      The observing geometry via keywords `rh`, `delta`.  [AU, AU]
    sun : float, optional
      Use this value for the solar flux density at 1 AU, or None to
      use calib.solar_flux().  [same units as flux parameter]
    bandpass : dict, optional
      Instead of using `sun`, set to a dictionary of keywords to pass,
      along with the solar spectrum, to `util.bandpass`.

    Returns
    -------
    afrho : float or ndarray
      The Afrho parameter.  [cm]

    Notes
    -----
    Farnham, Schleicher, and A'Hearn (2000), Hale-Bopp
    filter set:

      UC = 0.3449 um, qUC = 2.716e17 -> 908.9 W/m2/um
      BC = 0.4453 um, qBC = 1.276e17 -> 1934 W/m2/um
      GC = 0.5259 um, qGC = 1.341e17 -> 1841 W/m2/um
      RC = 0.7133 um, qRC = 1.975e17 -> 1250 W/m2/um

    """

    from . import util
    from . import calib
    import astropy.constants as const

    try:
        deltacm = geom['delta'].to(u.cm).value
        rh = geom['rh'].to(u.au).value
    except AttributeError:
        deltacm = geom['delta'] * const.au.to(u.cm).value
        rh = geom['rh']

    if sun is None:
        if bandpass is None:
            sun = calib.solar_flux(wave * u.um, unit=u.Unit('W/(m2 um)')).value
        else:
            sw, sf = calib.e490(smooth=True, unit=u.Unit('W/(m2 um)'))
            sun = util.bandpass(sw.micrometer, sf.value, **bandpass)[1]

    afrho = 4 * deltacm * 206265. / rho * fluxd * rh**2 / sun
    return afrho

def fluxd2efrho(wave, flux, rho, geom, Tscale=1.1):
    """Convert flux density to efrho (epsilon * f * rho).

    efrho is defined by Kelley et al. (2013, Icarus 225, 475-494).

    Parameters
    ----------
    wave : float or array
      The wavelength of flux.  [micron]
    flux : float or array
      The flux density of the comet.  [W/m2/um]
    rho : float
      The aperture radius.  [arcsec]
    geom : dictionary or ephem.Geom
      The observing geometry via keywords `rh`, `delta`.  [AU, AU]
    Tscale : float, optional
      Use a continuum temperature of `Tscale * 278 / sqrt(rh)` K.
      Kelley et al. (2013) suggest a default of 1.1.

    Returns
    -------
    efrho : float or ndarray
      The epsfrho parameter.  [cm]

    """

    from . import util
    import astropy.constants as const

    try:
        deltacm = geom['delta'].to(u.cm).value
        rh = geom['rh'].to(u.au).value
    except AttributeError:
        deltacm = geom['delta'] * const.au.to(u.cm).value
        rh = geom['rh']

    B = util.planck(wave, Tscale * 278. / np.sqrt(rh),
                    unit=u.Unit('W/(m2 um sr)'))
    B = B.value
    _rho = rho / 206265. * deltacm  # cm
    Om = np.pi * (rho / 206265.)**2  # sr
    I = flux / Om  # W/m2/um/sr
    return I * _rho / B

def m2afrho(m, g, C=8.5e17, m_sun=-27.1):
    """Convert JPL/HORIZONS apparent magnitude, m, to Afrho.

    *** EXPERIMENTAL ***

    Based on a few comets.  See MSK's notes.

    Parameters
    ----------
    m : float
      Comet's apparent magnitude from JPL.
    g : dict of Quantity, or ephem.Geom
      The observing circumstances (rh and delta).
    C : float
      Conversion constant. [cm]
    m_sun : float
      Apparent magnitude of the Sun.

    Returns
    -------
    Afrho : float
      Afrho.  [cm]

    """
    print("    *** EXPERIMENTAL ***   ")
    #M = m - 5 * np.log10(geom['rh'].to(u.au).value
    #                     * geom['delta'].to(u.au).value)
    #return 4.0e6 * 10**(M / -2.5)
    afrho = (C * g['delta'].to(u.au)**2 * g['rh'].to(u.au)**2 / u.au**4
             * 10**(-0.4 * (m - m_sun))) * u.cm
    return afrho

def M2afrho1(M1):
    """Convert JPL's absolute magnitude, M1, to Afrho at 1 AU.

    Based on an empirical correlation between M1 and Afrho as measured
    by A'Hearn et al. (1995).  There is easily a factor of 4 scatter
    about the trend line.

    Parameters
    ----------
    M1 : float
      Comet's absolute magnitude from JPL.

    Returns
    -------
    Afrho1 : float
      Afrho at 1 AU.  [cm]

    """
    return 10**(-0.208 * M1 + 4.687)

def m2qh2o(m_H):
    """Convert helocentric magnitude, m_H, to Q(H2O).

    Based on an empirical correlation between heliocentric magnitude
    and Q(H2O) by Jorda et al. (2008, ACM, 8046).  Scatter about the
    trend: 1-sigma = 1.6, 2-sigma = 2.4, 3-sigma = 3.7.

    Parameters
    ----------
    m_H : float
      Comet's heliocentric magnitude = m_V - 5 * log10(Delta).

    Returns
    -------
    Q : float
      Q(H2O) at 1 AU.  [molecules/s]

    """
    return 10**(30.675 - 0.2453 * m_H)

def Q2flux(Q, wave, geom, g, rap, v):
    """Convert Q to line emission.

    Use when rho << gas photodissociation lengthscale.

    Q = 2 N v / (pi rho)

    Parameters
    ----------
    Q : Quantity
      The production rate (inverse time).
    wave : Quantity
      The wavelength of the line/band.
    geom : dict of Quantity, or ephem.Geom
      The observing circumstances (rh and delta).
    g : Quantity
      The g-factor of the line/band at 1 AU.
    rap : Quantity
      The radius of the aperture used to measure the flux.  May be an
      angle or length.
    v : Quantity
      The expansion velocity of the gas.

    Returns
    -------
    flux : Quantity
      The total line flux.

    """

    from numpy import pi

    assert isinstance(Q, u.Quantity)
    assert isinstance(wave, u.Quantity)
    assert isinstance(geom['delta'], u.Quantity)
    assert isinstance(geom['rh'], u.Quantity)
    assert isinstance(g, u.Quantity)
    assert isinstance(rap, u.Quantity)
    assert isinstance(v, u.Quantity)

    hc = 1.9864457e-25 * u.J * u.m

    if rap.unit.is_equivalent(u.radian):
        rho = (rap * geom['delta'].to(u.au).value
               * 725 * u.km / u.arcsec).decompose()
    elif rap.unit.is_equivalent(u.meter):
        rho = rap
    else:
        raise ValueError("rap must have angular or length units.")

    F = (Q * rho * hc * g / v / geom['delta']**2 / wave / 8.0
         / geom['rh'].to(u.au).value**2)

    return F.to('W/m2')

# update module docstring
from .util import autodoc
autodoc(globals())
del autodoc

