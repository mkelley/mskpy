# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
instruments --- Cameras, spectrometers, etc. for astronomy.
===========================================================

Instruments can be used observe a `SolarSysObject`.

   Classes
   -------
   Instrument
   Camera
   Spectrometer
   LongSlitSpectrometer

"""

__all__ = [
    'Instrument',
    'Camera',
    'CircularApertureSpectrometer',
    'LongSlitSpectrometer'
]

import numpy as np
import astropy.units as u

class Instrument(object):
    """Base class for any astronomical instrument.

    Parameters
    ----------
    wave : Quantity
      The wavelengths of the instrument.
    location : SolarSysObject, optional
      The location of the instrument, or `None` to assume Earth.

    Methods
    -------
    sed : Spectral energy distribution of a target.
    lightcurve : Secular light curve of a target.

    """

    def __init__(self, wave, location=None):
        from ..ephem import Earth, SolarSysObject

        self.wave = wave

        if location is None:
            self.location = Earth
        else:
            self.location = location

        assert isinstance(self.wave, u.Quantity)
        assert isinstance(self.location, SolarSysObject)

    def sed(self, target, date, **kwargs):
        """Spectral energy distribution of a target.

        Parameters
        ----------
        target : SolarSysObject
          The target to consider.
        date : various
          The date of the observation in a format acceptable to
          `SolarSysObject.fluxd`.
        **kwargs
          Keywords to pass to `SolarSysObject.fluxd`.

        Returns
        -------
        sed : ndarray

        """

        return target.fluxd(self.location, date, self.wave, **kwargs)

    def lightcurve(self, target, dates, **kwargs):
        """Secular light curve of a target.

        Parameters
        ----------
        target : SolarSysObject
          The target to consider.
        dates : various
          The dates of the observation in a format acceptable to
          `SolarSysObject.lightcurve`.
        **kwargs
          Keywords to pass to `SolarSysObject.lightcurve`.

        Returns
        -------
        lc : astropy Table

        """

        return target.lightcurve(self.location, dates, self.wave, **kwargs)

class Camera(Instrument):
    """Cameras.

    Parameters
    ----------
    wave : Quantity
      The wavelengths of the camera's filters.
    shape : array
      The dimensions of the array.
    ps : Quantity
      Angular size of one pixel, either square or rectangular `(y,
      x)`.
    location : SolarSysObject, optional
      The location of the camera.

    Methods
    -------
    sed : Spectral energy distribution of a target.
    lightcurve : Secular light curve of a target.

    """

    def __init__(self, wave, shape, ps, location=None):

        self.shape = shape

        if not np.iterable(ps):
            self.ps = np.ones(2) * ps.value * ps.unit
        else:
            self.ps = ps

        Instrument.__init__(self, wave, location=location)

    def sed(self, target, date, **kwargs):
        rap = (self.ps[0] + self.ps[1]) * 2.5
        kwargs['rap'] = kwargs.pop('rap', rap)
        return target.fluxd(self.location, date, self.wave, **kwargs)

    sed.__doc__ = Instrument.sed.__doc__ + """        Notes
        -----
        Default aperture radius is 2.5 pixels.

        """

    def lightcurve(self, target, dates, **kwargs):
        rap = (self.ps[0] + self.ps[1]) * 2.5
        kwargs['rap'] = kwargs.pop('rap', rap)
        return target.lightcurve(self.location, dates, self.wave, **kwargs)

    lightcurve.__doc__ = Instrument.sed.__doc__ + """        Notes
        -----
        Default aperture radius is 2.5 pixels.

        """

class CircularApertureSpectrometer(Instrument):
    """Circular-aperture spectrometers.

    The aperture is assumed to be circular, i.e., no spatial
    resolution.

    Parameters
    ----------
    waves : Quantity
      Instrument wavelengths.
    rap : Quantity
      Angular radius of the circular aperture.
    location : SolarSysObject, optional
      Location of the camera.

    Methods
    -------
    sed : Spectral energy distribution of a target.
    lightcurve : Secular light curve of a target.

    """

    def __init__(self, waves, rap, location=None):
        self.waves = u.Quantity(waves, u.um)
        self.rap = u.Quantity(rap, u.arcsec)

        Instrument.__init__(self, waves, location=location)

    def sed(self, target, date, **kwargs):
        kwargs['rap'] = kwargs.pop('rap', self.rap)
        return target.fluxd(self.location, date, self.waves, **kwargs)

    def lightcurve(self, target, dates, **kwargs):
        kwargs['rap'] = kwargs.pop('rap', self.rap)
        w = [3.3, 4.8, 10, 12] * self.waves.unit
        return target.lightcurve(self.location, dates, w, **kwargs)

    lightcurve.__doc__ = Instrument.sed.__doc__ + """        Notes
        -----
        Default aperture radius is half the slit width.

        `lightcurve` wavelengths are fixed.

        """

class LongSlitSpectrometer(Instrument):
    """Long-slit spectrometers.

    Parameters
    ----------
    wave : Quantity
      Center wavelength of the spectrum.  The spectral range will be
      computed from `shape[1]` and `dlam`.
    shape : array
      Dimensions of the array.  The first dimension is spatial, the
      second is spectral.
    ps : Quantity
      Angular size of one pixel.
    slit : float
      Slit width in pixels.
    dlam : Quantity
      Spectral size of one pixel.
    R : float, optional
      Spectral resolution, or `None` for a resolution of infinity.
    location : SolarSysObject, optional
      Location of the camera.

    Attributes
    ----------
    waves : Quantity
      The wavelengths of the spectrometer, computed from `wave`,
      `shape[1]`, and `dlam`.

    Methods
    -------
    sed : Spectral energy distribution of a target.
    lightcurve : Secular light curve of a target.

    Notes
    -----
    Presently, `R` has no consequences on the spectrum.

    """

    def __init__(self, wave, shape, ps, slit, dlam, R=None, location=None):
        self.shape = shape

        self.ps = ps
        if not np.iterable(ps):
            self.ps = np.ones(2) * ps.value * ps.unit
        else:
            self.ps = ps

        self.slit = slit
        self.dlam = dlam
        self.R = R
        assert isinstance(self.ps, u.Quantity)
        assert isinstance(self.dlam, u.Quantity)

        Instrument.__init__(self, wave, location=location)

        self.dlam = self.dlam.to(self.wave.unit)

    @property
    def waves(self):
        pixels = np.arange(self.shape[1])
        pixels -= np.median(pixels)
        return self.dlam.value * pixels * self.dlam.unit + self.wave

    def sed(self, target, date, **kwargs):
        rap = (self.ps[0] + self.ps[1]) / 2.0 * 2.5
        kwargs['rap'] = kwargs.pop('rap', rap)
        return target.fluxd(self.location, date, self.waves, **kwargs)

    sed.__doc__ = Instrument.sed.__doc__ + """        Notes
        -----
        Default aperture radius is half the slit width.

        """

    def lightcurve(self, target, dates, **kwargs):
        rap = (self.ps[0] + self.ps[1]) / 2.0 * 2.5
        kwargs['rap'] = kwargs.pop('rap', rap)
        w = self.waves.value
        w = [min(w), self.wave.value, max(w)] * self.wave.unit
        return target.lightcurve(self.location, dates, w, **kwargs)

    lightcurve.__doc__ = Instrument.sed.__doc__ + """        Notes
        -----
        Default aperture radius is half the slit width.

        `lightcurve` will only consider the start, center, and end
        wavelengths of the spectrum.

        """

# update module docstring
from ..util import autodoc
autodoc(globals())
del autodoc
