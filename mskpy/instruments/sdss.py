"""
Sloan Digital Sky Survey.
"""

from enum import Enum
import astropy.units as u


class Bands(Enum):
    u = 0
    g = 1
    r = 2
    i = 3
    z = 4

    def __str__(self):
        return str(self.name)

    def __eq__(self, band):
        return self.value == band.value

    def __lt__(self, band):
        return self.value < band.value

    def __gt__(self, band):
        return self.value > band.value

    @property
    def wave(self):
        # lambda_mean from Willmer 2018
        wavelengths = [0.3562, 0.4719, 0.6185, 0.7500, 0.8961] * u.um
        return wavelengths[self.value]
