"""
Rubin observatory.
"""

from enum import Enum


class Bands(Enum):
    u = 0
    g = 1
    r = 2
    i = 3
    z = 4
    y = 5

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
        wavelengths = [372.4, 480.7, 622.1, 755.9, 868.0, 975.3]
        return wavelengths[self.value]
