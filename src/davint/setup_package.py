from glob import glob
from numpy.distutils.core import Extension


def get_extensions():
    return [
        Extension(
            name='mskpy.lib.davint',
            sources=glob('*.f')
        )
    ]
