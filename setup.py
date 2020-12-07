#!/usr/bin/env python
from numpy.distutils.core import setup, Extension
from distutils.command.install import install


class my_install(install):
    def run(self):
        install.run(self)
        import mskpy.config


# class PyTest(Command):
#     user_options = []

#     def initialize_options(self):
#         pass

#     def finalize_options(self):
#         pass

#     def run(self):
#         import sys
#         import subprocess
#         errno = subprocess.call([sys.executable, 'setup.py', 'build_ext',
#                                  '--inplace'])
#         errno = subprocess.call([sys.executable, 'setup.py', 'build'])
#         errno = subprocess.call(['ipython', 'runtests.py', 'tests/'])
#         raise SystemExit(errno)


def find_data_files():
    import os
    filelist = []
    for root, dirnames, files in os.walk('mskpy/data/'):
        dirlist = []
        for f in files:
            for suffix in ['.dat', '.txt', '.tbl', '.fits']:
                if f.endswith(suffix):
                    dirlist.append(os.path.join(root, f))
        if len(dirlist) > 0:
            filelist.append((root, dirlist))
    return filelist


if __name__ == "__main__":
    from glob import glob

    ext1 = Extension('mskpy.lib.davint', glob('src/davint/*.f'))
    files = find_data_files()

    setup(name='mskpy',
          version='3.0.10',
          description='General purpose and astronomy related tools',
          author="Michael S. P. Kelley",
          author_email="msk@astro.umd.edu",
          url="https://github.com/mkelley/mskpy",
          packages=['mskpy', 'mskpy.lib', 'mskpy.models', 'mskpy.image',
                    'mskpy.ephem', 'mskpy.instruments', 'mskpy.observing',
                    'mskpy.photometry'],
          data_files=files,
          scripts=['scripts/ephemeris', 'scripts/transit', 'scripts/comet-fest',
                   'scripts/center-target', 'scripts/H2D', 'scripts/ds9-ext',
                   'scripts/ads'],
          install_requires=['numpy', 'scipy',
                            'astropy', 'spiceypy>1.1', 'pytz'],
          ext_modules=[ext1],
          #cmdclass={'test': PyTest, 'install': my_install},
          license='BSD',
          classifiers=[
              'Intended Audience :: Science/Research',
              "License :: OSI Approved :: BSD License",
              'Operating System :: OS Independent',
              "Programming Language :: Python :: 3",
              'Topic :: Scientific/Engineering :: Astronomy'
          ]
          )
