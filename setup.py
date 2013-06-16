#!/usr/bin/env python
if __name__ == "__main__":
    from glob import glob
    from numpy.distutils.core import setup, Extension

    ext1 = Extension('mskpy.lib.davint', glob('src/davint/*.f'))

    setup(name='mskpy',
          version='2.0.dev',
          description='General purpose and astronomy related tools',
          author="Michael S. Kelley",
          author_email="msk@astro.umd.edu",
          url="https://github.com/mkelley/mskpy",
          packages=['mskpy', 'mskpy.lib'],
          requires=['numpy', 'scipy', 'astropy'],
          install_requires=['numpy'],
          ext_modules=[ext1],
          license='BSD',
          classifiers = [
              'Intended Audience :: Science/Research',
              "License :: OSI Approved :: BSD License",
              'Operating System :: OS Independent',
              "Programming Language :: Python :: 2.7",
              'Topic :: Scientific/Engineering :: Astronomy'
          ],
      )
