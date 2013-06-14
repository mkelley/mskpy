#!/usr/bin/env python
from distutils.core import setup

if __name__ == "__main__":
    setup(name='mskpy',
          version='2.0.dev',
          description='General purpose and astronomy related tools',
          author="Michael S. Kelley",
          author_email="msk@astro.umd.edu",
          url="https://github.com/mkelley/mskpy",
          packages=['mskpy'],
          require==['numpy', 'scipy', 'astropy'],
          license='BSD',
          classifiers = [
              'Intended Audience :: Science/Research',
              "License :: OSI Approved :: BSD License",
              'Operating System :: OS Independent',
              "Programming Language :: Python :: 2.7",
              'Topic :: Scientific/Engineering :: Astronomy'
          ]
      )
