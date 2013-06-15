#!/usr/bin/env python
def configuration(parent_package='mskpy', top_path=None):
    from numpy.distutils.misc_util import Configuration
    from glob import glob
    config = Configuration('', parent_package, top_path)
    config.add_extension('davint', sources=glob('src/davint/*.f'))
    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(name='mskpy',
          version='2.0.dev',
          description='General purpose and astronomy related tools',
          author="Michael S. Kelley",
          author_email="msk@astro.umd.edu",
          url="https://github.com/mkelley/mskpy",
          packages=['mskpy'],
          requires=['numpy', 'scipy', 'astropy'],
          install_requires=['numpy'],
          configuration=configuration,
          license='BSD',
          classifiers = [
              'Intended Audience :: Science/Research',
              "License :: OSI Approved :: BSD License",
              'Operating System :: OS Independent',
              "Programming Language :: Python :: 2.7",
              'Topic :: Scientific/Engineering :: Astronomy'
          ],
      )
