#!/usr/bin/env python

def find_data_files():
    import os
    filelist = []
    for root, dirnames, files in os.walk('data/'):
        dirlist = []
        for f in files:
            for suffix in ['.dat', '.txt', '.tbl']:
                if f.endswith(suffix):
                    dirlist.append(os.path.join(root, f))
        if len(dirlist) > 0:
            filelist.append((os.path.join('mskpy', root), dirlist))
    return filelist

if __name__ == "__main__":
    from glob import glob
    from numpy.distutils.core import setup, Extension

    ext1 = Extension('mskpy.lib.davint', glob('src/davint/*.f'))
    files = find_data_files()
    print files

    setup(name='mskpy',
          version='2.0.dev',
          description='General purpose and astronomy related tools',
          author="Michael S. Kelley",
          author_email="msk@astro.umd.edu",
          url="https://github.com/mkelley/mskpy",
          packages=['mskpy', 'mskpy.lib', 'mskpy.tests'],
          data_files=files,
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
