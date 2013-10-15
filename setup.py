#!/usr/bin/env python
from numpy.distutils.core import setup, Extension, Command

class PyTest(Command):
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        import sys
        import subprocess
        errno = subprocess.call([sys.executable, 'setup.py', 'build_ext',
                                 '--inplace'])
        errno = subprocess.call([sys.executable, 'setup.py', 'build'])
        errno = subprocess.call(['ipython', 'runtests.py', 'tests/'])
        raise SystemExit(errno)

def find_data_files():
    import os
    filelist = []
    for root, dirnames, files in os.walk('mskpy/data/'):
        dirlist = []
        for f in files:
            for suffix in ['.dat', '.txt', '.tbl']:
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
          version='2.0.0',
          description='General purpose and astronomy related tools',
          author="Michael S. Kelley",
          author_email="msk@astro.umd.edu",
          url="https://github.com/mkelley/mskpy",
          packages=['mskpy', 'mskpy.lib', 'mskpy.models', 'mskpy.image',
                    'mskpy.ephem', 'mskpy.instruments'],
          data_files=files,
          requires=['numpy', 'scipy', 'astropy'],
          ext_modules=[ext1],
          cmdclass={'test': PyTest},
          license='BSD',
          classifiers = [
              'Intended Audience :: Science/Research',
              "License :: OSI Approved :: BSD License",
              'Operating System :: OS Independent',
              "Programming Language :: Python :: 2.7",
              'Topic :: Scientific/Engineering :: Astronomy'
          ],
      )
