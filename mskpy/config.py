# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
config --- mskpy configuration.
===============================

"""

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

def _find_config():
    """Locate the config file."""
    import os
    path = os.path.join(os.path.expanduser("~"), '.config', 'mskpy',
                        'mskpy.cfg')
    if not os.path.exists(path):
        _create_config(path)
    return path

def _create_config(fn):
    import os
    path = os.path.dirname(fn)
    d = path.split(os.path.sep)
    for i in range(len(d)):
        x = os.path.sep.join(d[:i+1])
        if len(x) == 0:
            continue
        if not os.path.exists(x):
            os.mkdir(x)

    home = os.path.expanduser("~")
    config = configparser.RawConfigParser()
    config.add_section('ephem.core')
    config.set('ephem.core', 'kernel_path',
               os.path.sep.join([home, 'data', 'kernels']))
    config.add_section('calib')
    config.set('calib', 'cohen_path',
               os.path.sep.join([home, 'data', 'mid-ir']))
    config.add_section('spex')
    config.set('spex', 'spextool_path',
               os.path.sep.join([home, 'local', 'idl', 'irtf', 'Spextool']))

    with open(fn, 'w') as outf:
        config.write(outf)

config_file = _find_config()
config = configparser.RawConfigParser()
config.read(config_file)
