# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
config --- mskpy configuration.
===============================

"""

import os.path
import configparser

def _find_config():
    """Locate the config file."""
    fn = os.path.join(os.path.expanduser("~"), '.config', 'mskpy',
                      'mskpy.cfg')
    if not os.path.exists(fn):
        _create_config(fn)

    return fn

_defaults = (
    # (section, parameter, value)
    ('ephem.core', 'kernel_path', os.path.sep.join([home, 'data', 'kernels'])),
    ('calib', 'cohen_path', os.path.sep.join([home, 'data', 'mid-ir'])),
    ('spex', 'spextool_path', os.path.sep.join([home, 'local', 'idl', 'irtf', 'Spextool'])),
)

def _create_config(fn):
    from configparser import DuplicateSectionError
    path = os.path.dirname(fn)
    d = path.split(os.path.sep)
    for i in range(len(d)):
        x = os.path.sep.join(d[:i+1])
        if len(x) == 0:
            continue
        if not os.path.exists(x):
            os.mkdir(x)

    home = os.path.expanduser("~")
    config = _verify_config(configparser.RawConfigParser())

    with open(fn, 'w') as outf:
        config.write(outf)

def _verify_config(c):
    """Verifies that all sections are contained in `c`, else adds them."""
    from configparser import DuplicateSectionError

    updated = False
    for section, parameter, value in _defaults:
        try:
            c.add_section(section)
            updated = True
        except DuplicateSectionError:
            pass

        if c.get(section, parameter, fallback=None) is None:
            c.set(section, parameter, value)
            updated = True

    return c, updated

config_file = _find_config()
config = configparser.RawConfigParser()
config.read(config_file)
c, updated = _verify_config(config)
if updated:
    config = c
    with open(fn, 'w') as outf:
        config.write(config_file)
