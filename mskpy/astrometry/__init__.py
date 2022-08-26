import os
from ginga.misc.Bunch import Bunch

plugin_path = os.path.split(__file__)[0]


def setup_mskpyastrometry():
    spec = Bunch(path=os.path.join(plugin_path, 'ginga_plugin.py'),
                 module='ginga_plugin', klass='MSKpyAstrometry',
                 ptype='local', workspace='dialogs',
                 category="Analysis")
    return spec
