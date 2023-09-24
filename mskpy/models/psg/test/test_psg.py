from astropy.utils.data import get_pkg_data_filename
from mskpy.models import psg

def test_parse():
    filename = get_pkg_data_filename("psg_cfg.txt", "mskpy.models.psg.test")
    config = psg.PSGConfig.from_file(filename)

    assert config["OBJECT"] == "Comet"
    assert config["OBJECT-NAME"] == "22P"
    assert config["OBJECT-OBS-VELOCITY"] == -5.887
    assert config["ATMOSPHERE-TYPE"] == ["GSFC[CH4]", "GSFC[C2H6]", "GSFC[CH3OH]", "GSFC[CH3OH_V9]"]
    assert config["ATMOSPHERE-WEIGHT"] == 541.40
    assert config["ATMOSPHERE-NGAS"] == 4
    assert config["ATMOSPHERE-PRESSURE"] == 1e27
    assert config["ATMOSPHERE-AEROS"] == None

def test_str():
    config = psg.PSGConfig()
    config["OBJECT"] = "Comet"
    config["OBJECT-NAME"] = "22P"
    config["OBJECT-OBS-VELOCITY"] = -5.887
    config["ATMOSPHERE-TYPE"] = ["GSFC[CH4]", "GSFC[C2H6]", "GSFC[CH3OH]", "GSFC[CH3OH_V9]"]
    config["ATMOSPHERE-WEIGHT"] = 541.40
    config["ATMOSPHERE-NGAS"] = 4
    config["ATMOSPHERE-PRESSURE"] = 1e27
    config["ATMOSPHERE-AEROS"] = None

    s = str(config)
    assert s == """<OBJECT>Comet
<OBJECT-NAME>22P
<OBJECT-OBS-VELOCITY>-5.887
<ATMOSPHERE-TYPE>GSFC[CH4],GSFC[C2H6],GSFC[CH3OH],GSFC[CH3OH_V9]
<ATMOSPHERE-WEIGHT>541.4
<ATMOSPHERE-NGAS>4
<ATMOSPHERE-PRESSURE>1e+27
<ATMOSPHERE-AEROS>
"""