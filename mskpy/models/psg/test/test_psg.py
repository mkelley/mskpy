from astropy.utils.data import get_pkg_data_filename
from mskpy.models import psg


def test_parse_config():
    filename = get_pkg_data_filename("psg_cfg.txt", "mskpy.models.psg.test")
    config = psg.PSGConfig.from_file(filename)

    assert config["OBJECT"] == "Comet"
    assert config["OBJECT-NAME"] == "22P"
    assert config["OBJECT-OBS-VELOCITY"] == -5.887
    assert config["ATMOSPHERE-TYPE"] == ["GSFC[CH4]",
                                         "GSFC[C2H6]", "GSFC[CH3OH]", "GSFC[CH3OH_V9]"]
    assert config["ATMOSPHERE-WEIGHT"] == 541.40
    assert config["ATMOSPHERE-NGAS"] == 4
    assert config["ATMOSPHERE-PRESSURE"] == 1e27
    assert config["ATMOSPHERE-AEROS"] == None


def test_config_str():
    config = psg.PSGConfig()
    config["OBJECT"] = "Comet"
    config["OBJECT-NAME"] = "22P"
    config["OBJECT-OBS-VELOCITY"] = -5.887
    config["ATMOSPHERE-TYPE"] = ["GSFC[CH4]",
                                 "GSFC[C2H6]", "GSFC[CH3OH]", "GSFC[CH3OH_V9]"]
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


def test_parse_model():
    filename = get_pkg_data_filename("psg_rad.txt", "mskpy.models.psg.test")
    model = psg.PSGModel(filename)
    columns = ["wave", "Total", "22P", "Nucleus", "Dust",
               "GSFC[CH4]", "GSFC[C2H6]", "GSFC[CH3OH]", "GSFC[CH3OH_V9]"]
    values = [3.267386380, 1.4972492e-05, 3.34337e-02, 2.60666e-02,
              7.35209e-03, 1.48372e-06, 2.71464e-07, 1.34468e-06, 1.19062e-05]
    for k, v in zip(columns, values):
        assert model[k][521] == v
        assert model[521][k] == v

    assert model["CH4"][521] == values[5]
    assert model["C2H6"][521] == values[6]
    assert model["CH3OH"][521] == values[7] + values[8]
