import os

import dsacalib.calib as dc

from pkg_resources import package_data

MSNAME = package_data("dsacalib", "data/test.ms")
SOURCENAME = "test"


def test_delay_calibration_worker():
    refant = 0
    t = 'inf'
    combine_spw = True
    name = 'kcalinf'

    error = dc.delay_calibration_worker(MSNAME, SOURCENAME, refant, t, combine_spw, name)
    assert error == 0
    assert os.path.exists(f"{MSNAME}_{SOURCENAME}_{name}")

    t = '60s'
    name = 'cal60'

    error = dc.delay_calibration_worker(MSNAME, SOURCENAME, refant, t, combine_spw, name)
    assert error == 0
    assert os.path.exists(f"{MSNAME}_{SOURCENAME}_{name}")


def test_delay_caibration():
    refants = [0, 1]

    error = dc.delay_calibration(MSNAME, SOURCENAME, refants)
    assert error == 0
    assert os.path.exists(f"{MSNAME}_{SOURCENAME}_kcal")
    assert os.path.exists(f"{MSNAME}_{SOURCENAME}_2kcal")


def test_gain_calibration():
    refant = 0
    caltables = [{
        "table": f"{MSNAME}_{SOURCENAME}_kcal",
        "type": "K",
        "spwmap": [-1]
    }]
    if not os.path.exists(caltables[0]['table']):
        test_delay_caibration()

    error = dc.gain_calibration(MSNAME, SOURCENAME, refant, caltables)
    assert error == 0
    for calsuffix in ['gacal', 'gpcal', 'bacal', 'bpcal']:
        assert os.path.exists(f"{MSNAME}_{SOURCENAME}_{calsuffix}")


def test_bandpass_calibration():
    refant = 0
    if not os.path.exists(f"{MSNAME}_{SOURCENAME}_kcal"):
        test_delay_caibration()

    caltables = [{
        "table": f"{MSNAME}_{SOURCENAME}_kcal",
        "type": "K",
        "spwmap": [-1],
    }]

    error = dc.bandpass_calibration(MSNAME, SOURCENAME, refant, caltables)

    assert error == 0
    for calsuffix in ['bacal', 'bpcal']:
        assert os.path.exists(f"{MSNAME}_{SOURCENAME}_{calsuffix}")
