import pytest

import astropy.units as u
import numpy as np
from dsacalib import constants
from astropy.utils import iers
iers.conf.iers_auto_url_mirror = constants.IERS_TABLE
iers.conf.auto_max_age = None
from astropy.time import Time
from dsacalib import utils

def test_siderealtime():
    st = Time.now().sidereal_time('apparent', longitude=constants.OVRO_LON*u.rad).radian
    assert st > 0

def test_src():
    ss = utils.src('test', '0d0m0.00s', '3h0m0.00s')
    assert ss.name == 'test'
    assert ss.ra.to_value(u.rad) == 0.
    assert ss.dec.to_value(u.rad) == np.pi/4
    ss = utils.src('test', 0.*u.deg, 45.*u.deg)
    assert ss.name == 'test'
    assert ss.ra.to_value(u.rad) == 0.
    assert ss.dec.to_value(u.rad) == np.pi/4
