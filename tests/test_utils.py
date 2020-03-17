import pytest

import astropy.units as u
from astropy.time import Time
from astropy.utils import iers
from dsacalib import utils, constants
import numpy as np
iers.conf.iers_auto_url_mirror = constants.iers_table

def test_sideraltime():
    st = Time.now().sidereal_time('apparent', longitude=constants.ovro_lon*u.rad).radian
    assert st > 0


def test_src():
    ss = utils.src('test', '0d0m0.00s', '3h0m0.00s')
    assert ss.name == 'test'
    assert ss.ra.to_value(u.rad) == 0.
    assert ss.dec.to_value(u.rad) == np.pi/4
    ss = utils.src('test', 0., 45.)
    assert ss.name == 'test'
    assert ss.ra.to_value(u.rad) == 0.
    assert ss.dec.to_value(u.rad) == np.pi/4
