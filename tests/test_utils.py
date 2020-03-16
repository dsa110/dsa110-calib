import pytest

import astropy.units as u
from astropy.time import Time
from astropy.utils import iers
from dsacalib import utils, constants
iers.conf.iers_auto_url_mirror = constants.iers_table

def test_sideraltime():
    st = Time.now().sidereal_time('apparent', longitude=constants.ovro_lon*u.rad).radian
    assert st > 0


def test_src():
    ss = utils.src('test', 0., 45.)
    assert ss.name == 'test'
    assert src.ra.to_value(u.rad) == 0.
    assert src.dec.to_value(u.rad) == np.pi/4
