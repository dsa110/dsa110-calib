import pytest

import astropy.units as u
from astropy.time import Time
from astropy.utils import iers
import dsacalib.constants as ct

iers.conf.iers_auto_url_mirror = ct.iers_table


def test_sideraltime():
    st = Time.now().sidereal_time('apparent', longitude=ct.ovro_lon*u.rad).radian
    assert st > 0
