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
    tol = 1e-8
    ss = utils.src('test', '0d0m0.00s', '3h0m0.00s')
    assert ss.name == 'test'
    assert np.abs(ss.ra.to_value(u.rad) - 0.) < tol
    assert np.abs(ss.dec.to_value(u.rad) - np.pi/4) < tol
    ss = utils.src('test', 0.*u.deg, 45.*u.deg)
    assert ss.name == 'test'
    assert np.abs(ss.ra.to_value(u.rad) - 0.) < tol
    assert np.abs(ss.dec.to_value(u.rad) - np.pi/4) < tol

def test_todeg():
    tol = 1e-8
    assert np.abs(utils.to_deg('0d0m0.00s').to_value(u.rad) - 0.) < 1e-8
    assert np.abs(utils.to_deg('3h0m0.00s').to_value(u.rad) - np.pi/4) < 1e-8
