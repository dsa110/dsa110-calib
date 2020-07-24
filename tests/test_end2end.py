"""
Simple tests to ensure calibration files are created. Do not test validity of
solutions.
"""
import numpy as np
import astropy.units as u
import dsacalib.routines as dr
from dsacalib.utils import src
import dsacalib
from astropy.time import Time

def __init__():
    return

def test_dsa10(tmpdir):
    datadir = '{0}/data/'.format(dsacalib.__path__[0])
    pt_dec = 49.033386*np.pi/180.
    fname = '{0}/J0542+4951_test.fits'.format(datadir)
    cal = src('3C147', '05h42m36.137916s', '49d51m07.233560s', 22.8796)
    msname = '{0}/{1}'.format(tmpdir, cal.name)
    badants = [1, 3, 4, 7, 10]
    antpos = '{0}/antpos_ITRF.txt'.format(datadir)
    refant = '2'
    dr.dsa10_cal(fname, msname, cal, pt_dec, antpos, refant, badants)
    
def test_3ant(tmpdir):
    M87 = src('M87','12h30m49.4233s','+12d23m28.043s',138.4870)
    obs_params = {'fname': '{0}/data/{1}_test.fits'.format(dsacalib.__path__[0],
                                                           M87.name),
                  'msname': '{0}/{1}'.format(tmpdir, M87.name),
                  'cal': M87,
                  'utc_start': Time('2020-04-16T06:09:42')}
    ant_params = {'pt_dec': (12.391123*u.deg).to_value(u.rad),
                  'antenna_order': [9, 2, 6],
                  'refant': '2',
                  'antpos': '{0}/data/antpos_ITRF.txt'.format(
                      dsacalib.__path__[0])}
    status, caltime = dr.triple_antenna_cal(obs_params, ant_params)
    assert status == 0

def test_3ant_sefd(tmpdir):
    M87 = src('M87', '12h30m49.4233s', '+12d23m28.043s', 138.4870)
    obs_params = {'fname': '{0}/data/{1}_test.fits'.format(
        dsacalib.__path__[0], M87.name),
                  'msname': '{0}'.format(M87.name),
                  'cal': M87,
                  'utc_start': Time('2020-04-16T06:09:42')}
    ant_params = {'pt_dec': (12.391123*u.deg).to_value(u.rad),
                  'antenna_order': [9, 2, 6],
                  'refant': '2',
                  'antpos': '{0}/data/antpos_ITRF.txt'.format(
                      dsacalib.__path__[0])}
    status, caltime = dr.triple_antenna_cal(obs_params, ant_params, sefd=True)
    assert status == 0
    # Current file is too short to fit the SEFD
    # sefds, ant_gains, ant_transit_time = calculate_sefd(obs_params, ant_params)

