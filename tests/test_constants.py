import dsacalib.constants as ct
import astropy.units as u
import os
import numpy as np
import dsautils.cnf as dsc


def test_casa_location():
    ovro_lon = -118.283400*u.deg
    ovro_lat = 37.233386*u.deg
    ovro_height = 1188*u.m
    assert np.abs(ct.OVRO_LON - ovro_lon.to_value(u.rad)) < 0.1
    assert np.abs(ct.OVRO_LAT - ovro_lat.to_value(u.rad)) < 0.1
    assert np.abs(ct.OVRO_ALT - ovro_height.to_value(u.m)) < 1

def test_data():
    assert os.path.exists(ct.IERS_TABLE.replace('file://', ''))
    assert os.path.exists('{0}/template_gcal_ant'.format(ct.PKG_DATA_PATH))

def test_cnf():
    conf = dsc.Conf()
    params = conf.get('corr')
    assert 'ch0' in params.keys()
    assert 'antenna_order' in params.keys()
    corr_list = list(params['ch0'].keys())
    corr_list = [int(cl.strip('corr')) for cl in corr_list]
    antennas_plot = np.array(list(params['antenna_order'].values()))
