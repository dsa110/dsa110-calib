"""
Simple tests to ensure calibration files are created.
Do not test validity of solutions.
"""
import numpy as np
import astropy.units as u
import dsacalib
from dsacalib.utils import *
from dsacalib.calib import *
from dsacalib.plotting import *
from dsacalib.fringestopping import *
import os
import shutil

def __init__():
    return

def test_dsa10(tmpdir):
    datadir= '{0}/data/'.format(dsacalib.__path__[0])
    pt_dec = 49.033386*np.pi/180.
    fl     = '{0}/J0542+4951_test.fits'.format(datadir)
    cal    = src('3C147','05h42m36.137916s','49d51m07.233560s',22.8796)
    msname = '{0}/{1}'.format(tmpdir,cal.name)

    fobs, blen, bname, tstart, tstop, tsamp, vis, \
        mjd, lst, transit_idx,antenna_order = \
        read_psrfits_file(fl,cal,dur=1*u.min,antpos='{0}/antpos_ITRF.txt'
        .format(datadir),badants=[1,3,4,7,10])
    nant = len(antenna_order)

    fringestop(vis,blen,cal,mjd,fobs,pt_dec)
    amp_model = amplitude_sky_model(cal,lst,pt_dec,fobs)

    if os.path.exists('{0}.ms'.format(msname)):
        shutil.rmtree('{0}.ms'.format(msname))
    convert_to_ms(cal,vis,tstart,msname,
                           bname,antenna_order,tsamp=ct.tsamp,nint=25,
                           antpos='{0}/antpos_ITRF.txt'.format(datadir),
                           model = np.tile(amp_model[np.newaxis,:,:,np.newaxis],
                           (vis.shape[0],1,1,vis.shape[-1])))
    assert os.path.exists('{0}.ms'.format(msname))

    flag_zeros(msname)
    flag_antenna(msname,'8',pol='A')


    if os.path.exists('{0}_{1}_kcal'.format(msname,cal.name)):
        shutil.rmtree('{0}_{1}_kcal'.format(msname,cal.name))
    delay_calibration(msname,cal.name)
    assert os.path.exists('{0}_{1}_kcal'.format(msname,cal.name))
    
    bad_times,times = get_bad_times(msname,cal.name,nant)
    flag_badtimes(msname,times,bad_times,nant)

    for tbl in ['gacal','gpcal','bcal']:
        if os.path.exists('{0}_{1}_{2}'.format(msname,cal.name,tbl)):
            shutil.rmtree('{0}_{1}_{2}'.format(msname,cal.name,tbl))
    gain_calibration(msname,cal.name,tga='60s',refant='1')
    for tbl in ['gacal','gpcal','bcal']:
        assert os.path.exists('{0}_{1}_{2}'.format(msname,cal.name,tbl))

    
def test_3ant(tmpdir):
    datadir = '{0}/data/'.format(dsacalib.__path__[0])
    fl     = '{0}/M87_test.fits'.format(datadir)
    cal    = src('M87','12h30m49.4233s','+12d23m28.043s',138.4870)
    pt_dec = cal.dec.to_value(u.rad)
    utc_start = Time('2020-04-16T06:09:42')
    antenna_order = [9,2,6]
    msname = '{0}/{1}'.format(tmpdir,cal.name)

    fobs, blen, bname, tstart, tstop, tsamp, vis, \
        mjd, lst, transit_idx, antenna_order = \
        read_psrfits_file(fl,cal,antenna_order = antenna_order,
                          autocorrs=False,dur=1*u.min,
                          utc_start = utc_start,dsa10=False,
                          antpos='{0}/antpos_ITRF.txt'.format(datadir))
    (nbl,nt,nf,npol) = vis.shape
    nant = len(antenna_order)


    fringestop(vis,blen,cal,mjd,fobs,pt_dec)
    amp_model = amplitude_sky_model(cal,lst,pt_dec,fobs)

    if os.path.exists('{0}.ms'.format(msname)):
        shutil.rmtree('{0}.ms'.format(msname))
    convert_to_ms(cal,vis,mjd[0],msname,bname,antenna_order,tsamp,
                  nint=25,antpos='{0}/antpos_ITRF.txt'.format(datadir),
                  dsa10=False,model=np.tile(amp_model[np.newaxis,:,:,np.newaxis],
                                            (vis.shape[0],1,1,vis.shape[-1])))
    assert os.path.exists('{0}.ms'.format(msname))
    
    flag_zeros(msname)

    if os.path.exists('{0}_{1}_kcal'.format(msname,cal.name)):
        shutil.rmtree('{0}_{1}_kcal'.format(msname,cal.name))
    delay_calibration(msname,cal.name,refant='2')
    assert os.path.exists('{0}_{1}_kcal'.format(msname,cal.name))

    bad_times,times = get_bad_times(msname,cal.name,nant,refant='2')
    flag_badtimes(msname,times,bad_times,nant)

    for tbl in ['gacal','gpcal','bcal']:
        if os.path.exists('{0}_{1}_{2}'.format(msname,cal.name,tbl)):
            shutil.rmtree('{0}_{1}_{2}'.format(msname,cal.name,tbl))
    gain_calibration_blbased(msname,cal.name,refant='2',tga='10s')
    for tbl in ['gacal','gpcal','bcal']:
        assert os.path.exists('{0}_{1}_{2}'.format(msname,cal.name,tbl))
    



