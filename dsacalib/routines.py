import numpy as np
import astropy.units as u
from dsacalib.utils import *
from dsacalib.calib import * 
from dsacalib.plotting import *
from dsacalib.fringestopping import *
from astropy.time import Time
import shutil
import os
import casatools as cc
import dsacalib

def triple_antenna_cal(obs_params,ant_params):
    
    fname         = obs_params['fname']
    msname        = obs_params['msname']
    cal           = obs_params['cal']
    utc_start     = obs_params['utc_start']
    pt_dec        = ant_params['pt_dec']
    antenna_order = ant_params['antenna_order']
    refant        = ant_params['refant']
    antpos        = ant_params['antpos']
    
    # Remove files that we will create so that things will fail
    # if casa doesn't write a table
    casa_dirnames = ['{0}.ms'.format(msname),
                    '{0}_{1}_kcal'.format(msname,cal.name),
                    '{0}_{1}_2kcal'.format(msname,cal.name),
                    '{0}_{1}_bcal'.format(msname,cal.name),
                    '{0}_{1}_gpcal'.format(msname,cal.name),
                    '{0}_{1}_gacal'.format(msname,cal.name),
                    '{0}_{1}_gcal_ant'.format(msname,cal.name)]
    for dirname in casa_dirnames:
        if os.path.exists(dirname):
            shutil.rmtree(dirname)
    
    fobs, blen, bname, tstart, tstop, tsamp, vis, mjd, lst, \
        transit_idx, antenna_order = \
        read_psrfits_file(fname,cal,antenna_order = antenna_order,
                          autocorrs=False,dur=10*u.min,
                          utc_start = utc_start,dsa10=False,
                          antpos='{0}/data/antpos_ITRF.txt'.format(dsacalib.__path__[0]))
    (nbl,nt,nf,npol) = vis.shape
    nant = len(antenna_order)

    assert nant==3, \
        "triple_antenna_cal only works with a triplet of antennas"
    
    fringestop(vis,blen,cal,mjd,fobs,pt_dec)
    amp_model = amplitude_sky_model(cal,lst,pt_dec,fobs)
    amp_model = np.tile(amp_model[np.newaxis,:,:,np.newaxis],
                        (vis.shape[0],1,1,vis.shape[-1]))
        
    convert_to_ms(cal,vis,mjd[0],'{0}'.format(msname),
                   bname,antenna_order,tsamp,nint=25,
                   antpos='/home/dsa/data/antpos_ITRF.txt',dsa10=False,
                   model=amp_model)
 
    ### Calibrate ###
    flag_zeros(msname)
    
    # Antenna-based delay calibration
    delay_calibration(msname,cal.name,refant=refant)
    bad_times,times = get_bad_times(msname,cal.name,nant,refant=refant)
    times, a_delays, kcorr = plot_antenna_delays(
            msname,cal.name,antenna_order,
            outname="./figures/{0}_{1}".format(msname,cal.name),
            show=False)
    flag_badtimes(msname,times,bad_times,nant)
    
    # Baseline-based gain calibration
    gain_calibration_blbased(msname,cal.name,refant=refant,
                             tga='10s')
    tamp,gamp,gphase,bname,t0=plot_gain_calibration(
            msname,cal.name,antenna_order,blbased=True,
            outname="./figures/{0}_{1}".format(msname,cal.name),
            show=False)
    bpass = plot_bandpass(
            cal.name,cal.name,antenna_order,fobs,blbased=True,
            outname="./figures/{0}_{1}".format(msname,cal.name),
            show=False)
   
    # Calculate antenna gains
    gains = gamp.T*gphase.T
    gains = fill_antenna_gains(gains)

    shutil.copytree('{0}/data/template_gcal_ant'.format(
        dsacalib.__path__[0]),
        '{0}_{0}_gcal_ant'.format(cal.name))

    # Write out a new gains table
    tb = cc.table()
    tb.open('{0}_{0}_gcal_ant'.format(cal.name),nomodify=False)
    tb.putcol('TIME',
              np.ones(6)*np.median(mjd)*ct.seconds_per_day)
    tb.putcol('CPARAM',np.median(gains.T,axis=1,keepdims=True))
    tb.close()
    
    for dirname in casa_dirnames:
        assert os.path.exists(dirname), \
          "{0} is missing".format(dirname)
    
