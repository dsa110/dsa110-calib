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

import dsautils.dsa_syslog as dsl 

logger = dsl.DsaSyslogger()    
logger.subsystem("software")
logger.app("dsacalib")

def __init__():
    return

def exception_logger(task):
    logger.error('Exception occured during {0}')#,exc_info=True)
    
def check_path(fname):
    assert os.path.exists(fname), \
      'File {0} does not exist'.format(fname)
    
def triple_antenna_cal(obs_params,ant_params,show_plots=False):
    """ May 20th:
    Turn assert statements into error codes
    Change the etcd writing to match the new specifications
    """
    status = 0
    
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
    
    # READ IN DATA 
    try:
        fobs, blen, bname, tstart, tstop, tsamp, vis, mjd, lst, \
            transit_idx, antenna_order = \
            read_psrfits_file(fname,cal,antenna_order = antenna_order,
                              autocorrs=False,dur=10*u.min,
                              utc_start = utc_start,dsa10=False,
                              antpos='{0}/data/antpos_ITRF.txt'.format(dsacalib.__path__[0]))
        caltime = mjd[transit_idx]
        (nbl,nt,nf,npol) = vis.shape
        nant = len(antenna_order)
        nbls = (nant*(nant+1))//2
        assert nant==3, \
            "triple_antenna_cal only works with a triplet of antennas"
        assert int(refant) in antenna_order, \
            "refant {0} not in visibilities".format(refant)
    except Exception as e:
        # 1 - input file is unable to be read in 

        status = 1
        exception_logger('read and verification of visibility file')
        return status,None
    
    # FRINGESTOP DATA
    try:
        fringestop(vis,blen,cal,mjd,fobs,pt_dec)
    except Exception as e:
        # 2 - error in fringestopping
        status = 2
        exception_logger("fringestopping")
        return status,caltime
    
    # CONVERT DATA TO MS
    try:
        amp_model = amplitude_sky_model(cal,lst,pt_dec,fobs)
        amp_model = np.tile(amp_model[np.newaxis,:,:,np.newaxis],
                            (vis.shape[0],1,1,vis.shape[-1]))

        convert_to_ms(cal,vis,mjd[0],'{0}'.format(msname),
                       bname,antenna_order,tsamp,nint=25,
                       antpos='/home/dsa/data/antpos_ITRF.txt',dsa10=False,
                       model=amp_model)
        check_path('{0}.ms'.format(msname))
    except Exception as e:
        # 3 - error in write to ms
        status = 3
        exception_logger("write to ms")
        return status,caltime

    # FLAG DATA
    try:
        flag_zeros(msname)
        if 8 in antenna_order:
            flag_antenna(msname,'8',pol='A')
    except Exception as e:
        # 4 - error in flagger
        status = 4
        exception_logger("flagging of ms data")
        return status,caltime
    
    # DELAY CALIBRATION
    try:
        # Antenna-based delay calibration
        delay_calibration(msname,cal.name,refant=refant)
        check_path('{0}_{1}_kcal'.format(msname,cal.name))
    except Exception as e:
        # 5 - delay calibration
        status = 5
        exception_logger("delay calibration")
        return status,caltime
    
    # FLAG DATA 
    try:
        bad_times,times = get_bad_times(msname,cal.name,nant,refant=refant)
        times, a_delays, kcorr = plot_antenna_delays(
                msname,cal.name,antenna_order,
                outname="./figures/{0}_{1}".format(msname,cal.name),
                show=show_plots)
        flag_badtimes(msname,times,bad_times,nant)
        check_path('{0}_{1}_2kcal'.format(msname,cal.name))
    except Exception as e:
        #4 - error in flagger
        status = 4
        exception_logger("flagging of ms data")
        return status,caltime
    
    # GAIN CALIBRATION - BASELINE BASED
    try:
        gain_calibration_blbased(msname,cal.name,refant=refant,
                                 tga='10s')
        for fname in ['{0}_{1}_bcal'.format(msname,cal.name),
                      '{0}_{1}_gpcal'.format(msname,cal.name),
                      '{0}_{1}_gacal'.format(msname,cal.name)]:
            check_path(fname)
    except Exception as e:
        # 5 - gain calibration
        status = 5
        exception_logger("baseline-based bandpass or gain calibration")
        return status,caltime
    
    # PLOT GAIN CALIBRATION SOLUTIONS
    try:
        tamp,gamp,gphase,bname,t0=plot_gain_calibration(
                msname,cal.name,antenna_order,blbased=True,
                outname="./figures/{0}_{1}".format(msname,cal.name),
                show=show_plots)
        bpass = plot_bandpass(
                cal.name,cal.name,antenna_order,fobs,blbased=True,
                outname="./figures/{0}_{1}".format(msname,cal.name),
                show=show_plots)
    except Exception as e:
        # Not a fatal error
        exception_logger("plotting gain calibration solutions")
   
    # CALCULATE ANTENNA GAINS
    try:
        tamp,gamp,famp = read_caltable('{0}_{1}_gacal'.format(msname,cal.name),
                                      nbls,cparam=True)
        tphase,gphase,fphase = read_caltable('{0}_{1}_gpcal'.format(msname,cal.name),
                                            nbls,cparam=True)
        gains = gamp.T*gphase.T
        flags = famp.T*fphase.T
        gains,flags = fill_antenna_gains(gains,flags)
        mask = np.ones(flags.shape)
        mask[flags==1]=np.nan
        gains = np.nanmedian(gains*mask,axis=1,keepdims=True)
        flags = np.min(flags,axis=1,keepdims=True)

        if 8 in antenna_order:
            flags[...,0]=1

        shutil.copytree('{0}/data/template_gcal_ant'.format(
            dsacalib.__path__[0]),
            '{0}_{1}_gcal_ant'.format(msname,cal.name))

        # Write out a new gains table
        tb = cc.table()
        tb.open('{0}_{1}_gcal_ant'.format(msname,cal.name),nomodify=False)
        tb.putcol('TIME',
                  np.ones(6)*np.median(mjd)*ct.seconds_per_day)
        tb.putcol('FLAG',flags.T)
        tb.putcol('CPARAM',gains.T)
        tb.close()
        check_path('{0}_{1}_gcal_ant'.format(msname,cal.name))
    except Exception as e:
        # 5 - gain and bandpass calibration
        status = 5
        exception_logger("calculation of antenna gains")
        return status, caltime

    return status,caltime

def calibration_master(obs_params,ant_params,show_plots=False):
    logger.info('Beginning calibration of ms {0}.ms (start time {1}) using source {2}'.
                format(obs_params['msname'],obs_params['utc_start'].isot,
                       obs_params['cal'].name))
    status,caltime = triple_antenna_cal(obs_params,ant_params,show_plots)
    logger.info('Ending calibration of ms {0}.ms (start time {1}) using source {2} with status {3}'.
               format(obs_params['msname'],obs_params['utc_start'].isot,
                       obs_params['cal'].name,status))
    caltable_to_etcd(obs_params['msname'],obs_params['cal'].name,
                     ant_params['antenna_order'],caltime, status,baseline_cal=True)


