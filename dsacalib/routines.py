"""Calibration routine for DSA-110 calibration with CASA.

Author: Dana Simard, dana.simard@astro.caltech.edu, 2020/06
"""

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
import dsautils.calstatus as cs

import dsautils.dsa_syslog as dsl 

logger = dsl.DsaSyslogger()    
logger.subsystem("software")
logger.app("dsacalib")

def __init__():
    return
    
def check_path(fname):
    assert os.path.exists(fname), \
      'File {0} does not exist'.format(fname)
    
def triple_antenna_cal(obs_params,ant_params,show_plots=False,throw_exceptions=True):
    """Performs delay and complex gain calibration for a triplet of antennas. 
    
    Uses CASA to perform baseline-based calibration of visibilties recorded using 
    the 6-input DSA-110 correlator.  Calculates the antenna gains from these
    baseline-based solutions.
    
    Parameters
    ----------
    obs_params : dictionary
        The observation parameters.  The following keys must be defined:
        
        obs_params['fname'] : str
            The full path to the fits or hdf5 file containing the visibilities.
        obs_params['msname'] : str
            The name to use for the ms containing the fringestopped 
            visibilities.
        obs_params['cal'] : src class instance
            The calibrator source.
        obs_params['utc_start'] : astropy Time instance
            The start time of the correlator run.
    ant_params : dictionary
        The antenna parameters during the observation. The following keys must 
        be defined:
        
        ant_params['antenna_order'] : list
            The names of the antennas, in the order used in the correlator.
        ant_params['refant'] : str
            The name of the referance antenna. Note that if an integer is 
            passed instead of a string, it will be treated as the index of 
            the reference antenna in the CASA MS
        ant_params['antpos'] : str
            The full path to the antenna positions file.
    show_plots : boolean
        If set to ``True``, shows plots generated during calibration 
        (e.g. if working in a notebook or interactive python environment).
        Defaults ``False``.
    throw_exceptions : boolean
        If set to ``True``, throws exceptions. If set to ``False``, handles 
        them quietly.  In both cases, exception information is written to the 
        system logs.  Defaults ``True``.
        
    Returns
    -------
    status : int 
        The status code.  A non-zero status means at least one error has occured.
        Decode status codes using dsa110-pytools (`dsatools.calstatus`).
    caltime : float
        The time of the meridian crossing of the calibrator in MJD.
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
    daz           = ant_params['azimuth_offsets']
    dha           = daz_dha(dec=pt_dec,daz=daz)

    
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
                              antpos='{0}/data/antpos_ITRF.txt'.format(
                                  dsacalib.__path__[0]))
        caltime = mjd[transit_idx]
    except Exception as e:
        status = cs.update(status,
                    ['infile_err','inv_antnum','inv_time',
                     'inv_gainamp_p1','inv_gainamp_p2','inv_gainphase_p1',
                     'inv_gainphase_p2','inv_delay_p1','inv_delay_p2',
                     'inv_gaincaltime','inv_delaycaltime'])
        caltime = Time.now().mjd
        exception_logger('opening visibility file',e,throw_exceptions)
        return status,caltime
 
    try:
        (nbl,nt,nf,npol) = vis.shape
        assert nt>0, "calibrator not in file"
    except Exception as e:
        status = cs.update(status,
                    ['cal_missing_err','inv_time','inv_gainamp_p1',
                     'inv_gainamp_p2','inv_gainphase_p1','inv_gainphase_p2',
                     'inv_delay_p1','inv_delay_p2','inv_gaincaltime',
                     'inv_delaycaltime'])
        exception_logger('verification of visibility file',e,throw_exceptions)

    try:
        nant = len(antenna_order)
        nbls = (nant*(nant+1))//2
        assert nant==3, \
            "triple_antenna_cal only works with a triplet of antennas"
        assert int(refant) in antenna_order, \
            "refant {0} not in visibilities".format(refant)
    except Exception as e:
        status = cs.update(status,
                    ['infile_format_err','inv_gainamp_p1','inv_gainamp_p2',
                     'inv_gainphase_p1','inv_gainphase_p2','inv_delay_p1',
                     'inv_delay_p2','inv_gaincaltime','inv_delaycaltime'])
        exception_logger('read and verification of visibility file',e,throw_exceptions)
        return status,caltime
    
    try: 
        maskf,fraction_flagged =  mask_bad_bins(vis,axis=2,thresh=2.0,medfilt=True,
                                      nmed=129)
        maskt,fraction_flagged =  mask_bad_bins(vis,axis=1,thresh=2.0,medfilt=True,
                                          nmed=129)
        maskp,fraction_flagged = mask_bad_pixels(vis,thresh=6.0,mask=maskt*maskf)
        mask = maskt*maskf*maskp
        vis*=mask
    except Exception as e:
        status = cs.update(status,
                          ['flagging_err'])
        exception_logger("flagging of ms data",e,throw_exceptions)
    
    # FRINGESTOP DATA
    try:
        fringestop(vis,blen,cal,mjd,fobs,pt_dec)
    except Exception as e:
        status = cs.update(status,
                    ['fringes_err','inv_gainamp_p1','inv_gainamp_p2',
                     'inv_gainphase_p1','inv_gainphase_p2','inv_delay_p1',
                     'inv_delay_p2','inv_gaincaltime','inv_delaycaltime'])
        exception_logger("fringestopping",e,throw_exceptions)
        return status,caltime
    
    # CONVERT DATA TO MS
    try:
        antenna_amp_models = [
            amplitude_sky_model(cal,lst+dha[0],pt_dec,fobs),
            amplitude_sky_model(cal,lst+dha[1],pt_dec,fobs),
            amplitude_sky_model(cal,lst+dha[2],pt_dec,fobs)]
        amp_model = np.zeros((6,lst.shape[0],fobs.shape[0]))
        k=0
        for i in range(3):
            for j in range(i,3):
                amp_model[k] = np.sqrt(
                    antenna_amp_models[i]*antenna_amp_models[j])
                k+=1
                
        #amp_model = amplitude_sky_model(cal,lst,pt_dec,fobs)
        amp_model = np.tile(amp_model[:,:,:,np.newaxis],
                            (1,1,1,vis.shape[-1]))

        convert_to_ms(cal,vis,mjd[0],'{0}'.format(msname),
                       bname,antenna_order,tsamp,nint=25,
                       antpos='/home/dsa/data/antpos_ITRF.txt',dsa10=False,
                       model=amp_model)
        check_path('{0}.ms'.format(msname))
    except Exception as e:
        status = cs.update(status,
                    ['ms_write_err','inv_gainamp_p1','inv_gainamp_p2',
                     'inv_gainphase_p1','inv_gainphase_p2','inv_delay_p1',
                     'inv_delay_p2','inv_gaincaltime','inv_delaycaltime'])
        exception_logger("write to ms",e,throw_exceptions)
        return status,caltime

    # FLAG DATA
    try:
        flag_zeros(msname)
        if 8 in antenna_order:
            flag_antenna(msname,'8',pol='A')
    except Exception as e:
        status = cs.update(status,
                          ['flagging_err'])
        exception_logger("flagging of ms data",e,throw_exceptions)
    
    # DELAY CALIBRATION
    try:
        # Antenna-based delay calibration
        error = delay_calibration(msname,cal.name,refant=refant)
        if error > 0:
            status = cs.update(status,['delay_cal_err'])
            logger.info('Non-fatal error occured in delay calibration.')
        check_path('{0}_{1}_kcal'.format(msname,cal.name))
    except Exception as e:
        status = cs.update(status,
           ['delay_cal_err','inv_gainamp_p1','inv_gainamp_p2',
            'inv_gainphase_p1','inv_gainphase_p2','inv_delay_p1','inv_delay_p2',
            'inv_gaincaltime','inv_delaycaltime'])
        exception_logger("delay calibration",e,throw_exceptions)
        return status,caltime
    
    # FLAG DATA 
    try:
        bad_times,times,error = get_bad_times(msname,cal.name,nant,refant=refant)
        if error > 0:
            status = cs.update(status,['flagging_err'])
            logger.info('Non-fatal error occured in calculation of delays on short timescales.')

        times, a_delays, kcorr = plot_antenna_delays(
                msname,cal.name,antenna_order,
                outname="./figures/{0}_{1}".format(msname,cal.name),
                show=show_plots)
        error = flag_badtimes(msname,times,bad_times,nant)
        if error > 0:
            status = cs.update(status,['flagging_err'])
            logger.info('Non-fatal error occured in flagging of bad timebins')
        check_path('{0}_{1}_2kcal'.format(msname,cal.name))
    except Exception as e:
        status = cs.update(status,['flagging_err'])
        exception_logger("flagging of ms data",e,throw_exceptions)
    
    # GAIN CALIBRATION - BASELINE BASED
    try:
        error = gain_calibration_blbased(msname,cal.name,refant=refant,
                                 tga='30s',tgp='inf')
        if error > 0:
            status = cs.update(status,['gain_bp_cal_err'])
            logger.info('Non-fatal error occured in gain/bandpass calibration.')
        for fname in ['{0}_{1}_bcal'.format(msname,cal.name),
                      '{0}_{1}_gpcal'.format(msname,cal.name),
                      '{0}_{1}_gacal'.format(msname,cal.name)]:
            check_path(fname)
    except Exception as e:
        status = cs.update(status,
                    ['gain_bp_cal_err','inv_gainamp_p1','inv_gainamp_p2',
                     'inv_gainphase_p1','inv_gainphase_p2',
                     'inv_gaincaltime'])
        exception_logger("baseline-based bandpass or gain calibration",e,throw_exceptions)
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
        exception_logger("plotting gain calibration solutions",e,throw=False)
   
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
        status = cs.update(status,
                    ['gain_bp_cal_err','inv_gainamp_p1','inv_gainamp_p2',
                     'inv_gainphase_p1','inv_gainphase_p2',
                     'inv_gaincaltime'])
        exception_logger("calculation of antenna gains",e,throw_exceptions)
        return status, caltime

    return status,caltime


def calibration_master(obs_params,ant_params,show_plots=False,write_to_etcd=False,
                      throw_exceptions=True):
    """Calibrates data and writes the solutions to etcd.
    
    Parameters
    ----------
    obs_params : dictionary
        The observation parameters.  The following keys must be defined:
        
        obs_params['fname'] : str
            The full path to the fits or hdf5 file containing the visibilities.
        obs_params['msname'] : str
            The name to use for the ms containing the fringestopped 
            visibilities.
        obs_params['cal'] : src class instance
            The calibrator source.
        obs_params['utc_start'] : astropy Time instance
            The start time of the correlator run.
    ant_params : dictionary
        The antenna parameters during the observation. The following keys must 
        be defined:
        
        ant_params['antenna_order'] : list
            The names of the antennas, in the order used in the correlator.
        ant_params['refant'] : str
            The name of the referance antenna. Note that if an integer is 
            passed instead of a string, it will be treated as the index of 
            the reference antenna in the CASA MS
        ant_params['antpos'] : str
            The full path to the antenna positions file.
    show_plots : boolean
        If set to ``True``, shows plots generated during calibration 
        (e.g. if working in a notebook or interactive python environment).
        Defaults ``False``.
    write_to_etcd : boolean
        If set to ``True``, the calibration tables are read in and written 
        to etcd after calibrating is complete.  Defaults ``False``.
    throw_exceptions : boolean
        If set to ``True``, throws exceptions. If set to ``False``, handles 
        them quietly.  In both cases, exception information is written to the 
        system logs.  Defaults ``True``.
    
    Returns
    -------
    status : int 
        The status code.  A non-zero status means at least one error has occured.
        Decode statuses using dsa110-pytools (`dsatools.calstatus`).
    """
    if throw_exceptions is None:
        throw_exceptions = not write_to_etcd
    logger.info('Beginning calibration of ms {0}.ms (start time {1}) using source {2}'.
                format(obs_params['msname'],obs_params['utc_start'].isot,
                       obs_params['cal'].name))
    status,caltime = triple_antenna_cal(obs_params,ant_params,show_plots,throw_exceptions)
    logger.info('Ending calibration of ms {0}.ms (start time {1}) using source {2} with status {3}'.
               format(obs_params['msname'],obs_params['utc_start'].isot,
                       obs_params['cal'].name,status))
    print('Status: {0}'.format(cs.decode(status)))
    if write_to_etcd:
        caltable_to_etcd(obs_params['msname'],obs_params['cal'].name,
                     ant_params['antenna_order'],caltime, status,baseline_cal=True)
    return status



