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
    
def triple_antenna_cal(obs_params,ant_params,show_plots=False,
                       throw_exceptions=True,sefd=False):
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
        caldur = 60*u.min if sefd else 10*u.min
        fobs, blen, bname, tstart, tstop, tsamp, vis, mjd, lst, \
            transit_idx, antenna_order = \
            read_psrfits_file(fname,cal,antenna_order = antenna_order,
                              autocorrs=False,dur=caldur,
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
    
    # Flag data
    # Ideal thresholds?
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
        amp_model = amplitude_sky_model(cal,lst,pt_dec,fobs)
        amp_model = np.tile(amp_model[np.newaxis,:,:,np.newaxis],
                            (vis.shape[0],1,1,vis.shape[-1]))
        convert_to_ms(cal,vis,mjd[0],'{0}'.format(msname),
                       bname,antenna_order,tsamp,nint=25,
                       antpos='/home/dsa/data/antpos_ITRF.txt',dsa10=False,
                       model=None if sefd else amp_model)
        check_path('{0}.ms'.format(msname))
    except Exception as e:
        status = cs.update(status,
                    ['write_ms_err','inv_gainamp_p1','inv_gainamp_p2',
                     'inv_gainphase_p1','inv_gainphase_p2','inv_delay_p1',
                     'inv_delay_p2','inv_gaincaltime','inv_delaycaltime'])
        exception_logger("write to ms",e,throw_exceptions)
        return status,caltime

    # FLAG DATA
    try:
        error = flag_zeros(msname)
        if error > 0:
            status = cs.update(status,['flagging_err'])
            logger.info("Non-fatal error in zero flagging")
        if 8 in antenna_order:
            error = flag_antenna(msname,'8',pol='A')
            if error > 0:
                status = cs.update(status,['flagging_err'])
                logger.info("Non-fatal error in antenna 8 flagging")
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
        status = cs.update(status,'flagging_err')
        exception_logger("flagging of ms data",e,throw_exceptions)

    # GAIN CALIBRATION - BASELINE BASED
    try:
        print(os.path.exists('J1042+1203_1_J1042+1203_gacal'))
        error = gain_calibration_blbased(msname,cal.name,refant=refant,
                                 tga='30s',tgp='30s')
        if error > 0:
            status = cs.update(status,['gain_bp_cal_err'])
            logger.info('Non-fatal error occured in gain/bandpass calibration.')
        print(os.path.exists('J1042+1203_1_J1042+1203_gacal'))
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
        print(gamp.shape,type(gamp),gamp.dtype)
        tphase,gphase,fphase = read_caltable('{0}_{1}_gpcal'.format(msname,cal.name),
                                            nbls,cparam=True)
        gains = gamp.T*gphase.T
        flags = famp.T*fphase.T
        gains,flags = fill_antenna_gains(gains,flags)
        
        gx = np.abs(gains).T.astype(np.complex128)
        gx = gx.reshape(gx.shape[0],-1)
        print(gx.shape,type(gx),gx.dtype)
        tb = cc.table()
        tb.open('{0}_{1}_gacal'.format(msname,cal.name),
                     nomodify=False)
        tb.putcol('CPARAM',gx)
        tb.close()
        
        gx = np.exp(1.j*np.angle(gains).T)
        gx = gx.reshape(gx.shape[0],-1)
        tb.open('{0}_{1}_gpcal'.format(msname,cal.name),
                     nomodify=False)
        tb.putcol('CPARAM',gx)
        tb.close()
        
        if not sefd:
            # reduce to a single value to use 
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
                      throw_exceptions=None,sefd=False):
    if throw_exceptions is None:
        throw_exceptions = not write_to_etcd
    logger.info('Beginning calibration of ms {0}.ms (start time {1}) using source {2}'.
                format(obs_params['msname'],obs_params['utc_start'].isot,
                       obs_params['cal'].name))
    status,caltime = triple_antenna_cal(obs_params,ant_params,show_plots,throw_exceptions,sefd)
    logger.info('Ending calibration of ms {0}.ms (start time {1}) using source {2} with status {3}'.
               format(obs_params['msname'],obs_params['utc_start'].isot,
                       obs_params['cal'].name,status))
    print('Status: {0}'.format(cs.decode(status)))
    print('')
    if write_to_etcd:
        caltable_to_etcd(obs_params['msname'],obs_params['cal'].name,
                     ant_params['antenna_order'],caltime, status,baseline_cal=True)
    return status

from scipy.optimize import curve_fit
from dsacalib.utils import daz_dha

def gauss(x, a, x0, sigma,c):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))+c

def get_delay_bp_cal_vis(msname,calname,nbls):
    error = 0
    cb = cc.calibrater()
    error += not cb.open('{0}.ms'.format(msname))
    error += not cb.setapply(type='K',table='{0}_{1}_kcal'.
                             format(msname,calname))
    error += not cb.setapply(type='MF',table='{0}_{1}_bcal'.
                            format(msname,calname))
    error += not cb.correct()
    error += not cb.close()
    if error > 0:
        print('{0} errors occured during calibration'.format(error))
    vis_uncal,vis_cal,time,freq,flag=extract_vis_from_ms(msname,nbls)
    mask = (1-flag).astype(float)
    mask[mask<0.5] = np.nan
    vis_cal = vis_cal*mask
    return vis_cal,time,freq

def calculate_sefd(obs_params,ant_params,
                   fmin=1.35,fmax=1.45,baseline_cal=True,
                  showplots=False):
    # Change so figures saved if showplots is False
    status = 0
    
    fname         = obs_params['fname']
    msname        = obs_params['msname']
    cal           = obs_params['cal']
    utc_start     = obs_params['utc_start']
    pt_dec        = ant_params['pt_dec']
    antenna_order = ant_params['antenna_order']
    refant        = ant_params['refant']
    antpos        = ant_params['antpos']
    nant = len(antenna_order)
    nbls = (nant*(nant+1))//2
    npol = 2
    
    # Get the visibilities (for autocorrs)
    vis,tvis,fvis = get_delay_bp_cal_vis(msname,cal.name,nbls)

    #vis = np.mean(vis,axis=2)
    
    
    # Open the gain files and read in the gains
    time,gain,flag = read_caltable('{0}_{1}_gacal'.format(
                       msname,cal.name),nbls,cparam=True)
    gain = np.abs(gain) # Should already be purely real
    autocorr_idx = get_autobl_indices(nant,casa=True)
    #autocorr_idx = [(nbls-1)-aidx for aidx in autocorr_idx]
    if baseline_cal:
        time = time[...,autocorr_idx[0]]
        gain = gain[...,autocorr_idx]     
        flag = flag[...,autocorr_idx]
    vis = vis[autocorr_idx,...]
    idxl = np.searchsorted(fvis,fmin)
    idxr = np.searchsorted(fvis,fmax)
    vis = vis[...,idxl:idxr,:]
    vis = np.abs(vis)
    
    # assert np.all(vis.imag/vis.real < 1e-3)

    # Complex gain includes an extra relative delay term
    # in the phase, but we really only want the amplitude
    # We will ignore the phase for now
    
    ant_gains_on        = np.zeros((nant,npol))
    eant_gains_on       = np.zeros((nant,npol))
    ant_gains_off_meas  = np.zeros((nant,npol))
    eant_gains_off_meas = np.zeros((nant,npol))
    ant_transit_time    = np.zeros((nant,npol))
    eant_transit_time   = np.zeros((nant,npol))
    ant_transit_width   = np.zeros((nant,npol))
    eant_transit_width  = np.zeros((nant,npol))
    offbins_before      = np.zeros((nant,npol),dtype=int)
    offbins_after       = np.zeros((nant,npol),dtype=int)
    autocorr_gains_off  = np.zeros((nant,npol))
    
    if showplots:
        fig,ax = plt.subplots(1,3,figsize=(8*3,8*1))
        ccyc=plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Fit a Gaussian to the gains 
    for i in range(nant):
        for j in range(npol):
            if showplots:
                ax[i].plot(time-time[0],gain[j,:,i],
                           '.',color=ccyc[j])
            initial_params = [np.max(gain[j,:,i]),
                             (time[-1]-time[0])/2,
                             0.0035, #all should be similar
                             0]
            params, cov = curve_fit(gauss,time-time[0],
                                    gain[j,:,i],
                                    p0=initial_params)

            ant_gains_on[i,j] = params[0]+params[3]
            eant_gains_on[i,j] = np.sqrt(cov[0,0])+np.sqrt(cov[3,3]) 
            
            ant_transit_time[i,j] = time[0]+params[1]
            eant_transit_time[i,j] = np.sqrt(cov[1,1])
            ant_transit_width[i,j] = params[2]
            eant_transit_width[i,j] = np.sqrt(cov[2,2])
            
            offbins_before[i,j] = np.searchsorted(time,
                                    ant_transit_time[i,j]-ant_transit_width[i,j]*3)
            offbins_after[i,j]  = len(time) - np.searchsorted(time,
                                    ant_transit_time[i,j]+ant_transit_width[i,j]*3)
            #Need an automatic way to fit toffidx
            #ant_gains_off_meas[i,j] = np.mean(np.concatenate(
            #    (gain[j,:offbins_before[i,j],i],gain[j,offbins_after[i,j]:,i]),
            #    axis=0))
            #eant_gains_off_meas[i,j] = np.std(np.concatenate(
            #    (gain[j,:offbins_before[i,j],i],gain[j,offbins_after[i,j]:,i]),
            #    axis=0),ddof=1)
            
            # The array of time is going to be different - can I get that
            idxl = np.searchsorted(tvis, ant_transit_time[i,j]-ant_transit_width[i,j]*3)
            idxr = np.searchsorted(tvis, ant_transit_time[i,j]-ant_transit_width[i,j]*3)
            autocorr_gains_off[i,j] = np.nanmean(np.concatenate((vis[i,:idxl,:,j],vis[i,idxr:,:,j]),
                                                     axis=0))
            print(params[3],autocorr_gains_off[i,j])
            print('{0} {1}: on {2:.2f}'.format(
                antenna_order[i],'A' if j==0 else 'B',
                ant_gains_on[i,j]))#,ant_gains_off_meas[i,j]))
            if showplots:
                ax[i].plot(time-time[0],gauss(time-time[0],*params),'-',color=ccyc[j],
                      label='{0} {1}: {2:.4f}/Jy'.format(
                          antenna_order[i],'A' if j==0 else 'B',
                    (ant_gains_on[i,j]#-ant_gains_off_meas[i,j]
                    )/cal.I))

                ax[i].legend()
                ax[i].set_xlabel("Time (d)")
                ax[i].set_ylabel("Unnormalized power")
    # Change to meas
    ant_gains = (ant_gains_on# - ant_gains_off_meas
                )/cal.I
    
    # How do we get the autocorr_gains_off here??
    # Do we have to read in the casa ms and calculate this?
    # Does casa give the autocorr gains at all? 
    sefds     = autocorr_gains_off/ant_gains
    return sefds, ant_gains, ant_transit_time