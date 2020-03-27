"""
CALIB.PY

Dana Simard, dana.simard@astro.caltech.edu, 10/2019

Casa-based routines to calibrate visibilities using point sources
"""

import casatools as cc
import astropy.units as u
import numpy as np
import astropy.constants as c
from . import constants as ct
from scipy.fftpack import fft,fftshift,fftfreq

def delay_calibration(msname,sourcename,refant='0',t='inf'):
    """Calibrate delays using CASA and write the calibrated 
    visibilities to the corrected_data column of the measurement set
    
    Args:
        msname: str
          the name of the measurement set (will open <msname>.ms)
        sourcename: str
          the name of the calibrator source
          the calibration table will be written to <msname>_<sourcename>_kcal
        refant: int
          the reference antenna
        t: str
          a CASA-understood time to integrate by. e.g. 'inf' or '60s'

    Returns:
    """
    error = 0
    cb = cc.calibrater()
    error += not cb.open('{0}.ms'.format(msname))
    error += not cb.setsolve(type='K',t=t,
            refant=refant,table='{0}_{1}_kcal'.format(msname,sourcename))
    error += not cb.solve()
    error += not cb.setapply(type='K',table='{0}_{1}_kcal'.format(msname,
                                                                 sourcename))
    error += not cb.correct()
    error += not cb.close()
    if error > 0:
        print('{0} errors occured during calibration'.format(error))
    return

def gain_calibration_blbased(msname,sourcename,tga='600s',tgp='inf',
                            refant='0'):
    """Use Self-Cal to calibrate bandpass and complex gain solutions. 
    Saves solutions to calibration tables.
    Calibrates the measurement set by applying delay, bandpass, 
    and complex gain solutions.
    
    Args:
        msname: str
          the measurement set.  will open <msname>.ms
        sourcename: str
          the name of the calibrator source
          the calibration table will be written to <msname>_<sourcename>_kcal
        tga: str
          a CASA-understood time to integrate by. e.g. 'inf' or '60s'
          the integration time for the amplitude gain solutions
        tgp: str
          a CASA-understood time to integrate by. e.g. 'inf' or '60s'
          the integration time for the phase gain solutions
        refant: str
          the name of the reference antenna to use in calibration
          
    Returns:
    """
    error = 0
    
    # Solve for bandpass calibration
    cb = cc.calibrater()
    error += not cb.open('{0}.ms'.format(msname))
    error += not cb.setapply(type='K',table='{0}_{1}_kcal'.
                             format(msname,sourcename))
    error += not cb.setsolve(type='MF',table='{0}_{1}_bcal'.
                             format(msname,sourcename),
                             refant=refant,apmode='a')#,solnorm=True)
                            #solint='651.04167kHz')
    error += not cb.solve()
    error += not cb.close()
    
    # Solve for phase calibration
    cb = cc.calibrater()
    error += not cb.open('{0}.ms'.format(msname))
    error += not cb.setapply(type='K',table='{0}_{1}_kcal'.
                            format(msname,sourcename))
    error += not cb.setapply(type='MF',table='{0}_{1}_bcal'.
                            format(msname,sourcename))
    error += not cb.setsolve(type='M',table='{0}_{1}_gpcal'.
                            format(msname,sourcename),t=tgp,
                            refant=refant,apmode='p')
    error += not cb.solve()
    error += not cb.close()

    # Solve for gain calibration on 10 minute timescale
    cb = cc.calibrater()
    error += not cb.open('{0}.ms'.format(msname))
    error += not cb.setapply(type='K',table='{0}_{1}_kcal'.
                             format(msname,sourcename))
    error += not cb.setapply(type='MF',table='{0}_{1}_bcal'.
                            format(msname,sourcename))
    error += not cb.setapply(type='M',table='{0}_{1}_gpcal'.
                             format(msname,sourcename))
    error += not cb.setsolve(type='M',table='{0}_{1}_gacal'.
                             format(msname,sourcename), t=tga,
                             refant=refant,apmode='a')
    error += not cb.solve()
    error += not cb.close()

    # Apply calibration
    cb = cc.calibrater()
    error += not cb.open('{0}.ms'.format(msname))
    error += not cb.setapply(type='K',table='{0}_{1}_kcal'.
                             format(msname,sourcename))
    error += not cb.setapply(type='MF',table='{0}_{1}_bcal'.
                            format(msname,sourcename))
    error += not cb.setapply(type='M',table='{0}_{1}_gpcal'.
                             format(msname,sourcename))
    error += not cb.setapply(type='M',table='{0}_{1}_gacal'.
                             format(msname,sourcename))
    error += not cb.correct()
    error += not cb.close()
    if error > 0:
        print('{0} errors occured during calibration'.format(error))
    return    

def gain_calibration(msname,sourcename,tga='600s',tgp='inf',
                     refant='0'):
    """Use Self-Cal to calibrate bandpass and complex gain solutions. 
    Saves solutions to calibration tables.
    Calibrates the measurement set by applying delay, bandpass, 
    and complex gain solutions.
    
    Args:
        msname: str
          the measurement set.  will open <msname>.ms
        sourcename: str
          the name of the calibrator source
          the calibration table will be written to <msname>_<sourcename>_kcal
        tga: str
          a CASA-understood time to integrate by. e.g. 'inf' or '60s'
          the integration time for the amplitude gain solutions
        tgp: str
          a CASA-understood time to integrate by. e.g. 'inf' or '60s'
          the integration time for the phase gain solutions
        refant: str
          the name of the reference antenna to use in calibration
          
    Returns:
    """
    error = 0

    cb = cc.calibrater()
    error += not cb.open('{0}.ms'.format(msname))
    error += not cb.setapply(type='K',table='{0}_{1}_kcal'.
               format(msname,sourcename))
    error += not cb.setsolve(type='B',table='{0}_{1}_bcal'.format(msname,sourcename),
           refant=refant,apmode='a',solnorm=True)
    error += not cb.solve()
    error += not cb.close()
    
    # Solve for phase calibration over entire obs
    cb = cc.calibrater()
    error += not cb.open('{0}.ms'.format(msname))
    error += not cb.setapply(type='K',table='{0}_{1}_kcal'.
                             format(msname,sourcename))
    error += not cb.setapply(type='B',table='{0}_{1}_bcal'.
                            format(msname,sourcename))
    error += not cb.setsolve(type='G',table='{0}_{1}_gpcal'.
                             format(msname,sourcename),t=tgp,
                     minblperant=1,refant=refant,apmode='p')
    error += not cb.solve()
    error += not cb.close()

    # Solve for gain calibration on 10 minute timescale
    cb = cc.calibrater()
    error += not cb.open('{0}.ms'.format(msname))
    error += not cb.setapply(type='K',table='{0}_{1}_kcal'.
                             format(msname,sourcename))
    error += not cb.setapply(type='B',table='{0}_{1}_bcal'.
                            format(msname,sourcename))
    error += not cb.setapply(type='G',table='{0}_{1}_gpcal'.
                             format(msname,sourcename))
    error += not cb.setsolve(type='G',table='{0}_{1}_gacal'.
                             format(msname,sourcename), t=tga,
                     minblperant=1,refant=refant,apmode='a')
    error += not cb.solve()
    error += not cb.close()

    # Apply calibration
    cb = cc.calibrater()
    error += not cb.open('{0}.ms'.format(msname))
    error += not cb.setapply(type='K',table='{0}_{1}_kcal'.
                             format(msname,sourcename))
    error += not cb.setapply(type='B',table='{0}_{1}_bcal'.
                            format(msname,sourcename))
    error += not cb.setapply(type='G',table='{0}_{1}_gpcal'.
                             format(msname,sourcename))
    error += not cb.setapply(type='G',table='{0}_{1}_gacal'.
                             format(msname,sourcename))
    error += not cb.correct()
    error += not cb.close()
    if error > 0:
        print('{0} errors occured during calibration'.format(error))
    return

def flag_antenna(msname,antenna,datacolumn='data',pol=None):
    """Flag antennas in a measurement set using CASA.
    
    Args:
        ms: str
          the name of the measurement set (will open <ms>.ms)
        antenna: str or int 
          if str, a CASA-understood list of antennas to flag. If int,
          the index of a single antenna to flag

    Returns:
    """
    if type(antenna) is int:
        antenna = str(antenna)
    error = 0
    ag = cc.agentflagger()
    error += not ag.open('{0}.ms'.format(msname))
    error += not ag.selectdata()
    rec = {}
    rec['mode']='clip'
    rec['clipoutside']=False
    rec['datacolumn']=datacolumn
    rec['antenna']=antenna
    if pol is not None:
        rec['polarization_type']='XX' if pol=='A' else 'YY'
    error += not ag.parseagentparameters(rec)
    error += not ag.init()
    error += not ag.run()
    error += not ag.done()
    if error > 0:
        print('{0} errors occured during calibration'.format(error))
    return

def flag_zeros(msname,datacolumn='data'):
    error = 0 
    ag = cc.agentflagger()
    error += not ag.open('{0}.ms'.format(msname))
    error += not ag.selectdata()
    rec = {}
    rec['mode']='clip'
    rec['clipzeros']=True
    rec['datacolumn']=datacolumn
    error += not ag.parseagentparameters(rec)
    error += not ag.init()
    error += not ag.run()
    error += not ag.done()
    if error > 0:
        print('{0} errors occured during flagging'.format(error))
    return

def flag_badtimes(msname,times,bad,nant,datacolumn='data',
                  verbose=False):
    """Flag antennas in a measurement set using CASA
    
    Args:
      msname: str
        the name of the measurement set (will open <msname>.ms)
      times : float array
        the times of each calibration solution, MJD seconds
      bad   : boolean array
        dimensions (ntimes, nantennas), whether or not to flag a time bin
      verbose: boolean
        if True, will print information about the antenna/time pairs being flagged

    Returns:
    """
    error = 0
    tdiff = np.median(np.diff(times))
    ag = cc.agentflagger()
    error += not ag.open('{0}.ms'.format(msname))
    error += not ag.selectdata()
    for i in range(nant):
        rec = {}
        rec['mode']='clip'
        rec['clipoutside']=False
        rec['datacolumn']=datacolumn
        rec['antenna']=str(i)
        rec['polarization_type']='XX'
        tstr = ''
        for j in range(len(times)):
            if bad[0,j,i]:
                if len(tstr)>0:
                    tstr += '; '
                tstr += '{0}~{1}'.format(times[j]-tdiff/2,times[j]+tdiff/2)
        if verbose:
            print('For antenna {0}, flagged: {1}'.format(i,tstr))
        error += not ag.parseagentparameters(rec)
        error += not ag.init()
        error += not ag.run()
        
        rec['polarization_type']='YY'
        tstr = ''
        for j in range(len(times)):
            if bad[1,j,i]:
                if len(tstr)>0:
                    tstr += '; '
                tstr += '{0}~{1}'.format(times[j]-tdiff/2,times[j]+tdiff/2)
        if verbose:
            print('For antenna {0}, flagged: {1}'.format(i,tstr))
        error += not ag.parseagentparameters(rec)
        error += not ag.init()
        error += not ag.run()
    error += not ag.done()
    if error > 0:
        print('{0} errors occured during calibration'.format(error))
    return



def calc_delays(vis,df,nfavg=5,tavg=True):
    """Calculate delays from the visibilities.
    
    Args:
        vis: complex array
          the visibilities
        df: float
          the size of a single frequency channel in GHz
        nfavg: int
          the number of frequency channels to avg by
    
    Returns:
        vis_ft: complex array
          the Fourier transform of the time-averaged 
          visibilities, dimensions (baselines, delay)
        delay_arr: real array
          the values of the delay pixels in nanoseconds
    """
    nfbins = vis.shape[-2]//nfavg*nfavg
    if tavg:
        print(vis.shape)
        vis_ft = fftshift(fft(np.pad(vis[...,:nfbins,:].mean(1),
                                ((0,0), (0,nfbins),(0,0))),axis=-2),axes=-2)
        vis_ft = vis_ft.reshape(vis_ft.shape[0],-1,2*nfavg,2).mean(-2)
    else:
        vis_ft = fftshift(fft(np.pad(vis[...,:nfbins,:],
                                ((0,0),(0,0),(0,nfbins),(0,0)))
                              ,axis=-2),axes=-2)
        vis_ft = vis_ft.reshape(vis_ft.shape[0],
                                vis_ft.shape[1],
                                -1,2*nfavg,2).mean(-2)
    delay_arr = fftshift(fftfreq(nfbins))/df
    delay_arr = delay_arr.reshape(-1,nfavg).mean(-1)
    
    return vis_ft, delay_arr

def get_bad_times(msname,sourcename,nant,tint='59s'):
    """Use delays on short time periods to flag bad antenna/time
    pairs in the calibrator data. 
    
    Args:
        msname: str
          the prefix of the measurement set.  Will open <msname>.ms
        sourcename: str
          the name of the calibrator
          will extract the delay solutions on 'inf' timescales from 
          <msname>_<sourcename>_kcal 
          and save delay solutions on tint timescales in 
          <msname>_<sourcename>_2kcal
        nant: int
          the number of antennas in the array
    
    Returns:
        bad_times: boolean array 
          shape (ntimes, nant), whether a time-antenna 
          pair should be flagged
        times: float array
          the time (mjd) for each delay solution
    """
    error = 0
    # Solve the calibrator data on minute timescales
    cb = cc.calibrater()
    error += not cb.open('{0}.ms'.format(msname))
    error += not cb.setsolve(type='K',t=tint,
        refant=0,table='{0}_{1}_2kcal'.format(msname,sourcename))
    error += not cb.solve()
    error += not cb.close()
    # Pull the solutions for the entire timerange and the 
    # 60-s data from the measurement set tables
    tb = cc.table()
    error += not tb.open('{0}_{1}_2kcal'.format(msname,sourcename))
    antenna_delays = tb.getcol('FPARAM')
    npol = antenna_delays.shape[0]
    antenna_delays = antenna_delays.reshape(npol,-1,nant)
    times = (tb.getcol('TIME').reshape(-1,nant)[:,0]*u.s).to_value(u.d)
    error += not tb.close()
    tb = cc.table()
    error += not tb.open('{0}_{1}_kcal'.format(msname,sourcename))
    kcorr = tb.getcol('FPARAM').reshape(npol,-1,nant)
    error += not tb.close()
    threshold = nant*npol//2
    bad_times = (np.abs(antenna_delays-kcorr)>1.5)
    bad_times[:,np.sum(np.sum(bad_times,axis=0),axis=1)
              >threshold,:] = np.ones((npol,1,nant))
    if error > 0:
        print('{0} errors occured during calibration'.format(error))
    return bad_times,times

def apply_calibration(msname,calname,msnamecal=None):
    """Apply the calibration solution from the calibrator
    to a measurement set.  Applies delay, bandpass, and complex 
    gain solutions.
    
    Args:
      msname: str
        the name of the measurement set to apply calibration solutions
        to.  Will open <msname>.ms
      calname: str
        the name of the calibrator. used to identify the correct 
        calibration tables.
      msnamecal: str
        the name of the measurement set containing the calibrator
        visibilities.  calibration tables prefixed with 
        <msnamecal>_<calname> will be opened and applied.  If not 
        given, it is assumed that msnamecal = msname
          
    Returns:
    """
    if msnamecal is None:
        msnamecal = msname
    error = 0
    cb = cc.calibrater()
    error += not cb.open('{0}.ms'.format(msname))
    error += not cb.setapply(type='K',
                             table='{0}_{1}_kcal'.format(msnamecal,calname))
    error += not cb.setapply(type='B',
                            table='{0}_{1}_bcal'.format(msnamecal,calname))
    error += not cb.setapply(type='G',
                             table='{0}_{1}_gacal'.format(msnamecal,calname))
    error += not cb.setapply(type='G',
                             table='{0}_{1}_gpcal'.format(msnamecal,calname))
    error += not cb.correct()
    error += not cb.close()
    if error > 0:
        print('{0} errors occured during calibration'.format(error))
    return
