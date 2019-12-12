"""
DSA_CALIB.PY

Dana Simard, dana.simard@astro.caltech.edu, 10/2019

Casa-based routines to calibrate visibilities using point sources
"""

import __casac__ as cc
import astropy.units as u
import numpy as np
import astropy.constants as c
from . import constants as ct
from scipy.fftpack import fft,fftshift,fftfreq

def delay_calibration(msname,sourcename,refant=0,t='inf',fskcal=False):
    """Calibrate delays using CASA and write the calibrated 
    visibilities to the corrected_data column of the measurement set
    
    Args:
        msname: str
          the name of the measurement set (will open <ms>.ms)
        refant: int
          the index of the reference antenna
        t: str
          a CASA-understood time to integrate by
        prefix: str
          The prefix to use for the calibration table.  The table 
          will be written to <prefix>kcal.  If fringestopping, should be the source.name
          for the fringestopping source
        fskcal: Boolean
          if True, the delay table fs<prefix>kcal, containing the delays
          required to fringestop on the source coordinates, will be applied before 
          solving for calibration paramaters and calibrating

    Returns:
    """
    error = 0
    cb = cc.calibrater.calibrater()
    error += not cb.open('{0}.ms'.format(msname))
    if fskcal:
        error += not cb.setapply(type='K',table='{0}_{1}_fscal'.
                                format(msname,sourcename))
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

def gain_calibration(msname,sourcename,tga='600s',fskcal=False):
    """Use Self-Cal to calibrate gains and save to calibration tables
    
    Args:
        msname: str
          the measurement set.  will open <msname>.ms
        source: src class instance
          the calibrator, will save the tables to
          <src.name>kcal (for delays), <src.name>gpcal (for phases) and
          <src.name>gacal (for amplitudes)
        fskcal: boolean
          if True, the delay table fs<source.name>kcal, containing the delays
          required to fringestop on the source coordinates, will be applied before 
          solving for calibration paramaters and calibrating
    
    Returns:
    """
    error = 0
    # Solve for phase calibration over entire obs
    cb = cc.calibrater.calibrater()
    error += not cb.open('{0}.ms'.format(msname))
    if fskcal:
        error += not cb.setapply(type='K',table='{0}_{1}_fscal'.
                                format(msname,sourcename))
    error += not cb.setapply(type='K',table='{0}_{1}_kcal'.
                             format(msname,sourcename))
    error += not cb.setsolve(type='G',table='{0}_{1}_gpcal'.
                             format(msname,sourcename),t='inf',
                     minblperant=1,refant='0',apmode='p')
    error += not cb.solve()
    error += not cb.close()

    # Solve for gain calibration on 10 minute timescale
    cb = cc.calibrater.calibrater()
    error += not cb.open('{0}.ms'.format(msname))
    if fskcal:
        error += not cb.setapply(type='K',table='{0}_{1}_fscal'.
                                format(msname,sourcename))
    error += not cb.setapply(type='K',table='{0}_{1}_kcal'.
                             format(msname,sourcename))
    error += not cb.setapply(type='G',table='{0}_{1}_gpcal'.
                             format(msname,sourcename))
    error += not cb.setsolve(type='G',table='{0}_{1}_gacal'.
                             format(msname,sourcename), t=tga,
                     minblperant=1,refant='0',apmode='a')
    error += not cb.solve()
    error += not cb.close()

    # Apply calibration
    cb = cc.calibrater.calibrater()
    error += not cb.open('{0}.ms'.format(msname))
    if fskcal:
        error += not cb.setapply(type='K',table='{0}_{1}_fscal'.
                                format(msname,sourcename))
    error += not cb.setapply(type='K',table='{0}_{1}_kcal'.
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

def flag_antenna(ms,antenna):
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
    ag = cc.agentflagger.agentflagger()
    error += not ag.open('{0}.ms'.format(ms))
    error += not ag.selectdata()
    rec = {}
    rec['mode']='clip'
    rec['clipoutside']=False
    rec['datacolumn']='data'
    rec['antenna']=antenna
    error += not ag.parseagentparameters(rec)
    error += not ag.init()
    error += not ag.run()
    rec['datacolumn']='corrected'
    error += not ag.parseagentparameters(rec)
    error += not ag.init()
    error += not ag.run()
    error += not ag.done()
    if error > 0:
        print('{0} errors occured during calibration'.format(error))
    return

def flag_badtimes(ms,times,bad,nant,verbose=False):
    """Flag antennas in a measurement set using CASA
    
    Args:
      ms: str
        the name of the measurement set (will open <ms>.ms)
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
    ag = cc.agentflagger.agentflagger()
    error += not ag.open('{0}.ms'.format(ms))
    error += not ag.selectdata()
    for i in range(nant):
        rec = {}
        rec['mode']='clip'
        rec['clipoutside']=False
        rec['datacolumn']='data'
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
        rec['datacolum']='corrected'
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
        rec['datacolum']='corrected'
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

def get_bad_times(msname,sourcename,nant,fskcal=False):
    """Use delays on short time periods to flag bad antenna/time
    pairs in the calibrator data. These won't be used in the 
    gain calibration.
    
    Args:
        msname: str
          the prefix of the measurement set.  Will open <msname>.ms
        source: src class
          the calibrator. will create and read 
          calibration tables that are prefixed by src.name
        nant: int
          the number of antennas in the array
        fskcal: boolean
          if True, the delay table fs<source.name>kcal, containing the delays
          required to fringestop on the source coordinates, will be applied before 
          solving for calibration paramaters and calibrating
    
    Returns:
        bad_times: boolean array 
          shape (ntimes, nant), whether a time-antenna 
          pair should be flagged
        times: float array
          the time (mjd) for each delay solution
    """
    error = 0
    # Solve the calibrator data on minute timescales
    cb = cc.calibrater.calibrater()
    error += not cb.open('{0}.ms'.format(msname))
    if fskcal:
        error += not cb.setapply(type='K',table='{0}_{1}_fscal'.
                                format(msname,sourcename))
    error += not cb.setsolve(type='K',t='59s',
        refant=0,table='{0}_{1}_2kcal'.format(msname,sourcename))
    error += not cb.solve()
    error += not cb.close()
    # Pull the solutions for the entire timerange and the 
    # 60-s data from the measurement set tables
    tb = cc.table.table()
    error += not tb.open('{0}_{1}_2kcal'.format(msname,sourcename))
    antenna_delays = tb.getcol('FPARAM')
    npol = antenna_delays.shape[0]
    antenna_delays = antenna_delays.reshape(npol,-1,nant)
    times = (tb.getcol('TIME').reshape(-1,nant)[:,0]*u.s).to_value(u.d)
    error += not tb.close()
    tb = cc.table.table()
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

def apply_calibration(msname,calname,msnamecal=None,fskcal=False,fsname=None):
    """Apply the calibration solution from the calibrator
    to the source.
    
    Args:
        source: src class instance
          the target source.  Will open the measurement set 
          <src.name>.ms
        cal: src class instance
          the calibrator source.  Will open calibration
          tables prefixed with <src.name> and ending in kcal,
          gacal and gpcal
          
    Returns:
    """
    if msnamecal is None:
        msnamecal = msname
    error = 0
    cb = cc.calibrater.calibrater()
    error += not cb.open('{0}.ms'.format(msname))
    if fskcal:
        if not fsname: fsname = calname
        error += not cb.setapply(type='K',table='{0}_{1}_fscal'.
                                format(msname,fsname))
    error += not cb.setapply(type='K',
                             table='{0}_{1}_kcal'.format(msnamecal,calname))
    error += not cb.setapply(type='G',
                             table='{0}_{1}_gacal'.format(msnamecal,calname))
    error += not cb.setapply(type='G',
                             table='{0}_{1}_gpcal'.format(msnamecal,calname))
    error += not cb.correct()
    error += not cb.close()
    if error > 0:
        print('{0} errors occured during calibration'.format(error))
    return
