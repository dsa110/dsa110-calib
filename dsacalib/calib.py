"""Functions for calibration of DSA-110 visibilities.

These functions use the CASA package casatools to calibrate
visibilities stored as measurement sets.

Author: Dana Simard, dana.simard@astro.caltech.edu, 10/2019

"""
# Always import scipy before casatools
import scipy 
import casatools as cc
import astropy.units as u
import numpy as np
import astropy.constants as c
from dsacalib import constants as ct
from dsacalib.utils import read_caltable
from scipy.fftpack import fft,fftshift,fftfreq

def delay_calibration(msname,sourcename,refant,t='inf'):
    """Calibrates delays using CASA.

    Uses CASA to calibrate delays and write the calibrated 
    visibilities to the corrected_data column of the measurement set.
    
    Parameters
    ----------
    msname : str
        The name of the measurement set.  The measurement set `msname`.ms 
        will be opened.
    sourcename : str 
        The name of the calibrator source. The calibration table will be 
        written to `msname`\_`sourcename`\_kcal.
    refant : str
        The reference antenna to use in calibration. If  type *str*, 
        this is the name of the antenna.  If type *int*, it is the
        index of the antenna in the measurement set.
    t : str
        The integration time to use before calibrating, e.g. ``'inf'`` 
        or ``'60s'``.  See the CASA documentation for more examples.  
        Defaults to ``'inf'`` (averaging over the entire observation time).
          
    Returns
    -------
    error : int
        the number of errors that occured during calibration
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

    return error

def gain_calibration_blbased(msname,sourcename,tga,tgp,refant):
    """Use CASA to calculate bandpass and complex gain solutions. 

    Saves solutions to calibration tables and calibrates the 
    measurement set by applying delay, bandpass, 
    and complex gain solutions.  Uses baseline-based calibration routines
    within CASA.
    
    Parameters
    ----------
    msname : str
        The name of the measurement set.  The MS `msname`.ms will be opened.
    sourcename : str
        The name of the calibrator source.  The calibration table will be 
        written to `msname`\_`sourcename`\_kcal.
    tga : str
        The integration time to use before calibrating the amplitudes of the 
        complex gain, e.g. ``'inf'`` or ``'60s'``.  See the CASA 
        documentation for more examples.  
    tgp : str
        The integration time to use before calibrating
        the amplitudes of the complex gain, e.g. ``'inf'``
        or ``'60s'``.  See the CASA documentation for more examples.  
    refant : str 
        The reference antenna to use in calibration.  If type *str*, 
        this is the name of the antenna.  If type *int*, it is the
        index of the antenna in the measurement set.
    
    Returns 
    -------
    error : int
        The number of errors that occured during calibration.
    """
    error = 0
    
    # Solve for bandpass calibration
    cb = cc.calibrater()
    error += not cb.open('{0}.ms'.format(msname))
    error += not cb.setapply(type='K',table='{0}_{1}_kcal'.
                             format(msname,sourcename))
    error += not cb.setsolve(type='MF',table='{0}_{1}_bcal'.
                             format(msname,sourcename),
                             refant=refant,apmode='a',solnorm=True)
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

    return error

def gain_calibration(msname,sourcename,tga,tgp,refant):
    """Use CASA to calculate bandpass and complex gain solutions. 

    Saves solutions to calibration tables and calibrates the 
    measurement set by applying delay, bandpass, 
    and complex gain solutions. 
    
    Parameters
    ----------
    msname : str
        The name of the measurement set.
        The MS `msname`.ms will be opened.
    sourcename : str
        The name of the calibrator source.
        The calibration table will be written to 
        `msname`\_`sourcename`\_kcal.
    tga : str
        The integration time to use before calibrating
        the amplitudes of the complex gain, e.g. ``'inf'`` 
        or ``'60s'``.  See the CASA documentation for more examples.  
    tgp : str
        The integration time to use before calibrating
        the amplitudes of the complex gain, e.g. ``'inf'``
        or ``'60s'``.  See the CASA documentation for more examples.  
    refant : str
        The reference antenna to use in calibration. If type *str*, 
        this is the name of the antenna.  If type *int*, it is the
        index of the antenna in the measurement set.
    
    Returns
    -------
    error : int
        The number of errors that occured during calibration.
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

    return error

def flag_antenna(msname,antenna,datacolumn='data',pol=None):
    """Flags an antenna in a measurement set using CASA.
    
    Parameters
    ----------
    msname : str
        The name of the measurement set.  The MS `msname`.ms will be opened.
    antenna : str
        The antenna to flag. If type *str*, this is the name of the antenna.  
        If type *int*, the index of the antenna in the measurement set.
    datacolumn : str
        The column of the measurement set to flag.
        Options are ``'data'``,``'model'``,``'corrected'`` for the 
        uncalibrated visibilities, the visibility model (used by 
        CASA to calculate calibration solutions), the calibrated
        visibilities.  Defaults to ``'data'``.
    pol : str 
        The polarization to flag.  Must be `'A'` 
        (which is mapped to polarization 'XX' of the CASA measurement 
        set) or `'B'` (mapped to polarization 'YY').  Can also be `None`, 
        for which both polarizations are flagged.  Defaults to `None`.
    
    Returns
    -------
    error : int
        The number of errors that occured during calibration.
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

    return error

def reset_flags(msname,datacolumn=None):
    """Resets all flags in a measurement set, so that all data is unflagged.
    
    Parameters
    ----------
    msname : str
        The name of the measurement set. The MS `msname`.ms will be opened.
    datacolumn : str
        The column of the measurement set to flag.
        Options are ``'data'``,``'model'``,``'corrected'`` for the 
        uncalibrated visibilities, the visibility model (used by 
        CASA to calculate calibration solutions), the calibrated
        visibilities.  Defaults to ``'data'``.
    
    Returns
    -------
    error : int
        The number of errors that occured during calibration.
    """
    error = 0 
    ag = cc.agentflagger.agentflagger()
    error += not ag.open('{0}.ms'.format(msname))
    error += not ag.selectdata()
    rec = {}
    rec['mode']='unflag'
    if datacolumn is not None:
        rec['datacolumn']=datacolumn
    error += not ag.parseagentparameters(rec)
    error += not ag.init()
    error += not ag.run()
    error += not ag.done()

    return error

def flag_zeros(msname,datacolumn='data'):
    """Flags all zeros in a measurement set.
    
    Parameters
    ----------
    msname : str
        The name of the measurement set. The MS `msname`.ms will be opened.
    datacolumn : str
        The column of the measurement set to flag.
        Options are ``'data'``,``'model'``,``'corrected'`` for the 
        uncalibrated visibilities, the visibility model (used by 
        CASA to calculate calibration solutions), the calibrated
        visibilities.  Defaults to ``'data'``.
    
    Returns
    -------
    error : int
        The number of errors that occured during calibration.
    """
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

    return error

# Change times to not use mjds, but mjd instead
def flag_badtimes(msname,times,bad,nant,datacolumn='data',
                  verbose=False):
    """Flags bad time bins for each antenna in a measurement set using CASA.
    
    Parameters
    ----------
    msname : str 
        The name of the measurement set. The MS `msname`.ms will be opened.
    times : ndarray
        A 1-D array of times, type float, seconds since MJD=0. 
        Times should be equally spaced and cover the entire time range 
        of the measurement set, but can be coarser than the resolution of
        the measurement set.
    bad : ndarray
        A 1-D boolean array with dimensions (len(`times`), `nant`). 
        Should have a value of ``True`` if the corresponding timebins should be
        flagged.
    nant : int 
        The number of antennas in the measurement set.
    datacolumn : str
        The column of the measurement set to flag.
        Options are ``'data'``,``'model'``,``'corrected'`` for the 
        uncalibrated visibilities, the visibility model (used by 
        CASA to calculate calibration solutions), the calibrated
        visibilities.  Defaults to ``'data'``.
    verbose : boolean
        If ``True``, will print information about the 
        antenna/time pairs being flagged.  Defaults to ``False``.
            
    Returns
    -------
    error : int
        The number of errors that occured during calibration.
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

    return error

def calc_delays(vis,df,nfavg=5,tavg=True):
    """Calculates power as a function of delay from the visibilities.
    
    This uses scipy fftpack to fourier transform the visibilities 
    along the frequency axis.  The power as a function of delay can 
    then be used in fringe-fitting.
    
    Parameters
    ----------
    vis : ndarray
        The complex visibilities. 4 dimensions,
        (baseline,time,frequency,polarization).
    df : float
        The width of the frequency channels in GHz.
    nfavg : int 
        The number of frequency channels to average by
        after the Fourier transform.  Defaults to 5.
    tavg : boolean
        If ``True``, the visibilities are averaged 
        in time before the Fourier transform. Defaults to ``True``.
    
    Returns
    -------
    vis_ft : ndarray
        The complex visibilities, Fourier-transformed
        along the time axis.  3 (or 4, if `tavg` is set to False) 
        dimensions, (baseline,delay,polarization) (or 
        (baseline,time,delay,polarization) if `tavg` is set to False)
    delay_arr : ndarray
        Float, the values of the delay pixels in nanoseconds
    """
    nfbins = vis.shape[-2]//nfavg*nfavg
    npol = vis.shape[-1]
    if tavg:
        vis_ft = fftshift(fft(np.pad(vis[...,:nfbins,:].mean(1),
                                ((0,0), (0,nfbins),(0,0))),axis=-2),axes=-2)
        vis_ft = vis_ft.reshape(vis_ft.shape[0],-1,2*nfavg,npol).mean(-2)
    else:
        vis_ft = fftshift(fft(np.pad(vis[...,:nfbins,:],
                                ((0,0),(0,0),(0,nfbins),(0,0)))
                              ,axis=-2),axes=-2)
        vis_ft = vis_ft.reshape(vis_ft.shape[0],
                                vis_ft.shape[1],
                                -1,2*nfavg,npol).mean(-2)
    delay_arr = fftshift(fftfreq(nfbins))/df
    delay_arr = delay_arr.reshape(-1,nfavg).mean(-1)
    
    return vis_ft, delay_arr

# Change refant to no longer default
def get_bad_times(msname,sourcename,nant,refant,tint='59s'):
    """Flags bad times in the calibrator data.

    Calculates delays on short time periods and compares them to the
    delay calibration solution. Can only be run after delay calibration.
    
    Parameters
    ----------
    msname : str
        The name of the measurement set. The MS `msname`.ms will be opened.
    sourcename : str
        The name of the calibrator source.  The calibration table will 
        be written to `msname`\_`sourcename`\_kcal.
    nant : int
        The number of antennas in the array.
    refant : str
        The reference antenna to use in calibration.  If type *str*, the 
        name of the reference antenna, if type *int*, the 
        index of the antenna in the CASA measurement set.  This must
        be the same as the reference antenna used in the delay calibration,
        or unexpected errors may occur.
    tint : str
        The timescale on which to calculate the delay solutions (and 
        evaluate the data quality).  Must be a CASA-interpreted string,
        e.g. ``'inf'`` (average all of the data) or ``'60s'`` (average 
        data to 60-second bins before delay calibration).  Defaults to 
        ``'59s'``.

    Returns
    -------
    bad_times : ndarray
        Booleans, ``True`` if the data quality is poor and the
        time-bin should be flagged, ``False`` otherwise.  Dimensions 
        (time,antenna)
    times : ndarray
        Floats, the time (mjd) for each delay solution
    error :int
        The number of errors that occured during calibration.
    """
    error = 0
    # Solve the calibrator data on minute timescales
    cb = cc.calibrater()
    error += not cb.open('{0}.ms'.format(msname))
    error += not cb.setsolve(type='K',t=tint,
        refant=refant,table='{0}_{1}_2kcal'.format(msname,sourcename))
    error += not cb.solve()
    error += not cb.close()
    # Pull the solutions for the entire timerange and the 
    # 60-s data from the measurement set tables
    times,antenna_delays,flags = read_caltable('{0}_{1}_2kcal'
                                    .format(msname,sourcename),nant,
                                    cparam=False)
    npol = antenna_delays.shape[0]
    tkcorr,kcorr,flags = read_caltable('{0}_{1}_kcal'
                                      .format(msname,sourcename),nant,
                                      cparam=False)
    threshold = nant*npol//2
    bad_times = (np.abs(antenna_delays-kcorr)>1.5)
    bad_times[:,np.sum(np.sum(bad_times,axis=0),axis=1)
              >threshold,:] = np.ones((npol,1,nant))

    return bad_times,times,error

def apply_calibration(msname,calname,msnamecal=None):
    """Applies the calibration solution.

    Applies delay, bandpass and complex gain tables
    to a measurement set.  
    
    Parameters
    ----------
    msname : str
        The name of the measurement set to apply 
        calibration solutions to.  Opens `msname`.ms
    calname : str
        The name of the calibrator.  Tables that start
        with `msnamecal`\_`calname` will be applied to the measurement
        set.
    msnamecal : str
        The name of the measurement set used to model the
        calibraiton solutions.  Calibration tables prefixed with 
        `msnamecal`\_`calname` will be opened and applied. 
        If ``None``, `msnamecal` is set to `msname`. Defaults to ``None``.
          
    Returns
    -------
    error : int
        The number of errors that occured during calibration.
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

    return error 

def fill_antenna_gains(gains,flags=None):
    """Fills in the antenna gains for triple-antenna calibration.

    Takes the gains from baseline-based calibration for a trio of
    antennas and calculates the corresponding antenna gains using
    produces of the baseline gains.  Also propagates flag information 
    for the input baseline gains to the antenna gains.
    
    Parameters
    ----------
    gains : narray
        The complex gains matrix, first dimension
        is baseline.  Indices 1, 2 and 4 contain the gains for the
        cross-correlations. Information in indices 0, 3 and 5 is
        ignored and overwritten.
    flags : ndarray
        A boolean array, containing flag information 
        for the `gains` array. 1 if the data is flagged, 0 otherwise.
        If ``None``, assumes no flag information available.  The first 
        dimension is baseline.  Indices 1, 2 and 4 contain the flags
        for the cross-correlations.  Information in indices 0, 3 and 5
        is ignored and overwritten.
    
    Returns
    -------
    gains : ndarray
        The complex gains matrix, first dimension
        is baseline.  Indices 1, 2 and 4 contain the gains for the
        cross-correlations. Indices 0, 3 and 5
        contain the calculated values for the antennas.
    flags : ndarray
        A boolean array, containing flag information 
        for the `gains` array.  1 if the data is flagged, 0 otherwise.
        If None, assumes no flag information available.  The first 
        dimension is baseline.  Indices 1, 2 and 4 contain the flags
        for the cross-correlations.  Indices 0,3 and 5 contain
        the calculated values for the antennas.
    """
    assert gains.shape[0]==6,'Will only calculate antenna gains for trio'
    gains[0] = np.conjugate(gains[1])*gains[2]/gains[4]
    gains[3] = gains[1]*gains[4]/gains[2]
    gains[5] = gains[2]*np.conjugate(gains[4])/gains[1]
    
    if flags is not None:
        flags[[0,3,5],...] = np.min(np.array([flags[1]+flags[2]+flags[4],
                          np.ones(flags[0].shape,dtype=int)]),axis=0)
    
        return gains,flags
    return gains
