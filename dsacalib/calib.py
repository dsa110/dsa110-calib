"""
DSA_CALIB.PY
Dana Simard, dana.simard@astro.caltech.edu, 10/2019
Casa-based routines to calibrate visibilities using point sources
"""

import __casac__ as cc
import astropy.units as u
import numpy as np
import astropy.constants as c
from scipy.fftpack import fft,fftshift,fftfreq

def to_deg(string):
    """ Converts a string direction to degrees.
    Parameters:
    -----------
    string : string, of format e.g. "12h00m19.21s" or "+73d00m45.7s"
    Returns:
    --------
    deg : the input angle in degrees, astropy quantity
    """
    if 'h' in string:
        h,m,s = string.strip('s').strip('s').replace('h','m').split('m')
        deg = (float(h)+(float(m)+float(s)/60)/60)*15*u.deg
    else:
        sign = -1 if '-' in string else 1
        d,m,s = string.strip('+-s').strip('s').replace('d','m').split('m')
        deg = (float(d)+(float(m)+float(s)/60)/60)*u.deg*sign
    return deg

class src():
    """ Simple class for holding source parameters.
    Parameters:
    -----------
    name : some ascii identifier of your source
    I    : the flux in Jy, float
    ra   : the right ascension in a string format 
           e.g. "12h00m19.21s"
    dec  : the declination in a string format
           e.g. "+73d00m45.7s"
    epoch : the epoch of the ra and dec, string, 
           default "J2000"
    """
    def __init__(self,name,I,ra,dec,epoch='J2000'):
        self.name = name
        self.I = I
        self.ra = to_deg(ra)
        self.dec = to_deg(dec)
        self.epoch = epoch

def calc_uvw(b, tobs, src_epoch, src_lon, src_lat):
    """ Calculates the uvw coordinates of a source.
    Parameters:
    -----------
    b       : baseline, array of shape (nbaselines, 3), 
              units m in ITRF coords
    tobs    : time of observation in mjd; 1-d array
    src_epoch : str, casa-recognized descriptor of the source coordinates
              e.g. 'J2000' or 'HADEC'
    src_lon : longitude of source, astropy quantity
    src_lat : latitute of source, astropy quantity
    Returns:
    ---------
    u : u, array of shape (nbaselines, ntimes), units m
    v : v, array of shape (nbaselines, ntimes), units m
    w : w, array of shape (nbaselines, ntimes), units m
    """
    if type(tobs)!=np.ndarray: tobs = np.array(tobs)
    if tobs.ndim < 1: tobs = tobs[np.newaxis]
    nt = tobs.shape[0] 
    if type(b)==list: b = np.array(b)
    if b.ndim < 2: b = b[np.newaxis,:]
    nb = b.shape[0] 
    
    bu = np.zeros((nt,nb))
    bv = np.zeros((nt,nb))
    bw = np.zeros((nt,nb))
    
    # Define the reference frame
    me = cc.measures.measures()
    qa = cc.quanta.quanta()
    me.doframe(me.observatory('OVRO_MMA'))
    me.doframe(me.direction(src_epoch, 
                            qa.quantity(src_lon.to_value(u.deg),'deg'), 
                            qa.quantity(src_lat.to_value(u.deg),'deg')))

    contains_nans=False

    for i in range(nt):
        me.doframe(me.epoch('UTC',qa.quantity(tobs[i],'d'))) 
        for j in range(nb):
            bl = me.baseline('itrf',qa.quantity(b[j,0],'m'),
                  qa.quantity(b[j,1],'m'),qa.quantity(b[j,2],'m'))
            # Get the uvw coordinates
            try:
                uvw=me.touvw(bl)[1]['value']
                bu[i,j],bv[i,j],bw[i,j]=uvw[0],uvw[1],uvw[2]
            except KeyError:
                contains_nans=True
                bu[i,j],bv[i,j],bw[i,j]=np.nan,np.nan,np.nan
    if contains_nans:
        print('Warning: some solutions not found for u,v,w coordinates')
            
    return bu.T,bv.T,bw.T
    
def visibility_model(b, sources, tobs, fobs, phase_only=False):
    """
    Calculates the model of the visibilities along a baseline.
    Parameters:
    -----------
    b     : baseline, numpy array shape (nbaselines, 3), 
            units m in ITRF coords
    src_model : list of sources in the field (each source should 
             be a src class instance)
    tobs   : time of observation in mjd, float or array
    fobs   : frequency of observation in GHz
    Returns:
    --------
    vis_model : array, the modelled visibilities, dimensions 
             (baselines, frequency, time)
    """
  
    # Calculate the total visibility by adding each source
    # Eventually we will also have to include the primary beam here
    # Choose dimensions for vis_model
    
    if type(fobs)!=np.ndarray: fobs = np.array(fobs)
    if fobs.ndim < 1: fobs = fobs[np.newaxis]
    nf = fobs.shape[0] 
    if type(tobs)!=np.ndarray: tobs = np.array(tobs)
    if tobs.ndim < 1: tobs = tobs[np.newaxis]
    nt = tobs.shape[0] 
    if type(b)==list: b = np.array(b)
    if b.ndim < 2: b = b[np.newaxis,:]
    nb = b.shape[0] 
    
    # Model the flux of the source using a simple spectral index 
    if phase_only:
        famp = 1.
    else:
        spec_idx = -0.7
        f0 = 1.4
        famp = ((fobs/f0)**(spec_idx))
    
    vis_model = np.zeros((nb,nt,nf),dtype=complex)

    for src in sources:
        bu,bv,bw = calc_uvw(b, tobs, src.epoch, src.ra, src.dec)

        vis_model += famp * (1. if phase_only else src.I) * \
            np.exp(2j*np.pi/c.c.to_value(u.GHz*u.m)*
                   fobs*(bw[:,:,np.newaxis]))

    return vis_model

def delay_calibration(ms,refant=0,t='inf',prefix=''):
    """Calibrate delays using CASA and writes the calibrated 
    visibilities to the corrected_data column of the measurement set
    Parameters:
    -----------
    ms : str, the name of the measurement set (will open <ms>.ms)
    refant : int, the index of the reference antenna
    t  : str, a CASA-understood time to integrate by
    Returns nothing
    """
    error = 0
    cb = cc.calibrater.calibrater()
    error += not cb.open('{0}.ms'.format(ms))
    error += not cb.setsolve(type='K',t=t,
            refant=refant,table='{0}kcal'.format(prefix))
    error += not cb.solve()
    error += not cb.setapply(type='K',table='{0}kcal'.format(prefix))
    error += not cb.correct()
    error += not cb.close()
    if error > 0:
        print('{0} errors occured during calibration'.format(error))
    return

def gain_calibration(source):
    """Use Self-Cal to calibrate gains and save to calibration tables
    Parameters:
    -----------
    source: src class, the calibrator
    """
    error = 0
    # Solve for phase calibration over entire obs
    cb = cc.calibrater.calibrater()
    error += not cb.open('{0}.ms'.format(source.name))
    error += not cb.setapply(type='K',table='{0}kcal'.
                             format(source.name))
    error += not cb.setsolve(type='G',table='{0}gpcal'.
                             format(source.name),t='inf',
                     minblperant=1,refant='0',apmode='p')
    error += not cb.solve()
    error += not cb.close()

    # Solve for gain calibration on 10 minute timescale
    cb = cc.calibrater.calibrater()
    error += not cb.open('{0}.ms'.format(source.name))
    error += not cb.setapply(type='K',table='{0}kcal'.
                             format(source.name))
    error += not cb.setapply(type='G',table='{0}gpcal'.
                             format(source.name))
    error += not cb.setsolve(type='G',table='{0}gacal'.
                             format(source.name), t='600s',
                     minblperant=1,refant='0',apmode='a')
    error += not cb.solve()
    error += not cb.close()

    # Apply calibration
    cb = cc.calibrater.calibrater()
    error += not cb.open('{0}.ms'.format(source.name))
    error += not cb.setapply(type='K',table='{0}kcal'.
                             format(source.name))
    error += not cb.setapply(type='G',table='{0}gpcal'.
                             format(source.name))
    error += not cb.setapply(type='G',table='{0}gacal'.
                             format(source.name))
    error += not cb.correct()
    error += not cb.close()
    if error > 0:
        print('{0} errors occured during calibration'.format(error))
    return

def flag_antenna(ms,antenna):
    """Flag antennas in a measurement set using CASA
    Parameters:
    -----------
    ms : str, the name of the measurement set (will open <ms>.ms)
    antenna : str (a CASA-understood list of antennas to flag) or int 
         (the index of a single antenna to flag)
    Returns nothing
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
    Parameters:
    -----------
    ms : str, the name of the measurement set (will open <ms>.ms)
    times : float array, the times of each calibration solution, MJD seconds
    bad   : boolean array, (ntimes, nantennas), whether or not to flag a time bin
    verbose : boolean, default False
    Returns nothing
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
    """Calculate delays from the visibilities
    Parameters:
    -----------
    vis    : complex array, the visibilities
    df     : float, the size of a single frequency channel
             in GHz
    nfavg  : the number of frequency channels to avg by
    Returns:
    -----------
    vis_ft     : complex array, the Fourier transform of the time-averaged 
                 visibilities, dimensions (baselines, delay)
    delay_arr  : array, the values of the delay pixels
    """
    nfbins = vis.shape[-1]//nfavg*nfavg
    if tavg:
        vis_ft = fftshift(fft(np.pad(vis[...,:nfbins].mean(1),
                                ((0,0), (0,nfbins))),axis=-1),axes=-1)
        vis_ft = vis_ft.reshape(vis_ft.shape[0],-1,2*nfavg).mean(-1)
    else:
        vis_ft = fftshift(fft(np.pad(vis[...,:nfbins],
                                ((0,0),(0,0),(0,nfbins)))
                              ,axis=-1),axes=-1)
        vis_ft = vis_ft.reshape(vis_ft.shape[0],
                                vis_ft.shape[1],
                                -1,2*nfavg).mean(-1)
    delay_arr = fftshift(fftfreq(nfbins))/df
    delay_arr = delay_arr.reshape(-1,nfavg).mean(-1)
    
    return vis_ft, delay_arr

def get_bad_times(source,nant):
    """Use delays on short time periods to flag bad antenna/time
    pairs in the calibrator data. These won't be used in the 
    gain calibration.
    Parameters:
    -----------
    source: src class
    nant  : int, the number of antennas in the array
    Returns:
    ---------
    bad_times: boolean array, (ntimes, nant), whether a time-antenna 
           pair should be flagged
    times: array, the time (in s, past the boolean day) for each 
           delay solution
    """
    error = 0
    # Solve the calibrator data on minute timescales
    cb = cc.calibrater.calibrater()
    error += not cb.open('{0}.ms'.format(source.name))
    error += not cb.setsolve(type='K',t='59s',
        refant=0,table='{0}2kcal'.format(source.name))
    error += not cb.solve()
    error += not cb.close()
    # Pull the solutions for the entire timerange and the 
    # 60-s data from the measurement set tables
    tb = cc.table.table()
    error += not tb.open('{0}2kcal'.format(source.name))
    antenna_delays = tb.getcol('FPARAM')
    npol = antenna_delays.shape[0]
    antenna_delays = antenna_delays.reshape(npol,-1,nant)
    times = tb.getcol('TIME').reshape(-1,nant)[:,0]
    error += not tb.close()
    tb = cc.table.table()
    error += not tb.open('{0}kcal'.format(source.name))
    kcorr = tb.getcol('FPARAM').reshape(npol,-1,nant)
    error += not tb.close()
    threshold = nant*npol//2
    bad_times = (np.abs(antenna_delays-kcorr)>1.5)
    bad_times[:,np.sum(np.sum(bad_times,axis=0),axis=1)
              >threshold,:] = np.ones((npol,1,nant))
    if error > 0:
        print('{0} errors occured during calibration'.format(error))
    return bad_times,times

def apply_calibration(source,cal):
    """Apply the calibration solution from the calibrator
    to the source.
    Parameters:
    -----------
    source : src class, the target source
    cal    : src class, the calibrator source
    """
    error = 0
    cb = cc.calibrater.calibrater()
    error += not cb.open('{0}.ms'.format(source.name))
    error += not cb.setapply(type='K',
                             table='{0}kcal'.format(cal.name))
    error += not cb.setapply(type='G',
                             table='{0}gacal'.format(cal.name))
    error += not cb.setapply(type='G',
                             table='{0}gpcal'.format(cal.name))
    error += not cb.correct()
    error += not cb.close()
    if error > 0:
        print('{0} errors occured during calibration'.format(error))
    return