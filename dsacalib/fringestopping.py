"""
DSACALIB/FRINGESTOPPING.PY

Dana Simard, dana.simard@astro.caltech.edu 11/2019

Casa-based routines for calculating and applying fringe-stopping phases
to visibilities
"""

import __casac__ as cc
import numpy as np
import astropy.units as u
from . import constants as ct
from .utils import to_deg
from numba import jit

def calc_uvw(b, tobs, src_epoch, src_lon, src_lat):
    """ Calculates the uvw coordinates of a source.
    
    Args:
        b: real array
          the baseline, shape (nbaselines, 3), m in ITRF coords
        tobs: real array
          times to calculate coordinates at
        src_epoch: str
          casa-recognized descriptor of the source coordinates
          e.g. 'J2000' or 'HADEC'
        src_lon: astropy quantity
          longitude of source
        src_lat: astropy quantity
          latitute of source

    Returns:
        u: real array
          u, shape (nbaselines, ntimes), units m
        v: real array
          v, shape (nbaselines, ntimes), units m
        w: real array 
          w, shape (nbaselines, ntimes), units m
    """
    tobs, b = set_dimensions(tobs=tobs, b=b)
    nt = tobs.shape[0]
    nb  = b.shape[0]
    bu = np.zeros((nt,nb))
    bv = np.zeros((nt,nb))
    bw = np.zeros((nt,nb))
    

    # Define the reference frame
    me = cc.measures.measures()
    qa = cc.quanta.quanta()
    me.doframe(me.observatory('OVRO_MMA'))
    
    if src_lon.ndim > 0:
        assert src_lon.ndim == 1
        assert src_lon.shape[0] == nt
        assert src_lat.shape[0] == nt
        direction_set = False
    else:
        if (src_epoch == 'HADEC') and (nt>1):
            raise TypeError('HA and DEC must be specified at each time in tobs.')
        me.doframe(me.direction(src_epoch, 
                            qa.quantity(src_lon.to_value(u.deg),'deg'), 
                            qa.quantity(src_lat.to_value(u.deg),'deg')))
        direction_set = True

    contains_nans=False

    for i in range(nt):
        me.doframe(me.epoch('UTC',qa.quantity(tobs[i],'d'))) 
        if not direction_set:
            me.doframe(me.direction(src_epoch, 
                qa.quantity(src_lon[i].to_value(u.deg),'deg'), 
                qa.quantity(src_lat[i].to_value(u.deg),'deg')))
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

def generate_fringestopping_table(b,nint=ct.nint,tsamp=ct.tsamp,pt_dec=ct.pt_dec,
                                 method='B',mjd0=58849.0):
    """Generates a table of the w vectors towards a source to use in fringe-
    stopping.  And writes it to a numpy pickle file named 
    fringestopping_table.npz
    
    Args:
      b: array or list
        shape (nbaselines, 3), the lengths of the baselines in ITRF coordinates
      nint: int
        the number of time integrations to calculate the table for
      tsamp: float
        the sampling time in seconds
      pt_dec: str
        the pointing declination, eg '+73d00m00.0000s'
      method: str
        either 'A' (reference w to ha=0 dec=0 at midpoint of observation), 
        'B' (reference w to ha=0, dec=pt_dec at midpoint of observation)
      mjd0: float
        the mjd of the start time
        
    Returns:
      Nothing
    """
    dur = tsamp*nint
    dt  = (np.arange(nint)+0.5-nint/2)*tsamp
    tobs= mjd0 + (dt*u.s).to_value(u.d)
    ha  = (dt*u.s*15*u.deg/u.h).to(u.deg)
    if method=='A':
        ref = calc_uvw(b,tobs[len(tobs)//2],'HADEC',0*u.deg,0*u.deg)
    if method=='B':
        ref = calc_uvw(b,tobs[len(tobs)//2],'HADEC',0*u.deg,to_deg(pt_dec))
    bu,bv,bw = np.subtract(calc_uvw(b, tobs, 'HADEC', ha, 
        np.ones(len(ha))*to_deg(pt_dec)),ref)
    np.savez('fringestopping_table',
             dec=pt_dec,ha=ha,bw=bw,ref=ref)
    return

@jit(nopython=True)
def visibility_sky_model(vis,vis_model,bws,famps,f0,spec_idx,fobs):
    """A worker to contain the for loop in the visibility model calcuulation.
        
    Args:
      vis: complex array
        the visibilities, dimensions (baselines, time, frequency, polarization)
      vis_model: complex array
        an array initialized to all zeros, the same dimensions as the array
        of visibilities
      bws: real array
        the w component of the baselines towards the phasing direction, or the
        the w comonents of the baselines towards each source in the sky model
      famps: real array
        the intensities of each source at f0
      f0: float
        the reference frequency for famps, in GHz
      spec_idx: float
        the spectral index to use in the source model
      fobs: 
        the central frequency of each channel, in GHz

    Returns:
      Nothing.  Modifies vis_model and vis in-place.
    """
    for i in range(bws.shape[0]):
        vis_model += famps[i] * ((fobs/f0)**(spec_idx)) * np.exp(2j*np.pi/ct.c_GHz_m * fobs * bws[i,...])
    vis /= vis_model
    return 

def _py_visibility_sky_model(vis,vis_model,bws,famps,f0,spec_idx,fobs):
    """A pure python version of visibility_model_worker for timing against.
    """
    for i in range(bws.shape[0]):
        vis_model += famps[i] * ((fobs/f0)**(spec_idx)) * np.exp(2j*np.pi/ct.c_GHz_m * fobs * bws[i,...])
    vis /= vis_model
    return 

def set_dimensions(fobs=None,tobs=None,b=None):
    """Ensures that fobs, tobs and b are ndarrays with the correct
    dimensions.
    
    Args:
      fobs: float or ndarray(float)
        The central frequency of each channel, in GHz
      tobs: float or ndarray(float)
        The central time of each subintegration, in mjd
      b: list or ndarray
        The baselines in ITRF coordinates, dimensions (nbaselines,3)
        or (3) if a single baseline
    
    Returns:
      fobs: ndarray
        dimensions (nchannels,1)
      tobs: ndarray
        dimensions (nt)
      b: ndarray
        dimensions (nbaselines, 3) 
    """
    to_return = []
    if fobs is not None:
        if type(fobs)!=np.ndarray: fobs = np.array(fobs)
        if fobs.ndim < 1: fobs = fobs[np.newaxis]
        fobs = fobs[:,np.newaxis]
        to_return += [fobs]
        
    if tobs is  not None:
        if type(tobs)!=np.ndarray: tobs = np.array(tobs)
        if tobs.ndim < 1: tobs = tobs[np.newaxis]
        to_return += [tobs]
        
    if b is not None:
        if type(b)==list: b = np.array(b)
        if b.ndim < 2: b = b[np.newaxis,:]
        to_return += [b]
    return to_return

def divide_visibility_sky_model(vis,b, sources, tobs, fobs, prefs=True,
                                fstable='fringestopping_table.npz',
                            phase_only=False,return_model=False):
    """ Calculates the sky model visibilities the baselines b and divides the 
    input visibilities by the sky model
    
    Args:
        vis: complex array
          the visibilities, (baselines,time,frequency)
          will be updated in place to the fringe-stopped visibilities
        b: real array
          baselines to calculate visibilities for, shape (nbaselines, 3), 
          units m in ITRF coords
        sources: list(src)
          list of sources to include in the sky model 
        tobs: float or arr(float)
          times to calculate visibility model for, mjd
        fobs: float or arr(float)
          frequency to calculate model for, in GHz
        phase_only: Boolean
          if True, the fluxes of all sources will be set to 1 with a
          spectral index of 0.  For faster performance, fringestop_on_zenith 
          and fringestop_multiple_phase_centers are preferred.
        return_model: Boolean
          if True, the visibility model will be returned
    
    Returns:
        vis_model: complex array
          the modelled visibilities, dimensions (baselines, time, frequency)
          returned only if return_model is true
    """
  
    # Calculate the total visibility by adding each source
    # Eventually we will also have to include the primary beam here
    
    fobs, tobs, b = set_dimensions(fobs,tobs, b)
    
    # Calculate the w-vector and flux towards each source
    # Array indices must be nsources, nbaselines, nt, nf, npol
    bws = np.zeros((len(sources),len(b),len(tobs),1,1))
    famps = np.zeros(len(sources)) 
    for i,src in enumerate(sources):
        bu,bv,bw       = calc_uvw(b, tobs, src.epoch, src.ra, src.dec)
        bws[i,:,:,0,0] = bw
        # Need to add amplitude model of the primary beam here
        famps[i,...]   = 1. if phase_only else src.I
    if prefs:
        data = np.load(fstable)
        bur, bvr, bwr = data['ref']
        bws = bws - bwr[...,np.newaxis,np.newaxis]
    # Calculate the sky model using jit
    vis_model = np.zeros(vis.shape,dtype=vis.dtype)
    visibility_sky_model(vis,vis_model,bws,famps,ct.f0,0. if phase_only else ct.spec_idx,fobs)
    if return_model:
        return vis_model
    else:
        return
    
def fringestop_on_zenith(vis,fobs,bw_file='fringestopping_table.npz',
                        return_model=False,nint=ct.nint):
    """Fringestops on HA=0, DEC=pointing declination for the midpoint of each   
    integration and then integrates the data.  The number of samples to integrate 
    by is set by the length of the bw array in bw_file.
    
    Args:
      vis: complex array
        The input visibilities, dimensions (baselines,time,freq,pol)
      fobs: real array
        The central frequency of each channel, in GHz
      bw_file: str
        The path to the .npz file containing the bw array (the length of the
        w-vector in m)
      nint: int
        the number of samples (in time) to integrate
      
    Returns:
      vis: complex array
        The fringe-stopped and integrated visibilities
    """
    fobs, = set_dimensions(fobs)
    
    data = np.load(bw_file)
    bws  = data['bw']
    assert bws.shape[0] == vis.shape[0], \
        'w vector and visibility have different numbers of baselines'
    nint = bws.shape[1]
    
    # Reshape visibilities
    # nbaselines, nsubint, nt, nf, npol
    npad = nint - vis.shape[1]%nint
    if npad == nint: npad = 0 
    vis = np.pad(vis,((0,0),(0,npad),
                    (0,0),(0,0)),
                            mode='constant',constant_values=
                    (np.nan,)).reshape(vis.shape[0],-1,nint,
                    vis.shape[2],vis.shape[3])
    vis_model = np.exp(2j*np.pi/ct.c_GHz_m * fobs * 
                       bws[:,np.newaxis,:,np.newaxis,np.newaxis])
    vis = vis/vis_model
    # This mean is slow - need to change order of the axes at some point
    #out = np.zeros((vis.shape[0],vis.shape[1],vis.shape[3],vis.shape[4]),dtype=vis.dtype)
    #jitmean_fszenith(vis,out)
    vis = vis.mean(axis=2)
    if return_model:
        return vis,vis_model
    else:
        return vis

def fringestop_multiple_phase_centers(vis,b, sources, tobs, fobs, 
                            phase_only=False,return_model=False,prefs=True,
                                     fstable='fringestopping_table.npz'):
    """Fringestops on multiple phase centers.
    
    Args:
        vis: complex array
          the visibilities, (baselines,time,frequency)
          will be updated in place to the fringe-stopped visibilities
        b: real array
          baselines to calculate visibilities for, shape (nbaselines, 3), 
          units m in ITRF coords
        sources: list(src)
          list of sources to include in the sky model 
        tobs: float or arr(float)
          times to calculate visibility model for, mjd
        fobs: float or arr(float)
          frequency to calculate model for, in GHz
        phase_only: Boolean
          if True, the fluxes of all sources will be set to 1 with a
          spectral index of 0.  For faster performance, fringestop_on_zenith 
          and fringestop_multiple_phase_centers are preferred.
        return_model: Boolean
          if True, the visibility model will be returned
    
    Returns:
        vis: complex array
          the fringe-stopped visibilities, dimensions 
          (ncenters, baselines, time, frequency)
        vis_model: complex array
          the modelled visibilities, dimensions (ncenters, baselines, time, frequency)
          returned only if return_model is true
    """
    fobs,tobs,b = set_dimensions(fobs,tobs,b)
    
    # Array indices must be nsources, nbaselines, nt, nf, npol
    bws = np.zeros((len(sources),len(b),len(tobs),1,1))
    famps = np.zeros((len(sources),1,1,1,1)) 
    for i,src in enumerate(sources):
        bu,bv,bw       = calc_uvw(b, tobs, src.epoch, src.ra, src.dec)
        bws[i,:,:,0,0] = bw
        famps[i,...]   = src.I
    if prefs:
        data = np.load(fstable)
        bur, bvr, bwr = data['ref']
        bws = bws - bwr[...,np.newaxis,np.newaxis]
    if phase_only:
        vis_model = np.exp(2j*np.pi/ct.c_GHz_m * fobs * bws)
    else:
        vis_model = famps * ((fobs/ct.f0)**(ct.spec_idx)) * np.exp(2j*np.pi/ct.c_GHz_m * fobs * bws)
    vis = vis/vis_model
    
    if return_model:
        return vis, vis_model
    else:
        return vis
    
def fast_calc_uvw(src_ra, src_dec, mjd, blen):
    '''Calculates the projected uvw track for the baseline blen.
    Negative u points West, Negative v points South.

    Args:
      src_ra: float
        the ra of the phase center, radians
      src_dec: float
        the dec of the phase center, radians
      mjd: float or array(float)
        the time to calculate uvw coordinates at
      blen: array(float)
        the itrf coordinates of the baselines

    Returns:
        uvw: array
          the u, v and w tracks for each baseline
    '''
    src_dec = np.array(src_dec).reshape(1)
    if type(mjd)==float:
        H = (Time(mjd).sidereal_time('mean',longitude=ct.ovro_lon*u.rad
                                    ).to_value(u.rad)-src_ra)%2*np.pi
    else:
        H = (Time(mjd[0]).sidereal_time('mean',longitude=ct.ovro_lon*u.rad
                                      ).to_value(u.rad) - src_ra 
             + ((mjd- mjd[0])*ct.seconds_per_day/ct.seconds_per_sidereal_day
               )*2*np.pi)%(2*np.pi)
    trans = np.array([[ np.sin(H),                 np.cos(H),                0],
                      [-np.sin(src_dec)*np.cos(H), np.sin(src_dec)*np.sin(H),np.cos(src_dec)],
                      [ np.cos(src_dec)*np.cos(H),-np.cos(src_dec)*np.sin(H),np.sin(src_dec)]])
    return np.dot(trans,b)

def write_fs_delay_table(msname,source,blen,tobs,nant,fstable='fringestopping_table.npz'):
    """Writes the delays needed to fringestop on a source to a delay calibration
    table in casa format. 
    
    Args:
      msname: str
        The prefix of the ms for which this table is generated
        Note: doesn't open the ms
      source: src class
        The source (or location) to fringestop on
      blen: array, float
        The ITRF coordinates of the baselines
      tobs: array, float
        The observation time of each time bin in mjd
      nant: int
        The number of antennas in the array
      fstable: str
        The path to the numpy file containing the delays used to 
        fringestop at zenith
        
    Returns:
    """
    nt = tobs.shape[0]

    data = np.load(fstable)
    bur,bvr,bwr = data['ref']
    
    ant_delay = np.zeros((nt,nant))
    # Does the order here make sense? Casa will apply the negative, so it
    # will subtract the reference delay and then add the new delay - that makes sense
    ant_delay[:,1:] = -(bw[:nant-1,:]-bwr[:nant-1,:]).T/ct.c_GHz_m
    
    error = 0
    tb = cc.table.table()
    error += not tb.open('{0}/templatekcal'.format(ct.pkg_data_path))
    error += not tb.copy('{0}_{1}_fscal'.format(msname,source.name))
    error += not tb.close()
    
    error += not tb.open('{0}_{1}_fscal'.format(msname,source.name),nomodify=False)
    error += not tb.addrows(nant*nt - tb.nrows())
    error += not tb.flush()
    assert(tb.nrows() == nant*nt)
    error += not tb.putcol('TIME',np.tile((tobs*u.d).to_value(u.s).reshape(-1,1),(1,nant)).flatten())
    error += not tb.putcol('FIELD_ID',np.zeros(nt*nant,dtype=np.int32))
    error += not tb.putcol('SPECTRAL_WINDOW_ID',np.zeros(nt*nant,dtype=np.int32))
    error += not tb.putcol('ANTENNA1',np.tile(np.arange(nant,dtype=np.int32).reshape(1,nant),(nt,1)).flatten())
    error += not tb.putcol('ANTENNA2',-1*np.ones(nt*nant,dtype=np.int32))
    error += not tb.putcol('INTERVAL',np.zeros(nt*nant,dtype=np.int32))
    error += not tb.putcol('SCAN_NUMBER',np.ones(nt*nant,dtype=np.int32))
    error += not tb.putcol('OBSERVATION_ID',np.zeros(nt*nant,dtype=np.int32))
    error += not tb.putcol('FPARAM',np.tile(ant_delay.reshape(1,-1),(2,1)).reshape(2,1,-1))
    error += not tb.putcol('PARAMERR',np.zeros((2,1,nt*nant),dtype=np.float32))
    error += not tb.putcol('FLAG',np.zeros((2,1,nt*nant),dtype=bool))
    error += not tb.putcol('SNR',np.zeros((2,1,nt*nant),dtype=np.float64))
    #error += not tb.putcol('WEIGHT',np.ones((nt*nant),dtype=np.float64)) 
    # For some reason, WEIGHT is empty in the template ms, so we don't need to
    # modify it
    error += not tb.flush()
    error += not tb.close()
    
    if error > 0:
        print('{0} errors occured during calibration'.format(error))
    
    return