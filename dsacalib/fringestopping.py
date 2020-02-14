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
from astropy.time import Time
from scipy.special import j1
from astropy.coordinates.angle_utilities import angular_separation

def calc_uvw(b, tobs, src_epoch, src_lon, src_lat,obs='OVRO_MMA'):
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
    if obs is not None:
        me.doframe(me.observatory(obs))
    
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

def generate_fringestopping_table(b,nint=ct.nint,tsamp=ct.tsamp,pt_dec=ct.pt_dec,outname='fringestopping_table',                                mjd0=58849.0,transpose=False):
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
    dt = np.arange(nint)*tsamp
    dt = dt - np.median(dt)
    ha = dt * 360/ct.seconds_per_sidereal_day
    bu,bv,bw = calc_uvw(b,mjd0+dt/ct.seconds_per_day,'HADEC',
                       ha*u.deg, np.ones(ha.shape)*to_deg(ct.pt_dec))
    if nint%2 == 1:
        bwref = bw[:,(nint-1)//2]
    else:
        bu,bv,bwref = calc_uvw(b,mjd0,'HADEC',
                              0.*u.deg,to_deg(ct.pt_dec))
        bwref = bwref.squeeze()
    bw = bw - bwref[:,np.newaxis]
    if transpose:
        bw = bw.T
    np.savez(outname,
             dec=pt_dec,ha=ha,bw=bw,bwref=bwref)
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
        vis_model += famps[i,...] * ((fobs/f0)**(spec_idx)) * np.exp(2j*np.pi/ct.c_GHz_m * fobs * bws[i,...])
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

def divide_visibility_sky_model(vis,b, sources, tobs, fobs,lst,pt_dec,
                            phase_only=False,return_model=False,pbcorr=True):
    """ Calculates the sky model visibilities the baselines b and divides the 
    input visibilities by the sky model
    
    Args:
        vis: complex array
          the visibilities, (baselines,time,frequency,polarization)
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
          the modelled visibilities, dimensions (baselines, time, frequency,polarization)
          returned only if return_model is true
    """
  
    # Calculate the total visibility by adding each source
    # Eventually we will also have to include the primary beam here
    
    fobs, tobs, b = set_dimensions(fobs,tobs, b)
    
    # Calculate the w-vector and flux towards each source
    # Array indices must be nsources, nbaselines, nt, nf, npol
    bws = np.zeros((len(sources),len(b),len(tobs),1,1))
    famps = np.zeros((len(sources),1,len(tobs),len(fobs),1)) 
    for i,src in enumerate(sources):
        bu,bv,bw       = calc_uvw(b, tobs, src.epoch, src.ra, src.dec)
        bws[i,:,:,0,0] = bw
        # Need to add amplitude model of the primary beam here
        if phase_only:
            famps[i,0,:,:,0] = 1.
        elif pbcorr:
            famps[i,0,:,:,0]=src.I*pb_resp(lst,pt_dec,
                                       src.ra.to_value(u.rad),
                                       src.dec.to_value(u.rad),
                                       fobs.squeeze())
        else:
            famps[i,0,:,:,0]=src.I

    # Calculate the sky model using jit
    vis_model = np.zeros(vis.shape,dtype=vis.dtype)
    visibility_sky_model(vis,vis_model,bws,famps,ct.f0,0. if phase_only else ct.spec_idx,fobs)
    if return_model:
        return vis_model,bws
    else:
        return
    
def amplitude_sky_model(src,lst,pt_dec,fobs):
    return src.I * pb_resp(lst,pt_dec,src.ra.to_value(u.rad),
                           src.dec.to_value(u.rad),fobs)

def fringestop_on_zenith_worker_T(vis,vis_model,nint,nbl,nchan,npol):
    vis.shape=(-1,nint,nbl,nchan,npol)
    vis /= vis_model
    return vis.mean(axis=1)
    
def fringestop_on_zenith_worker(vis,vis_model,nint,nbl,nchan,npol):
    vis.shape=(nbl,-1,nint,nchan,npol)
    vis /= vis_model
    return vis.mean(axis=2)

def zenith_visibility_model_T(fobs,fstable='fringestopping_table.npz'):
    data = np.load(fstable)
    bws = data['bw']
    vis_model = np.exp(2j*np.pi/ct.c_GHz_m * fobs[:,np.newaxis] *
                       bws[np.newaxis,:,:,np.newaxis,np.newaxis])
    return vis_model

def zenith_visibility_model(fobs,fstable='fringestopping_table.npz'):
    data = np.load(fstable)
    bws = data['bw']
    vis_model = np.exp(2j*np.pi/ct.c_GHz_m * fobs[:,np.newaxis] * 
                       bws[:,np.newaxis,:,np.newaxis,np.newaxis])
    return vis_model

def fringestop_on_zenith_T(vis,vis_model,nint):
    if vis.shape[0]%nint != 0:
        npad = nint - vis.shape[0]%nint
        print('Warning: Padding array to integrate.  Last bin contains only {0}% real data.'.format((nint-npad)/nint*100))
        vis = np.pad(vis,((0,npad),(0,0),(0,0),(0,0)),mode='constant',
                 constant_values=(0.,))
    nt,nbl,nchan,npol = vis.shape
    vis = fringestop_on_zenith_worker_T(vis,vis_model,nint,nbl,nchan,npol)
    return vis

def fringestop_on_zenith(vis,fobs,fstable='fringestopping_table.npz',
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
    
    # Create the visibility model
    vis_model = zenith_visibility_model(fobs,fstable)
    
    # Reshape visibilities
    # nbaselines, nsubint, nt, nf, npol
    npad = nint - vis.shape[1]%nint
    if npad == nint: npad = 0 
    if npad != 0: print('Warning: Padding array to integrate.  Last bin contains only {0}% real data.'.format((nint-npad)/nint*100))
    vis = np.pad(vis,((0,0),(0,npad),
                    (0,0),(0,0)),
                            mode='constant',constant_values=
                    (0.,))

    nbl,nt,nchan,npol = vis.shape
    vis = fringestop_on_zenith_worker(vis,vis_model,nint,nbl,nchan,npol)
    
    if return_model:
        return vis,vis_model
    else:
        return vis

def fringestop_multiple_phase_centers(vis,b, sources, tobs, fobs, 
                            phase_only=False,return_model=False,
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
          (ncenters, baselines, time, frequency, polarization)
        vis_model: complex array
          the modelled visibilities, dimensions (ncenters, baselines, time, frequency,polarization)
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

def write_fs_delay_table(msname,source,blen,tobs,nant):
    """Writes the delays needed to fringestop on a source to a delay calibration
    table in casa format. 
    
    Not tested.
    
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
        
    Returns:
    """
    nt = tobs.shape[0]
    bu,bw,bw = calc_uvw(blen, tobs, source.epoch, source.ra, source.dec)
    
    ant_delay = np.zeros((nt,nant))
    ant_delay[:,1:] = bw[:nant-1,:].T/ct.c_GHz_m
    
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

def pb_resp(ant_ra,ant_dec,src_ra,src_dec,freq,dish_dia=4.65):
    """ Compute the primary beam response
    Args:
      ant_ra: float
        antenna right ascension pointing in radians
      ant_dec: float
        antenna declination pointing in radiants
      src_ra: float
        source right ascension in radians
      src_dec: float
        source declination in radians
      freq: array(float)
        the frequency of each channel in GHz
      dish_dia: float
        the dish diameter in m 
      
    Returns:
      pb: array(float)
        dimensions distance, freq
        the primary beam response
    """
    dis = angular_separation(ant_ra,ant_dec,src_ra,src_dec)
    lam  = 0.299792458/freq
    print(dis.shape)
    print(lam.shape)
    pb   = (2.0*j1(np.pi*dis[:,np.newaxis]*dish_dia/lam)/(np.pi*dis[:,np.newaxis]*dish_dia/lam))**2
    return pb