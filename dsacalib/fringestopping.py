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
                    phase_only=True,return_model=False):
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
          spectral index of 0.  
          if False, the fluxes of all sources, spectral index, 
          and primary beam correction will all be included in the sky
          model
        return_model: Boolean
          if True, the visibility model will be returned
    
    Returns:
        vis_model: complex array
          the modelled visibilities, dimensions (baselines, time, frequency,polarization)
          returned only if return_model is true
    """
  
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
        else:
            famps[i,0,:,:,0] = src.I*pb_resp(lst,pt_dec,
                                       src.ra.to_value(u.rad),
                                       src.dec.to_value(u.rad),
                                       fobs.squeeze())

    # Calculate the sky model using jit
    vis_model = np.zeros(vis.shape,dtype=vis.dtype)
    visibility_sky_model(vis,vis_model,bws,famps,ct.f0,0. if phase_only else ct.spec_idx,fobs)
    if return_model:
        return vis_model
    else:
        return
    
def amplitude_sky_model(src,lst,pt_dec,fobs):
    return src.I * pb_resp(lst,pt_dec,src.ra.to_value(u.rad),
                           src.dec.to_value(u.rad),fobs)
    
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
    pb   = (2.0*j1(np.pi*dis[:,np.newaxis]*dish_dia/lam)/(np.pi*dis[:,np.newaxis]*dish_dia/lam))**2
    return pb