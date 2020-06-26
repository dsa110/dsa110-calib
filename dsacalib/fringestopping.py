"""Functions used in fringestopping of DSA-110 visibilities.

Author: Dana Simard, dana.simard@astro.caltech.edu 11/2019

These functions use casatools to build sky models, divide
visibilities by sky models, and fringestop visibilities.

"""

# always import scipy before importing casatools
import scipy
import casatools as cc
import numpy as np
import astropy.units as u
from . import constants as ct
from .utils import to_deg
from numba import jit
from scipy.special import j1
from astropy.coordinates.angle_utilities import angular_separation

def calc_uvw(b, tobs, src_epoch, src_lon, src_lat,obs='OVRO_MMA'):
    """Calculates uvw coordinates.
    
    Uses CASA to calculate the u,v,w coordinates of the 
    baselines `b` towards a source or phase center
    (specified by `src_epoch`, `src_lon` and `src_lat`) at 
    the specified time and observatory.
    
    Parameters
    ----------
    b : ndarray
        The ITRF coordinates of the baselines.  Type float,
        shape (nbaselines, 3), units of meters.
    tobs : ndarray 
        An array of floats, the times in MJD for which to 
        calculate the uvw coordinates.
    src_epoch : str
        The epoch of the source or phase-center, as a 
        CASA-recognized string e.g. ``'J2000'`` or ``'HADEC'``
    src_lon : astropy quantity
        The longitude of the source or phase-center, in 
        degrees or an equivalent unit.  
    src_lat : astropy quantity
        The latitude of the source or phase-center, in 
        degrees or an equivalent unit.

    Returns
    -------
    u : ndarray
        The u-value for each time and baseline, in meters.
        Shape is ``(len(b), len(tobs))``.
    v : ndarray
        The v-value for each time and baseline, in meters.
        Shape is ``(len(b), len(tobs))``.
    w : ndarray
        The w-value for each time and baseline, in meters.
        Shape is ``(len(b), len(tobs))``.
    """
    tobs, b = set_dimensions(tobs=tobs, b=b)
    nt = tobs.shape[0]
    nb  = b.shape[0]
    bu = np.zeros((nt,nb))
    bv = np.zeros((nt,nb))
    bw = np.zeros((nt,nb))
    

    # Define the reference frame
    me = cc.measures()
    qa = cc.quanta()
    if obs is not None:
        me.doframe(me.observatory(obs))
    
    if type(src_lon.ndim) is not float and src_lon.ndim > 0:
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
def visibility_sky_model_worker(vis_model,bws,famps,f0,spec_idx,fobs):
    """Builds complex sky models.

    This is a worker to contain the for loop in the visibility model 
    calculation using jit. Modifies the input array `vis_model` in place.
        
    Parameters
    ----------
    vis_model : ndarray
        A placeholder for the output.  A complex array initialized to zeros,
        with the same shape as the array of visibilities you wish to model.  
        Dimensions (baseline,time,freq,polarization).
    bws : ndarray 
        The w component of the baselines towards the phase-center or towards 
        each source in the sky model.  Dimensions (sources,baselines).
    famps : ndarray
        The flux of each source at the reference frequency, in Jy.
    f0 : float
        The reference frequency, in GHz.
    spec_idx : float
        The spectral index for the frequency dependence of the source flux.
    fobs : ndarray
        The central frequency of each channel of the visibilities, in GHz.
    """
    for i in range(bws.shape[0]):
        vis_model += famps[i,...] * ((fobs/f0)**(spec_idx)) * np.exp(2j*np.pi/ct.c_GHz_m * fobs * bws[i,...])
    return 

def _py_visibility_sky_model_worker(vis_model,bws,famps,f0,spec_idx,fobs):
    """Builds complex sky models.

    A pure python version of `visibility_model_worker` for timing against.
    Modifies the input array `vis_model` in place.
    
    Parameters
    ----------
    vis_model : ndarray
        A placeholder for the output.  A complex array initialized to zeros,
        with the same shape as the array of visibilities you wish to model.  
        Dimensions (baseline,time,freq,polarization).
    bws : ndarray 
        The w component of the baselines towards the phase-center or towards 
        each source in the sky model.  Dimensions (sources,baselines).
    famps : ndarray
        The flux of each source at the reference frequency, in Jy.
    f0 : float
        The reference frequency, in GHz.
    spec_idx : float
        The spectral index for the frequency dependence of the source flux.
    fobs : ndarray
        The central frequency of each channel of the visibilities, in GHz.
    """
    for i in range(bws.shape[0]):
        vis_model += famps[i,...] * ((fobs/f0)**(spec_idx)) * np.exp(2j*np.pi/ct.c_GHz_m * fobs * bws[i,...])
    return 

def set_dimensions(fobs=None,tobs=None,b=None):
    """Sets the dimensions of arrays for fringestopping.

    Ensures that `fobs`, `tobs` and `b` are ndarrays with the correct
    dimensions for fringestopping using jit.  Any combination of these 
    arrays may be passed.  If no arrays are passed, an empty list is returned
    
    Parameters
    ----------
    fobs : float or ndarray
        The central frequency of each channel in GHz.  Defaults ``None``.
    tobs : float or ndarray
        The central time of each subintegration in MJD.  Defaults ``None``.
    b : ndarray
        The baselines in ITRF coordinates with dimensions (nbaselines,3)
        or (3) if a single baseline.  Defaults ``None``.
    
    Returns
    -------
    list 
        A list of the arrays passed, altered to contain the correct dimensions
        for fringestopping.  The list may contain:
        
        fobs : ndarray
            The central frequency of each channel in GHz with
            dimensions (channels,1).  Included if `fobs` is not set to ``None``.
            
        tobs : ndarray
            The central time of each subintegration in MJD with dimensions (time).
            Included if `tobs` is not set to ``None``.
            
        b : ndarray
            The baselines in ITRF coordaintes with dimensions (baselines, 3) 
            or (1,3) if a single baseline.  Included if ``b`` is not set to None.
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

def visibility_sky_model(vis_shape,vis_dtype,b,sources,tobs,fobs,lst,pt_dec):
    """Calculates the sky model visibilities.

    Calculates the sky model visibilities on the baselines `b` 
    and at times `tobs`.  Ensures that the returned sky model is the 
    same datatype and the same shape as specified by `vis_shape` and 
    `vis_dtype` to ensure compatability with jit.
    
    Parameters
    ----------
    vis_shape : tuple
        The shape of the visibilities: (baselines,time,frequency,polarization).
    vis_dtype: numpy datatype
        The datatype of the visibilities.
    b: real array
        The baselines to calculate visibilities for, shape (nbaselines, 3), 
        units of meters in ITRF coords.
    sources : list(src class instances)
        The sources to include in the sky model.
    tobs : float or ndarray
        The times at which to calculate the visibility model, in MJD.
    fobs : float or ndarray
        The frequencies at which to calculate the visibility model, in GHz.
    lst : ndarray
        The local sidereal time in radians for each time in tobs.
    pt_dec : float
        The antenna pointing declination in radians.
    
    Returns
    -------
    vis_model : ndarray
        The modelled complex visibilities, dimensions (baselines, 
        time, frequency,polarization).
    """
    fobs, tobs, b = set_dimensions(fobs,tobs, b)
    # Calculate the w-vector and flux towards each source
    # Array indices must be nsources, nbaselines, nt, nf, npol
    bws = np.zeros((len(sources),len(b),len(tobs),1,1))
    famps = np.zeros((len(sources),1,len(tobs),len(fobs),1)) 
    for i,src in enumerate(sources):
        bu,bv,bw       = calc_uvw(b, tobs, src.epoch, src.ra, src.dec)
        bws[i,:,:,0,0] = bw
        famps[i,0,:,:,0] = src.I*pb_resp(lst,pt_dec,
                                       src.ra.to_value(u.rad),
                                       src.dec.to_value(u.rad),fobs.squeeze())
    # Calculate the sky model using jit
    vis_model = np.zeros(vis_shape,dtype=vis_dtype)                                 
    visibility_sky_model_worker(vis_model,bws,famps,ct.f0,ct.spec_idx,fobs)
    return vis_model

def fringestop(vis,b,source,tobs,fobs,pt_dec,return_model=False):
    """Fringestops on a source.

    Fringestops on a source (or sky position) by dividing the input
    visibilities by a phase only model.  The input visibilities, vis, 
    are modified in place.
    
    Parameters
    ----------
    vis : ndarray
        The visibilities t be fringestopped, with dimensions
        (baselines,time,freq,pol). `vis` is modified in place.
    b : ndarray
        The ITRF coordinates of the baselines in meters, with dimensions
        (3,baselines).
    source : src class instance
        The source to fringestop on.
    tobs : ndarray
        The observation time (the center of each bin) in MJD.
    fobs : ndarray
        The observing frequency (the center of each bin) in GHz.
    pt_dec : float
        The pointing declination of the array, in radians.
    return_model : boolean
        If ``True``, the fringestopping model is returned.  Defaults 
        ``False``.

    Returns
    -------
    vis_model : ndarray
        The phase-only visibility model by which the
        visiblities were divided.  Returned only if 
        `return_model` is set to ``True``.
    """
    fobs,tobs,b = set_dimensions(fobs,tobs,b)
    bws = np.zeros((len(b),len(tobs),1,1))
    bu,bv,bw  = calc_uvw(b, tobs, source.epoch, source.ra, source.dec)
    # note that the time shouldn't matter bleow
    bup,bvp, bwp = calc_uvw(b,tobs[len(tobs)//2],'HADEC',0.*u.rad,pt_dec*u.rad)
    bw = bw - bwp
    vis_model = np.exp(2j*np.pi/ct.c_GHz_m * fobs * bw[...,np.newaxis,np.newaxis])
    vis /= vis_model
    if return_model:
        return vis_model
    else: 
        return
    

def divide_visibility_sky_model(vis,b, sources, tobs, fobs,lst,pt_dec,
                    phase_only=True,return_model=False):
    """Calculates and applies the sky model visibilities.

    Calculates the sky model visibilities on the baselines `b` and at
    times `tobs`.  Divides the input visibilities, `vis`, by the sky model.
    `vis` is modified in-place.
    
    Parameters
    ----------
    vis : ndarray
        The observed complex visibilities, with dimensions 
        (baselines,time,frequency,polarization).  `vis`
        will be updated in place to the fringe-stopped visibilities
    b : ndarray
        The baselines for which to calculate visibilities, shape (nbaselines, 3), 
        units of meters in ITRF coords.
    sources : list(src class objects)
        The list of sources to include in the sky model.
    tobs : float or ndarray
        The times for which to calculate the visibility model in MJD.
    fobs: float or arr(float)
        The frequency for which to calculate the model in GHz.
    phase_only: Boolean
        If  set to ``True``, the fluxes of all sources will be set to 1 with a
        spectral index of 0. If set to ``False``, the fluxes of all sources, 
        spectral index, and primary beam correction will all be included 
        in the sky model.  Defaults to ``True``.
    return_model: boolean
        If set to ``True``, the visibility model will be returned.  
        Defaults to ``False``.
    
    Returns
    -------
    vis_model: ndarray
        The modelled visibilities, dimensions 
        (baselines, time, frequency,polarization).
        Returned only if `return_model` is set to True.
    """
    vis_model = visibility_sky_model(vis.shape,vis.dtype,b,sources,tobs,fobs,lst,pt_dec)
    vis /= vis_model
    if return_model:
        return vis_model
    else:
        return
    
def amplitude_sky_model(source,ant_ra,pt_dec,fobs,
                        dish_dia=4.65,spind=0.7):
    """Computes the amplitude sky model due to the primary beam.

    Computes the amplitude sky model for a single source due to the primary 
    beam response of an antenna.
    
    Parameters
    ----------
    source : src class instance
        The source to model.  The source flux (`src.I`), right ascension
        (`src.ra`) and declination (`src.dec`) must be specified.
    ant_ra : ndarray
        The right ascension pointing of the antenna in each time bin
        of the observation, in radians.  If az=0 deg or az=180 deg, this is 
        the lst of each time bin in the observation in radians.
    pt_dec : float
        The pointing declination of the observation in radians.
    fobs : array
        The observing frequency of the center of each channel in GHz.
    
    Returns
    -------
    ndarray
        The calculated amplitude sky model.
    """
    # Should add spectral index 
    return source.I * (fobs/1.4)**(-spind) * pb_resp(ant_ra,pt_dec,
                source.ra.to_value(u.rad),
                source.dec.to_value(u.rad),fobs,dish_dia)
    
def pb_resp(ant_ra,ant_dec,src_ra,src_dec,freq,dish_dia=4.65):
    """Computes the primary beam response towards a direction on the sky.
    
    Returns a value between 0 and 1 for each value passed in `ant_ra`. 

    Parameters
    ----------
    ant_ra : float or ndarray
        The antenna right ascension pointing in radians.  If an array, must be 
        one-dimensional.
    ant_dec : float 
        The antenna declination pointing in radians.  
    src_ra : float
        The source right ascension in radians.
    src_dec : float
        The source declination in radians.
    freq : ndarray
        The frequency of each channel in GHz.
    dish_dia : float
        The dish diameter in meters.  Defaults to ``4.65``.
      
    Returns
    -------
    pb : ndarray
        The primary beam response, dimensions (`ant_ra`,`freq`).
    """
    dis = angular_separation(ant_ra,ant_dec,src_ra,src_dec)
    lam  = 0.299792458/freq
    pb   = (2.0*j1(np.pi*dis[:,np.newaxis]*dish_dia/lam)/(np.pi*dis[:,np.newaxis]*dish_dia/lam))**2
    return pb
