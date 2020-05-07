"""
DSA_UTILS.PY

Dana Simard, dana.simard@astro.caltech.edu, 10/2019

Modified for python3 from DSA-10 routines written by 
Vikram Ravi, Harish Vendantham

Routines to interact w/ fits visibilities recorded by DSA-10,
hdf5 visibilities recorded by DSA-110, and
visibility in CASA measurement sets
"""

# To do:
# Replace to_deg w/ astropy versions

# always import scipy before importing casatools
import scipy
import casatools as cc
import astropy.io.fits as pf
import astropy.units as u
import numpy as np
from astropy.time import Time
from . import constants as ct
from antpos.utils import *
import h5py
from astropy.utils import iers
iers.conf.iers_auto_url_mirror = ct.iers_table
from scipy.ndimage.filters import median_filter
from dsautils import dsa_store
import dsautis.dsa_syslog as dsl

logger = dsl.DsaSyslogger()    
logger.subsystem("software")
logger.app("dsacalib")
de = dsa_store.DsaStore()

class src():
    """ Simple class for holding source parameters.
    
    Args:
        name: str
          identifier for your source
        I: float
          the flux in Jy
        ra: str
          right ascension e.g. "12h00m19.21s"
        dec: str
          declination e.g. "+73d00m45.7s"
        epoch: str
          the epoch of the ra and dec, default "J2000"
        pa: float
          the position angle in degrees
        maj_axis: float
          the major axis in arcseconds
        min_axis: float
          the minor axis in arcseconds
           
    Returns:
    """
    def __init__(self,name,ra,dec,I=1.,epoch='J2000',
                pa=None,maj_axis=None,min_axis=None):
        self.name = name
        self.I = I
        if type(ra) is str:
            self.ra = to_deg(ra)
        else:
            self.ra = ra
        if type(dec) is str:
            self.dec = to_deg(dec)
        else:
            self.dec = dec
        self.epoch = epoch
        self.pa = pa
        if maj_axis is None:
            self.maj_axis = None
        else:
            self.maj_axis = maj_axis*u.arcsecond
        if min_axis is None:
            self.min_axis = None
        else:
            self.min_axis = min_axis*u.arcsecond

def to_deg(string):
    """ Converts a string direction to degrees.

    Args:
        string: str
          ra or dec in string format e.g. "12h00m19.21s" or "+73d00m45.7s"

    Returns:
        deg: astropy quantity
          the angle in degrees
    """
    if 'h' in string:
        h,m,s = string.strip('s').strip('s').replace('h','m').split('m')
        deg = (float(h)+(float(m)+float(s)/60)/60)*15*u.deg
    else:
        sign = -1 if '-' in string else 1
        d,m,s = string.strip('+-s').strip('s').replace('d','m').split('m')
        deg = (float(d)+(float(m)+float(s)/60)/60)*u.deg*sign
    return deg

def read_hdf5_file(fl,source=None,dur=50*u.min,autocorrs=True,
                   badants=None,quiet=True):
    
    if source is not None:
        stmid = source.ra.to_value(u.rad)
        seg_len = (dur/2*(15*u.deg/u.h)).to_value(u.rad)
        
    with h5py.File(fl, 'r') as f:
        antenna_order = list(f['antenna_order'][...])
        nant = len(antenna_order)
        fobs = f['fobs_GHz'][...]
        mjd = (f['time_seconds'][...]+f['tstart_mjd_seconds'])/ct.seconds_per_day 
        nt = len(mjd)
        tsamp  = (mjd[-1] - mjd[0])/(nt-1)*ct.seconds_per_day


        st0 = Time(mjd[0], format='mjd').sidereal_time(
            'apparent',longitude=ct.ovro_lon*u.rad).radian 
    
        st  = np.angle(np.exp(1j*(st0 + 2*np.pi/ct.seconds_per_sidereal_day*
                          np.arange(nt)*tsamp)))
    
        st1 = Time(mjd[-1],format='mjd').sidereal_time(
            'apparent',longitude=ct.ovro_lon).radian

        if source is not None:
            if not quiet:
                print("\n-------------EXTRACT DATA--------------------")
                print("Extracting data around {0}".format(stmid*180/np.pi))
                print("{0} Time samples in data".format(nt))
                print("LST range: {0:.1f} --- ({1:.1f}-{2:.1f}) --- {3:.1f}deg".format(st[0]*180./np.pi,(stmid-seg_len)*180./np.pi, (stmid+seg_len)*180./np.pi,st[-1]*180./np.pi))

            I1 = np.argmax(np.absolute(np.exp(1j*st)+
                           np.exp(1j*stmid)*np.exp(-1j*seg_len)))
            I2 = np.argmax(np.absolute(np.exp(1j*st)+
                           np.exp(1j*stmid)*np.exp(1j*seg_len)))
            transit_idx = np.argmax(np.absolute(np.exp(1j*st)+
                           np.exp(1j*(stmid))))
            
            mjd = mjd[I1:I2]
            st  = st[I1:I2]
            vis = f['vis'][I1:I2,...]
            if not quiet:
                print("Extract: {0} ----> {1} sample; transit at {2}".format(I1,I2,I0))
                print("----------------------------------------------")

        else:
            vis = f['vis'][...]
            transit_idx = None

    df_bls = get_baselines(antenna_order,autocorrs=True,casa_order=True)
    blen   = np.array([df_bls['x_m'],df_bls['y_m'],df_bls['z_m']]).T
    bname  = np.array([bn.split('-') for bn in df_bls['bname']])
    bname  = bname.astype(int)
    
    if not autocorrs:
        cross_bls = list(range((nant*(nant+1))//2))
        i=-1
        for j in range(1,nant+1):
            i += j
            cross_bls.remove(i)  
        
        vis = vis[:,cross_bls,...]
        blen = blen[cross_bls,...]
        bname = bname[cross_bls,...]
       
        assert vis.shape[0] == len(mjd)
        assert vis.shape[1] == len(cross_bls)
    
    if badants is not None:
        good_idx = list(range(len(bname)))
        for i,bn in enumerate(bname):
            if (bn[0] in badants) or (bn[1] in badants):
                good_idx.remove(i)
        vis = vis[:,good_idx,...]
        blen = blen[good_idx,...]
        bname = bname[good_idx,...]

        for badant in badants:
            antenna_order.remove(badant)
           
    assert vis.shape[0] == len(mjd)
    vis = vis.swapaxes(0,1)
    dt = np.median(np.diff(mjd))
    if len(mjd)>0:
        tstart = mjd[0]-dt/2
        tstop = mjd[-1]+dt/2
    else:
        tstart = None
        tstop = None
    
    bname = bname.tolist()
        
    return fobs, blen, bname, tstart, tstop, vis, mjd, transit_idx, antenna_order, tsamp

def read_psrfits_file(fl,source,dur=50*u.min,antenna_order=None,
                      antpos='./data/antpos_ITRF.txt',
                      utc_start = None,
                     autocorrs=False,badants=None,quiet=True,
                     dsa10=True):
    """Reads in the psrfits header and data
    
    Args:
        fl: str
          the full path to the psrfits file
        source: src class 
          the source to retrieve data for
        dur: astropy quantity in minutes or equivalent
          amount of time to extract
        antenna_order: list
          the order of the antennas in the correlator.  only used
          if dsa10 is False
        antpos: str
          the full path of the text file containing the 
          antenna positions
        utc_start: astropy time object
          the start time of the observation in UTC.  only used if dsa10 is False
        autocorrs: boolean
          if True, auto + cross correlations will be returned.
          if False, only the cross correlations will be returned.
        badants: None or list(int)
          The antennas for which you do not want data to be returned
        quiet: Boolean
          If True, additional info on the file printed 
        dsa10: Boolean
          If True, fits file is in dsa10 format, otherwise in the 3-antenna
          correlator output format
             
    Returns:
        fobs: arr(float)
          the frequency of the channels in GHz
        blen: arr(float) 
          the itrf coordinates of the baselines, shape (nbaselines, 3)
        bname: list(int)
          the station pairs for each baseline (in the same order as blen)
          shape (nbaselines, 2)
        tstart: float
          the start time in MJD
        tstop: float
          the stop time in MJD
        vis: complex array
          the requested visibilities, dimensions (baselines, time, frequency)
        mjd: arr(float)
          the midpoint mjd of each integration in the visibilities
        transit_idx: int
          the index of the transit in the time axis of the visibilities
    """
    # Open the file and get info from the header and visibilities
    fo = pf.open(fl,ignore_missing_end=True)
    f  = fo[1]
    if dsa10:
        nchan, fobs, nt, blen, bname, tstart, tstop, tsamp, antenna_order = \
            get_header_info(f,verbose=True,
            antpos=antpos)
        vis,lst,mjd,transit_idx = extract_vis_from_psrfits(f,
                        source.ra.to_value(u.rad),
                         (dur/2*(15*u.deg/u.h)).to_value(u.rad),
                        antenna_order,tstart,tstop,quiet)
    else:
        assert antenna_order is not None, 'Antenna order must be provided'
        assert utc_start is not None, 'Start time must be provided'
        nchan, fobs, nt, blen, bname, tstart_offset, tstop_offset, tsamp,\
            antenna_order = \
            get_header_info(f,verbose=True,
            antpos=antpos,antenna_order=antenna_order,dsa10=False)
        tstart = (utc_start + tstart_offset*u.s).mjd
        tstop = (utc_start + tstop_offset*u.s).mjd
        vis,lst,mjd,transit_idx = extract_vis_from_psrfits(f,
                    source.ra.to_value(u.rad),
                    (dur/2*(15*u.deg/u.h)).to_value(u.rad),
                    antenna_order,tstart,tstop,quiet)
    fo.close()
    
    # Now we have to extract the correct baselines
    nant = len(antenna_order)
    if not autocorrs:
        basels = list(range((nant*(nant+1))//2))
        auto_bls = get_autobl_indices(nant)
        if not dsa10:        
            auto_bls = [(len(basels)-1)-auto_bl for auto_bl in auto_bls]
        for i in auto_bls:
            basels.remove(i)
        vis = vis[basels,...]
        blen = blen[basels,...]
        bname = [bname[i] for i in basels]
    
    # Reorder the visibilities to fit with CASA ms convention
    if dsa10:
        vis = vis[::-1,...]
        bname = bname[::-1]
        blen = blen[::-1,...]
        antenna_order=antenna_order[::-1]

    if badants is not None:
        blen = np.array(blen)
        good_idx = list(range(len(bname)))
        for i,bn in enumerate(bname):
            if (bn[0] in badants) or (bn[1] in badants):
                good_idx.remove(i)
        vis = vis[good_idx,...]
        blen = blen[good_idx,...]
        bname = [bname[i] for i in good_idx]

        
    if badants is not None:
        for badant in badants:
            antenna_order.remove(badant)
            
    dt = np.median(np.diff(mjd))
    if len(mjd)>0:
        tstart = mjd[0]-dt/2
        tstop = mjd[-1]+dt/2
    else:
        tstart = None
        tstop = None
    
    if type(bname) is not list:
        bname = bname.tolist()
    return fobs, blen, bname, tstart, tstop, tsamp, vis, mjd, lst, transit_idx, antenna_order

def get_header_info(f,antpos='./data/antpos_ITRF.txt',verbose=False,
                   antenna_order=None,dsa10=True):
    """ Returns important header info from a visibility fits file.
    
    Args:
        f: pyfits table handler 
          the visibility data table
        antpos: str
          the path to the text file containing the antenna positions
        verbose: boolean
          whether to print information on the pyfits file
        antenna_order: list
          the order of the antennas in the correlator.  Required if 
          dsa10 is False
        dsa10: Boolean
          if True, dsa10 fits format, otherwise 3-antenna correlator
          fits format

    Returns:
        nchan: int
          the number of frequency channels
        fobs: arr(float)
          the midpoint frequency of each channel in GHz
        nt: int
          the number of time samples
        blen: arr(float)
          the itrf coordinates of the baselines, shape (nbaselines, 3)
        bname: list(int)
          the station pairs for each baseline (in the same order as blen), 
          shape (nbaselines, 2), numbering starts at 1
        tstart: float
          if dsa10: the start time in MJD
          else: the start time in seconds past the utc start time
        tstop: float
          if dsa10: the stop time in MJD
          else: the start time in seconds past the utc start time
    """
    if dsa10:
        aname = f.header['ANTENNAS'].split('-')
        aname = [int(an) for an in aname]
    else:
        assert antenna_order is not None, 'Antenna order must be provided'
        aname = antenna_order

    nchan = f.header['NCHAN']
    if dsa10:
        fobs  = ((f.header['FCH1']*1e6-(np.arange(nchan)+0.5)*2.*2.5e8/nchan)*u.Hz
            ).to_value(u.GHz)
    else:
        fobs = ((f.header['FCH1']*1e6-
                (np.arange(nchan)+0.5)*2.5e8/8192)*
                u.Hz).to_value(u.GHz)
    nt    = f.header['NAXIS2']
    tsamp = f.header['TSAMP']

    tp    = np.loadtxt(antpos)
    blen  = []
    bname = []
#     for i in np.arange(len(aname)-1)+1:
#         for j in np.arange(i+1):
    if dsa10:
        # currently doesn't include autocorrelations
        for i in np.arange(10):
            for j in np.arange(i+1):
                a1 = int(aname[i])-1
                a2 = int(aname[j])-1
                bname.append([a1+1,a2+1])
                blen.append(tp[a1,1:]-tp[a2,1:])
    else:
        for j in range(len(aname)):
            for i in range(j,len(aname)):
                a1 = int(aname[i])-1
                a2 = int(aname[j])-1
                bname.append([a2+1,a1+1])
                blen.append(tp[a2,1:]-tp[a1,1:])
    blen  = np.array(blen)

    if dsa10:
        tstart = f.header['MJD'] + ct.time_offset/ct.seconds_per_day
        tstop  = tstart + nt*tsamp/ct.seconds_per_day
    else:
        tstart = tsamp*f.header['NBLOCKS']
        tstop = tstart + nt*tsamp

    if verbose:
        if dsa10:
            print('File covers {0:.2f} hours from MJD {1} to {2}'.format(
            ((tstop-tstart)*u.d).to(u.h),tstart,tstop))
        else:
            print('File covers {0:.2f} h from {1} s to {2} s'.format(
            ((tstop-tstart)*u.s).to(u.h),tstart,tstop))
    return nchan, fobs, nt, blen, bname, tstart, tstop, tsamp, aname

def extract_vis_from_psrfits(f,stmid,seg_len,antenna_order,
                             mjd0,mjd1,
                             quiet=True):
    """ Routine to extract visibilities from a fits file output
    by the DSA-10 system.  
    
    Based on clip.extract_segment from DSA-10 routines

    Args:
        f: pyfits table handler
          the fits Visibility table
        stmid: float
          the LST around which to extract visibilities, in radians
        seg_len: float
          the duration of visibilities to extract, in radians
        antenna_order: list
          the order of the antennas in the correlator
        mjd0: float
          the start time of the file in mjd
        mjd1: float
          the stop time of the file in mjd
        quiet: boolean
          if False, information on the file will be printed

    Returns:
        odata: complex array
          the requested visibilities, dimensions (baselines, time, frequency)
        st: arr(float)
          the lst of each integration in the visibilities, in radians
        mjd: arr(float)
          the midpoint mjd of each integration in the visibilities
        I0-I1: int
          the index of the transit
    """
    dat   = f.data['VIS']
    nt    = f.header['NAXIS2']
    nchan = f.header['NCHAN']
    tsamp = f.header['TSAMP']
    nant = len(antenna_order)
    
    #mjd0 = f.header['MJD'] + ct.time_offset/ct.seconds_per_day 
    #mjd1 = mjd0 + ct.tsamp*nt/ct.seconds_per_day  
    if (mjd1-mjd0)>=1:
        print("Data covers > 1 sidereal day. Only the first segment "+
              "will be extracted")
    
    st0 = Time(mjd0, format='mjd').sidereal_time(
        'apparent',longitude=ct.ovro_lon*u.rad).radian 
    mjd = mjd0 + (np.arange(nt)+0.5) * tsamp / ct.seconds_per_day
    
    st  = np.angle(np.exp(1j*(st0 + 2*np.pi/ct.seconds_per_sidereal_day*
                          np.arange(nt+0.5)*tsamp)))
    
    st1 = Time(mjd1,format='mjd').sidereal_time(
            'apparent',longitude=ct.ovro_lon).radian

    if not quiet:
        print("\n-------------EXTRACT DATA--------------------")
        print("Extracting data around {0}".format(stmid*180/np.pi))
        print("{0} Time samples in data".format(nt))
        print("LST range: {0:.1f} --- ({1:.1f}-{2:.1f}) --- {3:.1f}deg".format(st[0]*180./np.pi,(stmid-seg_len)*180./np.pi, (stmid+seg_len)*180./np.pi,st[-1]*180./np.pi))

    I1 = np.argmax(np.absolute(np.exp(1j*st)+
                           np.exp(1j*stmid)*np.exp(-1j*seg_len)))
    I2 = np.argmax(np.absolute(np.exp(1j*st)+
                           np.exp(1j*stmid)*np.exp(1j*seg_len)))
    I0 = np.argmax(np.absolute(np.exp(1j*st)+
                           np.exp(1j*(stmid))))
    
    mjd = mjd[I1:I2]
    st  = st[I1:I2]
    dat = dat.reshape((nt,(nant*(nant+1))//2,nchan,2,2))[I1:I2,:,:,:,:]

    if not quiet:
        print("Extract: {0} ----> {1} sample; transit at {2}".format(I1,I2,I0))
        print("----------------------------------------------")
    
    # Fancy indexing can have downfalls and may change in future numpy versions
    # See issue here https://github.com/numpy/numpy/issues/9450
    # odata = dat[:,basels,:,:,0]+ 1j*dat[:,basels,:,:,1]
    odata = dat[...,0]+1j*dat[...,1]
    odata = odata.swapaxes(0,1)
                  
    return odata,st,mjd,I0-I1

def get_autobl_indices(nant):
    auto_bls = []
    i=-1
    for j in range(1,nant+1):
        i+=j
        auto_bls += [i]
    return auto_bls

def get_antpos_itrf(antpos='data/antpos_ITRF.txt'):
    # Read in baseline info and order it as needed
    if antpos[-4:]=='.txt':
        anum,xx,yy,zz = np.loadtxt(antpos).transpose()
        anum = anum.astype(int)+1
        anum,xx,yy,zz = zip(*sorted(zip(anum,xx,yy,zz)))
    elif antpos[-4:]=='.csv':
        df = get_itrf(antpos)
        anum = np.array(df.index)
        xx = np.array(df[['dx_m']])
        yy = np.array(df[['dy_m']])
        zz = np.array(df[['dz_m']])
    return anum,xx,yy,zz

def simulate_ms(ofile,tname,anum,xx,yy,zz,diam,mount,
               pos_obs,spwname,freq,deltafreq,freqresolution,
               nchannels,integrationtime,obstm,dt,src,
               stoptime):
    me = cc.measures()
    qa = cc.quanta()
    sm = cc.simulator()
    sm.open(ofile)
    sm.setconfig(telescopename=tname, x=xx, y=yy, z=zz, 
                 dishdiameter=diam, mount=mount, antname=anum, 
                 coordsystem='global', referencelocation=pos_obs)
    sm.setspwindow(spwname=spwname, freq=freq, deltafreq=deltafreq, 
                   freqresolution=freqresolution, 
                   nchannels=nchannels, stokes='XX YY')
    sm.settimes(integrationtime=integrationtime, usehourangle=False, 
                referencetime=me.epoch('utc', qa.quantity(obstm-dt,'d')))
    sm.setfield(sourcename=src.name, 
                sourcedirection=me.direction(src.epoch, 
                                             qa.quantity(src.ra.to_value(u.rad),'rad'), 
                                             qa.quantity(src.dec.to_value(u.rad),'rad')))
    sm.setauto(autocorrwt=0.0)
    print('simulation parameters set.')
    print('observing')
    sm.observe(src.name, spwname, starttime='0s', stoptime=stoptime)
    print('observation done')
    sm.close()

def convert_to_ms(source, vis, obstm, ofile, bname, antenna_order,
                  tsamp = ct.tsamp*ct.nint, nint=1,
                  antpos='data/antpos_ITRF.txt',model=None,
                 dt = ct.casa_time_offset,dsa10=True):
    """ Writes visibilities to an ms. 
    
    Uses the casa simulator tool to write the metadata to an ms,
    then uses the casa ms tool to replace the visibilities with 
    the observed data.
    
    Args:
        source: src instance 
          parameters for the source (or position) used for fringe-stopping
        vis: complex array
          the visibilities, shape (baseline,time, channel)
        obstm: float
          start time of observation in MJD 
        ofile: str
          name for the created ms.  will write to <ofile>.ms
        bname: list(int)
          the list of baselines names in the form [[ant1, ant2],...]
        antenna_order: list()
        tsamp: float
          the sampling time of the input visibilities in seconds
        nint: int
          the number of time bins to integrate before saving to a 
          measurement set, default 25
        antpos: str
          the full path to the text file containing itrf antenna
          positions
        model: complex array
          same shape as visibilities, the model to write to the 
          measurement set (and against which gain calibration will be done).
          If not provided an array of ones will be used at the model.

    Returns:
    """

    nant = len(antenna_order)
    
    me = cc.measures()
    qa = cc.quanta()

    # Observatory parameters 
    tname   = 'OVRO_MMA'
    diam    = 4.5 # m
    obs     = 'OVRO_MMA'
    mount   = 'alt-az'
    pos_obs = me.observatory(obs)
    
    # Backend
    if dsa10:
        spwname   = 'L_BAND'
        freq      = '1.4871533196875GHz'
        deltafreq = '-0.244140625MHz'
        freqresolution = deltafreq
    else:
        spwname = 'L_BAND'
        freq      = '1.28GHz'
        deltafreq = '40.6901041666667kHz'
        freqresolution = deltafreq
    nchannels = vis.shape[-2]
    npol      = vis.shape[-1]
    
    # Rebin visibilities 
    integrationtime = '{0}s'.format(tsamp*nint) 
    if nint != 1:
        npad = nint - vis.shape[1]%nint
        if npad == nint: npad = 0 
        vis = np.nanmean(np.pad(vis,((0,0),(0,npad),
                    (0,0),(0,0)),
                            mode='constant',constant_values=
                    (np.nan,)).reshape(vis.shape[0],-1,nint,
                    vis.shape[2],vis.shape[3]),axis=2)
        if model is not None:
            model = np.nanmean(np.pad(model,((0,0),(0,npad),
                    (0,0),(0,0)),
                    mode='constant',constant_values=
                    (np.nan,)).reshape(model.shape[0],-1,nint,
                    model.shape[2],model.shape[3]),axis=2)
    stoptime  = '{0}s'.format(vis.shape[1]*tsamp*nint)
    
    anum,xx,yy,zz = get_antpos_itrf(antpos)
    
    # Sort the antenna positions
    idx_order = sorted([int(a)-1 for a in antenna_order])
    anum = np.array(anum)[idx_order]
    xx = np.array(xx)
    yy = np.array(yy)
    zz = np.array(zz)
    xx = xx[idx_order]
    yy = yy[idx_order]
    zz = zz[idx_order]
    
    nints = np.zeros(nant,dtype=int)
    for i in range(nant):
        nints[i] = np.sum(np.array(bname)[:,0]==anum[i])
    nints, anum, xx, yy, zz = zip(*sorted(zip(nints,anum,xx,yy,zz),reverse=True))

    # Check that the visibilities are ordered correctly by 
    # checking the order of baselines in bname
    idx_order = []
    autocorr = True if [anum[0],anum[0]] in list(bname) else False

    for i in range(len(anum)):
        for j in range(i if autocorr else i+1,len(anum)):
            idx_order += [bname.index([anum[i],anum[j]])]
    assert idx_order == list(np.arange(len(bname),dtype=int)), \
        'Visibilities not ordered by baseline'
    anum = [str(a) for a in anum]
    
    print('beginning simulation')
    simulate_ms('{0}.ms'.format(ofile),tname,anum,xx,yy,zz,diam,mount,
               pos_obs,spwname,freq,deltafreq,freqresolution,
               nchannels,integrationtime,obstm,dt,source,
               stoptime)
    
    print('simulation done')
    print('checking time')

    # Check that the time is correct
    ms = cc.ms()
    ms.open('{0}.ms'.format(ofile))
    tstart_ms  = ms.summary()['BeginTime']
    ms.close()
    
    if np.abs(tstart_ms - obstm) > 1e-10:
        dt = dt + (tstart_ms - obstm)
        print('Updating casa time offset to {0}s'.format(dt*ct.seconds_per_day))
        print('Rerunning simulator')
        simulate_ms('{0}.ms'.format(ofile),tname,anum,xx,yy,zz,diam,mount,
               pos_obs,spwname,freq,deltafreq,freqresolution,
               nchannels,integrationtime,obstm,dt,source,
               stoptime)
    
    print('time correction done')
    print('modifying visibilities')
    # Reopen the measurement set and write the observed visibilities
    ms = cc.ms()
    ms.open('{0}.ms'.format(ofile),nomodify=False)
    ms.selectinit(datadescid=0)
    
    rec = ms.getdata(["data"]) 
    # rec['data'] has shape [scan, channel, [time*baseline]]
    vis = vis.T.reshape((npol,nchannels,-1))
    rec['data'] = vis
    ms.putdata(rec)
    ms.close()
    
    ms = cc.ms()
    ms.open('{0}.ms'.format(ofile),nomodify=False)
    if model is None:
        model = np.ones(vis.shape,dtype=complex)
    else:
        model = model.T.reshape((npol,nchannels,-1))
    rec = ms.getdata(["model_data"])
    rec['model_data'] = model
    ms.putdata(rec)
    ms.close()
    print('visibilities modified')
    
    print('confirming time correction')
    # Check that the time is correct
    ms = cc.ms()
    ms.open('{0}.ms'.format(ofile))
    tstart_ms  = ms.summary()['BeginTime']
    tstart_ms2 = ms.getdata('TIME')['time'][0]/ct.seconds_per_day
    ms.close()

    assert np.abs(tstart_ms - (tstart_ms2-tsamp*nint/ct.seconds_per_day/2)) < 1e-10, \
        'Data start time does not agree with MS start time'
    
    assert np.abs(tstart_ms - obstm) < 1e-10 , \
        'Measurement set start time does not agree with input tstart'
    print('done ms conversion')
    return

def extract_vis_from_ms(ms_name,nbls,nchan,npol=2):
    """ Extract calibrated and uncalibrated visibilities from 
    measurement set.
    
    Args:
        ms_name: str
          the name of the measurement set (will open <ms>.ms)
    
    Returns:
        vis_uncal: complex array
          the 'observed' data from the measurement set (baselines,time,freq)
        vis_cal: complex array
          the 'corrected' data from the measurement set (baselines,time,freq)
    """
    error = 0
    ms = cc.ms()
    error += not ms.open('{0}.ms'.format(ms_name))
    vis_uncal= (ms.getdata(["data"])
                ['data'].reshape(npol,nchan,-1,nbls).T)
    vis_cal  = (ms.getdata(["corrected_data"])
            ['corrected_data'].reshape(npol,nchan,-1,nbls).T)
    error += not ms.close()
    if error > 0:
        logger.info('{0} errors occured during calibration'.format(error))
    return vis_uncal, vis_cal

def initialize_hdf5_file(f,fobs,antenna_order,t0,nbls,nchan,npol,nant):
    """Initialize the hdf5 file.
    
    Args:
        f: hdf5 file handler
            the file
        fobs: array(float)
            the center frequency of each channel
        antenna_order: array(int)
            the order of the antennas in the correlator
        t0: float
            the time of the first sample in mjd seconds
        nbls: int
            the number of baselines
        nchan: int
            the number of channels
        npol: int
            the number of polarizations
        nant: int
            the number of antennas
        
    Returns:
        vis_ds: hdf5 dataset
            the dataset for the visibilities
        t_ds: hdf5 dataset
            the dataset for the times
    """
    ds_fobs = f.create_dataset("fobs_GHz",(nchan,),dtype=np.float32,data=fobs)
    ds_ants = f.create_dataset("antenna_order",(nant,),dtype=np.int,data=antenna_order)
    t_st = f.create_dataset("tstart_mjd_seconds",
                           (1,),maxshape=(1,),
                           dtype=int,data=t0)

    vis_ds = f.create_dataset("vis", 
                            (0,nbls,nchan,npol), 
                            maxshape=(None,nbls,nchan,npol),
                            dtype=np.complex64,chunks=True,
                            data = None)
    t_ds = f.create_dataset("time_seconds",
                           (0,),maxshape=(None,),
                           dtype=np.float32,chunks=True,
                           data = None)
    return vis_ds, t_ds

def mask_bad_bins(vis,axis,thresh=6.0,medfilt=False,nmed=129):
    """Mask bad channels or time bins in visibility data
    
    Args:
      vis: array(complex)
        the visibility array 
        dimensions (nbls,nt,nf,npol)
      axis: int
        the axis to flag in
        axis=1 will flag bad time bins
        axis=2 will flag bad frequency bins
      thresh: float
        the threshold above which to flag data 
        anything that deviates from the median by more than
        thresh x the standard deviation is flagged
      medfilt: Boolean
        whether to median filter to remove an average trend
        if True, will median filter
        if False, subtract the median for the baseline/pol pair
      nmed: int
        the size of the median filter to use
        only used in medfilt is True
        must be odd
      
    Returns:
      good_bins: array(boolean)
        if axis==2: (nbls,1,nf,npol) 
        if axis==1: (nbls,nt,1,npol)
        returns 1 where the bin is good, 
        0 where the bin should be flagged
      fraction_flagged: array(float), dims (nbls,npol)
        the amount of data binned for each baseline/pol
    """
    assert axis==1 or axis==2 
    avg_axis = 1 if axis==2 else 2

    # average over time first
    vis_avg = np.abs(np.mean(vis,axis=avg_axis))
    # median filter over frequency and remove the median trend
    # or remove the median
    if medfilt:
        vis_avg_mf = median_filter(vis_avg.real,size=(1,nmed,1))
        vis_avg -= vis_avg_mf
    else:
        vis_avg -= np.median(vis_avg,axis=1)
    # calculate the standard deviation along the frequency axis
    vis_std = np.std(vis_avg,axis=1,keepdims=True)
    # get good channels
    good_bins = np.abs(vis_avg)<thresh*vis_std
    fraction_flagged = (1-good_bins.sum(axis=1)/good_bins.shape[1])
    if avg_axis==1:
        good_bins = good_bins[:,np.newaxis,:,:] 
    else:
        good_bins = good_bins[:,:,np.newaxis,:]
    return good_bins,fraction_flagged

# def mask_bad_times(vis,thresh=6.0,medfilt=False):
#     """Mask bad channels in visibility data
#     Args:
#       vis: array(complex)
#         the visibility array 
#         dimensions (nbls,nt,nf,npol)
#       thresh: float
#         the threshold above which to flag data 
#         anything that deviates from the median by more than
#         thresh x the standard deviation is flagged
      
#     Returns:
#       good_times: array(boolean)
#         (nbls,nt,1,npol)
#         returns 1 where the timebin is good, 
#         0 where the timebin should be flagged
#     """
#     #vis2 = np.copy(vis)
#     tbin_mean = vis.mean(axis=2,keepdims=True)
#     total_mean = tbin_mean.mean(axis=1,keepdims=True)
#     total_std = tbin_mean.std(axis=1,keepdims=True)
#     #fstd = np.std(vis2,axis=1)
#     #rmean = medfilt(fstd,(1,nsamples,1))
#     #good_channels = (fstd-rmean)<cutoff
#     good_times = np.abs(tbin_mean-total_mean)<thresh*total_std
#     return good_times


def mask_bad_pixels(vis,thresh=6.0,mask=None):
    """Masks bad pixels that are more than thresh*std above the
    median in each visibility.
    
    Args:
      vis: array(complex)
        dims (nbls,nt, nf, npol)
        the visiblities to flag
      thresh: float
        the threshold in stds above which to flag data
      mask: array(bool)
        optional, dims (nbls,nt,nf,npol)
        a mask for data already flagged
      
    Returns:
      good_pixels: array(bool)  
        (nbls,nt,nf,npol)
      fraction_flagged: array(float)
        (nbls,npol)
      
    """
    (nbls,nt,nf,npol) = vis.shape
    vis = np.abs(vis.reshape(nbls,-1,npol))
    vis = vis-np.median(vis,axis=1,keepdims=True)
    if mask is not None:
        vis = vis*mask.reshape(nbls,-1,npol)
    std = np.std(vis,axis=1,keepdims=True)
    good_pixels = np.abs(vis)<thresh*std
    fraction_flagged = 1 - good_pixels.sum(1)/good_pixels.shape[1]
    good_pixels = good_pixels.reshape(nbls,nt,nf,npol)
    return good_pixels,fraction_flagged

# def mask_bad_pixels(vis,ntbin=100,thresh=6.0):
#     """Masks pixels thresh*std above the median in each visibility after 
#     binning in time.
    
#     Args:
#       vis: array(complex)
#         (nbls, nt, nf, npol)
#         the complex visibilities
#       ntbin: int
#         the number of time bins to average by 
#       thresh: float
#         the number of stddevs above which to flag data
#     Returns:
#       mask: array(boolean)
#         same dimensions as vis
#         1 for pixels which are good, 0 for pixels which should be flagged
#       fraction_flagged: array(float), dims (nbls,npol)
#         the fraction of data flagged on each baseline
#       """
#     (nbls,nt,nchan,npol)=vis.shape
#     bindata = np.copy(vis)
#     bindata = bindata[:,:nt//ntbin*ntbin,...].reshape(nbls,nt//ntbin,ntbin,nchan,npol).mean(axis=2)
#     bindata = 10*(np.log10(np.abs(bindata)))
#     std = np.std(bindata,axis=2,keepdims=True)
#     med = np.median(bindata,axis=2,keepdims=True)
#     good_pixels = np.abs(bindata-med)<thresh*std
#     mask = np.tile(good_pixels[:,:,np.newaxis,:,:],
#                   (1,1,ntbin,1,1)).reshape(nbls,-1,nchan,npol)
#     if mask.shape[1] < nt:
#         mask = np.append(mask,np.zeros((nbls,
#                                        nt-mask.shape[1],nchan,npol)),
#                  axis=1)
#     print('{0} % of data flagged.'.format(
#         100*(1-np.sum(mask)/(mask.flatten().shape[0]))))
#     fraction_flagged = (1-mask.sum(1).sum(1)/(
#         mask.reshape(mask.shape[0],-1,mask.shape[-1]).shape[1]))
#     return mask,fraction_flagged


def read_caltable(tablename,nbls,cparam=False):
    """
    Read a casa calibration table and return 
    the time (or None if not in the table) and the value of the 
    calibration parameter.
    
    Args:
      tablename: str
        the full path to the casa table
      nbls: int
        the number of baselines in the casa table 
        calculate from the number of antennas, nant:
          for delay calibration, nbls=nant
          for antenna-based gain/bandpass calibration, nbls=nant 
          for baseline-based gain/bandpass calibration, nbls=(nant*(nant+1))//2
      cparam: boolean
        whether the parameter of interest is complex (cparam=True) or
        a float (cparam=False)
          for delay calibration, cparam=False
          for gain/bandpass calibratino, cparam=True
          
    Returns:
      time: array(float) or None
        shape (nt,nbls)
        the times at which each solution is calculated, mjd
        None if no time given in the table
      val: array(float) or array(complex)
        shape may be (npol,nt,nbls) or (npol,1,nbls) (for delay or gain cal)
          or (npol,nf,nbls) where nf is frequency (for bandpass cal)
        the values of the calibration parameter
    """
    param_type = 'CPARAM' if cparam else 'FPARAM'
    
    tb = cc.table()
    logger.info('Opening table {0} as type {1}'.format(tablename, 
                                                    param_type))
    tb.open(tablename)
    if 'TIME' in tb.colnames():
        time = (tb.getcol('TIME').reshape(-1,nbls)*u.s).to_value(u.d)
    else:
        time = None
    vals = tb.getcol(param_type)
    vals = vals.reshape(vals.shape[0],-1,nbls)
    tb.close()
    
    return time,vals

def caltable_to_etcd(msname,calname,antenna_order,
                    baseline_cal=False,pols=['A','B']):
    """ Copy calibration values from table to etcd
    
    Not working yet.
    """
    nant = len(antenna_order)
    
    # Number of baselines included in the gain and bandpass calibrations
    if baseline_cal:
        nbls = (nant*(nant+1))//2
    else:
        nbls = nant
    
    # Complex gains for each antenna
    tamp,amps = read_caltable('{0}_{1}_gcal_ant'.format(msname,calname),
                             nbls,cparam=True)
    if baseline_cal:
        raise NotImplementedError
        # get the correct indices for the autocorrelations
    
    # Check the output shapes
    assert tamp.shape[0]==amps.shape[1]
    assert tamp.shape[1]==amps.shape[2]
    assert tamp.shape[1]==len(antenna_order)
    assert amps.shape[0]==len(pols)
    
    # Reduce tamp to a single value, amps to a single value for each ant/pol
    assert np.all(np.equal.reduce(tamp)==np.ones(len(antenna_order)))
    tamp = tamp[:,0]
    if tamp.shape[0]>1:
        tamp = np.median(tamp)
    if amps.ndim == 3 : 
        amps = np.median(amps,axis=1)
    
    # Delays for each antenna
    tdel,delays = read_caltable('{0}_{1}_kcal'.format(msname,calname),
                               nant,cparam=False)
    assert tdel.shape[0]==delays.shape[1]
    assert tdel.shape[1]==delays.shape[2]
    assert tdel.shape[1]==len(antenna_order)
    assert delays.shape[0]==len(pols)
    
    # Reduce tdel to a single value, delays to a single value for each ant/pol
    assert np.all(np.equal.reduce(tdel)==np.ones(len(antenna_order)))
    tdel = tdel[:,0]
    if tdel.shape[0]>1:
        tdel = np.median(tdel)
    if delays.ndim == 3: 
        delays = np.median(delays,axis=1)
    
    for i, antnum in enumerate(antenna_order):
        for j,pol in enumerate(pols):
            gainamp = np.abs(amps[j,i])
            gainphase = np.angle(amps[j,i])
            delay = delays[j,i]
            dd = {'gainamp': gainamp, 'gainphase': gainphase, 'delay': delay, 'calsource': calname, 'gaincaltime': tamp, 'delaycaltime': tdel}
            de.put_dict('/mon/calibration/{0}{1}'.format(antnum, pol.lower()), dd)

        
