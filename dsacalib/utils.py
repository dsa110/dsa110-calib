"""
DSA_UTILS.PY

Dana Simard, dana.simard@astro.caltech.edu, 10/2019

Modified for python3 from DSA-10 routines written by 
Vikram Ravi, Harish Vendantham

Routines to interact w/ fits visibilities recorded by DSA-10
"""

import __casac__ as cc
import astropy.io.fits as pf
import astropy.units as u
import numpy as np
from astropy.time import Time
from . import constants as ct
from antpos.utils import *
import h5py

from astropy.utils import iers
iers.conf.iers_auto_url_mirror = ct.iers_table

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
           
    Returns:
    """
    def __init__(self,name,ra,dec,I=1.,epoch='J2000'):
        self.name = name
        self.I = I
        self.ra = to_deg(ra)
        self.dec = to_deg(dec)
        self.epoch = epoch

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

def read_hdf5_file(fl,source=None,dur=50*u.min,autocorrs=False,
                   badants=None,quiet=True):
    if source is not None:
        stmid = source.ra.to_value(u.rad)
        seg_len = (dur/2*(15*u.deg/u.h)).to_value(u.rad)
        
    with h5py.File(fl, 'r') as f:
        antenna_order = f['antenna_order'][...]
        nant = len(antenna_order)
        fobs = f['fobs_GHz'][...]
        mjd = f['time_mjd_seconds'][...]/ct.seconds_per_day
        nt = len(f['time_mjd_seconds'])
        tsamp  = (mjd[0] - mjd[-1])/(nt-1)*ct.seconds_per_day


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
            dat = f['vis'][:,I1:I2,...]
            if not quiet:
                print("Extract: {0} ----> {1} sample; transit at {2}".format(I1,I2,I0))
                print("----------------------------------------------")

        else:
            dat = f['vis'][...]
            transit_idx = None

    # Now we have to extract the correct baselines
    auto_bls = []
    cross_bls = list(range(int(nant/2*(nant+1))))
    i=-1
    for j in range(1,nant+1):
        i += j
        auto_bls += [i]
        cross_bls.remove(i)

    basels = auto_bls if autocorrs else cross_bls
    
    # Fancy indexing can have downfalls and may change in future numpy versions
    # See issue here https://github.com/numpy/numpy/issues/9450
    vis = dat[basels,...]

    
    if autocorrs:
        bname = np.array([[a,a] for a in antenna_order])
        blen  = np.zeros((len(antenna_order),3))
        if badants is not None:
            badants = [str(ba) for ba in badants]
            good_idx = list(range(len(antenna_order)))
            for badant in badants:
                good_idx.remove(antenna_order.index(badant))
            vis = vis[good_idx,...]
            bname = bname[good_idx,...]
            blen = blen[good_idx,...]
    
    if not autocorrs:
        df_bls = get_baselines(antenna_order,autocorrs=False,casa_order=True)
        blen   = np.array([df_bls['x_m'],df_bls['y_m'],df_bls['z_m']]).T
        bname  = [bn.split('-') for bn in df_bls['bname']]
        if badants is not None:
            bname = np.array(bname)
            blen = np.array(blen)
            good_idx = list(range(len(bname)))
            for i,bn in enumerate(bname):
                if (bn[0] in badants) or (bn[1] in badants):
                    good_idx.remove(i)
            vis = vis[good_idx,...]
            blen = blen[good_idx,...]
            bname = bname[good_idx,...]
        
    if badants is not None:
        badants = [str(ba) for ba in badants]
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
        
    return fobs, blen, bname, tstart, tstop, vis, mjd, transit_idx, antenna_order

def read_psrfits_file(fl,source,dur=50*u.min,antpos='./data/antpos_ITRF.txt',
                     autocorrs=False,badants=None,quiet=True):
    """Reads in the psrfits header and data
    
    Args:
        fl: str
          the full path to the psrfits file
        source: src class 
          the source to retrieve data for
        dur: astropy quantity in minutes or equivalent
          amount of time to extract
        antpos: str
          the full path of the text file containing the 
          antenna positions
        autocorrs: boolean
          if True, only the autocorrelations will be returned.
          if False, only the cross correlations will be returned.
        badants: None or list(int)
          The antennas for which you do not want data to be returned
             
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
    nchan, fobs, nt, blen, bname, tstart, tstop, antenna_order = \
        get_header_info(f,verbose=True,
            antpos=antpos)
    vis,lst,mjd,transit_idx = extract_vis_from_psrfits(f,source.ra.to_value(u.rad),
                         (dur/2*(15*u.deg/u.h)).to_value(u.rad),
                        antenna_order,quiet,autocorrs)
    fo.close()
    
    # Reorder the visibilities to fit with CASA ms convention
    vis = vis[::-1,...]
    bname = bname[::-1]
    blen = blen[::-1,...]
    antenna_order=antenna_order[::-1]
    

    if autocorrs:
        bname = np.array([[a,a] for a in antenna_order])
        blen  = np.zeros((len(antenna_order),3))
        if badants is not None:
            badants = [str(ba) for ba in badants]
            good_idx = list(range(len(antenna_order)))
            for badant in badants:
                good_idx.remove(antenna_order.index(badant))
            vis = vis[good_idx,...]
            bname = bname[good_idx,...]
            blen = blen[good_idx,...]
    
    if not autocorrs:
        if badants is not None:
            bname = np.array(bname)
            blen = np.array(blen)
            good_idx = list(range(len(bname)))
            for i,bn in enumerate(bname):
                if (bn[0] in badants) or (bn[1] in badants):
                    good_idx.remove(i)
            vis = vis[good_idx,...]
            blen = blen[good_idx,...]
            bname = bname[good_idx,...]
        
    if badants is not None:
        badants = [str(ba) for ba in badants]
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
    return fobs, blen, bname, tstart, tstop, vis, mjd, transit_idx, antenna_order

def get_header_info(f,antpos='./data/antpos_ITRF.txt',verbose=False):
    """ Returns important header info from a visibility fits file.
    
    Args:
        f: pyfits table handler 
          the visibility data table
        antpos: str
          the path to the text file containing the antenna positions
        verbose: boolean
          whether to print information on the pyfits file

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
          the start time in MJD
        tstop: float
          the stop time in MJD
    """

    aname = f.header['ANTENNAS'].split('-')
    nchan = f.header['NCHAN']
    fobs  = ((f.header['FCH1']*1e6-(np.arange(nchan)+0.5)*2.*2.5e8/2048.)*u.Hz
            ).to_value(u.GHz)
    nt    = f.header['NAXIS2']

    tp    = np.loadtxt(antpos)
    blen  = []
    bname = []
    for i in np.arange(9)+1:
        for j in np.arange(i):
            a1 = int(aname[i])-1
            a2 = int(aname[j])-1
            bname.append([a1+1,a2+1])
            blen.append(tp[a1,1:]-tp[a2,1:])
    blen  = np.array(blen)

    tstart = f.header['MJD'] + ct.time_offset/ct.seconds_per_day
    tstop  = tstart + nt*ct.tsamp/ct.seconds_per_day

    if verbose:
        print('File covers {0:.2f} hours from MJD {1} to {2}'.format(
            ((tstop-tstart)*u.d).to(u.h),tstart,tstop))
        
    return nchan, fobs, nt, blen, bname, tstart, tstop, aname

def extract_vis_from_psrfits(f,stmid,seg_len,antenna_order,
                             quiet=True,
                            autocorrs=False,badants=None):
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
        quiet: boolean
          if False, information on the file will be printed
        autocorrs: boolean
          if True, extracts autocorrelations, else extracts 
          crosscorrelations
        badants: list
          list of bad antenna names, (base 1, actual names not 
          indices)

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
    nant = len(antenna_order)
    
    mjd0 = f.header['MJD'] + ct.time_offset/ct.seconds_per_day 
    mjd1 = mjd0 + ct.tsamp*nt/ct.seconds_per_day  
    if (mjd1-mjd0)>=1:
        print("Data covers > 1 sidereal day. Only the first segment "+
              "will be extracted")
    
    st0 = Time(mjd0, format='mjd').sidereal_time(
        'apparent',longitude=ct.ovro_lon*u.rad).radian 
    mjd = mjd0 + (np.arange(nt)+0.5) * ct.tsamp / ct.seconds_per_day
    
    st  = np.angle(np.exp(1j*(st0 + 2*np.pi/ct.seconds_per_sidereal_day*
                          np.arange(nt+0.5)*ct.tsamp)))
    
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
    dat = dat.reshape((nt,55,nchan,2,2))[I1:I2,:,:,:,:]

    if not quiet:
        print("Extract: {0} ----> {1} sample; transit at {2}".format(I1,I2,I0))
        print("----------------------------------------------")

    # Now we have to extract the correct baselines
    auto_bls = []
    cross_bls = list(range(int(nant/2*(nant+1))))
    i=-1
    for j in range(1,nant+1):
        i += j
        auto_bls += [i]
        cross_bls.remove(i)

    basels = auto_bls if autocorrs else cross_bls
    
    # Fancy indexing can have downfalls and may change in future numpy versions
    # See issue here https://github.com/numpy/numpy/issues/9450
    odata = dat[:,basels,:,:,0]+ 1j*dat[:,basels,:,:,1]
    return odata,st,mjd,I0-I1

def simulate_ms(ofile,tname,anum,xx,yy,zz,diam,mount,
               pos_obs,spwname,freq,deltafreq,freqresolution,
               nchannels,integrationtime,obstm,dt,src,
               stoptime):
    me = cc.measures.measures()
    qa = cc.quanta.quanta()
    sm = cc.simulator.simulator()
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
    sm.observe(src.name, spwname, starttime='0s', stoptime=stoptime)
    sm.close()

def convert_to_ms(src, vis, obstm, ofile, bname, antenna_order,
                  tsamp = ct.tsamp*ct.nint, nint=1,
                  antpos='data/antpos_ITRF.txt',model=None,
                 dt = ct.casa_time_offset):
    """ Writes visibilities to an ms. 
    
    Uses the casa simulator tool to write the metadata to an ms,
    then uses the casa ms tool to replace the visibilities with 
    the observed data.
    
    Args:
        src: src instance 
          parameters for the source (or position) used for fringe-stopping
        vis: complex array
          the visibilities, shape (baseline,time, channel)
        obstm: float
          start time of observation in MJD 
        ofile: str
          name for the created ms
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
    
    me = cc.measures.measures()
    qa = cc.quanta.quanta()

    # Observatory parameters 
    tname   = 'OVRO_MMA'
    diam    = 4.5 # m
    obs     = 'OVRO_MMA'
    mount   = 'alt-az'
    pos_obs = me.observatory(obs)
    
    # Backend
    spwname   = 'L_BAND'
    freq      = '1.4871533196875GHz'
    deltafreq = '-0.244140625MHz'
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
    
    # Read in baseline info and order it as needed
    anum,xx,yy,zz = np.loadtxt(antpos).transpose()
    anum = anum.astype(int)+1
    anum,xx,yy,zz = zip(*sorted(zip(anum,xx,yy,zz)))
    idx_order = sorted([int(a)-1 for a in antenna_order])
    anum = np.array(anum)[idx_order]
    xx = np.array(xx)[idx_order]
    yy = np.array(yy)[idx_order]
    zz = np.array(zz)[idx_order]
    
    nints = np.zeros(nant,dtype=int)
    for i in range(nant):
        nints[i] = sum(np.array(bname)[:,0]==anum[i])
    nints, anum, xx, yy, zz = zip(*sorted(zip(nints,anum,xx,yy,zz),reverse=True))

    # Check that the visibilities are ordered correctly by 
    # checking the order of baselines in bname
    idx_order = []
    for i in range(len(anum)):
        for j in range(i+1,len(anum)):
            idx_order += [bname.index([anum[i],anum[j]])]
    assert idx_order == list(np.arange(len(bname),dtype=int)), \
        'Visibilities not ordered by baseline'
    anum = [str(a) for a in anum]
    
    simulate_ms(ofile,tname,anum,xx,yy,zz,diam,mount,
               pos_obs,spwname,freq,deltafreq,freqresolution,
               nchannels,integrationtime,obstm,dt,src,
               stoptime)

    # Check that the time is correct
    ms = cc.ms.ms()
    ms.open(ofile)
    tstart_ms  = ms.summary()['BeginTime']
    ms.close()
    
    if np.abs(tstart_ms - obstm) > 1e-10:
        dt = dt + (tstart_ms - obstm)
        print('Updating casa time offset to {0}s'.format(dt*ct.seconds_per_day))
        print('Rerunning simulator')
        simulate_ms(ofile,tname,anum,xx,yy,zz,diam,mount,
               pos_obs,spwname,freq,deltafreq,freqresolution,
               nchannels,integrationtime,obstm,dt,src,
               stoptime)
    
    # Reopen the measurement set and write the observed visibilities
    ms = cc.ms.ms()
    ms.open(ofile,nomodify=False)
    ms.selectinit(datadescid=0)
    
    rec = ms.getdata(["data"]) 
    # rec['data'] has shape [scan, channel, [time*baseline]]
    vis = vis.T.reshape((npol,nchannels,-1))
    rec['data'] = vis
    ms.putdata(rec)
    if model is None:
        model = np.ones(vis.shape,dtype=complex)
    else:
        model = model.T.reshape((npol,nchannels,-1))
    rec = ms.getdata(["model_data"])
    rec['model_data'] = model
    ms.putdata(rec)
    ms.close()
    
    # Check that the time is correct
    ms = cc.ms.ms()
    ms.open(ofile)
    tstart_ms  = ms.summary()['BeginTime']
    tstart_ms2 = ms.getdata('TIME')['time'][0]/ct.seconds_per_day
    ms.close()

    assert np.abs(tstart_ms - (tstart_ms2-tsamp*nint/ct.seconds_per_day/2)) < 1e-10, \
        'Data start time does not agree with MS start time'
    
    assert np.abs(tstart_ms - obstm) < 1e-10 , \
        'Measurement set start time does not agree with input tstart'
    
    return

def extract_vis_from_ms(ms_name,nant):
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
    ms = cc.ms.ms()
    error += not ms.open('{0}.ms'.format(ms_name))
    vis_uncal= (ms.getdata(["data"])
                ['data'].reshape(2,625,-1,(nant*(nant-1))//2).T)
    vis_cal  = (ms.getdata(["corrected_data"])
            ['corrected_data'].reshape(2,625,-1,(nant*(nant-1))//2).T)
    error += not ms.close()
    if error > 0:
        print('{0} errors occured during calibration'.format(error))
    return vis_uncal, vis_cal
