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
import matplotlib.pyplot as plt
from astropy.time import Time
import os
from . import constants as ct

def get_header_info(f,antpos='./data/antpos_ITRF.txt',verbose=False):
    """ Returns important header info from a visibility fits file.
    Parameters:
    -----------
    f : the pyfits table handler for the visibility data table
    antpos  : str, the location of the text file containing the 
             antenna positions
    verbose : Boolean, default False
    
    Returns:
    -----------
    nchan  : int, the number of frequency channels
    fobs   : array, float, the frequency of the channels in GHz
    nt     : int, the number of time samples
    blen   : array, float, the itrf coordinates of the baselines,
             shape (nbaselines, 3)
    bname  : list, int, the station pairs for each baseline (in the same 
             order as blen), shape (nbaselines, 2)
    tstart : float, the start time in MJD
    tstop  : float, the stop time in MJD
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
            bname.append([a1,a2])
            blen.append(tp[a1,1:]-tp[a2,1:])
    blen  = np.array(blen)

    tstart = f.header['MJD']
    tstop  = f.header['MJD'] + nt*ct.tsamp/(1*u.d).to_value(u.s)

    if verbose:
        print('File covers {0:.2f} hours from MJD {1} to {2}'.format(
            ((tstop-tstart)*u.d).to(u.h),tstart,tstop))
        
    return nchan, fobs, nt, blen, bname, tstart, tstop

def extractVis(f,stmid,seg_len,pol,quiet=True):
    """ Routine to extract visibilities from a fits file output
    by the DSA-10 system.
    Based on clip.extract_segment from DSA-10 routines
    Parameters:
    -----------
    f       : the pyfits handler for the fits Visibility table
    stmid   : float, the LST around which to extract visibilities, in rad
    seg_len : float, the duration of visibilities to extract, in rad
    pol     : str, 'A' or 'B', the polarization to extract
    quiet   : boolean, default True
    Returns:
    -----------
    odata   : complex array, the requested visibilities, dimensions
              (baselines, time, frequency)
    st      : array, the lst of each integration in the visibilities, in rad
    mjd     : array, the mjd of each integration in the visibilities
    I0-I1   : integer, the index of the transit
    """
    dat   = f.data['VIS']
    nt    = f.header['NAXIS2']
    nchan = f.header['NCHAN']
    
    mjd0 = f.header['MJD'] + ct.time_offset/ct.seconds_per_day 
    mjd1 = mjd0 + ct.tsamp*nt/ct.seconds_per_day  
    if (mjd1-mjd0)>=1:
        print("Data covers > 1 sidereal day. Only the first segment "+
              "will be extracted")
    
    st0 = Time(mjd0, format='mjd').sidereal_time(
        'apparent',longitude=ct.ovro_lon).radian 
    mjd = mjd0 + np.arange(nt) * ct.tsamp / ct.seconds_per_day
    
    st  = np.angle(np.exp(1j*(st0 + 2*np.pi/ct.seconds_per_sidereal_day*
                          np.arange(nt)*ct.tsamp)))
    
    st1 = Time(mjd1,format='mjd').sidereal_time(
            'apparent',longitude=ct.ovro_lon).radian

    if not quiet:
        print("\n-------------EXTRACT DATA--------------------")
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

    basels = [1,3,4,6,7,8,10,11,12,13,15,16,17,18,19,21,
              22,23,24,25,26,28,29,30,31,32,33,34,36,37,
              38,39,40,41,42,43,45,46,47,48,49,50,51,52,53]
    opol = 0 if pol == 'A' else 1 
    odata = dat[:,basels,:,opol,0]+ 1j*dat[:,basels,:,opol,1]

    return odata,st,mjd,I0-I1


def plotFreqAmp(f,tmid=None,tspan=10*u.s,f1=None,f2=None):
    """
    Vikram's code to read in visibility fits files and plot amp vs freq
    Parameters:
    -----------
    f : pyfits file handler; the visibility file
    tmid : astropy time; the time to extract visibilities about
    tspan : astropy quantity, time; the duration of data to plot
    f1 : ??
    f2 : ??
    """
    nrow = int((f.header['NAXIS2']))
    mjd = Time(f.header['MJD'],format='mjd')
    nchan = f.header['NCHAN']
    fch1 = f.header['FCH1']-(nchan*2-1)*250./2048.

    if (f1 is not None) and (f2 is not None):
        if1 = np.floor((-fch1+f1)/(500./2048.)).astype('int')
        if2 = np.floor((-fch1+f2)/(500./2048.)).astype('int')
    else:
        if1 = 0
        if2 = nchan
    freqs = (np.arange(nchan)*(500./2048.)+fch1)[if1:if2]
            
    if tmid is None:
        t1 = 0
        t2 = np.floor((tspan/ct.tsamp).to(u.dimensionless_unscaled).value).astype('int')
    else:
        time = (tmid-(-7.*u.hour)).mjd
        t1 = np.floor(((time-mjd)*24.*3600.-tspan/2.)/ct.tsamp).astype('int')
        t2 = np.floor(((time-mjd)*24.*3600.+tspan/2.)/ct.tsamp).astype('int')
        
    print('Reading in data . ', end='\r')
    data = np.flip(f.data['VIS'].reshape((nrow,55,nchan,2,2)),axis=2)[t1:t2,:,if1:if2,:,:]
    print('Reading in data . . ', end='\r')
    data = data.mean(axis=0) 
    print('Reading in data . . . ', end='\r')
    amps = 5.*(np.log10(data[...,0]**2.+data[...,1]**2.))
    print('Data read            ')

    ants = f.header['ANTENNAS'].split('-')
    bases = []
    for i in range(10):
        for j in range(i+1):
            bases.append(ants[i]+'-'+ants[j])
    
    fig,ax = plt.subplots(11,5,figsize=(5*4,11*4),sharex=True,sharey=True)
    ax = ax.flatten()
    for pl in range(55):

        ax[pl].plot(freqs,amps[pl,:,0])
        ax[pl].plot(freqs,amps[pl,:,1])
        ax[pl].set_title(bases[pl])

    return (freqs,amps)

def convert_to_ms(src, vis, obstm, ofile, bname, nint=25,
                  antpos='data/antpos_ITRF.txt',model=None):
    """ Writes visibilities to an ms. 
    Uses the casa simulator tool to write the metadata to an ms,
    then uses the casa ms tool to replace the visibilities with 
    the observed data.
    
    Parameters:
    -----------
    src   : dsa_calib.src instance containing source parameters
    vis   : numpy array, the visibilities, 
            shape [baseline,time, channel]
    obstm : float, start time of observation in MJD 
    ofile : str, name for the created ms
    bname : list, the list of baselines names in the form [[ant1, ant2],...]
            where ant1 and ant2 are integers
    nint  : integer, the number of time bins to integrate before saving
            to a measurement set, default 25
    antpos: str, the full path to the text file containing itrf antenna
            positions
    model : array, same shape as visibilities, the model to write to the 
            measurement set (and against which gain calibration will be done),
            if not provided an array of ones will be used at the model
    Returns Nothing
    """
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
    nchannels = 625
    
    # Rebin visibilities 
    integrationtime = '{0}s'.format(ct.tsamp*nint) 
    if nint != 1:
        npad = nint - vis.shape[1]%nint
        if npad == nint: npad = 0 
        vis = np.nanmean(np.pad(vis,((0,0),(0,npad),
                    (0,0)),mode='constant',constant_values=
                    (np.nan,)).reshape(vis.shape[0],-1,nint,
                    vis.shape[2]),axis=2)
        if model is not None:
            model = np.nanmean(np.pad(model,((0,0),(0,npad),
                    (0,0)),mode='constant',constant_values=
                    (np.nan,)).reshape(model.shape[0],-1,nint,
                    model.shape[2]),axis=2)
        obstm += ct.tsamp*nint/2/ct.seconds_per_day
    stoptime  = '{0}s'.format(vis.shape[1]*ct.tsamp*nint)
    
    # Read in baseline info and order it as needed
    anum,xx,yy,zz = np.loadtxt(antpos).transpose()
    anum = anum.astype(int)
    anum,xx,yy,zz = zip(*sorted(zip(anum,xx,yy,zz)))
    
    nints = np.zeros(10,dtype=int)
    for i in range(10):
        nints[i] = sum(np.array(bname)[:,0]==i)
    nints, anum, xx, yy, zz = zip(*sorted(zip(nints,anum,xx,yy,zz),reverse=True))

    # Check that the visibilities are ordered correctly by 
    # checking the order of baselines in bname
    idx_order = []
    for i in range(len(anum)):
        for j in range(i+1,len(anum)):
            idx_order += [bname.index([anum[i],anum[j]])]
    assert idx_order == list(np.arange(45,dtype=int)), \
        'Visibilities not ordered by baseline'
    anum = [str(a) for a in anum]
    
    # make new ms 
    sm = cc.simulator.simulator()
    sm.open(ofile)
    sm.setconfig(telescopename=tname, x=xx, y=yy, z=zz, 
                 dishdiameter=diam, mount=mount, antname=tname, 
                 coordsystem='global', referencelocation=pos_obs)
    sm.setspwindow(spwname=spwname, freq=freq, deltafreq=deltafreq, 
                   freqresolution=freqresolution, 
                   nchannels=nchannels, stokes='I')
    sm.settimes(integrationtime=integrationtime, usehourangle=False, 
                referencetime=me.epoch('utc', qa.quantity(obstm,'d')))
    sm.setfield(sourcename=src.name, 
                sourcedirection=me.direction(src.epoch, 
                                             qa.quantity(src.ra.to_value(u.rad),'rad'), 
                                             qa.quantity(src.dec.to_value(u.rad),'rad')))
    sm.setauto(autocorrwt=0.0)
    sm.observe(src.name, spwname, starttime='0s', stoptime=stoptime)
    sm.close()

    # Reopen the measurement set and write the observed visibilities
    ms = cc.ms.ms()
    ms.open(ofile,nomodify=False)
    ms.selectinit(datadescid=0)
    rec = ms.getdata(["data"]) 
    
    # rec['data'] has shape [scan, channel, [time*baseline]]
    vis = vis.T.reshape((nchannels,-1))
    rec['data'][0,:,:] = vis
    ms.putdata(rec)
    if model is None:
        model = np.ones(vis.shape,dtype=complex)
    else:
        model = model.T.reshape((nchannels,-1))
    rec = ms.getdata(["model_data"])
    rec['model_data'][0,...] = model
    ms.putdata(rec)
    ms.close()
    
    return

def extract_vis_from_ms(ms_name):
    """ Extract calibrated and uncalibrated visibilities from 
    measurement set.
    Parameters:
    -----------
    ms_name : str, the name of the measurement set (will open <ms>.ms)
    Returns:
    -----------
    vis_uncal : array, the 'observed' data from the measurement set, 
                (baselines,time,freq)
    vis_cal   : array, the 'corrected' data from the measurement set
                (baselines,time,freq)
    """
    error = 0
    ms = cc.ms.ms()
    error += not ms.open('{0}.ms'.format(ms_name))
    vis_uncal= (ms.getdata(["data"])
                ['data'].reshape(625,-1,45).T)
    vis_cal  = (ms.getdata(["corrected_data"])
            ['corrected_data'].reshape(625,-1,45).T)
    error += not ms.close()
    if error > 0:
        print('{0} errors occured during calibration'.format(error))
    return vis_uncal, vis_cal