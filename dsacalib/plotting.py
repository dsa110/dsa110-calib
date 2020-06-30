"""Visibility and calibration solution plotting routines.

Plotting routines to visualize the visibilities and
the calibration solutions for DSA-110.

Author: Dana Simard, dana.simard@astro.caltech.edu, 10/2019
"""

import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import dsacalib.constants as ct
# Always import scipy before importing casatools
import scipy 
import casatools as cc
from dsacalib.utils import get_autobl_indices,read_caltable


def plot_dyn_spec(vis,fobs,mjd,bname,normalize=False,
                  outname=None,show=True,nx=None):
    """Plots the dynamic spectrum of the real part of the visibility.
    
    Parameters
    ----------
    vis : ndarray
        The visibilities to plot.  Dimensions
        (baseline,time,freq,polarization).
    fobs : ndarray
        The center frequency of each channel in GHz.
    mjd : ndarray
        The center of each subintegration in MJD.
    bname : list
        The name of each baseline in the visibilities.
    normalize : boolean
        If set to ``True``, the visibilities are normalized
        before plotting.  Defaults to ``False``.
    outname : str
        If provided and not set to ``None``, 
        the plot will be saved to the file
        `outname`_dynspec.png Defaults to ``None``.
    show : boolean
        If set to ``False`` the plot will be closed after
        rendering.  If set to ``True`` the plot is left open.
        Defaults to ``True``.
    nx : int
        The number of subplots in the horizontal direction 
        of the figure.  If not provided, or set to ``None``,
        `nx` is set to the minimum of 5 and the number of
        baselines in the visibilities.  Defaults to ``None``.
    """
    (nbl, nt, nf, npol) = vis.shape
    if nx is None:
        nx = min(nbl,5)
    ny = (nbl*2)//nx
    if (nbl*2)%nx != 0: ny += 1
    
    fig,ax = plt.subplots(ny,nx,figsize=(8*nx,8*ny))
    ax = ax.flatten()
    
    if len(mjd)>125:
        dplot = np.nanmean(vis[:,:nt//125*125,...].reshape(nbl,125,-1,nf,npol),2)
    else:
        dplot = vis.copy()
    if len(fobs) > 125:
        dplot = np.nanmean(dplot[:,:,:nf//125*125,:].reshape(nbl,dplot.shape[1],
                                                          125,-1,npol),3)
    dplot = dplot.real    
    dplot = dplot/dplot.reshape(nbl,-1,npol).mean(axis=1)[:,np.newaxis,np.newaxis,:]
    
    if normalize:
        dplot = dplot/np.abs(dplot)
        vmin = -1 
        vmax = 1
    else:
        vmin = -100
        vmax =  100
    dplot = dplot - 1
    
    if len(mjd)>125:
        x = mjd[:mjd.shape[0]//125*125].reshape(125,-1).mean(-1)
    else: 
        x = mjd
    x = ((x - x[0])*u.d).to_value(u.min)
    if len(fobs)>125:
        y = fobs[:nf//125*125].reshape(125,-1).mean(-1)
    else:
        y = fobs
    for i in range(nbl):
        for j in range(npol):
            ax[j*nbl+i].imshow(dplot[i,:,:,j].T,origin='lower',
                    interpolation='none',aspect='auto',
                    vmin=vmin,vmax=vmax,
                    extent=[x[0],x[-1],y[0],y[-1]])
            ax[j*nbl+i].text(0.1,0.9,
                             '{0}, pol {1}'.format(bname[i],'A' if j==0 else 'B'),
                             transform=ax[j*nbl+i].transAxes,
                             size=22,color='white')
    plt.subplots_adjust(wspace=0.1,hspace=0.1)
    for i in range((ny-1)*nx,ny*nx):
        ax[i].set_xlabel('time (min)')
    for i in np.arange(ny)*nx:
        ax[i].set_ylabel('freq (GHz)')
    if outname is not None:
        plt.savefig('{0}_dynspec.png'.format(outname))
    if not show:
        plt.close()
    return

def plot_vis_freq(vis,fobs,bname,outname=None,show=True,nx=None):
    """Plots visibilities against frequency.
    
    Creates plots of the amplitude and phase of the 
    visibilities `vis` as a function of frequency `fobs`.
    Two separate figures are opened, one for the amplitude 
    and one for the phases.  If `outname` is passed, these
    are saved as `outname`\_amp_freq.png and 
    `outname`\_phase_freq.png
    
    vis : ndarray
        The visibilities to plot.  Dimensions
        (baseline,time,freq,polarization).
    fobs : ndarray
        The center frequency of each channel in GHz.
    bname : list
        The name of each baseline in the visibilities.
    outname : str
        If provided and not set to ``None``, 
        the plots will be saved to the files
        `outname`\_amp_freq.png and `outname`\_phase_freq.png 
        Defaults to ``None``.
    show : boolean
        If set to ``False`` the plot will be closed after
        rendering.  If set to ``True`` the plot is left open.
        Defaults to ``True``.
    nx : int
        The number of subplots in the horizontal direction 
        of the figure.  If not provided, or set to ``None``,
        `nx` is set to the minimum of 5 and the number of
        baselines in the visibilities.  Defaults to ``None``.
    """
    (nbl,nt,nf,npol) = vis.shape
    if nx is None:
        nx = min(nbl,5)
    ny = nbl//nx
    if nbl%nx != 0: ny += 1
    
    dplot = vis.mean(1) 
    x = fobs
        
    fig,ax = plt.subplots(ny,nx,figsize=(8*nx,8*ny))
    ax = ax.flatten()
    for i in range(nbl):
        ax[i].plot(x,np.abs(dplot[i,:,0]),label='A')
        ax[i].plot(x,np.abs(dplot[i,:,1]),label='B')
        ax[i].text(0.1,0.9,'{0}: amp'.format(bname[i]),transform=ax[i].transAxes,
                  size=22)
    plt.subplots_adjust(wspace=0.1,hspace=0.1)
    ax[0].legend()
    for i in range((ny-1)*nx,ny*nx):
        ax[i].set_xlabel('freq (GHz)')
    if outname is not None:
        plt.savefig('{0}_amp_freq.png'.format(outname))
    if not show:
        plt.close()
        
    fig,ax = plt.subplots(ny,nx,figsize=(8*nx,8*ny))
    ax = ax.flatten()
    for i in range(nbl):
        ax[i].plot(x,np.angle(dplot[i,:,0]),label='A')
        ax[i].plot(x,np.angle(dplot[i,:,1]),label='B')
        ax[i].text(0.1,0.9,'{0}: phase'.format(bname[i]),transform=ax[i].transAxes,
                  size=22)
    plt.subplots_adjust(wspace=0.1,hspace=0.1)
    ax[0].legend()
    for i in range((ny-1)*nx,ny*nx):
        ax[i].set_xlabel('freq (GHz)')
    if outname is not None:
        plt.savefig('{0}_phase_freq.png'.format(outname))
    if not show:
        plt.close()
    return

def plot_vis_time(vis,mjd,bname,outname=None,show=True,nx=None):
    """Plots visibilities against time of observation.
    
    Creates plots of the amplitude and phase of the 
    visibilities `vis` as the time of observation `mjd`.
    Two separate figures are opened, one for the amplitude 
    and one for the phases.  If `outname` is passed, these
    are saved as `outname`\_amp_time.png and 
    `outname`\_phase_time.png    
    
    Parameters
    ----------
    vis : ndarray
        The visibilities to plot.  Dimensions
        (baseline,time,freq,polarization).
    mjd : ndarray
        The center of each subintegration in MJD.
    bname : list
        The name of each baseline in the visibilities.
    outname : str
        If provided and not set to ``None``, 
        the plot will be saved to the file
        `outname`\_dynspec.png Defaults to ``None``.
    show : boolean
        If set to ``False`` the plot will be closed after
        rendering.  If set to ``True`` the plot is left open.
        Defaults to ``True``.
    nx : int
        The number of subplots in the horizontal direction 
        of the figure.  If not provided, or set to ``None``,
        `nx` is set to the minimum of 5 and the number of
        baselines in the visibilities.  Defaults to ``None``.
    """
    (nbl,nt,nf,npol) = vis.shape
    if nx is None:
        nx  = min(nbl,5)
    ny = nbl//nx
    
    if nbl%nx != 0: ny += 1
    dplot = vis.mean(-2)
    x = ((mjd-mjd[0])*u.d).to_value(u.min)
    
    fig,ax = plt.subplots(ny,nx,figsize=(8*nx,8*ny))
    ax = ax.flatten()
    for i in range(nbl):
        ax[i].plot(x,np.abs(dplot[i,:,0]),label='A')
        ax[i].plot(x,np.abs(dplot[i,:,1]),label='B')
        ax[i].text(0.1,0.9,'{0}: amp'.format(bname[i]),transform=ax[i].transAxes,
                  size=22)
    plt.subplots_adjust(wspace=0.1,hspace=0.1)
    ax[0].legend()
    for i in range((ny-1)*nx,ny):
        ax[i].set_xlabel('time (min)')
    if outname is not None:
        plt.savefig('{0}_amp_time.png'.format(outname))
    if not show:
        plt.close()
        
    fig,ax = plt.subplots(ny,nx,figsize=(8*nx,8*ny))
    ax = ax.flatten()
    for i in range(nbl):
        ax[i].plot(x,np.angle(dplot[i,:,0]),label='A')
        ax[i].plot(x,np.angle(dplot[i,:,1]),label='B')
        ax[i].text(0.1,0.9,'{0}: phase'.format(bname[i]),transform=ax[i].transAxes,
                  size=22)
    plt.subplots_adjust(wspace=0.1,hspace=0.1)
    ax[0].legend()
    for i in range((ny-1)*nx,ny):
        ax[i].set_xlabel('time (min)')
    if outname is not None:
        plt.savefig('{0}_phase_time.png'.format(outname))
    if not show:
        plt.close()
    return

def plot_uv_track(bu,bv,outname=None,show=True):
    """Plots the uv track provided.
    
    Parameters
    ----------
    bu : ndarray
        The u-coordinates of the baselines in meters.
        Dimensions (baselines,time).
    bv : ndarray
        The v-coordinates of the baselines in meters.
        Dimensions (baselines, time)
    outname : str
        If provided and not set to ``None``, 
        the plot will be saved to the file
        `outname`_dynspec.png Defaults to ``None``.
    show : boolean
        If set to ``False`` the plot will be closed after
        rendering.  If set to ``True`` the plot is left open.
        Defaults to ``True``.
    """
    fig,ax = plt.subplots(1,1,figsize=(8,8))
    for i in range(bu.shape[0]):
        ax.plot(bu[i,:],bv[i,:])
        ax.set_xlim(-1500,1500)
        ax.set_ylim(-1500,1500)
        ax.text(-1200,1200,'UV Coverage')
        ax.set_xlabel('$u$ (m)')
        ax.set_ylabel('$v$ (m)')
    if outname is not None:
        plt.savefig('{0}_uv.png'.format(outname))
    if not show:
        plt.close()
    return

def rebin_vis(arr,nb1,nb2):
    """Rebins a 2-D array for plotting.  
    
    Excess bins along either axis are discarded.
    
    Parameters
    ----------
    arr : ndarray
        The two-dimensional array to rebin.
    nb1 : int
        The number of bins to rebin by along axis 0.
    nb2 : int
        The number of bins to rebin by along the axis 1.

    Returns
    -------
    arr: ndarray
        The rebinned array.
    """
    arr = arr[:arr.shape[0]//nb1*nb1,:arr.shape[1]//nb2*nb2].reshape(
        -1,nb1,arr.shape[1]).mean(1)
    arr = arr.reshape(arr.shape[0],-1,nb2).mean(-1)
    return arr

def plot_calibrated_vis(vis,vis_cal,mjd,fobs,bidx,pol=0,
                        outname=None,show=True):
    """Plots the calibrated and uncalibrated visibilities for comparison.
    
    
    Parameters
    ----------
    vis : ndarray
        The complex, uncalibrated visibilities, with dimensions (time, freq).  
    vis_cal : ndarray
        The complex calibrated visibilities, with dimensions (time, freq).
    mjd : ndarray
        The midpoint time of each subintegration in MJD.
    fobs : ndarray
        The center frequency of each channel in GHz.
    pol : int
        The index along the polarization index to plot.  Defaults to ``0``.
    outname : str
        The base to use for the name of the 
        png file the plot is saved to.  The plot will 
        be saved to `outname`_cal_vis.png if `outname` is
        provided, otherwise no plot will be saved.  Defaults ``None``.
    show : boolean
        If `show` is passed ``False``, the plot will be closed after 
        being generated.  Otherwise, it is left open.  Defaults ``True``.
    """
    x = mjd[:mjd.shape[0]//128*128].reshape(-1,128).mean(-1)
    x = ((x-x[0])*u.d).to_value(u.min)
    y = fobs[:fobs.shape[0]//5*5].reshape(-1,5).mean(-1)
    
    fig,ax = plt.subplots(2,2,figsize=(16,16), sharex=True, sharey=True)
    
    vplot = rebin_vis(vis[bidx,...,pol],128,5).T
    ax[0,0].imshow(vplot.real,
                   interpolation='none',origin='lower',
                   aspect='auto',vmin=-1,vmax=1,
                   extent=[x[0],x[-1],y[0],y[-1]])
    ax[0,0].text(0.1,0.9,'Before cal, real',transform=ax[0,0].transAxes,
                  size=22,color='white')
    ax[1,0].imshow(vplot.imag,
               interpolation='none',origin='lower',
               aspect='auto',vmin=-1,vmax=1,
               extent=[x[0],x[-1],y[0],y[-1]])
    ax[1,0].text(0.1,0.9,'Before cal, imag',transform=ax[1,0].transAxes,
                  size=22,color='white')
        
    vplot = rebin_vis(vis_cal[bidx,...,pol],128,5).T
    ax[0,1].imshow(vplot.real,
               interpolation='none',origin='lower',
               aspect='auto',vmin=-1,vmax=1,
               extent=[x[0],x[-1],y[0],y[-1]])
    ax[0,1].text(0.1,0.9,'After cal, real',transform=ax[0,1].transAxes,
                  size=22,color='white')
    ax[1,1].imshow(vplot.imag,
               interpolation='none',origin='lower',
               aspect='auto',vmin=-1,vmax=1,
               extent=[x[0],x[-1],y[0],y[-1]])
    ax[1,1].text(0.1,0.9,'After cal, imag',transform=ax[1,1].transAxes,
                  size=22,color='white')
    for i in range(2):
        ax[1,i].set_xlabel('time (min)')
        ax[i,0].set_ylabel('freq (GHz)')
    plt.subplots_adjust(hspace=0,wspace=0)
    if outname is not None:
        plt.savefig('{0}_{1}_cal_vis.png'.format(outname,'A' if pol==0 else 'B'))
    if not show:
        plt.close()
    return


def plot_delays(vis_ft,labels,delay_arr,bname,nx=None,outname=None,show=True):
    """Plots the visibility amplitude against delay.

    For a given visibility Fourier transformed along the frequency axis, 
    plots the amplitude against delay and calculates the location of the 
    fringe peak in delay.  Returns the peak delay for each visibility, 
    which can be used to calculate the antenna delays. 

    Parameters
    ----------
    vis_ft : ndarray
        The complex visibilities, dimensions (visibility type, baselines, delay).
        Note that the visibilities must have been Fourier transformed along the
        frequency axis and scrunched along the time axis before being passed
        to `plot_delays`.
    labels : list
        The labels of the types of visibilities passed.  For example, 
        ``['uncalibrated','calibrated']``.
    delay_arr : ndarray
        The center of the delay bins in nanoseconds.
    bname : list
        The baseline labels.
    nx : int
        The number of plots to tile along the horizontal axis. If `nx` is 
        given a value of ``None``, this is set to the number of baselines or 5, 
        if there are more than 5 baselines.
    outname : str
        The base to use for the name of the png file the plot is saved to.  
        The plot will be saved to `outname`_delays.png if an outname is
        provided.  If `outname` is given a value of ``None``, 
        no plot will be saved.  Defaults ``None``.
    show : boolean
        If `show` is given a value of ``False`` the plot will be closed.  
        Otherwise, the plot will be left open. Defaults ``True``.

    Returns
    -------
    delays : ndarray
        The peak delay for each visibility, in nanoseconds.
    """
    nvis = vis_ft.shape[0]
    nbl  = vis_ft.shape[1]
    npol = vis_ft.shape[-1]
    if nx is None:
        nx = min(nbl,5)
    ny = nbl//nx
    if nbl%nx != 0: ny+= 1
    
    alpha = 0.5 if nvis>2 else 1
    delays = delay_arr[np.argmax(np.abs(vis_ft),axis=2)]
    # could use scipy.signal.find_peaks instead
    for pidx in range(npol):
        fig,ax = plt.subplots(ny,nx,figsize=(8*nx,8*ny),sharex=True)#,sharey=True)
        ax = ax.flatten()
        for i in range(nbl):
            ax[i].axvline(0,color='black')
            for j in range(nvis):
                ax[i].plot(delay_arr, np.log10(np.abs(vis_ft[j,i,:,pidx])),
                      label=labels[j],alpha=alpha)
                ax[i].axvline(delays[j,i,pidx],color='red')
            ax[i].text(0.1,0.9,'{0}: {1}'.format(bname[i], 
                                                 'A' if pidx==0 
                                                 else 'B'), 
                       transform=ax[i].transAxes,
                       size=22)
        plt.subplots_adjust(wspace=0.1,hspace=0.1)
        ax[0].legend()
        for i in range((ny-1)*nx,ny*nx):
            ax[i].set_xlabel('delay (ns)')
        if outname is not None:
            plt.savefig('{0}_{1}_delays.png'.format(outname,'A' if pidx==0 else 'B'))
        if not show:
            plt.close()
            
    return delays

def plot_image(msname,imtype,sr0,verbose=False,outname=None,
               show=True,npix=256):
    """Uses CASA to grid and image visibilities.
    
    Parameters
    ----------
    msname : str
        The prefix of the measurement set name.  Opens 
        `msname`.ms.
    imtype : str
        The visibilities to image.  Can be 'observed', 
        'corrected', or 'model'.
    sr0 : src instance
        The source to use as the phase center in the image.
    verbose : boolean
        If ``True``, information about the image, including the
        location and SNR of the peak, are printed.  Default ``False``.
    outname : str
        The base to use for the name of the 
        png file the plot is saved to.  The plot will 
        be saved to `outname`_`imtype`.png if an outname is
        provided.  If `outname` is ``None``, the image is not 
        saved.  Defaults ``None``.
    show : boolean
        If `show` is ``False``, the plot will be closed after 
        being generated.  Defaults ``True``.
    npix : int
        The number of pixels per side to plot in the image. 
        Defaults 256.
    """
    error = 0
    im = cc.imager()
    error += not im.open('{0}.ms'.format(msname))
    me = cc.measures()
    qa = cc.quanta()
    direction = me.direction(sr0.epoch, 
            qa.quantity(sr0.ra.to_value(u.deg),'deg'), 
            qa.quantity(sr0.dec.to_value(u.deg),'deg'))
    error += not im.defineimage(nx=npix,ny=npix,
                    cellx='1arcsec',celly='1arcsec',)
    error += not im.makeimage(type=imtype,image='{0}_{1}.im'
             .format(msname,imtype))
    error += not im.done()
    
    ia = cc.image()
    error += not ia.open('{0}_{1}.im'.format(msname,imtype))
    dd = ia.summary()
    npixx,npixy,nch,npol = dd['shape']
    if verbose:
        print('Image shape: {0}'.format(dd['shape']))
    imvals = ia.getchunk(0, int(npixx))[:,:,0,0]
    error += ia.done()
    if verbose:
        peakx, peaky = np.where(imvals.max() == imvals)
        print('Peak SNR at pix ({0},{1}) = {2}'.format(
            peakx[0], peaky[0], imvals.max()/imvals.std()))
        print('Value at peak: {0}'.format(imvals.max()))
        print('Value at origin: {0}'.format(imvals[imvals.shape[0]//2,
                                                  imvals.shape[1]//2]))

    fig,ax = plt.subplots(1,1,figsize=(15,8))
    pim = ax.imshow(imvals.transpose(), interpolation='none', 
               origin='lower',
              extent=[-imvals.shape[0]/2,imvals.shape[0]/2,
                     -imvals.shape[1]/2,imvals.shape[1]/2])
    plt.colorbar(pim)
    ax.axvline(0,color='white',alpha=0.5)
    ax.axhline(0,color='white',alpha=0.5)
    ax.set_xlabel('l (arcsec)')
    ax.set_ylabel('m (arcsec)')
    if outname is not None:
        plt.savefig('{0}_{1}.png'.format(outname,imtype))
    if not show:
        plt.close()
    if error > 0:
        print('{0} errors occured during imaging'.format(error))
    return

def plot_antenna_delays(msname,calname,antenna_order,outname=None,show=True):
    """Plots antenna delay variations between two delay calibrations.
    
    Compares the antenna delays used in the calibration solution on the timescale 
    of the entire calibrator pass (assumed to be in a CASA table ending in 'kcal') 
    to those calculated on a shorter (e.g. 60s) timescale (assumed ot be in a CASA
    table ending in '2kcal').
    
    Parameters
    ----------
    msname : str
        The prefix of the measurement set (`msname`.ms), used to identify 
        the correct delay calibration tables.
    calname : str
        The calibrator source name, used to identify the correct delay calibration 
        tables.  The tables `msname`\_`calname`\_kcal (for a single delay 
        calculated using the entire calibrator pass) and 
        `msname`\_`calname`\_2kcal (for delays
        calculated on a shorter timescale) will be opened.
    antenna_order : ndarray
        The antenna names in order.
    outname : str
        The base to use for the name of the 
        png file the plot is saved to.  The plot will 
        be saved to `outname`\_antdelays.png.  If `outname` is set to ``None``,
        the plot is not saved.  Defaults ``None``.
    show : boolean
        If set to ``False`` the plot will be closed after it is generated.
        Defaults ``True``.

    Returns
    -------
    times : ndarray
        The times at which the antenna delays (on short timescales) were calculated, 
        in MJD.
    antenna_delays : ndarray
        The delays of the antennas on short timescales, in nanoseconds.  
        Dimensions (polarization,time,antenna).
    kcorr : ndarray
        The delay correction calculated using the entire observation time,
        in nanoseconds.  Dimensions (polarization,1,antenna).
    """
    nant = len(antenna_order)
    ccyc = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Pull the solutions for the entire timerange and the 
    # 60-s data from the measurement set tables
    times,antenna_delays,flags = read_caltable('{0}_{1}_2kcal'
                                               .format(msname,calname),
                                               nant,cparam=False)
    npol = antenna_delays.shape[0]
    times = times[:,0]
    tkcorr,kcorr,flags = read_caltable('{0}_{1}_kcal'
                                               .format(msname,calname),
                                               nant,cparam=False)
    
    tplot = (times-times[0])*ct.seconds_per_day/60.
    fig,ax = plt.subplots(1,1,figsize=(10,8))
    for i in range(nant):
        plt.plot(tplot,antenna_delays[0,:,i]-kcorr[0,0,i],'.',
            label=antenna_order[i],alpha=0.5,
                color=ccyc[i%len(ccyc)])
        plt.plot(tplot,antenna_delays[1,:,i]-kcorr[1,0,i],'x',
            alpha=0.5,color=ccyc[i%len(ccyc)])
    plt.ylim(-5,5)
    plt.ylabel('delay (ns)')
    plt.legend(ncol=3,fontsize='medium')
    plt.xlabel('time (min)')
    plt.axhline(1.5)
    plt.axhline(-1.5)
    if outname is not None:
        plt.savefig('{0}_antdelays.png'.format(outname))
    if not show:
        plt.close()

    return times, antenna_delays, kcorr

def plot_gain_calibration(msname,calname,antenna_order,
                          bname=None, blbased=False,
                          plabels=['A','B'],
                          outname=None,show=True):
    """Plots the gain calibration solutions.
    
    Parameters
    ----------
    msname : str
        The name of the measurement set.  Used to identify the 
        gain calibration tables.
    calname : str
        The calibrator used in gain calibration.  Used to identify
        the gain calibration tables.  The tables 
        `msname`\_`calname`\_gacal (assumed to contain the gain amplitude
        soutions) and `msname`\_`calname`\_gpcal (assumed to contain
        the phase amplitude solutions) will be opened.
    antenna_order : ndarray or list
        The antennas in their order in the measurement set.
    bname : list
        The names of the baselines in the calibration solutions.
    blbased : boolean
        Set to ``True`` if the gain solutions are baseline-based,
        ``False`` if they are antenna-based.  Used to properly parse 
        the gain table.
    plabels : list
        The names of the polarizations in the calibration solutions.  
        Defaults ``['A','B']``.
    outname : str
        The base to use for the name of the 
        png file the plot is saved to.  The plot is saved to 
        `outname`\_gaincal.png if `outname` is not ``None``.  if `outname` is
        set to ``None, the plot is not saved.  Defaults ``None``.
    show : boolean
        If set to ``False`` the plot is closed after being generated.
    """
    nant = len(antenna_order)
    if blbased:
        labels = []
        for i in range(nant):
            for j in range(i,nant):
                labels += [[i,j]]
    else:
        labels = antenna_order
    
    nlab = len(labels)
    idxs_to_plot = list(range(nlab))
    if blbased:
        autocorr_idx = get_autobl_indices(nant)
        autocorr_idx = [(nlab-1)-aidx for aidx in autocorr_idx]
        for idx in autocorr_idx:
            idxs_to_plot.remove(idx)
    
    time_phase,gain_phase,flags = read_caltable('{0}_{1}_gpcal'.
                                               format(msname,calname),
                                               nlab,cparam=True)
    time_phase = time_phase[:,0]
    npol = gain_phase.shape[0]
    
    time,gain_amp,flags = read_caltable('{0}_{1}_gacal'.
                                       format(msname,calname),
                                       nlab,cparam=True)
    time = time[:,0]
    t0 = time[0]
    time = ((time - t0)*u.d).to_value(u.min)
    time_phase = ((time_phase - t0)*u.d).to_value(u.min)
    
    ccyc = plt.rcParams['axes.prop_cycle'].by_key()['color']
    lcyc = ['-',':']
    fig,ax = plt.subplots(1,2,figsize=(16,6),sharex=True)
    
    if gain_amp.shape[1]>1:
        tplot = time
        gplot = gain_amp
    else:
        tplot = [0,1]
        gplot = np.tile(gain_amp,[1,2,1])
        
    for i,bidx in enumerate(idxs_to_plot):
        for pidx in range(npol):
            ax[0].plot(tplot,np.abs(gplot[pidx,:,bidx]),
                       label='{0} {1}'.format(labels[bidx],
                        plabels[pidx]),
                       color=ccyc[i%len(ccyc)],ls=lcyc[pidx])

            
    if gain_phase.shape[1]>1:
        tplot = time_phase
        gplot = gain_phase
    else:
        tplot = [tplot[0],tplot[-1]] # use the time from the gains
        gplot = np.tile(gain_phase,[1,2,1])

    for i,bidx in enumerate(idxs_to_plot):
        for pidx in range(npol):
            ax[1].plot(tplot,np.angle(gplot[pidx,:,bidx]),
                       label='{0} {1}'.format(labels[bidx],
                            plabels[pidx]),
                       color=ccyc[i%len(ccyc)],
                      ls = lcyc[pidx])

    ax[0].set_xlim(tplot[0],tplot[-1])
    ax[1].set_ylim(-np.pi,np.pi)
    ax[0].legend(ncol=3,fontsize=12)
    ax[0].set_xlabel('time (min)')
    ax[1].set_xlabel('time (min)')
    ax[0].set_ylabel('Abs of gain')
    ax[1].set_ylabel('Phase of gain')
    if outname is not None:
        plt.savefig('{0}_gaincal.png'.format(outname))
    if not show:
        plt.close()
    return time,gain_amp,gain_phase,labels,t0

def plot_bandpass(msname,calname,antenna_order,fobs,
                  blbased=False,plabels=['A','B'],
                 outname=None,show=True):
    """Plots the bandpass calibration solutions.
    
    Parameters
    ----------
    msname : str
        The name of the measurement set.  Used to identify
        the calibration table to plot.
    calname : str
        The name of the calibrator used in calibration.  The 
        calibration table `msname`\_`calname`\_bcal is opened.
    antenna_order : array
        The antenna names in the order they appear in the 
        calibration table.
    fobs : array
        The frequency of each channel in GHz.
    blbased : boolean
        Set to ``True`` if baseline-based routines were used 
        to calculate the bandpass solutions, ``False`` if 
        antenna-based routines were used. 
        Defaults ``False``.
    plabels : list
        The labels for the polarizations.  Defaults ``['A','B']``.
    outname : str
        The base to use for the name of the 
        png file the plot is saved to.  The plot is saved to 
        `outname`\_bandpass.png if an `outname` is not set to ``None``.
        If `outname` is set to ``None``, the plot is not saved.  
        Defaults ``None``.
    show : boolean
        If set to ``False``, the plot is closed after it is generated.
    """    
    nant = len(antenna_order)
    if blbased:
        labels = []
        for i in range(nant):
            for j in range(i,nant):
                labels += [[i,j]]
    else:
        labels = antenna_order
    
    nlab = len(labels)
    idxs_to_plot = list(range(nlab))
    if blbased:
        autocorr_idx = get_autobl_indices(nant)
        autocorr_idx = [(nlab-1)-aidx for aidx in autocorr_idx]
        for idx in autocorr_idx:
            idxs_to_plot.remove(idx)
    
    tbpass,bpass,flags = read_caltable('{0}_{1}_bcal'.format(msname,calname),
                                      nlab,cparam=True)
    npol = bpass.shape[0]
    
    if bpass.shape[1]!=fobs.shape[0]:
        nint = fobs.shape[0]//bpass.shape[1]
        fobs_plot = np.mean(fobs[:nint]) + \
            np.arange(bpass.shape[1])*np.median(np.diff(fobs))*nint
    else:
        fobs_plot = fobs.copy()

    ccyc = plt.rcParams['axes.prop_cycle'].by_key()['color']
    lcyc = ['-',':']
    fig,ax = plt.subplots(1,2,figsize=(16,6),sharex=True)
    for i,bidx in enumerate(idxs_to_plot):
        for pidx in range(npol): 
            ax[0].plot(fobs_plot,np.abs(bpass[pidx,:,bidx]),'.',
                 label='{0}{1}'.format(labels[bidx],
                            plabels[pidx]),
                alpha=0.5,ls=lcyc[pidx],
                color=ccyc[i%len(ccyc)])
            ax[1].plot(fobs_plot,np.angle(bpass[pidx,:,bidx]),'.',
                 label='{0}{1}'.format(labels[bidx],
                                        plabels[pidx]),
                 alpha=0.5,ls=lcyc[pidx],
                 color=ccyc[i%len(ccyc)])
    ax[0].set_xlabel('freq (GHz)')
    ax[1].set_xlabel('freq (GHz)')
    ax[0].set_ylabel('B cal amp')
    ax[1].set_ylabel('B cal phase')
    ax[0].legend(ncol=3,fontsize='medium')
    if outname is not None:
        plt.savefig('{0}_bandpass.png'.format(outname))
    if not show:
        plt.close()
    
    return bpass
    
