"""
DSA_PLOTTING.PY

Dana Simard, dana.simard@astro.caltech.edu, 10/2019

Visibility plotting routines for DSA-10
"""

import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import dsacalib.constants as ct
import __casac__ as cc


def plot_dyn_spec(vis,fobs,mjd,bname,normalize=False,
                  outname=None,show=True):
    """Plots the dynamic spectrum of the real part for each visibility
    
    Args:
        vis: complex arr
          the visibilities to plot the dynamic spectra of
        fobs: real arr
          the midpoint frequency of each channel in GHz
        mjd: real arr
          the midpoint mjd of each subintegration
        bname: list(str)
          the name for each baseline
        normalize: boolean
          whether to normalize the visibilities before plotting
        outname: str
          if not None the plot will be saved to a file <outname>_dynspec.png
        show: boolean
          if False the plot will be closed.  If using a notebook
          and show is True, the plot will show in the notebook
          
    Returns:
    """
    (nbl, nt, nf, npol) = vis.shape
    ny = (nbl*2)//5
    if (nbl*2)%5 != 0: ny += 1
    
    fig,ax = plt.subplots(ny,5,figsize=(8*5,8*ny))
    ax = ax.flatten()
    
    dplot = (np.nanmean(np.nanmean(vis[:,:nt//125*125,:nf//125*125,:].reshape(
            nbl,125,-1,nf,npol),2).reshape(nbl,125,125,-1,npol),3)).real
    dplot = dplot/dplot.reshape(nbl,-1,npol).mean(axis=1)[:,np.newaxis,np.newaxis,:]
    
    if normalize:
        dplot = dplot/np.abs(dplot)
        vmin = -1 
        vmax = 1
    else:
        vmin = -100
        vmax =  100
    dplot = dplot - 1
    
    x = mjd[:mjd.shape[0]//125*125].reshape(125,-1).mean(-1)
    x = ((x - x[0])*u.d).to_value(u.min)
    y = fobs[:nf//125*125].reshape(125,-1).mean(-1)
    for i in range(nbl):
        for j in range(npol):
            ax[i*npol+j].imshow(dplot[i,:,:,j].T,origin='lower',
                    interpolation='none',aspect='auto',
                    vmin=vmin,vmax=vmax,
                    extent=[x[0],x[-1],y[0],y[-1]])
            ax[i*npol+j].text(0.1,0.9,
                             '{0}, pol {1}'.format(bname[i],'A' if j==0 else 'B'),
                             transform=ax[i].transAxes,
                             size=22,color='white')
    plt.subplots_adjust(wspace=0.1,hspace=0.1)
    for i in range((ny-1)*5,ny*5):
        ax[i].set_xlabel('time (min)')
    for i in np.arange(ny)*5:
        ax[i].set_ylabel('freq (GHz)')
    if outname is not None:
        plt.savefig('{0}_dynspec.png'.format(outname))
    if not show:
        plt.close()
    return

def plot_vis_freq(vis,fobs,bname,outname=None,show=True):
    """Plots visibility against frequency of observation for 
    all baselines.  Bins to 125 points for each baseline.
    
    Args:
        vis: complex arr
          the visibilities, dimensions (pol,baselines, time, frequency)
        fobs: real arr
          the midpoint frequency of each channel in GHz
        bname: list(str)
          the name of each baseline 
        outname: str, optional 
          the base to use for the name of the 
          png file the plot is saved to.  The plot will 
          be saved to <outname>_freq.png if an outname is
          provided, otherwise no plot will be saved.
        show: boolean, optional
          if False the plot will be closed.  If using a notebook
          and show is True, the plot will show in the notebook

    Returns:
    """
    (nbl,nt,nf,npol) = vis.shape
    ny = nbl//5
    if nbl%5 != 0: ny += 1
    
    dplot = vis[:,:,:nf//125*125,:].mean(1).reshape(nbl,125,-1,npol).mean(3)
    x = fobs[:nf//125*125].reshape(125,-1).mean(-1)
        
    fig,ax = plt.subplots(ny,5,figsize=(8*5,8*ny))
    ax = ax.flatten()
    for i in range(nbl):
        ax[i].plot(x,np.abs(dplot[i,:,0]),label='A')
        ax[i].plot(x,np.abs(dplot[i,:,1]),label='B')
        ax[i].text(0.1,0.9,bname[i],transform=ax[i].transAxes,
                  size=22)
    plt.subplots_adjust(wspace=0.1,hspace=0.1)
    ax[0].legend()
    for i in range((ny-1)*5,ny*5):
        ax[i].set_xlabel('freq (GHz)')
    if outname is not None:
        plt.savefig('{0}_amp_freq.png'.format(outname))
    if not show:
        plt.close()
        
    fig,ax = plt.subplots(ny,5,figsize=(8*5,8*ny))
    ax = ax.flatten()
    for i in range(nbl):
        ax[i].plot(x,np.angle(dplot[i,:,0]),label='A')
        ax[i].plot(x,np.angle(dplot[i,:,1]),label='B')
        ax[i].text(0.1,0.9,bname[i],transform=ax[i].transAxes,
                  size=22)
    plt.subplots_adjust(wspace=0.1,hspace=0.1)
    ax[0].legend()
    for i in range((ny-1)*5,ny*5):
        ax[i].set_xlabel('freq (GHz)')
    if outname is not None:
        plt.savefig('{0}_phase_freq.png'.format(outname))
    if not show:
        plt.close()
    return

def plot_vis_time(vis,mjd,bname,outname=None,show=True):
    """Plots visibility against time of observation for 
    all baselines.  
    
    Args:
        vis: complex arr
          the visibilities, dimensions (pol,baselines, time, frequency)
        mjd: real arr
          the midpoint MJD of each subintegration
        bname: list(str)
          the name of each baseline 
        outname: str
          the base to use for the name of the 
          png files the plots is saved to.  The plots will 
          be saved to <outname>_amp_time.png and outname_phase_time.png
          if an outname is provided; otherwise no plot will be saved.
        show: boolean
          if False the plot will be closed.  If using a notebook
          and show is True, the plot will show in the notebook
               
    Returns:
    """
    (nbl,nt,nf,npol) = vis.shape
    ny = nbl//5
    
    if nbl%5 != 0: ny += 1
    dplot = vis.mean(-2)
    x = ((mjd-mjd[0])*u.d).to_value(u.min)
    
    fig,ax = plt.subplots(ny,5,figsize=(8*5,8*ny))
    ax = ax.flatten()
    for i in range(nbl):
        ax[i].plot(x,np.abs(dplot[i,:,0]),label='A')
        ax[i].plot(x,np.abs(dplot[i,:,1]),label='B')
        ax[i].text(0.1,0.9,bname[i],transform=ax[i].transAxes,
                  size=22)
    plt.subplots_adjust(wspace=0.1,hspace=0.1)
    ax[0].legend()
    for i in range((ny-1)*5,ny):
        ax[i].set_xlabel('time (min)')
    if outname is not None:
        plt.savefig('{0}_abs_time.png'.format(outname))
    if not show:
        plt.close()
        
    fig,ax = plt.subplots(ny,5,figsize=(8*5,8*ny))
    ax = ax.flatten()
    for i in range(nbl):
        ax[i].plot(x,np.angle(dplot[i,:,0]),label='A')
        ax[i].plot(x,np.angle(dplot[i,:,1]),label='B')
        ax[i].text(0.1,0.9,bname[i],transform=ax[i].transAxes,
                  size=22)
    plt.subplots_adjust(wspace=0.1,hspace=0.1)
    ax[0].legend()
    for i in range((ny-1)*5,ny):
        ax[i].set_xlabel('time (min)')
    if outname is not None:
        plt.savefig('{0}_phase_time.png'.format(outname))
    if not show:
        plt.close()
    return

def plot_uv_track(bu,bv,outname=None,show=True):
    """Plots the uv track provided
    
    Args:
        bu: real arr
          the values of u of the baselines in m,
          dimensions (baselines,time)
        bv: real arr
          the values of v of the baselines in m,
          dimension (baselines, time)
        outname: str
          the base to use for the name of the 
          png file the plot is saved to.  The plot will 
          be saved to <outname>_freq.png if an outname is
          provided, otherwise no plot will be saved.
        show: boolean
          If true, the plot will be shown in an inline 
          notebook.  If not using a notebook, the plot will never be shown.

    Returns:
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
    """ Rebins a 2-D array for plotting.  Excess bins along either axis
    are discarded.
    
    Args:
        arr: arr
          the two-dimensional array to rebin
        nb1: int
          the number of bins to rebin by along the 0th axis
        nb2: int
          the number of bins to rebin by along the 1st axis

    Returns:
        arr: arr
          the rebinned array
    """
    arr = arr[:arr.shape[0]//nb1*nb1,:arr.shape[1]//nb2*nb2].reshape(
        -1,nb1,arr.shape[1]).mean(1)
    arr = arr.reshape(arr.shape[0],-1,nb2).mean(-1)
    return arr

def plot_calibrated_vis(vis,vis_cal,mjd,fobs,bidx,pol=0,
                        outname='None',show=True):
    """Plots two visibilities for comparison for e.g. a calibrated
    and uncalibrated visibility
    
    Args:
        vis: complex arr
          dimensions (time, freq), the uncalibrated visibility to plot
        vis_cal: complex arr
          dimensions (time, freq), the calibrated visibility to plot
        mjd: real arr
          the midpoint MJD of each subintegration
        fobs: real arr
          the midpoint frequency of each channel in GHz
        pol: int
          the polarization index to plot
        outname: str
          the base to use for the name of the 
          png file the plot is saved to.  The plot will 
          be saved to <outname>_cal_vis.png if an outname is
          provided, otherwise no plot will be saved.
        show: boolean
          if False the plot will be closed.  If using a notebook
          and show is True, the plot will show in the notebook

    Returns:
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


def plot_delays(vis_ft,labels,delay_arr,bname,pol=0,outname=None,show=True):
    """Make amp vs delay plots for each visibility

    Args:
        vis_ft: complex arr
          the visibilities, dimensions (vis, baselines, delay)
        labels: list(str)
          the labels for the visibilities passed
        delay_arr: real array
          the delay bins in nanoseconds
        bname: list(str)
          the baseline labels
        pol: int
          the polarization index to plot
        outname: str
          the base to use for the name of the 
          png file the plot is saved to.  The plot will 
          be saved to <outname>_delays.png if an outname is
          provided, otherwise no plot will be saved.
        show: boolean
          if False the plot will be closed.  If using a notebook
          and show is True, the plot will show in the notebook

    Returns:
    """
    nvis = vis_ft.shape[0]
    nbl  = vis_ft.shape[1]
    ny = nbl//5
    if nbl%5 != 0: ny+= 1
    
    alpha = 0.5 if nvis>2 else 1
    fig,ax = plt.subplots(ny,5,figsize=(8*5,8*ny),sharex=True)#,sharey=True)
    ax = ax.flatten()
    for i in range(nbl):
        ax[i].axvline(0,color='red')
        for j in range(nvis):
            ax[i].plot(delay_arr, np.log10(np.abs(vis_ft[j,i,:,pol])),
                      label=labels[j],alpha=alpha)
        ax[i].text(0.1,0.9,bname[i],transform=ax[i].transAxes,
                   size=22)
    plt.subplots_adjust(wspace=0.1,hspace=0.1)
    ax[0].legend()
    for i in range(45-5,45):
        ax[i].set_xlabel('delay (ns)')
    if outname is not None:
        plt.savefig('{0}_{1}_delays.png'.format(outname,'A' if pol==0 else 'B'))
    if not show:
        plt.close()
    return

def plot_image(msname,imtype,sr0,verbose=False,outname=None,
               show=True,npix=256):
    """Uses CASA to grid an image and the plot the image.
    
    Args:
        msname: str
          the prefix of the measurement set 
        imtype: str
          the type of image to produce.  Can be 'observed', 
          'corrected', or 'model'
        sr0: src class
          the source to use as the phase center in the image
        verbose: boolean
          whether to print information about the produced image
        outname: str
          the base to use for the name of the 
          png file the plot is saved to.  The plot will 
          be saved to <outname>_<imtype>.png if an outname is
          provided, otherwise no plot will be saved.
        show: boolean
          if False the plot will be closed.  If using a notebook
          and show is True, the plot will show in the notebook
        npix: int
          the number of pixels per side to plot

    Returns:
    """
    error = 0
    im = cc.imager.imager()
    error += not im.open('{0}.ms'.format(msname))
    me = cc.measures.measures()
    qa = cc.quanta.quanta()
    direction = me.direction(sr0.epoch, 
            qa.quantity(sr0.ra.to_value(u.deg),'deg'), 
            qa.quantity(sr0.dec.to_value(u.deg),'deg'))
    error += not im.defineimage(nx=npix,ny=npix,
                    cellx='1arcsec',celly='1arcsec',)
    error += not im.makeimage(type=imtype,image='{0}_{1}.im'
             .format(msname,imtype))
    error += not im.done()
    
    ia = cc.image.image()
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
    """Plots the antenna delays on a 59s timescale relative to the antenna
    delays used in the calibration solution
    
    Args:
        prefix: str
          the prefix of the kcal measurement set
        antenna_order: int array or list(int)
          the order of the antennas 
        outname: str
          the base to use for the name of the 
          png file the plot is saved to.  The plot will 
          be saved to <outname>_antdelays.png if an outname is
          provided, otherwise no plot will be saved.
        show: boolean
          if False the plot will be closed.  If using a notebook
          and show is True, the plot will show in the notebook

    Returns:
        times: arr, real
          the times at which the antenna delays are calculated in mjd
        antenna_delays: arr, real
          the delays of the antennas solved every 59s in nanoseconds
        kcorr: arr, real
          the applied delay correction calculated using the entire time
          in nanoseconds
    """
    nant = len(antenna_order)
    ccyc = plt.rcParams['axes.prop_cycle'].by_key()['color']
    error = 0
    
    # Pull the solutions for the entire timerange and the 
    # 60-s data from the measurement set tables
    tb = cc.table.table()
    print('opening {0}_{1}_2kcal'.format(msname,calname))
    error += not tb.open('{0}_{1}_2kcal'.format(msname,calname))
    antenna_delays = tb.getcol('FPARAM')
    npol = antenna_delays.shape[0]
    antenna_delays = antenna_delays.reshape(npol,-1,nant)
    times = (tb.getcol('TIME').reshape(-1,nant)[:,0]*u.s).to_value(u.d)
    error += not tb.close()
    tb = cc.table.table()
    print('opening {0}_{1}_kcal'.format(msname,calname))
    error += not tb.open('{0}_{1}_kcal'.format(msname,calname))
    kcorr = tb.getcol('FPARAM').reshape(npol,-1,nant)
    error += not tb.close()
    
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
    if error > 0:
        print('{0} errors occured'.format(error))
    return times, antenna_delays, kcorr

def plot_gain_calibration(msname,calname,antenna_order,
                          outname=None,show=True):
    """Plot the gain calibration solution
    
    Args:
        source: src class
          the calibrator
        antenna_order: arr(int) or list(int)
          the order of the antennas 
        outname: str
          the base to use for the name of the 
          png file the plot is saved to.  The plot will 
          be saved to <outname>_gaincal.png if an outname is
          provided, otherwise no plot will be saved.
        show: boolean
          if False the plot will be closed.  If using a notebook
          and show is True, the plot will show in the notebook
    
    Returns:
    """
    nant = len(antenna_order)
    ccyc = plt.rcParams['axes.prop_cycle'].by_key()['color']
    error = 0
    
    tb = cc.table.table()
    error += not tb.open('{0}_{1}_gpcal'.format(msname,calname))
    gain_phase = tb.getcol('CPARAM')
    npol = gain_phase.shape[0]
    gain_phase = gain_phase.reshape(npol,-1,nant)
    error += not tb.close()

    tb = cc.table.table()
    error += not tb.open('{0}_{1}_gacal'.format(msname,calname))
    gain_amp = tb.getcol('CPARAM')
    gain_amp = gain_amp.reshape(npol,-1,nant)
    time = tb.getcol('TIME').reshape(-1,nant)[:,0]
    error += not tb.close()
    time = ((time - time[0])*u.s).to_value(u.min)
    
    fig,ax = plt.subplots(1,2,figsize=(16,6),sharex=True)
    for i in range(nant):
        if gain_amp.shape[1]>1:
            ax[0].plot(time,np.abs(gain_amp[0,:,i]),
                       label=antenna_order[i],
                       color=ccyc[i%len(ccyc)])
            ax[0].plot(time,np.abs(gain_amp[1,:,i]),
                       color=ccyc[i%len(ccyc)],
                      ls=':')
        else:
            ax[0].plot([time[0],time[-1]],
                       [np.abs(gain_amp[0,0,i]),
                        np.abs(gain_amp[0,0,i])],
                          color=ccyc[i%len(ccyc)])
            ax[0].plot([time[0],time[-1]],
                       [np.abs(gain_amp[1,0,i]),
                        np.abs(gain_amp[1,0,i])],
                          color=ccyc[i%len(ccyc)],ls = ':')
        if gain_phase.shape[1]>1:
            ax[1].plot(time,
                   np.angle(gain_phase[0,:,i]),
                       color=ccyc[i%len(ccyc)])
            ax[1].plot(time,
                   np.angle(gain_phase[1,:,i]),
                       color=ccyc[i%len(ccyc)],
                  ls = ':')
        else:
            ax[1].plot([time[0],time[-1]],
                       [np.angle(gain_phase[0,0,i]),
                        np.angle(gain_phase[0,0,i])],
                  color=ccyc[i%len(ccyc)])
            ax[1].plot([time[0],time[-1]],
                       [np.angle(gain_phase[1,0,i]),
                        np.angle(gain_phase[1,0,i])],
                          color=ccyc[i%len(ccyc)],ls = ':')
    ax[0].set_xlim(time[0],time[-1])
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
    if error > 0:
        print('{0} errors occured'.format(error))
    return

def plot_bandpass(msname,calname,antenna_order,fobs,
                 outname=None,show=True):
    nant = len(antenna_order)
    ccyc = plt.rcParams['axes.prop_cycle'].by_key()['color']
    error = 0
    
    tb = cc.table.table()
    error += not tb.open('{0}_{1}_bcal'.format(msname,calname))
    bpass = tb.getcol('CPARAM')
    error += not tb.close()
    
    fig,ax = plt.subplots(1,2,figsize=(16,6),sharex=True)
    for i in range(nant):
        ax[0].plot(fobs,np.abs(bpass[0,:,i]),'.',
                 label=antenna_order[i],alpha=0.5,
                color=ccyc[i%len(ccyc)])
        ax[0].plot(fobs,np.abs(bpass[1,:,i]),'x',
                alpha=0.5,
                color=ccyc[i%len(ccyc)])
        ax[1].plot(fobs,np.angle(bpass[0,:,i]),'.',
                 label=antenna_order[i],alpha=0.5,
                color=ccyc[i%len(ccyc)])
        ax[1].plot(fobs,np.angle(bpass[1,:,i]),'x',
                alpha=0.5,
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
    if error > 0:
        print('{0} errors occured'.format(error))
    return
    