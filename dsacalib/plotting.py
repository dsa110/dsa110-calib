"""
DSA_PLOTTING.PY
Dana Simard, dana.simard@astro.caltech.edu, 10/2019
Visibility plotting routines for DSA-10
"""

import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import __casac__ as cc


def plot_dyn_spec(vis,fobs,mjd,bname,normalize=False,
                  outname=None,show=True):
    fig,ax = plt.subplots(9,5,figsize=(8*5,8*9))
    ax = ax.flatten()
    dplot = (np.nanmean(np.nanmean(vis[:,:vis.shape[1]//125*125,:].reshape(
            45,125,-1,625),2).reshape(45,125,125,5),-1)).real
    dplot = dplot/dplot.reshape(45,-1).mean(axis=-1)[:,np.newaxis,np.newaxis]
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
    y = fobs.reshape(125,5).mean(-1)
    for i in range(45):
        ax[i].imshow(dplot[i].T,origin='lower',
                    interpolation='none',aspect='auto',
                    vmin=vmin,vmax=vmax,
                    extent=[x[0],x[-1],y[0],y[-1]])
        ax[i].text(0.1,0.9,bname[i],transform=ax[i].transAxes,
                  size=22,color='white')
    plt.subplots_adjust(wspace=0.1,hspace=0.1)
    for i in range(45-5,45):
        ax[i].set_xlabel('time (min)')
    for i in np.arange(9)*5:
        ax[i].set_ylabel('freq (GHz)')
    if outname is not None:
        plt.savefig('{0}_dynspec.png'.format(outname))
    if not show:
        plt.close()
    return

def plot_vis_freq(vis,fobs,bname,outname=None,show=True):
    """Plots visibility against frequency of observation for 
    all baselines.  Bins to 125 points for each baseline.
    Parameters:
    -----------
    vis      : complex array, the visibilities, dimensions 
               (baselines, time, frequency)
    fobs     : real array, the frequency of observation in GHz
    bname    : list, the name of each baseline 
    outname  : string, optional the base to use for the name of the 
               png file the plot is saved to.  The plot will 
               be saved to <outname>_freq.png if an outname is
               provided, otherwise no plot will be saved.
    show     : boolean, optional, default True.  If true, the plot
               will be shown in an inline notebook.  If not using 
               a notebook, the plot will never be shown.
    Returns none
    """
    fig,ax = plt.subplots(9,5,figsize=(8*5,8*9))
    ax = ax.flatten()
    dplot = vis.mean(1).reshape(45,125,5).mean(-1)
    x = fobs.reshape(125,5).mean(-1)
    for i in range(45):
        ax[i].plot(x,dplot[i].real,label='real')
        ax[i].plot(x,dplot[i].imag,label='imag')
        ax[i].text(0.1,0.9,bname[i],transform=ax[i].transAxes,
                  size=22)
    plt.subplots_adjust(wspace=0.1,hspace=0.1)
    ax[0].legend()
    for i in range(45-5,45):
        ax[i].set_xlabel('freq (GHz)')
    if outname is not None:
        plt.savefig('{0}_freq.png'.format(outname))
    if not show:
        plt.close()
    return

def plot_vis_time(vis,mjd,bname,outname=None,show=True):
    """Plots visibility against time of observation for 
    all baselines.  
    Parameters:
    -----------
    vis      : complex array, the visibilities, dimensions 
               (baselines, time, frequency)
    mjd      : real array, the MJD of each integration
    bname    : list, the name of each baseline 
    outname  : string, optional the base to use for the name of the 
               png file the plot is saved to.  The plot will 
               be saved to <outname>_freq.png if an outname is
               provided, otherwise no plot will be saved.
    show     : boolean, optional, default True.  If true, the plot
               will be shown in an inline notebook.  If not using 
               a notebook, the plot will never be shown.
    Returns none
    """
    fig,ax = plt.subplots(9,5,figsize=(8*5,8*9))
    ax = ax.flatten()
    dplot = vis.mean(-1)
    x = ((mjd-mjd[0])*u.d).to_value(u.min)
    for i in range(45):
        ax[i].plot(x,dplot[i].real,label='real')
        ax[i].plot(x,dplot[i].imag,label='imag')
        ax[i].text(0.1,0.9,bname[i],transform=ax[i].transAxes,
                  size=22)
    plt.subplots_adjust(wspace=0.1,hspace=0.1)
    ax[0].legend()
    for i in range(45-5,45):
        ax[i].set_xlabel('time (min)')
    if outname is not None:
        plt.savefig('{0}_time.png'.format(outname))
    if not show:
        plt.close()
    return

def plot_uv_track(bu,bv,outname=None,show=True):
    """Plots the uv track provided
    Parameters:
    -----------
    bu      : real array, the values of u of the baselines in m,
              dimensions (baselines,time)
    bv      : real array, the values of v of the baselines in m,
              dimension (baselines, time)
    outname : string, optional the base to use for the name of the 
              png file the plot is saved to.  The plot will 
              be saved to <outname>_freq.png if an outname is
              provided, otherwise no plot will be saved.
    show    : boolean, optional, default True.  If true, the plot
              will be shown in an inline notebook.  If not using 
              a notebook, the plot will never be shown.
    Returns none
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
    Parameters:
    -----------
    arr : two-dimensional array
    nb1 : the number of bins to rebin by along the 0th axis
    nb2 : the number of bins to rebin by along the 1st axis
    Returns:
    -----------
    arr : the rebinned array
    """
    arr = arr[:arr.shape[0]//nb1*nb1,:arr.shape[1]//nb2*nb2].reshape(
        -1,nb1,arr.shape[1]).mean(1)
    arr = arr.reshape(arr.shape[0],-1,nb2).mean(-1)
    return arr

def plot_calibrated_vis(vis,vis_cal,mjd,fobs,bidx,
                        outname='None',show=True):
    """Plots two visibilities for comparison
    Parameters:
    -----------
    vis     : complex array, dimensions (time, freq), the uncalibrated
              visibility to plot
    vis_cal : complex array, dimensions (time, freq), the calibrated
              visibility to plot
    mjd     : array, the MJD of each subintegration
    fobs    : array, the frequency of each channel in GHz
    outname : string, optional the base to use for the name of the 
              png file the plot is saved to.  The plot will 
              be saved to <outname>_freq.png if an outname is
              provided, otherwise no plot will be saved.
    show    : boolean, optional, default True.  If true, the plot
              will be shown in an inline notebook.  If not using 
              a notebook, the plot will never be shown.
    Returns none
    """
    x = mjd[:mjd.shape[0]//128*128].reshape(-1,128).mean(-1)
    x = ((x-x[0])*u.d).to_value(u.min)
    y = fobs[:fobs.shape[0]//5*5].reshape(-1,5).mean(-1)
    
    fig,ax = plt.subplots(2,2,figsize=(16,16), sharex=True, sharey=True)
    
    vplot = rebin_vis(vis[bidx,...],128,5).T
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
        
    vplot = rebin_vis(vis_cal[bidx,...],128,5).T
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
        plt.savefig('{0}_cal_vis.png'.format(outname))
    if not show:
        plt.close()
    return


def plot_delays(vis_ft,labels,delay_arr,bname,outname='None',show=True):
    """Make amp vs delay plots for each visibility
    Parameters:
    -----------
    vis_ft  : complex array, the visibilities, dimensions 
              (vis, baselines, delay)
    labels  : list of strings, the labels for the visibilities passed
    delay_arr : array, the delay bins in nanoseconds
    bname   : list of baselines labels
    outname : string, optional the base to use for the name of the 
              png file the plot is saved to.  The plot will 
              be saved to <outname>_freq.png if an outname is
              provided, otherwise no plot will be saved.
    show    : boolean, optional, default True.  If true, the plot
              will be shown in an inline notebook.  If not using 
              a notebook, the plot will never be shown.
    Returns Nothing
    """
    nvis = vis_ft.shape[0]
    alpha = 0.5 if nvis>2 else 1
    fig,ax = plt.subplots(9,5,figsize=(8*5,8*9),sharex=True)#,sharey=True)
    ax = ax.flatten()
    for i in range(45):
        ax[i].axvline(0,color='red')
        for j in range(nvis):
            ax[i].plot(delay_arr, np.log10(np.abs(vis_ft[j,i,:])),
                      label=labels[j],alpha=alpha)
        ax[i].text(0.1,0.9,bname[i],transform=ax[i].transAxes,
                   size=22)
    plt.subplots_adjust(wspace=0.1,hspace=0.1)
    ax[0].legend()
    for i in range(45-5,45):
        ax[i].set_xlabel('delay (ns)')
    if outname is not None:
        plt.savefig('{0}_delays.png'.format(outname))
    if not show:
        plt.close()
    return

def plot_image(msname,imtype,sr0,verbose=False,outname=None,
               show=True,npix=256):
    """Uses CASA to grid an image and the plot the image.
    Parameters:
    ----------
    msname  : str, the prefix of the measurement set 
    imtype  : str, the type of image to produce.  Can be 'observed', 
              'corrected', or 'model'
    sr0     : src class, the source to use as the phase center in the image
    verbose : Boolean, whether to print information about the
              produced image, default False
    outname : string, optional the base to use for the name of the 
              png file the plot is saved to.  The plot will 
              be saved to <outname>_freq.png if an outname is
              provided, otherwise no plot will be saved.
    show    : boolean, optional, default True.  If true, the plot
              will be shown in an inline notebook.  If not using 
              a notebook, the plot will never be shown.
    npix    : int, the number of pixels per side to plot
    Returns Nothing
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
        plt.savefig('{0}_{1}.png'.format(msname,imtype))
    if not show:
        plt.close()
    if error > 0:
        print('{0} errors occured during imaging'.format(error))
    return