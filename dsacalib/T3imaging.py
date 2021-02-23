"""Creating and manipulating measurement sets from T3 visibilities.
"""
import os
import yaml
import numpy as np
from pkg_resources import resource_filename
import astropy.units as u
from casacore.tables import table, tablefromascii
import casatools as cc
from casatasks import virtualconcat
from antpos.utils import get_itrf
from dsacalib.dispersion import disperse, dedisperse
import dsacalib.utils as du
from dsacalib.ms_io import simulate_ms

# TODO: create a single DEDISPERSION table for the MMS 
# TODO: combine applied_dm with dispersion_measure to only read file once
def T3_dedisperse_ms(
    paramfile, msname, dispersion_measure, ref_freq=1.405*u.GHz
):
    """Dedisperses T3 measurement sets one correlator node at a time.

    Parameters
    ----------
    paramfile : str
        Full path to the T3corr parameter file.
    msname : str
        Name of the ms file. (Not full path, and omitting the `.ms` extension.)
    dispersion_measure : astropy quantity
        The dispersion measure to dedisperse to.
    ref_freq : astropy quantity
        Reference frequency relative to which the dispersion delay is
        evaluated.
    """
    yamlf = open(paramfile)
    params = yaml.load(yamlf, Loader=yaml.FullLoader)['T3corr']
    yamlf.close()
    for corr in params['ch0'].keys():
        corr_ms = '{0}/{1}.ms/SUBMSS/{1}_{2}'.format(
            params['msdir'],
            msname,
            corr
        )
        dispersion_table = '{0}/{1}.ms/SUBMSS/{1}_{2}.ms/DEDISPERSION'.format(
            params['msdir'],
            msname,
            corr
        )
        dm_table_exists = os.path.exists(dispersion_table)
        if dm_table_exists:
            with table(dispersion_table) as tb:
                applied_dm = np.array(tb.DISP_MEAS[:])[-1]*u.pc*u.cm**(-3)
                applied_reffreq = np.array(tb.REF_FREQ[:])[-1]*u.Hz
            disperse(corr_ms, applied_dm, ref_freq=applied_reffreq)
        dedisperse(corr_ms, dispersion_measure)
        if not dm_table_exists:
            dispfile = resource_filename(
                'dsacalib',
                'data/template_dispersion_table.txt'
            )
            with tablefromascii(
                dispersion_table,
                dispfile
            ) as tb:
                pass
        with table(dispersion_table, readonly=False) as tb:
            tb.putcol(
                'DISP_MEAS',
                np.array([dispersion_measure.to_value(u.pc*u.cm**(-3))])
            )
            tb.putcol(
                'REF_FREQ',
                np.array([ref_freq.to_value(u.Hz)])
            )

def T3_initialize_ms(
    paramfile, msname, tstart, sourcename, ra, dec, ntint, nfint
):
    """Initialize a ms to write correlated data from the T3 system to.

    Parameters
    ----------
    paramfile : str
        The full path to a yaml parameter file. See package data for a
        template.
    msname : str
        The name of the measurement set. Will write to `msdir`/`msname`.ms.
        `msdir` is defined in `paramfile`.
    tstart : astropy.time.Time object
        The start time of the observation.
    sourcename : str
        The name of the source or field.
    ra : astropy quantity
        The right ascension of the pointing, units deg or equivalent.
    dec : astropy quantity
        The declination of the pointing, units deg or equivalent.
    """
    yamlf = open(paramfile)
    params = yaml.load(yamlf, Loader=yaml.FullLoader)['T3corr']
    yamlf.close()
    source = du.src(
        name=sourcename,
        ra=ra,
        dec=dec
    )
    ant_itrf = get_itrf().loc[params['antennas']]
    xx = ant_itrf['dx_m']
    yy = ant_itrf['dy_m']
    zz = ant_itrf['dz_m']
    antenna_names = [str(a) for a in params['antennas']]
    fobs = params['f0_GHz']+params['deltaf_MHz']*1e-3*nfint*(
        np.arange(params['nchan']//nfint)+0.5)
    me = cc.measures()
    filenames = []
    for corr, ch0 in params['ch0'].items():
        fobs_corr = fobs[ch0//nfint:(ch0+params['nchan_corr'])//nfint]
        simulate_ms(
            ofile='{0}/{1}_{2}.ms'.format(params['msdir'], msname, corr),
            tname='OVRO_MMA',
            anum=antenna_names,
            xx=xx,
            yy=yy,
            zz=zz,
            diam=4.5,
            mount='alt-az',
            pos_obs=me.observatory('OVRO_MMA'),
            spwname='L_BAND',
            freq='{0}GHz'.format(fobs_corr[0]),
            deltafreq='{0}MHz'.format(params['deltaf_MHz']*nfint),
            freqresolution='{0}MHz'.format(
                np.abs(params['deltaf_MHz']*nfint)
            ),
            nchannels=params['nchan_corr']//nfint,
            integrationtime='{0}s'.format(params['deltat_s']*ntint),
            obstm=tstart.mjd,
            dt=0.0004282407317077741,
            source=source,
            stoptime='{0}s'.format(params['deltat_s']*params['nsubint']),
            autocorr=True,
            fullpol=True
        )
        filenames += ['{0}/{1}_{2}.ms'.format(params['msdir'], msname, corr)]
    virtualconcat(
        filenames,
        '{0}/{1}.ms'.format(params['msdir'], msname)
    )

def write_T3data_to_ms(msname, datapaths, msdir):
    """Copies data from the T3 cpucorrelator to a measurement set.

    Parameters
    ----------
    msname : str
        The name used to title the ms.
    datapaths : dictionary
        The path to the data file for each correlator node. e.g.
        {'corr01': '/mnt/data/dsa110/cpucorr/corr01.dat',
         'corr02': '/mnt/data/dsa110/cpucorr/corr02.dat'}
        Note that correlators that do not have a specified data file will not
        be overwritten or flagged, and the simulated data will contaminate that
        band.
    msdir : str
        The path to the measurement set. Will modify `msdir`/`msname`.ms.
    
    """
    for corr, datapath in datapaths.items():
        shutil.copyfile(
            datapath,
            '{0}/{1}.ms/SUBMSS/{1}_{2}.ms/table.f2_TSM1'.format(
                msdir,
                msname,
                corr
            )
        )