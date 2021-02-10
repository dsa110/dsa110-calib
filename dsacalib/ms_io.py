"""
Dsacalib/MS_IO.PY

Dana Simard, dana.simard@astro.caltech.edu, 10/2019

Routines to interact with CASA measurement sets and calibration tables.
"""

# To do:
# Replace to_deg w/ astropy versions

# Always import scipy before importing casatools.
import shutil
import os
import glob
#from scipy.interpolate import interp1d
import numpy as np
#from pkg_resources import resource_filename
import yaml
import scipy # pylint: disable=unused-import
import astropy.units as u
import astropy.constants as c
import casatools as cc
from casatasks import importuvfits, virtualconcat
from casacore.tables import addImagingColumns, table
from pyuvdata import UVData
from dsautils import dsa_store
from dsautils import calstatus as cs
import dsautils.cnf as dsc
from dsamfs.fringestopping import calc_uvw_blt
from dsacalib import constants as ct
import dsacalib.utils as du
from dsacalib.fringestopping import calc_uvw, amplitude_sky_model
from antpos.utils import get_itrf # pylint: disable=wrong-import-order
from astropy.utils import iers # pylint: disable=wrong-import-order
iers.conf.iers_auto_url_mirror = ct.IERS_TABLE
iers.conf.auto_max_age = None
from astropy.time import Time # pylint: disable=wrong-import-position wrong-import-order

de = dsa_store.DsaStore()

CONF = dsc.Conf()
CORR_PARAMS = CONF.get('corr')

def T3_initialize_ms(paramfile, msname, tstart, sourcename, ra, dec, ntint, nfint):
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

def simulate_ms(ofile, tname, anum, xx, yy, zz, diam, mount, pos_obs, spwname,
                freq, deltafreq, freqresolution, nchannels, integrationtime,
                obstm, dt, source, stoptime, autocorr, fullpol):
    """Simulates a measurement set with cross-correlations only.

    WARNING: Not simulating autocorrelations correctly regardless of inclusion
    of autocorr parameter.

    Parameters
    ----------
    ofile : str
        The full path to which the measurement set will be written.
    tname : str
        The telescope name.
    xx, yy, zz : arrays
        The X, Y and Z ITRF coordinates of the antennas, in meters.
    diam : float
        The dish diameter in meters.
    mount : str
        The mount type, e.g. 'alt-az'.
    pos_obs : CASA measure instance
        The location of the observatory.
    spwname : str
        The name of the spectral window, e.g. 'L-band'.
    freq : str
        The central frequency, as a CASA-recognized string, e.g. '1.4GHz'.
    deltafreq : str
        The size of each channel, as a CASA-recognized string, e.g. '1.24kHz'.
    freqresolution : str
        The frequency resolution, as a CASA-recognized string, e.g. '1.24kHz'.
    nchannels : int
        The number of frequency channels.
    integrationtime : str
        The subintegration time, i.e. the width of each time bin, e.g. '1.4s'.
    obstm : float
        The start time of the observation in MJD.
    dt : float
        The offset between the CASA start time and the true start time in days.
    source : dsacalib.utils.source instance
        The source observed (or the phase-center).
    stoptime : float
        The end time of the observation in MJD. DS: should be s?
    autocorr : boolean
        Set to ``True`` if the visibilities include autocorrelations, ``False``
        if the only include crosscorrelations.
    """
    me = cc.measures()
    qa = cc.quanta()
    sm = cc.simulator()
    sm.open(ofile)
    sm.setconfig(
        telescopename=tname,
        x=xx,
        y=yy,
        z=zz,
        dishdiameter=diam,
        mount=mount,
        antname=anum,
        coordsystem='global',
        referencelocation=pos_obs
    )
    sm.setspwindow(
        spwname=spwname,
        freq=freq,
        deltafreq=deltafreq,
        freqresolution=freqresolution,
        nchannels=nchannels,
        stokes='XX YY XY YX' if fullpol else 'XX YY'
    )
    sm.settimes(
        integrationtime=integrationtime,
        usehourangle=False,
        referencetime=me.epoch('utc', qa.quantity(obstm-dt, 'd'))
    )
    sm.setfield(
        sourcename=source.name,
        sourcedirection=me.direction(
            source.epoch,
            qa.quantity(source.ra.to_value(u.rad), 'rad'),
            qa.quantity(source.dec.to_value(u.rad), 'rad')
        )
    )
    sm.setauto(autocorrwt=1.0 if autocorr else 0.0)
    sm.observe(source.name, spwname, starttime='0s', stoptime=stoptime)
    sm.close()

def convert_to_ms(source, vis, obstm, ofile, bname, antenna_order,
                  tsamp=ct.TSAMP*ct.NINT, nint=1, antpos=None, model=None,
                  dt=ct.CASA_TIME_OFFSET, dsa10=True):
    """ Writes visibilities to an ms.

    Uses the casa simulator tool to write the metadata to an ms, then uses the
    casa ms tool to replace the visibilities with the observed data.

    Parameters
    ----------
    source : source class instance
        The calibrator (or position) used for fringestopping.
    vis : ndarray
        The complex visibilities, dimensions (baseline, time, channel,
        polarization).
    obstm : float
        The start time of the observation in MJD.
    ofile : str
        The name for the created ms.  Writes to `ofile`.ms.
    bname : list
        The list of baselines names in the form [[ant1, ant2],...].
    antenna_order: list
        The list of the antennas, in CASA ordering.
    tsamp : float
        The sampling time of the input visibilities in seconds.  Defaults to
        the value `tsamp`*`nint` as defined in `dsacalib.constants`.
    nint : int
        The number of time bins to integrate by before saving to a measurement
        set.  Defaults 1.
    antpos : str
        The full path to the text file containing ITRF antenna positions or the
        csv file containing the station positions in longitude and latitude.
        Defaults `dsacalib.constants.PKG_DATA_PATH`/antpos_ITRF.txt.
    model : ndarray
        The visibility model to write to the measurement set (and against which
        gain calibration will be done). Must have the same shape as the
        visibilities `vis`. If given a value of ``None``, an array of ones will
        be used as the model.  Defaults ``None``.
    dt : float
        The offset between the CASA start time and the data start time in days.
        Defaults to the value of `casa_time_offset` in `dsacalib.constants`.
    dsa10 : boolean
        Set to ``True`` if the data are from the dsa10 correlator.  Defaults
        ``True``.
    """
    if antpos is None:
        antpos = '{0}/antpos_ITRF.txt'.format(ct.PKG_DATA_PATH)
    vis = vis.astype(np.complex128)
    if model is not None:
        model = model.astype(np.complex128)

    nant = len(antenna_order)

    me = cc.measures()

    # Observatory parameters
    tname = 'OVRO_MMA'
    diam = 4.5 # m
    obs = 'OVRO_MMA'
    mount = 'alt-az'
    pos_obs = me.observatory(obs)

    # Backend
    if dsa10:
        spwname = 'L_BAND'
        freq = '1.4871533196875GHz'
        deltafreq = '-0.244140625MHz'
        freqresolution = deltafreq
    else:
        spwname = 'L_BAND'
        freq = '1.28GHz'
        deltafreq = '40.6901041666667kHz'
        freqresolution = deltafreq
    (_, _, nchannels, npol) = vis.shape

    # Rebin visibilities
    integrationtime = '{0}s'.format(tsamp*nint)
    if nint != 1:
        npad = nint-vis.shape[1]%nint
        if npad == nint:
            npad = 0
        vis = np.nanmean(np.pad(vis, ((0, 0), (0, npad), (0, 0), (0, 0)),
                                mode='constant',
                                constant_values=(np.nan, )).reshape(
                                    vis.shape[0], -1, nint, vis.shape[2],
                                    vis.shape[3]), axis=2)
        if model is not None:
            model = np.nanmean(np.pad(model,
                                      ((0, 0), (0, npad), (0, 0), (0, 0)),
                                      mode='constant',
                                      constant_values=(np.nan, )).reshape(
                                          model.shape[0], -1, nint,
                                          model.shape[2], model.shape[3]),
                               axis=2)
    stoptime = '{0}s'.format(vis.shape[1]*tsamp*nint)

    anum, xx, yy, zz = du.get_antpos_itrf(antpos)
    # Sort the antenna positions
    idx_order = sorted([int(a)-1 for a in antenna_order])
    anum = np.array(anum)[idx_order]
    xx = np.array(xx)
    yy = np.array(yy)
    zz = np.array(zz)
    xx = xx[idx_order]
    yy = yy[idx_order]
    zz = zz[idx_order]

    nints = np.zeros(nant, dtype=int)
    for i, an in enumerate(anum):
        nints[i] = np.sum(np.array(bname)[:, 0] == an)
    nints, anum, xx, yy, zz = zip(*sorted(zip(nints, anum, xx, yy, zz),
                                          reverse=True))

    # Check that the visibilities are ordered correctly by checking the order
    # of baselines in bname
    idx_order = []
    autocorr = bname[0][0] == bname[0][1]

    for i in range(nant):
        for j in range(i if autocorr else i+1, nant):
            idx_order += [bname.index([anum[i], anum[j]])]
    assert idx_order == list(np.arange(len(bname), dtype=int)), \
        'Visibilities not ordered by baseline'
    anum = [str(a) for a in anum]

    simulate_ms(
        '{0}.ms'.format(ofile), tname, anum, xx, yy, zz, diam, mount, pos_obs,
        spwname, freq, deltafreq, freqresolution, nchannels, integrationtime,
        obstm, dt, source, stoptime, autocorr, fullpol=False
    )

    # Check that the time is correct
    ms = cc.ms()
    ms.open('{0}.ms'.format(ofile))
    tstart_ms = ms.summary()['BeginTime']
    ms.close()

    print('autocorr :', autocorr)

    if np.abs(tstart_ms-obstm) > 1e-10:
        dt = dt+(tstart_ms-obstm)
        print('Updating casa time offset to {0}s'.format(
            dt*ct.SECONDS_PER_DAY))
        print('Rerunning simulator')
        simulate_ms(
            '{0}.ms'.format(ofile), tname, anum, xx, yy, zz, diam, mount,
            pos_obs, spwname, freq, deltafreq, freqresolution, nchannels,
            integrationtime, obstm, dt, source, stoptime, autocorr,
            fullpol=False
        )

    # Reopen the measurement set and write the observed visibilities
    ms = cc.ms()
    ms.open('{0}.ms'.format(ofile), nomodify=False)
    ms.selectinit(datadescid=0)

    rec = ms.getdata(["data"])
    # rec['data'] has shape [scan, channel, [time*baseline]]
    vis = vis.T.reshape((npol, nchannels, -1))
    rec['data'] = vis
    ms.putdata(rec)
    ms.close()

    ms = cc.ms()
    ms.open('{0}.ms'.format(ofile), nomodify=False)
    if model is None:
        model = np.ones(vis.shape, dtype=complex)
    else:
        model = model.T.reshape((npol, nchannels, -1))
    rec = ms.getdata(["model_data"])
    rec['model_data'] = model
    ms.putdata(rec)
    ms.close()

    # Check that the time is correct
    ms = cc.ms()
    ms.open('{0}.ms'.format(ofile))
    tstart_ms = ms.summary()['BeginTime']
    tstart_ms2 = ms.getdata('TIME')['time'][0]/ct.SECONDS_PER_DAY
    ms.close()

    assert np.abs(tstart_ms-(tstart_ms2-tsamp*nint/ct.SECONDS_PER_DAY/2)) \
        < 1e-10, 'Data start time does not agree with MS start time'

    assert np.abs(tstart_ms - obstm) < 1e-10, \
        'Measurement set start time does not agree with input tstart'
    print('Visibilities writing to ms {0}.ms'.format(ofile))

def extract_vis_from_ms(msname, data='data'):
    """Extracts visibilities from a CASA measurement set.

    Parameters
    ----------
    msname : str
        The measurement set. Opens `msname`.ms
    data : str
        The visibilities to extract. Can be `data`, `model` or `corrected`.

    Returns
    -------
    vals : ndarray
        The visibilities, dimensions (baseline, time, spw, freq, pol).
    time : array
        The time of each integration in days.
    fobs : array
        The frequency of each channel in GHz.
    flags : ndarray
        Flags for the visibilities, same shape as vals. True if flagged.
    ant1, ant2 : array
        The antenna indices for each baselines in the visibilities.
    pt_dec : float
        The pointing declination of the array. (Note: Not the phase center, but
        the physical pointing of the antennas.)
    """
    with table('{0}.ms'.format(msname)) as tb:
        ant1 = np.array(tb.ANTENNA1[:])
        ant2 = np.array(tb.ANTENNA2[:])
        vals = np.array(tb.getcol(data.upper())[:])
        flags = np.array(tb.FLAG[:])
        time = np.array(tb.TIME[:])
        spw = np.array(tb.DATA_DESC_ID[:])
    with table('{0}.ms/SPECTRAL_WINDOW'.format(msname)) as tb:
        fobs = (np.array(tb.col('CHAN_FREQ')[:])/1e9).reshape(-1)

    baseline = 2048*(ant1+1)+(ant2+1)+2**16

    time, vals, flags, ant1, ant2 = reshape_calibration_data(
        vals, flags, ant1, ant2, baseline, time, spw)

    with table('{0}.ms/FIELD'.format(msname)) as tb:
        pt_dec = tb.PHASE_DIR[:][0][0][1]

    return vals, time/ct.SECONDS_PER_DAY, fobs, flags, ant1, ant2, pt_dec

def read_caltable(tablename, cparam=False):
    """Requires that each spw has the same number of frequency channels.

    Parameters
    ----------
    tablename : str
        The full path to the calibration table.
    cparam : bool
        If True, reads the column CPARAM in the calibrtion table. Otherwise
        reads FPARAM.

    Returns
    -------
    vals : ndarray
        The visibilities, dimensions (baseline, time, spw, freq, pol).
    time : array
        The time of each integration in days.
    flags : ndarray
        Flags for the visibilities, same shape as vals. True if flagged.
    ant1, ant2 : array
        The antenna indices for each baselines in the visibilities.
    """
    with table(tablename) as tb:
        try:
            spw = np.array(tb.SPECTRAL_WINDOW_ID[:])
        except AttributeError:
            spw = np.array([0])
        time = np.array(tb.TIME[:])
        if cparam:
            vals = np.array(tb.CPARAM[:])
        else:
            vals = np.array(tb.FPARAM[:])
        flags = np.array(tb.FLAG[:])
        ant1 = np.array(tb.ANTENNA1[:])
        ant2 = np.array(tb.ANTENNA2[:])
    baseline = 2048*(ant1+1)+(ant2+1)+2**16

    time, vals, flags, ant1, ant2 = reshape_calibration_data(
        vals, flags, ant1, ant2, baseline, time, spw)

    return vals, time/ct.SECONDS_PER_DAY, flags, ant1, ant2

def reshape_calibration_data(vals, flags, ant1, ant2, baseline, time, spw):
    """Reshape calibration or measurement set data.

    Reshapes the 0th axis of the input data `vals` and `flags` from a
    combined (baseline-time-spw) axis into 3 axes (baseline, time, spw).

    Parameters
    ----------
    vals : ndarray
        The input values, shape (baseline-time-spw, freq, pol).
    flags : ndarray
        Flag array, same shape as vals.
    ant1, ant2 : array
        The antennas in the baseline, same length as the 0th axis of `vals`.
    baseline : array
        The baseline index, same length as the 0th axis of `vals`.
    time : array
        The time of each integration, same length as the 0th axis of `vals`.
    spw : array
        The spectral window index of each integration, same length as the 0th
        axis of `vals`.

    Returns
    -------
    time : array
        Unique times, same length as the time axis of the output `vals`.
    vals, flags : ndarray
        The reshaped input arrays, dimensions (baseline, time, spw, freq, pol)
    ant1, ant2 : array
        ant1 and ant2 for unique baselines, same length as the baseline axis of
        the output `vals`.
    """
    if len(np.unique(ant1))==len(np.unique(ant2)):
        nbl = len(np.unique(baseline))
    else:
        nbl = max([len(np.unique(ant1)), len(np.unique(ant2))])
    nspw = len(np.unique(spw))
    ntime = len(time)//nbl//nspw
    nfreq = vals.shape[-2]
    npol = vals.shape[-1]
    if np.all(baseline[:ntime*nspw] == baseline[0]):
        if np.all(time[:nspw] == time[0]):
            # baseline, time, spw
            time = time.reshape(nbl, ntime, nspw)[0, :, 0]
            vals = vals.reshape(nbl, ntime, nspw, nfreq, npol)
            flags = flags.reshape(nbl, ntime, nspw, nfreq, npol)
            ant1 = ant1.reshape(nbl, ntime, nspw)[:, 0, 0]
            ant2 = ant2.reshape(nbl, ntime, nspw)[:, 0, 0]
        else:
            # baseline, spw, time
            assert np.all(spw[:ntime] == spw[0])
            time = time.reshape(nbl, nspw, ntime)[0, 0, :]
            vals = vals.reshape(nbl, nspw, ntime, nfreq, npol).swapaxes(1, 2)
            flags = flags.reshape(nbl, nspw, ntime, nfreq, npol).swapaxes(1, 2)
            ant1 = ant1.reshape(nbl, nspw, ntime)[:, 0, 0]
            ant2 = ant2.reshape(nbl, nspw, ntime)[:, 0, 0]
    elif np.all(time[:nspw*nbl] == time[0]):
        if np.all(baseline[:nspw] == baseline[0]):
            # time, baseline, spw
            time = time.reshape(ntime, nbl, nspw)[:, 0, 0]
            vals = vals.reshape(ntime, nbl, nspw, nfreq, npol).swapaxes(0, 1)
            flags = flags.reshape(ntime, nbl, nspw, nfreq, npol).swapaxes(0, 1)
            ant1 = ant1.reshape(ntime, nbl, nspw)[0, :, 0]
            ant2 = ant2.reshape(ntime, nbl, nspw)[0, :, 0]
        else:
            assert np.all(spw[:nbl] == spw[0])
            time = time.reshape(ntime, nspw, nbl)[:, 0, 0]
            vals = vals.reshape(ntime, nspw, nbl, nfreq, npol).swapaxes(
                1, 2).swapaxes(0, 1)
            flags = flags.reshape(ntime, nspw, nbl, nfreq, npol).swapaxes(
                1, 2).swapaxes(0, 1)
            ant1 = ant1.reshape(ntime, nspw, nbl)[0, 0, :]
            ant2 = ant2.reshape(ntime, nspw, nbl)[0, 0, :]
    else:
        assert np.all(spw[:nbl*ntime] == spw[0])
        if np.all(baseline[:ntime] == baseline[0]):
            # spw, baseline, time
            time = time.reshape(nspw, nbl, ntime)[0, 0, :]
            vals = vals.reshape(nspw, nbl, ntime, nfreq, npol).swapaxes(
                0, 1).swapaxes(1, 2)
            flags = flags.reshape(nspw, nbl, ntime, nfreq, npol).swapaxes(
                0, 1).swapaxes(1, 2)
            ant1 = ant1.reshape(nspw, nbl, ntime)[0, :, 0]
            ant2 = ant2.reshape(nspw, nbl, ntime)[0, :, 0]
        else:
            assert np.all(time[:nbl] == time[0])
            # spw, time, bl
            time = time.reshape(nspw, ntime, nbl)[0, :, 0]
            vals = vals.reshape(nspw, ntime, nbl, nfreq, npol).swapaxes(0, 2)
            flags = flags.reshape(nspw, ntime, nbl, nfreq, npol).swapaxes(0, 2)
            ant1 = ant1.reshape(nspw, ntime, nbl)[0, 0, :]
            ant2 = ant2.reshape(nspw, ntime, nbl)[0, 0, :]
    return time, vals, flags, ant1, ant2

def caltable_to_etcd(
    msname, calname, caltime, status, pols=None, logger=None
):
    r"""Copies calibration values from delay and gain tables to etcd.

    The dictionary passed to etcd should look like: {"ant_num": <i>,
    "time": <d>, "pol", [<s>, <s>], "gainamp": [<d>, <d>],
    "gainphase": [<d>, <d>], "delay": [<i>, <i>], "calsource": <s>,
    "gaincaltime_offset": <d>, "delaycaltime_offset": <d>, 'sim': <b>,
    'status': <i>}

    Parameters
    ----------
    msname : str
        The measurement set name, will use solutions created from the
        measurement set `msname`.ms.
    calname : str
        The calibrator name.  Will open the calibration tables
        `msname`\_`calname`\_kcal and `msname`\_`calname`\_gcal_ant.
    caltime : float
        The time of calibration transit in mjd.
    status : int
        The status of the calibration. Decode with dsautils.calstatus.
    pols : list
        The names of the polarizations. If ``None``, will be set to
        ``['B', 'A']``. Defaults ``None``.
    logger : dsautils.dsa_syslog.DsaSyslogger() instance
        Logger to write messages too. If None, messages are printed.
    """
    if pols is None:
        pols = ['B', 'A']

    try:
        # Complex gains for each antenna.
        amps, tamp, flags, ant1, ant2 = read_caltable(
            '{0}_{1}_gacal'.format(msname, calname),
            cparam=True
        )
        mask = np.ones(flags.shape)
        mask[flags == 1] = np.nan
        amps = amps*mask

        phase, _tphase, flags, ant1, ant2 = read_caltable(
            '{0}_{1}_gpcal'.format(msname, calname),
            cparam=True
        )
        mask = np.ones(flags.shape)
        mask[flags == 1] = np.nan
        phase = phase*mask

        if np.all(ant2 == ant2[0]):
            antenna_order_amps = ant1
        if not np.all(ant2 == ant2[0]):
            idxs = np.where(ant1 == ant2)[0]
            tamp = tamp[idxs]
            amps = amps[idxs, ...]
            antenna_order_amps = ant1[idxs]

        # Check the output shapes.
        print(tamp.shape, amps.shape)
        assert amps.shape[0] == len(antenna_order_amps)
        assert amps.shape[1] == tamp.shape[0]
        assert amps.shape[2] == 1
        assert amps.shape[3] == 1
        assert amps.shape[4] == len(pols)

        amps = np.nanmedian(
            amps.squeeze(axis=2).squeeze(axis=2),
            axis=1
        )*np.nanmedian(
            phase.squeeze(axis=2).squeeze(axis=2),
            axis=1
        )
        tamp = np.median(tamp)
        gaincaltime_offset = (tamp-caltime)*ct.SECONDS_PER_DAY

    except Exception as exc:
        tamp = np.nan
        amps = np.ones((0, len(pols)))*np.nan
        gaincaltime_offset = 0.
        antenna_order_amps = np.zeros(0, dtype=np.int)
        status = cs.update(
            status,
            cs.GAIN_TBL_ERR |
            cs.INV_GAINAMP_P1 |
            cs.INV_GAINAMP_P2 |
            cs.INV_GAINPHASE_P1 |
            cs.INV_GAINPHASE_P2 |
            cs.INV_GAINCALTIME
        )
        du.exception_logger(logger, 'caltable_to_etcd', exc, throw=False)

    # Delays for each antenna.
    try:
        delays, tdel, flags, antenna_order_delays, ant2 = read_caltable(
            '{0}_{1}_kcal'.format(msname, calname), cparam=False)
        mask = np.ones(flags.shape)
        mask[flags == 1] = np.nan
        delays = delays*mask

        # Check the output shapes.
        assert delays.shape[0] == len(antenna_order_delays)
        assert delays.shape[1] == tdel.shape[0]
        assert delays.shape[2] == 1
        assert delays.shape[3] == 1
        assert delays.shape[4] == len(pols)

        delays = np.nanmedian(delays.squeeze(axis=2).squeeze(axis=2), axis=1)
        tdel = np.median(tdel)
        delaycaltime_offset = (tdel-caltime)*ct.SECONDS_PER_DAY

    except Exception as exc:
        tdel = np.nan
        delays = np.ones((0, len(pols)))*np.nan
        delaycaltime_offset = 0.
        status = cs.update(
            status,
            cs.DELAY_TBL_ERR |
            cs.INV_DELAY_P1 |
            cs.INV_DELAY_P2 |
            cs.INV_DELAYCALTIME
        )
        antenna_order_delays = np.zeros(0, dtype=np.int)
        du.exception_logger(logger, 'caltable_to_etcd', exc, throw=False)

    antenna_order = np.unique(
        np.array([antenna_order_amps,
                  antenna_order_delays])
    )

    for antnum in antenna_order:

        # Everything needs to be cast properly.
        gainamp = []
        gainphase = []
        ant_delay = []

        if antnum in antenna_order_amps:
            idx = np.where(antenna_order_amps == antnum)[0][0]
            for amp in amps[idx, :]:
                if not np.isnan(amp):
                    gainamp += [float(np.abs(amp))]
                    gainphase += [float(np.angle(amp))]
                else:
                    gainamp += [None]
                    gainphase += [None]
        else:
            gainamp = [None]*len(pols)
            gainphase = [None]*len(pols)

        if antnum in antenna_order_delays:
            idx = np.where(antenna_order_delays == antnum)[0][0]
            for delay in delays[idx, :]:
                if not np.isnan(delay):
                    ant_delay += [int(np.rint(delay))]
                else:
                    ant_delay += [None]
        else:
            ant_delay = [None]*len(pols)

        dd = {
            'ant_num': int(antnum+1),
            'time': float(caltime),
            'pol': pols,
            'gainamp': gainamp,
            'gainphase': gainphase,
            'delay': ant_delay,
            'calsource': calname,
            'gaincaltime_offset': float(gaincaltime_offset),
            'delaycaltime_offset': float(delaycaltime_offset),
            'sim': False,
            'status':status
        }
        required_keys = dict({
            'ant_num': [cs.INV_ANTNUM, 0],
            'time': [cs.INV_DELAYCALTIME, 0.],
            'pol': [cs.INV_POL, ['B', 'A']],
            'calsource': [cs.INV_CALSOURCE, 'Unknown'],
            'sim': [cs.INV_SIM, False],
            'status': [cs.UNKNOWN_ERR, 0]
        })
        for key, value in required_keys.items():
            if dd[key] is None:
                print('caltable_to_etcd: key {0} must not be None to write to '
                      'etcd'.format(key))
                status = cs.update(status, value[0])
                dd[key] = value[1]

        for pol in dd['pol']:
            if pol is None:
                print('caltable_to_etcd: pol must not be None to write to '
                      'etcd')
                status = cs.update(status, cs.INV_POL)
                dd['pol'] = ['B', 'A']
        de.put_dict('/mon/cal/{0}'.format(antnum+1), dd)

def get_antenna_gains(gains, ant1, ant2, refant=0):
    """Calculates antenna gains, g_i, from CASA table of G_ij=g_i g_j*.

    Currently does not support baseline-based gains.
    Refant only used for baseline-based case.

    Parameters
    ----------
    gains : ndarray
        The gains read in from the CASA gain table. 0th axis is baseline or
        antenna.
    ant1, ant2 : ndarray
        The antenna pair for each entry along the 0th axis in gains.
    refant : int
        The reference antenna index to use to get antenna gains from baseline
        gains.

    Returns
    -------
    antennas : ndarray
        The antenna indices.
    antenna_gains : ndarray
        Gains for each antenna in `antennas`.
    """
    antennas = np.unique(np.concatenate((ant1, ant2)))
    output_shape = list(gains.shape)
    output_shape[0] = len(antennas)
    antenna_gains = np.zeros(tuple(output_shape), dtype=gains.dtype)
    if np.all(ant2 == ant2[0]):
        for i, ant in enumerate(antennas):
            antenna_gains[i] = 1/gains[ant1==ant]
    else:
        assert len(antennas) == 3, ("Baseline-based only supported for trio of"
                                    "antennas")
        for i, ant in enumerate(antennas):
            ant1idxs = np.where(ant1==ant)[0]
            ant2idxs = np.where(ant2==ant)[0]
            otheridx = np.where((ant1!=ant) & (ant2!=ant))[0][0]
            # phase
            sign = 1
            idx_phase = np.where((ant1==ant) & (ant2==refant))[0]
            if len(idx_phase) == 0:
                idx_phase = np.where((ant2==refant) & (ant1==ant))[0]
                assert len(idx_phase) == 1
                sign = -1
            # amplitude
            if len(ant1idxs) == 2:
                g01 = gains[ant1idxs[0]]
                g20 = np.conjugate(gains[ant1idxs[1]])
                if ant1[otheridx] == ant2[ant1idxs[1]]:
                    g21 = gains[otheridx]
                else:
                    g21 = np.conjugate(gains[otheridx])
            if len(ant1idxs) == 1:
                g01 = gains[ant1idxs[0]]
                g20 = gains[ant2idxs[0]]
                if ant1[otheridx] == ant1[ant2idxs[0]]:
                    g21 = gains[otheridx]
                else:
                    g21 = np.conjugate(gains[otheridx])
            else:
                g01 = np.conjugate(gains[ant2idxs[0]])
                g20 = gains[ant2idxs[1]]
                if ant1[otheridx] == ant1[ant2idxs[1]]:
                    g21 = gains[otheridx]
                else:
                    g21 = np.conjugate(gains[otheridx])
            antenna_gains[i] = (np.sqrt(np.abs(g01*g20/g21))*np.exp(
                sign*1.0j*np.angle(gains[idx_phase])))**(-1)
    return antennas, antenna_gains

def write_beamformer_weights(msname, calname, caltime, antennas, outdir,
                             corr_list, antenna_flags, tol=0.3):
    """Writes weights for the beamformer.

    Parameters
    ----------
    msname : str
        The prefix of the measurement set.  Will open `msname`.ms
    calname : str
        The name of the calibrator source.
    antennas : list
        The names of the antennas to extract solutions for.  Order must be the
        same as the order in the beamformer.
    outdir : str
        The directory to write the beamformer weights in.
    corr_list : list
        The indices of the correlator machines to write beamformer weights for.
        For now, these must be ordered so that the frequencies are contiguous
        and they are in the same order or the reverse order as in the ms. The
        bandwidth of each correlator is pulled from dsa110-meridian-fs package
        data.
    antenna_flags : ndarray(bool)
        Dimensions (antennas, pols). True where flagged, False otherwise.
    tol : float
        The fraction of data for a single antenna/pol flagged that the
        can be flagged in the beamformer. If more data than this is flagged as
        having bad solutions, the entire antenna/pol pair is flagged.

    Returns
    -------
    corr_list : list
    bu : array
        The length of the baselines in the u direction for each antenna
        relative to antenna 24.
    fweights : ndarray
        The frequencies corresponding to the beamformer weights, dimensions
        (correlator, frequency).
    filenames : list
        The names of the file containing the beamformer weights.
    """
    # Get the frequencies we want to write solutions for.
    # corr_settings = resource_filename("dsamfs", "data/dsa_parameters.yaml")
    # params = yaml.safe_load(fhand)
    ncorr = len(corr_list)
    weights = np.ones((ncorr, len(antennas), 48, 2), dtype=np.complex64)
    fweights = np.ones((ncorr, 48), dtype=np.float32)
    nchan = CORR_PARAMS['nchan']
    dfreq = CORR_PARAMS['bw_GHz']/nchan
    if CORR_PARAMS['chan_ascending']:
        fobs = CORR_PARAMS['f0_GHz']+np.arange(nchan)*dfreq
    else:
        fobs = CORR_PARAMS['f0_GHz']-np.arange(nchan)*dfreq
    nchan_spw = CORR_PARAMS['nchan_spw']
    for i, corr_id in enumerate(corr_list):
        ch0 = CORR_PARAMS['ch0']['corr{0:02d}'.format(corr_id)]
        fobs_corr = fobs[ch0:ch0+nchan_spw]
        fweights[i, :] = fobs_corr.reshape(
            fweights.shape[1],
            -1
        ).mean(axis=1)

    antpos_df = get_itrf()
    blen = np.zeros((len(antennas), 3))
    for i, ant in enumerate(antennas):
        blen[i, 0] = antpos_df['x_m'].loc[ant]-antpos_df['x_m'].loc[24]
        blen[i, 1] = antpos_df['y_m'].loc[ant]-antpos_df['y_m'].loc[24]
        blen[i, 2] = antpos_df['z_m'].loc[ant]-antpos_df['z_m'].loc[24]
    bu, _, _ = calc_uvw(blen, 59000., 'HADEC', 0.*u.rad, 0.6*u.rad)
    bu = bu.squeeze().astype(np.float32)

    with table('{0}.ms/SPECTRAL_WINDOW'.format(msname)) as tb:
        fobs = np.array(tb.CHAN_FREQ[:])/1e9
    fobs = fobs.reshape(fweights.size, -1).mean(axis=1)
    f_reversed = not np.all(
        np.abs(fobs-fweights.ravel())/fweights.ravel() < 1e-5
    )
    if f_reversed:
        assert np.all(
            np.abs(fobs[::-1]-fweights.ravel())/fweights.ravel() < 1e-5
        )

    gains, _time, flags, ant1, ant2 = read_caltable(
        '{0}_{1}_gacal'.format(msname, calname), True)
    gains[flags] = np.nan
    gains = np.nanmean(gains, axis=1)
    phases, _, flags, ant1p, ant2p = read_caltable(
        '{0}_{1}_gpcal'.format(msname, calname), True)
    phases[flags] = np.nan
    phases = np.nanmean(phases, axis=1)
    assert np.all(ant1p == ant1)
    assert np.all(ant2p == ant2)
    gantenna, gains = get_antenna_gains(gains*phases, ant1, ant2)

    bgains, _, flags, ant1, ant2 = read_caltable(
        '{0}_{1}_bcal'.format(msname, calname), True)
    bgains[flags] = np.nan
    bgains = np.nanmean(bgains, axis=1)
    bantenna, bgains = get_antenna_gains(bgains, ant1, ant2)
    assert np.all(bantenna == gantenna)

    nantenna = gains.shape[0]
    npol = gains.shape[-1]

    gains = gains*bgains
    print(gains.shape)
    gains = gains.reshape(nantenna, -1, npol)
    if f_reversed:
        gains = gains[:, ::-1, :]
    gains = gains.reshape(nantenna, ncorr, -1, npol)
    nfint = gains.shape[2]//weights.shape[2]
    assert gains.shape[2]%weights.shape[2]==0

    gains = np.nanmean(
        gains.reshape(
            gains.shape[0], gains.shape[1], -1, nfint, gains.shape[3]
        ), axis=3
    )
    if not np.all(ant2==ant2[0]):
        idxs = np.where(ant1==ant2)
        gains = gains[idxs]
        ant1 = ant1[idxs]
    for i, antid in enumerate(ant1):
        if antid+1 in antennas:
            idx = np.where(antennas==antid+1)[0][0]
            weights[:, idx, ...] = gains[i, ...]

#     # interpolate over missing values
#     # Not needed anymore because we are interpolating the bandpass solutions
#     med = np.nanmedian(weights, axis=2, keepdims=True)
#     std = np.nanstd(weights.real, axis=2, keepdims=True)+\
#         1j*np.std(weights.imag, axis=2, keepdims=True)
#     weights[np.abs((weights-med).real) > 3*std.real] = np.nan
#     weights[np.abs((weights-med).imag) > 3*std.imag] = np.nan
#     for i in range(weights.shape[0]):
#         for j in range(weights.shape[1]):
#             for k in range(weights.shape[-1]):
#                 idx = np.where(np.isnan(weights[i, j, :, k]))[0]
#                 if len(idx) > 0:
#                     x = np.where(~np.isnan(weights[i, j, :, k]))[0]
#                     if len(x) > 24:
#                         fr = interp1d(
#                             x,
#                             weights[i, j, x, k].real,
#                             bounds_error=False,
#                             kind='nearest'
#                         )
#                         fi = interp1d(
#                             x,
#                             weights[i, j, x, k].imag,
#                             bounds_error=False,
#                             kind='nearest'
#                         )
#                         weights[i, j, idx, k] = fr(idx) + 1j*fi(idx)
#     weights[np.isnan(weights)] = 0.
    fracflagged = np.sum(np.sum(np.isnan(weights), axis=2), axis=0)\
        /(weights.shape[0]*weights.shape[2])
    antenna_flags_badsolns = fracflagged > tol
    weights[np.isnan(weights)] = 0.

    # Divide by the first non-flagged antenna
    idx0, idx1 = np.nonzero(
        np.logical_not(
            antenna_flags + antenna_flags_badsolns
        )
    )
    weights = (
        weights/weights[:, idx0[0], ..., idx1[0]][:, np.newaxis, :, np.newaxis]
    )
    weights[np.isnan(weights)] = 0.

    filenames = []
    for i, corr_idx in enumerate(corr_list):
        wcorr = weights[i, ...].view(np.float32).flatten()
        wcorr = np.concatenate([bu, wcorr], axis=0)
        fname = 'beamformer_weights_corr{0:02d}'.format(corr_idx)
        fname = '{0}_{1}_{2}'.format(
            fname,
            calname,
            caltime.isot
        )
        if os.path.exists('{0}/{1}.dat'.format(outdir, fname)):
            os.unlink('{0}/{1}.dat'.format(outdir, fname))
        with open('{0}/{1}.dat'.format(outdir, fname), 'wb') as f:
            f.write(bytes(wcorr))
        filenames += ['{0}.dat'.format(fname)]
    return corr_list, bu, fweights, filenames, antenna_flags_badsolns

def get_delays(antennas, msname, calname, applied_delays):
    r"""Returns the delays to be set in the correlator.

    Based on the calibrated delays and the currently applied delays.

    Parameters
    ----------
    antennas : list
        The antennas to get delays for.
    msname : str
        The path to the measurement set containing the calibrator pass.
    calname : str
        The name of the calibrator. Will open `msname`\_`calname`\_kcal.
    applied_delays : ndarray
        The currently applied delays for every antenna/polarization, in ns.
        Dimensions (antenna, pol).

    Returns
    -------
    delays : ndarray
        The delays to be applied for every antenna/polarization in ns.
        Dimensions (antenna, pol).
    flags : ndarray
        True if that antenna/pol data is flagged in the calibration table.
        In this case, the delay should be set to 0. Dimensions (antenna, pol).
    """
    delays, _time, flags, ant1, _ant2 = read_caltable(
        '{0}_{1}_kcal'.format(msname, calname)
    )
    delays = delays.squeeze()
    flags = flags.squeeze()
    print('delays: {0}'.format(delays.shape))
    # delays[flags] = np.nan
    ant1 = list(ant1)
    idx = [ant1.index(ant-1) for ant in antennas]
    delays = delays[idx]
    flags = flags[idx]
    delays = delays + applied_delays
    delays = delays - np.nanmin(delays)
    delays = (np.rint(delays/2)*2)
    # delays[flags] = 0
    return delays.astype(np.int), flags

def write_beamformer_solutions(
    msname, calname, caltime, antennas, applied_delays,
    corr_list=np.arange(1, 17),
    outdir='/home/user/beamformer_weights/',
    flagged_antennas=None,
    pols=None
):
    """Writes beamformer solutions to disk.

    Parameters
    ----------
    msname : str
        The name of the measurement set used for calibration.
    calname : str
        The name of the calibrator source used for calibration. Will open
        tables that start with `msname`_`calname`
    caltime : astropy.time.Time object
        The transit time of the calibrator.
    antennas : list
        The antenna names for which to write beamformer solutions, in order.
    applied_delays : ndarray
        The currently applied delays at the time of the calibration, in ns.
        Dimensions (antenna, pol). The antenna axis should be in the order
        specified by antennas.
    corr_list : list
        The indices of the correlator machines to write beamformer weights for.
        For now, these must be ordered so that the frequencies are contiguous
        and they are in the same order or the reverse order as in the ms. The
        bandwidth of each correlator is pulled from dsa110-meridian-fs package
        data.
    flagged_antennas : list
        A list of antennas to flag in the beamformer solutions. Should include
        polarizations. e.g. ['24 B', '32 A']
    outdir : str
        The directory to write the beamformer weights in.
    pols : list
        The order of the polarizations.

    Returns
    -------
    flags : ndarray(boolean)
        Dimensions (antennas, pols). True where the data is flagged, and should
        not be used. Compiled from the ms flags as well as `flagged_antennas`.
    """
    if pols is None:
        pols = ['B', 'A']
    beamformer_flags = {}
    delays, flags = get_delays(antennas, msname, calname, applied_delays)
    print('delay flags:', flags.shape)
    if flagged_antennas is not None:
        for item in flagged_antennas:
            ant, pol = item.split(' ')
            flags[antennas==ant, pols==pol] = 1
            beamformer_flags['{0} {1}'.format(ant, pol)] = ['flagged by user']
    delays = delays-np.min(delays[~flags])
    while not np.all(delays[~flags] < 1024):
        if np.sum(delays[~flags] > 1024) < np.nansum(delays[~flags] < 1024):
            argflag = np.argmax(delays[~flags])
        else:
            argflag = np.argmin(delays[~flags])
        argflag = np.where(~flags.flatten())[0][argflag]
        flag_idxs = np.unravel_index(argflag, flags.shape)
        flags[np.unravel_index(argflag, flags.shape)] = 1
        key = '{0} {1}'.format(antennas[flag_idxs[0]], pols[flag_idxs[1]])
        if key not in beamformer_flags.keys():
            beamformer_flags[key] = []
        beamformer_flags[key] += ['delay exceeds snap capabilities']
        delays = delays-np.min(delays[~flags])

    caltime.precision = 0
    corr_list, eastings, _fobs, weights_files, flags_badsolns = \
        write_beamformer_weights(msname, calname, caltime, antennas, outdir,
        corr_list, flags)
    idxant, idxpol = np.nonzero(flags_badsolns)
    for i, ant in enumerate(idxant):
        key = '{0} {1}'.format(antennas[ant], pols[idxpol[i]])
        if key not in beamformer_flags.keys():
            beamformer_flags[key] = []
        beamformer_flags[key] += ['casa solutions flagged']

    calibration_dictionary = {
        'cal_solutions':
        {
            'source': calname,
            'caltime': float(caltime.mjd),
            'antenna_order': [int(ant) for ant in antennas],
            'corr_order': [int(corr) for corr in corr_list],
            'pol_order': ['B', 'A'],
            'delays': [
                [
                    int(delay[0]//2),
                    int(delay[1]//2)
                ] for delay in delays
            ],
            'eastings': [float(easting) for easting in eastings],
            'weights_axis0': 'antenna',
            'weights_axis1': 'frequency',
            'weights_axis2': 'pol',
            'weight_files': weights_files,
            'flagged_antennas': beamformer_flags
        }
    }

    with open(
        '{0}/beamformer_weights_{1}_{2}.yaml'.format(
            outdir,
            calname,
            caltime.isot
        ),
        'w'
    ) as file:
        yaml.dump(calibration_dictionary, file)
    return flags

def convert_calibrator_pass_to_ms(
    cal, date, files, duration, msdir='/mnt/data/dsa110/calibration/',
    hdf5dir='/mnt/data/dsa110/correlator/',
    logger=None
):
    r"""Converts hdf5 files near a calibrator pass to a CASA ms.

    Parameters
    ----------
    cal : dsacalib.utils.src instance
        The calibrator source.
    date : str
        The date (to day precision) of the calibrator pass. e.g. '2020-10-06'.
    files : list
        The hdf5 filenames corresponding to the calibrator pass. These should
        be date strings to second precision.
        e.g. ['2020-10-06T12:35:04', '2020-10-06T12:50:04']
        One ms will be written per filename in `files`. If the length of
        `files` is greater than 1, the mss created will be virtualconcated into
        a single ms.
    duration : astropy quantity
        Amount of data to extract, unit minutes or equivalent.
    msdir : str
        The full path to the directory to place the measurement set in. The ms
        will be written to `msdir`/`date`\_`cal.name`.ms
    logger : dsautils.dsa_syslog.DsaSyslogger() instance
        Logger to write messages too. If None, messages are printed.
    """
    msname = '{0}/{1}_{2}'.format(msdir, date, cal.name)
    if len(files) == 1:
        try:
            uvh5_to_ms(
                sorted(glob.glob(
                    '{0}/corr??/{1}*.hdf5'.format(hdf5dir, files[0][:-4])
                )),
                msname,
                ra=cal.ra,
                dec=cal.dec,
                flux=cal.I,
                dt=duration,
                logger=logger
            )
            message = 'Wrote {0}.ms'.format(msname)
            if logger is not None:
                logger.info(message)
            else:
                print(message)
        except (ValueError, IndexError):
            message = 'No data for {0} transit on {1}'.format(date, cal.name)
            if logger is not None:
                logger.info(message)
            else:
                print(message)
    elif len(files) > 0:
        msnames = []
        for filename in files:
            try:
                uvh5_to_ms(
                    sorted(
                        glob.glob(
                            '{0}/corr??/{1}*.hdf5'.format(
                                hdf5dir,
                                filename[:-4]
                            )
                        )
                    ),
                    '{0}/{1}'.format(msdir, filename),
                    ra=cal.ra,
                    dec=cal.dec,
                    flux=cal.I,
                    dt=duration,
                    logger=logger
                )
                msnames += ['{0}/{1}'.format(msdir, filename)]
            except (ValueError, IndexError):
                pass
        if os.path.exists('{0}.ms'.format(msname)):
            for root, _dirs, walkfiles in os.walk(
                '{0}.ms'.format(msname),
                topdown=False
            ):
                for name in walkfiles:
                    os.unlink(os.path.join(root, name))
            shutil.rmtree('{0}.ms'.format(msname))
        if len(msnames) > 1:
            virtualconcat(
                ['{0}.ms'.format(msn) for msn in msnames],
                '{0}.ms'.format(msname))
            message = 'Wrote {0}.ms'.format(msname)
            if logger is not None:
                logger.info(message)
            else:
                print(message)
        elif len(msnames) == 1:
            os.rename('{0}.ms'.format(msnames[0]), '{0}.ms'.format(msname))
            message = 'Wrote {0}.ms'.format(msname)
            if logger is not None:
                logger.info(message)
            else:
                print(message)
        else:
            message = 'No data for {0} transit on {1}'.format(date, cal.name)
            if logger is not None:
                logger.info(message)
            else:
                print(message)
    else:
        message = 'No data for {0} transit on {1}'.format(date, cal.name)
        if logger is not None:
            logger.info(message)
        else:
            print(message)

def uvh5_to_ms(fname, msname, ra=None, dec=None, dt=None, antenna_list=None,
               flux=None, logger=None):
    """
    Converts a uvh5 data to a uvfits file.

    Parameters
    ----------
    fname : str
        The full path to the uvh5 data file.
    msname : str
        The name of the ms to write. Data will be written to `msname`.ms
    ra : astropy quantity
        The RA at which to phase the data. If None, will phase at the meridian
        of the center of the uvh5 file.
    dec : astropy quantity
        The DEC at which to phase the data. If None, will phase at the pointing
        declination.
    dt : astropy quantity
        Duration of data to extract. Default is to extract the entire file.
    antenna_list : list
        Antennas for which to extract visibilities from the uvh5 file. Default
        is to extract all visibilities in the uvh5 file.
    flux : float
        The flux of the calibrator in Jy. If included, will write a model of
        the primary beam response to the calibrator source to the model column
        of the ms. If not included, a model of a constant response over
        frequency and time will be written instead of the primary beam model.
    logger : dsautils.dsa_syslog.DsaSyslogger() instance
        Logger to write messages too. If None, messages are printed.
    """
    print(fname)
    # zenith_dec = 0.6503903199825691*u.rad
    UV = UVData()

    # Read in the data
    if antenna_list is not None:
        UV.read(fname, file_type='uvh5', antenna_names=antenna_list,
                run_check_acceptability=False, strict_uvw_antpos_check=False)
    else:
        UV.read(fname, file_type='uvh5', run_check_acceptability=False,
                strict_uvw_antpos_check=False)
    print( UV.ant_1_array[UV.ant_1_array==UV.ant_2_array][:25])
    pt_dec = UV.extra_keywords['phase_center_dec']*u.rad
    lamb = c.c/(UV.freq_array*u.Hz)
    if ra is None:
        ra = UV.lst_array[UV.Nblts//2]*u.rad
    if dec is None:
        dec = pt_dec

    if dt is not None:
        extract_times(UV, ra, dt)
    time = Time(UV.time_array, format='jd')

    # Set antenna positions
    # This should already be done by the writer but for some reason they
    # are being converted to ICRS
    df_itrf = get_itrf(height=UV.telescope_location_lat_lon_alt[-1])
    if len(df_itrf['x_m']) != UV.antenna_positions.shape[0]:
        message = 'Mismatch between antennas in current environment ({0}) and correlator environment ({1}) for file {2}'.format(
            len(df_itrf['x_m']),
            UV.antenna_positions.shape[0],
            fname
        )
        if logger is not None:
            logger.info(message)
        else:
            print(message)
    UV.antenna_positions[:len(df_itrf['x_m'])] = np.array([
        df_itrf['x_m'],
        df_itrf['y_m'],
        df_itrf['z_m']
    ]).T-UV.telescope_location
    antenna_positions = UV.antenna_positions + UV.telescope_location
    blen = np.zeros((UV.Nbls, 3))
    for i, ant1 in enumerate(UV.ant_1_array[:UV.Nbls]):
        ant2 = UV.ant_2_array[i]
        blen[i, ...] = UV.antenna_positions[ant2, :] - \
            UV.antenna_positions[ant1, :]

    uvw_m = calc_uvw_blt(blen, time[:UV.Nbls].mjd, 'HADEC',
                         np.zeros(UV.Nbls)*u.rad, np.ones(UV.Nbls)*pt_dec)

    # Use antenna positions since CASA uvws are slightly off from pyuvdatas
    # uvw_z = calc_uvw_blt(blen, time[:UV.Nbls].mjd, 'HADEC',
    #                      np.zeros(UV.Nbls)*u.rad, np.ones(UV.Nbls)*zenith_dec)
    # dw = (uvw_z[:, -1] - uvw_m[:, -1])*u.m
    # phase_model = np.exp((2j*np.pi/lamb*dw[:, np.newaxis, np.newaxis])
    #                      .to_value(u.dimensionless_unscaled))
    # UV.uvw_array = np.tile(uvw_z[np.newaxis, :, :], (UV.Ntimes, 1, 1)
    #                    ).reshape(-1, 3)
    # UV.data_array = (UV.data_array.reshape(
    #     UV.Ntimes, UV.Nbls, UV.Nspws,UV.Nfreqs, UV.Npols
    # )/phase_model[np.newaxis, ..., np.newaxis]).reshape(
    #     UV.Nblts, UV.Nspws, UV.Nfreqs, UV.Npols
    # )
    # UV.phase(ra.to_value(u.rad), dec.to_value(u.rad), use_ant_pos=True)
    # Below is the manual calibration which can be used instead.
    # Currently using because casa uvws are more accurate than pyuvdatas, which
    # aren't true uvw coordinates.
    blen = np.tile(blen[np.newaxis, :, :], (UV.Ntimes, 1, 1)).reshape(-1, 3)
    uvw = calc_uvw_blt(blen, time.mjd, 'RADEC', ra.to(u.rad), dec.to(u.rad))
    dw = (uvw[:, -1] - np.tile(uvw_m[np.newaxis, :, -1], (UV.Ntimes, 1)
                             ).reshape(-1))*u.m
    phase_model = np.exp((2j*np.pi/lamb*dw[:, np.newaxis, np.newaxis])
                         .to_value(u.dimensionless_unscaled))
    UV.uvw_array = uvw
    UV.data_array = UV.data_array/phase_model[..., np.newaxis]
    UV.phase_type = 'phased'
    UV.phase_center_dec = dec.to_value(u.rad)
    UV.phase_center_ra = ra.to_value(u.rad)
    UV.phase_center_epoch = 2000.
    # Look for missing channels
    freq = UV.freq_array.squeeze()
    # The channels may have been reordered by pyuvdata so check that the
    # parameter UV.channel_width makes sense now.
    ascending = np.median(np.diff(freq)) > 0
    if ascending:
        assert np.all(np.diff(freq) > 0)
    else:
        assert np.all(np.diff(freq) < 0)
        UV.freq_array = UV.freq_array[:, ::-1]
        UV.data_array = UV.data_array[:, :, ::-1, :]
        freq = UV.freq_array.squeeze()
    UV.channel_width = np.abs(UV.channel_width)
    # Are there missing channels?
    if not np.all(np.diff(freq)-UV.channel_width < 1e-5):
        # There are missing channels!
        nfreq = int(np.rint(np.abs(freq[-1]-freq[0])/UV.channel_width+1))
        freq_out = freq[0] + np.arange(nfreq)*UV.channel_width
        existing_idxs = np.rint((freq-freq[0])/UV.channel_width).astype(int)
        data_out = np.zeros((UV.Nblts, UV.Nspws, nfreq, UV.Npols),
                            dtype=UV.data_array.dtype)
        nsample_out = np.zeros((UV.Nblts, UV.Nspws, nfreq, UV.Npols),
                                dtype=UV.nsample_array.dtype)
        flag_out = np.zeros((UV.Nblts, UV.Nspws, nfreq, UV.Npols),
                             dtype=UV.flag_array.dtype)
        data_out[:, :, existing_idxs, :] = UV.data_array
        nsample_out[:, :, existing_idxs, :] = UV.nsample_array
        flag_out[:, :, existing_idxs, :] = UV.flag_array
        # Now write everything
        UV.Nfreqs = nfreq
        UV.freq_array = freq_out[np.newaxis, :]
        UV.data_array = data_out
        UV.nsample_array = nsample_out
        UV.flag_array = flag_out

    if os.path.exists('{0}.fits'.format(msname)):
        os.remove('{0}.fits'.format(msname))

    UV.write_uvfits('{0}.fits'.format(msname),
                    spoof_nonessential=True)
    # Get the model to write to the data
    if flux is not None:
        fobs = UV.freq_array.squeeze()/1e9
        lst = UV.lst_array
        model = amplitude_sky_model(du.src('cal', ra, dec, flux),
                                    lst, pt_dec, fobs)
        model = np.tile(model[:, :, np.newaxis], (1, 1, UV.Npols))
    else:
        model = np.ones((UV.Nblts, UV.Nfreqs, UV.Npols), dtype=np.complex64)

    if os.path.exists('{0}.ms'.format(msname)):
        shutil.rmtree('{0}.ms'.format(msname))
    importuvfits('{0}.fits'.format(msname),
                 '{0}.ms'.format(msname))

    with table('{0}.ms/ANTENNA'.format(msname), readonly=False) as tb:
        tb.putcol('POSITION', antenna_positions)

    addImagingColumns('{0}.ms'.format(msname))
    #if flux is not None:
    with table('{0}.ms'.format(msname), readonly=False) as tb:
        tb.putcol('MODEL_DATA', model)
        tb.putcol('CORRECTED_DATA', tb.getcol('DATA')[:])

def extract_times(UV, ra, dt):
    """Extracts data from specified times from an already open UVData instance.

    This is an alternative to opening the file with the times specified using
    pyuvdata.UVData.open().

    Parameters
    ----------
    UV : pyuvdata.UVData() instance
        The UVData instance from which to extract data. Modified in-place.
    ra : float
        The ra of the source around which to extract data, in radians.
    dt : astropy quantity
        The amount of data to extract, units seconds or equivalent.
    """
    lst_min = (ra - (dt*2*np.pi*u.rad/(ct.SECONDS_PER_SIDEREAL_DAY*u.s))/2
              ).to_value(u.rad)%(2*np.pi)
    lst_max = (ra + (dt*2*np.pi*u.rad/(ct.SECONDS_PER_SIDEREAL_DAY*u.s))/2
              ).to_value(u.rad)%(2*np.pi)
    if lst_min < lst_max:
        idx_to_extract = np.where((UV.lst_array >= lst_min) &
                                  (UV.lst_array <= lst_max))[0]
    else:
        idx_to_extract = np.where((UV.lst_array >= lst_min) |
                                  (UV.lst_array <= lst_max))[0]
    if len(idx_to_extract) == 0:
        raise ValueError("No times in uvh5 file match requested timespan "
                         "with duration {0} centered at RA {1}.".format(
                         dt, ra))
    idxmin = min(idx_to_extract)
    idxmax = max(idx_to_extract)+1
    assert (idxmax-idxmin)%UV.Nbls == 0
    UV.uvw_array = UV.uvw_array[idxmin:idxmax, ...]
    UV.data_array = UV.data_array[idxmin:idxmax, ...]
    UV.time_array = UV.time_array[idxmin:idxmax, ...]
    UV.lst_array = UV.lst_array[idxmin:idxmax, ...]
    UV.nsample_array = UV.nsample_array[idxmin:idxmax, ...]
    UV.flag_array = UV.flag_array[idxmin:idxmax, ...]
    UV.ant_1_array = UV.ant_1_array[idxmin:idxmax, ...]
    UV.ant_2_array = UV.ant_2_array[idxmin:idxmax, ...]
    UV.baseline_array = UV.baseline_array[idxmin:idxmax, ...]
    UV.integration_time = UV.integration_time[idxmin:idxmax, ...]
    UV.Nblts = int(idxmax-idxmin)
    assert UV.data_array.shape[0]==UV.Nblts
    UV.Ntimes = UV.Nblts//UV.Nbls

def average_beamformer_solutions(
    fnames, ttime, outdir, corridxs=None, tol=0.3, logger=None
):
    """Averages written beamformer solutions.

    Parameters
    ----------
    fnames : list
    ttime : astropy.time.Time object
        A time to use in the filename of the solutions, indicating when they
        were written or are useful. E.g. the transit time of the most recent
        source being averaged over.
    outdir : str
        The directory in which the beamformer solutions are written, and into
        which new solutions should be written.
    corridxs : list
        The correlator nodes for which to average beamformer solutions.
        Defaults to 1 through 16 inclusive.
    logger : dsautils.dsa_syslog.DsaSyslogger() instance
        Logger to write messages too. If None, messages are printed.

    Returns
    -------
    written_files : list
        The names of the written beamformer solutions (one for each correlator
        node).
    antenna_flags_badsolns:
        Flags for antenna/polarization dimensions of gains.
    """
    if corridxs is None:
        corridxs = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
        ]
    gainshape = (64, 48, 2, 2)
    gains = np.ones(
            (len(fnames), len(corridxs), gainshape[0], gainshape[1],
             gainshape[2], gainshape[3]),
            dtype='<f4'
        )*np.nan
    antenna_flags = [None]*len(fnames)
    eastings = None
    for i, fname in enumerate(fnames):
        tmp_antflags = []
        filepath = '{0}/beamformer_weights_{1}.yaml'.format(outdir, fname)
        if os.path.exists(filepath):
            with open(filepath) as f:
                calibration_params = yaml.load(
                    f, Loader=yaml.FullLoader
                )['cal_solutions']
                antenna_order = calibration_params['antenna_order']
                for key in calibration_params['flagged_antennas']:
                    if 'casa solutions flagged' in \
                        calibration_params['flagged_antennas'][key]:
                        antname = int(key.split(' ')[0])
                        tmp_antflags.append(antenna_order.index(antname))
            antenna_flags[i] = sorted(tmp_antflags)

        for j, corr in enumerate(corridxs):
            if os.path.exists(
                '{0}/beamformer_weights_corr{1:02d}_{2}.dat'.format(
                outdir,
                corr,
                fname
                )
            ):
                with open(
                    '{0}/beamformer_weights_corr{1:02d}_{2}.dat'.format(
                        outdir,
                        corr,
                        fname
                    ),
                    'rb'
                ) as f:
                    data = np.fromfile(f, '<f4')
                    eastings = data[:64]
                    gains[i, j, ...] = data[64:].reshape(gainshape)
                if antenna_flags[i] is not None:
                    gains[i, :, antenna_flags[i], ... ] = np.nan
            else:
                message = \
                    '{0} not found during beamformer weight averaging'.format(
                        '{0}/beamformer_weights_corr{1:02d}_{2}.dat'.format(
                        outdir,
                        corr,
                        fname
                    ))
                if logger is not None:
                    logger.info(message)
                else:
                    print(message)
        if antenna_flags[i] is not None:
            gains[i, :, antenna_flags[i], ... ] = np.nan

    gains = np.nanmean(gains, axis=0) #np.nanmedian(gains, axis=0)
    print(gains.shape) # corr, antenna, freq, pol, complex
    fracflagged = np.sum(np.sum(np.sum(
        np.isnan(gains),
        axis=4), axis=2), axis=0)\
        /(gains.shape[0]*gains.shape[2]*gains.shape[4])
    antenna_flags_badsolns = fracflagged > tol
    gains[np.isnan(gains)] = 0.
    written_files = []
    if eastings is not None:
        for i, corr in enumerate(corridxs):
            fnameout = 'beamformer_weights_corr{0:02d}_{1}'.format(
                corr, ttime.isot
            )
            wcorr = gains[i, ...].flatten()
            wcorr = np.concatenate([eastings, wcorr], axis=0)
            with open('{0}/{1}.dat'.format(outdir, fnameout), 'wb') as f:
                f.write(bytes(wcorr))
            written_files += ['{0}.dat'.format(fnameout)]
    return written_files, antenna_flags_badsolns
