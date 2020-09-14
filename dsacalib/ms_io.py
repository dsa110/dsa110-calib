"""
DSACALIB/MS_IO.PY

Dana Simard, dana.simard@astro.caltech.edu, 10/2019

Routines to interact with CASA measurement sets and calibration tables.
"""

# To do:
# Replace to_deg w/ astropy versions

# Always import scipy before importing casatools.
import numpy as np
import scipy # pylint: disable=unused-import
import casatools as cc
from dsautils import dsa_store
from dsautils import calstatus as cs
import astropy.units as u
from dsacalib import constants as ct
from dsacalib.utils import get_antpos_itrf, get_autobl_indices

de = dsa_store.DsaStore()

def simulate_ms(ofile, tname, anum, xx, yy, zz, diam, mount, pos_obs, spwname,
                freq, deltafreq, freqresolution, nchannels, integrationtime,
                obstm, dt, source, stoptime, autocorr):
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
        The end time of the observation in MJD.
    autocorr : boolean
        Set to ``True`` if the visibilities include autocorrelations, ``False``
        if the only include crosscorrelations.
    """
    me = cc.measures()
    qa = cc.quanta()
    sm = cc.simulator()
    sm.open(ofile)
    sm.setconfig(telescopename=tname, x=xx, y=yy, z=zz, dishdiameter=diam,
                 mount=mount, antname=anum, coordsystem='global',
                 referencelocation=pos_obs)
    sm.setspwindow(spwname=spwname, freq=freq, deltafreq=deltafreq,
                   freqresolution=freqresolution, nchannels=nchannels,
                   stokes='XX YY')
    sm.settimes(integrationtime=integrationtime, usehourangle=False,
                referencetime=me.epoch('utc', qa.quantity(obstm-dt, 'd')))
    sm.setfield(
        sourcename=source.name, sourcedirection=me.direction(
            source.epoch, qa.quantity(source.ra.to_value(u.rad), 'rad'),
            qa.quantity(source.dec.to_value(u.rad), 'rad')))
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

    anum, xx, yy, zz = get_antpos_itrf(antpos)
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

    simulate_ms('{0}.ms'.format(ofile), tname, anum, xx, yy, zz, diam, mount,
                pos_obs, spwname, freq, deltafreq, freqresolution, nchannels,
                integrationtime, obstm, dt, source, stoptime, autocorr)

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
        simulate_ms('{0}.ms'.format(ofile), tname, anum, xx, yy, zz, diam,
                    mount, pos_obs, spwname, freq, deltafreq, freqresolution,
                    nchannels, integrationtime, obstm, dt, source, stoptime,
                    autocorr)

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

def extract_vis_from_ms(ms_name, nbls):
    """ Extracts visibilities from a CASA measurement set.

    Parameters
    ----------
    ms_name : str
        The name of the measurement set.  Will open `ms_name`.ms.
    nbls : int
        The number of baselines in the measurement set.

    Returns
    -------
    vis_uncal : ndarray
        The 'observed' visibilities from the measurement set.  Dimensions
        (baseline, time, freq, polarization).
    vis_cal : ndarray
        The 'corrected' visibilities from the measurement set.  Dimensions
        (baseline, time, freq, polarization).
    time : array
        The time of each subintegration, in MJD.
    freq : array
        The frequency of each channel, in GHz.
    flags : ndarray
        The flags for the visibility data, same dimensions as `vis_uncal` and
        `vis_cal`. `flags` is 1 (or ``True``) where the data is flagged
        (invalid) and 0 (or ``False``) otherwise.
    """
    error = 0
    ms = cc.ms()
    error += not ms.open('{0}.ms'.format(ms_name))
    axis_info = ms.getdata(['axis_info'])['axis_info']
    freq = axis_info['freq_axis']['chan_freq'].squeeze()/1e9
    time = ms.getdata(['time'])['time'].reshape(-1, nbls)[..., 0]
    time = time/ct.SECONDS_PER_DAY

    vis_uncal = ms.getdata(["data"])['data']
    vis_uncal = vis_uncal.reshape(vis_uncal.shape[0], vis_uncal.shape[1], -1,
                                  nbls).T
    vis_cal = ms.getdata(["corrected_data"])['corrected_data']
    vis_cal = vis_cal.reshape(vis_cal.shape[0], vis_cal.shape[1], -1, nbls).T

    flags = ms.getdata(["flag"])['flag']
    flags = flags.reshape(flags.shape[0], flags.shape[1], -1, nbls).T
    error += not ms.close()
    if error > 0:
        print('{0} errors occured during calibration'.format(error))
    return vis_uncal, vis_cal, time, freq, flags

def read_caltable(tablename, nbls, cparam=False):
    """Extracts calibration solution from a CASA calibration table.

    Parameters
    ----------
    tablename : str
        The full path to the CASA calibration table.
    nbls : int
        The number of baselines or antennas in the CASA calibration solutions.
        This can be calculated from the number of antennas, nant:

            For delay calibration, nbls=nant
            For antenna-based gain/bandpass calibration, nbls=nant
            For baseline-based gain/bandpass calibration,
            nbls=(nant*(nant+1))//2

    cparam : boolean
        Whether the parameter of interest is complex (set to ``True``) or a
        float (set to ``False``).  For delay calibration, set to ``False``. For
        gain/bandpass calibration, set to ``True``.  Defaults ``False``.

    Returns
    -------
    time : array
        The times at which each solution is calculated in MJD. Dimensions
        (time, baselines). If no column 'TIME' is in the calibration table,
        ``None`` is returned.
    vals : ndarray
        The calibration solutions. Dimensions may be (polarization, time,
        baselines) or (polarization, 1, baselines) (for delay or gain
        calibration) or (polarization,frequency,baselines) (for bandpass cal).
    flags : ndarray
        Flags corresponding to the calibration solutions.  Same dimensions as
        `vals`. A value of 1 or ``True`` if the data is flagged (invalid), 0 or
        ``False`` otherwise.
    """
    param_type = 'CPARAM' if cparam else 'FPARAM'
    tb = cc.table()
    print('Opening table {0} as type {1}'.format(tablename, param_type))
    tb.open(tablename)
    if 'TIME' in tb.colnames():
        time = (tb.getcol('TIME').reshape(-1, nbls)*u.s).to_value(u.d)
    else:
        time = None
    vals = tb.getcol(param_type)
    vals = vals.reshape(vals.shape[0], -1, nbls)
    flags = tb.getcol('FLAG')
    flags = flags.reshape(flags.shape[0], -1, nbls)
    tb.close()

    return time, vals, flags

def caltable_to_etcd(msname, calname, antenna_order, caltime, status,
                     baseline_cal=False, pols=None):
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
    antenna_order : list or array
        The antenna numbers, in CASA ordering.
    baseline_cal : boolean
        Set to ``True`` if the gain table was calculated using baseline-based
        calibration, ``False`` if the gain table was calculated using
        antenna-based calibration. Defaults ``False``.
    pols : list
        The names of the polarizations. If ``None``, will be set to
        ``['A', 'B']``. Defaults ``None``.
    """
    if pols is None:
        pols = ['A', 'B']
    nant = len(antenna_order)
    # Number of baselines included in the gain and bandpass calibrations.
    if baseline_cal:
        nbls = (nant*(nant+1))//2
    else:
        nbls = nant

    try:
        # Complex gains for each antenna.
        tamp, amps, flags = read_caltable('{0}_{1}_gcal_ant'.format(
            msname, calname), nbls, cparam=True)
        mask = np.ones(flags.shape)
        mask[flags == 1] = np.nan
        amps = amps*mask
        if baseline_cal:
            autocorr_idx = get_autobl_indices(nant)
            tamp = tamp[..., autocorr_idx]
            amps = amps[..., autocorr_idx]

        # Check the output shapes.
        assert tamp.shape[0] == amps.shape[1]
        assert tamp.shape[1] == amps.shape[2]
        assert tamp.shape[1] == len(antenna_order)
        assert amps.shape[0] == len(pols)

        # Reduce tamp to a single value, amps to a single value for each
        # ant/pol pair.
        assert np.all(np.equal.reduce(tamp) == np.ones(len(antenna_order)))
        tamp = np.median(tamp[:, 0])
        if amps.ndim == 3:
            amps = np.nanmedian(amps, axis=1)
        gaincaltime_offset = (tamp-caltime)*ct.SECONDS_PER_DAY
    except Exception:
        tamp = np.nan
        amps = np.ones((len(pols), nant))*np.nan
        gaincaltime_offset = 0.
        status = cs.update(status, ['gain_tbl_err', 'inv_gainamp_p1',
                                    'inv_gainamp_p2', 'inv_gainphase_p1',
                                    'inv_gainphase_p2', 'inv_gaincaltime'])

    # Delays for each antenna.
    try:
        tdel, delays, flags = read_caltable('{0}_{1}_kcal'.format(
            msname, calname), nant, cparam=False)
        mask = np.ones(flags.shape)
        mask[flags == 1] = np.nan
        delays = delays*mask
        assert tdel.shape[0] == delays.shape[1]
        assert tdel.shape[1] == delays.shape[2]
        assert tdel.shape[1] == len(antenna_order)
        assert delays.shape[0] == len(pols)
        # Reduce tdel to a single value, delays to a single value for each
        # ant/pol pair.
        assert np.all(np.equal.reduce(tdel) == np.ones(len(antenna_order)))
        tdel = np.median(tdel[:, 0])
        if delays.ndim == 3:
            delays = np.nanmedian(delays, axis=1)
        delaycaltime_offset = (tdel-caltime)*ct.SECONDS_PER_DAY
    except Exception:
        tdel = np.nan
        delays = np.ones((len(pols), nant))*np.nan
        delaycaltime_offset = 0.
        status = cs.update(status, ['delay_tbl-err', 'inv_delay_p1',
                                    'inv_delay_p2', 'inv_delaycaltime'])

    # Update antenna 24:
    for i, antnum in enumerate(antenna_order):
        # Deal with old/new numbering system for now.
        if antnum == 2:
            antnum = 24

        # Everything needs to be cast properly.
        gainamp = []
        gainphase = []
        ant_delay = []
        for amp in amps[:, i]:
            if not np.isnan(amp):
                gainamp += [np.abs(amp)]
                gainphase += [np.angle(amp)]
            else:
                gainamp += [None]
                gainphase += [None]
        for delay in delays[:, i]:
            if not np.isnan(delay):
                ant_delay += [int(np.rint(delay))]
            else:
                ant_delay += [None]

        dd = {'ant_num': antnum,
              'time': caltime,
              'pol': pols,
              'gainamp': gainamp,
              'gainphase': gainphase,
              'delay': ant_delay,
              'calsource': calname,
              'gaincaltime_offset': gaincaltime_offset,
              'delaycaltime_offset': delaycaltime_offset,
              'sim': False,
              'status':status}
        required_keys = dict({'ant_num':['inv_antnum', 0],
                              'time':['inv_time', 0.],
                              'pol':['inv_pol', ['A', 'B']],
                              'calsource':['inv_calsource', 'Unknown'],
                              'sim':['inv_sim', False],
                              'status':['other_err', 0]})
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
                status = cs.update(status, 'inv_pol')
                dd['pol'] = ['A', 'B']

        de.put_dict('/mon/cal/{0}'.format(antnum), dd)
