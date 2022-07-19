"""
Dsacalib/MS_IO.PY

Dana Simard, dana.simard@astro.caltech.edu, 10/2019

Routines to interact with CASA measurement sets and calibration tables.
"""

# To do:
# Replace to_deg w/ astropy versions

import glob
import os
import shutil
import traceback
from typing import List

import astropy.units as u
from astropy.time import Time
import scipy  # noqa
import casatools as cc
import numpy as np
from casacore.tables import table
from casatasks import virtualconcat
from dsautils import calstatus as cs
from dsautils import dsa_store

import dsacalib.utils as du
from dsacalib.uvh5_to_ms import uvh5_to_ms
from dsacalib import constants as ct


def convert_calibrator_pass_to_ms(
        cal, date, files, msdir, hdf5dir, refmjd, antenna_list=None, logger=None, overwrite=True):
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
    msdir : str
        The full path to the directory to place the measurement set in. The ms
        will be written to `msdir`/`date`\_`cal.name`.ms
    hdf5dir : str
        The full path to the directory containing subdirectories with correlated
        hdf5 data.
    antenna_list : list
        The names of the antennas to include in the measurement set. Names should
        be strings.  If not passed, all antennas in the hdf5 files are included.
    logger : dsautils.dsa_syslog.DsaSyslogger() instance
        Logger to write messages too. If None, messages are printed.
    """
    msname = f"{msdir}/{date}_{cal.name}"
    print(f"looking for files: {' '.join(files)}")
    if len(files) == 1:
        try:
            reftime = Time(files[0])
            hdf5files = []
            # TODO: improve this search so there are no edge cases
            for hdf5f in sorted(glob.glob(f"{hdf5dir}/{files[0][:-6]}*.hdf5")):
                filetime = Time(hdf5f[:-5].split("/")[-1].split('_')[0])
                if abs(filetime - reftime) < 2.5 * u.min:
                    hdf5files += [hdf5f]
            assert len(hdf5files) < 17
            assert len(hdf5files) > 1
            print(f"found {len(hdf5files)} hdf5files for {files[0]}")
            uvh5_to_ms(
                hdf5files,
                msname,
                refmjd,
                ra=cal.ra,
                dec=cal.dec,
                flux=cal.flux,
                antenna_list=antenna_list,
                logger=logger
            )
            message = f"Wrote {msname}.ms"
            if logger is not None:
                logger.info(message)
            print(message)
        except (ValueError, IndexError) as exception:
            tbmsg = "".join(traceback.format_tb(exception.__traceback__))
            message = f'No data for {date} transit on {cal.name}. ' +\
                f'Error {type(exception).__name__}. Traceback: {tbmsg}'
            if logger is not None:
                logger.info(message)
            print(message)
    elif len(files) > 0:
        msnames = []
        for filename in files:
            print(filename)
            if overwrite or not os.path.exists(f'{msdir}/{filename}.ms'):
                try:
                    reftime = Time(filename)
                    hdf5files = []
                    for hdf5f in sorted(glob.glob(f"{hdf5dir}/{filename[:-6]}*.hdf5")):
                        filetime = Time(
                            hdf5f[:-5].split('/')[-1].split('_')[0])
                        if abs(filetime - reftime) < 2.5 * u.min:
                            hdf5files += [hdf5f]
                    print(f"found {len(hdf5files)} hdf5files for {filename}")
                    uvh5_to_ms(
                        hdf5files,
                        f"{msdir}/{filename}",
                        refmjd,
                        ra=cal.ra,
                        dec=cal.dec,
                        flux=cal.flux,
                        antenna_list=antenna_list,
                        logger=logger
                    )
                    msnames += [f"{msdir}/{filename}"]
                except (ValueError, IndexError) as exception:
                    message = (
                        f"No data for {filename}. Error {type(exception).__name__}. "
                        f"Traceback: {''.join(traceback.format_tb(exception.__traceback__))}"
                    )
                    if logger is not None:
                        logger.info(message)
                    print(message)
            else:
                print(f"Not doing {filename}")
        if os.path.exists(f"{msname}.ms"):
            for root, _dirs, walkfiles in os.walk(
                    f"{msname}.ms",
                    topdown=False
            ):
                for name in walkfiles:
                    os.unlink(os.path.join(root, name))
            shutil.rmtree(f"{msname}.ms")
        if len(msnames) > 1:
            virtualconcat(
                [f"{msn}.ms" for msn in msnames],
                f"{msname}.ms"
            )
            message = f"Wrote {msname}.ms"
            if logger is not None:
                logger.info(message)
            print(message)
        elif len(msnames) == 1:
            os.rename(f"{msnames[0]}.ms", f"{msname}.ms")
            message = f"Wrote {msname}.ms"
            if logger is not None:
                logger.info(message)
            print(message)
        else:
            message = f"No data for {date} transit on {cal.name}"
            if logger is not None:
                logger.info(message)
            print(message)
    else:
        message = f"No data for {date} transit on {cal.name}"
        if logger is not None:
            logger.info(message)
        print(message)


def simulate_ms(
        ofile, tname, anum, xx, yy, zz, diam, mount, pos_obs, spwname, freq, deltafreq,
        freqresolution, nchannels, integrationtime, obstm, dt, source, stoptime, 
        autocorr, fullpol):
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
        coordsystem="global",
        referencelocation=pos_obs,
    )
    sm.setspwindow(
        spwname=spwname,
        freq=freq,
        deltafreq=deltafreq,
        freqresolution=freqresolution,
        nchannels=nchannels,
        stokes="XX XY YX YY" if fullpol else "XX YY",
    )
    sm.settimes(
        integrationtime=integrationtime,
        usehourangle=False,
        referencetime=me.epoch("utc", qa.quantity(obstm - dt, "d")),
    )
    sm.setfield(
        sourcename=source.name,
        sourcedirection=me.direction(
            source.epoch,
            qa.quantity(source.ra.to_value(u.rad), "rad"),
            qa.quantity(source.dec.to_value(u.rad), "rad"),
        ),
    )
    sm.setauto(autocorrwt=1.0 if autocorr else 0.0)
    sm.observe(source.name, spwname, starttime="0s", stoptime=stoptime)
    sm.close()


def convert_to_ms(
    source,
    vis,
    obstm,
    ofile,
    bname,
    antenna_order,
    tsamp=ct.TSAMP * ct.NINT,
    nint=1,
    antpos=None,
    model=None,
    dt=ct.CASA_TIME_OFFSET,
    dsa10=True,
):
    """Writes visibilities to an ms.

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
        antpos = f"{ct.PKG_DATA_PATH}/antpos_ITRF.txt"
    vis = vis.astype(np.complex128)
    if model is not None:
        model = model.astype(np.complex128)

    nant = len(antenna_order)

    me = cc.measures()

    # Observatory parameters
    tname = "OVRO_MMA"
    diam = 4.5  # m
    obs = "OVRO_MMA"
    mount = "alt-az"
    pos_obs = me.observatory(obs)

    # Backend
    if dsa10:
        spwname = "L_BAND"
        freq = "1.4871533196875GHz"
        deltafreq = "-0.244140625MHz"
        freqresolution = deltafreq
    else:
        spwname = "L_BAND"
        freq = "1.28GHz"
        deltafreq = "40.6901041666667kHz"
        freqresolution = deltafreq
    (_, _, nchannels, npol) = vis.shape

    # Rebin visibilities
    integrationtime = f"{tsamp * nint}s"
    if nint != 1:
        npad = nint - vis.shape[1] % nint
        if npad == nint:
            npad = 0
        vis = np.nanmean(
            np.pad(
                vis,
                ((0, 0), (0, npad), (0, 0), (0, 0)),
                mode="constant",
                constant_values=(np.nan,),
            ).reshape((vis.shape[0], -1, nint, vis.shape[2], vis.shape[3])),
            axis=2,
        )
        if model is not None:
            model = np.nanmean(
                np.pad(
                    model,
                    ((0, 0), (0, npad), (0, 0), (0, 0)),
                    mode="constant",
                    constant_values=(np.nan,),
                ).reshape((model.shape[0], -1, nint, model.shape[2], model.shape[3])),
                axis=2,
            )
    stoptime = f"{vis.shape[1] * tsamp * nint}s"

    anum, xx, yy, zz = du.get_antpos_itrf(antpos)
    # Sort the antenna positions
    idx_order = sorted([int(a) - 1 for a in antenna_order])
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
    nints, anum, xx, yy, zz = zip(
        *sorted(zip(nints, anum, xx, yy, zz), reverse=True))

    # Check that the visibilities are ordered correctly by checking the order
    # of baselines in bname
    idx_order = []
    autocorr = bname[0][0] == bname[0][1]

    for i in range(nant):
        for j in range(i if autocorr else i + 1, nant):
            idx_order += [bname.index([anum[i], anum[j]])]
    assert idx_order == list(
        np.arange(len(bname), dtype=int)
    ), "Visibilities not ordered by baseline"
    anum = [str(a) for a in anum]

    simulate_ms(
        f"{ofile}.ms",
        tname,
        anum,
        xx,
        yy,
        zz,
        diam,
        mount,
        pos_obs,
        spwname,
        freq,
        deltafreq,
        freqresolution,
        nchannels,
        integrationtime,
        obstm,
        dt,
        source,
        stoptime,
        autocorr,
        fullpol=False,
    )

    # Check that the time is correct
    ms = cc.ms()
    ms.open(f"{ofile}.ms")
    tstart_ms = ms.summary()["BeginTime"]
    ms.close()

    print("autocorr :", autocorr)

    if np.abs(tstart_ms - obstm) > 1e-10:
        dt = dt + (tstart_ms - obstm)
        print(f"Updating casa time offset to {dt * ct.SECONDS_PER_DAY}s")
        print("Rerunning simulator")
        simulate_ms(
            f"{ofile}.ms",
            tname,
            anum,
            xx,
            yy,
            zz,
            diam,
            mount,
            pos_obs,
            spwname,
            freq,
            deltafreq,
            freqresolution,
            nchannels,
            integrationtime,
            obstm,
            dt,
            source,
            stoptime,
            autocorr,
            fullpol=False,
        )

    # Reopen the measurement set and write the observed visibilities
    ms = cc.ms()
    ms.open(f"{ofile}.ms", nomodify=False)
    ms.selectinit(datadescid=0)

    rec = ms.getdata(["data"])
    # rec['data'] has shape [scan, channel, [time*baseline]]
    vis = vis.T.reshape((npol, nchannels, -1))
    rec["data"] = vis
    ms.putdata(rec)
    ms.close()

    ms = cc.ms()
    ms.open(f"{ofile}.ms", nomodify=False)
    if model is None:
        model = np.ones(vis.shape, dtype=complex)
    else:
        model = model.T.reshape((npol, nchannels, -1))
    rec = ms.getdata(["model_data"])
    rec["model_data"] = model
    ms.putdata(rec)
    ms.close()

    # Check that the time is correct
    ms = cc.ms()
    ms.open(f"{ofile}.ms")
    tstart_ms = ms.summary()["BeginTime"]
    tstart_ms2 = ms.getdata("TIME")["time"][0] / ct.SECONDS_PER_DAY
    ms.close()

    assert (
        np.abs(tstart_ms - (tstart_ms2 - tsamp *
               nint / ct.SECONDS_PER_DAY / 2)) < 1e-10
    ), "Data start time does not agree with MS start time"

    assert (
        np.abs(tstart_ms - obstm) < 1e-10
    ), "Measurement set start time does not agree with input tstart"
    print(f"Visibilities writing to ms {ofile}.ms")


def get_visiblities_time(
    msname, a1, a2, time, duration, datacolumn="CORRECTED_DATA", npol=2
):
    """Calculates the off-source gains from the autocorrs in the ms.

    Parameters
    ----------
    msname : str
        The path to the measurement set, with the .ms extension omitted.
    time : astropy.time.Time
        Time around which to extract in days.
    duration : astropy Quantity
        Amount of data to extract.
    npol : int
        The number of polarizations.

    Returns
    -------
    ndarray
        The visibilities. Same dimensions as `ant_transit_time`.
    """
    _, tvis, fvis, _, ant1, ant2, _pt_dec, spw, orig_shape = extract_vis_from_ms(
        msname, datacolumn.upper(), metadataonly=True
    )
    assert orig_shape == ["time", "baseline", "spw"]
    nspw = len(spw)
    assert nspw == 1
    nbls = len(ant1)
    nfreqs = len(fvis)

    idx0 = np.argmin(np.abs(tvis - (time - duration // 2).mjd))
    idx1 = np.argmin(np.abs(tvis - (time + duration // 2).mjd))
    ntimes = idx1 - idx0
    tidxs = np.arange(idx0, idx1)

    blidx = np.where((ant1 == a1) & (ant2 == a2))[0][0]

    vis = np.zeros((ntimes, nfreqs, npol), dtype=complex)
    with table(f"{msname}.ms") as tb:
        for j, tidx in enumerate(tidxs):
            idx = int(tidx * (nbls * nspw) + blidx * nspw)
            try:
                if datacolumn.upper() == "CORRECTED_DATA":
                    tmp = tb.CORRECTED_DATA[idx]
                elif datacolumn.upper() == "DATA":
                    tmp = tb.DATA[idx]
                elif datacolumn.upper() == "MODEL_DATA":
                    tmp = tb.MODEL_DATA[idx]
                else:
                    raise RuntimeError(
                        f"No column {datacolumn} in {msname}.ms")
            except IndexError:
                vis[j, :, :] = np.nan
                print(f"No data for tidx {tidx}, blidx {blidx}")
            else:
                vis[j, :, :] = tmp

    return vis


def extract_vis_from_ms(msname, data="data", swapaxes=True, metadataonly=False):
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
    spw : array
        The spectral window indices.
    orig_shape : list
        The order of the first three axes in the ms.
    """
    with table(f"{msname}.ms") as tb:
        ant1 = np.array(tb.ANTENNA1[:])
        ant2 = np.array(tb.ANTENNA2[:])
        if metadataonly:
            vals = None
            flags = None
        else:
            vals = np.array(tb.getcol(data.upper())[:])
            flags = np.array(tb.FLAG[:])
        time = np.array(tb.TIME[:])
        spw = np.array(tb.DATA_DESC_ID[:])
    fobs = du.freq_GHz_from_ms(msname)

    baseline = 2048 * (ant1 + 1) + (ant2 + 1) + 2**16

    time, vals, flags, ant1, ant2, spw, orig_shape = reshape_calibration_data(
        vals, flags, ant1, ant2, baseline, time, spw, swapaxes
    )

    with table(f"{msname}.ms/FIELD") as tb:
        pt_dec = tb.PHASE_DIR[:][0][0][1]

    return (
        vals,
        time / ct.SECONDS_PER_DAY,
        fobs,
        flags,
        ant1,
        ant2,
        pt_dec,
        spw,
        orig_shape,
    )


def read_caltable(tablename, cparam=False, reshape=True):
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
    baseline = 2048 * (ant1 + 1) + (ant2 + 1) + 2**16

    if reshape:
        time, vals, flags, ant1, ant2, _, _ = reshape_calibration_data(
            vals, flags, ant1, ant2, baseline, time, spw
        )

    return vals, time / ct.SECONDS_PER_DAY, flags, ant1, ant2


def reshape_calibration_data(
        vals, flags, ant1, ant2, baseline, time, spw, swapaxes=True
):
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
    orig_shape : list
        The original order of the time, baseline and spw axes in the ms.
    """
    if vals is None:
        assert flags is None
    if len(np.unique(ant1)) == len(np.unique(ant2)):
        nbl = len(np.unique(baseline))
    else:
        nbl = max([len(np.unique(ant1)), len(np.unique(ant2))])
    nspw = len(np.unique(spw))
    ntime = len(time) // nbl // nspw
    nfreq = vals.shape[-2] if vals is not None else 1
    npol = vals.shape[-1] if vals is not None else 1

    if np.all(time[: nspw * nbl] == time[0]):
        if np.all(baseline[:nspw] == baseline[0]):
            # time, baseline, spw
            orig_shape = ["time", "baseline", "spw"]
            time = time.reshape(ntime, nbl, nspw)[:, 0, 0]
            if vals is not None:
                vals = vals.reshape(ntime, nbl, nspw, nfreq, npol)
                flags = flags.reshape(ntime, nbl, nspw, nfreq, npol)
            if swapaxes:
                if vals is not None:
                    vals = vals.swapaxes(0, 1)
                    flags = flags.swapaxes(0, 1)
            ant1 = ant1.reshape(ntime, nbl, nspw)[0, :, 0]
            ant2 = ant2.reshape(ntime, nbl, nspw)[0, :, 0]
            spw = spw.reshape(ntime, nbl, nspw)[0, 0, :]
        else:
            orig_shape = ["time", "spw", "baseline"]
            assert np.all(spw[:nbl] == spw[0])
            time = time.reshape(ntime, nspw, nbl)[:, 0, 0]
            if vals is not None:
                vals = vals.reshape(ntime, nspw, nbl, nfreq, npol)
                flags = flags.reshape(ntime, nspw, nbl, nfreq, npol)
            if swapaxes:
                if vals is not None:
                    vals = vals.swapaxes(1, 2).swapaxes(0, 1)
                    flags = flags.swapaxes(1, 2).swapaxes(0, 1)
            ant1 = ant1.reshape(ntime, nspw, nbl)[0, 0, :]
            ant2 = ant2.reshape(ntime, nspw, nbl)[0, 0, :]

    elif np.all(baseline[: ntime * nspw] == baseline[0]):
        if np.all(time[:nspw] == time[0]):
            orig_shape = ["baseline", "time", "spw"]
            # baseline, time, spw
            time = time.reshape(nbl, ntime, nspw)[0, :, 0]
            if vals is not None:
                vals = vals.reshape(nbl, ntime, nspw, nfreq, npol)
                flags = flags.reshape(nbl, ntime, nspw, nfreq, npol)
            ant1 = ant1.reshape(nbl, ntime, nspw)[:, 0, 0]
            ant2 = ant2.reshape(nbl, ntime, nspw)[:, 0, 0]
            spw = spw.reshape(nbl, ntime, nspw)[0, 0, :]
        else:
            # baseline, spw, time
            orig_shape = ["baseline", "spw", "time"]
            assert np.all(spw[:ntime] == spw[0])
            time = time.reshape(nbl, nspw, ntime)[0, 0, :]
            if vals is not None:
                vals = vals.reshape(nbl, nspw, ntime, nfreq, npol)
                flags = flags.reshape(nbl, nspw, ntime, nfreq, npol)
            if swapaxes:
                if vals is not None:
                    vals = vals.swapaxes(1, 2)
                    flags = flags.swapaxes(1, 2)
            ant1 = ant1.reshape(nbl, nspw, ntime)[:, 0, 0]
            ant2 = ant2.reshape(nbl, nspw, ntime)[:, 0, 0]
            spw = spw.reshape(nbl, nspw, ntime)[0, :, 0]

    else:
        assert np.all(spw[: nbl * ntime] == spw[0])
        if np.all(baseline[:ntime] == baseline[0]):
            # spw, baseline, time
            orig_shape = ["spw", "baseline", "time"]
            time = time.reshape(nspw, nbl, ntime)[0, 0, :]
            if vals is not None:
                vals = vals.reshape(nspw, nbl, ntime, nfreq, npol)
                flags = flags.reshape(nspw, nbl, ntime, nfreq, npol)
            if swapaxes:
                if vals is not None:
                    vals = vals.swapaxes(0, 1).swapaxes(1, 2)
                    flags = flags.swapaxes(0, 1).swapaxes(1, 2)
            ant1 = ant1.reshape(nspw, nbl, ntime)[0, :, 0]
            ant2 = ant2.reshape(nspw, nbl, ntime)[0, :, 0]
            spw = spw.reshape(nspw, nbl, ntime)[:, 0, 0]
        else:
            assert np.all(time[:nbl] == time[0])
            # spw, time, bl
            orig_shape = ["spw", "time", "baseline"]
            time = time.reshape(nspw, ntime, nbl)[0, :, 0]
            if vals is not None:
                vals = vals.reshape(nspw, ntime, nbl, nfreq, npol)
                flags = flags.reshape(nspw, ntime, nbl, nfreq, npol)
            if swapaxes:
                if vals is not None:
                    vals = vals.swapaxes(0, 2)
                    flags = flags.swapaxes(0, 2)
            ant1 = ant1.reshape(nspw, ntime, nbl)[0, 0, :]
            ant2 = ant2.reshape(nspw, ntime, nbl)[0, 0, :]
            spw = spw.reshape(nspw, ntime, nbl)[:, 0, 0]

    return time, vals, flags, ant1, ant2, spw, orig_shape


def caltable_to_etcd(msname, calname, caltime, status, pols=None, logger=None):
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
    de = dsa_store.DsaStore()

    if pols is None:
        pols = ["B", "A"]

    try:
        # Complex gains for each antenna.
        amps, tamp, flags, ant1, ant2 = read_caltable(
            f"{msname}_{calname}_gacal", cparam=True
        )
        mask = np.ones(flags.shape)
        mask[flags == 1] = np.nan
        amps = amps * mask

        phase, _tphase, flags, ant1, ant2 = read_caltable(
            f"{msname}_{calname}_gpcal", cparam=True
        )
        mask = np.ones(flags.shape)
        mask[flags == 1] = np.nan
        phase = phase * mask

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
            amps.squeeze(axis=2).squeeze(axis=2), axis=1
        ) * np.nanmedian(phase.squeeze(axis=2).squeeze(axis=2), axis=1)
        tamp = np.median(tamp)
        gaincaltime_offset = (tamp - caltime) * ct.SECONDS_PER_DAY

    except Exception as exc:
        tamp = np.nan
        amps = np.ones((0, len(pols))) * np.nan
        gaincaltime_offset = 0.0
        antenna_order_amps = np.zeros(0, dtype=np.int)
        status = cs.update(
            status,
            cs.GAIN_TBL_ERR |
            cs.INV_GAINAMP_P1 |
            cs.INV_GAINAMP_P2 |
            cs.INV_GAINPHASE_P1 |
            cs.INV_GAINPHASE_P2 |
            cs.INV_GAINCALTIME,
        )
        du.exception_logger(logger, "caltable_to_etcd", exc, throw=False)

    # Delays for each antenna.
    try:
        delays, tdel, flags, antenna_order_delays, ant2 = read_caltable(
            f"{msname}_{calname}_kcal", cparam=False
        )
        mask = np.ones(flags.shape)
        mask[flags == 1] = np.nan
        delays = delays * mask

        # Check the output shapes.
        assert delays.shape[0] == len(antenna_order_delays)
        assert delays.shape[1] == tdel.shape[0]
        assert delays.shape[2] == 1
        assert delays.shape[3] == 1
        assert delays.shape[4] == len(pols)

        delays = np.nanmedian(delays.squeeze(axis=2).squeeze(axis=2), axis=1)
        tdel = np.median(tdel)
        delaycaltime_offset = (tdel - caltime) * ct.SECONDS_PER_DAY

    except Exception as exc:
        tdel = np.nan
        delays = np.ones((0, len(pols))) * np.nan
        delaycaltime_offset = 0.0
        status = cs.update(
            status,
            cs.DELAY_TBL_ERR | cs.INV_DELAY_P1 | cs.INV_DELAY_P2 | cs.INV_DELAYCALTIME,
        )
        antenna_order_delays = np.zeros(0, dtype=np.int)
        du.exception_logger(logger, "caltable_to_etcd", exc, throw=False)

    antenna_order = np.unique(
        np.array([antenna_order_amps, antenna_order_delays]))

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
            gainamp = [None] * len(pols)
            gainphase = [None] * len(pols)

        if antnum in antenna_order_delays:
            idx = np.where(antenna_order_delays == antnum)[0][0]
            for delay in delays[idx, :]:
                if not np.isnan(delay):
                    ant_delay += [int(np.rint(delay))]
                else:
                    ant_delay += [None]
        else:
            ant_delay = [None] * len(pols)

        dd = {
            "ant_num": int(antnum + 1),
            "time": float(caltime),
            "pol": pols,
            "gainamp": gainamp,
            "gainphase": gainphase,
            "delay": ant_delay,
            "calsource": calname,
            "gaincaltime_offset": float(gaincaltime_offset),
            "delaycaltime_offset": float(delaycaltime_offset),
            "sim": False,
            "status": status,
        }
        required_keys = dict(
            {
                "ant_num": [cs.INV_ANTNUM, 0],
                "time": [cs.INV_DELAYCALTIME, 0.0],
                "pol": [cs.INV_POL, ["B", "A"]],
                "calsource": [cs.INV_CALSOURCE, "Unknown"],
                "sim": [cs.INV_SIM, False],
                "status": [cs.UNKNOWN_ERR, 0],
            }
        )
        for key, value in required_keys.items():
            if dd[key] is None:
                print(
                    f"caltable_to_etcd: key {key} must not be None to write to "
                    "etcd"
                )
                status = cs.update(status, value[0])
                dd[key] = value[1]

        for pol in dd["pol"]:
            if pol is None:
                print("caltable_to_etcd: pol must not be None to write to " "etcd")
                status = cs.update(status, cs.INV_POL)
                dd["pol"] = ["B", "A"]
        de.put_dict(f"/mon/cal/{antnum + 1}", dd)


def get_antenna_gains(
        gains: np.ndarray, ant1: np.ndarray, ant2: np.ndarray, antennas: List[str], refant=0
) -> np.ndarray:
    """Calculates antenna gains, g_i, from CASA table of G_ij=1/(g_i g_j*) for `antennas`.

    Currently does not support baseline-based gains.
    Refant only used for baseline-based case.

    Parameters
    ----------
    gains : ndarray
        The gains read in from the CASA gain table. 0th axis is baseline or
        antenna.
    ant1, ant2 : ndarray
        The antenna pair for each entry along the 0th axis in gains.  Antennas are indexed starting
        at 0.
    antennas: List[str]
        The antennas (names, starting at 1) for which to extract gains.
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
    if np.all(ant2 == ant2[0]):
        # Antenna-based solutions already
        ant1 = list(ant1)
        antenna_gains = 1 / gains
        antenna_gains = antenna_gains[[ant1.index(
            int(ant) - 1) for ant in antennas], ...]
        return antenna_gains

    # Baseline-based solutions
    output_shape = list(gains.shape)
    output_shape[0] = len(antennas)
    antenna_gains = np.zeros(tuple(output_shape), dtype=gains.dtype)

    assert len(np.unique(np.concatenate((ant1, ant2)))) == 3, (
        "Baseline-based only supported for trio of antennas")

    for i, ant in enumerate(antennas):
        ant1idxs = np.where(ant1 == ant)[0]
        ant2idxs = np.where(ant2 == ant)[0]
        otheridx = np.where((ant1 != ant) & (ant2 != ant))[0][0]
        # phase
        sign = 1
        idx_phase = np.where((ant1 == ant) & (ant2 == refant))[0]
        if len(idx_phase) == 0:
            idx_phase = np.where((ant2 == refant) & (ant1 == ant))[0]
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
        antenna_gains[i] = (
            np.sqrt(np.abs(g01 * g20 / g21)) *
            np.exp(sign * 1.0j * np.angle(gains[idx_phase]))
        ) ** (-1)
        return antenna_gains


def get_delays(antennas, caltable_prefix, applied_delays):
    r"""Returns the delays to be set in the correlator.

    Based on the calibrated delays and the currently applied delays.

    Parameters
    ----------
    antennas : list
        The antennas to get delays for.
    caltable_prefix: str
        Will open `caltable_prefix`\_kcal to load the delays.
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
        f"{caltable_prefix}_kcal"
    )
    delays = delays.squeeze()
    flags = flags.squeeze()
    print(f"delays: {delays.shape}".format(delays.shape))
    ant1 = list(ant1)
    idx = [ant1.index(ant - 1) for ant in antennas]
    delays = delays[idx]
    flags = flags[idx]
    newdelays = applied_delays - delays
    newdelays = newdelays - np.nanmin(newdelays)
    newdelays = np.rint(newdelays / 2) * 2
    return newdelays.astype(np.int), flags
