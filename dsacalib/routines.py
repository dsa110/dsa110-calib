"""Calibration routine for DSA-110 calibration with CASA.

Author: Dana Simard, dana.simard@astro.caltech.edu, 2020/06
"""
import glob
import os
import shutil

import scipy
import astropy.units as u  # pylint: disable=wrong-import-order
import dsautils.calstatus as cs
import numpy as np
import pandas
from astropy.coordinates import Angle
from astropy.utils import iers  # pylint: disable=wrong-import-order
from casacore.tables import table

import dsacalib.calib as dc
import dsacalib.constants as ct
import dsacalib.fits_io as dfio
import dsacalib.fringestopping as df
import dsacalib.ms_io as dmsio
import dsacalib.plotting as dp
import dsacalib.utils as du
from dsacalib.ms_io import extract_vis_from_ms

iers.conf.iers_auto_url_mirror = ct.IERS_TABLE
iers.conf.auto_max_age = None
from astropy.time import Time  # pylint: disable=wrong-import-position


def __init__():
    return


def _check_path(fname):
    """Raises an AssertionError if the path `fname` does not exist.

    Parameters
    ----------
    fname : str
        The file to check existence of.
    """
    assert os.path.exists(fname), "File {0} does not exist".format(fname)


def triple_antenna_cal(
    obs_params, ant_params, throw_exceptions=True, sefd=False, logger=None
):
    r"""Calibrate visibilities from 3 antennas.

    Assumes visbilities are stored using dsa-10 or dsa-110 fits format.
    The caltable\_to\_etcd function should be able to handle this, but I haven't
    tested that yet.

    Parameters
    ----------
    obs_params : dict
        Observing parameters
    ant_params : dict
    show_plots : Boolean
        If set to ``True``, plots of the delay and gain calibration solutions
        will be shown. Defaults ``False``.
    throw_exception : Boolean
        If set to ``True``, exceptions will be thrown after being logged in
        syslog. If set to ``False``, the exceptions will not be thrown, but
        will still be logged in syslog. Defaults ``True``.
    sefd : Boolean
        If set to ``True``, enough data (60 minutes) will be included in the
        measurement set to calculate the off-source power (60 minutes) and the
        calibration solutions will be solved against a model of ones. If set to
        ``False``, only 10 minutes will be included in the measurement set and
        the calibration solutison will be solved against a sky model.
    logger : dsautils.dsa_syslog.DsaSyslogger() instance
        Logger to write messages too. If None, messages are printed.

    Returns
    -------
    status : int
        The status code of the pipeline. Decode with dsautils.calstatus.
    caltime : float
        The meridian crossing time of the source in MJD. If the input file
        could not be opened, ``None`` will be returned.
    """
    # TODO: Only keep one of the gain tables in the end, on a fine timescale.
    status = 0
    current_error = cs.UNKNOWN_ERR
    calstring = "initialization"

    try:
        fname = obs_params["fname"]
        msname = obs_params["msname"]
        cal = obs_params["cal"]
        utc_start = obs_params["utc_start"]
        pt_dec = ant_params["pt_dec"]
        antenna_order = ant_params["antenna_order"]
        refant = ant_params["refant"]
        antpos = ant_params["antpos"]

        # Remove files that we will create so that things will fail if casa
        # doesn't write a table.
        casa_dirnames = [
            "{0}.ms".format(msname),
            "{0}_{1}_kcal".format(msname, cal.name),
            "{0}_{1}_2kcal".format(msname, cal.name),
            "{0}_{1}_bcal".format(msname, cal.name),
            "{0}_{1}_gpcal".format(msname, cal.name),
            "{0}_{1}_gacal".format(msname, cal.name),
            "{0}_{1}_gcal_ant".format(msname, cal.name),
        ]
        for dirname in casa_dirnames:
            if os.path.exists(dirname):
                shutil.rmtree(dirname)

        calstring = "opening visibility file"
        current_error = (
            cs.INFILE_ERR
            | cs.INV_ANTNUM
            | cs.INV_GAINAMP_P1
            | cs.INV_GAINAMP_P2
            | cs.INV_GAINPHASE_P1
            | cs.INV_GAINPHASE_P2
            | cs.INV_DELAY_P1
            | cs.INV_DELAY_P2
            | cs.INV_GAINCALTIME
            | cs.INV_DELAYCALTIME
        )
        caldur = 60 * u.min if sefd else 10 * u.min
        (
            fobs,
            blen,
            bname,
            _tstart,
            _tstop,
            tsamp,
            vis,
            mjd,
            lst,
            transit_idx,
            antenna_order,
        ) = dfio.read_psrfits_file(
            fname,
            cal,
            antenna_order=antenna_order,
            autocorrs=True,
            dur=caldur,
            utc_start=utc_start,
            dsa10=False,
            antpos=antpos,
        )
        caltime = mjd[transit_idx]

        calstring = "read and verification of visibility file"
        current_error = (
            cs.CAL_MISSING_ERR
            | cs.INV_GAINAMP_P1
            | cs.INV_GAINAMP_P2
            | cs.INV_GAINPHASE_P1
            | cs.INV_GAINPHASE_P2
            | cs.INV_DELAY_P1
            | cs.INV_DELAY_P2
            | cs.INV_GAINCALTIME
            | cs.INV_DELAYCALTIME
        )

        nt = vis.shape[1]
        assert nt > 0, "calibrator not in file"
        current_error = (
            cs.INFILE_FORMAT_ERR
            | cs.INV_GAINAMP_P1
            | cs.INV_GAINAMP_P2
            | cs.INV_GAINPHASE_P1
            | cs.INV_GAINPHASE_P2
            | cs.INV_DELAY_P1
            | cs.INV_DELAY_P2
            | cs.INV_GAINCALTIME
            | cs.INV_DELAYCALTIME
        )

        nant = len(antenna_order)
        assert nant == 3, "triple_antenna_cal only works with a triplet of " "antennas"
        assert int(refant) in antenna_order, "refant {0} not in " "visibilities".format(
            refant
        )

        calstring = "flagging of ms data"
        current_error = (
            cs.FLAGGING_ERR
            | cs.INV_GAINAMP_P1
            | cs.INV_GAINAMP_P2
            | cs.INV_GAINPHASE_P1
            | cs.INV_GAINPHASE_P2
            | cs.INV_DELAY_P1
            | cs.INV_DELAY_P2
            | cs.INV_GAINCALTIME
            | cs.INV_DELAYCALTIME
        )
        #         maskf, _fraction_flagged = du.mask_bad_bins(
        #             vis,
        #             axis=2,
        #             thresh=2.0,
        #             # medfilt=True, # currently not supported
        #             nmed=129
        #         )
        #         maskt, _fraction_flagged = du.mask_bad_bins(
        #             vis,
        #             axis=1,
        #             thresh=2.0,
        #             # medfilt=True, # currently not supported
        #             nmed=129
        #         )
        maskp, _fraction_flagged = du.mask_bad_pixels(
            vis,
            thresh=6.0,
            # mask=maskt*maskf
        )
        #         mask = maskt*maskf*maskp
        #         vis *= mask
        vis *= maskp

        calstring = "fringestopping"
        current_error = (
            cs.FRINGES_ERR
            | cs.INV_GAINAMP_P1
            | cs.INV_GAINAMP_P2
            | cs.INV_GAINPHASE_P1
            | cs.INV_GAINPHASE_P2
            | cs.INV_DELAY_P1
            | cs.INV_DELAY_P2
            | cs.INV_GAINCALTIME
            | cs.INV_DELAYCALTIME
        )
        df.fringestop(vis, blen, cal, mjd, fobs, pt_dec)

        calstring = "writing to ms"
        current_error = (
            cs.MS_WRITE_ERR
            | cs.INV_GAINAMP_P1
            | cs.INV_GAINAMP_P2
            | cs.INV_GAINPHASE_P1
            | cs.INV_GAINPHASE_P2
            | cs.INV_DELAY_P1
            | cs.INV_DELAY_P2
            | cs.INV_GAINCALTIME
            | cs.INV_DELAYCALTIME
        )
        amp_model = df.amplitude_sky_model(cal, lst, pt_dec, fobs)
        amp_model = np.tile(
            amp_model[np.newaxis, :, :, np.newaxis], (vis.shape[0], 1, 1, vis.shape[-1])
        )
        dmsio.convert_to_ms(
            cal,
            vis,
            mjd[0],
            "{0}".format(msname),
            bname,
            antenna_order,
            tsamp,
            nint=25,
            antpos=antpos,
            dsa10=False,
            model=None if sefd else amp_model,
        )
        _check_path("{0}.ms".format(msname))

        calstring = "flagging of ms data"
        current_error = (
            cs.FLAGGING_ERR
            | cs.INV_GAINAMP_P1
            | cs.INV_GAINAMP_P2
            | cs.INV_GAINPHASE_P1
            | cs.INV_GAINPHASE_P2
            | cs.INV_DELAY_P1
            | cs.INV_DELAY_P2
            | cs.INV_GAINCALTIME
            | cs.INV_DELAYCALTIME
        )
        error = dc.flag_zeros(msname)
        if error > 0:
            status = cs.update(status, ["flagging_err"])
            message = "Non-fatal error in zero flagging"
            if logger is not None:
                logger.info(message)
            else:
                print(message)
        if 8 in antenna_order:
            error = dc.flag_antenna(msname, "8", pol="A")
            if error > 0:
                status = cs.update(status, ["flagging_err"])
                message = "Non-fatal error in antenna 8 flagging"
                if logger is not None:
                    logger.info(message)
                else:
                    print(message)

        # Antenna-based delay calibration
        calstring = "delay calibration"
        current_error = (
            cs.DELAY_CAL_ERR
            | cs.INV_GAINAMP_P1
            | cs.INV_GAINAMP_P2
            | cs.INV_GAINPHASE_P1
            | cs.INV_GAINPHASE_P2
            | cs.INV_DELAY_P1
            | cs.INV_DELAY_P2
            | cs.INV_GAINCALTIME
            | cs.INV_DELAYCALTIME
        )
        error = dc.delay_calibration(msname, cal.name, refants=[refant])
        if error > 0:
            status = cs.update(status, ["delay_cal_err"])
            message = "Non-fatal error occured in delay calibration."
            if logger is not None:
                logger.info(message)
            else:
                print(message)
        _check_path("{0}_{1}_kcal".format(msname, cal.name))

        calstring = "flagging of ms data"
        current_error = (
            cs.FLAGGING_ERR
            | cs.INV_GAINAMP_P1
            | cs.INV_GAINAMP_P2
            | cs.INV_GAINPHASE_P1
            | cs.INV_GAINPHASE_P2
            | cs.INV_GAINCALTIME
        )
        if error > 0:
            status = cs.update(status, ["flagging_err"])
            message = (
                "Non-fatal error occured in calculation of delays on short timescales."
            )
            if logger is not None:
                logger.info(message)
            else:
                print(message)
        if error > 0:
            status = cs.update(status, ["flagging_err"])
            message = "Non-fatal error occured in flagging of bad timebins"
            if logger is not None:
                logger.info(message)
            else:
                print(message)
        _check_path("{0}_{1}_2kcal".format(msname, cal.name))
        calstring = "baseline-based bandpass and gain calibration"
        current_error = (
            cs.GAIN_BP_CAL_ERR
            | cs.INV_GAINAMP_P1
            | cs.INV_GAINAMP_P2
            | cs.INV_GAINPHASE_P1
            | cs.INV_GAINPHASE_P2
            | cs.INV_GAINCALTIME
        )
        error = dc.calibrate_gain(
            msname,
            cal.name,
            "{0}_{1}".format(msname, cal.name),
            refant,
            tga="inf",
            tgp="inf",
            blbased=True,
            combined=False,
        )
        if error > 0:
            status = cs.update(status, ["gain_bp_cal_err"])
            message = "Non-fatal error occured in gain/bandpass calibration."
            if logger is not None:
                logger.info(message)
            else:
                print(message)
        for fname in [
            "{0}_{1}_bcal".format(msname, cal.name),
            "{0}_{1}_gpcal".format(msname, cal.name),
            "{0}_{1}_gacal".format(msname, cal.name),
        ]:
            _check_path(fname)
        calstring = "calculation of antenna gains"
        gamp, _tamp, famp, _ant1, _ant2 = dmsio.read_caltable(
            "{0}_{1}_gacal".format(msname, cal.name), cparam=True
        )
        gphase, _tphase, fphase, _ant1, _ant2 = dmsio.read_caltable(
            "{0}_{1}_gpcal".format(msname, cal.name), cparam=True
        )
        gains = (gamp * gphase).squeeze(axis=2)
        flags = (famp * fphase).squeeze(axis=2)
        # This makes some assumptions about the bl order! Should add some
        # statements to make sure it's true
        gains, flags = dc.fill_antenna_gains(gains, flags)

        # These tables will contain the results on fine time-scales.
        gamp = np.abs(gains).astype(np.complex128)
        gamp = gamp.reshape(gamp.shape[0], -1)
        # tb = cc.table()
        with table("{0}_{1}_gacal".format(msname, cal.name), readonly=False) as tb:
            shape = np.array(tb.CPARAM[:]).shape
            tb.putcol("CPARAM", gamp.reshape(shape))

        gphase = np.exp(1.0j * np.angle(gains))
        with table("{0}_{1}_gpcal".format(msname, cal.name), readonly=False) as tb:
            shape = np.array(tb.CPARAM[:]).shape
            tb.putcol("CPARAM", gphase.reshape(shape))

        if not sefd:
            # reduce to a single value to use
            mask = np.ones(flags.shape)
            mask[flags == 1] = np.nan
            gains = np.nanmedian(gains * mask, axis=1, keepdims=True)
            flags = np.min(flags, axis=1, keepdims=True)

            if 8 in antenna_order:
                flags[..., 0] = 1

            shutil.copytree(
                "{0}/template_gcal_ant".format(ct.PKG_DATA_PATH),
                "{0}_{1}_gcal_ant".format(msname, cal.name),
            )

            # Write out a new gains that is a single value.
            with table(
                "{0}_{1}_gcal_ant".format(msname, cal.name), readonly=False
            ) as tb:
                tb.putcol("TIME", np.ones(6) * np.median(mjd) * ct.SECONDS_PER_DAY)
                tb.putcol("FLAG", flags.squeeze(axis=1))
                tb.putcol("CPARAM", gains.squeeze(axis=1))
            _check_path("{0}_{1}_gcal_ant".format(msname, cal.name))

    except Exception as exc:
        status = cs.update(status, current_error)
        du.exception_logger(logger, calstring, exc, throw_exceptions)
        try:
            caltime
        except NameError:
            caltime = Time.now().mjd

    return status, caltime


def plot_solutions(msname, calname, figure_path, show_plots=False, logger=None):
    r"""Plots the antenna delay, gain and bandpass calibration solutions.

    Creates separate files for all solutions.  To create one plot with all
    solutions, use plotting.summary_plot.

    Parameters
    ----------
    msname : str
        The name of the measurement set. Used to identify the calibration
        tables.
    calname : str
        The name of the calibrator. Used to identify the calibration tables.
    antenna_order : list
        The antenna names, in order.
    fobs : array
        The central frequency of each channel, in GHz.
    blbased : boolean
        True of the calibration was baseline-based.
    figure_dir : str
        The location to save the figures.  Defaults ``./figures``.
    show_plots : boolean
        If False, plots are closed after being saved. Defaults False.
    logger : dsautils.dsa_syslog.DsaSyslogger() instance
        Logger to write messages too. If None, messages are printed.
    """
    try:
        _ = dp.plot_antenna_delays(
            msname, calname, outname=figure_path, show=show_plots
        )
    except RuntimeError:
        message = "Plotting antenna delays failed for {0}".format(msname)
        if logger is not None:
            logger.info(message)
        else:
            print(message)
    try:
        _ = dp.plot_gain_calibration(
            msname, calname, outname=figure_path, show=show_plots
        )
    except RuntimeError:
        message = "Plotting gain calibration solutions failed for {0}".format(msname)
        if logger is not None:
            logger.info(message)
        else:
            print(message)
    try:
        _ = dp.plot_bandpass(msname, calname, outname=figure_path, show=show_plots)
    except RuntimeError:
        message = "Plotting bandpass calibration solutions failed for {0}".format(
            msname
        )
        if logger is not None:
            logger.info(message)
        else:
            print(message)


def calibration_head(
    obs_params,
    ant_params,
    write_to_etcd=False,
    throw_exceptions=None,
    sefd=False,
    logger=None,
):
    """Controls calibrtion of a dsa10 or dsa110 dataset.

    After calibration, results are writen to etcd.

    Parameters
    ----------
    obs_params : list
        The observing parameters.
    ant_params : list
        The antenna configuration.
    write_to_etcd : boolean
        If set to ``True``, the results of the calibration are pushed to etcd.
        Defaults ``False``.
    throw_exceptions : boolean
        If set to ``False``, exceptions are not raised after being logged to
        syslog. Instead, `calibration_head` and `triple_antenna_cal` return the
        status value. If set to ``None``, `throw_exceptions` will be set to
        ``not write_to_etcd``.
    sefd : boolean
        If set to ``True``, the solutions will be solved against a model of
        ones in order to allow fitting of the source pass to the antenna gains
        and 60 minutes will be saved to the measurement set.  If set to
        ``False``, a sky model will be used in calibration and only 10 minutes
        of data is saved to the measurement set.
    logger : dsautils.dsa_syslog.DsaSyslogger() instance
        Logger to write messages too. If None, messages are printed.

    Returns
    -------
    int
        The status code. Decode with dsautils.calstatus.
    """
    if throw_exceptions is None:
        throw_exceptions = not write_to_etcd
    message = (
        "Beginning calibration of ms {0}.ms (start time {1}) using source {2}".format(
            obs_params["msname"], obs_params["utc_start"].isot, obs_params["cal"].name
        )
    )
    if logger is not None:
        logger.info(message)
    else:
        print(message)
    status, caltime = triple_antenna_cal(
        obs_params, ant_params, throw_exceptions, sefd, logger=logger
    )
    message = "Ending calibration of ms {0}.ms (start time {1}) using source {2} with status {3}".format(
        obs_params["msname"],
        obs_params["utc_start"].isot,
        obs_params["cal"].name,
        status,
    )
    if logger is not None:
        logger.info(message)
    else:
        print(message)
    print("Status: {0}".format(cs.decode(status)))
    print("")
    if write_to_etcd:
        dmsio.caltable_to_etcd(
            obs_params["msname"],
            obs_params["cal"].name,
            ant_params["antenna_order"],
            caltime,
            status,
            logger=logger,
        )
    return status


def dsa10_cal(fname, msname, cal, pt_dec, antpos, refant, badants=None):
    """Calibrate dsa10 data.

    Parameters
    ----------
    fname : str
        The fits file containing the correlated dsa10 data.
    msname : str
        The measurement set containing the correlated dsa10 data.
    cal : dsautils.src instance
        The calibrator source.
    pt_dec : float
        The pointing declination of the array in radians.
    antpos : str
        The path to the ITRF file containing the antenna positions.
    refant : str or int
        The reference antenna name (if str) or index (if int).
    badants : list(str)
        The naems of antennas that should be flagged before calibration.
    """
    # TODO: get header information from the ms instead of the fits file.
    if badants is None:
        badants = []

    for file_path in [
        "{0}.ms".format(msname),
        "{0}_{1}_kcal".format(msname, cal.name),
        "{0}_{1}_gacal".format(msname, cal.name),
        "{0}_{1}_gpcal".format(msname, cal.name),
        "{0}_{1}_bcal".format(msname, cal.name),
        "{0}_{1}_2kcal".format(msname, cal.name),
    ]:
        if os.path.exists(file_path):
            shutil.rmtree(file_path)

    (
        fobs,
        blen,
        bname,
        tstart,
        _tstop,
        tsamp,
        vis,
        mjd,
        lst,
        _transit_idx,
        antenna_order,
    ) = dfio.read_psrfits_file(
        fname, cal, dur=10 * u.min, antpos=antpos, badants=badants
    )

    df.fringestop(vis, blen, cal, mjd, fobs, pt_dec)
    amp_model = df.amplitude_sky_model(cal, lst, pt_dec, fobs)
    amp_model = np.tile(
        amp_model[np.newaxis, :, :, np.newaxis], (vis.shape[0], 1, 1, vis.shape[-1])
    )

    dmsio.convert_to_ms(
        cal,
        vis,
        tstart,
        msname,
        bname,
        antenna_order,
        tsamp=tsamp,
        nint=25,
        antpos=antpos,
        model=amp_model,
    )
    _check_path("{0}.ms".format(msname))

    dc.flag_zeros(msname)
    if "8" in antenna_order:
        dc.flag_antenna(msname, "8", pol="A")

    dc.delay_calibration(msname, cal.name, [refant])
    _check_path("{0}_{1}_kcal".format(msname, cal.name))

    dc.gain_calibration(msname, cal.name, refant=refant, forsystemhealth=True)
    for tbl in ["gacal", "gpcal", "bcal"]:
        _check_path("{0}_{1}_{2}".format(msname, cal.name, tbl))


def flag_pixels(msname, thresh=6.0, logger=None):
    """Flags pixels using dsautils.mask_bad_pixels.

    Parameters
    ----------
    msname : str
        The path to the measurement set. Opens `msname`.ms
    thresh : float
        The RFI threshold in units of standard deviation. Anything above
        thresh*stddev + mean will be flagged.
    """
    # Flag RFI - only for single spw
    vis, _, _, flags, ant1, ant2, _, _, orig_shape = extract_vis_from_ms(
        msname,
    )
    good_pixels, fraction_flagged = du.mask_bad_pixels(
        vis.squeeze(2), mask=~flags.squeeze(2), thresh=thresh
    )

    # # Not properly account for shape - getting repeat messages
    # (idx1s, idx2s) = np.where(fraction_flagged > 0.3)
    # for idx1 in idx1s:
    #     for idx2 in idx2s:
    #         message = \
    #             'Baseline {0}-{1} {2}: {3} percent of data flagged'.format(
    #                 ant1[idx1],
    #                 ant2[idx1],
    #                 'A' if idx2==1 else 'B',
    #                 fraction_flagged[idx1, idx2]*100
    #             )
    #         if logger is not None:
    #             logger.info(message)
    #         else:
    #             print(message)

    flags = flags + ~good_pixels[:, :, np.newaxis, :, :]
    if orig_shape[0] == "time":
        flags = flags.swapaxes(0, 1)
    with table("{0}.ms".format(msname), readonly=False) as tb:
        shape = np.array(tb.getcol("FLAG")[:]).shape
        tb.putcol("FLAG", flags.reshape(shape))


def flag_antennas_using_delays(
    antenna_delays, kcorr, msname, kcorr_thresh=0.3, logger=None
):
    """Flags antennas by comparing the delay on short times to the delay cal.

    Parameters
    ----------
    antenna_delays : ndarray
        The antenna delays from the 2kcal calibration file, calculated on short
        timescales.
    kcorr : ndarray
        The antenna delays from the kcal calibration file, calculated over the
        entire calibration pass.
    msname : str
        The path to the measurement set. Will open `msname`.ms
    kcorr_thresh : float
        The tolerance for descrepancies between the antenna_delays and kcorr,
        in nanoseconds.
    logger : dsautils.dsa_syslog.DsaSyslogger() instance
        Logger to write messages too. If None, messages are printed.
    """
    error = 0
    percent_bad = (np.abs(antenna_delays - kcorr) > 1).sum(1).squeeze(1).squeeze(
        1
    ) / antenna_delays.shape[1]
    for i in range(percent_bad.shape[0]):
        for j in range(percent_bad.shape[1]):
            if percent_bad[i, j] > kcorr_thresh:
                error += not dc.flag_antenna(
                    msname, "{0}".format(i + 1), pol="A" if j == 0 else "B"
                )
                message = "Flagged antenna {0}{1} in {2}".format(
                    i + 1, "A" if j == 0 else "B", msname
                )
                if logger is not None:
                    logger.info(message)
                else:
                    print(message)
    return error
