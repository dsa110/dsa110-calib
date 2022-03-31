"""Calibration routine for DSA-110 calibration with CASA.

Author: Dana Simard, dana.simard@astro.caltech.edu, 2020/06
"""
import glob
import os
import shutil

import scipy # pylint: disable=unused-import
import dsautils.calstatus as cs
import numpy as np
import pandas
import astropy.units as u
from astropy.coordinates import Angle
from astropy.utils import iers
from casacore.tables import table

import dsacalib.calib as dc
import dsacalib.constants as ct
import dsacalib.plotting as dp
import dsacalib.utils as du

iers.conf.iers_auto_url_mirror = ct.IERS_TABLE
iers.conf.auto_max_age = None
from astropy.time import Time  # pylint: disable=wrong-import-order,ungrouped-imports,wrong-import-position


def __init__():
    return


def calibrate_measurement_set(
        msname: str,
        cal: du.src,
        refants: list,
        throw_exceptions: bool=True,
        bad_antennas: bool=None,
        bad_uvrange: str="2~50m",
        keepdelays: bool=False,
        forsystemhealth: bool=False,
        interp_thresh: float=1.5,
        interp_polyorder: int=7,
        blbased: bool=False,
        manual_flags: str=None,
        logger: "DsaSyslogger"=None,
        t2: str="60s",
) -> int:
    r"""Calibrates the measurement set.

    Calibration can be done with the aim of monitoring system health (set
    `forsystemhealth=True`), obtaining beamformer weights (set
    `forsystemhealth=False` and `keepdelays=False`), or obtaining delays (set
    `forsystemhealth=False` and `keepdelays=True`, new beamformer weights will
    be generated as well).

    Parameters
    ----------
    msname : str
        The name of the measurement set. Will open `msname`.ms
    cal : dsacalib.utils.src instance
        The calibration source. Calibration tables will begin with
        `msname`\_`cal.name`
    refant : str or int
        The reference antenna name (if str) or index (if int) for calibration.
    throw_exceptions : bool
        If set to False, exceptions will not be thrown, although they will be
        logged to syslog. Defaults True.
    bad_antennas : list(str)
        Antennas (names) to be flagged before calibration.
    bad_uvrange : str
        Baselines with lengths within bad_uvrange will be flagged before
        calibration. Must be a casa-understood string with units.
    keepdelays : bool
        Only used if `forsystemhealth` is False. If `keepdelays` is set to
        False and `forsystemhealth` is set to False, then delays are integrated
        into the bandpass solutions and the kcal table is set to all zeros. If
        `keepdelays` is set to True and `forsystemhealth` is set to False, then
        delays are kept at 2 nanosecond resolution.  If `forsystemhealth` is
        set to True, delays are kept at full resolution regardless of the
        keepdelays parameter. Defaults False.
    forsystemhealth : bool
        Set to True for full-resolution delay and bandpass solutions to use to
        monitor system health, or to False to generate beamformer weights and
        delays. Defaults False.
    interp_thresh: float
        Used if `forsystemhealth` is False, when smoothing bandpass gains.
        The gain amplitudes and phases are fit using a polynomial after any
        points more than interp_thresh*std away from the median-filtered trend
        are flagged.
    interp_polyorder : int
        Used if `forsystemhealth` is False, when smoothing bandpass gains.
        The gain amplitudes and phases are fit using a polynomial of order
        interp_polyorder.
    blbased : boolean
        Set to True for baseline-based calibration, False for antenna-based
        calibration.
    manual_flags : list(str)
        Include any additional flags to be done prior to calibration, as
        CASA-understood strings.
    logger : dsautils.dsa_syslog.DsaSyslogger() instance
        Logger to write messages too. If None, messages are printed.

    Returns
    -------
    int
        A status code. Decode with dsautils.calstatus
    """
    if isinstance(refants, (int, str)):
        refant = refants
        refants = [refant]
    else:
        refant = refants[0]

    print("entered calibration")
    status = 0
    current_error = cs.UNKNOWN_ERR
    calstring = "initialization"

    try:
        # Remove files that we will create so that things will fail if casa
        # doesn't write a table.
        print("removing files")
        tables_to_remove = [
            f"{msname}_{cal.name}_2kcal",
            f"{msname}_{cal.name}_kcal",
            f"{msname}_{cal.name}_bkcal",
            f"{msname}_{cal.name}_gacal",
            f"{msname}_{cal.name}_gpcal",
            f"{msname}_{cal.name}_bcal",
        ]
        if forsystemhealth:
            tables_to_remove += [f"{msname}_{cal.name}_2gcal"]
        for path in tables_to_remove:
            if os.path.exists(path):
                shutil.rmtree(path)

        print("flagging of ms data")
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
        print("resetting flags")
        # Reset flags in the measurement set
        dc.reset_flags(msname, datacolumn="data")
        dc.reset_flags(msname, datacolumn="model")
        dc.reset_flags(msname, datacolumn="corrected")

        print("flagging baselines")
        current_error = cs.FLAGGING_ERR
        error = dc.flag_baselines(msname, uvrange=bad_uvrange)
        if error > 0:
            message = f"Non-fatal error occured in flagging short baselines of {msname}."
            if logger is not None:
                logger.warning(message)
            else:
                print(message)
        print("flagging zeros")
        error = dc.flag_zeros(msname)
        if error > 0:
            message = f"Non-fatal error occured in flagging zeros of {msname}."
            if logger is not None:
                logger.warning(message)
            else:
                print(message)
        print("flagging antennas")
        if bad_antennas is not None:
            for ant in bad_antennas:
                error = dc.flag_antenna(msname, ant)
                if error > 0:
                    message = f"Non-fatal error occured in flagging ant {ant} of {msname}."
                    if logger is not None:
                        logger.warning(message)
                    else:
                        print(message)
        if manual_flags is not None:
            for entry in manual_flags:
                dc.flag_manual(msname, entry[0], entry[1])
        print("flagging rfi")
        dc.flag_rfi(msname)
        if error > 0:
            message = f"Non-fatal error occured in flagging bad pixels of {msname}."
            if logger is not None:
                logger.warning(message)
            else:
                print(message)

        print("delay cal")
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
        error = dc.delay_calibration(msname, cal.name, refants=refants, t2=t2)
        if error > 0:
            status = cs.update(status, cs.DELAY_CAL_ERR)
            message = f"Non-fatal error occured in delay calibration of {msname}."
            if logger is not None:
                logger.warning(message)
            else:
                print(message)
        _check_path(f"{msname}_{cal.name}_kcal")

        print("flagging based on delay cal")
        calstring = "flagging of ms data"
        current_error = (
            cs.FLAGGING_ERR
            | cs.INV_GAINAMP_P1
            | cs.INV_GAINAMP_P2
            | cs.INV_GAINPHASE_P1
            | cs.INV_GAINPHASE_P2
            | cs.INV_GAINCALTIME
        )
        _times, antenna_delays, kcorr, _ant_nos = dp.plot_antenna_delays(
            msname, cal.name, show=False
        )
        error += dc.flag_antennas_using_delays(antenna_delays, kcorr, msname)
        if error > 0:
            status = cs.update(status, cs.FLAGGING_ERR)
            message = f"Non-fatal error occured in flagging of bad timebins on {msname}"
            if logger is not None:
                logger.warning(message)
            else:
                print(message)
        try:
            _check_path(f"{msname}_{cal.name}_2kcal")
        except AssertionError:
            status = cs.update(status, cs.FLAGGING_ERR)
            message = f"Non-fatal error occured in flagging of bad timebins on {msname}"
            if logger is not None:
                logger.warning(message)
            else:
                print(message)
        
        print("delay cal again")
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
        shutil.rmtree(f"{msname}_{cal.name}_kcal")
        shutil.rmtree(f"{msname}_{cal.name}_2kcal")
        error = dc.delay_calibration(msname, cal.name, refants=refants, t2=t2)
        if error > 0:
            status = cs.update(status, cs.DELAY_CAL_ERR)
            message = f"Non-fatal error occured in delay calibration of {msname}."
            if logger is not None:
                logger.warning(message)
            else:
                print(message)
        _check_path(f"{msname}_{cal.name}_kcal")

        print("bandpass and gain cal")
        calstring = "bandpass and gain calibration"
        current_error = (
            cs.GAIN_BP_CAL_ERR
            | cs.INV_GAINAMP_P1
            | cs.INV_GAINAMP_P2
            | cs.INV_GAINPHASE_P1
            | cs.INV_GAINPHASE_P2
            | cs.INV_GAINCALTIME
        )
        

    except Exception as exc:
        status = cs.update(status, current_error)
        du.exception_logger(logger, calstring, exc, throw_exceptions)
    print("end of cal routine")
    return status

@persistent(
    "bandpass and gain calibration",
    (
        cs.GAIN_BP_CAL_ERR
        | cs.INV_GAINAMP_P1
        | cs.INV_GAINAMP_P2
        | cs.INV_GAINPHASE_P1
        | cs.INV_GAINPHASE_P2
        | cs.INV_GAINCALTIME
    ),
    cs.GAIN_BP_CAL_ERR
)
def bandpass_and_gain_cal(msname, cal, refant ):
    error = dc.gain_calibration(
            msname,
            cal.name,
            refant,
            blbased=blbased,
            forsystemhealth=forsystemhealth,
            keepdelays=keepdelays,
            interp_thresh=interp_thresh,
            interp_polyorder=interp_polyorder,
            tbeam=t2,
        )
        if error > 0:
            status = cs.update(status, cs.GAIN_BP_CAL_ERR)
            message = f"Non-fatal error occured in gain/bandpass calibration of {msname}."
            if logger is not None:
                logger.warning(message)
            else:
                print(message)
        fnames = [
            f"{msname}_{cal.name}_bcal",
            f"{msname}_{cal.name}_bacal",
            f"{msname}_{cal.name}_bpcal",
            f"{msname}_{cal.name}_gpcal",
            f"{msname}_{cal.name}_gacal",
        ]
        if forsystemhealth:
            fnames += [f"{msname}_{cal.name}_2gcal"]
        if not keepdelays and not forsystemhealth:
            fnames += [f"{msname}_{cal.name}_bkcal"]
        for fname in fnames:
            _check_path(fname)
        print("combining bandpass and delay solns")
        # Combine bandpass solutions and delay solutions
        with table(f"{msname}_{cal.name}_bacal") as tb:
            bpass = np.array(tb.CPARAM[:])
        with table(f"{msname}_{cal.name}_bpcal") as tb:
            bpass *= np.array(tb.CPARAM[:])
        if not forsystemhealth:
            with table(f"{msname}_{cal.name}_bkcal") as tb:
                bpass = np.array(tb.CPARAM[:])
        with table(f"{msname}_{cal.name}_bcal", readonly=False) as tb:
            tb.putcol("CPARAM", bpass)
            if not forsystemhealth:
                tbflag = np.array(tb.FLAG[:])
                tb.putcol("FLAG", np.zeros(tbflag.shape, tbflag.dtype))


def persistent(target: "Callable", name: str, errorcode: int, nonfatal_errorcode: int=None):
    @wraps(target)
    def inner(status, *args, **kwargs):
        try:
            error = target(*args, **kwargs)
        except Exception as exc:
            status = cs.update(status, errorcode)
            du.exception_logger(logger, name, exc, throw_exceptions)
            error = 0
        if error:
            status = cs.update(status, nonfatal_errorcode)
            message = f'Non-fatal error occured in {name}.'
            info_logger(logger, message)
        return status
    return


def _check_path(fname: str) -> None:
    """Raises an AssertionError if the path `fname` does not exist.

    Parameters
    ----------
    fname : str
        The file to check existence of.
    """
    assert os.path.exists(fname), f"File {fname} does not exist"


def plot_solutions(
        msname: str, calname: str, figure_path: str, show_plots: bool=False,
        logger: "DsaSyslogger"=None
) -> None:
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
        message = f"Plotting antenna delays failed for {msname}"
        if logger is not None:
            logger.info(message)
        else:
            print(message)
    try:
        _ = dp.plot_gain_calibration(
            msname, calname, outname=figure_path, show=show_plots
        )
    except RuntimeError:
        message = f"Plotting gain calibration solutions failed for {msname}"
        if logger is not None:
            logger.info(message)
        else:
            print(message)
    try:
        _ = dp.plot_bandpass(msname, calname, outname=figure_path, show=show_plots)
    except RuntimeError:
        message = f"Plotting bandpass calibration solutions failed for {msname}"
        if logger is not None:
            logger.info(message)
        else:
            print(message)


def cal_in_datetime(
        dt: str, transit_time: "Time", duration: "Quantity" = 5*u.min,
        filelength: "Quantity" = 15*u.min
) -> bool:
    """Check to see if a transit is in a given file.

    Parameters
    ----------
    dt : str
        The start time of the file, given as an isot string.
        E.g. '2020-10-06T23:19:02'
    transit_time : astropy.time.Time instance
        The transit time of the source.
    duration : astropy quantity
        The amount of time around transit you are interested in, in minutes or
        seconds.
    filelength : astropy quantity
        The length of the hdf5 file, in minutes or seconds.

    Returns
    -------
    bool
        True if at least part of the transit is within the file, else False.
    """
    filestart = Time(dt)
    fileend = filestart + filelength
    transitstart = transit_time - duration / 2
    transitend = transit_time + duration / 2

    # For any of these conditions,
    # the file contains data that we want
    if (filestart < transitstart) and (fileend > transitend):
        transit_file = True
    elif (filestart > transitstart) and (fileend < transitend):
        transit_file = True
    elif (fileend > transitstart) and (fileend - transitstart < duration):
        transit_file = True
    elif (filestart < transitend) and (transitend - filestart) < duration:
        transit_file = True
    else:
        transit_file = False
    return transit_file


def get_files_for_cal(
        caltable: str,
        refcorr: str = "01",
        duration: "Quantity" = 5*u.min,
        filelength: "Quantity" = 15*u.min,
        hdf5dir: str = "/mnt/data/dsa110/correlator/",
        date_specifier: str = "*"
) -> dict:
    """Returns a dictionary containing the filenames for each calibrator pass.

    Parameters
    ----------
    caltable : str
        The path to the csv file containing calibrators of interest.
    refcorr : str
        The reference correlator to search for recent hdf5 files from. Searches
        the directory `hdf5dir`/corr`refcorr`/
    duration : astropy quantity
        The duration around transit which you are interested in extracting, in
        minutes or seconds.
    filelength : astropy quantity
        The length of the hdf5 files, in minutes or seconds.
    hdf5dir : str
        The path to the hdf5 files.
    date_specifier : str
        A specifier to include to limit the dates for which you are interested
        in. Should be something interpretable by glob and should be to the
        second precision. E.g. `2020-10-06*`, `2020-10-0[678]*` and
        `2020-10-06T01:03:??` are all valid.

    Returns
    -------
    dict
        A dictionary specifying the hdf5 filenames that correspond to the
        requested datesand calibrators.
    """
    calsources = pandas.read_csv(caltable, header=0)
    files = sorted(glob.glob(f"{hdf5dir}/corr{refcorr}/{date_specifier}.hdf5"))
    datetimes = [f.split("/")[-1][:19] for f in files]
    if len(np.unique(datetimes)) != len(datetimes):
        print("Multiple files exist for the same time.")
    dates = np.unique([dt[:10] for dt in datetimes])

    filenames = {}
    for date in dates:
        filenames[date] = {}
        for _index, row in calsources.iterrows():
            if isinstance(row["ra"], str):
                rowra = row["ra"]
            else:
                rowra = row["ra"] * u.deg
            if isinstance(row["dec"], str):
                rowdec = row["dec"]
            else:
                rowdec = row["dec"] * u.deg
            cal = du.src(
                row["source"], ra=Angle(rowra), dec=Angle(rowdec), I=row["flux (Jy)"]
            )

            midnight = Time(f"{date}T00:00:00")
            delta_lst = -1 * (cal.direction.hadec(midnight.mjd)[0]) % (2 * np.pi)
            transit_time = (
                midnight + delta_lst / (2 * np.pi) * ct.SECONDS_PER_SIDEREAL_DAY * u.s
            )
            assert transit_time.isot[:10] == date

            # Get the filenames for each calibrator transit
            transit_files = []
            for dt in datetimes:
                if cal_in_datetime(dt, transit_time, duration, filelength):
                    transit_files += [dt]

            filenames[date][cal.name] = {
                "cal": cal,
                "transit_time": transit_time,
                "files": transit_files,
            }
    return filenames
