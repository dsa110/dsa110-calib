"""Calibration routine for DSA-110 calibration with CASA.

Author: Dana Simard, dana.simard@astro.caltech.edu, 2020/06
"""
import glob
from typing import List

import scipy  # noqa
import numpy as np
import pandas
import astropy.units as u
from astropy.coordinates import Angle
from astropy.time import Time
from casacore.tables import table

import dsautils.dsa_syslog as dsl
import dsautils.calstatus as cs
import dsacalib.constants as ct
import dsacalib.utils as du
from dsacalib.calibrator_observation import CalibratorObservation
from dsacalib.weights import write_beamformer_solutions
from dsacalib.calib import combine_tables


def calibrate_measurement_set(
        msname: str, cal: du.CalibratorSource, refants: List[str],
        delay_bandpass_cal_prefix: str = "", logger: dsl.DsaSyslogger = None,
        throw_exceptions: bool = True, **kwargs
) -> int:
    r"""Calibrates the measurement set.

    Parameters
    ----------
    msname : str
        The name of the measurement set. Will open `msname`.ms
    cal : dsacalib.utils.src instance
        The calibration source. Calibration tables will begin with
        `msname`\_`cal.name`
    logger : dsautils.dsa_syslog.DsaSyslogger() instance
        Logger to write messages too. If None, messages are printed.
    throw_exceptions : bool
        If set to False, exceptions will not be thrown, although they will be
        logged to syslog. Defaults True.
    refants : str or int
        The reference antenna name (if str) or index (if int) for calibration.
    bad_antennas : list(str)
        Antennas (names) to be flagged before calibration.
    bad_uvrange : str
        Baselines with lengths within bad_uvrange will be flagged before
        calibration. Must be a casa-understood string with units.
    blbased : boolean
        Set to True for baseline-based calibration, False for antenna-based
        calibration.
    manual_flags : list(list(str))
        Include any additional flags to be done prior to calibration, as
        CASA-understood strings.

    Returns
    -------
    int
        A status code. Decode with dsautils.calstatus
    """
    calobs = CalibratorObservation(msname, cal, refants)
    calobs.set_calibration_parameters(**kwargs)

    print("entered calibration")
    status = 0
    current_error = cs.UNKNOWN_ERR
    calstring = "initialization"

    try:
        print("removing files")
        calobs.reset_calibration()

        if not calobs.config["reuse_flags"]:
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
            error = calobs.set_flags()
            if error > 0:
                status = cs.update(status, cs.FLAGGING_ERR)
                message = f"{error} non-fatal errors occured in flagging of {msname}."
                du.warning_logger(logger, message)

        if not delay_bandpass_cal_prefix:
            print("delay cal")
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
            error = calobs.delay_calibration()
            if error > 0:
                status = cs.update(status, cs.DELAY_CAL_ERR)
                message = f"{error} non-fatal errors occured in delay calibration of {msname}."
                du.warning_logger(logger, message)

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

        if not delay_bandpass_cal_prefix:
            error = calobs.bandpass_calibration()
            error += calobs.gain_calibration()
            error += calobs.bandpass_calibration()

        else:
            error = calobs.gain_calibration(delay_bandpass_cal_prefix)

        if error > 0:
            status = cs.update(status, cs.GAIN_BP_CAL_ERR)
            message = f"{error} non-fatal errors occured in gain calibration of {msname}."
            du.warning_logger(logger, message)

        current_error = cs.GAIN_BP_CAL_ERR
        combine_tables(msname, f"{msname}_{cal.name}", delay_bandpass_cal_prefix)

    except Exception as exc:
        status = cs.update(status, current_error)
        du.exception_logger(logger, calstring, exc, throw_exceptions)

    print("end of cal routine")

    return status


def quick_bfweightcal(
        msname: str, antennas: List[str], antennas_not_in_bf: List[str], corr_list: List[str],
        cal: du.CalibratorSource = None, **kwargs) -> int:
    """Calibrate delays and gains, and generate bfweights."""
    if not cal:
        cal = get_cal_from_msname(msname)

    config = {
        "antennas": antennas,
        "antennas_not_in_bf": antennas_not_in_bf,
        "corr_list": [int(cl.strip("corr")) for cl in corr_list],
    }

    for key in ["reuse_flags"]:
        if key in kwargs:
            raise RuntimeError(
                f"Input arg {key} not compatible with quick_bfweightcal")
    kwargs["reuse_flags"] = True

    calobs = CalibratorObservation(msname, cal)
    calobs.set_calibration_parameters(**kwargs)
    calobs.reset_calibration()
    error = 0
    error += calobs.quick_delay_calibration()
    error += calobs.bandpass_calibration()
    error += calobs.gain_calibration()
    error += calobs.bandpass_calibration()
    combine_tables(msname, cal.name)

    with table(f"{msname}.ms") as tb:
        caltime = Time((tb.TIME_CENTROID[tb.nrows() // 2] * u.s).to(u.d), format='mjd')

    write_beamformer_solutions(
        msname,
        cal.name,
        caltime,
        config["antennas"],
        applied_delays=None,
        flagged_antennas=config["antennas_not_in_bf"])

    return int


def quick_calibration(msname: str, cal: du.CalibratorSource = None, **kwargs) -> int:
    """Calibrate without resetting flags."""
    if not cal:
        cal = get_cal_from_msname(msname)

    calobs = CalibratorObservation(msname, cal)

    for key in ["reuse_flags"]:
        if key in kwargs and not kwargs[key]:
            raise RuntimeError(
                f"Input arg {key}: {kwargs[key]} not compatible with quick_calibration")
    kwargs["reuse_flags"] = True
    calobs.set_calibration_parameters(**kwargs)
    calobs.reset_calibration()
    error = 0
    error += calobs.quick_delay_calibration()
    error += calobs.bandpass_calibration()
    error += calobs.gain_calibration()
    error += calobs.bandpass_calibration()

    return error


def get_cal_from_msname(msname: str) -> du.CalibratorSource:
    """Construct a CalibratorSource objct based on the msname.

    Assumes that the msname includes the calibrator source, and does
    not include the suffix, for e.g., `/path/to/directory/{date}_{calname}`
    """
    calname = msname.split('/')[-1] if '/' in msname else msname
    if '_' in calname:
        calname = calname.split('_')[-1]
    else:
        calname = 'cal'
    cal = du.generate_calibrator_source(calname, ra=None, dec=None)
    return cal


def cal_in_datetime(
        dt: str, transit_time: Time, duration: u.Quantity = 5 * u.min,
        filelength: u.Quantity = 15 * u.min
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
        caltable: str, refcorr: str = "03", duration: u.Quantity = 5 * u.min,
        filelength: u.Quantity = 15 * u.min, hdf5dir: str = "/mnt/data/dsa110/correlator/",
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
            cal = du.generate_calibrator_source(
                row["source"], ra=Angle(rowra), dec=Angle(rowdec), flux=row["flux (Jy)"])

            midnight = Time(f"{date}T00:00:00")
            delta_lst = -1 * (cal.direction.hadec(midnight.mjd)[0]) % (2 * np.pi)
            transit_time = (
                midnight + delta_lst / (2 * np.pi) * ct.SECONDS_PER_SIDEREAL_DAY * u.s)
            assert transit_time.isot[:10] == date

            # Get the filenames for each calibrator transit
            transit_files = []
            for dt in datetimes:
                if cal_in_datetime(dt, transit_time, duration, filelength):
                    transit_files += [dt]

            filenames[date][cal.name] = {
                "cal": cal,
                "transit_time": transit_time,
                "files": transit_files}

    return filenames
