"""Calibration routine for DSA-110 calibration with CASA.

Author: Dana Simard, dana.simard@astro.caltech.edu, 2020/06
"""
import shutil
import os
import glob
from typing import List

import numpy as np
from astropy.coordinates import Angle
import astropy.units as u
from astropy.time import Time
import pandas

import scipy # pylint: disable=unused-import
from casacore.tables import table

import dsautils.calstatus as cs
import dsacalib.utils as du
import dsacalib.ms_io as dmsio
import dsacalib.fits_io as dfio
import dsacalib.calib as dc
import dsacalib.plotting as dp
import dsacalib.flagging as dfl
import dsacalib.fringestopping as df
import dsacalib.constants as ct
from dsacalib.ms_io import extract_vis_from_ms


class PipelineComponent:
    """A component of the real-time pipeline."""

    def __init__(self, logger: "DsaSyslogger", throw_exceptions: bool):
        self.description = 'Pipeline component'
        self.error_code = 0
        self.nonfatal_error_code = 0
        self.logger = logger
        self.throw_exceptions = throw_exceptions

    def target(self):
        return 0

    def __call__(self, status: int, *args) -> int:
        """Handle fatal and nonfatal errors.

        Return an updated status that reflects any errors that occurred in `target`.
        """
        try:
            error = self.target(msname, *args)

        except Exception as exc:
            status = cs.update(status, self.error_code)
            du.exception_logger(self.logger, self.description, exc, self.throw_exceptions)

        else:
            if error > 0:
                status = cs.update(status, self.nonfatal_error_code)
                message = f'Non-fatal error occured in {self.description} on {msname}'
                du.warning_logger(self.logger, message)

        return status


class Flagger(PipelineComponent):
    """The ms flagger.  Flags baselines, zeros, bad antennas, and rfi."""

    def __init__(self, logger: "DsaSyslogger", throw_exceptions: bool):
        """Describe Flagger and error code if fails."""
        super().__init__(logger, throw_exceptions)
        self.description = 'flagging'
        self.error_code = (
            cs.FLAGGING_ERR |
            cs.INV_GAINAMP_P1 |
            cs.INV_GAINAMP_P2 |
            cs.INV_GAINPHASE_P1 |
            cs.INV_GAINPHASE_P2 |
            cs.INV_DELAY_P1 |
            cs.INV_DELAY_P2 |
            cs.INV_GAINCALTIME |
            cs.INV_DELAYCALTIME )
        self.nonfatal_error_code = cs.FLAGGING_ERR

    def target(self, calobs: "CalibratorObservation"):
        """Flag data in the measurement set."""
        error = calobs.set_flags()
        return error

class DelayCalibrater(PipelineComponent):
    def __init__(self, logger: "DsaSyslogger", throw_exceptions: bool):
        super().__init__(logger, throw_exceptions)
        self.description = 'delay calibration'
        self.error_code = (
            cs.DELAY_CAL_ERR |
            cs.INV_GAINAMP_P1 |
            cs.INV_GAINAMP_P2 |
            cs.INV_GAINPHASE_P1 |
            cs.INV_GAINPHASE_P2 |
            cs.INV_DELAY_P1 |
            cs.INV_DELAY_P2 |
            cs.INV_GAINCALTIME |
            cs.INV_DELAYCALTIME )
        self.nonfatal_error_code = cs.DELAY_CAL_ERR

    def target(self, calobs):
        error = calobs.delay_calibration()
        return error


class BandpassGainCalibrater(PipelineComponent):
    def __init__(self, logger: "DsaSyslogger", throw_exceptions: bool):
        super().__init__(logger, throw_exceptions)
        self.description = 'bandpass and gain calibration'
        self.error_code = (
            cs.GAIN_BP_CAL_ERR |
            cs.INV_GAINAMP_P1 |
            cs.INV_GAINAMP_P2 |
            cs.INV_GAINPHASE_P1 |
            cs.INV_GAINPHASE_P2 |
            cs.INV_GAINCALTIME)
        self.nonfatal_error_code = cs.GAIN_BP_CAL_ERR

    def target(self, calobs, delay_bandpass_table_prefix: str = ""):
        if not delay_bandpass_table_prefix:
            error = calobs.bandpass_calibration()
            error += calobs.gain_calibration()
            error += calobs.bandpass_calibration()

        else:
            error += calobs.gain_calibration(delay_bandpass_table_prefix)

        return error

# TODO: delay_bandpass_table_prefix should be part of calobs
def calibrate_measurement_set(
        msname: str, cal: "CalibratorSource", delay_bandpass_table_prefix: str = "",
        logger: "DsaSyslogger" = None, throw_exceptions: bool = True, **kwargs
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
    manual_flags : list(list(str))
        Include any additional flags to be done prior to calibration, as
        CASA-understood strings.

    Returns
    -------
    int
        A status code. Decode with dsautils.calstatus
    """
    calobs = CalibratorObservation(msname, cal)
    calobs.set_calibration_parameters(**kwargs)
    flag = Flagger(logger, throw_exceptions)
    delaycal = DelayCalibrater(logger, throw_exceptions)
    bpgaincal = BandpassGainCalibrater(logger, throw_exceptions)

    print("entered calibration")
    print("removing files")
    
    status = 0
    calobs.reset_calibration()

    if not calobs.config["reuse_flags"]:
        print("flagging of ms data")
        status |= flagger(calobs)
           
    if not delay_bandpass_table_prefix:
        print("delay cal")
        status |= delaycal(calobs)
    
    print("bp and gain cal")    
    status |= bpgaincal(calobs, delay_bandpass_table_prefix)

    combine_tables(msname, f"{msname}_{cal.name}", delay_bandpass_table_prefix)

    print("end of cal routine")
    return status


def cal_in_datetime(
        dt: str, transit_time: "Time", duration: "Quantity" = 5*u.min,
        filelength: "Quantity" = 15*u.min
) -> bool:
    """Check to see if a transit is in a given file.

    Parameters
    ----------
    dt : str
        The start time of the file, given as a string.
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
    fileend = filestart+filelength
    transitstart = transit_time-duration/2
    transitend = transit_time+duration/2

    # For any of these conditions,
    # the file contains data that we want
    if (filestart < transitstart) and (fileend > transitend):
        transit_file = True
    elif (filestart > transitstart) and (fileend < transitend):
        transit_file = True
    elif (fileend > transitstart) and \
        (fileend-transitstart < duration):
        transit_file = True
    elif (filestart < transitend) and \
        (transitend-filestart) < duration:
        transit_file = True
    else:
        transit_file = False
    return transit_file

def get_files_for_cal(
        caltable: str, refcorr: str = "03", duration: "Quantity" = 5*u.min,
        filelength: "Quantity" = 15*u.min, hdf5dir: str = "/mnt/data/dsa110/correlator/",
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
    files = sorted(
        glob.glob(
            '{0}/corr{1}/{2}.hdf5'.format(
            hdf5dir,
            refcorr,
            date_specifier
            )
        )
    )
    datetimes = [f.split('/')[-1][:19] for f in files]
    if len(np.unique(datetimes)) != len(datetimes):
        print('Multiple files exist for the same time.')
    dates = np.unique([dt[:10] for dt in datetimes])

    filenames = dict()
    for date in dates:
        filenames[date] = dict()
        for _index, row in calsources.iterrows():
            if isinstance(row['ra'], str):
                rowra = row['ra']
            else:
                rowra = row['ra']*u.deg
            if isinstance(row['dec'], str):
                rowdec = row['dec']
            else:
                rowdec = row['dec']*u.deg
            cal = du.src(
                row['source'],
                ra=Angle(rowra),
                dec=Angle(rowdec),
                I=row['flux (Jy)']
            )

            midnight = Time('{0}T00:00:00'.format(date))
            delta_lst = -1*(
                cal.direction.hadec(midnight.mjd)[0]
            )%(2*np.pi)
            transit_time = (
                midnight + delta_lst/(2*np.pi)*ct.SECONDS_PER_SIDEREAL_DAY*u.s
            )
            assert transit_time.isot[:10]==date

            # Get the filenames for each calibrator transit
            transit_files = []
            for dt in datetimes:
                if cal_in_datetime(dt, transit_time, duration, filelength):
                    transit_files += [dt]

            filenames[date][cal.name] = {
                'cal': cal,
                'transit_time': transit_time,
                'files': transit_files
            }
    return filenames
