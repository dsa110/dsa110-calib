"""Calibration routine for DSA-110 calibration with CASA.

Author: Dana Simard, dana.simard@astro.caltech.edu, 2020/06
"""
import shutil
import os
import glob
from typing import List, Union
from pathlib import Path

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
        if not calobs.config['delay_bandpass_table_prefix']:
            error = calobs.bandpass_calibration()
            error += calobs.gain_calibration()
            error += calobs.bandpass_calibration()

        else:
            error += calobs.gain_calibration()

        return error


class H5File:
    """An hdf5 file containing correlated data."""

    h5path = Path(get_h5path())

    def __init__(self, corrname: str, remote_path: str):
        self.corrname = corrname
        self.remote_path = Path(remote_path)
        self.stem = self.remote_path.stem
        self.local_path = self.h5path/{self.corrname}/f"{self.stem}.hdf5"

    def copy(self):
        rsync_string = (
            f"{self.corrname}.sas.pvt:{self.remote_path} {self.local_path}")
        rsync_file(rsync_string)


class Scan:
    """A scan (multiple correlator hdf5 files that cover the same time)."""

    corr_list = get_corr_list()
    filelength = get_filelength()
    cal_sidereal_span = get_cal_sidereal_span()

    def __init__(self, h5file: H5File):
        """Instantiate the Scan.

        Parameters
        ----------
        h5file : H5File
            A h5file to be included in the scan.
        """
        self.files = [None]*len(self.corr_list)
        self.nfiles = 0

        self.start_time = Time(h5file.stem)
        self.add(h5file)

        self.start_sidereal_time = None
        self.end_sidereal_time = None
        self.pt_dec = None
        self.source = None

    def add(self, h5file: H5File) -> None:
        """Add an hdf5file to the list of files in the scan.

        Parameters
        ----------
        h5file : H5file
            The h5file to be added.
        """
        self.files[self.corr_list.index(h5file.corrname)] = h5file
        self.nfiles += 1

    def assess(self):
        """Assess if the scan should be converted to a ms for calibration."""
        self.start_sidereal_time, self.end_sidereal_time = [
            (self.start_time + offset).sidereal_time(
            'apparent', longitude=ct.OVRO_LON*u.rad)
            for offset in 0*u.min, self.filelength]

        first_file = first_true(self.files)
        self.pt_dec = get_pointing_dec(first_file.local_path)

        caltable = update_caltable(pt_dec)
        calsources = pandas.read_csv(caltable, header=0)

        self.source = self.check_for_source(calsources)

    def check_for_source(self, calsources: "pandas.DataFrame") -> "pandas.DataFrame":
        """Determine if there is any calibrator source of interest in the scan.

        Parameters
        ----------
        calsources : pandas.DataFrame
            A dataframe containing sources at the pointing declination of the scan.

        Returns
        -------
        pandas.DataFrame
            The row of the data frame with the first source that is in the scan.  If no source is
            found, `None` is returned.
        """
        ras = calsources['ra']
        if not isinstance(ras[0], str):
            ras = [ra*u.deg for ra in ras]

        delta_lst_start = [
            sidereal_time_delta(self.start_sidereal_time, Angle(ra)) for ra in ras]
        delta_lst_end = [
            sidereal_time_delta(self.end_sidereal_time, Angle(ra)) for ra in ras]
        
        source_index = delta_lst_start < self.cal_sidereal_span < delta_lst_end
        if True in source_index:
            return calsources.iloc[source_index.index(True)]


def convert_to_ms(scan: Scan, logger: "DsaSyslogger" = None) -> str:
    """Convert a scan to a measurement set.

    Parameters
    ----------
    scan : Scan
        A scan containing part of the calibrator pass of interest.
    logger : DsaSyslogger
        The logging interface.  If `None`, messages are only printed.
    """
    date = scan.start_time.strftime("%Y-%m-%d")
    msname = f"{config['msdir']}/{date}_{scan.source}"
    if os.path.exists(f"{msname}.ms"):
        message = f"{msname}.ms already exists.  Not recreating."
        du.info_logger(logger, message)
        return

    file = first_true(scan.files)
    directory = file.parents[0]
    hdf5dir = file.parents[1]
    filenames = get_files_for_cal(
        scan.source, directory, f"{date}*", config['caltime'], config['filelength'])

    convert_calibrator_pass_to_ms(
        cal=filenames[date][calname]["cal"],
        date=date,
        files=filenames[date][calname]["files"],
        msidr=msdir,
        hdf5dir=hdf5dir,
        logger=logger)

    return msname, filenames[date][calname]['cal']


def calibrate_measurement_set(
        msname: str, cal: "CalibratorSource", scan: "Scan" = None,
        logger: "DsaSyslogger" = None, throw_exceptions: bool = False, **kwargs) -> int:
    calobs = CalibratorObservation(msname, cal, scan)
    calobs.set_calibration_parameters(**kwargs)
    flag = Flagger(logger, throw_exceptions)
    delaycal = DelayCalibrater(logger, throw_exceptions)
    bpgaincal = BandpassGainCalibrater(logger, throw_exceptions)

    print("entered calibration")
    print("removing files")
    
    status = 0
    calobs.reset_calibration()

    if not calobs.config['reuse_flags']:
        print("flagging of ms data")
        status |= flagger(calobs)
           
    if not calobs.config['delay_bandpass_table_prefix']:
        print("delay cal")
        status |= delaycal(calobs)
    
    print("bp and gain cal")    
    status |= bpgaincal(calobs)

    combine_tables(msname, f"{msname}_{cal.name}", calobs.config['delay_bandpass_table_prefix'])
    calobs.create_beamformer_weights()

    print("end of cal routine")
    return status

def sidereal_time_delta(time1: "Angle", time2: "Angle") -> float:
    """Get the sidereal rotation between two LSTs. (time1-time2)

    Parameters
    ----------
    time1, time2 : Angle
        The LSTs to compare.

    Returns
    -------
    float
        The difference in the sidereal times, `time1` and `time2`, in radians.
    """
    time_delta = (time1-time2).to_value(u.rad)%(2*np.pi)
    if time_delta > np.pi:
        time_delta -= 2*np.pi
    return time_delta


def get_h5path() -> Path:
    """Return the path to the hdf5 (correlator) directory."""
    conf = cnf.Conf()
    return Path(conf.get('cal')['hdf5_dir'])


def get_corr_list() -> List[str]:
    """Return the list of correlators, in freqency order (highest to lowest)."""
    conf = cnf.Conf()
    corr_conf = conf.get('corr')
    return list(corr_conf['ch0'].keys())


def get_filelength() -> "Quantity":
    """Return the filelength of the hdf5 files."""
    conf = cnf.Conf()
    return conf.get('fringe')['filelength_minutes']*u.min


def get_cal_sidereal_span() -> float:
    """Return the sidereal span desired for the calibration pass, in radians."""
    conf = cnf.Conf()
    caltime = conf.get('cal')['caltime_minutes']*u.min
    return (caltime*np.pi*u.rad / (ct.SECONDS_PER_SIDEREAL_DAY*u.s)).to_value(u.rad)


def get_pointing_dec(filepath: Union[str, Path]) -> "Quantity":
    """Extract the pointing declination from an h5 file."""
    with h5py.File(str(filepath), mode='r') as h5file:
        pt_dec = h5file['Header']['extra_keywords']['phase_center_dec'].value*u.rad
    return pt_dec
