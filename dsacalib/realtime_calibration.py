"""Calibration routine for DSA-110 calibration with CASA.

Author: Dana Simard, dana.simard@astro.caltech.edu, 2020/06
"""
import shutil
import os
import glob
import numpy as np
from astropy.coordinates import Angle
import pandas
import scipy # pylint: disable=unused-import
from casacore.tables import table
import dsautils.calstatus as cs
import dsacalib.utils as du
import dsacalib.ms_io as dmsio
import dsacalib.fits_io as dfio
import dsacalib.calib as dc
import dsacalib.plotting as dp
import dsacalib.fringestopping as df
import dsacalib.constants as ct
from dsacalib.ms_io import extract_vis_from_ms
import astropy.units as u # pylint: disable=wrong-import-order
from astropy.utils import iers # pylint: disable=wrong-import-order
iers.conf.iers_auto_url_mirror = ct.IERS_TABLE
iers.conf.auto_max_age = None
from astropy.time import Time # pylint: disable=wrong-import-position


class PipelineComponent:
    def __init__(self, logger, throw_exceptions):
        self.description = 'Pipline component'
        self.error_code = 0
        self.nonfatal_error_code = 0
        self.logger = logger
        self.throw_exceptions = throw_exceptions

    def target(self):
        return 0

    def __call__(self, status, *args):
        """Handle fatal and nonfatal errors."""
        try:
            error = self.target(msname, *args)

        except Exception as exc:
            status = cs.update(status, self.error_code)
            du.exception_logger(self.logger, self.description, exc, self.throw_exceptions)

        else:
            if error > 0:
                status = cs.update(status, self.nonfatal_error_code)
                message = f'Non-fatal error occured in {self.description} on {msname}'
                if logger is not None:
                    logger.warning(message)
                else:
                    print(message)

        return status

class Flagger(PipelineComponent):
    def __init__(self, logger, throw_exceptions):
        """Describe Flagger and error code if fails."""
        super.__init__(logger, throw_exceptions)
        self.description = 'Flagging of ms data'
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

    def target(self, msname, bad_uvrange):
        """Flag data in the measurement set."""
        dc.reset_flags(msname, datacolumn='data')
        dc.reset_flags(msname, datacolumn='model')
        dc.reset_flags(msname, datacolumn='corrected')        
        self.current_error = cs.FLAGGING_ERR

        error = 0
        error += dc.flag_baselines(msname, uvrange=bad_uvrange)
        error += dc.flag_zeros(msname)

        if bad_antennas is not None:
            for ant in bad_antennas:
                error += dc.flag_antenna(msname, ant)

        if manual_flags is not None:
            for entry in manual_flags:
                error += dc.flag_manual(msname, entry[0], entry[1])

        flag_pixels(msname)

        return error

class DelayCalibrator(PipelineComponent):
    def __init__(self, logger, throw_exceptions):
        super.__init__(logger, throw_exceptions)
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

    def flag_using_delay_calibration(self, msname, cal, refants, t2):
        error = 0
        error += dc.delay_calibration(msname, cal.name, refants=refants, t2=t2)
        _check_path('{0}_{1}_kcal'.format(msname, cal.name))
        _times, antenna_delays, kcorr, _ant_nos = dp.plot_antenna_delays(
            msname, cal.name, show=False)
        _check_path('{0}_{1}_2kcal'.format(msname, cal.name))
        error += flag_antennas_using_delays(antenna_delays, kcorr, msname)
        shutil.rmtree('{0}_{1}_kcal'.format(msname, cal.name))
        shutil.rmtree('{0}_{1}_2kcal'.format(msname, cal.name))
        return error

    def delay_calibrate(self, msname, cal, refants, t2):
        error = 0
        error += dc.delay_calibration(msname, cal.name, refants=refants, t2=t2)
        _check_path('{0}_{1}_kcal'.format(msname, cal.name))
        return error

    def target(self, msname, cal, refants, t2):
        error = 0
        error += self.flag_using_delay_calibration(msname, cal, refants, t2)
        error += self.delay_calibrate(msname, cal, refants, t2)
        return error

class BandpassGainCalibrator(PipelineComponent):
    def __init__(self, logger, throw_exceptions):
        super.__init__(logger, throw_exceptions)
        self.description = 'bandpass and gain calibration'
        self.error_code = (
            cs.GAIN_BP_CAL_ERR |
            cs.INV_GAINAMP_P1 |
            cs.INV_GAINAMP_P2 |
            cs.INV_GAINPHASE_P1 |
            cs.INV_GAINPHASE_P2 |
            cs.INV_GAINCALTIME)
        self.nonfatal_error_code = cs.GAIN_BP_CAL_ERR

    def check_tables_created(self, msname, cal, forsystemhealth, keepdelays)
        fnames = [
            '{0}_{1}_bcal'.format(msname, cal.name),
            '{0}_{1}_bacal'.format(msname, cal.name),
            '{0}_{1}_bpcal'.format(msname, cal.name),
            '{0}_{1}_gpcal'.format(msname, cal.name),
            '{0}_{1}_gacal'.format(msname, cal.name)
        ]
        if forsystemhealth:
            fnames += [
                '{0}_{1}_2gcal'.format(msname, cal.name)
            ]
        if not keepdelays and not forsystemhealth:
            fnames += [
                '{0}_{1}_bkcal'.format(msname, cal.name)
            ]
        for fname in fnames:
            _check_path(fname)

    def combine_tables(self, msname, cal, forsystemhealth):
        print('combining bandpass and delay solns')
        # Combine bandpass solutions and delay solutions
        with table('{0}_{1}_bacal'.format(msname, cal.name)) as tb:
            bpass = np.array(tb.CPARAM[:])
        with table('{0}_{1}_bpcal'.format(msname, cal.name)) as tb:
            bpass *= np.array(tb.CPARAM[:])
        if not forsystemhealth:
            with table('{0}_{1}_bkcal'.format(msname, cal.name)) as tb:
                bpass = np.array(tb.CPARAM[:])
        with table(
            '{0}_{1}_bcal'.format(msname, cal.name),
            readonly=False
        ) as tb:
            tb.putcol('CPARAM', bpass)
            if not forsystemhealth:
                tbflag = np.array(tb.FLAG[:])
                tb.putcol('FLAG', np.zeros(tbflag.shape, tbflag.dtype))

    def target(self, msname, cal, refant, blbased, forsystemhealth, keepdelays,
               interp_thresh, interp_polyorder, t2):
        error = 0
        error += dc.gain_calibration(
            msname,
            cal.name,
            refant,
            blbased=blbased,
            forsystemhealth=forsystemhealth,
            keepdelays=keepdelays,
            interp_thresh=interp_thresh,
            interp_polyorder=interp_polyorder,
            tbeam=t2)

        self.check_tables_created(msname, cal, forsystemhealth, keepdelays)

        self.combine_tables(msname, cal, forsystemhealth)

        return error

def calibrate_measurement_set(
    msname, cal, refants, throw_exceptions=True, bad_antennas=None,
    bad_uvrange='2~27m', keepdelays=False, forsystemhealth=False,
    interp_thresh=1.5, interp_polyorder=7, blbased=False, manual_flags=None,
    logger=None, t2='60s'
):
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
    if isinstance(refants, (int,str)):
        refant = refants
        refants = [refant]
    else:
        refant = refants[0]

    status = 0

    # Remove files that we will create so that things will fail if casa
    # doesn't write a table.
    tables_to_remove = [
        '{0}_{1}_2kcal'.format(msname, cal.name),
        '{0}_{1}_kcal'.format(msname, cal.name),
        '{0}_{1}_bkcal'.format(msname, cal.name),
        '{0}_{1}_gacal'.format(msname, cal.name),
        '{0}_{1}_gpcal'.format(msname, cal.name),
        '{0}_{1}_bcal'.format(msname, cal.name)
    ]
    if forsystemhealth:
        tables_to_remove += [
            '{0}_{1}_2gcal'.format(msname, cal.name)
        ]
    for path in tables_to_remove:
        if os.path.exists(path):
            shutil.rmtree(path)

    flagger = Flagger(logger, throw_exceptions)
    delaycal = DelayCalibrator(logger, throw_exceptions)
    gaincal = BandpassGainCalibrator(logger, throw_exceptions)

    status = flagger(status, msname, bad_uvrange)
    status = delaycal(status, msname, cal, refants, t2)
    status = gaincal(status, msname, cal, refant, blbased, forsystemhealth,
                     keepdelays, interp_thresh, interp_polyorder, t2)

    return status

def cal_in_datetime(dt, transit_time, duration=5*u.min, filelength=15*u.min):
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
    caltable, refcorr='01', duration=5*u.min, filelength=15*u.min,
    hdf5dir='/mnt/data/dsa110/correlator/', date_specifier='*'):
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

def _check_path(fname):
    """Raises an AssertionError if the path `fname` does not exist.

    Parameters
    ----------
    fname : str
        The file to check existence of.
    """
    assert os.path.exists(fname), 'File {0} does not exist'.format(fname)
