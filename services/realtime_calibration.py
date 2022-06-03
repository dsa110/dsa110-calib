"""Calibration service"""

from typing import List, Union
from pathlib import Path
from dsacalib.realtime_calibration import get_files_for_cal

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


def convert_to_ms(scan: Scan, logger: "DsaSyslogger" = None) -> None:
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
