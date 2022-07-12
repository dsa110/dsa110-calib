"""Calibration service"""
from typing import List, Tuple
from pathlib import Path
import time
import os

from astropy.time import Time
import astropy.units as u
from astropy.coordinates import Angle
from dask.distributed import Client, Queue, Future
import h5py
import numpy as np
import pandas
from dsautils import dsa_store, dsa_syslog

import dsacalib.constants as ct
from dsacalib.routines import calibrate_measurement_set
from dsacalib.preprocess import rsync_file, first_true, update_caltable

"""How do we want this to work?
1. ETCD callback - create H5File
2. copy file
3. add file to the appropriate scan
4. check if scan is full
5. if scan is full, assess it, then remove it from the cache of scans
6. if assess is positive, convert to the measurement set
7. calibrate
8. create beamformer weights

Todo:
Add logging
Add plotting
"""

class H5File:
    """An hdf5 file containing correlated data."""

    def __init__(self, corrname: str, path: str):
        self.corrname = corrname
        self.path = Path(path)
        self.timestamp, self.subband = self.path.stem.split('_')
    
    def copy(self):
        rsync_string = (
            f"{self.corrname}.sas.pvt:{self.remote_path} {self.local_path}")
        rsync_file(rsync_string)

    @property
    def pointing_dec(self) -> u.Quantity:
        """Extract the pointing declination from an h5 file."""
        with h5py.File(str(self.path), mode='r') as h5file:
            pt_dec = h5file['Header']['extra_keywords']['phase_center_dec'].value * u.rad
        return pt_dec

    def check_for_source(self, calsources: pandas.DataFrame) -> pandas.DataFrame:
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
            ras = [ra * u.deg for ra in ras]

        delta_lst_start = [
            sidereal_time_delta(self.start_sidereal_time, Angle(ra)) for ra in ras]
        delta_lst_end = [
            sidereal_time_delta(self.end_sidereal_time, Angle(ra)) for ra in ras]

        source_index = delta_lst_start < self.cal_sidereal_span < delta_lst_end
        if True in source_index:
            return calsources.iloc[source_index.index(True)]


class Scan:

    def __init__(self, h5files: List[H5File], source=None, nsubbands=16):
        """Instantiate the Scan.

        Parameters
        ----------
        h5file : H5File
            A h5file to be included in the scan.
        """
        self.files = [None] * len(nsubbands)
        self.nfiles = 0
        self.source = source

        self.start_time = Time(h5files[0].timestamp)
        for h5file in h5files:
            self.add(h5file)

    def add(self, h5file: H5File) -> None:
        """Add an hdf5file to the list of files in the scan.

        Parameters
        ----------
        h5file : H5file
            The h5file to be added.
        """
        index = int(h5file.subband.strip('sb'))
        self.files[index] = h5file
        self.nfiles += 1


class ScanCache:
    """Hold scans until they have collected all files."""

    def __init__(self, max_scans: int):
        self.max_scans = max_scans
        self.scans = [None]*max_scans
        self.futures = [[] for i in range(max_scans)]
        self.next_index = 0

    def get_scan_from_file(self, h5file: H5File, copy_future: Future) -> Tuple[Scan, List[Future]]:
        """Get the scan and corresponding copy futures for an h5file."""
        for i, scan in enumerate(self.scans):
            if scan is not None:
                if abs(Time(h5file.stem) - scan.start_time) < 3 * u.min:
                    scan.add(h5file)
                    self.futures[i].append(copy_future)
                    return scan, self.futures[i]

        scan = CalibratorScan([h5file])
        scan_futures = [copy_future]
        self.scans[self.next_index] = scan
        self.futures[self.next_index] = scan_futures
        self.next_index = (self.next_index + 1) % self.max_scans
        return scan, scan_futures

    def remove(self, scan_to_remove: Scan):
        """Remove references to a scan and corresponding copy futures from the cache."""
        to_remove = None
        for i, scan in enumerate(self.scans):
            if scan.start_time == scan_to_remove.start_time:
                to_remove = i

        if to_remove:
            self.scans[to_remove] = None
            self.futures[to_remove] = None


class CalibratorScan(Scan):
    """A scan (multiple correlator hdf5 files that cover the same time)."""

    def __init__(self, h5files: List[H5File]):

        super().__init__(h5files)

        self.cal_sidereal_span = get_cal_sidereal_span()
        self.start_sidereal_time = None
        self.end_sidereal_time = None
        self.pt_dec = None

    def assess(self):
        """Assess if the scan should be converted to a ms for calibration."""
        self.start_sidereal_time, self.end_sidereal_time = [
            (self.start_time + offset).sidereal_time(
                'apparent', longitude=ct.OVRO_LON * u.rad)
            for offset in [0 * u.min, self.filelength]]

        first_file = first_true(self.files)
        self.pt_dec = first_file.pointing_dec

        caltable = update_caltable(self.pt_dec)
        calsources = pandas.read_csv(caltable, header=0)

        self.source = self.check_for_source(calsources)
    
    def convert_to_ms(self):
        """Convert this scan and the previous one to a ms."""


class CalibrationManager:
    """Manage calibration of files in realtime in the realtime system."""
    def __init__(self, logger: dsa_syslog.DsaSyslogger = None):
        self.client = Client()
        self.scan_cache = ScanCache(max_scans=12)
        self.futures = []
        self.logger = logger

    def remove_done_futures(self):
        # We track futures, instead of using fire_and_forget, so that we can
        # cancel them on keyboard interrupt.  This means we have to remove
        # references to them when they are completed.
        for future in self.futures:
            if future.done():
                self.futures.remove(future)

    def process_file(self, h5file: H5File):
        copy_future = self.client.submit(h5file.copy())
        scan, scan_futures = self.scan_cache.get_scan_from_file(h5file, copy_future)

        if scan.nfiles == 16:
            self.scan_cache.remove(scan)
            self.futures.append(self.client.submit(self.process_scan, scan, *scan_futures))

    def process_scan(self, scan, *futures):
        """Process a scan and calibrate it if it contains a source."
        
        The list of futures is unused but is required to handle the dependencies on the availability of files.
        """
        scan.assess()
        if scan.source is not None:
            msname = convert_to_ms(scan, self.logger)
            status = calibrate_measurement_set(msname, scan, self.logger)
        return status

    def __del__(self):
        self.client.cancel(self.futures)




def convert_to_ms(scan: Scan, logger: dsa_syslog.DsaSyslogger = None) -> str:
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
        msdir=msdir,
        hdf5dir=hdf5dir,
        logger=logger)

    return msname, filenames[date][calname]['cal']


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
    time_delta = (time1 - time2).to_value(u.rad) % (2 * np.pi)
    if time_delta > np.pi:
        time_delta -= 2 * np.pi
    return time_delta


def handle_etcd_triggers():
    """Main process to handle etcd triggers under /cmd/cal"""

    etcd = dsa_store.DsaStore()
    logger = dsa_syslog.DsaSyslogger()
    calmanager = CalibrationManager(logger) 

    def etcd_callback(etcd_dict):
        """Note that each callback is run in a new thread.

        All of the work is handled by dask, but we need the scan lookup to be thread-safe.
        """
        cmd = etcd_dict['cmd']
        val = etcd_dict['val']
        if cmd == 'rsync':
            h5file = H5File(val['hostname'], val['filename'])
            calmanager.process_file(h5file)
        elif cmd == 'field':
            trigname = val['trigname']
            trigmjd = val['mjds']
            calmanager.process_field_request(trigname, trigmjd)
    
    etcd.add_watch("/cmd/cal", etcd_callback)

    while True:
        calmanager.remove_done_futures()
        time.sleep(60)


if __name__ == "__main__":
    handle_etcd_triggers()   
