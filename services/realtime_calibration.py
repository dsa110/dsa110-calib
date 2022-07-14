"""Calibration service"""
from typing import List
from pathlib import Path
import time
from itertools import chain

from astropy.time import Time
import astropy.units as u
from dask.distributed import Client, Future
from dsautils import dsa_store, dsa_syslog

import dsacalib.constants as ct
from dsacalib.config import Configuration
from dsacalib.routines import calibrate_measurement_set
from dsacalib.preprocess import first_true
from dsacalib.realtime_calibration import Scan, ScanCache, H5File 


class CalibrationManager:
    """Manage calibration of files in realtime in the realtime system."""

    def __init__(self, msdir: str, h5dir: str, refmjd: float, nsubbands: int, logger: dsa_syslog.DsaSyslogger = None):
        self.client = Client()
        self.scan_cache = ScanCache(max_scans=12)
        self.futures = []
        self.logger = logger
        self.msdir = msdir
        self.h5dir = h5dir
        self.refmjd = refmjd
        self.nsubbands = nsubbands

    def process_file(self, hostname: str, remote_path: str):
        """Process an H5File that is newly written on the corr nodes."""
        local_path = Path(self.h5dir)/Path(filename).name
        h5file = H5File(local_path, hostname, remote_path)
        
        copy_future = self.client.submit(h5file.copy)
        scan, scan_futures = self.scan_cache.get_scan_from_file(h5file, copy_future)

        if scan.nfiles == 16:
            self.scan_cache.remove(scan)
            self.futures.append(self.client.submit(self.process_scan, scan, *scan_futures))

    def process_scan(self, scan: Scan, *futures: List[Future]):
        """Process a scan and calibrate it if it contains a source."
        
        The list of futures is unused but is required to handle the dependencies on the availability of files.
        """
        scan.assess()
        if scan.source is not None:
            self.calibrate_scan(scan, self.msdir)

    def calibrate_scan(self, scan: Scan):
        """Convert a scan to ms and calibrate it."""
        if scan.source is not None:
            msname = scan.convert_to_ms(scan, self.msdir, self.refmjd, self.logger)
            status = calibrate_measurement_set(msname, scan, self.logger)
        return status

    def process_field_request(self, trigname: str, trigmjd: float):
        """Create and calibrate a field ms."""
        caltime = Time(trigmjd, format='mjd')
        callst = caltime.sidereal_time('apparent', longitude=ct.OVRO_LON)
        hdf5dir = Path(self.h5dir)

        # Check current time to see if trigmjd has passed yet.  If not, wait
        to_wait = (caltime - Time.now()).to_value(u.s)
        if to_wait > 0:
            time.sleep(to_wait)
    
        # Check if correct files exists yet.  If not, wait with timeout of 10 minutes
        thishour = caltime.strftime("%Y-%m-%dT%H")
        lasthour = (caltime + 1 * u.h).strftime("%Y-%m-%dT%H")
        nexthour = (caltime + 1 * u.h).strftime("%Y-%m-%dT%H")

        # Get the list of H5Files
        counter = 0
        h5files = []
        while counter < 10:
            allfiles = chain(
                hdf5dir.glob(f"{thishour}.hdf5"),
                hdf5dir.glob(f"{lasthour}.hdf5"),
                hdf5dir.glob(f"{nexthour}.hdf5"))
            for file in allfiles:
                h5file = H5File(file)
                if abs(h5file.start_time - caltime).to_value(u.min) < 2.5:
                    if h5file not in h5files:
                        h5files.append(h5file)
            if len(h5files) >= self.nsubbands:
                break
            
            time.sleep(60) 

        # Define the scan
        calsource = {
            'source': trigname,
            'ra': callst.to_value(u.deg),
            'dec': first_true(h5files).pointing_dec.to_value(u.deg)
        }
        scan = Scan(h5files, calsource)

        # Calibrate the scan
        self.calibrate_scan(scan, self.msdir)

    def remove_done_futures(self):
        """Remove futures that are done from the list of futures.
        
        We track futures, instead of using fire_and_forget, so that we can
        cancel them on keyboard interrupt.  This means we have to remove
        references to them when they are completed.
        """
        for future in self.futures:
            if future.done():
                self.futures.remove(future)

    def __del__(self):
        self.client.cancel(self.futures)
        self.client.close()


def handle_etcd_triggers():
    """Main process to handle etcd triggers under /cmd/cal"""

    etcd = dsa_store.DsaStore()
    logger = dsa_syslog.DsaSyslogger()
    config = Configuration()
    calmanager = CalibrationManager(config.msdir, config.hdf5dir, config.refmjd, config.ncorr, logger) 

    def etcd_callback(etcd_dict: dict):
        """Note that each callback is run in a new thread.

        All of the work is handled by dask, but we need the scan lookup to be thread-safe.
        """
        cmd = etcd_dict['cmd']
        val = etcd_dict['val']
        if cmd == 'rsync':
            calmanager.process_file_request(val['hostname'], val['filename'])
        elif cmd == 'field':

            calmanager.process_field_request(val['trigname'], val['mjds'])
    
    etcd.add_watch("/cmd/cal", etcd_callback)

    while True:
        calmanager.remove_done_futures()
        time.sleep(60)


if __name__ == "__main__":
    handle_etcd_triggers()   
