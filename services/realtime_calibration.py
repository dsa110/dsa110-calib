"""Calibration service"""
from typing import List, Tuple
import time

from astropy.time import Time
import astropy.units as u
from dask.distributed import Client, Queue, Future

from dsautils import dsa_store, dsa_syslog

from dsacalib.realtime_calibration import (
    H5File, Scan, calibrate_measurement_set, convert_to_ms)

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

    def process_file(self, h5file):
        scan, scan_futures = self.scan_cache.get_scan_from_file(h5file)

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
    
    def process_field_request(self, candname, candmjd):
        """Create and calibrate a field measurement set at the time of a candidate.
        """
        # Get hdf5 files corresponding to the time
        # Create the scan
        msname = convert_to_ms(scan, self.logger)
        status = calibrate_field_ms(msname, scan, self.logger)
        pass

    def __del__(self):
        self.client.cancel(self.futures)


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
