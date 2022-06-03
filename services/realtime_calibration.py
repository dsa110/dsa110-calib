"""Calibration service"""

from dask.distributed import Client, Queue, wait

from dsautils import dsa_store

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
FILEQ = Queue('FileQ', client)


def process_file_queue(twait: int):
    scan_cache = ScanCache(max_scans=12)
    futures = []

    try:
        while True:
            # Track futures, instead of using fire_and_forget, so that we can 
            # cancel them on keyboard interrupt.

            for future in futures:
                if future.done():
                    futures.remove(future)

            if FILEQ.qsize() == 0:
                time.sleep(twait)
                continue

            h5file, copy_future = FILEQ.get()
            scan, scan_futures = scan_cache.get_scan_from_file(h5file, copy_future)

            if scan.nfiles == 16:
                scan_cache.remove(scan)
                futures.append(client.submit(process_scan, scan, *scan_futures))

    except KeyBoardInterrupt:
        client.cancel(futures)


def process_scan(scan, *futures):
    """Process a scan and calibrate it if it contains a source."""
    scan.assess()
    if scan.source is not None:
        msname, source = convert_to_ms(scan, logger)
        status = calibrate_measurement_set(msname, source, scan, logger)


def etcd_callback(etcd_dict):
    """Note that each callback is run in a new thread.

    All of the work is handled by dask, but we need the scan lookup to be thread-safe.
    """
    cmd = etcd_dict['cmd']
    val = etcd_dict['val']
    if cmd == 'rsync':
        h5file = H5File(val['hostname'], val['filename'])
        copy_future = client.submit(h5file.copy)
        FILEQ.put((h5file, copy_future))


class ScanCache:
    """Hold scans until they have collected all files."""

    def __init__(self, max_scans: int):
        self.max_scans = max_scans
        self.scans = [None]*max_scans
        self.futures = [[] for i in range(max_scans)]
        self.next_index = 0

    def get_scan_from_file(self, h5file: H5File, copy_future: future) -> Tuple[Scan, List[future]]:
        """Get the scan and corresponding copy futures for an h5file."""
        for i, scan in enumerate(self.scans):
            if scan is not None:
                if abs(Time(h5file.stem) - scan.start_time) < 3*u.min:
                    scan.add(h5file)
                    self.futures[i].append(copy_future)
                    return scan, self.futures[i]

        scan = Scan(h5file)
        scan_futures = [copy_future]
        self.scans[self.next_index] = scan
        self.futures[self.next_index] = scan_futures
        self.next_index = (self.next_index + 1 ) % self.max_scans
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

def __main__():
    etcd = dsa_store.DsaStore()
    etcd.add_watch('/cmd/cal', etcd_callback)
    process_file_queue()
