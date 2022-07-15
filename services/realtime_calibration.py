"""Calibration service"""
from typing import List
from pathlib import Path
import time
from itertools import chain
import yaml

from astropy.time import Time
import astropy.units as u
from dask.distributed import Client, Future
from dsautils import dsa_store, dsa_syslog

import dsacalib.constants as ct
from dsacalib.config import Configuration
from dsacalib.routines import calibrate_measurement_set
from dsacalib.preprocess import first_true
from dsacalib.realtime_calibration import (
    Scan, ScanCache, H5File, generate_averaged_beamformer_solns
)
from dsacalib.plotting import generate_summary_plot, plot_bandpass_phases, plot_beamformer_weights
from dsacalib.hdf5_io import extract_applied_delays
from dsacalib.weights import write_beamformer_solutions
from dsacalib.ms_io import caltable_to_etcd


class CalibrationManager:
    """Manage calibration of files in realtime in the realtime system."""

    def __init__(
            self, config: Configuration, logger: dsa_syslog.DsaSyslogger, store: dsa_store.DsaStore):
        self.client = Client()
        self.scan_cache = ScanCache(max_scans=12)
        self.futures = []
        self.logger = logger
        self.store = store
        self.config = config.copy()

    def process_file(self, hostname: str, remote_path: str):
        """Process an H5File that is newly written on the corr nodes."""
        local_path = Path(self.h5dir) / Path(remote_path).name
        h5file = H5File(local_path, hostname, remote_path)

        copy_future = self.client.submit(h5file.copy)
        scan, scan_futures = self.scan_cache.get_scan_from_file(
            h5file, copy_future)

        if scan.nfiles == 16:
            self.scan_cache.remove(scan)
            self.futures.append(self.client.submit(
                self.process_scan, scan, *scan_futures))

    def process_scan(self, scan: Scan, *futures: List[Future]):
        """Process a scan and calibrate it if it contains a source."

        The list of futures is unused but is required to handle the dependencies on the availability of files.
        """
        scan.assess()
        if scan.source is None:
            return

        self.store.put_dict(
            "/mon/calibration",
            {
                "scan_time": scan.start_time.mjd,
                "calibration_source": scan.source.source,
                "status": -1
            }
        )

        msname, cal, calstatus = self.calibrate_scan(scan, self.msdir)

        self.store.put_dict(
            "/mon/calibration",
            {
                "scan_time": scan.start_time.mjd,
                "calibration_source": scan.source.source,
                "status": calstatus
            }
        )

        # Upload solutions to etcd
        caltable_to_etcd(
            msname, cal.name, scan.start_time.mjd, calstatus, logger=self.logger)

        # Make summary plots
        generate_summary_plot(
            scan.start_time.strftime("%Y-%m-%d"), msname, cal.name, self.config.antennas, self.config.tempplots,
            self.config.webplots)

        # Make beamformer weights
        applied_delays = extract_applied_delays(
            first_true(scan.files).path, self.config.antennas)

        # Write beamformer solutions for one source
        caltime = Time(scan.start_time.isot, precision=0)
        write_beamformer_solutions(
            msname, cal.name, scan.start_time.mjd, self.config.antennas, applied_delays, self.config.beamformer_dir,
            self.config.pols, self.config.nchan, self.config.nchan_spw, self.config.bw_GHz,
            self.config.chan_ascending, self.config.f0_GHz, self.config.ch0, self.config.refmjd,
            flagged_antennas=self.config.antennas_not_in_bf)

        ref_bfweights = self.store.get_dict("/mon/cal/bfweights")
        beamformer_solns, beamformer_names = generate_averaged_beamformer_solns(
            self.config.snap_start_time, scan.start_time, self.config.beamformer_dir,
            self.config.antennas, self.config.antennas_core, self.config.pols, self.config.refants[0],
            self.config.refmjd, ref_bfweights)

        if not beamformer_solns:
            return

        with open(
                f"{self.config.beamformer_dir}/beamformer_weights_{caltime.isot}.yaml",
                "w",
                encoding="utf-8"
        ) as file:
            yaml.dump(beamformer_solns, file)

        self.store.put_dict(
            "/mon/cal/bfweights",
            {
                "cmd": "update_weights",
                "val": beamformer_solns["cal_solutions"]})

        # Plot the beamformer solutions
        figure_prefix = f"{self.config.tempplots}/{caltime}"
        plot_beamformer_weights(
            beamformer_names, self.config.antennas, self.config.beamformer_dir,
            outname=figure_prefix, show=False)

        # Plot evolution of the phase over the day
        plot_bandpass_phases(
            beamformer_names,
            self.config.msdir,
            self.config.antennas,
            outname=figure_prefix,
            show=False
        )

    def calibrate_scan(self, scan: Scan):
        """Convert a scan to ms and calibrate it."""
        if scan.source is None:
            raise UndefinedcalError(
                f"No calibrator source defined for scan {scan.start_time.isot}")

        msname, cal = scan.convert_to_ms(
            scan, self.msdir, self.refmjd, self.logger)

        status = calibrate_measurement_set(
            msname, cal, refants=self.refants, logger=self.logger)

        return msname, cal, status

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
        self.calibrate_scan(scan)

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


class UndefinedcalError(RuntimeError):
    pass


def handle_etcd_triggers():
    """Main process to handle etcd triggers under /cmd/cal"""

    store = dsa_store.DsaStore()
    logger = dsa_syslog.DsaSyslogger()
    config = Configuration()
    calmanager = CalibrationManager(
        config.msdir, config.hdf5dir, config.refmjd, config.ncorr, config.refants, logger)

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

    store.add_watch("/cmd/cal", etcd_callback)

    while True:
        calmanager.remove_done_futures()
        time.sleep(60)


if __name__ == "__main__":
    handle_etcd_triggers()
