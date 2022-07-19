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

from dsacalib.uvh5_to_ms import add_single_source_model_to_ms, add_multisource_model_to_ms
from dsacalib.config import Configuration
import dsacalib.constants as ct
from dsacalib.hdf5_io import extract_applied_delays
from dsacalib.ms_io import caltable_to_etcd
from dsacalib.plotting import generate_summary_plot, plot_bandpass_phases, plot_beamformer_weights
from dsacalib.preprocess import first_true
from dsacalib.realtime_calibration import (
    Scan, ScanCache, H5File, generate_averaged_beamformer_solns
)
from dsacalib.routines import calibrate_measurement_set
from dsacalib.weights import write_beamformer_solutions

SCHEDULER_IP = "10.41.0.104:8786"


class CalibrationManager:
    """Manage calibration of files in realtime in the realtime system."""

    def __init__(self):
        self.logger = dsa_syslog.DsaSyslogger()
        # TODO: Configuration needs to update the snap start time when they are restarted
        self.config = Configuration()
        self.client = Client(SCHEDULER_IP)
        self.scan_cache = ScanCache(max_scans=12)
        self.futures = []

    def process_file(self, hostname: str, remote_path: str):
        """Process an H5File that is newly written on the corr nodes."""
        local_path = Path(self.config.hdf5dir) / Path(remote_path).name
        h5file = H5File(local_path, hostname, remote_path)

        copy_future = self.client.submit(h5file.copy, resources={'MEMORY': 10e9})
        scan, scan_futures = self.scan_cache.get_scan_from_file(
            h5file, copy_future)

        self.futures.append(copy_future)

        if scan.nfiles == 16:
            self.scan_cache.remove(scan)
            assess_future = self.client.submit(assess_scan, scan, *scan_futures,  resources={'MEMORY': 10e9})
            process_future = self.client.submit(
                process_scan, scan, self.config, assess_future,  resources={'MEMORY': 60e9})
            self.futures.append(assess_future)
            self.futures.append(process_future)

    def process_field_request(self, trigname: str, trigmjd: float):
        """Create and calibrate a field ms."""
        caltime = Time(trigmjd, format='mjd')
        print(f"Making field ms for {caltime.isot}")
        process_future = self.client.submit(create_field_ms, caltime, trigname, self.config, resources={'MEMORY': 60e9})
        self.futures.append(process_future)

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


def assess_scan(scan, *futures):
    scan.assess()


def create_field_ms(caltime, calname, config):
    """Create a field measurement set."""
    hdf5dir = Path(config.hdf5dir)
    callst = caltime.sidereal_time('apparent', longitude=ct.OVRO_LON)

    # Check current time to see if trigmjd has passed yet.  If not, wait
    to_wait = (caltime - Time.now()).to_value(u.s)
    if to_wait > 0:
        print(f"Waiting {to_wait}s for {caltime.isot}")
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
            hdf5dir.glob(f"{thishour}*.hdf5"),
            hdf5dir.glob(f"{lasthour}*.hdf5"),
            hdf5dir.glob(f"{nexthour}*.hdf5"))
        for file in allfiles:
            h5file = H5File(file)
            if abs(h5file.start_time - caltime).to_value(u.min) < 2.5:
                if h5file not in h5files:
                    h5files.append(h5file)
        print(f"Found {len(h5files)} files for {caltime.isot}")
        if len(h5files) >= config.ncorr:
            break

        time.sleep(60)

    # Define the scan
    calsource = {
        'source': calname,
        'ra': callst.to_value(u.deg),
        'dec': first_true(h5files).pointing_dec.to_value(u.deg)
    }
    scan = Scan(h5files, calsource)

    # Calibrate the scan
    print(f"Calibrating {caltime.isot}")
    calibrate_scan(scan, config, 'field')


def process_scan(scan: Scan, config: Configuration, *futures: List[Future]):
    """Process a scan and calibrate it if it contains a source."

    The list of futures is unused but is required to handle the dependencies on the availability of files.
    """
    store = dsa_store.DsaStore()
    logger = dsa_syslog.DsaSyslogger()

    if scan.source is None:
        return

    store.put_dict(
        "/mon/calibration",
        {
            "scan_time": scan.start_time.mjd,
            "calibration_source": scan.source.source,
            "status": -1
        }
    )

    msname, cal, calstatus = calibrate_scan(scan, config, 'calibrator')

    store.put_dict(
        "/mon/calibration",
        {
            "scan_time": scan.start_time.mjd,
            "calibration_source": scan.source.source,
            "status": calstatus
        }
    )

    # Upload solutions to etcd
    caltable_to_etcd(
        msname, cal.name, scan.start_time.mjd, calstatus, logger=logger)

    # Make summary plots
    generate_summary_plot(
        scan.start_time.strftime("%Y-%m-%d"), msname, cal.name, config.antennas, config.tempplots,
        config.webplots)

    # Make beamformer weights
    applied_delays = extract_applied_delays(
        first_true(scan.files).path, config.antennas)

    # Write beamformer solutions for one source
    caltime = Time(scan.start_time.isot, precision=0)
    write_beamformer_solutions(
        msname, cal.name, scan.start_time.mjd, config.antennas, applied_delays, config.beamformer_dir,
        config.pols, config.nchan, config.nchan_spw, config.bw_GHz,
        config.chan_ascending, config.f0_GHz, config.ch0, config.refmjd,
        flagged_antennas=config.antennas_not_in_bf)

    ref_bfweights = store.get_dict("/mon/cal/bfweights")
    beamformer_solns, beamformer_names = generate_averaged_beamformer_solns(scan.start_time, config, ref_bfweights)

    if not beamformer_solns:
        return

    with open(
            f"{config.beamformer_dir}/beamformer_weights_{caltime.isot}.yaml",
            "w",
            encoding="utf-8"
    ) as file:
        yaml.dump(beamformer_solns, file)

    store.put_dict(
        "/mon/cal/bfweights",
        {
            "cmd": "update_weights",
            "val": beamformer_solns["cal_solutions"]})

    # Plot the beamformer solutions
    figure_prefix = f"{config.tempplots}/{caltime}"
    plot_beamformer_weights(
        beamformer_names, config.antennas, config.beamformer_dir,
        outname=figure_prefix, show=False)

    # Plot evolution of the phase over the day
    plot_bandpass_phases(
        beamformer_names,
        config.msdir,
        config.antennas,
        outname=figure_prefix,
        show=False
    )


def calibrate_scan(scan: Scan, config: Configuration, caltype: str):
    """Convert a scan to ms and calibrate it.

    Parameters
    ----------
    scan : Scan
        The scan to include in the measurement set.  Will also include the previous scan.
    config : Configuration
        The current configuration.
    caltype : str
        Must be either 'calibrator' or 'field'. If 'calibrator', single source model is used
        and delay/bandpass calibration is done.  If 'field', multi source model is used
        and delay/bandpass calibration is not done.
    """
    logger = dsa_syslog.DsaSyslogger()

    if scan.source is None:
        raise UndefinedcalError(
            f"No calibrator source defined for scan {scan.start_time.isot}")

    msname, cal = scan.convert_to_ms(
        config.msdir, config.refmjd, simplemodel=caltype == 'calibrator', logger=logger)

    if caltype == 'calibrator':
        # add_single_source_model_to_ms(msname, cal.name, first_true(scan.files))
        delay_bandpass_prefix = ''
    else:
        _ = add_multisource_model_to_ms(msname)
        delay_bandpass_prefix = config.delay_bandpass_prefix
        if not delay_bandpass_prefix:
            return msname, cal, 0

    status = calibrate_measurement_set(
        msname, cal.name, config.refants, delay_bandpass_prefix, logger=logger)



    return msname, cal, status


class UndefinedcalError(RuntimeError):
    pass


def handle_etcd_triggers():
    """Main process to handle etcd triggers under /cmd/cal"""

    store = dsa_store.DsaStore()
    calmanager = CalibrationManager()

    def etcd_callback(etcd_dict: dict):
        """Note that each callback is run in a new thread.

        All of the work is handled by dask, but we need the scan lookup to be thread-safe.
        """
        cmd = etcd_dict['cmd']
        val = etcd_dict['val']
        if cmd == 'rsync':
            pass
            # calmanager.process_file_request(val['hostname'], val['filename'])
        elif cmd == 'field':
            calmanager.process_field_request(val['trigname'], val['mjds'])

    store.add_watch("/cmd/cal", etcd_callback)

    while True:
        calmanager.remove_done_futures()
        print(f"{len(calmanager.futures)} tasks underway")
        time.sleep(60)


if __name__ == "__main__":
    handle_etcd_triggers()
