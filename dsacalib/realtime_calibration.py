"""Calibration service"""
from typing import List, Tuple
from pathlib import Path
import os

import yaml
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import Angle
from dask.distributed import Future
import h5py
import numpy as np
import pandas
from dsautils import dsa_syslog

import dsacalib.constants as ct
from dsacalib.utils import generate_calibrator_source
from dsacalib.ms_io import convert_calibrator_pass_to_ms
from dsacalib.preprocess import rsync_file, first_true, update_caltable
from dsacalib.weights import (
    get_good_solution, filter_beamformer_solutions, average_beamformer_solutions, consistent_correlator)


class H5File:
    """An hdf5 file containing correlated data."""

    def __init__(self, local_path: str, hostname: str = None, remote_path: str = None):
        self.hostname = hostname
        if remote_path:
            self.remote_path = Path(remote_path)
        else:
            self.remote_path = None
        self.path = Path(local_path)
        self.timestamp, self.subband = self.path.stem.split('_')
        self.start_time = Time(self.timestamp)

    def copy(self):
        assert self.hostname, "No hostname defined for remote file."
        assert self.remote_path, "No path defined for remote file."
        rsync_string = (
            f"{self.hostname}.sas.pvt:{self.remote_path} {self.path}")
        rsync_file(rsync_string)

    @property
    def pointing_dec(self) -> u.Quantity:
        """Extract the pointing declination from an h5 file."""
        with h5py.File(str(self.path), mode='r') as h5file:
            pt_dec = h5file['Header']['extra_keywords']['phase_center_dec'][()] * u.rad
        return pt_dec


class Scan:
    """A scan of 16 files collected at the same time, each for a different subband."""

    def __init__(self, h5files: List[H5File], source=None, nsubbands=16):
        """Instantiate the Scan.

        Parameters
        ----------
        h5file : H5File
            A h5file to be included in the scan.
        """
        self.files = [None] * nsubbands
        self.nfiles = 0
        self.source = source

        self.start_time = h5files[0].start_time
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

    def convert_to_ms(self, msdir: str, refmjd: float, logger: dsa_syslog.DsaSyslogger = None) -> str:
        """Convert a scan to a measurement set.

        Parameters
        ----------
        scan : Scan
            A scan containing part of the calibrator pass of interest.
        logger : DsaSyslogger
            The logging interface.  If `None`, messages are only printed.
        """
        date = self.start_time.strftime("%Y-%m-%d")
        cal = generate_calibrator_source(
            self.source['source'], self.source['ra']*u.deg, self.source['dec']*u.deg)
        msname = f"{msdir}/{date}_{self.source['source']}"
        if os.path.exists(f"{msname}.ms"):
            message = f"{msname}.ms already exists.  Not recreating."
            if logger:
                logger.info(message)
            print(logger)
            return msname, cal

        file = first_true(self.files)
        hdf5dir = file.path.parents[0]
        filenames = [
            (self.start_time - 5 * u.min).strftime("%Y-%m-%dT%H:%M:%S"),
            self.start_time.strftime("%Y-%m-%dT%H:%M:%S")]
        convert_calibrator_pass_to_ms(
            cal=cal,
            date=date,
            files=filenames,
            msdir=msdir,
            hdf5dir=hdf5dir,
            refmjd=refmjd,
            logger=logger)

        return msname, cal


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
                if abs(h5file.start_time - scan.start_time) < 3 * u.min:
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
            if scan is scan_to_remove:
                to_remove = i
                break

        if to_remove is not None:
            self.scans[to_remove] = None
            self.futures[to_remove] = None


class CalibratorScan(Scan):
    """A scan (multiple correlator hdf5 files that cover the same time)."""

    def __init__(self, h5files: List[H5File]):

        super().__init__(h5files)

        self.start_sidereal_time = None
        self.end_sidereal_time = None
        self.pt_dec = None
        self.scan_length = 5 * u.min

    def assess(self):
        """Assess if the scan should be converted to a ms for calibration."""
        self.start_sidereal_time, self.end_sidereal_time = [
            (self.start_time + offset).sidereal_time(
                'apparent', longitude=ct.OVRO_LON * u.rad)
            for offset in [0 * u.min, self.scan_length]]

        first_file = first_true(self.files)
        self.pt_dec = first_file.pointing_dec

        caltable = update_caltable(self.pt_dec)
        calsources = pandas.read_csv(caltable, header=0)

        self.source = self.check_for_source(calsources)

    def check_for_source(self, calsources: pandas.DataFrame) -> pandas.DataFrame:
        """Determine if there is any calibrator source of interest is withing +/- 2.5 minutes of the start of the scan.

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
        delta_lst = delta_lst_end - delta_lst_start

        source_index = abs(delta_lst_start) < delta_lst // 2
        if True in source_index:
            return calsources.iloc[source_index.index(True)]


def sidereal_time_delta(time1: Angle, time2: Angle) -> float:
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


def generate_averaged_beamformer_solns(
        start_time: Time, caltime: Time, beamformer_dir: str, antennas: List[int], antennas_core: List[int],
        pols: List[str], refant: int, refmjd: float, ref_bfweights: str, refsb: str = 'sb01'):
    """Generate an averaged beamformer solution.

    Uses only calibrator passes within the last 24 hours or since the snaps
    were restarted.
    """

    if caltime - start_time > 24 * u.h:
        start_time = caltime - 24 * u.h

    # Now we want to find all sources in the last 24 hours
    # start by updating our list with calibrators from the day before
    beamformer_names = get_good_solution(beamformer_dir, refsb, antennas, refant, antennas_core=antennas_core)
    beamformer_names, latest_solns = filter_beamformer_solutions(
        beamformer_names, start_time.mjd, beamformer_dir)

    if len(beamformer_names) == 0:
        return None, None

    try:
        add_reference_bfname(ref_bfweights, beamformer_names, latest_solns,
                             start_time, beamformer_dir)
    except:
        print("could not get reference bname. continuing...")

    averaged_files, avg_flags = average_beamformer_solutions(
        beamformer_names, caltime, beamformer_dir, antennas, refmjd)

    update_solution_dictionary(
        latest_solns, beamformer_names, averaged_files, avg_flags, antennas, pols)
    latest_solns["cal_solutions"]["time"] = caltime.mjd
    latest_solns["cal_solutions"]["bfname"] = caltime.isot

    beamformer_names += [averaged_files[0].split("_")[-1].strip(".dat")]

    return latest_solns, beamformer_names


def update_solution_dictionary(
        latest_solns, beamformer_names, averaged_files, avg_flags, antennas, pols
):
    """Update latest_solns to reflect the averaged beamformer weight parameters."""

    latest_solns["cal_solutions"]["weight_files"] = averaged_files
    latest_solns["cal_solutions"]["source"] = [
        bf.split("_")[0] for bf in beamformer_names
    ]
    latest_solns["cal_solutions"]["caltime"] = [
        float(Time(bf.split("_")[1]).mjd) for bf in beamformer_names
    ]

    # Remove the old bad casas solutions from flagged_antennas
    for key, value in latest_solns["cal_solutions"]["flagged_antennas"].items():
        if "casa solutions flagged" in value:
            value = value.remove("casa solutions flagged")

    # Flag new bad solutions
    idxant, idxpol = np.nonzero(avg_flags)
    for i, ant in enumerate(idxant):
        key = f"{antennas[ant]} {pols[idxpol[i]]}"

        latest_solns["cal_solutions"]["flagged_antennas"][key] = (
            latest_solns["cal_solutions"]["flagged_antennas"].get(key, [])
            + ["casa solutions flagged"])

    # Remove any empty keys in the flagged_antennas dictionary
    to_remove = []
    for key, value in latest_solns["cal_solutions"]["flagged_antennas"].items():
        if not value:
            to_remove += [key]
    for key in to_remove:
        del latest_solns["cal_solutions"]["flagged_antennas"][key]


def add_reference_bfname(ref_bfweights, beamformer_names, latest_solns, start_time, beamformer_dir):
    """
    If the setup of the current beamformer weights matches that of the latest file,
    add the current weights to the beamformer_names list
    """

    if "bfname" in ref_bfweights["val"]:
        ref_bfname = ref_bfweights["val"]["bfname"]
    else:
        # parse from name like "beamformer_weights_sb01_2022-03-18T04:40:15.dat"
        ref_bfname = ref_bfweights["val"]["weights_files"].rstrip(
            ".dat").split("_")[-1]
    print(f"Got reference bfname of {ref_bfname}. Checking solutions...")

    with open(
            f"{beamformer_dir}/beamformer_weights_{ref_bfname}.yaml",
            encoding="utf-8"
    ) as f:
        ref_solns = yaml.load(f, Loader=yaml.FullLoader)

    if consistent_correlator(ref_solns, latest_solns, start_time.mjd):
        beamformer_names.append(ref_bfname)
