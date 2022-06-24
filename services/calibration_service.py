"""A service to create measurement sets and calibrate data.
"""
from typing import List
import time
import datetime
import os
import shutil
import sys
import socket
from pathlib import Path
import warnings
from multiprocessing import Process, Queue
import yaml

import h5py
import numpy as np
import astropy.units as u
from astropy.time import Time
import dsautils.dsa_store as ds
import dsautils.dsa_syslog as dsl

import matplotlib
matplotlib.use("Agg")  # noqa
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from dsacalib.preprocess import first_true, update_caltable
from dsacalib.utils import exception_logger
from dsacalib.routines import get_files_for_cal, calibrate_measurement_set
from dsacalib.ms_io import convert_calibrator_pass_to_ms, caltable_to_etcd
from dsacalib.hdf5_io import extract_applied_delays
from dsacalib.weights import (
    write_beamformer_solutions, average_beamformer_solutions,
    filter_beamformer_solutions,
    get_good_solution, consistent_correlator
)
from dsacalib.plotting import (
    summary_plot, plot_bandpass_phases, plot_beamformer_weights)
from dsacalib.config import Configuration

warnings.filterwarnings("ignore")

# Logger
LOGGER = dsl.DsaSyslogger()
LOGGER.subsystem("software")
LOGGER.app("dsacalib")

TSLEEP = 60
CALIB_Q = Queue()


def calibrate_file(calname, flist, **kwargs):
    """Calibrate a calibrator pass."""

    config = Configuration()
    etcd = ds.DsaStore()
    date = Path(first_true(flist)).stem.split("T")[0]

    msname = f"{config.msdir}/{date}_{calname}"
    date_specifier = f"{date}*"

    # Get the pointing declination from the file
    with h5py.File(first_true(flist), mode="r") as h5file:
        pt_dec = (
            h5file["Header"]["extra_keywords"]["phase_center_dec"][()] * u.rad)

    # Get the list of sources at the current pointing dec
    caltable = update_caltable(pt_dec)
    # Find the ones that have files
    filenames = get_files_for_cal(
        caltable,
        config.hdf5dir,
        duration=config.caltime,
        filelength=config.filelength,
        date_specifier=date_specifier,
    )
    caltime = filenames[date][calname]["transit_time"]
    caltime.precision = 0
    etcd.put_dict(
        "/mon/cal/calibration",
        {
            "transit_time": filenames[date][calname]["transit_time"].mjd,
            "calibration_source": calname,
            "filelist": flist,
            "status": -1
        }
    )

    # Generate the measurement set
    message = f"Creating {msname}.ms at dec {pt_dec}"
    LOGGER.info(message)
    print(message)
    if not os.path.exists(f"{msname}.ms"):
        print("writing ms")
        convert_calibrator_pass_to_ms(
            cal=filenames[date][calname]["cal"],
            date=date,
            files=filenames[date][calname]["files"],
            msdir=config.msdir,
            hdf5dir=config.hdf5dir,
            refmjd=config.refmjd,
            logger=LOGGER,
        )
        print("done writing ms")
        message = f"{msname}.ms created."
        LOGGER.info(message)
        print(message)
    else:
        message = f"{msname}.ms already exists.  Not recreating."
        LOGGER.info(message)
        print(message)

    # Calibrate for system health & flagging
    status = calibrate_measurement_set(
        msname,
        filenames[date][calname]["cal"],
        config.refants,
        delay_bandpass_cal_prefix="",
        logger=LOGGER,
        throw_exceptions=False,
    )

    # Write solutions to etcd
    caltable_to_etcd(
        msname,
        calname,
        filenames[date][calname]["transit_time"].mjd,
        status,
        logger=LOGGER
    )

    etcd.put_dict(
        "/mon/cal/calibration",
        {
            "transit_time": filenames[date][calname]["transit_time"].mjd,
            "calibration_source": calname,
            "filelist": flist,
            "status": status
        }
    )

    LOGGER.info(
        f"Calibrated {msname}.ms for system health with status {status}"
    )

    try:
        generate_summary_plot(
            date, msname, calname, config.antennas, config.tempplots, config.webplots)
    except Exception as exc:
        exception_logger(
            LOGGER,
            f"plotting of calibration solutions for {msname}.ms",
            exc,
            throw=False)

    try:
        applied_delays = extract_applied_delays(
            first_true(flist), config.antennas)
        # Write beamformer solutions for one source
        write_beamformer_solutions(
            msname, calname, caltime, config.antennas, applied_delays, config.beamformer_dir,
            config.pols, config.nchan, config.nchan_spw, config.bw_GHz,
            config.chan_ascending, config.f0_GHz, config.ch0, config.refmjd,
            flagged_antennas=config.antennas_not_in_bf)
    except Exception as exc:
        exception_logger(
            LOGGER,
            f"calculation of beamformer weights for {msname}.ms",
            exc,
            throw=False)
    try:
        ref_bfweights = etcd.get_dict("/mon/cal/bfweights")
        beamformer_solns, beamformer_names = generate_averaged_beamformer_solns(
            config.snap_start_time, caltime, config.beamformer_dir,
            config.antennas, config.antennas_core, config.pols, config.refants[0], config.refmjd, ref_bfweights)

        if beamformer_solns:
            with open(
                    f"{config.beamformer_dir}/beamformer_weights_{caltime.isot}.yaml",
                    "w",
                    encoding="utf-8"
            ) as file:
                yaml.dump(beamformer_solns, file)

            etcd.put_dict(
                "/mon/cal/bfweights",
                {
                    "cmd": "update_weights",
                    "val": beamformer_solns["cal_solutions"]})

            # Plot the beamformer solutions
            figure_prefix = f"{config.tempplots}/{caltime}"
            plot_beamformer_weights(
                beamformer_names, config.antennas, config.beamformer_dir,
                outname=figure_prefix, show=False)
            # store_file(
            #     f"{figure_prefix}_averagedweights.png",
            #     f"{config.webplots}/bfw_current.png",
            #     remove_source_files=False)
            # store_file(
            #     f"{figure_prefix}_averagedweights.png",
            #     f"{config.webplots}/allpngs/{caltime}_averagedweights.png",
            #     remove_source_files=True)

            # Plot evolution of the phase over the day
            plot_bandpass_phases(
                beamformer_names,
                config.msdir,
                config.antennas,
                outname=figure_prefix,
                show=False
            )
            plt.close("all")
            # store_file(
            #     f"{figure_prefix}_phase.png",
            #     f"{config.webplots}/phase_current.png",
            #     remove_source_files=False)
            # store_file(
            #     f"{figure_prefix}_phase.png",
            #     f"{config.webplots}/allpngs/{caltime}_phase.png",
            #     remove_source_files=True)
    except Exception as exc:
        exception_logger(
            LOGGER,
            f"averaging of beamformer weights for {msname}.ms",
            exc,
            throw=False)                


def generate_averaged_beamformer_solns(
        start_time: Time, caltime: Time, beamformer_dir: str, antennas: List[int], antennas_core: List[int], pols: List[str],
        refant: int, refmjd: float, ref_bfweights: str, refsb: str = 'sb01'):
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


def generate_summary_plot(date, msname, calname, antennas, tempdir, webplots):
    """Generate a summary plot and put it in webplots."""

    figure_path = f"{tempdir}/{date}_{calname}.pdf"
    with PdfPages(figure_path) as pdf:
        for j in range(len(antennas) // 10 + 1):
            fig = summary_plot(
                msname,
                calname,
                2,
                ["B", "A"],
                antennas[j * 10:(j + 1) * 10]
            )
            pdf.savefig(fig)
            plt.close(fig)

    # store_file(
    #     figure_path, f"{webplots}/allpngs/{date}_{calname}.pdf", remove_source_files=False)
    # store_file(
    #     figure_path, f"{webplots}/summary_current.pdf", remove_source_files=True)


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


def calibrate_file_manager(inqueue=CALIB_Q):
    """Manages the queue and creates subprocesses for calibration.
    """
    while True:
        if not inqueue.empty():
            try:
                val = inqueue.get()
                calname = val["calname"]
                flist = val["flist"]
            except Exception as exc:
                exception_logger(
                    LOGGER,
                    "attempt to retrieve calibration task from queue",
                    exc,
                    throw=False)
            else:
                # start a subprocess
                calib_process = Process(
                    target=calibrate_file,
                    args=(calname, flist),
                    daemon=True)
                calib_process.start()
                calib_process.join()
            time.sleep(TSLEEP)


def populate_queue(etcd_dict, outqueue=CALIB_Q):
    """Populates the calibration queue.
    """
    if etcd_dict["cmd"] == "calibrate":
        outqueue.put(etcd_dict["val"])


# def store_file(source: str, target: str, remove_source_files: bool = False) -> None:
#     """Sends an etcd command for a file to be stored on dsa-storage."""
#     if socket.gethostname() == "dsa-storage":
#         shutil.copyfile(source, target)
#     else:
#         ETCD.put_dict("/cmd/store", {
#             'cmd': 'rsync',
#             'val': {
#                 'source': source,
#                 'dest': target,
#                 'remove_source_files': remove_source_files}})


# def copy_figure(source, target):
#     """Copy figure to new path."""
#     if os.path.exists(target):
#         os.unlink(target)
#     shutil.copyfile(source, target)


def watch_for_calibration():
    """Watch for calibration commands from etcd.
    """
    etcd = ds.DsaStore()
    etcd.add_watch("/cmd/cal", populate_queue)
    while True:
        etcd.put_dict(
            "/mon/service/calibration",
            {
                "cadence": 60,
                "time": Time(datetime.datetime.utcnow()).mjd
            }
        )
        time.sleep(60)


if __name__ == "__main__":
    processes = {}
    processes["calibrate"] = Process(
        target=calibrate_file_manager,
        args=(CALIB_Q,)
    )
    processes["calibrate"].start()
    print("calibration process started")
    processes["watch"] = Process(
        target=watch_for_calibration,
        daemon=True
    )
    processes["watch"].start()
    print("watch process started")

    try:
        while True:
            try:
                assert processes["watch"].is_alive(), "Watch process has died"  # needs a timeout
                assert processes["calibrate"].is_alive(), "Calibration process has died"  # needs a timeout
            except AssertionError as exc:
                print(f'Caught exception: {exc}. Unsure if this is true. Continuing.')

            print(f"{CALIB_Q.qsize()} objects in calibration queue")
            time.sleep(5 * 60)
            print('Processes have died!')

    except (KeyboardInterrupt, SystemExit, AssertionError) as exc:
        # Terminate non-daemon processes
        print(f'Caught exception: {exc}')
        print('Exiting calibration')
        processes["calibrate"].terminate()
        processes["calibrate"].join()
        sys.exit()
