"""A service to preprcocess hdf5 files before calibration.
"""
import datetime
import sys
import warnings
from multiprocessing import Process, Queue
import time
from functools import partial
import os

import pandas
import h5py
import numpy as np
from astropy.time import Time
from astropy.coordinates import Angle
import astropy.units as u

import dsautils.dsa_store as ds
import dsautils.dsa_syslog as dsl

import dsacalib.constants as ct
from dsacalib import config
from dsacalib.preprocess import rsync_file, first_true
from dsacalib.preprocess import update_caltable
from dsacalib.utils import exception_logger

import fnmatch

# make sure warnings do not spam syslog
warnings.filterwarnings("ignore")

# Logger
LOGGER = dsl.DsaSyslogger()
LOGGER.subsystem("software")
LOGGER.app("dsacalib")

# ETCD interface
ETCD = ds.DsaStore()

# FIFO Queues for rsync, freq scrunching, calibration
RSYNC_Q = Queue()
GATHER_Q = Queue()
ASSESS_Q = Queue()
CALIB_Q = Queue()

# Maximum number of files per correlator that can be assessed for calibration
# needs at one time.
MAX_ASSESS = 4

# Maximum amount of time that gather_files will wait for all correlator files
# to be gathered, in seconds
MAX_WAIT = 10 * 60

# Time to sleep if a queue is empty before trying to get an item
TSLEEP = 30

# Configuration
CONFIG = config.Configuration()


def populate_queue(etcd_dict, queue=RSYNC_Q):
    """Populates the fscrunch and rsync queues using etcd.

    Etcd watch callback function.
    """
    cmd = etcd_dict['cmd']
    val = etcd_dict['val']
    if cmd != 'rsync':
        return
    rsync_string = f"{val['hostname']}.pro.pvt:{val['filename']} {CONFIG.hdf5dir}/"
    queue.put(rsync_string)


def rsync_handler(inqueue, outqueue=None):
    """Handles in and out queues of preprocessing tasks.

    Parameters
    ----------
    task_fn : function
        The function to execute, with a single argument.
    inqueue : multiprocessing.Queue instance
        The queue containing the arguments to `task_fn`.
    outqueue : multiprocessing.Queue instance
        The queue to write the otuput of `task_fn` to.
    """
    while True:
        if not inqueue.empty():
            fname = inqueue.get()
            try:
                fname = rsync_file(fname, logger=LOGGER)
                # test for spl in name
                if outqueue is not None:
                    if not fnmatch.fnmatch(fname,"*spl*"):
                        outqueue.put(fname)
            except Exception as exc:
                exception_logger(
                    LOGGER,
                    f"preprocessing of file {fname}",
                    exc,
                    throw=False
                )
        else:
            time.sleep(TSLEEP)


def gather_worker(inqueue, outqueue, ncorr=CONFIG.ncorr):
    """Gather all files that match a filename.

    Will wait for a maximum of 15 minutes from the time the first file is
    received.

    Parameters
    ----------
    inqueue : multiprocessing.Queue instance
        The queue containing the filenames, max size of 16 (i.e. one file per
        corr node).
    outqueue : multiprocessing.Queue instance
        The queue in which to place the gathered files (as a list).
    """
    nfiles = 0
    filelist = []
    # Times out after 15 minutes
    end = time.time() + 60 * 15
    while nfiles < ncorr and time.time() < end:
        if not inqueue.empty():
            fname = inqueue.get()
            filelist.append(fname)
            nfiles += 1
        time.sleep(1)
    outqueue.put(filelist)


def gather_files(inqueue, outqueue, ncorr=CONFIG.ncorr, max_assess=MAX_ASSESS, tsleep=TSLEEP):
    """Gather files from all correlators.

    Will wait for a maximum of 15 minutes from the time the first file is
    received.

    Parameters
    ----------
    inqueue : multiprocessing.Queue instance
        The queue containing the ungathered filenames .
    outqueue : multiprocessing.Queue instance
        The queue in which to place the gathered files (as a list).
    """
    gather_queues = [Queue(ncorr) for idx in range(max_assess)]
    gather_names = [None] * max_assess
    gather_processes = [None] * max_assess
    nfiles_assessed = 0
    while True:
        if not inqueue.empty():
            try:
                fname = inqueue.get()
                print(fname)
                basename = os.path.splitext(os.path.basename(fname))[0]
                basename = basename.split('_')[0][:-2]
                if not basename in gather_names:
                    gather_names[nfiles_assessed % max_assess] = basename
                    gather_processes[nfiles_assessed % max_assess] = Process(
                        target=gather_worker,
                        args=(
                            gather_queues[nfiles_assessed % max_assess],
                            outqueue
                        ),
                        daemon=True)
                    gather_processes[nfiles_assessed % max_assess].start()
                    nfiles_assessed += 1
                gather_queues[
                    gather_names.index(basename)
                ].put(fname)
            except Exception as exc:
                exception_logger(
                    LOGGER,
                    f"preprocessing of file {fname}",
                    exc,
                    throw=False
                )
        else:
            time.sleep(tsleep)


def assess_file(inqueue, outqueue, caltime=CONFIG.caltime, filelength=CONFIG.filelength):
    """Decides whether calibration is necessary.

    Sends a command to etcd using the monitor point /cmd/cal if the file should
    be calibrated.

    Parameters
    ----------
    inqueue : multiprocessing.Queue instance
        The queue containing the gathered filenames.
    outqueue : multiprocessing.Queue instance
        The queue to which the calname and gathered filenames (as a tuple) if
        the file is appropriate for calibration.
    caltime : astropy quantity
        The amount of time around the calibrator to be converted to
        a measurement set for calibration. Used to assess whether any part of
        the desired calibrator pass is in a given file.
    """
    # TODO: also pass the prefix for the delay_bandpass_cal to calibration
    while True:
        if not inqueue.empty():
            try:
                flist = inqueue.get()
                fname = first_true(flist)
                print(f"Assessing {len(flist)} files {fname}")
                datet = fname.split('/')[-1][:19]
                tstart = Time(datet).sidereal_time(
                    'apparent',
                    longitude=ct.OVRO_LON * u.rad
                )
                tend = (Time(datet) + filelength).sidereal_time(
                    'apparent',
                    longitude=ct.OVRO_LON * u.rad
                )
                a0 = (
                    caltime * np.pi * u.rad
                    / (ct.SECONDS_PER_SIDEREAL_DAY * u.s)).to_value(u.rad)
                with h5py.File(fname, mode='r') as h5file:
                    pt_dec = h5file['Header']['extra_keywords']['phase_center_dec'][()] * u.rad
                caltable = update_caltable(pt_dec)
                calsources = pandas.read_csv(caltable, header=0)
                for _index, row in calsources.iterrows():
                    if isinstance(row['ra'], str):
                        rowra = Angle(row['ra'])
                    else:
                        rowra = Angle(row['ra'] * u.deg)
                    delta_lst_start = (
                        tstart - rowra
                    ).to_value(u.rad) % (2 * np.pi)
                    if delta_lst_start > np.pi:
                        delta_lst_start -= 2 * np.pi
                    delta_lst_end = (
                        tend - rowra
                    ).to_value(u.rad) % (2 * np.pi)
                    if delta_lst_end > np.pi:
                        delta_lst_end -= 2 * np.pi
                    if delta_lst_start < a0 < delta_lst_end:
                        calname = row['source']
                        print(f"Calibrating {calname}")
                        outqueue.put((calname, flist))
                    else:
                        print(f"Not calibrating {row['source']} with ra {rowra.to(u.deg)} using lst {tstart.to(u.deg)}")

            except Exception as exc:
                exception_logger(
                    LOGGER,
                    f"preprocessing of file {fname}",
                    exc,
                    throw=False
                )
        else:
            time.sleep(TSLEEP)


if __name__ == "__main__":
    # Start etcd watch
    ETCD.add_watch('/cmd/cal', populate_queue)
    processes = {
        'rsync': {
            'task_fn': rsync_handler,
            'queue': RSYNC_Q,
            'outqueue': GATHER_Q,
            'daemon': False,
            'process': None
        },
        'gather': {
            'task_fn': gather_files,
            'queue': GATHER_Q,
            'outqueue': ASSESS_Q,
            'daemon': False,
            'process': None
        },
        'assess': {
            'task_fn': assess_file,
            'queue': ASSESS_Q,
            'outqueue': CALIB_Q,
            'daemon': True,
            'process': None
        }}
    try:

        for key in ['rsync', 'gather', 'assess']:
            pdict = processes[key]
            pdict['process'] = Process(
                target=pdict['task_fn'],
                args=(
                    pdict['queue'],
                    pdict['outqueue']
                ),
                daemon=pdict['daemon']
            )
            pdict['process'].start()

        while True:
            for name, pinfo in processes.items():
                ETCD.put_dict(
                    f'/mon/cal/{name}_process',
                    {
                        "queue_size": pinfo['queue'].qsize(),
                        "ntasks_alive": sum([
                            pinfo['process'].is_alive()
                        ]),
                    }
                )
            ETCD.put_dict(
                '/mon/service/calpreprocess',
                {
                    "cadence": 60,
                    "time": Time(datetime.datetime.utcnow()).mjd
                }
            )
            while not CALIB_Q.empty():
                (calname_fromq, flist_fromq) = CALIB_Q.get()
                ETCD.put_dict(
                    '/cmd/cal',
                    {
                        'cmd': 'calibrate',
                        'val': {
                            'calname': calname_fromq,
                            'flist': flist_fromq
                        }
                    }
                )
            time.sleep(60)

    except (KeyboardInterrupt, SystemExit):
        processes['gather']['process'].terminate()
        processes['gather']['process'].join()
        sys.exit()
