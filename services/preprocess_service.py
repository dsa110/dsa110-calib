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
MAX_ASSESS = 8   # was 4: must exceed the number of workers that can be
                 # alive at once (MAX_WAIT / set cadence), or the ring
                 # wraps onto a live worker and its queue gets drained
                 # by two processes at once.

# Maximum amount of time that gather_files will wait for all correlator files
# to be gathered, in seconds
MAX_WAIT = 10 * 60

# Time to sleep if a queue is empty before trying to get an item
TSLEEP = 30

# Tolerance, in seconds, for deciding that two correlator files belong to the
# same set. Each node stamps the filename from its own clock, so a set is
# routinely spread over a few seconds.
GATHER_TOL = 60.0

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
    print(rsync_string)
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

    Will wait for a maximum of MAX_WAIT (10 min) from the time the first
    file is received.

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
    # Honour the module-level MAX_WAIT (10 min) instead of a hardcoded
    # 15 min. The hardcoded value exceeded 3x the ~5 min set cadence, so
    # three workers were always holding slots concurrently.
    end = time.time() + MAX_WAIT
    while nfiles < ncorr and time.time() < end:
        if not inqueue.empty():
            fname = inqueue.get()
            filelist.append(fname)
            nfiles += 1
        time.sleep(1)
    outqueue.put(filelist)


def gather_key(fname):
    """Timestamp of a correlator file, used to group files into one set.

    Parsed as a naive UTC datetime so that sets are matched on how far apart
    they actually are. The previous key was the filename truncated to the
    minute (``basename.split('_')[0][:-2]``), which grouped by string
    equality: files a second apart but on opposite sides of a minute
    boundary landed in different sets, and each partial set was then
    assessed on its own once MAX_WAIT expired. Observed 2026-08-01, where
    02:23:59 and 02:24:00 gathered as 7 and 9 files instead of 16.

    Parameters
    ----------
    fname : str
        Path to a correlator hdf5 file, e.g. ``.../2026-08-01T02:24:00_sb03.hdf5``.

    Returns
    -------
    datetime.datetime
        The timestamp encoded in the filename.
    """
    basename = os.path.splitext(os.path.basename(fname))[0]
    return datetime.datetime.strptime(
        basename.split('_')[0], "%Y-%m-%dT%H:%M:%S")


def gather_files(inqueue, outqueue, ncorr=CONFIG.ncorr, max_assess=MAX_ASSESS, tsleep=TSLEEP):
    """Gather files from all correlators.

    Will wait for a maximum of MAX_WAIT (10 min) from the time the first
    file is received.

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
                key = gather_key(fname)
                # Reclaim slots whose worker has exited. gather_worker holds
                # its slot for up to MAX_WAIT while waiting for files that may
                # never arrive, so several slots can be occupied at once by
                # finished-but-uncleared or still-waiting workers. Without
                # this, `nfiles_assessed % max_assess` eventually lands on a
                # slot whose worker is STILL ALIVE, and that slot's queue is
                # then drained by two processes at once: each takes some of
                # the files, neither ever reaches ncorr, and no set is
                # assessed again. Observed 2026-08-01 03:30, where all 16
                # subbands were on disk but no gather ever fired.
                for idx, proc in enumerate(gather_processes):
                    if proc is not None and not proc.is_alive():
                        proc.join(timeout=0)
                        gather_names[idx] = None
                        gather_processes[idx] = None
                # Match against an open set by proximity rather than by an
                # exact key, so that a set spanning a minute boundary is not
                # split in two.
                islot = next(
                    (
                        idx for idx, open_key in enumerate(gather_names)
                        if open_key is not None
                        and abs((open_key - key).total_seconds()) < GATHER_TOL
                    ),
                    None
                )
                if islot is None:
                    # Prefer a genuinely free slot; only fall back to the
                    # round-robin index when every worker is still busy.
                    try:
                        islot = gather_names.index(None)
                    except ValueError:
                        islot = nfiles_assessed % max_assess
                        LOGGER.warning(
                            "all %d gather slots busy; reusing slot %d whose "
                            "worker is still alive", max_assess, islot)
                    gather_names[islot] = key
                    gather_processes[islot] = Process(
                        target=gather_worker,
                        args=(
                            gather_queues[islot],
                            outqueue
                        ),
                        daemon=True)
                    gather_processes[islot].start()
                    nfiles_assessed += 1
                gather_queues[islot].put(fname)
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
