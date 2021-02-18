"""A service to preprocess hdf5 files before calibration.
"""

import warnings
# make sure warnings do not spam syslog
warnings.filterwarnings("ignore")
from multiprocessing import Process, Queue
import time
from pkg_resources import resource_filename
import pandas
import numpy as np
from astropy.time import Time
from astropy.coordinates import Angle
import astropy.units as u
import dsautils.dsa_store as ds
import dsautils.dsa_syslog as dsl
import dsacalib.constants as ct
from dsacalib.preprocess import rsync_file, fscrunch_file, first_true
from dsacalib.utils import exception_logger

# Logger
LOGGER = dsl.DsaSyslogger()
LOGGER.subsystem("software")
LOGGER.app("dsacalib")

# ETCD interface
ETCD = ds.DsaStore()

# FIFO Queues for rsync, freq scrunching, calibration
FSCRUNCH_Q = Queue()
RSYNC_Q = Queue()
GATHER_Q = Queue()
ASSESS_Q = Queue()
CALIB_Q = Queue()

# Number of correlators
NCORR = 16

# Maximum number of files per correlator that can be assessed for calibration
# needs at one time.
MAX_ASSESS = 4

# Maximum amount of time that gather_files will wait for all correlator files
# to be gathered, in seconds
MAX_WAIT = 5*60

# Time to sleep if a queue is empty before trying to get an item
TSLEEP = 10

def populate_queue(etcd_dict):
    """Populates the fscrunch and rsync queues using etcd.

    Etcd watch callback function.
    """
    cmd = etcd_dict['cmd']
    val = etcd_dict['val']
    if cmd == 'rsync':
        rsync_string = '{0}.sas.pvt:{1} /mnt/data/dsa110/correlator/{0}/'.format(
            val['hostname'],
            val['filename']
        )
        RSYNC_Q.put(rsync_string)

def task_handler(task_fn, inqueue, outqueue=None):
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
                fname = task_fn(fname)
                if outqueue is not None:
                    outqueue.put(fname)
            except Exception as exc:
                exception_logger(
                    LOGGER,
                    'preprocessing of file {0}'.format(fname),
                    exc,
                    throw=False
                )
        else:
            time.sleep(TSLEEP)

def gather_worker(inqueue, outqueue):
    # TODO: Replace corr 21 with corr 4
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
    filelist = [None]*NCORR
    nfiles = 0
    # Times out after 15 minutes
    end = time.time() + 60*15
    while nfiles < NCORR and time.time() < end:
        if not inqueue.empty():
            fname = inqueue.get()
            corrid = int(fname.split('/')[5].strip('corr'))
            # 21 is currently replacing 4
            if corrid==21:
                filelist[4-1] = fname
            else:
                filelist[corrid-1] = fname
            nfiles += 1
            #print(filelist)
        time.sleep(1)
    outqueue.put(filelist)

def gather_files(inqueue, outqueue):
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
    gather_queues = [Queue(NCORR) for idx in range(MAX_ASSESS)]
    gather_names = [None]*MAX_ASSESS
    gather_processes = [None]*MAX_ASSESS
    nfiles_assessed = 0
    while True:
        if not inqueue.empty():
            try:
                fname = inqueue.get()
                print(fname)
                if not fname.split('/')[-1][:-7] in gather_names:
                    gather_names[nfiles_assessed%MAX_ASSESS] = \
                        fname.split('/')[-1][:-7]
                    gather_processes[nfiles_assessed%MAX_ASSESS] = \
                        Process(
                            target=gather_worker,
                            args=(
                                gather_queues[nfiles_assessed%MAX_ASSESS],
                                outqueue
                            )
                        )
                    gather_processes[nfiles_assessed%MAX_ASSESS].start()
                    nfiles_assessed += 1
                gather_queues[
                    gather_names.index(fname.split('/')[-1][:-7])
                ].put(fname)
            except Exception as exc:
                exception_logger(
                    LOGGER,
                    'preprocessing of file {0}'.format(fname),
                    exc,
                    throw=False
                )
        else:
            time.sleep(TSLEEP)

def assess_file(inqueue, outqueue, caltime=15*u.min):
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
    caltable = resource_filename(
        'dsacalib',
        'data/calibrator_sources.csv'
    )
    calsources = pandas.read_csv(caltable, header=0)
    while True:
        if not inqueue.empty():
            try:
                flist = inqueue.get()
                fname = first_true(flist)
                datetime = fname.split('/')[-1][:19]
                tstart = Time(datetime).sidereal_time(
                    'apparent',
                    longitude=ct.OVRO_LON*u.rad
                )
                tend = (Time(datetime)+15*u.min).sidereal_time(
                    'apparent',
                    longitude=ct.OVRO_LON*u.rad
                )
                a0 = (caltime*np.pi*u.rad/
                      (ct.SECONDS_PER_SIDEREAL_DAY*u.s)).to_value(u.rad)
                for _index, row in calsources.iterrows():
                    delta_lst_start = (
                        tstart-Angle(row['ra'])
                    ).to_value(u.rad)%(2*np.pi)
                    if delta_lst_start > np.pi:
                        delta_lst_start -= 2*np.pi
                    delta_lst_end = (
                        tend-Angle(row['ra'])
                    ).to_value(u.rad)%(2*np.pi)
                    if delta_lst_end > np.pi:
                        delta_lst_end -= 2*np.pi
                    if delta_lst_start < a0 < delta_lst_end:
                        calname = row['source']
                        print('Calibrating {0}'.format(calname))
                        outqueue.put((calname, flist))
            except Exception as exc:
                exception_logger(
                    LOGGER,
                    'preprocessing of file {0}'.format(fname),
                    exc,
                    throw=False
                )
        else:
            time.sleep(TSLEEP)

if __name__=="__main__":
    processes = {
        'rsync': {
            'nthreads': 1,
            'task_fn': rsync_file,
            'queue': RSYNC_Q,
            'outqueue': FSCRUNCH_Q,
            'processes': []
        },
        'fscrunch': {
            'nthreads': 1,
            'task_fn': fscrunch_file,
            'queue': FSCRUNCH_Q,
            'outqueue': GATHER_Q,
            'processes': []
        },
    }
    # Start etcd watch
    ETCD.add_watch('/cmd/cal', populate_queue)
    # Start all threads
    for name in processes.keys():
        for i in range(processes[name]['nthreads']):
            processes[name]['processes'] += [Process(
                target=task_handler,
                args=(
                    processes[name]['task_fn'],
                    processes[name]['queue'],
                    processes[name]['outqueue']
                ),
                daemon=True
            )]
        for pinst in processes[name]['processes']:
            pinst.start()

    processes['gather'] = {
        'nthreads': 1,
        'task_fn': gather_files,
        'queue': GATHER_Q,
        'outqueue': ASSESS_Q,
        'processes': []
    }
    processes['gather']['processes'] += [Process(
        target=gather_files,
        args=(
            GATHER_Q,
            ASSESS_Q
            )
    )]
    processes['gather']['processes'][0].start()

    processes['assess'] = {
        'nthreads': 1,
        'task_fn': assess_file,
        'queue': ASSESS_Q,
        'outqueue': CALIB_Q,
         'processes': []
    }
    processes['assess']['processes'] += [Process(
        target=assess_file,
        args=(
            ASSESS_Q,
            CALIB_Q
        )
    )]
    processes['assess']['processes'][0].start()

    while True:
        for name in processes.keys():
            ETCD.put_dict(
                '/mon/cal/{0}_process'.format(name),
                {
                    "queue_size": processes[name]['queue'].qsize(),
                    "ntasks_alive": sum([
                        pinst.is_alive() for pinst in
                        processes[name]['processes']
                    ]),
                    "ntasks_total": processes[name]['nthreads']
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
