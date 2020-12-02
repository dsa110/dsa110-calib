"""A service to preprocess hdf5 files before calibration.
"""

import warnings
# make sure warnings do not spam syslog
warnings.filterwarnings("ignore")
from multiprocessing import Process, Queue
import time
from pkg_resources import resource_filename
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas
import numpy as np
from astropy.time import Time
from astropy.coordinates import Angle
import astropy.units as u
import dsautils.dsa_store as ds
import dsautils.dsa_syslog as dsl
import dsacalib.constants as ct
from dsacalib.preprocess import rsync_file, fscrunch_file, first_true #, assess_file, calibrate_file
from dsacalib.utils import exception_logger
from dsacalib.routines import get_files_for_cal, calibrate_measurement_set
from dsacalib.ms_io import convert_calibrator_pass_to_ms
from dsacalib.plotting import summary_plot, plot_current_beamformer_solutions

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
    """
    cmd = etcd_dict['cmd']
    val = etcd_dict['val']
    if cmd == 'rsync':
        rsync_string = '{0}:{1} /mnt/data/dsa110/{0}/'.format(
            val['hostname'],
            val['filename']
        )
        RSYNC_Q.put(rsync_string)
    # THis part doesnt seem to be happening
    elif cmd == 'calibrate':
        print(
            'adding to calibration queue: {0} {1}'.format(
                val['calname'],
                val['flist']
            )
        )
        CALIB_Q.put([
            val['calname'],
            val['flist']
        ])

def task_handler(task_fn, inqueue, outqueue=None):
    """Threaded frequency scrunching.
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
    """Gather all files that match a filename.
    """
    filelist = [None]*NCORR
    nfiles = 0
    # Times out after 15 minutes
    end = time.time() + 60*15
    #print(inqueue.qsize())
    #print(nfiles, NCORR, end, time.time())
    while nfiles < NCORR and time.time() < end:
        if not inqueue.empty():
            fname = inqueue.get()
            corrid = int(fname.split('/')[4].strip('corr'))
            # 21 is currently replacing 4
            if corrid==21:
                filelist[4-1] = fname
            else:
                filelist[corrid-1] = fname
            nfiles += 1
            #print(filelist)
        time.sleep(1)
    #print(filelist)
    outqueue.put(filelist)

def gather_files(inqueue, outqueue):
    """Gather files from all correlators.
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
                    gather_names[nfiles_assessed%MAX_ASSESS] = fname.split('/')[-1][:-7]

                    # Start subprocess to watch this queue
                    # seems to do this
                    # print(gather_names)
                    # but not the rest of this - it is throwing an exception
                    gather_processes[nfiles_assessed%MAX_ASSESS] = Process(
                        target=gather_worker,
                        args=(
                            gather_queues[nfiles_assessed%MAX_ASSESS],
                            outqueue
                        )
                    )
                    gather_processes[nfiles_assessed%MAX_ASSESS].start()
                    nfiles_assessed += 1
                    # print('nfiles_assessed: {0}'.format(nfiles_assessed))
                gather_queues[
                    gather_names.index(fname.split('/')[-1][:-7])
                ].put(fname)
                # print([gq.qsize() for gq in gather_queues])
                # print([gp.is_alive() if gp is not None else 'None' for gp in gather_processes])
            except Exception as exc:
                exception_logger(
                    LOGGER,
                    'preprocessing of file {0}'.format(fname),
                    exc,
                    throw=False
                )
        else:
            time.sleep(TSLEEP)

def assess_file(flist, caltime=15*u.min):
    """Decides on calibration that is necessary.

    Sends a command to etcd using the monitor point /cmd/cal if the file should
    be calibrated.
    """
    myetcd = ds.DsaStore()
    caltable = resource_filename('dsacalib', 'data/calibrator_sources.csv')
    calsources = pandas.read_csv(caltable, header=0)
    fname = first_true(flist)
    print(fname)
    datetime = fname.split('/')[-1][:19]
    tstart = Time(datetime).sidereal_time(
        'apparent',
        longitude=ct.OVRO_LON*u.rad
    )
    tend = (Time(datetime)+15*u.min).sidereal_time(
        'apparent',
        longitude=ct.OVRO_LON*u.rad
    )
    a0 = (caltime*np.pi*u.rad/(ct.SECONDS_PER_SIDEREAL_DAY*u.s)).to_value(u.rad)

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
        # Possibilities:
        # Source transits in the file (delta_lst_start between -a0 and 0) and 
        #     transit ends in the file (delta_lst_end > a0) -> calibrate
        #
        # Source doesnt transit in the file (delta_lst_start between 0 and a0) but
        #     transit ends in the file (delta_lst_end > a0) -> calibrate
        #
        # Source transits in the file (delta_lst_start between -a0 and 0) but 
        #     transit doenst end in the file (delta_lst_end < a0)
        # 
        if delta_lst_start < a0 and delta_lst_end > a0:
            calname = row['source']
            print('Calibrating {0}'.format(calname))
            # This is failing - the assess queue gets stuck 
            # and doesnt assess any more files, or I get a segfault
            myetcd.put_dict(
                '/cmd/cal',
                {
                    'cmd': 'calibrate',
                    'val': {
                        'calname': calname,
                        'flist': flist
                    }
                }
            )
            print(myetcd.get_dict('/cmd/cal'))
    return fname

def calibrate_file(calib_list, caltime=15*u.min, refant='102'):
    """Generates and calibrates a measurement set.
    """
    myetcd = ds.DsaStore()
    calname = calib_list[0]
    flist = calib_list[1]
    date = first_true(flist).split('/')[-1][:-14]
    caltable = resource_filename('dsacalib', 'data/calibrator_sources.csv')

    # Parameters that are mostly constant
    refcorr = '01'
    filelength = 15*u.min
    msdir = '/mnt/data/dsa110/msfiles/'
    # hdf5dir = '/mnt/data/dsa110/'
    msname = '{0}/{1}_{2}'.format(msdir, date, calname)
    date_specifier = '{0}*'.format(date)

    LOGGER.info('Creating {0}.ms'.format(msname))
    
    filenames = get_files_for_cal(
        caltable,
        refcorr,
        caltime,
        filelength,
        date_specifier=date_specifier,
    )

#    print('getting ready to put a dict to etcd')
#     myetcd.put_dict(
#         '/mon/cal/calibration',
#         {
#             "transit_time": filenames[date][calname]['transit_time'].mjd,
#             "calibration_source": calname,
#             "filelist": flist,
#             "status": -1
#         }
#     )
    
    convert_calibrator_pass_to_ms(
        cal=filenames[date][calname]['cal'],
        date=date,
        files=filenames[date][calname]['files'],
        duration=caltime
    )

    LOGGER.info('{0}.ms created'.format(msname))

    status = calibrate_measurement_set(
        msname,
        filenames[date][calname]['cal'],
        refant=refant,
        bad_antennas=None,
        bad_uvrange='2~27m',
        forsystemhealth=True
    )

#     myetcd.put_dict(
#         '/mon/cal/calibration',
#         {
#             "transit_time": filenames[date][calname]['transit_time'].mjd,
#             "calibration_source": calname,
#             "filelist": flist,
#             "status": status
#         }
#     )
    
    # This should be made more general for more antennas
    antennas = np.array(
        [24, 25, 26, 27, 28, 29, 30, 31, 32,
         33, 34, 35, 20, 19, 18, 17, 16, 15,
         14, 13, 100, 101, 102, 116, 103]
    )
    figure_path = '{0}/figures/{1}_{2}'.format(msdir, date, calname)
    with PdfPages('{0}.pdf'.format(figure_path)) as pdf:
        for j in range(len(antennas)//10+1):
            fig = summary_plot(
                msname,
                calname,
                2,
                ['B', 'A'],
                antennas[j*10:(j+1)*10]
            )
            pdf.savefig(fig)
            plt.close()
    plot_current_beamformer_solutions(
        filenames[date][calname]['files'],
        calname,
        date,
        corrlist=[1, 2, 3, 21, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        outname=figure_path,
        show=False
    )

    LOGGER.info('Calibrated {0}.ms with status {1}'.format(msname, status))

    return calib_list

if __name__=="__main__":
    # CONFIGURATION OF PIPELINE IS DEFINED BY THIS LIST
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
        'assess': {
            'nthreads': 1,
            'task_fn': assess_file,
            'queue': ASSESS_Q,
            'outqueue': None,
            'processes': []
        },
        'calibrate': {
            'nthreads': 1, 
            'task_fn': calibrate_file,
            'queue': CALIB_Q,
            'outqueue': None,
            'processes': []
        }
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

    while True:
        for name in processes.keys():
            ETCD.put_dict(
                '/mon/cal/{0}_process'.format(name),
                {
                    "queue_size": processes[name]['queue'].qsize(),
                    "ntasks_alive": sum([
                        pinst.is_alive() for pinst in processes[name]['processes']
                    ]),
                    "ntasks_total": processes[name]['nthreads']
                }
            )
        time.sleep(60)
