"""Service for updating beamformer weights.
"""

import datetime
import warnings
from multiprocessing import Process, Queue
import time
from astropy.time import Time
import dsautils.dsa_store as ds
import dsautils.dsa_syslog as dsl
import dsautils.cnf as dsc
from dsacalib.preprocess import rsync_file
from dsacalib.utils import exception_logger
warnings.filterwarnings("ignore")

# Logger
LOGGER = dsl.DsaSyslogger()
LOGGER.subsystem("software")
LOGGER.app("dsacalib")

# ETCD interface
ETCD = ds.DsaStore()

DATE_STR = '28feb21'
VOLTAGE_DIR = '/mnt/data/dsa110/T3/'
CORR_LIST = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
RSYNC_Q = Queue()
TSLEEP = 10

CONF = dsc.Conf()
PARAMS = CONF.get('corr')

def rsync_handler(inqueue):
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
                rsync_file(
                    fname,
                    remove_source_files=False
                )
            except Exception as exc:
                exception_logger(
                    LOGGER,
                    f"copying of voltage trigger {fname}",
                    exc,
                    throw=False
                )
        else:
            time.sleep(TSLEEP)

def populate_queue(etcd_dict):
    """Copies voltage triggers from corr machines.
    """
    time.sleep(5*60) # Allow correlator voltage service to create metadata
    for specnum in etcd_dict.keys():
        specnum = (int(specnum)-477)*16
        for corr in PARAMS['ch0'].keys():
            fname = f"{corr}.sas.pvt:/home/ubuntu/data/fl*.out.{specnum}"
            fnameout = f"/mnt/data/dsa110/T3/{corr}/{DATE_STR}/"
            print(f"{fname} {fnameout}")
            RSYNC_Q.put(f"{fname} {fnameout}")
            fname = f"{corr}.sas.pvt:/home/ubuntu/data/fl*.out.{specnum}.json"
            print(f"{fname} {fnameout}")
            RSYNC_Q.put(f"{fname} {fnameout}")
            LOGGER.info(f"Copied voltage trigger {specnum} from {corr}")

if __name__ == "__main__":
    processes = {
        'rsync': {
            'nthreads': 2,
            'queue': RSYNC_Q,
            'processes': []
        },
    }
    # Start etcd watch
    ETCD.add_watch('/mon/corr/1/trigger', populate_queue)
    # Start all threads
    for name, process in processes.items():
        for i in range(process['nthreads']):
            process['processes'] += [Process(
                target=rsync_handler,
                args=(
                    process['queue'],
                ),
                daemon=True
            )]
        for pinst in process['processes']:
            pinst.start()

    while True:
        ETCD.put_dict(
            '/mon/cal/voltagecopy',
            {
                "alive": True,
                "cadence": 60,
                "time": Time(datetime.datetime.utcnow()).isot
            }
        )
        time.sleep(60)
