"""Sync files from other machines to dsastorage on change of /cmd/store key.
"""
import datetime
import warnings
import time
import subprocess

from astropy.time import Time

import dsautils.dsa_store as ds
import dsautils.dsa_syslog as dsl

# make sure warnings do not spam syslog
warnings.filterwarnings("ignore")

# Logger
LOGGER = dsl.DsaSyslogger()
LOGGER.subsystem("software")
LOGGER.app("dsacalib")

# ETCD interface
ETCD = ds.DsaStore()


def rsync_handler(etcd_dict: dict):
    """Etcd watch callback function.

    Rsyncs the file if requested.
    """
    cmd = etcd_dict['cmd']
    val = etcd_dict['val']
    if cmd == 'rsync':
        rsync_file(val['source'], val['dest'], val['remove_source_files'])


def rsync_file(source: str, dest: str, remove_source_files: bool = False) -> None:
    """Rsyncs a file from the correlator machines to dsastorage.

    Parameters
    ----------
    source : str
        E.g. 'corr06.sas.pvt:/home/ubuntu/data/2020-06-24T12:32:06.hdf5'
    dest: str
        E.g. '/mnt/data/dsa110/correlator/corr06/'
    remove_source_files : bool
        If true, source files are removed after the rsync.
    """
    command = " ".join([
        ". ~/.keychain/dsa-storage-sh; rsync -avv ",
        "--remove-source-files " if remove_source_files else "",
        f"{source} {dest}"])

    with subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True
    ) as process:
        proc_stdout = str(process.communicate()[0].strip())

    print(proc_stdout)
    LOGGER.info(proc_stdout)


if __name__=="__main__":

    ETCD.add_watch('/cmd/store', rsync_handler)

    while True:
        ETCD.put_dict(
            '/mon/service/store',
            {
                "cadence": 60,
                "time": Time(datetime.datetime.utcnow()).mjd
            }
        )
        time.sleep(60)
