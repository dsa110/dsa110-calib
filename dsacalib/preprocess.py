"""Functions used in preprocessing of hdf5 files.

Prepares hdf5 files written by dsa-meridian-fs for conversion to ms.
"""
import re
import os
import subprocess
import numpy as np
import astropy.units as u
from pyuvdata import UVData
# import dsautils.dsa_store as ds
import dsautils.dsa_syslog as dsl

# ETCD interface
# ETCD = ds.DsaStore()

# Logger
LOGGER = dsl.DsaSyslogger()
LOGGER.subsystem("software")
LOGGER.app("dsacalib")

# parameters for freq scrunching
NFREQ = 48
OUTRIGGER_DELAYS = {
    100:  3.6-1.2,
    101:  3.6-1.2,
    102:  3.5-1.2,
    103:  2.2-1.2,
    104:  4.3-1.2,
    105: 12.9-1.2,
    106: 10.7-1.2,
    107: 11.6-1.2,
    108: 12.3-1.2,
    109: 13.4-1.2,
    110: 17.3-1.2,
    111: 16.3-1.2,
    112: 17.1-1.2,
    113: 18.6-1.2,
    114: 20.5-1.2,
    115: 22.3-1.2,
    116:  5.5-1.2,
    117:  6.5-1.2
}

def first_true(iterable, default=False, pred=None):
    """Returns the first true value in the iterable.

    If no true value is found, returns *default*

    If *pred* is not None, returns the first item
    for which pred(item) is true.

    """
    # first_true([a,b,c], x) --> a or b or c or x
    # first_true([a,b], x, f) --> a if f(a) else b if f(b) else x
    return next(filter(pred, iterable), default)

def rsync_file(rsync_string):
    """Rsyncs a file from the correlator machines to dsastorage.
    """
    fname, fdir = rsync_string.split(' ')
    output = subprocess.run(
        [
            'rsync',
#            '--dry-run',
            '-avv',
            '--remove-source-files',
            fname,
            fdir
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    LOGGER.info(
        'rsync of {0} completed\noutput: {1}'.format(
            fname, output.stdout
        )
    )
    fname = fname.split('/')[-1]
    return '{0}{1}'.format(fdir, fname)

def fscrunch_file(fname):
    """Removes outrigger delays before averaging in frequency.

    Leaves file untouched if the number of frequency bins is not divisible
    by the desired number of frequency bins (NFREQ), or is equal to the desired
    number of frequency bins.
    """
    # Process the file
    # print(fname)
    UV = UVData()
    UV.read_uvh5(fname)
    nint = UV.Nfreqs//NFREQ
    if nint > 1 and UV.Nfreqs%nint == 0:
        fobs = UV.freq_array*u.Hz
        # Remove delays for the outrigger antennas
        for ant, delay in OUTRIGGER_DELAYS.items():
            phase_model = np.exp(
                (
                    2.j*np.pi*fobs*(delay*u.microsecond)
                ).to_value(u.dimensionless_unscaled)
            ).reshape(1, 1, 384, 1)
            UV.data_array[
                (UV.ant_1_array!=UV.ant_2_array) &
                (UV.ant_2_array==ant-1)
            ] *= phase_model
            UV.data_array[
                (UV.ant_1_array!=UV.ant_2_array) &
                (UV.ant_1_array==ant-1)
            ] /= phase_model
        # Scrunch in frequency by factor of nint
        UV.frequency_average(n_chan_to_avg=nint)
        if os.path.exists(fname.replace('.hdf5', '_favg.hdf5')):
            os.remove(fname.replace('.hdf5', '_favg.hdf5'))
        UV.write_uvh5(fname.replace('.hdf5', '_favg.hdf5'))
        # Move the original data to a new directory
        corrname = re.findall('corr\d\d', fname)[0]
        os.rename(
            fname,
            fname.replace(
                '{0}'.format(corrname),
                '{0}/full_freq_resolution/'.format(corrname)
            )
         )
        os.rename(
            fname.replace('.hdf5', '_favg.hdf5'),
            fname
        )
    return fname
