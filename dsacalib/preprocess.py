"""Functions used in preprocessing of hdf5 files.

Prepares hdf5 files written by dsa-meridian-fs for conversion to ms.
"""
import re
import os
import subprocess
import numpy as np
import astropy.units as u
from pyuvdata import UVData
import dsautils.cnf as cnf
CONF = cnf.Conf()
MFS_CONF = CONF.get('fringe')
# parameters for freq scrunching
NFREQ = MFS_CONF['nfreq_scrunch']
# Outrigger delays are those estimated by Morgan Catha based on the cable
# length.
OUTRIGGER_DELAYS = MFS_CONF['outriggger_delays']

def first_true(iterable, default=False, pred=None):
    """Returns the first true value in the iterable.

    If no true value is found, returns ``default``
    If ``pred`` is not None, returns the first item
    for which pred(item) is true.

    Parameters
    ----------
    iterable : list
        The list for which to find the first True item.
    default :
        If not False, then this is returned if no true value is found.
        Defaults False.
    pred :
        If not None, then the first item for which pred(item) is True is
        returned.
    """
    # first_true([a,b,c], x) --> a or b or c or x
    # first_true([a,b], x, f) --> a if f(a) else b if f(b) else x
    return next(filter(pred, iterable), default)

def rsync_file(rsync_string, remove_source_files=True):
    """Rsyncs a file from the correlator machines to dsastorage.

    Parameters
    ----------
    rsync_string : str
        E.g. 'corr06.sas.pvt:/home/ubuntu/data/2020-06-24T12:32:06.hdf5 /mnt/data/dsa110/correlator/corr06/'
    """
    fname, fdir = rsync_string.split(' ')
    if remove_source_files:
        output = subprocess.run(
            [
                'rsync',
                '-avv',
                '--remove-source-files',
                fname,
                fdir
            ],
            #check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
    else:
        output = subprocess.run(
            [
                'rsync',
                '-avv',
                fname,
                fdir
            ],
            #check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT
        )
    fname = fname.split('/')[-1]
    if output.returncode != 0:
        print(output)
    return '{0}{1}'.format(fdir, fname)

def fscrunch_file(fname):
    """Removes outrigger delays before averaging in frequency.

    Leaves file untouched if the number of frequency bins is not divisible
    by the desired number of frequency bins (NFREQ), or is equal to the desired
    number of frequency bins.

    Parameters
    ----------
    fname : str
        The full path to the file to process.
    """
    # Process the file
    # print(fname)
    UV = UVData()
    UV.read_uvh5(fname)
    nint = UV.Nfreqs//NFREQ
    if 'applied_delays_ns' in UV.extra_keywords.keys():
        applied_delays = np.array(
            UV.extra_keywords['applied_delays_ns'].split(' ')
        ).astype(np.int).reshape(-1, 2)
    else:
        applied_delays = np.zeros((UV.Nants_telescope, 2), np.int)
    if nint > 1 and UV.Nfreqs%nint == 0:
        fobs = UV.freq_array*u.Hz
        # Remove delays for the outrigger antennas
        for ant, delay in OUTRIGGER_DELAYS.items():
            phase_model = np.exp(
                (
                    2.j*np.pi*fobs*(delay*u.nanosecond)
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
            applied_delays[ant-1, :] += int(delay)
        if 'applied_delays_ns' in UV.extra_keywords.keys():
            UV.extra_keywords['applied_delays_ns'] = np.string_(
                ' '.join([str(d) for d in applied_delays.flatten()])
            )
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
