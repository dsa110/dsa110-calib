"""
meridian_fringestopping.py
dana.simard@astro.caltech.edu, Feb 2020

This script reads correlated data from a psrdada
ringbuffer.  It fringestops on the meridian for each integrated
sample, before integrating the data and writing it to a hdf5 file.
"""

import numpy as np
from psrdada import Reader
from dsacalib.psrdada_utils import *
from dsacalib.fringestopping import *
from dsacalib.utils import *
from antpos.utils import *
import h5py
import os
from datetime import datetime
from meridian_fringestopping_parameters import *

# Get the visibility model
vis_model = load_visibility_model(fs_table,antenna_order,nint,nbls,fobs)

if test:
    sample_rate = 1/0.134217728
    header_size = 4096
    buffer_size = int(4*nbls*npol*nchan*samples_per_frame*2)
    data_rate = buffer_size*(sample_rate/samples_per_frame)/1e6
    os.system('dada_db -a {0} -b {1} -k {2}'.format(
        header_size, buffer_size, key_string))

print('Initializing reader')
reader = Reader(key)

if test:
    print('Writing data to psrdada buffer')
    os.system('dada_junkdb -r {0} -t 60 -k {2} {1}'.format(
        data_rate,'test_header.txt',key_string))

# Get the start time and the sample time from the reader
#tstart, tsamp = read_header(reader)
#tstart       += nint*tsamp/2
tstart = 58871.66878472222*ct.seconds_per_day
tsamp  = 0.134217728
tstart += nint*tsamp/2
t0     = int(tstart)
tstart -= t0
sample_rate_out = 1/(tsamp*nint)
# 
data_in = np.zeros((samples_per_frame_out*nint,nbls,nchan,npol),
                  dtype=np.complex64
                  ).reshape(-1,samples_per_frame,nbls,nchan,npol)

print('Opening output file')
with h5py.File('{0}.hdf5'.format(fname), 'w') as f:
    # Create output dataset
    print('Output file open')
    vis_ds, t_ds = initialize_hdf5_file(f,fobs,antenna_order,t0,
                                       nbls,nchan,npol,nant)
    
    print('Reading rest of dada buffer')
    idx_frame_out = 0
    while reader.isConnected and not reader.isEndOfData:
        for i in range(data_in.shape[0]):
            try:
                data_in[i,...] = read_buffer(reader,nbls,nchan,npol)
            except:
                # Have to deal with the psrdada buffer ending not 
                # on the right number of frames
                exit
        
        data = fringestop_on_zenith_T(
            data_in.reshape(-1,nbls,nchan,npol),
                                      vis_model,nint)

            # Write out the data 
        t,tstart = update_time(tstart,
                               samples_per_frame_out,sample_rate_out)
        f["vis"].resize((idx_frame_out+1)*samples_per_frame_out,axis=0)
        print(f["vis"].shape)
        f["time_seconds"].resize((idx_frame_out+1)*
                                 samples_per_frame_out,axis=0)
        f["vis"][idx_frame_out*samples_per_frame_out:,...]=data
        f["time_seconds"][idx_frame_out*samples_per_frame_out:]=t
        idx_frame_out += 1
        
    reader.disconnect() 
    if test:
        os.system('dada_db -d -k {0}'.format(key_string))

