import numpy as np
from psrdada import Reader
from dsacalib.psrdada_utils import *
from dsacalib.fringestopping import *
import h5py
import os

# input parameters - should be parsed from command line arguments
test = True
nant  = 32
nchan = 2048
npol  = 2
fname = 'test'
nint  = 5
antenna_order = [8,5,7,4,3,0,9,1,6,2]

nbls  = nant*(nant+1)/2

# Read the most recent baseline table
blen,bname = get_antpos(antenna_order,'/home/simard/dsa_calib/data/antpos_ITRF.txt')
# Check that there is a fringestopping table
# Generate a new one if not
generate_fringestopping_table(blen,nint)


if test:
    samples_per_frame = 25
    sample_rate = 25
    header_size = 4096
    buffer_size = int(4*nbls*npol*nchan*samples_per_frame)
    data_rate = buffer_size*(sample_rate/samples_per_frame)/1e6
    print('dada_db -d create -a {0} -b {1}'.format(
    header_size, buffer_size))
    os.system('dada_db -d create -a {0} -b {1}'.format(
    header_size, buffer_size))


# Should add the context manager info to the classes to the reader so 
# it closes nicely
reader = Reader()

if test:
    print('dada_junkdb -r {0} -t 60 {1}'.format(
        data_rate,'test_header.txt'))
    os.system('dada_junkdb -r {0} -t 60 {1}'.format(
        data_rate,'test_header.txt'))

# tstart,sample_rate,sample_idx = read_header(reader)
# Eventually, should change this to use integers to conserve times?
tstart      = 0.  # mjs
sample_rate = 25 # in Hz
samples_per_frame = 25

# Other things
samples_per_frame_out = samples_per_frame//nint
sample_rate_out = sample_rate//nint
tstart     += nint/sample_rate/2

# Read the first frame
data = read_buffer(reader,nant,nchan,npol)
#data = integrate(data)
data = fringestop_on_zenith(data,fobs,nint)

with h5py.File('{0}.hdf5'.format(fname), 'w') as f:
    # create output dataset 
    vis_ds = f.create_dataset("vis", 
                            (samples_per_frame_out,nbls,nchan,npol), 
                            maxshape=(None,nbls,nchan,npol),
                            dtype=np.complex64,chunks=True,
                            data = data)
    
    t,tstart = update_time(tstart,samples_per_frame_out,sample_rate_out)
    
    t_ds = f.create_dataset("time_mjd_seconds",
                           (samples_per_frame_out),maxshape=(None),
                           dtype=np.float32,chunks=True,
                           data = t)
    nframes = 1
    
    while reader.isConnected and not reader.isEndOfData:
        data = read_buffer(reader,nant,nchan,npol)
        # Do stuff here to the data
        #data = integrate(data)
        data = fringestop_on_zenith(data,fobs,nint)
        # Write out the data 
        t,tstart = update_time(tstart,samples_per_frame_out,sample_rate_out)
        f["vis"].resize(samples_per_frame_out*(nframes+1),axis=0)
        f["time_mjd_seconds"].resize(samples_per_frame_out*(nframes+1),axis=0)
        f["vis"][samples_per_frame_out*nframes:samples_per_frame_out*(nframes+1),...]=data
        f["time_mjd_seconds"][samples_per_frame_out*nframes:samples_per_frame_out*(nframes+1)]=t
        nframes +=1 
        
    reader.disconnect() 
    if test:
        os.system('dada_db -d destroy')
