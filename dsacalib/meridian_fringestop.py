import numpy as np
from psrdada import Reader
from dsacalib.psrdada_utils import *
from dsacalib.fringestopping import *
from antpos.utils import *
import h5py
import os
from datetime import datetime

# input parameters - should be parsed from command line arguments
test = True
key_string  = 'dbdb'
key = 0xdbdb
nant  = 16
nchan = 1536*4
npol  = 2
fname = 'test'
nint  = 10
#antenna_order = [8,5,7,4,3,0,9,1,6,2]
antenna_order = np.arange(1,nant+1,1)
fobs = 1.13 + np.arange(nchan)*0.400/nchan
nbls  = nant*(nant+1)//2
print('{0} baselines'.format(nbls))

try:
    fs_data = np.load('fringestopping_table.npz')
    assert fs_data['bw'].shape == (nint,nbls)
except (FileNotFoundError, AssertionError) as e:
    print('Creating new fringestopping table.')
    df_bls = get_baselines(antenna_order,autocorrs=True,casa_order=False)
    blen   = np.array([df_bls['x_m'],df_bls['y_m'],df_bls['z_m']]).T
    generate_fringestopping_table(blen,nint,transpose=True)
    os.link('fringestopping_table.npz','fringestopping_table_{0}.npz'.format(
        datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')))
vis_model = zenith_visibility_model_T(fobs,'fringestopping_table.npz')
    
if test:
    samples_per_frame = 10
    sample_rate = 1/0.134217728
    header_size = 4096
    buffer_size = int(4*nbls*npol*nchan*samples_per_frame*2)
    # Need to check that this size makes sense and that my code is
    # interpreting things correctily
    data_rate = buffer_size*(sample_rate/samples_per_frame)/1e6
    print('')
    print('Run the next command in a new terminal, then hit enter in this one')
    print('dada_db -a {0} -b {1} -k {2}'.format(
        header_size, buffer_size, key_string))
    s = input('Enter --> ')


# Should add the context manager info to the classes to the reader so 
# it closes nicely
reader = Reader(key)

if test:
    print('')
    print('Run the next command in another window, then hit enter in this one')
    print('dada_junkdb -r {0} -t 60 -k {2} {1}'.format(
        data_rate,'test_header.txt',key_string))
    s = input('Enter --> ')

tstart, tsamp = read_header(reader)
tstart       += nint*tsamp/2

# Read the first frame
data              = read_buffer(reader,nbls,nchan,npol)
samples_per_frame = data.shape[0]
assert samples_per_frame%nint == 0
samples_per_frame_out = samples_per_frame//nint
sample_rate_out = 1/(tsamp*nint)

data = fringestop_on_zenith_T(data,vis_model,nint)

with h5py.File('{0}.hdf5'.format(fname), 'w') as f:
    # create output dataset
    print('Output file open')
    ds_fobs = f.create_dataset("fobs_GHz",(nchan,),dtype=np.float32,data=fobs)
    ds_ants = f.create_dataset("antenna_order",(nant,),dtype=np.int,data=antenna_order)

    vis_ds = f.create_dataset("vis", 
                            (samples_per_frame_out,nbls,nchan,npol), 
                            maxshape=(None,nbls,nchan,npol),
                            dtype=np.complex64,chunks=True,
                            data = data)
    
    t,tstart = update_time(tstart,samples_per_frame_out,sample_rate_out)
    
    t_ds = f.create_dataset("time_mjd_seconds",
                           (samples_per_frame_out,),maxshape=(None,),
                           dtype=np.float32,chunks=True,
                           data = t)
    nsamp_out = samples_per_frame_out

    print('Reading rest of dada buffer')
    while reader.isConnected and not reader.isEndOfData:
        
        data = read_buffer(reader,nbls,nchan,npol)
        #data = integrate(data,nint)
        data = fringestop_on_zenith_T(data,vis_model,nint)
        nsamp_frame = data.shape[0]
        # Write out the data 
        t,tstart = update_time(tstart,nsamp_frame,sample_rate_out)
        f["vis"].resize(nsamp_out+nsamp_frame,axis=0)
        print(f["vis"].shape)
        f["time_mjd_seconds"].resize(nsamp_out+nsamp_frame,axis=0)
        f["vis"][nsamp_out:nsamp_out+nsamp_frame,...]=data
        f["time_mjd_seconds"][nsamp_out:nsamp_out+nsamp_frame]=t
        nsamp_out += nsamp_frame
        
    reader.disconnect() 
    if test:
        print('')
        print('Destroy the dada buffer with:')
        print('dada_db -d -k {0}'.format(key_string))
