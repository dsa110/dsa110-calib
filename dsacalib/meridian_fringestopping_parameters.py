# meridian_fringestopping_parameters.py
# Parameter file for meridian_fringestopping.py
import numpy as np

# Whether or not to create & populate the psrdada buffer
# If true, the buffer is created, populated using dada_junkdb
# and then destroyed at the end of the script.
test = False 

# The key of the psrdada buffer, in both string and numeric form
key_string  = 'adad'
key = 0xadad

# The number of antennas in the array
nant  = 16
# The order of the antennas in the correlator
antenna_order = np.arange(1,nant+1,1)

# The number of frequency channels
nchan = 6144//4 #1536
# The lowest edge of the lowest freq channel, GHz
f0 = 1.28
# BW, GHz
bw = 0.250/4
# Whether the data is ordered from lowest to highest f, 
# or highest to lowest.  If True, data ordered from 
# lowest to highest frequency
chan_ascending = True

# The number of polarizations
npol  = 2

# The prefix for the hdf5 file to contain the output of
# meridian_fringestopping.py 
fname = 'test_psrdada'

# The number of time samples per frame in the psrdada buffer
samples_per_frame = 1
# The number of time samples to integrate after fringestopping
nint  = 1
# The number of frames to output each time after fringestopping
samples_per_frame_out = 1
assert samples_per_frame_out*nint%samples_per_frame==0

# Path to fringestopping table
fs_table = 'fringestopping_table.npz'

# additional parameters set using the ones set above
df = bw/nchan
fobs = f0 + df/2 + np.arange(nchan)*df
if not chan_ascending:
    fobs = fobs[::-1]
nbls = nant*(nant+1)//2

