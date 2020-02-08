"""
A quick script to write to a psrdada buffer in order to 
test a psrdada reader.
"""

import numpy as np
from psrdada import Writer
import os,subprocess
from datetime import datetime
from time import sleep

key_string  = 'adad'
key = 0xadad
nant  = 16
nchan = 1536 #*4
npol  = 2
nbls  = nant*(nant+1)//2

vis_temp = np.arange(nbls*nchan*npol*2,dtype=np.float32)

# Define the data rate, including the buffer size
# and the header size
samples_per_frame = 1
sample_rate = 1/0.134217728
header_size = 4096
buffer_size = int(4*nbls*npol*nchan*samples_per_frame*2)
assert buffer_size==vis_temp.nbytes, \
    "Sample data size and buffer size do not match"

# Create the buffer
data_rate = buffer_size*(sample_rate/samples_per_frame)/1e6
os.system('dada_db -a {0} -b {1} -k {2}'.format(
        header_size, buffer_size, key_string))
print('Buffer created')

# Start the reader
read = 'python ./meridian_fringestop.py'
read_log = open('/home/dsa/tmp/write.log','w')
read_proc = subprocess.Popen(read,shell=True,
                              stdout=read_log,
                              stderr=read_log)
print('Reader started')
sleep(0.1)

# Write to the buffer
writer = Writer(key)
print('Writer created')
for i in range(10):
    page = writer.getNextPage()
    data = np.asarray(page)
    data[...] = vis_temp.view(np.int8)
    if i < 9:
        writer.markFilled()
    else:
        writer.markEndOfData()
    vis_temp += 1
    # Wait to allow reader to clear pages
    sleep(1)

writer.disconnect()
os.system('dada_db -d -k {0}'.format(key_string))
