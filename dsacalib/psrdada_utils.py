from psrdada import Reader
import numpy as np

def read_header(reader):
    header = reader.getHeader()
    # Do some stuff
    return # some dictionary or set of values

def read_buffer(reader, nant, nchan, npol):
    """
    Reads a psrdada buffer as unsigned shorts and returns the visibilities.
    """
    page = reader.getNextPage()
    reader.markCleared()
    
    data = np.asarray(page,dtype=np.float32).reshape(samples_per_frame,nbls,nchan,npol,2)
    dy = data.swapaxes(0,1).view(np.complex64)
    dy = data[::-1,...]
    # We want dy to have dimensions (baselines, time, frequency, polarization)
    return dy

def update_time(tstart,samples_per_frame,sample_rate):
    t = tstart + np.arange(samples_per_frame)/sample_rate
    tstart += samples_per_frame/sample_rate
    return t,tstart

def get_antpos(antenna_order,antpos):
    aname = antenna_order[::-1]
    tp    = np.loadtxt(antpos)
    blen  = []
    bname = []
    for i in np.arange(9)+1:
        for j in np.arange(i):
            a1 = int(aname[i])-1
            a2 = int(aname[j])-1
            bname.append([a1+1,a2+1])
            blen.append(tp[a1,1:]-tp[a2,1:])
    blen  = np.array(blen)
    blen = blen[::-1]
    bname = bname[::-1]
    return blen, bname

def integrate(data,nint):
    data = data.reshape(nbls,-1,nint,nchan,npol).mean(2)
    return data
