from psrdada import Reader
import dsacalib.constants as ct
import numpy as np

def read_header(reader):
    header = reader.getHeader()
    tsamp = float(header['TSAMP'])
    tstart = float(header['MJD_START'])*ct.seconds_per_day
    return tsamp, tstart

def read_buffer(reader, nbls, nchan, npol):
    """
    Reads a psrdada buffer as unsigned shorts and returns the visibilities.
    
    Args:
      reader: psrdada Reader instance
        an instance of the Reader class for the psrdada buffer you want to read
      nbls: int
        the number of baselines
      nchan: int
        the number of frequency channels
      npol: int 
        the number of polarizations

    Returns:
      data: ndarray
        dimensions time, baselines, channels, polarization
    """
    page = reader.getNextPage()
    reader.markCleared()
    
    data = np.asarray(page,dtype=np.float32).reshape(-1,2).view(np.complex64).squeeze(axis=-1)
    try:
        data = data.reshape(-1,nbls,nchan,npol)
    except ValueError:
        print('incomplete data: {0} out of {1} samples'.format(data.shape[0]%(nbls*nchan*npol),
                                                               nbls*nchan*npol))
        data = data[:data.shape[0]//(nbls*nchan*npol)*(nbls*nchan*npol)].reshape(-1,nbls,nchan,npol)
    # Remove reshaping needed for casa order
    #data = data.swapaxes(0,1)
    #data = data[::-1,...]
    return data

def update_time(tstart,samples_per_frame,sample_rate):
    t = tstart + np.arange(samples_per_frame)/sample_rate
    tstart += samples_per_frame/sample_rate
    return t,tstart

def get_antpos(antenna_order,antpos,casaorder=True):
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
    if casaorder:
        blen = blen[::-1]
        bname = bname[::-1]
    return blen, bname

def integrate(data,nint):
    (nbls,nt,nchan,npol) = data.shape
    data = data.reshape(nbls,-1,nint,nchan,npol).mean(2)
    return data
