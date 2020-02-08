import dsacalib.constants as ct
from dsacalib.fringestopping import *
import numpy as np

def read_header(reader):
    """
    Reads a psrdada header.
    
    Args:
        reader: psrdada reader instance
            the reader instance connected to the psrdada buffer
    
    Returns:
        tstart: float
            the start time in mjd seconds
        tsamp: float
            the sample time in seconds
    """
    header = reader.getHeader()
    tsamp = float(header['TSAMP'])
    tstart = float(header['MJD_START'])*ct.seconds_per_day
    return tstart,tsamp

def read_buffer(reader, nbls, nchan, npol):
    """
    Reads a psrdada buffer as float32 and returns the visibilities.
    
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
    
    data = np.asarray(page)
    data = data.view(np.float32)
    data = data.reshape(-1,2).view(np.complex64).squeeze(axis=-1)
    try:
        data = data.reshape(-1,nbls,nchan,npol)
    except ValueError:
        print('incomplete data: {0} out of {1} samples'.format(data.shape[0]%(nbls*nchan*npol),
                                                               nbls*nchan*npol))
        data = data[:data.shape[0]//(nbls*nchan*npol)*(nbls*nchan*npol)].reshape(-1,nbls,nchan,npol)
    return data

def update_time(tstart,samples_per_frame,sample_rate):
    """
    Update the start time and the array of sample times for
    a dataframe.
    
    Args:
        tstart: float
            the start time of the frame, seconds
        samples_per_frame: int
            the number of time samples in the frame
        sample_rate: float
            the sampling rate, seconds
    
    Returns:
        t: array(float)
            the center of the time bin for each sample, seconds
        tstart: float
            the start time of the next dataframe, seconds
    """
    t = tstart + np.arange(samples_per_frame)/sample_rate
    tstart += samples_per_frame/sample_rate
    return t,tstart

# def get_antpos(antenna_order,antpos,casaorder=True):
#     aname = antenna_order[::-1]
#     tp    = np.loadtxt(antpos)
#     blen  = []
#     bname = []
#     for i in np.arange(9)+1:
#         for j in np.arange(i):
#             a1 = int(aname[i])-1
#             a2 = int(aname[j])-1
#             bname.append([a1+1,a2+1])
#             blen.append(tp[a1,1:]-tp[a2,1:])
#     blen  = np.array(blen)
#     if casaorder:
#         blen = blen[::-1]
#         bname = bname[::-1]
#     return blen, bname

def integrate(data,nint):
    """
    A simple integration for testing and benchmarking.  
    Integrates along the time axis
    
    Args:
        data: array
            the data to integrate
            dimensions (time, baseline, channel, polarization)
        nint: int
            the number of consecutive time samples to combine
    
    Returns:
        data: array
           the integrated data
           dimensions (time, baseline, channel, polarization)
        
    """
    (nt,nbls,nchan,npol) = data.shape
    data = data.reshape(-1,nint,nbls,nchan,npol).mean(1)
    return data

def load_visibility_model(fs_table,antenna_order,nint,nbls,fobs):
    """
    Load the visibility model for fringestopping.  If the path
    to the file does not exist or if the model is for a 
    different number of integrations or baselines a new model will
    be created and saved to the file path.
    
    Args:
        fs_table: str
            the full path to the .npz file containing the
            fringestopping model 
        antenna_order: array(int)
            the order of the antennas in the correlator
        nint: int
            the number of time samples to integrate
        nbls: int
            the number of baselines
            
    Returns:
        vis_model: array(complex)
            the visibility model to use for fringestopping
    """
    try:
        fs_data = np.load(fs_table)
        assert fs_data['bw'].shape == (nint,nbls)
    except (FileNotFoundError, AssertionError) as e:
        print('Creating new fringestopping table.')
        df_bls = get_baselines(antenna_order,autocorrs=True,casa_order=False)
        blen   = np.array([df_bls['x_m'],df_bls['y_m'],df_bls['z_m']]).T
        generate_fringestopping_table(blen,nint,transpose=True,
                                     outname=fs_table)
        os.link(fs_table,
            '{0}_{1}.npz'.format(fs_table.strip('.npz'),
            datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')))
    vis_model = zenith_visibility_model_T(fobs,fs_table)
    return vis_model