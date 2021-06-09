"""Creating and manipulating measurement sets from T3 visibilities.
"""
import yaml
import h5py
import numpy as np
from pkg_resources import resource_filename
import astropy.units as u
from antpos.utils import get_itrf
from pyuvdata import UVData
from dsautils import cnf
from dsamfs.io import initialize_uvh5_file, update_uvh5_file
from dsacalib.ms_io import uvh5_to_ms
from dsacalib.fringestopping import calc_uvw
import dsacalib.constants as ct
from dsacalib.preprocess import remove_outrigger_delays

PARAMFILE = resource_filename('dsacalib', 'data/T3_parameters.yaml')
with open(PARAMFILE) as YAMLF:
    T3PARAMS = yaml.load(YAMLF, Loader=yaml.FullLoader)['T3corr']

MYCONF = cnf.Conf()
CORRPARAMS = MYCONF.get('corr')

def get_mjd(armed_mjd, utc_start, specnum):
    tstart = (armed_mjd+utc_start*4*8.192e-6/86400+
              (1/(250e6/8192/2)*specnum/ct.SECONDS_PER_DAY))
    return tstart

def get_blen(antennas):
    ant_itrf = get_itrf(
        latlon_center=(ct.OVRO_LAT*u.rad, ct.OVRO_LON*u.rad, ct.OVRO_ALT*u.m)
    ).loc[antennas]
    xx = np.array(ant_itrf['dx_m'])
    yy = np.array(ant_itrf['dy_m'])
    zz = np.array(ant_itrf['dz_m'])
    # Get uvw coordinates
    nants = len(antennas)
    nbls = (nants*(nants+1))//2
    blen = np.zeros((nbls, 3))
    bname = []
    k = 0
    for i in range(nants):
        for j in range(i, nants):
            blen[k, :] = np.array([
                xx[i]-xx[j],
                yy[i]-yy[j],
                zz[i]-zz[j]
            ])
            bname += ['{0}-{1}'.format(
                antennas[i],
                antennas[j]
            )]
            k += 1
    return blen, bname

def generate_T3_ms(name, pt_dec, tstart, ntint, nfint, filelist, params=T3PARAMS, start_offset=None, end_offset=None):
    """Generates a measurement set from the T3 correlations.
    
    Parameters
    ----------
    name : str
        The name of the measurement set.
    pt_dec : quantity
        The pointing declination in degrees or equivalient.
    tstart : astropy.time.Time instance
        The start time of the correlated data.
    ntint : float
        The number of time bins that have been binned together (compared to the
        native correlator resolution).
    nfint : float
        The number of frequency bins that have been binned together (compared
        to the native resolution).
    filelist : dictionary
        The correlator data files for each node.
    params : dictionary
        T3 parameters.
    """
    msname = '{0}/{1}'.format(params['msdir'], name)
    antenna_order = params['antennas']
    fobs = params['f0_GHz']+params['deltaf_MHz']*1e-3*nfint*(
        np.arange(params['nchan']//nfint)+0.5)
    antenna_order = params['antennas']
    nant = len(antenna_order)
    nbls = (nant*(nant+1))//2
    tsamp = params['deltat_s']*ntint*u.s
    tobs = tstart + (np.arange(params['nsubint']//ntint)+0.5)*tsamp
    if start_offset is not None:
        assert end_offset is not None
        tobs = tobs[start_offset:end_offset]
    blen, bname = get_blen(params['antennas'])
    bu, bv, bw = calc_uvw(
        blen,
        tobs.mjd,
        'HADEC',
        np.zeros(len(tobs))*u.rad,
        np.ones(len(tobs))*pt_dec
    )
    buvw = np.array([bu, bv, bw]).T
    hdf5_files = []
    for corr, ch0 in params['ch0'].items():
        fobs_corr = fobs[ch0//nfint:(ch0+params['nchan_corr'])//nfint]
        data = np.fromfile(
            filelist[corr],
            dtype=np.float32
        )
        data = data.reshape(-1, 2)
        data = data[..., 0] + 1.j*data[..., 1]
        data = data.reshape(-1, nbls, len(fobs_corr), 4)[..., [0, -1]]
        if start_offset is not None:
            data = data[start_offset:end_offset, ...]
        outname = '{2}/{1}_{0}.hdf5'.format(corr, name, params['msdir'])
        with h5py.File(outname, 'w') as fhdf5:
            initialize_uvh5_file(
                fhdf5,
                len(fobs_corr),
                2,
                pt_dec.to_value(u.rad),
                antenna_order,
                fobs_corr
            )
            update_uvh5_file(
                fhdf5,
                data,
                tobs.jd,
                tsamp,
                bname,
                buvw,
                np.ones(data.shape, np.float32)
            )
        UV = UVData()
        UV.read(outname, file_type='uvh5')
        remove_outrigger_delays(UV)
        UV.write_uvh5(outname, clobber=True)
        hdf5_files += [outname]
    uvh5_to_ms(
        hdf5_files,
        msname
    )
    return msname
