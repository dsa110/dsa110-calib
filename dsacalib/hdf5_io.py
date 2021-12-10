"""
DSACALIB/HDF5_UTILS.PY

Dana Simard, dana.simard@astro.caltech.edu, 10/2019

Routines to interact w/ hdf5 visibilities recorded by DSA-110.
"""
# Always import scipy before importing casatools.
import numpy as np
import h5py
from antpos.utils import get_baselines
# pylint will complain about this, but iers.conf.iers_auto_url_mirror must be
# set before astropy.time.Time is imported.
import astropy.units as u
from dsacalib import constants as ct
from astropy.utils import iers
iers.conf.iers_auto_url_mirror = ct.IERS_TABLE
iers.conf.auto_max_age = None
from astropy.time import Time # pylint: disable=wrong-import-position

def read_hdf5_file(
    fl, source=None, dur=50*u.min, autocorrs=True, badants=None, quiet=True
):
    """Reads visibilities from a hdf5 file written by dsa110-fringestopping.

    Parameters
    ----------
    fl : str
        Full path to the hdf5 file.
    source : source instance
        The source to extract from the hdf5 file. If set to ``None``, the
        entire file is extracted. Defaults ``None``.
    dur : astropy quantity
        The duration of the observation to extract, in minutes or an equivalent
        unit. Only used if `source` is not set to ``None``.  Defaults
        ``50*u.min``.
    autocorrs : Boolean
        If set to ``True``, both the autocorrelations and the crosscorrelations
        are extracted from the hdf5 file.  If set to ``False``, only the
        crosscorrelations are extracted.  Defaults ``True``.
    badants : list
        Antennas that have been flagged as bad or offline.  If provied,
        baselines that include these antennas will not be extracted.  If set to
        ``None``, all baselines in the hdf5 file are extracted.  Defaults
        ``None``.
    quiet : Boolean
        If set to ``False``, information about the file will be printed to
        stdout. Defaults ``True``.

    Returns
    -------
    fobs : array
        The observing frequency of the center of each channel in the
        visibilities, in GHz.
    blen : ndarray
        The ITRF baseline lengths, dimensions (nbaselines, 3).
    bname : list(str)
        The name of each baseline.
    tstart : float
        The start time of the extracted visibilities in MJD. If the data does
        not contain the specified ``source``, a value of ``None`` is returned.
    tstop : float
        The stop time of the extracted visibilities in MJD. If the data does
        not contain the specified ``source``, a value of ``None`` is returned.
    vis : ndarray
        The extracted visibilities, dimensions (baselines, time, frequency,
        polarization).
    mjd : array
        The time of the center of each timebin in the visibilities, in MJD.
    transit_idx : int
        The index along the time axis corresponding to the meridian crossing of
        the source given by `source`.  If `source` is set to ``None``, a value
        of ``None`` is returned.
    antenna_order : list
        The antenna names, in the order they are in in the hdf5 visibilities.
    tsamp : float
        The sampling time in seconds.
    """
    with h5py.File(fl, 'r') as f:
        antenna_order = list(f['antenna_order'][...])
        nant = len(antenna_order)
        fobs = f['fobs_GHz'][...]
        mjd = (f['time_seconds'][...]+f['tstart_mjd_seconds'])/ \
            ct.SECONDS_PER_DAY
        nt = len(mjd)
        tsamp = (mjd[-1]-mjd[0])/(nt-1)*ct.SECONDS_PER_DAY
        lst0 = Time(mjd[0], format='mjd').sidereal_time(
            'apparent', longitude=ct.OVRO_LON*u.rad).to_value(u.rad)
        lst = np.angle(np.exp(
            1j*(lst0+2*np.pi/ct.SECONDS_PER_SIDEREAL_DAY*np.arange(nt)*tsamp)))

        if source is not None:
            lstmid = lst0 - source.direction.hadec(obstime=mjd[0])[0]
            seg_len = (dur/2*(15*u.deg/u.h)).to_value(u.rad)
            if not quiet:
                print("\n-------------EXTRACT DATA--------------------")
                print("Extracting data around {0}".format(lstmid*180/np.pi))
                print("{0} Time samples in data".format(nt))
                print("LST range: {0:.1f} --- ({1:.1f}-{2:.1f}) --- {3:.1f}deg"
                      .format(lst[0]*180./np.pi, (lstmid-seg_len)*180./np.pi,
                              (lstmid+seg_len)*180./np.pi, lst[-1]*180./np.pi))
            idxl = np.argmax(np.absolute(np.exp(1j*lst)+
                                         np.exp(1j*lstmid)*
                                         np.exp(-1j*seg_len)))
            idxr = np.argmax(np.absolute(np.exp(1j*lst)+
                                         np.exp(1j*lstmid)*np.exp(1j*seg_len)))
            transit_idx = np.argmax(np.absolute(np.exp(1j*lst)+
                                                np.exp(1j*(lstmid))))

            mjd = mjd[idxl:idxr]
            vis = f['vis'][idxl:idxr, ...]
            if not quiet:
                print("Extract: {0} ----> {1} sample; transit at {2}".format(
                    idxl, idxr, transit_idx))
                print("----------------------------------------------")

        else:
            vis = f['vis'][...]
            transit_idx = None

    df_bls = get_baselines(antenna_order, autocorrs=True, casa_order=True)
    blen = np.array([df_bls['x_m'], df_bls['y_m'], df_bls['z_m']]).T
    bname = np.array([bn.split('-') for bn in df_bls['bname']])
    bname = bname.astype(int)

    if not autocorrs:
        cross_bls = list(range((nant*(nant+1))//2))
        i = -1
        for j in range(1, nant+1):
            i += j
            cross_bls.remove(i)
        vis = vis[:, cross_bls, ...]
        blen = blen[cross_bls, ...]
        bname = bname[cross_bls, ...]

        assert vis.shape[0] == len(mjd)
        assert vis.shape[1] == len(cross_bls)

    if badants is not None:
        good_idx = list(range(len(bname)))
        for i, bn in enumerate(bname):
            if (bn[0] in badants) or (bn[1] in badants):
                good_idx.remove(i)
        vis = vis[:, good_idx, ...]
        blen = blen[good_idx, ...]
        bname = bname[good_idx, ...]

        for badant in badants:
            antenna_order.remove(badant)

    assert vis.shape[0] == len(mjd)
    vis = vis.swapaxes(0, 1)
    dt = np.median(np.diff(mjd))
    if len(mjd) > 0:
        tstart = mjd[0]-dt/2
        tstop = mjd[-1]+dt/2
    else:
        tstart = None
        tstop = None

    bname = bname.tolist()
    return fobs, blen, bname, tstart, tstop, vis, mjd, transit_idx, \
        antenna_order, tsamp

def initialize_hdf5_file(
    fhdf, fobs, antenna_order, t0, nbls, nchan, npol, nant
):
    """Initializes the hdf5 file for the fringestopped visibilities.

    Parameters
    ----------
    fhdf : hdf5 file handler
        The file to initialize.
    fobs : array
        The center frequency of each channel in GHz.
    antenna_order : array
        The order of the antennas in the correlator.
    t0 : float
        The time of the first subintegration in MJ seconds.
    nbls : int
        The number of baselines in the visibilities.
    nchan : int
        The number of channels in the visibilities.
    npol : int
        The number of polarizations in the visibilities.
    nant : int
        The number of antennas in the visibilities.

    Returns
    -------
    vis_ds : hdf5 dataset
        The dataset for the visibilities.
    t_ds : hdf5 dataset
        The dataset for the times.
    """
    _ds_fobs = fhdf.create_dataset("fobs_GHz", (nchan, ), dtype=np.float32,
                                   data=fobs)
    _ds_ants = fhdf.create_dataset("antenna_order", (nant, ), dtype=np.int,
                                   data=antenna_order)
    _t_st = fhdf.create_dataset("tstart_mjd_seconds", (1, ), maxshape=(1, ),
                                dtype=int, data=t0)
    vis_ds = fhdf.create_dataset("vis", (0, nbls, nchan, npol),
                                 maxshape=(None, nbls, nchan, npol),
                                 dtype=np.complex64, chunks=True, data=None)
    t_ds = fhdf.create_dataset("time_seconds", (0, ), maxshape=(None, ),
                               dtype=np.float32, chunks=True, data=None)
    return vis_ds, t_ds

def extract_applied_delays(file, antennas):
    """Extracts the current snap delays from the hdf5 file.

    If delays are not set in the hdf5 file, uses the most recent delays in
    the beamformer weights directory instead.

    Parameters
    ----------
    file : str
        The full path to the hdf5 file.
    antennas : list
        Order of antennas in the hdf5 file.

    Returns
    -------
    ndarray
        The applied delays in ns.
    """
    with h5py.File(file, 'r') as f:
        delaystring = (
            f['Header']['extra_keywords']['applied_delays_ns']
            [()]
        ).astype(np.str)
        applied_delays = np.array(
            delaystring.split(' ')
        ).astype(np.int).reshape(-1, 2)
        applied_delays = applied_delays[np.array(antennas)-1, :]
    return applied_delays
