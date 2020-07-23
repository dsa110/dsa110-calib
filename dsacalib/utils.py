"""
DSA_UTILS.PY

Dana Simard, dana.simard@astro.caltech.edu, 10/2019

Modified for python3 from DSA-10 routines written by Vikram Ravi, Harish
Vendantham.

Routines to interact w/ fits visibilities recorded by DSA-10, hdf5 visibilities
recorded by DSA-110, and visibility in CASA measurement sets.
"""

# To do:
# Replace to_deg w/ astropy versions

# Always import scipy before importing casatools.
import traceback
from scipy.ndimage.filters import median_filter
import numpy as np
import h5py
import casatools as cc
from dsacalib import constants as ct
import dsacalib
from antpos.utils import get_itrf, get_baselines
from dsautils import dsa_store
from dsautils import calstatus as cs
# pylint will complain about this, but iers.conf.iers_auto_url_mirror must be
# set before astropy.time.Time is imported.
import astropy.io.fits as pf
import astropy.units as u
from astropy.utils import iers
iers.conf.iers_auto_url_mirror = ct.IERS_TABLE
from astropy.time import Time # pylint: disable=wrong-import-position

de = dsa_store.DsaStore()

def exception_logger(logger, task, exception, throw):
    """Logs exception traceback to syslog using the dsa_syslog module.

    Parameters
    ----------
    logger : dsa_syslog.DsaSyslogger() instance
        The logger used for within the reduction pipeline.
    task : str
        A short description of where in the pipeline the error occured.
    e : Exception
        The exception that occured.
    throw : boolean
        If set to True, the exception is raised after the traceback is written
        to syslogs.
    """
    logger.error(
        'During {0}, {1} occurred:\n{2}'.format(
            task, type(exception).__name__, ''.join(
                traceback.format_tb(exception.__traceback__))))
    if throw:
        raise exception

class src():
    """Simple class for holding source parameters.
    """

    def __init__(self, name, ra, dec, I=1., epoch='J2000', pa=None,
                 maj_axis=None, min_axis=None):
        """Initializes the src class.

        Parameters
        ----------
        name : str
            Identifier for the source.
        ra : str
            The right ascension of the source. e.g. "12h00m19.21s".
        dec : str
            The declination of the source. e.g. "+73d00m45.7s".
        I : float
            The flux of the source in Jy.  Defaults 1.
        epoch : str
            The epoch of `ra` and `dec`. Defaults "J2000".
        pa : float
            The position angle in degrees.  Defaults ``None``.
        maj_axis : float
            The major axis in arcseconds.  Defaults ``None``.
        min_axis : float
            The minor axis in arcseconds.  Defaults ``None``.
        """
        self.name = name
        self.I = I
        if isinstance(ra, str):
            self.ra = to_deg(ra)
        else:
            self.ra = ra
        if isinstance(dec, str):
            self.dec = to_deg(dec)
        else:
            self.dec = dec
        self.epoch = epoch
        self.pa = pa
        if maj_axis is None:
            self.maj_axis = None
        else:
            self.maj_axis = maj_axis*u.arcsecond
        if min_axis is None:
            self.min_axis = None
        else:
            self.min_axis = min_axis*u.arcsecond

def to_deg(string):
    """Converts a string representation of RA or DEC to degrees.

    Parameters
    ----------
    string : str
        RA or DEC in string format e.g. "12h00m19.21s" or "+73d00m45.7s".

    Returns
    -------
    deg : astropy quantity
        The angle in degrees.
    """
    if 'h' in string:
        h, m, s = string.strip('s').strip('s').replace('h', 'm').split('m')
        deg = (float(h)+(float(m)+float(s)/60)/60)*15*u.deg
    else:
        sign = -1 if '-' in string else 1
        d, m, s = string.strip('+-s').strip('s').replace('d', 'm').split('m')
        deg = (float(d)+(float(m)+float(s)/60)/60)*u.deg*sign
    return deg

def read_hdf5_file(fl, source=None, dur=50*u.min, autocorrs=True, badants=None,
                   quiet=True):
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
    if source is not None:
        lstmid = source.ra.to_value(u.rad)
        seg_len = (dur/2*(15*u.deg/u.h)).to_value(u.rad)

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

def read_psrfits_file(fl, source, dur=50*u.min, antenna_order=None,
                      antpos=None, utc_start=None, autocorrs=False,
                      badants=None, quiet=True, dsa10=True):
    r"""Reads in the psrfits header and data.

    Parameters
    ----------
    fl : str
        The full path to the psrfits file.
    source : src class instance
        The source to retrieve data for.
    dur : astropy quantity in minutes or equivalent
        The amount of time to extract around the source meridian passing.
        Defaults ``50*u.min``.
    antenna_order : list
        The order of the antennas in the correlator.  Only used if dsa10 is set
        to ``False``.  Defaults ``None``.
    antpos : str
        The full path of the text file containing the antenna ITRF positions.
        If ``None`` set to `dsacalib.__path__[0]`_/data_/antpos_ITRF.txt. Defaults
        ``None`.
    utc_start : astropy time object
        The start time of the observation in UTC.  Only used if dsa10 is set to
        ``False``. Defaults ``None``.
    autocorrs : boolean
        If set to ``True``, both auto and cross correlations will be returned.
        If set to False, only the cross correlations will be returned. Defaults
        ``False``.
    badants : list(int)
        The antennas for which you do not want data to be returned. If set to
        ``None``, all antennas are returned.  Defaults ``None``.
    quiet : boolean
        If set to ``True``, infromation on the file printed to stdout.
    dsa10 : boolean
        If set to ``True``, assumes fits file is in dsa10 format, otherwise
        assumes fits file is in the 3-antenna correlator output format.
        Defaults ``True``.

    Returns
    -------
    fobs : array
        The frequency of the channels in GHz.
    blen : array
        The itrf coordinates of the baselines, shape (nbaselines, 3).
    bname : list
        The station pairs for each baseline (in the same order as blen). Shape
        (nbaselines, 2).
    tstart : float
        The start time of the extracted data in MJD.
    tstop : float
        The stop time of the extracted data in MJD.
    vis : ndarray
        The requested visibilities, dimensions (baseline, time, frequency,
        polarization).
    mjd : array
        The midpoint MJD of each subintegration in the visibilities.
    transit_idx : int
        The index of the meridian passing in the time axis of the visibilities.
    """
    if antpos is None:
        antpos = '{0}/data/antpos_ITRF.txt'.format(dsacalib.__path__[0])

    fo = pf.open(fl, ignore_missing_end=True)
    f = fo[1]
    if dsa10:
        _nchan, fobs, _nt, blen, bname, tstart, tstop, tsamp, antenna_order = \
            get_header_info(f, verbose=True, antpos=antpos)
        vis, lst, mjd, transit_idx = extract_vis_from_psrfits(
            f, source.ra.to_value(u.rad),
            (dur/2*(15*u.deg/u.h)).to_value(u.rad), antenna_order, tstart,
            tstop, quiet)
    else:
        assert antenna_order is not None, 'Antenna order must be provided'
        assert utc_start is not None, 'Start time must be provided'
        _nchan, fobs, _nt, blen, bname, tstart_offset, tstop_offset, tsamp, \
            antenna_order = get_header_info(f, verbose=True, antpos=antpos,
                                            antenna_order=antenna_order,
                                            dsa10=False)
        tstart = (utc_start+tstart_offset*u.s).mjd
        tstop = (utc_start+tstop_offset*u.s).mjd
        vis, lst, mjd, transit_idx = extract_vis_from_psrfits(
            f, source.ra.to_value(u.rad),
            (dur/2*(15*u.deg/u.h)).to_value(u.rad), antenna_order, tstart,
            tstop, quiet)
    fo.close()

    # Now we have to extract the correct baselines
    nant = len(antenna_order)
    if not autocorrs:
        basels = list(range((nant*(nant+1))//2))
        auto_bls = get_autobl_indices(nant)
        if not dsa10:
            auto_bls = [(len(basels)-1)-auto_bl for auto_bl in auto_bls]
        for i in auto_bls:
            basels.remove(i)
        vis = vis[basels, ...]
        blen = blen[basels, ...]
        bname = [bname[i] for i in basels]

    # Reorder the visibilities to fit with CASA ms convention
    if dsa10:
        vis = vis[::-1, ...]
        bname = bname[::-1]
        blen = blen[::-1, ...]
        antenna_order = antenna_order[::-1]

    if badants is not None:
        blen = np.array(blen)
        good_idx = list(range(len(bname)))
        for i, bn in enumerate(bname):
            if (bn[0] in badants) or (bn[1] in badants):
                good_idx.remove(i)
        vis = vis[good_idx, ...]
        blen = blen[good_idx, ...]
        bname = [bname[i] for i in good_idx]

    if badants is not None:
        for badant in badants:
            antenna_order.remove(badant)

    dt = np.median(np.diff(mjd))
    if len(mjd) > 0:
        tstart = mjd[0]-dt/2
        tstop = mjd[-1]+dt/2
    else:
        tstart = None
        tstop = None

    if not isinstance(bname, list):
        bname = bname.tolist()
    return fobs, blen, bname, tstart, tstop, tsamp, vis, mjd, lst, \
        transit_idx, antenna_order

def get_header_info(f, antpos=None, verbose=False, antenna_order=None,
                    dsa10=True):
    """Extracts important header info from a visibility fits file.

    Parameters
    ----------
    f : pyfits table handle
        The visibility data from the correlator.
    antpos : str
        The path to the text file containing the antenna positions. If set to
        ``None``, will be set to `dsacalib.__path__[0]`/data/antpos_ITRF.txt.
        Defaults ``None``.
    verbose : boolean
        If ``True``, information on the fits file is printed to stdout.
    antenna_order : list
        The order of the antennas in the correlator.  Required if `dsa10` is
        set to ``False``. Defaults ``None``.
    dsa10 : Boolean
        Set to ``True`` if the fits file is in dsa10 correlator format,
        ``False`` if the file is in 6-input correlator format.

    Returns
    -------
    nchan : int
        The number of frequency channels.
    fobs : array
        The midpoint frequency of each channel in GHz.
    nt: int
        The number of time subintegrations.
    blen : ndarray
        The ITRF coordinates of the baselines, shape (nbaselines, 3).
    bname : list
        The station pairs for each baseline (in the same order as blen), shape
        (nbaselines, 2).
    tstart : float
        The start time. If `dsa10` is set to ``True``, `tstart` is the start
        time in MJD. If `dsa10` is set to ``False``, `tstart` is the start time
        in seconds past the utc start time of the correlator run.
    tstop : float
        The stop time. If `dsa10` is set to ``True``, `tstart` is the stop time
        in MJD. If `dsa10` is set to ``False``, `tstart` is the stop time in
        seconds past the utc start time of the correlator run.
    """
    if antpos is None:
        antpos = '{0}/data/antpos_ITRF.txt'.format(dsacalib.__path__[0])
    if dsa10:
        aname = f.header['ANTENNAS'].split('-')
        aname = [int(an) for an in aname]
    else:
        assert antenna_order is not None, 'Antenna order must be provided'
        aname = antenna_order
    nant = len(aname)

    nchan = f.header['NCHAN']
    if dsa10:
        fobs = ((f.header['FCH1']*1e6-(np.arange(nchan)+0.5)*2.*2.5e8/nchan)*
                u.Hz).to_value(u.GHz)
    else:
        fobs = ((f.header['FCH1']*1e6-(np.arange(nchan)+0.5)*2.5e8/8192)*
                u.Hz).to_value(u.GHz)
    nt = f.header['NAXIS2']
    tsamp = f.header['TSAMP']

    tel_pos = np.loadtxt(antpos)
    blen = []
    bname = []
#     for i in np.arange(len(aname)-1)+1:
#         for j in np.arange(i+1):
    if dsa10:
        # currently doesn't include autocorrelations
        for i in np.arange(10):
            for j in np.arange(i+1):
                a1 = int(aname[i])-1
                a2 = int(aname[j])-1
                bname.append([a1+1, a2+1])
                blen.append(tel_pos[a1, 1:]-tel_pos[a2, 1:])
    else:
        for j in range(nant):
            for i in range(j, nant):
                a1 = int(aname[i])-1
                a2 = int(aname[j])-1
                bname.append([a2+1, a1+1])
                blen.append(tel_pos[a2, 1:]-tel_pos[a1, 1:])
    blen = np.array(blen)

    if dsa10:
        tstart = f.header['MJD']+ct.TIME_OFFSET/ct.SECONDS_PER_DAY
        tstop = tstart+nt*tsamp/ct.SECONDS_PER_DAY
    else:
        tstart = tsamp*f.header['NBLOCKS']
        tstop = tstart+nt*tsamp

    if verbose:
        if dsa10:
            print('File covers {0:.2f} hours from MJD {1} to {2}'.format(
                ((tstop-tstart)*u.d).to(u.h), tstart, tstop))
        else:
            print('File covers {0:.2f} h from {1} s to {2} s'.format(
                ((tstop-tstart)*u.s).to(u.h), tstart, tstop))
    return nchan, fobs, nt, blen, bname, tstart, tstop, tsamp, aname

def extract_vis_from_psrfits(f, lstmid, seg_len, antenna_order, mjd0, mjd1,
                             quiet=True):
    """Extracts visibilities from a fits file.

    Based on clip.extract_segment from DSA-10 routines.

    Parameters
    ----------
    f : pyfits table handle
        The fits file containing the visibilities.
    lstmid : float
        The LST around which to extract visibilities, in radians.
    seg_len : float
        The duration (in LST) of visibilities to extract, in radians.
    antenna_order : list
        The order of the antennas in the correlator.
    mjd0 : float
        The start time of the file in MJD.
    mjd1 : float
        The stop time of the file in MJD.
    quiet : boolean
        If set to ``False``, information on the file will be printed. Defaults
        ``True``.

    Returns
    -------
    vis : ndarray
        The requested visibilities, dimensions (baselines, time, frequency).
    lst : array
        The lst of each integration in the visibilities, in radians.
    mjd : array
        The midpoint mjd of each integration in the visibilities.
    idxmid : int
        The index along the time axis corresponding to the `lstmid`. (i.e the
        index corresponding to the meridian transit of the source.)
    """
    vis = f.data['VIS']
    nt = f.header['NAXIS2']
    nchan = f.header['NCHAN']
    tsamp = f.header['TSAMP']
    nant = len(antenna_order)

    if (mjd1-mjd0) >= 1:
        print("Data covers > 1 sidereal day. Only the first segment will be "+
              "extracted")

    lst0 = Time(mjd0, format='mjd').sidereal_time(
        'apparent', longitude=ct.OVRO_LON*u.rad).to_value(u.rad)
    mjd = mjd0+(np.arange(nt)+0.5)*tsamp/ct.SECONDS_PER_DAY
    lst = np.angle(np.exp(1j*(lst0+2*np.pi/ct.SECONDS_PER_SIDEREAL_DAY*
                              np.arange(nt+0.5)*tsamp)))

    if not quiet:
        print("\n-------------EXTRACT DATA--------------------")
        print("Extracting data around {0}".format(lstmid*180/np.pi))
        print("{0} Time samples in data".format(nt))
        print("LST range: {0:.1f} --- ({1:.1f}-{2:.1f}) --- {3:.1f}deg".format(
            lst[0]*180./np.pi, (lstmid-seg_len)*180./np.pi,
            (lstmid+seg_len)*180./np.pi, lst[-1]*180./np.pi))

    idxl = np.argmax(np.absolute(np.exp(1j*lst)+np.exp(1j*lstmid)*
                                 np.exp(-1j*seg_len)))
    idxr = np.argmax(np.absolute(np.exp(1j*lst)+np.exp(1j*lstmid)*
                                 np.exp(1j*seg_len)))
    idx0 = np.argmax(np.absolute(np.exp(1j*lst)+np.exp(1j*(lstmid))))
    idxmid = idxl-idx0

    mjd = mjd[idxl:idxr]
    lst = lst[idxl:idxr]
    vis = vis.reshape((nt, (nant*(nant+1))//2, nchan, 2, 2)) \
        [idxl:idxr, :, :, :, :]

    if not quiet:
        print("Extract: {0} ----> {1} sample; transit at {2}".format(
            idxl, idxr, idx0))
        print("----------------------------------------------")

    # Fancy indexing can have downfalls and may change in future numpy versions
    # See issue here https://github.com/numpy/numpy/issues/9450
    # odata = dat[:,basels,:,:,0]+ 1j*dat[:,basels,:,:,1]
    vis = vis[..., 0]+1j*vis[..., 1]
    vis = vis.swapaxes(0, 1)

    return vis, lst, mjd, idxmid

def get_autobl_indices(nant, casa=False):
    """Returns a list of the indices containing the autocorrelations.

    Can return the index for either correlator-ordered visibilities (`casa` set
    to ``False``) or CASA-ordered visibilities (`casa` set to ``True``).

    Parameters
    ----------
    nant : int
        The number of antennas in the visibility set.
    casa : boolean
        Whether the visibilities follow CASA ordering standards (`casa` set to
        ``True``) or DSA-10/DSA-110 correlator ordering standards (`casa` set
        to ``False``). Defaults to ``False``, or correlator ordering standards.

    Returns
    -------
    auto_bls : list
        The list of indices in the visibilities corresponding to
        autocorrelations.
    """
    auto_bls = []
    i = -1
    for j in range(1, nant+1):
        i += j
        auto_bls += [i]
    if casa:
        nbls = (nant*(nant+1))//2
        auto_bls = [(nbls-1)-aidx for aidx in auto_bls]
        auto_bls = auto_bls[::-1]
    return auto_bls

def get_antpos_itrf(antpos):
    """Reads and orders antenna positions from a text or csv file.

    Parameters
    ----------
    antpos : str
        The path to the text or csv file containing the antenna positions.

    Returns
    -------
    anum : list(int)
        The antenna numbers, in numerical order.
    xx, yy, zz : list(float)
        The ITRF coordinates of the antennas, in meters.
    """
    if antpos[-4:] == '.txt':
        anum, xx, yy, zz = np.loadtxt(antpos).transpose()
        anum = anum.astype(int)+1
        anum, xx, yy, zz = zip(*sorted(zip(anum, xx, yy, zz)))
    elif antpos[-4:] == '.csv':
        df = get_itrf(antpos)
        anum = np.array(df.index)
        xx = np.array(df[['dx_m']])
        yy = np.array(df[['dy_m']])
        zz = np.array(df[['dz_m']])
    return anum, xx, yy, zz

def simulate_ms(ofile, tname, anum, xx, yy, zz, diam, mount, pos_obs, spwname,
                freq, deltafreq, freqresolution, nchannels, integrationtime,
                obstm, dt, source, stoptime, autocorr):
    """Simulates a measurement set with cross-correlations only.
    
    WARNING: Not simulating autocorrelations correctly regardless of inclusion
    of autocorr parameter.

    Parameters
    ----------
    ofile : str
        The full path to which the measurement set will be written.
    tname : str
        The telescope name.
    xx, yy, zz : arrays
        The X, Y and Z ITRF coordinates of the antennas, in meters.
    diam : float
        The dish diameter in meters.
    mount : str
        The mount type, e.g. 'alt-az'.
    pos_obs : CASA measure instance
        The location of the observatory.
    spwname : str
        The name of the spectral window, e.g. 'L-band'.
    freq : str
        The central frequency, as a CASA-recognized string, e.g. '1.4GHz'.
    deltafreq : str
        The size of each channel, as a CASA-recognized string, e.g. '1.24kHz'.
    freqresolution : str
        The frequency resolution, as a CASA-recognized string, e.g. '1.24kHz'.
    nchannels : int
        The number of frequency channels.
    integrationtime : str
        The subintegration time, i.e. the width of each time bin, e.g. '1.4s'.
    obstm : float
        The start time of the observation in MJD.
    dt : float
        The offset between the CASA start time and the true start time in days.
    source : dsacalib.utils.source instance
        The source observed (or the phase-center).
    stoptime : float
        The end time of the observation in MJD.
    autocorr : boolean
        Set to ``True`` if the visibilities include autocorrelations, ``False``
        if the only include crosscorrelations.
    """
    me = cc.measures()
    qa = cc.quanta()
    sm = cc.simulator()
    sm.open(ofile)
    sm.setconfig(telescopename=tname, x=xx, y=yy, z=zz, dishdiameter=diam,
                 mount=mount, antname=anum, coordsystem='global',
                 referencelocation=pos_obs)
    sm.setspwindow(spwname=spwname, freq=freq, deltafreq=deltafreq,
                   freqresolution=freqresolution, nchannels=nchannels,
                   stokes='XX YY')
    sm.settimes(integrationtime=integrationtime, usehourangle=False,
                referencetime=me.epoch('utc', qa.quantity(obstm-dt, 'd')))
    sm.setfield(
        sourcename=source.name, sourcedirection=me.direction(
            source.epoch, qa.quantity(source.ra.to_value(u.rad), 'rad'),
            qa.quantity(source.dec.to_value(u.rad), 'rad')))
    sm.setauto(autocorrwt=1.0 if autocorr else 0.0)
    sm.observe(source.name, spwname, starttime='0s', stoptime=stoptime)
    sm.close()

def convert_to_ms(source, vis, obstm, ofile, bname, antenna_order,
                  tsamp=ct.TSAMP*ct.NINT, nint=1, antpos=None, model=None,
                  dt=ct.CASA_TIME_OFFSET, dsa10=True):
    """ Writes visibilities to an ms.

    Uses the casa simulator tool to write the metadata to an ms, then uses the
    casa ms tool to replace the visibilities with the observed data.

    Parameters
    ----------
    source : source class instance
        The calibrator (or position) used for fringestopping.
    vis : ndarray
        The complex visibilities, dimensions (baseline, time, channel,
        polarization).
    obstm : float
        The start time of the observation in MJD.
    ofile : str
        The name for the created ms.  Writes to `ofile`.ms.
    bname : list
        The list of baselines names in the form [[ant1, ant2],...].
    antenna_order: list
        The list of the antennas, in CASA ordering.
    tsamp : float
        The sampling time of the input visibilities in seconds.  Defaults to
        the value `tsamp`*`nint` as defined in `dsacalib.constants`.
    nint : int
        The number of time bins to integrate by before saving to a measurement
        set.  Defaults 1.
    antpos : str
        The full path to the text file containing ITRF antenna positions or the
        csv file containing the station positions in longitude and latitude.
        Defaults `dsacalib.__path__[0]`/data/antpos_ITRF.txt.
    model : ndarray
        The visibility model to write to the measurement set (and against which
        gain calibration will be done). Must have the same shape as the
        visibilities `vis`. If given a value of ``None``, an array of ones will
        be used as the model.  Defaults ``None``.
    dt : float
        The offset between the CASA start time and the data start time in days.
        Defaults to the value of `casa_time_offset` in `dsacalib.constants`.
    dsa10 : boolean
        Set to ``True`` if the data are from the dsa10 correlator.  Defaults
        ``True``.
    """
    if antpos is None:
        antpos = '{0}/data/antpos_ITRF.txt'.format(dsacalib.__path__[0])
    vis = vis.astype(np.complex128)
    if model is not None:
        model = model.astype(np.complex128)

    nant = len(antenna_order)

    me = cc.measures()

    # Observatory parameters
    tname = 'OVRO_MMA'
    diam = 4.5 # m
    obs = 'OVRO_MMA'
    mount = 'alt-az'
    pos_obs = me.observatory(obs)

    # Backend
    if dsa10:
        spwname = 'L_BAND'
        freq = '1.4871533196875GHz'
        deltafreq = '-0.244140625MHz'
        freqresolution = deltafreq
    else:
        spwname = 'L_BAND'
        freq = '1.28GHz'
        deltafreq = '40.6901041666667kHz'
        freqresolution = deltafreq
    (_, _, nchannels, npol) = vis.shape

    # Rebin visibilities
    integrationtime = '{0}s'.format(tsamp*nint)
    if nint != 1:
        npad = nint-vis.shape[1]%nint
        if npad == nint:
            npad = 0
        vis = np.nanmean(np.pad(vis, ((0, 0), (0, npad), (0, 0), (0, 0)),
                                mode='constant',
                                constant_values=(np.nan, )).reshape(
                                    vis.shape[0], -1, nint, vis.shape[2],
                                    vis.shape[3]), axis=2)
        if model is not None:
            model = np.nanmean(np.pad(model,
                                      ((0, 0), (0, npad), (0, 0), (0, 0)),
                                      mode='constant',
                                      constant_values=(np.nan, )).reshape(
                                          model.shape[0], -1, nint,
                                          model.shape[2], model.shape[3]),
                               axis=2)
    stoptime = '{0}s'.format(vis.shape[1]*tsamp*nint)

    anum, xx, yy, zz = get_antpos_itrf(antpos)
    # Sort the antenna positions
    idx_order = sorted([int(a)-1 for a in antenna_order])
    anum = np.array(anum)[idx_order]
    xx = np.array(xx)
    yy = np.array(yy)
    zz = np.array(zz)
    xx = xx[idx_order]
    yy = yy[idx_order]
    zz = zz[idx_order]

    nints = np.zeros(nant, dtype=int)
    for i, an in enumerate(anum):
        nints[i] = np.sum(np.array(bname)[:, 0] == an)
    nints, anum, xx, yy, zz = zip(*sorted(zip(nints, anum, xx, yy, zz),
                                          reverse=True))

    # Check that the visibilities are ordered correctly by checking the order
    # of baselines in bname
    idx_order = []
    autocorr = [anum[0], anum[0]] in list(bname)

    for i in range(nant):
        for j in range(i if autocorr else i+1, nant):
            idx_order += [bname.index([anum[i], anum[j]])]
    assert idx_order == list(np.arange(len(bname), dtype=int)), \
        'Visibilities not ordered by baseline'
    anum = [str(a) for a in anum]

    simulate_ms('{0}.ms'.format(ofile), tname, anum, xx, yy, zz, diam, mount,
                pos_obs, spwname, freq, deltafreq, freqresolution, nchannels,
                integrationtime, obstm, dt, source, stoptime, autocorr)

    # Check that the time is correct
    ms = cc.ms()
    ms.open('{0}.ms'.format(ofile))
    tstart_ms = ms.summary()['BeginTime']
    ms.close()

    print('autocorr :', autocorr)

    if np.abs(tstart_ms-obstm) > 1e-10:
        dt = dt+(tstart_ms-obstm)
        print('Updating casa time offset to {0}s'.format(
            dt*ct.SECONDS_PER_DAY))
        print('Rerunning simulator')
        simulate_ms('{0}.ms'.format(ofile), tname, anum, xx, yy, zz, diam,
                    mount, pos_obs, spwname, freq, deltafreq, freqresolution,
                    nchannels, integrationtime, obstm, dt, source, stoptime,
                    autocorr)

    # Reopen the measurement set and write the observed visibilities
    ms = cc.ms()
    ms.open('{0}.ms'.format(ofile), nomodify=False)
    ms.selectinit(datadescid=0)

    rec = ms.getdata(["data"])
    # rec['data'] has shape [scan, channel, [time*baseline]]
    vis = vis.T.reshape((npol, nchannels, -1))
    rec['data'] = vis
    ms.putdata(rec)
    ms.close()

    ms = cc.ms()
    ms.open('{0}.ms'.format(ofile), nomodify=False)
    if model is None:
        model = np.ones(vis.shape, dtype=complex)
    else:
        model = model.T.reshape((npol, nchannels, -1))
    rec = ms.getdata(["model_data"])
    rec['model_data'] = model
    ms.putdata(rec)
    ms.close()

    # Check that the time is correct
    ms = cc.ms()
    ms.open('{0}.ms'.format(ofile))
    tstart_ms = ms.summary()['BeginTime']
    tstart_ms2 = ms.getdata('TIME')['time'][0]/ct.SECONDS_PER_DAY
    ms.close()

    assert np.abs(tstart_ms-(tstart_ms2-tsamp*nint/ct.SECONDS_PER_DAY/2)) \
        < 1e-10, 'Data start time does not agree with MS start time'

    assert np.abs(tstart_ms - obstm) < 1e-10, \
        'Measurement set start time does not agree with input tstart'
    print('Visibilities writing to ms {0}.ms'.format(ofile))

def extract_vis_from_ms(ms_name, nbls):
    """ Extracts visibilities from a CASA measurement set.

    Parameters
    ----------
    ms_name : str
        The name of the measurement set.  Will open `ms_name`.ms.
    nbls : int
        The number of baselines in the measurement set.

    Returns
    -------
    vis_uncal : ndarray
        The 'observed' visibilities from the measurement set.  Dimensions
        (baseline, time, freq, polarization).
    vis_cal : ndarray
        The 'corrected' visibilities from the measurement set.  Dimensions
        (baseline, time, freq, polarization).
    time : array
        The time of each subintegration, in MJD.
    freq : array
        The frequency of each channel, in GHz.
    flags : ndarray
        The flags for the visibility data, same dimensions as `vis_uncal` and
        `vis_cal`. `flags` is 1 (or ``True``) where the data is flagged
        (invalid) and 0 (or ``False``) otherwise.
    """
    error = 0
    ms = cc.ms()
    error += not ms.open('{0}.ms'.format(ms_name))
    axis_info = ms.getdata(['axis_info'])['axis_info']
    freq = axis_info['freq_axis']['chan_freq'].squeeze()/1e9
    time = ms.getdata(['time'])['time'].reshape(-1, nbls)[..., 0]
    time = time/ct.SECONDS_PER_DAY

    vis_uncal = ms.getdata(["data"])['data']
    vis_uncal = vis_uncal.reshape(vis_uncal.shape[0], vis_uncal.shape[1], -1,
                                  nbls).T
    vis_cal = ms.getdata(["corrected_data"])['corrected_data']
    vis_cal = vis_cal.reshape(vis_cal.shape[0], vis_cal.shape[1], -1, nbls).T

    flags = ms.getdata(["flag"])['flag']
    flags = flags.reshape(flags.shape[0], flags.shape[1], -1, nbls).T
    error += not ms.close()
    if error > 0:
        print('{0} errors occured during calibration'.format(error))
    return vis_uncal, vis_cal, time, freq, flags

def initialize_hdf5_file(fhdf, fobs, antenna_order, t0, nbls, nchan, npol,
                         nant):
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

def mask_bad_bins(vis, axis, thresh=6.0, medfilt=False, nmed=129):
    """Masks bad channels or time bins in visibility data.

    Parameters
    ----------
      vis: array(complex)
        The visibility array, with dimensions (baselines, time, frequency,
        polarization)
      axis: int
        The axis to flag along. `axis`=1 will flag bad time bins. `axis`=2 will
        flag bad frequency bins.
      thresh: float
        The threshold above which to flag data. Anything that deviates from the
        median by more than `thresh` multiplied by the standard deviation is
        flagged.
      medfilt: Boolean
        Whether to median filter to remove an average trend. If ``True``, will
        median filter. If ``False``, will subtract the median for the
        baseline/pol pair.
      nmed: int
        The size of the median filter to use. Only used in medfilt is ``True``.
        Must be an odd integer.

    Returns
    -------
      good_bins : array(boolean)
          Has a value of 1 where the bin is good, and 0 where the bin should be
          flagged. If `axis`==2, the dimensions are (baselines, 1, frequency,
          polarization). If `axis`==1, the dimensions are (baselines, time, 1,
          polarization).
      fraction_flagged: array(float)
          The fraction of data flagged for each baseline/polarization pair.
          Dimensions (baselines, polarization).
    """
    assert axis in (1, 2)
    avg_axis = 1 if axis == 2 else 2

    # Average over time (or frequency) first.
    vis_avg = np.abs(np.mean(vis, axis=avg_axis))
    # Median filter over frequency (or time) and remove the median trend or
    # remove the median.
    if medfilt:
        vis_avg_mf = median_filter(vis_avg.real, size=(1, nmed, 1))
        vis_avg -= vis_avg_mf
    else:
        vis_avg -= np.median(vis_avg, axis=1)
    # Calculate the standard deviation along the frequency (or time) axis.
    vis_std = np.std(vis_avg, axis=1, keepdims=True)
    # Get good channels.
    good_bins = np.abs(vis_avg) < thresh*vis_std
    fraction_flagged = 1-good_bins.sum(axis=1)/good_bins.shape[1]
    if avg_axis == 1:
        good_bins = good_bins[:, np.newaxis, :, :]
    else:
        good_bins = good_bins[:, :, np.newaxis, :]
    return good_bins, fraction_flagged

def mask_bad_pixels(vis, thresh=6.0, mask=None):
    r"""Masks pixels with values above a SNR threshold within each visibility.

    Parameters
    ----------
    vis : ndarray
        The complex visibilities. Dimensions (baseline, time, frequency,
        polarization).
    thresh : float
        The threshold above which to flag data. Data above `thresh`\*the
        standard deviation in each channel of each visiblity is flagged.
        Defaults 6.
    mask : ndarray
        A mask for data that is already flagged. Should be 0 where data has
        been flagged, 1 otherwise. Same dimensions as `vis`.  Data previously
        flagged is not used in the calculation of the channel standard
        deviations.

    Returns
    -------
    good_pixels : ndarray
        Whether a given pixel in `vis` is good (1 or ``True``) or bad (i.e.
        above the threshold: 0 or ``False``). Same dimensions as ``vis``.
    fraction_flagged : array
        The ratio of the flagged data to the total number of pixels for each
        baseline/polarization.
    """
    (nbls, nt, nf, npol) = vis.shape
    vis = np.abs(vis.reshape(nbls, -1, npol))
    vis = vis-np.median(vis, axis=1, keepdims=True)
    if mask is not None:
        vis = vis*mask.reshape(nbls, -1, npol)
    std = np.std(vis, axis=1, keepdims=True)
    good_pixels = np.abs(vis) < thresh*std
    fraction_flagged = 1 - good_pixels.sum(1)/good_pixels.shape[1]
    good_pixels = good_pixels.reshape(nbls, nt, nf, npol)
    return good_pixels, fraction_flagged

def read_caltable(tablename, nbls, cparam=False):
    """Extracts calibration solution from a CASA calibration table.

    Parameters
    ----------
    tablename : str
        The full path to the CASA calibration table.
    nbls : int
        The number of baselines or antennas in the CASA calibration solutions.
        This can be calculated from the number of antennas, nant:

            For delay calibration, nbls=nant
            For antenna-based gain/bandpass calibration, nbls=nant
            For baseline-based gain/bandpass calibration,
                nbls=(nant*(nant+1))//2
    cparam : boolean
        Whether the parameter of interest is complex (set to ``True``) or a
        float (set to ``False``).  For delay calibration, set to ``False``. For
        gain/bandpass calibration, set to ``True``.  Defaults ``False``.

    Returns
    -------
    time : array
        The times at which each solution is calculated in MJD. Dimensions
        (time, baselines). If no column 'TIME' is in the calibration table,
        ``None`` is returned.
    vals : ndarray
        The calibration solutions. Dimensions may be (polarization, time,
        baselines) or (polarization, 1, baselines) (for delay or gain
        calibration) or (polarization,frequency,baselines) (for bandpass cal).
    flags : ndarray
        Flags corresponding to the calibration solutions.  Same dimensions as
        `vals`. A value of 1 or ``True`` if the data is flagged (invalid), 0 or
        ``False`` otherwise.
    """
    param_type = 'CPARAM' if cparam else 'FPARAM'
    tb = cc.table()
    print('Opening table {0} as type {1}'.format(tablename, param_type))
    tb.open(tablename)
    if 'TIME' in tb.colnames():
        time = (tb.getcol('TIME').reshape(-1, nbls)*u.s).to_value(u.d)
    else:
        time = None
    vals = tb.getcol(param_type)
    vals = vals.reshape(vals.shape[0], -1, nbls)
    flags = tb.getcol('FLAG')
    flags = flags.reshape(flags.shape[0], -1, nbls)
    tb.close()

    return time, vals, flags

def caltable_to_etcd(msname, calname, antenna_order, caltime, status,
                     baseline_cal=False, pols=None):
    r"""Copies calibration values from delay and gain tables to etcd.

    The dictionary passed to etcd should look like: {"ant_num": <i>,
    "time": <d>, "pol", [<s>, <s>], "gainamp": [<d>, <d>],
    "gainphase": [<d>, <d>], "delay": [<i>, <i>], "calsource": <s>,
    "gaincaltime_offset": <d>, "delaycaltime_offset": <d>, 'sim': <b>,
    'status': <i>}

    Parameters
    ----------
    msname : str
        The measurement set name, will use solutions created from the
        measurement set `msname`.ms.
    calname : str
        The calibrator name.  Will open the calibration tables
        `msname`\_`calname`\_kcal and `msname`\_`calname`\_gcal_ant.
    antenna_order : list or array
        The antenna numbers, in CASA ordering.
    baseline_cal : boolean
        Set to ``True`` if the gain table was calculated using baseline-based
        calibration, ``False`` if the gain table was calculated using
        antenna-based calibration. Defaults ``False``.
    pols : list
        The names of the polarizations. If ``None``, will be set to
        ``['A', 'B']``. Defaults ``None``.
    """
    if pols is None:
        pols = ['A', 'B']
    nant = len(antenna_order)
    # Number of baselines included in the gain and bandpass calibrations.
    if baseline_cal:
        nbls = (nant*(nant+1))//2
    else:
        nbls = nant

    try:
        # Complex gains for each antenna.
        tamp, amps, flags = read_caltable('{0}_{1}_gcal_ant'.format(
            msname, calname), nbls, cparam=True)
        mask = np.ones(flags.shape)
        mask[flags == 1] = np.nan
        amps = amps*mask
        if baseline_cal:
            autocorr_idx = get_autobl_indices(nant)
            tamp = tamp[..., autocorr_idx]
            amps = amps[..., autocorr_idx]

        # Check the output shapes.
        assert tamp.shape[0] == amps.shape[1]
        assert tamp.shape[1] == amps.shape[2]
        assert tamp.shape[1] == len(antenna_order)
        assert amps.shape[0] == len(pols)

        # Reduce tamp to a single value, amps to a single value for each
        # ant/pol pair.
        assert np.all(np.equal.reduce(tamp) == np.ones(len(antenna_order)))
        tamp = np.median(tamp[:, 0])
        if amps.ndim == 3:
            amps = np.nanmedian(amps, axis=1)
        gaincaltime_offset = (tamp-caltime)*ct.SECONDS_PER_DAY
    except Exception:
        tamp = np.nan
        amps = np.ones((len(pols), nant))*np.nan
        gaincaltime_offset = 0.
        status = cs.update(status, ['gain_tbl_err', 'inv_gainamp_p1',
                                    'inv_gainamp_p2', 'inv_gainphase_p1',
                                    'inv_gainphase_p2', 'inv_gaincaltime'])

    # Delays for each antenna.
    try:
        tdel, delays, flags = read_caltable('{0}_{1}_kcal'.format(
            msname, calname), nant, cparam=False)
        mask = np.ones(flags.shape)
        mask[flags == 1] = np.nan
        delays = delays*mask
        assert tdel.shape[0] == delays.shape[1]
        assert tdel.shape[1] == delays.shape[2]
        assert tdel.shape[1] == len(antenna_order)
        assert delays.shape[0] == len(pols)
        # Reduce tdel to a single value, delays to a single value for each
        # ant/pol pair.
        assert np.all(np.equal.reduce(tdel) == np.ones(len(antenna_order)))
        tdel = np.median(tdel[:, 0])
        if delays.ndim == 3:
            delays = np.nanmedian(delays, axis=1)
        delaycaltime_offset = (tdel-caltime)*ct.SECONDS_PER_DAY
    except Exception:
        tdel = np.nan
        delays = np.ones((len(pols), nant))*np.nan
        delaycaltime_offset = 0.
        status = cs.update(status, ['delay_tbl-err', 'inv_delay_p1',
                                    'inv_delay_p2', 'inv_delaycaltime'])

    # Update antenna 24:
    for i, antnum in enumerate(antenna_order):
        # Deal with old/new numbering system for now.
        if antnum == 2:
            antnum = 24

        # Everything needs to be cast properly.
        gainamp = []
        gainphase = []
        ant_delay = []
        for amp in amps[:, i]:
            if not np.isnan(amp):
                gainamp += [np.abs(amp)]
                gainphase += [np.angle(amp)]
            else:
                gainamp += [None]
                gainphase += [None]
        for delay in delays[:, i]:
            if not np.isnan(delay):
                ant_delay += [int(np.rint(delay))]
            else:
                ant_delay += [None]

        dd = {'ant_num': antnum,
              'time': caltime,
              'pol': pols,
              'gainamp': gainamp,
              'gainphase': gainphase,
              'delay': ant_delay,
              'calsource': calname,
              'gaincaltime_offset': gaincaltime_offset,
              'delaycaltime_offset': delaycaltime_offset,
              'sim': False,
              'status':status}
        required_keys = dict({'ant_num':['inv_antnum', 0],
                              'time':['inv_time', 0.],
                              'pol':['inv_pol', ['A', 'B']],
                              'calsource':['inv_calsource', 'Unknown'],
                              'sim':['inv_sim', False],
                              'status':['other_err', 0]})
        for key, value in required_keys.items():
            if dd[key] is None:
                print('caltable_to_etcd: key {0} must not be None to write to '
                      'etcd'.format(key))
                status = cs.update(status, value[0])
                dd[key] = value[1]

        for pol in dd['pol']:
            if pol is None:
                print('caltable_to_etcd: pol must not be None to write to '
                      'etcd')
                status = cs.update(status, 'inv_pol')
                dd['pol'] = ['A', 'B']

        de.put_dict('/mon/cal/{0}'.format(antnum), dd)

def daz_dha(dec, daz=None, dha=None, lat=ct.OVRO_LAT):
    """Converts an offset between azimuth and hour angle.

    Assumes that the offset in azimuth or hour angle from an azimuth of pi or
    hour angle of 0 is small.  One of `daz` or `dha` must be provided, the
    other is calculated.

    Parameters
    ----------
    dec : float
        The pointing declination of the antenna in radians.
    daz : float
        The azimuth offset in radians. ``None`` may also be passed, in which
        case the azimuth offset is calculated and returned. Defaults to
        ``None``.
    dha : float
        The hour angle offset in radians. ``None`` may also be passed, in which
        case the hour angle offset is calculated and returned. Defaults to
        ``None``.
    lat : float
        The latitude of the antenna in radians.  Defaults to the value of
        ``ovro_lat`` defined in ``dsacalib.constants``.

    Returns
    -------
    float
        The converted offset. If the value of `daz` passed was not ``None``,
        this is the hour angle offset corresponding to the azimuth offset
        `daz`. If the value of `dha` passed was not ``None``, this is the
        azimuth offset corresonding to the hour angle offset `dha`.

    Raises
    ------
    RuntimeError
        If neither `daz or `dha` is defined.
    """
    factor = -1*(np.sin(lat)-np.tan(dec)*np.cos(lat))
    if daz is not None:
        assert dha is None, "daz and dha cannot both be defined."
        ans = daz*factor
    elif dha is not None:
        ans = dha/factor
    else:
        raise RuntimeError('One of daz or dha must be defined')
    return ans
