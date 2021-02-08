"""
DSACALIB/FITS_IO.PY

Dana Simard, dana.simard@astro.caltech.edu, 10/2019

Modified for python3 from DSA-10 routines written by Vikram Ravi, Harish
Vendantham.

Routines to interact w/ fits visibilities recorded by DSA-10.
"""

# TODO: Replace to_deg w/ astropy versions

import warnings
import numpy as np
from dsacalib import constants as ct
from dsacalib.utils import get_autobl_indices
# pylint will complain about this, but iers.conf.iers_auto_url_mirror must be
# set before astropy.time.Time is imported.
import astropy.io.fits as pf
import astropy.units as u
from astropy.utils import iers
iers.conf.iers_auto_url_mirror = ct.IERS_TABLE
iers.conf.auto_max_age = None
from astropy.time import Time # pylint: disable=wrong-import-position

warnings.warn(
    "the fits_io module is deprecated and will be removed in v2.0.0",
    DeprecationWarning,
    stacklevel=2
)

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
        Defaults `dsacalib.constants.PKG_DATA_PATH`/antpos_ITRF.txt.
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
    tsamp : float
        The sampling time in seconds.
    vis : ndarray
        The requested visibilities, dimensions (baseline, time, frequency,
        polarization).
    mjd : array
        The midpoint MJD of each subintegration in the visibilities.
    lst : array
        The midpoint LST of each subintegration in the visibilities.
    transit_idx : int
        The index of the meridian passing in the time axis of the visibilities.
    antenna_order : list
        The antenna indices, in the order that they are in in the visibilities.
    """
    if antpos is None:
        antpos = '{0}/antpos_ITRF.txt'.format(ct.PKG_DATA_PATH)
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
        The path to the text file containing the antenna positions. Defaults
        `dsacalib.constants.PKG_DATA_PATH`.
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
    tsamp : float
        The sampling time in seconds.
    aname : list
        The antenna names, in the order they are in in the visibilities.
    """
    if antpos is None:
        antpos = '{0}/antpos_ITRF.txt'.format(ct.PKG_DATA_PATH)
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
