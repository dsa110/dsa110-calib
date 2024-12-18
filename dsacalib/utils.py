"""
DSACALIB/UTILS.PY

Dana Simard, dana.simard@astro.caltech.edu, 10/2019

Modified for python3 from DSA-10 routines written by Vikram Ravi, Harish
Vendantham.

Routines to interact w/ fits visibilities recorded by DSA-10, hdf5 visibilities
recorded by DSA-110, and visibility in CASA measurement sets.
"""

import traceback
from collections import namedtuple

from scipy.ndimage.filters import median_filter
import astropy.units as u
import casatools as cc
import numpy as np
from antpos.utils import get_itrf
from astropy.coordinates import Angle

from dsacalib import constants as ct
from dsacalib.preprocess import read_nvss_catalog

NVSS = None

CalibratorSource = namedtuple(
    "CalibratorSource", "name ra dec flux epoch direction pa maj_axis min_axis")


def generate_calibrator_source(
        name, ra, dec, flux=1.0, epoch="J2000", pa=None, maj_axis=None, min_axis=None
):
    """Return a named tuple containing calibrator parameters.

    Parameters
    ----------
    name : str
        Identifier for the source.
    ra : str
        The right ascension of the source. e.g. "12h00m19.21s".Astropy
        quantity also accepted.
    dec : str
        The declination of the source. e.g. "+73d00m45.7s". Astropy
        quantity also accepted.
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

    assert epoch == "J2000"

    if isinstance(ra, str):
        ra = to_deg(ra)
    if isinstance(dec, str):
        dec = to_deg(dec)
    if ra:
        direction = Direction("J2000", ra.to_value(u.rad), dec.to_value(u.rad))
    else:
        direction = None

    if maj_axis:
        maj_axis = maj_axis * u.arcsecond
    if min_axis:
        min_axis = min_axis * u.arcsecond

    return CalibratorSource(
        name, ra, dec, flux, epoch, direction, pa, maj_axis, min_axis)


def update_nvss():
    global NVSS
    NVSS = read_nvss_catalog()


def calibrator_source_from_name(name):
    if NVSS is None:
        update_nvss()
    record = NVSS.loc[f"NVSS {name}"]
    return generate_calibrator_source(
        name, record["ra"]*u.deg, record["dec"]*u.deg, record["flux_20_cm"]*1e-3)


class Direction:
    """Class for holding sky coordinates and converting between ICRS and FK5.

    Parameters
    ----------
    epoch : str
        'J2000' (for ICRS or J2000 coordinates) or 'HADEC' (for FK5 coordinates
        at an equinox of obstime)
    lon : float
        The longitude (right ascension or hour angle) in radians
    lat : float
        The latitude (declination) in radians
    obstime : float
        The observation time in mjd.
    observatory : str
        The name of the observatory
    """

    def __init__(self, epoch, lon, lat, obstime=None, observatory="OVRO_MMA"):

        assert epoch in ["J2000", "HADEC"]
        if epoch == "HADEC":
            assert obstime is not None
        self.epoch = epoch
        self.lon = lon
        self.lat = lat
        self.obstime = obstime
        self.observatory = observatory

    def J2000(self, obstime=None, observatory=None):
        """Provides direction in J2000 coordinates.

        Parameters
        ----------
        obstime : float
            Time of observation in mjd.
        location : str
            Name of the observatory.

        Returns
        -------
        tuple
            ra, dec at J2000 in units of radians.
        """
        if self.epoch == "J2000":
            return self.lon, self.lat

        assert self.epoch == "HADEC"
        if obstime is None:
            assert self.obstime is not None
            obstime = self.obstime
        if observatory is None:
            assert self.observatory is not None
            observatory = self.observatory

        me = cc.measures()
        epoch = me.epoch("UTC", f"{obstime}d")
        location = me.observatory(observatory)
        source = me.direction("HADEC", f"{self.lon}rad", f"{self.lat}rad")
        me.doframe(epoch)
        me.doframe(location)
        output = me.measure(source, "J2000")
        assert output["m0"]["unit"] == "rad"
        assert output["m1"]["unit"] == "rad"
        return output["m0"]["value"], output["m1"]["value"]

    def hadec(self, obstime=None, observatory=None):
        """Provides direction in HADEC (FK5) at `obstime`.

        Parameters
        ----------
        obstime : float
            Time of observation in mjd.
        location : str
            Name of the observatory.

        Returns
        -------
        tuple
            ha, dec at obstime in units of radians.
        """
        if self.epoch == "HADEC":
            assert obstime is None
            return self.lon, self.lat

        assert self.epoch == "J2000"
        if obstime is None:
            assert self.obstime is not None
            obstime = self.obstime
        if observatory is None:
            assert self.observatory is not None
            observatory = self.observatory
        me = cc.measures()
        epoch = me.epoch("UTC", f"{obstime}d")
        location = me.observatory(observatory)
        source = me.direction("J2000", f"{self.lon}rad", f"{self.lat}rad")
        me.doframe(epoch)
        me.doframe(location)
        output = me.measure(source, "HADEC")
        assert output["m0"]["unit"] == "rad"
        assert output["m1"]["unit"] == "rad"
        return output["m0"]["value"], output["m1"]["value"]


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
    return Angle(string).to(u.deg)


def exception_logger(logger, task, exception, throw):
    """Logs exception traceback to syslog using the dsa_syslog module.

    Parameters
    ----------
    logger : dsa_syslog.DsaSyslogger() instance
        The logger used for within the reduction pipeline.
    task : str
        A short description of where in the pipeline the error occured.
    exception : Exception
        The exception that occured.
    throw : boolean
        If set to True, the exception is raised after the traceback is written
        to syslogs.
    """
    error_string = (
        f"During {task}, {type(exception).__name__} occurred:\n"
        f"{''.join(traceback.format_tb(exception.__traceback__))}")

    if logger:
        logger.error(error_string)
    print(error_string)

    if throw:
        raise exception


def warning_logger(logger, message):
    """Print and log (if a logger is specified) a message."""
    if logger:
        logger.warning(message)
    print(message)


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
    for j in range(1, nant + 1):
        i += j
        auto_bls += [i]
    if casa:
        nbls = (nant * (nant + 1)) // 2
        auto_bls = [(nbls - 1) - aidx for aidx in auto_bls]
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
    if antpos[-4:] == ".txt":
        anum, xx, yy, zz = np.loadtxt(antpos).transpose()
        anum = anum.astype(int) + 1
        anum, xx, yy, zz = zip(*sorted(zip(anum, xx, yy, zz)))
    elif antpos[-4:] == ".csv":
        df = get_itrf(antpos)
        anum = np.array(df.index)
        xx = np.array(df[["dx_m"]])
        yy = np.array(df[["dy_m"]])
        zz = np.array(df[["dz_m"]])
    return anum, xx, yy, zz


def mask_bad_bins(vis, axis, thresh=6.0, medfilt=False, nmed=129):
    """Masks bad channels or time bins in visibility data.

    Parameters
    ----------
    vis : ndarray
        The visibility array, with dimensions (baselines, time, frequency,
        polarization)
    axis : int
        The axis to flag along. `axis` set to 1 will flag bad time bins. `axis`
        set to 2 will flag bad frequency bins.
    thresh : float
        The threshold above which to flag data. Anything that deviates from the
        median by more than `thresh` multiplied by the standard deviation is
        flagged.
    medfilt : Boolean
        Whether to median filter to remove an average trend. If ``True``, will
        median filter. If ``False``, will subtract the median for the
        baseline/pol pair.
    nmed : int
        The size of the median filter to use. Only used in medfilt is ``True``.
        Must be an odd integer.

    Returns
    -------
    good_bins : ndarray
          Has a value of 1 where the bin is good, and 0 where the bin should be
          flagged. If `axis` is 2, the dimensions are (baselines, 1, frequency,
          polarization). If `axis` is 1, the dimensions are (baselines, time, 1,
          polarization).
    fraction_flagged : ndarray
          The fraction of data flagged for each baseline/polarization pair.
          Dimensions (baselines, polarization).
    """
    # TODO: Update medfilt to use the correct axis
    assert not medfilt
    assert axis in (1, 2)
    avg_axis = 1 if axis == 2 else 2

    # Average over time (or frequency) first.
    vis_avg = np.abs(np.mean(vis, axis=avg_axis, keepdims=True))
    # Median filter over frequency (or time) and remove the median trend or
    # remove the median.
    if medfilt:
        vis_avg_mf = median_filter(vis_avg.real, size=(1, nmed, 1))
        vis_avg -= vis_avg_mf
    else:
        vis_avg -= np.median(vis_avg, axis=1, keepdims=True)
    # Calculate the standard deviation along the frequency (or time) axis.
    vis_std = np.std(vis_avg, axis=1, keepdims=True)
    # Get good channels.
    good_bins = np.abs(vis_avg) < thresh * vis_std
    fraction_flagged = (1 - good_bins.sum(axis=axis) / good_bins.shape[axis]).squeeze()
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
    vis = vis - np.median(vis, axis=1, keepdims=True)
    if mask is not None:
        vis = vis * mask.reshape(nbls, -1, npol)
    std = np.std(np.abs(vis), axis=1, keepdims=True)
    good_pixels = np.abs(vis) < thresh * std
    fraction_flagged = 1 - good_pixels.sum(1) / good_pixels.shape[1]
    good_pixels = good_pixels.reshape(nbls, nt, nf, npol)
    return good_pixels, fraction_flagged


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
    factor = -1 * (np.sin(lat) - np.tan(dec) * np.cos(lat))
    if daz is not None:
        assert dha is None, "daz and dha cannot both be defined."
        ans = daz * factor
    elif dha is not None:
        ans = dha / factor
    else:
        raise RuntimeError("One of daz or dha must be defined")
    return ans
