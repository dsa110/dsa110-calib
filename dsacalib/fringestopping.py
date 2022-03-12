"""Functions used in fringestopping of DSA-110 visibilities.

Author: Dana Simard, dana.simard@astro.caltech.edu 11/2019

These functions use casatools to build sky models, divide visibilities by sky
models, and fringestop visibilities.

"""

# always import scipy before importing casatools
from scipy.special import j1

import astropy.units as u
import casatools as cc
import numpy as np
from astropy.coordinates.angle_utilities import angular_separation
from numba import jit

from dsacalib import constants as ct


def calc_uvw(blen, tobs, src_epoch, src_lon, src_lat, obs="OVRO_MMA"):
    """Calculates uvw coordinates.

    Uses CASA to calculate the u,v,w coordinates of the baselines `b` towards a
    source or phase center (specified by `src_epoch`, `src_lon` and `src_lat`)
    at the specified time and observatory.

    Parameters
    ----------
    blen : ndarray
        The ITRF coordinates of the baselines.  Type float, shape (nbaselines,
        3), units of meters.
    tobs : ndarray
        An array of floats, the times in MJD for which to calculate the uvw
        coordinates.
    src_epoch : str
        The epoch of the source or phase-center, as a CASA-recognized string
        e.g. ``'J2000'`` or ``'HADEC'``
    src_lon : astropy quantity
        The longitude of the source or phase-center, in degrees or an
        equivalent unit.
    src_lat : astropy quantity
        The latitude of the source or phase-center, in degrees or an equivalent
        unit.
    obs : string
        The name of the observatory in CASA.

    Returns
    -------
    bu : ndarray
        The u-value for each time and baseline, in meters. Shape is
        ``(len(b), len(tobs))``.
    bv : ndarray
        The v-value for each time and baseline, in meters. Shape is
        ``(len(b), len(tobs))``.
    bw : ndarray
        The w-value for each time and baseline, in meters. Shape is
        ``(len(b), len(tobs))``.
    """
    tobs, blen = set_dimensions(tobs=tobs, blen=blen)
    nt = tobs.shape[0]
    nb = blen.shape[0]
    bu = np.zeros((nt, nb))
    bv = np.zeros((nt, nb))
    bw = np.zeros((nt, nb))

    # Define the reference frame
    me = cc.measures()
    qa = cc.quanta()
    if obs is not None:
        me.doframe(me.observatory(obs))

    if not isinstance(src_lon.ndim, float) and src_lon.ndim > 0:
        assert src_lon.ndim == 1
        assert src_lon.shape[0] == nt
        assert src_lat.shape[0] == nt
        direction_set = False
    else:
        if (src_epoch == "HADEC") and (nt > 1):
            raise TypeError("HA and DEC must be specified at each time in " + "tobs.")
        me.doframe(
            me.direction(
                src_epoch,
                qa.quantity(src_lon.to_value(u.deg), "deg"),
                qa.quantity(src_lat.to_value(u.deg), "deg"),
            )
        )
        direction_set = True

    contains_nans = False

    for i in range(nt):
        me.doframe(me.epoch("UTC", qa.quantity(tobs[i], "d")))
        if not direction_set:
            me.doframe(
                me.direction(
                    src_epoch,
                    qa.quantity(src_lon[i].to_value(u.deg), "deg"),
                    qa.quantity(src_lat[i].to_value(u.deg), "deg"),
                )
            )
        for j in range(nb):
            bl = me.baseline(
                "itrf",
                qa.quantity(blen[j, 0], "m"),
                qa.quantity(blen[j, 1], "m"),
                qa.quantity(blen[j, 2], "m"),
            )
            # Get the uvw coordinates
            try:
                uvw = me.touvw(bl)[1]["value"]
                bu[i, j], bv[i, j], bw[i, j] = uvw[0], uvw[1], uvw[2]
            except KeyError:
                contains_nans = True
                bu[i, j], bv[i, j], bw[i, j] = np.nan, np.nan, np.nan
    if contains_nans:
        print("Warning: some solutions not found for u, v, w coordinates")
    return bu.T, bv.T, bw.T

def calc_uvw_interpolate(
        blen: np.ndarray, tobs: "Time array", epoch: str, lon: "Quantity",
        lat: "Quantity") -> np.ndarray:
    """Calculate uvw coordinates with linear interpolation."""
    ntimebins = len(tobs)
    buvw_start = calc_uvw(blen, tobs.mjd[0], epoch, lon, lat)
    buvw_start = np.array(buvw_start).T

    buvw_end = calc_uvw(blen, tobs.mjd[-1], epoch, lon, lat)
    buvw_end = np.array(buvw_end).T

    buvw = buvw_start + ((buvw_end-buvw_start)/(ntimebins-1))*np.arange(ntimebins)[:, np.newaxis, np.newaxis]

    return buvw

@jit(nopython=True)
def visibility_sky_model_worker(vis_model, bws, famps, f0, spec_idx, fobs):
    """Builds complex sky models.

    This is a worker to contain the for loop in the visibility model
    calculation using jit. Modifies the input array `vis_model` in place.

    Parameters
    ----------
    vis_model : ndarray
        A placeholder for the output.  A complex array initialized to zeros,
        with the same shape as the array of visibilities you wish to model.
        Dimensions (baseline, time, freq, polarization).
    bws : ndarray
        The w component of the baselines towards the phase-center or towards
        each source in the sky model.  Dimensions (sources,baselines).
    famps : ndarray
        The flux of each source at the reference frequency, in Jy.
    f0 : float
        The reference frequency, in GHz.
    spec_idx : float
        The spectral index for the frequency dependence of the source flux.
    fobs : ndarray
        The central frequency of each channel of the visibilities, in GHz.
    """
    for i in range(bws.shape[0]):
        vis_model += (
            famps[i, ...]
            * ((fobs / f0) ** (spec_idx))
            * np.exp(2j * np.pi / ct.C_GHZ_M * fobs * bws[i, ...])
        )


def _py_visibility_sky_model_worker(vis_model, bws, famps, f0, spec_idx, fobs):
    """Builds complex sky models.

    A pure python version of `visibility_model_worker` for timing against.
    Modifies the input array `vis_model` in place.

    Parameters
    ----------
    vis_model : ndarray
        A placeholder for the output.  A complex array initialized to zeros,
        with the same shape as the array of visibilities you wish to model.
        Dimensions (baseline, time, freq, polarization).
    bws : ndarray
        The w component of the baselines towards the phase-center or towards
        each source in the sky model.  Dimensions (sources, baselines).
    famps : ndarray
        The flux of each source at the reference frequency, in Jy.
    f0 : float
        The reference frequency, in GHz.
    spec_idx : float
        The spectral index for the frequency dependence of the source flux.
    fobs : ndarray
        The central frequency of each channel of the visibilities, in GHz.
    """
    for i in range(bws.shape[0]):
        vis_model += (
            famps[i, ...]
            * ((fobs / f0) ** (spec_idx))
            * np.exp(2j * np.pi / ct.C_GHZ_M * fobs * bws[i, ...])
        )


def set_dimensions(fobs=None, tobs=None, blen=None):
    """Sets the dimensions of arrays for fringestopping.

    Ensures that `fobs`, `tobs` and `blen` are ndarrays with the correct
    dimensions for fringestopping using jit.  Any combination of these arrays
    may be passed.  If no arrays are passed, an empty list is returned

    Parameters
    ----------
    fobs : float or ndarray
        The central frequency of each channel in GHz.  Defaults ``None``.
    tobs : float or ndarray
        The central time of each subintegration in MJD.  Defaults ``None``.
    blen : ndarray
        The baselines in ITRF coordinates with dimensions (nbaselines, 3) or
        (3) if a single baseline.  Defaults ``None``.

    Returns
    -------
    list
        A list of the arrays passed, altered to contain the correct dimensions
        for fringestopping.  The list may contain:

        fobs : ndarray
            The central frequency of each channel in GHz with dimensions
            (channels, 1). Included if `fobs` is not set to ``None``.

        tobs : ndarray
            The central time of each subintegration in MJD with dimensions
            (time). Included if `tobs` is not set to ``None``.

        b : ndarray
            The baselines in ITRF coordaintes with dimensions (baselines, 3) or
            (1, 3) if a single baseline.  Included if ``b`` is not set to None.
    """
    to_return = []
    if fobs is not None:
        if not isinstance(fobs, np.ndarray):
            fobs = np.array(fobs)
        if fobs.ndim < 1:
            fobs = fobs[np.newaxis]
        fobs = fobs[:, np.newaxis]
        to_return += [fobs]

    if tobs is not None:
        if not isinstance(tobs, np.ndarray):
            tobs = np.array(tobs)
        if tobs.ndim < 1:
            tobs = tobs[np.newaxis]
        to_return += [tobs]

    if blen is not None:
        if isinstance(blen, list):
            blen = np.array(blen)
        if blen.ndim < 2:
            blen = blen[np.newaxis, :]
        to_return += [blen]
    return to_return


def visibility_sky_model(vis_shape, vis_dtype, blen, sources, tobs, fobs, lst, pt_dec):
    """Calculates the sky model visibilities.

    Calculates the sky model visibilities on the baselines `b` and at times
    `tobs`.  Ensures that the returned sky model is the same datatype and the
    same shape as specified by `vis_shape` and `vis_dtype` to ensure
    compatability with jit.

    Parameters
    ----------
    vis_shape : tuple
        The shape of the visibilities: (baselines, time, frequency,
        polarization).
    vis_dtype: numpy datatype
        The datatype of the visibilities.
    blen: real array
        The baselines to calculate visibilities for, shape (nbaselines, 3),
        units of meters in ITRF coords.
    sources : list(src class instances)
        The sources to include in the sky model.
    tobs : float or ndarray
        The times at which to calculate the visibility model, in MJD.
    fobs : float or ndarray
        The frequencies at which to calculate the visibility model, in GHz.
    lst : ndarray
        The local sidereal time in radians for each time in tobs.
    pt_dec : float
        The antenna pointing declination in radians.

    Returns
    -------
    vis_model : ndarray
        The modelled complex visibilities, dimensions (directions, baselines,
        time, frequency, polarization).
    """
    # Something seriously wrong here.
    raise NotImplementedError
    fobs, tobs, blen = set_dimensions(fobs, tobs, blen)
    bws = np.zeros((len(sources), len(blen), len(tobs), 1, 1))
    famps = np.zeros((len(sources), 1, len(tobs), len(fobs), 1))
    for i, src in enumerate(sources):
        _, _, bw = calc_uvw(blen, tobs, src.epoch, src.ra, src.dec)
        bws[i, :, :, 0, 0] = bw
        famps[i, 0, :, :, 0] = src.I * pb_resp(
            lst, pt_dec, src.ra.to_value(u.rad), src.dec.to_value(u.rad), fobs.squeeze()
        )
    # Calculate the sky model using jit
    vis_model = np.zeros(vis_shape, dtype=vis_dtype)
    visibility_sky_model_worker(vis_model, bws, famps, ct.F0, ct.SPEC_IDX, fobs)
    return vis_model


def fringestop(vis, blen, source, tobs, fobs, pt_dec, return_model=False):
    """Fringestops on a source.

    Fringestops on a source (or sky position) by dividing the input
    visibilities by a phase only model.  The input visibilities, vis, are
    modified in place.

    Parameters
    ----------
    vis : ndarray
        The visibilities t be fringestopped, with dimensions (baselines, time,
        freq, pol). `vis` is modified in place.
    blen : ndarray
        The ITRF coordinates of the baselines in meters, with dimensions (3,
        baselines).
    source : src class instance
        The source to fringestop on.
    tobs : ndarray
        The observation time (the center of each bin) in MJD.
    fobs : ndarray
        The observing frequency (the center of each bin) in GHz.
    pt_dec : float
        The pointing declination of the array, in radians.
    return_model : boolean
        If ``True``, the fringestopping model is returned.  Defaults ``False``.

    Returns
    -------
    vis_model : ndarray
        The phase-only visibility model by which the visiblities were divided.
        Returned only if `return_model` is set to ``True``.
    """
    fobs, tobs, blen = set_dimensions(fobs, tobs, blen)
    _, _, bw = calc_uvw(blen, tobs, source.epoch, source.ra, source.dec)
    # note that the time shouldn't matter below
    _, _, bwp = calc_uvw(
        blen, tobs[len(tobs) // 2], "HADEC", 0.0 * u.rad, pt_dec * u.rad
    )
    bw = bw - bwp
    vis_model = np.exp(2j * np.pi / ct.C_GHZ_M * fobs * bw[..., np.newaxis, np.newaxis])
    vis /= vis_model
    if return_model:
        return vis_model
    return


def divide_visibility_sky_model(
    vis, blen, sources, tobs, fobs, lst, pt_dec, return_model=False
):
    """Calculates and applies the sky model visibilities.

    Calculates the sky model visibilities on the baselines `blen` and at times
    `tobs`.  Divides the input visibilities, `vis`, by the sky model. `vis` is
    modified in-place.

    Parameters
    ----------
    vis : ndarray
        The observed complex visibilities, with dimensions (baselines, time,
        frequency, polarization).  `vis` will be updated in place to the
        fringe-stopped visibilities
    blen : ndarray
        The baselines for which to calculate visibilities, shape (nbaselines,
        3), units of meters in ITRF coords.
    sources : list(src class objects)
        The list of sources to include in the sky model.
    tobs : float or ndarray
        The times for which to calculate the visibility model in MJD.
    fobs: float or arr(float)
        The frequency for which to calculate the model in GHz.
    lst : fload or ndarray
        The lsts at which to calculate the visibility model.
    pt_dec : float
        The pointing declination in radians.
    return_model: boolean
        If set to ``True``, the visibility model will be returned. Defaults to
        ``False``.

    Returns
    -------
    vis_model: ndarray
        The modelled visibilities, dimensions (baselines, time, frequency,
        polarization). Returned only if `return_model` is set to True.
    """
    vis_model = visibility_sky_model(
        vis.shape, vis.dtype, blen, sources, tobs, fobs, lst, pt_dec
    )
    vis /= vis_model
    if return_model:
        return vis_model
    return


def complex_sky_model(
    source, ant_ra, pt_dec, fobs, tobs, blen, dish_dia=4.65, spind=0.7, pointing_ra=None
):
    """Computes the complex sky model, taking into account pointing errors.

    Use when the pointing error is large enough to introduce a phase error in
    the visibilities if not properly accounted for.

    Parameters
    ----------
    source : src class instance
        The source to model.  The source flux (`src.I`), right ascension
        (`src.ra`) and declination (`src.dec`) must be specified.
    ant_ra : ndarray
        The right ascension pointing of the antenna in each time bin of the
        observation, in radians.  If az=0 deg or az=180 deg, this is the lst of
        each time bin in the observation in radians.
    pt_dec : float
        The pointing declination of the observation in radians.
    fobs : array
        The observing frequency of the center of each channel in GHz.
    tobs : array
        The observing time of the center of each timebin in MJD.
    blen : ndarray
        The ITRF dimensions of each baseline, in m.
    dish_dia : float
        The dish diameter in m.  Defaults 4.65.
    spind : float
        The spectral index of the source.  Defaults 0.7.
    pt_ra : float
        The pointing (either physically or through fringestopping) right
        ascension in radians. If None, defaults to the pointing of the antenna,
        `ant_ra`. In other words, assumes fringestopping on the meridian.

    Returns
    -------
    ndarray
        The calculated complex sky model.
    """
    raise NotImplementedError
    if pointing_ra is None:
        pointing_ra = ant_ra
    model_amplitude = amplitude_sky_model(
        source, ant_ra, pt_dec, fobs, dish_dia=dish_dia, spind=spind
    )
    fobs, tobs, blen = set_dimensions(fobs, tobs, blen)
    _, _, bw = calc_uvw(blen, tobs, source.epoch, source.ra, source.dec)
    _, _, bwp = calc_uvw(blen, tobs, "RADEC", pointing_ra * u.rad, pt_dec * u.rad)
    bw = bw - bwp
    model_phase = np.exp(
        2j * np.pi / ct.C_GHZ_M * fobs * bw[..., np.newaxis, np.newaxis]
    )
    return model_amplitude * model_phase


def amplitude_sky_model(source, ant_ra, pt_dec, fobs, dish_dia=4.65, spind=0.7):
    """Computes the amplitude sky model due to the primary beam.

    Computes the amplitude sky model for a single source due to the primary
    beam response of an antenna.

    Parameters
    ----------
    source : src class instance
        The source to model.  The source flux (`src.I`), right ascension
        (`src.ra`) and declination (`src.dec`) must be specified.
    ant_ra : ndarray
        The right ascension pointing of the antenna in each time bin of the
        observation, in radians.  If az=0 deg or az=180 deg, this is the lst of
        each time bin in the observation in radians.
    pt_dec : float
        The pointing declination of the observation in radians.
    fobs : array
        The observing frequency of the center of each channel in GHz.
    dish_dia : float
        The dish diameter in m.  Defaults 4.65.
    spind : float
        The spectral index of the source.  Defaults 0.7.

    Returns
    -------
    ndarray
        The calculated amplitude sky model.
    """
    # Should add spectral index
    return (
        source.I
        * (fobs / 1.4) ** (-spind)
        * pb_resp(
            ant_ra,
            pt_dec,
            source.ra.to_value(u.rad),
            source.dec.to_value(u.rad),
            fobs,
            dish_dia,
        )
    )


def pb_resp_uniform_ill(ant_ra, ant_dec, src_ra, src_dec, freq, dish_dia=4.65):
    """Computes the primary beam response towards a direction on the sky.

    Assumes uniform illumination of the disk. Returns a value between 0 and 1
    for each value passed in `ant_ra`.

    Parameters
    ----------
    ant_ra : float or ndarray
        The antenna right ascension pointing in radians.  If an array, must be
        one-dimensional.
    ant_dec : float
        The antenna declination pointing in radians.
    src_ra : float
        The source right ascension in radians.
    src_dec : float
        The source declination in radians.
    freq : ndarray
        The frequency of each channel in GHz.
    dish_dia : float
        The dish diameter in meters.  Defaults to ``4.65``.

    Returns
    -------
    pb : ndarray
        The primary beam response, dimensions (`ant_ra`, `freq`).
    """
    dis = angular_separation(ant_ra, ant_dec, src_ra, src_dec)
    lam = 0.299792458 / freq
    pb = (
        2.0
        * j1(np.pi * dis[:, np.newaxis] * dish_dia / lam)
        / (np.pi * dis[:, np.newaxis] * dish_dia / lam)
    ) ** 2
    return pb


def pb_resp(ant_ra, ant_dec, src_ra, src_dec, freq, dish_dia=4.34):
    """Computes the primary beam response towards a direction on the sky.

    Assumes tapered illumination of the disk. Returns a value between 0 and 1
    for each value passed in `ant_ra`.

    Parameters
    ----------
    ant_ra : float or ndarray
        The antenna right ascension pointing in radians.  If an array, must be
        one-dimensional.
    ant_dec : float
        The antenna declination pointing in radians.
    src_ra : float
        The source right ascension in radians.
    src_dec : float
        The source declination in radians.
    freq : ndarray
        The frequency of each channel in GHz.
    dish_dia : float
        The dish diameter in meters.  Defaults to ``4.65``.

    Returns
    -------
    pb : ndarray
        The primary beam response, dimensions (`ant_ra`, `freq`) if an array is
        passed to `ant_ra`, or (`freq`) if a float is passed to `ant_ra`.
    """
    dis = np.array(angular_separation(ant_ra, ant_dec, src_ra, src_dec))
    if dis.ndim > 0 and dis.shape[0] > 1:
        dis = dis[:, np.newaxis]  # prepare for broadcasting

    lam = 0.299792458 / freq
    arg = 1.2 * dis * dish_dia / lam
    pb = (np.cos(np.pi * arg) / (1 - 4 * arg**2)) ** 2
    return pb
