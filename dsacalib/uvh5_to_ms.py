"""
Create a measurement set from a uvh5 file.
"""
import shutil
import os

import numpy as np
import scipy # pylint: disable=unused-import
import astropy.units as u
import astropy.constants as c
from astropy.time import Time
from casatasks import importuvfits
from casacore.tables import addImagingColumns, table
from pyuvdata import UVData

from antpos.utils import get_itrf
import dsautils.cnf as dsc
from dsamfs.fringestopping import calc_uvw_blt

from dsacalib.fringestopping import calc_uvw_interpolate
from dsacalib import constants as ct
from dsacalib.utils import Direction, generate_calibrator_source
from dsacalib.fringestopping import amplitude_sky_model


def get_refmjd() -> float:
    conf = dsc.Conf()
    return conf.get('fringe')['refmjd']


def uvh5_to_ms(
        fname, msname, ra=None, dec=None, dt=None, antenna_list=None,
        flux=None, fringestop=True, logger=None, refmjd=None
):
    """
    Converts a uvh5 data to a uvfits file.

    Parameters
    ----------
    fname : str
        The full path to the uvh5 data file.
    msname : str
        The name of the ms to write. Data will be written to `msname`.ms
    ra : astropy quantity
        The RA at which to phase the data. If None, will phase at the meridian
        of the center of the uvh5 file.
    dec : astropy quantity
        The DEC at which to phase the data. If None, will phase at the pointing
        declination.
    dt : astropy quantity
        Duration of data to extract. Default is to extract the entire file.
    antenna_list : list
        Antennas for which to extract visibilities from the uvh5 file. Default
        is to extract all visibilities in the uvh5 file.
    flux : float
        The flux of the calibrator in Jy. If included, will write a model of
        the primary beam response to the calibrator source to the model column
        of the ms. If not included, a model of a constant response over
        frequency and time will be written instead of the primary beam model.
    logger : dsautils.dsa_syslog.DsaSyslogger() instance
        Logger to write messages too. If None, messages are printed.
    refmjd : float
        The mjd used in the fringestopper.
    """

    if refmjd is None:
        refmjd = get_refmjd()

    uvdata, pt_dec, ra, dec = load_uvh5_file(fname, antenna_list, dt, ra, dec)

    antenna_positions = set_antenna_positions(uvdata, logger)

    phase_visibilities(uvdata, ra, dec, fringestop, refmjd=refmjd)

    fix_descending_missing_freqs(uvdata)

    write_UV_to_ms(uvdata, msname, antenna_positions)

    set_ms_model_column(msname, uvdata, pt_dec, ra, dec, flux)


def phase_visibilities(
        uvdata, phase_ra, phase_dec, fringestop=True, interpolate_uvws=False, refmjd=None
):
    """Phase a UVData instance.

    If fringestop is False, then no phasing is done,
    but the phase centre is set to the meridian at the midpoint of the observation,
    and the uvdata object is modified to indicate that it is phased.
    """
    blen = get_blen(uvdata)
    lamb = c.c/(uvdata.freq_array*u.Hz)
    time = Time(uvdata.time_array, format='jd')

    if refmjd is None:
        refmjd = np.mean(time.mjd)

    pt_dec = uvdata.extra_keywords['phase_center_dec']*u.rad
    uvw_m = calc_uvw_blt(
        blen, np.tile(refmjd, (uvdata.Nbls)), 'HADEC',
        np.zeros(uvdata.Nbls)*u.rad, np.tile(pt_dec, (uvdata.Nbls)))

    if fringestop:
        # Calculate uvw coordinates
        if interpolate_uvws:
            uvw = calc_uvw_interpolate(
                blen, time[::uvdata.Nbls], 'RADEC', phase_ra.to(u.rad), phase_dec.to(u.rad))
            uvw = uvw.reshape(-1, 3)
        else:
            blen = np.tile(blen[np.newaxis, :, :], (uvdata.Ntimes, 1, 1)).reshape(-1, 3)
            uvw = calc_uvw_blt(
                blen, time.mjd, 'RADEC', phase_ra.to(u.rad), phase_dec.to(u.rad)
            )

        # Fringestop and phase
        phase_model = generate_phase_model_antbased(
            uvw, uvw_m, uvdata.Nbls, uvdata.Ntimes, lamb, uvdata.ant_1_array[:uvdata.Nbls],
            uvdata.ant_2_array[:uvdata.Nbls])
        uvdata.data_array = uvdata.data_array/phase_model[..., np.newaxis]

    else:
        uvw = calc_uvw_blt(
            blen, np.tile(np.mean(time.mjd), (uvdata.Nbls)), 'J2000',
            np.tile(phase_ra, (uvdata.Nbls)), np.tile(phase_dec, (uvdata.Nbls)))
        phase_model = generate_phase_model_antbased(
            uvw, uvw_m, uvdata.Nbls, 1, lamb, uvdata.ant_1_array[:uvdata.Nbls],
            uvdata.ant_2_array[:uvdata.Nbls])
        uvdata.data_array = uvdata.data_array/phase_model[..., np.newaxis]
        uvw = np.tile(uvw.reshape((1, uvdata.Nbls, 3)),
                      (1, uvdata.Ntimes, 1)).reshape((uvdata.Nblts, 3))

    uvdata.uvw_array = uvw
    uvdata.phase_type = 'phased'
    uvdata.phase_center_dec = phase_dec.to_value(u.rad)
    uvdata.phase_center_ra = phase_ra.to_value(u.rad)
    uvdata.phase_center_epoch = 2000.
    uvdata.phase_center_frame = 'icrs'
    try:
        uvdata._set_app_coords_helper()
    except AttributeError:
        pass


def load_uvh5_file(
        fname: str,
        antenna_list: list=None,
        dt: "astropy.Quantity"=None,
        phase_ra: "Quantity"=None,
        phase_dec: "Quantity"=None,
        phase_time: "Time"=None
) -> "UVData":
    """Load specific antennas and times for a uvh5 file.

    phase_ra and phase_dec are set here, but the uvh5 file is not phased.
    """
    if (phase_ra is None and phase_dec is not None) or (phase_ra is not None and phase_dec is None):
        raise RuntimeError(
            "Only one of phase_ra and phase_dec defined.  Please define both or neither."
        )
    if phase_time is not None and phase_ra is not None:
        raise RuntimeError(
            "Please specific only one of phase_time and phasing direction (phase_ra + phase_dec)"
        )

    uvdata = UVData()

    # Read in the data
    if antenna_list is not None:
        uvdata.read(fname, file_type='uvh5', antenna_names=antenna_list,
                run_check_acceptability=False, strict_uvw_antpos_check=False)
    else:
        uvdata.read(fname, file_type='uvh5', run_check_acceptability=False,
                strict_uvw_antpos_check=False)

    pt_dec = uvdata.extra_keywords['phase_center_dec']*u.rad

    # Get pointing information
    if phase_ra is None:
        if phase_time is None:
            phase_time = Time(np.mean(uvdata.time_array), format='jd')
        pointing = Direction(
            'HADEC',
            0.,
            pt_dec.to_value(u.rad),
            phase_time.mjd)

        phase_ra = pointing.J2000()[0]*u.rad
        phase_dec = pointing.J2000()[1]*u.rad

    if dt is not None:
        extract_times(uvdata, phase_ra, dt)

    return uvdata, pt_dec, phase_ra, phase_dec


def extract_times(uvdata, ra, dt):
    """Extracts data from specified times from an already open UVData instance.

    This is an alternative to opening the file with the times specified using
    pyuvdata.UVData.open().

    Parameters
    ----------
    UV : pyuvdata.UVData() instance
        The UVData instance from which to extract data. Modified in-place.
    ra : float
        The ra of the source around which to extract data, in radians.
    dt : astropy quantity
        The amount of data to extract, units seconds or equivalent.
    """
    lst_min = (
        ra - (dt*2*np.pi*u.rad/(ct.SECONDS_PER_SIDEREAL_DAY*u.s))/2
    ).to_value(u.rad)%(2*np.pi)
    lst_max = (
        ra + (dt*2*np.pi*u.rad/(ct.SECONDS_PER_SIDEREAL_DAY*u.s))/2
    ).to_value(u.rad)%(2*np.pi)
    if lst_min < lst_max:
        idx_to_extract = np.where(
            (uvdata.lst_array >= lst_min) & (uvdata.lst_array <= lst_max)
        )[0]
    else:
        idx_to_extract = np.where(
            (uvdata.lst_array >= lst_min) | (uvdata.lst_array <= lst_max)
        )[0]
    if len(idx_to_extract) == 0:
        raise ValueError(
            "No times in uvh5 file match requested timespan "
            f"with duration {dt} centered at RA {ra}."
        )
    idxmin = min(idx_to_extract)
    idxmax = max(idx_to_extract)+1
    assert (idxmax-idxmin)%uvdata.Nbls == 0

    uvdata.uvw_array = uvdata.uvw_array[idxmin:idxmax, ...]
    uvdata.data_array = uvdata.data_array[idxmin:idxmax, ...]
    uvdata.time_array = uvdata.time_array[idxmin:idxmax, ...]
    uvdata.lst_array = uvdata.lst_array[idxmin:idxmax, ...]
    uvdata.nsample_array = uvdata.nsample_array[idxmin:idxmax, ...]
    uvdata.flag_array = uvdata.flag_array[idxmin:idxmax, ...]
    uvdata.ant_1_array = uvdata.ant_1_array[idxmin:idxmax, ...]
    uvdata.ant_2_array = uvdata.ant_2_array[idxmin:idxmax, ...]
    uvdata.baseline_array = uvdata.baseline_array[idxmin:idxmax, ...]
    uvdata.integration_time = uvdata.integration_time[idxmin:idxmax, ...]
    uvdata.Nblts = int(idxmax-idxmin)
    assert uvdata.data_array.shape[0] == uvdata.Nblts
    uvdata.Ntimes = uvdata.Nblts//uvdata.Nbls


def set_antenna_positions(uvdata: "UVData", logger: "DsaSyslogger"=None) -> "np.ndarray":
    """Set and return the antenna positions.

    This should already be done by the writer but for some reason they
    are being converted to ICRS, so we set them using antpos here.
    """
    df_itrf = get_itrf(
        latlon_center=(ct.OVRO_LAT*u.rad, ct.OVRO_LON*u.rad, ct.OVRO_ALT*u.m)
    )
    if len(df_itrf['x_m']) != uvdata.antenna_positions.shape[0]:
        message = 'Mismatch between antennas in current environment '+\
            f'({len(df_itrf["x_m"])}) and correlator environment '+\
            f'({uvdata.antenna_positions.shape[0]})'
        if logger is not None:
            logger.info(message)
        else:
            print(message)
    uvdata.antenna_positions[:len(df_itrf['x_m'])] = np.array([
        df_itrf['x_m'],
        df_itrf['y_m'],
        df_itrf['z_m']
    ]).T-uvdata.telescope_location
    antenna_positions = uvdata.antenna_positions + uvdata.telescope_location
    return antenna_positions


def get_blen(uvdata: "UVData") -> "np.ndarray":
    """Calculate baseline lenghts using antenna positions in the UVData file."""
    blen = np.zeros((uvdata.Nbls, 3))
    for i, ant1 in enumerate(uvdata.ant_1_array[:uvdata.Nbls]):
        ant2 = uvdata.ant_2_array[i]
        blen[i, ...] = uvdata.antenna_positions[ant2, :] - \
            uvdata.antenna_positions[ant1, :]
    return blen


def fix_descending_missing_freqs(uvdata: "UVData") -> None:
    """Flip descending freq arrays, and fills in missing channels."""

    # Look for missing channels
    freq = uvdata.freq_array.squeeze()
    # The channels may have been reordered by pyuvdata so check that the
    # parameter uvdata.channel_width makes sense now.
    ascending = np.median(np.diff(freq)) > 0
    if ascending:
        assert np.all(np.diff(freq) > 0)
    else:
        assert np.all(np.diff(freq) < 0)
        uvdata.freq_array = uvdata.freq_array[:, ::-1]
        uvdata.data_array = uvdata.data_array[:, :, ::-1, :]
        freq = uvdata.freq_array.squeeze()

    # TODO: Need to update this for missing on either side as well
    uvdata.channel_width = np.abs(uvdata.channel_width)
    # Are there missing channels?
    if not np.all(np.diff(freq)-uvdata.channel_width < 1e-5):
        # There are missing channels!
        nfreq = int(np.rint(np.abs(freq[-1]-freq[0])/uvdata.channel_width+1))
        freq_out = freq[0] + np.arange(nfreq)*uvdata.channel_width
        existing_idxs = np.rint((freq-freq[0])/uvdata.channel_width).astype(int)
        data_out = np.zeros((uvdata.Nblts, uvdata.Nspws, nfreq, uvdata.Npols),
                            dtype=uvdata.data_array.dtype)
        nsample_out = np.zeros((uvdata.Nblts, uvdata.Nspws, nfreq, uvdata.Npols),
                               dtype=uvdata.nsample_array.dtype)
        flag_out = np.zeros((uvdata.Nblts, uvdata.Nspws, nfreq, uvdata.Npols),
                            dtype=uvdata.flag_array.dtype)
        data_out[:, :, existing_idxs, :] = uvdata.data_array
        nsample_out[:, :, existing_idxs, :] = uvdata.nsample_array
        flag_out[:, :, existing_idxs, :] = uvdata.flag_array
        # Now write everything
        uvdata.Nfreqs = nfreq
        uvdata.freq_array = freq_out[np.newaxis, :]
        uvdata.data_array = data_out
        uvdata.nsample_array = nsample_out
        uvdata.flag_array = flag_out


def write_UV_to_ms(uvdata: "UVData", msname: "str", antenna_positions: "np.ndarray") -> None:
    """Write a UVData object to a ms.

    Uses a fits file as the intermediate between UVData and ms, which is removed after
    the measurement set is written.
    """
    if os.path.exists(f'{msname}.fits'):
        os.remove(f'{msname}.fits')

    uvdata.write_uvfits(
        f'{msname}.fits',
        spoof_nonessential=True,
        run_check_acceptability=False,
        strict_uvw_antpos_check=False
    )

    if os.path.exists(f'{msname}.ms'):
        shutil.rmtree(f'{msname}.ms')
    importuvfits(f'{msname}.fits', f'{msname}.ms')

    with table(f'{msname}.ms/ANTENNA', readonly=False) as tb:
        tb.putcol('POSITION', antenna_positions)

    addImagingColumns(f'{msname}.ms')

    os.remove(f'{msname}.fits')


def set_ms_model_column(msname: str, uvdata: "UVData", pt_dec: "Quantity", ra: "Quantity",
                        dec: "Quantity", flux_Jy: float) -> None:
    """Set the measurement model column."""
    if flux_Jy is not None:
        fobs = uvdata.freq_array.squeeze()/1e9
        lst = uvdata.lst_array
        source = generate_calibrator_source('cal', ra, dec, flux_Jy)
        model = amplitude_sky_model(source, lst, pt_dec, fobs)
        model = np.tile(model[:, :, np.newaxis], (1, 1, uvdata.Npols))
    else:
        model = np.ones((uvdata.Nblts, uvdata.Nfreqs, uvdata.Npols), dtype=np.complex64)

    with table(f'{msname}.ms', readonly=False) as tb:
        tb.putcol('MODEL_DATA', model)
        tb.putcol('CORRECTED_DATA', tb.getcol('DATA')[:])

def coordinates_differ(meridian, phase, tol=1e-7):
    """Determine if meridian and phase coordinates differ to within tol."""
    phase_ra, phase_dec = phase
    meridian_ra, meridian_dec = meridian
    if (phase_ra is None or
            np.abs(phase_ra.to_value(u.rad) - meridian_ra.to_value(u.rad)) < tol):
        if (phase_dec is None or
                np.abs(phase_dec.to_value(u.rad) - meridian_dec.to_value(u.rad)) < tol):
            return False
    return True


def generate_phase_model(uvw, uvw_m, nts, lamb):
    """Generates a phase model to apply using baseline-based delays.

    Parameters
    ----------
    uvw : np.ndarray
        The uvw coordinates at each time bin (baseline, 3)
    uvw_m : np.ndarray
        The uvw coordinates at the meridian, (time, baseline, 3)
    nts : int
        The number of unique times.
    lamb : astropy quantity
        The observing wavelength of each channel.
    ant1, ant2 : list
        The antenna indices in order.

    Returns:
    --------
    np.ndarray
        The phase model to apply.
    """
    dw = (uvw[:, -1] - np.tile(uvw_m[np.newaxis, :, -1], (nts, 1, 1)).reshape(-1))*u.m
    phase_model = np.exp((2j*np.pi/lamb*dw[:, np.newaxis, np.newaxis]
                         ).to_value(u.dimensionless_unscaled))
    return phase_model


def generate_phase_model_antbased(uvw, uvw_m, nbls, nts, lamb, ant1, ant2):
    """Generates a phase model to apply using antenna-based geometric delays.

    Parameters
    ----------
    uvw : np.ndarray
        The uvw coordinates at each time bin (baseline, 3)
    uvw_m : np.ndarray
        The uvw coordinates at the meridian, (time, baseline, 3)
    nbls, nts : int
        The number of unique baselines, times.
    pt_dec : astropy quantity
        The pointing declination of the array.
    ra, dec : astropy quantities
        The position to phase to in J2000 RA, DEC
    """
    uvw_m = calc_uvw_blt(
        blen,
        mjds[:nbls],
        'HADEC',
        np.zeros(nbls)*u.rad,
        np.ones(nbls)*pt_dec
    )
    blen = np.tile(
        blen[np.newaxis, :, :],
        (nts, 1, 1)
    ).reshape(-1, 3)
    uvw = calc_uvw_blt(
        blen,
        mjds,
        'RADEC',
        ra.to(u.rad),
        dec.to(u.rad)
    )
    dw = (
        uvw[:, -1] - np.tile(
            uvw_m[np.newaxis, :, -1],
            (nts, 1)
        ).reshape(-1)
    )*u.m
    phase_model = np.exp((
        2j*np.pi/lamb*dw[:, np.newaxis, np.newaxis]
    ).to_value(u.dimensionless_unscaled))
    return uvw, phase_model

def generate_phase_model_antbased(blen, mjds, nbls, nts, pt_dec, ra, dec, lamb, ant1, ant2):
    """Generates a phase model to apply.

    Parameters
    ----------
    blen : ndarray(float)
        The lengths of all baselines, shape (nbls, 3)
    mjds : array(float)
        The mjd of every sample, shape (nts*nbls)
    nbls, nts : int
        The number of unique baselines, times.
    pt_dec : astropy quantity
        The pointing declination of the array.
    ra, dec : astropy quantities
        The position to phase to in J2000 RA, DEC
    ant1, ant2 : list
        The antenna indices in order
    """
    uvw_m = calc_uvw_blt(
        blen,
        mjds[:nbls],
        'HADEC',
        np.zeros(nbls)*u.rad,
        np.ones(nbls)*pt_dec
    )
    # Need ant1 and ant2 to be passed here
    # Need to check that this gets the correct refidxs
    refant = 23 # ant1[0]
    refidxs = np.where(ant1 == refant)[0]

    antenna_order = list(ant2[refidxs])

    antenna_w_m = uvw_m[refidxs, -1]
    uvw_delays = uvw.reshape((nts, nbls, 3))
    antenna_w = uvw_delays[:, refidxs, -1]
    antenna_dw = antenna_w-antenna_w_m[np.newaxis, :]
    dw = np.zeros((nts, nbls))
    for i, a1 in enumerate(ant1):
        a2 = ant2[i]
        dw[:, i] = antenna_dw[:, antenna_order.index(a2)] - \
                   antenna_dw[:, antenna_order.index(a1)]
    dw = dw.reshape(-1)*u.m
    phase_model = np.exp((2j*np.pi/lamb*dw[:, np.newaxis, np.newaxis]
                         ).to_value(u.dimensionless_unscaled))
    return phase_model


def get_meridian_coords(pt_dec, time_mjd):
    """Get coordinates for the meridian in J2000."""
    pointing = Direction(
        'HADEC', 0., pt_dec.to_value(u.rad), time_mjd)
    meridian_ra, meridian_dec = pointing.J2000()
    meridian_ra = meridian_ra*u.rad
    meridian_dec = meridian_dec*u.rad
    return meridian_ra, meridian_dec
