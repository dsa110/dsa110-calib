"""
Create a measurement set from a uvh5 file.
"""
import shutil
import os
import numpy as np
import scipy # pylint: disable=unused-import
import astropy.units as u
import astropy.constants as c
from casatasks import importuvfits
from casacore.tables import addImagingColumns, table
from pyuvdata import UVData
from dsautils import dsa_store
import dsautils.cnf as dsc
from dsamfs.fringestopping import calc_uvw_blt
from dsacalib.fringestopping import calc_uvw_interpolate
from dsacalib import constants as ct
import dsacalib.utils as du
from dsacalib.fringestopping import amplitude_sky_model
from antpos.utils import get_itrf # pylint: disable=wrong-import-order
from astropy.utils import iers # pylint: disable=wrong-import-order
iers.conf.iers_auto_url_mirror = ct.IERS_TABLE
iers.conf.auto_max_age = None
from astropy.time import Time # pylint: disable=wrong-import-position wrong-import-order

de = dsa_store.DsaStore()

CONF = dsc.Conf()
CORR_PARAMS = CONF.get('corr')
REFMJD = CONF.get('fringe')['refmjd']

def uvh5_to_ms(fname, msname, ra=None, dec=None, dt=None, antenna_list=None,
               flux=None, fringestop=True, logger=None):
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

    UV, pt_dec, ra, dec = load_uvh5_file(fname, antenna_list, dt, ra, dec)

    antenna_positions = set_antenna_positions(UV, logger)

    phase_visibilities(UV, fringestop, ra, dec)

    fix_descending_missing_freqs(UV)

    write_UV_to_ms(UV, msname, antenna_positions)

    set_ms_model_column(msname, UV, pt_dec, ra, dec, flux)

def phase_visibilities(UV, fringestop=True, phase_ra=None, phase_dec=None, interpolate_uvws=False):
    """Phase a UVData instance.

    If fringestop is False, then no phasing is done,
    but the phase centre is set to the meridian at the midpoint of the observation,
    and the UV object is modified to indicate that it is phased.
    """
    blen = get_blen(UV)
    lamb = c.c/(UV.freq_array*u.Hz)
    time = Time(UV.time_array, format='jd')
    meantime = np.mean(time.mjd)
    pt_dec = UV.extra_keywords['phase_center_dec']*u.rad
    uvw_m = calc_uvw_blt(
        blen, np.tile(meantime, (UV.Nbls)), 'HADEC',
        np.zeros(UV.Nbls)*u.rad, np.tile(pt_dec, (UV.Nbls)))

    if phase_ra is None or phase_dec is None or not fringestop:
        meridian_ra, meridian_dec = get_meridian_coords(pt_dec, meantime)

    if fringestop:
        if phase_ra is None:
            phase_ra = meridian_ra
        if phase_dec is None:
            phase_dec = meridian_dec

        # Calculate uvw coordinates
        if interpolate_uvws:
            uvw = calc_uvw_interpolate(
                blen, time[::UV.Nbls], 'RADEC', phase_ra.to(u.rad), phase_dec.to(u.rad))
            uvw = uvw.reshape(-1, 3)
        else:
            blen = np.tile(blen[np.newaxis, :, :], (UV.Ntimes, 1, 1)).reshape(-1, 3)
            uvw = calc_uvw_blt(
                blen, time.mjd, 'RADEC', phase_ra.to(u.rad), phase_dec.to(u.rad))            

        # Fringestop and phase
        phase_model = generate_phase_model_antbased(
            uvw, uvw_m, UV.Nbls, UV.Ntimes, lamb, UV.ant_1_array[:UV.Nbls],
            UV.ant_2_array[:UV.Nbls])
        UV.data_array = UV.data_array/phase_model[..., np.newaxis]

    else:

        if not coordinates_differ(
                (meridian_ra, meridian_dec), (phase_ra, phase_dec), tol=1e-7):
            phase_ra = meridian_ra
            phase_dec = meridian_dec
            uvw = np.tile(uvw_m.reshape((1, UV.Nbls, 3)), (1, UV.Ntimes, 1)
                         ).reshape((UV.Nblts, 3))

        else: # Coordinates differ so we need to change the phase centre
            uvw = calc_uvw_blt(
                blen, np.tile(meantime, (UV.Nbls)), 'J2000',
                np.tile(phase_ra, (UV.Nbls)), np.tile(phase_dec, (UV.Nbls)))
            phase_model = generate_phase_model_antbased(
                uvw, uvw_m, UV.Nbls, 1, lamb, UV.ant_1_array[:UV.Nbls],
                UV.ant_2_array[:UV.Nbls])
            UV.data_array = UV.data_array/phase_model[..., np.newaxis]
            uvw = np.tile(uvw.reshape((1, UV.Nbls, 3)),
                          (1, UV.Ntimes, 1)).reshape((UV.Nblts, 3))

    UV.uvw_array = uvw
    UV.phase_type = 'phased'
    UV.phase_center_dec = phase_dec.to_value(u.rad)
    UV.phase_center_ra = phase_ra.to_value(u.rad)
    UV.phase_center_epoch = 2000.
    UV.phase_center_frame = 'icrs'
    UV._set_app_coords_helper()

def load_uvh5_file(fname: str, antenna_list: list=None, dt: "astropy.Quantity"=None,
                   phase_ra: "Quantity"=None, phase_dec: "Quantity"=None) -> "UVData":
    """Load specific antennas and times for a uvh5 file.

    phase_ra and phase_dec are set here, but the uvh5 file is not phased.
    """
    UV = UVData()

    # Read in the data
    if antenna_list is not None:
        UV.read(fname, file_type='uvh5', antenna_names=antenna_list,
                run_check_acceptability=False, strict_uvw_antpos_check=False)
    else:
        UV.read(fname, file_type='uvh5', run_check_acceptability=False,
                strict_uvw_antpos_check=False)

    # Get pointing information
    meantime = Time(np.mean(UV.time_array), format='jd')
    pt_dec = UV.extra_keywords['phase_center_dec']*u.rad
    pointing = du.direction(
        'HADEC',
        0.,
        pt_dec.to_value(u.rad),
        meantime.mjd
    )

    if phase_ra is None:
        phase_ra = pointing.J2000()[0]*u.rad
    if phase_dec is None:
        phase_dec = pointing.J2000()[1]*u.rad

    if dt is not None:
        extract_times(UV, phase_ra, dt)

    return UV, pt_dec, phase_ra, phase_dec

def extract_times(UV, ra, dt):
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
    lst_min = (ra - (dt*2*np.pi*u.rad/(ct.SECONDS_PER_SIDEREAL_DAY*u.s))/2
              ).to_value(u.rad)%(2*np.pi)
    lst_max = (ra + (dt*2*np.pi*u.rad/(ct.SECONDS_PER_SIDEREAL_DAY*u.s))/2
              ).to_value(u.rad)%(2*np.pi)
    if lst_min < lst_max:
        idx_to_extract = np.where((UV.lst_array >= lst_min) &
                                  (UV.lst_array <= lst_max))[0]
    else:
        idx_to_extract = np.where((UV.lst_array >= lst_min) |
                                  (UV.lst_array <= lst_max))[0]
    if len(idx_to_extract) == 0:
        raise ValueError("No times in uvh5 file match requested timespan "
                         f"with duration {dt} centered at RA {ra}.")
    idxmin = min(idx_to_extract)
    idxmax = max(idx_to_extract)+1
    assert (idxmax-idxmin)%UV.Nbls == 0
    UV.uvw_array = UV.uvw_array[idxmin:idxmax, ...]
    UV.data_array = UV.data_array[idxmin:idxmax, ...]
    UV.time_array = UV.time_array[idxmin:idxmax, ...]
    UV.lst_array = UV.lst_array[idxmin:idxmax, ...]
    UV.nsample_array = UV.nsample_array[idxmin:idxmax, ...]
    UV.flag_array = UV.flag_array[idxmin:idxmax, ...]
    UV.ant_1_array = UV.ant_1_array[idxmin:idxmax, ...]
    UV.ant_2_array = UV.ant_2_array[idxmin:idxmax, ...]
    UV.baseline_array = UV.baseline_array[idxmin:idxmax, ...]
    UV.integration_time = UV.integration_time[idxmin:idxmax, ...]
    UV.Nblts = int(idxmax-idxmin)
    assert UV.data_array.shape[0] == UV.Nblts
    UV.Ntimes = UV.Nblts//UV.Nbls

def set_antenna_positions(UV: "UVData", logger: "DsaSyslogger"=None) -> "np.ndarray":
    """Set and return the antenna positions.

    This should already be done by the writer but for some reason they
    are being converted to ICRS, so we set them using antpos here.
    """
    df_itrf = get_itrf(
        latlon_center=(ct.OVRO_LAT*u.rad, ct.OVRO_LON*u.rad, ct.OVRO_ALT*u.m)
    )
    if len(df_itrf['x_m']) != UV.antenna_positions.shape[0]:
        message = 'Mismatch between antennas in current environment '+\
            f'({len(df_itrf["x_m"])}) and correlator environment '+\
            f'({UV.antenna_positions.shape[0]})'
        if logger is not None:
            logger.info(message)
        else:
            print(message)
    UV.antenna_positions[:len(df_itrf['x_m'])] = np.array([
        df_itrf['x_m'],
        df_itrf['y_m'],
        df_itrf['z_m']
    ]).T-UV.telescope_location
    antenna_positions = UV.antenna_positions + UV.telescope_location
    return antenna_positions

def get_blen(UV: "UVData") -> "np.ndarray":
    """Calculate baseline lenghts using antenna positions in the UVData file."""
    blen = np.zeros((UV.Nbls, 3))
    for i, ant1 in enumerate(UV.ant_1_array[:UV.Nbls]):
        ant2 = UV.ant_2_array[i]
        blen[i, ...] = UV.antenna_positions[ant2, :] - \
            UV.antenna_positions[ant1, :]
    return blen

def fix_descending_missing_freqs(UV: "UVData") -> None:
    """Flip descending freq arrays, and fills in missing channels."""
    # Look for missing channels
    freq = UV.freq_array.squeeze()
    # The channels may have been reordered by pyuvdata so check that the
    # parameter UV.channel_width makes sense now.
    ascending = np.median(np.diff(freq)) > 0
    if ascending:
        assert np.all(np.diff(freq) > 0)
    else:
        assert np.all(np.diff(freq) < 0)
        UV.freq_array = UV.freq_array[:, ::-1]
        UV.data_array = UV.data_array[:, :, ::-1, :]
        freq = UV.freq_array.squeeze()

    # TODO: Need to update this for missing on either side as well
    UV.channel_width = np.abs(UV.channel_width)
    # Are there missing channels?
    if not np.all(np.diff(freq)-UV.channel_width < 1e-5):
        # There are missing channels!
        nfreq = int(np.rint(np.abs(freq[-1]-freq[0])/UV.channel_width+1))
        freq_out = freq[0] + np.arange(nfreq)*UV.channel_width
        existing_idxs = np.rint((freq-freq[0])/UV.channel_width).astype(int)
        data_out = np.zeros((UV.Nblts, UV.Nspws, nfreq, UV.Npols),
                            dtype=UV.data_array.dtype)
        nsample_out = np.zeros((UV.Nblts, UV.Nspws, nfreq, UV.Npols),
                               dtype=UV.nsample_array.dtype)
        flag_out = np.zeros((UV.Nblts, UV.Nspws, nfreq, UV.Npols),
                            dtype=UV.flag_array.dtype)
        data_out[:, :, existing_idxs, :] = UV.data_array
        nsample_out[:, :, existing_idxs, :] = UV.nsample_array
        flag_out[:, :, existing_idxs, :] = UV.flag_array
        # Now write everything
        UV.Nfreqs = nfreq
        UV.freq_array = freq_out[np.newaxis, :]
        UV.data_array = data_out
        UV.nsample_array = nsample_out
        UV.flag_array = flag_out

def write_UV_to_ms(UV: "UVData", msname: "str", antenna_positions: "np.ndarray") -> None:
    """Write a UVData object to a ms.

    Uses a fits file as the intermediate between UVData and ms, which is removed after
    the measurement set is written.
    """
    if os.path.exists('{0}.fits'.format(msname)):
        os.remove('{0}.fits'.format(msname))

    UV.write_uvfits('{0}.fits'.format(msname),
                    spoof_nonessential=True,
                    run_check_acceptability=False,
                    strict_uvw_antpos_check=False
                   )

    if os.path.exists('{0}.ms'.format(msname)):
        shutil.rmtree('{0}.ms'.format(msname))
    importuvfits('{0}.fits'.format(msname),
                 '{0}.ms'.format(msname))

    with table('{0}.ms/ANTENNA'.format(msname), readonly=False) as tb:
        tb.putcol('POSITION', antenna_positions)

    addImagingColumns('{0}.ms'.format(msname))

    os.remove('{0}.fits'.format(msname))

def set_ms_model_column(msname: str, UV: "UVData", pt_dec: "Quantity", ra: "Quantity",
                        dec: "Quantity", flux_Jy: float) -> None:
    """Set the measurement model column."""
    if flux_Jy is not None:
        fobs = UV.freq_array.squeeze()/1e9
        lst = UV.lst_array
        model = amplitude_sky_model(du.src('cal', ra, dec, flux_Jy),
                                    lst, pt_dec, fobs)
        model = np.tile(model[:, :, np.newaxis], (1, 1, UV.Npols))
    else:
        model = np.ones((UV.Nblts, UV.Nfreqs, UV.Npols), dtype=np.complex64)

    with table('{0}.ms'.format(msname), readonly=False) as tb:
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
    lamb : astropy quantity
        The observing wavelength of each channel.
    ant1, ant2 : list
        The antenna indices in order.

    Returns:
    --------
    np.ndarray
        The phase model to apply.
    """
    # Need ant1 and ant2 to be passed here
    # Need to check that this gets the correct refidxs
    refant = ant1[0]
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
    pointing = du.direction(
        'HADEC', 0., pt_dec.to_value(u.rad), time_mjd)
    meridian_ra, meridian_dec = pointing.J2000()
    meridian_ra = meridian_ra*u.rad
    meridian_dec = meridian_dec*u.rad
    return meridian_ra, meridian_dec
