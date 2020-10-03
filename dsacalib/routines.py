"""Calibration routine for DSA-110 calibration with CASA.

Author: Dana Simard, dana.simard@astro.caltech.edu, 2020/06
"""
import shutil
import os
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import astropy.units as u
import scipy # pylint: disable=unused-import
import casatools as cc
from casacore.tables import table
import dsautils.calstatus as cs
import dsautils.dsa_syslog as dsl
import dsacalib.utils as du
import dsacalib.ms_io as dmsio
import dsacalib.fits_io as dfio
import dsacalib.calib as dc
import dsacalib.plotting as dp
import dsacalib.fringestopping as df
import dsacalib.constants as ct
from astropy.utils import iers
iers.conf.iers_auto_url_mirror = ct.IERS_TABLE
iers.conf.auto_max_age = None
from astropy.time import Time # pylint: disable=wrong-import-position

logger = dsl.DsaSyslogger()
logger.subsystem("software")
logger.app("dsacalib")

def __init__():
    return

# Error handling global variables.
# status is the current status of the reduction.
# when exit is 1, the code will exit.
# current_error is the error to add to status at any time in the redcution.
# status = 0
# exit = 0
# current_error = 0

# def _calib_error_handler(exc_type, exc_value, exc_traceback):
#     global status, exit
#     if issubclass(exc_type, KeyboardInterrupt):
#         sys.__excepthook__(exc_type, exc_value, exc_traceback)
#         return
#     # Add the error to the status and set exit to 1.
#     status = status | error
#     exit = 1

#     # Print the error to syslog
#     message = traceback.format_exception(exc_type, exc_value,
#                                    exc_traceback)[1:]
#     message = ''.join(message)
#     message = "Uncaught exception occured.\n{0}".format(message)
#     if logger is not None:
#         logger.info(message)
#     else:
#         sys.stderr.write(message)

# sys.excepthook = _calib_error_handler

def _check_path(fname):
    """Raises an AssertionError if the path `fname` does not exist.
    """
    assert os.path.exists(fname), 'File {0} does not exist'.format(fname)

def triple_antenna_cal(obs_params, ant_params, throw_exceptions=True,
                       sefd=False):
    r"""Calibrate visibilities from 3 antennas.

    TODO: Only keep one of the gain tables in the end, on a fine timescale.
    The caltable\_to\_etcd function should be able to handle this, but I haven't
    tested that yet.

    Parameters
    ----------
    obs_params : dict
    ant_params : dict
    show_plots : Boolean
        If set to ``True``, plots of the delay and gain calibration solutions
        will be shown. Defaults ``False``.
    throw_exception : Boolean
        If set to ``True``, exceptions will be thrown after being logged in
        syslog. If set to ``False``, the exceptions will not be thrown, but
        will still be logged in syslog. Defaults ``True``.
    sefd : Boolean
        If set to ``True``, enough data (60 minutes) will be included in the
        measurement set to calculate the off-source power (60 minutes) and the
        calibration solutions will be solved against a model of ones. If set to
        ``False``, only 10 minutes will be included in the measurement set and
        the calibration solutison will be solved against a sky model.

    Returns
    -------
    status : int
        The status code of the pipeline. Decode with dsautils.calstatus.
    caltime : float
        The meridian crossing time of the source in MJD. If the input file
        could not be opened, ``None`` will be returned.
    """
    status = 0
    current_error = cs.UNKNOWN_ERR
    calstring = 'initialization'

    try:
        fname = obs_params['fname']
        msname = obs_params['msname']
        cal = obs_params['cal']
        utc_start = obs_params['utc_start']
        pt_dec = ant_params['pt_dec']
        antenna_order = ant_params['antenna_order']
        refant = ant_params['refant']
        antpos = ant_params['antpos']

        # Remove files that we will create so that things will fail if casa
        # doesn't write a table.
        casa_dirnames = ['{0}.ms'.format(msname),
                         '{0}_{1}_kcal'.format(msname, cal.name),
                         '{0}_{1}_2kcal'.format(msname, cal.name),
                         '{0}_{1}_bcal'.format(msname, cal.name),
                         '{0}_{1}_gpcal'.format(msname, cal.name),
                         '{0}_{1}_gacal'.format(msname, cal.name),
                         '{0}_{1}_gcal_ant'.format(msname, cal.name)]
        for dirname in casa_dirnames:
            if os.path.exists(dirname):
                shutil.rmtree(dirname)

        calstring = 'opening visibility file'
        current_error = (cs.INFILE_ERR | cs.INV_ANTNUM |
                         cs.INV_GAINAMP_P1 | cs.INV_GAINAMP_P2 |
                         cs.INV_GAINPHASE_P1 | cs.INV_GAINPHASE_P2 |
                         cs.INV_DELAY_P1 | cs.INV_DELAY_P2 |
                         cs.INV_GAINCALTIME | cs.INV_DELAYCALTIME)
        caldur = 60*u.min if sefd else 10*u.min
        fobs, blen, bname, _tstart, _tstop, tsamp, vis, mjd, lst, \
            transit_idx, antenna_order = dfio.read_psrfits_file(
                fname, cal, antenna_order=antenna_order, autocorrs=True,
                dur=caldur, utc_start=utc_start, dsa10=False, antpos=antpos)
        caltime = mjd[transit_idx]

        calstring = 'read and verification of visibility file'
        current_error = (cs.CAL_MISSING_ERR | cs.INV_GAINAMP_P1 |
                         cs.INV_GAINAMP_P2 | cs.INV_GAINPHASE_P1 |
                         cs.INV_GAINPHASE_P2 | cs.INV_DELAY_P1 |
                         cs.INV_DELAY_P2 | cs.INV_GAINCALTIME |
                         cs.INV_DELAYCALTIME)

        nt = vis.shape[1]
        assert nt > 0, "calibrator not in file"
        current_error = (cs.INFILE_FORMAT_ERR | cs.INV_GAINAMP_P1 |
                         cs.INV_GAINAMP_P2 | cs.INV_GAINPHASE_P1 |
                         cs.INV_GAINPHASE_P2 | cs.INV_DELAY_P1 |
                         cs.INV_DELAY_P2 | cs.INV_GAINCALTIME |
                         cs.INV_DELAYCALTIME)

        nant = len(antenna_order)
        nbls = (nant*(nant+1))//2
        assert nant == 3, ("triple_antenna_cal only works with a triplet of "
                           "antennas")
        assert int(refant) in antenna_order, ("refant {0} not in "
                                              "visibilities".format(refant))

        calstring = "flagging of ms data"
        current_error = (cs.FLAGGING_ERR | cs.INV_GAINAMP_P1 |
                         cs.INV_GAINAMP_P2 | cs.INV_GAINPHASE_P1 |
                         cs.INV_GAINPHASE_P2 | cs.INV_DELAY_P1 |
                         cs.INV_DELAY_P2 | cs.INV_GAINCALTIME |
                         cs.INV_DELAYCALTIME)
        maskf, _fraction_flagged = du.mask_bad_bins(vis, axis=2, thresh=2.0,
                                                    medfilt=True, nmed=129)
        maskt, _fraction_flagged = du.mask_bad_bins(vis, axis=1, thresh=2.0,
                                                    medfilt=True, nmed=129)
        maskp, _fraction_flagged = du.mask_bad_pixels(vis, thresh=6.0,
                                                      mask=maskt*maskf)
        mask = maskt*maskf*maskp
        fraction_flagged = 1-np.sum(mask)/mask.size
        if fraction_flagged > 0.15:
            logger.info("{0}% of data flagged".format(fraction_flagged))
        vis *= mask

        calstring = 'fringestopping'
        current_error = (cs.FRINGES_ERR | cs.INV_GAINAMP_P1 |
                         cs.INV_GAINAMP_P2 | cs.INV_GAINPHASE_P1 |
                         cs.INV_GAINPHASE_P2 | cs.INV_DELAY_P1 |
                         cs.INV_DELAY_P2 | cs.INV_GAINCALTIME |
                         cs.INV_DELAYCALTIME)
        df.fringestop(vis, blen, cal, mjd, fobs, pt_dec)

        calstring = 'writing to ms'
        current_error = (cs.MS_WRITE_ERR | cs.INV_GAINAMP_P1 |
                         cs.INV_GAINAMP_P2 | cs.INV_GAINPHASE_P1 |
                         cs.INV_GAINPHASE_P2 | cs.INV_DELAY_P1 |
                         cs.INV_DELAY_P2 | cs.INV_GAINCALTIME |
                         cs.INV_DELAYCALTIME)
        amp_model = df.amplitude_sky_model(cal, lst, pt_dec, fobs)
        amp_model = np.tile(amp_model[np.newaxis, :, :, np.newaxis],
                            (vis.shape[0], 1, 1, vis.shape[-1]))
        dmsio.convert_to_ms(cal, vis, mjd[0], '{0}'.format(msname),
                         bname, antenna_order, tsamp, nint=25,
                         antpos=antpos, dsa10=False,
                         model=None if sefd else amp_model)
        _check_path('{0}.ms'.format(msname))

        calstring = 'flagging of ms data'
        current_error = (cs.FLAGGING_ERR | cs.INV_GAINAMP_P1 |
                         cs.INV_GAINAMP_P2 | cs.INV_GAINPHASE_P1 |
                         cs.INV_GAINPHASE_P2 | cs.INV_DELAY_P1 |
                         cs.INV_DELAY_P2 | cs.INV_GAINCALTIME |
                         cs.INV_DELAYCALTIME)
        error = dc.flag_zeros(msname)
        if error > 0:
            status = cs.update(status, ['flagging_err'])
            logger.info("Non-fatal error in zero flagging")
        if 8 in antenna_order:
            error = dc.flag_antenna(msname, '8', pol='A')
            if error > 0:
                status = cs.update(status, ['flagging_err'])
                logger.info("Non-fatal error in antenna 8 flagging")

        # Antenna-based delay calibration
        calstring = 'delay calibration'
        current_error = (cs.DELAY_CAL_ERR | cs.INV_GAINAMP_P1 |
                         cs.INV_GAINAMP_P2 | cs.INV_GAINPHASE_P1 |
                         cs.INV_GAINPHASE_P2 | cs.INV_DELAY_P1 |
                         cs.INV_DELAY_P2 | cs.INV_GAINCALTIME |
                         cs.INV_DELAYCALTIME)
        error = dc.delay_calibration(msname, cal.name, refant=refant)
        if error > 0:
            status = cs.update(status, ['delay_cal_err'])
            logger.info('Non-fatal error occured in delay calibration.')
        _check_path('{0}_{1}_kcal'.format(msname, cal.name))

        calstring = 'flagging of ms data'
        current_error = (cs.FLAGGING_ERR | cs.INV_GAINAMP_P1 |
                         cs.INV_GAINAMP_P2 | cs.INV_GAINPHASE_P1 |
                         cs.INV_GAINPHASE_P2 | cs.INV_GAINCALTIME)
        bad_times, times, error = dc.get_bad_times(msname, cal.name, refant)
        if error > 0:
            status = cs.update(status, ['flagging_err'])
            logger.info('Non-fatal error occured in calculation of delays on '
                        'short timescales.')
        error = dc.flag_badtimes(msname, times, bad_times, nant)
        if error > 0:
            status = cs.update(status, ['flagging_err'])
            logger.info('Non-fatal error occured in flagging of bad timebins')
        _check_path('{0}_{1}_2kcal'.format(msname, cal.name))

        calstring = 'baseline-based bandpass and gain calibration'
        current_error = (cs.GAIN_BP_CAL_ERR | cs.INV_GAINAMP_P1 |
                         cs.INV_GAINAMP_P2 | cs.INV_GAINPHASE_P1 |
                         cs.INV_GAINPHASE_P2 | cs.INV_GAINCALTIME)
        print(os.path.exists('J1042+1203_1_J1042+1203_gacal'))
        error = dc.gain_calibration_blbased(msname, cal.name, refant=refant,
                                            tga='30s', tgp='30s')
        if error > 0:
            status = cs.update(status, ['gain_bp_cal_err'])
            logger.info('Non-fatal error occured in gain/bandpass calibration.')
        print(os.path.exists('J1042+1203_1_J1042+1203_gacal'))
        for fname in ['{0}_{1}_bcal'.format(msname, cal.name),
                      '{0}_{1}_gpcal'.format(msname, cal.name),
                      '{0}_{1}_gacal'.format(msname, cal.name)]:
            _check_path(fname)
        calstring = 'calculation of antenna gains'
        gamp, _tamp, famp, _ant1, _ant2 = dmsio.read_caltable(
            '{0}_{1}_gacal'.format(msname, cal.name), cparam=True)
        gphase, _tphase, fphase, _ant1, _ant2 = dmsio.read_caltable(
            '{0}_{1}_gpcal'.format(msname, cal.name), cparam=True)
        gains = (gamp*gphase).squeeze(axis=2)
        flags = (famp*fphase).squeeze(axis=2)
        # This makes some assumptions about the bl order! Should add some
        # statements to make sure it's true
        gains, flags = dc.fill_antenna_gains(gains, flags)

        # These tables will contain the results on fine time-scales.
        gamp = np.abs(gains).astype(np.complex128)
        gamp = gamp.reshape(gamp.shape[0], -1)
        # tb = cc.table()
        with table('{0}_{1}_gacal'.format(msname, cal.name),
                   readonly=False) as tb:
            # tb.open('{0}_{1}_gacal'.format(msname, cal.name), nomodify=False)
            shape = np.array(tb.CPARAM[:]).shape
            tb.putcol('CPARAM', gamp.reshape(shape))
        # tb.close()

        gphase = np.exp(1.j*np.angle(gains))
        with table('{0}_{1}_gpcal'.format(msname, cal.name),
                   readonly=False) as tb:
            # tb.open('{0}_{1}_gpcal'.format(msname, cal.name), nomodify=False)
            shape = np.array(tb.CPARAM[:]).shape
            tb.putcol('CPARAM', gphase.reshape(shape))
        # tb.close()

        if not sefd:
            # reduce to a single value to use
            mask = np.ones(flags.shape)
            mask[flags == 1] = np.nan
            gains = np.nanmedian(gains*mask, axis=1, keepdims=True)
            flags = np.min(flags, axis=1, keepdims=True)

            if 8 in antenna_order:
                flags[..., 0] = 1

            shutil.copytree('{0}/template_gcal_ant'.format(
                ct.PKG_DATA_PATH), '{0}_{1}_gcal_ant'.format(msname,
                                                                 cal.name))

            # Write out a new gains that is a single value.
            #tb = cc.table()
            with table('{0}_{1}_gcal_ant'.format(msname, cal.name),
                       readonly=False) as tb:
                # tb.open('{0}_{1}_gcal_ant'.format(msname, cal.name),
                #     nomodify=False)
                tb.putcol('TIME', np.ones(6)*np.median(mjd)*ct.SECONDS_PER_DAY)
                tb.putcol('FLAG', flags.T)
                tb.putcol('CPARAM', gains.T)
            # tb.close()
            _check_path('{0}_{1}_gcal_ant'.format(msname, cal.name))

    except Exception as exc:
        status = cs.update(status, current_error)
        du.exception_logger(logger, calstring, exc, throw_exceptions)
        try:
            caltime
        except NameError:
            caltime = Time.now().mjd

    return status, caltime

def plot_solutions(msname, calname, figure_dir="./figures/",
                   show_plots=False):
    r"""Plots the antenna delay, gain and bandpass calibration solutions.

    Parameters
    ----------
    msname : str
        The name of the measurement set. Used to identify the calibration
        tables.
    calname : str
        The name of the calibrator. Used to identify the calibration tables.
    antenna_order : list
        The antenna names, in order.
    fobs : array
        The central frequency of each channel, in GHz.
    blbased : boolean
        True of the calibration was baseline-based.
    figure_dir : str
        The location to save the figures.  Defaults ``./figures``.
    show_plots : boolean
        If False, plots are closed after being saved. Defaults False.
    """
    outname = '{0}/{1}_{2}'.format(figure_dir, msname, calname)
    _ = dp.plot_antenna_delays(msname, calname, outname=outname,
                               show=show_plots)
    _ = dp.plot_gain_calibration(msname, calname, outname=outname,
                                 show=show_plots)
    _ = dp.plot_bandpass(msname, calname,
                         outname=outname, show=show_plots)

def calibration_head(obs_params, ant_params, write_to_etcd=False,
                     throw_exceptions=None, sefd=False):
    """Controls calibraiton of a dataset and writing the results to etcd.

    Parameters
    ----------
    obs_params : list
        The observing parameters.
    ant_params : list
        The antenna configuration.
    write_to_etcd : boolean
        If set to ``True``, the results of the calibration are pushed to etcd.
        Defaults ``False``.
    throw_exceptions : boolean
        If set to ``False``, exceptions are not raised after being logged to
        syslog. Instead, `calibration_head` and `triple_antenna_cal` return the
        status value. If set to ``None``, `throw_exceptions` will be set to
        ``not write_to_etcd``.
    sefd : boolean
        If set to ``True``, the solutions will be solved against a model of
        ones in order to allow fitting of the source pass to the antenna gains
        and 60 minutes will be saved to the measurement set.  If set to
        ``False``, a sky model will be used in calibration and only 10 minutes
        of data is saved to the measurement set.

    Returns
    -------
    int
        The status code. Decode with dsautils.calstatus.
    """
    if throw_exceptions is None:
        throw_exceptions = not write_to_etcd
    logger.info('Beginning calibration of ms {0}.ms (start time {1}) using '
                'source {2}'.format(obs_params['msname'],
                                    obs_params['utc_start'].isot,
                                    obs_params['cal'].name))
    status, caltime = triple_antenna_cal(obs_params, ant_params,
                                         throw_exceptions, sefd)
    logger.info('Ending calibration of ms {0}.ms (start time {1}) using '
                'source {2} with status {3}'.format(
                    obs_params['msname'], obs_params['utc_start'].isot,
                    obs_params['cal'].name, status))
    print('Status: {0}'.format(cs.decode(status)))
    print('')
    if write_to_etcd:
        dmsio.caltable_to_etcd(obs_params['msname'], obs_params['cal'].name,
                            ant_params['antenna_order'], caltime, status)
    return status

def _gauss(xvals, amp, mean, sigma, offset):
    """Calculates the value of a Gaussian at the locations `x`.
    """
    return amp*np.exp(-(xvals-mean)**2/(2*sigma**2))+offset

def calculate_sefd(msname, cal, fmin=None, fmax=None,
                   baseline_cal=False, showplots=False,
                   msname_delaycal=None, calname_delaycal=None, halfpower=False):
    r"""Calculates the SEFD from a measurement set.

    The measurement set must have been calibrated against a model of ones and
    must include autocorrelations.

    Parameters
    ----------
    obs_params : dictionary
        Parameters of the observing run.  The following must be defined:

            msname : str
                The measurement set name.  The measurement set `msname`.ms will
                be opened.
            cal : src class instance
                The calibrator source.  Will be used to identify the correct
                calibration tables.  The table `msname`\_`cal.name`\_gacal will
                be opened.

    fmin : float
        The lowest frequency to consider when calculating the off-source power
        to use in the SEFD calculation, in GHz. Channels below this frequency
        will be flagged. Defaults 1.35.
    fmax : float
        The greatest frequency to consider when calculating the off-source
        power to use in the SEFD calculation, in GHz.  Channels above this
        frequency will be flagged.  Defaults 1.45.
    baseline_cal : Boolean
        Set to ``True`` if the gain tables were derived using baseline-based
        calibration. Set to ``False`` if the gain tables were derived using
        antenna-based calibration. Defaults ``True``.
    showplots : Boolean
        If set to ``True``, plots will be generated that show the Gaussian fits
        to the gains. Defaults ``False``.
    halfpower : Boolean
        If True, will calculate the sefd using the half-power point instead of
        using the off-source power.  Defaults False.

    Returns
    -------
    sefds : ndarray
        The SEFD of each antenna/polarization pair, in Jy. Dimensions (antenna,
        polarization).
    ant_gains : ndarray
        The antenna gains in 1/Jy. Dimensions (antenna, polarization).
    ant_transit_time : ndarray
        The meridian transit time of the source as seen by each antenna/
        polarization pair, in MJD. Dimensions (antenna, polarization).
    """
    # Change so figures saved if showplots is False
    if msname_delaycal is None:
        msname_delaycal = msname
    if calname_delaycal is None:
        calname_delaycal = cal.name
    npol = 2

    # Get the visibilities (for autocorrs)
    dc.apply_delay_bp_cal(msname, calname_delaycal, msnamecal=msname_delaycal,
                         blbased=baseline_cal)
    vis, tvis, fvis, flag, ant1, ant2 = dmsio.extract_vis_from_ms(
        msname, 'CORRECTED_DATA')
    mask = (1-flag).astype(float)
    mask[mask < 0.5] = np.nan
    vis = vis*mask
    vis = vis[ant1 == ant2, ...]
    antenna_order = ant1[ant1 == ant2]
    nant = len(antenna_order)
    # Note that these are antenna idxs, not names

    # Open the gain files and read in the gains
    gain, time, flag, ant1, ant2 = dmsio.read_caltable(
        '{0}_{1}_gacal'.format(msname, cal.name), cparam=True)
    gain[flag] = np.nan
    # phase, _, flag, ant1p, ant2p = dmsio.read_caltable(
    #     '{0}_{1}_gpcal'.format(msname, cal.name), cparam=True)
    # phase[flag] = np.nan
    # assert np.all(ant1p == ant1)
    # assert np.all(ant2p == ant2)
    # gain = gain*phase
    antenna, gain = dmsio.get_antenna_gains(gain, ant1, ant2)
    antenna = list(antenna)
    idxs = [antenna.index(ant) for ant in antenna_order]
    gain = gain[idxs, ...]
    assert gain.shape[0] == nant
    gain = np.abs(gain*np.conjugate(gain))
    gain = np.abs(np.nanmean(gain, axis=2)).squeeze(axis=2)
    idxl = np.searchsorted(fvis, fmin) if fmin is not None else 0
    idxr = np.searchsorted(fvis, fmax) if fmax is not None else vis.shape[-2]
    if idxl < idxr:
        vis = vis[..., idxl:idxr, :]
    else:
        vis = vis[..., idxr:idxl, :]
    imag_fraction = np.nanmean((vis.imag/vis.real).reshape(nant, -1),
                               axis=-1)
    assert np.nanmax(np.abs(imag_fraction) < 1e-4), ("Autocorrelations have "
                                           "non-negligable imaginary "
                                           "components.")
    vis = np.abs(vis)

    # Complex gain includes an extra relative delay term
    # in the phase, but we really only want the amplitude
    # We will ignore the phase for now

    ant_gains_on = np.zeros((nant, npol))
    eant_gains_on = np.zeros((nant, npol))
    ant_transit_time = np.zeros((nant, npol))
    eant_transit_time = np.zeros((nant, npol))
    ant_transit_width = np.zeros((nant, npol))
    eant_transit_width = np.zeros((nant, npol))
    offbins_before = np.zeros((nant, npol), dtype=int)
    offbins_after = np.zeros((nant, npol), dtype=int)
    autocorr_gains_off = np.zeros((nant, npol))

    if showplots:
        nx = 3
        ny = nant//nx
        if nant%nx != 0:
            ny += 1
        _fig, ax = plt.subplots(ny, nx, figsize=(8*nx, 8*ny))
        ccyc = plt.rcParams['axes.prop_cycle'].by_key()['color']
        ax = ax.flatten()

    # Fit a Gaussian to the gains
    for i in range(nant):
        for j in range(npol):
            if showplots:
                ax[i].plot(time-time[0], gain[i, :, j], '.', color=ccyc[j])
            initial_params = [np.max(gain[i, :, j]), (time[-1]-time[0])/2,
                              0.0035, 0]
            params, cov = curve_fit(_gauss, time-time[0], gain[i, :, j],
                                    p0=initial_params)

            ant_gains_on[i, j] = params[0]+params[3]
            eant_gains_on[i, j] = np.sqrt(cov[0, 0])+np.sqrt(cov[3, 3])

            ant_transit_time[i, j] = time[0]+params[1]
            eant_transit_time[i, j] = np.sqrt(cov[1, 1])
            ant_transit_width[i, j] = params[2]
            eant_transit_width[i, j] = np.sqrt(cov[2, 2])
            if not halfpower:
                offbins_before[i, j] = np.searchsorted(
                    time, ant_transit_time[i, j]-ant_transit_width[i, j]*3)
                offbins_after[i, j] = len(time)-np.searchsorted(
                    time, ant_transit_time[i, j]+ant_transit_width[i, j]*3)
                idxl = np.searchsorted(
                    tvis, ant_transit_time[i, j]-ant_transit_width[i, j]*3)
                idxr = np.searchsorted(
                    tvis, ant_transit_time[i, j]+ant_transit_width[i, j]*3)
                autocorr_gains_off[i, j] = np.nanmedian(
                    np.concatenate(
                        (vis[i, :idxl, :, j], vis[i, idxr:, :, j]), axis=0))
            else:
                hwhm = np.sqrt(2*np.log(2))*ant_transit_width[i, j]
                idxl = np.searchsorted(tvis, ant_transit_time[i, j]-hwhm)
                idxr = np.searchsorted(tvis, ant_transit_time[i, j]+hwhm)
                autocorr_gains_off[i, j] = np.nanmedian(
                    np.concatenate(
                        (vis[i, idxl-10:idxl+10, :, j],
                         vis[i, idxr-10:idxr+10, :, j]), axis=0))
            # print('{0} {1}: on {2:.2f}'.format(
            #    antenna_order[i], 'A' if j == 0 else 'B', ant_gains_on[i, j]))
            if showplots:
                ax[i].plot(time-time[0], _gauss(time-time[0], *params), '-',
                           color=ccyc[j],
                           label='{0} {1}: {2:.4f}/Jy'.format(
                               antenna_order[i], 'A' if j == 0 else 'B',
                               ant_gains_on[i, j]/cal.I))
                ax[i].legend()
                ax[i].set_xlabel("Time (d)")
                ax[i].set_ylabel("Unnormalized power")

    ant_gains = ant_gains_on/cal.I
    if halfpower:
        sefds = autocorr_gains_off/(ant_gains/2)
    else:
        sefds = autocorr_gains_off/ant_gains

    return antenna_order, sefds, ant_gains, ant_transit_time

def dsa10_cal(fname, msname, cal, pt_dec, antpos, refant, badants=None):
    """Calibrate dsa10 data.
    """
    if badants is None:
        badants = []

    for file_path in ['{0}.ms'.format(msname),
                      '{0}_{1}_kcal'.format(msname, cal.name),
                      '{0}_{1}_gacal'.format(msname, cal.name),
                      '{0}_{1}_gpcal'.format(msname, cal.name),
                      '{0}_{1}_bcal'.format(msname, cal.name),
                      '{0}_{1}_2kcal'.format(msname, cal.name)]:
        if os.path.exists(file_path):
            shutil.rmtree(file_path)

    fobs, blen, bname, tstart, _tstop, tsamp, vis, mjd, lst, _transit_idx, \
        antenna_order = dfio.read_psrfits_file(
            fname, cal, dur=10*u.min, antpos=antpos, badants=badants)
    nant = len(antenna_order)

    df.fringestop(vis, blen, cal, mjd, fobs, pt_dec)
    amp_model = df.amplitude_sky_model(cal, lst, pt_dec, fobs)
    amp_model = np.tile(amp_model[np.newaxis, :, :, np.newaxis],
                        (vis.shape[0], 1, 1, vis.shape[-1]))

    dmsio.convert_to_ms(cal, vis, tstart, msname, bname, antenna_order,
                     tsamp=tsamp, nint=25, antpos=antpos,
                     model=amp_model)
    _check_path('{0}.ms'.format(msname))

    dc.flag_zeros(msname)
    if '8' in antenna_order:
        dc.flag_antenna(msname, '8', pol='A')

    dc.delay_calibration(msname, cal.name, refant)
    _check_path('{0}_{1}_kcal'.format(msname, cal.name))

    bad_times, times, _error = dc.get_bad_times(msname, cal.name, refant)
    dc.flag_badtimes(msname, times, bad_times, nant)

    dc.gain_calibration(msname, cal.name, refant=refant, tga="10s", tgp="inf")
    for tbl in ['gacal', 'gpcal', 'bcal']:
        _check_path('{0}_{1}_{2}'.format(msname, cal.name, tbl))
