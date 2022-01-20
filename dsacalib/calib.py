"""Functions for calibration of DSA-110 visibilities.

These functions use the CASA package casatools to calibrate
visibilities stored as measurement sets.

Author: Dana Simard, dana.simard@astro.caltech.edu, 10/2019

"""
import os
# Always import scipy before casatools
from scipy.fftpack import fft, fftshift, fftfreq
from scipy.signal import medfilt
import numpy as np
import casatools as cc
from casacore.tables import table
from dsacalib.ms_io import read_caltable

def delay_calibration_worker(msname, sourcename, refant, t, combine_spw, name):
    r"""Calibrates delays using CASA.

    Uses CASA to calibrate delays and write the calibrated visibilities to the
    corrected_data column of the measurement set.

    Parameters
    ----------
    msname : str
        The name of the measurement set.  The measurement set `msname`.ms will
        be opened.
    sourcename : str
        The name of the calibrator source. The calibration table will be
        written to `msname`\_`sourcename`\_kcal.
    refant : str
        The reference antenna to use in calibration. If  type *str*, this is
        the name of the antenna.  If type *int*, it is the index of the antenna
        in the measurement set.
    t : str
        The integration time to use before calibrating, e.g. ``'inf'`` or
        ``'60s'``.  See the CASA documentation for more examples. Defaults to
        ``'inf'`` (averaging over the entire observation time).
    combine_spw : boolean
        If True, distinct spws in the same ms will be combined before delay
        calibration.
    name : str
        The suffix for the calibration table.

    Returns
    -------
    int
        The number of errors that occured during calibration.
    """
    if combine_spw:
        combine = 'field,scan,obs,spw'
    else:
        combine = 'field,scan,obs'
    error = 0
    cb = cc.calibrater()
    error += not cb.open('{0}.ms'.format(msname))
    error += not cb.setsolve(
        type='K',
        t=t,
        refant=refant,
        combine=combine,
        table='{0}_{1}_{2}'.format(msname, sourcename, name)
    )
    error += not cb.solve()
    error += not cb.close()
    return error

def delay_calibration(msname, sourcename, refants, t1='inf', t2='60s',
                      combine_spw=False):
    r"""Calibrates delays using CASA.

    Uses CASA to calibrate delays and write the calibrated visibilities to the
    corrected_data column of the measurement set.

    Parameters
    ----------
    msname : str
        The name of the measurement set.  The measurement set `msname`.ms will
        be opened.
    sourcename : str
        The name of the calibrator source. The calibration table will be
        written to `msname`\_`sourcename`\_kcal.
    refants : list(str)
        The reference antennas to use in calibration. If list items are type
        *str*, this is the name of the antenna.  If type *int*, it is the index
        of the antenna in the measurement set. An average is done over all
        reference antennas to get the final delay table.
    t1 : str
        The integration time to use before calibrating to generate the final
        delay table, e.g. ``'inf'`` or ``'60s'``.  See the CASA documentation
        for more examples. Defaults to ``'inf'`` (averaging over the entire
        observation time).
    t2 : str
        The integration time to use before fast delay calibrations used to flag
        antennas with poor delay solutions.
    combine_spw : boolean
        If True, distinct spws in the same ms will be combined before delay
        calibration.

    Returns
    -------
    int
        The number of errors that occured during calibration.
    """
    # TODO : revisit and get this to work with a list of antennas
    assert isinstance(refants, list)
    error = 0
    refant = None
    for t in [t1, t2]:
        kcorr = None
        for refant in refants:
            if isinstance(refant, str):
                refantidx = int(refant)-1
            else:
                refantidx = refant
            error += delay_calibration_worker(
                msname,
                sourcename,
                refant,
                t,
                combine_spw,
                'ref{0}_{1}kcal'.format(refant, '' if t==t1 else '2')
            )
            if kcorr is None:
                kcorr, _, flags, _, ant2 = read_caltable(
                    '{0}_{1}_ref{2}_{3}kcal'.format(
                        msname,
                        sourcename,
                        refant,
                        '' if t==t1 else '2'
                    ),
                    cparam=False,
                    reshape=False
                )
            else:
                kcorrtmp, _, flagstmp, _, ant2tmp = read_caltable(
                    '{0}_{1}_ref{2}_{3}kcal'.format(
                        msname,
                        sourcename,
                        refant,
                        '' if t==t1 else '2'
                    ),
                    cparam=False,
                    reshape=False
                )
                antflags = np.abs(
                    flags.reshape(flags.shape[0], -1).mean(axis=1)-1) < 1e-5
                assert antflags[refantidx] == 0, \
                    'Refant {0} is flagged in kcorr!'.format(refant) + \
                    'Choose refants that are separated in uv-space.'
                kcorr[antflags, ...] = kcorrtmp[antflags, ...]-\
                                       kcorr[refantidx, ...]
                ant2[antflags, ...] = ant2tmp[antflags, ...]
                flags[antflags, ...] = flagstmp[antflags, ...]
        # write out to a table
        with table(
            '{0}_{1}_ref{2}_{3}kcal'.format(
                msname,
                sourcename,
                refant,
                '' if t==t1 else '2'
            ),
            readonly=False
        ) as tb:
            tb.putcol('FPARAM', kcorr)
            tb.putcol('FLAG', flags)
            tb.putcol('ANTENNA2', ant2)
        os.rename(
            '{0}_{1}_ref{2}_{3}kcal'.format(
                msname,
                sourcename,
                refant,
                '' if t==t1 else '2'
            ),
            '{0}_{1}_{2}kcal'.format(
                msname,
                sourcename,
                '' if t==t1 else '2'
            )
        )
    return error

def gain_calibration(
    msname, sourcename, refant, blbased=False, forsystemhealth=False,
    keepdelays=False, tbeam='30s', interp_thresh=1.5, interp_polyorder=7
):
    r"""Use CASA to calculate bandpass and complex gain solutions.

    Saves solutions to calibration tables and calibrates the measurement set by
    applying delay, bandpass, and complex gain solutions.  Uses baseline-based
    calibration routines within CASA.

    Parameters
    ----------
    msname : str
        The name of the measurement set.  The MS `msname`.ms will be opened.
    sourcename : str
        The name of the calibrator source.  The calibration table will be
        written to `msname`\_`sourcename`\_kcal.
    refant : str
        The reference antenna to use in calibration.  If type *str*, this is
        the name of the antenna.  If type *int*, it is the index of the antenna
        in the measurement set.
    blbased : boolean
        Set to True if baseline-based calibration desired.
    forsystemhealth : boolean
        Set to True if gain calibration is for system health monitoring. Delays
        will be kept at full resolution. If set to False, then at least some of
        the delay will be incorporated into the bandpass gain table.
    keepdelays : boolean
        Set to True if you want to update the delays currently set in the
        system. In this case, delay changes of integer 2 ns will be kept in the
        delay calibration table, and residual delays will be incorporated into
        the bandpass gain table. If set to False, all of the delay will be
        incorporated into the bandpass gain table.
    tbeam : str
        The integration time to use when measuring gain variations over time,
        e.g. ``'inf'`` or ``'60s'``.  See the CASA documentation for more
        examples.
    interp_thresh : float
        Sets flagging of bandpass solutions before interpolating in order to
        smooth the solutions. After median baselining, any points that deviate
        by more than interp_thresh*std are flagged.
    interp_polyorder : int
        The order of the polynomial used to smooth bandpass solutions.

    Returns
    -------
    int
        The number of errors that occured during calibration.
    """
    combine = 'field,scan,obs'
    spwmap = [-1]
    error = 0
    fref_snaps = 0.03 # SNAPs correct to freq of 30 MHz

    # Convert delay calibration into a bandpass representation
    caltables = [{
        'table': '{0}_{1}_kcal'.format(msname, sourcename),
        'type': 'K',
        'spwmap': spwmap
    }]

    if not forsystemhealth:
        with table('{0}.ms/SPECTRAL_WINDOW'.format(msname)) as tb:
            fobs = np.array(tb.CHAN_FREQ[:]).squeeze(0)/1e9
            fref = np.array(tb.REF_FREQUENCY[:])/1e9
        cb = cc.calibrater()
        error += not cb.open('{0}.ms'.format(msname))
        error += apply_calibration_tables(cb, caltables)
        error += not cb.setsolve(
            type='MF' if blbased else 'B',
            combine=combine,
            table='{0}_{1}_bkcal'.format(msname, sourcename),
            refant=refant,
            apmode='a',
            solnorm=True
        )
        error += not cb.solve()
        error += not cb.close()

        with table(
            '{0}_{1}_kcal'.format(msname, sourcename), readonly=False
        ) as tb:
            kcorr = np.array(tb.FPARAM[:])
            tb.putcol('FPARAM', np.zeros(kcorr.shape, kcorr.dtype))

        with table(
            '{0}_{1}_bkcal'.format(msname, sourcename),
            readonly=False
        ) as tb:
            bpass = np.array(tb.CPARAM[:])
            bpass = np.ones(bpass.shape, bpass.dtype)
            kcorr = kcorr.squeeze()
            bpass *= np.exp(
                2j*np.pi*(fobs[:, np.newaxis]-fref)*(
                    kcorr[:, np.newaxis, :]#-kcorr[int(refant)-1, :]
                )
            )
            tb.putcol('CPARAM', bpass)
        caltables += [
            {
                'table': '{0}_{1}_bkcal'.format(msname, sourcename),
                'type': 'B',
                'spwmap': spwmap
            }
        ]

    # Rough bandpass calibration
    cb = cc.calibrater()
    error += not cb.open('{0}.ms'.format(msname))
    error += apply_calibration_tables(cb, caltables)
    error += cb.setsolve(
        type='B',
        combine=combine,
        table='{0}_{1}_bcal'.format(msname, sourcename),
        refant=refant,
        apmode='ap',
        t='inf',
        solnorm=True
    )
    error += cb.solve()
    error += cb.close()

    caltables += [
        {
            'table': '{0}_{1}_bcal'.format(msname, sourcename),
            'type': 'B',
            'spwmap': spwmap
        }
    ]

    # Gain calibration
    cb = cc.calibrater()
    error += cb.open('{0}.ms'.format(msname))
    error += apply_calibration_tables(cb, caltables)
    error += cb.setsolve(
        type='G',
        combine=combine,
        table='{0}_{1}_gacal'.format(msname, sourcename),
        refant=refant,
        apmode='a',
        t='inf'
    )
    error += cb.solve()
    error += cb.close()

    caltables += [
        {
            'table': '{0}_{1}_gacal'.format(msname, sourcename),
            'type': 'G',
            'spwmap': spwmap
        }
    ]

    cb = cc.calibrater()
    error += cb.open('{0}.ms'.format(msname))
    error += apply_calibration_tables(cb, caltables)
    error += cb.setsolve(
        type='G',
        combine=combine,
        table='{0}_{1}_gpcal'.format(msname, sourcename),
        refant=refant,
        apmode='p',
        t='inf'
    )
    error += cb.solve()
    error += cb.close()

    # Final bandpass calibration
    caltables = [
        {
            'table': '{0}_{1}_gacal'.format(msname, sourcename),
            'type': 'G',
            'spwmap': spwmap
        },
        {
            'table': '{0}_{1}_gpcal'.format(msname, sourcename),
            'type': 'G',
            'spwmap': spwmap
        }
    ]

    if not forsystemhealth:
        caltables += [
            {
                'table': '{0}_{1}_bkcal'.format(msname, sourcename),
                'type': 'B',
                'spwmap': spwmap
            }
        ]
    caltables += [
        {
            'table': '{0}_{1}_kcal'.format(msname, sourcename),
            'type': 'K',
            'spwmap': spwmap
        }
    ]

    cb = cc.calibrater()
    error += cb.open('{0}.ms'.format(msname))
    error += apply_calibration_tables(cb, caltables)
    error += cb.setsolve(
        type='B',
        combine=combine,
        table='{0}_{1}_bacal'.format(msname, sourcename),
        refant=refant,
        apmode='a',
        t='inf',
        solnorm=True
    )
    error += cb.solve()
    error += cb.close()

    if not forsystemhealth:
        interpolate_bandpass_solutions(
            msname,
            sourcename,
            thresh=interp_thresh,
            polyorder=interp_polyorder,
            mode='a'
        )

    caltables += [
        {
            'table': '{0}_{1}_bacal'.format(msname, sourcename),
            'type': 'B',
            'spwmap': spwmap
        }
    ]

    cb = cc.calibrater()
    error += cb.open('{0}.ms'.format(msname))
    error += apply_calibration_tables(cb, caltables)
    error += cb.setsolve(
        type='B',
        combine=combine,
        table='{0}_{1}_bpcal'.format(msname, sourcename),
        refant=refant,
        apmode='p',
        t='inf',
        solnorm=True
    )
    error += cb.solve()
    error += cb.close()

    if not forsystemhealth: # and not keepdelays:
        interpolate_bandpass_solutions(
            msname,
            sourcename,
            thresh=interp_thresh,
            polyorder=interp_polyorder,
            mode='p'
        )

    if not forsystemhealth and keepdelays:
        with table(
            '{0}_{1}_kcal'.format(msname, sourcename),
            readonly=False
        ) as tb:
            fparam = np.array(tb.FPARAM[:])
            newparam = np.round(kcorr[:, np.newaxis, :]/2)*2
            print('kcal', fparam.shape, newparam.shape)
            tb.putcol('FPARAM', newparam)
        with table(
            '{0}_{1}_bkcal'.format(msname, sourcename),
            readonly=False
        ) as tb:
            bpass = np.array(tb.CPARAM[:])
            print(newparam.shape, bpass.shape, fobs.shape)
            bpass *= np.exp(
                -2j*np.pi*(fobs[:, np.newaxis]-fref_snaps)*
                    newparam
                )
            print(bpass.shape)
            tb.putcol('CPARAM', bpass)

    if forsystemhealth:
        caltables += [
            {
                'table': '{0}_{1}_bpcal'.format(msname, sourcename),
                'type': 'B',
                'spwmap': spwmap
            }
        ]
        cb = cc.calibrater()
        error += not cb.open('{0}.ms'.format(msname))
        error += apply_calibration_tables(cb, caltables)
        error += not cb.setsolve(
            type='M' if blbased else 'G',
            combine=combine,
            table='{0}_{1}_2gcal'.format(msname, sourcename),
            refant=refant,
            apmode='ap',
            t=tbeam
        )
        error += not cb.solve()
        error += not cb.close()

    return error


def flag_antenna(msname, antenna, datacolumn='data', pol=None):
    """Flags an antenna in a measurement set using CASA.

    Parameters
    ----------
    msname : str
        The name of the measurement set.  The MS `msname`.ms will be opened.
    antenna : str
        The antenna to flag. If type *str*, this is the name of the antenna. If
        type *int*, the index of the antenna in the measurement set.
    datacolumn : str
        The column of the measurement set to flag. Options are ``'data'``,
        ``'model'``, ``'corrected'`` for the uncalibrated visibilities, the
        visibility model (used by CASA to calculate calibration solutions), the
        calibrated visibilities.  Defaults to ``'data'``.
    pol : str
        The polarization to flag. Must be `'A'` (which is mapped to
        polarization 'XX' of the CASA measurement set) or `'B'` (mapped to
        polarization 'YY').  Can also be `None`, for which both polarizations
        are flagged.  Defaults to `None`.

    Returns
    -------
    int
        The number of errors that occured during flagging.
    """
    if isinstance(antenna, int):
        antenna = str(antenna)
    error = 0
    ag = cc.agentflagger()
    error += not ag.open('{0}.ms'.format(msname))
    error += not ag.selectdata()
    rec = {}
    rec['mode'] = 'manual'
    #rec['clipoutside'] = False
    rec['datacolumn'] = datacolumn
    rec['antenna'] = antenna
    if pol is not None:
        rec['correlation'] = 'XX' if pol == 'A' else 'YY'
    else:
        rec['correlation'] = 'XX,YY'
    error += not ag.parseagentparameters(rec)
    error += not ag.init()
    error += not ag.run()
    error += not ag.done()

    return error

def flag_manual(msname, key, value, datacolumn='data', pol=None):
    """Flags a measurement set in CASA using a flagging string.

    Parameters
    ----------
    msname : str
        The name of the measurement set.  The MS `msname`.ms will be opened.
    key : str
        The CASA-interpreted flagging specifier.
    datacolumn : str
        The column of the measurement set to flag. Options are ``'data'``,
        ``'model'``, ``'corrected'`` for the uncalibrated visibilities, the
        visibility model (used by CASA to calculate calibration solutions), the
        calibrated visibilities.  Defaults to ``'data'``.
    pol : str
        The polarization to flag. Must be `'A'` (which is mapped to
        polarization 'XX' of the CASA measurement set) or `'B'` (mapped to
        polarization 'YY').  Can also be `None`, for which both polarizations
        are flagged.  Defaults to `None`.

    Returns
    -------
    int
        The number of errors that occured during flagging.
    """
    error = 0
    ag = cc.agentflagger()
    error += not ag.open('{0}.ms'.format(msname))
    error += not ag.selectdata()
    rec = {}
    rec['mode'] = 'manual'
    rec['datacolumn'] = datacolumn
    rec[key] = value
    if pol is not None:
        rec['correlation'] = 'XX' if pol == 'A' else 'YY'
    else:
        rec['correlation'] = 'XX,YY'
    error += not ag.parseagentparameters(rec)
    error += not ag.init()
    error += not ag.run()
    error += not ag.done()

    return error

def flag_baselines(msname, datacolumn='data', uvrange='2~15m'):
    """Flags an antenna in a measurement set using CASA.

    Parameters
    ----------
    msname : str
        The name of the measurement set.  The MS `msname`.ms will be opened.
    datacolumn : str
        The column of the measurement set to flag. Options are ``'data'``,
        ``'model'``, ``'corrected'`` for the uncalibrated visibilities, the
        visibility model (used by CASA to calculate calibration solutions), the
        calibrated visibilities.  Defaults to ``'data'``.
    uvrange : str
        The uvrange to flag. Should be CASA-interpretable.

    Returns
    -------
    int
        The number of errors that occured during flagging.
    """
    error = 0
    ag = cc.agentflagger()
    error += not ag.open('{0}.ms'.format(msname))
    error += not ag.selectdata()
    rec = {}
    rec['mode'] = 'manual'
    rec['datacolumn'] = datacolumn
    rec['uvrange'] = uvrange
    error += not ag.parseagentparameters(rec)
    error += not ag.init()
    error += not ag.run()
    error += not ag.done()

    return error

def reset_flags(msname, datacolumn=None):
    """Resets all flags in a measurement set, so that all data is unflagged.

    Parameters
    ----------
    msname : str
        The name of the measurement set. The MS `msname`.ms will be opened.
    datacolumn : str
        The column of the measurement set to flag. Options are ``'data'``,
        ``'model'``, ``'corrected'`` for the uncalibrated visibilities, the
        visibility model (used by CASA to calculate calibration solutions), the
        calibrated visibilities.  Defaults to ``'data'``.

    Returns
    -------
    int
        The number of errors that occured during flagging.
    """
    error = 0
    ag = cc.agentflagger()
    error += not ag.open('{0}.ms'.format(msname))
    error += not ag.selectdata()
    rec = {}
    rec['mode'] = 'unflag'
    if datacolumn is not None:
        rec['datacolumn'] = datacolumn
    rec['antenna'] = ''
    error += not ag.parseagentparameters(rec)
    error += not ag.init()
    error += not ag.run()
    error += not ag.done()

    return error

def flag_zeros(msname, datacolumn='data'):
    """Flags all zeros in a measurement set.

    Parameters
    ----------
    msname : str
        The name of the measurement set. The MS `msname`.ms will be opened.
    datacolumn : str
        The column of the measurement set to flag. Options are ``'data'``,
        ``'model'``, ``'corrected'`` for the uncalibrated visibilities, the
        visibility model (used by CASA to calculate calibration solutions), the
        calibrated visibilities.  Defaults to ``'data'``.

    Returns
    -------
    int
        The number of errors that occured during flagging.
    """
    error = 0
    ag = cc.agentflagger()
    error += not ag.open('{0}.ms'.format(msname))
    error += not ag.selectdata()
    rec = {}
    rec['mode'] = 'clip'
    rec['clipzeros'] = True
    rec['datacolumn'] = datacolumn
    error += not ag.parseagentparameters(rec)
    error += not ag.init()
    error += not ag.run()
    error += not ag.done()

    return error

# TODO: Change times to not use mjds, but mjd instead
def flag_badtimes(msname, times, bad, nant, datacolumn='data', verbose=False):
    """Flags bad time bins for each antenna in a measurement set using CASA.

    Could use some work to select antennas to flag in a smarter way.

    Parameters
    ----------
    msname : str
        The name of the measurement set. The MS `msname`.ms will be opened.
    times : ndarray
        A 1-D array of times, type float, seconds since MJD=0. Times should be
        equally spaced and cover the entire time range of the measurement set,
        but can be coarser than the resolution of the measurement set.
    bad : ndarray
        A 1-D boolean array with dimensions (len(`times`), `nant`). Should have
        a value of ``True`` if the corresponding timebins should be flagged.
    nant : int
        The number of antennas in the measurement set (includes ones not in the
        visibilities).
    datacolumn : str
        The column of the measurement set to flag. Options are ``'data'``,
        ``'model'``, ``'corrected'`` for the uncalibrated visibilities, the
        visibility model (used by CASA to calculate calibration solutions), the
        calibrated visibilities.  Defaults to ``'data'``.
    verbose : boolean
        If ``True``, will print information about the antenna/time pairs being
        flagged.  Defaults to ``False``.

    Returns
    -------
    int
        The number of errors that occured during flagging.
    """
    error = 0
    tdiff = np.median(np.diff(times))
    ag = cc.agentflagger()
    error += not ag.open('{0}.ms'.format(msname))
    error += not ag.selectdata()
    for i in range(nant):
        rec = {}
        rec['mode'] = 'clip'
        rec['clipoutside'] = False
        rec['datacolumn'] = datacolumn
        rec['antenna'] = str(i)
        rec['polarization_type'] = 'XX'
        tstr = ''
        for j, timesj in enumerate(times):
            if bad[j]:
                if len(tstr) > 0:
                    tstr += '; '
                tstr += '{0}~{1}'.format(timesj-tdiff/2, timesj+tdiff/2)
        if verbose:
            print('For antenna {0}, flagged: {1}'.format(i, tstr))
        error += not ag.parseagentparameters(rec)
        error += not ag.init()
        error += not ag.run()

        rec['polarization_type'] = 'YY'
        tstr = ''
        for j, timesj in enumerate(times):
            if bad[j]:
                if len(tstr) > 0:
                    tstr += '; '
                tstr += '{0}~{1}'.format(timesj-tdiff/2, timesj+tdiff/2)
        if verbose:
            print('For antenna {0}, flagged: {1}'.format(i, tstr))
        error += not ag.parseagentparameters(rec)
        error += not ag.init()
        error += not ag.run()
    error += not ag.done()

    return error

def calc_delays(vis, df, nfavg=5, tavg=True):
    """Calculates power as a function of delay from the visibilities.

    This uses scipy fftpack to fourier transform the visibilities along the
    frequency axis.  The power as a function of delay can then be used in
    fringe-fitting.

    Parameters
    ----------
    vis : ndarray
        The complex visibilities. 4 dimensions, (baseline, time, frequency,
        polarization).
    df : float
        The width of the frequency channels in GHz.
    nfavg : int
        The number of frequency channels to average by after the Fourier
        transform.  Defaults to 5.
    tavg : boolean
        If ``True``, the visibilities are averaged in time before the Fourier
        transform. Defaults to ``True``.

    Returns
    -------
    vis_ft : ndarray
        The complex visibilities, Fourier-transformed along the time axis. 3
        (or 4, if `tavg` is set to False) dimensions, (baseline, delay,
        polarization) (or (baseline, time, delay, polarization) if `tavg` is
        set to False).
    delay_arr : ndarray
        Float, the values of the delay pixels in nanoseconds.
    """
    nfbins = vis.shape[-2]//nfavg*nfavg
    npol = vis.shape[-1]
    if tavg:
        vis_ft = fftshift(fft(np.pad(vis[..., :nfbins, :].mean(1),
                                     ((0, 0), (0, nfbins), (0, 0))), axis=-2),
                          axes=-2)
        vis_ft = vis_ft.reshape(vis_ft.shape[0], -1, 2*nfavg, npol).mean(-2)
    else:
        vis_ft = fftshift(fft(np.pad(vis[..., :nfbins, :],
                                     ((0, 0), (0, 0), (0, nfbins), (0, 0))),
                              axis=-2), axes=-2)
        vis_ft = vis_ft.reshape(vis_ft.shape[0], vis_ft.shape[1], -1, 2*nfavg,
                                npol).mean(-2)
    delay_arr = fftshift(fftfreq(nfbins))/df
    delay_arr = delay_arr.reshape(-1, nfavg).mean(-1)

    return vis_ft, delay_arr

# def get_bad_times(msname, sourcename, refant, tint='59s', combine_spw=False,
#                   nspw=1):
#     r"""Flags bad times in the calibrator data.

#     Calculates delays on short time periods and compares them to the delay
#     calibration solution. Can only be run after delay calibration.

#     Parameters
#     ----------
#     msname : str
#         The name of the measurement set. The MS `msname`.ms will be opened.
#     sourcename : str
#         The name of the calibrator source.  The calibration table will be
#         written to `msname`\_`sourcename`\_kcal.
#     refant : str
#         The reference antenna to use in calibration. If type *str*, the name of
#         the reference antenna, if type *int*, the index of the antenna in the
#         CASA measurement set. This must be the same as the reference antenna
#         used in the delay calibration, or unexpected errors may occur.
#     tint : str
#         The timescale on which to calculate the delay solutions (and evaluate
#         the data quality). Must be a CASA-interpreted string, e.g. ``'inf'``
#         (average all of the data) or ``'60s'`` (average data to 60-second bins
#         before delay calibration).  Defaults to ``'59s'``.
#     combine_spw : bool
#         Set to True if the spws were combined before delay calibration.
#     nspw : int
#         The number of spws in the measurement set.

#     Returns
#     -------
#     bad_times : array
#         Booleans, ``True`` if the data quality is poor and the time-bin should
#         be flagged, ``False`` otherwise.  Same dimensions as times.
#     times : array
#         Floats, the time (mjd) for each delay solution.
#     error : int
#         The number of errors that occured during calibration.
#     """
#     if combine_spw:
#         combine = 'field,scan,obs,spw'
#     else:
#         combine = 'field,scan,obs'
#     error = 0
#     # Solve the calibrator data on minute timescales
#     cb = cc.calibrater()
#     error += not cb.open('{0}.ms'.format(msname))
#     error += not cb.setsolve(type='K', t=tint, refant=refant, combine=combine,
#                              table='{0}_{1}_2kcal'.format(msname, sourcename))
#     error += not cb.solve()
#     error += not cb.close()
#     # Pull the solutions for the entire timerange and the 60-s data from the
#     # measurement set tables
#     antenna_delays, times, _flags, _ant1, _ant2 = read_caltable(
#         '{0}_{1}_2kcal'.format(msname, sourcename), cparam=False)
#     npol = antenna_delays.shape[-1]
#     kcorr, _tkcorr, _flags, _ant1, _ant2 = read_caltable(
#         '{0}_{1}_kcal'.format(msname, sourcename), cparam=False)
#     # Shapes (baseline, time, spw, frequency, pol)
#     # Squeeze on the freqeuncy axis
#     antenna_delays = antenna_delays.squeeze(axis=3)
#     kcorr = kcorr.squeeze(axis=3)
#     nant = antenna_delays.shape[0]
#     nspw = antenna_delays.shape[2]

#     threshold = nant*nspw*npol//2
#     bad_pixels = (np.abs(antenna_delays-kcorr) > 1.5)
#     bad_times = (bad_pixels.reshape(bad_pixels.shape[0],
#                                             bad_pixels.shape[1], -1)
#                          .sum(axis=-1).sum(axis=0) > threshold)
#     # bad_times[:, np.sum(np.sum(bad_times, axis=0), axis=1) > threshold, :] \
#     #    = np.ones((nant, 1, npol))

#     return bad_times, times, error

def apply_calibration(msname, calname, msnamecal=None, combine_spw=False,
                      nspw=1, blbased=False):
    r"""Applies the calibration solution.

    Applies delay, bandpass and complex gain tables to a measurement set.

    Parameters
    ----------
    msname : str
        The name of the measurement set to apply calibration solutions to.
        Opens `msname`.ms
    calname : str
        The name of the calibrator. Tables that start with
        `msnamecal`\_`calname` will be applied to the measurement set.
    msnamecal : str
        The name of the measurement set used to model the calibration solutions
        Calibration tables prefixed with `msnamecal`\_`calname` will be opened
        and applied. If ``None``, `msnamecal` is set to `msname`. Defaults to
        ``None``.
    combine_spw : bool
        Set to True if multi-spw ms and spws in the ms were combined before
        calibration.
    nspw : int
        The number of spws in the ms.
    blbased : bool
        Set to True if the calibration was baseline-based.

    Returns
    -------
    int
        The number of errors that occured during calibration.
    """
    if combine_spw:
        spwmap = [0]*nspw
    else:
        spwmap = [-1]
    if msnamecal is None:
        msnamecal = msname
    caltables = [{'table': '{0}_{1}_kcal'.format(msnamecal, calname),
                  'type': 'K',
                  'spwmap': spwmap},
                 {'table': '{0}_{1}_bcal'.format(msnamecal, calname),
                  'type': 'MF' if blbased else 'B',
                  'spwmap': spwmap},
                 {'table': '{0}_{1}_gacal'.format(msnamecal, calname),
                  'type': 'M' if blbased else 'G',
                  'spwmap': spwmap},
                 {'table': '{0}_{1}_gpcal'.format(msnamecal, calname),
                  'type': 'M' if blbased else 'G',
                  'spwmap': spwmap}]
    error = apply_calibration_tables(msname, caltables)
    return error

def apply_delay_bp_cal(
    msname, calname, blbased=False, msnamecal=None, combine_spw=False, nspw=1
):
    r"""Applies delay and bandpass calibration.

    Parameters
    ----------
    msname : str
        The name of the measurement set containing the visibilities. The
        measurement set `msname`.ms will be opened.
    calname : str
        The name of the calibrator source used in calibration of the
        measurement set. The tables `msname`\_`calname`_kcal and
        `msnamecal`\_`calname`_bcal will be applied to the measurement set.
    blbased : boolean
        Set to True if baseline-based calibration routines were done. Defaults
        False.
    msnamecal : str
        The prefix of the measurement set used to derive the calibration
        solutions. If None, set to `msname`.
    combine_spw : boolean
        Set to True if the spws were combined when deriving the solutions.
        Defaults False.
    nspw : int
        The number of spws in the dataset.  Only used if `combine_spw` is set
        to True. Defaults 1.

    Returns
    -------
    int
        The number of errors that occured during calibration.
    """
    if combine_spw:
        spwmap = [0]*nspw
    else:
        spwmap = [-1]
    if msnamecal is None:
        msnamecal = msname
    error = 0
    caltables = [{'table': '{0}_{1}_kcal'.format(msnamecal, calname),
                  'type': 'K',
                  'spwmap': spwmap},
                 {'table': '{0}_{1}_bcal'.format(msnamecal, calname),
                  'type': 'MF' if blbased else 'B',
                  'spwmap': spwmap}]
    error += apply_and_correct_calibrations(msname, caltables)
    return error


def fill_antenna_gains(gains, flags=None):
    """Fills in the antenna gains for triple-antenna calibration.

    Takes the gains from baseline-based calibration for a trio of antennas and
    calculates the corresponding antenna gains using products of the baseline
    gains. Also propagates flag information for the input baseline gains to
    the antenna gains.

    Parameters
    ----------
    gains : narray
        The complex gains matrix, first dimension is baseline. Indices 1, 2 and
        4 contain the gains for the cross-correlations. Information in indices
        0, 3 and 5 is ignored and overwritten.
    flags : ndarray
        A boolean array, containing flag information for the `gains` array. 1
        if the data is flagged, 0 otherwise. If ``None``, assumes no flag
        information available.  The first dimension is baseline.  Indices 1, 2
        and 4 contain the flags for the cross-correlations.  Information in
        indices 0, 3 and 5 is ignored and overwritten.

    Returns
    -------
    gains : ndarray
        The complex gains matrix, first dimension is baseline. Indices 1, 2 and
        4 contain the gains for the cross-correlations. Indices 0, 3 and 5
        contain the calculated values for the antennas.
    flags : ndarray
        A boolean array, containing flag information for the `gains` array. 1
        if the data is flagged, 0 otherwise. If None, assumes no flag
        information available. The first dimension is baseline.  Indices 1, 2
        and 4 contain the flags for the cross-correlations.  Indices 0,3 and 5
        contain the calculated values for the antennas.
    """
    assert gains.shape[0] == 6, 'Will only calculate antenna gains for trio'
#   use ant1 and ant2 to do this?
#     for i in range(6):
#         if ant1[i] != ant2[i]:
#             idxp = np.where((ant1 == ant1[i]) & (ant2 == ant1[i]))[0][0]
#             idxn = np.where((ant1 == ant2[i]) & (ant2 == ant2[i]))[0][0]
#             idxd = np.where((ant1 == ant2) & (ant1 != ant1[i]) &
#                             (ant1 != ant2[i]))[0][0]
#             gains[i] = np.conjugate(gains[idxn])*gains[idxp]/gains[idxd]
    gains[0] = np.conjugate(gains[1])*gains[2]/gains[4]
    gains[3] = gains[1]*gains[4]/gains[2]
    gains[5] = gains[2]*np.conjugate(gains[4])/gains[1]

    if flags is not None:
        flags[[0, 3, 5], ...] = np.min(np.array([flags[1]+flags[2]+flags[4],
                                                 np.ones(flags[0].shape,
                                                         dtype=int)]), axis=0)
        return gains, flags
    return gains

def calibrate_gain(msname, calname, caltable_prefix, refant, tga, tgp,
                   blbased=False, combined=False):
    """Calculates gain calibration only.

    Uses existing solutions for the delay and bandpass.

    Parameters
    ----------
    msname : str
        The name of the measurement set for gain calibration.
    calname : str
        The name of the calibrator used in calibration.
    caltable_prefix : str
        The prefix of the delay and bandpass tables to be applied.
    refant : str
        The name of the reference antenna.
    tga : str
        A casa-understood integration time for gain amplitude calibration.
    tgp : str
        A casa-understood integration time for gain phase calibration.
    blbased : boolean
        Set to True if using baseline-based calibration for gains. Defaults
        False.
    combined : boolean
        Set to True if spectral windows are combined for calibration.

    Returns
    -------
    int
        The number of errors that occured during calibration.
    """
    if combined:
        spwmap = [0]
    else:
        spwmap = [-1]
    if blbased:
        gtype = 'M'
        bptype = 'MF'
    else:
        gtype = 'G'
        bptype = 'B'
    combine='scan,field,obs'
    caltables = [{'table': '{0}_kcal'.format(caltable_prefix),
                  'type': 'K',
                  'spwmap': spwmap}
                ]
    error = 0
    cb = cc.calibrater()
    error += not cb.open('{0}.ms'.format(msname))
    error += apply_calibration_tables(cb, caltables)
    error += not cb.setsolve(type=bptype, combine=combine,
                             table='{0}_{1}_bcal'.format(msname, calname),
                             minblperant=1, refant=refant)
    error += not cb.solve()
    error += not cb.close()
    caltables += [
        {'table': '{0}_bcal'.format(caltable_prefix),
         'type': bptype,
         'spwmap': spwmap}
    ]
    cb = cc.calibrater()
    error += not cb.open('{0}.ms'.format(msname))
    error += apply_calibration_tables(cb, caltables)
    error += not cb.setsolve(type=gtype, combine=combine,
                             table='{0}_{1}_gpcal'.format(msname, calname),
                             t=tgp, minblperant=1, refant=refant, apmode='p')
    error += not cb.solve()
    error += not cb.close()
    cb = cc.calibrater()
    error += not cb.open('{0}.ms'.format(msname))
    caltables += [{'table': '{0}_{1}_gpcal'.format(msname, calname),
                   'type': gtype,
                   'spwmap': spwmap}]
    error += apply_calibration_tables(cb, caltables)
    error += not cb.setsolve(type=gtype, combine=combine,
                             table='{0}_{1}_gacal'.format(msname, calname),
                             t=tga, minblperant=1, refant=refant, apmode='a')
    error += not cb.solve()
    error += not cb.close()
    return error

def apply_and_correct_calibrations(msname, calibration_tables):
    """Applies and corrects calibration tables in an ms.

    Parameters
    ----------
    msname : str
        The measurement set filepath. Will open `msname`.ms.
    calibration_tables : list
        Calibration tables to apply. Each entry is a dictionary containing the
        keywords 'type' (calibration type, e.g. 'K'), 'spwmap' (spwmap for the
        calibration), and 'table' (full path to the calibration table).

    Returns
    -------
    int
        The number of errors that occured during calibration.
    """
    error = 0
    cb = cc.calibrater()
    error += not cb.open('{0}.ms'.format(msname))
    error += apply_calibration_tables(cb, calibration_tables)
    error += not cb.correct()
    error += not cb.close()
    return error

def apply_calibration_tables(cb, calibration_tables):
    """Applies calibration tables to an open calibrater object.

    Parameters
    ----------
    cb : cc.calibrater() instance
        Measurement set should be opened already.
    calibration_tables : list
        Calibration tables to apply. Each entry is a dictionary containing the
        keywords 'type' (calibration type, e.g. 'K'), 'spwmap' (spwmap for the
        calibration), and 'table' (full path to the calibration table).

    Returns
    -------
    int
        The number of errors that occured during calibration.
    """
    error = 0
    for caltable in calibration_tables:
        error += not cb.setapply(type=caltable['type'],
                                 spwmap=caltable['spwmap'],
                                 table=caltable['table'])
    return error

def interpolate_bandpass_solutions(
    msname, calname, thresh=1.5, polyorder=7, mode='ap'
):
    r"""Interpolates bandpass solutions.

    Parameters
    ----------
    msname : str
        The measurement set filepath (with the `.ms` extension omitted).
    calname : str
        The name of the calibrator source. Calibration tables starting with
        `msname`\_`calname` will be opened.
    thresh : float
        Sets flagging of bandpass solutions before interpolating in order to
        smooth the solutions. After median baselining, any points that deviate
        by more than interp_thresh*std are flagged.
    polyorder : int
        The order of the polynomial used to smooth bandpass solutions.
    mode : str
        The bandpass calibration mode. Must be one of "a", "p" or "ap".
    """
    if mode=='a':
        tbname = 'bacal'
    elif mode=='p':
        tbname = 'bpcal'
    elif mode=='ap':
        tbname = 'bcal'
    else:
        raise RuntimeError('mode must be one of "a", "p" or "ap"')

    with table('{0}_{1}_{2}'.format(msname, calname, tbname)) as tb:
        bpass = np.array(tb.CPARAM[:])
        flags = np.array(tb.FLAG[:])

    with table('{0}.ms'.format(msname)) as tb:
        antennas = np.unique(np.array(tb.ANTENNA1[:]))

    with table('{0}.ms/SPECTRAL_WINDOW'.format(msname)) as tb:
        fobs = np.array(tb.CHAN_FREQ[:]).squeeze(0)/1e9

    bpass_amp = np.abs(bpass)
    bpass_ang = np.angle(bpass)
    bpass_amp_out = np.ones(bpass.shape, dtype=bpass.dtype)
    bpass_ang_out = np.zeros(bpass.shape, dtype=bpass.dtype)

    # Interpolate amplitudes
    if mode in ('a', 'ap'):
        std = bpass_amp.std(axis=1, keepdims=True)
        for ant in antennas:
            for j in range(bpass.shape[-1]):
                offset = np.abs(
                    bpass_amp[ant-1, :, j]-medfilt(
                        bpass_amp[ant-1, :, j], 9
                    )
                )/std[ant-1, :, j]
                idx = offset < thresh
                idx[flags[ant-1, :, j]] = 1
                if sum(idx) > 0:
                    z_fit = np.polyfit(
                        fobs[idx],
                        bpass_amp[ant-1, idx, j],
                        polyorder
                    )
                    p_fit = np.poly1d(z_fit)
                    bpass_amp_out[ant-1, :, j] = p_fit(fobs)

    # Interpolate phase
    if mode in ('p', 'ap'):
        std = bpass_ang.std(axis=1, keepdims=True)
        for ant in antennas:
            for j in range(bpass.shape[-1]):
                offset = np.abs(
                    bpass_ang[ant-1, :, j]-medfilt(
                        bpass_ang[ant-1, :, j], 9
                    )
                )/std[ant-1, :, j]
                idx = offset < thresh
                idx[flags[ant-1, :, j]] = 1
                if sum(idx) > 0:
                    z_fit = np.polyfit(
                        fobs[idx],
                        bpass_ang[ant-1, idx, j],
                        7
                    )
                    p_fit = np.poly1d(z_fit)
                    bpass_ang_out[ant-1, :, j] = p_fit(fobs)

    with table(
        '{0}_{1}_{2}'.format(msname, calname, tbname), readonly=False
    ) as tb:
        tb.putcol('CPARAM', bpass_amp_out*np.exp(1j*bpass_ang_out))
        # Reset flags for the interpolated solutions
        tbflag = np.array(tb.FLAG[:])
        tb.putcol('FLAG', np.zeros(tbflag.shape, tbflag.dtype))

def calibrate_phases(filenames, refant, msdir='/mnt/data/dsa110/calibration/'):
    """Calibrate phases only for a group of calibrator passes.

    Parameters
    ----------
    filenames : dict
        A dictionary containing information on the calibrator passes to be
        calibrated. Same format as dictionary returned by
        dsacalib.utils.get_filenames()
    refant : str
        The reference antenna name to use. If int, will be interpreted as the
        reference antenna index instead.
    msdir : str
        The full path to the measurement set, with the `.ms` extension omitted.
    """
    for date in filenames.keys():
        for cal in filenames[date].keys():
            msname = '{0}/{1}_{2}'.format(msdir, date, cal)
            if os.path.exists('{0}.ms'.format(msname)):
                reset_flags(msname)
                flag_baselines(msname, '2~27m')
                cb = cc.calibrater()
                cb.open('{0}.ms'.format(msname))
                cb.setsolve(
                    type='B',
                    combine='field,scan,obs',
                    table='{0}_{1}_bpcal'.format(msname, cal),
                    refant=refant,
                    apmode='p',
                    t='inf'
                )
                cb.solve()
                cb.close()

def calibrate_phase_single_ms(msname, refant, calname):
    cb = cc.calibrater()
    cb.open('{0}.ms'.format(msname))
    cb.setsolve(
        type='B',
        combine='field,scan,obs',
        table='{0}_{1}_bpcal'.format(msname, calname),
        refant=refant,
        apmode='p',
        t='inf'
    )
    cb.solve()
    cb.close()
