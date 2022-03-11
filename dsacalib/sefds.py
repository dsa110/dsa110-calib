"""Routines for calculating and plotting SEFD measurements.
"""

import glob
import os

import astropy.units as u
import casatools as cc
import matplotlib.pyplot as plt
import numpy as np
from casacore.tables import table
from dsautils import cnf
from scipy.optimize import curve_fit

from dsacalib.calib import apply_calibration_tables, apply_delay_bp_cal
from dsacalib.fringestopping import amplitude_sky_model
from dsacalib.ms_io import extract_vis_from_ms, get_antenna_gains, read_caltable

MYCONF = cnf.Conf()
CALPARAMS = MYCONF.get("cal")
CORRPARAMS = MYCONF.get("corr")
REFANT = CALPARAMS["refant"][0]
POLS = CORRPARAMS["pols_corr"]


def _gauss_offset(xvals, amp, mean, sigma, offset):
    """Calculates the value of a Gaussian at the locations `x`.

    Parameters
    ----------
    xvals : array
        The x values at which to evaluate the Gaussian.
    amp, mean, sigma, offset : float
        Define the Gaussian: amp * exp(-(x-mean)**2/(2 sigma**2)) + offset

    Returns
    -------
    array
        The values of the Gaussian function defined evaluated at xvals.
    """
    return amp * np.exp(-((xvals - mean) ** 2) / (2 * sigma**2)) + offset


def _gauss(xvals, amp, mean, sigma):
    """Calculates the value of a Gaussian at the locations `x`.


    Parameters
    ----------
    xvals : array
        The x values at which to evaluate the Gaussian.
    amp, mean, sigma : float
        Define the Gaussian: amp * exp(-(x-mean)**2/(2 sigma**2))

    Returns
    -------
    array
        The values of the Gaussian function defined evaluated at xvals.
    """
    return _gauss_offset(xvals, amp, mean, sigma, 0.0)


def remove_model(msname):
    """Removes the model from the ms and replaces with ones.

    Parameters
    ----------
    msname : str
        The path to the measurement set, with the extension omitted.
    """
    with table("{0}.ms".format(msname), readonly=False) as tb:
        model = np.array(tb.MODEL_DATA[:])
        if not os.path.exists(f"{msname}.ms/model.npz"):
            np.savez(f"{msname}.ms/model", model)
        tb.putcol("MODEL_DATA", np.ones(model.shape, model.dtype))


def solve_gains(msname, calname, msname_delaycal, calname_delaycal, refant=REFANT):
    """Solves for complex gains on 30s timescales.

    Parameters
    ----------
    msname : str
        The path to the measurement set, with the .ms extension omitted.
    calname : str
        The name of the calibrator.
    msname_delaycal : str
        The path to the measurement set for delay and bandpass calibration,
        with the .ms extension omitted.
    calname_delaycal : str
        The name of the calibrator for `msname_delaycal`.
    """
    caltables = [
        {
            "table": "{0}_{1}_kcal".format(msname_delaycal, calname_delaycal),
            "type": "K",
            "spwmap": [-1],
        },
        {
            "table": "{0}_{1}_bcal".format(msname_delaycal, calname_delaycal),
            "type": "B",
            "spwmap": [-1],
        },
    ]
    cb = cc.calibrater()
    cb.open("{0}.ms".format(msname))
    apply_calibration_tables(cb, caltables)
    cb.setsolve(
        type="G",
        combine="scan, field, obs",
        table="{0}_{1}_2gcal".format(msname, calname),
        t="30s",
        refant=refant,
        apmode="ap",
    )
    cb.solve()


def read_gains(msname, calname, msname_delaycal, calname_delaycal, antenna_order):
    """Reads the gain and bandpass tables and gets the antenna gains.

    Parameters
    ----------
    msname : str
        The path to the measurement set, with the .ms extension omitted.
    calname : str
        The name of the calibrator.
    msname_delaycal : str
        The path to the measurement set for bandpass calibration, with the .ms
        extension omitted.
    calname_delaycal : str
        The name of the calibrator used for the bandpass calibration ms.
    antenna_order : list
        The order of the antennas, base 0.

    Returns
    -------
    time : array
        The times corresponding to the gain measurements.
    gain : ndarray
        The gains, shape (antenna, time, freq, pol).
    """
    # Open the gain files and read in the gains
    # Note that this doesn't have the bandpass gain
    gain, time, flag, ant1, ant2 = read_caltable(
        f"{msname}_{calname}_2gcal", cparam=True
    )
    gain[flag] = np.nan
    bpass, _, flag, ant1, ant2 = read_caltable(
        f"{msname_delaycal}_{calname_delaycal}_bcal", cparam=True
    )
    bpass[flag] = np.nan
    gain = gain * bpass
    antenna, gain = get_antenna_gains(gain, ant1, ant2)
    gain = 1 / gain
    antenna = list(antenna)
    idxs = [antenna.index(ant) for ant in antenna_order]
    gain = gain[idxs, ...]
    gain = np.abs(gain * np.conjugate(gain))
    gain = np.abs(gain).squeeze(axis=2)
    return time, gain


def get_autocorr_gains_off(
    msname,
    tvis,
    antenna_order,
    ant1,
    ant2,
    nspw,
    idxl0,
    idxr0,
    ant_transit_time,
    hwhms,
    nfint,
    npol,
):
    """Calculates the off-source gains from the autocorrs in the ms.

    Parameters
    ----------
    msname : str
        The path to the measurement set, with the .ms extension omitted.
    tvis : array
        The times corresponding to each sample in the ms, in mjd.
    antenna_order : list
        The antennas for which to get autocorr gains.
    ant1, ant2 : list
        The antennas for each baseline in the ms.
    nspw : int
        The number of spws in the ms
    idxl0, idxr0 : int
        The start and stop indices for the requested frequency rang.e
    ant_transit_time : ndarray
        The transit time for each freq, antenna, pol. Dimensions
        (nfint, nant, npol).
    hwhms : ndarray
        The hwhm of transit for each freq, antenna, pol in days. Same
        dimensions as `ant_transit_time`.
    nfint, npol : int
        The number of frequency bins and polarizations.

    Returns
    -------
    ndarray
        The autocorr gains. Same dimensions as `ant_transit_time`.
    """
    nbls = len(ant1)
    nant = len(antenna_order)
    autocorr_gains_off = np.zeros((nfint, nant, 40, npol), dtype=complex)
    with table(f"{msname}.ms") as tb:
        for i, ant in enumerate(antenna_order):
            blidx = np.where((ant1 == ant) & (ant2 == ant))
            anttt = np.median(ant_transit_time[:, i, :])
            anthwhm = np.median(hwhms[:, i, :])
            tidxl = np.searchsorted(tvis, anttt - anthwhm)
            tidxr = np.searchsorted(tvis, anttt + anthwhm)
            tidxs = np.concatenate(
                (tidxl + np.arange(-10, 10), tidxr + np.arange(-10, 10))
            )
            for j, tidx in enumerate(tidxs):
                idx = int(tidx * (nbls * nspw) + blidx * nspw)
                try:
                    tmp = tb.CORRECTED_DATA[idx]
                except IndexError:
                    autocorr_gains_off[:, i, j, :] = np.nan
                    print(f"No data for tidx {tidx}, blidx {blidx}")
                else:
                    tmp = np.median(
                        tmp[idxl0:idxr0, :].reshape(nfint, -1, npol), axis=1
                    )
                    autocorr_gains_off[:, i, j, :] = tmp
    eautocorr_gains_off = np.abs(np.std(autocorr_gains_off, axis=2))
    autocorr_gains_off = np.abs(np.nanmedian(autocorr_gains_off, axis=2))

    return autocorr_gains_off, eautocorr_gains_off


def calculate_sefd(
    msname,
    cal,
    fmin=None,
    fmax=None,
    nfint=1,
    showplots=False,
    msname_delaycal=None,
    calname_delaycal=None,
    repeat=False,
    refant=REFANT,
):
    r"""Calculates the SEFD from a measurement set.

    The measurement set must have been calibrated against a model of ones and
    must include autocorrelations.

    Parameters
    ----------
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
    nfint : int
        The number of frequency channels to split the data into before fitting
        the SEFD.
    showplots : Boolean
        If set to ``True``, plots that show the Gaussian fits to the gains will
        be left open.  Otherwise they are saved. Defaults ``False``.
    msname_delaycal : str
        The name of the measurement set from which delay solutions should be
        applied. Defaults to `msname`.
    calname_delaycal : str
        The name of the calibrator source from which delay solutions should be
        applied. Defaults to `calname`.

    Returns
    -------
    antenna_names : list
        The names of the antennas in their order in `sefds`.
    sefds : ndarray
        The SEFD of each antenna/polarization pair, in Jy. Dimensions (antenna,
        polarization).
    ant_gains : ndarray
        The antenna gains in 1/Jy. Dimensions (antenna, polarization).
    ant_transit_time : ndarray
        The meridian transit time of the source as seen by each antenna/
        polarization pair, in MJD. Dimensions (antenna, polarization).
    fref : float
        The reference frequency of the SEFD measurements in GHz.
    hwhms : float
        The hwhms of the calibrator transits in days.
    """
    # TODO: Change beam shape to something that more closely matches than a gaussian

    if msname_delaycal is None:
        msname_delaycal = msname
    if calname_delaycal is None:
        calname_delaycal = cal.name
    outname = msname.split("/")[-1]
    npol = 2
    pols = ["B", "A"]

    if not repeat:
        # Get rid of the model column in the measurement set
        for msn in sorted(glob.glob(f"{msname}.ms/SUBMSS/*.ms")):
            remove_model(msn[:-3])

        # Solve for complex gains
        solve_gains(msname, cal.name, msname_delaycal, calname_delaycal, refant=refant)

    # Get metadata from the measurement set
    _, tvis, fvis, _, ant1, ant2, pt_dec, spw, orig_shape = extract_vis_from_ms(
        msname, "CORRECTED_DATA", metadataonly=True
    )
    assert orig_shape == ["time", "baseline", "spw"]
    nspw = len(spw)
    antenna_order = ant1[ant1 == ant2]
    nant = len(antenna_order)

    # Figure out the frequencies that we want
    idxl0 = np.searchsorted(fvis, fmin) if fmin is not None else 0
    idxr0 = np.searchsorted(fvis, fmax) if fmax is not None else fvis.shape[0]
    width = idxr0 - idxl0
    assert (
        width % nfint == 0
    ), f"the number of frequency channels ({width}) between fmin ({fmin}) and fmax ({fmax}) must be divisible by nfint ({nfint})"
    fvis = fvis[idxl0:idxr0].reshape(nfint, -1)
    fref = np.median(fvis, axis=-1)
    max_flux = amplitude_sky_model(cal, cal.ra.to_value(u.rad), pt_dec, fref)

    # Read the gains and reshape to the desired frequencies
    time, gain = read_gains(
        msname, cal.name, msname_delaycal, calname_delaycal, antenna_order
    )
    gain = np.nanmean(
        gain[:, :, idxl0:idxr0, :].reshape(nant, len(time), nfint, -1, npol), axis=3
    )

    # Define some things for fitting the gains + plotting
    ant_gains_on = np.zeros((nfint, nant, npol))
    eant_gains_on = np.zeros((nfint, nant, npol))
    ant_transit_time = np.zeros((nfint, nant, npol))
    eant_transit_time = np.zeros((nfint, nant, npol))
    ant_transit_width = np.zeros((nfint, nant, npol))
    eant_transit_width = np.zeros((nfint, nant, npol))
    x = time - time[0]
    med = np.median(x)
    nx = 3
    ny = nant // nx
    if nant % nx != 0:
        ny += 1

    # Fit the gains on a frequency x antenna x pol basis
    # One figure is made and shown or saved per frequency
    for ifreq in range(nfint):

        fig, ax = plt.subplots(ny, nx, figsize=(8 * nx, 8 * ny), sharey=True)
        ccyc = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        ax = ax.flatten()

        # Fit a Gaussian to the gains for each freq/ant/pol
        # TODO: scale cov matrix more appropriately
        for i in range(nant):
            for j in range(npol):
                ax[i].plot(time - time[0], gain[i, :, ifreq, j], ".", color=ccyc[j])
                initial_params = [np.max(gain[i, :, ifreq, j]), med, 0.0035]  # , 0]
                try:
                    y = gain[i, :, ifreq, j]
                    idx = ~np.isnan(y)
                    assert len(idx) >= 4
                    params, cov = curve_fit(_gauss, x[idx], y[idx], p0=initial_params)
                    ypred = _gauss(x[idx], *params)
                    sigma = np.std(ypred - y[idx])
                    err = sigma * np.sqrt(cov)
                except (RuntimeError, ValueError, AssertionError):
                    params = initial_params.copy()
                    err = np.ones((len(params), len(params))) * np.inf

                # Save fit for calculating the sefd later
                ant_gains_on[ifreq, i, j] = params[0]
                eant_gains_on[ifreq, i, j] = err[0, 0]
                ant_transit_time[ifreq, i, j] = time[0] + params[1]
                eant_transit_time[ifreq, i, j] = err[1, 1]
                ant_transit_width[ifreq, i, j] = params[2]
                eant_transit_width[ifreq, i, j] = err[2, 2]

                xmb = (time - time[0]) - params[1]
                earg = -(xmb**2) / (2 * params[2]) ** 2

                ypred = _gauss(time - time[0], *params)
                eypred = np.sqrt(
                    (err[0, 0] * np.exp(earg)) ** 2
                    + (err[1, 1] * params[0] * np.exp(earg) * xmb / params[2] ** 2) ** 2
                    + (err[2, 2] * params[0] * np.exp(earg) * xmb**2 / params[2] ** 3)
                    ** 2
                )

                ax[i].fill_between(
                    time - time[0],
                    ypred - eypred,
                    ypred + eypred,
                    color=ccyc[j],
                    alpha=0.4,
                )
                ax[i].plot(
                    time - time[0],
                    ypred,
                    "-",
                    color=ccyc[j],
                    label=f"{antenna_order[i]+1} {pols[j]} {fref[ifreq]}",
                )

                ax[i].legend()
                ax[i].set_xlabel("Time (d)")
                ax[i].set_ylabel("Unnormalized power")
                max_gain = np.nanmax(ant_gains_on[ifreq, ...])
                ax[0].set_ylim(-0.1 * max_gain, 1.1 * max_gain)

        if not showplots:
            plt.savefig(f"{outname}_{fref[ifreq]}GHz_of_{nfint}_sefds.png")
            plt.close(fig)

    # Calculate the gains, and hwhms
    ant_gains = ant_gains_on / max_flux[:, np.newaxis, np.newaxis]
    eant_gains = eant_gains_on / max_flux[:, np.newaxis, np.newaxis]
    hwhms = np.sqrt(2 * np.log(2)) * ant_transit_width
    ehwhms = np.sqrt(2 * np.log(2)) * eant_transit_width

    # Get the autocorr gains at the half power point
    if not repeat:
        apply_delay_bp_cal(msname, calname_delaycal, msnamecal=msname_delaycal)

    autocorr_gains_off, eautocorr_gains_off = get_autocorr_gains_off(
        msname,
        tvis,
        antenna_order,
        ant1,
        ant2,
        nspw,
        idxl0,
        idxr0,
        ant_transit_time,
        hwhms,
        nfint,
        npol,
    )

    # Calculate the sefd
    sefds = autocorr_gains_off / ant_gains - max_flux[:, np.newaxis, np.newaxis] / 2
    esefds = sefds * np.sqrt(
        (eautocorr_gains_off / autocorr_gains_off) ** 2 + (eant_gains / ant_gains) ** 2
    )

    # Save the sefds so we can read them in later
    np.savez(
        f"{outname}_nfint{nfint}_sefds",
        antennas=antenna_order + 1,
        sefds_Jy=sefds,
        esefds_Jy=esefds,
        gains=ant_gains,
        egains=eant_gains,
        transits=ant_transit_time,
        etranstis=eant_transit_time,
        freqs_GHz=fref,
        hwhms_days=hwhms,
        ehwhms_days=ehwhms,
    )

    return (
        antenna_order + 1,
        sefds,
        esefds,
        ant_gains,
        eant_gains,
        ant_transit_time,
        eant_transit_time,
        fref,
        hwhms,
        ehwhms,
    )


def plot_sefds(antennas, sefds, esefds, fref, ymax=None):
    """Plots the sefds.

    Parameters
    ----------
    antennas : list
        The antennas for which you have sefds.
    sefds : ndarray
        The sefds for each fref x antenna x pol.
    fref : array
        The frequencies for which you have sefds.

    Returns
    -------
    figure object
    """
    ccyc = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    lcyc = ["-", ":"]
    fig, ax = plt.subplots(4, 2, figsize=(10, 4 * 4))
    if ymax is None:
        ymax = np.nanmax(sefds) * 1.1
    for axi in ax[-1, :]:
        axi.set_xlabel("freq (GHz)")
    for axi in ax[:, 0]:
        axi.set_ylabel("SEFD (Jy)")
    ax = ax.flatten()
    for i in range(64):
        for j in range(2):
            ax[i // 10].plot(
                fref,
                sefds[:, i, j],
                label=antennas[i] if j == 0 else None,
                color=ccyc[i % 10],
                ls=lcyc[j],
            )
            ax[i // 10].fill_between(
                fref,
                sefds[:, i, j] - esefds[:, i, j],
                sefds[:, i, j] + esefds[:, i, j],
                color=ccyc[i % 10],
                alpha=0.3,
            )
    for i in range(7):
        ax[i].legend()
        ax[i].set_ylim(0, ymax)
    for axi in ax:
        x0, x1 = axi.get_xlim()
        y0, y1 = axi.get_ylim()
        axi.fill_between([x0, x1], [10200, 10200], [y1, y1], color="lightgrey")
    return fig
