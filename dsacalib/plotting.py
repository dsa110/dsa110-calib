"""Visibility and calibration solution plotting routines.

Plotting routines to visualize the visibilities and
the calibration solutions for DSA-110.

Author: Dana Simard, dana.simard@astro.caltech.edu, 10/2019
"""
import glob
import os

import astropy.units as u
import h5py
import matplotlib.pyplot as plt
import numpy as np
from casacore.tables import table
from dsautils import cnf

import dsacalib.constants as ct
from dsacalib.ms_io import extract_vis_from_ms, read_caltable

CONF = cnf.Conf()


def plot_dyn_spec(
    vis, fobs, mjd, bname, normalize=False, outname=None, show=True, nx=None
):
    """Plots the dynamic spectrum of the real part of the visibility.

    Parameters
    ----------
    vis : ndarray
        The visibilities to plot.  Dimensions (baseline, time, freq,
        polarization).
    fobs : ndarray
        The center frequency of each channel in GHz.
    mjd : ndarray
        The center of each subintegration in MJD.
    bname : list
        The name of each baseline in the visibilities.
    normalize : boolean
        If set to ``True``, the visibilities are normalized before plotting.
        Defaults to ``False``.
    outname : str
        If provided and not set to ``None``, the plot will be saved to the file
        `outname`_dynspec.png Defaults to ``None``.
    show : boolean
        If set to ``False`` the plot will be closed after rendering. If set to
        ``True`` the plot is left open. Defaults to ``True``.
    nx : int
        The number of subplots in the horizontal direction of the figure. If
        not provided, or set to ``None``, `nx` is set to the minimum of 5 and
        the number of baselines in the visibilities.  Defaults to ``None``.
    """
    (nbl, nt, nf, npol) = vis.shape
    if nx is None:
        nx = min(nbl, 5)
    ny = (nbl * 2) // nx
    if (nbl * 2) % nx != 0:
        ny += 1

    _, ax = plt.subplots(ny, nx, figsize=(8 * nx, 8 * ny))
    ax = ax.flatten()

    if len(mjd) > 125:
        dplot = np.nanmean(
            vis[:, : nt // 125 * 125, ...].reshape(nbl, 125, -1, nf, npol), 2
        )
    else:
        dplot = vis.copy()
    if len(fobs) > 125:
        dplot = np.nanmean(
            dplot[:, :, : nf // 125 * 125, :].reshape(
                nbl, dplot.shape[1], 125, -1, npol
            ),
            3,
        )
    dplot = dplot.real
    dplot = dplot / (
        dplot.reshape(nbl, -1, npol).mean(axis=1)[:, np.newaxis, np.newaxis, :]
    )

    if normalize:
        dplot = dplot / np.abs(dplot)
        vmin = -1
        vmax = 1
    else:
        vmin = -100
        vmax = 100
    dplot = dplot - 1

    if len(mjd) > 125:
        x = mjd[: mjd.shape[0] // 125 * 125].reshape(125, -1).mean(-1)
    else:
        x = mjd
    x = ((x - x[0]) * u.d).to_value(u.min)
    if len(fobs) > 125:
        y = fobs[: nf // 125 * 125].reshape(125, -1).mean(-1)
    else:
        y = fobs
    for i in range(nbl):
        for j in range(npol):
            ax[j * nbl + i].imshow(
                dplot[i, :, :, j].T,
                origin="lower",
                interpolation="none",
                aspect="auto",
                vmin=vmin,
                vmax=vmax,
                extent=[x[0], x[-1], y[0], y[-1]],
            )
            ax[j * nbl + i].text(
                0.1,
                0.9,
                f"{bname[i]}, pol {'B' if j else 'A'}",
                transform=ax[j * nbl + i].transAxes,
                size=22,
                color="white",
            )
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    for i in range((ny - 1) * nx, ny * nx):
        ax[i].set_xlabel("time (min)")
    for i in np.arange(ny) * nx:
        ax[i].set_ylabel("freq (GHz)")
    if outname is not None:
        plt.savefig(f"{outname}_dynspec.png")
    if not show:
        plt.close()


def plot_vis_freq(vis, fobs, bname, outname=None, show=True, nx=None):
    r"""Plots visibilities against frequency.

    Creates plots of the amplitude and phase of the visibilities `vis` as a
    function of frequency `fobs`. Two separate figures are opened, one for the
    amplitude and one for the phases.  If `outname` is passed, these are saved
    as `outname`\_amp_freq.png and `outname`\_phase_freq.png

    Parameters
    ----------
    vis : ndarray
        The visibilities to plot.  Dimensions (baseline, time, freq,
        polarization).
    fobs : ndarray
        The center frequency of each channel in GHz.
    bname : list
        The name of each baseline in the visibilities.
    outname : str
        If provided and not set to ``None``, the plots will be saved to the
        files `outname`\_amp_freq.png and `outname`\_phase_freq.png Defaults to
        ``None``.
    show : boolean
        If set to ``False`` the plot will be closed after rendering.  If set to
        ``True`` the plot is left open. Defaults to ``True``.
    nx : int
        The number of subplots in the horizontal direction of the figure.  If
        not provided, or set to ``None``, `nx` is set to the minimum of 5 and
        the number of baselines in the visibilities.  Defaults to ``None``.
    """
    nbl = vis.shape[0]
    if nx is None:
        nx = min(nbl, 5)
    ny = nbl // nx
    if nbl % nx != 0:
        ny += 1

    dplot = vis.mean(1)
    x = fobs

    _, ax = plt.subplots(ny, nx, figsize=(8 * nx, 8 * ny))
    ax = ax.flatten()
    for i in range(nbl):
        ax[i].plot(x, np.abs(dplot[i, :, 0]), label="A")
        ax[i].plot(x, np.abs(dplot[i, :, 1]), label="B")
        ax[i].text(0.1, 0.9, f"{bname[i]}: amp", transform=ax[i].transAxes, size=22)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    ax[0].legend()
    for i in range((ny - 1) * nx, ny * nx):
        ax[i].set_xlabel("freq (GHz)")
    if outname is not None:
        plt.savefig(f"{outname}_amp_freq.png")
    if not show:
        plt.close()

    _, ax = plt.subplots(ny, nx, figsize=(8 * nx, 8 * ny))
    ax = ax.flatten()
    for i in range(nbl):
        ax[i].plot(x, np.angle(dplot[i, :, 0]), label="A")
        ax[i].plot(x, np.angle(dplot[i, :, 1]), label="B")
        ax[i].text(0.1, 0.9, f"{bname[i]}: phase", transform=ax[i].transAxes, size=22)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    ax[0].legend()
    for i in range((ny - 1) * nx, ny * nx):
        ax[i].set_xlabel("freq (GHz)")
    if outname is not None:
        plt.savefig(f"{outname}_phase_freq.png")
    if not show:
        plt.close()


def plot_vis_time(vis, mjd, bname, outname=None, show=True, nx=None):
    r"""Plots visibilities against time of observation.

    Creates plots of the amplitude and phase of the visibilities `vis` as the
    time of observation `mjd`. Two separate figures are opened, one for the
    amplitude and one for the phases.  If `outname` is passed, these are saved
    as `outname`\_amp_time.png and `outname`\_phase_time.png

    Parameters
    ----------
    vis : ndarray
        The visibilities to plot.  Dimensions (baseline, time, freq,
        polarization).
    mjd : ndarray
        The center of each subintegration in MJD.
    bname : list
        The name of each baseline in the visibilities.
    outname : str
        If provided and not set to ``None``, the plot will be saved to the file
        `outname`\_dynspec.png Defaults to ``None``.
    show : boolean
        If set to ``False`` the plot will be closed after rendering.  If set to
        ``True`` the plot is left open. Defaults to ``True``.
    nx : int
        The number of subplots in the horizontal direction of the figure.  If
        not provided, or set to ``None``, `nx` is set to the minimum of 5 and
        the number of baselines in the visibilities.  Defaults to ``None``.
    """
    nbl = vis.shape[0]
    if nx is None:
        nx = min(nbl, 5)
    ny = nbl // nx

    if nbl % nx != 0:
        ny += 1
    dplot = vis.mean(-2)
    x = ((mjd - mjd[0]) * u.d).to_value(u.min)

    _, ax = plt.subplots(ny, nx, figsize=(8 * nx, 8 * ny))
    ax = ax.flatten()
    for i in range(nbl):
        ax[i].plot(x, np.abs(dplot[i, :, 0]), label="A")
        ax[i].plot(x, np.abs(dplot[i, :, 1]), label="B")
        ax[i].text(0.1, 0.9, f"{bname[i]}: amp", transform=ax[i].transAxes, size=22)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    ax[0].legend()
    for i in range((ny - 1) * nx, ny):
        ax[i].set_xlabel("time (min)")
    if outname is not None:
        plt.savefig(f"{outname}_amp_time.png")
    if not show:
        plt.close()

    _, ax = plt.subplots(ny, nx, figsize=(8 * nx, 8 * ny))
    ax = ax.flatten()
    for i in range(nbl):
        ax[i].plot(x, np.angle(dplot[i, :, 0]), label="A")
        ax[i].plot(x, np.angle(dplot[i, :, 1]), label="B")
        ax[i].text(0.1, 0.9, f"{bname[i]}: phase", transform=ax[i].transAxes, size=22)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    ax[0].legend()
    for i in range((ny - 1) * nx, ny):
        ax[i].set_xlabel("time (min)")
    if outname is not None:
        plt.savefig(f"{outname}_phase_time.png")
    if not show:
        plt.close()


def plot_uv_track(bu, bv, outname=None, show=True):
    """Plots the uv track provided.

    Parameters
    ----------
    bu : ndarray
        The u-coordinates of the baselines in meters. Dimensions (baselines,
        time).
    bv : ndarray
        The v-coordinates of the baselines in meters. Dimensions (baselines,
        time).
    outname : str
        If provided and not set to ``None``, the plot will be saved to the file
        `outname`_dynspec.png Defaults to ``None``.
    show : boolean
        If set to ``False`` the plot will be closed after rendering.  If set to
        ``True`` the plot is left open. Defaults to ``True``.
    """
    _, ax = plt.subplots(1, 1, figsize=(8, 8))
    for i in range(bu.shape[0]):
        ax.plot(bu[i, :], bv[i, :])
        ax.set_xlim(-1500, 1500)
        ax.set_ylim(-1500, 1500)
        ax.text(-1200, 1200, "UV Coverage")
        ax.set_xlabel("$u$ (m)")
        ax.set_ylabel("$v$ (m)")
    if outname is not None:
        plt.savefig(f"{outname}_uv.png")
    if not show:
        plt.close()


def rebin_vis(arr, nb1, nb2):
    """Rebins a 2-D array for plotting.

    Excess bins along either axis are discarded.

    Parameters
    ----------
    arr : ndarray
        The two-dimensional array to rebin.
    nb1 : int
        The number of bins to rebin by along axis 0.
    nb2 : int
        The number of bins to rebin by along the axis 1.

    Returns
    -------
    arr: ndarray
        The rebinned array.
    """
    arr = (
        arr[: arr.shape[0] // nb1 * nb1, : arr.shape[1] // nb2 * nb2]
        .reshape(-1, nb1, arr.shape[1])
        .mean(1)
    )
    arr = arr.reshape(arr.shape[0], -1, nb2).mean(-1)
    return arr


def plot_calibrated_vis(vis, vis_cal, mjd, fobs, bidx, pol, outname=None, show=True):
    """Plots the calibrated and uncalibrated visibilities for comparison.

    Parameters
    ----------
    vis : ndarray
        The complex, uncalibrated visibilities, with dimensions
        (baseline, time, freq, pol).
    vis_cal : ndarray
        The complex calibrated visibilities, with dimensions
        (baseline, time, freq, pol).
    mjd : ndarray
        The midpoint time of each subintegration in MJD.
    fobs : ndarray
        The center frequency of each channel in GHz.
    bidx : int
        The index along the baseline dimension to plot.
    pol : int
        The index along the polarization dimension to plot.
    outname : str
        The base to use for the name of the png file the plot is saved to.  The
        plot will be saved to `outname`_cal_vis.png if `outname` is provided,
        otherwise no plot will be saved.  Defaults ``None``.
    show : boolean
        If `show` is passed ``False``, the plot will be closed after being
        generated.  Otherwise, it is left open.  Defaults ``True``.
    """
    x = mjd[: mjd.shape[0] // 128 * 128].reshape(-1, 128).mean(-1)
    x = ((x - x[0]) * u.d).to_value(u.min)
    y = fobs[: fobs.shape[0] // 5 * 5].reshape(-1, 5).mean(-1)

    _, ax = plt.subplots(2, 2, figsize=(16, 16), sharex=True, sharey=True)

    vplot = rebin_vis(vis[bidx, ..., pol], 128, 5).T
    ax[0, 0].imshow(
        vplot.real,
        interpolation="none",
        origin="lower",
        aspect="auto",
        vmin=-1,
        vmax=1,
        extent=[x[0], x[-1], y[0], y[-1]],
    )
    ax[0, 0].text(
        0.1,
        0.9,
        "Before cal, real",
        transform=ax[0, 0].transAxes,
        size=22,
        color="white",
    )
    ax[1, 0].imshow(
        vplot.imag,
        interpolation="none",
        origin="lower",
        aspect="auto",
        vmin=-1,
        vmax=1,
        extent=[x[0], x[-1], y[0], y[-1]],
    )
    ax[1, 0].text(
        0.1,
        0.9,
        "Before cal, imag",
        transform=ax[1, 0].transAxes,
        size=22,
        color="white",
    )

    vplot = rebin_vis(vis_cal[bidx, ..., pol], 128, 5).T
    ax[0, 1].imshow(
        vplot.real,
        interpolation="none",
        origin="lower",
        aspect="auto",
        vmin=-1,
        vmax=1,
        extent=[x[0], x[-1], y[0], y[-1]],
    )
    ax[0, 1].text(
        0.1,
        0.9,
        "After cal, real",
        transform=ax[0, 1].transAxes,
        size=22,
        color="white",
    )
    ax[1, 1].imshow(
        vplot.imag,
        interpolation="none",
        origin="lower",
        aspect="auto",
        vmin=-1,
        vmax=1,
        extent=[x[0], x[-1], y[0], y[-1]],
    )
    ax[1, 1].text(
        0.1,
        0.9,
        "After cal, imag",
        transform=ax[1, 1].transAxes,
        size=22,
        color="white",
    )
    for i in range(2):
        ax[1, i].set_xlabel("time (min)")
        ax[i, 0].set_ylabel("freq (GHz)")
    plt.subplots_adjust(hspace=0, wspace=0)
    if outname is not None:
        plt.savefig(f"{outname}_{'B' if pol else 'A'}_cal_vis.png")
    if not show:
        plt.close()


def plot_delays(vis_ft, labels, delay_arr, bname, nx=None, outname=None, show=True):
    """Plots the visibility amplitude against delay.

    For a given visibility Fourier transformed along the frequency axis,
    plots the amplitude against delay and calculates the location of the
    fringe peak in delay.  Returns the peak delay for each visibility,
    which can be used to calculate the antenna delays.

    Parameters
    ----------
    vis_ft : ndarray
        The complex visibilities, dimensions (visibility type, baselines,
        delay). Note that the visibilities must have been Fourier transformed
        along the frequency axis and scrunched along the time axis before being
        passed to `plot_delays`.
    labels : list
        The labels of the types of visibilities passed.  For example,
        ``['uncalibrated', 'calibrated']``.
    delay_arr : ndarray
        The center of the delay bins in nanoseconds.
    bname : list
        The baseline labels.
    nx : int
        The number of plots to tile along the horizontal axis. If `nx` is
        given a value of ``None``, this is set to the number of baselines or 5,
        if there are more than 5 baselines.
    outname : str
        The base to use for the name of the png file the plot is saved to. The
        plot will be saved to `outname`_delays.png if an outname is provided.
        If `outname` is given a value of ``None``, no plot will be saved.
        Defaults ``None``.
    show : boolean
        If `show` is given a value of ``False`` the plot will be closed.
        Otherwise, the plot will be left open. Defaults ``True``.

    Returns
    -------
    delays : ndarray
        The peak delay for each visibility, in nanoseconds.
    """
    nvis = vis_ft.shape[0]
    nbl = vis_ft.shape[1]
    npol = vis_ft.shape[-1]
    if nx is None:
        nx = min(nbl, 5)
    ny = nbl // nx
    if nbl % nx != 0:
        ny += 1

    alpha = 0.5 if nvis > 2 else 1
    delays = delay_arr[np.argmax(np.abs(vis_ft), axis=2)]
    # could use scipy.signal.find_peaks instead
    for pidx in range(npol):
        _, ax = plt.subplots(ny, nx, figsize=(8 * nx, 8 * ny), sharex=True)
        ax = ax.flatten()
        for i in range(nbl):
            ax[i].axvline(0, color="black")
            for j in range(nvis):
                ax[i].plot(
                    delay_arr,
                    np.log10(np.abs(vis_ft[j, i, :, pidx])),
                    label=labels[j],
                    alpha=alpha,
                )
                ax[i].axvline(delays[j, i, pidx], color="red")
            ax[i].text(
                0.1,
                0.9,
                f"{bname[i]}: {'B' if pidx else 'A'}",
                transform=ax[i].transAxes,
                size=22,
            )
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        ax[0].legend()
        for i in range((ny - 1) * nx, ny * nx):
            ax[i].set_xlabel("delay (ns)")
        if outname is not None:
            plt.savefig(
                f"{outname}_{'B' if pidx else 'A'}_delays.png",
                bbox_inches="tight",
            )
        if not show:
            plt.close()

    return delays


def plot_antenna_delays(msname, calname, plabels=None, outname=None, show=True):
    r"""Plots antenna delay variations between two delay calibrations.

    Compares the antenna delays used in the calibration solution on the
    timescale of the entire calibrator pass (assumed to be in a CASA table
    ending in 'kcal') to those calculated on a shorter (e.g. 60s) timescale
    (assumed to be in a CASA table ending in '2kcal').

    Parameters
    ----------
    msname : str
        The prefix of the measurement set (`msname`.ms), used to identify the
        correct delay calibration tables.
    calname : str
        The calibrator source name, used to identify the correct delay
        calibration tables.  The tables `msname`\_`calname`\_kcal (for a single
        delay calculated using the entire calibrator pass) and
        `msname`\_`calname`\_2kcal (for delays calculated on a shorter
        timescale) will be opened.
    antenna_order : ndarray
        The antenna names in order.
    outname : str
        The base to use for the name of the png file the plot is saved to.  The
        plot will be saved to `outname`\_antdelays.png.  If `outname` is set to
        ``None``, the plot is not saved.  Defaults ``None``.
    plabels : list
        The labels along the polarization axis.  Defaults to ['B', 'A'].
    outname : string
        The base to use for the name of the png file the plot is saved to. The
        plot will be saved to `outname`_antdelays.png if an outname is provided.
        If `outname` is ``None``, the image is not saved.  Defaults ``None``.
    show : boolean
        If set to ``False`` the plot will be closed after it is generated.
        Defaults ``True``.

    Returns
    -------
    times : ndarray
        The times at which the antenna delays (on short timescales) were
        calculated, in MJD.
    antenna_delays : ndarray
        The delays of the antennas on short timescales, in nanoseconds.
        Dimensions (polarization, time, antenna).
    kcorr : ndarray
        The delay correction calculated using the entire observation time, in
        nanoseconds.  Dimensions (polarization, 1, antenna).
    antenna_order : list
        The antenna indices.
    """
    if plabels is None:
        plabels = ["B", "A"]

    ccyc = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Pull the solutions for the entire timerange and the
    # 60-s data from the measurement set tables
    antenna_delays, times, _flags, _ant1, _ant2 = read_caltable(
        f"{msname}_{calname}_2kcal", cparam=False
    )
    npol = antenna_delays.shape[-1]
    kcorr, _tkcorr, _flags, antenna_order, _ant2 = read_caltable(
        f"{msname}_{calname}_kcal", cparam=False
    )
    nant = len(antenna_order)

    val_to_plot = (antenna_delays - kcorr).squeeze(axis=3).mean(axis=2)
    mean_along_time = np.abs(val_to_plot.reshape(nant, -1, npol).mean(1))
    idx_to_plot = np.where(
        (mean_along_time[..., 0] > 1e-10) | (mean_along_time[..., 1] > 1e-10)
    )[0]
    tplot = (times - times[0]) * ct.SECONDS_PER_DAY / 60.0
    ny = max(len(idx_to_plot) // 10 + 1, 2)
    _, ax = plt.subplots(ny, 1, figsize=(10, 8 * ny))
    lcyc = [".", "x"]
    for i, bidx in enumerate(idx_to_plot):
        for j in range(npol):
            ax[i // 10].plot(
                tplot,
                val_to_plot[bidx, :, j],
                marker=lcyc[j % len(lcyc)],
                label=f"{antenna_order[bidx]+1} {plabels[j]}",
                alpha=0.5,
                color=ccyc[i % len(ccyc)],
            )
    for i in range(ny):
        ax[i].set_ylim(-5, 5)
        ax[i].set_ylabel("delay (ns)")
        ax[i].legend(
            ncol=len(idx_to_plot) // 15 + 1,
            fontsize="small",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )
        ax[i].set_xlabel("time (min)")
        ax[i].axhline(1.5)
        ax[i].axhline(-1.5)
    if outname is not None:
        plt.savefig(f"{outname}_antdelays.png", bbox_inches="tight")
    if not show:
        plt.close()

    return times, antenna_delays, kcorr, antenna_order


def plot_gain_calibration(msname, calname, plabels=None, outname=None, show=True):
    r"""Plots the gain calibration solutions from the gacal and gpcal tables.

    Parameters
    ----------
    msname : str
        The name of the measurement set.  Used to identify the gain calibration
        tables.
    calname : str
        The calibrator used in gain calibration.  Used to identify the gain
        calibration tables.  The tables `msname`\_`calname`\_gacal (assumed to
        contain the gain amplitude soutions) and `msname`\_`calname`\_gpcal
        (assumed to contain the phase amplitude solutions) will be opened.
    plabels : list
        The names of the polarizations in the calibration solutions. Defaults
        to ``['A', 'B']``.
    outname : str
        The base to use for the name of the png file the plot is saved to. The
        plot is saved to `outname`\_gaincal.png if `outname` is not ``None``.
        If `outname` is set to ``None, the plot is not saved.  Defaults
        ``None``.
    show : boolean
        If set to ``False`` the plot is closed after being generated.

    Returns
    -------
    time_days : ndarray
        The times of gain calibration, in days.
    gain_amp : ndarray
        The gain amplitudes. Dimensions (antenna or baseline, time, pol).
    gain_phase : ndarray
        The gain phases. Dimensions (antenna or baseline, time, pol).
    labels : ndarray
        The labels of the antennas or baselines along the 0th axis of the gain
        solutions.
    """
    if plabels is None:
        plabels = ["A", "B"]

    gain_phase, time_phase, _flags, ant1, ant2 = read_caltable(
        f"{msname}_{calname}_gpcal", cparam=True
    )
    gain_phase = gain_phase.squeeze(axis=3)
    gain_phase = gain_phase.mean(axis=2)
    npol = gain_phase.shape[-1]
    if np.all(ant2 == ant2[0]):
        labels = ant1
    else:
        labels = np.array([ant1, ant2]).T
    nlab = labels.shape[0]

    gain_amp, time, _flags, _ant1, _ant2 = read_caltable(
        f"{msname}_{calname}_gacal", cparam=True
    )
    gain_amp = gain_amp.squeeze(axis=3)
    gain_amp = gain_amp.mean(axis=2)
    time_days = time.copy()
    t0 = time[0]
    time = ((time - t0) * u.d).to_value(u.min)
    time_phase = ((time_phase - t0) * u.d).to_value(u.min)

    idx_to_plot = np.where(np.abs(gain_amp.reshape(nlab, -1).mean(1) - 1) > 1e-10)[0]

    ccyc = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    lcyc = ["-", ":"]
    _, ax = plt.subplots(1, 2, figsize=(16, 6), sharex=True)

    if gain_amp.shape[1] > 1:
        tplot = time
        gplot = gain_amp
    else:
        tplot = [0, 1]
        gplot = np.tile(gain_amp, [1, 2, 1])

    for i, bidx in enumerate(idx_to_plot):
        for pidx in range(npol):
            ax[0].plot(
                tplot,
                np.abs(gplot[bidx, :, pidx]),
                label=f"{labels[bidx] + 1} {plabels[pidx]}",
                color=ccyc[i % len(ccyc)],
                ls=lcyc[pidx],
            )

    if gain_phase.shape[1] > 1:
        tplot = time_phase
        gplot = gain_phase
    else:
        tplot = [tplot[0], tplot[-1]]  # use the time from the gains
        gplot = np.tile(gain_phase, [1, 2, 1])

    for i, bidx in enumerate(idx_to_plot):
        for pidx in range(npol):
            ax[1].plot(
                tplot,
                np.angle(gplot[bidx, :, pidx]),
                label=f"{labels[bidx] + 1} {plabels[pidx]}",
                color=ccyc[i % len(ccyc)],
                ls=lcyc[pidx],
            )

    ax[0].set_xlim(tplot[0], tplot[-1])
    ax[1].set_ylim(-np.pi, np.pi)
    ax[0].legend(
        ncol=10, fontsize="x-small", bbox_to_anchor=(0.05, -0.1), loc="upper left"
    )
    ax[0].set_xlabel("time (min)")
    ax[1].set_xlabel("time (min)")
    ax[0].set_ylabel("Abs of gain")
    ax[1].set_ylabel("Phase of gain")
    if outname is not None:
        plt.savefig(f"{outname}_gaincal.png", bbox_inches="tight")
    if not show:
        plt.close()
    return time_days, gain_amp, gain_phase, labels


def plot_bandpass(msname, calname, plabels=None, outname=None, show=True):
    r"""Plots the bandpass calibration solutions in the bcal table.

    Parameters
    ----------
    msname : str
        The name of the measurement set. Used to identify the calibration table
        to plot.
    calname : str
        The name of the calibrator used in calibration. The calibration table
        `msname`\_`calname`\_bcal is opened.
    plabels : list
        The labels for the polarizations.  Defaults ``['A', 'B']``.
    outname : str
        The base to use for the name of the png file the plot is saved to. The
        plot is saved to `outname`\_bandpass.png if an `outname` is not set to
        ``None``. If `outname` is set to ``None``, the plot is not saved.
        Defaults ``None``.
    show : boolean
        If set to ``False``, the plot is closed after it is generated.

    Returns
    -------
    bpass : ndarray
        The bandpass solutions, dimensions (antenna or baseline, frequency).
    fobs : array
        The frequencies at which the bandpass solutions are calculated, in GHz.
    labels : ndarray
        The antenna or baseline labels along the 0th axis of the bandpass
        solutions.
    """
    if plabels is None:
        plabels = ["A", "B"]

    bpass, _tbpass, _flags, ant1, ant2 = read_caltable(
        f"{msname}_{calname}_bcal", cparam=True
    )
    # squeeze along the time axis
    # baseline, time, spw, frequency, pol
    bpass = bpass.squeeze(axis=1)
    bpass = bpass.reshape(bpass.shape[0], -1, bpass.shape[-1])
    npol = bpass.shape[-1]

    with table(f"{msname}.ms/SPECTRAL_WINDOW") as tb:
        fobs = (np.array(tb.col("CHAN_FREQ")[:]) / 1e9).reshape(-1)

    if bpass.shape[1] != fobs.shape[0]:
        nint = fobs.shape[0] // bpass.shape[1]
        fobs_plot = (
            np.mean(fobs[:nint])
            + np.arange(bpass.shape[1]) * np.median(np.diff(fobs)) * nint
        )
    else:
        fobs_plot = fobs.copy()

    if np.all(ant2 == ant2[0]):
        labels = ant1 + 1
    else:
        labels = np.array([ant1 + 1, ant2 + 1]).T
    nant = len(ant1)

    idxs_to_plot = np.where(np.abs(np.abs(bpass).reshape(nant, -1).mean(1) - 1) > 1e-5)[
        0
    ]

    ccyc = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    lcyc = ["-", ":"]
    _, ax = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
    for i, bidx in enumerate(idxs_to_plot):
        for pidx in range(npol):
            ax[0].plot(
                fobs_plot,
                np.abs(bpass[bidx, :, pidx]),
                ".",
                label=f"{labels[bidx]} {plabels[pidx]}",
                alpha=0.5,
                ls=lcyc[pidx],
                color=ccyc[i % len(ccyc)],
            )
            ax[1].plot(
                fobs_plot,
                np.angle(bpass[bidx, :, pidx]),
                ".",
                label=f"{labels[bidx]} {plabels[pidx]}",
                alpha=0.5,
                ls=lcyc[pidx],
                color=ccyc[i % len(ccyc)],
            )
    ax[0].set_xlabel("freq (GHz)")
    ax[1].set_xlabel("freq (GHz)")
    ax[0].set_ylabel("B cal amp")
    ax[1].set_ylabel("B cal phase (rad)")
    ax[0].legend(ncol=3, fontsize="small")
    if outname is not None:
        plt.savefig(f"{outname}_bandpass.png".format(outname), bbox_inches="tight")
    if not show:
        plt.close()

    return bpass, fobs, labels


def plot_autocorr(UV):
    """Plots autocorrelations from UVData object.

    Parameters
    ----------
    UV : UVData object
        The UVData object for which to plot autocorrelations.
    """
    freq = UV.freq_array.squeeze()
    ant1 = UV.ant_1_array.reshape(UV.Ntimes, -1)[0, :]
    ant2 = UV.ant_2_array.reshape(UV.Ntimes, -1)[0, :]
    time = UV.time_array.reshape(UV.Ntimes, -1)[:, 0]
    vis = UV.data_array.reshape(UV.Ntimes, -1, UV.Nfreqs, UV.Npols)
    autocorrs = np.where(ant1 == ant2)[0]
    ccyc = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    _, ax = plt.subplots(
        len(autocorrs) // len(ccyc) + 1,
        1,
        figsize=(8, 4 * len(autocorrs) // len(ccyc) + 1),
        sharex=True,
        sharey=True,
    )
    for j in range(len(autocorrs) // len(ccyc) + 1):
        for i, ac in enumerate(autocorrs[len(ccyc) * j : len(ccyc) * (j + 1)]):
            ax[j].plot(
                freq.squeeze() / 1e9,
                np.abs(np.nanmean(vis[:, ac, ..., 0], axis=0)),
                alpha=0.5,
                color=ccyc[i % len(ccyc)],
                ls="-",
                label=ant1[ac] + 1,
            )
            ax[j].plot(
                freq.squeeze() / 1e9,
                np.abs(np.nanmean(vis[:, ac, ..., 1], axis=0)),
                alpha=0.5,
                color=ccyc[i % len(ccyc)],
                ls=":",
            )
        ax[j].legend()
    ax[-1].set_xlabel("freq (GHz)")
    ax[0].set_yscale("log")
    plt.subplots_adjust(hspace=0)

    _, ax = plt.subplots(
        len(autocorrs) // len(ccyc) + 1,
        1,
        figsize=(8, 4 * len(autocorrs) // len(ccyc) + 1),
        sharex=True,
        sharey=True,
    )
    for j in range(len(autocorrs) // len(ccyc) + 1):
        for i, ac in enumerate(autocorrs[len(ccyc) * j : len(ccyc) * (j + 1)]):
            ax[j].plot(
                (time - time[0]) * 24 * 60,
                np.abs(vis[:, ac, ..., 0].mean(axis=1)),
                alpha=0.5,
                color=ccyc[i % len(ccyc)],
                ls="-",
                label=ant1[ac] + 1,
            )
            ax[j].plot(
                (time - time[0]) * 24 * 60,
                np.abs(vis[:, ac, ..., 1].mean(axis=1)),
                alpha=0.5,
                color=ccyc[i % len(ccyc)],
                ls=":",
            )
        ax[j].legend()
    ax[-1].set_xlabel("time (min)")
    ax[0].set_yscale("log")
    plt.subplots_adjust(hspace=0)


def summary_plot(msname, calname, npol, plabels, antennas):
    r"""Generates a summary plot showing the calibration solutions.

    Parameters
    ----------
    msname : str
        The path to the measurement set is ``msname``.ms.
    calname : str
        The name of the calibrator. Calibration tables starting with
        ``msname``\_``calname`` will be opened.
    npol : int
        The number of polarization indices. Currently not in use.
    plabels : list
        The labels of the polarizations.
    antennas: list
        The antennas to plot.

    Returns
    -------
    matplotlib.pyplot.figure
        The handle for the generated figure.
    """
    # TODO: remove npol, and assert that plabels and npol have the same shapes
    ny = len(antennas) // 10
    if len(antennas) % 10 != 0:
        ny += 1
    ccyc = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    mcyc = [".", "x"]
    lcyc = ["-", "--"]

    fig, ax = plt.subplots(4, ny * 2, figsize=(12, 12))
    ax = ax.reshape(4, ny, 2).swapaxes(0, 1)
    ax[0, 0, 0].axis("off")

    # Plot kcal
    if os.path.exists(f"{msname}_{calname}_2kcal"):
        antenna_delays, times, _flags, _ant1, _ant2 = read_caltable(
            f"{msname}_{calname}_2kcal", cparam=False
        )
        npol = antenna_delays.shape[-1]
        kcorr, _tkcorr, _flags, _antenna_order, _ant2 = read_caltable(
            f"{msname}_{calname}_kcal", cparam=False
        )
        val_to_plot = (antenna_delays - kcorr).squeeze(axis=3).mean(axis=2)
        tplot = (times - times[0]) * ct.SECONDS_PER_DAY / 60.0

        for i, ant in enumerate(antennas):
            for j in range(npol):
                ax[i // 10, 0, 1].plot(
                    tplot,
                    val_to_plot[ant - 1, :, j],
                    marker=mcyc[j % len(mcyc)],
                    linestyle=lcyc[j % len(lcyc)],
                    label=f"{ant} {plabels[j]}",
                    alpha=0.5,
                    color=ccyc[i % len(ccyc)],
                )
        for i in range(ny):
            ax[i, 0, 1].set_ylim(-5, 5)
            ax[i, 0, 1].axhline(1.5)
            ax[i, 0, 1].axhline(-1.5)
        ax[0, 0, 1].legend(ncol=3, loc="upper left", bbox_to_anchor=(-1, 1))
    else:
        tplot = None

    if os.path.exists(f"{msname}_{calname}_bcal"):
        bpass, _tbpass, _flags, ant1, ant2 = read_caltable(
            f"{msname}_{calname}_bcal", cparam=True
        )
        bpass = bpass.squeeze(axis=1)
        bpass = bpass.reshape(bpass.shape[0], -1, bpass.shape[-1])
        npol = bpass.shape[-1]

        with table(f"{msname}.ms/SPECTRAL_WINDOW") as tb:
            fobs = (np.array(tb.col("CHAN_FREQ")[:]) / 1e9).reshape(-1)

        if bpass.shape[1] != fobs.shape[0]:
            nint = fobs.shape[0] // bpass.shape[1]
            fobs_plot = (
                np.mean(fobs[:nint])
                + np.arange(bpass.shape[1]) * np.median(np.diff(fobs)) * nint
            )
        else:
            fobs_plot = fobs.copy()

        for i, ant in enumerate(antennas):
            for pidx in range(npol):
                ax[i // 10, 1, 0].plot(
                    fobs_plot,
                    np.abs(bpass[ant - 1, :, pidx]),
                    label=f"{ant} {plabels[pidx]}",
                    alpha=0.5,
                    ls=lcyc[pidx % len(lcyc)],
                    color=ccyc[i % len(ccyc)],
                )
                ax[i // 10, 2, 0].plot(
                    fobs_plot,
                    np.angle(bpass[ant - 1, :, pidx]),
                    label=f"{ant} {plabels[pidx]}",
                    alpha=0.5,
                    ls=lcyc[pidx],
                    color=ccyc[i % len(ccyc)],
                )
                ax[i // 10, 1, 0].set_yscale("log")

    if os.path.exists(f"{msname}_{calname}_2gcal"):
        gain, time, _flags, ant1, ant2 = read_caltable(
            f"{msname}_{calname}_2gcal", cparam=True
        )
        gain = gain.squeeze(axis=3)
        gain = gain.mean(axis=2)
        t0 = time[0]
        time = ((time - t0) * u.d).to_value(u.min)

        for i, ant in enumerate(antennas):
            for pidx in range(npol):
                ax[i // 10, 1, 1].plot(
                    time,
                    np.abs(gain[ant - 1, :, pidx]),
                    label=f"{ant} {plabels[pidx]}",
                    color=ccyc[i % len(ccyc)],
                    ls=lcyc[pidx % len(lcyc)],
                    marker=mcyc[pidx % len(mcyc)],
                )

        for i, ant in enumerate(antennas):
            for pidx in range(npol):
                ax[i // 10, 2, 1].plot(
                    time,
                    np.angle(gain[ant - 1, :, pidx]),
                    label=f"{ant} {plabels[pidx]}",
                    color=ccyc[i % len(ccyc)],
                    ls=lcyc[pidx],
                )
        for i in range(ny):
            if tplot:
                ax[i, 1, 1].set_xlim(tplot[0], tplot[-1])
                ax[i, 2, 1].set_xlim(tplot[0], tplot[-1])
            ax[i, 2, 1].set_ylim(-np.pi / 10, np.pi / 10)
    else:
        t0 = None

    vis, time, fobs, _, ant1, ant2, _, _, _ = extract_vis_from_ms(msname)
    autocorr_idx = np.where(ant1 == ant2)[0]
    vis_autocorr = vis[autocorr_idx, ...]
    vis_time = np.median(
        vis_autocorr.reshape(vis_autocorr.shape[0], vis_autocorr.shape[1], -1, 2),
        axis=-2,
    )
    vis_freq = np.median(
        vis_autocorr.reshape(vis_autocorr.shape[0], vis_autocorr.shape[1], -1, 2),
        axis=1,
    )
    if t0 is None:
        t0 = time[0]
    time = ((time - t0) * u.d).to_value(u.min)
    vis_ant_order = ant1[autocorr_idx]
    for i, ant in enumerate(antennas):
        vis_idx = np.where(vis_ant_order == ant - 1)[0]
        if len(vis_idx) > 0:
            vis_idx = vis_idx[0]
            for pidx in range(npol):
                ax[i // 10, 3, 1].plot(
                    time - time[0],
                    np.abs(vis_time[vis_idx, :, pidx]),
                    label=f"{ant} {plabels[pidx]}",
                    color=ccyc[i % len(ccyc)],
                    ls=lcyc[pidx % len(lcyc)],
                    alpha=0.5,
                )
                ax[i // 10, 3, 0].plot(
                    fobs,
                    np.abs(vis_freq[vis_idx, :, pidx]),
                    label=f"{ant} {plabels[pidx]}",
                    color=ccyc[i % len(ccyc)],
                    ls=lcyc[pidx % len(lcyc)],
                    alpha=0.5,
                )

    for i in range(ny):
        ax[i, 3, 1].set_xlabel("time (min)")
        ax[i, 3, 0].set_xlabel("freq (GHz)")
        ax[i, 3, 1].set_ylabel("autocorr power")
        ax[i, 3, 0].set_ylabel("autocorr power")
        ax[i, 3, 0].set_yscale("log")
        ax[i, 3, 1].set_yscale("log")
        ax[i, 1, 1].set_ylabel("Abs of gain")
        ax[i, 2, 1].set_ylabel("Phase of gain")
        ax[i, 1, 0].set_ylabel("B cal amp")
        ax[i, 2, 0].set_ylabel("B cal phase (rad)")
        ax[i, 0, 1].set_ylabel("delay (ns)")
    fig.suptitle(f"{msname}")
    return fig


def plot_current_beamformer_solutions(
    filenames,
    calname,
    date,
    beamformer_name,
    corrlist=np.arange(1, 16 + 1),
    antennas_to_plot=None,
    antennas=None,
    outname=None,
    show=True,
    gaindir="/home/user/beamformer_weights/",
    hdf5dir="/mnt/data/dsa110/correlator/",
):
    r"""Plots the phase difference between the two polarizations.

    Applies the beamformer weights to the given hdf5 files, and then plots the
    remaining phase difference between the two polarizations for each antenna.

    Parameters
    ----------
    filenames : list
        A list of the hdf5 filenames for which to plot. Each filename should
        omit the directory and the .hdf5 extension. E.g.
        `['2020-10-06T15:32:01', '2020-10-06T15:47:01']` would a valid
        argument to plot data for 30 minutes starting at 2020-10-06T16:32:01.
    calname : str
        The name of the source or event that you are plotting. Used in the
        title of the plot.
    date : str
        The date of the source or event that you are plotting. Used in the
        title of the plot. e.g. '2020-10-06'
    beamformer_name : str
        The title of the beamformer weights.
        e.g. 'J141120+521209_2021-02-19T12:05:51'
    corrlist : list(int)
        A list of the correlator indices to plot.  Defaults to correlators 01
        through 16.
    antennas_to_plot : array(int)
        The names of the antennas to plot beamformer solutions for. Defaults to
        `antennas`.
    antennas : array(int)
        The names of the antennas for which beamformer solutions were
        generated.  Defaults to values in dsautils.cnf
    outname : str
        The base to use for the name of the png file the plot is saved to. The
        plot is saved to `outname`\_beamformerweights.png if `outname` is not
        ``None``. f `outname` is set to ``None, the plot is not saved.
        Defaults `None``.
    show : boolean
        If set to ``False`` the plot is closed after being generated.
    gaindir : str
        The full path to the directory in which the beamformer weights are
        stored.
    hdf5dir : str
        The full path to the directory in which the correlated hdf5 files are
        stored. Files were be searched for in `hdf5dir`/corr??/
    """
    if antennas is None:
        antennas = np.array(list(CONF.get("corr")["antenna_order"].values))
    assert len(antennas) == 64
    if antennas_to_plot is None:
        antennas_to_plot = antennas
    # Should be generalized to different times, baselines
    visdata_corr = np.zeros((len(filenames) * 280, 325, 16, 48, 2), dtype=np.complex)
    for corridx, corr in enumerate(corrlist):
        visdata = np.zeros((len(filenames), 91000, 1, 48, 2), dtype=np.complex)
        for i, filename in enumerate(filenames):
            files = sorted(
                glob.glob(
                    f"{hdf5dir}/corr{corr:02d}/{filename[:-2]}??.hdf5"
                )
            )
            if len(files) > 0:
                with h5py.File(files[0], "r") as f:
                    visdata[i, ...] = np.array(f["Data"]["visdata"][:])
                    ant1 = np.array(f["Header"]["ant_1_array"][:])
                    ant2 = np.array(f["Header"]["ant_2_array"][:])
        visdata = visdata.reshape((-1, 325, 48, 2))
        ant1 = ant1.reshape(-1, 325)[0, :]
        ant2 = ant2.reshape(-1, 325)[0, :]
        with open(
                f"{gaindir}/beamformer_weights_corr{corr:02d}_{beamformer_name}.dat",
                "rb",
        ) as f:
            data = np.fromfile(f, "<f4")
        gains = data[64:].reshape(64, 48, 2, 2)
        gains = gains[..., 0] + 1.0j * gains[..., 1]
        my_gains = np.zeros((325, 48, 2), dtype=np.complex)
        for i in range(325):
            idx1 = np.where(antennas == ant1[i] + 1)[0][0]
            idx2 = np.where(antennas == ant2[i] + 1)[0][0]
            my_gains[i, ...] = np.conjugate(gains[idx1, ...]) * gains[idx2, ...]
        visdata_corr[:, :, corridx, ...] = visdata * my_gains[np.newaxis, :, :, :]
    visdata_corr = visdata_corr.reshape((-1, 325, 16 * 48, 2))

    fig, ax = plt.subplots(5, 4, figsize=(25, 12), sharex=True, sharey=True)
    for axi in ax[-1, :]:
        axi.set_xlabel("freq channel")
    for axi in ax[:, 0]:
        axi.set_ylabel("time bin")
    ax = ax.flatten()
    for i, ant in enumerate(antennas_to_plot - 1):
        idx = np.where((ant1 == 23) & (ant2 == ant))[0][0]
        ax[i].imshow(
            np.angle(visdata_corr[:, idx, :, 0] / visdata_corr[:, idx, :, 1]),
            aspect="auto",
            origin="lower",
            interpolation="none",
            cmap=plt.get_cmap("RdBu"),
            vmin=-np.pi,
            vmax=np.pi,
        )
        avg_phase = np.angle(
            np.mean(visdata_corr[:, idx, :, 0] / visdata_corr[:, idx, :, 1])
        )
        ax[i].set_title(f"24-{ant + 1}, {avg_phase:.2f}")
    fig.suptitle(f"{date} {calname}")
    if outname is not None:
        plt.savefig(f"{outname}_beamformerweights.png")
    if not show:
        plt.close()


def plot_bandpass_phases(
    beamformer_names,
    antennas,
    refant=24,
    outname=None,
    show=True,
    msdir="/mnt/data/dsa110/calibration/",
):
    r"""Plot the bandpass phase observed over multiple calibrators.

    Parameters:
    -----------
    beamformer_names : dict
        The beamformer weight filenames for which to plot the phases.
    antennas : list
        The antenna names (as ints) to plot.
    refant : int
        The reference antenna to plot phases against.
    outname : str
        The base to use for the name of the png file the plot is saved to. The
        plot is saved to `outname`\_phase.png if `outname` is not ``None``. If
        `outname` is set to ``None, the plot is not saved. Defaults `None``.
    show : boolean
        If set to ``False`` the plot is closed after being generated.
    """


    # Parse cal name and date information from the beamformer names
    cals = []
    transit_times = []
    dates = []
    for beamformer_name in beamformer_names:
        try:
            cal, transit_time = beamformer_name.split("_")
        except ValueError:
            print(f'Not plotting phases for {beamformer_name}')
        else:
            cals += [cal]
            transit_times += [transit_time]
            dates += [transit_time.split("T")[0]]

    # Sort beamformer weights by transit time
    transit_times, cals, dates = zip(*sorted(zip(transit_times, cals, dates)))

    nentries = len(cals)
    # Read in gain phase calibration tables for each calibrator pass
    calnames = []
    gains = [None] * nentries
    gshape = None
    for i, cal in enumerate(cals):
        date = dates[i]
        calnames += [f"{date}_{cal}"]
        msname = f"{msdir}/{date}_{cal}"

        bpcal_table = f"{msname}_{cal}_bpcal"
        if os.path.exists(bpcal_table):
            with table(bpcal_table) as tb:
                gains[i] = np.array(tb.CPARAM[:])
            if gshape is None:
                gshape = gains[i].shape

    if gshape is None:  # We have no data to plot
        return

    # Fill missing data with default to safely convert to an array
    for i, gain in enumerate(gains):
        if gain is None:
            gains[i] = np.zeros(gshape, dtype=np.complex64)
    gains = np.array(gains)

    # Set up the plot
    nx = 4
    ny = len(antennas) // nx
    if len(antennas) % nx > 0:
        ny += 1
    fig, ax = plt.subplots(ny, nx, figsize=(3 * nx, 3 * ny), sharex=True, sharey=True)

    # Set the yticks to display the calibrator pass details
    ax[0, 0].set_yticks(np.arange(nentries))
    for axi in ax[0, :]:
        axi.set_yticklabels(calnames)

    # Plot the gains for each antenna
    ax = ax.flatten()
    for i in np.arange(len(antennas)):
        ax[i].imshow(
            np.angle(gains[:, antennas[i] - 1, :, 0] / gains[:, refant - 1, :, 0]),
            vmin=-np.pi,
            vmax=np.pi,
            aspect="auto",
            origin="lower",
            interpolation="None",
            cmap=plt.get_cmap("RdBu"),
        )
        ax[i].annotate(str(antennas[i]), (0, 1), xycoords="axes fraction")
        ax[i].set_xlabel("Frequency channel")

    if outname is not None:
        plt.savefig(f"{outname}_phases.png")
    if not show:
        plt.close(fig)


def plot_beamformer_weights(
    beamformer_names,
    corrlist=np.arange(1, 16 + 1),
    antennas_to_plot=None,
    antennas=None,
    outname=None,
    pols=None,
    show=True,
    gaindir="/home/user/beamformer_weights/",
):
    """Plot beamformer weights from a number of beamformer solutions.

    Parameters
    ----------
    beamformer_names : list(str)
        The postfixes of the beamformer weight files to plot. Will open
        beamformer_weights_corr??_`beamformer_name`.dat for each item in
        beamformer_names.
    corrlist : list(int)
        The corrnode numbers.
    antennas_to_plot : list(int)
        The names of the antennas to plot. Defaults to `antennas`.
    antennas : list(int)
        The names of the antennas in the beamformer weight files. Defaults to
        list in dsautils.cnf
    outname : str
        The prefix of the file to save the plot to. If None, no figure is saved.
    pols : list(str)
        The order of the pols in the beamformer weight files.
    show : bool
        If False, the plot is closed after saving.
    gaindir : str
        The directory in which the beamformer weight files are saved.

    Returns
    -------
    ndarray
        The beamformer weights.
    """
    if pols is None:
        pols = ["B", "A"]
    if antennas is None:
        antennas = np.array(list(CONF.get("corr")["antenna_order"].values()))
    assert len(antennas) == 64
    if antennas_to_plot is None:
        antennas_to_plot = antennas
    # Set shape of the figure
    nplots = 4
    nx = 5
    ny = len(antennas_to_plot) // nx
    if len(antennas_to_plot) % nx != 0:
        ny += 1
    gains = np.zeros(
        (len(beamformer_names), len(antennas), len(corrlist), 48, 2), dtype=np.complex
    )
    for i, beamformer_name in enumerate(beamformer_names):
        for corridx, corr in enumerate(corrlist):
            with open(
                    f"{gaindir}/beamformer_weights_corr{corr:02d}_{beamformer_name}.dat",
                    "rb",
            ) as f:
                data = np.fromfile(f, "<f4")
            temp = data[64:].reshape(64, 48, 2, 2)
            gains[i, :, corridx, :, :] = temp[..., 0] + 1.0j * temp[..., 1]
    gains = gains.reshape((len(beamformer_names), len(antennas), len(corrlist) * 48, 2))

    # Phase, polarization B
    fig, ax = plt.subplots(
        nplots * ny, nx, figsize=(6 * nx, 2.5 * ny * nplots), sharex=True, sharey=False
    )
    for axi in ax[-1, :]:
        axi.set_xlabel("freq channel")
    for nplot in range(nplots):
        polidx = nplot % 2
        angle = nplot // 2
        axi = ax[ny * nplot : ny * (nplot + 1), :]
        for axii in axi[:, 0]:
            axii.set_ylabel("phase (rad)" if angle else "amplitude (arb)")
        axi = axi.flatten()
        for i, ant in enumerate(antennas_to_plot):
            for bnidx, beamformer_name in enumerate(beamformer_names):
                idx = np.where(antennas == ant)[0][0]
                axi[i].plot(
                    np.angle(gains[bnidx, idx, :, polidx])
                    if angle
                    else np.abs(gains[bnidx, idx, :, polidx]),
                    alpha=0.4,
                    ls="None",
                    marker=".",
                    label=beamformer_name,
                )
                axi[i].set_title(f"{ant} {pols[polidx]}: {'phase' if angle else 'amp'}")
            if angle:
                axi[i].set_ylim(-np.pi, np.pi)

        axi[0].legend()

    if outname is not None:
        plt.savefig(f"{outname}_averagedweights.png")
    if not show:
        plt.close(fig)

    return gains
