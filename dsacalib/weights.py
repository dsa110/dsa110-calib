"""Determine and inspect beamformer weights."""

from typing import List
import glob
import os
from datetime import datetime, timedelta, timezone

import astropy.units as u
from astropy.time import Time
import matplotlib.pyplot as plt
import numpy as np
import yaml
from antpos.utils import get_itrf

import dsacalib.constants as ct
from dsacalib.fringestopping import calc_uvw
from dsacalib.ms_io import get_antenna_gains, get_delays, read_caltable, freq_GHz_from_ms


def get_good_solution(
        beamformer_dir, refsb, antennas, refant, select=None,
        plot=False, threshold_ants=45, threshold_angle=10, selectcore=True, antennas_core=None
):
    """Find good set of solutions and calculate average gains.
    TODO: go from good bfnames to average gains.
    """
    if selectcore and not antennas_core:
        raise RuntimeError(
            "Specified selection of weights should be done using core antennas but core antennas "
            "not specified.")

    if select is None:
        today = datetime.now(timezone.utc)
        select = [f"{today.year}-{today.month:02}-{today.day:02}"]
        today = today - timedelta(days=1)
        select += [f"{today.year}-{today.month:02}-{today.day:02}"]

    print(f"Selecting for string {select}")
    if isinstance(select, str):
        bfnames = get_bfnames(beamformer_dir, refsb, select=select)
    else:
        bfnames = get_bfnames(beamformer_dir, refsb, select=select[0])
        if len(select) > 1:
            for sel in select[1:]:
                bfnames += get_bfnames(beamformer_dir, refsb, select=sel)
    if not bfnames:
        return bfnames
    times = [bfname.split("_")[1] for bfname in bfnames]
    times, bfnames = zip(*sorted(zip(times, bfnames), reverse=True))
    if selectcore:
        gains = read_gains(bfnames, antennas, beamformer_dir,
                           selectants=antennas_core)
    else:
        gains = read_gains(bfnames, antennas, beamformer_dir)
    good = find_good_solutions(
        bfnames,
        gains,
        refant,
        antennas,
        antennas_core=antennas_core,
        plot=plot,
        threshold_ants=threshold_ants,
        threshold_angle=threshold_angle,
        selectcore=selectcore,
    )
    if plot:
        show_gains(bfnames, gains, good, refant, antennas,
                   antennas_core, selectcore=selectcore)

    return [bfnames[gidx] for gidx in good]


def get_bfnames(beamformer_dir: str, refsb: str = "sb01", select: List[str] = None):
    """Run on dsa-storage to get file names for beamformer weight files.

    Returns list of calibrator-times that are used in data file name.

    Parameters
    ----------
    beamformer_dir : str
        Path to beamformer weight files.
    refsb : str
        Subband to use as reference.
    select : list of str, optional
        List of substrings of the beamformer weights to select.  For example,
        ["2022-06-01", "2022-06-02"] would select all weights with names containing
        those two dates.

    Returns
    -------
    list of str
        List of beamformer weight file names selected.
    """
    fn_dat = glob.glob(
        os.path.join(beamformer_dir, f"beamformer_weights*{refsb}*.dat"))

    bfnames = []
    for fn in fn_dat:
        sp = fn.split(f"_{refsb}_")
        if len(sp) > 1:
            if "_" in sp[1]:
                bfnames.append(sp[1].rstrip(".dat"))

    # select subset
    if select is not None:
        bfnames2 = [bfn for bfn in bfnames if select in bfn]
        print(f"Selecting {len(bfnames2)} from {len(bfnames)} gain files.")
        return bfnames2

    print(f"Selecting all {len(bfnames)} gain files.")
    return bfnames


def read_gains(
        bfnames, antennas, beamformer_dir, selectants=None, path=None, nsubbands=16, nchanspw=48,
        npol=2):
    """Reads gain for each of the data files in bfnames.

    Returns gain array with same length as bfnames.
    path can overload the location of files stored in cnf.

    Parameters
    ----------
    bfnames : list of str
        List of beamformer weight file names.
    antennas : list of str
        List of antenna names.
    beamformer_dir : str
        Path to beamformer weight files.
    selectants : list of str, optional
        List of antenna names to use.  If None, use all antennas.
    path : str, optional
        Path to beamformer weight files.  If None, use beamformer_dir.
    nsubbands : int, optional
        Number of subbands.  If None, use default.
    nchanspw : int, optional
        Number of channels per subband.  If None, use default.
    npol : int, optional
        Number of polarizations.  If None, use default.

    Returns
    -------
    np.ndarray
        Array of gains.
    """
    if selectants is None:
        selectants = antennas

    gains = np.zeros(
        (len(bfnames), len(antennas), nsubbands, nchanspw, npol), dtype=np.complex
    )

    if path is None:
        path = beamformer_dir

    for i, beamformer_name in enumerate(bfnames):
        for subband in range(nsubbands):
            with open(
                    f"{path}/beamformer_weights_sb{subband:02d}_{beamformer_name}.dat",
                    "rb",
            ) as f:
                data = np.fromfile(f, "<f4")
            temp = data[64:].reshape(64, 48, 2, 2)
            gains[i, :, subband, :, :] = temp[..., 0] + 1.0j * temp[..., 1]
    gains = gains.reshape((len(bfnames), len(antennas), nsubbands * 48, 2))
    select = [list(antennas).index(i) for i in selectants]
    print(f"Using {len(bfnames)} to get gain array of shape {gains.shape}.")
    return gains.take(select, axis=1)


def find_good_solutions(
        bfnames, gains, refant, antennas, antennas_core=None, threshold_ants=45, threshold_angle=10,
        mask_bothpols=True, plot=False, selectcore=True,
):
    """Given names and gain array, calc good set.
    Returns indices of bfnames argument that are good.
    """
    if selectcore:
        if not antennas_core:
            raise RuntimeError(
                "selectcore specified but no core antennas passed with antennas_core")
        ants = antennas_core
    else:
        ants = antennas

    try:
        refant_ind = ants.index(int(refant))
    except:
        refant_ind = 0
        print(f"Using first listed antenna ({ants[0]}) as refant")

    nchan = gains.shape[2]
    grads = np.zeros((len(bfnames), len(ants), 2))
    for i in np.arange(2):
        for j in np.arange(len(ants)):
            #            angles = np.zeros((len(bfnames), nchan))
            for k in np.arange(len(bfnames)):
                if gains[k, j, :, i].any():
                    #                    angle = np.angle(gains[k, j, :, i]/gains[0, j, :, i])
                    angle = np.angle(
                        gains[k, j, :, i] / gains[k, refant_ind, :, i])
                    #                    angles[k] = angle
                    medgrad = np.median(np.gradient(angle))
                    grads[k, j, i] = medgrad
    #            angles = np.ma.masked_equal(angles, 0)
    #            angles = np.ma.masked_invalid(angles, 0)
    # TODO: find bad k from comparing over all k
    # for k in np.arange(len(bfnames)):
    grads = np.ma.masked_equal(grads, 0)
    grads = np.ma.masked_invalid(grads)

    if mask_bothpols:
        maskor = np.ma.mask_or(grads.mask[:, :, 0], grads.mask[:, :, 1])
        grads.mask[:, :, 0] = maskor
        grads.mask[:, :, 1] = maskor

    # select good sets
    keep = []
    for k in np.arange(len(bfnames)):
        ngood = len(ants) - sum(
            grads[k, :, 0].mask
        )  # assumes pol=0 has useful flagging info

        print(f"Good phase gradients: {k}, {bfnames[k]}, {ngood} antennas")
        if ngood >= threshold_ants:
            keep.append(k)
            print(f"{bfnames[k]}: good")
        else:
            print(f"{bfnames[k]}: rejected")

    gains = np.ma.masked_array(gains)
    for i in range(nchan):
        gains[..., i, :].mask = grads.mask

    # calc relative phase change across all pairs of solutions to find outliers
    #    for i in np.arange(2):
    bad = []
    for k in np.arange(len(bfnames)):
        for j in np.arange(k + 1, len(bfnames)):
            if k not in keep or j not in keep:
                continue
            for p in [0, 1]:
                angle = np.degrees(
                    np.angle(gains[k, :, :, p] / gains[j, :, :, p]).mean()
                )
                if np.abs(angle) >= threshold_angle:
                    bad.append(k)
                    bad.append(j)
                    status_pair = "bad"
                else:
                    status_pair = "good"
                print(
                    f"Rel phase for cal pairs ({k}, {j}), ({bfnames[k]}, {bfnames[j]}), "
                    f"pol {p}: {angle} degrees => {status_pair}"
                )
    if bad:
        keepcount = 2 * (len(keep) - 1)
        for k in np.arange(len(bfnames)):
            if k not in keep:
                continue
            if bad.count(k) >= keepcount:
                keep.remove(k)
                status_sol = "reject"
            else:
                status_sol = "keep"
            print(
                f"{bfnames[k]} has {bad.count(k)} bad relative gains => {status_sol}")

    if plot:
        # visualize grads
        pl, (ax0, ax1) = plt.subplots(1, 2)
        ax0.imshow(grads[:, :, 0], origin="lower")
        ax0.set_xlabel("antenna (pol 0)")
        ax0.set_ylabel("calibrator")
        ax1.imshow(grads[:, :, 1], origin="lower")
        ax1.set_xlabel("antenna (pol 1)")
        ax1.set_ylabel("calibrator")
        plt.show()

    print(f"keep set: {keep}, {[bfnames[k] for k in keep]}")
    return keep


def show_gains(
        bfnames, gains, keep, refant, antennas, antennas_core, selectcore=True, ret=False,
        show=True):
    """Given bfnames and gains, plot the good gains.
    Default uses mpl show. Optionally can return figure with ret=True.
    """
    if selectcore:
        if not antennas_core:
            raise RuntimeError(
                "selectcore specified but no core antennas passed as antennas_core")
        ants = antennas_core
    else:
        ants = antennas

    try:
        refant_ind = ants.index(int(refant))
    except ValueError:
        refant_ind = 0
        print(f"Using first listed antenna ({ants[0]}) as refant")

    nants = len(ants)
    nx = 4
    ny = nants // nx
    if nants % nx > 0:
        ny += 1
    fig, ax = plt.subplots(ny, nx, figsize=(
        3 * nx, 4 * ny), sharex=True, sharey=True)
    ax[0, 0].set_yticks(np.arange(len(keep)))

    for axi in ax[0, :]:
        axi.set_yticklabels(
            [bfn for i, bfn in enumerate(bfnames) if i in keep])
    ax = ax.flatten()

    for i in np.arange(nants):
        ax[i].imshow(
            np.angle(gains[:, i, :, 0] /
                     gains[:, refant_ind, :, 0]).take(keep, axis=0),
            vmin=-np.pi,
            vmax=np.pi,
            aspect="auto",
            origin="lower",
            interpolation="None",
            cmap=plt.get_cmap("RdBu"),
        )
        ax[i].annotate(f"{ants[i]}", (0, 1), xycoords="axes fraction")
        ax[i].set_xlabel("Frequency channel")

    if show:
        plt.show()
    if ret:
        return fig


def calc_eastings(antennas, refmjd):
    """Calculate the eastings (u).

    Generates them for baselines composed of antennas in `antennas`.
    """
    antpos_df = get_itrf(
        latlon_center=(ct.OVRO_LAT * u.rad, ct.OVRO_LON *
                       u.rad, ct.OVRO_ALT * u.m)
    )
    blen = np.zeros((len(antennas), 3))
    for i, ant in enumerate(antennas):
        blen[i, 0] = antpos_df["x_m"].loc[ant] - antpos_df["x_m"].loc[24]
        blen[i, 1] = antpos_df["y_m"].loc[ant] - antpos_df["y_m"].loc[24]
        blen[i, 2] = antpos_df["z_m"].loc[ant] - antpos_df["z_m"].loc[24]
    bu, _, _ = calc_uvw(blen, refmjd, "HADEC", 0.0 * u.rad, 0.6 * u.rad)
    bu = bu.squeeze().astype(np.float32)
    return bu


def sort_beamformer_names(beamformer_names):
    """Sort beamformer names based on calibrator transit time."""
    calnames = []
    caltimes = []
    for beamformer_name in beamformer_names:
        calname, caltime = beamformer_name.split("_")
        calnames += [calname]
        caltimes += [caltime]
    caltimes, calnames = zip(*sorted(zip(caltimes, calnames), reverse=True))
    beamformer_names = []
    for i, calname in enumerate(calnames):
        caltime = caltimes[i]
        beamformer_names += [f"{calname}_{caltime}"]
    return beamformer_names


def filter_beamformer_solutions(beamformer_names, start_time, beamformer_dir):
    """Removes beamformer solutions inconsistent with the latest solution."""

    if len(beamformer_names) == 0:
        return beamformer_names, None

    beamformer_names = sort_beamformer_names(beamformer_names)

    if not os.path.exists(
        f"{beamformer_dir}/beamformer_weights_{beamformer_names[0]}.yaml"
    ):
        return [], None

    with open(
            f"{beamformer_dir}/beamformer_weights_{beamformer_names[0]}.yaml",
            encoding="utf-8"
    ) as f:
        latest_solns = yaml.load(f, Loader=yaml.FullLoader)

    for bfname in beamformer_names[1:].copy():
        try:
            with open(
                    f"{beamformer_dir}/beamformer_weights_{bfname}.yaml",
                    encoding="utf-8"
            ) as f:
                solns = yaml.load(f, Loader=yaml.FullLoader)
            assert consistent_correlator(solns, latest_solns, start_time)
        except (AssertionError, FileNotFoundError):
            beamformer_names.remove(bfname)

    return beamformer_names, latest_solns


def pull_out_cal_solutions(input_dict):
    """Extract the cal solutions dictionary from `input_dict`

    If cal_solutions is the the keys of `input_dict`, it is extracted
    and returned.  Otherwise `input_dict` is returned.
    """
    key = "cal_solutions"
    output_dict = input_dict[key] if key in input_dict else input_dict
    return output_dict


def consistent_correlator(full_solns, full_latest_solns, start_time):
    """
    Return True if new beamformer weight solution have the same correlator setup
    as `latest_solns`.
    """
    # pull from cal_solutions, if present

    solns = pull_out_cal_solutions(full_solns)
    latest_solns = pull_out_cal_solutions(full_latest_solns)

    if (
            solns['antenna_order'] != latest_solns['antenna_order'] or
            solns['delays'] != latest_solns['delays'] or
            solns['eastings'] != latest_solns['eastings'] or
            solns['caltime'] < latest_solns['caltime'] - 1 or
            solns['caltime'] < start_time):
        return False

    return True


def average_beamformer_solutions(
        fnames: List[str], ttime: Time, beamformer_dir: str, antennas: List[int], refmjd: float,
        tol: float = 0.3, nsubbands: int = 16, nchanspw: int = 48, npol: int = 2):
    """Averages written beamformer solutions.

    Parameters
    ----------
    fnames : list
    ttime : astropy.time.Time object
        A time to use in the filename of the solutions, indicating when they
        were written or are useful. E.g. the transit time of the most recent
        source being averaged over.
    beamformer_dir : str
        The directory where temporary beamformer solutions are stored.
    antennas : list
        The order of the antennas in the beamformer solutions.
    refmjd : float
        The reference mjd used in the beamformer solutions.
    tol : float
        The percentage of weights for a given antenna that can be flagged
        before the entire antenna is flagged.
    nsubbands : int
        The number of subbands in the beamformer solutions.

    Returns
    -------
    written_files : list
        The names of the written beamformer solutions (one for each correlator
        node).
    antenna_flags_badsolns:
        Flags for antenna/polarization dimensions of gains.
    """
    gains = read_gains(fnames, antennas, beamformer_dir, nsubbands=nsubbands)
    nants = len(antennas)
    print(gains.shape)

    antenna_flags = [None] * len(fnames)
    for i, fname in enumerate(fnames):
        tmp_antflags = []
        filepath = f"{beamformer_dir}/beamformer_weights_{fname}.yaml"
        with open(filepath, encoding="utf-8") as f:
            calibration_params = yaml.load(f, Loader=yaml.FullLoader)[
                "cal_solutions"]
            antenna_order = calibration_params["antenna_order"]
            for key in calibration_params["flagged_antennas"]:
                if (
                    "casa solutions flagged"
                    in calibration_params["flagged_antennas"][key]
                ):
                    antname = int(key.split(" ")[0])
                    tmp_antflags.append(antenna_order.index(antname))
        antenna_flags[i] = sorted(tmp_antflags)
        gains[i, antenna_flags[i], ...] = np.nan
    eastings = calc_eastings(antenna_order, refmjd)

    gains = np.nanmean(gains, axis=0)
    fracflagged = np.sum(np.isnan(gains), axis=1) / (gains.shape[1])
    antenna_flags_badsolns = fracflagged > tol
    gains[np.isnan(gains)] = 0.0
    gains = gains.astype(np.complex64).view(
        np.float32).reshape(nants, nsubbands, nchanspw, npol, 2)
    print(gains.shape)
    written_files = []
    if eastings is not None:
        for subband in range(nsubbands):
            fnameout = f"beamformer_weights_sb{subband:02d}_{ttime.isot}"
            wcorr = gains[:, subband, ...].flatten()
            wcorr = np.concatenate([eastings, wcorr], axis=0)
            with open(f"{beamformer_dir}/{fnameout}.dat", "wb") as f:
                f.write(bytes(wcorr))
            written_files += [f"{fnameout}.dat"]
    return written_files, antenna_flags_badsolns


def set_freq_beamformer_weights(
        nchan: int, nchan_spw: int, bw_GHz: float, chan_ascending: bool,
        f0_GHz: float, ch0dict: dict):
    """Set the frequency for the beamformer weights."""
    ncorr = len(ch0dict)
    fweights = np.ones((ncorr, 48), dtype=np.float32)

    dfreq = bw_GHz / nchan
    if chan_ascending:
        fobs = f0_GHz + np.arange(nchan) * dfreq
    else:
        fobs = f0_GHz - np.arange(nchan) * dfreq

    for sbnum, ch0 in enumerate(ch0dict.values()):
        fobs_corr = fobs[ch0:ch0 + nchan_spw]
        fweights[sbnum, :] = fobs_corr.reshape(
            fweights.shape[1], -1).mean(axis=1)

    return fweights


def write_beamformer_weights(
        msname, calname, caltime, antennas, beamformer_dir, nchan, nchan_spw, bw_GHz,
        chan_ascending, f0_GHz, ch0, refmjd, tol=0.3
):
    """Writes weights for the beamformer.

    Parameters
    ----------
    msname : str
        The prefix of the measurement set.  Will open `msname`.ms
    calname : str
        The name of the calibrator source.
    antennas : list
        The names of the antennas to extract solutions for.  Order must be the
        same as the order in the beamformer.
    tol : float
        The fraction of data for a single antenna/pol flagged that the
        can be flagged in the beamformer. If more data than this is flagged as
        having bad solutions, the entire antenna/pol pair is flagged.

    Returns
    -------
    bu : array
        The length of the baselines in the u direction for each antenna
        relative to antenna 24.
    filenames : list
        The names of the file containing the beamformer weights.
    antenna_flags_badsolns : np.ndarray, dimensions(nant, npol)
        1 where the antenna/pol pair is flagged for having bad beamformer weight solutions
    """
    ncorr = len(ch0)
    nant = len(antennas)
    npol = 2
    nfreq = 48

    fweights = set_freq_beamformer_weights(
        nchan, nchan_spw, bw_GHz, chan_ascending, f0_GHz, ch0)
    bu = calc_eastings(antennas, refmjd)

    fobs = freq_GHz_from_ms(msname)
    fobs = fobs.reshape(fweights.size, -1).mean(axis=1)
    f_reversed = not np.allclose(fobs, fweights.ravel())
    if f_reversed:
        assert np.allclose(fobs[::-1], fweights.ravel())

    gains, _time, flags, ant1, ant2 = read_caltable(
        f"{msname}_{calname}_bcal", cparam=True)
    gains[flags] = np.nan
    gains = get_antenna_gains(gains, ant1, ant2, antennas)
    assert gains.shape[0] == nant
    assert gains.shape[-1] == npol

    if f_reversed:
        print("Frequencies are reversed. Changing order of weights.")
        gains = gains.reshape((nant, -1, npol))[:, ::-1, :]
    gains = gains.reshape(nant, ncorr, -1, npol)
    assert gains.shape[2] % nfreq == 0

    gains = np.nanmean(gains.reshape(nant, ncorr, nfreq, -1, npol), axis=3)
    weights = gains.swapaxes(0, 1).astype(np.complex64).copy()

    fracflagged = (
        np.sum(np.sum(np.isnan(weights), axis=2), axis=0) / (weights.shape[0] * weights.shape[2]))
    antenna_flags_badsolns = fracflagged > tol
    weights[np.isnan(weights)] = 0.0

    filenames = []
    for i in range(ncorr):
        wcorr = weights[i, ...].view(np.float32).flatten()
        wcorr = np.concatenate([bu, wcorr], axis=0)
        fname = f"beamformer_weights_sb{i:02d}"
        fname = f"{fname}_{calname}_{caltime.isot}"
        if os.path.exists(f"{beamformer_dir}/{fname}.dat"):
            os.unlink(f"{beamformer_dir}/{fname}.dat")
        print(f"{beamformer_dir}/{fname}.dat")
        with open(f"{beamformer_dir}/{fname}.dat", "wb") as f:
            f.write(bytes(wcorr))
        filenames += [f"{fname}.dat".format(fname)]

    return bu, filenames, antenna_flags_badsolns


def write_beamformer_solutions(
        msname, calname, caltime, antennas, applied_delays, beamformer_dir, pols, nchan,
        nchan_spw, bw_GHz, chan_ascending, f0_GHz, ch0, refmjd, flagged_antennas=None):
    """Writes beamformer solutions to disk.

    Parameters
    ----------
    msname : str
        The name of the measurement set used for calibration.
    calname : str
        The name of the calibrator source used for calibration. Will open
        tables that start with `msname`_`calname`
    caltime : astropy.time.Time object
        The transit time of the calibrator.
    antennas : list
        The antenna names for which to write beamformer solutions, in order.
    applied_delays : ndarray
        The currently applied delays at the time of the calibration, in ns.
        Dimensions (antenna, pol). The antenna axis should be in the order
        specified by antennas.
    flagged_antennas : list
        A list of antennas to flag in the beamformer solutions. Should include
        polarizations. e.g. ['24 B', '32 A']
    pols : list
        The order of the polarizations.
    """
    # Set the delays and flags for large delays
    beamformer_flags = {}

    delays, flags = get_delays(antennas, msname, calname, applied_delays)

    if flagged_antennas is not None:
        for item in flagged_antennas:
            ant, pol = item.split(" ")
            flags[antennas == ant, pols == pol] = 1
            beamformer_flags[f"{ant} {pol}"] = ["flagged by user"]
    delays = delays - np.min(delays[~flags])
    # TODO: this is currently flagging antennas that it shouldn't be
    # while not np.all(delays[~flags] < 1024):
    #    if np.sum(delays[~flags] > 1024) < np.nansum(delays[~flags] < 1024):
    #        argflag = np.argmax(delays[~flags])
    #    else:
    #        argflag = np.argmin(delays[~flags])
    #    argflag = np.where(~flags.flatten())[0][argflag]
    #    flag_idxs = np.unravel_index(argflag, flags.shape)
    #    flags[np.unravel_index(argflag, flags.shape)] = 1
    #    key = '{0} {1}'.format(antennas[flag_idxs[0]], pols[flag_idxs[1]])
    #    if key not in beamformer_flags.keys():
    #        beamformer_flags[key] = []
    #    beamformer_flags[key] += ['delay exceeds snap capabilities']
    #    delays = delays-np.min(delays[~flags])

    caltime.precision = 0
    eastings, weights_files, flags_badsolns = write_beamformer_weights(
        msname, calname, caltime, antennas, beamformer_dir, nchan, nchan_spw, bw_GHz,
        chan_ascending, f0_GHz, ch0, refmjd)

    idxant, idxpol = np.nonzero(flags_badsolns)
    for i, ant in enumerate(idxant):
        key = f"{antennas[ant]} {pols[idxpol[i]]}"
        if key not in beamformer_flags:
            beamformer_flags[key] = []
        beamformer_flags[key] += ["casa solutions flagged"]

    calibration_dictionary = {
        "cal_solutions": {
            "source": calname,
            "caltime": float(caltime.mjd),
            "antenna_order": [int(ant) for ant in antennas],
            "pol_order": ["B", "A"],
            "delays": [[int(delay[0] // 2), int(delay[1] // 2)] for delay in delays],
            "eastings": [float(easting) for easting in eastings],
            "weights_axis0": "antenna",
            "weights_axis1": "frequency",
            "weights_axis2": "pol",
            "weight_files": weights_files,
            "flagged_antennas": beamformer_flags,
        }
    }

    with open(
            f"{beamformer_dir}/beamformer_weights_{calname}_{caltime.isot}.yaml",
            "w",
            encoding="utf-8"
    ) as file:
        yaml.dump(calibration_dictionary, file)
