"""Determine and inspect beamformer weights."""

import glob
import os
from collections import namedtuple

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import yaml
from antpos.utils import get_itrf
import scipy  #pylint: disable=unused-import # must come before casacore
from casacore.tables import table
from dsautils import cnf

import dsacalib.constants as ct
from dsacalib.fringestopping import calc_uvw
from dsacalib.ms_io import get_antenna_gains, get_delays, read_caltable


def get_refmjd() -> float:
    conf = cnf.Conf()
    return conf.get("fringe")["refmjd"]


def get_config() -> "Config":
    """Retrieve antenna parameters from cnf."""
    ConfParams = namedtuple(
        "Config", "antennas antennas_core antennas_not_in_bf refants pols corr_list refcorr "
        "beamformer_dir")

    conf = cnf.Conf()
    corr_params = conf.get("corr")
    cal_params = conf.get("cal")

    antennas = np.array(list(corr_params["antenna_order"].values()))
    antennas_core = [ant for ant in antennas if ant < 100]
    antennas_not_in_bf = cal_params["antennas_not_in_bf"]
    refants = cal_params["refant"]

    pols = corr_params["pols_voltage"]

    corr_list = list(corr_params["ch0"].keys())
    refcorr = corr_list[0]
    corr_list = [int(cl.strip("corr")) for cl in corr_list]

    beamformer_dir = conf.get("cal")["beamformer_dir"]

    return ConfParams(
        antennas, antennas_core, antennas_not_in_bf, refants, pols, corr_list, refcorr,
        beamformer_dir)


def get_good_solution(
    select=None, plot=False, threshold_ants=45, threshold_angle=10, selectcore=True
):
    """Find good set of solutions and calculate average gains.
    TODO: go from good bfnames to average gains.
    """
    from datetime import datetime, timedelta, timezone

    config = get_config()

    if select is None:
        today = datetime.now(timezone.utc)
        select = [f"{today.year}-{today.month:02}-{today.day:02}"]
        today = today - timedelta(days=1)
        select += [f"{today.year}-{today.month:02}-{today.day:02}"]

    print(f"Selecting for string {select}")
    if isinstance(select, str):
        bfnames = get_bfnames(select=select)
    else:
        bfnames = get_bfnames(select=select[0])
        if len(select) > 1:
            for sel in select[1:]:
                bfnames += get_bfnames(select=sel)
    if not bfnames:
        return bfnames
    times = [bfname.split("_")[1] for bfname in bfnames]
    times, bfnames = zip(*sorted(zip(times, bfnames), reverse=True))
    if selectcore:
        gains = read_gains(bfnames, selectants=config.antennas_core)
    else:
        gains = read_gains(bfnames)
    good = find_good_solutions(
        bfnames,
        gains,
        plot=plot,
        threshold_ants=threshold_ants,
        threshold_angle=threshold_angle,
        selectcore=selectcore,
    )
    if plot:
        show_gains(bfnames, gains, good, selectcore=selectcore)

    return [bfnames[gidx] for gidx in good]


def get_bfnames(select=None):
    """Run on dsa-storage to get file names for beamformer weight files.

    Returns list of calibrator-times that are used in data file name.
    """
    config = get_config()

    # Use first corr to be robust against corr name changes
    fn_dat = glob.glob(
        os.path.join(config.beamformer_dir, f"beamformer_weights*{config.refcorr}*.dat"))

    bfnames = []
    for fn in fn_dat:
        sp = fn.split(f"_{config.refcorr}_")
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


def read_gains(bfnames, selectants=None, path=None):
    """Reads gain for each of the data files in bfnames.
    Returns gain array with same length as bfnames.
    path can overload the location of files assumed etcd value.
    """
    config = get_config()
    if selectants is None:
        selectants = config.antennas

    gains = np.zeros(
        (len(bfnames), len(config.antennas), len(config.corr_list), 48, 2), dtype=np.complex
    )

    if path is None:
        path = config.beamformer_dir

    for i, beamformer_name in enumerate(bfnames):
        for corridx, corr in enumerate(config.corr_list):
            with open(
                    f"{path}/beamformer_weights_corr{corr:02d}_{beamformer_name}.dat",
                    "rb",
            ) as f:
                data = np.fromfile(f, "<f4")
            temp = data[64:].reshape(64, 48, 2, 2)
            gains[i, :, corridx, :, :] = temp[..., 0] + 1.0j * temp[..., 1]
    gains = gains.reshape((len(bfnames), len(config.antennas), len(config.corr_list) * 48, 2))
    select = [config.antennas.tolist().index(i) for i in selectants]
    print(f"Using {len(bfnames)} to get gain array of shape {gains.shape}.")
    return gains.take(select, axis=1)


def find_good_solutions(
    bfnames,
    gains,
    threshold_ants=45,
    threshold_angle=10,
    mask_bothpols=True,
    plot=False,
    selectcore=True,
):
    """Given names and gain array, calc good set.
    Returns indices of bfnames argument that are good.
    """
    config = get_config()

    if selectcore:
        ants = config.antennas_core
    else:
        ants = config.antennas

    try:
        refant_ind = ants.index(int(config.refants[0]))
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
                    angle = np.angle(gains[k, j, :, i] / gains[k, refant_ind, :, i])
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
            print(f"{bfnames[k]} has {bad.count(k)} bad relative gains => {status_sol}")

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


def show_gains(bfnames, gains, keep, selectcore=True, ret=False, show=True):
    """Given bfnames and gains, plot the good gains.
    Default uses mpl show. Optionally can return figure with ret=True.
    """
    config = get_config()
    if selectcore:
        ants = config.antennas_core
    else:
        ants = config.antennas

    try:
        refant_ind = ants.index(int(config.refants[0]))
    except ValueError:
        refant_ind = 0
        print(f"Using first listed antenna ({ants[0]}) as refant")

    nants = len(ants)
    nx = 4
    ny = nants // nx
    if nants % nx > 0:
        ny += 1
    fig, ax = plt.subplots(ny, nx, figsize=(3 * nx, 4 * ny), sharex=True, sharey=True)
    ax[0, 0].set_yticks(np.arange(len(keep)))

    for axi in ax[0, :]:
        axi.set_yticklabels([bfn for i, bfn in enumerate(bfnames) if i in keep])
    ax = ax.flatten()

    for i in np.arange(nants):
        ax[i].imshow(
            np.angle(gains[:, i, :, 0] / gains[:, refant_ind, :, 0]).take(keep, axis=0),
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


def calc_eastings(antennas):
    """Calculate the eastings (u).

    Generates them for baselines composed of antennas in `antennas`.
    """
    antpos_df = get_itrf(
        latlon_center=(ct.OVRO_LAT * u.rad, ct.OVRO_LON * u.rad, ct.OVRO_ALT * u.m)
    )
    blen = np.zeros((len(antennas), 3))
    for i, ant in enumerate(antennas):
        blen[i, 0] = antpos_df["x_m"].loc[ant] - antpos_df["x_m"].loc[24]
        blen[i, 1] = antpos_df["y_m"].loc[ant] - antpos_df["y_m"].loc[24]
        blen[i, 2] = antpos_df["z_m"].loc[ant] - antpos_df["z_m"].loc[24]
    bu, _, _ = calc_uvw(blen, get_refmjd(), "HADEC", 0.0 * u.rad, 0.6 * u.rad)
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


def filter_beamformer_solutions(beamformer_names, start_time):
    """Removes beamformer solutions inconsistent with the latest solution."""
    config = get_config()
    beamformer_dir = config["beamformer_dir"]

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

    if solns['antenna_order'] != latest_solns['antenna_order'] or \
        solns['corr_order'] != latest_solns['corr_order'] or \
        solns['delays'] != latest_solns['delays'] or \
        solns['eastings'] != latest_solns['eastings'] or \
        solns['caltime'] < latest_solns['caltime']-1 or \
        solns['caltime'] < start_time:
        return False

    return True

def average_beamformer_solutions(
    fnames, ttime, corridxs=None, tol=0.3
):
    """Averages written beamformer solutions.

    Parameters
    ----------
    fnames : list
    ttime : astropy.time.Time object
        A time to use in the filename of the solutions, indicating when they
        were written or are useful. E.g. the transit time of the most recent
        source being averaged over.
    corridxs : list
        The correlator nodes for which to average beamformer solutions.
        Defaults to 1 through 16 inclusive.

    Returns
    -------
    written_files : list
        The names of the written beamformer solutions (one for each correlator
        node).
    antenna_flags_badsolns:
        Flags for antenna/polarization dimensions of gains.
    """
    config = get_config()
    beamformer_dir = config.beamformer_dir


    if corridxs is None:
        corridxs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    gains = read_gains(fnames)
    print(gains.shape)
    antenna_flags = [None] * len(fnames)
    for i, fname in enumerate(fnames):
        tmp_antflags = []
        filepath = f"{beamformer_dir}/beamformer_weights_{fname}.yaml"
        with open(filepath, encoding="utf-8") as f:
            calibration_params = yaml.load(f, Loader=yaml.FullLoader)["cal_solutions"]
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
    eastings = calc_eastings(antenna_order)

    gains = np.nanmean(gains, axis=0)
    fracflagged = np.sum(np.isnan(gains), axis=1) / (gains.shape[1])
    antenna_flags_badsolns = fracflagged > tol
    gains[np.isnan(gains)] = 0.0
    gains = gains.astype(np.complex64).view(np.float32).reshape(64, 16, 48, 2, 2)
    print(gains.shape)
    written_files = []
    if eastings is not None:
        for i, corr in enumerate(corridxs):
            fnameout = f"beamformer_weights_corr{corr:02d}_{ttime.isot}"
            wcorr = gains[:, i, ...].flatten()
            wcorr = np.concatenate([eastings, wcorr], axis=0)
            with open(f"{beamformer_dir}/{fnameout}.dat", "wb") as f:
                f.write(bytes(wcorr))
            written_files += [f"{fnameout}.dat"]
    return written_files, antenna_flags_badsolns


def write_beamformer_weights(
    msname, calname, caltime, antennas, corr_list, antenna_flags, tol=0.3
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
    corr_list : list
        The indices of the correlator machines to write beamformer weights for.
        For now, these must be ordered so that the frequencies are contiguous
        and they are in the same order or the reverse order as in the ms. The
        bandwidth of each correlator is pulled from dsa110-meridian-fs package
        data.
    antenna_flags : ndarray(bool)
        Dimensions (antennas, pols). True where flagged, False otherwise.
    tol : float
        The fraction of data for a single antenna/pol flagged that the
        can be flagged in the beamformer. If more data than this is flagged as
        having bad solutions, the entire antenna/pol pair is flagged.

    Returns
    -------
    corr_list : list
    bu : array
        The length of the baselines in the u direction for each antenna
        relative to antenna 24.
    fweights : ndarray
        The frequencies corresponding to the beamformer weights, dimensions
        (correlator, frequency).
    filenames : list
        The names of the file containing the beamformer weights.
    """
    ncorr = len(corr_list)
    weights = np.ones((ncorr, len(antennas), 48, 2), dtype=np.complex64)
    fweights = np.ones((ncorr, 48), dtype=np.float32)

    config = get_config()
    beamformer_dir = config["beamformer_dir"]

    conf = cnf.Conf()
    corr_params = conf.get('corr')
    nchan = corr_params["nchan"]
    dfreq = corr_params["bw_GHz"] / nchan
    if corr_params["chan_ascending"]:
        fobs = corr_params["f0_GHz"] + np.arange(nchan) * dfreq
    else:
        fobs = corr_params["f0_GHz"] - np.arange(nchan) * dfreq
    nchan_spw = corr_params["nchan_spw"]
    for i, corr_id in enumerate(corr_list):
        ch0 = corr_params["ch0"][f"corr{corr_id:02d}"]
        fobs_corr = fobs[ch0 : ch0 + nchan_spw]
        fweights[i, :] = fobs_corr.reshape(fweights.shape[1], -1).mean(axis=1)

    bu = calc_eastings(antennas)

    with table(f"{msname}.ms/SPECTRAL_WINDOW") as tb:
        fobs = np.array(tb.CHAN_FREQ[:]) / 1e9
    fobs = fobs.reshape(fweights.size, -1).mean(axis=1)
    f_reversed = not np.all(np.abs(fobs - fweights.ravel()) / fweights.ravel() < 1e-5)
    if f_reversed:
        assert np.all(np.abs(fobs[::-1] - fweights.ravel()) / fweights.ravel() < 1e-5)

    gains, _time, flags, ant1, ant2 = read_caltable(f"{msname}_{calname}_gacal", True)
    gains[flags] = np.nan
    gains = np.nanmean(gains, axis=1)
    phases, _, flags, ant1p, ant2p = read_caltable(f"{msname}_{calname}_gpcal", True)
    phases[flags] = np.nan
    phases = np.nanmean(phases, axis=1)
    assert np.all(ant1p == ant1)
    assert np.all(ant2p == ant2)
    gantenna, gains = get_antenna_gains(gains * phases, ant1, ant2)

    bgains, _, flags, ant1, ant2 = read_caltable(f"{msname}_{calname}_bcal", True)
    bgains[flags] = np.nan
    bgains = np.nanmean(bgains, axis=1)
    bantenna, bgains = get_antenna_gains(bgains, ant1, ant2)
    assert np.all(bantenna == gantenna)

    nantenna = gains.shape[0]
    npol = gains.shape[-1]

    gains = gains * bgains
    print(gains.shape)
    gains = gains.reshape((nantenna, -1, npol))
    if f_reversed:
        gains = gains[:, ::-1, :]
    gains = gains.reshape(nantenna, ncorr, -1, npol)
    nfint = gains.shape[2] // weights.shape[2]
    assert gains.shape[2] % weights.shape[2] == 0

    gains = np.nanmean(
        gains.reshape(gains.shape[0], gains.shape[1], -1, nfint, gains.shape[3]), axis=3
    )
    if not np.all(ant2 == ant2[0]):
        idxs = np.where(ant1 == ant2)
        gains = gains[idxs]
        ant1 = ant1[idxs]
    for i, antid in enumerate(ant1):
        if antid + 1 in antennas:
            idx = np.where(antennas == antid + 1)[0][0]
            weights[:, idx, ...] = gains[i, ...]

    fracflagged = np.sum(np.sum(np.isnan(weights), axis=2), axis=0) / (
        weights.shape[0] * weights.shape[2]
    )
    antenna_flags_badsolns = fracflagged > tol
    weights[np.isnan(weights)] = 0.0

    # Divide by the first non-flagged antenna
    idx0, idx1 = np.nonzero(np.logical_not(antenna_flags + antenna_flags_badsolns))
    weights = weights / weights[:, idx0[0], ..., idx1[0]][:, np.newaxis, :, np.newaxis]
    weights[np.isnan(weights)] = 0.0

    filenames = []
    for i, corr_idx in enumerate(corr_list):
        wcorr = weights[i, ...].view(np.float32).flatten()
        wcorr = np.concatenate([bu, wcorr], axis=0)
        fname = f"beamformer_weights_corr{corr_idx:02d}"
        fname = f"{fname}_{calname}_{caltime.isot}"
        if os.path.exists(f"{beamformer_dir}/{fname}.dat"):
            os.unlink(f"{beamformer_dir}/{fname}.dat")
        print(f"{beamformer_dir}/{fname}.dat")
        with open(f"{beamformer_dir}/{fname}.dat", "wb") as f:
            f.write(bytes(wcorr))
        filenames += [f"{fname}.dat".format(fname)]
    return corr_list, bu, fweights, filenames, antenna_flags_badsolns


def write_beamformer_solutions(
    msname,
    calname,
    caltime,
    antennas,
    applied_delays,
    corr_list=np.arange(1, 17),
    flagged_antennas=None,
    pols=None,
):
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
    corr_list : list
        The indices of the correlator machines to write beamformer weights for.
        For now, these must be ordered so that the frequencies are contiguous
        and they are in the same order or the reverse order as in the ms. The
        bandwidth of each correlator is pulled from dsa110-meridian-fs package
        data.
    flagged_antennas : list
        A list of antennas to flag in the beamformer solutions. Should include
        polarizations. e.g. ['24 B', '32 A']
    pols : list
        The order of the polarizations.

    Returns
    -------
    flags : ndarray(boolean)
        Dimensions (antennas, pols). True where the data is flagged, and should
        not be used. Compiled from the ms flags as well as `flagged_antennas`.
    """
    config = get_config()
    beamformer_dir = config.beamformer_dir

    if pols is None:
        pols = ["B", "A"]
    beamformer_flags = {}
    delays, flags = get_delays(antennas, msname, calname, applied_delays)
    print("delay flags:", flags.shape)
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
    (
        corr_list,
        eastings,
        _fobs,
        weights_files,
        flags_badsolns,
    ) = write_beamformer_weights(msname, calname, caltime, antennas, corr_list, flags)
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
            "corr_order": [int(corr) for corr in corr_list],
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
    return flags
