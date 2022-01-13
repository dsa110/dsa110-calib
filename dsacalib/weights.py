import glob
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from antpos.utils import get_itrf
from casacore.tables import table
from dsautils import cnf
import dsacalib.constants as ct
from dsacalib.fringestopping import calc_uvw
from dsacalib.ms_io import read_caltable, get_antenna_gains, get_delays

CONF = cnf.Conf()

CORR_PARAMS = CONF.get('corr')
CAL_PARAMS = CONF.get('cal')
MFS_PARAMS = CONF.get('fringe')
REFANTS = CAL_PARAMS['refant']
BEAMFORMER_DIR = CAL_PARAMS['beamformer_dir']
ANTENNAS = np.array(list(CORR_PARAMS['antenna_order'].values()))
ANTENNAS = [ant for ant in ANTENNAS if ant < 100]  # core only
POLS = CORR_PARAMS['pols_voltage']
ANTENNAS_NOT_IN_BF = CAL_PARAMS['antennas_not_in_bf']
CORR_LIST = list(CORR_PARAMS['ch0'].keys())
REFCORR = CORR_LIST[0]
CORR_LIST = [int(cl.strip('corr')) for cl in CORR_LIST]
REFMJD = MFS_PARAMS['refmjd']


def get_good_solution(select=None, plot=False, threshold_ants=60, threshold_angle=10):
    """ Find good set of solutions and calculate average gains.
    TODO: go from good bfnames to average gains.
    """
    from datetime import datetime, timedelta, timezone

    if select is None:
        today = datetime.now(timezone.utc)
        select = [f'{today.year}-{today.month}-{today.day:02}']
        today = today-timedelta(days=1)
        select += [f'{today.year}-{today.month}-{today.day:02}']

    print(f'Selecting for string {select}')
    if isinstance(select, str):
        bfnames = get_bfnames(select=select)
    else:
        bfnames = get_bfnames(select=select[0])
        if len(select) > 1:
            for sel in select[1:]:
                bfnames += get_bfnames(select=sel)
    if not len(bfnames):
        return bfnames
    times = [bfname.split('_')[1] for bfname in bfnames]
    times, bfnames = zip(*sorted(zip(times, bfnames), reverse=True))
    gains = read_gains(bfnames)
    good = find_good_solutions(bfnames, gains, plot=plot, threshold_ants=threshold_ants, threshold_angle=threshold_angle)
    if plot:
        show_gains(bfnames, gains, good)

    return [bfnames[gidx] for gidx in good]


def get_bfnames(select=None):
    """ Run on dsa-storage to get file names for beamformer weight files.
    Returns list of calibrator-times that are used in data file name.
    """
    # Use first corr to be robust against corr name changes
    fn_dat = glob.glob(
        os.path.join(BEAMFORMER_DIR, f'beamformer_weights*{REFCORR}*.dat')
    )

    bfnames = []
    for fn in fn_dat:
        sp = fn.split(f'_{REFCORR}_')
        if len(sp) > 1:
            if '_' in sp[1]:
                bfnames.append(sp[1].rstrip('.dat'))

    # select subset
    if select is not None:
        bfnames2 = [bfn for bfn in bfnames if select in bfn]
        print(f'Selecting {len(bfnames2)} from {len(bfnames)} gain files.')
        return bfnames2
    else:
        print(f'Selecting all {len(bfnames)} gain files.')
        return bfnames


def read_gains(bfnames):
    """ Reads gain for each of the data files in bfnames.
    Returns gain array with same length as bfnames.
    """
    gains = np.zeros((len(bfnames), len(ANTENNAS), len(CORR_LIST), 48, 2),
                     dtype=np.complex)

    for i, beamformer_name in enumerate(bfnames):
        for corridx, corr in enumerate(CORR_LIST):
            with open(
                '{0}/beamformer_weights_corr{1:02d}_{2}.dat'.format(
                    BEAMFORMER_DIR,
                    corr,
                    beamformer_name),
                'rb'
            ) as f:
                data = np.fromfile(f, '<f4')
            temp = data[64:].reshape(64, 48, 2, 2)
            gains[i, :, corridx, :, :] = temp[..., 0]+1.0j*temp[..., 1]
    gains = gains.reshape((len(bfnames), len(ANTENNAS), len(CORR_LIST)*48, 2))
    print(f'Using {len(bfnames)} to get gain array of shape {gains.shape}.')
    return gains


def find_good_solutions(bfnames, gains, threshold_ants=60, threshold_angle=10, mask_bothpols=True, plot=False):
    """ Given names and gain array, calc good set.
    Returns indices of bfnames argument that are good.
    """

    refant_ind = ANTENNAS.tolist().index(int(REFANTS[0]))
    nchan = gains.shape[2]
    grads = np.zeros((len(bfnames), len(ANTENNAS), 2))
    for i in np.arange(2):
        for j in np.arange(len(ANTENNAS)):
#            angles = np.zeros((len(bfnames), nchan))
            for k in np.arange(len(bfnames)):
                if gains[k, j, :, i].any():
#                    angle = np.angle(gains[k, j, :, i]/gains[0, j, :, i])
                    angle = np.angle(gains[k, j, :, i]/gains[k, refant_ind, :, i])
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
        maskor = np.ma.mask_or(grads.mask[:,:,0], grads.mask[:,:,1])
        grads.mask[:,:,0] = maskor
        grads.mask[:,:,1] = maskor

    # select good sets
    keep = []
    for k in np.arange(len(bfnames)):
        ngood = len(ANTENNAS)-sum(grads[k, :, 0].mask)  # assumes pol=0 has useful flagging info

        print(f'Good phase gradients: {k}, {bfnames[k]}, {ngood} antennas')
        if ngood >= threshold_ants:
            keep.append(k)
            print(f'{bfnames[k]}: good')
        else:
            print(f'{bfnames[k]}: rejected')

    gains = np.ma.masked_array(gains)
    for i in range(nchan):
        gains[...,i,:].mask = grads.mask

    # calc relative phase change across all pairs of solutions to find outliers
    #    for i in np.arange(2):
    bad = []
    for k in np.arange(len(bfnames)):
        for j in np.arange(k+1, len(bfnames)):
            if k not in keep or j not in keep:
                continue
            angle = np.degrees(np.angle(gains[k, :, :, 0]/gains[j, :, :, 0]).mean())
            if np.abs(angle) >= threshold_angle:
                bad.append(k)
                bad.append(j)
                status_pair = 'bad'
            else:
                status_pair = 'good'
            print(f'Rel phase for cal pairs ({k}, {j}), ({bfnames[k]}, {bfnames[j]}): {angle} degrees => {status_pair}')
    if len(bad):
        keepcount = len(keep)
        for k in np.arange(len(bfnames)):
            if k not in keep:
                continue
            if bad.count(k) >= keepcount - 1:
                keep.remove(k)
                status_sol = 'reject'
            else:
                status_sol = 'keep'
            print(f'{bfnames[k]} has {bad.count(k)} bad relative gains => {status_sol}')
            
    if plot:
        # visualize grads
        pl, (ax0, ax1) = plt.subplots(1,2)
        ax0.imshow(grads[:,:,0], origin='lower')
        ax0.set_xlabel('antenna (pol 0)')
        ax0.set_ylabel('calibrator')
        ax1.imshow(grads[:,:,1], origin='lower')
        ax1.set_xlabel('antenna (pol 1)')
        ax1.set_ylabel('calibrator')
        plt.show()

    return keep


def show_gains(bfnames, gains, keep):
    """ Given bfnames and gains, plot the good gains.
    """

    refant_ind = ANTENNAS.tolist().index(int(REFANTS[0]))
    nx = 4
    ny = len(ANTENNAS)//nx
    if len(ANTENNAS)%nx > 0:
        ny += 1
    _, ax = plt.subplots(ny, nx, figsize=(3*nx, 4*ny),
                         sharex=True, sharey=True)
    ax[0, 0].set_yticks(np.arange(len(bfnames)))

    for axi in ax[0, :]:
        axi.set_yticklabels(bfnames)
    ax = ax.flatten()

    for i in np.arange(len(ANTENNAS)):
        ax[i].imshow(
            np.angle(gains[:, i, :, 0]/gains[:, refant_ind, :, 0]).take(keep, axis=0),
            vmin=-np.pi, vmax=np.pi, aspect='auto', origin='lower',
            interpolation='None', cmap=plt.get_cmap('RdBu')
        )
        ax[i].annotate('{0}'.format(ANTENNAS[i]), (0, 1), xycoords='axes fraction')
        ax[i].set_xlabel('Frequency channel')
    plt.show()


def calc_eastings(antennas):
    antpos_df = get_itrf(
        latlon_center=(ct.OVRO_LAT*u.rad, ct.OVRO_LON*u.rad, ct.OVRO_ALT*u.m)
    )
    blen = np.zeros((len(antennas), 3))
    for i, ant in enumerate(antennas):
        blen[i, 0] = antpos_df['x_m'].loc[ant]-antpos_df['x_m'].loc[24]
        blen[i, 1] = antpos_df['y_m'].loc[ant]-antpos_df['y_m'].loc[24]
        blen[i, 2] = antpos_df['z_m'].loc[ant]-antpos_df['z_m'].loc[24]
    bu, _, _ = calc_uvw(blen, REFMJD, 'HADEC', 0.*u.rad, 0.6*u.rad)
    bu = bu.squeeze().astype(np.float32)
    return bu


def filter_beamformer_solutions(beamformer_names, start_time):
    """Removes beamformer solutions inconsistent with the latest solution.
    """
    if os.path.exists(
        '{0}/beamformer_weights_{1}.yaml'.format(
            BEAMFORMER_DIR,
            beamformer_names[0]
        )
    ):
        with open(
            '{0}/beamformer_weights_{1}.yaml'.format(
                BEAMFORMER_DIR,
                beamformer_names[0]
            )
        ) as f:
            latest_solns = yaml.load(f, Loader=yaml.FullLoader)
        for bfname in beamformer_names[1:].copy():
            try:
                with open(
                    '{0}/beamformer_weights_{1}.yaml'.format(
                        BEAMFORMER_DIR,
                        bfname
                    )
                ) as f:
                    solns = yaml.load(f, Loader=yaml.FullLoader)
                assert solns['cal_solutions']['antenna_order'] == \
                    latest_solns['cal_solutions']['antenna_order']
                assert solns['cal_solutions']['corr_order'] == \
                    latest_solns['cal_solutions']['corr_order']
                assert solns['cal_solutions']['delays'] == \
                    latest_solns['cal_solutions']['delays']
                assert solns['cal_solutions']['eastings'] == \
                    latest_solns['cal_solutions']['eastings']
                assert solns['cal_solutions']['caltime'] > \
                    latest_solns['cal_solutions']['caltime']-1
                assert solns['cal_solutions']['caltime'] > start_time
            except (AssertionError, FileNotFoundError):
                beamformer_names.remove(bfname)
    else:
        beamformer_names = []
        latest_solns = None
    return beamformer_names, latest_solns

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
    if corridxs is None:
        corridxs = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
        ]
    gains = read_gains(fnames)
    print(gains.shape)
    antenna_flags = [None]*len(fnames)
    for i, fname in enumerate(fnames):
        tmp_antflags = []
        filepath = f'{BEAMFORMER_DIR}/beamformer_weights_{fname}.yaml'
        with open(filepath) as f:
            calibration_params = yaml.load(
                f, Loader=yaml.FullLoader
            )['cal_solutions']
            antenna_order = calibration_params['antenna_order']
            for key in calibration_params['flagged_antennas']:
                if 'casa solutions flagged' in \
                    calibration_params['flagged_antennas'][key]:
                    antname = int(key.split(' ')[0])
                    tmp_antflags.append(antenna_order.index(antname))
        antenna_flags[i] = sorted(tmp_antflags)
        gains[i, antenna_flags[i], ... ] = np.nan
    eastings = calc_eastings(antenna_order)
    
    gains = np.nanmean(gains, axis=0)
    fracflagged = np.sum(np.isnan(gains), axis=1)/(gains.shape[1])
    antenna_flags_badsolns = fracflagged > tol
    gains[np.isnan(gains)] = 0.
    gains = gains.astype(np.complex64).view(np.float32).reshape(64, 16, 48, 2, 2)
    print(gains.shape)
    written_files = []
    if eastings is not None:
        for i, corr in enumerate(corridxs):
            fnameout = f'beamformer_weights_corr{corr:02d}_{ttime.isot}'
            wcorr = gains[:, i, ...].flatten()
            wcorr = np.concatenate([eastings, wcorr], axis=0)
            with open(f'{BEAMFORMER_DIR}/{fnameout}.dat', 'wb') as f:
                f.write(bytes(wcorr))
            written_files += ['{0}.dat'.format(fnameout)]
    return written_files, antenna_flags_badsolns

def write_beamformer_weights(msname, calname, caltime, antennas,
                             corr_list, antenna_flags, tol=0.3):
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
    # Get the frequencies we want to write solutions for.
    # corr_settings = resource_filename("dsamfs", "data/dsa_parameters.yaml")
    # params = yaml.safe_load(fhand)
    ncorr = len(corr_list)
    weights = np.ones((ncorr, len(antennas), 48, 2), dtype=np.complex64)
    fweights = np.ones((ncorr, 48), dtype=np.float32)
    nchan = CORR_PARAMS['nchan']
    dfreq = CORR_PARAMS['bw_GHz']/nchan
    if CORR_PARAMS['chan_ascending']:
        fobs = CORR_PARAMS['f0_GHz']+np.arange(nchan)*dfreq
    else:
        fobs = CORR_PARAMS['f0_GHz']-np.arange(nchan)*dfreq
    nchan_spw = CORR_PARAMS['nchan_spw']
    for i, corr_id in enumerate(corr_list):
        ch0 = CORR_PARAMS['ch0']['corr{0:02d}'.format(corr_id)]
        fobs_corr = fobs[ch0:ch0+nchan_spw]
        fweights[i, :] = fobs_corr.reshape(
            fweights.shape[1],
            -1
        ).mean(axis=1)

    bu = calc_eastings(antennas)

    with table('{0}.ms/SPECTRAL_WINDOW'.format(msname)) as tb:
        fobs = np.array(tb.CHAN_FREQ[:])/1e9
    fobs = fobs.reshape(fweights.size, -1).mean(axis=1)
    f_reversed = not np.all(
        np.abs(fobs-fweights.ravel())/fweights.ravel() < 1e-5
    )
    if f_reversed:
        assert np.all(
            np.abs(fobs[::-1]-fweights.ravel())/fweights.ravel() < 1e-5
        )

    gains, _time, flags, ant1, ant2 = read_caltable(
        '{0}_{1}_gacal'.format(msname, calname), True)
    gains[flags] = np.nan
    gains = np.nanmean(gains, axis=1)
    phases, _, flags, ant1p, ant2p = read_caltable(
        '{0}_{1}_gpcal'.format(msname, calname), True)
    phases[flags] = np.nan
    phases = np.nanmean(phases, axis=1)
    assert np.all(ant1p == ant1)
    assert np.all(ant2p == ant2)
    gantenna, gains = get_antenna_gains(gains*phases, ant1, ant2)

    bgains, _, flags, ant1, ant2 = read_caltable(
        '{0}_{1}_bcal'.format(msname, calname), True)
    bgains[flags] = np.nan
    bgains = np.nanmean(bgains, axis=1)
    bantenna, bgains = get_antenna_gains(bgains, ant1, ant2)
    assert np.all(bantenna == gantenna)

    nantenna = gains.shape[0]
    npol = gains.shape[-1]

    gains = gains*bgains
    print(gains.shape)
    gains = gains.reshape(nantenna, -1, npol)
    if f_reversed:
        gains = gains[:, ::-1, :]
    gains = gains.reshape(nantenna, ncorr, -1, npol)
    nfint = gains.shape[2]//weights.shape[2]
    assert gains.shape[2]%weights.shape[2]==0

    gains = np.nanmean(
        gains.reshape(
            gains.shape[0], gains.shape[1], -1, nfint, gains.shape[3]
        ), axis=3
    )
    if not np.all(ant2==ant2[0]):
        idxs = np.where(ant1==ant2)
        gains = gains[idxs]
        ant1 = ant1[idxs]
    for i, antid in enumerate(ant1):
        if antid+1 in antennas:
            idx = np.where(antennas==antid+1)[0][0]
            weights[:, idx, ...] = gains[i, ...]

    fracflagged = np.sum(np.sum(np.isnan(weights), axis=2), axis=0)\
        /(weights.shape[0]*weights.shape[2])
    antenna_flags_badsolns = fracflagged > tol
    weights[np.isnan(weights)] = 0.

    # Divide by the first non-flagged antenna
    idx0, idx1 = np.nonzero(
        np.logical_not(
            antenna_flags + antenna_flags_badsolns
        )
    )
    weights = (
        weights/weights[:, idx0[0], ..., idx1[0]][:, np.newaxis, :, np.newaxis]
    )
    weights[np.isnan(weights)] = 0.

    filenames = []
    for i, corr_idx in enumerate(corr_list):
        wcorr = weights[i, ...].view(np.float32).flatten()
        wcorr = np.concatenate([bu, wcorr], axis=0)
        fname = 'beamformer_weights_corr{0:02d}'.format(corr_idx)
        fname = '{0}_{1}_{2}'.format(
            fname,
            calname,
            caltime.isot
        )
        if os.path.exists(f'{BEAMFORMER_DIR}/{fname}.dat'):
            os.unlink(f'{BEAMFORMER_DIR}/{fname}.dat')
        with open(f'{BEAMFORMER_DIR}/{fname}.dat', 'wb') as f:
            f.write(bytes(wcorr))
        filenames += ['{0}.dat'.format(fname)]
    return corr_list, bu, fweights, filenames, antenna_flags_badsolns

def write_beamformer_solutions(
    msname, calname, caltime, antennas, applied_delays,
    corr_list=np.arange(1, 17),
    flagged_antennas=None,
    pols=None
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
    if pols is None:
        pols = ['B', 'A']
    beamformer_flags = {}
    delays, flags = get_delays(antennas, msname, calname, applied_delays)
    print('delay flags:', flags.shape)
    if flagged_antennas is not None:
        for item in flagged_antennas:
            ant, pol = item.split(' ')
            flags[antennas==ant, pols==pol] = 1
            beamformer_flags['{0} {1}'.format(ant, pol)] = ['flagged by user']
    delays = delays-np.min(delays[~flags])
    # TODO: this is currently flagging antennas that it shouldn't be
    #while not np.all(delays[~flags] < 1024):
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
    corr_list, eastings, _fobs, weights_files, flags_badsolns = \
        write_beamformer_weights(msname, calname, caltime, antennas,
        corr_list, flags)
    idxant, idxpol = np.nonzero(flags_badsolns)
    for i, ant in enumerate(idxant):
        key = '{0} {1}'.format(antennas[ant], pols[idxpol[i]])
        if key not in beamformer_flags.keys():
            beamformer_flags[key] = []
        beamformer_flags[key] += ['casa solutions flagged']

    calibration_dictionary = {
        'cal_solutions':
        {
            'source': calname,
            'caltime': float(caltime.mjd),
            'antenna_order': [int(ant) for ant in antennas],
            'corr_order': [int(corr) for corr in corr_list],
            'pol_order': ['B', 'A'],
            'delays': [
                [
                    int(delay[0]//2),
                    int(delay[1]//2)
                ] for delay in delays
            ],
            'eastings': [float(easting) for easting in eastings],
            'weights_axis0': 'antenna',
            'weights_axis1': 'frequency',
            'weights_axis2': 'pol',
            'weight_files': weights_files,
            'flagged_antennas': beamformer_flags
        }
    }

    with open(
        f'{BEAMFORMER_DIR}/beamformer_weights_{calname}_{caltime.isot}.yaml',
        'w'
    ) as file:
        yaml.dump(calibration_dictionary, file)
    return flags
