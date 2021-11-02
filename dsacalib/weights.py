import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy import time, coordinates, units
from dsautils import cnf

CONF = cnf.Conf()

CORR_PARAMS = CONF.get('corr')
CAL_PARAMS = CONF.get('cal')
MFS_PARAMS = CONF.get('fringe')
REFANTS = CAL_PARAMS['refant']
BEAMFORMER_DIR = CAL_PARAMS['beamformer_dir']
ANTENNAS = np.array(list(CORR_PARAMS['antenna_order'].values()))
POLS = CORR_PARAMS['pols_voltage']
ANTENNAS_NOT_IN_BF = CAL_PARAMS['antennas_not_in_bf']
CORR_LIST = list(CORR_PARAMS['ch0'].keys())
CORR_LIST = [int(cl.strip('corr')) for cl in CORR_LIST]


def get_bfnames(select=None):
    """ Run on dsa-storage to get file names for beamformer weight files.
    Returns list of calibrator-times that are used in data file name.
    """

    fn_dat = glob.glob(os.path.join(BEAMFORMER_DIR, 'beamformer_weights*corr07*.dat'))

    bfnames = []
    for fn in fn_dat:
        sp = fn.split('_corr07_')
        if len(sp) > 1:
            if '_' in sp[1]:
                bfnames.append(sp[1].rstrip('.dat'))
    bfnames = sorted(bfnames)

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
            with open('{0}/beamformer_weights_corr{1:02d}_{2}.dat'.format(BEAMFORMER_DIR,
                                                                          corr,
                                                                          beamformer_name),
                      'rb') as f:
                data = np.fromfile(f, '<f4')
            temp = data[64:].reshape(64, 48, 2, 2)
            gains[i, :, corridx, :, :] = temp[..., 0]+1.0j*temp[..., 1]
    gains = gains.reshape((len(bfnames), len(ANTENNAS), len(CORR_LIST)*48, 2))
    print(f'Using {len(bfnames)} to get gain array of shape {gains.shape}.')
    return gains


def find_good_solutions(bfnames, gains, threshold=60, plot=False):
    """ Given names and gain array, calc good set.
    Returns indices of bfnames argument that are good.
    """

    grads = np.zeros((len(ANTENNAS), len(bfnames)))
    for i in np.arange(len(ANTENNAS)):
        for j in np.arange(len(bfnames)):
            ss = gains[:, 0, :, 0].sum()
            if ss != 0.+0j:
                angle = np.angle(gains[:, i, :, 0]/gains[:, 0, :, 0])
                medgrad = np.median(np.gradient(angle[j]))
                grads[i, j] = medgrad
    grads = np.ma.masked_equal(grads, 0)
    grads = np.ma.masked_invalid(grads)

    # select good sets
    keep = []
    for j in np.arange(len(bfnames)):
        ngood = len(ANTENNAS)-sum(grads[:, j].mask)
        print(j, bfnames[j], ngood)
        if ngood > threshold:
            keep.append(j)
            print(f'{bfnames[j]}: good')
        else:
            print(f'{bfnames[j]}: rejected')

    if plot:
        # visualize grads
        plt.imshow(grads.transpose(), origin='lower')
        plt.xlabel('antenna')
        plt.ylabel('calibrator')
        plt.show()
        
    return keep


def show_gains(bfnames, gains, keep):
    """ Given bfnames and gains, plot the good gains.
    """
    
    refant = 24
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
        ax[i].imshow(np.angle(gains[:, i, :, 0]/gains[:, 0, :, 0]).take(keep, axis=0),
                     vmin=-np.pi, vmax=np.pi, aspect='auto', origin='lower', interpolation='None',
                     cmap=plt.get_cmap('RdBu'))
        ax[i].annotate('{0}'.format(ANTENNAS[i]), (0, 1), xycoords='axes fraction')
        ax[i].set_xlabel('Frequency channel')
    plt.show()


def get_good_solution(select=None, plot=False):
    """ Find good set of solutions and calculate average gains.
    TODO: go from good bfnames to average gains.
    """

    from datetime import date

    if select is None:
        today = date.today()
        select = f'{today.year}-{today.month}-{today.day:02}'

    print(f'Selecting for string {select}')
    bfnames = weights.get_bfnames(select=select)
    gains = weights.read_gains(bfnames)
    good = weights.find_good_solutions(bfnames, gains, plot=plot, threshold=60)
    if plot:
        weights.show_gains(bfnames, gains, good)

    return bfnames[good]
