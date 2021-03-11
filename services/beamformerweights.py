"""Service for updating beamformer weights.
"""

import shutil
import datetime
import warnings
import time
import yaml
import numpy as np
from astropy.time import Time
import dsautils.dsa_store as ds
import dsautils.dsa_syslog as dsl
import dsautils.cnf as dsc
from dsacalib.preprocess import rsync_file
warnings.filterwarnings("ignore")

# Logger
LOGGER = dsl.DsaSyslogger()
LOGGER.subsystem("software")
LOGGER.app("dsacalib")

# ETCD interface
ETCD = ds.DsaStore()

CONF = dsc.Conf()
CORR_PARAMS = CONF.get('corr')
CAL_PARAMS = CONF.get('cal')
CORR_LIST = list(CORR_PARAMS['ch0'].keys())
CORR_LIST = [int(cl.strip('corr')) for cl in CORR_LIST]

ANTENNAS_PLOT = np.array(list(CORR_PARAMS['antenna_order'].values()))
ANTENNAS = list(np.concatenate((
    ANTENNAS_PLOT,
    np.arange(36, 36+39))
))
BFDIR = CAL_PARAMS['beamformer_dir']
WEIGHTFILE = CAL_PARAMS['weightfile']
FLAGFILE = CAL_PARAMS['flagfile']
BFARCHIVEDIR = CAL_PARAMS['bfarchivedir']

def update_beamformer_weights(etcd_dict):
    """Updates beamformer weights and antenna flags on core machines.

    Also archives the beamformer weights in /mnt/data/dsa110/T3/calibs/
    """
    cmd = etcd_dict['cmd']
    val = etcd_dict['val']

    if cmd == 'update_weights':
        bfsolns = val
        # Put antenna flags in the way needed by the bf
        antenna_flags = np.zeros((len(ANTENNAS)), np.int)
        for key in bfsolns['flagged_antennas']:
            antenna_flags[
               ANTENNAS.index(int(key.split(' ')[0]))
            ] = 1
        antenna_flags = np.where(antenna_flags)[0]
        with open('antenna_flags.txt', 'w') as f:
            f.write('\n'.join([str(af) for af in antenna_flags]))
            f.write('\n')
        tstamp = Time(datetime.datetime.utcnow())
        tstamp.precision = 0
        with open(
            '{0}/beamformer_weights_{1}.yaml'.format(BFARCHIVEDIR, tstamp.isot), 'w'
        ) as file:
            _ = yaml.dump(bfsolns, file)
        for i, corr in enumerate(CORR_LIST):
            fname = '{0}/{1}'.format(
                BFDIR,
                bfsolns['weight_files'][i]
            )
            fnamearchive = '{0}/beamformer_weights_corr{1:02d}_{2}.dat'.format(
                BFARCHIVEDIR,
                corr,
                tstamp.isot
            )
            fnameout = 'corr{0:02d}.sas.pvt:{1}'.format(
                corr,
                WEIGHTFILE
            )
            flagsout = 'corr{0:02d}.sas.pvt:{1}'.format(
                corr,
                FLAGFILE
            )
            rsync_file(
                '{0} {1}'.format(fname, fnameout),
                remove_source_files=False
            )
            rsync_file(
                'antenna_flags.txt {0}'.format(flagsout),
                remove_source_files=False
            )
            shutil.copyfile(fname, fnamearchive)
        LOGGER.info(
            'Updated beamformer weights using {0}'.format(
                bfsolns['weight_files']
            )
        )

if __name__ == "__main__":
    ETCD.add_watch('/cmd/corr/1/bf', update_beamformer_weights)
    while True:
        ETCD.put_dict(
            '/mon/corr/1/bf',
            {
                "alive": True,
                "cadence": 60,
                "time": Time(datetime.datetime.utcnow()).isot
            }
        )
        time.sleep(60)
