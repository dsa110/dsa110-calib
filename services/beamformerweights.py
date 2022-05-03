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

ANTENNAS_PLOT = list(CORR_PARAMS['antenna_order'].values())
ANTENNAS = ANTENNAS_PLOT
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
            ant = int(key.split(' ')[0])
            if ant in ANTENNAS:
                antenna_flags[
                   ANTENNAS.index(ant)
                ] = 1
        antenna_flags = np.where(antenna_flags)[0]
        with open('antenna_flags.txt', 'w', encoding="utf-8") as f:
            f.write('\n'.join([str(af) for af in antenna_flags]))
            f.write('\n')
        tstamp = Time(datetime.datetime.utcnow())
        tstamp.precision = 0
        with open(
                f"{BFARCHIVEDIR}/beamformer_weights_{tstamp.isot}.yaml",
                "w",
                encoding="utf-8"
        ) as file:
            _ = yaml.dump(bfsolns, file)
        for i, corr in enumerate(CORR_LIST):
            fname = f"{BFDIR}/{bfsolns['weight_files'][i]}"
            fnamearchive = f"{BFARCHIVEDIR}/beamformer_weights_corr{corr:02d}_{tstamp.isot}.dat"
            fnameout = f"corr{corr:02d}.sas.pvt:{WEIGHTFILE}"
            flagsout = f"corr{corr:02d}.sas.pvt:{FLAGFILE}"
            rsync_file(
                f"{fname} {fnameout}",
                remove_source_files=False
            )
            rsync_file(
                f"antenna_flags.txt {flagsout}",
                remove_source_files=False
            )
            shutil.copyfile(fname, fnamearchive)
        LOGGER.info(
            f"Updated beamformer weights using {bfsolns['weight_files']}"
        )

if __name__ == "__main__":
    ETCD.add_watch('/mon/cal/bfweights', update_beamformer_weights)
    while True:
        ETCD.put_dict(
            '/mon/service/bfweightcopy',
            {
                "cadence": 60,
                "time": Time(datetime.datetime.utcnow()).mjd
            }
        )
        time.sleep(60)
