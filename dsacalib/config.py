import astropy.units as u
import numpy as np

from dsautils import cnf


class Configuration:
    """Configuration for dsacalib."""

    def __init__(self):
        """Set parameters for conf."""
        dsaconf = cnf.Conf()

        cal_params = dsaconf.get("cal")
        corr_params = dsaconf.get("corr")
        mfs_params = dsaconf.get("fringe")

        self.refants = cal_params["refant"]
        self.antennas = np.array(list(corr_params["antenna_order"].values))  # in plotting
        self.nfreq_scrunch = mfs_params["nfreq_scrunch"]
        self.outrigger_delays = mfs_params["outrigger_delays"]
        self.ch0 = corr_params['ch0']
        self.corr_list = list(self.ch0.keys())
        self.ncorr = len(self.corr_list)
        self.caltime = cal_params['caltime_minutes'] * u.min
        self.filelength = mfs_params['filelength_minutes'] * u.min
        self.hdf5dir = cal_params['hdf5_dir']
        self.antennas_not_in_bf = cal_params['antennas_not_in_bf']
        self.refmjd = mfs_params['refmjd']
        self.antennas_core = [ant for ant in self.antennas if ant < 100]
        self.refcorr = self.corr_list[0]
        self.beamformer_dir = cal_params['beamformer_dir']
        self.pols = corr_params['pols_voltage']
