import socket
import astropy.units as u
from astropy.time import Time

from dsautils import cnf, dsa_store


class Configuration:
    """Configuration for dsacalib."""

    def __init__(self):
        """Set parameters for conf."""
        dsaconf = cnf.Conf()
        etcd = dsa_store.DsaStore()

        cal_params = dsaconf.get('cal')
        corr_params = dsaconf.get('corr')
        mfs_params = dsaconf.get('fringe')

        # Antennas
        self.refants = cal_params['refant']
        self.antennas = list(corr_params['antenna_order'].values())
        self.antennas_not_in_bf = cal_params['antennas_not_in_bf']
        self.antennas_core = [ant for ant in self.antennas if ant < 100]

        # Correlator
        self.ch0 = corr_params['ch0']
        self.corr_list = list(self.ch0.keys())
        self.ncorr = len(self.corr_list)
        self.pols = corr_params['pols_voltage']
        self.refcorr = self.corr_list[0]
        self.nchan = corr_params['nchan']
        self.nchan_spw = corr_params['nchan_spw']
        self.bw_GHz = corr_params['bw_GHz']
        self.chan_ascending = corr_params['chan_ascending']
        self.f0_GHz = corr_params['f0_GHz']

        # Correlator & Fringestopping system
        self.nfreq_scrunch = mfs_params['nfreq_scrunch']
        self.outrigger_delays = mfs_params['outrigger_delays']
        self.caltime = cal_params['caltime_minutes'] * u.min
        self.filelength = mfs_params['filelength_minutes'] * u.min
        self.refmjd = mfs_params['refmjd']
        self.snap_start_time = Time(
            etcd.get_dict("/mon/snap/1/armed_mjd")['armed_mjd'], format="mjd")
        # Directories
        self.msdir = cal_params['msdir']
        self.beamformer_dir = cal_params['beamformer_dir']
        self.hdf5dir = cal_params['hdf5_dir']
        self.webplots = "/mnt/data/dsa110/webPLOTS/calibration/"
        self.tempplots = (
            "/home/user/temp" if socket.gethostname() == "dsa-storage"
            else "/home/ubuntu/caldata/temp")

    def __repr__(self):
        string_repr = "Configuration:\n" + "\n".join([f"{key}: {value}" for key, value in self.__dict__.items()])
        return string_repr

