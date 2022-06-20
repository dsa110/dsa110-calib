import socket
import astropy.units as u
from astropy.time import Time
from dsacalib.config import Configuration

LOCALTESTHOST = 'corr23'


def test_configuration():
    if not socket.gethostname() == LOCALTESTHOST:
        return
    config = Configuration()
    assert isinstance(config.refants, list)
    assert isinstance(config.antennas, list)
    assert isinstance(config.antennas_not_in_bf, list)
    assert isinstance(config.antennas_core, list)

    assert isinstance(config.ch0, dict)
    assert isinstance(config.corr_list, list)
    assert isinstance(config.ncorr, int)
    assert isinstance(config.pols, list)
    assert isinstance(config.refcorr, str)
    assert isinstance(config.nchan, int)
    assert isinstance(config.nchan_spw, int)
    assert isinstance(config.bw_GHz, float)
    assert isinstance(config.chan_ascending, bool)
    assert isinstance(config.f0_GHz, float)

    assert isinstance(config.nfreq_scrunch, int)
    assert isinstance(config.outrigger_delays, dict)
    assert isinstance(config.caltime, u.Quantity)
    assert isinstance(config.filelength, u.Quantity)
    assert isinstance(config.refmjd, float)
    assert isinstance(config.snap_start_time, Time)

    assert isinstance(config.msdir, str)
    assert isinstance(config.beamformer_dir, str)
    assert isinstance(config.hdf5dir, str)
    assert isinstance(config.webplots, str)
    assert isinstance(config.tempplots, str)
