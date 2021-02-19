import pytest

import datetime
import numpy as np
from antpos.utils import get_itrf, get_baselines
import scipy
import casatools as cc
from casacore.tables import table
from dsacalib import constants
from dsacalib import utils
from dsacalib.fringestopping import calc_uvw
from dsacalib import ms_io as msio
import astropy.units as u
from astropy.utils import iers
iers.conf.iers_auto_url_mirror = constants.IERS_TABLE
iers.conf.auto_max_age = None
from astropy.time import Time

def test_simulate_ms(tmpdir):
    ntint = 32*32*4
    nfint = 8*8*4
    antennas = np.array([24, 25, 26])
    blen_df = get_baselines(antennas[::-1], autocorrs=True, casa_order=True)
    blen = np.array([blen_df['x_m'], blen_df['y_m'], blen_df['z_m']]).T
    ant_itrf = get_itrf().loc[antennas]
    xx = ant_itrf['dx_m']
    yy = ant_itrf['dy_m']
    zz = ant_itrf['dz_m']
    antenna_names = [str(a) for a in antennas]
    top_of_channel = 1.53*u.GHz + (-250*u.MHz/8192)*1024
    deltaf = -0.030517578125*u.MHz
    nchan = 6144//nfint
    fobs = top_of_channel + deltaf*nfint*(np.arange(nchan)+0.5)
    tstart = Time(datetime.datetime.utcnow())
    source = utils.src(
        name='3C273',
        ra=187.27916667*u.deg,
        dec=2.0625*u.deg
    )
    me = cc.measures()
    msio.simulate_ms(
        ofile='{0}/test.ms'.format(tmpdir),
        tname='OVRO_MMA',
        anum=antenna_names,
        xx=xx,
        yy=yy,
        zz=zz,
        diam=4.5,
        mount='alt-az',
        pos_obs=me.observatory('OVRO_MMA'),
        spwname='L_BAND',
        freq='{0}GHz'.format(fobs[0].to_value(u.GHz)),
        deltafreq='{0}MHz'.format((deltaf*nfint).to_value(u.MHz)),
        freqresolution='{0}MHz'.format(np.abs((deltaf*nfint).to_value(u.MHz))),
        nchannels=6144//nfint,
        integrationtime='{0}s'.format(0.000032768*ntint),
        obstm=tstart.mjd,
        dt=0.000429017462010961+7.275957614183426e-12-7.767375791445374e-07,
        source=source,
        stoptime='{0}s'.format(0.000032768*122880),
        autocorr=True, 
        fullpol=True
    )
    with table('{0}/test.ms/POLARIZATION'.format(tmpdir)) as tb:
        assert np.all(tb.CORR_TYPE[:][0] == np.array([9, 12, 10, 11]))
        assert np.all(
            tb.CORR_PRODUCT[:][0] == np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
        )
        assert tb.NUM_CORR[:][0] == 4
    with table('{0}/test.ms'.format(tmpdir)) as tb:
        ant1 = np.array(tb.ANTENNA1[:])
        ant2 = np.array(tb.ANTENNA2[:])
        tobs = np.array(tb.TIME[:])/constants.SECONDS_PER_DAY
        uvw = np.array(tb.UVW[:])
    with table('{0}/test.ms/SPECTRAL_WINDOW'.format(tmpdir)) as tb:
        fobs_out = np.array(tb.CHAN_FREQ[:])
    tobs = tobs.reshape(-1, 6)
    assert np.all(np.abs(tobs[0, :]-tobs[0, 0]) < 1e-15)
    assert np.abs(tobs[0, 0]-(tstart.mjd+0.000032768*ntint/2/constants.SECONDS_PER_DAY)) < 1e-11
    assert tobs.shape[0] == 122880//ntint
    assert fobs_out.shape == (1, 6144//nfint)
    assert np.all(np.abs(np.diff(fobs_out)-(-0.030517578125*nfint*1e6)) < 1e-15)
    assert np.all(np.abs(fobs_out[0, :]-fobs.to_value(u.Hz)) < 1e-15)
    bu, bv, bw = calc_uvw(
        blen,
        tobs.reshape(-1, 6)[:, 0],
        source.epoch,
        source.ra,
        source.dec
    )
    # Note that antpos gives A* B, casa gives A B*
    # Need to confirm which order we are doing
    assert np.all(np.abs(
        uvw[:, 0].reshape(-1, 6) - bu.T) < 1e-8)
    assert np.all(np.abs(
        uvw[:, 1].reshape(-1, 6) - bv.T) < 1e-8)
    assert np.all(np.abs(
        uvw[:, 2].reshape(-1, 6) - bw.T) < 1e-8)
