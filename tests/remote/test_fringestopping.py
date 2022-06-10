import datetime
import numpy as np
from astropy.time import Time
import astropy.units as u
from antpos.utils import get_itrf
from dsacalib.fringestopping import calc_uvw_interpolate, calc_uvw
import dsacalib.constants as ct

def test_calc_uvw_interpolate():
    ntimes = 100
    deltat = 2.*u.s/100.
    pt_dec = 70.*u.rad
    
    tobs = Time(datetime.datetime.utcnow()) + np.arange(ntimes)*deltat

    blen, bname = get_blen([1, 24, 35, 51, 100, 105, 115, 116])
    buvw = calc_uvw_interpolate(blen, tobs, 'HADEC', 0.*u.rad, pt_dec)
    buvw2 = calc_uvw(blen, tobs.mjd, 'HADEC', np.tile(0*u.rad, ntimes), np.tile(pt_dec, ntimes))
    buvw2 = np.array(buvw2).T
    assert buvw2.shape == buvw.shape
    assert np.allclose(buvw2, buvw)
    
def get_blen(antennas: list) -> tuple:
    """Gets the baseline lengths for a subset of antennas.

    Parameters
    ----------
    antennas : list
        The antennas used in the array.

    Returns
    -------
    blen : array
        The ITRF coordinates of all of the baselines.
    bname : list
        The names of all of the baselines.
    """
    ant_itrf = get_itrf(
        latlon_center=(ct.OVRO_LAT*u.rad, ct.OVRO_LON*u.rad, ct.OVRO_ALT*u.m)
    ).loc[antennas]
    xx = np.array(ant_itrf['dx_m'])
    yy = np.array(ant_itrf['dy_m'])
    zz = np.array(ant_itrf['dz_m'])

    nants = len(antennas)
    nbls = (nants*(nants+1))//2
    blen = np.zeros((nbls, 3))
    bname = []
    k = 0
    for i in range(nants):
        for j in range(i, nants):
            blen[k, :] = np.array([
                xx[i]-xx[j],
                yy[i]-yy[j],
                zz[i]-zz[j]
            ])
            bname += ['{0}-{1}'.format(
                antennas[i],
                antennas[j]
            )]
            k += 1
    return blen, bname