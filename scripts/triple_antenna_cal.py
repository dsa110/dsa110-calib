import numpy as np
import astropy.units as u
from dsacalib.utils import *
from dsacalib.routines import triple_antenna_cal
from caltools import caltools

# Parameters that will need to be passed in or saved somehow 
datadir= '/home/dsa/data/'
cal    = src('M87','12h30m49.4233s','+12d23m28.043s',138.4870)
caltbl = caltools.list_calibrators(cal.ra,cal.dec,
                extent_info=True,radius=1/60)['NVSS']
cal.pa = caltbl[0]['position_angle']
cal.min_axis = caltbl[0]['minor_axis']
cal.maj_axis = caltbl[0]['major_axis']

obs_params = {'fname':'{0}/M87_1.fits'.format(datadir),
              'msname':'M87_1',
              'cal':cal,
              'utc_start':Time('2020-04-16T06:09:42')}

ant_params = {'pt_dec':cal.dec.to_value(u.rad),
              'antenna_order':[9,2,6],
              'refant':'2',
              'antpos':'/home/dsa/data/antpos_ITRF.txt'}

ptoffsets = {'dracosdec':(np.array([[0.61445538, 0.54614568], [0.23613347, 0.31217943], [0.24186434, 0.20372287]])*u.deg).to_value(u.rad),
             'rdec':(12.39*u.deg).to_value(u.rad),
             'ddec':(0*u.deg).to_value(u.rad)}

triple_antenna_cal(obs_params,ant_params)

caltable_to_etcd(obs_params['msname'],obs_params['cal'].name,
                 ant_params['antenna_order'],baseline_cal=True)

