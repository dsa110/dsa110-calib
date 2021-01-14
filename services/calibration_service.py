"""A service to create measurement sets and calibrate data.
"""

import warnings
# make sure warnings do not spam syslog
warnings.filterwarnings("ignore")
import datetime # pylint: disable=wrong-import-position
import time # pylint: disable=wrong-import-position
import matplotlib # pylint: disable=wrong-import-position
matplotlib.use('Agg')
import matplotlib.pyplot as plt # pylint: disable=wrong-import-position
from matplotlib.backends.backend_pdf import PdfPages # pylint: disable=wrong-import-position
from pkg_resources import resource_filename # pylint: disable=wrong-import-position
import numpy as np # pylint: disable=wrong-import-position
import astropy.units as u # pylint: disable=wrong-import-position
from astropy.time import Time # pylint: disable=wrong-import-position
import dsautils.dsa_store as ds # pylint: disable=wrong-import-position
import dsautils.dsa_syslog as dsl # pylint: disable=wrong-import-position
from dsacalib.preprocess import first_true # pylint: disable=wrong-import-position
from dsacalib.utils import exception_logger # pylint: disable=wrong-import-position
from dsacalib.routines import get_files_for_cal, calibrate_measurement_set # pylint: disable=wrong-import-position
from dsacalib.ms_io import convert_calibrator_pass_to_ms, caltable_to_etcd # pylint: disable=wrong-import-position
from dsacalib.plotting import summary_plot, plot_current_beamformer_solutions # pylint: disable=wrong-import-position

# Logger
LOGGER = dsl.DsaSyslogger()
LOGGER.subsystem("software")
LOGGER.app("dsacalib")

# ETCD interface
ETCD = ds.DsaStore()

# These should be put somewhere else eventually
CALTIME = 15*u.min
REFANT = '102'

def calibrate_file(etcd_dict):
    """Generates and calibrates a measurement set.
    """
    cmd = etcd_dict['cmd']
    val = etcd_dict['val']
    if cmd == 'calibrate':
        calname = val['calname']
        flist = val['flist']
        date = first_true(flist).split('/')[-1][:-14]
        caltable = resource_filename('dsacalib', 'data/calibrator_sources.csv')

        # Parameters that are mostly constant
        refcorr = '01'
        filelength = 15*u.min
        msdir = '/mnt/data/dsa110/calibration/'
        msname = '{0}/{1}_{2}'.format(msdir, date, calname)
        date_specifier = '{0}*'.format(date)

        LOGGER.info('Creating {0}.ms'.format(msname))

        filenames = get_files_for_cal(
            caltable,
            refcorr,
            CALTIME,
            filelength,
            date_specifier=date_specifier,
        )

        ETCD.put_dict(
            '/mon/cal/calibration',
            {
                "transit_time": filenames[date][calname]['transit_time'].mjd,
                "calibration_source": calname,
                "filelist": flist,
                "status": -1
            }
        )

        convert_calibrator_pass_to_ms(
            cal=filenames[date][calname]['cal'],
            date=date,
            files=filenames[date][calname]['files'],
            duration=CALTIME
        )

        LOGGER.info('{0}.ms created'.format(msname))

        status = calibrate_measurement_set(
            msname,
            filenames[date][calname]['cal'],
            refant=REFANT,
            bad_antennas=None,
            bad_uvrange='2~27m',
            forsystemhealth=True
        )

        caltable_to_etcd(
            msname,
            calname,
            filenames[date][calname]['transit_time'].mjd,
            status
        )

        ETCD.put_dict(
            '/mon/cal/calibration',
            {
                "transit_time": filenames[date][calname]['transit_time'].mjd,
                "calibration_source": calname,
                "filelist": flist,
                "status": status
            }
        )

        # This should be made more general for more antennas
        antennas = np.array(
            [24, 25, 26, 27, 28, 29, 30, 31, 32,
             33, 34, 35, 20, 19, 18, 17, 16, 15,
             14, 13, 100, 101, 102, 116, 103]
        )
        figure_path = '{0}/figures/{1}_{2}'.format(msdir, date, calname)
        try:
            with PdfPages('{0}.pdf'.format(figure_path)) as pdf:
                for j in range(len(antennas)//10+1):
                    fig = summary_plot(
                        msname,
                        calname,
                        2,
                        ['B', 'A'],
                        antennas[j*10:(j+1)*10]
                    )
                    pdf.savefig(fig)
                    plt.close()
            # Index error occured - some files could not be found.
            plot_current_beamformer_solutions(
                filenames[date][calname]['files'],
                calname,
                date,
                corrlist=[1, 2, 3, 21, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                outname=figure_path,
                show=False
            )
        except Exception as exc:
            exception_logger(
                LOGGER,
                'plotting of calibration solutions for {0}.ms'.format(msname),
                exc,
                throw=False
            )

        LOGGER.info('Calibrated {0}.ms with status {1}'.format(msname, status))

if __name__=="__main__":
    ETCD.add_watch('/cmd/cal', calibrate_file)
    while True:
        ETCD.put_dict(
            '/mon/cal/calibrate_process',
            {
                "alive": True,
                "cadence": 60,
                "time": Time(datetime.datetime.utcnow()).isot
            }
        )
        time.sleep(60)
