"""A service to create measurement sets and calibrate data.
"""

import warnings
# make sure warnings do not spam syslog
warnings.filterwarnings("ignore")
import os
import datetime # pylint: disable=wrong-import-position
import time # pylint: disable=wrong-import-position
import yaml # pylint: disable=wrong-import-position
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
from dsacalib.calib import calibrate_phases # pylint: disable=wrong-import-position
from dsacalib.routines import get_files_for_cal, calibrate_measurement_set # pylint: disable=wrong-import-position
from dsacalib.ms_io import convert_calibrator_pass_to_ms, caltable_to_etcd, write_beamformer_solutions, average_beamformer_solutions # pylint: disable=wrong-import-position
from dsacalib.plotting import summary_plot, plot_current_beamformer_solutions, plot_bandpass_phases # pylint: disable=wrong-import-position

# Logger
LOGGER = dsl.DsaSyslogger()
LOGGER.subsystem("software")
LOGGER.app("dsacalib")

# ETCD interface
ETCD = ds.DsaStore()

# These should be put somewhere else eventually
CALTIME = 15*u.min
REFANT = '102'
REFCORR = '01'
FILELENGTH = 15*u.min
MSDIR = '/mnt/data/dsa110/calibration/'
BEAMFORMER_DIR = '/home/user/beamformer_weights/'
# This should be made more general for more antennas
ANTENNAS_PLOT = np.array(
    [24, 25, 26, 27, 28, 29, 30, 31, 32,
     33, 34, 35, 20, 19, 18, 17, 16, 15,
     14, 13, 100, 101, 102, 116, 103]
)
ANTENNAS = np.concatenate((
    ANTENNAS_PLOT,
    np.arange(36, 36+39)
))
ANTENNAS_NOT_IN_BF = ['103', '101', '100', '116', '102']
CALTABLE = resource_filename('dsacalib', 'data/calibrator_sources.csv')

def calibrate_file(etcd_dict):
    """Generates and calibrates a measurement set.
    """
    cmd = etcd_dict['cmd']
    val = etcd_dict['val']
    if cmd == 'calibrate':
        calname = val['calname']
        flist = val['flist']
        date = first_true(flist).split('/')[-1][:-14]
        msname = '{0}/{1}_{2}'.format(MSDIR, date, calname)
        date_specifier = '{0}*'.format(date)

        # Get the start time for the snaps
        # start_time = Time(
        #    ETCD.get_dict('/mon/snap/1/armed_mjd')['armed_mjd'], format='mjd'
        # )
        start_time = Time('2021-01-06T09:00:00')
        LOGGER.info('Creating {0}.ms'.format(msname))

        filenames = get_files_for_cal(
            CALTABLE,
            REFCORR,
            CALTIME,
            FILELENGTH,
            date_specifier=date_specifier,
        )
        ttime = filenames[date][calname]['transit_time']
        # Only use calibrators within the last 24 hours or since the snaps
        # were restarted
        if ttime-start_time > 24*u.h:
            start_time = ttime - 24*u.h
        ttime.precision = 0

        ETCD.put_dict(
            '/mon/cal/calibration',
            {
                "transit_time": filenames[date][calname]['transit_time'].mjd,
                "calibration_source": calname,
                "filelist": flist,
                "status": -1
            }
        )
        print('writing ms')
        convert_calibrator_pass_to_ms(
            cal=filenames[date][calname]['cal'],
            date=date,
            files=filenames[date][calname]['files'],
            duration=CALTIME
        )
        print('done writing ms')
        LOGGER.info('{0}.ms created'.format(msname))

        status = calibrate_measurement_set(
            msname,
            filenames[date][calname]['cal'],
            refant=REFANT,
            bad_antennas=None,
            bad_uvrange='2~27m',
            forsystemhealth=True,
            throw_exceptions=False
        )
        print('done calibration')
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
        print('solns written to etcd')
        LOGGER.info(
            'Calibrated {0}.ms for system health with status {1}'
            .format(msname, status)
        )
        print('creating figures')
        figure_path = '{0}/figures/{1}_{2}'.format(MSDIR, date, calname)
        try:
            with PdfPages('{0}.pdf'.format(figure_path)) as pdf:
                for j in range(len(ANTENNAS_PLOT)//10+1):
                    fig = summary_plot(
                        msname,
                        calname,
                        2,
                        ['B', 'A'],
                        ANTENNAS_PLOT[j*10:(j+1)*10]
                    )
                    pdf.savefig(fig)
                    plt.close()
            # Index error occured - some files could not be found.
            plot_current_beamformer_solutions(
                filenames[date][calname]['files'],
                calname,
                date,
                corrlist=[1, 2, 3, 21, 5, 6, 7, 8, 9, 10,
                          11, 12, 13, 14, 15, 16],
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
        print('calibration for bf')
        status = calibrate_measurement_set(
            msname,
            filenames[date][calname]['cal'],
            refant=REFANT,
            bad_antennas=None,
            bad_uvrange='2~27m',
            keepdelays=False,
            forsystemhealth=False,
            throw_exceptions=False
        )
        LOGGER.info(
            'Calibrated {0}.ms for beamformer weights with status {1}'
            .format(msname, status)
        )
        print('calculating beamformer weights')
        try:
            # These delays should be placed in the file itself instead
            current_solns = '{0}/beamformer_weights.yaml'.format(BEAMFORMER_DIR)
            with open(current_solns) as file:
                calibration_params = yaml.load(file,  Loader=yaml.FullLoader)['cal_solutions']

            applied_delays = np.array(calibration_params['delays'])*2
            print('applied_delays: {0}'.format(applied_delays.shape))
            # Write beamformer solutions for one source
            _ = write_beamformer_solutions(
                msname,
                calname,
                ttime,
                ANTENNAS,
                applied_delays,
                flagged_antennas=ANTENNAS_NOT_IN_BF,
                outdir=BEAMFORMER_DIR
            )
        except Exception as exc:
            exception_logger(
                LOGGER,
                'calculation of beamformer weights for {0}.ms'.format(msname),
                exc,
                throw=False
            )
        print('getting list of calibrators')
        # Now we want to find all sources in the last 24 hours
        # start by updating our list with calibrators from the day before
        today = date
        yesterday = (ttime-1*u.d).isot.split('T')[0]
        filenames_yesterday = get_files_for_cal(
            CALTABLE,
            REFCORR,
            CALTIME,
            FILELENGTH,
            date_specifier='{0}*'.format(yesterday),
        )
        filenames[yesterday] = filenames_yesterday[yesterday]
        # Get rid of calibrators after the snap start time or without files
        for date in filenames.keys():
            for cal in list(filenames[date].keys()):
                if filenames[date][cal]['transit_time'] < start_time or \
                    len(filenames[date][cal]['files'])==0 or \
                    filenames[date][cal]['transit_time'] > ttime:
                    filenames[date].pop(cal)
        # Sort the filenames by time
        assert len(filenames.keys()) < 3
        filenames_sorted = {}
        for date in sorted(filenames.keys(), reverse=True):
            filenames_sorted[date] = {}
        # What is the order that we will get here
        # We want the most recent cal to be last
        times = {
            cal: filenames[today][cal]['transit_time']
            for cal in filenames[today].keys()
        }
        ordered_times = {
            k: v for k, v in sorted(
                times.items(),
                key=lambda item: item[1],
                reverse=True
            )
        }
        for cal in ordered_times.keys():
            filenames_sorted[today][cal] = filenames[today][cal]
        times = {
            cal: filenames[yesterday][cal]['transit_time']
            for cal in filenames[yesterday].keys()
        }
        ordered_times = {
            k: v for k, v in sorted(
                times.items(),
                key=lambda item: item[1],
                reverse=True
            )
        }
        for cal in ordered_times.keys():
            if cal not in filenames_sorted[today].keys():
                filenames_sorted[yesterday][cal] = filenames[yesterday][cal]
        # Average beamformer solutions
        beamformer_names = []
        for date in filenames_sorted.keys():
            for cal in filenames_sorted[date].keys():
                cal_ttime = filenames_sorted[date][cal]['transit_time']
                cal_ttime.precision = 0
                beamformer_names += [
                    '{0}_{1}'.format(
                        cal,
                        cal_ttime.isot
                    )
                ]
        # Open yaml files 
        print('opening yaml files')
        if os.path.exists(
            '{0}/beamformer_weights_{1}.yaml'.format(
                BEAMFORMER_DIR,
                beamformer_names[0]
            )
        ):
            with open(
                '{0}/beamformer_weights_{1}.yaml'.format(BEAMFORMER_DIR, beamformer_names[0])
            ) as f:
                latest_solns = yaml.load(f, Loader=yaml.FullLoader)
            for bfname in beamformer_names[1:].copy():
                try:
                    with open('{0}/beamformer_weights_{1}.yaml'.format(BEAMFORMER_DIR, bfname)) as f:
                        solns = yaml.load(f, Loader=yaml.FullLoader)
                    assert solns['cal_solutions']['antenna_order'] == \
                        latest_solns['cal_solutions']['antenna_order']
                    assert solns['cal_solutions']['corr_order'] == \
                        latest_solns['cal_solutions']['corr_order']
                    assert solns['cal_solutions']['delays'] == \
                        latest_solns['cal_solutions']['delays']
                    assert solns['cal_solutions']['eastings'] == \
                        latest_solns['cal_solutions']['eastings']
                except (AssertionError, FileNotFoundError):
                    beamformer_names.remove(bfname)
            # Average beamformer solutions
            print('averaging beamformer weights')
            averaged_files = average_beamformer_solutions(
                beamformer_names,
                ttime,
                outdir=BEAMFORMER_DIR,
                corridxs=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            )
            print('setting parameters for new yaml file')
            # Make the final yaml file
            latest_solns['cal_solutions']['weight_files'] = averaged_files
            latest_solns['cal_solutions']['source'] = [
                bf.split('_')[0] for bf in beamformer_names
            ]
            latest_solns['cal_solutions']['caltime'] = [
                float(Time(bf.split('_')[1]).mjd) for bf in beamformer_names
            ]
            print('opening yaml file')
            with open(
                '{0}/beamformer_weights.yaml'.format(BEAMFORMER_DIR), 'w'
            ) as file:
                print('writing bf weights')
                _ = yaml.dump(latest_solns, file)
            print('done writing')
            os.system(
                "cd {0} ; "
                "git add beamformer_weights.yaml ; "
                "git commit -m {1} ; "
                "cd /home/user/proj/dsa110-shell/dsa110-calib/services/".format(
                    BEAMFORMER_DIR,
                    beamformer_names[0]
                )
            )

            # Plot evolution of the phase over the day
            calibrate_phases(filenames_sorted, REFANT)
            plot_bandpass_phases(
                filenames_sorted,
                ANTENNAS_PLOT,
                outname='{0}/figures/{1}'.format(MSDIR, ttime)
            )

if __name__=="__main__":
    # try:
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
    # except ds.etcd3.exceptions.ConnectionFailedError:
