"""A service to create measurement sets and calibrate data.
"""

import os
import warnings
# make sure warnings do not spam syslog
warnings.filterwarnings("ignore")
import datetime # pylint: disable=wrong-import-position
import time # pylint: disable=wrong-import-position
import yaml # pylint: disable=wrong-import-position
import h5py # pylint: disable=wrong-import-position
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
ANTENNAS_NOT_IN_BF = ['103 A', '103 B', '101 A', '101 B', '100 A', '100 B',
                      '116 A', '116 B', '102 A', '102 B']
CALTABLE = resource_filename('dsacalib', 'data/calibrator_sources.csv')

def sort_filenames(filenames):
    """Sort list of calibrator passes.
    """
    filenames_sorted = {}
    yesterday, today = sorted(filenames.keys())
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
    return filenames_sorted

def find_bf_solns_to_avg(today, filenames, ttime, start_time):
    """Find all previous calibrator passes to average.
    """
    # TODO: Just use a glob of the beamformer directory instead since the
    # names contain the transit pass time and calibrator names.
    yesterday = (ttime-1*u.d).isot.split('T')[0]
    filenames_yesterday = get_files_for_cal(
        CALTABLE,
        REFCORR,
        CALTIME,
        FILELENGTH,
        date_specifier='{0}*'.format(yesterday),
    )
    if yesterday in filenames_yesterday.keys():
        filenames[yesterday] = filenames_yesterday[yesterday]
    else:
        filenames[yesterday] = {}
    # Get rid of calibrators after the snap start time or without files
    for date in filenames.keys():
        for cal in list(filenames[date].keys()):
            if filenames[date][cal]['transit_time'] < start_time or \
                len(filenames[date][cal]['files'])==0 or \
                filenames[date][cal]['transit_time'] > ttime:
                filenames[date].pop(cal)
    # Sort the filenames by time
    assert len(filenames.keys()) < 3
    filenames = sort_filenames(filenames)
    # Average beamformer solutions
    beamformer_names = []
    for date in filenames.keys():
        for cal in filenames[date].keys():
            cal_ttime = filenames[date][cal]['transit_time']
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
            '{0}/beamformer_weights_{1}.yaml'.format(
                BEAMFORMER_DIR,
                beamformer_names[0]
            )
        ) as f:
            latest_solns = yaml.load(f, Loader=yaml.FullLoader)
        for bfname in beamformer_names[1:].copy():
            try:
                with open(
                    '{0}/beamformer_weights_{1}.yaml'.format(
                        BEAMFORMER_DIR,
                        bfname
                    )
                ) as f:
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
    else:
        beamformer_names = []
    return beamformer_names, latest_solns

def extract_applied_delays(file):
    """Extracts the current snap delays from the hdf5 file.
    
    If delays are not set in the hdf5 file, uses the most recent delays in
    the beamformer weights directory instead.

    Parameters
    ----------
    file : str
        The full path to the hdf5 file.

    Returns
    -------
    ndarray
        The applied delays in ns.
    """
    with h5py.File(file, 'r') as f:
        if 'applied_delays_ns' in f['Header']['extra_keywords'].keys():
            delaystring = (
                f['Header']['extra_keywords']['applied_delays_ns']
                .value
            )
            applied_delays = np.array(
                delaystring.split(' ')
            ).astype(np.int).reshape(-1, 2)
            applied_delays = applied_delays[ANTENNAS-1, :]
        else:
            current_solns = '{0}/beamformer_weights.yaml'.format(BEAMFORMER_DIR)
            with open(current_solns) as file:
                calibration_params = yaml.load(
                    file,
                    Loader=yaml.FullLoader
                )['cal_solutions']
            applied_delays = np.array(calibration_params['delays'])*2
            LOGGER.error(
                'Error extracting snap delays from uvh5 files. '
                'Using delays in {0}'.format(current_solns)
            )
    return applied_delays

# TODO: Etcd watch robust to etcd connection failures.
def calibrate_file(etcd_dict):
    """Generates and calibrates a measurement set.
    
    An etcd watch callback function.
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
        start_time = Time(
           ETCD.get_dict('/mon/snap/1/armed_mjd')['armed_mjd'], format='mjd'
        )
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

#         ETCD.put_dict(
#             '/mon/cal/calibration',
#             {
#                 "transit_time": filenames[date][calname]['transit_time'].mjd,
#                 "calibration_source": calname,
#                 "filelist": flist,
#                 "status": -1
#             }
#         )
#         print('writing ms')
#         convert_calibrator_pass_to_ms(
#             cal=filenames[date][calname]['cal'],
#             date=date,
#             files=filenames[date][calname]['files'],
#             duration=CALTIME
#         )
#         print('done writing ms')
#         LOGGER.info('{0}.ms created'.format(msname))

        status = calibrate_measurement_set(
            msname,
            filenames[date][calname]['cal'],
            refant=REFANT,
            bad_antennas=None,
            bad_uvrange='2~27m',
            forsystemhealth=True,
            throw_exceptions=True
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
# TODO: Get beamformer weight filenames from etcd
#             # Index error occured - some files could not be found. corr04
#             plot_current_beamformer_solutions(
#                 filenames[date][calname]['files'],
#                 calname,
#                 date,
#                 # beamformer name,
#                 corrlist=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
#                           11, 12, 13, 14, 15, 16],
#                 outname=figure_path,
#                 show=False
#             )
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
            applied_delays = extract_applied_delays(flist[0])
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
        beamformer_names, latest_solns = find_bf_solns_to_avg(date, filenames, ttime, start_time)
        # Average beamformer solutions
        # TODO: Implement averaging that takes into account flags
        # For now, just using the most recent one instead
        if len(beamformer_names) > 0:
            print('averaging beamformer weights')
            averaged_files = average_beamformer_solutions(
                [beamformer_names[0]],
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
            filenames,
            ANTENNAS_PLOT,
            outname='{0}/figures/{1}'.format(MSDIR, ttime)
        )

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
