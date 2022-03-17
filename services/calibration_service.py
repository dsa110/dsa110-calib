"""A service to create measurement sets and calibrate data.
"""
import os
import sys
import shutil
import warnings
from multiprocessing import Process, Queue
import datetime
import time
import yaml
import h5py
import numpy as np
import astropy.units as u
from astropy.time import Time
import dsautils.dsa_store as ds
import dsautils.dsa_syslog as dsl
import dsautils.cnf as dsc
from dsacalib.preprocess import first_true, update_caltable
from dsacalib.utils import exception_logger
from dsacalib.calib import calibrate_phase_single_ms
from dsacalib.routines import get_files_for_cal, calibrate_measurement_set
from dsacalib.ms_io import convert_calibrator_pass_to_ms, caltable_to_etcd
from dsacalib.hdf5_io import extract_applied_delays
from dsacalib.weights import write_beamformer_solutions, average_beamformer_solutions, filter_beamformer_solutions, get_good_solution, \
    consistent_correlator
from dsacalib.plotting import summary_plot, plot_bandpass_phases, \
    plot_beamformer_weights

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
warnings.filterwarnings("ignore")

# Logger
LOGGER = dsl.DsaSyslogger()
LOGGER.subsystem("software")
LOGGER.app("dsacalib")

CONF = dsc.Conf()
CORR_PARAMS = CONF.get('corr')
CAL_PARAMS = CONF.get('cal')
MFS_PARAMS = CONF.get('fringe')

# These should be put somewhere else eventually
CALTIME = CAL_PARAMS['caltime_minutes']*u.min
REFANTS = CAL_PARAMS['refant']
if isinstance(REFANTS, (str, int)):
    REFANTS = [REFANTS]
FILELENGTH = MFS_PARAMS['filelength_minutes']*u.min
MSDIR = CAL_PARAMS['msdir']
BEAMFORMER_DIR = CAL_PARAMS['beamformer_dir']
HDF5DIR = CAL_PARAMS['hdf5_dir']
# This should be made more general for more antennas
ANTENNAS = list(CORR_PARAMS['antenna_order'].values())
POLS = CORR_PARAMS['pols_voltage']
ANTENNAS_IN_MS = CAL_PARAMS['antennas_in_ms']
ANTENNAS_NOT_IN_BF = CAL_PARAMS['antennas_not_in_bf']
CORR_LIST = list(CORR_PARAMS['ch0'].keys())
CORR_LIST = [int(cl.strip('corr')) for cl in CORR_LIST]
REFCORR = '{0:02d}'.format(CORR_LIST[0])
WEBPLOTS = '/mnt/data/dsa110/webPLOTS/calibration/'
PLOTDIR = f'{WEBPLOTS}/allpngs/'

TSLEEP = 60
CALIB_Q = Queue()

def calibrate_file(calname, flist):
    """Calibrate a calibrator pass.
    """
    etcd = ds.DsaStore()
    date = first_true(flist).split('/')[-1][:-14]
    msname = '{0}/{1}_{2}'.format(MSDIR, date, calname)
    date_specifier = '{0}*'.format(date)
    # Get the start time for the snaps
    start_time = Time(
       etcd.get_dict('/mon/snap/1/armed_mjd')['armed_mjd'], format='mjd'
    )
    with h5py.File(first_true(flist), mode='r') as h5file:
        pt_dec = h5file['Header']['extra_keywords']['phase_center_dec'].value*u.rad
    caltable = update_caltable(pt_dec)
    LOGGER.info('Creating {0}.ms at dec {1}'.format(msname, pt_dec))
    filenames = get_files_for_cal(
        caltable,
        REFCORR,
        CALTIME,
        FILELENGTH,
        hdf5dir=HDF5DIR,
        date_specifier=date_specifier,
    )
    ttime = filenames[date][calname]['transit_time']
    # Only use calibrators within the last 24 hours or since the snaps
    # were restarted
    if ttime-start_time > 24*u.h:
        start_time = ttime - 24*u.h
    ttime.precision = 0
    etcd.put_dict(
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
        duration=CALTIME,
        # antenna_list=ANTENNAS_IN_MS,
        logger=LOGGER,
        msdir=MSDIR
    )
    print('done writing ms')
    LOGGER.info('{0}.ms created'.format(msname))

    status = calibrate_measurement_set(
        msname,
        filenames[date][calname]['cal'],
        refants=REFANTS,
        bad_antennas=None,
        bad_uvrange='2~27m',
        forsystemhealth=True,
        throw_exceptions=True,
        logger=LOGGER
    )
    print('done calibration')
    caltable_to_etcd(
        msname,
        calname,
        filenames[date][calname]['transit_time'].mjd,
        status,
        logger=LOGGER
    )

    etcd.put_dict(
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
    figure_path = '{0}/{1}_{2}'.format(PLOTDIR, date, calname)
    try:
        with PdfPages('{0}.pdf'.format(figure_path)) as pdf:
            for j in range(len(ANTENNAS)//10+1):
                fig = summary_plot(
                    msname,
                    calname,
                    2,
                    ['B', 'A'],
                    ANTENNAS[j*10:(j+1)*10]
                )
                pdf.savefig(fig)
                plt.close(fig)
        target = f'{WEBPLOTS}/summary_current.pdf'
        if os.path.exists(target):
            os.unlink(target)
        shutil.copyfile(
            f'{PLOTDIR}/{date}_{calname}.pdf',
            target
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
        refants=REFANTS,
        bad_antennas=None,
        bad_uvrange='2~27m',
        keepdelays=False,
        forsystemhealth=False,
        throw_exceptions=False,
        logger=LOGGER
    )
    LOGGER.info(
        'Calibrated {0}.ms for beamformer weights with status {1}'
        .format(msname, status)
    )
    print('calculating beamformer weights')
    try:
        applied_delays = extract_applied_delays(first_true(flist), ANTENNAS)
        # Write beamformer solutions for one source
        _ = write_beamformer_solutions(
            msname,
            calname,
            ttime,
            ANTENNAS,
            applied_delays,
            flagged_antennas=ANTENNAS_NOT_IN_BF,
            corr_list=np.array(CORR_LIST)
        )
    except Exception as exc:
        exception_logger(
            LOGGER,
            'calculation of beamformer weights for {0}.ms'.format(msname),
            exc,
            throw=False
        )

    # Create the bandpass phase table for plotting later
    calibrate_phase_single_ms(msname, REFANTS[0], calname)

    print('getting list of calibrators')
    # Now we want to find all sources in the last 24 hours
    # start by updating our list with calibrators from the day before
    beamformer_names = get_good_solution()
    beamformer_names, latest_solns = filter_beamformer_solutions(
        beamformer_names, start_time.mjd)

    # Average beamformer solutions
    if len(beamformer_names) > 0:
        print('checking reference gains')
        add_reference_bfname(beamformer_names, latest_solns, start_time)

        print('averaging beamformer weights')
        averaged_files, avg_flags = average_beamformer_solutions(
            beamformer_names,
            ttime,
            corridxs=CORR_LIST
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
        # Remove the old bad cal solutions
        for key, value in \
            latest_solns['cal_solutions']['flagged_antennas'].items():
            if 'casa solutions flagged' in value:
                value = value.remove('casa solutions flagged')
        # Flag new bad solutions
        idxant, idxpol = np.nonzero(avg_flags)
        for i, ant in enumerate(idxant):
            key = '{0} {1}'.format(ANTENNAS[ant], POLS[idxpol[i]])
            if key not in \
                latest_solns['cal_solutions']['flagged_antennas'].keys():
                latest_solns['cal_solutions']['flagged_antennas'][key] = []
            latest_solns['cal_solutions']['flagged_antennas'][key] += \
                ['casa solutions flagged']
        latest_solns['cal_solutions']['flagged_antennas'] = {
            key: value for key, value in
            latest_solns['cal_solutions']['flagged_antennas'].items()
            if len(value) > 0
        }
        print('opening yaml file')
        with open(
            '{0}/beamformer_weights_{1}.yaml'.format(
                BEAMFORMER_DIR, ttime.isot
            ),
            'w'
        ) as file:
            print('writing bf weights')
            _ = yaml.dump(latest_solns, file)
        latest_solns['cal_solutions']['time'] = ttime.mjd
        latest_solns['cal_solutions']['bfname'] = ttime.isot
        etcd.put_dict(
            '/mon/cal/bfweights',
            {
                'cmd': 'update_weights',
                'val': latest_solns['cal_solutions']
            }
        )
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
        beamformer_names += [averaged_files[0].split('_')[-1].strip(".dat")]
        _ = plot_beamformer_weights(
            beamformer_names,
            antennas_to_plot=np.array(ANTENNAS),
            outname='{0}/{1}'.format(PLOTDIR, ttime),
            corrlist=np.array(CORR_LIST),
            show=False
        )

        # Copy plot to "current" WebPLOTS
        target = f'{WEBPLOTS}/bfw_current.png'
        if os.path.exists(target):
            os.unlink(target)
        shutil.copyfile(
            f'{PLOTDIR}/{ttime}_averagedweights.png',
            target
        )

    # Plot evolution of the phase over the day
    plot_bandpass_phases(
        beamformer_names,
        np.array(ANTENNAS),
        outname='{0}/{1}'.format(PLOTDIR, ttime),
        show=False
    )
    plt.close('all')

    # Copy plot to "current" WebPLOTS
    target = f'{WEBPLOTS}/phase_current.png'
    if os.path.exists(target):
        os.unlink(target)
    shutil.copyfile(
        f'{PLOTDIR}/{ttime}_phases.png',
        target
    )

def add_reference_bfname(beamformer_names, latest_solns, start_time):
    """
    If the setup of the current beamformer weights matches that of the latest file, 
    add the current weights to the beamformer_names list
    """

    etcd = ds.DsaStore()
    ref_bfname = etcd.get_dict('/mon/cal/bfweights')['bfname']

    with open(
        "{0}/beamformer_weights_{1}.yaml".format(BEAMFORMER_DIR, ref_bfname)
    ) as f:
        ref_solns = yaml.load(f, Loader=yaml.FullLoader)

    if consistent_correlator(ref_solns, latest_solns, start_time.mjd):
        beamformer_names.append(ref_bfname)

# TODO: Etcd watch robust to etcd connection failures.
def calibrate_file_manager(inqueue=CALIB_Q):
    """Manages the queue and creates subprocesses for calibration.
    """
    while True:
        if not inqueue.empty():
            try:
                val = inqueue.get()
                calname = val['calname']
                flist = val['flist']
            except Exception as exc:
                exception_logger(
                    LOGGER,
                    'attempt to retrieve calibration task from queue',
                    exc,
                    throw=False
                )
            else:
                print('flist[0]: {0}, {1}'.format(
                    first_true(flist), type(first_true(flist))
                ))
                # start a subprocess
                calib_process = Process(
                    target=calibrate_file,
                    args=(
                        calname,
                        flist
                    ),
                    daemon=True
                )
                calib_process.start()
                calib_process.join()
            time.sleep(TSLEEP)

def populate_queue(etcd_dict, outqueue=CALIB_Q):
    """Populates the calibration queue.
    """
    if etcd_dict['cmd'] == 'calibrate':
        outqueue.put(etcd_dict['val'])

def watch_for_calibration():
    """Watch for calibration commands from etcd.
    """
    etcd = ds.DsaStore()
    etcd.add_watch('/cmd/cal', populate_queue)
    while True:
        etcd.put_dict(
            '/mon/service/calibration',
            {
                "cadence": 60,
                "time": Time(datetime.datetime.utcnow()).mjd
            }
        )
        time.sleep(60)

if __name__=="__main__":
    processes = {}
    processes['calibrate'] = Process(
        target=calibrate_file_manager,
        args=(CALIB_Q,)
    )
    processes['calibrate'].start()
    print('calibration process started')
    processes['watch'] = Process(
        target=watch_for_calibration,
        daemon=True
    )
    processes['watch'].start()
    print('watch process started')

    try:
        while True:
            assert processes['watch'].is_alive() # needs a timeout
            assert processes['calibrate'].is_alive() # needs a timeout
            print(f'{CALIB_Q.qsize()} objects in calibration queue')
            #sys.stdout.flush()
            #sys.stderr.flush()
            time.sleep(5*60)
    except (KeyboardInterrupt, SystemExit, AssertionError):
        # Terminate non-daemon processes
        processes['calibrate'].terminate()
        processes['calibrate'].join()
        sys.exit()
