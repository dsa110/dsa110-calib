from dsacalib.ms_io import convert_calibrator_pass_to_ms
from dsacalib.routines import get_files_for_cal
import astropy.units as u
from pkg_resources import resource_filename
import dsacalib.config as configuration
config = configuration.Configuration()

msdir = '/operations/calibration/manual_cal'
date = '2022-06-28'
calname = '1331+305'
dec = '+030p5'
duration = 40.*u.min

calsources = resource_filename(
    'dsacalib',
    f'data/calibrator_sources_dec{dec}.csv'
)

filenames = get_files_for_cal(
    calsources,
    config.hdf5dir,
    'sb01',
    duration,
    config.filelength,
    date_specifier = '{0}*'.format(date)
)
print(filenames)
cal = filenames[date][calname]['cal']
convert_calibrator_pass_to_ms(cal,date,filenames[date][calname]['files'],msdir=msdir,hdf5dir=config.hdf5dir,refmjd=config.refmjd)

