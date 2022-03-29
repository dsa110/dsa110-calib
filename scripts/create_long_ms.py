from dsacalib.ms_io import convert_calibrator_pass_to_ms
from dsacalib.routines import get_files_for_cal
import astropy.units as u
from pkg_resources import resource_filename

msdir = '/mnt/data/dsa110/calibration/manual_cal/'
hdf5dir = '/mnt/data/dsa110/correlator/'
calpasses = [
    {
        'date': '2021-12-23',
        'name': 'J141120+521209',
        'dec': '+52p06',
        'duration': 15*u.min,
    },
]

for i, calpass in enumerate(calpasses):
    print(f'working on {calpass}')
    date = calpass['date']
    dec = calpass['dec']
    duration = calpass['duration']
    name = calpass['name']
    calsources = resource_filename(
        'dsacalib',
        'data/calibrator_sources_dec{0}.csv'.format(dec)
    )
    filenames = get_files_for_cal(
        calsources,
        '03',
        duration,
        5*u.min,
        date_specifier='{0}*'.format(date)
    )
    cal = filenames[date][name]['cal']
    files = filenames[date][name]['files']
    convert_calibrator_pass_to_ms(
        cal,
        date,
        files,
#        duration, # Note - I got rid of this, since it's not being used
        msdir=msdir,
        hdf5dir=hdf5dir
    )
