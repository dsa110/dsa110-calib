import os
import glob
import tqdm
import datetime

from dsautils import cnf

corrdir = cnf.Conf().get('cal')['hdf5_dir']
cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=7)
cutoff = cutoff.strftime('%Y-%m-%d')
print(f'Removing hdf5 files in {corrdir} from before {cutoff}')

for subdir in glob.glob(f'{corrdir}corr??'):
    files = glob.glob(f'{subdir}/????-??-??T??:??:??.hdf5')
    print(f'For directory {subdir} working on {len(files)} files')
    for i in tqdm.tqdm(range(len(files))):
        file = files[i]
        date = file.split('/')[-1][:10]
        if date < cutoff:
            os.unlink(file)
