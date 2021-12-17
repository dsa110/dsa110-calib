import json
import glob
import pandas
from astropy.time import Time
import astropy.units as u
import os
import glob
import tqdm
import numpy as np

corrdir = '/mnt/data/dsa110/correlator/'
savedir = '/mnt/data/dsa110/candidate_corrdata/'
duration = 5*u.min
hours_tosave = 24

df = pandas.DataFrame()
for file in glob.iglob('/media/ubuntu/data/dsa110/T3/*/[0-9][0-9][0-9][0-9][0-9][0-9][a-z][a-z][a-z][a-z].json'):
    with open(file) as f:
        data = json.load(f)
    if data.get('save', False):
        df = pandas.concat([
            df,
            pandas.DataFrame(dict({
                k: [v] for k, v in data.items()}))
        ])

corrfiles_cands = {}
for index, row in df.iterrows():
    candtime = Time(row['mjds'], format='mjd')
    candtime.precision = 0
    hdf5files = []
    for h in np.arange(-hours_tosave/2, hours_tosave/2+1):
        hdf5files += glob.glob(f'/data/dsa110/correlator/corr??/{(candtime+h*u.h).isot[:-5]}*hdf5')
    if len(hdf5files) > 0:
        print(candtime, len(hdf5files))
        tokeep = []
        for file in hdf5files:
            filetime = Time(file.split('/')[-1][:-5])
            if filetime + duration > candtime - hours_tosave/2*u.h and filetime < candtime + hours_tosave/2*u.h :
                tokeep += [file]
                corrfiles_cands[row['trigname']] = tokeep
with open('corrfiles_cands.json', 'w') as f:
    json.dump(corrfiles_cands, f)
