import os
import tqdm
import json

corrdir = '/mnt/data/dsa110/correlator/'
savedir = '/mnt/data/dsa110/candidate_corrdata/'

with open('corrfiles_cands.json') as f:
    data = json.load(f)

for candname, files in data.items():
    print(f'Working on {len(files)} files for {candname}')
    if not os.path.exists(f'{savedir}{candname}'):
        os.mkdir(f'{savedir}{candname}')
    for file in tqdm.tqdm(files):
        corr, fname = file.split('/')[-2:]
        if not os.path.exists(f'{savedir}{candname}/{corr}'):
            os.mkdir(f'{savedir}{candname}/{corr}')
        if not os.path.exists(f'{savedir}{candname}/{corr}/{fname}'):
            os.link(
                f'{corrdir}{corr}/{fname}',
                f'{savedir}{candname}/{corr}/{fname}'
            )
