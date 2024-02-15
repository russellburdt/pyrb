
"""
create single csv from model artifacts to run business case calculations
- only works with models predicting collision as behavior-id = 47
"""

import os
import utils
import pickle
import pandas as pd
import numpy as np
from collections import defaultdict
from lytx import get_conn
from tqdm import tqdm
from glob import glob
from shutil import copy2 as copy
from tqdm import tqdm
from ipdb import set_trace

# model artifacts
src = r'/mnt/home/russell.burdt/data/collision-model/app/artifacts-12'
assert os.path.isdir(src)

# population and collision data
if os.path.isfile(os.path.join(src, 'population-data.p')):
    dp = pd.read_pickle(os.path.join(src, 'population-data.p'))
else:
    assert os.path.isfile(os.path.join(src, 'population-test-data.p'))
    dp = pd.read_pickle(os.path.join(src, 'population-test-data.p'))
dc = pd.read_pickle(os.path.join(src, 'collisions.p'))
utils.validate_dcm_dp(dp, dc)

# clean population data
assert all(dp['collision-47'] == dp['outcome'])
dp['collision'] = dp.pop('collision-47')
dp['collision-idx'] = dp.pop('collision-47-idx')
for x in ['CreationDate', 'DeletedDate', 'collision-45', 'collision-45-idx', 'collision-46', 'collision-46-idx', 'outcome']:
    del dp[x]

# severity data
lab = get_conn('lytx-lab')
ds = pd.read_sql_query(f"""
    SELECT *
    FROM insurance_model.severity_data""", lab)
ds.columns = ['event_id', 'company_id', 'date', 'severity', 'vehicle_type']

# scan over collisions
dx = defaultdict(list)
for index, row in tqdm(dp.loc[dp['collision']].iterrows(), desc='scanning collisions', total=dp['collision'].sum()):

    # validate idx, initialize collision data
    assert row['collision-idx'].size > 0
    nc, sev1, sev2, sev3, sev4 = 0, 0, 0, 0, 0
    ts = np.array([]).astype(np.datetime64)
    sev_order = np.array([]).astype('int')

    # scan over collisions for vid
    for x in row['collision-idx']:

        # collision data
        dcx = dc.loc[x]
        cid = dcx['CustomerEventIdString']
        ts = np.hstack((ts, np.array([dcx['RecordDate']], dtype=np.datetime64)))
        nc += 1

        # identify collision severity
        dsx = ds.loc[ds['event_id'] == cid]
        assert dsx.shape[0] < 2
        if dsx.shape[0] == 1:
            sev = dsx.iloc[0]['severity']
            if sev == 'Level 1':
                sev1 += 1
                sev_order = np.hstack((sev_order, np.array([1], dtype='int')))
            elif sev == 'Level 2':
                sev2 += 1
                sev_order = np.hstack((sev_order, np.array([2], dtype='int')))
            elif sev == 'Level 3':
                sev3 += 1
                sev_order = np.hstack((sev_order, np.array([3], dtype='int')))
            elif sev == 'Level 4':
                sev4 += 1
                sev_order = np.hstack((sev_order, np.array([4], dtype='int')))
            # null case
            else:
                sev_order = np.hstack((sev_order, np.array([-1], dtype='int')))
        # null case
        else:
            sev_order = np.hstack((sev_order, np.array([-1], dtype='int')))

    # update dx
    assert sev1 + sev2 + sev3 + sev4 <= nc
    dx['index'].append(index)
    dx['collision_ts'].append(ts)
    dx['num collisions'].append(nc)
    dx['num severity 1'].append(sev1)
    dx['num severity 2'].append(sev2)
    dx['num severity 3'].append(sev3)
    dx['num severity 4'].append(sev4)
    dx['severity order'].append(sev_order)
dx = pd.DataFrame(dx)
dx.index = dx.pop('index')
dx.index.name = None

# join count and severity info with population data
dp = pd.merge(left=dp, right=dx, left_index=True, right_index=True, how='left')
cn = np.array([np.isnan(dp[x]).sum() for x in ['num collisions'] + [f'num severity {x}' for x in range(1, 5)]])
assert np.unique(cn).size == 1
assert (~dp['collision']).sum() == cn[0]

# prediction probabilities
dm = pd.read_pickle(os.path.join(src, 'model-prediction-probabilities.p'))
assert all(dm['actual outcome'] == dp['collision'])
dp['prediction probability'] = dm['prediction probability']
pmax = dp['prediction probability'].max()
pmin = dp['prediction probability'].min()
dp['rating factor, %'] = 100 * (1 - (dp['prediction probability'] - pmin) / (pmax - pmin))

# save as pickle and csv
for x in ['num severity 1', 'num severity 2', 'num severity 3', 'num severity 4', 'severity order']:
    del dp[x]
dp.to_pickle(os.path.join(os.path.split(src)[0], os.path.split(src)[1] + '.p'))
dp.to_csv(os.path.join(os.path.split(src)[0], os.path.split(src)[1] + '.csv'), index=False)
