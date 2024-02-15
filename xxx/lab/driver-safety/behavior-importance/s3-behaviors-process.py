
"""
process behaviors data and store example videos
"""

import os
import lytx
import numpy as np
import pandas as pd
from shutil import rmtree
from tqdm import tqdm
from glob import glob
from datetime import datetime
from ipdb import set_trace


# parameters and spark session
population = 'dist300'
spark = lytx.spark_session(memory='32g', cores='*')
videos = True

# datadir and metadata objects
datadir = os.path.join(r'/mnt/home/russell.burdt/data/driver-safety/behavior-importance', population)
assert os.path.isdir(datadir)

# population DataFrame
dp = pd.read_pickle(os.path.join(datadir, 'population-dataframe.p'))
# dp = dp.loc[dp['IndustryDesc'] == 'Distribution'].reset_index(drop=True)

# eval table
de = dp[['VehicleId', 'ta', 'tb']].rename(columns={'ta': 't0', 'tb': 't1'}).reset_index(drop=False)
de['t0'] = de['t0'].values.astype('datetime64[s]').astype('int')
de['t1'] = de['t1'].values.astype('datetime64[s]').astype('int')
spark.createDataFrame(de).createOrReplaceTempView('eval')

# behaviors data
behaviors = spark.read.parquet(os.path.join(datadir, 'behaviors.parquet'))
behaviors.createOrReplaceTempView('behaviors')

# behavior counts
db = lytx.behavior_count_metrics(spark)
assert all(db.values[:, 0] == db.values[:, 1:].sum(axis=1))
del db['nbehaviors']
db.to_pickle(os.path.join(datadir, 'behavior-counts.p'))

# filter behaviors that are concurrent with collisions
dba = db.copy()
cxs = dba.loc[dba['nbehaviors_47'].astype('bool')].index.to_numpy()
for cx in tqdm(cxs, desc='filter behaviors'):
    assert dba.loc[cx, 'nbehaviors_47'] > 0
    fn = glob(os.path.join(datadir, 'behaviors.parquet', f"""VehicleId={dp.loc[cx, 'VehicleId']}""", '*.parquet'))
    assert len(fn) == 1
    df = pd.read_parquet(fn[0])
    df['RecordDate'] = [pd.Timestamp(x) for x in df['RecordDate']]
    df = df.loc[(df['RecordDate'] > dp.loc[cx, 'ta']) & (df['RecordDate'] < dp.loc[cx, 'tb'])].reset_index(drop=True)
    assert (df['BehaviorId'] == '47').any()
    for xe in df.loc[df['BehaviorId'] == '47', 'EventId'].values:
        for bx in [x for x in df.loc[df['EventId'] == xe, 'BehaviorId'].values if x != '47']:
            assert dba.loc[cx, f'nbehaviors_{bx}'] > 0
            dba.loc[cx, f'nbehaviors_{bx}'] -= 1
assert (db.shape[0] == dba.shape[0]) and all(db.columns == dba.columns)
dba.to_pickle(os.path.join(datadir, 'behavior-counts-mod.p'))

# n random videos of behaviors
if videos:
    n = 10
    bxs = np.array([int(x.split('_')[1]) for x in db.columns])
    bdir = os.path.join(datadir, 'behavior-videos')
    if os.path.isdir(bdir):
        rmtree(bdir)
    os.mkdir(bdir)
    for bx in tqdm(bxs, desc='behavior videos'):
        xdir = os.path.join(bdir, f'behavior-{bx}')
        os.mkdir(xdir)
        cxs = db.loc[db[f'nbehaviors_{bx}'] > 0].index.to_numpy()
        for x in range(n):
            cx = np.random.choice(cxs)
            assert db.loc[cx, f'nbehaviors_{bx}'] > 0
            fn = glob(os.path.join(datadir, 'behaviors.parquet', f"""VehicleId={dp.loc[cx, 'VehicleId']}""", '*.parquet'))
            assert len(fn) == 1
            df = pd.read_parquet(fn[0])
            df['RecordDate'] = [pd.Timestamp(x) for x in df['RecordDate']]
            df = df.loc[(df['RecordDate'] > dp.loc[cx, 'ta']) & (df['RecordDate'] < dp.loc[cx, 'tb'])].reset_index(drop=True)
            assert (df['BehaviorId'] == str(bx)).any()
            record = df.loc[np.random.choice(df.loc[df['BehaviorId'] == str(bx)].index.to_numpy())].squeeze()
            lytx.extract_and_save_video(record=record, fn=os.path.join(xdir, f'behavior-{x}.mkv'))
