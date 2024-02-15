
"""
sample videos for gen-ai experiments
"""

import os
import lytx
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from datetime import datetime
from shutil import rmtree


# datadir and parameters
datadir = r'/mnt/home/russell.burdt/data/gen-ai'
assert os.path.isdir(datadir)
population = 'dtft'
time0, time1 = '10/1/2023', '11/1/2023'

# clear existing data and videos
for src in ['events', 'behaviors']:
    if os.path.isdir(os.path.join(datadir, f'{src}.parquet')):
        rmtree(os.path.join(datadir, f'{src}.parquet'))
    if os.path.isdir(os.path.join(datadir, f'{src}-videos')):
        rmtree(os.path.join(datadir, f'{src}-videos'))

# event recorder associations
time0 = pd.Timestamp(datetime.strptime(time0, '%m/%d/%Y'))
time1 = pd.Timestamp(datetime.strptime(time1, '%m/%d/%Y'))
assert time0.day == time1.day == 1
dx = lytx.event_recorder_associations_window(population=lytx.population_dict(population), time0=time0, time1=time1)
dx = dx.loc[(dx['CreationDate'] < time0) & (dx['DeletedDate'] > time1)].reset_index(drop=True)
assert pd.unique(dx['VehicleId']).size == pd.unique(dx['EventRecorderId']).size == dx.shape[0]

# filter by data rights
dg = pd.read_csv(r'/mnt/home/russell.burdt/data/gen-ai/Account_202308300904.csv')
dx = dx.loc[dx['CompanyId'].isin(dg['companyId'])].reset_index(drop=True)

# DataFrame of n random vehicles from each industry
n = 3000
dn = pd.DataFrame()
for industry in pd.unique(dx['IndustryDesc']):
    ok = dx['IndustryDesc'] == industry
    assert ok.sum() > n
    vids = np.random.choice(dx.loc[ok, 'VehicleId'].values, size=n, replace=False)
    dn = pd.concat((dn, dx.loc[dx['VehicleId'].isin(vids)]))
dn = dn.reset_index(drop=True)
del dn['desc']

# spark session and events raw data
spark = lytx.spark_session(memory='16g', cores='4')
df = dn[['VehicleId', 'time0', 'time1']].rename(columns={'time0': 't0', 'time1': 't1'})
lytx.distributed_data_extraction(datadir=datadir, dataset='events', df=df, nx=100, nd=60, spark=spark, distributed=True)
lytx.merge_parquet(spark=spark, loc=os.path.join(datadir, 'events.parquet'), partition='VehicleId')
events = spark.read.parquet(os.path.join(datadir, 'events.parquet'))
events.createOrReplaceTempView('events')

# behaviors raw data
lytx.parquet_events_to_behaviors(path=datadir, spark=spark, dataset='behaviors', nv=200)
lytx.merge_parquet(spark=spark, loc=os.path.join(datadir, 'behaviors.parquet'), partition='VehicleId')
behaviors = spark.read.parquet(os.path.join(datadir, 'behaviors.parquet'))
behaviors.createOrReplaceTempView('behaviors')

# scan over events, record up to n random videos
n = 40
edir = os.path.join(datadir, 'events-videos')
os.mkdir(edir)
de = events.toPandas()
de = de.loc[~pd.isnull(de['EventFileName']) & ~pd.isnull(de['EventFilePath'])].reset_index(drop=True)
exs = pd.unique(de['EventTriggerTypeId'])
for x in tqdm(exs, desc='event-id'):
    ok = de['EventTriggerTypeId'] == x
    size = min(n, ok.sum())
    idx = np.random.choice(de.loc[ok].index, size=size, replace=False)
    for xx in idx:
        record = de.loc[xx]
        fn = os.path.join(edir, f'{xx}.mkv')
        lytx.extract_and_save_video(record, fn)
dex = de.loc[np.array([os.path.split(x)[1][:-4] for x in glob(os.path.join(edir, '*'))]).astype('int')].sort_index()
dex.to_csv(os.path.join(edir, 'key.csv'), index=True)

# scan over behaviors, record up to n random videos
bdir = os.path.join(datadir, 'behaviors-videos')
os.mkdir(bdir)
db = behaviors.toPandas()
db = db.loc[~pd.isnull(db['EventFileName']) & ~pd.isnull(db['EventFilePath'])].reset_index(drop=True)
bxs = pd.unique(db['BehaviorId'])
for x in tqdm(bxs, desc='behavior-id'):
    ok = db['BehaviorId'] == x
    size = min(n, ok.sum())
    idx = np.random.choice(db.loc[ok].index, size=size, replace=False)
    for xx in idx:
        record = db.loc[xx]
        fn = os.path.join(bdir, f'{xx}.mkv')
        lytx.extract_and_save_video(record, fn)
dbx = db.loc[np.array([os.path.split(x)[1][:-4] for x in glob(os.path.join(bdir, '*'))]).astype('int')].sort_index()
dbx.to_csv(os.path.join(bdir, 'key.csv'), index=True)
