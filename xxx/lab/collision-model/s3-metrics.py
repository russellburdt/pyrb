
"""
metrics for collision prediction model
"""

import os
import lytx
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime
from ipdb import set_trace


# datadir and spark session
datadir = r'/mnt/home/russell.burdt/data/collision-model/dft'
assert os.path.isdir(datadir)
spark = lytx.spark_session(memory='300g', cores='*')

# population DataFrame and eval tables
dcm = pd.read_pickle(os.path.join(datadir, 'population-dataframe.p'))
de = {}
de['predictor'] = dcm[['VehicleId', 'ta', 'tb']].rename(columns={'ta': 't0', 'tb': 't1'}).reset_index(drop=False)
de['collision'] = dcm[['VehicleId', 'tb', 'tc']].rename(columns={'tb': 't0', 'tc': 't1'}).reset_index(drop=False)
for window in ['predictor', 'collision']:
    de[window]['t0'] = de[window]['t0'].values.astype('datetime64[s]').astype('int')
    de[window]['t1'] = de[window]['t1'].values.astype('datetime64[s]').astype('int')

# model outcome based on collision interval, initialize metrics DataFrame
spark.createDataFrame(de['collision']).createOrReplaceTempView('eval')
spark.read.parquet(os.path.join(datadir, 'behaviors.parquet')).createOrReplaceTempView('behaviors')
db = lytx.behavior_count_metrics(spark)
nb47 = db.pop('nbehaviors_47').astype('bool')
dm = nb47.to_frame().rename(columns={'nbehaviors_47': 'outcome'})
print(f"""{dm['outcome'].sum()} collisions, {100 * dm['outcome'].sum() / dcm.shape[0]:.1f}% of evals""")

# all other metrics based on predictor interval
spark.createDataFrame(de['predictor']).createOrReplaceTempView('eval')

# behavior count metrics
db = lytx.behavior_count_metrics(spark)
assert all(db.iloc[:, 1:].sum(axis=1) == db.iloc[:, 0])
dm = pd.merge(left=dm, right=db, left_index=True, right_index=True, how='outer')

# event count and speed metrics
spark.read.parquet(os.path.join(datadir, 'events.parquet')).createOrReplaceTempView('events')
dm = pd.merge(left=dm, right=lytx.event_count_metrics(spark), left_index=True, right_index=True, how='outer')
dm = pd.merge(left=dm, right=lytx.event_speed_metrics(spark), left_index=True, right_index=True, how='outer')

# dce score metrics
spark.read.parquet(os.path.join(datadir, 'dce_scores.parquet')).createOrReplaceTempView('dce_scores')
dm = pd.merge(left=dm, right=lytx.dce_score_metrics(spark), left_index=True, right_index=True, how='outer')

# core gps metrics
spark.read.parquet(os.path.join(datadir, 'gps2.parquet')).createOrReplaceTempView('gps')
dm = pd.merge(left=dm, right=lytx.gps_metrics(spark), left_index=True, right_index=True, how='outer')

# save metrics DataFrame
dm.to_pickle(os.path.join(datadir, 'metrics.p'))
