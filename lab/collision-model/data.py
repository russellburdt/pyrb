
"""
raw data for collision prediction model
"""

import os
import lytx
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime
from ipdb import set_trace


# datadir, metadata
datadir = r'/mnt/home/russell.burdt/data/collision-model/dtft'
assert os.path.isdir(datadir)
dx = pd.read_pickle(os.path.join(datadir, 'event-recorder-associations.p'))
dt = pd.read_pickle(os.path.join(datadir, 'predictor-collision-intervals.p'))
dcm = pd.read_pickle(os.path.join(datadir, 'population-dataframe.p'))
dbv = pd.read_pickle(os.path.join(datadir, 'time-bounds-vehicle.p'))
dbe = pd.read_pickle(os.path.join(datadir, 'time-bounds-eventrecorder.p'))

# spark session
spark = lytx.spark_session(memory='32g', cores='8')

# events data over predictor and collision intervals
# df = dbv[['VehicleId', 'ta', 'tc']].rename(columns={'ta': 't0', 'tc': 't1'})
# v0 = np.array([os.path.split(x)[1][10:] for x in glob(os.path.join(datadir, 'events.parquet', 'VehicleId=*'))])
# df = df.loc[~df['VehicleId'].isin(v0)].reset_index(drop=True)
# lytx.distributed_data_extraction(datadir=datadir, dataset='events', df=df, nx=300, nd=60, spark=spark, distributed=True, overwrite=False)
# lytx.merge_parquet(spark=spark, loc=os.path.join(datadir, 'events.parquet'), partition='VehicleId')
# events = spark.read.parquet(os.path.join(datadir, 'events.parquet'))
# events.createOrReplaceTempView('events')
# dxe = lytx.records_per_vehicle_per_day(spark=spark, df=df, table='events')

# behaviors data over predictor and collision intervals
# lytx.parquet_events_to_behaviors(path=datadir, spark=spark, dataset='behaviors', nv=5000)
# lytx.merge_parquet(spark=spark, loc=os.path.join(datadir, 'behaviors.parquet'), partition='VehicleId')
# behaviors = spark.read.parquet(os.path.join(datadir, 'behaviors.parquet'))
# behaviors.createOrReplaceTempView('behaviors')
# dxb = lytx.records_per_vehicle_per_day(spark=spark, df=df, table='behaviors')

# dce-scores over predictor and collision intervals
# df = dbv[['VehicleId', 'ta', 'tc']].rename(columns={'ta': 't0', 'tc': 't1'})
# lytx.distributed_data_extraction(datadir=datadir, dataset='dce_scores', df=df, nx=200, nd=60, spark=spark, distributed=False)
# lytx.merge_parquet(spark=spark, loc=os.path.join(datadir, 'dce_scores.parquet'), partition='VehicleId')
# dce = spark.read.parquet(os.path.join(datadir, 'dce_scores.parquet'))
# dce.createOrReplaceTempView('dce')
# xdce = lytx.coverage_dce_scores_events(spark)
# dxc = lytx.records_per_vehicle_per_day(spark=spark, df=df, table='dce')

# gps over predictor and collision intervals
df = dbv[['VehicleId', 'ta', 'tc']].rename(columns={'ta': 't0', 'tc': 't1'})
lytx.distributed_data_extraction(datadir=datadir, dataset='gps', df=df, nx=6000, nd='all', spark=spark, distributed=False)

# lytx.merge_parquet(spark=spark, loc=os.path.join(datadir, 'gps.parquet'), partition='VehicleId')
# gps = spark.read.parquet(os.path.join(datadir, 'gps.parquet'))
# gps.createOrReplaceTempView('gps')
# dxg = lytx.records_per_vehicle_per_day(spark=spark, df=df, table='gps')
