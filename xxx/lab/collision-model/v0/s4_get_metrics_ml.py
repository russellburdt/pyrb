
"""
create collision prediction model metrics DataFrame based on Parquet datasets
- full metrics used by ML model
"""

import os
import lytx
import utils
import numpy as np
import pandas as pd
from datetime import datetime
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import broadcast


# model datadir and metadata
datadir = r'/mnt/home/russell.burdt/data/collision-model/gwcc/v1'
assert os.path.isdir(datadir)
dc = pd.read_pickle(os.path.join(datadir, 'metadata', 'model_params.p'))
dp = pd.read_pickle(os.path.join(datadir, 'metadata', 'positive_instances.p'))
dcm = pd.read_pickle(os.path.join(datadir, 'dcm-gwcc.p'))
dm = utils.get_population_metadata(dcm, dc, datadir=None)

# Spark Session
conf = SparkConf()
conf.set('spark.driver.memory', '64g')
conf.set('spark.sql.execution.arrow.pyspark.enabled', 'true')
conf.set('spark.sql.parquet.enableVectorizedReader', 'false')
conf.set('spark.sql.session.timeZone', 'UTC')
conf.set('spark.local.dir', r'/mnt/home/russell.burdt/rbin')
conf.set('spark.sql.shuffle.partitions', 20000)
spark = SparkSession.builder.config(conf=conf).getOrCreate()
spark.sparkContext.setLogLevel('ERROR')

# Parquet datasets
spark.read.parquet(os.path.join(datadir, 'trips.parquet')).createOrReplaceTempView('trips')
spark.read.parquet(os.path.join(datadir, 'dce_scores.parquet')).createOrReplaceTempView('dce_scores')
spark.read.parquet(os.path.join(datadir, 'events.parquet')).createOrReplaceTempView('events')
# spark.read.parquet(os.path.join(datadir, 'triggers.parquet')).createOrReplaceTempView('triggers')
spark.read.parquet(os.path.join(datadir, 'behaviors.parquet')).createOrReplaceTempView('behaviors')
spark.read.parquet(os.path.join(datadir, 'gps2.parquet')).createOrReplaceTempView('gps')

# vehicle evaluation windows by VehicleId (dfv) and EventRecorderId (dfe) as Spark DataFrames and views
dcm.index.name = 'rid'
dcm = dcm.reset_index(drop=False)
dfv = dcm.loc[:, ['rid', 'VehicleId', 'time0', 'time1']].copy()
dfe = dcm.loc[:, ['rid', 'EventRecorderId', 'time0', 'time1']].copy()
dfv['time0'] = [int((x - datetime(1970, 1, 1)).total_seconds()) for x in dfv['time0']]
dfv['time1'] = [int((x - datetime(1970, 1, 1)).total_seconds()) for x in dfv['time1']]
dfv = broadcast(spark.createDataFrame(dfv))
dfe = broadcast(spark.createDataFrame(dfe))
dfv.cache()
dfe.cache()
dfv.createOrReplaceTempView('dfv')
dfe.createOrReplaceTempView('dfe')

# trips metrics
dt = lytx.trip_metrics(spark)
# dce score metrics
dce = lytx.dce_score_metrics(spark)
# event count metrics
de_all = lytx.event_count_metrics(spark)
# event speed metrics
dex_all = lytx.event_speed_metrics(spark)
# trigger count metrics
# dtc_all = lytx.trigger_count_metrics(spark)
# trigger speed metrics
# dtx_all = lytx.trigger_speed_metrics(spark)
# behavior count metrics
db_all = lytx.behavior_count_metrics(spark)
# core gps usage metrics
dgps = lytx.gps_metrics(spark)
# enriched gps metrics
# assert os.path.isfile(os.path.join(datadir, 'enriched_gps_metrics.p'))
# dgpse = pd.read_pickle(os.path.join(datadir, 'enriched_gps_metrics.p'))

# merge collision model population and metrics DataFrames
df = pd.merge(left=dcm, right=de_all, on='rid', how='left')
df = pd.merge(left=df, right=dt, on='rid', how='left')
df = pd.merge(left=df, right=dce, on='rid', how='left')
df = pd.merge(left=df, right=dex_all, on='rid', how='left')
# df = pd.merge(left=df, right=dtc_all, on='rid', how='left')
# df = pd.merge(left=df, right=dtx_all, on='rid', how='left')
df = pd.merge(left=df, right=db_all, on='rid', how='left')
df = pd.merge(left=df, right=dgps, on='rid', how='left')
# df = pd.merge(left=df, right=dgpse, on='rid', how='left')
assert df.shape[0] == dcm.shape[0]

# clean and save df
for col in dcm.columns:
    del df[col]
df.to_pickle(os.path.join(datadir, 'df-gwcc.p'))
