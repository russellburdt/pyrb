
"""
create collision prediction model metrics DataFrame based on Parquet datasets
- minimal metrics used by data app only
"""

import os
import lytx
import utils
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import broadcast


# model datadir and metadata
datadir = r'/mnt/home/russell.burdt/data/collision-model/gwcc'
assert os.path.isdir(datadir)
dc = pd.read_pickle(os.path.join(datadir, 'metadata', 'model_params.p'))
dp = pd.read_pickle(os.path.join(datadir, 'metadata', 'positive_instances.p'))
dcm = pd.read_pickle(os.path.join(datadir, 'dcm.p'))
dcm = dcm[~dcm['oversampled']].reset_index(drop=True)
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

# vehicle evaluation windows by VehicleId (dfv) as Spark DataFrame and view
dcm.index.name = 'rid'
dcm = dcm.reset_index(drop=False)
dfv = dcm.loc[:, ['rid', 'VehicleId', 'time0', 'time1']].copy()
dfv['time0'] = [int((x - datetime(1970, 1, 1)).total_seconds()) for x in dfv['time0']]
dfv['time1'] = [int((x - datetime(1970, 1, 1)).total_seconds()) for x in dfv['time1']]
dfv = broadcast(spark.createDataFrame(dfv))
dfv.cache()
dfv.createOrReplaceTempView('dfv')

# vehicle evals with any gps data indicating any 2D motion
spark.read.parquet(os.path.join(datadir, 'gps2.parquet')).createOrReplaceTempView('gps')
r0 = spark.sql(f"""
    SELECT
        dfv.rid,
        COUNT(DISTINCT(gps.longitude)) AS nlon,
        COUNT(DISTINCT(gps.latitude)) AS nlat,
        MAX(gps.cumulative_distance_miles) - MIN(gps.cumulative_distance_miles) AS dmax
    FROM gps JOIN dfv
        ON gps.VehicleId = dfv.VehicleId
        AND gps.TS_SEC >= dfv.time0
        AND gps.TS_SEC <= dfv.time1
    WHERE gps.segmentId IS NOT NULL
    GROUP BY dfv.rid""").toPandas()
ok = (r0['nlon'] > 1) & (r0['nlat'] > 1) & (r0['dmax'] > 0)
rok = r0.loc[ok, 'rid'].values
dcm = dcm.loc[dcm['rid'].isin(rok)].reset_index(drop=True)
dcm.drop('rid', axis=1).to_pickle(os.path.join(datadir, 'dcm-gps.p'))

# process events dataset, handle null case
events = False
if len(glob(os.path.join(datadir, 'events.parquet', '*'))) > 0:
    spark.read.parquet(os.path.join(datadir, 'events.parquet')).createOrReplaceTempView('events')
    de = lytx.event_count_metrics(spark, span='all', suffix=None)
    events = True

# process behaviors dataset, handle null case
behaviors = False
if len(glob(os.path.join(datadir, 'behaviors.parquet', '*'))) > 0:
    spark.read.parquet(os.path.join(datadir, 'behaviors.parquet')).createOrReplaceTempView('behaviors')
    db = lytx.behavior_count_metrics(spark, span='all', suffix=None)
    behaviors = True

# merge collision model population and metrics DataFrames
if not events and not behaviors:
    df = dcm.copy()
elif not events and behaviors:
    df = pd.merge(left=dcm, right=db, on='rid', how='left')
elif events and not behaviors:
    df = pd.merge(left=dcm, right=de, on='rid', how='left')
elif events and behaviors:
    df = pd.merge(left=dcm, right=de, on='rid', how='left')
    df = pd.merge(left=df, right=db, on='rid', how='left')
assert df.shape[0] == dcm.shape[0]

# clean and save df
for col in dcm.columns:
    del df[col]
df.to_pickle(os.path.join(datadir, 'df-app.p'))
