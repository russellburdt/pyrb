
"""
create clean time-series DataFrames by CompanyId and metric
"""

import os
import lytx
import utils
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime
from functools import partial
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import broadcast
from ipdb import set_trace


# model datadir
datadir = r'/mnt/home/russell.burdt/data/utilization/amt'
assert os.path.isdir(datadir)

# metadata and population
dc = pd.read_pickle(os.path.join(datadir, 'metadata', 'model_params.p'))
dp = pd.read_pickle(os.path.join(datadir, 'dp.p'))
dm = utils.get_population_metadata(dp, dc, datadir=datadir)
ds = pd.read_pickle(os.path.join(datadir, 'coverage', 'gps_segmentation_metrics.p'))

# SparkSession object
conf = SparkConf()
conf.set('spark.driver.memory', '32g')
conf.set('spark.driver.maxResultSize', 0)
conf.set('spark.sql.execution.arrow.pyspark.enabled', 'true')
conf.set('spark.sql.session.timeZone', 'UTC')
conf.set('spark.sql.shuffle.partitions', 2000)
conf.set('spark.local.dir', r'/mnt/home/russell.burdt/rbin')
spark = SparkSession.builder.config(conf=conf).getOrCreate()
spark.sparkContext.setLogLevel('ERROR')

# read Parquet datasets
loc = os.path.join(datadir, f'gps.parquet')
gps = spark.read.parquet(loc)
gps.createOrReplaceTempView('gps')

# daily evaluation table, pandas and Spark
tmin = pd.Timestamp(datetime.strptime(dm['window start time'], r'%m/%d/%Y'))
tmax = pd.Timestamp(datetime.strptime(dm['window end time'], r'%m/%d/%Y'))
time0 = pd.date_range(tmin, tmax - pd.Timedelta(days=1), freq=pd.Timedelta(days=1))
pdf = pd.DataFrame({'time0': time0, 'time1': time0 + pd.Timedelta(days=1)}).reset_index(drop=False)
pdf['ts0'] = [(x - pd.Timestamp('1970-1-1')).total_seconds() for x in pdf['time0']]
pdf['ts1'] = [(x - pd.Timestamp('1970-1-1')).total_seconds() for x in pdf['time1']]
sdf = broadcast(spark.createDataFrame(pdf))
sdf.cache()
sdf.createOrReplaceTempView('sdf')

# vehicles with gps records on all days, update dp DataFrame
v0 = spark.sql(f"""
    WITH vc AS (
        SELECT
            sdf.time0,
            sdf.time1,
            gps.VehicleId
        FROM sdf
            INNER JOIN gps
            ON gps.TS_SEC > sdf.ts0
            AND gps.TS_SEC <= sdf.ts1
        GROUP BY sdf.time0, sdf.time1, gps.VehicleId)
    SELECT VehicleId from vc GROUP BY VehicleId HAVING COUNT(*)={pdf.shape[0]}""").toPandas()
dp = dp.loc[dp['VehicleId'].isin(v0.values.flatten())].reset_index(drop=True)

# scan over companies
ml = {}
for company in pd.unique(dp['CompanyName']):
    ml[company] = {}
    vids = dp.loc[dp['CompanyName'] == company, 'VehicleId'].values
    vids = pd.DataFrame({'VehicleId': vids, 'vx': range(1, vids.size + 1)})
    ml[company]['vids'] = vids
    spark.createDataFrame(vids).createOrReplaceTempView('vx')
    dx = spark.sql(f"""
        SELECT
            sdf.time0,
            CONCAT('V', vx.vx) AS vx,
            SUM(distance_interval_miles) AS miles
        FROM sdf
            INNER JOIN gps
            ON gps.TS_SEC > sdf.ts0
            AND gps.TS_SEC <= sdf.ts1
            INNER JOIN vx
            ON vx.VehicleId = gps.VehicleId
        GROUP BY sdf.time0, vx.vx
        ORDER BY sdf.time0, vx.vx""").toPandas()
    assert dx.shape[0] / pdf.shape[0] == vids.shape[0]
    dx = dx.pivot(index='time0', columns='vx', values='miles')
    dx.columns.name = None
    dx = dx.fillna(0)
    ml[company]['data'] = dx

# save data
# with open(os.path.join(datadir, 'datasets.p'), 'wb') as fid:
#     pickle.dump(ml, fid)
