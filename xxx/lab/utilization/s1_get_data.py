
"""
extract and clean raw data based on vehicle utilization prediction model population DataFrame
"""

import os
import lytx
import utils
import numpy as np
import pandas as pd
from datetime import datetime
from functools import partial
from pyspark import SparkConf
from pyspark.sql import SparkSession
from ipdb import set_trace


# model datadir
datadir = r'/mnt/home/russell.burdt/data/utilization/amt'
assert os.path.isdir(datadir)

# metadata and population
dc = pd.read_pickle(os.path.join(datadir, 'metadata', 'model_params.p'))
dp = pd.read_pickle(os.path.join(datadir, 'dp.p'))
dm = utils.get_population_metadata(dp, dc, datadir=datadir)
assert pd.unique(dp['VehicleId']).size == dp.shape[0]
assert all([pd.unique(dp[x]).size == 1 for x in ['time0', 'time1']])

# distributed gps data extraction
lytx.distributed_data_extraction(dataset='gps', datadir=datadir, df=dp, xid='VehicleId', n=100, distributed=True, assert_new=True)

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

# create coverage data folder
if not os.path.isdir(os.path.join(datadir, 'coverage')):
    os.mkdir(os.path.join(datadir, 'coverage'))

# time bounds and daily record count
bounds = lytx.get_bounds_by_src_vid(spark)
count = lytx.records_per_day_vehicle(spark, dp, 'gps')
bounds.to_pickle(os.path.join(datadir, 'coverage', 'bounds.p'))
count.to_pickle(os.path.join(datadir, 'coverage', 'gps_coverage.p'))

# gps segmentation
lytx.gps_segmentation(spark, loc)
gps = spark.read.parquet(loc)
gps.createOrReplaceTempView('gps')
r0 = lytx.count_all(spark, 'gps')

# gps data segmentation metrics
ds = lytx.gps_segmentation_metrics(dp, spark)
ds.to_pickle(os.path.join(datadir, 'coverage', 'gps_segmentation_metrics.p'))

# interpolate gps data
lytx.interpolate_gps(spark, loc)
gps = spark.read.parquet(loc)
gps.createOrReplaceTempView('gps')
assert lytx.count_all(spark, 'gps') > r0
r0 = lytx.count_all(spark, 'gps')

# gps interval metrics
lytx.gps_interval_metrics(spark, loc, include_daily_coverage=True)
gps = spark.read.parquet(loc)
gps.createOrReplaceTempView('gps')
assert lytx.count_all(spark, 'gps') == r0

# geocode gps data, reload and validate
xs = spark.sql(f'SELECT DISTINCT VehicleId FROM gps')
xs.groupby('VehicleId').applyInPandas(partial(lytx.distributed_geocode, loc=loc), schema=xs.schema).toPandas()
gps = spark.read.parquet(loc)
gps.createOrReplaceTempView('gps')
assert lytx.count_all(spark, 'gps') == r0
