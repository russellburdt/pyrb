
"""
enrich collision prediction model gps data
- run on single-node EC2 instance
- requires dcm.p and gps.parquet at datadir
"""

import os
import lytx
import utils
import numpy as np
import pandas as pd
from datetime import datetime
from pyspark import SparkConf
from pyspark.sql.functions import broadcast
from pyspark.sql import SparkSession


# spark session
conf = SparkConf()
conf.set('spark.driver.memory', '64g')
conf.set('spark.driver.maxResultSize', 0)
conf.set('spark.sql.execution.arrow.pyspark.enabled', 'true')
conf.set('spark.sql.parquet.enableVectorizedReader', 'false')
conf.set('spark.sql.session.timeZone', 'UTC')
conf.set('spark.sql.shuffle.partitions', 20000)
conf.set('spark.local.dir', r'/mnt/home/russell.burdt/rbin')
spark = SparkSession.builder.master('local[*]').config(conf=conf).getOrCreate()
spark.sparkContext.setLogLevel('ERROR')

# collision prediction model population DataFrame
datadir = r'/mnt/home/russell.burdt/data/collision-model/gwcc/v1'
assert os.path.isdir(datadir)
dcm = pd.read_pickle(os.path.join(datadir, 'dcm-gwcc.p'))

# read and count gps data
gps = spark.read.parquet(os.path.join(datadir, f'gps.parquet'))
gps.createOrReplaceTempView('gps')
gc = gps.count()

# test ETL pattern
lytx.gps_test_etl_pattern(spark=spark, datadir=datadir, src='gps.parquet', dst='gpstest.parquet', service='EC2')
gps = spark.read.parquet(os.path.join(datadir, 'gpstest.parquet'))
assert gps.count() == gc

# gps segmentation
lytx.gps_segmentation(spark=spark, datadir=datadir, src='gps.parquet', dst='gps1.parquet', service='EC2', time_interval_sec=61, distance_interval_miles=1.0)
gps = spark.read.parquet(os.path.join(datadir, 'gps1.parquet'))
gps.createOrReplaceTempView('gps')
assert gps.count() == gc

# gps segmentation metrics
ds = lytx.gps_segmentation_metrics(dcm, spark)
ds.to_pickle(os.path.join(datadir, 'gps_segmentation_metrics.p'))

# gps interval metrics
lytx.gps_interval_metrics(spark=spark, datadir=datadir, src='gps1.parquet', dst='gps2.parquet', service='EC2')
gps = spark.read.parquet(os.path.join(datadir, 'gps2.parquet'))
gps.createOrReplaceTempView('gps')
assert gps.count() == gc

# # legacy gps enrichment
# lytx.gps_enrich_dc(spark=spark, datadir=datadir, src='gps2.parquet', dst='gpse.parquet', service='EC2')
# gps = spark.read.parquet(os.path.join(datadir, 'gpse.parquet'))
# gps.createOrReplaceTempView('gps')
# assert gps.count() == gc

# # distance-normalized gps enrichment
# lytx.gps_enrich_dc_normalized(spark=spark, datadir=datadir, src='gps2.parquet', service='EC2')
# gpse = spark.read.parquet(os.path.join(datadir, 'gpse.parquet'))
# gpse.createOrReplaceTempView('gpse')
# gpsm = spark.read.parquet(os.path.join(datadir, 'gpsm.parquet'))
# gpsm.createOrReplaceTempView('gpsm')
# assert gpse.count() > gpsm.count()

# # record enrichment time
# dx = spark.sql(f"""
#     SELECT
#         VehicleId,
#         FIRST(enrichment_minutes) AS minutes,
#         COUNT(*) AS n_records
#     FROM gpse
#     GROUP BY VehicleId""").toPandas()
# dx.to_pickle(os.path.join(datadir, 'gps_enrichment_time.p'))

# # vehicle evaluation windows and enriched gps metrics
# dcm.index.name = 'rid'
# dcm = dcm.reset_index(drop=False)
# dfv = dcm.loc[:, ['rid', 'VehicleId', 'time0', 'time1']].copy()
# dfv['time0'] = [int((x - datetime(1970, 1, 1)).total_seconds()) for x in dfv['time0']]
# dfv['time1'] = [int((x - datetime(1970, 1, 1)).total_seconds()) for x in dfv['time1']]
# dfv = broadcast(spark.createDataFrame(dfv))
# dfv.cache()
# dfv.createOrReplaceTempView('dfv')
# # dxx = lytx.gpse_metrics(spark, prefix='gpse')
# # dxx.to_pickle(os.path.join(datadir, 'enriched_gps_metrics.p'))
# # dx = pd.read_pickle(os.path.join(datadir, 'enriched_gps_metrics.p'))

# # core gps metrics and distance-normalized metrics
# dg = lytx.gps_metrics(spark)
# dxx = lytx.gpsn_metrics(spark, prefix='gpsn')
# dxx.to_pickle(os.path.join(datadir, 'enriched_gps_metrics.p'))
