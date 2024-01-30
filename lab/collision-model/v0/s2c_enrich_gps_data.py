
"""
enrich collision prediction model gps data
- run on EMR cluster
- requires dcm.p and gps.parquet at s3://russell-s3/<s3dir>/
"""

import os
import lytx
import utils
import boto3
import numpy as np
import pandas as pd
from datetime import datetime
from pyspark import SparkConf
from pyspark.sql.functions import broadcast
from pyspark.sql import SparkSession


# spark session
spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel('ERROR')
spark.conf.set('spark.sql.shuffle.partitions', 20000)

# s3 datadir
s3dir = 'osm221107-2023-3-v2-dft'

# collision prediction model population DataFrame
boto3.client('s3').download_file(Bucket='russell-s3', Key=f'{s3dir}/dcm.p', Filename='/mnt1/s3/dcm.p')
dcm = pd.read_pickle('/mnt1/s3/dcm.p')
os.remove('/mnt1/s3/dcm.p')

# read and count gps data
gps = spark.read.parquet(f's3a://russell-s3/{s3dir}/gps2.parquet')
gps.createOrReplaceTempView('gps')
gc = gps.count()

# # test ETL pattern
# lytx.gps_test_etl_pattern(spark=spark, datadir=None, service='EMR', src=f'{s3dir}/gps.parquet', dst=f'{s3dir}/gpstest.parquet')
# gps = spark.read.parquet(f's3a://russell-s3/{s3dir}/gpstest.parquet')
# assert gps.count() == gc

# # gps segmentation
# lytx.gps_segmentation(spark=spark, datadir=None, service='EMR', src=f'{s3dir}/gps.parquet', dst=f'{s3dir}/gps1.parquet')
# gps = spark.read.parquet(f's3a://russell-s3/{s3dir}/gps1.parquet')
# gps.createOrReplaceTempView('gps')
# assert gps.count() == gc

# # gps segmentation metrics
# ds = lytx.gps_segmentation_metrics(dcm, spark)
# ds.to_pickle('/mnt1/s3/gps_segmentation_metrics.p')
# boto3.client('s3').upload_file(Filename='/mnt1/s3/gps_segmentation_metrics.p', Bucket='russell-s3', Key=f'{s3dir}/gps_segmentation_metrics.p')
# os.remove('/mnt1/s3/gps_segmentation_metrics.p')

# # gps interval metrics
# lytx.gps_interval_metrics(spark=spark, datadir=None, service='EMR', src=f'{s3dir}/gps1.parquet', dst=f'{s3dir}/gps2.parquet')
# gps = spark.read.parquet(f's3a://russell-s3/{s3dir}/gps2.parquet')
# gps.createOrReplaceTempView('gps')
# assert gps.count() == gc

# gps enrichment via Dennis Cheng SQL function
lytx.gps_enrich_dc(spark=spark, datadir=None, service='EMR', src=f'{s3dir}/gps2.parquet', dst=f'{s3dir}/gpse.parquet')
gps = spark.read.parquet(f's3a://russell-s3/{s3dir}/gpse.parquet')
gps.createOrReplaceTempView('gps')
assert gps.count() == gc

# record enrichment time
dx = spark.sql(f"""
    SELECT
        VehicleId,
        FIRST(enrichment_minutes) AS minutes,
        COUNT(*) AS n_records
    FROM gps
    GROUP BY VehicleId""").toPandas()
dx.to_pickle('/mnt1/s3/gps_enrichment_time.p')
boto3.client('s3').upload_file(Filename='/mnt1/s3/gps_enrichment_time.p', Bucket='russell-s3', Key=f'{s3dir}/gps_enrichment_time.p')
os.remove('/mnt1/s3/gps_enrichment_time.p')

# vehicle evaluation windows and enriched gps metrics
dcm.index.name = 'rid'
dcm = dcm.reset_index(drop=False)
dfv = dcm.loc[:, ['rid', 'VehicleId', 'time0', 'time1']].copy()
dfv['time0'] = [int((x - datetime(1970, 1, 1)).total_seconds()) for x in dfv['time0']]
dfv['time1'] = [int((x - datetime(1970, 1, 1)).total_seconds()) for x in dfv['time1']]
dfv = broadcast(spark.createDataFrame(dfv))
dfv.cache()
dfv.createOrReplaceTempView('dfv')
dxx = utils.gpse_metrics(spark, prefix='gpse')
dxx.to_pickle(r'/mnt1/s3/enriched_gps_metrics.p')
boto3.client('s3').upload_file(Filename='/mnt1/s3/enriched_gps_metrics.p', Bucket='russell-s3', Key=f'{s3dir}/enriched_gps_metrics.p')
os.remove('/mnt1/s3/enriched_gps_metrics.p')
