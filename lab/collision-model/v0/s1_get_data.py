
"""
extract raw data based on collision prediction model population DataFrame
"""

import os
import lytx
import utils
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlalchemy as sa
import pyarrow.parquet as pq
from collections import defaultdict
from pyrb.mpl import open_figure, format_axes, largefonts, save_pngs
from ipdb import set_trace
from datetime import datetime
from lytx import get_conn
from glob import glob
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, BooleanType
from shutil import rmtree
from ipdb import set_trace
from tqdm import tqdm


# model datadir
datadir = r'/mnt/home/russell.burdt/data/collision-model/gwcc/v1'
assert os.path.isdir(datadir)

# metadata and population
dc = pd.read_pickle(os.path.join(datadir, 'metadata', 'model_params.p'))
dp = pd.read_pickle(os.path.join(datadir, 'metadata', 'positive_instances.p'))
dcm = pd.read_pickle(os.path.join(datadir, 'dcm-gwcc.p'))
dm = utils.get_population_metadata(dcm, dc, datadir=datadir, oversampled_info=True)
utils.validate_dcm_dp(dcm, dp)

# filter dcm for data already collected
# v0 = np.array([x.split('=')[1] for x in glob(os.path.join(datadir, 'gps.parquet', 'VehicleId=*'))])
# dcm = dcm.loc[~dcm['VehicleId'].isin(v0)].reset_index(drop=True)
# assert dm['num unique vehicle-id'] - pd.unique(dcm['VehicleId']).size == v0.size
# v0 = np.array([x.split('=')[1] for x in glob(os.path.join(datadir, 'triggers.parquet', 'EventRecorderId=*'))])
# dcm = dcm.loc[~dcm['EventRecorderId'].isin(v0)].reset_index(drop=True)
# assert dm['num unique event-recorder-id'] - pd.unique(dcm['EventRecorderId']).size == v0.size

# single window by xid
dv = utils.modify_dcm_for_data_extraction(dcm, xid='VehicleId', dc=dc)
de = utils.modify_dcm_for_data_extraction(dcm, xid='EventRecorderId', dc=dc)

# raw data extraction
# lytx.distributed_data_extraction('events', datadir, df=dv, xid='VehicleId', n=200, distributed=False, assert_new=False)
# lytx.distributed_data_extraction('behaviors', datadir, df=dv, xid='VehicleId', n=200, distributed=False, assert_new=False)
# lytx.distributed_data_extraction('trips', datadir, df=dv, xid='VehicleId', n=200, distributed=False, assert_new=False)
# lytx.distributed_data_extraction('triggers', datadir, df=de, xid='EventRecorderId', n=2000, distributed=False, assert_new=False)
# lytx.distributed_data_extraction('dce_scores', datadir, df=dv, xid='VehicleId', n=200, distributed=False, assert_new=False)
# lytx.distributed_data_extraction('gps', datadir, df=dv, xid='VehicleId', n=200, distributed=False, assert_new=False)

# SparkSession object
conf = SparkConf()
conf.set('spark.driver.memory', '32g')
conf.set('spark.driver.maxResultSize', 0)
conf.set('spark.sql.execution.arrow.pyspark.enabled', 'true')
conf.set('spark.sql.parquet.enableVectorizedReader', 'false')
conf.set('spark.sql.session.timeZone', 'UTC')
conf.set('spark.sql.shuffle.partitions', 20000)
conf.set('spark.local.dir', r'/mnt/home/russell.burdt/rbin')
spark = SparkSession.builder.config(conf=conf).getOrCreate()
spark.sparkContext.setLogLevel('ERROR')

# read events data and assert consistent schema
loc = os.path.join(datadir, f'events.parquet')
events = spark.read.parquet(loc)
events.createOrReplaceTempView('events')
lytx.validate_consistent_parquet_schema(spark, loc=loc, src='events', xid='VehicleId')

# read behaviors data and assert consistent schema
loc = os.path.join(datadir, f'behaviors.parquet')
behaviors = spark.read.parquet(loc)
behaviors.createOrReplaceTempView('behaviors')
lytx.validate_consistent_parquet_schema(spark, loc=loc, src='behaviors', xid='VehicleId')

# read trips data and assert consistent schema
loc = os.path.join(datadir, f'trips.parquet')
trips = spark.read.parquet(loc)
trips.createOrReplaceTempView('trips')
lytx.validate_consistent_parquet_schema(spark, loc=loc, src='trips', xid='VehicleId')

# # read triggers data and assert consistent schema
# loc = os.path.join(datadir, f'triggers.parquet')
# triggers = spark.read.parquet(loc)
# triggers.createOrReplaceTempView('triggers')
# lytx.validate_consistent_parquet_schema(spark, loc=loc, src='triggers', xid='EventRecorderId')

# read dce_scores data and assert consistent schema
loc = os.path.join(datadir, f'dce_scores.parquet')
dce_scores = spark.read.parquet(loc)
dce_scores.createOrReplaceTempView('dce_scores')
lytx.validate_consistent_parquet_schema(spark, loc=loc, src='dce_scores', xid='VehicleId')

# read gps data and assert consistent schema
loc = os.path.join(datadir, f'gps.parquet')
gps = spark.read.parquet(loc)
gps.createOrReplaceTempView('gps')
lytx.validate_consistent_parquet_schema(spark, loc=loc, src='gps', xid='VehicleId')

# create coverage data folder
if not os.path.isdir(os.path.join(datadir, 'coverage')):
    os.mkdir(os.path.join(datadir, 'coverage'))

# raw data time bounds by data source
bounds = {}
bounds['events'] = lytx.get_time_bounds(spark, dcm, src='events', xid='VehicleId', ts_min='TS_SEC', ts_max='TS_SEC')
bounds['behaviors'] = lytx.get_time_bounds(spark, dcm, src='behaviors', xid='VehicleId', ts_min='TS_SEC', ts_max='TS_SEC')
bounds['trips'] = lytx.get_time_bounds(spark, dcm, src='trips', xid='VehicleId', ts_min='TS_SEC1', ts_max='TS_SEC0')
# bounds['triggers'] = lytx.get_time_bounds(spark, dcm, src='triggers', xid='EventRecorderId', ts_min='TS_SEC', ts_max='TS_SEC')
bounds['dce_scores'] = lytx.get_time_bounds(spark, dcm, src='dce_scores', xid='VehicleId', ts_min='TS_SEC', ts_max='TS_SEC')
bounds['gps'] = lytx.get_time_bounds(spark, dcm, src='gps', xid='VehicleId', ts_min='TS_SEC', ts_max='TS_SEC')
with open(os.path.join(datadir, 'coverage', 'bounds.p'), 'wb') as fid:
    pickle.dump(bounds, fid)

# daily record counts by data source
records = {}
records['events'] = lytx.records_per_day_vid(datadir, spark, dcm, src='events', ts='TS_SEC')
records['behaviors'] = lytx.records_per_day_vid(datadir, spark, dcm, src='behaviors', ts='TS_SEC')
records['trips'] = lytx.records_per_day_vid(datadir, spark, dcm, src='trips', ts='TS_SEC0')
# records['triggers'] = lytx.records_per_day_vid(datadir, spark, dcm, src='triggers', ts='TS_SEC', xid='EventRecorderId')
records['dce_scores'] = lytx.records_per_day_vid(datadir, spark, dcm, src='dce_scores', ts='TS_SEC')
records['gps'] = lytx.records_per_day_vid(datadir, spark, dcm, src='gps', ts='TS_SEC')
with open(os.path.join(datadir, 'coverage', 'records.p'), 'wb') as fid:
    pickle.dump(records, fid)

# coverage of dce scores data relative to events data
dce = lytx.get_dce_scores_events_coverage_dataframe(spark)
dce.to_pickle(os.path.join(datadir, 'coverage', 'dce_scores_events_coverage.p'))

# # coverage of triggers data relative to events data
# dte = lytx.get_triggers_events_coverage_dataframe(spark)
# dte.to_pickle(os.path.join(datadir, 'coverage', 'triggers_events_coverage.p'))

# event and behavior decoders
decoder = {}
decoder['event-type'] = pd.read_sql_query(con=get_conn('edw'),
    sql=sa.text(f'SELECT Id, Name FROM hs.EventTriggerTypes_i18n'))
decoder['event-sub-type'] = pd.read_sql_query(con=get_conn('edw'),
    sql=sa.text(f'SELECT EventTriggerSubTypeId AS Id, UILabel AS Name FROM hs.EventTriggerSubTypes_i18n'))
decoder['behaviors'] = pd.read_sql_query(con=get_conn('edw'),
    sql=sa.text(f'SELECT Id, Name FROM hs.Behaviors_i18n'))
with open(os.path.join(datadir, 'decoder.p'), 'wb') as fid:
    pickle.dump(decoder, fid)
