
"""
enrich gps data
"""

import os
import lytx
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime
from tqdm import tqdm
from ipdb import set_trace


# datadir and population DataFrame
datadir = r'/mnt/home/russell.burdt/data/collision-model/dft'
assert os.path.isdir(datadir)
dcm = pd.read_pickle(os.path.join(datadir, 'population-dataframe.p'))

# spark session, read parquet dataset
spark = lytx.spark_session(memory='300g', cores='*')
gps = spark.read.parquet(os.path.join(datadir, 'gps.parquet'))
gps.createOrReplaceTempView('gps')
gc = gps.count()

# gps segmentation
lytx.gps_segmentation(spark=spark, datadir=datadir, src='gps.parquet', dst='gps1.parquet', service='EC2', time_interval_sec=61, distance_interval_miles=1.0)
gps = spark.read.parquet(os.path.join(datadir, 'gps1.parquet'))
gps.createOrReplaceTempView('gps')
assert gps.count() <= gc

# gps segmentation metrics
ds = lytx.gps_segmentation_metrics(dcm, spark)
ds['t0'] = ds['t0'].astype('datetime64[s]')
ds['t1'] = ds['t1'].astype('datetime64[s]')
ds = ds.rename(columns={'t0': 'ta', 't1': 'tb'})
ds = pd.merge(ds, dcm[['VehicleId', 'ta', 'tb']].reset_index(drop=False), on=['VehicleId', 'ta', 'tb'], how='inner')
ds.index = ds.pop('index')
ds.index.name = None
ds.to_pickle(os.path.join(datadir, 'gps_segmentation_metrics.p'))

# gps interval metrics
gps = spark.read.parquet(os.path.join(datadir, 'gps1.parquet'))
gps.createOrReplaceTempView('gps')
gc = gps.count()
lytx.gps_interval_metrics(spark=spark, datadir=datadir, src='gps1.parquet', dst='gps2.parquet', service='EC2')
gps = spark.read.parquet(os.path.join(datadir, 'gps2.parquet'))
gps.createOrReplaceTempView('gps')
assert gps.count() == gc
