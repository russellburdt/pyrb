
"""
clean events raw data, extract behaviors data for behavior-importance analysis
"""

import os
import lytx
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime
from ipdb import set_trace


# parameters and spark session
population = 'dist300'
spark = lytx.spark_session(memory='32g', cores='*')

# datadir and metadata objects
datadir = os.path.join(r'/mnt/home/russell.burdt/data/driver-safety/behavior-importance', population)
assert os.path.isdir(datadir)
dbv = pd.read_pickle(os.path.join(datadir, 'time-bounds-vehicle.p'))

# clean events data
df = dbv[['VehicleId', 'ta', 'tb']].rename(columns={'ta': 't0', 'tb': 't1'})
lytx.merge_parquet(spark=spark, loc=os.path.join(datadir, 'events.parquet'), partition='VehicleId')
events = spark.read.parquet(os.path.join(datadir, 'events.parquet'))
events.createOrReplaceTempView('events')

# extract behaviors data
lytx.parquet_events_to_behaviors(path=datadir, spark=spark, dataset='behaviors', nv=5000)
lytx.merge_parquet(spark=spark, loc=os.path.join(datadir, 'behaviors.parquet'), partition='VehicleId')
