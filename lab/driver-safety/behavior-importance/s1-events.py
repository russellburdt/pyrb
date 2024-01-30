
"""
events data for behavior-importance analysis
* limit concurrent queries to between 4 to 8
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
spark = lytx.spark_session(memory='32g', cores='8')

# datadir and metadata objects
datadir = os.path.join(r'/mnt/home/russell.burdt/data/driver-safety/behavior-importance', population)
assert os.path.isdir(datadir)
dbv = pd.read_pickle(os.path.join(datadir, 'time-bounds-vehicle.p'))

# extract events data
df = dbv[['VehicleId', 'ta', 'tb']].rename(columns={'ta': 't0', 'tb': 't1'})
lytx.distributed_data_extraction(datadir=datadir, dataset='events', df=df, nx=300, nd=90, spark=spark, distributed=True)
