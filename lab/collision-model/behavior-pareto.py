
"""
pareto of behaviors
"""

import os
import lytx
import numpy as np
import pandas as pd
from ipdb import set_trace


# datadir and spark session
datadir = r'/mnt/home/russell.burdt/data/collision-model/dtft2021'
assert os.path.isdir(datadir)
spark = lytx.spark_session(memory='16g', cores='*')

# load metadata
dx = pd.read_pickle(os.path.join(datadir, 'event-recorder-associations.p'))
dt = pd.read_pickle(os.path.join(datadir, 'predictor-collision-intervals.p'))
dcm = pd.read_pickle(os.path.join(datadir, 'population-dataframe.p'))
db = pd.read_pickle(os.path.join(datadir, 'time-bounds.p'))

# 100k random VehicleId
assert pd.unique(dcm['VehicleId']).size > 100e3
vids = pd.DataFrame({'VehicleId': np.random.choice(pd.unique(dcm['VehicleId']), size=100000, replace=False)})
spark.createDataFrame(vids).createOrReplaceTempView('vids')

# behavior pareto for 100k random VehicleId
behaviors = spark.read.parquet(os.path.join(datadir, 'behaviors.parquet'))
behaviors.createOrReplaceTempView('behaviors')
dbs = spark.sql(f"""
    SELECT BehaviorId, BehaviorName, COUNT(*) AS nb
    FROM behaviors
        INNER JOIN vids ON behaviors.VehicleId = vids.VehicleId
    GROUP BY BehaviorId, BehaviorName
    ORDER BY nb DESC""").toPandas()
dbs.to_pickle(r'/mnt/home/russell.burdt/data/dbs2020.p')
