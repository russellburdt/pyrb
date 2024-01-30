
"""
metadata for collision prediction model
"""

import os
import lytx
import utils
import numpy as np
import pandas as pd
from datetime import datetime
from ipdb import set_trace


# model params
population = 'dtft'
time0, time1 = '1/1/2023', '9/1/2023'
predictor_interval_days, collision_interval_days = 30, 90

# datadir and spark session
datadir = fr'/mnt/home/russell.burdt/data/collision-model/{population}'
assert os.path.isdir(datadir)
spark = lytx.spark_session(memory='32g')

# event recorder associations in time window
time0 = pd.Timestamp(datetime.strptime(time0, '%m/%d/%Y'))
time1 = pd.Timestamp(datetime.strptime(time1, '%m/%d/%Y'))
dx = lytx.event_recorder_associations_window(population=lytx.population_dict(population), time0=time0, time1=time1)

# filter event-recorder associations
dx = dx.loc[dx['days'] > predictor_interval_days + collision_interval_days].reset_index(drop=True).copy()
dx = lytx.event_recorder_associations_overlap(dx=dx, spark=spark)
tdm = dx.groupby('VehicleId')['td'].min()
dx = dx.loc[~dx['VehicleId'].isin(tdm[tdm < -1].index.values)].reset_index(drop=True)
for x in ['td', 'days']:
    del dx[x]

# predictor/collision intervals, population dataframe, min/max time-bounds
dt = utils.collision_prediction_model_windows(
    time0=time0, time1=time1, predictor_interval_days=predictor_interval_days, collision_interval_days=collision_interval_days, overlap_days=30)
dcm = utils.collision_prediction_model_dataframe(dx=dx, dt=dt)
dbv = pd.merge(on='VehicleId', how='inner', left=pd.merge(on='VehicleId', how='inner',
    left=dcm.groupby('VehicleId')['ta'].min().reset_index(drop=False),
    right=dcm.groupby('VehicleId')['tb'].max().reset_index(drop=False)),
    right=dcm.groupby('VehicleId')['tc'].max().reset_index(drop=False)).sort_values(['ta', 'tb', 'tc']).reset_index(drop=True)
dbe = pd.merge(on='EventRecorderId', how='inner', left=pd.merge(on='EventRecorderId', how='inner',
    left=dcm.groupby('EventRecorderId')['ta'].min().reset_index(drop=False),
    right=dcm.groupby('EventRecorderId')['tb'].max().reset_index(drop=False)),
    right=dcm.groupby('EventRecorderId')['tc'].max().reset_index(drop=False)).sort_values(['ta', 'tb', 'tc']).reset_index(drop=True)

# save metadata
dx.to_pickle(os.path.join(datadir, 'event-recorder-associations.p'))
dt.to_pickle(os.path.join(datadir, 'predictor-collision-intervals.p'))
dcm.to_pickle(os.path.join(datadir, 'population-dataframe.p'))
dbv.to_pickle(os.path.join(datadir, 'time-bounds-vehicle.p'))
dbe.to_pickle(os.path.join(datadir, 'time-bounds-eventrecorder.p'))
