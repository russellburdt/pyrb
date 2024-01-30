
"""
metadata for behavior-importance analysis
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
time0, time1 = '6/1/2022', '7/1/2023'
spark = lytx.spark_session(memory='32g', cores='*')

# datadir
datadir = os.path.join(r'/mnt/home/russell.burdt/data/driver-safety/behavior-importance', population)
assert os.path.isdir(datadir)

# event-recorder-assocations
time0 = pd.Timestamp(datetime.strptime(time0, '%m/%d/%Y'))
time1 = pd.Timestamp(datetime.strptime(time1, '%m/%d/%Y'))
assert time0.day == time1.day == 1
dx = lytx.event_recorder_associations_window(population=lytx.population_dict(population), time0=time0, time1=time1)
dx = lytx.event_recorder_associations_overlap(dx=dx, spark=spark)
tdm = dx.groupby('VehicleId')['td'].min()
dx = dx.loc[~dx['VehicleId'].isin(tdm[tdm < -1].index.values)].reset_index(drop=True)
for x in ['td', 'days']:
    del dx[x]

# calendar month time intervals
dt = np.hstack((time0, np.array([pd.Timestamp(x) for x in pd.date_range(time0, time1, freq='M') + pd.Timedelta(days=1)])))
dt = pd.DataFrame(data={'ta': dt[:-1], 'tb': dt[1:]})

# driver-safety population dataframe, time-bounds dataframe
dp = []
for _, (ta, tb) in dt.iterrows():
    dxx = dx.loc[(dx['CreationDate'] < ta) & (dx['DeletedDate'] > tb)].copy()
    assert pd.unique(dxx['VehicleId']).size == dxx.shape[0]
    assert pd.unique(dxx['EventRecorderId']).size == dxx.shape[0]
    dxx['ta'], dxx['tb'] = ta, tb
    dp.append(dxx)
dp = pd.concat(dp).sort_values(['VehicleId', 'ta']).reset_index(drop=True)
dbv = pd.merge(on='VehicleId', how='inner',
    left=dp.groupby('VehicleId')['ta'].min().reset_index(drop=False),
    right=dp.groupby('VehicleId')['tb'].max().reset_index(drop=False)).sort_values(['ta', 'tb']).reset_index(drop=True)
dbe = pd.merge(on='EventRecorderId', how='inner',
    left=dp.groupby('EventRecorderId')['ta'].min().reset_index(drop=False),
    right=dp.groupby('EventRecorderId')['tb'].max().reset_index(drop=False)).sort_values(['ta', 'tb']).reset_index(drop=True)
dx.to_pickle(os.path.join(datadir, 'event-recorder-associations.p'))
dt.to_pickle(os.path.join(datadir, 'intervals.p'))
dp.to_pickle(os.path.join(datadir, 'population-dataframe.p'))
dbv.to_pickle(os.path.join(datadir, 'time-bounds-vehicle.p'))
dbe.to_pickle(os.path.join(datadir, 'time-bounds-eventrecorder.p'))
