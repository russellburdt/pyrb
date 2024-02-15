
"""
collision prediction model metadata and population DataFrame based on model parameters
"""

import os
import lytx
import utils
import numpy as np
import pandas as pd


# model datadir
datadir = r'/mnt/home/russell.burdt/data/collision-model/gwcc/v1'
assert os.path.isdir(datadir)

# collision prediction model metadata
dc, desc, dv, dp, de, dem = utils.model_metadata(
    population=lytx.get_population('gwcc'),
    collision_intervals=['11/2021', '12/2021', '1/2022', '2/2022', '3/2022', '4/2022', '5/2022', '6/2022', '7/2022'],
    devices=['ER-SF300', 'ER-SF300V2', 'ER-SF64'],
    predictor_interval_days=90)

# collision prediction model population DataFrame
dcm = utils.model_population(dc, dv, dp, de)
# dcm = dcm.loc[dcm['VehicleId'] == '9100FFFF-48A9-CC63-7A15-A8A3E03F0000'].reset_index(drop=True)

# validate and get population metadata
utils.validate_dcm_dp(dcm, dp)
dm = utils.get_population_metadata(dcm, dc, datadir=datadir)

# save metadata and population DataFrame
sdir = os.path.join(datadir, 'metadata')
if not os.path.isdir(sdir):
    os.mkdir(sdir)
dc.to_pickle(os.path.join(sdir, 'model_params.p'))
dv.to_pickle(os.path.join(sdir, 'vehicle_metadata.p'))
dp.to_pickle(os.path.join(sdir, 'positive_instances.p'))
de.to_pickle(os.path.join(sdir, 'event_recorder_associations.p'))
dem.to_pickle(os.path.join(sdir, 'event_recorder_association_metrics.p'))
with open(os.path.join(sdir, 'desc.txt'), 'w') as fid:
    fid.write(desc)
dcm.to_pickle(os.path.join(datadir, 'dcm.p'))
