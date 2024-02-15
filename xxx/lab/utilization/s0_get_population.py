
"""
vehicle utilization prediction model metadata and population DataFrame based on model parameters
"""

import os
import lytx
import utils
import numpy as np
import pandas as pd


# model datadir
datadir = r'/mnt/home/russell.burdt/data/utilization/amt'
assert os.path.isdir(datadir)

# vehicle utilization prediction model metadata
dc, dv, de, dem = utils.model_metadata(
    population=lytx.get_population('amt'),
    devices=['ER-SF300', 'ER-SF64'],
    window=['8/2/2021', '7/30/2022'])

# vehicle utilization prediction model population DataFrame
dp = utils.model_population(dc, dv, de)

# save metadata and population DataFrame
sdir = os.path.join(datadir, 'metadata')
if not os.path.isdir(sdir):
    os.mkdir(sdir)
dc.to_pickle(os.path.join(sdir, 'model_params.p'))
dv.to_pickle(os.path.join(sdir, 'vehicle_metadata.p'))
de.to_pickle(os.path.join(sdir, 'event_recorder_associations.p'))
dem.to_pickle(os.path.join(sdir, 'event_recorder_association_metrics.p'))
dp.to_pickle(os.path.join(datadir, 'dp.p'))
