
"""
misc dce processing
"""

import os
import pickle
import numpy as np
import pandas as pd
from dceutils import dceinfo4

# get DCEFile object
fn = r'c:/Users/russell.burdt/Downloads/2021-11-05 QM221_dvr_testpull - Outside.dce'
dce = dceinfo4.DCEFile(fn)

# extract video data as mp4 files
for tag in ['VFWE', 'VRWE']:
    form = dce.getForms(tag)
    assert len(form) == 1
    form = form[0]
    src = os.path.join(os.path.split(fn)[0], 'bytes')
    with open(src, 'wb') as fid:
        fid.write(form.extractData())
    dst = os.path.join(os.path.split(fn)[0], '{}.mp4'.format(tag))
    cmd = 'ffmpeg -framerate 10 -i {} -vcodec copy -acodec copy {}'.format(src, dst)
    os.system(cmd)
    os.remove(src)

# extract sensor data as a dictionary of pandas DataFrames
sensor = {}
for tag in ['VNTR', 'GPSR']:
    form = dce.getForms(tag)
    assert len(form) == 1
    form = form[0]
    sensor[tag] = pd.DataFrame(dceinfo4.FormParser(form).parse())
with open(os.path.join(os.path.split(fn)[0], 'sensor.p'), 'wb') as fid:
    pickle.dump(sensor, fid)
