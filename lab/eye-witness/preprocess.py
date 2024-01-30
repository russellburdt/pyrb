
"""
preprocess dce file for eye-witness viewer
"""

import os
import cv2
import pytz
import pandas as pd
import numpy as np
from bokeh.io import output_file
from pyrb.bokeh import MapInterface
from datetime import datetime
from glob import glob
from dceutils import dceinfo4
from shutil import rmtree
from scipy.interpolate import interp1d


# dce file and timezone
# fn = r'/mnt/home/russell.burdt/data/eye-witness/case-7-13-23/MV00222705.dce'
# fn = r'/mnt/home/russell.burdt/data/eye-witness/case-7-13-23/MV00728113.dce'
# fn = r'/mnt/home/russell.burdt/data/eye-witness/case-7-13-23/QM00537913.dce'
fn = r'/mnt/home/russell.burdt/data/eye-witness/case-7-13-23/SF80180511.dce'
tz = 'US/Central'

# validate and read dce file, validate timezone
assert os.path.isfile(fn)
assert fn[-4:] == '.dce'
dce = dceinfo4.DCEFile(fn)
assert tz in pytz.all_timezones

# initialize datadir
datadir = fn[:-4]
if os.path.isdir(datadir):
    rmtree(datadir)
os.mkdir(datadir)

# outside video as mp4 at 10 frames per second (hard-coded)
video = dce.getForms('VFWE')
assert len(video) == 1
video = video[0]
src = os.path.join(datadir, 'bytes')
with open(src, 'wb') as fid:
    fid.write(video.extractData())
dst = os.path.join(datadir, 'video.mp4')
cmd = f'ffmpeg -framerate 10 -i {src} -vcodec copy -acodec copy -y {dst}'
os.system(cmd)
os.remove(src)

# video frames as pngs
cmd = f"""ffmpeg -i {dst} -vf fps=10 {os.path.join(datadir, 'frame%07d.png')}"""
os.system(cmd)
frames = len(glob(os.path.join(datadir, 'frame*.png')))
f0 = os.path.join(datadir, 'frame0000001.png')
assert os.path.isfile(f0)
height, width = cv2.imread(f0).shape[:2]

# gps data
gps = dce.getForms('UGPS')
assert len(gps) == 1
gps = gps[0]
gps = pd.DataFrame(dceinfo4.FormParser(gps).parse())
gps['time'] = gps['time'].astype('float')
gps = gps.sort_values('time').reset_index(drop=True)
sec = gps['time'].max() - gps['time'].min()
td = (frames / 10) - sec
assert td < 2

# convert gps data to same shape as number of frames
dx = pd.DataFrame(data={'fn': [os.path.split(x)[1] for x in sorted(glob(os.path.join(datadir, 'frame*.png')))]})
time = gps['time'].values.copy()
time -= time[0]
for col in ['latitude', 'longitude', 'speed', 'heading', 'time']:
    interp = interp1d(x=time, y=gps[col].values, kind='linear')
    dx[col] = interp(np.linspace(0, time.max(), frames))
dx['speed, mph'] = 0.621371 * dx.pop('speed')
dx['utc'] = [pd.Timestamp(datetime.utcfromtimestamp(x)) for x in dx['time']]
dx['localtime'] = [x.tz_localize('UTC').astimezone(pytz.timezone(tz)).replace(tzinfo=None) for x in dx['utc']]
dx['tz'] = tz
dx.to_pickle(os.path.join(datadir, 'gps.p'))
