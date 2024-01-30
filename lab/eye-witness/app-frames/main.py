"""
eye witness app for vehicles within location and time window
"""

import os
import cv2
import numpy as np
import pandas as pd
from functools import partial
from shutil import rmtree, copy2
from glob import glob
from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Button, Slider
from pyrb.bokeh import MapInterface, largefonts
from pyrb.processing import gps_to_webm
from tqdm import tqdm
from ipdb import set_trace


# datadir and devices
datadir = r'/mnt/home/russell.burdt/data/eye-witness/case-7-13-23'
assert os.path.isdir(datadir)
fns = glob(os.path.join(datadir, '*.dce'))
devices = [os.path.split(x)[1][:-4] for x in fns]

# static data dir
sdir = os.path.join(os.getcwd(), 'app-frames')
assert os.path.isdir(sdir)
sdir = os.path.join(sdir, 'static')
if os.path.isdir(sdir):
    rmtree(sdir)
os.mkdir(sdir)

# load and validate data for devices
gps = {}
frames = {}
urls = {}
for device in tqdm(devices, 'copy frames'):
    vdir = os.path.join(datadir, device)
    assert os.path.isdir(vdir)
    fxs = glob(os.path.join(vdir, '*'))
    assert len([fx for fx in fxs if os.path.split(fx)[1] == 'video.mp4']) == 1
    fgps = [fx for fx in fxs if os.path.split(fx)[1] == 'gps.p']
    assert len(fgps) == 1
    gps[device] = pd.read_pickle(fgps[0])
    pngs = sorted([fx for fx in fxs if os.path.split(fx)[1][:5] == 'frame'])
    assert (len(pngs) > 0) and (len(pngs) == gps[device].shape[0])
    [copy2(png, os.path.join(sdir, '-'.join(png.split(os.sep)[-2:]))) for png in pngs]
    url = 'http://10.144.240.35:5013/app-frames/static'
    frames[device] = np.array([os.path.join(url, '-'.join(png.split(os.sep)[-2:])) for png in pngs])
    assert frames[device].size == gps[device].shape[0]
    urls[device] = ColumnDataSource(data={'frame': [frames[device][0]]})

# common lat/lon window
lat0 = np.min([gps[device]['latitude'].min() for device in devices])
lat1 = np.min([gps[device]['latitude'].max() for device in devices])
lon0 = np.min([gps[device]['longitude'].min() for device in devices])
lon1 = np.min([gps[device]['longitude'].max() for device in devices])

def increment(sender):

    # update slider
    if (sender[1] == 'plus1') and (dx.loc[sender[0], 'slider'].value < frames[devices[sender[0]]].size):
        dx.loc[sender[0], 'slider'].value += 1
    elif (sender[1] == 'plus10') and (dx.loc[sender[0], 'slider'].value < frames[devices[sender[0]]].size - 10):
        dx.loc[sender[0], 'slider'].value += 10
    elif (sender[1] == 'minus1') & (dx.loc[sender[0], 'slider'].value > 1):
        dx.loc[sender[0], 'slider'].value -= 1
    elif (sender[1] == 'minus10') & (dx.loc[sender[0], 'slider'].value > 11):
        dx.loc[sender[0], 'slider'].value -= 10

def slide(attr, old, new, sender):

    # update frame
    urls[devices[sender]].data = {'frame': [
        os.path.join('http://10.144.240.35:5013/app-frames/static', f'{devices[sender]}-frame{new:07d}.png')]}

    # update frame title
    device = devices[sender]
    title = f"""{device}, {gps[device].loc[new - 1, 'localtime'].strftime('%m/%d/%Y %H:%M:%S.%f')} {gps[device].loc[new - 1, 'tz']}"""
    dx.loc[sender, 'frames'].title.text = title

    # update map position
    lon = gps[device].loc[new - 1, 'longitude']
    lat = gps[device].loc[new - 1, 'latitude']
    lon, lat = gps_to_webm(lon, lat)
    dx.loc[sender, 'maps'].position.data = {'lon': np.array([lon]), 'lat': np.array([lat])}

# layout objects for each dce
dx = pd.DataFrame(index=range(len(devices)), columns=['frames', 'maps', 'maps-fig', 'minus10', 'minus1', 'slider', 'plus1', 'plus10'])
for nd, device in enumerate(devices):

    # frames object
    height, width = cv2.imread(sorted(glob(os.path.join(datadir, device, 'frame*.png')))[0]).shape[:2]
    fig = figure(width=width, height=height, tools='pan,box_zoom,wheel_zoom,save,reset', toolbar_location='left', )
    fig.toolbar.logo = None
    fig.grid.visible = False
    fig.axis.visible = False
    fig.outline_line_color = None
    fig.x_range.start, fig.y_range.start = 0, 0
    fig.x_range.end, fig.y_range.end = width, height
    title = f"""{device}, {gps[device].loc[0, 'localtime'].strftime('%m/%d/%Y %H:%M:%S.%f')} {gps[device].loc[0, 'tz']}"""
    fig.title.text = title
    fig.image_url(url='frame', source=urls[device], x=0, y=0, w=width, h=height, anchor='bottom_left')
    largefonts(fig, 14)
    dx.loc[nd, 'frames'] = fig

    # map-interface object
    lon, lat = gps[device]['longitude'].values, gps[device]['latitude'].values
    lon, lat = gps_to_webm(lon, lat)
    mapx = MapInterface(width=width, height=height, lon0=lon0, lon1=lon1, lat0=lat0, lat1=lat1)
    mapx.path.data = {'lon': lon, 'lat': lat}
    mapx.position.data = {'lon': lon[:1], 'lat': lat[:1]}
    dx.loc[nd, 'maps'] = mapx
    if nd > 0:
        dx.loc[nd, 'maps'].fig.x_range = dx.loc[nd - 1, 'maps'].fig.x_range
        dx.loc[nd, 'maps'].fig.y_range = dx.loc[nd - 1, 'maps'].fig.y_range
    dx.loc[nd, 'maps-fig'] = mapx.fig

    # frame button and slider objects
    dx.loc[nd, 'minus1'] = Button(label='-1', width=40, height=40)
    dx.loc[nd, 'minus10'] = Button(label='-10', width=40, height=40)
    dx.loc[nd, 'slider'] = Slider(start=1, end=frames[device].size, step=1, value=1, width=300, height=40, title='Frame number')
    dx.loc[nd, 'plus1'] = Button(label='+1', width=40, height=40)
    dx.loc[nd, 'plus10'] = Button(label='+10', width=40, height=40)
    dx.loc[nd, 'minus1'].on_click(partial(increment, sender=(nd, 'minus1')))
    dx.loc[nd, 'minus10'].on_click(partial(increment, sender=(nd, 'minus10')))
    dx.loc[nd, 'plus1'].on_click(partial(increment, sender=(nd, 'plus1')))
    dx.loc[nd, 'plus10'].on_click(partial(increment, sender=(nd, 'plus10')))
    dx.loc[nd, 'slider'].on_change('value', partial(slide, sender=nd))

# app layout and document object
layout = column([row(x.tolist()) for x in dx[['frames', 'maps-fig', 'minus10', 'minus1', 'slider', 'plus1', 'plus10']].values])
doc = curdoc()
doc.add_root(layout)
doc.title = 'eye-witness frames app'
