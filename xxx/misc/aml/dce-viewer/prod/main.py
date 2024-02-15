
"""
dce viewer app
"""

import os
import sys
import boto3
import config
import numpy as np
import pandas as pd
from dceutils import dceinfo4
from shutil import rmtree, move
from bokeh.plotting import figure
from bokeh.io import show, curdoc
from bokeh.tile_providers import Vendors, get_provider
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Div, TextInput, Button
from bokeh.models import PanTool, WheelZoomTool, ResetTool
from datetime import datetime
from pyproj import Transformer
from glob import glob
from ipdb import set_trace



def empty_rbin(rbin):
    """
    empty directory 'rbin'
    """
    if not os.path.isdir(rbin):
        os.mkdir(rbin)
        return
    rmtree(rbin)
    os.mkdir(rbin)

def clear_s3():
    """
    clear dce viewer interface and video file if exists
    """

    # empty recyclebin and clear interface
    empty_rbin(rbin)
    url['text input'].value = ''
    video['video'].text = config.VIDEO_STYLE0 + '' + config.VIDEO_STYLE1

    # remove video files
    for fn in glob(os.path.join(config.SDIR, f'video_{sid}_*.mp4')):
        os.remove(fn)

    # reset maps glyph and limits
    gps.data = {'latitude': np.array([]), 'longitude': np.array([])}
    maps['osm'].x_range.start = x0
    maps['osm'].x_range.end = x1
    maps['osm'].y_range.start = y0
    maps['osm'].y_range.end = y1

def update_s3():
    """
    update dce viewer interface
    """
    empty_rbin(rbin)

    # valide text input value
    fn = url['text input'].value
    if (not url['text input'].value) or (os.sep not in fn):
        clear_s3()
        return
    fn = [x for x in fn.split(os.sep) if x]
    try:
        assert fn[0] == 's3:'
        assert s3.meta.client.head_bucket(Bucket=fn[1])['ResponseMetadata']['HTTPStatusCode'] == 200
    except:
        clear_s3()
        return

    # download s3 file
    try:
        sx = s3.Object(bucket_name=bucket, key=os.sep.join(fn[2:]))
        sx.download_file(os.path.join(rbin, fn[-1]))
    except:
        clear_s3()
        return

    # get merged video and audio directly using dceutils.dce2mkv in 'ffmpeg3' environment,
    # this is a different conda environment as ffmpeg v3 is not compatible with bokeh v2, install this environment as:
    # conda create -c conda-forge -n ffmpeg3
    # conda install -c conda-forge python=3.6
    # conda install -c conda-forge ffmpeg=3
    cmd = f'conda run -n ffmpeg3 --cwd {rbin} '
    cmd += 'python /mnt/home/russell.burdt/miniconda3/envs/ffmpeg3/lib/python3.6/site-packages/dceutils/dce2mkv.py '
    cmd += os.path.join(rbin, fn[-1])
    os.system(cmd)
    try:
        assert os.path.isfile(os.path.join(rbin, fn[-1][:-4] + '_discrete.mkv'))
        assert os.path.isfile(os.path.join(rbin, fn[-1][:-4] + '_merged.mkv'))
    except:
        clear_s3()
        return

    # clean up dce2mkv and move merged video and audio mp4 to sdir
    dt = datetime.now().strftime('%s')
    os.remove(os.path.join(rbin, fn[-1][:-4] + '_discrete.mkv'))
    move(src=os.path.join(rbin, fn[-1][:-4] + '_merged.mkv'), dst=os.path.join(config.SDIR, f'video_{sid}_{dt}.mp4'))

    # update html video tag to combined file in sdir
    src = os.path.join(config.VDIR, 'static', f'video_{sid}_{dt}.mp4')
    video['video'].text = f"""
        <video style="background-color:#222222;width:{config.VIDEO_WIDTH}px;height:{config.VIDEO_HEIGHT}px;color:white;border:0" controls autoplay>
        <source src="{src}" type="video/mp4"></video>"""

    # get high-res GPS data
    dce = dceinfo4.DCEFile(data=sx.get()['Body'].read(), filename=None)
    form = dce.getForms('GPSR')
    if len(form) != 1:
        return
    df = pd.DataFrame(dceinfo4.FormParser(form[0]).parse())

    # convert GPS units, update glyph and map limits
    webm = np.full((df.shape[0], 2), np.nan)
    for xi, (lon, lat) in df[['longitude', 'latitude']].iterrows():
        webm[xi, :] = transform(lon, lat)
    df[['longitude', 'latitude']] = webm
    gps.data = {'latitude': df['latitude'].values, 'longitude': df['longitude'].values}
    lon0, lon1 = df['longitude'].min(), df['longitude'].max()
    lat0, lat1 = df['latitude'].min(), df['latitude'].max()
    dlon, dlat = lon1 - lon0, lat1 - lat0
    if (dlon == 0) or (dlat == 0):
        return
    if dlon >= dlat:
        maps['osm'].x_range.start = lon0 - dlon * config.GPS_VIEW_SCALE
        maps['osm'].x_range.end = lon1 + dlon * config.GPS_VIEW_SCALE
        margin = (2 * dlon * config.GPS_VIEW_SCALE * aspect) - dlat
        assert margin > 0
        maps['osm'].y_range.start = lat0 - margin / 2
        maps['osm'].y_range.end = lat1 + margin / 2
    else:
        maps['osm'].y_range.start = lat0 - dlat * config.GPS_VIEW_SCALE
        maps['osm'].y_range.end = lat1 + dlat * config.GPS_VIEW_SCALE
        margin = (2 * dlat * config.GPS_VIEW_SCALE / aspect) - dlon
        assert margin > 0
        maps['osm'].x_range.start = lon0 - margin / 2
        maps['osm'].x_range.end = lon1 + margin / 2

# bokeh document and 'recyclebin' based on session id
doc = curdoc()
sid = doc.session_context.id
rbin = os.path.join(config.RBIN, sid)
os.mkdir(rbin)

# create static dir if needed
if not os.path.isdir(config.SDIR):
    os.mkdir(config.SDIR)

# s3 resource and bucket for dce files
s3 = boto3.resource(service_name='s3')
bucket = 'lytx-amlnas-us-west-2'

# s3 URL interface
url = {}
url['div'] = Div(text='<strong>S3 URI to DCE</strong>')
url['text input'] = TextInput(width=520)
url['run'] = Button(label='run', width=100, button_type='success')
url['run'].on_click(update_s3)
url['clear'] = Button(label='clear', width=100, button_type='warning')
url['clear'].on_click(clear_s3)

# video interface
video = {}
video['div'] = Div(text='<strong>merged video and audio</strong>')
video['video'] = Div(text=config.VIDEO_STYLE0 + '' + config.VIDEO_STYLE1)

# map interface
transform = Transformer.from_crs(crs_from=4326, crs_to=3857, always_xy=True).transform
maps = {}
gps = ColumnDataSource(data={'latitude': np.array([]), 'longitude': np.array([])})
kws = {
    'width': config.MAP_WIDTH, 'height': config.MAP_HEIGHT, 'toolbar_location': 'above',
    'x_axis_type': 'mercator', 'y_axis_type': 'mercator',
    'x_range': config.MAP_X_RANGE, 'y_range': config.MAP_Y_RANGE}
for name, vendor, title in zip(['osm', 'sat'], ['OSM', 'ESRI_IMAGERY'], ['Open Street Map', 'ESRI Satellite Map']):

    # tools, fig, and map provider objects
    pan, wheel, reset = PanTool(), WheelZoomTool(), ResetTool()
    maps[name] = figure(tools=[pan, wheel, reset], title=title, **kws)
    maps[name].add_tile(tile_source=get_provider(getattr(Vendors, vendor)))
    maps[name].toolbar.active_drag = pan
    maps[name].toolbar.active_scroll = wheel

    # glyph for high-res GPS data
    maps[name].circle('longitude', 'latitude', source=gps, size=10, color='red')

    # clean up
    maps[name].toolbar.logo = None
    maps[name].outline_line_width = 2
    maps[name].xaxis.visible = False
    maps[name].yaxis.visible = False
    maps[name].grid.visible = False
    maps[name].title.text_font_size = '10pt'
maps['osm'].x_range = maps['sat'].x_range
maps['osm'].y_range = maps['sat'].y_range
x0, x1 = maps['osm'].x_range.start, maps['osm'].x_range.end
y0, y1 = maps['osm'].y_range.start, maps['osm'].y_range.end
aspect = (y1 - y0) / (x1 - x0)

# layout and deploy
a = row(url['div'], url['text input'], url['run'], url['clear'])
b = video['div']
c = video['video']
d = row(maps['osm'], maps['sat'])
layout = column(a, b, c, d)
doc.add_root(layout)
