
"""
Lytx-vehicle eye-witness application
"""

import os
import ffmpeg
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
from shutil import copy, rmtree, copytree
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import Div, Select, LegendItem, Span, CrosshairTool, Button, TextInput, Tabs, TabPanel, Slider, RadioGroup
from bokeh.events import MouseMove, MouseLeave
from pyrb.bokeh import console_str_objects, MapInterface2, MultiLineInterface, FrameInterface
from dceutils import dceinfo4
from pyproj import Transformer, Geod
from geopandas import points_from_xy
from scipy.interpolate import interp1d
from glob import glob
from ipdb import set_trace


# datadir
datadir = r'/mnt/home/russell.burdt/data/collision-reconstruction/app'
assert os.path.isdir(datadir)
fdir = os.path.join(datadir, 'frames')

# clear cached data
if os.path.isdir(fdir):
    rmtree(fdir)
os.mkdir(fdir)
for fn in glob(os.path.join(os.getcwd(), 'app', 'static', 'collision*.mkv')):
    os.remove(fn)
for xs in ['collision_forward', 'collision_rear', 'nearby_forward', 'nearby_rear']:
    if os.path.isdir(os.path.join(os.getcwd(), 'app', 'static', xs)):
        rmtree(os.path.join(os.getcwd(), 'app', 'static', xs))

# load data
dm = pd.read_pickle(os.path.join(datadir, 'population.p'))
days, vehicles = (dm.loc[0, 't1'] - dm.loc[0, 't0']).days, dm.shape[0]
dx = pd.read_pickle(os.path.join(datadir, 'collisions.p'))

# app params, geo-objects
w0 = 360
w1, h1 = 640, 400
w2, h2 = 460, 220
w3 = 400
w4, h4 = 700, 400
w5, h5 = 500, 275
tf = '%m/%d/%Y %H:%M:%S'
transform = Transformer.from_crs(crs_from=4326, crs_to=3857, always_xy=True).transform
geod = Geod(ellps='WGS84')

def nearby_vehicle_definition_select(attr, old, new):
    global dn

    # nearby vehicles DataFrame
    td, xd = [int(x.strip().split(' ')[0]) for x in nv_select.value.split(',')]
    dn = pd.read_pickle(os.path.join(datadir, f'nearby_vehicles_td{td}_xd{xd}.p'))
    assert all(dn['td'] == td) and all(dn['xd'] == xd)

    # collision select interface
    ok = pd.unique(dn['id'])
    cx_select.remove_on_change('value', collision_select)
    cx_select.title = f'{ok.size}x collisions with nearby vehicles'
    cx_select.options = [f'collision{x}' for x in ok]
    cx_select.value = cx_select.options[0]
    cx_select.on_change('value', collision_select)
    collision_select(None, None, None)

def collision_select(attr, old, new):

    # collision metadata
    cx = int(cx_select.value[9:])
    cs = f"""
        RecordDate, {dx.loc[cx, 'RecordDate'].strftime(tf)}<br>
        SerialNumber, {dx.loc[cx, 'SerialNumber']}<br>
        SpeedAtTrigger, {dx.loc[cx, 'SpeedAtTrigger']}<br>
        VehicleId, {dx.loc[cx, 'VehicleId']}<br>
        Model, {dx.loc[cx, 'Model']}<br>
        Company, {dx.loc[cx, 'CompanyName']}<br>
        publicroad1, {dx.loc[cx].to_dict().get('publicroad1', 'na')}<br>
        locality, {dx.loc[cx].to_dict().get('locality', 'na')}<br>
        timezone_locality_tzid, {dx.loc[cx].to_dict().get('timezone_locality_tzid', 'na')}<br>
        publicroad1_class, {dx.loc[cx].to_dict().get('publicroad1_class', 'na')}<br>"""
    c0, c1 = console_str_objects(w0)
    cx_metadata.text = c0 + cs + c1

    # copy collision video to app static dir
    fn = os.path.join(datadir, 'collision-videos', f'collision-{cx:04d}.mkv')
    assert os.path.isfile(fn)
    sdir = os.path.join(os.getcwd(), 'app', 'static')
    if not os.path.isdir(sdir):
        os.mkdir(sdir)
    dst = os.path.join(sdir, os.path.split(fn)[1])
    if not os.path.isfile(dst):
        copy(src=fn, dst=dst)
    doc.session_context.remove.append(dst)

    # html video tag
    path = os.path.join('http://10.140.72.174:5009/app', 'static', os.path.split(dst)[1])
    collision_video.text = f"""
        <video style="background-color:#222222;width:{w1}px;height:220px;color:white;border:0" controls>
        <source src="{path}" type="video/mp4"></video>"""

    # nearby vehicle select interface
    dv = dn.loc[dn['id'] == cx]
    vids = pd.unique(dv['VehicleId']).tolist()
    assert len(vids) == dv.shape[0]
    nx_select.remove_on_change('value', nearby_vehicle_select)
    nx_select.title = f'{dv.shape[0]}x nearby vehicles'
    nx_select.options = vids
    nx_select.value = vids[0]
    nx_select.on_change('value', nearby_vehicle_select)
    nearby_vehicle_select(None, None, None)

def nearby_vehicle_select(attr, old, new):
    global d1
    global d2

    # reset map
    mapx.legend.items = []
    mapx.position1.data = {'lon': np.array([]), 'lat': np.array([])}
    mapx.position2.data = {'lon': np.array([]), 'lat': np.array([])}

    # reset nearby vehicle data selection and interface
    pre_seconds.value = '10'
    post_seconds.value = '10'
    nearby_vehicle_data_status.text = ''

    # collision and nearby vehicle
    cx = int(cx_select.value[9:])
    collision = dx.loc[cx]
    nearby = dn.loc[(dn['id'] == cx) & (dn['VehicleId'] == nx_select.value)]
    assert nearby.shape[0] == 1
    nearby = nearby.squeeze()

    # update nearby vehicle metadata
    dmnv = dm.loc[dm['SerialNumber'] == nearby['SerialNumber']]
    assert dmnv.shape[0] == 1
    dmnv = dmnv.squeeze()
    cs = f"""
        SerialNumber, {nearby['SerialNumber']}<br>
        Company, {dmnv['CompanyName']}<br>
        {nearby['count']} nearby GPS records"""
    c0, c1 = console_str_objects(w0)
    nx_metadata.text = c0 + cs + c1

    # gps data for vehicle in collision from dce file
    fn = glob(os.path.join(datadir, 'collision-videos', f'collision-{cx:04d}.dce'))
    assert len(fn) == 1
    d1 = pd.DataFrame(dceinfo4.FormParser(dceinfo4.DCEFile(fn[0]).getForms('GPSR')[0]).parse())
    d1 = d1[['time', 'longitude', 'latitude', 'heading', 'speed(kph)']].copy().rename(columns={'time': 'ts_sec', 'speed(kph)': 'speed'})
    d1['ts_sec'] = d1['ts_sec'].astype('float')

    # set_trace()
    # (collision['RecordDate'] - pd.Timestamp(datetime(1970, 1, 1))).total_seconds()

    # gps data for vehicle in collision from Parquet dataset
    fn = glob(os.path.join(datadir, 'gps.parquet', f"""VehicleId={collision['VehicleId']}""", '*.parquet'))
    assert len(fn) <= 1
    if len(fn) == 1:
        d2 = pd.read_parquet(fn[0])
        d2 = d2[['ts_sec', 'longitude', 'latitude', 'heading', 'speed']].copy()
        d1 = pd.concat((d1, d2), axis=0)
    d1 = d1.sort_values('ts_sec').reset_index(drop=True)

    # gps data for nearby vehicle
    fn = glob(os.path.join(datadir, 'gps.parquet', f"""VehicleId={nearby['VehicleId']}""", '*.parquet'))
    assert (len(fn) == 1) and os.path.isfile(fn[0])
    d2 = pd.read_parquet(fn[0])[['ts_sec', 'longitude', 'latitude', 'heading', 'speed']].copy().sort_values('ts_sec').reset_index(drop=True)

    # consistent filter to gps data for vehicle in collision and nearby vehicle
    ta = ((collision['RecordDate'] - pd.Timedelta(seconds=3600)) - pd.Timestamp(datetime(1970, 1, 1))).total_seconds()
    tb = ((collision['RecordDate'] + pd.Timedelta(seconds=3600)) - pd.Timestamp(datetime(1970, 1, 1))).total_seconds()
    d1 = d1.loc[(d1['ts_sec'] >= ta) & (d1['ts_sec'] <= tb)].reset_index(drop=True)
    d2 = d2.loc[(d2['ts_sec'] >= ta) & (d2['ts_sec'] <= tb)].reset_index(drop=True)
    assert (d1.size > 0) & (d2.size > 0)

    # map gps path data source for vehicle in collision
    lon, lat = transform(xx=d1['longitude'].values, yy=d1['latitude'].values)
    mapx.path1.data = {'lon': lon, 'lat': lat}
    mapx.legend.items.append(LegendItem(label='gps data, vehicle in collision', renderers=[x for x in mapx.fig.renderers if x.name == 'path1']))
    lon0, lon1 = lon.min(), lon.max()
    lat0, lat1 = lat.min(), lat.max()

    # map gps path data source for nearby collision
    lon, lat = transform(xx=d2['longitude'].values, yy=d2['latitude'].values)
    mapx.path2.data = {'lon': lon, 'lat': lat}
    mapx.legend.items.append(LegendItem(label='gps data, nearby vehicle', renderers=[x for x in mapx.fig.renderers if x.name == 'path2']))
    lon0, lon1 = min(lon.min(), lon0), max(lon.max(), lon1)
    lat0, lat1 = min(lat.min(), lat0), max(lat.max(), lat1)
    mapx.reset_map_view(lon0=lon0, lon1=lon1, lat0=lat0, lat1=lat1, convert=False)

    # collision marker
    lon, lat = transform(xx=collision['Longitude'], yy=collision['Latitude'])
    mapx.event.data = {'lon': np.array([lon]), 'lat': np.array([lat])}
    mapx.legend.items.append(LegendItem(label='collision location', renderers=[x for x in mapx.fig.renderers if x.name == 'event']))

    # distance from collision for vehicle in collision and nearby vehicle
    _, _, d1['distance'] = geod.inv(lons1=np.full(d1.shape[0], collision['Longitude']), lats1=np.full(d1.shape[0], collision['Latitude']), lons2=d1['longitude'].values, lats2=d1['latitude'].values)
    _, _, d2['distance'] = geod.inv(lons1=np.full(d2.shape[0], collision['Longitude']), lats1=np.full(d2.shape[0], collision['Latitude']), lons2=d2['longitude'].values, lats2=d2['latitude'].values)

    # update multi-line interface
    mx1.data_sources[0].data = {
        'x': np.array([pd.Timestamp(datetime.utcfromtimestamp(x)) for x in d1['ts_sec']]),
        'y': d1['distance'].values}
    mx2.data_sources[0].data = {
        'x': np.array([pd.Timestamp(datetime.utcfromtimestamp(x)) for x in d2['ts_sec']]),
        'y': d2['distance'].values}
    mx1.fig.x_range.start = datetime.utcfromtimestamp(min(d1['ts_sec'].min(), d2['ts_sec'].min()))
    mx1.fig.x_range.end = datetime.utcfromtimestamp(max(d1['ts_sec'].max(), d2['ts_sec'].max()))

def mapx_tiles(attr, old, new):
    mapx.update_tile_source(mapx_radio_group.labels[mapx_radio_group.active])

def mapv_tiles(attr, old, new):
    mapv.update_tile_source(mapv_radio_group.labels[mapv_radio_group.active])

def mouse_move(event):

    # null cases
    if event.y < 0:
        mapx.position1.data = {'lon': np.array([]), 'lat': np.array([])}
        mapx.position2.data = {'lon': np.array([]), 'lat': np.array([])}
        return
    epoch = (1e-3) * event.x
    c0 = epoch < d1.iloc[0]['ts_sec']
    c1 = epoch > d1.iloc[-1]['ts_sec']
    c2 = epoch < d2.iloc[0]['ts_sec']
    c3 = epoch > d2.iloc[-1]['ts_sec']
    if (c0 or c1) and (c2 or c3):
        mapx.position1.data = {'lon': np.array([]), 'lat': np.array([])}
        mapx.position2.data = {'lon': np.array([]), 'lat': np.array([])}
        return

    # closest record in d1 and d2
    x1 = np.abs(d1['ts_sec'] - epoch).idxmin()
    x2 = np.abs(d2['ts_sec'] - epoch).idxmin()

    # update position glyphs on map
    lon, lat = transform(xx=d1.loc[x1, 'longitude'], yy=d1.loc[x1, 'latitude'])
    mapx.position1.data = {'lon': np.array([lon]), 'lat': np.array([lat])}
    lon, lat = transform(xx=d2.loc[x2, 'longitude'], yy=d2.loc[x2, 'latitude'])
    mapx.position2.data = {'lon': np.array([lon]), 'lat': np.array([lat])}

def mouse_leave(event):
    mapx.position1.data = {'lon': np.array([]), 'lat': np.array([])}
    mapx.position2.data = {'lon': np.array([]), 'lat': np.array([])}

def dce_form_video_to_frames(video, alignment, d0, d1):

    # validate tags
    if video.tag == 'VFWE':
        assert alignment.tag in ['UFVI', 'FNDX']
    elif video.tag == 'VRWE':
        assert alignment.tag in ['UIVI', 'RNDX']
    else:
        raise ValueError

    # dce form to mp4
    src = os.path.join(d0, 'bytes')
    with open(src, 'wb') as fid:
        fid.write(video.extractData())
    dst = os.path.join(d0, f'{video.tag}.mp4')
    cmd = f'ffmpeg -hide_banner -loglevel error -framerate 10 -i {src} -vcodec copy -acodec copy -y {dst}'
    rv = os.system(cmd)
    assert rv == 0
    os.remove(src)
    sec_mp4 = float(ffmpeg.probe(dst)['format']['duration'])

    # mp4 to frames, include timestamp in filename to avoid browser caching issues
    now = str((pd.Timestamp(datetime.utcnow()) - pd.Timestamp(datetime(1970, 1, 1))).total_seconds()).replace('.', '')
    assert not os.path.isdir(os.path.join(d0, d1))
    os.mkdir(os.path.join(d0, d1))
    cmd = f"""ffmpeg -hide_banner -loglevel error -i {dst} -vf fps=10 {os.path.join(d0, d1, f'frame{now}_%07d.png')}"""
    rv = os.system(cmd)
    assert rv == 0
    os.remove(dst)
    frames = np.array(sorted(glob(os.path.join(d0, d1, '*.png'))))
    assert sec_mp4 == frames.size / 10

    # alignment data, validate frames consistent with more than 95% of alignment records
    da = pd.DataFrame(dceinfo4.FormParser(alignment).parse())
    assert frames.size <= da.shape[0]
    if frames.size < da.shape[0]:
        assert 1 - ((da.shape[0] - frames.size) / da.shape[0]) > 0.95
        xd = da.shape[0] - frames.size
        da = da.loc[xd:].reset_index(drop=True)
        assert da.shape[0] == frames.size

    return pd.DataFrame(data={'frames': frames, 'time': da['time'].values.astype('float')})

def dce_form_gps_to_df(dce):

    assert dce.tag in ['UGPS', 'GPSR']
    gps = pd.DataFrame(dceinfo4.FormParser(dce).parse())
    gps['time'] = gps['time'].astype('float')
    gps = gps.sort_values('time').reset_index(drop=True)
    gps['utc'] = [datetime.fromtimestamp(x) for x in gps['time'].values]
    return gps[['utc', 'time', 'latitude', 'longitude']].copy()

def merge_gps_frames(forward, rear, gps):

    # merge gps and forward DataFrames
    gps = gps[['time', 'latitude', 'longitude']].copy()
    forward = forward.rename(columns={'frames': 'frames-forward'})
    df = pd.merge(gps, forward, on='time', how='outer').sort_values('time').reset_index(drop=True)
    assert pd.unique(df['time']).size == df.shape[0]

    # merge with rear if provided
    if rear is not None:
        rear = rear.rename(columns={'frames': 'frames-rear'})
        df = pd.merge(df, rear, on='time', how='outer').sort_values('time').reset_index(drop=True)
        assert pd.unique(df['time']).size == df.shape[0]
    else:
        df['frames-rear'] = None

    # clean up and return
    df['utc'] = [datetime.fromtimestamp(x) for x in df['time']]
    return df

def nearby_vehicle_extract_data_dce():
    """
    create frames and gps data for nearby vehicle and vehicle in collision
    - uses manually-saved dce file for nearby vehicle (from Video Search)
    """
    print('extracting data for nearby vehicle')

    # clear frames dir
    if os.path.isdir(fdir):
        rmtree(fdir)
    os.mkdir(fdir)

    # collision and nearby vehicle
    cx = int(cx_select.value[9:])
    collision = dx.loc[cx]
    nearby = dn.loc[(dn['id'] == cx) & (dn['VehicleId'] == nx_select.value)]
    assert nearby.shape[0] == 1
    nearby = nearby.squeeze()

    # time-window to save frames and gps data
    ta = collision['RecordDate'] - pd.Timedelta(seconds=float(pre_seconds.value))
    tb = collision['RecordDate'] + pd.Timedelta(seconds=float(post_seconds.value))

    # validate dce file for nearby vehicle, handle null case
    fn = os.path.join(os.path.split(datadir)[0], 'nearby-vehicle-dce', f"""{nearby['SerialNumber']}outside.dce""")
    if not os.path.isfile(fn):
        c0, c1 = console_str_objects(w2)
        nearby_vehicle_data_status.text = c0 + 'no dce for nearby vehicle' + c1
        return

    # gps data for nearby vehicle
    dce = dceinfo4.DCEFile(fn)
    forms = [x.tag for x in dce.getForms()]
    assert 'UGPS' in forms
    gps = dce.getForms('UGPS')
    assert len(gps) == 1
    gps = dce_form_gps_to_df(gps[0])
    assert gps.iloc[0]['utc'] < collision['RecordDate'] < gps.iloc[-1]['utc']

    # forward video frames
    assert ('VFWE' in forms) and ('UFVI' in forms)
    vfwe, ufvi = dce.getForms('VFWE'), dce.getForms('UFVI')
    assert (len(vfwe) == 1) and (len(ufvi) == 1)
    forward = dce_form_video_to_frames(vfwe[0], ufvi[0], d0=fdir, d1='nearby_forward')

    # process rear video if possible
    rear = None
    rn = os.path.join(os.path.split(datadir)[0], 'nearby-vehicle-dce', f"""{nearby['SerialNumber']}inside.dce""")
    if os.path.isfile(rn):

        # validate same gps data
        dce = dceinfo4.DCEFile(rn)
        forms = [x.tag for x in dce.getForms()]
        assert 'UGPS' in forms
        rgps = dce.getForms('UGPS')
        assert len(rgps) == 1
        rgps = dce_form_gps_to_df(rgps[0])
        assert all(gps == rgps)

        # rear video frames
        assert ('VRWE' in forms) and ('UIVI' in forms)
        vrwe, uivi = dce.getForms('VRWE'), dce.getForms('UIVI')
        assert (len(vrwe) == 1) and (len(uivi) == 1)
        rear = dce_form_video_to_frames(vrwe[0], uivi[0], d0=fdir, d1='nearby_rear')

    # merge gps and frames data for nearby vehicle
    df = merge_gps_frames(forward=forward, rear=rear, gps=gps)

    # filter by time-window, remove unused frames, save data for nearby vehicle
    df = df.loc[(df['utc'] > ta) & (df['utc'] < tb)].reset_index(drop=True)
    keep = np.hstack((df['frames-forward'].values, df['frames-rear'].values))
    keep = keep[~pd.isnull(keep)]
    frames_all = forward['frames'].values
    if rear is not None:
        frames_all = np.hstack((frames_all, rear['frames'].values))
    [os.remove(x) for x in list(set(frames_all).difference(keep))]
    fns = df.loc[~pd.isnull(df['frames-forward']), 'frames-forward'].values
    fns = [os.path.join(os.path.split(os.path.split(x)[0])[1], os.path.split(x)[1]) for x in fns]
    df.loc[~pd.isnull(df['frames-forward']), 'frames-forward'] = fns
    if rear is not None:
        fns = df.loc[~pd.isnull(df['frames-rear']), 'frames-rear'].values
        fns = [os.path.join(os.path.split(os.path.split(x)[0])[1], os.path.split(x)[1]) for x in fns]
        df.loc[~pd.isnull(df['frames-rear']), 'frames-rear'] = fns
    df.to_pickle(os.path.join(fdir, 'nearby.p'))

    # validate dce file for vehicle in collision
    fn = os.path.join(datadir, 'collision-videos', f'collision-{cx:04d}.dce')
    assert os.path.isfile(fn)
    dce = dceinfo4.DCEFile(fn)
    forms = [x.tag for x in dce.getForms()]

    # gps data for vehicle in collision
    assert 'GPSR' in forms
    gps = dce.getForms('GPSR')
    assert len(gps) == 1
    gps = dce_form_gps_to_df(gps[0])
    assert gps.iloc[0]['utc'] < collision['RecordDate'] < gps.iloc[-1]['utc']

    # forward video frames
    assert ('VFWE' in forms) and ('FNDX' in forms)
    vfwe, fndx = dce.getForms('VFWE'), dce.getForms('FNDX')
    assert (len(vfwe) == 1) and (len(fndx) == 1)
    forward = dce_form_video_to_frames(vfwe[0], fndx[0], d0=fdir, d1='collision_forward')

    # process rear video frames if possible
    rear = None
    if ('VRWE' in forms) and ('RNDX' in forms):
        vrwe, rndx = dce.getForms('VRWE'), dce.getForms('RNDX')
        assert (len(vrwe) == 1) and (len(rndx) == 1)
        rear = dce_form_video_to_frames(vrwe[0], rndx[0], d0=fdir, d1='collision_rear')

    # merge gps and frames data for vehicle in collision
    df = merge_gps_frames(forward=forward, rear=rear, gps=gps)

    # filter by time-window, remove unused frames, save data for nearby vehicle
    df = df.loc[(df['utc'] > ta) & (df['utc'] < tb)].reset_index(drop=True)
    keep = np.hstack((df['frames-forward'].values, df['frames-rear'].values))
    keep = keep[~pd.isnull(keep)]
    frames_all = forward['frames'].values
    if rear is not None:
        frames_all = np.hstack((frames_all, rear['frames'].values))
    [os.remove(x) for x in list(set(frames_all).difference(keep))]
    fns = df.loc[~pd.isnull(df['frames-forward']), 'frames-forward'].values
    fns = [os.path.join(os.path.split(os.path.split(x)[0])[1], os.path.split(x)[1]) for x in fns]
    df.loc[~pd.isnull(df['frames-forward']), 'frames-forward'] = fns
    if rear is not None:
        fns = df.loc[~pd.isnull(df['frames-rear']), 'frames-rear'].values
        fns = [os.path.join(os.path.split(os.path.split(x)[0])[1], os.path.split(x)[1]) for x in fns]
        df.loc[~pd.isnull(df['frames-rear']), 'frames-rear'] = fns
    df.to_pickle(os.path.join(fdir, 'collision.p'))

    # save collision video mkv
    fn = os.path.join(datadir, 'collision-videos', f'collision-{cx:04d}.mkv')
    assert os.path.isfile(fn)
    copy(src=fn, dst=os.path.join(fdir, os.path.split(fn)[1]))

    # metadata for vehicle in collision and nearby vehicle
    dmv = pd.Series({
        'time-window': f"""{ta.strftime(tf)} to {tb.strftime(tf)}""",
        'vehicle in collision': collision['SerialNumber'],
        'collision record-date': collision['RecordDate'].strftime(tf),
        'speed at collision': collision['SpeedAtTrigger'],
        'company for vehicle in collision': collision['CompanyName'],
        'industry for vehicle in collision': collision['IndustryDesc'],
        'nearby vehicle': nearby['SerialNumber'],
        'company for nearby vehicle': dm.loc[dm['SerialNumber'] == nearby['SerialNumber'], 'CompanyName'].iloc[0],
        'industry for nearby vehicle':dm.loc[dm['SerialNumber'] == nearby['SerialNumber'], 'IndustryDesc'].iloc[0],
        'seconds from collision at closest distance for nearby vehicle':
            f"""{min([np.abs((pd.Timestamp(datetime.fromtimestamp(x)) - collision['RecordDate']).total_seconds()) for x in d2.loc[d2['distance'] == d2['distance'].min(), 'ts_sec']]):.1f} sec""",
        'closest distance to collision for nearby vehicle': f"""{d2['distance'].min():.2f} meters"""})
    dmv.to_pickle(os.path.join(fdir, 'metadata.p'))

    # update nearby vehicle data status console
    c0, c1 = console_str_objects(w2)
    cs = f"""
        saved frames and gps data<br>
        {ta.strftime(tf)} to {tb.strftime(tf)}"""
    nearby_vehicle_data_status.text = c0 + cs + c1

def load_frames_gps_data():
    global dcv
    global dnv
    global loc

    # reset interface
    mapv.reset_interface()
    mapv.reset_map_view()
    nearby_forward.reset_interface()
    nearby_rear.reset_interface()
    collision_forward.reset_interface()
    collision_rear.reset_interface()
    collision_video2.text = ''

    # clear frames in static dir
    for xs in ['collision_forward', 'collision_rear', 'nearby_forward', 'nearby_rear']:
        if os.path.isdir(os.path.join(os.getcwd(), 'app', 'static', xs)):
            rmtree(os.path.join(os.getcwd(), 'app', 'static', xs))

    # location for frames and gps data
    if frames_gps_select.value == 'default':
        loc = fdir
    else:
        loc = os.path.join(os.path.split(datadir)[0], 'app-archive', frames_gps_select.value)
    assert (loc is not None) and (os.path.isdir(loc))

    # null case
    if len(glob(os.path.join(loc, '*'))) == 0:
        c0, c1 = console_str_objects(w3)
        frames_gps_data_status.text = c0 + f'no saved data' + c1
        return

    # validate and load frames and gps data
    dcv = pd.read_pickle(os.path.join(loc, 'collision.p'))
    dnv = pd.read_pickle(os.path.join(loc, 'nearby.p'))
    dmv = pd.read_pickle(os.path.join(loc, 'metadata.p'))
    c0, c1 = console_str_objects(w3)
    cs = ''
    for key, value in dmv.items():
        cs += f'-- {key} --<br>'
        cs += f'&nbsp;&nbsp;&nbsp;{value}<br>'
    frames_gps_data_status.text = c0 + cs + c1

    # update slider
    tc = pd.Timestamp(datetime.strptime(dmv['collision record-date'], tf))
    tmin = np.array([dcv['utc'].min(), dnv['utc'].min()]).min()
    tmax = np.array([dcv['utc'].max(), dnv['utc'].max()]).max()
    assert (tmin < tc) and (tmax > tc)
    start = (tmin - tc).total_seconds()
    slider.start = 0.1 * np.floor(start * 10)
    end = (tmax - tc).total_seconds()
    slider.end = 0.1 * np.ceil(10 * end)
    slider.remove_on_change('value', update_frames_gps)
    slider.value = 0
    slider.on_change('value', update_frames_gps)

    # update dcv and dnv with seconds to collision data
    dcv['stc'] = [x.total_seconds() for x in dcv['utc'] - tc]
    dnv['stc'] = [x.total_seconds() for x in dnv['utc'] - tc]

    # map gps path data source for vehicle in collision
    lon, lat = dcv['longitude'].values, dcv['latitude'].values
    lon, lat = lon[~np.isnan(lon)], lat[~np.isnan(lat)]
    lon, lat = transform(xx=lon, yy=lat)
    mapv.path1.data = {'lon': lon, 'lat': lat}
    mapv.legend.items.append(LegendItem(label='gps data, vehicle in collision', renderers=[x for x in mapv.fig.renderers if x.name == 'path1']))
    lon0, lon1 = lon.min(), lon.max()
    lat0, lat1 = lat.min(), lat.max()

    # map gps path data source for nearby collision
    lon, lat = dnv['longitude'].values, dnv['latitude'].values
    lon, lat = lon[~np.isnan(lon)], lat[~np.isnan(lat)]
    lon, lat = transform(xx=lon, yy=lat)
    mapv.path2.data = {'lon': lon, 'lat': lat}
    mapv.legend.items.append(LegendItem(label='gps data, nearby vehicle', renderers=[x for x in mapv.fig.renderers if x.name == 'path2']))
    lon0, lon1 = min(lon.min(), lon0), max(lon.max(), lon1)
    lat0, lat1 = min(lat.min(), lat0), max(lat.max(), lat1)
    mapv.reset_map_view(lon0=lon0, lon1=lon1, lat0=lat0, lat1=lat1, convert=False)

    # copy frames to static dir
    for xx in ['frames-forward', 'frames-rear']:

        # vehicle in collision
        if any(~pd.isnull(dcv[xx])):
            src = os.path.join(loc, os.path.split(dcv.loc[~pd.isnull(dcv[xx]), xx].iloc[0])[0])
            dst = os.path.join(os.getcwd(), 'app', 'static', os.path.split(dcv.loc[~pd.isnull(dcv[xx]), xx].iloc[0])[0])
            assert not os.path.isdir(dst)
            copytree(src, dst)
            doc.session_context.remove.append(dst)

        # nearby vehicle
        if any(~pd.isnull(dnv[xx])):
            src = os.path.join(loc, os.path.split(dnv.loc[~pd.isnull(dnv[xx]), xx].iloc[0])[0])
            dst = os.path.join(os.getcwd(), 'app', 'static', os.path.split(dnv.loc[~pd.isnull(dnv[xx]), xx].iloc[0])[0])
            assert not os.path.isdir(dst)
            copytree(src, dst)
            doc.session_context.remove.append(dst)

    # copy collision video to static dir, update html video tag
    fn = glob(os.path.join(loc, 'collision-*.mkv'))
    assert len(fn) == 1
    dst = os.path.join(os.getcwd(), 'app', 'static', os.path.split(fn[0])[1])
    if os.path.isfile(dst):
        os.remove(dst)
    copy(src=fn[0], dst=dst)
    doc.session_context.remove.append(dst)
    path = os.path.join('http://10.140.72.174:5009/app', 'static', os.path.split(dst)[1])
    collision_video2.text = f"""
        <video style="background-color:#222222;width:{w1}px;height:220px;color:white;border:0" controls>
        <source src="{path}" type="video/mp4"></video>"""

    # initialize frames and position glyphs
    update_frames_gps(None, None, None)

def update_frames_gps(attr, old, new):

    # position glyph for vehicle in collision
    xx = np.abs(dcv.loc[~pd.isnull(dcv['latitude']), 'stc'] - slider.value).idxmin()
    cv = dcv.loc[xx]
    lon, lat = transform(xx=cv['longitude'], yy=cv['latitude'])
    mapv.position1.data = {'lon': np.array([lon]), 'lat': np.array([lat])}

    # position glyph for nearby vehicle
    xx = np.abs(dnv.loc[~pd.isnull(dnv['latitude']), 'stc'] - slider.value).idxmin()
    nv = dnv.loc[xx]
    lon, lat = transform(xx=nv['longitude'], yy=nv['latitude'])
    mapv.position2.data = {'lon': np.array([lon]), 'lat': np.array([lat])}

    # forward frame for vehicle in collision
    xx = np.abs(dcv.loc[~pd.isnull(dcv['frames-forward']), 'stc'] - slider.value).idxmin()
    cv = dcv.loc[xx]
    collision_forward.frame.data = {'frame': np.array([os.path.join('http://10.140.72.174:5009/app', 'static', cv['frames-forward'])])}

    # forward frame for nearby vehicle
    xx = np.abs(dnv.loc[~pd.isnull(dnv['frames-forward']), 'stc'] - slider.value).idxmin()
    nv = dnv.loc[xx]
    nearby_forward.frame.data = {'frame': np.array([os.path.join('http://10.140.72.174:5009/app', 'static', nv['frames-forward'])])}

    # rear frame for vehicle in collision
    if (~pd.isnull(dcv['frames-rear'])).any():
        xx = np.abs(dcv.loc[~pd.isnull(dcv['frames-rear']), 'stc'] - slider.value).idxmin()
        cv = dcv.loc[xx]
        collision_rear.frame.data = {'frame': np.array([os.path.join('http://10.140.72.174:5009/app', 'static', cv['frames-rear'])])}

    # rear frame for nearby vehicle
    if (~pd.isnull(dnv['frames-rear'])).any():
        xx = np.abs(dnv.loc[~pd.isnull(dnv['frames-rear']), 'stc'] - slider.value).idxmin()
        nv = dnv.loc[xx]
        nearby_rear.frame.data = {'frame': np.array([os.path.join('http://10.140.72.174:5009/app', 'static', nv['frames-rear'])])}

# metadata console
c0, c1 = console_str_objects(w0)
metadata_title = Div(text='<strong>Population Metadata</strong>', width=w0)
cs = f"""
    {dm.shape[0]} vehicles<br>
    {dm.loc[0, 't0'].strftime(tf)} to {dm.loc[0, 't1'].strftime(tf)}, {(dm.loc[0, 't1'] - dm.loc[0, 't0']).days} days<br>
    {dx.shape[0]} collisions"""
metadata = Div(text=c0 + cs + c1)

# nearby vehicle definitions
xn = glob(os.path.join(datadir, 'nearby_vehicles*.p'))
assert len(xn) > 0
xn = np.array([os.path.split(x)[1].split('_')[2:] for x in xn])
assert xn.shape[1] == 2
td = np.array([int(x[2:]) for x in xn[:, 0]])
xd = np.array([int(x[2:-2]) for x in xn[:, 1]])
ok = np.argsort(td)[::-1]
td, xd = td[ok], xd[ok]
nvs = [f'{tx} seconds, {xx} meters' for tx, xx in zip(td, xd)]
nv_select = Select(title='nearby vehicle definition', value=nvs[0], options=nvs, width=w0)
nv_select.on_change('value', nearby_vehicle_definition_select)

# collisions with nearby vehicles
cx_select = Select(width=w0)
cx_select.on_change('value', collision_select)
cx_metadata_title = Div(text='<strong>Collision Metadata</strong>', width=w0)
cx_metadata = Div()

# nearby vehicles
nx_select = Select(width=w0)
nx_select.on_change('value', nearby_vehicle_select)
nx_metadata_title = Div(text='<strong>Nearby Vehicle Metadata</strong>', width=w0)
nx_metadata = Div()

# map and collision video objects
mapx = MapInterface2(width=w1, height=h1, size=14)
mapx_radio_title = Div(text='<strong>Map Provider</strong>', width=w0)
mapx_radio_group = RadioGroup(labels=['OSM', 'ESRI'], active=0)
mapx_radio_group.on_change('active', mapx_tiles)
collision_video = Div()

# interface to extract data for nearby vehicle
button_extract_data = Button(button_type='primary', label='Extract Data for Nearby Vehicle', width=w2, disabled=False)
button_extract_data.on_click(nearby_vehicle_extract_data_dce)
pre_seconds = TextInput(title='seconds before collision', width=int(w2 / 2))
post_seconds = TextInput(title='seconds after collision', width=int(w2 / 2))
c0, c1 = console_str_objects(w2)
nearby_vehicle_data_status = Div()

# distance vs time interface for vehicle in collision and nearby vehicle
mx1 = MultiLineInterface(width=w2, height=h2, n=1, title='distance from collision, vehicle in collision',
    manual_xlim=True, datetime=True, dimensions='width', box_dimensions='height')
mx1.fig.on_event(MouseMove, mouse_move)
mx1.fig.on_event(MouseLeave, mouse_leave)
mx2 = MultiLineInterface(width=w2, height=h2, n=1, title='distance from collision, nearby vehicle',
    manual_xlim=True, datetime=True, dimensions='width', box_dimensions='height')
mx2.fig.on_event(MouseMove, mouse_move)
mx2.fig.on_event(MouseLeave, mouse_leave)
mx1.fig.x_range = mx2.fig.x_range
span = Span(dimension='height', line_width=2, line_color='black')
mx1.cross, mx2.cross = CrosshairTool(overlay=span), CrosshairTool(overlay=span)
mx1.fig.add_tools(mx1.cross)
mx2.fig.add_tools(mx2.cross)
mx1.fig.toolbar.active_inspect, mx2.fig.toolbar.active_inspect = mx1.cross, mx2.cross

# frames interface to load data
options = ['default'] + [os.path.split(x)[1] for x in glob(os.path.join(os.path.split(datadir)[0], 'app-archive', '*'))]
frames_gps_select = Select(title='location for frames and gps data', value='default', options=options, width=w3)
button_load_data = Button(button_type='primary', label='Load Frames and GPS Data', width=w3)
button_load_data.on_click(load_frames_gps_data)
frames_gps_data_status = Div()

# frames interface
collision_video2 = Div()
slider = Slider(width=w5, title='seconds from collision', start=0, end=1, value=0.1, step=0.2, format='0[.]0')
slider.on_change('value', update_frames_gps)
mapv = MapInterface2(width=w4, height=h4, size=14)
mapv_radio_title = Div(text='<strong>Map Provider</strong>', width=w0)
mapv_radio_group = RadioGroup(labels=['OSM', 'ESRI'], active=0)
mapv_radio_group.on_change('active', mapv_tiles)
nearby_forward = FrameInterface(title='nearby vehicle, forward view frames', width=w5, height=h5)
nearby_rear = FrameInterface(title='nearby vehicle, rear view frames', width=w5, height=h5)
collision_forward = FrameInterface(title='vehicle in collision, forward view frames', width=w5, height=h5)
collision_rear = FrameInterface(title='vehicle in collision, rear view frames', width=w5, height=h5)

# run app
layout = Tabs(tabs_location='above', tabs=[
    TabPanel(title='collision and nearby vehicle', child=row(
        column(metadata_title, metadata, nv_select, cx_select, cx_metadata_title, cx_metadata, nx_select, nx_metadata_title, nx_metadata, mapx_radio_title, mapx_radio_group),
        column(mapx.fig, collision_video),
        column(mx1.fig, mx2.fig, button_extract_data, row(pre_seconds, post_seconds), nearby_vehicle_data_status))),
    TabPanel(title='synchronized frames and gps', child=row(
        column(frames_gps_select, button_load_data, frames_gps_data_status),
        column(collision_video2),
        column(slider, mapv.fig, mapv_radio_title, mapv_radio_group),
        column(slider, collision_forward.fig, collision_rear.fig),
        column(slider, nearby_forward.fig, nearby_rear.fig)))])
doc = curdoc()
doc.session_context.remove = []
doc.add_root(layout)
doc.title = 'lytx vehicle eye-witness'
nearby_vehicle_definition_select(None, None, None)
