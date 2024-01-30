
"""
bokeh app to explore collision prediction model raw data by single vehicle-eval
"""

import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pickle
import boto3
from tempfile import NamedTemporaryFile
from datetime import datetime
from functools import partial
from glob import glob
from shutil import move
from botocore.exceptions import ClientError
from bokeh.io import show, curdoc
from bokeh.models import Tabs, Panel, Select, MultiSelect, CheckboxGroup, Div, Slider, Button, RadioGroup
from bokeh.layouts import column, row
from bokeh.events import MouseMove, MouseLeave
from pyrb.bokeh import MetricDistributionInterface, HorizontalBarChartInterface, MapInterface, MultiLineInterface
from pyrb.bokeh import str_axis_labels, link_axes
from ipdb import set_trace


# reset interfaces
def reset_interfaces(
        reset_gps=False,
        reset_events=False,
        reset_behaviors=False,
        reset_signal=False,
        reset_sources=False,
        reset_segments_multiselect=False,
        reset_events_multiselect=False,
        reset_events_selected_indices=False,
        reset_behaviors_multiselect=False,
        reset_behaviors_selected_indices=False,
        reset_all=False):
    """
    reset all or individual interfaces
    """
    global df   # gps
    global de   # events
    global dex  # selected events
    global db   # behaviors
    global dbx  # selected behaviors

    if reset_gps or reset_all:
        gps.reset_interface()
        gps.reset_map_view()

    if reset_events or reset_all:
        events.reset_interface()

    if reset_behaviors or reset_all:
        behaviors.reset_interface()

    if reset_signal or reset_all:
        signal.reset_interface()

    if reset_sources or reset_all:
        df = pd.DataFrame()     # gps
        de = pd.DataFrame()     # events
        dex = pd.DataFrame()    # selected events
        db = pd.DataFrame()     # behaviors
        dbx = pd.DataFrame()    # selected behaviors

    if reset_segments_multiselect or reset_all:
        segments.options = []
        segments.remove_on_change('value', segments_multiselect_callback)
        segments.value = []
        segments.on_change('value', segments_multiselect_callback)
        segments_all.active = []

    if reset_events_multiselect or reset_all:
        eselect.options = []
        eselect.remove_on_change('value', eselect_callback)
        eselect.value = []
        eselect.on_change('value', eselect_callback)
        eselect_all.active = []

    if reset_events_selected_indices or reset_all:
        events.events.selected.remove_on_change('indices', events_selected_callback)
        events.events.selected.indices = []
        events.events.selected.on_change('indices', events_selected_callback)
        video_events.text = f"""<div style="background-color:#000000;width:600px;height:220px;color:white;border:0"></div>"""

    if reset_behaviors_multiselect or reset_all:
        bselect.options = []
        bselect.remove_on_change('value', bselect_callback)
        bselect.value = []
        bselect.on_change('value', bselect_callback)
        bselect_all.active = []

    if reset_behaviors_selected_indices or reset_all:
        behaviors.events.selected.remove_on_change('indices', behaviors_selected_callback)
        behaviors.events.selected.indices = []
        behaviors.events.selected.on_change('indices', behaviors_selected_callback)
        video_behaviors.text = f"""<div style="background-color:#000000;width:600px;height:220px;color:white;border:0"></div>"""

# vehicle select callback
def vs_callback(attr, old, new, sender):
    """
    update vehicle eval select objects
    """

    # update status, reset all interfaces
    status.text = status_running
    reset_interfaces(reset_all=True)

    # update vehicle select menu
    if sender is None:
        vs['vehicle-select'].options = sorted(pd.unique(dcm['VehicleId']))
        vs['vehicle-select'].remove_on_change('value', vs_callback_vehicle_select)
        vs['vehicle-select'].value = vs['vehicle-select'].options[0]
        vs['vehicle-select'].on_change('value', vs_callback_vehicle_select)

    # identify vehicle eval
    ok = (dcm['VehicleId'] == vs['vehicle-select'].value)
    assert ok.sum() == 1
    vehicle = dcm.loc[ok].squeeze()

    # update interfaces
    update_gps_interface(vehicle)
    update_events_interface(vehicle)
    update_behaviors_interface(vehicle)

    # update status
    status.text = status_done

# event/behavior filter callback
def filter_callback(attr, old, new):
    """
    update dcm based on filter selection and re-initialize app
    """
    global dcm

    # filter dcm by selected events
    active = vs['filter-radio'].labels[vs['filter-radio'].active]
    if active == 'events':

        # any events selected
        if vs['filter'].value == 'any events':
            dcm = dcm0.copy()

        # specific event selected
        else:
            xid = new.split('(')[0].strip().split('-')
            if len(xid) > 1:
                ok = dfa[f'nevents_{xid[0]}_{xid[1]}'].values > 0
            else:
                ok = dfa[f'nevents_{xid[0]}'].values > 0
            dcm = dcm0.loc[ok].reset_index(drop=True)

    # filter dcm by selected behaviors
    elif active == 'behaviors':

        # any behaviors selected
        if vs['filter'].value == 'any behaviors':
            dcm = dcm0.copy()

        # specific behavior selected
        else:
            ok = dfa[f"""nbehaviors_{new.split('(')[0].strip()}"""] > 0
            dcm = dcm0.loc[ok].reset_index(drop=True)

    # re-initialize
    vs_callback(None, None, None, None)

# gps map interface callbacks
def update_gps_interface(vehicle):
    """
    update objects on GPS interface
    """
    global df

    # gps raw data parquet file for vehicle
    fn = glob(os.path.join(datadir, 'gps2.parquet', f"""VehicleId={vehicle['VehicleId']}""", '*.parquet'))
    if len(fn) == 0:
        return

    # read gps raw data
    assert len(fn) == 1
    df = pq.ParquetFile(fn[0]).read().to_pandas()

    # filter by vehicle eval time bounds, record in df
    time0 = int((vehicle['time0'] - datetime(1970, 1, 1)).total_seconds())
    time1 = int((vehicle['time1'] - datetime(1970, 1, 1)).total_seconds())
    df = df.loc[(df['TS_SEC'] > time0) & (df['TS_SEC'] < time1)]
    if df.size == 0:
        return
    df['time0'] = vehicle['time0']
    df['time1'] = vehicle['time1']

    # update gps path data
    sid = df['segmentId'].values
    nok = np.isnan(sid)
    lon = df['longitude'].values
    lat = df['latitude'].values
    lon[nok] = np.nan
    lat[nok] = np.nan
    gps.path.data = {'lon': lon, 'lat': lat}

    # update gps map view
    gps.reset_map_view(lon0=np.nanmin(lon), lon1=np.nanmax(lon), lat0=np.nanmin(lat), lat1=np.nanmax(lat), convert=False)

    # update signal data
    df['dt'] = np.array([datetime.utcfromtimestamp(x) for x in df['TS_SEC'].values])
    miles = df['cumulative_distance_miles'].values
    miles -= np.nanmin(miles)
    signal.data_sources[0].data = {'x': df['dt'].values, 'y': miles}
    signal.fig.x_range.start = 1000 * time0
    signal.fig.x_range.end = 1000 * time1
    signal.fig.y_range.start = 0
    signal.fig.y_range.end = 1.05 * np.nanmax(miles)

    # update GPS segments multiselect
    xs = np.unique(df['segmentId'])
    xs = np.sort(xs[~np.isnan(xs)]).astype('int')
    segments.options = [str(x) for x in xs]

def gps_interface_mouse_move(event):

    # do not update
    if mouse_move_checkbox.active == []:
        return

    # convert event time and handle null cases
    epoch = (1e-3) * event.x
    if (df.size == 0) or (epoch < df.iloc[0]['TS_SEC']) or (epoch > df.iloc[-1]['TS_SEC']):
        gps.position.data = {'lat': np.array([]), 'lon': np.array([])}
        return

    # update marker object
    delta = np.abs(epoch - df['TS_SEC'])
    if delta.min() > 3600:
        gps.position.data = {'lat': np.array([]), 'lon': np.array([])}
        return
    pos = np.argmin(delta)
    lat, lon = df.iloc[pos][['latitude', 'longitude']].values
    gps.position.data = {'lat': np.array([lat]), 'lon': np.array([lon])}

def gps_interface_mouse_leave(event):

    # do not update
    if mouse_move_checkbox.active == []:
        return

    # reset position glyph
    gps.position.data = {'lat': np.array([]), 'lon': np.array([])}

def segments_multiselect_callback(attr, old, new):
    """
    update GPS segments data source based on multiselect value
    """

    # null case
    if not new:
        gps.segments.data = {'lat': np.array([]), 'lon': np.array([])}
        signal.segments.data = {'x': np.array([]), 'y': np.array([])}
        return

    # build lat/lon data from selected segments, update source
    status.text = status_running
    lat = np.array([])
    lon = np.array([])
    time = np.array([]).astype(np.datetime64)
    value = np.array([])
    for sid in [int(x) for x in new]:
        ok = df['segmentId'] == sid
        assert ok.sum() > 0
        lat = np.hstack((lat, df.loc[ok, 'latitude'].values, np.nan))
        lon = np.hstack((lon, df.loc[ok, 'longitude'].values, np.nan))
        time = np.hstack((time, df.loc[ok, 'dt'].values, df.loc[ok, 'dt'].values[-1]))
        value = np.hstack((value, df.loc[ok, 'cumulative_distance_miles'].values, np.nan))
    gps.segments.data = {'lat': lat[:-1], 'lon': lon[:-1]}
    signal.segments.data = {'x': time[:-1], 'y': value[:-1]}
    status.text = status_done

# events interface callbacks
def update_events_interface(vehicle):
    """
    update objects on event interface
    """
    global de

    # sync events path data with gps
    events.path.data = dict(gps.path.data)

    # events raw data parquet file for vehicle
    fn = glob(os.path.join(datadir, 'events.parquet', f"""VehicleId={vehicle['VehicleId']}""", '*.parquet'))
    if len(fn) == 0:
        return

    # read events raw data
    assert len(fn) == 1
    de = pq.ParquetFile(fn[0]).read().to_pandas()

    # filter by vehicle eval time bounds
    time0 = int((vehicle['time0'] - datetime(1970, 1, 1)).total_seconds())
    time1 = int((vehicle['time1'] - datetime(1970, 1, 1)).total_seconds())
    de = de.loc[(de['TS_SEC'] > time0) & (de['TS_SEC'] < time1)]
    if de.size == 0:
        return

    # create tag column for multiselect
    de['tag'] = [
        f"""{name}-{sub} (accel-{dcs.loc[dcs['Id'] == sub, 'Name'].iloc[0]})""" if name == 30 else
        f"""{name} ({dce.loc[dce['Id'] == name, 'Name'].iloc[0]})""" for name, sub in zip(de['NameId'], de['SubId'])]

    # update events multiselect
    eselect.options = sorted(pd.unique(de['tag']))

def eselect_callback(attr, old, new):
    """
    update events data source based on multiselect value
    """
    global dex

    # clear events selected indices
    reset_interfaces(reset_events_selected_indices=True)

    # null case
    if not new:
        events.events.data = {'lon': np.array([]), 'lat': np.array([])}
        dex = pd.DataFrame()
        return

    # update events source with data for glyphs and hover tool
    status.text = status_running
    dex = de.loc[de['tag'].isin(new)].reset_index(drop=True)
    events.events.data = {
        'lat': dex['latitude'].values,
        'lon': dex['longitude'].values,
        'latitude': dex['latitude_gps'].values,
        'longitude': dex['longitude_gps'].values,
        'speed at trigger': dex['SpeedAtTrigger'].values,
        'event tag': dex['tag'].values,
        'eventdatetime': [pd.Timestamp(x).to_pydatetime() for x in dex['eventdatetime'].values]}
    status.text = status_done

def events_selected_callback(attr, old, new):
    """
    callback on selected events
    """

    # null cases
    if new == []:
        return
    if len(new) > 1:
        reset_interfaces(reset_events_selected_indices=True)
        return

    # extract event video as NamedTemporaryFile
    status.text = status_running
    event = dex.loc[new[0]]
    uri = [x for x in event['s3_uri'].split(os.sep) if x]
    assert uri[0] == 's3:'
    obj = s3.Object(bucket_name=uri[1], key=os.sep.join((os.sep.join(uri[2:-3]), os.sep.join(uri[-2:]))))
    try:
        obj.load()
        with NamedTemporaryFile(suffix='.DCE') as fx:
            obj.download_file(fx.name)
            # merged audio/video via dce2mkv in ffmpeg3 environment
            cmd = f'conda run -n ffmpeg3 --cwd {os.path.split(fx.name)[0]} '
            cmd += 'python /mnt/home/russell.burdt/miniconda3/envs/ffmpeg3/lib/python3.10/site-packages/dceutils/dce2mkv.py '
            cmd += fx.name
            os.system(cmd)
            assert os.path.isfile(fx.name[:-4] + '_merged.mkv')
            assert os.path.isfile(fx.name[:-4] + '_discrete.mkv')
            os.remove(fx.name[:-4] + '_discrete.mkv')
    except ClientError as err:
        video_events.text = """<div style="background-color:#000000;width:600px;height:220px;color:white;border:0">"""
        video_events.text += f"""video not found<br>{'<br>'.join(uri)}"""
        video_events.text += """</div>"""
        status.text = status_done
        return

    # move video to app static dir
    assert os.path.isdir(os.path.join(os.getcwd(), 'app-data'))
    sdir = os.path.join(os.getcwd(), 'app-data', 'static')
    if not os.path.isdir(sdir):
        os.mkdir(sdir)
    src = fx.name[:-4] + '_merged.mkv'
    dst = os.path.join(sdir, os.path.split(fx.name)[1][:-4] + '.mp4')
    move(src=src, dst=dst)
    doc.session_context.vids.append(dst)

    # html video tag
    path = os.path.join('http://10.144.240.35:5011/app-data', 'static', os.path.split(dst)[1])
    video_events.text = f"""
        <video style="background-color:#222222;width:600px;height:220px;color:white;border:0" controls autoplay>
        <source src="{path}" type="video/mp4"></video>"""
    status.text = status_done

# behaviors interface callbacks
def update_behaviors_interface(vehicle):
    """
    update objects on behavior interface
    """
    global db

    # sync behaviors path data with gps
    behaviors.path.data = dict(gps.path.data)

    # behaviors raw data parquet file for vehicle
    fn = glob(os.path.join(datadir, 'behaviors.parquet', f"""VehicleId={vehicle['VehicleId']}""", '*.parquet'))
    if len(fn) == 0:
        return

    # read behaviors raw data
    assert len(fn) == 1
    db = pq.ParquetFile(fn[0]).read().to_pandas()

    # filter by vehicle eval time bounds
    time0 = int((vehicle['time0'] - datetime(1970, 1, 1)).total_seconds())
    time1 = int((vehicle['time1'] - datetime(1970, 1, 1)).total_seconds())
    db = db.loc[(db['TS_SEC'] > time0) & (db['TS_SEC'] < time1)]
    if db.size == 0:
        return

    # create tag column for multiselect
    db['tag'] = [f"""{a} ({b})""" for a, b in zip(db['NameId'], db['Name'])]

    # update behaviors multiselect
    bselect.options = sorted(pd.unique(db['tag']))

def bselect_callback(attr, old, new):
    """
    update behaviors data source based on multiselect value
    """
    global dbx

    # clear events selected indices
    reset_interfaces(reset_behaviors_selected_indices=True)

    # null case
    if not new:
        behaviors.events.data = {'lon': np.array([]), 'lat': np.array([])}
        dbx = pd.DataFrame()
        return

    # update events source with data for glyphs and hover tool
    status.text = status_running
    dbx = db.loc[db['tag'].isin(new)].reset_index(drop=True)
    behaviors.events.data = {
        'lat': dbx['latitude'].values,
        'lon': dbx['longitude'].values,
        'latitude': dbx['latitude_gps'].values,
        'longitude': dbx['longitude_gps'].values,
        'behavior tag': dbx['tag'].values}
    status.text = status_done

def behaviors_selected_callback(attr, old, new):
    """
    callback on selected behaviors
    """

    # null cases
    if new == []:
        return
    if len(new) > 1:
        reset_interfaces(reset_behaviors_selected_indices=True)
        return

    # extract event video as NamedTemporaryFile
    status.text = status_running
    event = dbx.loc[new[0]]
    uri = [x for x in event['s3_uri'].split(os.sep) if x]
    assert uri[0] == 's3:'
    obj = s3.Object(bucket_name=uri[1], key=os.sep.join((os.sep.join(uri[2:-3]), os.sep.join(uri[-2:]))))
    try:
        obj.load()
        with NamedTemporaryFile(suffix='.DCE') as fx:
            obj.download_file(fx.name)
            # merged audio/video via dce2mkv in ffmpeg3 environment
            cmd = f'conda run -n ffmpeg3 --cwd {os.path.split(fx.name)[0]} '
            cmd += 'python /mnt/home/russell.burdt/miniconda3/envs/ffmpeg3/lib/python3.10/site-packages/dceutils/dce2mkv.py '
            cmd += fx.name
            os.system(cmd)
            assert os.path.isfile(fx.name[:-4] + '_merged.mkv')
            assert os.path.isfile(fx.name[:-4] + '_discrete.mkv')
            os.remove(fx.name[:-4] + '_discrete.mkv')
    except ClientError as err:
        for video in [video_events, video_behaviors]:
            video.text = """<div style="background-color:#000000;width:600px;height:220px;color:white;border:0">"""
            video.text += f"""video not found<br>{'<br>'.join(uri)}"""
            video.text += """</div>"""
        status.text = status_done
        return

    # move video to app static dir
    assert os.path.isdir(os.path.join(os.getcwd(), 'app-data'))
    sdir = os.path.join(os.getcwd(), 'app-data', 'static')
    if not os.path.isdir(sdir):
        os.mkdir(sdir)
    src = fx.name[:-4] + '_merged.mkv'
    dst = os.path.join(sdir, os.path.split(fx.name)[1][:-4] + '.mp4')
    move(src=src, dst=dst)
    doc.session_context.vids.append(dst)

    # html video tag
    path = os.path.join('http://10.144.240.35:5011/app-data', 'static', os.path.split(dst)[1])
    video_behaviors.text = f"""
        <video style="background-color:#222222;width:600px;height:220px;color:white;border:0" controls autoplay>
        <source src="{path}" type="video/mp4"></video>"""
    status.text = status_done

# single callback for all checkbox/radio objects
def checkbox(event, sender):
    """
    callback for all checkbox/radio objects
    """

    if sender == 'segments':
        if not segments.options:
            return
        segments.value = segments.options if event else []

    elif sender == 'events':
        if not eselect.options:
            return
        eselect.value = eselect.options if event else []

    elif sender == 'behaviors':
        if not bselect.options:
            return
        bselect.value = bselect.options if event else []

    elif sender == 'filter-radio':
        active = vs['filter-radio'].labels[vs['filter-radio'].active]
        if active == 'events':
            vs['filter'].options = ['any events'] + xe
            vs['filter'].value = 'any events'
        elif active == 'behaviors':
            vs['filter'].options = ['any behaviors'] + xb
            vs['filter'].value = 'any behaviors'

# collision prediction model datadir and validate
datadir = r'/mnt/home/russell.burdt/data/driver-safety/kenan20'
assert os.path.isdir(datadir)
assert all([os.path.isdir(os.path.join(datadir, x + '.parquet')) for x in ['gps2', 'events', 'behaviors']])
assert os.path.isfile(os.path.join(datadir, 'decoder.p'))
assert os.path.isfile(os.path.join(datadir, 'dv.p'))

# read data
dm = pd.read_pickle(os.path.join(datadir, 'dm.p'))
dcm = pd.read_pickle(os.path.join(datadir, 'dcm-gps.p'))
dfa = pd.read_pickle(os.path.join(datadir, 'dv.p'))
assert dcm.shape[0] == dfa.shape[0]
dcm0 = dcm.copy()
decoder = pd.read_pickle(os.path.join(datadir, 'decoder.p'))
dce = decoder['event-type']
dcs = decoder['event-sub-type']
dcb = decoder['behaviors']
s3 = boto3.resource(service_name='s3')

# unique event and behavior tags for all vehicle evals
xe = [x.split('_')[1:] for x in dfa.columns if 'nevents_' in x]
xe = [[x[0], None] if len(x) == 1 else x for x in xe]
xe = np.array([x for x in xe if x[1] != 'others'])
xe = np.unique(sorted([
    f"""{name}-{sub} (accel-{dcs.loc[dcs['Id'] == int(sub), 'Name'].iloc[0]})""" if name == '30' else
    f"""{name} ({dce.loc[dce['Id'] == int(name), 'Name'].iloc[0]})""" for name, sub in zip(xe[:, 0], xe[:, 1])])).tolist()
xb = np.array([x.split('_')[1] for x in dfa.columns if 'nbehaviors_' in x]).astype('int')
xb = np.unique(sorted([f"""{x} ({dcb.loc[dcb['Id'] == x, 'Name'].iloc[0]})""" for x in xb])).tolist()

# vehicle-eval select objects
vs = {}
title = f"""<strong>
    {dm['company']}<br>
    {dm['time0'].strftime('%d %b %Y')} to {dm['time1'].strftime('%d %b %Y')}<br>
    Select VehicleId</strong>"""
vs['main-title'] = Div(text=title, width=300)
vs['vehicle-select'] = Select(title='vehicle-id', width=300)
vs_callback_vehicle_select = partial(vs_callback, sender=vs['vehicle-select'])
vs['vehicle-select'].on_change('value', vs_callback_vehicle_select)
vs['filter-title'] = Div(text='Filter Vehicle Evals by', width=300)
vs['filter-radio'] = RadioGroup(labels=['events', 'behaviors'], active=0)
vs['filter-radio'].on_click(partial(checkbox, sender='filter-radio'))
vs['filter'] = Select(width=300, options=['any events'] + xe, value='any events')
vs['filter'].on_change('value', filter_callback)

# status object and messages
status = Div(width=300)
status_running = """<strong style="color:red">Status -- Running</strong>"""
status_done = """<strong style="color:blue">Status -- Done</strong>"""

# gps map interface objects
gps = MapInterface(width=600, height=300)
signal = MultiLineInterface(width=600, height=200, ylabel='miles', title='cumulative miles vs time',
    n=1, cross=True, datetime=True, manual_xlim=True, manual_ylim=True)
signal.fig.on_event(MouseMove, gps_interface_mouse_move)
signal.fig.on_event(MouseLeave, gps_interface_mouse_leave)
mouse_move_checkbox = CheckboxGroup(labels=['show position on map'], active=[])
segments = MultiSelect(title='GPS Segment(s)', width=160, height=120)
segments.on_change('value', segments_multiselect_callback)
segments_all = CheckboxGroup(labels=['All Segments'], active=[])
segments_all.on_click(partial(checkbox, sender='segments'))

# events interface objects
events = MapInterface(width=600, height=280, tap=True, hover=True)
link_axes(figs=[gps.fig, events.fig], axis='xy')
events.hover.renderers = [x for x in events.fig.renderers if x.name == 'events']
events.hover.tooltips = [
    ('latitude', '@latitude{%.4f}'),
    ('longitude', '@longitude{%.4f}'),
    ('speed at trigger', '@{speed at trigger}{%.1f}'),
    ('event utc', '@eventdatetime{%d %b %Y %H:%M}'),
    ('tag', '@{event tag}')]
events.hover.formatters = {'@latitude': 'printf', '@longitude': 'printf', '@{speed at trigger}': 'printf', '@eventdatetime': 'datetime'}
events.tap.renderers = [x for x in events.fig.renderers if x.name == 'events']
events.events.selected.on_change('indices', events_selected_callback)
eselect = MultiSelect(title='Event(s)', width=200, height=200)
eselect.on_change('value', eselect_callback)
eselect_all = CheckboxGroup(labels=['All Events'], active=[])
eselect_all.on_click(partial(checkbox, sender='events'))
video_events = Div()

# behaviors interface objects
behaviors = MapInterface(width=600, height=280, tap=True, hover=True)
link_axes(figs=[gps.fig, behaviors.fig], axis='xy')
behaviors.hover.renderers = [x for x in behaviors.fig.renderers if x.name == 'events']
behaviors.hover.tooltips = [
    ('latitude', '@latitude{%.4f}'),
    ('longitude', '@longitude{%.4f}'),
    ('tag', '@{behavior tag}')]
behaviors.hover.formatters = {'@latitude': 'printf', '@longitude': 'printf'}
behaviors.tap.renderers = [x for x in behaviors.fig.renderers if x.name == 'events']
behaviors.events.selected.on_change('indices', behaviors_selected_callback)
bselect = MultiSelect(title='Behavior(s)', width=200, height=200)
bselect.on_change('value', bselect_callback)
bselect_all = CheckboxGroup(labels=['All Behaviors'], active=[])
bselect_all.on_click(partial(checkbox, sender='behaviors'))
video_behaviors = Div()

# app layout based on panel objects
eval_select = column(
    vs['main-title'], vs['vehicle-select'], vs['filter-title'], vs['filter-radio'], vs['filter'], status)
layout_gps = row(
    eval_select,
    column(gps.fig, mouse_move_checkbox, signal.fig),
    column(segments, segments_all))
layout_events = row(
    eval_select,
    column(events.fig, video_events),
    column(eselect, eselect_all))
layout_behaviors = row(
    eval_select,
    column(behaviors.fig, video_behaviors),
    column(bselect, bselect_all))
layout = Tabs(tabs=[
    Panel(child=layout_gps, title='GPS Map Interface'),
    Panel(child=layout_events, title='Events Interface'),
    Panel(child=layout_behaviors, title='Behaviors Interface')])

# create document object and list for created video files
doc = curdoc()
doc.session_context.vids = []
doc.add_root(layout)
doc.title = 'collision model data app'

# initialize state
vs_callback(None, None, None, None)
