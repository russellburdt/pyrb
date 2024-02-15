
"""
bokeh app to explore collision prediction model gps-enriched data by single vehicle-eval
"""

import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pickle
from datetime import datetime
from functools import partial
from glob import glob
from pyspark import SparkConf
from pyspark.sql import SparkSession
from bokeh.io import show, curdoc
from bokeh.models import Select, MultiSelect, CheckboxGroup, Div, Slider, Button, RadioGroup
from bokeh.layouts import column, row
from bokeh.events import MouseMove, MouseLeave
from pyrb.bokeh import MetricDistributionInterface, HorizontalBarChartInterface, MapInterface, MultiLineInterface
from pyrb.processing import webm_to_gps, gps_to_webm
from geopandas.array import from_wkb
from ipdb import set_trace


# reset interfaces
def reset_interfaces(
        reset_gps=False,
        reset_gps_signal=False,
        reset_gpse=False,
        reset_gpse_signal=False,
        reset_sources=False,
        reset_segments_multiselect=False,
        reset_maneuvers_multiselect=False,
        reset_all=False):
    """
    reset all or individual interfaces
    """
    global df
    global dfe
    global dfm

    if reset_gps or reset_all:
        gps.reset_interface()
        gps.reset_map_view()

    if reset_gps_signal or reset_all:
        gps_signal.reset_interface()

    if reset_gpse or reset_all:
        gpse.reset_interface()
        gpse.reset_map_view()

    if reset_gpse_signal or reset_all:
        gpse_signal.reset_interface()

    if reset_sources or reset_all:
        df = pd.DataFrame()
        dfe = pd.DataFrame()
        dfm = pd.DataFrame()

    if reset_segments_multiselect or reset_all:
        segments.options = []
        segments.remove_on_change('value', segments_multiselect_callback)
        segments.value = []
        segments.on_change('value', segments_multiselect_callback)

    if reset_maneuvers_multiselect or reset_all:
        maneuvers.options = []
        maneuvers.remove_on_change('value', maneuvers_multiselect_callback)
        maneuvers.value = []
        maneuvers.on_change('value', maneuvers_multiselect_callback)

# vehicle select callback
def vs_callback(attr, old, new, sender):
    """
    update vehicle eval select objects
    """

    # update status, reset all interfaces
    status.text = status_running
    reset_interfaces(reset_all=True)

    # identify sender
    c0 = sender is None
    c1 = sender == vs['industry-select']
    c2 = sender == vs['company-select']
    c3 = sender == vs['vehicle-select']

    # update industry select menu
    if c0:
        vs['industry-select'].options = sorted(pd.unique(dcm['IndustryDesc']))
        vs['industry-select'].remove_on_change('value', vs_callback_industry_select)
        vs['industry-select'].value = vs['industry-select'].options[0]
        vs['industry-select'].on_change('value', vs_callback_industry_select)

    # update company select menu
    if c0 or c1:
        ok = dcm['IndustryDesc'] == vs['industry-select'].value
        assert ok.sum() > 0
        vs['company-select'].options = sorted(pd.unique(dcm.loc[ok, 'CompanyName']))
        vs['company-select'].remove_on_change('value', vs_callback_company_select)
        vs['company-select'].value = vs['company-select'].options[0]
        vs['company-select'].on_change('value', vs_callback_company_select)

    # update vehicle select menu
    if c0 or c1 or c2:
        ok = (dcm['IndustryDesc'] == vs['industry-select'].value) & (dcm['CompanyName'] == vs['company-select'].value)
        assert ok.sum() > 0
        vs['vehicle-select'].options = sorted(pd.unique(dcm.loc[ok, 'VehicleId']))
        vs['vehicle-select'].remove_on_change('value', vs_callback_vehicle_select)
        vs['vehicle-select'].value = vs['vehicle-select'].options[0]
        vs['vehicle-select'].on_change('value', vs_callback_vehicle_select)

    # update predictor interval select menu
    if c0 or c1 or c2 or c3:
        ok = (dcm['IndustryDesc'] == vs['industry-select'].value) & (dcm['CompanyName'] == vs['company-select'].value) & (dcm['VehicleId'] == vs['vehicle-select'].value)
        assert ok.sum() > 0
        vs['window-select'].options = sorted(list(dcm.loc[ok, 'window']))
        vs['window-select'].remove_on_change('value', vs_callback_window_select)
        vs['window-select'].value = vs['window-select'].options[0]
        vs['window-select'].on_change('value', vs_callback_window_select)

    # identify vehicle eval
    ok = (dcm['VehicleId'] == vs['vehicle-select'].value) & (dcm['window'] == vs['window-select'].value)
    assert ok.sum() == 1
    vehicle = dcm.loc[ok].squeeze()

    # update interfaces
    update_gps_interface(vehicle)
    update_gpse_interface(vehicle)

    # update status
    status.text = status_done

# gps map interface
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
    df = pq.ParquetFile(fn[0]).read().to_pandas().sort_values('TS_SEC').reset_index(drop=True)

    # filter by vehicle eval time bounds
    time0 = int((vehicle['time0'] - datetime(1970, 1, 1)).total_seconds())
    time1 = int((vehicle['time1'] - datetime(1970, 1, 1)).total_seconds())
    df = df.loc[(df['TS_SEC'] > time0) & (df['TS_SEC'] < time1)]
    if df.size == 0:
        return

    # update gps path data
    sid = df['segmentId'].values
    nok = np.isnan(sid)
    lon = df['longitude'].values
    lat = df['latitude'].values
    lon[nok] = np.nan
    lat[nok] = np.nan
    gps.path.data = {'lon': lon, 'lat': lat,
        'utc': df['utc'], 'longitude': df['longitude_gps'], 'latitude': df['latitude_gps']}

    # update gps map view
    gps.reset_map_view(lon0=np.nanmin(lon), lon1=np.nanmax(lon), lat0=np.nanmin(lat), lat1=np.nanmax(lat), convert=False)

    # update gps signal data
    miles = df['cumulative_distance_miles'].values
    miles -= miles.min()
    gps_signal.data_sources[0].data = {'x': df['utc'].values, 'y': miles}
    gps_signal.fig.x_range.start = 1000 * time0
    gps_signal.fig.x_range.end = 1000 * time1
    gps_signal.fig.y_range.start = 0
    gps_signal.fig.y_range.end = 1.05 * np.nanmax(miles)

    # update GPS segments multiselect
    xs = np.unique(df['segmentId'])
    xs = np.sort(xs[~np.isnan(xs)]).astype('int')
    segments.options = [str(x) for x in xs]

# gps-enriched map interface
def update_gpse_interface(vehicle):
    """
    update objects on GPS-enriched interface
    """
    global dfe
    global dfm

    # gpse raw data parquet file for vehicle
    fn = glob(os.path.join(datadir, 'gpse.parquet', f"""VehicleId={vehicle['VehicleId']}""", '*.parquet'))
    if len(fn) == 0:
        return
    assert len(fn) == 1
    dfe = pq.ParquetFile(fn[0]).read().to_pandas().sort_values('segment__id').reset_index(drop=True)
    assert all(np.sort(dfe['TS_SEC'].values) == dfe['TS_SEC'])

    # gpsm raw data parquet file for vehicle
    fn = glob(os.path.join(datadir, 'gpsm.parquet', f"""VehicleId={vehicle['VehicleId']}""", '*.parquet'))
    assert len(fn) == 1
    dfm = pq.ParquetFile(fn[0]).read().to_pandas().sort_values('sg_start').reset_index(drop=True)

    # filter by vehicle eval time bounds, record in df
    time0 = int((vehicle['time0'] - datetime(1970, 1, 1)).total_seconds())
    time1 = int((vehicle['time1'] - datetime(1970, 1, 1)).total_seconds())
    dfe = dfe.loc[(dfe['TS_SEC'] > time0) & (dfe['TS_SEC'] < time1)]
    if dfe.size == 0:
        return

    # update gpse path data
    sid = dfe['segmentId'].values
    xs = np.array([np.where(sid == x)[0][-1] for x in np.sort(np.unique(sid))])
    lon = dfe['longitude'].values
    lat = dfe['latitude'].values
    lon[xs] = np.nan
    lat[xs] = np.nan
    gpse.path.data = {
        'lon': lon,
        'lat': lat,
        'longitude': dfe['longitude_gps'],
        'latitude': dfe['latitude_gps'],
        'utc': dfe['utc'],
        'segment__id': dfe['segment__id']}

    # update gpse map view
    gpse.reset_map_view(lon0=np.nanmin(lon), lon1=np.nanmax(lon), lat0=np.nanmin(lat), lat1=np.nanmax(lat), convert=False)

    # update gpse signal data (insert nans at changes in segmentId)
    miles = dfe['cumulative_distance_miles'].values
    miles -= miles.min()
    gpse_signal.data_sources[0].data = {'x': dfe['utc'].values, 'y': miles}
    gpse_signal.fig.x_range.start = 1000 * time0
    gpse_signal.fig.x_range.end = 1000 * time1
    gpse_signal.fig.y_range.start = 0
    gpse_signal.fig.y_range.end = 1.05 * np.nanmax(miles)

    # update GPS maneuvers multiselect
    maneuvers.options = sorted(pd.unique(dfm['segmentgroup__maneuver']))

def mouse_move(event, sender):

    # idenfity sender
    if sender == 'gps':
        move_checkbox = gps_mouse_move_checkbox
        data = df
        gmap = gps
    elif sender == 'gpse':
        move_checkbox = gpse_mouse_move_checkbox
        data = dfe
        gmap = gpse

    # do not update
    if move_checkbox.active == []:
        return

    # convert event time and handle null cases
    epoch = (1e-3) * event.x
    if (data.size == 0) or (epoch < data.iloc[0]['TS_SEC']) or (epoch > data.iloc[-1]['TS_SEC']):
        gmap.position.data = {'lat': np.array([]), 'lon': np.array([])}
        return

    # update marker object
    delta = np.abs(epoch - data['TS_SEC'])
    if delta.min() > 3600:
        gmap.position.data = {'lat': np.array([]), 'lon': np.array([])}
        return
    pos = np.argmin(delta)
    lat, lon = data.iloc[pos][['latitude', 'longitude']].values
    gmap.position.data = {'lat': np.array([lat]), 'lon': np.array([lon])}

def mouse_leave(event, sender):

    # idenfity sender
    if sender == 'gps':
        move_checkbox = gps_mouse_move_checkbox
        gmap = gps
    elif sender == 'gpse':
        move_checkbox = gpse_mouse_move_checkbox
        gmap = gpse

    # do not update
    if move_checkbox.active == []:
        return

    # reset position glyph
    gmap.position.data = {'lat': np.array([]), 'lon': np.array([])}

def segments_multiselect_callback(attr, old, new):
    """
    update GPS segments data source based on multiselect value
    """

    # null case
    if not new:
        gps.segments.data = {'lat': np.array([]), 'lon': np.array([])}
        gps_signal.segments.data = {'x': np.array([]), 'y': np.array([])}
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
        time = np.hstack((time, df.loc[ok, 'utc'].values, df.loc[ok, 'utc'].values[-1]))
        value = np.hstack((value, df.loc[ok, 'cumulative_distance_miles'].values, np.nan))
    gps.segments.data = {'lat': lat[:-1], 'lon': lon[:-1]}
    gps_signal.segments.data = {'x': time[:-1], 'y': value[:-1]}
    status.text = status_done

def maneuvers_multiselect_callback(attr, old, new):
    """
    update GPS maneuvers data source based on multiselect value
    """

    # null case
    if not new:
        gpse.segments.data = {'lat': np.array([]), 'lon': np.array([])}
        gpse_signal.segments.data = {'x': np.array([]), 'y': np.array([])}
        return

    # build lat/lon data from selected maneuvers, update source
    status.text = status_running
    lat = np.array([])
    lon = np.array([])
    time = np.array([]).astype(np.datetime64)
    value = np.array([])
    for mx in new:
        ok = dfm['segmentgroup__maneuver'] == mx
        assert ok.sum() > 0
        for x0, x1 in zip(dfm.loc[ok, 'sg_start'].values, dfm.loc[ok, 'sg_end'].values):
            lat = np.hstack((lat, dfe.loc[x0 : x1, 'latitude'].values, np.nan))
            lon = np.hstack((lon, dfe.loc[x0 : x1, 'longitude'].values, np.nan))
            time = np.hstack((time, dfe.loc[x0 : x1, 'utc'].values, df.loc[ok, 'utc'].values[-1]))
            value = np.hstack((value, dfe.loc[x0 : x1, 'cumulative_distance_miles'].values, np.nan))
    gpse.segments.data = {'lat': lat[:-1], 'lon': lon[:-1]}
    gpse_signal.segments.data = {'x': time[:-1], 'y': value[:-1]}
    status.text = status_done

def convert_geom_callback():
    """
    expand highlighted maneuvers to full geometry objects
    """

    # null case
    if not maneuvers.value:
        return

    # current map view
    lon0, lat0 = webm_to_gps(gps.fig.x_range.start, gps.fig.y_range.start)
    lon1, lat1 = webm_to_gps(gps.fig.x_range.end, gps.fig.y_range.end)

    # build lat/lon data from selected maneuvers within current map view, update source
    status.text = status_running
    lat = np.array([])
    lon = np.array([])
    for mx in maneuvers.value:
        ok = \
            (dfm['segmentgroup__maneuver'] == mx) & \
            (dfm['latitude_gps'] >= lat0) & \
            (dfm['latitude_gps'] <= lat1) & \
            (dfm['longitude_gps'] >= lon0) & \
            (dfm['longitude_gps'] <= lon1)
        if ok.sum() == 0:
            continue
        for gm in dfm.loc[ok, 'segmentgroup__geomsegment4326'].values:
            gx = from_wkb([gm])[0]
            assert gx.geom_type == 'LineString'
            gx0, gx1 = gps_to_webm(np.array(gx.xy[0]), np.array(gx.xy[1]))
            lat = np.hstack((lat, gx1, np.nan))
            lon = np.hstack((lon, gx0, np.nan))
    gpse.segments.data = {'lat': lat[:-1], 'lon': lon[:-1]}
    status.text = status_done

# collision prediction model datadir and validate
datadir = r'/mnt/home/russell.burdt/data/collision-model/v3/rb'
assert os.path.isdir(datadir)
assert all([os.path.isdir(os.path.join(datadir, x + '.parquet')) for x in ['gps2', 'gpse']])

# read population data
dcm = pd.read_pickle(os.path.join(datadir, 'dcm-gps.p'))
dcm['window'] = [f"""
    {pd.Timestamp(a).strftime('%d %b %Y')} to
    {pd.Timestamp(b).strftime('%d %b %Y')}""" for a, b in zip(dcm['time0'], dcm['time1'])]

# vehicle-eval select objects
vs = {}
vs['main-title'] = Div(text='<strong>Select Vehicle Eval</strong>', width=300)
vs['industry-select'] = Select(title='industry', width=300)
vs_callback_industry_select = partial(vs_callback, sender=vs['industry-select'])
vs['industry-select'].on_change('value', vs_callback_industry_select)
vs['company-select'] = Select(title='company', width=300)
vs_callback_company_select = partial(vs_callback, sender=vs['company-select'])
vs['company-select'].on_change('value', vs_callback_company_select)
vs['vehicle-select'] = Select(title='vehicle-id', width=300)
vs_callback_vehicle_select = partial(vs_callback, sender=vs['vehicle-select'])
vs['vehicle-select'].on_change('value', vs_callback_vehicle_select)
vs['window-select'] = Select(title='predictor interval', width=300)
vs_callback_window_select = partial(vs_callback, sender=vs['window-select'])
vs['window-select'].on_change('value', vs_callback_window_select)

# status object and messages
status = Div(width=300)
status_running = """<strong style="color:red">Status -- Running</strong>"""
status_done = """<strong style="color:blue">Status -- Done</strong>"""

# gps map interface
gps = MapInterface(width=600, height=300, hover=True)
gps.hover.renderers = [x for x in gps.fig.renderers if x.name == 'path']
gps.hover.tooltips = [
    ('latitude', '@latitude{%.4f}'),
    ('longitude', '@longitude{%.4f}'),
    ('utc', '@utc{%d %b %Y %H:%M:%S}')]
gps.hover.formatters = {'@latitude': 'printf', '@longitude': 'printf', '@utc': 'datetime'}
gps_signal = MultiLineInterface(width=600, height=200, ylabel='miles', title='cumulative miles vs time',
    n=1, cross=True, datetime=True, manual_xlim=True, manual_ylim=True)
gps_signal.fig.on_event(MouseMove, partial(mouse_move, sender='gps'))
gps_signal.fig.on_event(MouseLeave, partial(mouse_leave, sender='gps'))
gps_mouse_move_checkbox = CheckboxGroup(labels=['show position on map'], active=[])

# gpse map interface
gpse = MapInterface(width=600, height=300, hover=True)
gpse.fig.x_range = gps.fig.x_range
gpse.fig.y_range = gps.fig.y_range
gpse.hover.renderers = [x for x in gpse.fig.renderers if x.name == 'path']
gpse.hover.tooltips = [
    ('segment__id', '@segment__id{%.0f}'),
    ('latitude', '@latitude{%.4f}'),
    ('longitude', '@longitude{%.4f}'),
    ('utc', '@utc{%d %b %Y %H:%M:%S}')]
gpse.hover.formatters = {'@segment__id': 'printf', '@latitude': 'printf', '@longitude': 'printf', '@utc': 'datetime'}
gpse_signal = MultiLineInterface(width=600, height=200, ylabel='miles', title='cumulative miles vs time',
    n=1, cross=True, datetime=True, manual_xlim=True, manual_ylim=True)
gpse_signal.fig.x_range = gps_signal.fig.x_range
gpse_signal.fig.y_range = gps_signal.fig.y_range
gpse_signal.fig.on_event(MouseMove, partial(mouse_move, sender='gpse'))
gpse_signal.fig.on_event(MouseLeave, partial(mouse_leave, sender='gpse'))
gpse_mouse_move_checkbox = CheckboxGroup(labels=['show position on map'], active=[])

# segment selection interface
segments = MultiSelect(title='GPS Segment(s)', width=200, height=120)
segments.on_change('value', segments_multiselect_callback)

# maneuver selection interface
maneuvers = MultiSelect(title='GPS Maneuver(s)', width=300, height=300)
maneuvers.on_change('value', maneuvers_multiselect_callback)
convert_geom = Button(label='convert maneuver geometry', button_type='success', width=200)
convert_geom.on_click(convert_geom_callback)

# app layout
layout = row(
    # vehicle-eval select and status
    column(vs['main-title'], vs['industry-select'], vs['company-select'], vs['vehicle-select'], vs['window-select'], status),
    # gps interface
    column(gps.fig, gps_mouse_move_checkbox, gps_signal.fig),
    # gpse interface
    column(gpse.fig, gpse_mouse_move_checkbox, gpse_signal.fig),
    # segments and maneuvers multi-select
    column(segments, maneuvers, convert_geom))

# create document object and list for created video files
doc = curdoc()
doc.add_root(layout)
doc.title = 'collision model gpse app'

# initialize state
vs_callback(None, None, None, None)
