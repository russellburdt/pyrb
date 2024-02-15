
"""
bokeh app to explore collision prediction model gps data by single vehicle-eval
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
from bokeh.models import Tabs, Panel, Select, MultiSelect, CheckboxGroup, Div, Slider, Button, RadioGroup
from bokeh.layouts import column, row
from bokeh.events import MouseMove, MouseLeave
from pyrb.bokeh import MetricDistributionInterface, HorizontalBarChartInterface, MapInterface, MultiLineInterface
from pyrb.bokeh import str_axis_labels, link_axes
from ipdb import set_trace


# reset interfaces
def reset_interfaces(
        reset_gps=False,
        reset_signal=False,
        reset_sources=False,
        reset_segments_multiselect=False,
        reset_all=False):
    """
    reset all or individual interfaces
    """
    global df   # gps

    if reset_gps or reset_all:
        gps.reset_interface()
        gps.reset_map_view()

    if reset_signal or reset_all:
        signal.reset_interface()

    if reset_sources or reset_all:
        df = pd.DataFrame()     # gps

    if reset_segments_multiselect or reset_all:
        segments.options = []
        segments.remove_on_change('value', segments_multiselect_callback)
        segments.value = []
        segments.on_change('value', segments_multiselect_callback)
        segments_all.active = []

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

    # update status
    status.text = status_done

# gps map interface callbacks
def update_gps_interface(vehicle):
    """
    update objects on GPS interface
    """
    global df   # gps

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
    gps.path.data = {'lon': lon, 'lat': lat,
        'utc': df['utc'], 'longitude': df['longitude_gps'], 'latitude': df['latitude_gps']}

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
        # gps.segments.data = {'lat': np.array([]), 'lon': np.array([])}
        gps.path.data = {'lat': np.array([]), 'lon': np.array([])}
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
    # gps.segments.data = {'lat': lat[:-1], 'lon': lon[:-1]}
    gps.path.data = {'lat': lat[:-1], 'lon': lon[:-1]}
    signal.segments.data = {'x': time[:-1], 'y': value[:-1]}
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

# collision prediction model datadir and validate
datadir = r'/mnt/home/russell.burdt/data/collision-model/v2/munich-re'
assert os.path.isdir(datadir)
assert os.path.isdir(os.path.join(datadir, 'gps2.parquet'))

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

# gps map interface objects
gps = MapInterface(width=600, height=300, hover=True)
gps.hover.renderers = [x for x in gps.fig.renderers if x.name == 'path']
gps.hover.tooltips = [
    ('latitude', '@latitude{%.4f}'),
    ('longitude', '@longitude{%.4f}'),
    ('utc', '@utc{%d %b %Y %H:%M:%S}')]
gps.hover.formatters = {'@latitude': 'printf', '@longitude': 'printf', '@utc': 'datetime'}
signal = MultiLineInterface(width=600, height=200, ylabel='miles', title='cumulative miles vs time',
    n=1, cross=True, datetime=True, manual_xlim=True, manual_ylim=True)
signal.fig.on_event(MouseMove, gps_interface_mouse_move)
signal.fig.on_event(MouseLeave, gps_interface_mouse_leave)
mouse_move_checkbox = CheckboxGroup(labels=['show position on map'], active=[])
segments = MultiSelect(title='GPS Segment(s)', width=160, height=120)
segments.on_change('value', segments_multiselect_callback)
segments_all = CheckboxGroup(labels=['All Segments'], active=[])
segments_all.on_click(partial(checkbox, sender='segments'))

# app layout based on panel objects
eval_select = column(
    vs['main-title'], vs['industry-select'], vs['company-select'], vs['vehicle-select'], vs['window-select'], status)
layout_gps = row(
    eval_select,
    column(gps.fig, mouse_move_checkbox, signal.fig),
    column(segments, segments_all))
layout = Tabs(tabs=[
    Panel(child=layout_gps, title='GPS Map Interface')])

# create document object and list for created video files
doc = curdoc()
doc.session_context.vids = []
doc.add_root(layout)
doc.title = 'collision model gps app'

# initialize state
vs_callback(None, None, None, None)
