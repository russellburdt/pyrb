
"""
vehicle utilization prediction model app
"""

import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pickle
from datetime import datetime
from functools import partial
from glob import glob
from shutil import move
from bokeh.io import show, curdoc
from bokeh.models import Tabs, Panel, Select, MultiSelect, CheckboxGroup, Div, Slider, Button, RadioGroup
from bokeh.layouts import column, row
from bokeh.events import MouseMove, MouseLeave
from pyrb.bokeh import MetricDistributionInterface, HorizontalBarChartInterface, MapInterface, MultiLineInterface
from pyrb.bokeh import str_axis_labels, link_axes, console_str_objects
from ipdb import set_trace


# reset interfaces
def reset_interfaces(
        reset_gps=False,
        reset_sources=False,
        reset_segments_multiselect=False,
        reset_nodes_multiselect=False,
        reset_links_multiselect=False,
        reset_all=False):
    """
    reset all or individual interfaces
    """
    global df   # gps
    global dn   # nodes
    global dx   # links
    global de   # events
    global dex  # selected events
    global db   # behaviors
    global dbx  # selected behaviors

    if reset_gps or reset_all:
        gps.reset_interface()
        gps.reset_map_view()
        signal.reset_interface()
        intervals.reset_interface()
        vehicle_state.reset_interface()
        vehicle_state_signal.reset_interface()

    if reset_sources or reset_all:
        df = pd.DataFrame()     # gps
        dn = pd.DataFrame()     # nodes
        dx = pd.DataFrame()     # links
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

    if reset_nodes_multiselect or reset_all:
        nodes.options = []
        nodes.remove_on_change('value', nodes_multiselect_callback)
        nodes.value = []
        nodes.on_change('value', nodes_multiselect_callback)
        nodes_all.active = []

    if reset_links_multiselect or reset_all:
        links.options = []
        links.remove_on_change('value', links_multiselect_callback)
        links.value = []
        links.on_change('value', links_multiselect_callback)
        links_all.active = []

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

    # update industry select menu
    if c0:
        vs['industry-select'].options = sorted(pd.unique(dp['IndustryDesc']))
        vs['industry-select'].remove_on_change('value', vs_callback_industry_select)
        vs['industry-select'].value = vs['industry-select'].options[0]
        vs['industry-select'].on_change('value', vs_callback_industry_select)

    # update company select menu
    if c0 or c1:
        ok = dp['IndustryDesc'] == vs['industry-select'].value
        assert ok.sum() > 0
        vs['company-select'].options = sorted(pd.unique(dp.loc[ok, 'CompanyName']))
        vs['company-select'].remove_on_change('value', vs_callback_company_select)
        vs['company-select'].value = vs['company-select'].options[0]
        vs['company-select'].on_change('value', vs_callback_company_select)

    # update vehicle select menu
    if c0 or c1 or c2:
        ok = (dp['IndustryDesc'] == vs['industry-select'].value) & (dp['CompanyName'] == vs['company-select'].value)
        assert ok.sum() > 0
        vs['vehicle-select'].options = sorted(pd.unique(dp.loc[ok, 'VehicleId']))
        vs['vehicle-select'].remove_on_change('value', vs_callback_vehicle_select)
        vs['vehicle-select'].value = vs['vehicle-select'].options[0]
        vs['vehicle-select'].on_change('value', vs_callback_vehicle_select)

    # identify vehicle eval
    ok = (dp['VehicleId'] == vs['vehicle-select'].value)
    assert ok.sum() == 1
    vehicle = dp.loc[ok].squeeze()

    # update gps interface then status
    update_gps_interface(vehicle)
    status.text = status_done

# gps map interface callbacks
def update_gps_interface(vehicle):
    """
    update objects on GPS interface
    """
    global df
    global dn
    global dx

    # gps raw data parquet file for vehicle
    vid = vehicle['VehicleId']
    fn = glob(os.path.join(datadir, 'gps.parquet', f"""VehicleId={vid}""", '*.parquet'))
    if len(fn) == 0:
        return

    # read gps raw data
    assert len(fn) == 1
    df = pq.ParquetFile(fn[0]).read().to_pandas()
    assert df.size > 0
    assert datetime.utcfromtimestamp(df['TS_SEC'].min()) > tmin
    assert datetime.utcfromtimestamp(df['TS_SEC'].max()) < tmax
    time0 = int((vehicle['time0'] - datetime(1970, 1, 1)).total_seconds())
    time1 = int((vehicle['time1'] - datetime(1970, 1, 1)).total_seconds())

    # update gps path data
    sid = df['segmentId'].values
    nok = np.isnan(sid)
    lon = df['longitude'].values
    lat = df['latitude'].values
    lon[nok] = np.nan
    lat[nok] = np.nan
    gps.path.data = {'lon': lon, 'lat': lat}

    # update gps segmentation console
    dsv = ds.loc[ds['VehicleId'] == vid]
    assert dsv.shape[0] == 1
    dsv = dsv.squeeze()
    c0, c1 = console_str_objects(300)
    desc = f"""
        left window boundary to first record, {dsv['left_window_to_first_segmented_record']:.1f}<br>
        last record to right window boundary, {dsv['last_segmented_record_to_right_window']:.1f}<br>
        number of segments, {dsv['n_segments']:.0f}<br>
        number of records in all segments, {dsv['n_records_segments']:.0f}<br>
        number of days covered by segments, {dsv['n_days_segments']:.1f}<br>
        number of days not covered by segments, {dsv['n_days_no_segments']:.1f}<br>
        total num of days, {dsv['total_days']:.0f}"""
    seg_desc.text = c0 + desc + c1

    # update gps map view and title
    gps.reset_map_view(lon0=np.nanmin(lon), lon1=np.nanmax(lon), lat0=np.nanmin(lat), lat1=np.nanmax(lat), convert=False)
    gps.fig.title.text = f"""{vehicle['VehicleId']}, {tmin.strftime('%d %b %Y')} to {tmax.strftime('%d %b %Y')}"""

    # update signal data
    miles = df['cumulative_distance_miles'].values
    miles -= np.nanmin(miles)
    signal.data_sources[0].data = {'x': df['utc'].values, 'y': miles}
    signal.fig.x_range.start = 1000 * (time0 - 86400)
    signal.fig.x_range.end = 1000 * (time1 + 86400)
    signal.fig.y_range.start = 0
    signal.fig.y_range.end = 1.05 * np.nanmax(miles)

    # update GPS segments multiselect
    xs = np.unique(df['segmentId'])
    xs = np.sort(xs[~np.isnan(xs)]).astype('int')
    segments.options = [str(x) for x in xs]

    # update distribution of GPS record intervals in seconds
    bins = np.arange(0, 1200, 30)
    seconds = df['time_interval_sec'].values
    seconds = seconds[~np.isnan(seconds)]
    if seconds.size > 0:
        sx = np.digitize(seconds, bins)
        assert (sx == 0).sum() == 0
        top = np.array([(sx == xi).sum() for xi in range(1, bins.size + 1)]).astype('float')
        width = np.diff(bins)[0]
        intervals.data.data = {'x': bins + width / 2, 'width': np.tile(width, top.size), 'top': top}
        intervals.fig.x_range.start = 0
        intervals.fig.x_range.end = 1200
        intervals.fig.y_range.start = 0
        intervals.fig.y_range.end = 1.1 * top.max()

    # vehicle state characterization data
    labels = np.array(['motion', 'off', 'idle'])
    right = np.array([df.loc[df['vehicle_state'] == x, 'time_interval_sec'].sum() / 86400 for x in labels])
    sid = np.sort(pd.unique(df['segmentId']))
    sid = sid[~np.isnan(sid)]
    total_days = (time1 - time0) / 86400
    days_to_left_window = (df.loc[df['segmentId'] == sid[0], 'TS_SEC'].min() - time0) / 86400
    days_to_right_window = (time1 - df.loc[df['segmentId'] == sid[-1], 'TS_SEC'].max()) / 86400
    sid = np.array([(
        np.where(df['segmentId'] == sid[x])[0][-1],
        np.where(df['segmentId'] == sid[x + 1])[0][0]) for x in np.arange(sid.size - 1)])
    days_no_segments = np.sum([np.diff(df.loc[xa:xb, 'TS_SEC']).sum() for xa, xb in sid]) / 86400
    assert np.isclose(days_to_left_window + days_to_right_window + days_no_segments + right.sum(), total_days)
    labels = np.hstack((labels, np.array(['to left boundary', 'to right boundary', 'no segments'])))[::-1]
    right = np.hstack((right, np.array([days_to_left_window, days_to_right_window, days_no_segments])))[::-1]
    vehicle_state.fig.title.text = f'days in vehicle state over {total_days:.0f} days'

    # update vehicle state horizontal bar chart interface
    height = 0.8
    vehicle_state.data.data = {
        'y': np.arange(right.size),
        'right': right,
        'height': np.tile(height, right.size)}
    vehicle_state.fig.yaxis.ticker.ticks = list(range(right.size))
    str_axis_labels(axis=vehicle_state.fig.yaxis, labels=labels)
    vehicle_state.fig.x_range.start = 0
    vehicle_state.fig.x_range.end = 1.1 * right.max()
    vehicle_state.fig.y_range.start = - height
    vehicle_state.fig.y_range.end = right.size - 1 + height
    vehicle_state.label_source.data = {
        'x': right,
        'y': np.arange(right.size),
        'text': [f'{x:0.2f}' for x in right]}
    vehicle_state_callback(None, None, None)

    # # IMN nodes raw data parquet file for vehicle
    # fn = glob(os.path.join(datadir, 'nodes.parquet', f"""VehicleId={vehicle['VehicleId']}""", '*.parquet'))
    # if len(fn) == 0:
    #     return

    # # read IMN nodes raw data and filter by vehicle eval time bounds
    # assert len(fn) == 1
    # dn = pq.ParquetFile(fn[0]).read().to_pandas()
    # dn = dn.loc[(dn['TS_SEC1'] > time0) & (dn['TS_SEC0'] < time1)].reset_index(drop=True)
    # if dn.size == 0:
    #     return

    # # update IMN nodes multiselect
    # nodes.options = [str(x) for x in dn.index]

    # # IMN links raw data parquet file for vehicle
    # fn = glob(os.path.join(datadir, 'links.parquet', f"""VehicleId={vehicle['VehicleId']}""", '*.parquet'))
    # if len(fn) == 0:
    #     return

    # # read IMN links raw data and filter by vehicle eval time bounds
    # assert len(fn) == 1
    # dx = pq.ParquetFile(fn[0]).read().to_pandas()
    # dx = dx.loc[(dx['TS_SEC1'] > time0) & (dx['TS_SEC0'] < time1)].reset_index(drop=True)
    # if dx.size == 0:
    #     return

    # # update IMN links multiselect
    # links.options = [str(x) for x in dx.index]

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
        time = np.hstack((time, df.loc[ok, 'utc'].values, df.loc[ok, 'utc'].values[-1]))
        value = np.hstack((value, df.loc[ok, 'cumulative_distance_miles'].values, np.nan))
    gps.segments.data = {'lat': lat[:-1], 'lon': lon[:-1]}
    signal.segments.data = {'x': time[:-1], 'y': value[:-1]}
    status.text = status_done

def nodes_multiselect_callback(attr, old, new):
    """
    update IMN nodes data source based on multiselect value
    """

    # null case
    if not new:
        gps.nodes.data = {'lon': [], 'lat': [], 'lon_centroids': np.array([]), 'lat_centroids': np.array([])}
        return

    # build lat/lon data from selected nodes, update source
    status.text = status_running
    lon, lat = [], []
    for nx in [int(x) for x in new]:
        node = dn.loc[nx]
        polygon = node['polygon']
        assert (polygon[:9] == 'POLYGON((') and (polygon[-2:] == '))')
        polygon = polygon[9:-2].split(',')
        polygon = [x.split(' ') for x in polygon]
        polygon = [[float(xi) for xi in x] for x in polygon]
        lon.append([x[0] for x in polygon])
        lat.append([x[1] for x in polygon])
    gps.nodes.data = {'lat': lat, 'lon': lon,
        'lat_centroids': [(x[0] + x[-1]) / 2 for x in lat],
        'lon_centroids': [(x[0] + x[-1]) / 2 for x in lon]}
    status.text = status_done

def links_multiselect_callback(attr, old, new):
    """
    update IMN links data source based on multiselect value
    """

    # null case
    if not new:
        gps.links.data = {'lon': np.array([]), 'lat': np.array([])}
        return

    # build lat/lon data from selected links, update source
    status.text = status_running
    lat = np.array([])
    lon = np.array([])
    for lx in [int(x) for x in new]:
        link = dx.loc[lx]
        lon = np.hstack((lon, np.array([link['lon0'], link['lon1']]), np.nan))
        lat = np.hstack((lat, np.array([link['lat0'], link['lat1']]), np.nan))
    gps.links.data = {'lat': lat[:-1], 'lon': lon[:-1]}
    status.text = status_done

def vehicle_state_callback(attr, old, new):
    """
    update vehicle state interface
    """

    # identify day indices
    x0 = df.assign(index=df.index).resample('D', on='utc')['index'].min().to_frame().rename(columns={'index': 'start index'})
    x1 = df.assign(index=df.index).resample('D', on='utc')['index'].max().to_frame().rename(columns={'index': 'end index'})
    assert all(x0.index == x1.index)
    days = pd.merge(x0, x1, how='inner', left_index=True, right_index=True)

    # validate
    kws = {'rule': 'D', 'on': 'utc', 'closed': 'right', 'label': 'left'}
    hours_per_day = df.resample(**kws)['all_time_interval_sec'].sum() / 3600
    assert all(hours_per_day[1:-1] == 24)
    x0 = df.loc[df['vehicle_state'].isin(['motion', 'idle', 'off'])].resample(**kws)['time_interval_sec'].sum() / 3600
    x1 = df.loc[pd.isnull(df['vehicle_state'])].resample(**kws)['all_time_interval_sec'].sum() / 3600
    assert all(np.nansum(pd.merge(x0, x1, left_index=True, right_index=True, how='outer').values, axis=1)[1:-1] == 24)

    # metric and data
    metric = vehicle_state_signal_select.value
    rule = 'D' if 'by day' in metric else 'W' if 'by week' in metric else None
    assert rule is not None
    kws = {'rule': rule, 'on': 'utc', 'closed': 'right', 'label': 'left'}
    if 'gps miles' in metric:
        data = df.resample(**kws)['distance_interval_miles'].sum()
    elif 'hours in motion' in metric:
        data = df.loc[df['vehicle_state'] == 'motion'].resample(**kws)['time_interval_sec'].sum() / 3600
    elif 'idle hours' in metric:
        data = df.loc[df['vehicle_state'] == 'idle'].resample(**kws)['time_interval_sec'].sum() / 3600
    elif 'off hours' in metric:
        data = df.loc[df['vehicle_state'] == 'off'].resample(**kws)['time_interval_sec'].sum() / 3600
    elif 'unresolved hours' in metric:
        data = df.loc[pd.isnull(df['vehicle_state'])].resample(**kws)['all_time_interval_sec'].sum() / 3600
    elif 'sum of time intervals' in metric:
        data = df.resample(**kws)['all_time_interval_sec'].sum() / 3600

    # update
    vehicle_state_signal.data_sources[0].data = {'x': data.index, 'y': data.values}
    vehicle_state_signal.fig.x_range.start = tmin - pd.Timedelta(days=1)
    vehicle_state_signal.fig.x_range.end = tmax + pd.Timedelta(days=1)
    vehicle_state_signal.fig.y_range.start = 0
    vehicle_state_signal.fig.y_range.end = 1.1 * data.values.max()
    vehicle_state_signal.fig.title.text = metric

# single callback for all checkbox/radio objects
def checkbox(event, sender):
    """
    callback for all checkbox/radio objects
    """

    if sender == 'segments':
        if not segments.options:
            return
        segments.value = segments.options if event else []

    elif sender == 'nodes':
        if not nodes.options:
            return
        nodes.value = nodes.options if event else []

    elif sender == 'links':
        if not links.options:
            return
        links.value = links.options if event else []

# vehicle utilization prediction model datadir
datadir = r'/mnt/home/russell.burdt/data/utilization/amt'
assert os.path.isdir(datadir)
assert os.path.isdir(os.path.join(datadir, 'gps.parquet'))

# population, window, and gps segmentation data
dp = pd.read_pickle(os.path.join(datadir, 'dp.p'))
assert all([pd.unique(dp[x]).size == 1 for x in ['time0', 'time1']])
tmin = dp.loc[0, 'time0']
tmax = dp.loc[0, 'time1']
ds = pd.read_pickle(os.path.join(datadir, 'coverage', 'gps_segmentation_metrics.p'))

# vehicle-eval select objects
vs = {}
vs['main-title'] = Div(text='<strong>Select Vehicle</strong>', width=300)
vs['industry-select'] = Select(title='industry', width=300)
vs_callback_industry_select = partial(vs_callback, sender=vs['industry-select'])
vs['industry-select'].on_change('value', vs_callback_industry_select)
vs['company-select'] = Select(title='company', width=300)
vs_callback_company_select = partial(vs_callback, sender=vs['company-select'])
vs['company-select'].on_change('value', vs_callback_company_select)
vs['vehicle-select'] = Select(title='vehicle-id', width=300)
vs_callback_vehicle_select = partial(vs_callback, sender=vs['vehicle-select'])
vs['vehicle-select'].on_change('value', vs_callback_vehicle_select)

# status object and messages
status = Div(width=300)
status_running = """<strong style="color:red">Status -- Running</strong>"""
status_done = """<strong style="color:blue">Status -- Done</strong>"""

# gps map interface objects
gps = MapInterface(width=600, height=300, size=8)
signal = MultiLineInterface(width=600, height=200, ylabel='miles', title='cumulative miles vs time',
    n=1, cross=True, datetime=True, manual_xlim=True, manual_ylim=True)
signal.fig.on_event(MouseMove, gps_interface_mouse_move)
signal.fig.on_event(MouseLeave, gps_interface_mouse_leave)
mouse_move_checkbox = CheckboxGroup(labels=['show position on map'], active=[])
segments = MultiSelect(title='GPS Segment(s)', width=160, height=120)
segments.on_change('value', segments_multiselect_callback)
segments_all = CheckboxGroup(labels=['All Segments'], active=[])
segments_all.on_click(partial(checkbox, sender='segments'))
nodes = MultiSelect(title='IMN Node(s)', width=160, height=120)
nodes.on_change('value', nodes_multiselect_callback)
nodes_all = CheckboxGroup(labels=['All Nodes'], active=[])
nodes_all.on_click(partial(checkbox, sender='nodes'))
links = MultiSelect(title='IMN Link(s)', width=160, height=120)
links.on_change('value', links_multiselect_callback)
links_all = CheckboxGroup(labels=['All Links'], active=[])
links_all.on_click(partial(checkbox, sender='links'))
seg_desc_title = Div(text='<strong>GPS Segmentation Metrics</strong>', width=300)
seg_desc = Div()

# gps metrics interface objects
intervals = MetricDistributionInterface(width=700, height=260, xlabel='seconds', ylabel='bin count', size=12, logscale=False,
    title='distribution of gps record intervals within segments', dimensions='height')
vehicle_state = HorizontalBarChartInterface(width=700, height=260, xlabel='days in vehicle state',
    size=12, include_nums=True, pan_dimensions='height')
vehicle_state_signal_select = Select(title='select vehicle state metric', width=300,
    value = 'gps miles by day', options=[
    'gps miles by day', 'hours in motion by day', 'idle hours by day', 'off hours by day', 'sum of time intervals by day', 'unresolved hours by day',
    'gps miles by week', 'hours in motion by week', 'idle hours by week', 'off hours by week', 'sum of time intervals by week', 'unresolved hours by week'])
vehicle_state_signal_select.on_change('value', vehicle_state_callback)
vehicle_state_signal = MultiLineInterface(
    width=700, height=260, n=1, datetime=True, manual_xlim=True, manual_ylim=True, dimensions='width', box_dimensions='width')

# app layout based on panel objects
eval_select = column(
    vs['main-title'], vs['industry-select'], vs['company-select'], vs['vehicle-select'], status)
layout_gps = row(
    column(eval_select, seg_desc_title, seg_desc),
    column(gps.fig, mouse_move_checkbox, signal.fig),
    column(segments, segments_all, nodes, nodes_all, links, links_all))
layout_metrics = row(
    column(eval_select, vehicle_state_signal_select),
    column(vehicle_state_signal.fig, vehicle_state.fig),
    column(intervals.fig))
layout = Tabs(tabs=[
    Panel(child=layout_gps, title='GPS Map Interface'),
    Panel(child=layout_metrics, title='GPS Metrics Interface')])

# create document object and list for created video files
doc = curdoc()
doc.session_context.vids = []
doc.add_root(layout)
doc.title = 'vehicle utilization prediction model app'

# initialize state
vs_callback(None, None, None, None)
