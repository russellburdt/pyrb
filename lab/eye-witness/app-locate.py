
"""
eye witness app for vehicles within location and time window
"""

import os
import lytx
import pytz
import sqlalchemy as sa
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from glob import glob
from shutil import rmtree
from functools import partial
from datetime import datetime
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import Div, Button, Select
from pyrb.bokeh import MapInterface, console_str_objects
from pyrb.processing import webm_to_gps
from ipdb import set_trace


# eye-witness time window and timezone
time0 = '2023-7-13 21:00:00'
time1 = '2023-7-13 22:00:00'
tz = 'US/Central'
time0, time1 = pd.Timestamp(time0), pd.Timestamp(time1)
assert time1 > time0
assert tz in pytz.all_timezones

# eye-witness time window, UTC and epoch
t0 = time0.tz_localize(tz).astimezone('UTC').replace(tzinfo=None)
t1 = time1.tz_localize(tz).astimezone('UTC').replace(tzinfo=None)
ta = int((t0 - datetime(1970, 1, 1)).total_seconds())
tb = int((t1 - datetime(1970, 1, 1)).total_seconds())

def app_callback():

    # eye-witness location window
    x0, x1 = gps.fig.x_range.start, gps.fig.x_range.end
    y0, y1 = gps.fig.y_range.start, gps.fig.y_range.end
    lon0, lat0 = webm_to_gps(x0, y0)
    lon1, lat1 = webm_to_gps(x1, y1)

    # vehicles within location and time window
    status_all.text = c0 + f"""identifying vehicles""" + c1
    snow = lytx.get_conn('snowflake')
    query = f"""
        SELECT DISTINCT(VEHICLE_ID)
        FROM GPS.GPS_ENRICHED
        WHERE TS_SEC BETWEEN {ta} AND {tb}
        AND LONGITUDE BETWEEN {lon0} AND {lon1}
        AND LATITUDE BETWEEN {lat0} AND {lat1}
        AND VEHICLE_ID <> '00000000-0000-0000-0000-000000000000'"""
    dx = pd.read_sql_query(query, snow)
    vids = np.unique([x.upper() for x in dx['VEHICLE_ID'].values])

    # null case (no vehicles found in window)
    if vids.size == 0:
        status_all.text = c0 + f"""
            zero vehicles identified in {cs}
            modify eye-witness location window on map""" + c1
        return

    # metadata for vehicles
    status_all.text = c0 + f"""identifying vehicle metadata""" + c1
    edw = lytx.get_conn('edw')
    vstr = ','.join([f"""'{x}'""" for x in vids])
    query = f"""
        SELECT
            ERA.EventRecorderId, ERA.VehicleId, ERA.CreationDate, ERA.DeletedDate, ERA.GroupId,
            G.Name as GroupName, C.CompanyId, C.CompanyName, C.IndustryDesc, D.SerialNumber
        FROM hs.EventRecorderAssociations AS ERA
            LEFT JOIN flat.Groups AS G ON ERA.GroupId = G.GroupId
            LEFT JOIN flat.Companies AS C ON G.CompanyId = C.CompanyId
            LEFT JOIN flat.Devices AS D ON D.DeviceId = ERA.EventRecorderId
        WHERE ERA.VehicleId IN ({vstr})
        AND ERA.CreationDate < '{t0}'
        AND ERA.DeletedDate > '{t1}'
        ORDER BY VehicleId"""
    dm = pd.read_sql_query(sa.text(query), edw)
    assert pd.unique(dm['VehicleId']).size == dm.shape[0] == vids.size

    # gps data for vehicles
    status_all.text = c0 + f"""extracting gps data""" + c1
    df = pd.DataFrame({'VehicleId': vids})
    df['time0'] = t0
    df['time1'] = t1
    lytx.distributed_data_extraction('gps', datadir, df, 'VehicleId', n=100, distributed=False, assert_new=False)

    # update status_all, disable update
    status_all.text = c0 + f"""
        {vids.size} vehicles identified in eye-witness location window<br>
        lower-left corner, ({lon0:.4f}, {lat0:.4f})<br>
        upper-right corner, ({lon1:.4f}, {lat1:.4f})<br>
        and {cs}""" + c1
    update.disabled = True

    # update vehicle-id select menu, set and run vehicle callback
    select.options = sorted(vids)
    select.value = select.options[0]
    select.on_change('value', partial(vehicle_callback, dm=dm))
    vehicle_callback(None, None, None, dm)

def vehicle_callback(attr, old, new, dm):

    # validate and update status_vehicle
    vid = select.value
    assert vid in dm['VehicleId'].values
    dx = dm.loc[dm['VehicleId'] == vid].squeeze()
    status_vehicle.text = c0 + f"""
        VehicleId, {vid}<br>
        SerialNumber, {dx['SerialNumber']}<br>
        CompanyName, {dx['CompanyName']}<br>
        IndustryDesc, {dx['IndustryDesc']}<br>""" + c1

    # update gps path data
    fn = glob(os.path.join(datadir, 'gps.parquet', f'VehicleId={vid}', '*.parquet'))
    assert len(fn) == 1
    df = pq.ParquetFile(fn[0]).read().to_pandas()
    localtime = np.array([datetime.utcfromtimestamp(x).astimezone(pytz.timezone(tz)).replace(tzinfo=None)
        for x in df['TS_SEC'].values])
    gps.path.data = {
        'lon': df['longitude'].values,
        'lat': df['latitude'].values,
        'longitude': df['longitude_gps'].values,
        'latitude': df['latitude_gps'].values,
        'localtime': localtime}

# datadir
datadir = r'/mnt/home/russell.burdt/data/eye-witness/app'
assert os.path.isdir(datadir)
if glob(os.path.join(datadir, '*')):
    rmtree(datadir)
    assert not os.path.isdir(datadir)
    os.mkdir(datadir)

# interface objects
lat, lon = 33.3720, -86.8321
delta = 0.010
lon0, lon1 = lon - delta, lon + delta
lat0, lat1 = lat - delta, lat + delta
gps = MapInterface(width=900, height=500, hover=True, size=16, lon0=lon0, lon1=lon1, lat0=lat0, lat1=lat1)
gps.hover.renderers = [x for x in gps.fig.renderers if x.name == 'path']
gps.hover.tooltips = [
    ('latitude', '@latitude{%.4f}'),
    ('longitude', '@longitude{%.4f}'),
    ('localtime', '@localtime{%d %b %Y %H:%M:%S}')]
gps.hover.formatters = {'@latitude': 'printf', '@longitude': 'printf', '@localtime': 'datetime'}
update = Button(label='Run eye-witness', button_type='success', width=300)
update.on_click(app_callback)
select = Select(title='select vehicle-id')
c0, c1 = console_str_objects(400)
cs = f"""
    eye-witness time window<br>
    {time0.strftime('%m/%d/%Y %H:%M:%S')} to {time1.strftime('%m/%d/%Y %H:%M:%S')} {tz}<br>"""
status_all = Div(text=c0 + cs + 'set eye-witness location window on map' + c1)
status_vehicle = Div(text=c0 + c1)

# app layout and document object
layout = row(gps.fig, column(status_all, update, select, status_vehicle))
doc = curdoc()
doc.add_root(layout)
doc.title = 'eye-witness locate app'
