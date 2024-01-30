
"""
enrich gps data by device serial number and time-window
"""

import os
import lytx
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import sqlalchemy as sa
from glob import glob
from datetime import datetime
from functools import partial
from shutil import rmtree
from pyspark import SparkConf
from pyspark.sql import SparkSession
from bokeh.tile_providers import Vendors, get_provider
from bokeh.io import curdoc
from bokeh.models import MultiSelect, Div, TileRenderer, RadioButtonGroup
from bokeh.layouts import column, row
from pyrb.bokeh import MapInterface, MultiLineInterface
from pyrb.processing import webm_to_gps, gps_to_webm
from geopandas.array import from_wkb
from ipdb import set_trace


# device and time-window
# device = 'MV00134446'
device = 'MV01000865'
# device = 'QM00020074'
time0 = pd.Timestamp('4/20/2023')
time1 = pd.Timestamp('4/25/2023')

# set and clear datadir
datadir = r'/mnt/home/russell.burdt/data/driver-safety/tmp'
assert os.path.isdir(datadir)
if glob(os.path.join(datadir, '*')):
    rmtree(datadir)
    assert not os.path.isdir(datadir)
    os.mkdir(datadir)

# event-recorder association during time-window
edw = lytx.get_conn('edw')
dm = pd.read_sql_query(sa.text(f"""
    SELECT
        ERA.VehicleId, ERA.CreationDate, ERA.DeletedDate, ERS.Description AS Status, D.SerialNumber,
        G.Name as GroupName, C.CompanyId, C.CompanyName, C.IndustryDesc, D.Model
    FROM hs.EventRecorderAssociations AS ERA
        LEFT JOIN flat.Devices AS D ON ERA.EventRecorderId = D.DeviceId
        LEFT JOIN flat.Groups AS G ON ERA.GroupId = G.GroupId
        LEFT JOIN flat.Companies AS C ON G.CompanyId = C.CompanyId
        LEFT JOIN hs.EventRecorderStatuses_i18n AS ERS ON ERA.EventRecorderStatusId = ERS.Id
    WHERE D.SerialNumber='{device}'
    AND ERA.CreationDate < '{time0.strftime('%m/%d/%Y')}'
    AND ERA.DeletedDate > '{time1.strftime('%m/%d/%Y')}'
    AND ERA.VehicleId <> '00000000-0000-0000-0000-000000000000'
    AND ERS.LocaleId = 9"""), edw)
assert dm.shape[0] == 1
dm = dm.squeeze()
dm['time0'] = time0
dm['time1'] = time1
dm.to_pickle(os.path.join(datadir, 'dm.p'))

# raw gps data
vids = np.array([dm['VehicleId']])
dcm = pd.DataFrame({'VehicleId': vids,'time0': [time0], 'time1': [time1]})
lytx.distributed_data_extraction(dataset='gps', datadir=datadir, df=dcm, xid='VehicleId', n=200, distributed=False, assert_new=True)
assert glob(os.path.join(datadir, 'gps.parquet', '*'))

# Spark Session object
conf = SparkConf()
conf.set('spark.sql.execution.arrow.pyspark.enabled', 'true')
conf.set('spark.sql.parquet.enableVectorizedReader', 'false')
conf.set('spark.sql.session.timeZone', 'UTC')
conf.set('spark.local.dir', r'/mnt/home/russell.burdt/rbin')
spark = SparkSession.builder.config(conf=conf).getOrCreate()
spark.sparkContext.setLogLevel('ERROR')

# read gps data and assert consistent schema
loc = os.path.join(datadir, f'gps.parquet')
gps = spark.read.parquet(loc)
gps.createOrReplaceTempView('gps')
lytx.validate_consistent_parquet_schema(spark, loc=loc, src='gps', xid='VehicleId')
gc = gps.count()
df = gps.toPandas().sort_values('TS_SEC').reset_index(drop=True)
assert datetime.utcfromtimestamp(df.iloc[0]['TS_SEC']) > time0
assert datetime.utcfromtimestamp(df.iloc[-1]['TS_SEC']) < time1

# enrich gps data
lytx.gps_segmentation(spark=spark, datadir=datadir, src='gps.parquet', dst='gps1.parquet', service='EC2', time_interval_sec=61, distance_interval_miles=1.0)
gps = spark.read.parquet(os.path.join(datadir, 'gps1.parquet'))
gps.createOrReplaceTempView('gps')
assert gps.count() == gc
ds = lytx.gps_segmentation_metrics(dcm, spark).squeeze()
assert ds['n_segments'] == 1
ds.to_pickle(os.path.join(datadir, 'ds.p'))
lytx.gps_interval_metrics(spark=spark, datadir=datadir, src='gps1.parquet', dst='gps2.parquet', service='EC2')
gps = spark.read.parquet(os.path.join(datadir, 'gps2.parquet'))
gps.createOrReplaceTempView('gps')
assert gps.count() == gc
lytx.gps_enrich_dc_normalized(spark=spark, datadir=datadir, src='gps2.parquet', dst='gpse.parquet', service='EC2')
gpse = spark.read.parquet(os.path.join(datadir, 'gpse.parquet'))
gpse.createOrReplaceTempView('gpse')
gpsm = spark.read.parquet(os.path.join(datadir, 'gpsm.parquet'))
gpsm.createOrReplaceTempView('gpsm')
assert gpse.count() > gpsm.count()

# load gps data
df = gps.toPandas()
dfe = gpse.toPandas()
dfm = gpsm.toPandas()

def maneuvers_multiselect_callback(attr, old, new):
    """
    update GPS maneuvers data source based on multiselect value
    """

    # null case
    if not new:
        gpse.segments.data = {'lat': np.array([]), 'lon': np.array([])}
        return

    # build lat/lon data from selected maneuvers, update source
    lat = np.array([])
    lon = np.array([])
    for mx in new:
        ok = dfm['segmentgroup__maneuver'] == mx
        assert ok.sum() > 0
        for x0, x1 in zip(dfm.loc[ok, 'sg_start'].values, dfm.loc[ok, 'sg_end'].values):
            lat = np.hstack((lat, dfe.loc[x0 : x1, 'latitude'].values, np.nan))
            lon = np.hstack((lon, dfe.loc[x0 : x1, 'longitude'].values, np.nan))
    gpse.segments.data = {'lat': lat[:-1], 'lon': lon[:-1]}

def tiles_callback(attr, old, new):

    # update gps tile source
    assert isinstance(gps.fig.renderers[0], TileRenderer)
    gps.fig.renderers.remove(gps.fig.renderers[0])
    gps.fig.renderers.insert(0, TileRenderer(tile_source=get_provider(getattr(Vendors, tiles.labels[tiles.active]))))

    # update gpse tile source
    assert isinstance(gpse.fig.renderers[0], TileRenderer)
    gpse.fig.renderers.remove(gpse.fig.renderers[0])
    gpse.fig.renderers.insert(0, TileRenderer(tile_source=get_provider(getattr(Vendors, tiles.labels[tiles.active]))))

# app metadata
text = f"""<strong>Device {dm['SerialNumber']}, {dm['CompanyName']}, {time0.strftime('%m/%d/%Y %H:%M')} to {time1.strftime('%m/%d/%Y %H:%M')}</strong>"""
da = Div(text=text, style={'font-size': '14pt'})

# gps map interface
gps = MapInterface(width=700, height=400, hover=True)
gps.hover.renderers = [x for x in gps.fig.renderers if x.name == 'path']
gps.hover.tooltips = [
    ('latitude', '@latitude{%.4f}'),
    ('longitude', '@longitude{%.4f}'),
    ('utc', '@utc{%d %b %Y %H:%M:%S}')]
gps.hover.formatters = {'@latitude': 'printf', '@longitude': 'printf', '@utc': 'datetime'}
gps.fig.title.text = 'Raw GPS Data'

# gpse map interface
gpse = MapInterface(width=700, height=400, hover=True)
gpse.fig.x_range = gps.fig.x_range
gpse.fig.y_range = gps.fig.y_range
gpse.hover.renderers = [x for x in gpse.fig.renderers if x.name == 'path']
gpse.hover.tooltips = [
    ('segment__id', '@segment__id{%.0f}'),
    ('latitude', '@latitude{%.4f}'),
    ('longitude', '@longitude{%.4f}'),
    ('utc', '@utc{%d %b %Y %H:%M:%S}')]
gpse.hover.formatters = {'@segment__id': 'printf', '@latitude': 'printf', '@longitude': 'printf', '@utc': 'datetime'}
gpse.fig.title.text = 'Distance-Normalized GPS Data'

# tile provider selection
tiles = RadioButtonGroup(labels=['OSM', 'ESRI_IMAGERY'], active=0, width=200)
tiles.on_change('active', tiles_callback)

# maneuver selection interface
maneuvers = MultiSelect(title='GPS Maneuver(s)', width=200, height=300)
maneuvers.on_change('value', maneuvers_multiselect_callback)

# update app data sources
lon = df['longitude'].values
lat = df['latitude'].values
gps.path.data = {'lon': lon, 'lat': lat, 'utc': df['utc'], 'longitude': df['longitude_gps'], 'latitude': df['latitude_gps']}
gps.reset_map_view(lon0=np.nanmin(lon), lon1=np.nanmax(lon), lat0=np.nanmin(lat), lat1=np.nanmax(lat), convert=False)
gpse.path.data = {'lon': dfe['longitude'], 'lat': dfe['latitude'], 'longitude': dfe['longitude_gps'], 'latitude': dfe['latitude_gps'], 'utc': dfe['utc'], 'segment__id': dfe['segment__id']}
maneuvers.options = sorted(pd.unique(dfm['segmentgroup__maneuver']))

# app layout and document object
layout = column(da, row(gps.fig, gpse.fig, maneuvers), tiles)
doc = curdoc()
doc.add_root(layout)
doc.title = 'device gpse app'

