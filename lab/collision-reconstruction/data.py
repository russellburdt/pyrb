
"""
data for Lytx-vehicle eye-witness application
- based on collions for a population of vehicles in a previous time window wrt datetime.utcnow
"""

import os
import lytx
import boto3
import numpy as np
import pandas as pd
import sqlalchemy as sa
import pytz
from glob import glob
from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType
from collections import defaultdict
from shutil import rmtree
from datetime import datetime
from tqdm import tqdm
from ipdb import set_trace

# datadir
datadir = r'/mnt/home/russell.burdt/data/collision-reconstruction/app'
if os.path.isdir(datadir):
    rmtree(datadir)
os.mkdir(datadir)

# spark session
conf = SparkConf()
conf.set('spark.sql.execution.arrow.pyspark.enabled', 'true')
spark = SparkSession.builder.config(conf=conf).getOrCreate()

# vehicle metadata, active vehicles between t0 and t1
days = 20
edw = lytx.get_conn('edw')
# t1 = datetime.strptime(datetime.utcnow().strftime('%m/%d/%Y %H:%M'), '%m/%d/%Y %H:%M')
# t0 = t1 - pd.Timedelta(days=days)
t0 = datetime.strptime('10/15/2023', '%m/%d/%Y')
t1 = datetime.strptime('10/22/2023', '%m/%d/%Y')
query = f"""
    SELECT
        ERA.EventRecorderId, ERA.VehicleId, ERA.CreationDate, ERA.DeletedDate, ERA.GroupId, ER.Model,
        G.Name as GroupName, C.CompanyId, C.CompanyName, C.IndustryDesc, D.SerialNumber
    FROM hs.EventRecorderAssociations AS ERA
        LEFT JOIN flat.Groups AS G ON ERA.GroupId = G.GroupId
        LEFT JOIN flat.Companies AS C ON G.CompanyId = C.CompanyId
        LEFT JOIN flat.Devices AS D ON D.DeviceId = ERA.EventRecorderId
        LEFT JOIN hs.EventRecorders AS ER ON ER.Id = ERA.EventRecorderId
    WHERE ERA.VehicleId <> '00000000-0000-0000-0000-000000000000'
    AND ERA.CreationDate < '{t0}'
    AND ERA.DeletedDate > '{t1}'
    AND ER.Model IN ('ER-SF300','ER-SF300V2','ER-SF64','ER-SF400')"""
dm = pd.read_sql_query(sa.text(query), edw)
assert pd.unique(dm['VehicleId']).size == pd.unique(dm['SerialNumber']).size == pd.unique(dm['EventRecorderId']).size == dm.shape[0]
dm['t0'], dm['t1'] = t0, t1
dm.to_pickle(os.path.join(datadir, 'population.p'))

# collision metadata for active vehicles between t0 and t1
query = f'WITH ERS AS (' + query + ')'
query += f"""
    SELECT B.RecordDate, B.Latitude, B.Longitude, B.EventId, B.EventRecorderId, value AS BehaviorId,
        B.SpeedAtTrigger, B.EventFilePath, B.EventFileName, HSB.Name AS BehaviorName, ERS.*
    FROM flat.Events AS B
        CROSS APPLY STRING_SPLIT(COALESCE(B.BehaviourStringIds, '-1'), ',')
        INNER JOIN ERS ON ERS.VehicleId = B.VehicleId
        LEFT JOIN hs.Behaviors_i18n AS HSB ON value = HSB.Id
    WHERE B.Deleted = 0
    AND B.RecordDate BETWEEN '{t0}' AND '{t1}'
    AND value = 47
    AND NOT ((B.Latitude = 0) AND (B.Longitude = 0))"""
dx = pd.read_sql_query(sa.text(query), edw)
assert dx['RecordDate'].max() < t1
assert len(set(dx['VehicleId'].values).intersection(dm['VehicleId'].values)) == pd.unique(dx['VehicleId']).size
dx = dx.sort_values('RecordDate').reset_index(drop=True)
dx = dx.loc[(~pd.isnull(dx['EventFilePath'])) & (~pd.isnull(dx['EventFileName']))].reset_index(drop=True)
keys = [f"""dce-files/{os.sep.join(path[15:].split(chr(92)))}/{name}""" for path, name in zip(dx['EventFilePath'], dx['EventFileName'])]
responses = [boto3.client('s3').list_objects_v2(Bucket='lytx-amlnas-us-west-2', Prefix=key) for key in tqdm(keys, desc='dce-files')]
ok = [('Contents' in response.keys()) and (len(response['Contents']) == 1) for response in responses]
dx = dx.loc[ok].reset_index(drop=True)

# geocode via Dennis Cheng enrichment function, include localhour and weekday
gc = []
conn = lytx.get_conn('unified-map')
for x, cx in tqdm(dx.iterrows(), desc='geocode collisions', total=dx.shape[0]):
    sql = f"""SELECT * FROM osm230807.lytxlabs_getlocationinfo_unifiedmap({cx['Longitude']}, {cx['Latitude']})"""
    dc = pd.read_sql_query(sql, conn)
    assert dc.shape[0] == 1
    dc = dc.squeeze().to_dict()
    dc['index'] = x
    assert dc['timezone_locality_tzid'] in pytz.all_timezones
    dc['localhour'] = cx['RecordDate'].tz_localize('UTC').astimezone(dc['timezone_locality_tzid']).tz_localize(None).hour
    dc['weekday'] = cx['RecordDate'].tz_localize('UTC').astimezone(dc['timezone_locality_tzid']).tz_localize(None).weekday()
    gc.append(dc)
gc = pd.DataFrame(gc)
gc.index = gc.pop('index')
assert len(set(gc.columns).intersection(dx.columns)) == 0
dx = pd.merge(dx, gc, left_index=True, right_index=True, how='inner')
dx.to_pickle(os.path.join(datadir, 'collisions.p'))

# collision videos
cdir = os.path.join(datadir, 'collision-videos')
os.mkdir(cdir)
for x, row in tqdm(dx.iterrows(), desc='collision videos', total=dx.shape[0]):
    fn = os.path.join(cdir, f'collision-{x:04d}.mkv')
    lytx.extract_and_save_video(row, fn, assert_exists=True, keep_dce=True)
assert len(glob(os.path.join(cdir, '*'))) == 2 * dx.shape[0]

# search parameters for nearby vehicles, td in sec, xd in meters
def nearby_vehicles(pdf):

    # snowflake connection, collision record
    snow = lytx.get_conn('snowflake')
    assert pdf.shape[0] == 1
    row = dx.loc[pdf.iloc[0]['id']]

    # GPS data for nearby vehicles
    t0 = int((row['RecordDate'] - pd.Timestamp(datetime(1970, 1, 1))).total_seconds())
    query = f"""
        SELECT
            UPPER(VEHICLE_ID) AS vehicleid, ts_sec, latitude, longitude, speed, heading, hdop, serial_number,
            ST_DISTANCE(ST_POINT({row['Longitude']}, {row['Latitude']}), ST_POINT(longitude, latitude)) AS distance
        FROM GPS.GPS_ENRICHED
        WHERE UPPER(VEHICLE_ID) <> '00000000-0000-0000-0000-000000000000'
        AND UPPER(VEHICLE_ID) <> '{row['VehicleId']}'
        AND TS_SEC BETWEEN {t0 - td} AND {t0 + td}
        AND ST_DISTANCE(ST_POINT({row['Longitude']}, {row['Latitude']}), ST_POINT(longitude, latitude)) < {xd}"""
    df = pd.read_sql_query(query, snow)

    # validate and join with dm
    if df.size > 0:
        assert df.groupby('vehicleid')['serial_number'].nunique().max() == 1
        assert df.groupby('vehicleid')['distance'].max().max() < xd
        dfm = pd.merge(left_index=True, right_index=True, how='inner',
            left=df.groupby('vehicleid')['serial_number'].first(),
            right=df.groupby('vehicleid')['ts_sec'].count().rename('count'))
        dfm = pd.merge(left=dfm, left_index=True, right_index=True, how='inner',
            right=df.groupby('vehicleid')['distance'].max().rename('max_d'))
        def func(xx):
            return np.abs(xx['ts_sec'].values - t0).max()
        dfm = pd.merge(left=dfm, left_index=True, right_index=True, how='inner',
            right = df.groupby('vehicleid').apply(func).rename('max_t')).reset_index(drop=False)
        dfx = pd.merge(left=dfm, right=dm[['VehicleId', 'SerialNumber']], left_on='vehicleid', right_on='VehicleId', how='left').drop('vehicleid', axis=1)
        assert dfm.shape[0] == dfx.shape[0]
        dfx = dfx.loc[~np.any(pd.isnull(dfx), axis=1)].reset_index(drop=True)
        assert all(dfx['serial_number'] == dfx['SerialNumber'])
        del dfx['serial_number']
        dfx['id'] = pdf.iloc[0]['id']
        return dfx

    # null case
    return pd.DataFrame(columns=['count', 'max_d', 'max_t', 'VehicleId', 'SerialNumber', 'id'])
sdf = spark.range(start=0, end=dx.shape[0], step=1, numPartitions=dx.shape[0])
schema = StructType([
    StructField('count', IntegerType(), nullable=False),
    StructField('max_d', DoubleType(), nullable=False),
    StructField('max_t', DoubleType(), nullable=False),
    StructField('VehicleId', StringType(), nullable=False),
    StructField('SerialNumber', StringType(), nullable=False),
    StructField('id', IntegerType(), nullable=False)])
config = (
    (3600, 40),     # +/- 1 hour, 40 meters
    (1800, 40),     # +/- 30 minute, 40 meters
    (300, 40),      # +/- 5 minute, 40 meters
    (60, 40),       # +/- 1 minute, 40 meters
    (30, 40),       # +/- 30 sec, 40 meters
    (10, 40))       # +/- 10 sec, 40 meters

# metadata for nearby vehicles
for td, xd in config:

    # DataFrame of nearby vehicles to collisions
    print(f'nearby vehicles, td={td}, xd={xd}')
    pdf = sdf.groupby('id').applyInPandas(nearby_vehicles, schema=schema).toPandas()
    # xdf = sdf.toPandas()
    # nearby_vehicles(xdf.loc[xdf.index == 0])

    # validate and save
    if pdf.size > 0:
        assert pdf.duplicated().sum() == 0
        assert len(set(pdf['VehicleId']).intersection(dm['VehicleId'])) == pd.unique(pdf['VehicleId']).size
        pdf['td'], pdf['xd'] = td, xd
        assert all(pdf['max_d'] <= xd) and all(pdf['max_t'] <= td)
        pdf = pd.merge(left=pdf, right=dm[['VehicleId', 'SerialNumber', 'CompanyName', 'CompanyId', 'IndustryDesc']], on=['VehicleId', 'SerialNumber'], how='left')
        pdf.to_pickle(os.path.join(datadir, f'nearby_vehicles_td{td}_xd{xd}.p'))

# GPS raw data parameters for nearby vehicles
gps = pd.DataFrame()
for td, xd in config:

    # metadata for nearby vehicles, gps raw data parameters
    print(f'GPS data for nearby vehicles, td={td}, xd={xd}')
    df = pd.read_pickle(os.path.join(datadir, f'nearby_vehicles_td{td}_xd{xd}.p'))
    for xx in pd.unique(df['id']):

        # time-window based on collision RecordDate and td
        ta = dx.loc[xx, 'RecordDate'] - pd.Timedelta(seconds=td)
        tb = dx.loc[xx, 'RecordDate'] + pd.Timedelta(seconds=td)

        # update gps parameters DataFrame
        vids = pd.unique(df.loc[df['id'] == xx, 'VehicleId'])
        assert dx.loc[xx, 'VehicleId'] not in vids
        vids = np.hstack((dx.loc[xx, 'VehicleId'], vids))
        gps = pd.concat((gps,
            pd.DataFrame(data={'VehicleId': vids, 'ta': np.full(vids.size, ta), 'tb': np.full(vids.size, tb)})))
gps = pd.merge(left_index=True, right_index=True, how='inner',
    left=gps.groupby('VehicleId')['ta'].min(),
    right=gps.groupby('VehicleId')['tb'].max()).reset_index(drop=False)

# distributed gps data extraction
def gps_data(pdf):

    # snowflake connection, collision record
    snow = lytx.get_conn('snowflake')
    assert pdf.shape[0] == 1
    row = gps.loc[pdf.iloc[0]['id']]

    # GPS data for vehicle
    ta = int((row['ta'] - pd.Timestamp(datetime(1970, 1, 1))).total_seconds())
    tb = int((row['tb'] - pd.Timestamp(datetime(1970, 1, 1))).total_seconds())
    query = f"""
        SELECT UPPER(vehicle_id) AS vehicleid, ts_sec, latitude, longitude, speed, heading, hdop, serial_number
        FROM GPS.GPS_ENRICHED
        WHERE UPPER(vehicle_id) = '{row['VehicleId']}'
        AND TS_SEC BETWEEN {ta} AND {tb}"""
    df = pd.read_sql_query(query, snow)
    df = df.rename(columns={'vehicleid': 'VehicleId'})
    df.to_parquet(path=os.path.join(datadir, 'gps.parquet'), engine='pyarrow', compression='snappy', index=False, partition_cols=['VehicleId'])
    return pdf
sdf = spark.range(start=0, end=gps.shape[0], step=1, numPartitions=gps.shape[0])
pdf = sdf.groupby('id').applyInPandas(gps_data, schema=sdf.schema).toPandas()
# xdf = sdf.toPandas()
# gps_data(xdf.loc[xdf.index == 0])

# merge parquet files
lytx.merge_parquet(spark=spark, loc=os.path.join(datadir, 'gps.parquet'), xid='VehicleId', ts='ts_sec')
