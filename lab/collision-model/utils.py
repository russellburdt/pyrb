
"""
collision prediction model utils
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow as pa
import pyarrow.parquet as pq
import pickle
import sqlalchemy as sa
from itertools import chain
from collections import defaultdict
from shutil import rmtree
from datetime import datetime, timedelta
from pyproj import Transformer, Geod
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import pandas_udf, broadcast
from pyspark.sql.types import StructType, StructField, DoubleType
from typing import Iterator
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from pyrb.mpl import open_figure, format_axes, largefonts, save_pngs
from pyrb.processing import get_folder_size, int_to_ordinal
from lytx import get_conn
from time import sleep
from glob import glob
from tqdm import tqdm
from ipdb import set_trace
plt.style.use('bmh')


# refactored Q4/23
def collision_prediction_model_windows(time0, time1, predictor_interval_days, collision_interval_days, overlap_days):
    """
    time-windows for collision prediction model as a pandas DataFrame
    """

    # validate and initialize cpm DataFrame
    assert isinstance(time0, pd.Timestamp) and isinstance(time1, pd.Timestamp)
    assert isinstance(predictor_interval_days, int)
    assert isinstance(collision_interval_days, (int, str))
    if isinstance(collision_interval_days, int):
        assert isinstance(overlap_days, int) or (overlap_days is None)
    if isinstance(collision_interval_days, str):
        assert isinstance(overlap_days, str)
        assert (collision_interval_days in ('calendar-month'))
        assert collision_interval_days == overlap_days
    cpm = defaultdict(list)

    # int collision-interval
    if isinstance(collision_interval_days, int):

        # first predictor and collision interval
        cpm['ta'].append(time0)
        cpm['tb'].append(time0 + pd.Timedelta(days=predictor_interval_days))
        cpm['tc'].append(time0 + pd.Timedelta(days=predictor_interval_days + collision_interval_days))
        assert cpm['tc'][-1] < time1

        # subsequent predictor and collision intervals
        while cpm['tc'][-1] <= time1:

            # start of predictor interval based on cases for overlap
            tx = cpm['ta'][-1] + pd.Timedelta(days=overlap_days) if isinstance(overlap_days, int) else cpm['tb'][-1]
            cpm['ta'].append(tx)
            cpm['tb'].append(tx + pd.Timedelta(days=predictor_interval_days))
            cpm['tc'].append(tx + pd.Timedelta(days=predictor_interval_days + collision_interval_days))

    # str collision interval
    if isinstance(collision_interval_days, str):

        # collision-intervals as calendar-months
        if collision_interval_days == 'calendar-month':

            # get 1st day of subsequent calendar month based on timestamp
            gcm = lambda tx: \
                pd.Timestamp(year=tx.year, month=tx.month + 1, day=1) if tx.month < 12 else pd.Timestamp(year=tx.year + 1, month=1, day=1)

            # first predictor and collision interval
            tx = time0 + pd.Timedelta(days=predictor_interval_days)
            cpm['tb'].append(gcm(tx))
            cpm['ta'].append(cpm['tb'][-1] - pd.Timedelta(days=predictor_interval_days))
            assert cpm['ta'][-1] > time0
            cpm['tc'].append(gcm(gcm(tx)))
            assert cpm['tc'][-1] < time1

            # subsequent predictor and collision intervals
            while cpm['tc'][-1] <= time1:
                tx = cpm['tc'][-1]
                cpm['tb'].append(tx)
                cpm['ta'].append(tx - pd.Timedelta(days=predictor_interval_days))
                cpm['tc'].append(gcm(tx))

    # remove last window and return
    return pd.DataFrame(cpm)[['ta', 'tb', 'tc']].sort_values('ta').reset_index(drop=True).iloc[:-1]

def collision_prediction_model_dataframe(dx, dt):
    """
    collision prediction model DataFrame based on event-recorder-associations as dx and time-windows as dt
    """

    # initialize dcm, scan over intervals
    dcm = pd.DataFrame()
    for _, row in dt.iterrows():
        ta, tb, tc = row['ta'], row['tb'], row['tc']
        dv = dx.loc[(dx['CreationDate'] < ta) & (dx['DeletedDate'] > tc)].copy()
        assert pd.unique(dv['VehicleId']).size == dv.shape[0]
        dv['ta'], dv['tb'], dv['tc'] = ta, tb, tc
        dcm = pd.concat((dcm, dv))

    return dcm.sort_values(['VehicleId', 'ta']).reset_index(drop=True)

# before Q4/23 refactor
def model_metadata(population, collision_intervals, devices, predictor_interval_days):
    """
    collision prediction model metadata, returns
    dc - population metadata Series
    desc - population str description
    dv - vehicle metadata DataFrame
    dp - collision metadata
    de - event recorder association DataFrame
    dem - event recorder association metrics DataFrame
    args
    population, eg lytx.get_population('amt')
    collision_intervals, eg ['11/2021', '12/2021']
    devices, eg ['ER-SF300', 'ER-SF64']
    predictor_interval_days, eg 90
    """
    import reverse_geocoder
    from timezonefinder import TimezoneFinder

    # population metadata Series
    dc = pd.Series({'desc': population['desc']})
    for key, value in population.items():
        if key == 'desc':
            continue
        dc[key] = ','.join(["""'{}'""".format(x) for x in [x.replace("""'""", """''""") for x in value]])
    dts = [datetime.strptime(x, r'%m/%Y') for x in collision_intervals]
    for year in np.sort(np.unique([x.year for x in dts])):
        xs = np.sort([x for x in dts if x.year == year])
        dc[f'collision intervals {year}'] = ','.join([x.strftime('%b') for x in xs])
    dc['predictor interval days'] = str(predictor_interval_days)
    dc['devices'] = ','.join(devices)

    # population str description
    desc = dc['desc']
    for key, value in dc.items():
        if key in population.keys():
            continue
        desc += f"""\n{key} - {value}"""

    # vehicle metadata DataFrame
    edw = get_conn('edw')
    now = datetime.now()
    query = f"""
        SELECT D.VehicleId, C.CompanyName, C.IndustryDesc, C.CompanyId"""
    query += f"""
        FROM flat.Companies AS C
        LEFT JOIN flat.Devices AS D
        ON C.CompanyId = D.CompanyId
        WHERE C.CompanyName <> 'DriveCam DC4DC Test Co'
        AND D.VehicleId <> '00000000-0000-0000-0000-000000000000'"""
    for field, xf in population.items():
        if field == 'desc':
            continue
        sf = ','.join(["""'{}'""".format(x) for x in [x.replace("""'""", """''""") for x in xf]])
        query += f"""\nAND {field} IN ({sf})"""
    if ('CompanyName' in population) and ('DriveCam DC4DC Test Co' in population['CompanyName']):
        query = query.replace(
            """WHERE C.CompanyName <> 'DriveCam DC4DC Test Co'\n        AND D.VehicleId <> '00000000-0000-0000-0000-000000000000'""",
            """WHERE D.VehicleId <> '00000000-0000-0000-0000-000000000000'""")
    dv = pd.read_sql_query(sa.text(query), edw).drop_duplicates().sort_values('VehicleId').reset_index(drop=True)
    assert pd.unique(dv['VehicleId']).size == dv.shape[0]
    print(f'query vehicle metadata, {(datetime.now() - now).total_seconds():.1f}sec')

    # query positive instances over full range of all collision intervals, for vehicles from metadata query
    now = datetime.now()
    dts = np.sort([datetime.strptime(x, r'%m/%Y') for x in collision_intervals])
    t0, t1 = dts[0], dts[-1]
    assert (t0.day == 1) and (t1.day == 1)
    t1 = datetime.strptime((t1 + timedelta(days=32)).strftime('%m/%Y'), '%m/%Y')
    assert t1.day == 1
    query1 = f"""
        WITH V AS ({query})
        SELECT
            E.VehicleId,
            E.RecordDate,
            E.Latitude,
            E.Longitude,
            E.EventId,
            E.CustomerEventIdString,
            value AS BehaviorId
        FROM flat.Events AS E
            CROSS APPLY STRING_SPLIT(COALESCE(E.BehaviourStringIds, '-1'), ',')
            INNER JOIN V ON V.VehicleId = E.VehicleId
        WHERE E.RecordDate > '{t0.strftime('%m-%d-%Y %H:%M:%S')}'
        AND E.RecordDate < '{t1.strftime('%m-%d-%Y %H:%M:%S')}'
        AND E.VehicleId <> '00000000-0000-0000-0000-000000000000'
        AND E.Deleted=0
        AND value IN (45,46,47)"""
    dp = pd.read_sql_query(sa.text(query1), edw).drop_duplicates().sort_values(['VehicleId', 'RecordDate']).reset_index(drop=True)
    if dp['RecordDate'].dtype.type == np.object_:
        dp['RecordDate'] = [pd.Timestamp(x) for x in dp['RecordDate']]
    else:
        assert dp['RecordDate'].dtype.type == np.datetime64
    dp = dp.groupby(['VehicleId', 'RecordDate']).first().reset_index(drop=False)
    assert dp.groupby(['VehicleId', 'RecordDate']).ngroups == dp.shape[0]
    dp['BehaviorId'] = dp['BehaviorId'].astype('int')
    if dp.size == 0:
        dp['RecordDate'] = dp['RecordDate'].astype('datetime64[ns]')
    print(f'query collisions, {(datetime.now() - now).total_seconds():.1f}sec')

    # query ER assocations, for vehicles from metadata query
    now = datetime.now()
    devices_str = ','.join(["""'{}'""".format(x) for x in devices])
    query2 = f"""
        WITH V AS ({query})
        SELECT
            ERA.VehicleId,
            ERA.EventRecorderId,
            ER.Model,
            ERA.CreationDate,
            ERA.DeletedDate,
            ERA.GroupId,
            G.Name AS GroupName
        FROM hs.EventRecorderAssociations AS ERA
            INNER JOIN V
            ON V.VehicleId = ERA.VehicleId
            LEFT JOIN hs.EventRecorders AS ER
            ON ER.Id = ERA.EventRecorderId
            LEFT JOIN hs.EventRecorderStatuses_i18n AS ERS
            ON ERS.Id = ERA.EventRecorderStatusId
            LEFT JOIN hs.Groups AS G
            ON G.Id = ERA.GroupId
        WHERE ER.Model IN ({devices_str})
        AND ERS.Description = 'In Service'
        AND ERS.LocaleId = 9"""
    de = pd.read_sql_query(sa.text(query2), edw).drop_duplicates().sort_values(['VehicleId', 'CreationDate']).reset_index(drop=True)
    print(f'query event recorder associations, {(datetime.now() - now).total_seconds():.1f}sec')

    # convert CreationDate
    if de['CreationDate'].dtype.type == np.object_:
        de['CreationDate'] = [pd.Timestamp(x) for x in de['CreationDate'].values]
    else:
        assert de['CreationDate'].dtype.type == np.datetime64

    # convert DeletedDate
    # max time based on https://stackoverflow.com/questions/32888124/pandas-out-of-bounds-nanosecond-timestamp-after-offset-rollforward-plus-adding-a
    assert de['DeletedDate'].dtype.type == np.object_
    if isinstance(de['DeletedDate'].loc[0], datetime):
        de['DeletedDate'] = [pd.Timestamp(x) if x.strftime(r'%Y-%m-%d') != '9999-01-01' else pd.Timestamp('2262-04-11 23:47:16.854775807') for x in de['DeletedDate'].values]
    else:
        de['DeletedDate'] = [pd.Timestamp(x) if x[:10] != '9999-01-01' else pd.Timestamp('2262-04-11 23:47:16.854775807') for x in de['DeletedDate'].values]

    # filter event recorder associations with zero or negative duration
    ok = (de['DeletedDate'].values - de['CreationDate'].values).astype('float') > 0
    de = de.loc[ok].reset_index(drop=True)

    # filter and modify bounds of event recorder associations to be within model parameters
    tmin = pd.Timestamp(t0) - pd.Timedelta(days=int(predictor_interval_days))
    tmax = pd.Timestamp(t1)
    de = de.loc[(de['DeletedDate'] > tmin) & (de['CreationDate'] < tmax)].reset_index(drop=True)
    assert np.all((de['DeletedDate'].values - de['CreationDate'].values).astype('float') > 0)

    # event recorder association metrics by vehicle-id
    now = datetime.now()
    def er_metrics(dx):
        x0 = (1e-9) * (dx['DeletedDate'].values - dx['CreationDate'].values).astype('float')
        if dx.shape[0] == 1:
            return pd.Series({
                'duration of ER associations in sec': x0,
                'sec between consecutive ER associations without metadata change': np.array([]),
                'sec between consecutive ER associations with metadata change': np.array([])})
        x1 = (1e-9) * (dx['CreationDate'].values[1:] - dx['DeletedDate'].values[:-1]).astype('float')
        x2 = np.logical_or(
            dx['GroupId'].values[1:] != dx['GroupId'].values[:-1],
            dx['EventRecorderId'].values[1:] != dx['EventRecorderId'].values[:-1])
        return pd.Series({
            'duration of ER associations in sec': x0,
            'sec between consecutive ER associations without metadata change': x1[~x2],
            'sec between consecutive ER associations with metadata change': x1[x2]})
    dem = de.groupby('VehicleId').apply(er_metrics).reset_index(drop=False)
    assert np.all(np.concatenate(dem['duration of ER associations in sec'].values) > 0)
    print(f'event recorder association metrics by vehicle-id, {(datetime.now() - now).total_seconds():.1f}sec')

    # filter vehicles in ER association DataFrames with any negative time between consecutive event recorder associations
    nok = np.hstack((
        dem.loc[np.where([np.any(x < 0) for x in dem['sec between consecutive ER associations without metadata change'].values])[0], 'VehicleId'].values,
        dem.loc[np.where([np.any(x < 0) for x in dem['sec between consecutive ER associations with metadata change'].values])[0], 'VehicleId'].values))
    vt = pd.unique(de['VehicleId'])
    print(f'{100 * nok.size / vt.size:.4f}% of vehicles with negative time between consecutive ER associations')
    assert pd.unique(de['VehicleId']).size == pd.unique(dem['VehicleId']).size
    de = de.loc[~de['VehicleId'].isin(nok)].reset_index(drop=True)
    dem = dem.loc[~dem['VehicleId'].isin(nok)].reset_index(drop=True)
    assert np.all(np.concatenate(dem['duration of ER associations in sec'].values) > 0)
    assert np.all(np.concatenate(dem['sec between consecutive ER associations without metadata change'].values) >= 0)
    assert np.all(np.concatenate(dem['sec between consecutive ER associations with metadata change'].values) >= 0)

    # filter DataFrames based on common set of vehicles
    vids = np.array(list(set(dv['VehicleId'].values).intersection(set(de['VehicleId'].values))))
    dv = dv.loc[dv['VehicleId'].isin(vids)].reset_index(drop=True)
    de = de.loc[de['VehicleId'].isin(vids)].reset_index(drop=True)
    dem = dem.loc[dem['VehicleId'].isin(vids)].reset_index(drop=True)
    vids = np.array(list(set(dp['VehicleId'].values).intersection(vids)))
    dp = dp.loc[dp['VehicleId'].isin(vids)].reset_index(drop=True)

    # data for geospatial event info, handle null case
    if dp.size == 0:
        return dc, desc, dv, dp, de, dem
    lat = dp['Latitude'].values
    lon = dp['Longitude'].values
    dt = [pd.Timestamp(x) for x  in dp['RecordDate'].values]
    assert (lat.size > 0) and (lon.size > 0) and all(~np.isnan(lon)) and all(~np.isnan(lat))

    # update dp - timezone, localtime, day-of-week, weekday
    tzf = TimezoneFinder()
    timezone = np.array([tzf.timezone_at(lng=a, lat=b) for a, b in zip(lon, lat)])
    localtime = np.array([a.tz_localize('UTC').astimezone(b).tz_localize(None) for a, b in zip(dt, timezone)])
    dow = np.array([x.strftime('%a') for x in localtime])
    weekday = np.array([False if x in ['Sat', 'Sun'] else True for x in dow])
    dp['timezone'] = timezone
    dp['localtime'] = localtime
    dp['day_of_week'] = dow
    dp['weekday'] = weekday

    # update dp - state, county, country
    rg = reverse_geocoder.RGeocoder()
    locations = rg.query([(a, b) for a, b in zip(lat, lon)])
    dp['state'] = np.array([x['admin1'] for x in locations])
    dp['county'] = np.array([x['admin2'] for x in locations])
    dp['country'] = np.array([x['cc'] for x in locations])

    return dc, desc, dv, dp, de, dem

def model_population(dc, dv, dp, de, daily=False, oversampled=False):
    """
    nominal collision prediction model population DataFrame
    - daily to include daily predictor / collision intervals
    - oversampled to oversample positive instances
    """

    # cannot use both 'daily' and 'oversampled'
    if daily:
        assert not oversampled
    if oversampled:
        assert not daily

    # predictor and collision intervals as a DataFrame
    dpc = [(a[-4:], b.split(',')) for a, b in dc.items() if 'collision interval' in a]
    dpc = [[datetime.strptime(f'{xi}/{x[0]}', r'%b/%Y') for xi in x[1]] for x in dpc]
    dpc = pd.DataFrame({'time1': np.sort(list(chain(*dpc)))})
    dpc['time0'] = dpc['time1'] - pd.Timedelta(days=int(dc['predictor interval days']))
    dpc['time2'] = [datetime.strptime((x + pd.Timedelta(days=32)).strftime('%m/%Y'), '%m/%Y') for x in dpc['time1']]
    dpc = dpc[['time0', 'time1', 'time2']]

    # daily predictor and collision intervals
    if daily:
        dd = defaultdict(list)
        for x0, x1 in zip(dpc.index.values[:-1], dpc.index.values[1:]):
            for xd in range(1, (dpc.loc[x1, 'time0'] - dpc.loc[x0, 'time0']).days):
                dd['time0'].append(dpc.loc[x0, 'time0'] + pd.Timedelta(days=xd))
                dd['time1'].append(dpc.loc[x0, 'time1'] + pd.Timedelta(days=xd))
                dd['time2'].append(dpc.loc[x0, 'time2'] + pd.Timedelta(days=xd))
        dpc = pd.concat((dpc, pd.DataFrame(dd)[['time0', 'time1', 'time2']])).sort_values('time0').reset_index(drop=True)
        assert dpc.duplicated().sum() == 0

    # initialize collision model population DataFrame, scan over predictor/collision intervals
    dcm = pd.DataFrame()
    for _, (t0, t1, t2) in tqdm(dpc.iterrows(), desc='collision model population', total=dpc.shape[0]):

        # valid ER associations
        dx = (de.loc[(de['CreationDate'] < t0) & (de['DeletedDate'] > t2)]).copy()
        dx0 = dx.shape[0]
        assert pd.unique(dx['VehicleId']).size == dx0

        # merge with vehicle metadata
        dx = pd.merge(left=dx, right=dv, how='left', on='VehicleId')
        assert dx.shape[0] == dx0

        # add predictor/collision interval bounds
        dx['time0'] = t0
        dx['time1'] = t1
        dx['time2'] = t2
        assert (dx['DeletedDate'] - dx['CreationDate']).min() > t2 - t0

        # merge with collision metadata
        for value in [45, 46, 47]:
            dx[f'collision-{value}'] = False
            c0 = dp['BehaviorId'] == value
            c1 = dp['RecordDate'] > t1
            c2 = dp['RecordDate'] < t2
            vids = pd.unique(dp.loc[c0 & c1 & c2, 'VehicleId'])
            dx.loc[dx['VehicleId'].isin(vids), f'collision-{value}'] = True
            groups = dp.loc[c0 & c1 & c2].groupby('VehicleId').groups
            groups = pd.DataFrame({'VehicleId': groups.keys(), f'collision-{value}-idx': [x.to_numpy() for x in groups.values()]})
            dx = pd.merge(left=dx, right=groups, on='VehicleId', how='left')
            if all(pd.isnull(dx[f'collision-{value}-idx'].values)):
                dx[f'collision-{value}-idx'] = [np.array([], dtype='int64') for _ in range(dx.shape[0])]
            else:
                dx[f'collision-{value}-idx'] = np.array([np.array([], dtype='int64') if np.all(np.isnan(x)) else x for x in dx.pop(f'collision-{value}-idx')], dtype='object')

        # update dcm
        dcm = pd.concat((dcm, dx))

    # reset index and validate
    dcm = dcm.reset_index(drop=True)
    assert pd.unique(dcm['time1'] - dcm['time0']).size == 1
    assert (dcm['time1'] - dcm['time0']).loc[0].days == int(dc['predictor interval days'])
    assert all(dcm['CreationDate'] < dcm['time0']) and all(dcm['DeletedDate'] > dcm['time2'])
    assert dcm[[x for x in dcm.columns if 'idx' not in x]].duplicated().sum() == 0

    # oversampled evaluations
    if oversampled:

        # initialize oversampled columns in dcm, scan over collisions
        dcm['oversampled'] = False
        dcm['oversampled index'] = None
        for index, row in tqdm(dcm.loc[dcm['collision-47']].iterrows(), desc='oversampled vehicle evals', total=dcm['collision-47'].sum()):
            assert row['collision-47-idx'].size > 0
            for x in row['collision-47-idx']:

                # collision record and oversampled time DataFrame up to the collision record
                collision = dp.loc[x]
                ta = row['time1'] + pd.Timedelta(days=1)
                tb = pd.Timestamp(datetime.strptime(collision['RecordDate'].strftime('%Y-%m-%d'), '%Y-%m-%d')) - pd.Timedelta(days=1)
                dts = pd.date_range(ta, tb, freq=pd.Timedelta(days=1))
                cols = [x for x in row.index if (x not in ['time0', 'time1', 'time2', 'oversampled']) and ('collision' not in x)]
                dx = pd.DataFrame({'time0': dts - pd.Timedelta(days=int(dc['predictor interval days'])), 'time1': dts, 'time2': row['time2']})

                # join oversampled DataFrame with rest of information in row, append to dcm
                for key, value in row[cols].iteritems():
                    dx[key] = value
                for value in [45, 46]:
                    dx[f'collision-{value}'] = row[f'collision-{value}']
                    dx[f'collision-{value}-idx'] = None
                dx['collision-47'] = True
                dx['collision-47-idx'] = x
                dx['oversampled'] = True
                dx['oversampled index'] = index
                assert sorted(dx.columns) == sorted(dcm.columns)
                dx = dx[dcm.columns]
                dcm = pd.concat((dcm, dx))

    # for compatibility if not oversampled
    else:
        dcm['oversampled'] = False
        dcm['oversampled index'] = None

    # reset index and validate
    dcm = dcm.reset_index(drop=True)
    assert all(dcm['time1'] - dcm['time0'] == pd.Timedelta(days=int(dc['predictor interval days'])))
    dx = pd.merge(dcm.loc[dcm['oversampled']], dp[['RecordDate']], left_on='collision-47-idx', right_index=True, how='left')
    assert all((dx['RecordDate'] > dx['time1']) & (dx['RecordDate'] < dx['time2']))
    dcm = dcm.loc[~dcm[[x for x in dcm.columns if 'idx' not in x]].duplicated()].reset_index(drop=True)

    return dcm

def get_population_metadata(dcm, dc, datadir=None, oversampled_info=True):
    """
    collision model population metadata as a pandas Series
    """

    # initialize population metadata based on dc
    dm = dc[[x for x in dc.index if x not in ['CompanyName', 'IndustryDesc']]]

    # update based on dcm0 (not oversampled)
    dcm0 = dcm.loc[~dcm['oversampled']].copy()
    dm['num of vehicle evals (wout oversamples)'] = dcm0.shape[0]
    for value in [45, 46, 47]:
        a = dcm0[f'collision-{value}'].sum()
        b = 100 * a / dcm0.shape[0]
        dm[f'num evals with any collision-{value} (wout oversamples)'] = f'{a} ({b:.3f})%'

    if 'collision-gwcc' in dcm0.columns:
        a = dcm0['collision-gwcc'].sum()
        b = 100 * a / dcm0.shape[0]
        dm[f'num evals with any collision-gwcc (wout oversamples)'] = f'{a} ({b:.3f})%'

    # update based on dcm (oversampled)
    dm['num of vehicle evals (with oversamples)'] = dcm.shape[0]
    for value in [45, 46, 47]:
        a = dcm[f'collision-{value}'].sum()
        b = 100 * a / dcm.shape[0]
        dm[f'num evals with any collision-{value} (with oversamples)'] = f'{a} ({b:.3f})%'

    # unique vehicle and event recorder / device ids
    dm['num unique vehicle-id'] = pd.unique(dcm['VehicleId']).size
    if 'EventRecorderId' in dcm.columns:
        dm['num unique event-recorder-id'] = pd.unique(dcm['EventRecorderId']).size
    if 'DeviceId' in dcm.columns:
        dm['num unique device-id'] = pd.unique(dcm['DeviceId']).size

    # data volume metadata
    if datadir is not None:
        psets = sorted(glob(os.path.join(datadir, '*.parquet')))
        for pset in psets:
            ps = os.path.split(pset)[1]
            dsize = (1e-9) * get_folder_size(pset)
            partitions = len(os.listdir(pset))
            dm[f'{ps} - GB, num of partitions'] = f'{dsize:.2f}GB, {partitions}'

    # clean up based on oversampled_info
    if not oversampled_info:
        for x in [x for x in dm.index if 'with oversamples' in x]:
            del dm[x]
        for x in [x for x in dm.index if 'wout oversamples' in x]:
            dm[x.replace(' (wout oversamples)', '')] = dm.pop(x)

    return dm

def validate_dcm_dp(dcm, dp):
    """
    validate collision indices in dcm align with full collision metadata in dp
    """

    # validate non-oversampled rows
    print('validate non-oversampled rows')
    dcm0 = dcm.loc[~dcm['oversampled']]
    for value in [45, 46, 47]:
        for _, row in dcm0.loc[dcm[f'collision-{value}']].iterrows():
            assert row[f'collision-{value}-idx'].size > 0
            for x in row[f'collision-{value}-idx']:
                assert row['VehicleId'] == dp.loc[x, 'VehicleId']
                assert dp.loc[x, 'BehaviorId'] == value
                assert row['time1'] < dp.loc[x, 'RecordDate'] < row['time2']

    # identify rows in dcm0 that may be oversampled
    dcm1 = dcm.loc[dcm['oversampled']]
    if dcm1.size == 0:
        return
    print('validate oversampled rows')
    c0 = np.sort(dcm0.loc[dcm0['collision-47']].index.to_numpy())
    c1 = np.sort(pd.unique(dcm1['oversampled index'])).astype('int')
    assert c1.size <= c0.size
    not_oversampled = np.sort(np.array(list(set(c0).difference(c1))))
    oversampled = np.sort(np.array(list(set(c0).intersection(c1))))
    if oversampled.size > 0:
        assert dcm0.index[-1] >= oversampled.max()
    if not_oversampled.size > 0:
        assert dcm0.index[-1] >= not_oversampled.max()

    # validate rows in dcm0 eligible to be oversampled but not oversampled
    for x in not_oversampled:
        row = dcm.loc[x]
        for cx in row['collision-47-idx']:
            days = (dp.loc[cx, 'RecordDate'] - row['time1']).total_seconds() / (60 * 60 * 24)
            assert days < 2

    # validate rows in dcm0 eligible to be oversampled and oversampled
    for x in oversampled:
        row = dcm.loc[x]
        xrows = dcm.loc[dcm['oversampled index'] == x]
        assert xrows.size > 0
        assert xrows['time1'].min() > row['time1']
        cdt = dp.loc[row['collision-47-idx'], 'RecordDate'].max()
        days = (cdt - xrows['time1'].max()).total_seconds() / (60 * 60 * 24)
        assert 1 < days < 2

def convert_v1_to_v2(dcm, dp, dc):
    """
    convert population DataFrames from 'v1' format to 'v2' format
    """
    dcm['oversampled'] = False
    dcm['collision-45'] = False
    dcm['collision-46'] = False
    dcm['collision-47'] = dcm.pop('collision')
    idx = []
    for x, row in tqdm(dcm.iterrows(), desc='v1 to v2', total=dcm.shape[0]):
        if not row['collision-47']:
            idx.append(np.array([]).astype('int'))
            continue
        dpx = dp.loc[(dp['VehicleId'] == row['VehicleId']) & (dp['RecordDate'].isin(row['collision_ts']))]
        assert dpx.shape[0] == row['collision_ts'].size
        idx.append(dpx.index.to_numpy())
    dcm['collision-47-idx'] = np.array(idx, dtype='object')
    dcm['time2'] = dcm['time1'] + pd.Timedelta(days=int(dc['POSITIVE_CLASS_DAYS']))
    dp['BehaviorId'] = dp['BehaviorId'].astype('int')

    return dcm, dp

def modify_dcm_for_data_extraction(dcm, xid, dc):
    """
    group by xid and sort to optimize query time in data extraction steps
    """
    dx = time_bounds_dataframe(df=dcm, xid=xid)

    # case where all row durations are same as nominal window duration
    if np.all((dx['time1'] - dx['time0']) == pd.Timedelta(days=int(dc['predictor interval days']))):
        return dx.sort_values('time0').reset_index(drop=True)

    # identify where window size first exceeds nominal window size
    x0 = np.where((dx['time1'] - dx['time0']) > pd.Timedelta(days=int(dc['predictor interval days'])))[0][0]

    # split into nominal and greater-than-nominal window size
    dx0 = dx.loc[: x0 - 1].reset_index(drop=True)
    dx1 = dx.loc[x0 :].reset_index(drop=True)
    assert np.all((dx0['time1'] - dx0['time0']) == pd.Timedelta(days=int(dc['predictor interval days'])))
    assert np.all((dx1['time1'] - dx1['time0']) > pd.Timedelta(days=int(dc['predictor interval days'])))

    # sort by start time and concat
    dx0 = dx0.sort_values(['time0', xid])
    dx1 = dx1.sort_values(['time0', xid])
    return pd.concat((dx0, dx1)).reset_index(drop=True)

def bootstrap_hypothesis_test_and_visualization(xa, xb, bins=np.linspace(-1, 1, 100)):
    """
    run bootstrap hypothesis test and create visualization
    null hypothesis:
    data in xa and in xb are drawn from same population
    """
    from dask import delayed, compute
    from dask.diagnostics import ProgressBar

    # validate
    assert isinstance(xa, np.ndarray)
    assert isinstance(xb, np.ndarray)
    assert xa.ndim == 1
    assert xb.ndim == 1

    # get data sizes and initial means
    xas = xa.size
    xbs = xb.size
    xam = xa.mean()
    xbm = xb.mean()

    # data embodiment of the null hypothesis
    data = np.hstack((xa, xb))
    ds = data.size
    dsx = np.arange(ds)

    def bootstrap(dsx, xas, data):
        oka = np.random.choice(dsx, size=xas, replace=False)
        okb = np.setdiff1d(dsx, oka)
        return data[oka].mean(), data[okb].mean()

    # run bootstrap hypothesis test
    niter = 10000
    print(f'bootstrap hypothesis test, {niter} iterations')
    with ProgressBar():
        rv = compute(*[delayed(bootstrap)(dsx, xas, data) for _ in range(niter)])
    rv = np.array(rv)
    rva = rv[:, 0]
    rvb = rv[:, 1]

    # bootstrap difference and nominal difference
    diff_b = rva - rvb
    diff_0 = xam - xbm

    # result of bootstrap hypothesis test
    fig, ax = open_figure('bootstrap hypothesis test', figsize=(10, 4))
    ax.hist(diff_b, bins=bins, label=f'distribution of {niter}\nbootstrap differences')
    ylim = ax.get_ylim()
    ax.plot(np.tile(diff_0, 2), ylim, '--', lw=3, label=f'observed difference, {diff_0:.2f}')
    title = f'Bootstrap Hypothesis Test of Population A/B'
    title += f'\nPopulation A size - {xas}, mean - {xam:.2f}'
    title += f'\nPopulation B size - {xbs}, mean - {xbm:.2f}'
    if xam - xbm > 0:
        pvalue = 100 * (diff_b >= diff_0).sum() / niter
    else:
        pvalue = 100 * (diff_b <= diff_0).sum() / niter
    title += f'\np-value is {pvalue:.2f}%'
    format_axes('difference of population means', 'bin count', title, ax)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), numpoints=3, handlelength=3)
    ax.set_yscale('log')
    largefonts(14)
    fig.tight_layout()

    return pvalue

def get_shap_chart_data(base, values, xr, n_features, cols):
    """
    return chart data and labels for a shap waterfall chart
    """

    # validate
    assert values.size == cols.size == xr.size

    # row counts for positive and negative rows
    positive = (values >= 0)
    negative = (values < 0)
    positive_rows = n_features + 1 if positive.sum() > n_features else positive.sum()
    negative_rows = n_features + 1 if negative.sum() > n_features else negative.sum()
    n_rows = positive_rows + negative_rows

    # y-data for positive and negative rows
    yp = np.arange(n_rows, negative_rows, -1) - 1
    yp = np.vstack((yp, yp)).T
    yn = np.arange(negative_rows - 1, -1, -1)
    yn = np.vstack((yn, yn)).T

    # positive rows
    ok = np.argsort(values[positive])[::-1]
    pvals = values[positive][ok]
    formatters = np.array(['.0f' if x.is_integer() else '.1f' for x in pvals])
    pcols = np.array([f'{a}={b:{fmt}}' for a, b, fmt in zip(cols[positive][ok], xr[positive][ok], formatters)])
    for x, (px, cx) in enumerate(zip(pvals[:n_features], pcols[:n_features])):
        if x == 0:
            positive = np.expand_dims(np.array([base, base + px]), axis=0)
        else:
            positive = np.vstack((positive, np.array([positive[-1, 1], px + positive[-1, 1]])))
    if positive_rows == n_features + 1:
        positive = np.vstack((positive, np.array([positive[-1, 1], pvals[n_features:].sum() + positive[-1, 1]])))
        pcols = np.hstack((pcols[:n_features], np.array(['all positive others'])))
    # null case
    if positive.ndim == 1:
        positive = yp.copy().astype('float')

    # negative rows
    ok = np.argsort(values[negative])
    nvals = values[negative][ok]
    formatters = np.array(['.0f' if x.is_integer() else '.1f' for x in nvals])
    ncols = np.array([f'{a}={b:{fmt}}' for a, b, fmt in zip(cols[negative][ok], xr[negative][ok], formatters)])
    for x, (nx, cx) in enumerate(zip(nvals[:n_features], ncols[:n_features])):
        if x == 0:
            base = base if positive.size == 0 else positive[-1, 1]
            negative = np.expand_dims(np.array([base + nx, base]), axis=0)
        else:
            negative = np.vstack((negative, np.array([nx + negative[-1, 0], negative[-1, 0]])))
    if negative_rows == n_features + 1:
        negative = np.vstack((negative, np.array([nvals[n_features:].sum() + negative[-1, 0], negative[-1, 0]])))
        ncols = np.hstack((ncols[:n_features], np.array(['all negative others'])))
    # null case
    if negative.ndim == 1:
        negative = yn.copy().astype('float')

    return yp, positive, pcols, yn, negative, ncols

def mpl_shap_waterfall_chart(yp, positive, pcols, yn, negative, ncols, title='shap waterfall chart',
        figsize=(16, 8), ms=12, lw=4, size=16, fontsize=14):
    """
    matplotlib version of a shap waterfall chart
    """

    # figure objects and base value
    fig, ax = open_figure(title, figsize=figsize)
    base = positive[0, 0] if positive.size > 0 else negative[0, 1]
    ax.plot(np.tile(base, 2), np.array([-0.5, pcols.size + ncols.size - 0.5]), '--', color='black', lw=4, label=f'base value, {base:.2f}')

    # positive contributors
    for x, (y, px) in enumerate(zip(yp, positive)):
        ax.plot(px, y, '-', lw=lw, color='green')
        p = ax.plot(px[1], y[1], '->', ms=ms, lw=lw, color='green')[0]
        ax.text(px[1], y[0], f'  +{np.diff(px)[0]:.3f}', ha='left', va='center', fontsize=fontsize, fontweight='bold')
        if x == 0:
            p.set_label('positive contributors')

    # negative contributors
    for x, (y, nx) in enumerate(zip(yn, negative)):
        ax.plot(nx, y, '-', lw=lw, color='red')
        p = ax.plot(nx[0], y[0], '-<', ms=ms, lw=lw, color='red')[0]
        ax.text(nx[0], y[0], f'-{np.diff(nx)[0]:.3f}  ', ha='right', va='center', fontsize=fontsize, fontweight='bold')
        if x == 0:
            p.set_label('negative contributors')

    # feature labels
    vs = np.hstack((negative.flatten(), positive.flatten()))
    ax.set_xlim(0.9 * vs.min(), 1.1 * vs.max())
    ax.set_ylim(-0.5, pcols.size + ncols.size - 0.5)
    ax.set_yticks(range(pcols.size + ncols.size))
    ax.set_yticklabels(np.hstack((pcols, ncols))[::-1])

    # clean up
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), numpoints=1, handlelength=4)
    format_axes('prediction probability', '', title, ax)
    largefonts(size)
    fig.tight_layout()

def mpl_shap_dependence_chart(feature, x, xs, x0, xp, title, figsize=(10, 6), ms=10, lw=2, mew=1, size=16):
    """
    matplotlib version of a shap dependence plot
    x - feature values
    xs - shap values for feature
    x0 - model actual outcome
    xp - model predicted probabilities
    """

    # validate
    assert (x.shape == xs.shape == x0.shape == xp.shape) and (x.ndim == 1)

    # fig and ax objects
    fig, ax = open_figure(title, 2, 1, figsize=figsize)

    # feature distribution
    bins = np.linspace(x.min(), 1.01 * x.max(), 50)
    xd = np.digitize(x, bins)
    height = np.array([(xd == xi).sum() for xi in range(1, bins.size + 1)])
    assert ((xd == 0).sum() == 0) and ((xd == bins.size).sum() == 0) and (height[-1] == 0)
    width = np.diff(bins)[0]
    ax[0].bar(x=bins, height=height, align='edge', width=width, alpha=0.8)
    centers = (bins[1:] + bins[:-1]) / 2
    ax[0].plot(centers, height[:-1], '-', lw=lw)
    format_axes(feature, 'bin count', f'distribution of {feature}', ax[0])

    # shap dependence plot
    ax[1].plot(x, xs, 'o', ms=ms, mew=mew, color='darkblue')
    format_axes(feature, f'shap value', f'shap dependence plot for {feature}', ax[1])

    # clean up
    for x in ax:
        x.set_xlim(bins[0], bins[-1])
    largefonts(size)
    fig.tight_layout()

    # fig = plt.figure(figsize=(17, 7))
    # fig.canvas.manager.set_window_title(f'dependence plots and distribution for {feature}')
    # ax = np.full(3, None).astype('object')
    # ax[0] = plt.subplot2grid((2, 2), (0, 0), rowspan=1, colspan=1)
    # ax[1] = plt.subplot2grid((2, 2), (1, 0), rowspan=1, colspan=1)
    # ax[2] = plt.subplot2grid((2, 2), (0, 1), rowspan=2, colspan=1)
    # x = X_test[feature].values
    # # partial dependence plot
    # ax[1].plot(x, rf.predict_proba(X_test)[:, 1], 'o', ms=8, mew=2)
    # format_axes(feature, f'prediction probability', f'partial dependence plot for {feature}', ax[1])

def get_classification_metrics(ytrue, ypred):
    """
    return binary classification metrics
    """
    acc = accuracy_score(ytrue, ypred)
    tn, fp, fn, tp = confusion_matrix(ytrue, ypred).ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    tnr = tn / (tn + fp)
    fnr = fn / (fn + tp)
    if (tp > 0) or (fp > 0):
        precision = tp / (tp + fp)
    else:
        precision = 0
    if (tn > 0) or (fn > 0):
        precision0 = tn / (tn + fn)
    else:
        precision0 = 0
    if (precision > 0) or (tpr > 0):
        f1 = (2 * precision * tpr) / (precision + tpr)
    else:
        f1 = 0
    return {
        'acc': acc,
        'precision': precision,
        'precision0': precision0,
        'f1': f1,
        'positive_rate': (tp + fp) / (tp + fp + tn + fn),
        'negative_rate': (tn + fn) / (tp + fp + tn + fn),
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'tpr': tpr,
        'fpr': fpr,
        'tnr': tnr,
        'fnr': fnr,
        'bacc': (tpr + tnr) / 2}

def get_learning_curve_data(df, y, model, preprocessor=None, n_splits=4, train_sizes=np.linspace(0.1, 1, 10)):
    """
    create learning curve data from a single model
    - store prediction probability and classification metrics for each train-size of each split
    - 0 < train_sizes <= 1 (must be sorted, max must be 1), represents fractions of training data to be used
    """
    from dask import delayed, compute
    from dask.diagnostics import ProgressBar

    # validation
    assert isinstance(df, pd.DataFrame)
    assert isinstance(y, np.ndarray)
    assert df.shape[0] == y.size
    assert train_sizes.min() > 0
    assert train_sizes.max() == 1
    assert np.all(np.sort(train_sizes) == train_sizes)

    # initialize dict of learning curve indices, KFold object, and scan over splits
    kf = StratifiedKFold(n_splits=n_splits)
    idx = defaultdict(list)
    for split, (train, test) in enumerate(kf.split(X=np.zeros(y.size), y=y)):

        # scan over train-sizes, get random train indices for each train-size, update index dict
        tn = train.size
        tx = np.arange(tn)
        np.random.shuffle(train)
        for ts in train_sizes:
            x = train[np.random.choice(tx, int(ts * tn), replace=False)]
            # x = train[: int(ts * tn)]
            assert np.intersect1d(test, x).size == 0
            idx['split'].append(split)
            idx['train indices fractional size'].append(ts)
            idx['train indices actual size'].append(x.size)
            idx['train indices'].append(x)
            idx['test indices actual size'].append(test.size)
            idx['test indices'].append(test)

    # convert to DataFrame
    idx = pd.DataFrame(idx)

    # validate train indices, get range of actual train size by fractional train size
    assert pd.unique(idx['train indices fractional size']).size == train_sizes.size
    left = idx.groupby('train indices fractional size')['train indices actual size'].min()
    right = idx.groupby('train indices fractional size')['train indices actual size'].max()
    da = pd.merge(left, right, left_index=True, right_index=True, how='inner', suffixes=(' min', ' max'))
    assert np.abs((left - right).values).max() <= 1

    # run and eval single ML model
    def run_model(model, preprocessor, row):

        # get local clone of model and preprocessor
        model = clone(model)
        if preprocessor is not None:
           preprocessor = clone(preprocessor)

        # extract train and test data
        df_train = df.loc[row['train indices']]
        df_test = df.loc[row['test indices']]
        y_train = y[row['train indices']]
        y_test = y[row['test indices']]
        assert np.any(y_train)
        assert np.any(y_test)

        # extract X_train/X_test via preprocessor
        if preprocessor is not None:
            X_train = preprocessor.fit_transform(df_train)
            X_test = preprocessor.transform(df_test)
        else:
            set_trace()

        # fit and eval
        model.fit(X_train, y_train)
        y_train_proba = model.predict_proba(X_train)[:, model.classes_ == 1].flatten()
        y_test_proba = model.predict_proba(X_test)[:, model.classes_ == 1].flatten()
        assert y_train_proba.size == y_train.size
        assert y_test_proba.size == y_test.size
        train = {'true': y_train, 'proba': y_train_proba}
        test = {'true': y_test, 'proba': y_test_proba}

        # DataFrame of train/test evaluation metrics and metadata
        metadata = {'split': row['split'], 'frac train size': row['train indices fractional size']}
        train = {**{'component': 'train'}, **metadata, **train}
        test = {**{'component': 'test'}, **metadata, **test}
        ds = pd.DataFrame((train, test))

        return ds

    # distributed processing of learning curve data
    print(f'distributed learning curve data')
    with ProgressBar():
        rv = compute(*[delayed(run_model)(model, preprocessor, row) for _, row in idx.iterrows()])

    # create return objects
    dlc = pd.concat(rv).reset_index(drop=True)
    assert np.all(pd.unique(dlc['frac train size']) == train_sizes)
    assert np.all(pd.unique(dlc['frac train size']) == da.index)
    dlc = pd.merge(left=dlc, right=da['train indices actual size min'], left_on='frac train size', right_index=True, how='inner')
    dlc['actual train size'] = dlc.pop('train indices actual size min')
    dlc['train frac all data'] = dlc['actual train size'] / df.shape[0]

    return dlc.sort_values(['component', 'actual train size', 'split']).reset_index(drop=True)

def get_roc_pr_data(ytrue, yprob, size):
    """
    return DataFrame of classification metrics vs prediction-threshold
    """
    thresh = np.logspace(np.log10(yprob.min()), np.log10(yprob.max()), size)
    dml = defaultdict(list)
    for x in thresh:
        ypred = (yprob >= x)
        metrics = get_classification_metrics(ytrue, ypred)
        dml['thresh'].append(x)
        for key, value in metrics.items():
            dml[key].append(value)
    return pd.DataFrame(dml)
