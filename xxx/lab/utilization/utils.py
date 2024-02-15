
"""
utils for vehicle utilization prediction model
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from datetime import datetime
from collections import defaultdict
from pyrb.processing import get_folder_size
from lytx import get_conn, get_columns
from tqdm import tqdm
from ipdb import set_trace
plt.style.use('bmh')


# model metadata and population
def model_metadata(population, devices, window):
    """
    vehicle utilization model metadata, returns
    dc - population metadata Series
    dv - vehicle metadata DataFrame
    de - event recorder association DataFrame
    dem - event recorder association metrics DataFrame
    args
    population, eg lytx.get_population('amt')
    devices, eg ['ER-SF300', 'ER-SF64']
    window, eg ['8/1/2021', '6/1/2022']
    """

    # population metadata Series
    dc = pd.Series({'desc': population['desc']})
    for key, value in population.items():
        if key == 'desc':
            continue
        dc[key] = ','.join(["""'{}'""".format(x) for x in [x.replace("""'""", """''""") for x in value]])
    dc['devices'] = ','.join(devices)
    dc['window start time'] = window[0]
    dc['window end time'] = window[1]

    # vehicle metadata DataFrame
    edw = get_conn('edw')
    now = datetime.now()
    query = f"""
        SELECT D.VehicleId, V.VehicleName, C.CompanyName, C.IndustryDesc, C.CompanyId
        FROM flat.Companies AS C
        LEFT JOIN flat.Devices AS D
        ON C.CompanyId = D.CompanyId
        LEFT JOIN hs.Vehicles AS V
        ON V.Id = D.VehicleId
        WHERE C.CompanyName <> 'DriveCam DC4DC Test Co'
        AND D.VehicleId <> '00000000-0000-0000-0000-000000000000'"""
    for field, xf in population.items():
        if field == 'desc':
            continue
        sf = ','.join(["""'{}'""".format(x) for x in [x.replace("""'""", """''""") for x in xf]])
        query += f"""\nAND {field} IN ({sf})"""
    dv = pd.read_sql_query(query, edw).drop_duplicates().sort_values('VehicleId').reset_index(drop=True)
    assert pd.unique(dv['VehicleId']).size == dv.shape[0]
    print(f'query vehicle metadata, {(datetime.now() - now).total_seconds():.1f}sec')

    # query ER assocations, for vehicles from metadata query
    now = datetime.now()
    devices_str = ','.join(["""'{}'""".format(x) for x in devices])
    query1 = f"""
        WITH V AS ({query})
        SELECT
            ERA.VehicleId,
            ERA.EventRecorderId,
            ERA.Id AS ERA_Id,
            ER.Model,
            ER.SerialNumber,
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
    de = pd.read_sql_query(query1, edw).drop_duplicates().sort_values(['VehicleId', 'CreationDate']).reset_index(drop=True)
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
    tmin = pd.Timestamp(datetime.strptime(window[0], r'%m/%d/%Y'))
    tmax = pd.Timestamp(datetime.strptime(window[1], r'%m/%d/%Y'))
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

    return dc, dv, de, dem

def model_population(dc, dv, de):
    """
    returns vehicle utilization prediction model population DataFrame
    """

    # initialize, get time window bounds
    dcm = defaultdict(list)
    tmin = pd.Timestamp(datetime.strptime(dc['window start time'], r'%m/%d/%Y'))
    tmax = pd.Timestamp(datetime.strptime(dc['window end time'], r'%m/%d/%Y'))

    # scan over vehicle id and event recorders
    for vid in tqdm(pd.unique(dv['VehicleId']), desc='model population'):
        for _, dev in de.loc[de['VehicleId'] == vid].iterrows():

            # identify single event recorders installed over full time window and record in population
            if (dev['CreationDate'] < tmin) and (dev['DeletedDate'] > tmax):
                dx = dv.loc[dv['VehicleId'] == vid].squeeze()
                del dx['VehicleId']
                del dev['VehicleId']
                dcm['VehicleId'].append(vid)
                dcm['time0'].append(tmin)
                dcm['time1'].append(tmax)
                for xa, xb in dev.iteritems():
                    dcm[xa].append(xb)
                for xa, xb in dx.iteritems():
                    dcm[xa].append(xb)

    # validate and return
    dcm = pd.DataFrame(dcm)
    assert pd.unique(dcm['VehicleId']).size == dcm.shape[0]
    return dcm

def get_population_metadata(dp, dc, datadir=None):
    """
    vehicle utilization prediction model population metadata as a pandas Series
    """

    # initialize population metadata based on dc
    dm = dc[[x for x in dc.index if x not in ['CompanyName', 'IndustryDesc']]]

    # update based on dp
    dm['num unique vehicle-id'] = pd.unique(dp['VehicleId']).size
    dm['num unique event-recorder-id'] = pd.unique(dp['EventRecorderId']).size
    dm['num unique company-id'] = pd.unique(dp['CompanyId']).size

    # data volume metadata
    if datadir is not None:
        psets = sorted(glob(os.path.join(datadir, '*.parquet')))
        for pset in psets:
            ps = os.path.split(pset)[1]
            dsize = (1e-9) * get_folder_size(pset)
            partitions = len(glob(os.path.join(pset, '*')))
            dm[f'{ps} - GB, num of partitions'] = f'{dsize:.2f}GB, {partitions}'

    return dm

# ...
