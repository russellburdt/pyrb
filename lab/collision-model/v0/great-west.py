
"""
Great-West POC
"""

import os
import lytx
import utils
import numpy as np
import pandas as pd
import sqlalchemy as sa
import pyarrow.parquet as pq
import boto3
import matplotlib.pyplot as plt
from shutil import rmtree
from glob import glob
from pyrb.mpl import metric_distribution, open_figure, format_axes, largefonts, save_pngs
from pyrb.processing import int_to_ordinal
from collections import defaultdict
from scipy.interpolate import interp1d
from Levenshtein import distance as lev
from datetime import datetime
from ipdb import set_trace
from tqdm import tqdm


# datadir
datadir = r'/mnt/home/russell.burdt/data/collision-model/gwcc/v0'
# datadir = r'c:/Users/russell.burdt/Downloads'
assert os.path.isdir(datadir)

# clean raw GWCC loss records, merge with Lytx VINs and event-recorder-associations
if False:

    # loss records provided by GW
    fn = os.path.join(datadir, 'gwcc-loss-records.csv')
    assert os.path.isfile(fn)
    d0 = pd.read_csv(fn)

    # filter columns
    d1 = d0[[
        'DOT_NUM', 'VIN', 'Company_Name', 'ACCD_DATM', 'ACCD_LOC_TEXT', 'ACCD_LOC_CITY', 'ACCD_LOC_STATE',
        'Loss_Status', 'Paid_Loss', 'Net_Paid_Loss', 'Incurred_Loss', 'Net_Incurred_Loss']].copy()

    # filter duplicates, VINs without 17 characters, convert ACCD_DATM
    d2 = d1.loc[~d1.duplicated()].reset_index(drop=True)
    d2['VIN'] = [x.strip() for x in d2['VIN']]
    d2 = d2.loc[np.array([len(x) == 17 for x in d2['VIN']])].reset_index(drop=True)
    d2 = d2.loc[~d2['VIN'].str.lower().str.contains('unknown')].reset_index(drop=True)
    d2['ACCD_DATM'] = [pd.Timestamp(x) for x in d2['ACCD_DATM'].values]

    # filter partial duplicates (same loss record recorded twice), keep loss record with higher loss amount
    d3 = defaultdict(list)
    for _, group in d2.groupby(['Company_Name', 'VIN', 'ACCD_DATM']):
        assert group.shape[0] in [1, 2]
        if group.shape[0] == 1:
            for name, value in group.items():
                if name[-5:] == '_Loss':
                    d3[name].append(str(int(value.iloc[0])))
                else:
                    d3[name].append(value.iloc[0])
        elif group.shape[0] == 2:
            for name in group.columns:
                if 'Loss' not in name:
                    d3[name].append(group.iloc[0][name])
                elif name == 'Loss_Status':
                    d3[name].append(','.join(group[name].astype(str).values))
                else:
                    d3[name].append(','.join(group[name].astype(int).astype(str).values))
    d3 = pd.DataFrame(d3)

    # Lytx VIN DataFrame
    dv = pd.read_sql_query(con=lytx.get_conn('lab'), sql=sa.text(f"""
        SELECT
            gwcc_insured_vehicle_vin AS vin,
            edw_hs_vehicles__id AS v0,
            vnet__vehicle_id AS v1,
            edw_installation__vin__edw__flat__devices__vehicleid AS v2
        FROM insurance_model_gwcc.gwcc__truck_in_force_4_30_2023__fuzzy_lytx_vin_match
        WHERE is_lytx_vin_exactly_matches_gwcc_vin = true"""))
    assert pd.isnull(dv['vin']).sum() == 0
    assert np.unique([len(x) for x in dv['vin']]) == np.array([17])
    dn = defaultdict(list)
    for group in dv.groupby('vin'):
        vid = np.unique(np.hstack((
            group[1].loc[~pd.isnull(group[1]['v0']), 'v0'].values,
            group[1].loc[~pd.isnull(group[1]['v1']), 'v1'].values,
            group[1].loc[~pd.isnull(group[1]['v2']), 'v2'].values)))
        if (vid.size == 1) and (vid[0] != '00000000-0000-0000-0000-000000000000'):
            dn['VIN'].append(group[0])
            dn['VehicleId'].append(vid[0])
    dn = pd.DataFrame(dn)
    assert pd.unique(dn['VIN']).size == pd.unique(dn['VehicleId']).size == dn.shape[0]

    # merge VIN data with GW loss records
    d4 = pd.merge(left=d3, right=dn, how='inner', on='VIN')
    d4 = d4.rename(columns={
        'DOT_NUM': 'GW-dotnum',
        'Company_Name': 'GW-companyname',
        'ACCD_DATM': 'GW-datetime',
        'ACCD_LOC_TEXT': 'GW-loc-desc',
        'ACCD_LOC_CITY': 'GW-loc-city',
        'ACCD_LOC_STATE': 'GW-loc-state',
        'Loss_Status': 'GW-loss-status',
        'Paid_Loss': 'GW-paid-loss',
        'Net_Paid_Loss': 'GW-net-paid-loss',
        'Incurred_Loss': 'GW-incurred-loss',
        'Net_Incurred_Loss': 'GW-net-incurred-loss'})

    # Lytx event-recorder-associations for VINs
    vstr = ','.join([f"""'{x}'""" for x in pd.unique(d4['VehicleId'])])
    query = f"""
        SELECT
            ERA.EventRecorderId, ERA.VehicleId, ERA.CreationDate, ERA.DeletedDate, ERA.GroupId,
            ER.Model, G.Name as GroupName, C.CompanyId, C.CompanyName, C.IndustryDesc
        FROM hs.EventRecorderAssociations AS ERA
            LEFT JOIN flat.Groups AS G ON ERA.GroupId = G.GroupId
            LEFT JOIN flat.Companies AS C ON G.CompanyId = C.CompanyId
            LEFT JOIN hs.EventRecorders AS ER ON ER.Id = ERA.EventRecorderId
        WHERE ERA.VehicleId IN ({vstr})"""
    de = pd.read_sql_query(sa.text(query), lytx.get_conn('edw'))
    assert de['DeletedDate'].dtype.type == np.object_
    if isinstance(de['DeletedDate'].loc[0], datetime):
        de['DeletedDate'] = [pd.Timestamp(x) if x.strftime(r'%Y-%m-%d') != '9999-01-01' else pd.Timestamp('2262-04-11 23:47:16.854775807') for x in de['DeletedDate'].values]
    else:
        de['DeletedDate'] = [pd.Timestamp(x) if x[:10] != '9999-01-01' else pd.Timestamp('2262-04-11 23:47:16.854775807') for x in de['DeletedDate'].values]
    ok = (de['DeletedDate'].values - de['CreationDate'].values).astype('float') > 0
    de = de.loc[ok].reset_index(drop=True)
    de = de.loc[de['Model'].isin(['ER-SF64', 'ER-SF300', 'ER-SF300V2'])].reset_index(drop=True)

    # merge GW loss records with Lytx event-recorder-associations
    d5 = defaultdict(list)
    for _, row in d4.iterrows():

        # all event-recorder-associations for VIN / VehicleId
        era = de.loc[de['VehicleId'] == row['VehicleId']]
        if era.size == 0:
            continue

        # equivalent cpm t0 / t1 / t2 based on GW-datetime
        t1 = row['GW-datetime']
        t1 = pd.Timestamp(year=t1.year, month=t1.month, day=1)
        t0 = t1 - pd.Timedelta(days=90)
        t2 = t1 + pd.Timedelta(days=32)
        t2 = pd.Timestamp(year=t2.year, month=t2.month, day=1)

        # valid event-recorder-associations for VIN / VehicleId
        ok = (era['CreationDate'] < t0) & (era['DeletedDate'] > t2)
        assert ok.sum() in [0, 1]
        if ok.sum() == 0:
            continue

        # save GW loss record and event-recorder-association
        era = era.loc[ok].squeeze()
        assert era['VehicleId'] == row['VehicleId']
        del era['VehicleId']
        for name, value in row.items():
            d5[name].append(value)
        for name, value in era.items():
            d5[name].append(value)
    d5 = pd.DataFrame(d5)

    # limit time window
    d5 = d5.loc[(d5['GW-datetime'] > pd.Timestamp('2021-11-1')) & (d5['GW-datetime'] < pd.Timestamp('2022-8-1'))].reset_index(drop=True)
    d5.to_pickle(os.path.join(datadir, 'gwcc-lytx-loss-records.p'))

    # print Lytx company-names
    for company in np.sort(pd.unique(d5['CompanyName'])):
        print(f"""\"\"\"{company}\"\"\",""")

# events and behaviors DataFrames for merged GWCC loss records
if False:

    # merged GWCC loss records, time window for events and behaviors data
    df = pd.read_pickle(os.path.join(datadir, 'gwcc-lytx-loss-records.p'))
    tmin = df['GW-datetime'].min() - pd.Timedelta(days=7)
    tmax = df['GW-datetime'].max() + pd.Timedelta(days=7)
    tmin = pd.Timestamp(year=tmin.year, month=tmin.month, day=tmin.day).strftime(r'%Y-%m-%d')
    tmax = pd.Timestamp(year=tmax.year, month=tmax.month, day=tmax.day).strftime(r'%Y-%m-%d')

    # events raw data
    vstr = ','.join([f"""'{x}'""" for x in pd.unique(df['VehicleId'])])
    query = f"""
        SELECT E.VehicleId, E.RecordDate, E.Latitude, E.Longitude, E.EventId, E.EventRecorderId,
            E.EventRecorderFileId, E.SpeedAtTrigger, E.EventTriggerTypeId AS NameId, E.EventTriggerSubTypeId AS SubId,
            E.BehaviourStringIds, E.EventFilePath, E.EventFileName, T.Name
        FROM flat.Events AS E
            LEFT JOIN hs.EventTriggerTypes_i18n AS T
            ON T.Id = E.EventTriggerTypeId
        WHERE E.Deleted=0
        AND E.RecordDate BETWEEN '{tmin}' AND '{tmax}'
        AND E.VehicleId IN ({vstr})"""
    de = pd.read_sql_query(sa.text(query), lytx.get_conn('edw'))
    de.to_pickle(os.path.join(datadir, 'gwcc-lytx-events.p'))

    # behaviors raw data
    query = f"""
        SELECT B.VehicleId, B.RecordDate, B.Latitude, B.Longitude, B.EventId, B.EventRecorderId, value AS BehaviorId,
            B.SpeedAtTrigger, B.EventFilePath, B.EventFileName, hsb.Name
        FROM flat.Events AS B
            CROSS APPLY STRING_SPLIT(COALESCE(B.BehaviourStringIds, '-1'), ',')
            LEFT JOIN hs.Behaviors_i18n AS hsb ON value = hsb.Id
        WHERE value <> -1
        AND Deleted = 0
        AND B.RecordDate BETWEEN '{tmin}' AND '{tmax}'
        AND B.VehicleId IN ({vstr})"""
    db = pd.read_sql_query(sa.text(query), lytx.get_conn('edw'))
    db.to_pickle(os.path.join(datadir, 'gwcc-lytx-behaviors.p'))

    # merge with DataFrame of num of specific events/behaviors within 24 hours of each GW loss record
    dc = defaultdict(list)
    for _, row in df.iterrows():
        tmin = row['GW-datetime'] - pd.Timedelta(hours=24)
        tmax = row['GW-datetime'] + pd.Timedelta(hours=24)
        dex = de.loc[(de['VehicleId'] == row['VehicleId']) & (de['RecordDate'] > tmin) & (de['RecordDate'] < tmax)]
        dbx = db.loc[(db['VehicleId'] == row['VehicleId']) & (db['RecordDate'] > tmin) & (db['RecordDate'] < tmax)]
        dc['num accel in 24h'].append((dex['NameId'] == 30).sum())
        dc['num collision-47 in 24h'].append((dbx['BehaviorId'] == '47').sum())
        dc['num collision-all in 24h'].append((dbx['BehaviorId'].isin(['45', '46', '47', '72', '142'])).sum())
    dm = pd.concat((df, pd.DataFrame(dc)), axis=1)

    # Lytx collisions without associated GW loss record
    dx = defaultdict(list)
    for _, row in db.loc[db['BehaviorId'] == '47'].iterrows():
        dxx = df.loc[df['VehicleId'] == row['VehicleId']]
        assert dxx.size > 0
        if np.abs((dxx['GW-datetime'] - row['RecordDate']).min().total_seconds()) > 24*3600:
            for name, value in row.items():
                dx[name].append(value)
    dx = pd.DataFrame(dx)

    # save results
    dm.to_pickle(os.path.join(datadir, 'gwcc-lytx-loss-records.p'))
    dx.to_pickle(os.path.join(datadir, 'gwcc-lytx-only-collision-47.p'))

# save collision videos
if False:

    # load collision records metadata
    dc = pd.read_pickle(os.path.join(datadir, 'gwcc-lytx-loss-records.p'))
    dx = pd.read_pickle(os.path.join(datadir, 'gwcc-lytx-only-collision-47.p'))
    de = pd.read_pickle(os.path.join(datadir, 'gwcc-lytx-events.p'))
    db = pd.read_pickle(os.path.join(datadir, 'gwcc-lytx-behaviors.p'))

    # Lytx collision videos
    vdir = os.path.join(datadir, 'videos')
    if os.path.isdir(vdir):
        rmtree(vdir)
    os.mkdir(vdir)

    # matched GWCC-Lytx collision videos
    df = dc.loc[dc['num collision-47 in 24h'] > 0]
    for x, row in tqdm(df.iterrows(), desc='matched collision videos', total=df.shape[0]):

        # Lytx collision records within 24 hours of GW-datetime
        tmin = row['GW-datetime'] - pd.Timedelta(hours=24)
        tmax = row['GW-datetime'] + pd.Timedelta(hours=24)
        dbx = db.loc[(db['VehicleId'] == row['VehicleId']) & (db['RecordDate'] > tmin) & (db['RecordDate'] < tmax) & (db['BehaviorId'] == '47')]
        assert dbx.shape[0] == row['num collision-47 in 24h']
        for _, event in dbx.iterrows():
            fn = os.path.join(vdir, f'gwcc-{row.name}-lytx-{event.name}.mkv')
            lytx.extract_and_save_video(record=event, fn=fn)

    # unmatched Lytx collision videos
    for x, event in tqdm(dx.iterrows(), desc='unmatched collision videos', total=dx.shape[0]):
        fn = os.path.join(vdir, f'lytx-{event.name}.mkv')
        lytx.extract_and_save_video(record=event, fn=fn)

# update collision prediction model population DataFrame (after s0, before s1)
if False:

    # GWCC and Lytx metadata, scan over GW loss records
    dcm = pd.read_pickle(os.path.join(datadir, 'dcm.p'))
    dc = pd.read_pickle(os.path.join(datadir, 'metadata', 'model_params.p'))
    dm0 = utils.get_population_metadata(dcm, dc, datadir=None, oversampled_info=False)
    df = pd.read_pickle(os.path.join(datadir, 'gwcc-lytx-loss-records.p'))
    dx = pd.read_pickle(os.path.join(datadir, 'gwcc-lytx-only-collision-47.p'))
    dcm['collision-gwcc'] = False
    dcm['collision-gwcc-idx'] = None

    # update dcm based on missing rows from GW loss records (df)
    da = defaultdict(list)
    for _, row in df.iterrows():

        # equivalent cpm t0 / t1 / t2 based on GW-datetime
        t1 = row['GW-datetime']
        t1 = pd.Timestamp(year=t1.year, month=t1.month, day=1)
        t0 = t1 - pd.Timedelta(days=90)
        t2 = t1 + pd.Timedelta(days=32)
        t2 = pd.Timestamp(year=t2.year, month=t2.month, day=1)

        # identify GW loss record in dcm
        c0 = dcm['VehicleId'] == row['VehicleId']
        c1 = dcm['time0'] == t0
        c2 = dcm['time1'] == t1
        c3 = dcm['time2'] == t2
        assert (c0 & c1 & c2 & c3).sum() in [0, 1]

        # update dcm, new row
        if (c0 & c1 & c2 & c3).sum() == 0:

            # new record in collision prediction model DataFrame
            da['VehicleId'].append(row['VehicleId'])
            da['EventRecorderId'].append(row['EventRecorderId'])
            da['Model'].append(row['Model'])
            da['CreationDate'].append(pd.NaT)
            da['DeletedDate'].append(pd.NaT)
            da['GroupId'].append(row['GroupId'])
            da['GroupName'].append(row['GroupName'])
            da['CompanyName'].append(row['CompanyName'])
            da['IndustryDesc'].append(row['IndustryDesc'])
            da['CompanyId'].append(row['CompanyId'])
            da['time0'].append(t0)
            da['time1'].append(t1)
            da['time2'].append(t2)
            da['collision-45'].append(False)
            da['collision-45-idx'].append(np.array([]))
            da['collision-46'].append(False)
            da['collision-46-idx'].append(np.array([]))
            da['collision-47'].append(False)
            da['collision-47-idx'].append(np.array([]))
            da['oversampled'].append(False)
            da['oversampled index'].append(None)
            da['collision-gwcc'].append(True)
            da['collision-gwcc-idx'].append(row.name)

        # update dcm, update collision outcome and index
        elif (c0 & c1 & c2 & c3).sum() == 1:
            dcm.loc[(c0 & c1 & c2 & c2), 'collision-gwcc'] = True
            idx = dcm.loc[(c0 & c1 & c2 & c2), 'collision-gwcc-idx'].iloc[0]
            if idx is None:
                dcm.loc[(c0 & c1 & c2 & c2), 'collision-gwcc-idx'] = str(row.name)
            else:
                assert isinstance(idx, str)
                dcm.loc[(c0 & c1 & c2 & c2), 'collision-gwcc-idx'] = idx + f',{row.name}'
    da = pd.DataFrame(da)
    assert da.groupby(['VehicleId', 'time0', 'time1', 'time2']).ngroups == da.shape[0]

    # update and save collision prediction model population DataFrame
    dcm1 = pd.concat((dcm, da), axis=0).copy().reset_index(drop=True)
    dcm1.to_pickle(os.path.join(datadir, 'dcm-gwcc.p'))

    # updated model metadata
    dm1 = utils.get_population_metadata(dcm1, dc, datadir=None, oversampled_info=False)

# collision prediction model metadata and model artifacts (after s5)
if False:

    # collision prediction model metadata
    dg = pd.read_pickle(os.path.join(datadir, 'gwcc-lytx-loss-records.p'))
    dx = pd.read_pickle(os.path.join(datadir, 'positive_instances.p'))
    dp = pd.read_pickle(os.path.join(datadir, 'artifacts-05', 'population-data.p'))
    yp = pd.read_pickle(os.path.join(datadir, 'artifacts-05', 'model-prediction-probabilities.p'))
    ypx = interp1d((yp['prediction probability'].min(), yp['prediction probability'].max()), (0, 100))
    yp['prediction probability'] = ypx(yp['prediction probability'].values)
    assert dp.shape[0] == 64858
    assert np.logical_or(dp['collision-47'], dp['collision-gwcc']).sum() == 562
    assert all(np.logical_or(dp['collision-47'], dp['collision-gwcc']).astype('int') == yp['actual outcome'])

    # ...
    dc = pd.merge(dp, yp, left_index=True, right_index=True, how='inner')
    assert all(dc['outcome'] == dc['actual outcome'])
    dcc = pd.merge(on='VehicleId', how='inner', suffixes=(' min', ' max'),
        left=dc.loc[dc['outcome'] == 0].groupby('VehicleId')['prediction probability'].min().to_frame().reset_index(drop=False),
        right=dc.loc[dc['outcome'] == 0].groupby('VehicleId')['prediction probability'].max().to_frame().reset_index(drop=False))
    dcc['yr'] = dcc['prediction probability max'] - dcc['prediction probability min']
    dcc = dcc.sort_values('yr', ascending=False).reset_index(drop=True)
    dcc = dcc.loc[~dcc['VehicleId'].isin(dg['VehicleId'].values)].reset_index(drop=True)

    # scan over gwcc VINs
    # vins = pd.unique(dg['VIN'])
    # for vin in tqdm(vins, desc='vins'):
    #     vin = '1FUJGLD56GLGY5780'

    #     # data for VIN
    #     gwcc = dg.loc[dg['VIN'] == vin]
    #     assert pd.unique(gwcc['VehicleId']).size == 1
    #     vid = gwcc.iloc[0]['VehicleId']
    for vid in tqdm(dcc['VehicleId'][:20].values, desc='vids'):
        vin = vid
        dpv = dp.loc[dp['VehicleId'] == vid].sort_values('time0')
        dpv = pd.merge(left=dpv, right=yp, left_index=True, right_index=True, how='left')
        assert all(dpv['outcome'] == dpv['actual outcome'])
        for col in ['actual outcome', 'collision-45', 'collision-45-idx', 'collision-46', 'collision-46-idx', 'oversampled', 'oversampled index']:
            del dpv[col]
        # if ~np.any(dpv['collision-gwcc']):
        #     continue

        # fig, ax = open_figure(f'collision-prediction model results for {vin}', figsize=(16, 6))
        # for x, (time0, time1, time2) in dpv[['time0', 'time1', 'time2']].reset_index(drop=True).iterrows():
        #     p0 = ax.fill_between(x=np.array([time0, time1]), y1=np.tile(x - 0.4, 2), y2=np.tile(x + 0.4, 2), color='darkblue', alpha=0.8)
        #     p1 = ax.fill_between(x=np.array([time1, time2]), y1=np.tile(x - 0.4, 2), y2=np.tile(x + 0.4, 2), color='orange', alpha=0.2)
        #     if x == 1:
        #         p0.set_label(f'predictor intervals')
        #         p1.set_label(f'collision intervals')
        # ax.legend(loc='upper left', bbox_to_anchor=(1, 1), numpoints=3, shadow=True, fancybox=True)
        # format_axes('', '', 'predictor and collision intervals in collision-prediction model', ax, apply_concise_date_formatter=True)
        # ax.set_yticks(range(dpv.shape[0]))
        # ax.set_ylim(-0.6, dpv.shape[0] - 0.4)
        # ax.set_yticklabels([f'{int_to_ordinal(x)} interval' for x in range(1, dpv.shape[0] + 1)])
        # ax.set_xlim(dpv['time0'].min() - pd.Timedelta(days=3), dpv['time2'].max() + pd.Timedelta(days=3))
        # largefonts(18)
        # fig.tight_layout()

        # collision prediction model results for vin
        fig, ax = open_figure(f'collision-prediction model results for {vin}', 2, 1, figsize=(20, 8))
        ax[0].plot(dpv['time1'].values, dpv['prediction probability'].values, 'x-', ms=12, lw=3, label='prediction probability')
        ylim = ax[0].get_ylim()
        # for dt in gwcc['GW-datetime']:
        #     ax[0].plot(np.tile(dt, 2), (0, 100), linestyle='dashed', lw=4, color='black', label=f"""GWCC loss record\n{dt.strftime('%m/%d/%Y')}""")
        for c47 in np.hstack(dpv['collision-47-idx'].values):
            dt = dx.loc[c47, 'RecordDate']
            ax[0].plot(np.tile(dt, 2), (0, 100), linestyle='dotted', lw=4, color='darkred', label=f"""Lytx collision-47\n{dt.strftime('%m/%d/%Y')}""")
        format_axes('', '', 'collision prediction model probability and known collision events', ax[0], apply_concise_date_formatter=True)
        ax[0].set_ylim(-2, 102)
        ax[0].set_yticks(np.arange(0, 101, 10))
        ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1), numpoints=3, shadow=True, fancybox=True, handlelength=4)
        for x, (time0, time1, time2) in dpv[['time0', 'time1', 'time2']].reset_index(drop=True).iterrows():
            p0 = ax[1].fill_between(x=np.array([time0, time1]), y1=np.tile(x - 0.4, 2), y2=np.tile(x + 0.4, 2), color='darkblue', alpha=0.8)
            p1 = ax[1].fill_between(x=np.array([time1, time2]), y1=np.tile(x - 0.4, 2), y2=np.tile(x + 0.4, 2), color='orange', alpha=0.2)
            if x == 1:
                p0.set_label(f'predictor intervals')
                p1.set_label(f'collision intervals')
        ax[1].legend(loc='upper left', bbox_to_anchor=(1, 1), numpoints=3, shadow=True, fancybox=True)
        format_axes('', '', 'predictor and collision intervals in collision-prediction model', ax[1], apply_concise_date_formatter=True)
        ax[1].set_yticks(range(dpv.shape[0]))
        ax[1].set_ylim(-0.6, dpv.shape[0] - 0.4)
        ax[1].set_yticklabels([f'{int_to_ordinal(x)} interval' for x in range(1, dpv.shape[0] + 1)])
        for x in ax:
            x.set_xlim(dpv['time0'].min() - pd.Timedelta(days=3), dpv['time2'].max() + pd.Timedelta(days=3))
        largefonts(18)
        fig.tight_layout()
        save_pngs(os.path.join(datadir, 'gwcc'))

# distribution of loss amount in great-west collision records (needs review)
if False:

    # GWCC collision records
    gwcc = pd.read_pickle(os.path.join(datadir, 'gwcc.p'))

    # loss amount distribution
    x = gwcc['GW-loss-amount'].values
    title = f'distribution of GW-loss-amount, {gwcc.shape[0]} loss records'
    metric_distribution(x=x, bins=np.linspace(0, 1.1e6, 120), title=title, xlabel='amount, $', logscale=True, figsize=(10, 4), size=20)

    # top n companies by sum of GW-loss-amount
    n = 20
    dx = gwcc.groupby('CompanyName')['GW-loss-amount'].sum().sort_values(ascending=False).iloc[0:n]
    companies = dx.index.to_numpy()
    amounts = dx.values[::-1]
    title = f'top {n} companies by sum of GW-loss-amount'
    fig, ax = open_figure(title, figsize=(14, 8))
    ax.barh(y=np.arange(companies.size), width=amounts, height=0.8)
    ax.set_yticks(np.arange(companies.size))
    ax.set_ylim(-0.5, companies.size - 0.5)
    ax.set_yticklabels(companies[::-1])
    ax.set_xlim(0, 1.05e6)
    ax.set_xticks((0, 200e3, 400e3, 600e3, 800e3, 1e6))
    ax.set_xticklabels(('$0', '$200k', '$400k', '$600k', '$800k', '$1M'))
    format_axes('sum of GW-loss-amount for all company collision records', '', title, ax)
    largefonts(20)
    fig.tight_layout()

    # top n companies by number of collision records
    dx = gwcc.groupby('CompanyName')['GW-loss-amount'].count().sort_values(ascending=False).iloc[0:n]
    companies = dx.index.to_numpy()
    amounts = dx.values[::-1]
    title = f'top {n} companies by number of collision records'
    fig, ax = open_figure(title, figsize=(14, 8))
    ax.barh(y=np.arange(companies.size), width=amounts, height=0.8)
    ax.set_yticks(np.arange(companies.size))
    ax.set_ylim(-0.5, companies.size - 0.5)
    ax.set_yticklabels(companies[::-1])
    format_axes('number of company collision records', '', title, ax)
    largefonts(20)
    fig.tight_layout()

    save_pngs(r'c:/Users/russell.burdt/Downloads')
    # plt.show()
