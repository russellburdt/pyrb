
"""
retrieve coaching notes from set of companies over set of time windows
"""

import os
import lytx
import boto3
import numpy as np
import pandas as pd
import sqlalchemy as sa
from functools import reduce


# datadir, edw connection
datadir = r'/mnt/home/russell.burdt/data/gen-ai/coach-notes'
assert os.path.isdir(datadir)
edw = lytx.get_conn('edw')

# companies from Distribution, fleet size of 100 to 500 vehicles
query = f"""
    SELECT C.CompanyName, C.CompanyId, COUNT(DISTINCT(ERA.VehicleId)) AS n_vehicle_id
    FROM hs.EventRecorderAssociations AS ERA
        LEFT JOIN flat.Groups AS G ON ERA.GroupId = G.GroupId
        LEFT JOIN flat.Companies AS C ON G.CompanyId = C.CompanyId
        LEFT JOIN hs.EventRecorders AS ER ON ER.Id = ERA.EventRecorderId
    WHERE C.IndustryDesc = 'Distribution'
    AND C.CompanyName <> 'DriveCam DC4DC Test Co'
    GROUP BY C.CompanyName, C.CompanyId
    HAVING COUNT(DISTINCT(ERA.VehicleId)) BETWEEN 100 AND 500"""
dc = pd.read_sql_query(con=edw, sql=sa.text(query))

# gold data-security filter
query = f"""
    SELECT HS_Company_ID__c AS companyId, Data_Rights_Level__c AS level
    FROM sf.Account
    WHERE HS_Company_ID__c IS NOT NULL
    AND Data_Rights_Level__c = 'Gold'"""
# dg = pd.read_sql_query(con=edw, sql=sa.text(query))
dg = pd.read_csv(os.path.join(os.path.split(datadir)[0], 'Account_202308300904.csv'))
dg['CompanyId'] = dg.pop('companyId')
dg = dg[~dg.duplicated()].reset_index(drop=True)
assert pd.merge(left=dc, right=dg, on='CompanyId', how='left').shape[0] == dc.shape[0]
dc = pd.merge(left=dc, right=dg, on='CompanyId', how='left')
dc = dc.loc[dc['level'] == 'Gold'].reset_index(drop=True)

def get_coaching_notes(time0, time1, desc):

    query = f"""
        SELECT
            N.NoteId, N.Note, N.EventId, N.CreationDate, N.CompanyId,
            E.VehicleId, E.Latitude, E.Longitude, E.SpeedAtTrigger, E.RecordDate,
            E.EventTriggerTypeId, E.EventTriggerSubTypeId, E.EventFilePath,
            E.EventFileName, E.EventRecorderId
        FROM flat.Notes AS N
            LEFT JOIN flat.Events AS E
            ON N.EventId = E.EventId
        WHERE N.CreationDate BETWEEN '{time0}' AND '{time1}'
        AND E.Deleted=0
        AND N.Type = 'Coach'
        AND N.CompanyId IN ({','.join(dc['CompanyId'].astype(str))})"""
    dx = pd.read_sql_query(con=edw, sql=sa.text(query))
    dx['desc'] = desc

    # high-level note metrics
    dm = pd.merge(left_index=True, right_index=True,
        left=dx.groupby('CompanyId')['Note'].nunique(),
        right=dx.groupby('CompanyId')['VehicleId'].nunique())
    dm = pd.merge(left_index=True, right_index=True, left=dm,
        right=dx.groupby('CompanyId')['EventId'].nunique())
    dm = dm.rename(columns={
        'Note': f'notes-{desc}',
        'VehicleId': f'vehicles-{desc}',
        'EventId': f'events-{desc}'})
    assert dx.groupby(['CompanyId', 'NoteId', 'EventId']).ngroups == dx.shape[0]

    return dx, dm

# coaching notes
dx1, dm1 = get_coaching_notes(time0=pd.Timestamp('4/1/2023'), time1=pd.Timestamp('5/1/2023'), desc='Apr23')
dx2, dm2 = get_coaching_notes(time0=pd.Timestamp('5/1/2023'), time1=pd.Timestamp('6/1/2023'), desc='May23')
dx3, dm3 = get_coaching_notes(time0=pd.Timestamp('6/1/2023'), time1=pd.Timestamp('7/1/2023'), desc='Jun23')
dm = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='inner'), (dm1, dm2, dm3))
dm = dm.reset_index(drop=False)
dm = pd.merge(left=dm, right=dc[['CompanyId', 'CompanyName']], on='CompanyId', how='left')
dm = dm[sorted(dm.columns)]
assert dx1.duplicated().sum() == dx2.duplicated().sum() == dx3.duplicated().sum() == 0
assert all(dx1.columns == dx2.columns) and all(dx2.columns == dx3.columns)
dx = pd.concat((dx1, dx2, dx3), axis=0).reset_index(drop=True)
assert dx.duplicated().sum() == 0
assert len(set(dx['CompanyId']).intersection(dm['CompanyId'])) == dm.shape[0]
dm = dm.loc[np.random.choice(dm.index, size=40, replace=False)].reset_index(drop=True)
dx = dx.loc[dx['CompanyId'].isin(dm['CompanyId'])].reset_index(drop=True)

# save data
dm.to_parquet(os.path.join(datadir, 'metadata.parquet'), engine='pyarrow', index=False)
dx.to_parquet(path=datadir, engine='pyarrow', compression='snappy', index=False, partition_cols=['CompanyId', 'desc'])
